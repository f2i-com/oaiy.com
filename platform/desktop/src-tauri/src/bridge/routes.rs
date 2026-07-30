//! HTTP surface for the Bridge Protocol v1.
//!
//! Mounted alongside the existing `/api/*` routes as its own `Router` with its
//! own state, so `http.rs` only has to merge it.
//!
//! | Route | Does |
//! |---|---|
//! | `GET  /api/bridge/capabilities` | what this runtime can do right now |
//! | `POST /api/bridge/runs` | reserve a run (idempotent) |
//! | `GET  /api/bridge/runs` | claimable runs, oldest first |
//! | `GET  /api/bridge/runs/:id` | one run |
//! | `POST /api/bridge/runs/:id/claim` | single-winner claim |
//! | `POST /api/bridge/runs/:id/cancel` | request cancellation |
//! | `GET  /api/plugins` | installed plugins + state + reason |
//! | `POST /api/bridge/connectors/:id/request` | gated connector command |
//!
//! # Status codes carry meaning here
//!
//! The protocol distinguishes outcomes a caller must branch on, so they get
//! distinct codes rather than a uniform 200 with a status field:
//!
//! - `201` a run was reserved and will execute.
//! - `200` **with `idempotent: true`** — this key already existed, nothing new ran.
//! - `409` the claim was lost, the run is already terminal, or a loop guard
//!   refused it. All three mean "do not proceed", and all three are things a
//!   correct caller can hit under normal concurrency.
//! - `422` a loop guard refused. Distinguished from a lost claim because the
//!   caller should stop retrying rather than back off.
//!
//! Collapsing these into 200 is how a consumer ends up treating "someone else is
//! already running this" as "I should run it".

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::ledger::{
    ClaimOutcome, LedgerHandle, LineageRef, ReserveOutcome, RunRequest, RunStatus, Runtime,
};
use crate::http::BRIDGE_PROTOCOL;
use crate::plugins::registry::PluginRegistryHandle;
use crate::plugins::{CallError, ForwardError, CONNECTOR_TIMEOUT};

#[derive(Clone)]
pub struct BridgeState {
    pub ledger: LedgerHandle,
    pub plugins: PluginRegistryHandle,
    /// The live-process side of the plugin story: start/stop, health, and the
    /// gated forwarding path. Shares `plugins` and `ledger` with this state.
    pub host: std::sync::Arc<crate::plugins::PluginHost>,
    pub flows: std::sync::Arc<super::worker::FlowStore>,
    /// Pairing: minting + validating the tokens an untrusted-origin consumer
    /// uses to reach privileged routes. Shared with the HTTP auth guard.
    pub pairing: super::pairing::PairingHandle,
    /// Stable per-install id, echoed in discovery so a consumer can tell two
    /// machines apart in run history.
    pub device_id: String,
}

/// `caller` per `protocol/v1/caller.schema.json`.
///
/// Only `product` is read. The rest is stored and echoed but never parsed — see
/// the protocol's genericity note. `deny_unknown_fields` is deliberate: it is
/// what makes FormLogic's `appContext` a 400 rather than a silently ignored field
/// that leaves the caller thinking it passed scope information.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Caller {
    pub product: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LineageBody {
    #[serde(default)]
    pub root_run_id: Option<String>,
    #[serde(default)]
    pub parent_run_id: Option<String>,
    #[serde(default)]
    pub binding_id: Option<String>,
    #[serde(default)]
    pub depth: u32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunRequestBody {
    pub protocol: String,
    pub caller: Caller,
    #[serde(default)]
    pub flow_id: Option<String>,
    #[serde(default)]
    pub graph: Option<serde_json::Value>,
    #[serde(default)]
    pub input: Option<serde_json::Value>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    pub correlation_id: String,
    pub idempotency_key: String,
    #[serde(default)]
    pub lineage: Option<LineageBody>,
    #[serde(default)]
    pub trigger_event: Option<String>,
}

/// Why a request was rejected before it reached the ledger.
#[derive(Debug, PartialEq, Eq)]
pub enum RequestRejection {
    ProtocolMismatch { got: String },
    NoFlow,
    BothFlowAndGraph,
    EmptyIdempotencyKey,
    UnknownMode { got: String },
}

impl RequestRejection {
    pub fn message(&self) -> String {
        match self {
            // Naming what we speak matters: a consumer seeing only "unsupported"
            // has to guess whether to upgrade or downgrade.
            RequestRejection::ProtocolMismatch { got } => format!(
                "unsupported protocol {got:?}; this runtime speaks {BRIDGE_PROTOCOL}"
            ),
            RequestRejection::NoFlow => "one of flowId or graph is required".into(),
            RequestRejection::BothFlowAndGraph => {
                "flowId and graph are mutually exclusive; send one".into()
            }
            RequestRejection::EmptyIdempotencyKey => {
                "idempotencyKey must be a non-empty, stable key for this logical event".into()
            }
            RequestRejection::UnknownMode { got } => {
                format!("unknown mode {got:?}; expected sync, async or queued")
            }
        }
    }
}

/// Validate a run request. Pure, so the rules are directly testable.
pub fn validate(body: &RunRequestBody) -> Result<(), RequestRejection> {
    if body.protocol != BRIDGE_PROTOCOL {
        return Err(RequestRejection::ProtocolMismatch {
            got: body.protocol.clone(),
        });
    }
    match (&body.flow_id, &body.graph) {
        (None, None) => return Err(RequestRejection::NoFlow),
        (Some(_), Some(_)) => return Err(RequestRejection::BothFlowAndGraph),
        _ => {}
    }
    if body.idempotency_key.trim().is_empty() {
        return Err(RequestRejection::EmptyIdempotencyKey);
    }
    if let Some(mode) = &body.mode {
        if !matches!(mode.as_str(), "sync" | "async" | "queued") {
            return Err(RequestRejection::UnknownMode { got: mode.clone() });
        }
    }
    Ok(())
}

fn to_run_request(body: &RunRequestBody) -> RunRequest {
    let lineage = body
        .lineage
        .as_ref()
        .map(|l| LineageRef {
            root_run_id: l.root_run_id.clone(),
            parent_run_id: l.parent_run_id.clone(),
            binding_id: l.binding_id.clone(),
            depth: l.depth,
        })
        .unwrap_or_default();
    RunRequest {
        caller_product: body.caller.product.clone(),
        flow_id: body.flow_id.clone(),
        inline_graph: body.graph.is_some(),
        input: body.input.clone(),
        timeout_ms: body.timeout_ms,
        mode: body.mode.clone().unwrap_or_else(|| "async".into()),
        correlation_id: body.correlation_id.clone(),
        idempotency_key: body.idempotency_key.clone(),
        lineage,
        trigger_event: body.trigger_event.clone(),
    }
}

/// The closed error taxonomy from `protocol/v1/error.schema.json`. A plugin may
/// name one of these; anything else the host authors itself.
fn is_taxonomy_code(code: &str) -> bool {
    matches!(
        code,
        "invalid_request"
            | "invalid_flow"
            | "flow_not_found"
            | "capability_denied"
            | "capability_unavailable"
            | "connection_missing"
            | "node_failed"
            | "timeout"
            | "cancelled"
            | "runtime_unavailable"
            | "internal"
    )
}

fn bridge_error(status: StatusCode, code: &str, message: String) -> axum::response::Response {
    (
        status,
        Json(json!({ "error": { "code": code, "message": message } })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------

async fn capabilities(State(st): State<BridgeState>) -> impl IntoResponse {
    // Plugin-contributed connector commands are real capabilities, so discovery
    // must reflect the plugin's live state — a stopped plugin's commands are
    // listed as unavailable WITH a reason rather than omitted, so a caller can
    // tell "installed but stopped" from "never heard of it".
    let mut caps: Vec<serde_json::Value> = Vec::new();

    if let Ok(mut reg) = st.plugins.lock() {
        // Scan here too, not only in `/api/plugins`.
        //
        // Without this, a fresh process reports ZERO capabilities until something
        // else happens to hit the plugins route — and discovery is the documented
        // FIRST call, so a consumer doing the right thing would conclude the
        // plugin was not installed. Discovery must not depend on call order.
        // `scan` preserves live state, so this cannot disturb a running plugin.
        reg.scan();
        for rec in reg.list() {
            let Some(m) = rec.manifest.as_ref() else {
                continue;
            };
            let usable = rec.state.accepts_commands();
            for conn in &m.connectors {
                for cmd in &conn.commands {
                    let id = format!("connector.{}.{}", conn.id, cmd);
                    if usable {
                        caps.push(json!({ "id": id, "available": true, "pluginId": rec.id }));
                    } else {
                        // The reason comes from the state, not a guess. Reporting
                        // `plugin_crashed` for a plugin that was never started
                        // tells someone their software broke when it is idle.
                        let reason = rec
                            .state
                            .unavailable_reason(rec.user_disabled)
                            .unwrap_or("service_stopped");
                        // `detail` must be actionable — the protocol requires it —
                        // so the registry's reason is prefixed with what to DO,
                        // rather than shipped alone as a bare status line.
                        let detail = match rec.reason.as_deref() {
                            Some(r) => format!(
                                "The {} plugin is not running ({r}) Start it in OAIY Desktop → Plugins.",
                                rec.id
                            ),
                            None => format!(
                                "The {} plugin is {:?}. Start it in OAIY Desktop → Plugins.",
                                rec.id, rec.state
                            ),
                        };
                        caps.push(json!({
                            "id": id,
                            "available": false,
                            "reason": reason,
                            "detail": detail,
                            "pluginId": rec.id,
                        }));
                    }
                }
            }
        }
    }

    Json(json!({
        "protocol": BRIDGE_PROTOCOL,
        "runtime": "desktop",
        "deviceId": st.device_id,
        "capabilities": caps,
    }))
}

async fn create_run(
    State(st): State<BridgeState>,
    Json(body): Json<RunRequestBody>,
) -> axum::response::Response {
    if let Err(rej) = validate(&body) {
        return bridge_error(StatusCode::BAD_REQUEST, "invalid_request", rej.message());
    }
    let req = to_run_request(&body);
    // Block-scoped: this handler awaits below, and a std MutexGuard anywhere in
    // the async fn's state — even one already drop()ed — makes the future !Send
    // and the whole route fail to compile as a handler. The guard must
    // provably end before the first await.
    let outcome = {
        let mut ledger = match st.ledger.lock() {
            Ok(l) => l,
            Err(_) => {
                return bridge_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    "run ledger lock poisoned".into(),
                )
            }
        };
        ledger.reserve(&req)
    };

    match outcome {
        ReserveOutcome::Reserved(rec) => {
            if req.mode == "sync" {
                // `sync` means the caller wants the terminal result, and the
                // protocol says so — the first cut validated the mode and then
                // ignored it, returning 201 Queued, which made sync
                // indistinguishable from async and broke every caller that
                // trusted the contract. Poll the ledger (the worker executes on
                // its own thread) up to the run's own budget plus scheduling
                // slack.
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_millis(req.timeout_ms.unwrap_or(30_000))
                    + std::time::Duration::from_secs(5);
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
                    let current = { st.ledger.lock().ok().and_then(|l| l.get(&rec.run_id)) };
                    match current {
                        Some(r) if r.status.is_terminal() => {
                            return (StatusCode::OK, Json(r)).into_response();
                        }
                        _ if std::time::Instant::now() >= deadline => {
                            // Honest partial answer: accepted, still running.
                            // 202 rather than 200 so a schema-driven caller can
                            // tell "result" from "still waiting".
                            let latest = { st.ledger.lock().ok().and_then(|l| l.get(&rec.run_id)) }
                                .unwrap_or_else(|| rec.clone());
                            return (StatusCode::ACCEPTED, Json(latest)).into_response();
                        }
                        _ => {}
                    }
                }
            }
            (StatusCode::CREATED, Json(rec)).into_response()
        }
        // 200, not 201: nothing was created and nothing will execute. The flag is
        // what stops a caller treating a dedupe as a fresh run.
        ReserveOutcome::Duplicate(mut rec) => {
            rec.idempotent = true;
            (StatusCode::OK, Json(rec)).into_response()
        }
        // 422 rather than 409: a guard refusal will never succeed on retry, so
        // the caller should stop rather than back off.
        ReserveOutcome::Refused { reason } => bridge_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request",
            format!("refused by a loop guard: {reason}"),
        ),
    }
}

async fn get_run(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.ledger.lock() {
        Ok(l) => match l.get(&id) {
            Some(rec) => (StatusCode::OK, Json(rec)).into_response(),
            None => bridge_error(
                StatusCode::NOT_FOUND,
                "invalid_request",
                format!("unknown run {id}"),
            ),
        },
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "run ledger lock poisoned".into(),
        ),
    }
}

async fn queued_runs(State(st): State<BridgeState>) -> axum::response::Response {
    match st.ledger.lock() {
        Ok(l) => (StatusCode::OK, Json(json!({ "runs": l.queued(100) }))).into_response(),
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "run ledger lock poisoned".into(),
        ),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaimBody {
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default)]
    worker: Option<String>,
}

async fn claim_run(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
    body: Option<Json<ClaimBody>>,
) -> axum::response::Response {
    let (runtime, worker) = match body {
        Some(Json(b)) => (
            match b.runtime.as_deref() {
                Some("browser") => Runtime::Browser,
                Some("cli") => Runtime::Cli,
                Some("cloud") => Runtime::Cloud,
                _ => Runtime::Desktop,
            },
            b.worker.unwrap_or_else(|| "oaiy-desktop".into()),
        ),
        None => (Runtime::Desktop, "oaiy-desktop".to_string()),
    };

    let mut ledger = match st.ledger.lock() {
        Ok(l) => l,
        Err(_) => {
            return bridge_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "run ledger lock poisoned".into(),
            )
        }
    };
    match ledger.claim(&id, runtime, &worker) {
        ClaimOutcome::Claimed(rec) => (StatusCode::OK, Json(rec)).into_response(),
        // 409 with the winner named: "someone else has it" is a normal outcome
        // under concurrency, and naming the holder makes it diagnosable.
        ClaimOutcome::AlreadyClaimed { claimed_by } => bridge_error(
            StatusCode::CONFLICT,
            "invalid_request",
            format!(
                "run {id} is already claimed by {}",
                claimed_by.unwrap_or_else(|| "another worker".into())
            ),
        ),
        ClaimOutcome::NotClaimable { status } => bridge_error(
            StatusCode::CONFLICT,
            "invalid_request",
            format!("run {id} is {status:?} and cannot be claimed"),
        ),
        ClaimOutcome::Unknown => bridge_error(
            StatusCode::NOT_FOUND,
            "invalid_request",
            format!("unknown run {id}"),
        ),
    }
}

async fn cancel_run(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
) -> axum::response::Response {
    let mut ledger = match st.ledger.lock() {
        Ok(l) => l,
        Err(_) => {
            return bridge_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "run ledger lock poisoned".into(),
            )
        }
    };
    match ledger.request_cancel(&id) {
        // 202 for a running run: cancellation is a REQUEST. Returning 200 would
        // imply the work had stopped, and a node mid-HTTP-call has not.
        Ok(RunStatus::Running) => (
            StatusCode::ACCEPTED,
            Json(json!({ "runId": id, "status": "running", "cancelRequested": true })),
        )
            .into_response(),
        Ok(status) => (
            StatusCode::OK,
            Json(json!({ "runId": id, "status": status })),
        )
            .into_response(),
        Err(e) => bridge_error(StatusCode::CONFLICT, "invalid_request", e),
    }
}

async fn list_plugins(State(st): State<BridgeState>) -> axum::response::Response {
    match st.plugins.lock() {
        Ok(mut reg) => {
            // Rescan on read so a plugin dropped into the folder appears without
            // a restart. `scan` preserves live state, so this cannot knock a
            // running plugin back to Installed.
            let report = reg.scan();
            (
                StatusCode::OK,
                Json(json!({
                    "plugins": reg.list(),
                    "root": reg.root().display().to_string(),
                    "scan": { "added": report.added, "unchanged": report.unchanged, "invalid": report.invalid },
                })),
            )
                .into_response()
        }
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "plugin registry lock poisoned".into(),
        ),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConnectorBody {
    command: String,
    /// Accepted and validated as part of the wire shape, but not yet forwarded —
    /// the supervised plugin process is not wired up, so the handler refuses
    /// typed rather than pretending to deliver it. Kept in the struct so a caller
    /// sending a correct request is not rejected for a field this build cannot
    /// act on yet.
    #[allow(dead_code)]
    #[serde(default)]
    payload: Option<serde_json::Value>,
    #[serde(default)]
    idempotency_key: Option<String>,
}

async fn connector_request(
    State(st): State<BridgeState>,
    Path(connector_id): Path<String>,
    Json(body): Json<ConnectorBody>,
) -> axum::response::Response {
    // The host gates first (state, allow-list, journalling), forwards second.
    // Run on a blocking thread: the plugin RPC legitimately takes seconds and
    // must not park an async executor thread for the duration.
    let host = st.host.clone();
    let result = tokio::task::spawn_blocking(move || {
        host.forward_connector(
            &connector_id,
            &body.command,
            body.payload,
            body.idempotency_key.as_deref(),
            CONNECTOR_TIMEOUT,
        )
    })
    .await;

    match result {
        Ok(Ok(value)) => (StatusCode::OK, Json(json!({ "ok": true, "result": value }))).into_response(),
        Ok(Err(ForwardError::Refused(refusal))) => {
            let status = match refusal.code() {
                "capability_denied" => StatusCode::FORBIDDEN,
                "invalid_request" => StatusCode::BAD_REQUEST,
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            bridge_error(status, refusal.code(), refusal.message())
        }
        Ok(Err(ForwardError::NotRunning { plugin_id })) => bridge_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "capability_unavailable",
            format!(
                "The {plugin_id} plugin stopped between the gate check and the call.                  Start it in OAIY Desktop → Plugins."
            ),
        ),
        Ok(Err(ForwardError::Call(CallError::Timeout { method, waited }))) => bridge_error(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            format!("the plugin did not answer {method} within {:.0}s", waited.as_secs_f32()),
        ),
        Ok(Err(ForwardError::Call(CallError::Plugin { message, typed, .. }))) => {
            // The plugin's typed code passes through only if it is IN the closed
            // taxonomy. A plugin is untrusted, and `error.code` is the field a
            // consumer branches on — echoing an arbitrary string lets a plugin
            // emit `capability_denied` (a verdict the host never made) or a code
            // outside the enum (which fails schema validation at a conforming
            // consumer, the exact failure the closed set exists to prevent).
            // Anything unrecognised becomes node_failed: the plugin's call
            // genuinely failed, we just will not let it name the reason falsely.
            let code = match typed.as_deref() {
                Some(t) if is_taxonomy_code(t) => t,
                _ => "node_failed",
            };
            bridge_error(StatusCode::BAD_GATEWAY, code, message)
        }
        Ok(Err(ForwardError::Call(e))) => {
            bridge_error(StatusCode::BAD_GATEWAY, "runtime_unavailable", e.to_string())
        }
        Ok(Err(ForwardError::Internal(m))) => {
            bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", m)
        }
        Err(join) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("forwarding task failed: {join}"),
        ),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FinishBody {
    status: String,
    #[serde(default)]
    output: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<crate::bridge::ledger::RunError>,
}

/// Finalise a run an EXTERNAL claimer executed.
///
/// The desktop's own worker finishes its runs in-process, which is why this
/// route did not exist at first — and its absence made `mode: "queued"` a trap:
/// a browser could claim a run and then had no way to ever record its outcome,
/// so the run sat `running` forever, immune even to cancellation (which only
/// flags; the finisher is whoever executes). The claim/finish pair is only a
/// pair if both halves are reachable.
async fn finish_run(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
    Json(body): Json<FinishBody>,
) -> axum::response::Response {
    let status = match body.status.as_str() {
        "succeeded" => RunStatus::Succeeded,
        "failed" => RunStatus::Failed,
        "timed_out" => RunStatus::TimedOut,
        "cancelled" => RunStatus::Cancelled,
        other => {
            return bridge_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                format!("{other:?} is not a terminal status (succeeded | failed | timed_out | cancelled)"),
            )
        }
    };
    let mut ledger = match st.ledger.lock() {
        Ok(l) => l,
        Err(_) => {
            return bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "run ledger lock poisoned".into())
        }
    };
    match ledger.finish(&id, status, body.output, body.error) {
        Ok(rec) => (StatusCode::OK, Json(rec)).into_response(),
        // The ledger's refusals here are all conflicts: already terminal, a
        // failure with no error, an unknown run. 409 tells the claimer its view
        // of the run is stale — re-read, don't retry blindly.
        Err(e) => bridge_error(StatusCode::CONFLICT, "invalid_request", e),
    }
}

// ------- plugin lifecycle -------

async fn start_plugin(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    let host = st.host.clone();
    // Blocking: spawn + handshake can take the full HANDSHAKE_TIMEOUT.
    match tokio::task::spawn_blocking(move || host.start(&id)).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => bridge_error(StatusCode::CONFLICT, "capability_unavailable", e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

async fn stop_plugin(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    let host = st.host.clone();
    match tokio::task::spawn_blocking(move || host.stop(&id)).await {
        Ok(Ok(())) => StatusCode::NO_CONTENT.into_response(),
        Ok(Err(e)) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
struct LogsQuery {
    tail: Option<usize>,
}

async fn plugin_logs(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
    axum::extract::Query(q): axum::extract::Query<LogsQuery>,
) -> axum::response::Response {
    match st.host.logs(&id, q.tail) {
        Some(lines) => (StatusCode::OK, Json(json!({ "lines": lines }))).into_response(),
        // Not running is not an error - logs simply do not exist yet. An empty
        // list keeps a UI polling logs from erroring the moment a plugin stops.
        None => (StatusCode::OK, Json(json!({ "lines": [] }))).into_response(),
    }
}

#[derive(Debug, Deserialize)]
struct EnableBody {
    enabled: bool,
}

async fn set_plugin_enabled(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
    Json(body): Json<EnableBody>,
) -> axum::response::Response {
    if !body.enabled {
        // Disabling a running plugin stops it first - a disabled-but-running
        // plugin would keep serving commands the user just declined.
        let host = st.host.clone();
        let id2 = id.clone();
        let _ = tokio::task::spawn_blocking(move || host.stop(&id2)).await;
    }
    match st.plugins.lock() {
        Ok(mut reg) => {
            reg.set_user_disabled(&id, !body.enabled);
            match reg.get(&id) {
                Some(rec) => (StatusCode::OK, Json(rec.clone())).into_response(),
                None => bridge_error(
                    StatusCode::NOT_FOUND,
                    "invalid_request",
                    format!("no plugin named {id:?}"),
                ),
            }
        }
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "registry lock poisoned".into()),
    }
}

// ------- triggers -------

async fn list_triggers(State(st): State<BridgeState>) -> axum::response::Response {
    match st.host.triggers.lock() {
        Ok(t) => (StatusCode::OK, Json(json!({ "bindings": t.list() }))).into_response(),
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "trigger store lock poisoned".into()),
    }
}

async fn upsert_trigger(
    State(st): State<BridgeState>,
    Json(binding): Json<crate::bridge::triggers::TriggerBinding>,
) -> axum::response::Response {
    match st.host.triggers.lock() {
        Ok(mut t) => match t.upsert(binding) {
            Ok(()) => (StatusCode::OK, Json(json!({ "bindings": t.list() }))).into_response(),
            Err(e) => bridge_error(StatusCode::BAD_REQUEST, "invalid_request", e),
        },
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "trigger store lock poisoned".into()),
    }
}

async fn delete_trigger(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.host.triggers.lock() {
        Ok(mut t) => match t.remove(&id) {
            Ok(true) => StatusCode::NO_CONTENT.into_response(),
            Ok(false) => bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no binding {id:?}")),
            Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e),
        },
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "trigger store lock poisoned".into()),
    }
}

// ------- events (polling) -------

#[derive(Debug, Deserialize)]
struct EventsQuery {
    #[serde(default)]
    since: u64,
    limit: Option<usize>,
}

async fn poll_events(
    State(st): State<BridgeState>,
    axum::extract::Query(q): axum::extract::Query<EventsQuery>,
) -> axum::response::Response {
    let events = st.host.events_since(q.since, q.limit.unwrap_or(100).min(500));
    let next = events.last().map(|e| e.seq).unwrap_or(q.since);
    (StatusCode::OK, Json(json!({ "events": events, "next": next }))).into_response()
}

// ------- flows -------

async fn list_flows(State(st): State<BridgeState>) -> axum::response::Response {
    let flows: Vec<serde_json::Value> = st
        .flows
        .list()
        .into_iter()
        .map(|(id, name)| json!({ "flowId": id, "name": name.unwrap_or_else(|| id.clone()) }))
        .collect();
    (StatusCode::OK, Json(json!({ "flows": flows }))).into_response()
}

async fn put_flow(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
    body: String,
) -> axum::response::Response {
    match st.flows.put(&id, &body) {
        Ok(_) => (StatusCode::OK, Json(json!({ "flowId": id }))).into_response(),
        Err(e) => bridge_error(StatusCode::BAD_REQUEST, "invalid_request", e),
    }
}

async fn delete_flow(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.flows.delete(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no flow {id:?}")),
        Err(e) => bridge_error(StatusCode::BAD_REQUEST, "invalid_request", e),
    }
}

// ------- pairing -------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PairingRequestBody {
    /// Consuming product, e.g. "formlogic". Displayed to the user; not trusted.
    product: String,
    #[serde(default)]
    label: Option<String>,
}

/// Raise a pairing request. UNAUTHENTICATED by design — its only power is to put
/// a prompt in front of the user, who is the actual trust boundary. The response
/// carries the code the consumer shows so the user can confirm it matches the
/// prompt. The Origin (a real browser cannot forge it) is captured for the
/// prompt.
async fn create_pairing(
    State(st): State<BridgeState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<PairingRequestBody>,
) -> axum::response::Response {
    if body.product.trim().is_empty() {
        return bridge_error(StatusCode::BAD_REQUEST, "invalid_request", "product is required".into());
    }
    let origin = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|o| o.to_str().ok())
        .map(str::to_string);
    match st.pairing.lock() {
        Ok(mut mgr) => {
            let req = mgr.request(&body.product, body.label, origin);
            (
                StatusCode::CREATED,
                Json(json!({ "pairingId": req.pairing_id, "code": req.code, "status": req.status })),
            )
                .into_response()
        }
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// The consumer polls for approval. Open — a pairingId is an unguessable handle,
/// and the token is only ever returned to whoever holds it. Returns the status
/// and, once approved, the token for the consumer to store. Re-fetchable by the
/// id-holder until the request ages out (so a dropped response can retry), not
/// single-use — the unguessable id and the TTL bound the exposure.
async fn poll_pairing(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mut mgr) => match mgr.poll(&id) {
            Some(req) => (
                StatusCode::OK,
                Json(json!({ "status": req.status, "token": req.token, "product": req.product })),
            )
                .into_response(),
            None => bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("unknown pairing {id}")),
        },
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// Pending requests for the approval UI. PRIVILEGED — only OAIY's own webview (or
/// a token holder) may see who is asking and act on it.
async fn list_pending_pairings(State(st): State<BridgeState>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mut mgr) => (StatusCode::OK, Json(json!({ "pending": mgr.pending() }))).into_response(),
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// Approve a pending request — the user's trust act. PRIVILEGED.
async fn approve_pairing(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mut mgr) => match mgr.approve(&id) {
            // The token is not returned here — the requester collects it by
            // polling, so it only travels to whoever holds the pairingId.
            Ok(_) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => bridge_error(StatusCode::CONFLICT, "invalid_request", e),
        },
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// Deny a pending request. PRIVILEGED.
async fn deny_pairing(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mut mgr) => match mgr.deny(&id) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(e) => bridge_error(StatusCode::NOT_FOUND, "invalid_request", e),
        },
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// The apps currently paired, secret-free. PRIVILEGED.
async fn list_paired(State(st): State<BridgeState>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mgr) => (StatusCode::OK, Json(json!({ "paired": mgr.paired() }))).into_response(),
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// Revoke a granted token by its public id (unpair). PRIVILEGED.
async fn revoke_pairing(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.pairing.lock() {
        Ok(mut mgr) => {
            if mgr.revoke(&id) {
                StatusCode::NO_CONTENT.into_response()
            } else {
                bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no paired app {id}"))
            }
        }
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "pairing lock poisoned".into()),
    }
}

/// The bridge router, ready to `.merge()` into the main app.
pub fn router(state: BridgeState) -> Router {
    Router::new()
        .route("/api/bridge/capabilities", get(capabilities))
        .route("/api/bridge/runs", get(queued_runs).post(create_run))
        .route("/api/bridge/runs/:id", get(get_run))
        .route("/api/bridge/runs/:id/claim", post(claim_run))
        .route("/api/bridge/runs/:id/finish", post(finish_run))
        .route("/api/bridge/runs/:id/cancel", post(cancel_run))
        .route("/api/plugins", get(list_plugins))
        .route(
            "/api/bridge/connectors/:id/request",
            post(connector_request),
        )
        .route("/api/plugins/:id/start", post(start_plugin))
        .route("/api/plugins/:id/stop", post(stop_plugin))
        .route("/api/plugins/:id/logs", get(plugin_logs))
        .route("/api/plugins/:id/enabled", post(set_plugin_enabled))
        .route("/api/bridge/triggers", get(list_triggers).post(upsert_trigger))
        .route("/api/bridge/triggers/:id", axum::routing::delete(delete_trigger))
        .route("/api/bridge/events", get(poll_events))
        .route("/api/bridge/flows", get(list_flows))
        .route(
            "/api/bridge/flows/:id",
            axum::routing::put(put_flow).delete(delete_flow),
        )
        .route("/api/bridge/pairing", get(list_pending_pairings).post(create_pairing))
        .route("/api/bridge/pairing/:id", get(poll_pairing))
        .route("/api/bridge/pairing/:id/approve", post(approve_pairing))
        .route("/api/bridge/pairing/:id/deny", post(deny_pairing))
        .route("/api/bridge/pairings", get(list_paired))
        .route("/api/bridge/pairings/:id", axum::routing::delete(revoke_pairing))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(json_str: &str) -> Result<RunRequestBody, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    fn valid() -> RunRequestBody {
        body(&format!(
            r#"{{"protocol":"{BRIDGE_PROTOCOL}","caller":{{"product":"formlogic"}},
                "flowId":"f","correlationId":"c","idempotencyKey":"k"}}"#
        ))
        .expect("fixture must parse")
    }

    #[test]
    fn a_well_formed_request_validates() {
        assert!(validate(&valid()).is_ok());
    }

    #[test]
    fn a_foreign_protocol_is_refused_and_says_what_we_speak() {
        let mut b = valid();
        b.protocol = "oaiy-bridge/2".into();
        let msg = validate(&b).unwrap_err().message();
        assert!(msg.contains("oaiy-bridge/2"), "{msg}");
        assert!(
            msg.contains(BRIDGE_PROTOCOL),
            "a consumer seeing only 'unsupported' has to guess which way to move: {msg}"
        );
    }

    #[test]
    fn neither_flow_nor_graph_is_refused() {
        let mut b = valid();
        b.flow_id = None;
        assert_eq!(validate(&b).unwrap_err(), RequestRejection::NoFlow);
    }

    #[test]
    fn both_flow_and_graph_is_refused() {
        let mut b = valid();
        b.graph = Some(json!({"nodes": []}));
        assert_eq!(
            validate(&b).unwrap_err(),
            RequestRejection::BothFlowAndGraph
        );
    }

    #[test]
    fn a_blank_idempotency_key_is_refused() {
        // A whitespace key would pass a naive is_empty check and then dedupe
        // every unrelated run against itself.
        for k in ["", "   ", "\t"] {
            let mut b = valid();
            b.idempotency_key = k.into();
            assert_eq!(
                validate(&b).unwrap_err(),
                RequestRejection::EmptyIdempotencyKey,
                "{k:?}"
            );
        }
    }

    #[test]
    fn an_unknown_mode_is_refused() {
        let mut b = valid();
        b.mode = Some("eventually".into());
        assert!(matches!(
            validate(&b).unwrap_err(),
            RequestRejection::UnknownMode { .. }
        ));
        for good in ["sync", "async", "queued"] {
            let mut b = valid();
            b.mode = Some(good.into());
            assert!(validate(&b).is_ok(), "{good}");
        }
    }

    #[test]
    fn formlogics_app_context_is_a_parse_error_not_a_silent_drop() {
        // The genericity guarantee, enforced at the edge. Ignoring the field
        // would leave a caller believing it had passed scope information.
        let err = body(&format!(
            r#"{{"protocol":"{BRIDGE_PROTOCOL}","caller":{{"product":"formlogic"}},
                "appContext":{{"appSlug":"receptionist"}},
                "flowId":"f","correlationId":"c","idempotencyKey":"k"}}"#
        ))
        .expect_err("appContext must be refused");
        assert!(err.to_string().contains("appContext"), "{err}");
    }

    #[test]
    fn an_unknown_caller_field_is_a_parse_error() {
        let err = body(&format!(
            r#"{{"protocol":"{BRIDGE_PROTOCOL}","caller":{{"product":"formlogic","appSlug":"x"}},
                "flowId":"f","correlationId":"c","idempotencyKey":"k"}}"#
        ))
        .expect_err("an unknown caller field must be refused");
        assert!(err.to_string().contains("appSlug"), "{err}");
    }

    #[test]
    fn opaque_caller_fields_are_accepted_and_preserved() {
        let b = body(&format!(
            r#"{{"protocol":"{BRIDGE_PROTOCOL}",
                "caller":{{"product":"formlogic","tenantId":"u_8814","scopeId":"app:receptionist","label":"Acme"}},
                "flowId":"f","correlationId":"c","idempotencyKey":"k"}}"#
        ))
        .expect("opaque fields are legitimate");
        assert_eq!(b.caller.scope_id.as_deref(), Some("app:receptionist"));
        assert_eq!(b.caller.tenant_id.as_deref(), Some("u_8814"));
    }

    #[test]
    fn lineage_is_carried_into_the_ledger_request() {
        let b = body(&format!(
            r#"{{"protocol":"{BRIDGE_PROTOCOL}","caller":{{"product":"formlogic"}},
                "flowId":"f","correlationId":"c","idempotencyKey":"k",
                "lineage":{{"rootRunId":"r","parentRunId":"p","bindingId":"b","depth":3}},
                "triggerEvent":"flow.succeeded"}}"#
        ))
        .expect("parse");
        let req = to_run_request(&b);
        // Without this the loop guards have nothing to work with, and a cycle
        // runs until the depth cap it can no longer see.
        assert_eq!(req.lineage.root_run_id.as_deref(), Some("r"));
        assert_eq!(req.lineage.binding_id.as_deref(), Some("b"));
        assert_eq!(req.lineage.depth, 3);
        assert_eq!(req.trigger_event.as_deref(), Some("flow.succeeded"));
    }

    #[test]
    fn a_request_without_lineage_is_a_direct_invocation() {
        let req = to_run_request(&valid());
        assert!(req.lineage.binding_id.is_none());
        assert_eq!(req.lineage.depth, 0);
    }

    #[test]
    fn every_rejection_explains_itself() {
        for r in [
            RequestRejection::ProtocolMismatch { got: "x".into() },
            RequestRejection::NoFlow,
            RequestRejection::BothFlowAndGraph,
            RequestRejection::EmptyIdempotencyKey,
            RequestRejection::UnknownMode { got: "x".into() },
        ] {
            let m = r.message();
            assert!(m.len() > 15, "too terse: {m:?}");
            assert!(!m.starts_with("error"), "lead with the problem: {m:?}");
        }
    }
}
