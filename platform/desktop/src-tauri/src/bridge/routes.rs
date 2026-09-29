//! HTTP surface for the Bridge Protocol v1.
//!
//! Mounted alongside the existing `/api/*` routes as its own `Router` with its
//! own state, so `http.rs` only has to merge it.
//!
//! | Route | Does |
//! |---|---|
//! | `GET  /api/bridge/capabilities` | what this runtime can do right now |
//! | `POST /api/bridge/runs` | reserve a run (idempotent) |
//! | `GET  /api/bridge/runs` | claimable runs oldest first; `?status=` for history, newest first |
//! | `GET  /api/bridge/runs/:id` | one run |
//! | `POST /api/bridge/runs/:id/claim` | single-winner claim |
//! | `POST /api/bridge/runs/:id/cancel` | request cancellation |
//! | `GET  /api/bridge/deadletters` | events that produced no work |
//! | `POST /api/bridge/deadletters/:id/redrive` | re-dispatch one |
//! | `GET  /api/plugins` | installed plugins + state + reason + package trust |
//! | `POST /api/plugins/:id/trust` | trust this exact unsigned package (privileged; takes only the id) |
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

use super::deadletters::DeadLetterHandle;
use super::ledger::{
    ClaimOutcome, LedgerHandle, LineageRef, ReserveOutcome, RunRequest, RunStatus, Runtime,
};
use crate::http::BRIDGE_PROTOCOL;
use crate::plugins::registry::PluginRegistryHandle;
use crate::plugins::{CallError, ForwardError, CONNECTOR_TIMEOUT};

#[derive(Clone)]
pub struct BridgeState {
    pub ledger: LedgerHandle,
    /// Events that arrived and produced no work.
    pub dead: DeadLetterHandle,
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
    /// The Node runtime the bundled CLI runs under — reported by readiness so a
    /// missing runtime is distinguishable from a missing CLI.
    pub node: Option<crate::services::node_runtime::NodeHandle>,
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

#[derive(Debug, Deserialize)]
struct RunsQuery {
    /// Comma-separated statuses, or `all`. Absent keeps the historical
    /// queued-only behaviour that pollers depend on.
    status: Option<String>,
    limit: Option<usize>,
}

/// The most rows one request will return, whatever `limit` asks for. The ledger
/// holds 20k runs and every record carries its input and output — serialising
/// the lot in one response is a memory spike, not a feature.
const MAX_RUNS_PAGE: usize = 200;

/// Parse `status=failed,timed_out` into the enum. An unrecognised name is an
/// error rather than an empty filter: silently returning "no runs" for a typo
/// reads exactly like "nothing failed", which is the one answer that must never
/// be wrong here.
fn parse_statuses(raw: &str) -> Result<Vec<RunStatus>, String> {
    if raw.trim().eq_ignore_ascii_case("all") {
        return Ok(Vec::new());
    }
    let names: Vec<&str> = raw.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    // `status=` or `status=,,,` named nothing. An empty Vec means "no filter"
    // downstream, so accepting it would answer a request for SOME runs with ALL
    // of them — a screen titled "Failed runs" listing successes. That is the
    // same silent-wrongness as an unknown name, pointing the other way.
    if names.is_empty() {
        return Err("status must name at least one state, or be `all`".into());
    }
    names
        .into_iter()
        .map(|s| match s.to_ascii_lowercase().as_str() {
            "queued" => Ok(RunStatus::Queued),
            "running" => Ok(RunStatus::Running),
            "succeeded" => Ok(RunStatus::Succeeded),
            "failed" => Ok(RunStatus::Failed),
            "timed_out" => Ok(RunStatus::TimedOut),
            "cancelled" => Ok(RunStatus::Cancelled),
            other => Err(format!("unknown status `{other}`")),
        })
        .collect()
}

/// `GET /api/bridge/runs` — queued runs by default (what a worker polls), or
/// history when `status` is given.
/// Forget finished runs. Queued and running work is kept — see
/// [`crate::bridge::ledger::Ledger::clear_history`].
async fn clear_runs(State(st): State<BridgeState>) -> axum::response::Response {
    match st.ledger.lock() {
        Ok(mut l) => {
            let cleared = l.clear_history();
            (StatusCode::OK, Json(json!({ "cleared": cleared, "total": l.len() }))).into_response()
        }
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "the run ledger lock is poisoned".to_string(),
        ),
    }
}

async fn queued_runs(
    State(st): State<BridgeState>,
    axum::extract::Query(q): axum::extract::Query<RunsQuery>,
) -> axum::response::Response {
    let limit = q.limit.unwrap_or(100).min(MAX_RUNS_PAGE);
    let statuses = match q.status.as_deref().map(parse_statuses).transpose() {
        Ok(s) => s,
        Err(msg) => return bridge_error(StatusCode::BAD_REQUEST, "invalid_request", msg),
    };

    match st.ledger.lock() {
        Ok(l) => {
            // No filter asked for: the original contract, oldest first, because a
            // claimer wants the longest-waiting run — not the newest.
            let (runs, counts) = match statuses {
                None => (l.queued(limit), None),
                Some(s) => (l.recent(limit, &s), Some(l.status_counts())),
            };
            let mut body = json!({ "runs": runs, "total": l.len() });
            if let Some(counts) = counts {
                // Keyed by the same snake_case names the filter accepts.
                let by_status: serde_json::Map<String, serde_json::Value> = counts
                    .into_iter()
                    .filter_map(|(k, v)| {
                        let key = serde_json::to_value(k).ok()?;
                        Some((key.as_str()?.to_string(), json!(v)))
                    })
                    .collect();
                body["byStatus"] = serde_json::Value::Object(by_status);
            }
            (StatusCode::OK, Json(body)).into_response()
        }
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
            // A plugin dropped in (or its manifest changed) can bring or take a module.
            crate::modules::refresh(&reg);
            crate::modules::poke();
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

/// Map a connector-forwarding failure onto the bridge error taxonomy.
///
/// Shared by the raw connector route and the service-action route so the two
/// cannot drift — a plugin failure must read identically however it was reached.
fn forward_error_response(e: ForwardError) -> axum::response::Response {
    match e {
        ForwardError::Refused(refusal) => {
            let status = match refusal.code() {
                "capability_denied" => StatusCode::FORBIDDEN,
                "invalid_request" => StatusCode::BAD_REQUEST,
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            bridge_error(status, refusal.code(), refusal.message())
        }
        ForwardError::NotRunning { plugin_id } => bridge_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "capability_unavailable",
            format!(
                "The {plugin_id} plugin stopped between the gate check and the call.                  Start it in OAIY Desktop → Plugins."
            ),
        ),
        ForwardError::Call(CallError::Timeout { method, waited }) => bridge_error(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            format!("the plugin did not answer {method} within {:.0}s", waited.as_secs_f32()),
        ),
        ForwardError::Call(CallError::Plugin { message, typed, .. }) => {
            // The plugin's typed code passes through only if it is IN the closed
            // taxonomy. A plugin is untrusted, and `error.code` is the field a
            // consumer branches on — echoing an arbitrary string lets a plugin
            // emit `capability_denied` (a verdict the host never made) or a code
            // outside the enum. Anything unrecognised becomes node_failed.
            let code = match typed.as_deref() {
                Some(t) if is_taxonomy_code(t) => t,
                _ => "node_failed",
            };
            bridge_error(StatusCode::BAD_GATEWAY, code, message)
        }
        ForwardError::Call(e) => {
            bridge_error(StatusCode::BAD_GATEWAY, "runtime_unavailable", e.to_string())
        }
        ForwardError::Internal(m) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", m),
    }
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
        Ok(Err(e)) => forward_error_response(e),
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

/// A success body for control routes, instead of 204 No Content.
///
/// A bare 204 is correct HTTP and a trap for browser clients: a fetch wrapper
/// that parses every reply as JSON sees an empty body, fails to parse, and
/// reports a transport error — so a start that WORKED is shown to the user as a
/// failure. A tiny body costs nothing and removes the whole class of bug.
fn ok_body() -> axum::response::Response {
    (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response()
}

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
        Ok(Ok(())) => ok_body(),
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
            // Turned off, its modules go now (and their leases with them); turned on, they are back.
            crate::modules::refresh(&reg);
            crate::modules::poke();
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

// ------- runtime readiness -------

/// `GET /api/bridge/status` — is this runtime actually able to run a flow?
///
/// `/api/health` deliberately answers a different question: it asserts identity
/// ("this really is OAIY Desktop") and is open so discovery works. It says
/// nothing about READINESS, so the only way to learn the flow runtime was
/// unusable was to submit a run and get `runtime_unavailable` back. This reports
/// the things that actually stop work: whether the CLI resolves, how deep the
/// queue is, and which plugins are not serving.
///
/// A restricted read — it names plugins and queue state, which is not for an
/// arbitrary remote page.
async fn runtime_status(State(st): State<BridgeState>) -> axum::response::Response {
    // Same resolution a real run performs, so this cannot report ready while a
    // run would fail (or the reverse).
    let cli = crate::bridge::worker::cli_status();
    let cli_kind = match &cli {
        Some(crate::bridge::worker::CliInvocation::Node { .. }) => "node",
        Some(crate::bridge::worker::CliInvocation::Binary { .. }) => "binary",
        None => "missing",
    };
    // And what that CLI RUNS flows on. Read from the probe's cache only — a
    // probe is a child process that hashes and compiles the engine, and this
    // handler is polled every few seconds by the UI. A cold cache is reported
    // as `unknown` and filled on a thread of its own (the worker also fills it
    // at start), so the next poll has the answer.
    let node_exe = st.node.as_ref().and_then(|n| n.resolve());
    let engine = cli
        .as_ref()
        .map(|c| crate::bridge::worker::cached_engine_probe(c, node_exe.as_deref()));
    if let (Some(c), Some(None)) = (&cli, &engine) {
        crate::bridge::worker::warm_engine_probe(c.clone(), node_exe.clone());
    }
    let engine_ready = matches!(engine, Some(Some(crate::bridge::worker::EngineProbe::Ready(_))));
    let engine_json = match &engine {
        Some(Some(crate::bridge::worker::EngineProbe::Ready(id))) => json!({
            "name": id.name, "release": id.release, "status": "ready",
        }),
        Some(Some(crate::bridge::worker::EngineProbe::Unavailable { reason })) => json!({
            "name": serde_json::Value::Null, "release": serde_json::Value::Null,
            "status": "unavailable", "reason": reason,
        }),
        // No CLI at all, or a cache still cold: nothing is known yet.
        _ => json!({
            "name": serde_json::Value::Null, "release": serde_json::Value::Null, "status": "unknown",
        }),
    };

    // `failed` is carried here so Overview can point at the run history without
    // paging it — a machine where flows are quietly failing should say so on the
    // first screen rather than only on the one you have to think to open.
    let (queued, total_runs, failed_runs) = match st.ledger.lock() {
        Ok(l) => {
            let counts = l.status_counts();
            let failed = counts.get(&RunStatus::Failed).copied().unwrap_or(0)
                + counts.get(&RunStatus::TimedOut).copied().unwrap_or(0);
            (
                counts.get(&RunStatus::Queued).copied().unwrap_or(0),
                l.len(),
                failed,
            )
        }
        Err(_) => (0, 0, 0),
    };

    let plugins: Vec<serde_json::Value> = match st.plugins.lock() {
        Ok(mut reg) => {
            reg.scan();
            reg.list()
                .into_iter()
                .map(|r| {
                    json!({ "id": r.id, "state": r.state, "reason": r.reason,
                            "servesCommands": r.state.accepts_commands() })
                })
                .collect()
        }
        Err(_) => Vec::new(),
    };
    let plugins_serving = plugins
        .iter()
        .filter(|p| p.get("servesCommands").and_then(serde_json::Value::as_bool) == Some(true))
        .count();

    // The bundled CLI is a Node script, so a resolved CLI is not enough: without
    // a Node runtime the spawn fails and the run dies with a confusing error.
    let node = st.node.as_ref().map(|n| n.snapshot());
    let node_ok = node.as_ref().map(|n| n.available).unwrap_or(true);
    // Ready means a run would work: a CLI, a Node to start it with, and that
    // CLI answering that it runs flows on ZIPP. A resolved CLI whose engine is
    // missing fails every run `runtime_unavailable`; saying "ready" over that
    // sends the user hunting for a link problem.
    //
    // The probe's `Ready` answer is not kept for ever on this claim's account:
    // a RUN that comes back `Unavailable` replaces it (worker.rs,
    // `remember_run_refusal`), so an engine that broke under an unchanged
    // `oaiy.mjs` — a quarantined or half-replaced wasm — turns this to
    // `unavailable` on the next poll rather than at the next restart.
    let ready = cli.is_some() && node_ok && engine_ready;
    (
        StatusCode::OK,
        Json(json!({
            "ready": ready,
            "deviceId": st.device_id,
            "flowRuntime": {
                "cliResolved": cli.is_some(),
                "cliKind": cli_kind,
                "engine": engine_json,
                // The fix, in the response, so a caller does not have to go
                // looking for what "not resolved" means.
                // Name the ACTUAL blocker: "no CLI", "no Node to run it with"
                // and "a CLI without its engine" need different fixes, and
                // conflating them sends the user the wrong way.
                "detail": if cli.is_none() {
                    Some("Install the `oaiy` CLI so it is on PATH, or set OAIY_CLI to the path of cli/bin/oaiy.mjs.".to_string())
                } else if !node_ok {
                    Some("A Node runtime is required to run the bundled CLI — install it from OAIY Desktop, or put node on PATH.".to_string())
                } else if let Some(Some(crate::bridge::worker::EngineProbe::Unavailable { reason })) = &engine {
                    Some(format!(
                        "the OAIY CLI does not run user logic on ZIPP: {reason}. {}",
                        crate::bridge::worker::engine_fix_hint()
                    ))
                } else if !engine_ready {
                    Some("checking which engine the OAIY CLI runs flows on…".to_string())
                } else {
                    None
                },
            },
            "nodeRuntime": node,
            "runs": { "queued": queued, "known": total_runs, "failed": failed_runs },
            "plugins": { "serving": plugins_serving, "total": plugins.len(), "detail": plugins },
        })),
    )
        .into_response()
}

// ------- plugin-contributed service definitions -------

/// Every service definition contributed by an installed plugin, with provenance.
fn collect_definitions(st: &BridgeState) -> Vec<crate::plugins::definitions::ServiceDefinition> {
    let Ok(mut reg) = st.plugins.lock() else { return Vec::new() };
    // Discovery must not depend on call order (the same reason capabilities
    // scans): a fresh process should list definitions on the first request.
    reg.scan();
    let mut out = Vec::new();
    for rec in reg.list() {
        // A plugin that is still running when its folder stops verifying keeps its
        // manifest (its process was started from it), but the definition files are read
        // from that folder now, and a swapped one could point an action at another of
        // the plugin's commands. Nothing is offered from a package that does not verify.
        if rec.refused_by_trust() {
            continue;
        }
        if let Some(m) = rec.manifest.as_ref() {
            out.extend(crate::plugins::definitions::load_for_plugin(&rec.dir, m));
        }
    }
    out
}

/// `GET /api/services/definitions` — the invocable action surfaces plugins add.
async fn list_service_definitions(State(st): State<BridgeState>) -> axum::response::Response {
    let defs = collect_definitions(&st);
    (StatusCode::OK, Json(json!({ "definitions": defs }))).into_response()
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct InvokeBody {
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    idempotency_key: Option<String>,
}

/// `POST /api/services/actions/:definition_id/:action_id/invoke`
///
/// Resolves the action to the plugin connector command it declares and forwards
/// it down the ORDINARY gated path — same capability check, same journalling
/// rule. A definition is a documented facade, not a way around the gate.
async fn invoke_service_action(
    State(st): State<BridgeState>,
    Path((definition_id, action_id)): Path<(String, String)>,
    body: Option<Json<InvokeBody>>,
) -> axum::response::Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let defs = collect_definitions(&st);
    let Some(def) = defs.iter().find(|d| d.id == definition_id) else {
        return bridge_error(
            StatusCode::NOT_FOUND,
            "invalid_request",
            format!("no service definition {definition_id:?}"),
        );
    };
    let Some(action) = def.action(&action_id) else {
        return bridge_error(
            StatusCode::NOT_FOUND,
            "invalid_request",
            format!("{definition_id:?} has no action {action_id:?}"),
        );
    };
    if action.transport.kind != "plugin-command" {
        return bridge_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("action {action_id:?} uses an unsupported transport {:?}", action.transport.kind),
        );
    }
    let Some(command) = action.transport.command.clone() else {
        return bridge_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            format!("action {action_id:?} declares no command"),
        );
    };

    let host = st.host.clone();
    let connector_id = def.plugin_id.clone();
    let payload = body.input;
    let key = body.idempotency_key;
    let timeout = action
        .timeout_ms
        .map(std::time::Duration::from_millis)
        .unwrap_or(CONNECTOR_TIMEOUT);
    let result = tokio::task::spawn_blocking(move || {
        host.forward_connector(&connector_id, &command, payload, key.as_deref(), timeout)
    })
    .await;

    match result {
        Ok(Ok(value)) => (StatusCode::OK, Json(json!({ "ok": true, "result": value }))).into_response(),
        Ok(Err(e)) => forward_error_response(e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

// ------- plugin install / uninstall -------

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InstallBody {
    /// A path ON THIS MACHINE: a plugin directory, or a `.tar.gz` of one.
    /// Deliberately not a URL — this route installs native code, and fetching
    /// that from the network would make it a remote-code-install primitive.
    source: String,
}

/// `POST /api/plugins/install` — install (or replace) a plugin from a local path.
///
/// PRIVILEGED: installing a plugin means installing native code that the host
/// will supervise, so a paired web page must never reach this — only OAIY's own
/// window or a token holder (see `is_ai_exec_path`'s sibling classification in
/// http.rs).
async fn install_plugin(
    State(st): State<BridgeState>,
    Json(body): Json<InstallBody>,
) -> axum::response::Response {
    let source = std::path::PathBuf::from(body.source.trim());
    if source.as_os_str().is_empty() {
        return bridge_error(StatusCode::BAD_REQUEST, "invalid_request", "a source path is required".into());
    }
    let (root, trust) = match st.plugins.lock() {
        Ok(reg) => (reg.root().to_path_buf(), reg.trust()),
        Err(_) => {
            return bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "plugin registry lock poisoned".into())
        }
    };

    // A running plugin holds its executable open, so replacing it would fail
    // mid-way. Stop it first if this install targets an id we already run.
    let host = st.host.clone();
    let installed = tokio::task::spawn_blocking(move || {
        // Peek at the id before copying, so we only stop what we are replacing.
        if let Ok(id) = crate::plugins::install::peek_id(&source) {
            let _ = host.stop(&id);
        }
        crate::plugins::install::install_from_path(&source, &root, &trust)
    })
    .await;

    match installed {
        Ok(Ok(out)) => {
            // Pick the new plugin up immediately rather than on the next poll.
            let mut setup = None;
            if let Ok(mut reg) = st.plugins.lock() {
                reg.scan();
                crate::modules::refresh(&reg);
                setup = reg.get(&out.id).and_then(crate::setup::declared).cloned();
            }
            crate::modules::poke();
            (StatusCode::OK, Json(install_reply(&out, setup.as_ref()))).into_response()
        }
        Ok(Err(e)) => bridge_error(StatusCode::BAD_REQUEST, "invalid_request", e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

/// What an install answers: the plugin installed, and when it declares a
/// setup, its version and title, so the window that installed it can open
/// its setup wizard (or, for an update, nudge when the version went up).
/// And what its package was found to be, so the window can say a plugin that
/// will not start until it is trusted is not broken.
fn install_reply(out: &crate::plugins::install::Installed, setup: Option<&crate::plugins::manifest::SetupDecl>) -> serde_json::Value {
    let mut reply = json!({
        "id": out.id,
        "name": out.name,
        "version": out.version,
        "replaced": out.replaced,
        "trust": out.trust,
    });
    if let Some(setup) = setup {
        reply["setup"] = json!({ "version": setup.version, "title": setup.title });
    }
    reply
}

/// `DELETE /api/plugins/:id` — stop a plugin and remove it from disk. PRIVILEGED.
async fn uninstall_plugin(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    let root = match st.plugins.lock() {
        Ok(reg) => reg.root().to_path_buf(),
        Err(_) => {
            return bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "plugin registry lock poisoned".into())
        }
    };
    let host = st.host.clone();
    let id2 = id.clone();
    let removed = tokio::task::spawn_blocking(move || {
        // Stop first: on Windows the running executable pins its own directory.
        let _ = host.stop(&id2);
        crate::plugins::install::uninstall(&id2, &root)
    })
    .await;

    match removed {
        Ok(Ok(())) => {
            if let Ok(mut reg) = st.plugins.lock() {
                reg.forget(&id);
                reg.scan();
                crate::modules::refresh(&reg);
            }
            crate::modules::poke();
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(Err(e)) => bridge_error(StatusCode::NOT_FOUND, "invalid_request", e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

/// `POST /api/plugins/:id/trust` — trust this exact, unsigned package. PRIVILEGED.
///
/// The one thing a person can do about a plugin a release build holds back because it
/// has no signature. It takes the plugin's id and nothing else: the folder is the one the
/// registry already holds for that id, and no path, URL or body from a caller is read,
/// so it cannot be pointed at something the person did not install. What it records is a
/// digest of every file in that folder (see `plugins::trust`): any change ends the trust.
///
/// A package that carries a signature is refused (409): it is verified or quarantined by
/// that signature, and no click turns a failed one into a good one.
async fn trust_plugin(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    let (dir, trust) = {
        let mut reg = match st.plugins.lock() {
            Ok(r) => r,
            Err(_) => {
                return bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "plugin registry lock poisoned".into())
            }
        };
        reg.scan();
        let Some(rec) = reg.get(&id) else {
            return bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no plugin named {id:?}"));
        };
        if rec.trust.is_none() {
            // Its manifest could not be loaded, so there is no plugin to trust.
            return bridge_error(
                StatusCode::CONFLICT,
                "invalid_request",
                format!("{id} cannot be trusted: {}", rec.reason.clone().unwrap_or_else(|| "its manifest is invalid".into())),
            );
        }
        (rec.dir.clone(), reg.trust())
    };

    // Reads and hashes every file of the package: off the async threads, and with no
    // registry lock held.
    let plugin_id = id.clone();
    let trusted = tokio::task::spawn_blocking(move || trust.trust_local(&dir, &plugin_id)).await;
    match trusted {
        Ok(Ok(_)) => match st.plugins.lock() {
            Ok(mut reg) => {
                // Brought in now, not on the next poll: it can be started.
                reg.scan();
                crate::modules::refresh(&reg);
                crate::modules::poke();
                match reg.get(&id) {
                    Some(rec) => (StatusCode::OK, Json(rec.clone())).into_response(),
                    None => bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no plugin named {id:?}")),
                }
            }
            Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "plugin registry lock poisoned".into()),
        },
        Ok(Err(e)) => bridge_error(StatusCode::CONFLICT, "invalid_request", e),
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

// ------- plugin-contributed UI -------

/// Content type for a plugin UI asset, by extension. Deliberately a small
/// allow-list: anything else is served as an opaque download rather than
/// something the webview will execute.
fn ui_content_type(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        _ => "application/octet-stream",
    }
}

/// `GET /api/plugins/:id/ui/:screen/*path` — serve one plugin-shipped UI asset.
///
/// A plugin may contribute screens (`manifest.ui.screens[]`): static HTML/CSS/JS
/// that the desktop hosts in an iframe, so a plugin can ship its own interface
/// (the Aokie plugin's "AI Receptionist", for example) instead of being limited
/// to the generic Plugins card.
///
/// Only files the screen DECLARES in its `files` list are served. That, not path
/// arithmetic, is the real guard: a traversal string simply won't be in the
/// allow-list, and neither will the plugin's own executables, DLLs or manifest.
/// The path is additionally rejected if it is absolute or contains `..`.
async fn plugin_ui_asset(
    State(st): State<BridgeState>,
    Path((id, screen, asset)): Path<(String, String, String)>,
) -> axum::response::Response {
    // Cheap structural rejects before touching the registry.
    let asset = asset.replace('\\', "/");
    if asset.starts_with('/') || asset.split('/').any(|seg| seg == ".." || seg == ".") {
        return bridge_error(StatusCode::BAD_REQUEST, "invalid_request", "invalid asset path".into());
    }

    let (dir, declared, trust, signed) = {
        let mut reg = match st.plugins.lock() {
            Ok(r) => r,
            Err(_) => {
                return bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "plugin registry lock poisoned".into())
            }
        };
        // The registry is populated lazily by scan(). Scan only on a MISS: an
        // iframe pulls a dozen assets per screen and rescanning each time would
        // stat the whole plugins tree over and over — but a screen must still
        // load in a fresh process that has not listed plugins yet, rather than
        // depending on which route the UI happened to call first.
        if reg.get(&id).is_none() {
            reg.scan();
        }
        let Some(rec) = reg.get(&id) else {
            return bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no plugin {id:?}"));
        };
        // A screen is code the dashboard runs, from a folder on disk. A plugin whose
        // package does not verify (one that was running when its folder changed keeps its
        // manifest, and so still names its screens) serves none of it: what it says of
        // itself is what the listing says, and the two must not disagree.
        if rec.refused_by_trust() {
            let why = rec.trust.as_ref().and_then(|t| t.reason.clone()).unwrap_or_else(|| "its package is not trusted".into());
            return bridge_error(StatusCode::FORBIDDEN, "capability_unavailable", format!("the screens of {id:?} are not served: {why}"));
        }
        let signed = rec.trust.as_ref().is_some_and(|t| t.state == crate::plugins::TrustState::Verified);
        // The declared file list for THIS screen, from the manifest's `ui` block
        // (retained verbatim in `extra`, since the host has no other opinion on it).
        let files: Vec<String> = rec
            .manifest
            .as_ref()
            .and_then(|m| m.extra.get("ui"))
            .and_then(|ui| ui.get("screens"))
            .and_then(|s| s.as_array())
            .and_then(|screens| {
                screens
                    .iter()
                    .find(|s| s.get("id").and_then(|v| v.as_str()) == Some(screen.as_str()))
            })
            .and_then(|s| s.get("files"))
            .and_then(|f| f.as_array())
            .map(|f| f.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        (rec.dir.clone(), files, reg.trust(), signed)
    };

    if declared.is_empty() {
        return bridge_error(
            StatusCode::NOT_FOUND,
            "invalid_request",
            format!("plugin {id:?} declares no screen {screen:?}"),
        );
    }
    if !declared.iter().any(|f| f.replace('\\', "/") == asset) {
        return bridge_error(
            StatusCode::FORBIDDEN,
            "capability_denied",
            format!("{asset:?} is not declared by screen {screen:?}"),
        );
    }

    // Read once, off the async threads. From a package that carries a signature the file
    // is served only if it is exactly what was signed: the scan's answer can be a few
    // seconds old, and a file swapped since then (with its size and time put back) is
    // not one it could tell. A package nobody signed has nothing to compare with; it was
    // let through above by its own verdict (a developer build, or the person's trust).
    let served = {
        let (dir, asset, id) = (dir.clone(), asset.clone(), id.clone());
        tokio::task::spawn_blocking(move || match trust.read_signed_file(&dir, &id, &asset) {
            Ok(Some(bytes)) => Ok(bytes),
            Ok(None) if signed => Err((StatusCode::FORBIDDEN, format!("the signature of {id:?} is gone, so {asset:?} cannot be checked"))),
            Ok(None) => std::fs::read(dir.join(&asset)).map_err(|e| (StatusCode::NOT_FOUND, format!("cannot read {asset:?}: {e}"))),
            Err(why) => Err((StatusCode::FORBIDDEN, why)),
        })
        .await
    };
    match served {
        Ok(Ok(bytes)) => (
            StatusCode::OK,
            [
                (axum::http::header::CONTENT_TYPE, ui_content_type(&asset)),
                // Plugin assets change when the plugin is updated, never mid-run.
                (axum::http::header::CACHE_CONTROL, "no-cache"),
            ],
            bytes,
        )
            .into_response(),
        Ok(Err((status, why))) => {
            let code = if status == StatusCode::FORBIDDEN { "capability_unavailable" } else { "invalid_request" };
            bridge_error(status, code, why)
        }
        Err(e) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()),
    }
}

// ------- triggers -------

#[derive(Debug, Deserialize)]
struct DeadQuery {
    limit: Option<usize>,
}

async fn list_dead_letters(
    State(st): State<BridgeState>,
    axum::extract::Query(q): axum::extract::Query<DeadQuery>,
) -> axum::response::Response {
    match st.dead.lock() {
        Ok(d) => (
            StatusCode::OK,
            Json(json!({ "deadLetters": d.list(q.limit.unwrap_or(100).min(200)), "total": d.len() })),
        )
            .into_response(),
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "dead-letter queue lock poisoned".into(),
        ),
    }
}

/// Re-dispatch one dead letter against the current bindings.
///
/// `200` either way — the interesting part is `reserved`, because a redrive that
/// legitimately fails again (the guard still refuses, the binding is still
/// broken) is not a request error. Reporting it as one would have callers
/// retrying a thing that will never work.
///
/// On a blocking thread: a redrive is a full dispatch, and a dispatch decides
/// its conditions on ZIPP — which takes a process-wide lock and, if no child is
/// up, starts one. That is seconds of blocking, and it must not sit on a tokio
/// worker.
async fn redrive_dead_letter(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
) -> axum::response::Response {
    let host = st.host.clone();
    let wanted = id.clone();
    let redriven = match tokio::task::spawn_blocking(move || host.redrive(&id)).await {
        Ok(r) => r,
        Err(e) => {
            return bridge_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                format!("the redrive did not run: {e}"),
            )
        }
    };
    match redriven {
        Some((outcomes, reserved)) => (
            StatusCode::OK,
            Json(json!({ "reserved": reserved, "outcomes": outcomes })),
        )
            .into_response(),
        None => bridge_error(
            StatusCode::NOT_FOUND,
            "invalid_request",
            format!("unknown dead letter {wanted}"),
        ),
    }
}

async fn delete_dead_letter(
    State(st): State<BridgeState>,
    Path(id): Path<String>,
) -> axum::response::Response {
    match st.dead.lock() {
        Ok(mut d) => {
            if d.remove(&id) {
                StatusCode::NO_CONTENT.into_response()
            } else {
                bridge_error(
                    StatusCode::NOT_FOUND,
                    "invalid_request",
                    format!("unknown dead letter {id}"),
                )
            }
        }
        Err(_) => bridge_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "dead-letter queue lock poisoned".into(),
        ),
    }
}

async fn list_triggers(State(st): State<BridgeState>) -> axum::response::Response {
    match st.host.triggers.lock() {
        Ok(t) => (StatusCode::OK, Json(json!({ "bindings": t.list() }))).into_response(),
        Err(_) => bridge_error(StatusCode::INTERNAL_SERVER_ERROR, "internal", "trigger store lock poisoned".into()),
    }
}

/// Save a trigger binding, checking its condition first.
///
/// The condition is JavaScript, and the only thing that can honestly say
/// whether it parses is the engine that will run it. That engine is a child
/// process behind a process-wide lock, so it is asked on a blocking thread and
/// BEFORE `st.host.triggers` is locked — holding the store's lock across a
/// script host that may be starting would stall `GET /api/bridge/triggers`,
/// whose whole job is to answer immediately.
///
/// An expression the engine refuses is a 400, not a saved binding with a
/// warning: a binding that cannot be evaluated looks correct in the list and is
/// incapable of ever firing, which is the trigger bug that is hardest to find.
/// Refusing it puts the error where the mistake is. An engine that could not be
/// REACHED is different — that is our problem, not the author's — so the binding
/// is saved and the answer says it went unchecked. Nothing unsafe follows:
/// dispatch decides every condition on ZIPP too, and fails closed.
async fn upsert_trigger(
    State(st): State<BridgeState>,
    Json(binding): Json<crate::bridge::triggers::TriggerBinding>,
) -> axum::response::Response {
    use crate::bridge::conditions::CheckOutcome;

    let mut warning: Option<String> = None;
    if let Some(expr) = crate::bridge::triggers::condition_of(&binding).map(str::to_string) {
        let scripts = st.host.script_evaluator();
        let source = expr.clone();
        let checked = tokio::task::spawn_blocking(move || {
            crate::bridge::conditions::check(scripts.as_ref(), &source)
        })
        .await
        .unwrap_or_else(|e| CheckOutcome::Unchecked(format!("the check did not run: {e}")));
        match checked {
            CheckOutcome::Parses => {}
            CheckOutcome::Rejected(why) => {
                return bridge_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("condition cannot be evaluated ({why}): {expr}"),
                )
            }
            CheckOutcome::Unchecked(why) => {
                warning = Some(format!("condition not checked: {why}"))
            }
        }
    }

    match st.host.triggers.lock() {
        Ok(mut t) => match t.upsert(binding) {
            Ok(()) => {
                let mut body = json!({ "bindings": t.list() });
                if let Some(warning) = warning {
                    body["warning"] = serde_json::Value::String(warning);
                }
                (StatusCode::OK, Json(body)).into_response()
            }
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

/// A stored flow, as it was put (the agent reads a tool flow's inputs from it).
async fn get_flow(State(st): State<BridgeState>, Path(id): Path<String>) -> axum::response::Response {
    match st.flows.get(&id) {
        Some(body) => ([(axum::http::header::CONTENT_TYPE, "application/json")], body).into_response(),
        None => bridge_error(StatusCode::NOT_FOUND, "invalid_request", format!("no flow {id:?}")),
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
            Ok(()) => ok_body(),
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
        .route("/api/bridge/leases/:name", post(super::leases::take_lease))
        .route("/api/bridge/status", get(runtime_status))
        .route(
            "/api/bridge/runs",
            get(queued_runs).post(create_run).delete(clear_runs),
        )
        .route("/api/bridge/runs/:id", get(get_run))
        .route("/api/bridge/runs/:id/claim", post(claim_run))
        .route("/api/bridge/runs/:id/finish", post(finish_run))
        .route("/api/bridge/runs/:id/cancel", post(cancel_run))
        .route("/api/bridge/deadletters", get(list_dead_letters))
        .route(
            "/api/bridge/deadletters/:id",
            axum::routing::delete(delete_dead_letter),
        )
        .route("/api/bridge/deadletters/:id/redrive", post(redrive_dead_letter))
        .route("/api/plugins", get(list_plugins))
        .route(
            "/api/bridge/connectors/:id/request",
            post(connector_request),
        )
        .route("/api/plugins/:id/start", post(start_plugin))
        .route("/api/plugins/:id/stop", post(stop_plugin))
        .route("/api/plugins/:id/logs", get(plugin_logs))
        // Install / remove a plugin. Privileged (see http.rs): this installs
        // native code the host will supervise.
        .route("/api/plugins/install", post(install_plugin))
        .route("/api/plugins/:id", axum::routing::delete(uninstall_plugin))
        // Trust this exact unsigned package. Privileged, and takes only the id.
        .route("/api/plugins/:id/trust", post(trust_plugin))
        // Service definitions a plugin contributes (its invocable action surface).
        .route("/api/services/definitions", get(list_service_definitions))
        .route(
            "/api/services/actions/:definition_id/:action_id/invoke",
            post(invoke_service_action),
        )
        // Plugin-contributed UI: static assets the desktop hosts in an iframe.
        // Open GET (an iframe navigation cannot carry a bearer), but restricted
        // to the files the screen declares — see plugin_ui_asset.
        .route("/api/plugins/:id/ui/:screen/*path", get(plugin_ui_asset))
        .route("/api/plugins/:id/enabled", post(set_plugin_enabled))
        .route("/api/bridge/triggers", get(list_triggers).post(upsert_trigger))
        .route("/api/bridge/triggers/:id", axum::routing::delete(delete_trigger))
        .route("/api/bridge/events", get(poll_events))
        .route("/api/bridge/flows", get(list_flows))
        .route(
            "/api/bridge/flows/:id",
            axum::routing::put(put_flow).delete(delete_flow).get(get_flow),
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
    fn an_install_says_the_setup_version_when_the_plugin_declares_one() {
        let out = crate::plugins::install::Installed {
            id: "aokie".into(),
            name: "Aokie Phone Bridge".into(),
            version: "0.1.0".into(),
            dir: std::path::PathBuf::from("plugins/aokie"),
            replaced: false,
            trust: crate::plugins::PackageTrust {
                state: crate::plugins::TrustState::Verified,
                publisher: Some("Aokie".into()),
                key_id: Some("fl-aokie-2026a".into()),
                version: Some("0.1.0".into()),
                reason: None,
                trusted_at: None,
            },
        };
        let plain = install_reply(&out, None);
        assert_eq!(
            plain,
            json!({
                "id": "aokie", "name": "Aokie Phone Bridge", "version": "0.1.0", "replaced": false,
                "trust": { "state": "verified", "publisher": "Aokie", "keyId": "fl-aokie-2026a", "version": "0.1.0" },
            })
        );
        let setup = crate::plugins::manifest::SetupDecl { version: 2, title: "Set up the AI Receptionist".into(), steps: vec![] };
        let with = install_reply(&out, Some(&setup));
        assert_eq!(with["setup"], json!({ "version": 2, "title": "Set up the AI Receptionist" }));
        assert!(with.get("dir").is_none(), "the folder (and the OS username in it) stays on the desktop");
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
    fn the_history_filter_accepts_the_protocols_own_status_names() {
        assert_eq!(parse_statuses("failed").unwrap(), vec![RunStatus::Failed]);
        assert_eq!(
            parse_statuses("failed, timed_out ,cancelled").unwrap(),
            vec![RunStatus::Failed, RunStatus::TimedOut, RunStatus::Cancelled]
        );
        // `all` is the explicit "no filter", distinct from omitting the param
        // (which keeps the queued-only default a worker polls).
        assert!(parse_statuses("all").unwrap().is_empty());
    }

    #[test]
    fn a_filter_that_names_nothing_is_an_error_not_everything() {
        // An empty Vec means "no filter" downstream, so accepting `status=`
        // would answer a request for SOME runs with ALL of them — the panel
        // headed "Failed runs" listing successes. Asking for nothing and
        // getting everything is the worst possible reading.
        assert!(parse_statuses("").is_err());
        assert!(parse_statuses("   ").is_err());
        assert!(parse_statuses(",,,").is_err());
        assert!(parse_statuses(" , , ").is_err());
        // `all` remains the explicit, deliberate way to say "no filter".
        assert!(parse_statuses("all").unwrap().is_empty());
    }

    #[test]
    fn an_unknown_status_is_an_error_not_an_empty_result() {
        // A typo must not answer "no runs failed" — that is indistinguishable
        // from good news, and it is the one thing this view exists to report.
        let err = parse_statuses("failedd").unwrap_err();
        assert!(err.contains("failedd"), "{err}");
        assert!(parse_statuses("succeeded,nonsense").is_err());
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
    // --- saving a trigger: the condition is checked by the engine ----------

    mod saving_a_trigger {
        use super::super::*;
        use crate::bridge::conditions::testing::{guest, FakeHost};
        use crate::bridge::script_host::{HostError, ScriptBatch};
        use serde_json::Value;
        use std::sync::Arc;

        struct Sandbox(std::path::PathBuf);
        impl Sandbox {
            fn new(tag: &str) -> Self {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                let p = std::env::temp_dir()
                    .join(format!("oaiy-routes-{tag}-{}-{n}", std::process::id()));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
        }
        impl Drop for Sandbox {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        fn state(tag: &str, evaluator: Arc<dyn ScriptBatch>) -> (Sandbox, BridgeState) {
            let sb = Sandbox::new(tag);
            let st = crate::build_bridge_state(
                sb.0.join("plugins"),
                sb.0.clone(),
                "device".into(),
                None,
            );
            st.host.set_script_evaluator(evaluator);
            (sb, st)
        }

        fn binding(condition: Option<&str>) -> crate::bridge::triggers::TriggerBinding {
            crate::bridge::triggers::TriggerBinding {
                id: "b1".into(),
                event: "aokie.call.incoming".into(),
                flow_id: "f".into(),
                mode: crate::bridge::triggers::BindingMode::Async,
                enabled: true,
                condition: condition.map(str::to_string),
                input_map: Default::default(),
                sort_order: 0,
            }
        }

        async fn read(response: axum::response::Response) -> (StatusCode, Value) {
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
        }

        #[tokio::test]
        async fn a_condition_the_engine_refuses_is_a_400_and_is_not_saved() {
            // A binding whose condition will never parse looks correct in the
            // list and is incapable of ever firing — the trigger bug that is
            // hardest to find. Refusing it where the mistake was made is the
            // whole point; saving it with a warning is not.
            let refusing =
                Arc::new(FakeHost::by_source(|_| Err(guest("expected one expression, and \";\" follows it"))));
            let (_sb, st) = state("refused", refusing);

            let (status, body) =
                read(upsert_trigger(State(st.clone()), Json(binding(Some("1); x = (1")))).await).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
            assert_eq!(body["error"]["code"], "invalid_request");
            assert!(
                body["error"]["message"].as_str().unwrap().contains("expected one expression"),
                "the author needs the engine's own reason: {body}"
            );
            assert!(st.host.triggers.lock().unwrap().list().is_empty(), "nothing was saved");
        }

        #[tokio::test]
        async fn an_engine_that_cannot_be_reached_saves_and_says_so() {
            // Not the author's fault, so not the author's 400. Saving is safe:
            // dispatch decides every condition on ZIPP too, and fails closed, so
            // an unchecked condition still cannot fire.
            let down = Arc::new(FakeHost::down("the CLI is not installed"));
            let (_sb, st) = state("unchecked", down);

            let (status, body) = read(
                upsert_trigger(State(st.clone()), Json(binding(Some("event.data.x === 1")))).await,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            let warning = body["warning"].as_str().expect("a warning: {body}");
            assert!(warning.starts_with("condition not checked:"), "{warning}");
            assert!(warning.contains("the CLI is not installed"), "{warning}");
            assert_eq!(st.host.triggers.lock().unwrap().list().len(), 1, "and it was saved");
        }

        #[tokio::test]
        async fn a_condition_the_engine_compiles_saves_without_a_warning() {
            let ok = Arc::new(FakeHost::always(Value::Null));
            let (_sb, st) = state("clean", ok);
            let (status, body) = read(
                upsert_trigger(State(st.clone()), Json(binding(Some("event.data.x === 1")))).await,
            )
            .await;
            assert_eq!(status, StatusCode::OK);
            assert!(body.get("warning").is_none(), "nothing went wrong: {body}");
            assert_eq!(body["bindings"].as_array().unwrap().len(), 1);
        }

        #[tokio::test]
        async fn a_binding_with_no_condition_never_asks_the_engine() {
            let ok = Arc::new(FakeHost::always(Value::Null));
            let (_sb, st) = state("nocond", ok.clone());
            let (status, _) = read(upsert_trigger(State(st.clone()), Json(binding(None))).await).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(ok.calls(), 0);
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
        async fn the_trigger_store_is_not_locked_while_the_engine_is_asked() {
            // `evaluate` takes a process-wide lock and may spawn a child. Holding
            // the trigger store's lock across that would stall every read of the
            // binding list behind an engine that is starting.
            let slow = Arc::new(FakeHost::raw(|_| {
                std::thread::sleep(std::time::Duration::from_millis(1500));
                Err(HostError::Unavailable { reason: "slow".into(), retry_after: None })
            }));
            let (_sb, st) = state("unlocked", slow);

            let saving = {
                let st = st.clone();
                tokio::spawn(async move {
                    upsert_trigger(State(st), Json(binding(Some("event.data.x === 1")))).await
                })
            };
            // Give the save a moment to get as far as the check.
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;

            let started = std::time::Instant::now();
            let (status, _) = read(list_triggers(State(st.clone())).await).await;
            assert_eq!(status, StatusCode::OK);
            assert!(
                started.elapsed() < std::time::Duration::from_millis(500),
                "listing waited {:?} — the store was locked across the engine call",
                started.elapsed()
            );
            saving.await.unwrap();
        }
    }

    // --- trusting a package nobody signed ------------------------------------

    mod trusting_a_plugin {
        use super::super::*;
        use crate::plugins::registry::PluginRegistry;
        use crate::plugins::trust::tests::{fill, TestKey};
        use crate::plugins::trust::{Publishers, TrustPolicy, TrustService};
        use axum::body::Body;
        use axum::http::{Method, Request};
        use serde_json::Value;
        use std::path::PathBuf;
        use tower::ServiceExt as _;

        pub(super) struct Sandbox(pub(super) PathBuf);
        impl Sandbox {
            fn new(tag: &str) -> Self {
                use std::sync::atomic::{AtomicU32, Ordering};
                static N: AtomicU32 = AtomicU32::new(0);
                let n = N.fetch_add(1, Ordering::Relaxed);
                let p = std::env::temp_dir().join(format!("oaiy-routes-trust-{tag}-{}-{n}", std::process::id()));
                let _ = std::fs::remove_dir_all(&p);
                std::fs::create_dir_all(&p).unwrap();
                Self(p)
            }
        }
        impl Drop for Sandbox {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        /// The bridge over a registry that holds a release build's rules. The host does not
        /// start plugins at boot (as `build_bridge_state`'s would), so a plugin the test puts
        /// on disk is started by the test and by nothing else.
        pub(super) fn release_state(tag: &str, publishers: Publishers) -> (Sandbox, BridgeState) {
            state_under(tag, TrustPolicy::release(), publishers)
        }

        /// The same under `policy`.
        pub(super) fn state_under(tag: &str, policy: TrustPolicy, publishers: Publishers) -> (Sandbox, BridgeState) {
            let sb = Sandbox::new(tag);
            let root = sb.0.join("plugins");
            std::fs::create_dir_all(&root).unwrap();
            let trust = TrustService::new(policy, publishers, root.join("trusted-plugins.json"));
            let plugins: PluginRegistryHandle = std::sync::Arc::new(std::sync::Mutex::new(PluginRegistry::with_trust(root, trust)));
            let ledger = crate::bridge::ledger::new_handle();
            let dead = crate::bridge::deadletters::open_handle(sb.0.join("deadletters.jsonl"));
            let triggers: crate::plugins::TriggerStoreHandle =
                std::sync::Arc::new(std::sync::Mutex::new(crate::plugins::TriggerStore::load(sb.0.join("triggers.json"))));
            let host = crate::plugins::PluginHost::assemble(plugins.clone(), ledger.clone(), triggers, dead.clone(), "0.0.0-test".into(), true);
            let st = BridgeState {
                ledger,
                dead,
                plugins,
                host,
                flows: std::sync::Arc::new(crate::bridge::worker::FlowStore::new(sb.0.join("flows"))),
                pairing: crate::bridge::pairing::open_handle(sb.0.join("pairings.json")),
                device_id: "device".into(),
                node: None,
            };
            (sb, st)
        }

        /// An unsigned plugin the demo fixtures make, under `plugins/<id>`, with the id in its manifest.
        fn unsigned_plugin(sb: &Sandbox, id: &str) -> PathBuf {
            let dir = sb.0.join("plugins").join(id);
            std::fs::create_dir_all(&dir).unwrap();
            fill(&dir);
            let manifest = std::fs::read_to_string(dir.join("manifest.json")).unwrap().replace("\"id\":\"demo\"", &format!("\"id\":\"{id}\""));
            std::fs::write(dir.join("manifest.json"), manifest).unwrap();
            dir
        }

        async fn read(response: axum::response::Response) -> (StatusCode, Value) {
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
        }

        fn trust_of(st: &BridgeState, id: &str) -> Option<crate::plugins::TrustState> {
            let mut reg = st.plugins.lock().unwrap();
            reg.scan();
            reg.get(id).and_then(|r| r.trust.as_ref().map(|t| t.state))
        }

        #[tokio::test]
        async fn trusting_an_unsigned_plugin_lets_it_be_started_and_answers_with_the_record() {
            let (sb, st) = release_state("trust-ok", Publishers::default());
            unsigned_plugin(&sb, "demo");
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::Unsigned));
            assert!(st.host.start("demo").unwrap_err().contains("Not signed"), "held back until trusted");

            let (status, body) = read(trust_plugin(State(st.clone()), Path("demo".into())).await).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["trust"]["state"], "trusted-local");
            assert_eq!(body["state"], "installed", "brought in now: it can be started");
            assert_eq!(body["manifest"]["id"], "demo");
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::TrustedLocal));
            // Started, it is past the trust check (the stub cannot run, so it is the launch that fails).
            let err = st.host.start("demo").unwrap_err();
            assert!(!err.contains("Not signed") && !err.contains("was not started"), "{err}");
        }

        #[tokio::test]
        async fn an_unknown_signed_or_broken_plugin_cannot_be_trusted() {
            let (sb, st) = release_state("trust-refused", Publishers::default());
            // Nothing by that name.
            let (status, _) = read(trust_plugin(State(st.clone()), Path("ghost".into())).await).await;
            assert_eq!(status, StatusCode::NOT_FOUND);
            // A path in the id's place is just an id nobody has.
            for id in ["..", "../demo", "C:\\plugins\\demo", "demo/../demo"] {
                let (status, _) = read(trust_plugin(State(st.clone()), Path(id.into())).await).await;
                assert_eq!(status, StatusCode::NOT_FOUND, "{id}");
            }
            // One that carries a signature is judged by it.
            let dir = unsigned_plugin(&sb, "signed");
            TestKey::generate("k").sign(&dir, "signed-plugin", "1.0.0");
            let (status, body) = read(trust_plugin(State(st.clone()), Path("signed".into())).await).await;
            assert_eq!(status, StatusCode::CONFLICT, "{body}");
            assert!(body["error"]["message"].as_str().unwrap().contains("cannot be trusted by hand"), "{body}");
            assert_eq!(trust_of(&st, "signed"), Some(crate::plugins::TrustState::Quarantined));
            // One with a manifest that does not load has no plugin to trust.
            let broken = sb.0.join("plugins").join("broken");
            std::fs::create_dir_all(&broken).unwrap();
            std::fs::write(broken.join("manifest.json"), b"{ not json").unwrap();
            let (status, body) = read(trust_plugin(State(st.clone()), Path("broken".into())).await).await;
            assert_eq!(status, StatusCode::CONFLICT, "{body}");
        }

        async fn send(app: &Router, path: &str, headers: &[(&str, &str)], body: Option<Value>) -> (StatusCode, Value) {
            let mut req = Request::builder().method(Method::POST).uri(path);
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            if body.is_some() {
                req = req.header("content-type", "application/json");
            }
            read(app.clone().oneshot(req.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap()).await.unwrap()).await
        }

        #[tokio::test]
        async fn the_route_is_closed_to_a_stranger_and_open_to_the_token_and_the_window() {
            let (sb, st) = release_state("trust-gate", Publishers::default());
            unsigned_plugin(&sb, "demo");

            // A headless server: the token or nothing.
            let headless = crate::http::guarded_for_tests(router(st.clone()), Some("desk-token".into()), false);
            for headers in [vec![], vec![("Authorization", "Bearer wrong")], vec![("Origin", "tauri://localhost")]] {
                let (status, _) = send(&headless, "/api/plugins/demo/trust", &headers, None).await;
                assert_eq!(status, StatusCode::FORBIDDEN, "{headers:?}");
            }
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::Unsigned), "nothing was trusted");

            // The desktop's window: its own origin, and not a web page or a caller with no origin.
            let gui = crate::http::guarded_for_tests(router(st.clone()), None, true);
            for headers in [vec![("Origin", "https://evil.example")], vec![("Origin", "null")], vec![]] {
                let (status, _) = send(&gui, "/api/plugins/demo/trust", &headers, None).await;
                assert_eq!(status, StatusCode::FORBIDDEN, "{headers:?}");
            }
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::Unsigned));

            let (status, body) = send(&gui, "/api/plugins/demo/trust", &[("Origin", "tauri://localhost")], None).await;
            assert_eq!((status, body["trust"]["state"].as_str()), (StatusCode::OK, Some("trusted-local")), "{body}");
        }

        #[tokio::test]
        async fn the_route_reads_no_path_or_url_from_the_caller() {
            // Two unsigned plugins; the caller names one in the URL and tries to smuggle
            // the other, and a folder of its own, in the body. Only the named plugin's
            // own folder is ever looked at.
            let (sb, st) = release_state("trust-id-only", Publishers::default());
            unsigned_plugin(&sb, "demo");
            let other = unsigned_plugin(&sb, "other");
            let elsewhere = sb.0.join("elsewhere");
            std::fs::create_dir_all(&elsewhere).unwrap();
            fill(&elsewhere);

            let app = crate::http::guarded_for_tests(router(st.clone()), Some("desk-token".into()), false);
            let body = json!({
                "id": "other", "dir": other.display().to_string(), "path": elsewhere.display().to_string(),
                "source": "https://example.com/plugin.zip", "url": "https://example.com/plugin.zip", "digest": "sha256:00",
            });
            let (status, reply) = send(&app, "/api/plugins/demo/trust", &[("Authorization", "Bearer desk-token")], Some(body)).await;
            assert_eq!(status, StatusCode::OK, "{reply}");
            assert_eq!(reply["id"], "demo");
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::TrustedLocal));
            assert_eq!(trust_of(&st, "other"), Some(crate::plugins::TrustState::Unsigned), "the id in the body was not read");
            let trusted = std::fs::read_to_string(sb.0.join("plugins").join("trusted-plugins.json")).unwrap();
            assert!(trusted.contains("\"demo\"") && !trusted.contains("\"other\"") && !trusted.contains("elsewhere"), "{trusted}");
        }

        #[tokio::test]
        async fn uninstalling_a_plugin_takes_the_persons_trust_with_it() {
            let (sb, st) = release_state("trust-uninstall", Publishers::default());
            unsigned_plugin(&sb, "demo");
            read(trust_plugin(State(st.clone()), Path("demo".into())).await).await;
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::TrustedLocal));

            let response = uninstall_plugin(State(st.clone()), Path("demo".into())).await;
            assert_eq!(response.status(), StatusCode::NO_CONTENT);
            // The same bytes installed again are a new decision.
            unsigned_plugin(&sb, "demo");
            assert_eq!(trust_of(&st, "demo"), Some(crate::plugins::TrustState::Unsigned));
        }

        #[tokio::test]
        async fn the_listing_carries_each_plugins_trust() {
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("trust-list", key.pinned_for("Demo Co", &["demo"]));
            let dir = unsigned_plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");
            unsigned_plugin(&sb, "loose");

            let (status, body) = read(list_plugins(State(st.clone())).await).await;
            assert_eq!(status, StatusCode::OK);
            let plugins = body["plugins"].as_array().unwrap();
            let by_id = |id: &str| plugins.iter().find(|p| p["id"] == id).unwrap().clone();
            assert_eq!(by_id("demo")["trust"]["state"], "verified");
            assert_eq!(by_id("demo")["trust"]["publisher"], "Demo Co");
            assert_eq!(by_id("loose")["trust"]["state"], "unsigned");
            assert!(by_id("loose")["trust"]["reason"].as_str().unwrap().contains("trust this exact package"));
            assert_eq!(by_id("loose")["state"], "disabled");
            assert!(by_id("loose").get("manifest").is_none(), "a held-back plugin offers no manifest to build on");
        }
    }

    // --- what a package's screens and definitions are served from ----------------

    mod screens_and_definitions_under_trust {
        use super::super::*;
        use super::trusting_a_plugin::{release_state, state_under, Sandbox};
        use crate::plugins::registry::PluginState;
        use crate::plugins::trust::tests::{fill, TestKey};
        use crate::plugins::trust::{Publishers, TrustPolicy};
        use serde_json::Value;
        use std::path::PathBuf;

        /// The page `fill` writes, and another of the same length.
        const SIGNED_PAGE: &str = "<p>hello</p>";
        const SWAPPED_PAGE: &str = "<p>evil!</p>";

        /// Plugin `id`, with a screen `main` that declares `ui/index.html`, and the service
        /// definition `definitions/phone.json` that maps its one action to a command.
        fn plugin(sb: &Sandbox, id: &str) -> PathBuf {
            let dir = sb.0.join("plugins").join(id);
            std::fs::create_dir_all(dir.join("definitions")).unwrap();
            fill(&dir);
            let mut manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
            manifest["id"] = json!(id);
            manifest["ui"] = json!({ "screens": [{ "id": "main", "title": "Main", "files": ["ui/index.html"] }] });
            manifest["serviceDefinitions"] = json!([{ "definitionFile": "definitions/phone.json" }]);
            std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
            std::fs::write(
                dir.join("definitions").join("phone.json"),
                json!({
                    "id": format!("{id}.phone"), "name": "Phone",
                    "actions": [{ "id": "call.dial", "transport": { "kind": "plugin-command", "command": "call.dial" } }],
                })
                .to_string(),
            )
            .unwrap();
            dir
        }

        async fn get(st: &BridgeState, id: &str, asset: &str) -> (StatusCode, String) {
            let response = plugin_ui_asset(State(st.clone()), Path((id.to_string(), "main".to_string(), asset.to_string()))).await;
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            (status, String::from_utf8_lossy(&bytes).to_string())
        }

        fn rescan(st: &BridgeState) {
            st.plugins.lock().unwrap().scan();
        }

        /// Rewrite a file with other bytes of the same length and put its modified time back.
        fn swap_keeping_size_and_time(path: &std::path::Path, bytes: &[u8]) {
            let before = std::fs::metadata(path).unwrap().modified().unwrap();
            assert_eq!(std::fs::metadata(path).unwrap().len(), bytes.len() as u64);
            std::fs::write(path, bytes).unwrap();
            std::fs::File::options().write(true).open(path).unwrap().set_modified(before).unwrap();
        }

        #[tokio::test]
        async fn a_verified_plugins_screen_is_served_exactly_as_signed() {
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("screen-ok", key.pinned_for("Demo Co", &["demo"]));
            let dir = plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");

            assert_eq!(get(&st, "demo", "ui/index.html").await, (StatusCode::OK, SIGNED_PAGE.to_string()));
            // Only what the screen declares, as before.
            assert_eq!(get(&st, "demo", "demo-plugin.exe").await.0, StatusCode::FORBIDDEN);
        }

        #[tokio::test]
        async fn a_screen_file_swapped_behind_the_scans_fingerprint_is_not_served() {
            // Same length, modified time put back: the scan's stat fingerprint cannot tell,
            // so the record still says verified. The file is checked against the signature
            // on the bytes that are served.
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("screen-swapped", key.pinned_for("Demo Co", &["demo"]));
            let dir = plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");
            assert_eq!(get(&st, "demo", "ui/index.html").await.0, StatusCode::OK);

            swap_keeping_size_and_time(&dir.join("ui").join("index.html"), SWAPPED_PAGE.as_bytes());
            rescan(&st);
            let listed = st.plugins.lock().unwrap().get("demo").unwrap().trust.as_ref().unwrap().state;
            assert_eq!(listed, crate::plugins::TrustState::Verified, "the listing cannot tell");

            let (status, body) = get(&st, "demo", "ui/index.html").await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
            assert!(!body.contains("evil"), "the swapped page was served: {body}");
            assert!(body.contains("ui/index.html is not what was signed"), "{body}");
        }

        #[tokio::test]
        async fn a_running_plugin_whose_folder_stopped_verifying_serves_no_screen() {
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("screen-quarantined", key.pinned_for("Demo Co", &["demo"]));
            let dir = plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");
            assert_eq!(get(&st, "demo", "ui/index.html").await.0, StatusCode::OK);

            // It is running, and its screen file is replaced (and a file added).
            st.plugins.lock().unwrap().set_state("demo", PluginState::Running, None);
            std::fs::write(dir.join("ui").join("index.html"), "<script>/* tampered */</script>").unwrap();
            std::fs::write(dir.join("note.txt"), "x").unwrap();
            rescan(&st);
            let rec = st.plugins.lock().unwrap().get("demo").cloned().unwrap();
            assert_eq!(rec.state, PluginState::Running, "the live process is left alone");
            assert_eq!(rec.trust.as_ref().unwrap().state, crate::plugins::TrustState::Quarantined);
            assert!(rec.manifest.is_some(), "and it still names its screens");

            let response = plugin_ui_asset(State(st.clone()), Path(("demo".into(), "main".into(), "ui/index.html".into()))).await;
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20).await.unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(status, StatusCode::FORBIDDEN);
            assert_eq!(body["error"]["code"], "capability_unavailable");
            let message = body["error"]["message"].as_str().unwrap();
            assert!(message.contains("not served") && message.contains("Quarantined"), "{message}");
            assert!(!String::from_utf8_lossy(&bytes).contains("tampered"));
        }

        #[tokio::test]
        async fn a_signature_that_disappears_serves_nothing() {
            // The record still says verified until the next scan; the signature is gone, so
            // nothing can be checked, and a swapped page would otherwise go out unchecked.
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("screen-unsigned-after", key.pinned_for("Demo Co", &["demo"]));
            let dir = plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");
            assert_eq!(get(&st, "demo", "ui/index.html").await.0, StatusCode::OK);

            std::fs::remove_file(dir.join("package-manifest.json")).unwrap();
            let (status, body) = get(&st, "demo", "ui/index.html").await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
            assert!(body.contains("signature") && body.contains("gone"), "{body}");
        }

        #[tokio::test]
        async fn an_unsigned_plugin_serves_its_screens_in_a_developer_build_as_before() {
            let (sb, st) = state_under("screen-dev", TrustPolicy::developer(), Publishers::default());
            plugin(&sb, "demo");
            assert_eq!(get(&st, "demo", "ui/index.html").await, (StatusCode::OK, SIGNED_PAGE.to_string()));
        }

        #[tokio::test]
        async fn a_plugin_a_release_build_holds_back_serves_no_screen() {
            let (sb, st) = release_state("screen-held", Publishers::default());
            plugin(&sb, "demo");
            let (status, body) = get(&st, "demo", "ui/index.html").await;
            assert_eq!(status, StatusCode::FORBIDDEN, "{body}");
            assert!(body.contains("not served") && body.contains("Not signed"), "it says why, as the listing does: {body}");
        }

        #[tokio::test]
        async fn service_definitions_are_not_offered_from_a_package_that_stopped_verifying() {
            let key = TestKey::generate("fl-test-2026a");
            let (sb, st) = release_state("defs", key.pinned_for("Demo Co", &["demo"]));
            let dir = plugin(&sb, "demo");
            key.sign(&dir, "demo-plugin", "1.0.0");
            assert_eq!(collect_definitions(&st).len(), 1, "a verified plugin offers its definition");

            // Running, with a definition file that no longer says what was signed.
            st.plugins.lock().unwrap().set_state("demo", PluginState::Running, None);
            std::fs::write(
                dir.join("definitions").join("phone.json"),
                json!({ "id": "demo.phone", "name": "Phone", "actions": [{ "id": "call.dial", "transport": { "kind": "plugin-command", "command": "sms.send" } }] })
                    .to_string(),
            )
            .unwrap();
            rescan(&st);
            assert!(st.plugins.lock().unwrap().get("demo").unwrap().manifest.is_some());
            assert!(collect_definitions(&st).is_empty(), "nothing is offered from a package that does not verify");
        }
    }
}
