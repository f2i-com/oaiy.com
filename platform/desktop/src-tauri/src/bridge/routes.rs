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

#[derive(Clone)]
pub struct BridgeState {
    pub ledger: LedgerHandle,
    pub plugins: PluginRegistryHandle,
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
        correlation_id: body.correlation_id.clone(),
        idempotency_key: body.idempotency_key.clone(),
        lineage,
        trigger_event: body.trigger_event.clone(),
    }
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
    match ledger.reserve(&req) {
        ReserveOutcome::Reserved(rec) => (StatusCode::CREATED, Json(rec)).into_response(),
        // 200, not 201: nothing was created and nothing will execute. The flag is
        // what stops a caller treating a dedupe as a fresh run.
        ReserveOutcome::Duplicate(rec) => {
            let mut v = serde_json::to_value(&rec).unwrap_or_else(|_| json!({}));
            v["idempotent"] = json!(true);
            (StatusCode::OK, Json(v)).into_response()
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
    let reg = match st.plugins.lock() {
        Ok(r) => r,
        Err(_) => {
            return bridge_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
                "plugin registry lock poisoned".into(),
            )
        }
    };

    // The gate runs before anything is forwarded. An undeclared command must
    // never reach the plugin process.
    match reg.gate(&connector_id, &body.command, body.idempotency_key.as_deref()) {
        Ok(_rec) => {
            // The supervised process and its stdio are not wired yet, so this is
            // an honest typed refusal rather than a fabricated success. Returning
            // {ok:true} with an empty result would be the "quiet green light"
            // this whole protocol exists to prevent.
            bridge_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "capability_unavailable",
                format!(
                    "The \"{connector_id}\" plugin passed the capability gate, but this build \
                     cannot forward commands yet — the supervised plugin process is not wired up. \
                     Nothing was sent to the plugin."
                ),
            )
        }
        Err(refusal) => {
            let status = match refusal.code() {
                "capability_denied" => StatusCode::FORBIDDEN,
                "invalid_request" => StatusCode::BAD_REQUEST,
                _ => StatusCode::SERVICE_UNAVAILABLE,
            };
            bridge_error(status, refusal.code(), refusal.message())
        }
    }
}

/// The bridge router, ready to `.merge()` into the main app.
pub fn router(state: BridgeState) -> Router {
    Router::new()
        .route("/api/bridge/capabilities", get(capabilities))
        .route("/api/bridge/runs", get(queued_runs).post(create_run))
        .route("/api/bridge/runs/:id", get(get_run))
        .route("/api/bridge/runs/:id/claim", post(claim_run))
        .route("/api/bridge/runs/:id/cancel", post(cancel_run))
        .route("/api/plugins", get(list_plugins))
        .route(
            "/api/bridge/connectors/:id/request",
            post(connector_request),
        )
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
