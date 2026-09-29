//! `POST /api/mcp`: MCP over Streamable HTTP, JSON-RPC 2.0, one request per
//! POST, answered as plain `application/json` (no server-sent stream).
//!
//! - `initialize`: the client's protocol version when it is one of
//!   [`PROTOCOL_VERSIONS`], else the newest; `serverInfo {name: "oaiy", version}`
//!   and `capabilities {tools: {}}`.
//! - `ping`, `tools/list`, `tools/call`.
//! - A notification (`notifications/initialized` and any other message without
//!   an id) is 202 with no body.
//! - A batch is refused (-32600); a body that is not JSON is -32700; an
//!   unknown method -32601; malformed params -32602. A tool that fails, or
//!   that does not exist, is a RESULT with `isError: true`, so the model reads
//!   what went wrong and how to fix it.

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Map, Value};

use super::{tools, Control, Session, SESSION_HEADER};

/// The protocol versions this server speaks, newest first.
pub const PROTOCOL_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;

fn error_body(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message.into() } })
}

fn reply(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

/// What the server tells a client when it starts: who is asking decides which tools there are.
const INSTRUCTIONS: &str = "OAIY's control tools: read how OAIY is set up (status first) and change it (models, engines, AI sources, services, plugins and their settings, flows, the calendar, the FormLogic link, setup). Which tools are offered depends on the X-OAIY-Session header: project and setup get all of them, runner only the read tools, call, sms and task none. A change tool refuses while the person has switched off 'Let the Agent change OAIY' (status shows agentMayChange); ask them to switch it on in Settings → Agent. Every change is logged for the person to see.";

pub(crate) async fn handle(State(control): State<Control>, headers: HeaderMap, body: Bytes) -> Response {
    let session = Session::from_header(headers.get(SESSION_HEADER).and_then(|v| v.to_str().ok()));
    let message: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return reply(StatusCode::BAD_REQUEST, error_body(Value::Null, PARSE_ERROR, format!("the body is not JSON: {e}"))),
    };
    if message.is_array() {
        return reply(
            StatusCode::BAD_REQUEST,
            error_body(Value::Null, INVALID_REQUEST, "batches are not accepted: send one JSON-RPC request per POST"),
        );
    }
    let Some(obj) = message.as_object() else {
        return reply(StatusCode::BAD_REQUEST, error_body(Value::Null, INVALID_REQUEST, "a JSON-RPC request is an object"));
    };
    // An id this server can echo: a string or a number.
    let id = obj.get("id").cloned();
    let usable_id = id.clone().filter(|i| i.is_string() || i.is_number());
    if obj.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return reply(StatusCode::BAD_REQUEST, error_body(usable_id.unwrap_or(Value::Null), INVALID_REQUEST, "jsonrpc must be \"2.0\""));
    }
    let method = obj.get("method").and_then(Value::as_str);
    let Some(id) = id else {
        // A notification (or a client's answer to a request of ours, which we never make): accepted, nothing to say.
        return StatusCode::ACCEPTED.into_response();
    };
    let Some(id) = usable_id else {
        return reply(StatusCode::BAD_REQUEST, error_body(Value::Null, INVALID_REQUEST, format!("the id must be a string or a number, not {id}")));
    };
    let Some(method) = method else {
        if obj.contains_key("result") || obj.contains_key("error") {
            return StatusCode::ACCEPTED.into_response();
        }
        return reply(StatusCode::BAD_REQUEST, error_body(id, INVALID_REQUEST, "a request needs a method"));
    };
    let body = match dispatch(&control, session, method, obj.get("params")).await {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err((code, message)) => error_body(id, code, message),
    };
    reply(StatusCode::OK, body)
}

/// `params` as an object: absent is empty, anything but an object is -32602.
fn params_object(params: Option<&Value>) -> Result<Map<String, Value>, (i64, String)> {
    match params {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(o)) => Ok(o.clone()),
        Some(other) => Err((INVALID_PARAMS, format!("params must be an object, not {other}"))),
    }
}

pub(crate) async fn dispatch(control: &Control, session: Session, method: &str, params: Option<&Value>) -> Result<Value, (i64, String)> {
    match method {
        "initialize" => {
            let p = params_object(params)?;
            let asked = match p.get("protocolVersion") {
                None => None,
                Some(Value::String(v)) => Some(v.as_str()),
                Some(other) => return Err((INVALID_PARAMS, format!("protocolVersion must be a string, not {other}"))),
            };
            let version = asked.filter(|v| PROTOCOL_VERSIONS.contains(v)).unwrap_or(PROTOCOL_VERSIONS[0]);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "oaiy", "title": "OAIY", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => {
            params_object(params)?;
            Ok(json!({}))
        }
        "tools/list" => {
            let p = params_object(params)?;
            if let Some(c) = p.get("cursor").filter(|c| !c.is_null() && !c.is_string()) {
                return Err((INVALID_PARAMS, format!("cursor must be a string, not {c}")));
            }
            // Every tool fits one page: there is no next cursor.
            Ok(json!({ "tools": tools::list(session) }))
        }
        "tools/call" => {
            let p = params_object(params)?;
            let name = match p.get("name") {
                Some(Value::String(n)) if !n.trim().is_empty() => n.trim().to_string(),
                Some(other) => return Err((INVALID_PARAMS, format!("name must be a tool's name, not {other}"))),
                None => return Err((INVALID_PARAMS, "tools/call needs the tool's name".into())),
            };
            let args = match p.get("arguments") {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(o)) => o.clone(),
                Some(other) => return Err((INVALID_PARAMS, format!("arguments must be an object, not {other}"))),
            };
            Ok(tools::call(control, session, &name, args).await)
        }
        other => Err((METHOD_NOT_FOUND, format!("method not found: {other}"))),
    }
}
