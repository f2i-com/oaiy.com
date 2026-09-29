//! The rest of the engines relay the control tools need, beside the one in
//! `http.rs` (state, catalog, downloads): which model each group has chosen
//! and which it can choose from, choosing one, starting and stopping the
//! language model, and the engines' own log.
//!
//!   GET  /api/engines/defaults              → {running, defaults: {group: model|null}, models: {group: [id]}}
//!   PUT  /api/engines/defaults {group, model} → {group, model}
//!   POST /api/engines/llm/:action           → {llm: {state, resident, models, error}} (start | stop | restart)
//!   GET  /api/engines/logs?source=studio|llm|media&lines=100 → {source, lines}
//!
//! The engines' control port answers only its own origin, so the desktop asks
//! it (as `http.rs` does). Choosing a model is what the Engines page does:
//! its configuration read, `default_model` set, and written back, which the
//! engines check (a model that is not one of the group's is refused). Their
//! configuration holds the gateway key and the Hugging Face token, so it is
//! only ever passed back to them, never answered here.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::Control;
use crate::http::studio_json;

/// The engine's groups, as its discovery document names them.
pub const GROUPS: [&str; 9] = ["llm", "image", "video", "speech", "music", "sound", "model3d", "background", "upscale"];

/// The message when no engines are there to ask.
pub const NOT_RUNNING: &str = "The engines are not running: they start with OAIY (a headless server is told where they are with OAIY_ENGINES_UI).";

pub fn router(control: Control) -> Router {
    Router::new()
        .route("/api/engines/defaults", get(defaults).put(set_default))
        .route("/api/engines/llm/:action", post(llm))
        .route("/api/engines/logs", get(logs))
        .with_state(control)
}

fn answer(result: Result<Value, (u16, String)>) -> Response {
    match result {
        Ok(v) => Json(v).into_response(),
        Err((status, message)) => {
            (StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY), Json(json!({ "error": message }))).into_response()
        }
    }
}

/// A model name the engines' configuration could hold: short, no control characters.
fn model_name(model: &str) -> bool {
    (1..=128).contains(&model.len()) && !model.chars().any(char::is_control)
}

/// The models chosen per group and the models each group has, from the
/// engines' discovery document (ids only).
pub(crate) async fn defaults_at(ui: Option<String>) -> Value {
    let Some(ui) = ui else {
        return json!({ "running": false, "error": NOT_RUNNING });
    };
    let doc = match studio_json(&ui, reqwest::Method::GET, "/api/discovery", None).await {
        Ok(d) => d,
        Err((_, e)) => return json!({ "running": false, "error": e }),
    };
    let mut defaults = Map::new();
    let mut models = Map::new();
    for group in GROUPS {
        let chosen = doc
            .pointer(&format!("/defaults/{group}"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map_or(Value::Null, |m| Value::String(m.to_string()));
        defaults.insert(group.into(), chosen);
        let ids: Vec<Value> = doc
            .pointer(&format!("/models/{group}"))
            .and_then(Value::as_array)
            .map(|list| list.iter().filter_map(|m| m.get("id").and_then(Value::as_str)).map(|s| Value::String(s.to_string())).collect())
            .unwrap_or_default();
        models.insert(group.into(), Value::Array(ids));
    }
    json!({ "running": true, "defaults": defaults, "models": models })
}

async fn defaults(State(control): State<Control>) -> Response {
    Json(defaults_at(control.engines_ui()).await).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DefaultBody {
    group: String,
    model: String,
}

/// Choose `model` for `group` in the engines at `ui`.
pub(crate) async fn set_default_at(ui: Option<String>, group: &str, model: &str) -> Result<Value, (u16, String)> {
    let section: &[&str] = match group {
        "llm" => &["llm"],
        "image" | "video" | "speech" | "music" | "sound" | "model3d" => &["media", group],
        "background" | "upscale" => {
            return Err((400, format!("{group} has one model, used as soon as it is downloaded: there is nothing to choose")));
        }
        other => return Err((400, format!("{other:?} is not an engine group: one of {}", GROUPS.join(", ")))),
    };
    let model = model.trim();
    if !model_name(model) {
        return Err((400, "a model is its name in the engines, as models_list shows it".into()));
    }
    let ui = ui.ok_or((409, NOT_RUNNING.to_string()))?;
    let mut config = studio_json(&ui, reqwest::Method::GET, "/api/config", None).await?;
    let mut at = &mut config;
    for key in section {
        at = at
            .get_mut(*key)
            .filter(|v| v.is_object())
            .ok_or_else(|| (502, format!("the engines' configuration has no {} section", section.join("."))))?;
    }
    at["default_model"] = Value::String(model.to_string());
    // The engines check it: a model that is not one of the group's is refused with why.
    studio_json(&ui, reqwest::Method::PUT, "/api/config", Some(config)).await?;
    Ok(json!({ "group": group, "model": model }))
}

async fn set_default(State(control): State<Control>, Json(body): Json<DefaultBody>) -> Response {
    answer(set_default_at(control.engines_ui(), body.group.trim(), &body.model).await)
}

/// Start, stop or restart the language model at `ui`.
pub(crate) async fn llm_at(ui: Option<String>, action: &str) -> Result<Value, (u16, String)> {
    if !matches!(action, "start" | "stop" | "restart") {
        return Err((404, format!("the language model can start, stop or restart, not {action:?}")));
    }
    let ui = ui.ok_or((409, NOT_RUNNING.to_string()))?;
    let status = studio_json(&ui, reqwest::Method::POST, &format!("/api/llm/{action}"), None).await?;
    Ok(json!({
        "llm": {
            "state": status.get("state"),
            "resident": status.get("resident"),
            "models": status.get("models"),
            "error": status.get("error"),
        }
    }))
}

async fn llm(State(control): State<Control>, Path(action): Path<String>) -> Response {
    answer(llm_at(control.engines_ui(), &action).await)
}

#[derive(Deserialize)]
struct LogsQuery {
    source: Option<String>,
    lines: Option<usize>,
}

/// The engines' own log (`studio`), or the language model's (`llm`) or the media worker's (`media`).
pub(crate) async fn logs_at(ui: Option<String>, source: &str, lines: usize) -> Result<Value, (u16, String)> {
    let query = match source {
        "studio" | "engines" => "",
        "llm" => "?source=llm",
        "media" => "?source=media",
        other => return Err((400, format!("the engines' logs are studio, llm or media, not {other:?}"))),
    };
    let ui = ui.ok_or((409, NOT_RUNNING.to_string()))?;
    let v = studio_json(&ui, reqwest::Method::GET, &format!("/api/logs{query}"), None).await?;
    let all: Vec<Value> = v
        .get("lines")
        .and_then(Value::as_array)
        .map(|l| l.iter().filter_map(|x| x.get("line").and_then(Value::as_str).or_else(|| x.as_str())).map(|s| Value::String(s.to_string())).collect())
        .unwrap_or_default();
    let tail = all[all.len().saturating_sub(lines)..].to_vec();
    Ok(json!({ "source": source, "lines": tail }))
}

async fn logs(State(control): State<Control>, Query(q): Query<LogsQuery>) -> Response {
    let lines = q.lines.unwrap_or(100).clamp(1, 1000);
    answer(logs_at(control.engines_ui(), q.source.as_deref().unwrap_or("studio"), lines).await)
}
