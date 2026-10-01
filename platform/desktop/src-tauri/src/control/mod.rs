//! OAIY's control API: the MCP server the Agent configures OAIY with.
//!
//!   POST /api/mcp                          → MCP over Streamable HTTP: JSON-RPC 2.0, answered as plain JSON
//!   GET  /api/mcp                          → 405 (there is no server-sent stream)
//!   GET  /api/control/settings             → {agentMayChange}
//!   PUT  /api/control/settings             {agentMayChange}
//!   GET  /api/control/log?limit=100        → {entries}: the change tools called, newest first
//!   GET  /api/control/desktop-log?lines=200 → {available, lines}: the end of the desktop's own log
//!
//! and, beside the engines relay in `http.rs` (see [`engines`]):
//!   GET  /api/engines/defaults             → the model chosen per group, and the models each group has
//!   PUT  /api/engines/defaults {group, model}
//!   POST /api/engines/llm/:action          → start | stop | restart the language model
//!   GET  /api/engines/logs?source=&lines=  → the engines' own log lines
//!
//! # How a tool runs
//!
//! Every tool calls the desktop's EXISTING routes, in-process: the finished
//! router (the one `http::serve` listens with, gate and all) takes the request
//! with `oneshot`, carrying this process's own bearer ([`crate::internal_token`]).
//! So a tool inherits each route's validation, gating, journalling and module
//! checks, and nothing here keeps a second copy of any of it.
//!
//! # Who may use which tools
//!
//! The Agent app says who is asking with `X-OAIY-Session`: `project` and
//! `setup` get every tool; `runner` (a flow's agent) only the read tools;
//! `call`, `sms` and `task` none, so a caller on the phone cannot talk the
//! receptionist into reconfiguring OAIY. No header counts as `project` (another
//! MCP client holding the desktop token); a value this OAIY does not know gets
//! nothing.
//!
//! # The switch and the log
//!
//! `<data>/control.json` holds `{"agentMayChange": true}` (the default on a local
//! install; off on a proxied or lan one, where nobody has said yet: design
//! 4.5.5). While it is false the change tools refuse and the read tools still
//! work. Every
//! change-tool call, refused or not, is one line of `<data>/control-log.jsonl`
//! (see [`audit`]); read tools are not logged.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

pub mod audit;
pub mod engines;
mod mcp;
mod tools;
#[cfg(test)]
mod tests;

/// The switch's file, in the data folder.
pub const SETTINGS_FILE: &str = "control.json";
/// The audit log's file, in the data folder.
pub const LOG_FILE: &str = "control-log.jsonl";
/// The header the Agent app says who is asking with.
pub const SESSION_HEADER: &str = "x-oaiy-session";
/// The Tauri event the dashboard navigates on: `{view, pluginId?, stepId?, contact?}`
/// (`contact`: with view `contacts`, the key of the person to open).
pub const NAVIGATE_EVENT: &str = "oaiy://navigate";
/// What a change tool answers while the switch is off.
pub const SWITCHED_OFF: &str = "The Agent may not change OAIY: switch it on in Settings → Agent";

/// Shows a page in the dashboard (the GUI emits [`NAVIGATE_EVENT`]); none on a headless server.
pub type Navigator = Arc<dyn Fn(Value) -> Result<(), String> + Send + Sync>;

static NAVIGATOR: OnceLock<Navigator> = OnceLock::new();

/// The GUI's way to show a page in the dashboard. Set once, at startup.
pub fn set_navigator(navigator: Navigator) {
    let _ = NAVIGATOR.set(navigator);
}

/// Who is asking, from `X-OAIY-Session`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Session {
    Project,
    Setup,
    Runner,
    Call,
    Sms,
    Task,
    /// A value this OAIY does not know: it gets nothing.
    Other,
}

impl Session {
    /// No header is `project`; a header is matched without regard to case or
    /// surrounding space, and anything else (an empty one included) is `Other`.
    pub fn from_header(value: Option<&str>) -> Session {
        let Some(value) = value else { return Session::Project };
        match value.trim().to_ascii_lowercase().as_str() {
            "project" => Session::Project,
            "setup" => Session::Setup,
            "runner" => Session::Runner,
            "call" => Session::Call,
            "sms" => Session::Sms,
            "task" => Session::Task,
            _ => Session::Other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Session::Project => "project",
            Session::Setup => "setup",
            Session::Runner => "runner",
            Session::Call => "call",
            Session::Sms => "sms",
            Session::Task => "task",
            Session::Other => "other",
        }
    }

    /// The read tools: `project`, `setup` and `runner`.
    pub fn may_read(self) -> bool {
        matches!(self, Session::Project | Session::Setup | Session::Runner)
    }

    /// The change tools: `project` and `setup` only.
    pub fn may_change(self) -> bool {
        matches!(self, Session::Project | Session::Setup)
    }
}

/// The control API's state: the switch, the audit log, the router the tools
/// call, and the way to show a page.
#[derive(Clone)]
pub struct Control {
    inner: Arc<Inner>,
}

struct Inner {
    data_dir: PathBuf,
    /// What the switch is when the person has not said: on for a local install, off for one that can be reached
    /// through a proxy or from a network (design 4.5.5).
    default_on: bool,
    /// One writer of `control.json` at a time.
    settings_lock: Mutex<()>,
    audit: audit::Log,
    /// The finished router, set once `http::serve` has built it.
    router: OnceLock<Router>,
    /// A navigator of this state's own (tests); otherwise the process's.
    navigator: Option<Navigator>,
    /// Where the engines are, for this state alone (tests: none, so a test
    /// never reaches engines another test started); otherwise the process's.
    engines: Option<Option<String>>,
}

impl Control {
    /// A local install: the switch is on unless the person switched it off.
    pub fn new(data_dir: &Path) -> Control {
        Control::build(data_dir, None, true)
    }

    /// For an install with this exposure: `agentMayChange` is what `control.json` says, and where the file
    /// (or its field) is absent, on for a local install (as it has always been) and **off** for a proxied or
    /// lan one, where a prompt-injected Agent could reach further than the person at the keyboard.
    pub fn with_exposure(data_dir: &Path, exposure: crate::auth::mode::Exposure) -> Control {
        Control::build(
            data_dir,
            None,
            exposure == crate::auth::mode::Exposure::Local,
        )
    }

    /// For tests: its own navigator, and no engines.
    #[cfg(test)]
    pub(crate) fn with_navigator(data_dir: &Path, navigator: Option<Navigator>) -> Control {
        let mut control = Control::build(data_dir, navigator, true);
        if let Some(inner) = Arc::get_mut(&mut control.inner) {
            inner.engines = Some(None);
        }
        control
    }

    fn build(data_dir: &Path, navigator: Option<Navigator>, default_on: bool) -> Control {
        Control {
            inner: Arc::new(Inner {
                data_dir: data_dir.to_path_buf(),
                default_on,
                settings_lock: Mutex::new(()),
                audit: audit::Log::new(data_dir.join(LOG_FILE)),
                router: OnceLock::new(),
                navigator,
                engines: None,
            }),
        }
    }

    /// Give the tools the finished router (the one the server listens with).
    pub fn set_router(&self, router: Router) {
        let _ = self.inner.router.set(router);
    }

    pub(crate) fn router(&self) -> Option<Router> {
        self.inner.router.get().cloned()
    }

    /// Where the engines' control pages are (none: not running).
    pub(crate) fn engines_ui(&self) -> Option<String> {
        match &self.inner.engines {
            Some(own) => own.clone(),
            None => crate::http::engines_ui(),
        }
    }

    pub(crate) fn audit(&self) -> &audit::Log {
        &self.inner.audit
    }

    fn settings_path(&self) -> PathBuf {
        self.inner.data_dir.join(SETTINGS_FILE)
    }

    /// The switch. No file (or no field) is the default: on for a local
    /// install, off for a proxied or lan one; a file that cannot be read is
    /// off, so a person who switched it off is never switched back on by a
    /// damaged file.
    pub fn agent_may_change(&self) -> bool {
        read_switch(&self.settings_path(), self.inner.default_on)
    }

    pub fn set_agent_may_change(&self, on: bool) -> Result<bool, String> {
        let _one = self.inner.settings_lock.lock().unwrap_or_else(|e| e.into_inner());
        let path = self.settings_path();
        write_atomic(&path, &json!({ "agentMayChange": on }))?;
        Ok(on)
    }

    /// Show a page in the dashboard, or say why there is none.
    pub(crate) fn navigate(&self, payload: Value) -> Result<(), String> {
        let navigator = self.inner.navigator.clone().or_else(|| NAVIGATOR.get().cloned());
        match navigator {
            Some(n) => n(payload),
            None => Err("There is no dashboard to show it in: this OAIY runs headless (oaiy-server). Tell the person to open it in the OAIY desktop app.".into()),
        }
    }
}

fn read_switch(path: &Path, default_on: bool) -> bool {
    match std::fs::read_to_string(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => default_on,
        Err(e) => {
            log::warn!("control: {} could not be read ({e}): the Agent may not change OAIY until it is saved again", path.display());
            false
        }
        Ok(text) => match serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
            Ok(v) => match v.get("agentMayChange") {
                None | Some(Value::Null) => default_on,
                Some(Value::Bool(b)) => *b,
                Some(_) => false,
            },
            Err(e) => {
                log::warn!("control: {} is not JSON ({e}): the Agent may not change OAIY until it is saved again", path.display());
                false
            }
        },
    }
}

/// Write `value` to `path` whole or not at all: a temporary file beside it, then a rename.
pub(crate) fn write_atomic(path: &Path, value: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
    }
    let tmp = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4().simple()));
    std::fs::write(&tmp, text).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("could not save {}: {e}", path.display())
    })
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/// `/api/mcp` and `/api/control/*`, and the engines relay the tools need.
/// Merged inside the guard layers: all of it is restricted (see `http.rs`).
pub fn router(control: Control) -> Router {
    Router::new()
        .route("/api/mcp", post(mcp::handle))
        .route("/api/control/settings", get(get_settings).put(put_settings))
        .route("/api/control/log", get(get_log))
        .route("/api/control/desktop-log", get(desktop_log))
        .with_state(control.clone())
        .merge(engines::router(control))
}

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

async fn get_settings(State(control): State<Control>) -> Response {
    Json(json!({ "agentMayChange": control.agent_may_change() })).into_response()
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SettingsBody {
    agent_may_change: bool,
}

async fn put_settings(State(control): State<Control>, Json(body): Json<SettingsBody>) -> Response {
    match control.set_agent_may_change(body.agent_may_change) {
        Ok(on) => {
            log::info!("control: the Agent {} change OAIY", if on { "may" } else { "may not" });
            Json(json!({ "agentMayChange": on })).into_response()
        }
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

#[derive(Deserialize)]
struct LogQuery {
    limit: Option<usize>,
}

async fn get_log(State(control): State<Control>, Query(q): Query<LogQuery>) -> Response {
    let limit = q.limit.unwrap_or(100).clamp(1, audit::MAX_READ);
    Json(json!({ "entries": control.audit().read(limit) })).into_response()
}

#[derive(Deserialize)]
struct LinesQuery {
    lines: Option<usize>,
}

/// The end of the desktop's own log file. The GUI writes one (`applog`); a
/// headless server logs to its console, so there is none to read.
async fn desktop_log(Query(q): Query<LinesQuery>) -> Response {
    let lines = q.lines.unwrap_or(200).clamp(1, 2000);
    let Some(path) = crate::applog::LOGGER.path() else {
        return Json(json!({
            "available": false,
            "lines": [],
            "reason": "This OAIY runs headless: its log goes to its console, not to a file.",
        }))
        .into_response();
    };
    let result = tokio::task::spawn_blocking(move || std::fs::read(&path)).await;
    match result {
        Ok(Ok(bytes)) => {
            let text = String::from_utf8_lossy(&bytes);
            let all: Vec<&str> = text.lines().collect();
            let tail: Vec<&str> = all[all.len().saturating_sub(lines)..].to_vec();
            Json(json!({ "available": true, "lines": tail })).into_response()
        }
        Ok(Err(e)) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("the desktop's log could not be read: {e}")),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
