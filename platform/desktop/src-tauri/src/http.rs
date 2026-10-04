//! Localhost HTTP API the oaiy-web flow editor talks to.
//!
//! Phase 1 surface:
//!   GET /api/health    → { status, product, protocol, version }
//!
//! Phase 2 surface (this file):
//!   GET    /api/services                  → list registered + running services
//!   POST   /api/services/:id/start        → spawn the service process
//!   POST   /api/services/:id/stop         → terminate it
//!   POST   /api/services/:id/autostart    → {enabled} start it with the app
//!   POST   /api/services/:id/install      → run install script (streams logs)
//!   POST   /api/services/:id/uninstall    → remove its installed files (clean reinstall)
//!   GET    /api/services/:id/logs[?tail]  → recent stdout+stderr lines
//!
//!   GET    /api/models                       → list known/downloaded models (+ root dir)
//!   POST   /api/models/download              → start an HF / direct-URL download
//!   GET    /api/models/downloads             → in-flight + recent downloads
//!   POST   /api/models/downloads/:id/pause   → pause an in-flight download
//!   POST   /api/models/downloads/:id/resume  → resume a paused download
//!   POST   /api/models/downloads/:id/cancel  → cancel + delete .part
//!   DELETE /api/models/:name                 → remove a model file
//!
//!   GET    /api/python                    → python runtime + venv status
//!   POST   /api/python/install            → install bundled python (PBS)
//!   GET    /api/python/logs[?tail]        → current job's logs
//!   POST   /api/python/venvs              → create or reuse a venv
//!   DELETE /api/python/venvs/:name        → remove a venv
//!
//! Phase 4 (after Playwright sidecar): /api/browser/*

use axum::{
    extract::{Path, Query, Request, State},
    http::{
        header::{AUTHORIZATION, ORIGIN},
        Method, StatusCode,
    },
    middleware::{self, Next},
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use crate::services::catalog::CatalogHandle;
use crate::services::downloads::DownloadsHandle;
use crate::services::python::PythonHandle;
use crate::services::registry::RegistryHandle;
use crate::services::template::ServiceTemplate;

/// Convenience alias for the error type returned by the server loop.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Read-only data-dir configuration the web app shows ("your models live at
/// X"). Built by a [`ConfigProvider`] so the HTTP layer stays host-agnostic —
/// the Tauri GUI backs it with AppHandle paths, the headless `oaiy-server` with
/// env vars. This is the `GET /api/config` response shape.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DesktopConfig {
    /// The dir this running process is actually using right now.
    pub active_dir: String,
    /// The OS default (what "Reset" goes back to).
    pub default_dir: String,
    /// The override currently written to the pointer file, if any.
    pub configured_dir: Option<String>,
    /// True when a custom dir is configured (differs from default).
    pub is_custom: bool,
    /// True when the configured dir differs from the active dir.
    pub restart_required: bool,
    /// The models dir this running process is actually using.
    pub models_active_dir: String,
    /// The default the models dir falls back to (`<activeDataDir>/models`).
    pub models_default_dir: String,
    /// The `modelsDir` override currently written to the pointer, if any.
    pub models_configured_dir: Option<String>,
    /// True when a custom models dir is configured.
    pub models_is_custom: bool,
    /// True when the configured models dir differs from the active one.
    pub models_restart_required: bool,
}

/// Supplies the [`DesktopConfig`] snapshot for `GET /api/config` without
/// binding the HTTP layer to any particular host (Tauri AppHandle vs env vars).
pub trait ConfigProvider: Send + Sync + 'static {
    fn snapshot(&self, registry: &RegistryHandle) -> DesktopConfig;
}

#[derive(Clone)]
struct AppState {
    config: Arc<dyn ConfigProvider>,
    registry: RegistryHandle,
    downloads: DownloadsHandle,
    python: PythonHandle,
    catalog: CatalogHandle,
    node: crate::services::node_runtime::NodeHandle,
}

/// The handshake every client uses to decide "is this actually us?".
///
/// A fixed loopback port is trivially squatted — this check has already caught a
/// different vendor's app answering `/api/health` with a compatible shape, which
/// turned the UI's status badge green while every authenticated call 401'd. So
/// the identity is asserted, not assumed.
///
/// `protocol` is the negotiable part: clients must branch on it rather than on
/// `version`, which moves for reasons that have nothing to do with the wire
/// format. See `protocol/README.md`.
#[derive(Serialize)]
// camelCase because consumers read `apiVersion` / `pluginApiVersion`. The
// single-word fields are unaffected, so this changes no existing name — but
// without it the two version fields ship as snake_case and read as `undefined`
// on the other side, while health still returns a healthy 200.
#[serde(rename_all = "camelCase")]
struct HealthResponse {
    status: &'static str,
    /// Stable machine identity. Never localise or re-word this.
    product: &'static str,
    /// The same identity under the name a consumer's desktop-detection probe
    /// looks for.
    ///
    /// Duplicated rather than renamed: `product` is this API's own long-standing
    /// field and something may already match on it, while `companion` is what a
    /// host app's loopback probe reads to decide a desktop is present at all.
    /// Without it OAIY answers health perfectly and is still reported as "no
    /// desktop", because the field the probe checks was simply absent.
    companion: &'static str,
    /// Bridge Protocol the rest of this API speaks.
    protocol: &'static str,
    version: &'static str,
    /// Surfaces the same numbers the probe records, so a consumer can refuse a
    /// desktop too old for it rather than failing later on a missing route.
    api_version: u32,
    plugin_api_version: u32,
}

/// Wire identity. Deliberately NOT derived from the crate name — renaming the
/// binary must not silently change what clients match on.
pub const PRODUCT_ID: &str = "oaiy-desktop";

/// Version of THIS HTTP API, for a consumer deciding whether it can talk to us.
pub const API_VERSION: u32 = 1;
/// Bump only on a breaking wire change; add fields freely without touching it.
pub const BRIDGE_PROTOCOL: &str = "oaiy-bridge/1";

/// `GET /api/health`'s answer: the fixed fields above and, when the access guard is in front of the
/// route, `access` (the access mode: anything but `scoped` is worth a banner) and `storage` (`ok`,
/// `low` or `full`: the disk of the credential store). Both are additive; nothing else changed.
#[derive(Serialize)]
struct HealthBody {
    #[serde(flatten)]
    base: HealthResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    access: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage: Option<&'static str>,
}

async fn health(extras: Option<axum::Extension<crate::auth::HealthExtras>>) -> Json<HealthBody> {
    let extras = extras.map(|e| e.0);
    Json(HealthBody {
        base: HealthResponse {
            status: "ok",
            product: PRODUCT_ID,
            companion: PRODUCT_ID,
            protocol: BRIDGE_PROTOCOL,
            version: env!("CARGO_PKG_VERSION"),
            api_version: API_VERSION,
            plugin_api_version: *crate::plugins::manifest::SUPPORTED_PLUGIN_API.end(),
        },
        access: extras.map(|e| e.access),
        storage: extras.map(|e| e.storage),
    })
}

// ------- services -------

async fn list_services(State(state): State<AppState>) -> impl IntoResponse {
    match state.registry.lock() {
        Ok(mut reg) => {
            // Pick up any package dropped into templates/ since last poll, so
            // services are dynamically loadable just by adding a file there.
            reg.reload_new_templates();
            (StatusCode::OK, Json(reg.snapshot())).into_response()
        }
        Err(_) => err500("registry mutex poisoned"),
    }
}

/// A success body for control routes, instead of 204 No Content.
///
/// A bare 204 is correct HTTP and a trap for browser clients: a fetch wrapper
/// that parses every reply as JSON sees an empty body, fails to parse, and
/// reports a transport error — so a start that WORKED is shown to the user as a
/// failure. A tiny body costs nothing and removes the whole class of bug.
fn ok_body() -> axum::response::Response {
    (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response()
}

async fn start_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.start(&id));
    match result {
        Ok(()) => ok_body(),
        Err(e) => err400(&e),
    }
}

async fn stop_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.stop(&id));
    match result {
        Ok(()) => ok_body(),
        Err(e) => err400(&e),
    }
}

/// Body of `POST /api/services/:id/autostart`.
#[derive(serde::Deserialize)]
struct AutostartBody {
    enabled: bool,
}

/// Tick or untick "start this service when the app starts".
///
/// A preference, not an action: it deliberately does NOT start or stop anything
/// now. Ticking a stopped service and having it spring to life would make the
/// checkbox a second Start button, and unticking a running one would stop a
/// service the user never asked to stop.
async fn set_service_autostart(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<AutostartBody>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.set_autostart(&id, body.enabled));
    match result {
        Ok(()) => ok_body(),
        Err(e) => err400(&e),
    }
}

/// Install via `Runner` (same pipe + log machinery used for service
/// processes) so the existing /logs endpoint streams progress in real
/// time — no extra plumbing on the UI side.
/// Clear a tripped crash breaker and start the service from a clean slate,
/// tearing down any process tree still holding its port.
async fn repair_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.repair(&id));
    match result {
        Ok(()) => ok_body(),
        Err(e) => err400(&e),
    }
}

async fn install_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.install_streaming(&id));
    match result {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(e) => err400(&e),
    }
}

/// Cancel an in-flight install (kills the install process tree). Logs are
/// kept so the user can still read where it stopped.
async fn cancel_install_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.cancel_install(&id));
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

#[derive(Deserialize)]
struct LogsQuery {
    tail: Option<usize>,
}

async fn service_logs(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<LogsQuery>,
) -> impl IntoResponse {
    match state.registry.lock() {
        Ok(reg) => match reg.logs(&id, q.tail) {
            Some(lines) => (StatusCode::OK, Json(lines)).into_response(),
            None => (StatusCode::OK, Json(Vec::<serde_json::Value>::new())).into_response(),
        },
        Err(_) => err500("registry mutex poisoned"),
    }
}

/// Create or replace a service template from a UI form. Body is the
/// ServiceTemplate JSON itself (same shape on-disk + over the wire).
async fn add_service(
    State(state): State<AppState>,
    Json(template): Json<ServiceTemplate>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.add_template(template));
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

/// Export a service as a self-contained, shareable package: the template with
/// every script it references inlined into `files`. POST the result back to
/// `/api/services` on any machine to install it — no recompile, no loose files.
async fn export_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|reg| reg.export_package(&id));
    match result {
        Ok(pkg) => Json(pkg).into_response(),
        Err(e) => err400(&e),
    }
}

async fn delete_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.delete_template(&id));
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

/// Remove a service's installed files (the template's `uninstall` paths) so the
/// user can clean-reinstall — e.g. swap an old build for a new one.
/// Privileged + destructive (gated like delete); leaves the template in place.
async fn uninstall_service(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let result = state
        .registry
        .lock()
        .map_err(|_| "registry mutex poisoned".to_string())
        .and_then(|mut reg| reg.uninstall(&id));
    match result {
        Ok(n) => (StatusCode::OK, Json(serde_json::json!({ "removed": n }))).into_response(),
        Err(e) => err400(&e),
    }
}

#[derive(Deserialize)]
struct EnsureByPortRequest {
    port: u16,
}

/// Start OAIY Desktop service that owns `port` if it isn't already
/// running. Called by oaiy-web before it hits a `127.0.0.1:<port>`
/// endpoint an OAIY Desktop service owns, so picking a stopped service in a
/// flow and running it "just works". Returns immediately after the
/// spawn — the flow's HTTP/LLM node retries while the server warms up.
async fn ensure_service_by_port(
    State(state): State<AppState>,
    Json(req): Json<EnsureByPortRequest>,
) -> impl IntoResponse {
    match state.registry.lock() {
        Ok(mut reg) => (StatusCode::OK, Json(reg.ensure_by_port(req.port))).into_response(),
        Err(_) => err500("registry mutex poisoned"),
    }
}

// ------- models -------

#[derive(Debug, Deserialize)]
struct ModelsQuery {
    offset: Option<usize>,
    limit: Option<usize>,
}

/// The most files one request returns. A library can hold thousands, and
/// serialising every one of them into a panel that polls is the difference
/// between a list that opens instantly and one that stutters. `total` is always
/// the real count, so a page is never mistaken for the whole library.
const MODELS_PAGE_MAX: usize = 500;

async fn list_models(
    State(state): State<AppState>,
    Query(q): Query<ModelsQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(MODELS_PAGE_MAX).min(MODELS_PAGE_MAX);
    match state.downloads.list_models_page(q.offset.unwrap_or(0), limit) {
        Ok(models) => (StatusCode::OK, Json(models)).into_response(),
        Err(e) => err500(&e),
    }
}

/// `camelCase` + `deny_unknown_fields` because this body carries a CHECKSUM.
/// Without the rename, `expectedSha256` deserialised to `None` and the download
/// ran unverified while the caller believed it had asked for verification —
/// silently, which is the worst possible way for a checksum to fail. Denying
/// unknown fields turns the next such typo into a 400 instead of a shrug.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ModelDownloadRequest {
    /// Either a HuggingFace URL ("https://huggingface.co/<repo>/resolve/<rev>/<file>")
    /// or a direct download URL.
    url: String,
    /// Destination filename. Defaults to last path segment of the URL.
    #[serde(default)]
    filename: Option<String>,
    /// Optional subdirectory under the models dir.
    #[serde(default)]
    subdir: Option<String>,
    /// Expected SHA-256 of the content, if the caller knows it. HuggingFace
    /// downloads don't need this — the origin publishes the digest and we use
    /// it automatically. Refused if malformed rather than ignored.
    #[serde(default)]
    expected_sha256: Option<String>,
}

async fn start_model_download(
    State(state): State<AppState>,
    Json(req): Json<ModelDownloadRequest>,
) -> impl IntoResponse {
    match state
        .downloads
        .start(
            &req.url,
            req.filename.as_deref(),
            req.subdir.as_deref(),
            req.expected_sha256.as_deref(),
        )
    {
        Ok(id) => (StatusCode::ACCEPTED, Json(serde_json::json!({ "downloadId": id })))
            .into_response(),
        Err(e) => err400(&e),
    }
}

async fn list_downloads(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.downloads.snapshot())).into_response()
}

async fn pause_download(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.downloads.pause(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

async fn resume_download(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.downloads.resume(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

async fn cancel_download(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.downloads.cancel(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

async fn delete_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.downloads.delete_model(&name) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

async fn model_catalog(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.catalog.snapshot())).into_response()
}

/// Read-only data-dir configuration so the web app can show "your models
/// live at X". Changing the dir is a desktop-only action (native picker +
/// restart), so there's intentionally no POST here.
async fn get_config(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.config.snapshot(&state.registry))).into_response()
}

// ------- node runtime -------

/// The Node runtime the bundled CLI runs under: system, portable, or absent.
async fn node_status(State(state): State<AppState>) -> impl IntoResponse {
    (StatusCode::OK, Json(state.node.snapshot())).into_response()
}

/// Download + install the pinned portable Node. Returns immediately; progress
/// streams through `/api/node/logs`, mirroring the Python installer.
async fn install_node(State(state): State<AppState>) -> impl IntoResponse {
    match state.node.install() {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(e) => err400(&e),
    }
}

async fn node_logs(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> impl IntoResponse {
    match state.node.current_logs(q.tail) {
        Some(lines) => (StatusCode::OK, Json(lines)).into_response(),
        None => (StatusCode::OK, Json(Vec::<crate::services::runner::LogLine>::new())).into_response(),
    }
}

// ------- python -------

async fn python_status(State(state): State<AppState>) -> impl IntoResponse {
    // Fold the registry's venv→service usage into each venv's
    // `bound_services` so the Python tab can show "used by …". The Python
    // module is registry-agnostic, so the join happens here where both
    // handles are in scope.
    let mut snap = state.python.snapshot();
    if let Ok(reg) = state.registry.lock() {
        let usage = reg.venv_usage();
        for v in &mut snap.venvs {
            if let Some(svcs) = usage.get(&v.name) {
                v.bound_services = svcs.clone();
            }
        }
    }
    (StatusCode::OK, Json(snap)).into_response()
}

async fn install_python(State(state): State<AppState>) -> impl IntoResponse {
    match state.python.install_runtime() {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(e) => err400(&e),
    }
}

/// Logs of the currently-running Python job (install or venv create).
/// Returns an empty array when nothing is in flight; the UI's LogsViewer
/// renders the same way for either case.
async fn python_logs(
    State(state): State<AppState>,
    Query(q): Query<LogsQuery>,
) -> impl IntoResponse {
    match state.python.current_logs(q.tail) {
        Some(lines) => (StatusCode::OK, Json(lines)).into_response(),
        None => (
            StatusCode::OK,
            Json(Vec::<crate::services::runner::LogLine>::new()),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct VenvRequest {
    name: String,
    /// Pip-installable packages to set up after venv creation.
    #[serde(default)]
    requirements: Vec<String>,
}

async fn create_venv(
    State(state): State<AppState>,
    Json(req): Json<VenvRequest>,
) -> impl IntoResponse {
    match state.python.create_or_reuse_venv(&req.name, &req.requirements) {
        Ok(path) => (StatusCode::OK, Json(serde_json::json!({ "path": path }))).into_response(),
        Err(e) => err400(&e),
    }
}

async fn delete_venv(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.python.delete_venv(&name) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

// ------- helpers -------

fn err400(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": msg })),
    )
        .into_response()
}

fn err500(msg: &str) -> axum::response::Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": msg })),
    )
        .into_response()
}

/// Whether a browser `Origin` is allowed to drive state-changing endpoints.
/// The localhost bind keeps non-browser callers out; this stops a *web page*
/// the user happens to have open from issuing drive-by POST/DELETE requests
/// (which would otherwise be possible since CORS is permissive for reads).
/// True only when `origin`'s HOST is exactly a loopback name — NOT a prefix.
/// `origin.starts_with("http://localhost")` would also accept the attacker-owned
/// `http://localhost.evil.com`, so we parse the host and compare it exactly.
/// Port-agnostic; handles bracketed IPv6 (`http://[::1]:port`).
fn is_loopback_origin(origin: &str) -> bool {
    let rest = match origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    {
        Some(r) => r,
        None => return false,
    };
    let host = rest.split('/').next().unwrap_or(rest);
    if let Some(inner) = host.strip_prefix('[') {
        // Bracketed IPv6: take the part before ']'.
        return inner.split(']').next() == Some("::1");
    }
    // host[:port] — strip a trailing :port (none of our loopback names contain ':').
    let host = host.rsplit_once(':').map(|(h, _)| h).unwrap_or(host);
    host == "localhost" || host == "127.0.0.1"
}

/// The schemes OAIY's own pages are served from: the agent, `oaiy`, and the
/// flow editor, `oaiyflows` (`embed.rs`, which serves them, has a test that
/// these are its `AGENT_SCHEME` and `FLOWS_SCHEME`).
pub const EMBEDDED_SCHEMES: [&str; 2] = ["oaiy", "oaiyflows"];

/// The exact origins of the agent's window and the flow editor's window: the
/// page of each scheme is `http://<scheme>.localhost` on Windows (WebView2 maps
/// a custom scheme to that) and `<scheme>://localhost` elsewhere. The same in a
/// debug build under `tauri dev` and in an installed one: both serve the pages
/// from these schemes, not from a dev server.
pub fn embedded_window_origins(windows: bool) -> Vec<String> {
    EMBEDDED_SCHEMES
        .iter()
        .map(|s| if windows { format!("http://{s}.localhost") } else { format!("{s}://localhost") })
        .collect()
}

/// Whether `origin` is one of the pages OAIY shows in its own window (the
/// agent, `oaiy`; the flow editor, `oaiyflows`), served from its own schemes.
pub fn is_embedded_origin(origin: &str) -> bool {
    EMBEDDED_SCHEMES.iter().any(|s| {
        origin == format!("{s}://localhost") || origin == format!("http://{s}.localhost") || origin == format!("https://{s}.localhost")
    })
}

/// `GET /api/update/status`: open on the headless server (like health), a restricted read on the desktop.
const UPDATE_STATUS_PATH: &str = "/api/update/status";

fn is_allowed_origin(origin: &str) -> bool {
    // Dev + locally-served oaiy-web (any loopback port).
    if is_loopback_origin(origin) {
        return true;
    }
    // OAIY's own pages in its window: the agent and the flow editor.
    if is_embedded_origin(origin) {
        return true;
    }
    // The provider this desktop is LINKED to. Linking is the user approving
    // that provider, and its web app is where they then expect to see and
    // control this machine. Derived from the link rather than hardcoded, so no
    // address is baked in and unlinking withdraws it.
    if crate::link::linked_origin().is_some_and(|o| o == origin) {
        return true;
    }
    // Tauri webview origins (in case OAIY Desktop's own UI ever calls over HTTP).
    if origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
    {
        return true;
    }
    // Production oaiy-web: https://oaiy.com and any subdomain (port-agnostic).
    if let Some(rest) = origin.strip_prefix("https://") {
        let host = rest.split('/').next().unwrap_or(rest);
        let host = host.split(':').next().unwrap_or(host);
        if host == "oaiy.com" || host.ends_with(".oaiy.com") {
            return true;
        }
    }
    false
}

/// Endpoints that DEFINE/INSTALL arbitrary code or DESTROY user data — i.e. the
/// real exec surface. A malicious local web page (any `http://localhost:<port>`)
/// must not be able to reach these: defining a service command and starting it
/// would be remote code execution. They get the stricter origin check below and
/// fail CLOSED on a missing `Origin`. Note: starting/stopping/installing an
/// ALREADY-DEFINED service (and ensure-by-port) stays on the broad allow-list —
/// those only run commands the user already added + reviewed, and the web app
/// relies on them.
fn is_privileged_path(method: &Method, path: &str) -> bool {
    match *method {
        Method::POST => {
            matches!(
                path,
                "/api/services"
                    | "/api/models/download"
                    | "/api/python/venvs"
                    | "/api/python/install"
                    | "/api/node/install"
                    // Starts a download of gigabytes into the engines' folder.
                    | "/api/engines/downloads"
                    // Makes OAIY ask GitHub for a newer release (at most every 30 seconds), and the Agent
                    // page's answer when it is asked to save its work before an update: a page the owner
                    // happens to have open must not do either. (Downloading and installing are commands of
                    // the dashboard's own window, not routes.)
                    | "/api/update/check"
                    | "/api/update/agent-flushed"
            ) || (path.starts_with("/api/services/") && path.ends_with("/uninstall"))
                || is_setup_path(path)
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
                || is_personal_path(path)
                || is_control_path(path)
                || is_engine_control_path(path)
                // The Agent page handing its storage over to a backup a session the desktop opened.
                || crate::backup::routes::is_backup_path(path)
        }
        Method::PATCH => is_personal_path(path),
        // PUT is only used by the bridge (flow documents). A flow doc is
        // executable code the worker hands to the CLI, so it is exec surface.
        Method::PUT => {
            is_bridge_exec_path(path) || is_personal_path(path) || is_setup_path(path) || is_control_path(path) || is_engine_control_path(path)
        }
        Method::DELETE => {
            path.starts_with("/api/services/")
                || path.starts_with("/api/models/")
                || path.starts_with("/api/python/venvs/")
                // Uninstalling a plugin removes native code from disk.
                || path.starts_with("/api/plugins/")
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
                || is_personal_path(path)
                || is_control_path(path)
        }
        _ => false,
    }
}

/// The control API (`control/`): the MCP server the Agent configures OAIY
/// with, the Agent's switch and the log of what it changed. Its tools reach
/// everything the other privileged routes do, so every change takes the
/// privileged gate and every read is a restricted read, like `/api/setup`.
fn is_control_path(path: &str) -> bool {
    path == "/api/mcp" || path == "/api/control" || path.starts_with("/api/control/")
}

/// Choosing the engines' models and starting or stopping their language
/// model (`control/engines.rs`): the engines' configuration, taken on the
/// privileged gate like their downloads.
fn is_engine_control_path(path: &str) -> bool {
    path == "/api/engines/defaults" || path.starts_with("/api/engines/llm/")
}

/// Where the engines' control pages are, once the desktop found or started them (see `engines.rs`).
static ENGINES_UI: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

pub fn set_engines_ui(url: &str) {
    if let Ok(mut g) = ENGINES_UI.write() {
        *g = Some(url.to_string());
    }
}

/// For tests: no engines are recorded again.
#[cfg(test)]
pub(crate) fn clear_engines_ui() {
    if let Ok(mut g) = ENGINES_UI.write() {
        *g = None;
    }
}

/// `GET /api/engines`: the engines' state for the window — the language model
/// (loaded or not, which one), the GPUs, where their pages are. Their control
/// port answers only its own origin, so the desktop asks it for the window.
async fn engines_status() -> axum::response::Response {
    let Some(ui) = ENGINES_UI.read().ok().and_then(|g| g.clone()) else {
        return Json(serde_json::json!({ "running": false })).into_response();
    };
    let state = match reqwest::Client::new().get(format!("{ui}/api/state")).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.ok(),
        _ => None,
    };
    let Some(state) = state else {
        return Json(serde_json::json!({ "running": false, "uiUrl": ui })).into_response();
    };
    let llm = state.get("llm").cloned().unwrap_or(serde_json::Value::Null);
    let system = match reqwest::Client::new().get(format!("{ui}/api/system")).timeout(std::time::Duration::from_secs(5)).send().await {
        Ok(r) if r.status().is_success() => r.json::<serde_json::Value>().await.unwrap_or(serde_json::Value::Null),
        _ => serde_json::Value::Null,
    };
    Json(serde_json::json!({
        "running": true,
        "uiUrl": ui,
        "llm": {
            "state": llm.get("state"),
            "resident": llm.get("resident"),
            "models": llm.get("models"),
            "loadSeconds": llm.get("load_seconds"),
        },
        "gpus": system.get("gpus"),
    }))
    .into_response()
}

// ------- the engines' catalog and downloads, relayed -------
//
// The setup wizard offers the language model (and what other groups need)
// from the engines' own catalog, and downloads it with progress, without the
// window reaching the engines' control port (it answers only its own origin).
// The model the person chose in Engines is the discovery document's
// `defaults.<group>` (the document the gateway serves as `/v1/discovery`,
// which the control port serves as `/api/discovery`): nothing here names a
// model of its own.

pub(crate) fn engines_ui() -> Option<String> {
    ENGINES_UI.read().ok().and_then(|g| g.clone())
}

/// The message when no engines are there to relay to.
const ENGINES_NOT_RUNNING: &str = "The engines are not running: they start with OAIY, or run oaiy-studio.";

pub(crate) async fn studio_json(ui: &str, method: reqwest::Method, path: &str, body: Option<serde_json::Value>) -> Result<serde_json::Value, (u16, String)> {
    let mut req = reqwest::Client::new().request(method, format!("{ui}{path}")).timeout(std::time::Duration::from_secs(10));
    if let Some(body) = body {
        req = req.json(&body);
    }
    let resp = req.send().await.map_err(|e| (502, format!("the engines did not answer: {e}")))?;
    let status = resp.status().as_u16();
    let value: serde_json::Value = resp.json().await.map_err(|e| (502, format!("the engines' answer could not be read: {e}")))?;
    if (200..300).contains(&status) {
        Ok(value)
    } else {
        let message = value.get("error").and_then(|e| e.as_str().map(str::to_string).or_else(|| e.get("message").and_then(|m| m.as_str()).map(str::to_string)));
        Err((status, message.unwrap_or_else(|| format!("the engines answered {status}"))))
    }
}

/// One download as the window reads it (camelCase, only what it shows).
fn engine_download(d: &serde_json::Value) -> serde_json::Value {
    if d.is_null() {
        return serde_json::Value::Null;
    }
    serde_json::json!({
        "id": d.get("id"),
        "status": d.get("status"),
        "done": d.get("done"),
        "total": d.get("total"),
        "file": d.get("file"),
        "filesDone": d.get("files_done"),
        "filesTotal": d.get("files_total"),
        "speed": d.get("speed"),
        "error": d.get("error"),
    })
}

/// One catalog model as the window reads it.
fn engine_model(m: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "id": m.get("id"),
        "group": m.get("group"),
        "name": m.get("name"),
        "about": m.get("about"),
        "license": m.get("license"),
        "sizeGb": m.get("size_gb"),
        "vramGb": m.get("vram_gb"),
        "recommended": m.get("recommended").and_then(|r| r.as_bool()).unwrap_or(false),
        // A language model the Agent can use its tools with (the catalog's `agent_tools`); the others chat.
        "agentTools": m.get("agent_tools").and_then(|r| r.as_bool()).unwrap_or(false),
        "needs": m.get("needs").cloned().unwrap_or_else(|| serde_json::json!([])),
        "installed": m.get("installed").and_then(|r| r.as_bool()).unwrap_or(false),
        "partial": m.get("partial").and_then(|r| r.as_bool()).unwrap_or(false),
        "download": engine_download(m.get("download").unwrap_or(&serde_json::Value::Null)),
    })
}

fn engine_downloads(state: &serde_json::Value) -> Vec<serde_json::Value> {
    state
        .get("models")
        .and_then(|m| m.as_array())
        .map(|models| models.iter().filter_map(|m| m.get("download").filter(|d| !d.is_null()).map(engine_download)).collect())
        .unwrap_or_default()
}

/// The catalog at `ui` (none: the engines are not running), with the models
/// chosen in Engines (`defaults`, from its discovery document; `null` for a
/// group with none).
pub(crate) async fn engines_catalog_at(ui: Option<String>) -> serde_json::Value {
    let Some(ui) = ui else {
        return serde_json::json!({ "running": false, "error": ENGINES_NOT_RUNNING });
    };
    let (state, discovery) = tokio::join!(
        studio_json(&ui, reqwest::Method::GET, "/api/downloads", None),
        studio_json(&ui, reqwest::Method::GET, "/api/discovery", None)
    );
    let state = match state {
        Ok(s) => s,
        Err((_, e)) => return serde_json::json!({ "running": false, "error": e }),
    };
    let defaults: serde_json::Map<String, serde_json::Value> = discovery
        .ok()
        .and_then(|d| d.get("defaults").and_then(|d| d.as_object()).cloned())
        .unwrap_or_default()
        .into_iter()
        .map(|(group, m)| (group, m.as_str().map(str::trim).filter(|m| !m.is_empty()).map_or(serde_json::Value::Null, |m| serde_json::Value::String(m.to_string()))))
        .collect();
    serde_json::json!({
        "running": true,
        "dir": state.get("dir"),
        "free": state.get("free"),
        "groups": state.get("groups").cloned().unwrap_or_else(|| serde_json::json!([])),
        "models": state.get("models").and_then(|m| m.as_array()).map(|m| m.iter().map(engine_model).collect::<Vec<_>>()).unwrap_or_default(),
        "defaults": defaults,
    })
}

/// The downloads at `ui`, as they stand.
pub(crate) async fn engines_downloads_at(ui: Option<String>) -> serde_json::Value {
    let Some(ui) = ui else {
        return serde_json::json!({ "running": false, "downloads": [], "error": ENGINES_NOT_RUNNING });
    };
    match studio_json(&ui, reqwest::Method::GET, "/api/downloads", None).await {
        Ok(state) => serde_json::json!({ "running": true, "downloads": engine_downloads(&state) }),
        Err((_, e)) => serde_json::json!({ "running": false, "downloads": [], "error": e }),
    }
}

/// A catalog id: what the engines' catalog names its models with.
fn catalog_id(id: &str) -> bool {
    (1..=80).contains(&id.len())
        && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

/// Start downloading catalog model `id` at `ui`, into the engines' own
/// download folder (no folder is ever passed on).
pub(crate) async fn engines_download_at(ui: Option<String>, id: &str) -> Result<serde_json::Value, (u16, String)> {
    if !catalog_id(id) {
        return Err((400, format!("{id:?} is not a catalog model id")));
    }
    let ui = ui.ok_or((409, ENGINES_NOT_RUNNING.to_string()))?;
    let state = studio_json(&ui, reqwest::Method::POST, "/api/downloads", Some(serde_json::json!({ "id": id }))).await?;
    Ok(serde_json::json!({ "running": true, "downloads": engine_downloads(&state) }))
}

/// `GET /api/engines/catalog`
async fn engines_catalog() -> axum::response::Response {
    Json(engines_catalog_at(engines_ui()).await).into_response()
}

/// `GET /api/engines/downloads`
async fn engines_downloads() -> axum::response::Response {
    Json(engines_downloads_at(engines_ui()).await).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EngineDownloadBody {
    id: String,
}

/// `POST /api/engines/downloads {id}`: download a model from the engines' catalog.
async fn engines_download_start(Json(body): Json<EngineDownloadBody>) -> axum::response::Response {
    match engines_download_at(engines_ui(), body.id.trim()).await {
        Ok(v) => Json(v).into_response(),
        Err((status, message)) => (
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            Json(serde_json::json!({ "error": message })),
        )
            .into_response(),
    }
}

/// The setup wizard's record and its checks (`setup.rs`). Changing it is
/// privileged: accepting a plugin's capabilities is a trust act, and a check
/// runs one of the plugin's commands. Reading it is a restricted read.
fn is_setup_path(path: &str) -> bool {
    path == "/api/setup" || path.starts_with("/api/setup/")
}

/// Calls, the calendar, the contacts and flows' tasks for the agent: callers' numbers and words,
/// customers' names and appointments, the person's notes about the people who ring and what the
/// receptionist remembered, what a flow asks. Reading them is a restricted read; changing them
/// (speaking on a live call, booking or deleting an appointment, naming or importing contacts)
/// takes the privileged gate.
fn is_personal_path(path: &str) -> bool {
    path.starts_with("/api/voice/")
        || path == "/api/calendar"
        || path.starts_with("/api/calendar/")
        || path == "/api/contacts"
        || path.starts_with("/api/contacts/")
        // Whom the receptionist may put through and to which devices, and the messages callers leave.
        || path == "/api/ring"
        || path.starts_with("/api/ring/")
        || path == "/api/messages"
        || path.starts_with("/api/messages/")
        || path.starts_with("/api/agent/")
}

/// Bridge + plugin routes that EXECUTE code, cause physical side effects, or
/// persist a foothold across restart. A security review found the entire bridge
/// surface was on the broad (loopback-permissive) allow-list: a local web page
/// could `PUT` a flow document and `POST /api/bridge/runs` to run it — remote
/// code execution of exactly the shape `is_privileged_path` already defends the
/// services routes against — or `POST` a connector command to send an SMS. So
/// these join the strict-origin set (OAIY's own webview / oaiy.com only; loopback
/// only in debug) and fail closed on a missing Origin. `/api/bridge/capabilities`
/// and `/api/health` are deliberately NOT here — they are the open discovery
/// handshake the protocol commits to, and carry no user data.
fn is_bridge_exec_path(path: &str) -> bool {
    path == "/api/bridge/runs"                       // reserve + execute a flow
        || path == "/api/bridge/triggers"            // create a persistent binding
        || path.starts_with("/api/bridge/runs/")     // claim / finish / cancel
        || path.starts_with("/api/bridge/flows/")    // define / delete a flow doc
        || path.starts_with("/api/bridge/triggers/") // delete a binding
        || path.starts_with("/api/bridge/connectors/") // physical side effects
        // Redrive re-dispatches a stored event through the ordinary trigger
        // path, so it RESERVES RUNS the worker then executes — the same power
        // as creating a run, and it was on the broad loopback allow-list.
        // DELETE on the same prefix destroys the only record that an event was
        // lost, which is not something a local page should be able to do either.
        || path.starts_with("/api/bridge/deadletters")
        // Companion device trust: these decide which phones may carry a live
        // call's audio, and rotation invalidates every existing pairing. A
        // local web page must not reach them just by being on loopback.
        || path.starts_with("/api/companion/")
        // Linking opens the user's browser and ends in a stored credential;
        // unlinking throws that credential away. Neither belongs to a local
        // page that happens to be on loopback.
        || path.starts_with("/api/link")
        // Pairing APPROVAL/denial is the user's trust act, and revoke unpairs an
        // app — only OAIY's own webview (or a token holder) may. But raising a
        // request (`POST /api/bridge/pairing`) and polling it
        // (`/api/bridge/pairing/<id>`) are OPEN — their whole job is to let an
        // untrusted consumer bootstrap, and neither grants anything without the
        // approval below. `pairings` (plural) is the granted-token surface.
        || path.ends_with("/approve")
        || path.ends_with("/deny")
        || path.starts_with("/api/bridge/pairings")
        // Installing a plugin installs NATIVE CODE this host will then supervise,
        // and removing one deletes it from disk — strictly more dangerous than
        // starting an already-reviewed plugin. A paired web page must never
        // reach either; only OAIY's own window or a token holder.
        || path == "/api/plugins/install"
        // Invoking a plugin-contributed action IS a connector command with
        // physical side effects (aokie.phone's call.dial places a real call), so
        // it takes the same gate as the raw connector route.
        || path.starts_with("/api/services/actions/")
        || (path.starts_with("/api/plugins/")
            && (path.ends_with("/start")
                || path.ends_with("/stop")
                || path.ends_with("/enabled")
                // Letting a package nobody signed run is the person's decision, and
                // only OAIY's own window (or a token holder) may make it for them.
                || path.ends_with("/trust")))
}

/// The AI gateway surface: provider CRUD + credential admin AND the chat/models
/// proxy that spends the stored key. ALL of it takes the exec-surface gate
/// (trusted origin OR bearer/paired token, fail-closed on a missing Origin) — an
/// anonymous local page must not reconfigure a provider, plant a key, or spend
/// the user's API credits. The path prefix covers `/api/ai/providers*` (config)
/// and `/api/ai/{v1,providers/:id/v1}/*` (the gateway). GET reads are handled by
/// `is_restricted_read_path`, not here.
fn is_ai_exec_path(path: &str) -> bool {
    path.starts_with("/api/ai/") || crate::ai::plugin_completion::is_route(path) || crate::voice::plugin_session::is_route(path)
}

/// GET /api/services/:id/export returns the FULL ServiceTemplate — including `run.env` (which a
/// user-authored service may hold an API key in) and the verbatim install/helper script bodies.
/// It's the read-twin of the privileged `add_service` POST, so it's gated like a privileged read
/// (trusted origin or token) rather than left on the open GET surface.
fn is_export_path(path: &str) -> bool {
    (path.starts_with("/api/services/") && path.ends_with("/export"))
        // The storage a restore left for the Agent page (whole conversations): its own page, the dashboard, or a token.
        || (crate::backup::routes::is_backup_path(path) && !crate::backup::routes::is_status_path(path))
}

/// GET reads that expose process output / absolute paths (the OS username via the data-dir path)
/// and so must not be readable by an arbitrary cross-origin page: the logs endpoints + the config
/// snapshot. Gated on the broad allow-list (blocks only a remote cross-origin page; loopback dev
/// tools + the native CLI still pass).
fn is_restricted_read_path(path: &str) -> bool {
    path == "/api/config"
        || path == "/api/python/logs"
        || path == "/api/node"
        || path == "/api/node/logs"
        || (path.starts_with("/api/services/") && path.ends_with("/logs"))
        // When the last backup was made and whether a restore waits: for OAIY's own pages.
        || crate::backup::routes::is_status_path(path)
        // Bridge/plugin reads carry real data an arbitrary remote page must not
        // scrape cross-origin: events hold plugin-supplied payloads (for Aokie,
        // caller phone numbers and message bodies), runs hold flow inputs and
        // outputs, and the plugins listing exposes each plugin's absolute `dir`
        // — the OS username, the exact leak `/api/config` is already gated for.
        // `capabilities` + `health` stay open (discovery); everything else under
        // these prefixes is gated to a non-remote origin.
        || path == "/api/plugins"
        // Which modules the plugins provide (the phone, the calendar) — same tier as /api/plugins.
        || path == "/api/modules"
        || path == "/api/modules/events"
        // Which plugins are installed and what they can do — same tier as /api/plugins.
        || path == "/api/services/definitions"
        // Readiness names plugins + queue depth: gated like the other bridge reads.
        || path == "/api/bridge/status"
        // A dead letter stores the WHOLE event envelope — the same
        // plugin-supplied payload `/api/bridge/events` is gated for, except
        // durable across restarts rather than a 500-entry ring.
        || path.starts_with("/api/bridge/deadletters")
        // Companion status names every trusted device and the desktop's own
        // thumbprint — the material an attacker would want in order to imitate
        // a pairing screen. Same tier as the other bridge reads.
        || path.starts_with("/api/companion/")
        // The link status names the provider, the account and the granted
        // scopes — an inventory of what this machine can reach.
        || path.starts_with("/api/link")
        || path == "/api/bridge/events"
        || path == "/api/bridge/runs"
        || path == "/api/bridge/flows"
        || path == "/api/bridge/triggers"
        || path.starts_with("/api/bridge/runs/")
        || (path.starts_with("/api/plugins/") && path.ends_with("/logs"))
        // Who is asking to pair, and which apps are paired, are the OAIY UI's to
        // see — not a remote page's. Note the exact match: the POLL route
        // `/api/bridge/pairing/<id>` is deliberately NOT here (a consumer must be
        // able to poll for its own token), only the listing `/api/bridge/pairing`.
        || path == "/api/bridge/pairing"
        || path == "/api/bridge/pairings"
        // The AI gateway's reads: sources union + provider listing (names, base
        // URLs, hasKey/enabled — no secret) and the models proxy. Not for an
        // arbitrary remote page; a paired token or a trusted origin passes.
        || path.starts_with("/api/ai/") || crate::ai::plugin_completion::is_route(path) || crate::voice::plugin_session::is_route(path)
        || is_personal_path(path)
        // The Agent's model (engine or ChatGPT). Already under `/api/agent/`,
        // named so it stays gated if that prefix ever narrows.
        || path == "/api/agent/preferences"
        // Which models are loaded, the GPUs, the engines' address; their
        // catalog, the models chosen and the downloads.
        || path == "/api/engines"
        || path.starts_with("/api/engines/")
        // How far setup got, which plugins were chosen, what was accepted.
        || is_setup_path(path)
        // The MCP server (a GET is 405 behind this), the Agent's switch, and what it changed.
        || is_control_path(path)
}

/// Stricter allow-list for privileged endpoints: OAIY Desktop's OWN webview and
/// oaiy.com only — never an arbitrary localhost page. Loopback origins are
/// allowed in debug builds (the dev UI is served from a localhost port) but NOT
/// in a release build, which is what ships.
fn is_allowed_origin_privileged(origin: &str) -> bool {
    if origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
        || is_embedded_origin(origin)
    {
        return true;
    }
    if let Some(rest) = origin.strip_prefix("https://") {
        let host = rest.split('/').next().unwrap_or(rest);
        let host = host.split(':').next().unwrap_or(host);
        if host == "oaiy.com" || host.ends_with(".oaiy.com") {
            return true;
        }
    }
    #[cfg(debug_assertions)]
    if is_loopback_origin(origin) {
        return true;
    }
    false
}

/// Extract a `Bearer <token>` from the Authorization header, if present.
fn bearer_token(req: &Request) -> Option<String> {
    req.headers()
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.trim().to_owned())
}

/// Compare the configured token to the supplied one without short-circuiting on
/// the first differing byte (so it can't be recovered prefix-by-prefix via
/// timing). Length still differs early — acceptable for a loopback secret.
fn token_eq(want: &str, got: &str) -> bool {
    let (w, g) = (want.as_bytes(), got.as_bytes());
    if w.len() != g.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..w.len() {
        diff |= w[i] ^ g[i];
    }
    diff == 0
}

/// Auth config for the origin guard: an optional bearer `token` (the only key
/// for privileged routes on a headless server that has one set) plus `gui_mode`,
/// which the GUI companion sets so its trusted webview still reaches privileged
/// routes via the origin allow-list even when a token is ALSO configured -- so
/// the CLI can drive OAIY Desktop without locking out its own UI.
#[derive(Clone)]
struct AuthConfig {
    token: Option<String>,
    gui_mode: bool,
    /// Tokens minted by the pairing flow. A consumer that paired (the user
    /// approved it in the OAIY UI) presents one of these as its bearer — the
    /// production path for an untrusted-origin consumer like FormLogic Web. The
    /// guard checks it alongside the single configured `token`.
    pairing: Option<crate::bridge::PairingHandle>,
}

impl AuthConfig {
    /// Does `presented` match the configured token, this process's own internal
    /// token, OR a live paired token?
    fn token_matches(&self, presented: &str) -> bool {
        // An empty bearer is never a match — a caller that sent nothing must not
        // pass because something on this side is also unset.
        if presented.is_empty() {
            return false;
        }
        if let Some(want) = self.token.as_deref() {
            if token_eq(want, presented) {
                return true;
            }
        }
        // The credential this process hands its own children (the CLI running a
        // flow). Same trust as the parent, by construction — it was spawned by
        // it — and it has no browser origin the guard could recognise instead.
        let internal = crate::internal_token();
        if !internal.is_empty() && token_eq(internal, presented) {
            return true;
        }
        if let Some(p) = &self.pairing {
            if let Ok(mgr) = p.lock() {
                return mgr.is_valid_token(presented);
            }
        }
        false
    }
}

/// Decide whether a privileged request is allowed. A matching bearer token
/// always passes. The trusted-origin allow-list is honored ONLY for the GUI
/// companion (gui_mode), which has a real, unspoofable webview origin. A headless
/// server has no webview — any local process can forge the `Origin` header — so it
/// trusts the token alone: headless WITH a token is token-only, and headless with
/// NO token has its privileged (command-defining / destructive) routes CLOSED (the
/// operator must set OAIY_SERVER_TOKEN to administer it; the CLI sends the bearer).
fn privileged_allowed(token_ok: bool, gui_mode: bool, _has_token: bool, origin_priv_ok: bool) -> bool {
    token_ok || (gui_mode && origin_priv_ok)
}

/// Gate mutating/exec requests (POST/PUT/DELETE/PATCH) on the `Origin` header.
/// Privileged (command-defining / destructive) paths require OAIY Desktop's own
/// origin and fail CLOSED on a missing Origin; other mutations keep the broad
/// loopback allow-list. GET reads and CORS preflight (OPTIONS) pass through.
async fn origin_guard(
    State(auth): State<AuthConfig>,
    req: Request,
    next: Next,
) -> axum::response::Response {
    let m = req.method().clone();
    let path = req.uri().path();
    let public = m == Method::OPTIONS
        || (m == Method::POST && path == "/api/bridge/pairing")
        || ((m == Method::GET || m == Method::HEAD)
            && (path == "/api/health"
                // Which version runs and whether a newer release exists: what health half says already.
                // Open, so the headless server, which only reports a newer release, can be asked without a
                // token. (The desktop's listener does not take this exemption: see the restricted read below.)
                || path == UPDATE_STATUS_PATH
                || path == "/api/bridge/capabilities"
                || path.strip_prefix("/api/bridge/pairing/")
                    .is_some_and(|id| !id.is_empty() && !id.contains('/'))));
    // Default-deny on headless/network listeners, including new routes and
    // HEAD requests. Only discovery and user-approved pairing bootstrap are public.
    if !auth.gui_mode && !public
        && !bearer_token(&req).is_some_and(|got| auth.token_matches(&got))
    {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "authentication required"
        }))).into_response();
    }
    let mutating =
        m == Method::POST || m == Method::PUT || m == Method::DELETE || m == Method::PATCH;
    // A tiny public-mutation allow-list: raising a pairing request is a POST that
    // MUST be reachable from an untrusted origin — bootstrapping a token is its
    // whole purpose, and it grants nothing without the user's approval (a
    // separate, privileged act). Without this exemption the ordinary mutation
    // guard below would 403 FormLogic's very first call. Everything else stays
    // gated.
    if mutating && m == Method::POST && req.uri().path() == "/api/bridge/pairing" {
        return next.run(req).await;
    }
    if mutating {
        let privileged = is_privileged_path(&m, req.uri().path());
        let origin = req
            .headers()
            .get(ORIGIN)
            .and_then(|o| o.to_str().ok())
            .map(str::to_owned);
        // A configured bearer token lets a headless/non-browser admin client
        // (the CLI, oaiy-server tooling) perform privileged ops the origin
        // allow-list would otherwise block — there's no browser origin on a
        // server. Compared without per-byte short-circuit (token_eq).
        // A configured token OR a paired token satisfies auth.
        let token_ok = bearer_token(&req).is_some_and(|got| auth.token_matches(&got));
        let allowed = if privileged {
            let origin_priv_ok =
                matches!(origin.as_deref(), Some(o) if is_allowed_origin_privileged(o));
            privileged_allowed(token_ok, auth.gui_mode, auth.token.is_some(), origin_priv_ok)
        } else {
            // Origin is a browser CSRF check, never a headless credential.
            let origin_ok = match origin.as_deref() {
                Some(o) => is_allowed_origin(o),
                None => true, // native/CLI caller: no browser Origin to check
            };
            token_ok || (auth.gui_mode && origin_ok)
        };
        if !allowed {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "origin not allowed" })),
            )
                .into_response();
        }
    } else if m == Method::GET || m == Method::HEAD {
        // A missing Origin proves nothing about a caller's location or authority.
        // Sensitive reads require a token outside the loopback GUI listener.
        let path = req.uri().path();
        let export_read = is_export_path(path);
        // The update status says whether a phone call is live (it is why "Restart to update" is off), so on the
        // desktop it is read like the calls themselves are: by OAIY's own pages or with the token, never by a page
        // the owner happens to have open. The headless server computes no such thing and answers it openly.
        let restricted_read = is_restricted_read_path(path) || (auth.gui_mode && path == UPDATE_STATUS_PATH);
        if export_read || restricted_read {
            let origin = req
                .headers()
                .get(ORIGIN)
                .and_then(|o| o.to_str().ok())
                .map(str::to_owned);
            let token_ok = bearer_token(&req).is_some_and(|got| auth.token_matches(&got));
            let origin_ok = origin.as_deref().is_some_and(|o| {
                if export_read { is_allowed_origin_privileged(o) } else { is_allowed_origin(o) }
            });
            let allowed = token_ok || (auth.gui_mode && origin_ok);
            if !allowed {
                return (
                    StatusCode::FORBIDDEN,
                    Json(serde_json::json!({ "error": "origin not allowed" })),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}

/// What the access guard decides with: the old guard's settings (what `legacy` mode judges by) and the
/// new guard.
#[derive(Clone)]
struct AccessState {
    legacy: AuthConfig,
    guard: Arc<crate::auth::Guard>,
}

/// The one guard of the listener.
///
/// In `legacy` mode a request for a route that existed before the access model, and a request for a route
/// that does not exist, go to [`origin_guard`] exactly as they always did (the guard itself is
/// unchanged), with the owner as the principal for any handler that reads one. Only a route the model
/// adds (`since: 2` in `auth/routes.rs`) is judged by the new pipeline. In `scoped` and `shadow` mode the
/// new pipeline judges every request, and `origin_guard` is never called.
async fn access_guard(
    State(s): State<AccessState>,
    mut req: Request,
    next: Next,
) -> axum::response::Response {
    if req.uri().path() == "/api/health" {
        req.extensions_mut().insert(s.guard.health_extras());
    }
    if s.guard.claims(&req) {
        s.guard.handle(req, next).await
    } else {
        req.extensions_mut().insert(crate::auth::principal::Principal::legacy_owner());
        origin_guard(State(s.legacy), req, next).await
    }
}

/// A reserved listener keeps an isolated launch's selected port owned from
/// bootstrap through serving. Ordinary callers still pass their existing u16.
pub enum ServerEndpoint {
    Port(u16),
    Reserved(std::net::TcpListener),
}
impl From<u16> for ServerEndpoint {
    fn from(port: u16) -> Self { Self::Port(port) }
}

pub async fn serve(
    endpoint: impl Into<ServerEndpoint>,
    // The address to listen on: loopback unless the person running it said otherwise.
    //
    // Never inferred: an address other machines can reach widens the surface from "this machine" to "anything
    // that can route here", which is a decision for the person running it, not for us (`OAIY_SERVER_BIND`,
    // and the startup rules of `auth::exposure` that go with it). The desktop always passes loopback.
    bind: std::net::IpAddr,
    config: Arc<dyn ConfigProvider>,
    // Optional bearer token gating privileged routes for non-browser clients.
    auth_token: Option<String>,
    // GUI companion: also accept its trusted webview origin for privileged
    // routes, so configuring a token (for CLI access) doesn't lock out the UI.
    gui_mode: bool,
    registry: RegistryHandle,
    downloads: DownloadsHandle,
    python: PythonHandle,
    catalog: CatalogHandle,
    // Bridge Protocol v1 surface. Passed in rather than constructed here so the
    // GUI and the headless server can share one ledger with their own lifetimes.
    bridge: crate::bridge::BridgeState,
    // Companion device trust, and the upstream that brokers admissions for it.
    // Built by the caller rather than here: the plugin host answers
    // `companion.admission` from the SAME identity store these routes
    // administer, and a second store would let the two disagree about which
    // phones are trusted.
    companion: crate::companion::routes::CompanionHandle,
    companion_upstream: crate::companion::upstream::UpstreamHandle,
    // The one outbound account link, and the descriptors it can be made with.
    link: crate::link::LinkHandle,
    // AI gateway provider store (holds provider API keys). Built at the call site
    // where the data dir is known; the AI router pairs it with the registry below.
    ai_providers: crate::ai::providers::ProviderStoreHandle,
    // The ChatGPT connector (a managed codex child, keyed to its own CODEX_HOME).
    ai_codex: crate::ai::CodexHandle,
    // The Node runtime the bundled CLI runs under.
    node: crate::services::node_runtime::NodeHandle,
    // What is known of newer releases, shared with whatever else (the desktop's window and tray) asks.
    updater: crate::update::UpdaterHandle,
    // The access mode (`legacy` keeps every route that existed before the access model exactly as it was).
    access: crate::auth::AccessSettings,
) -> Result<(), BoxError> {
    let (port, reserved) = match endpoint.into() {
        ServerEndpoint::Port(port) => (port, None),
        ServerEndpoint::Reserved(listener) => {
            let address = listener.local_addr()?;
            if address.ip() != bind || !address.ip().is_loopback() || address.port() == 0 {
                return Err("reserved listener must match the loopback bind".into());
            }
            listener.set_nonblocking(true)?;
            (address.port(), Some(listener))
        }
    };
    let isolated = crate::isolated::active();
    let bind_all = !crate::auth::clientip::unmap(bind).is_loopback();
    // A server that passed the startup rules has been told what a network bind needs (an owner login, and a
    // named proxy behind a public URL): the bearer token is no stand-in for a login. Anything else that binds
    // beyond loopback still needs the token it always needed.
    if access.config.is_none() {
        validate_listener_auth(bind_all, auth_token.as_deref())?;
    }
    // The access guard: in `legacy` mode it holds nothing on disk; otherwise it opens `<data>/auth`.
    let data_dir_for_auth = registry
        .lock()
        .map(|r| r.data_dir().to_path_buf())
        .map_err(|_| BoxError::from("the registry is poisoned: the data folder is not known"))?;
    let guard = crate::auth::build_guard(
        &access,
        &data_dir_for_auth,
        port,
        bind_all,
        gui_mode,
        auth_token.clone(),
        &|name| std::env::var(name).ok(),
    )?;
    // The upkeep of the credential store runs in every mode: in `legacy` the store is memory-only, and
    // `derive` (a route the model adds, judged by the new guard even there) makes credentials that must not
    // pile up.
    tokio::spawn(crate::auth::runtime::maintain_forever(guard.clone()));
    // The web login (the server's `web` feature): a cookie session per app host, setup, the console. Only where
    // the store is on disk (an enforcing mode) and not in the desktop, which has no login. It refuses, as the
    // store does, a file it cannot read (a mangled or unreadable owner file is a startup error, never setup-only).
    #[cfg(feature = "web")]
    let login = if access.mode.is_enforcing() && !gui_mode && crate::auth::login::can_host(&guard) {
        let opts = match &access.config {
            Some(config) => crate::auth::login::LoginOptions::from_config(config, port),
            None => crate::auth::login::LoginOptions::production(&|name| std::env::var(name).ok(), port)?,
        };
        let state = crate::auth::login::enable(&guard, &data_dir_for_auth.join("auth"), opts)?;
        tokio::spawn(crate::auth::login::maintain_forever(state.clone()));
        // One banner with no secret in it: not through the log facade, which the journal and the ring keep.
        if let Some(banner) = state.banner() {
            eprintln!("{banner}");
        }
        Some(state)
    } else {
        None
    };
    // CORS stays permissive so a hosted oaiy-web at any domain can READ the
    // API (the localhost bind keeps non-local processes out). State-changing
    // and exec endpoints are additionally gated by `origin_guard` below, so a
    // random web page the user has open can't issue drive-by POST/DELETE
    // requests against the loopback API.
    let cors = cors_layer();

    /// Answer Chrome's Private Network Access preflight.
    ///
    /// A page on a public or `.local` address calling `127.0.0.1` is a
    /// private-network request. Chrome sends
    /// `Access-Control-Request-Private-Network: true` on the preflight and
    /// BLOCKS the real request unless the reply carries
    /// `Access-Control-Allow-Private-Network: true`. Ordinary CORS headers are
    /// not enough, and the failure shows up as a bare network error — which is
    /// exactly what "Loading services…" forever looks like.
    ///
    /// Only added when asked for, and only on the preflight, so nothing changes
    /// for the loopback and Tauri callers that never trigger PNA. The existing
    /// origin gate still decides what the follow-up request may actually do.
    async fn allow_private_network(
        req: axum::extract::Request,
        next: axum::middleware::Next,
    ) -> axum::response::Response {
        const REQUEST: &str = "access-control-request-private-network";
        const ALLOW: &str = "access-control-allow-private-network";
        let asked = req.method() == Method::OPTIONS
            && req.headers().get(REQUEST).and_then(|v| v.to_str().ok()) == Some("true");
        let mut resp = next.run(req).await;
        if asked {
            if let Ok(value) = axum::http::HeaderValue::from_str("true") {
                resp.headers_mut().insert(ALLOW, value);
            }
        }
        resp
    }

    // The AI sources union needs to read the services registry; clone the handle
    // before `registry` is moved into AppState below.
    let registry_for_ai = registry.clone();
    let registry_for_voice = registry.clone();
    let registry_for_plugin_voice = registry.clone();
    // The backup routes (its status, and the Agent page handing over its storage) work on the data folder.
    let backup_data_dir = data_dir_for_auth.clone();
    // The calendar lives in the data folder, beside the flows it may defer to.
    if let Ok(dir) = registry.lock().map(|r| r.data_dir().to_path_buf()) {
        crate::calendar::init(&dir);
        crate::voice::voices::init(&dir);
        // The contacts (and the names callers are greeted by): the older names file is upgraded now.
        crate::voice::contacts::init(&dir);
        // Transferring calls to the owner and taking messages: the owner's settings, all off until turned on.
        crate::ring::init(&dir);
        // What callers leave for the owner.
        crate::messages::init(&dir);
    }
    // The Agent's control API: its switch and its log live in the data folder too.
    // Whether the Agent may change OAIY when the person has not said (no `control.json`) depends on how the install
    // can be reached: on for a local one, as it has always been, off for a proxied or lan one (design 4.5.5).
    let control = crate::control::Control::with_exposure(
        &data_dir_for_auth,
        guard.config().exposure,
    );
    // Which modules (the phone, the calendar) a plugin provides: worked out now, then kept up to date.
    crate::modules::start(bridge.plugins.clone());
    // The setup wizard's record. Made now when there is none, so a desktop
    // already in use is recorded as set up before any window asks.
    let setup_routes = {
        let data_dir = registry.lock().map(|r| r.data_dir().to_path_buf()).ok();
        let plugins_root = bridge.plugins.lock().map(|r| r.root().to_path_buf()).ok();
        let providers = ai_providers.lock().map(|s| s.list()).unwrap_or_default();
        let data_dir = data_dir.unwrap_or_else(|| data_dir_for_auth.clone());
        let store = crate::setup::Store::open(&data_dir, || {
            crate::setup::in_use_signal(&data_dir, plugins_root.as_deref().unwrap_or(&data_dir.join("plugins")), &providers)
        });
        if let Some(why) = &store.state().first_run.migrated {
            log::info!("setup: first-run wizard recorded as finished ({why})");
        }
        crate::setup::router(crate::setup::Ctx::new(store, bridge.plugins.clone(), bridge.host.clone()))
    };

    let state = AppState {
        config,
        registry,
        downloads,
        python,
        catalog,
        node,
    };

    // Merged rather than inlined: the bridge owns its own state (the run ledger
    // + the plugin registry) and should not be threaded through AppState, which
    // every services/models/python handler would then carry for no reason.
    // Capture the pairing handle before `bridge` is moved into its router, so
    // the auth guard can validate paired tokens.
    let pairing_for_auth = bridge.pairing.clone();
    let companion_routes =
        crate::companion::routes::router(companion.clone(), companion_upstream.clone());
    // The calendar syncs with FormLogic while this desktop is linked to it.
    if !isolated { crate::calendar::sync::spawn(link.clone()); }
    let link_routes = crate::link::routes::router(link);
    // Calls answered by the agent: the app's side here, Aokie's on the voice gateway (17872).
    let voice = {
        let host = bridge.host.clone();
        crate::voice::VoiceHub::new(crate::voice::engines::Engines::new(registry_for_voice), move |call| {
            let events: Vec<serde_json::Value> = host.events_since(0, 500).into_iter().map(|e| e.envelope).collect();
            crate::voice::caller_from_events(&events, call)
        })
    };
    let voice_routes = crate::voice::app_router(voice.clone());
    let plugin_voice_routes = crate::voice::plugin_session::router(registry_for_plugin_voice, bridge.host.clone(), isolated);
    // Putting a caller through to the owner: which Companions could take the call, and whether the owner is at the computer
    // (only where there is a window to ring).
    if let Some(ring) = crate::ring::shared() {
        ring.set_devices(std::sync::Arc::new(crate::ring::devices::CompanionDevices::new(companion.clone(), "aokie")));
        bridge.host.set_ring(ring.clone());
        // A caller who asks for the owner by name is asking for the owner: the name is the business's, when it is named for a person.
        ring.set_names(std::sync::Arc::new(|| crate::calendar::shared().map(|c| crate::ring::phrases::owner_names(&c.settings().business)).unwrap_or_default()));
        if gui_mode && !isolated {
            ring.set_presence(std::sync::Arc::new(crate::ring::presence::IdlePresence::os(ring.settings.clone())));
        }
    }
    let ring_routes = crate::ring::routes::router(crate::ring::shared().unwrap_or_else(|| crate::ring::Ring::in_memory(Default::default())));
    let plugin_ai_host = bridge.host.clone();
    let bridge_routes = crate::bridge::bridge_router(bridge);

    // The AI gateway is its own sub-router with its own state (provider store +
    // registry clone), merged INSIDE the guard layers like the bridge — provider
    // CRUD and the credential-injecting chat proxy need the same origin/token gate.
    let ai_state = crate::ai::AiState::new(ai_providers, registry_for_ai, ai_codex);
    let plugin_ai_routes = crate::ai::plugin_completion::router(ai_state.clone(), plugin_ai_host, isolated);
    // Aokie's gateway (17872): calls, and a provider's chat for Aokie's own speech lanes.
    if !isolated { tokio::spawn(crate::voice::serve_gateway(voice, crate::ai::provider_chat_router(ai_state.clone()))); }
    let ai_routes = crate::ai::ai_router(ai_state).merge(plugin_ai_routes);

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/config", get(get_config))
        // services
        .route("/api/services", get(list_services).post(add_service))
        .route("/api/services/ensure-by-port", post(ensure_service_by_port))
        .route("/api/services/:id", delete(delete_service))
        .route("/api/services/:id/start", post(start_service))
        .route("/api/services/:id/stop", post(stop_service))
        .route("/api/services/:id/repair", post(repair_service))
        .route("/api/services/:id/autostart", post(set_service_autostart))
        .route("/api/services/:id/install", post(install_service))
        .route("/api/services/:id/uninstall", post(uninstall_service))
        .route(
            "/api/services/:id/cancel-install",
            post(cancel_install_service),
        )
        .route("/api/services/:id/logs", get(service_logs))
        .route("/api/services/:id/export", get(export_service))
        // models
        .route("/api/models", get(list_models))
        .route("/api/models/catalog", get(model_catalog))
        .route("/api/models/download", post(start_model_download))
        .route("/api/models/downloads", get(list_downloads))
        .route("/api/models/downloads/:id/pause", post(pause_download))
        .route("/api/models/downloads/:id/resume", post(resume_download))
        .route("/api/models/downloads/:id/cancel", post(cancel_download))
        .route("/api/models/:name", delete(delete_model))
        // python
        .route("/api/node", get(node_status))
        .route("/api/node/install", post(install_node))
        .route("/api/node/logs", get(node_logs))
        .route("/api/python", get(python_status))
        .route("/api/python/install", post(install_python))
        .route("/api/python/logs", get(python_logs))
        .route("/api/python/venvs", post(create_venv))
        .route("/api/python/venvs/:name", delete(delete_venv))
        .with_state(state)
        // Merged INSIDE the guard layers, not outside: the bridge's POST routes
        // reserve runs and invoke connector commands, so they need the same
        // origin/token gate as the services routes. Adding them after `.layer()`
        // would leave them ungated — reachable by any web page the user has open.
        .merge(bridge_routes)
        .merge(voice_routes)
        .merge(plugin_voice_routes)
        .merge(crate::voice::contacts::routes::router(crate::voice::contacts::shared()))
        .merge(ring_routes)
        .merge(crate::messages::routes::router(crate::messages::shared()))
        .merge(crate::calendar::routes::router())
        .merge(crate::modules::routes::router())
        .merge(crate::agent_tasks::router())
        // Whether a newer release exists: read-only status, and a rate-limited check (never a download or an install).
        .merge(crate::update::routes::router(updater))
        .merge(crate::backup::routes::router(backup_data_dir))
        .merge(setup_routes)
        .route("/api/engines", axum::routing::get(engines_status))
        .route("/api/engines/catalog", axum::routing::get(engines_catalog))
        .route("/api/engines/downloads", axum::routing::get(engines_downloads).post(engines_download_start))
        .merge(companion_routes)
        .merge(link_routes)
        .merge(ai_routes)
        // The MCP server and the Agent's switch: inside the guard, like everything else.
        .merge(crate::control::router(control.clone()))
        // What the access model has built so far: who am I, derive a credential, what the server saw.
        .merge(crate::auth::api::router(guard.clone()));
    // The login's routes (and the console's), inside the guard like everything else.
    #[cfg(feature = "web")]
    let app = match &login {
        Some(state) => app.merge(crate::auth::login::router(state.clone())),
        None => app,
    };
    // Inside the unchanged access/CORS guards, so qualification never grants
    // an otherwise unauthorized caller access to plugin installation or commands.
    let app = if isolated { app.layer(middleware::from_fn(crate::isolated_policy::guard)) } else { app };
    let access_state = AccessState {
        // A network listener must never trust a forgeable Origin, even
        // when launched by the GUI. Its clients must present a credential.
        legacy: AuthConfig { token: auth_token, gui_mode: gui_mode && !bind_all, pairing: Some(pairing_for_auth) },
        guard: guard.clone(),
    };
    let app = if access.mode.is_enforcing() {
        // The new guard alone, and CORS decided by what is paired (never `Any`).
        app.layer(middleware::from_fn_with_state(access_state, access_guard))
            .layer(middleware::from_fn_with_state(guard.clone(), crate::auth::guard::scoped_cors))
    } else {
        app.layer(middleware::from_fn_with_state(access_state, access_guard))
            .layer(cors)
            // OUTSIDE the CORS layer so it runs after it and can add to the
            // preflight response CORS produced.
            .layer(axum::middleware::from_fn(allow_private_network))
    };
    // The control tools call the routes above in-process, through this same
    // router and its gate (see `control/`).
    control.set_router(app.clone());

    let addr = SocketAddr::new(bind, port);
    let listener = match reserved {
        Some(listener) => tokio::net::TcpListener::from_std(listener)?,
        None => tokio::net::TcpListener::bind(addr).await?,
    };

    // Every bound address is logged, and the exposure with it (design 4.5.5).
    let bound = listener.local_addr().unwrap_or(addr);
    if bind_all {
        log::warn!(
            "OAIY API listening on http://{bound} ({} install) — reachable from the NETWORK, not just this machine",
            guard.config().exposure.name()
        );
    } else {
        log::info!(
            "OAIY API listening on http://{bound} ({} install)",
            guard.config().exposure.name()
        );
    }
    // The console's credential and the port it is on are written only now that the listener is bound: a console that
    // finds them finds a server that answers. (A clean exit removes them; the binary does that.)
    #[cfg(feature = "web")]
    if let Some(state) = &login {
        let bound = listener.local_addr().map_or(port, |a| a.port());
        if let Err(e) = crate::auth::console::publish(state, bound) {
            log::warn!(
                "{}",
                crate::auth::scrub::scrub_line(&format!(
                    "auth: the console cannot reach this server: {e}"
                ))
            );
        }
    }
    // What this install is, in words, with no secret in it (the audit log's startup event has it too): now that the
    // credential store is open and the listener is bound, so that "listening on" is true. One banner, not through the
    // log facade, which the journal and the ring keep. (A server that stops before this says why in one line.)
    if let Some(config) = &access.config {
        for line in config.banner_lines() {
            eprintln!("{line}");
        }
    }
    // With the peer's address on each request: the new guard needs it (a `desk` credential works only from
    // loopback, an address is what the failed-bearer throttle counts).
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

fn validate_listener_auth(bind_all: bool, token: Option<&str>) -> Result<(), BoxError> {
    if bind_all && token.is_none_or(|t| t.trim().is_empty()) {
        return Err("network binding requires a non-empty OAIY_SERVER_TOKEN; use loopback or configure authentication".into());
    }
    Ok(())
}

/// The CORS layer `serve` puts on everything: what a page on another origin (the dashboard at `localhost:17973` or `tauri://`, the Agent, a hosted
/// web app) may ask of this API. Every method those pages send must be in it, or the browser's preflight fails and the request is never made
/// (a Messages page that cannot mark a message seen): the test below walks every method the pages' sources send.
pub(crate) fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            // (The Messages page marks a message seen or handled, and the calendar changes an appointment, with PATCH.)
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(Any)
        // The dashboard (another origin) reads /api/modules' ETag to ask again with If-None-Match.
        .expose_headers([axum::http::header::ETAG])
}

/// `router` behind the same gate `serve` puts in front of everything, for the
/// tests of other modules (the control tools' in-process calls go through it).
#[cfg(test)]
pub(crate) fn guarded_for_tests(router: Router, token: Option<String>, gui_mode: bool) -> Router {
    router.layer(middleware::from_fn_with_state(AuthConfig { token, gui_mode, pairing: None }, origin_guard))
}

/// A frozen copy of the guard as it was before the access model, and the differential test that holds the live
/// guard to it in `legacy` mode.
#[cfg(test)]
mod frozen_guard;
#[cfg(test)]
mod legacy_neutrality;

#[cfg(test)]
mod tests {
    use super::{
        is_allowed_origin, is_allowed_origin_privileged, is_privileged_path, is_restricted_read_path,
        HealthResponse, API_VERSION, BRIDGE_PROTOCOL, PRODUCT_ID,
        privileged_allowed, AuthConfig, ModelDownloadRequest,
    };
    use axum::http::Method;

    #[test]
    fn token_eq_compares_the_whole_token() {
        let want = "0123456789abcdef0123456789abcdef";
        assert!(super::token_eq(want, want));
        // The same length, one character different at the start, in the middle and at the end; every character different.
        for wrong in ["1123456789abcdef0123456789abcdef", "0123456789abcdee0123456789abcdef", "0123456789abcdef0123456789abcdee", "fedcba9876543210fedcba9876543210", "00000000000000000000000000000000"] {
            assert!(!super::token_eq(want, wrong), "{wrong}");
        }
        // Other lengths: a prefix, a longer one, and none.
        for wrong in ["0123456789abcdef0123456789abcde", "0123456789abcdef0123456789abcdef0", "", "0"] {
            assert!(!super::token_eq(want, wrong), "{wrong:?}");
        }
        assert!(!super::token_eq(want, ""));
        // Two empty ones are equal: a caller refuses an empty token itself (see `token_matches`).
        assert!(super::token_eq("", ""));
    }

    #[test]
    fn network_listener_requires_a_nonempty_token() {
        for token in [None, Some(""), Some("  ")] {
            assert!(super::validate_listener_auth(true, token).is_err());
        }
        assert!(super::validate_listener_auth(true, Some("secret")).is_ok());
        assert!(super::validate_listener_auth(false, None).is_ok());
    }

    #[tokio::test]
    async fn network_guard_rejects_missing_or_forged_origins_and_accepts_bearer() {
        use axum::{middleware, routing::any, Router};
        for token in [None, Some("test-admin".to_string())] {
            let expected_authenticated = if token.is_some() { 200 } else { 403 };
            let app = Router::new()
                .fallback(any(|| async { "allowed" }))
                .layer(middleware::from_fn_with_state(
                    AuthConfig { token, gui_mode: false, pairing: None },
                    super::origin_guard,
                ));
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let client = reqwest::Client::new();
            for (method, path) in [
                (Method::GET, "/api/config"),
                (Method::HEAD, "/api/bridge/runs"),
                (Method::GET, "/api/services/private/export"),
                (Method::GET, "/api/bridge/flows/private"),
                (Method::POST, "/api/models/download"),
                (Method::POST, "/api/plugins/install"),
                (Method::POST, "/api/bridge/pairing/id/approve"),
            ] {
                for origin in [None, Some("https://oaiy.com"), Some("tauri://localhost"), Some("http://localhost:3000")] {
                    let mut request = client.request(method.clone(), format!("{base}{path}"));
                    if let Some(origin) = origin { request = request.header("Origin", origin); }
                    assert_eq!(request.send().await.unwrap().status(), 403, "{method} {path} {origin:?}");
                }
            }
            for path in ["/api/health", "/api/bridge/capabilities", "/api/bridge/pairing/id"] {
                assert_eq!(client.get(format!("{base}{path}")).send().await.unwrap().status(), 200);
            }
            let response = client.get(format!("{base}/api/config"))
                .bearer_auth("test-admin").send().await.unwrap();
            // The no-token configuration remains closed even for a guessed bearer.
            assert_eq!(response.status(), expected_authenticated);
            server.abort();
        }
    }

    #[test]
    fn health_uses_the_field_names_consumers_actually_read() {
        // A desktop-detection probe matches on `companion` and gates on
        // `apiVersion`. Ship either under another name and health still returns
        // a cheerful 200 while the consumer concludes there is no desktop, or
        // one too old to talk to.
        let body = serde_json::to_value(HealthResponse {
            status: "ok",
            product: PRODUCT_ID,
            companion: PRODUCT_ID,
            protocol: BRIDGE_PROTOCOL,
            version: "0.0.0-test",
            api_version: API_VERSION,
            plugin_api_version: 1,
        })
        .unwrap();
        assert_eq!(body["companion"], "oaiy-desktop");
        assert_eq!(body["apiVersion"], 1);
        assert_eq!(body["pluginApiVersion"], 1);
        // The long-standing names must not have moved underneath anyone.
        assert_eq!(body["product"], "oaiy-desktop");
        assert_eq!(body["status"], "ok");
        assert!(body.get("api_version").is_none(), "snake_case must not leak");
    }

    #[test]
    fn the_windows_origins_are_the_two_pages_in_their_systems_form() {
        assert_eq!(
            super::embedded_window_origins(true),
            ["http://oaiy.localhost", "http://oaiyflows.localhost"],
            "WebView2 maps a custom scheme to http://<scheme>.localhost"
        );
        assert_eq!(super::embedded_window_origins(false), ["oaiy://localhost", "oaiyflows://localhost"]);
        // The desktop's own rule for "one of OAIY's windows" knows every one of them.
        for windows in [true, false] {
            for origin in super::embedded_window_origins(windows) {
                assert!(super::is_embedded_origin(&origin), "{origin}");
            }
        }
    }

    #[test]
    fn the_linked_providers_origin_is_trusted_and_nothing_else_new_is() {
        // Linking is the user approving that provider, and its web app is where
        // they expect to manage this desktop from. Derived from the link so no
        // address is hardcoded — and so unlinking withdraws it.
        assert!(!is_allowed_origin("http://formlogic.local"), "untrusted before linking");
        crate::link::set_linked_origin_for_tests(Some("http://formlogic.local"));
        assert!(is_allowed_origin("http://formlogic.local"));
        // Exact origin only: a sibling host or a different scheme/port is a
        // different origin and must not inherit the trust.
        assert!(!is_allowed_origin("http://evil.formlogic.local"));
        assert!(!is_allowed_origin("https://formlogic.local"));
        assert!(!is_allowed_origin("http://formlogic.local:8080"));
        // The trust is the broad kind only: a page of the linked provider is not one of OAIY's own windows, so the
        // routes that define code or destroy data (and `/api/relay/*`, design 4.16.7) do not take it.
        assert!(!is_allowed_origin_privileged("http://formlogic.local"));

        crate::link::set_linked_origin_for_tests(None);
        assert!(!is_allowed_origin("http://formlogic.local"), "unlinking withdraws it");
    }

    #[test]
    fn a_relay_is_never_a_trusted_web_origin_however_it_is_linked() {
        // Design 4.16.6. The relay's address is kept in a type of its own, `relay::link_store::RelayLink`, which no rule
        // of this file reads: a relay is a place this desktop sends its token to, not a page that may drive it. So a relay
        // enrolled at an address is no more trusted than the same address was before, by either rule, whatever the port
        // or the scheme, and enrolling one does not touch the cell that trusts the provider's origin.
        let dir = std::env::temp_dir().join(format!("oaiy-http-relay-origin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let relay = crate::relay::link_store::RelayStore::open(&dir);
        relay
            .set_relay(crate::relay::link_store::RelayLink {
                relay_url: "https://relay.example.com".into(),
                relay_id: "rly-1".into(),
                relay_thumbprint: "t".repeat(43),
                device_id: "dev-1".into(),
                token: "oaiyrt1.TOKEN".into(),
                name: "Reception PC".into(),
                enrolled_at: chrono::Utc::now(),
                calibration: None,
                other: Default::default(),
            })
            .unwrap();
        for origin in ["https://relay.example.com", "https://relay.example.com:443", "https://relay.example.com:8443", "http://relay.example.com", "http://relay.local"] {
            assert!(!is_allowed_origin(origin), "{origin}");
            assert!(!is_allowed_origin_privileged(origin), "{origin}");
        }
        assert_ne!(crate::link::linked_origin().as_deref(), Some("https://relay.example.com"), "enrolling a relay is not linking a provider");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn linking_is_privileged_and_the_status_is_a_restricted_read() {
        // Starting a link opens the user's browser and ends in a stored
        // credential; unlinking throws it away. A local page must reach
        // neither. Reading the status changes nothing, but it inventories what
        // this machine can reach, so it stays off the open GET surface.
        assert!(is_privileged_path(&Method::POST, "/api/link/start"));
        assert!(is_privileged_path(&Method::POST, "/api/link/cancel"));
        assert!(is_privileged_path(&Method::DELETE, "/api/link"));
        assert!(is_restricted_read_path("/api/link"));
    }

    #[test]
    fn clearing_run_history_is_privileged() {
        // DELETE on the runs collection destroys the record of what this
        // machine has done. It takes the same gate as creating a run, so a
        // local page cannot erase the evidence of one.
        assert!(is_privileged_path(&Method::DELETE, "/api/bridge/runs"));
        // Calls and the calendar hold people's numbers, words and appointments.
        assert!(is_privileged_path(&Method::POST, "/api/voice/calls/c1/say"));
        assert!(is_privileged_path(&Method::POST, "/api/calendar/appointments"));
        assert!(is_privileged_path(&Method::PATCH, "/api/calendar/appointments/a1"));
        assert!(is_privileged_path(&Method::DELETE, "/api/calendar/appointments/a1"));
        assert!(is_privileged_path(&Method::PUT, "/api/calendar/settings"));
        assert!(is_restricted_read_path("/api/voice/events"));
        assert!(is_restricted_read_path("/api/calendar"));
        assert!(is_restricted_read_path("/api/calendar/free"));
        assert!(!is_restricted_read_path("/api/calendars-elsewhere"));
        assert!(is_privileged_path(&Method::POST, "/api/bridge/runs"));
        // Reading history stays a restricted read, not a privileged one.
        assert!(is_restricted_read_path("/api/bridge/runs"));
    }

    #[test]
    fn the_contacts_are_restricted_reads_and_their_changes_privileged() {
        // The person's notes about the people who ring, and what the
        // receptionist remembered: gated like the calls and the calendar.
        for path in ["/api/contacts", "/api/contacts/0491570006", "/api/contacts/export.csv", "/api/contacts/%2B61491570006"] {
            assert!(is_restricted_read_path(path), "{path} must be a restricted read");
        }
        for (m, path) in [
            (Method::PUT, "/api/contacts/0491570006"),
            (Method::DELETE, "/api/contacts/0491570006"),
            (Method::POST, "/api/contacts/0491570006/facts"),
            (Method::DELETE, "/api/contacts/0491570006/facts/0"),
            (Method::POST, "/api/contacts/import"),
        ] {
            assert!(is_privileged_path(&m, path), "{m} {path} must be privileged");
        }
        assert!(!is_restricted_read_path("/api/contactsx"));
        assert!(!is_privileged_path(&Method::POST, "/api/contacts-elsewhere"));
    }

    #[test]
    fn the_rings_settings_and_the_messages_are_restricted_reads_and_their_changes_privileged() {
        // Whom the receptionist may put through, the owner's VIP numbers, and what callers said to leave.
        for path in ["/api/ring/settings", "/api/ring/active", "/api/messages", "/api/messages/msg_1"] {
            assert!(is_restricted_read_path(path), "{path} must be a restricted read");
        }
        for (m, path) in [
            (Method::PUT, "/api/ring/settings"),
            (Method::POST, "/api/ring/active/assist_1/respond"),
            (Method::POST, "/api/ring/notices/notice_1/dismiss"),
            (Method::PATCH, "/api/messages/msg_1"),
            (Method::DELETE, "/api/messages/msg_1"),
            (Method::POST, "/api/messages"),
            (Method::POST, "/api/voice/calls/call_1/message"),
        ] {
            assert!(is_privileged_path(&m, path), "{m} {path} must be privileged");
        }
        assert!(!is_restricted_read_path("/api/ringing") && !is_restricted_read_path("/api/messagesx"));
        assert!(!is_privileged_path(&Method::POST, "/api/ringing"));
    }

    /// The access model's table (`auth/routes.rs`) has a row for each route of the receptionist's transfers and messages, in the scope of its kind
    /// (reading callers' words and numbers is `calls.read`, as the call events are; the receptionist's own `take_message` is `calls.write`, as `say` and
    /// `finish` are; acting on a message or a ring is `calls.manage` and the owner's transfer settings are `calls.settings`, which the Agent page's
    /// preset holds neither of), with `since: 1` (the routes exist, and `legacy` mode keeps the guard they were built with), and the
    /// scoped CORS answers a paired page's preflight for each method, PATCH and DELETE included, and no page that has not paired.
    #[test]
    fn the_receptionists_routes_take_the_scope_of_their_kind_and_the_scoped_cors_lets_a_paired_page_use_them() {
        use crate::auth::routes::{pattern_existed_before, route_class, Class};
        let paired: std::collections::BTreeSet<String> = ["https://app.oaiy.com".to_string()].into();
        let routes = [
            (Method::GET, "/api/messages", "calls.read"),
            (Method::GET, "/api/messages/:id", "calls.read"),
            (Method::PATCH, "/api/messages/:id", "calls.manage"),
            (Method::DELETE, "/api/messages/:id", "calls.manage"),
            (Method::GET, "/api/ring/settings", "calls.settings"),
            (Method::PUT, "/api/ring/settings", "calls.settings"),
            (Method::GET, "/api/ring/preview", "calls.read"),
            (Method::GET, "/api/ring/active", "calls.read"),
            (Method::POST, "/api/ring/active/:id/respond", "calls.manage"),
            (Method::POST, "/api/ring/notices/:id/dismiss", "calls.manage"),
            (Method::POST, "/api/voice/calls/:id/message", "calls.write"),
        ];
        for (method, pattern, scope) in &routes {
            assert_eq!(route_class(method, pattern), Class::Scope(scope), "{method} {pattern}");
            assert!(pattern_existed_before(pattern), "{pattern}: the guard these routes were built with is the one `legacy` mode keeps");
            let cors = crate::auth::cors::decide(method, Some(pattern), Some("https://app.oaiy.com"), &paired, false);
            assert_eq!(cors.get("access-control-allow-origin"), Some("https://app.oaiy.com"), "{method} {pattern}: a paired page's preflight is answered");
            assert!(cors.get("access-control-allow-methods").is_some_and(|m| m.split(", ").any(|m| m == method.as_str())), "{method} {pattern}: {:?}", cors.get("access-control-allow-methods"));
            assert!(crate::auth::cors::decide(method, Some(pattern), Some("https://evil.example"), &paired, false).is_empty(), "{method} {pattern}: a page that has not paired gets nothing");
        }
        // Every route this module's routers register is one of those (a route added to them needs its row and its line here).
        let scanned = crate::auth::route_coverage::scan_main_router().found;
        for f in scanned.iter().filter(|f| f.file.starts_with("ring/routes.rs") || f.file.starts_with("messages/routes.rs")) {
            assert!(routes.iter().any(|(m, p, _)| m.as_str() == f.verb.as_str() && *p == f.pattern), "{} {} ({}:{}) has no line in this test: is its row in `auth/routes.rs` the scope of its kind?", f.verb.as_str(), f.pattern, f.file, f.line);
        }
    }

    /// The encrypted backup's eight routes (`backup/routes.rs`) have a row each in the access table, and each is `since: 1` (the routes exist, and `legacy`
    /// mode keeps the guard they were built with, the lines of `is_backup_path`): the status is `system.read`, beside the update status, and the seven of
    /// the Agent page's hand-over of its own storage are `agent.serve`, which the `agent` and `owner` presets hold and no other does; none of the
    /// scopes is dangerous (none of the routes restores or overwrites a setting: what a restore brings back is prepared from files the owner looked at and
    /// ticked, and applied by the desktop at its next start). The scoped CORS answers a paired page's preflight for each method and no page that has not
    /// paired. Every route `backup/routes.rs` registers is one of those, and every one of those is registered there.
    #[test]
    fn the_backups_routes_take_the_scope_of_their_kind_and_the_scoped_cors_lets_a_paired_page_use_them() {
        use crate::auth::presets::{Preset, ALL_PRESETS};
        use crate::auth::routes::{pattern_existed_before, route_class, Class};
        use crate::auth::scopes::ScopeSet;
        let paired: std::collections::BTreeSet<String> = ["https://app.oaiy.com".to_string()].into();
        let routes = [
            (Method::GET, "/api/backup/status", "system.read"),
            (Method::POST, "/api/backup/agent/:id/part", "agent.serve"),
            (Method::POST, "/api/backup/agent/:id/done", "agent.serve"),
            (Method::GET, "/api/backup/agent-import", "agent.serve"),
            (Method::GET, "/api/backup/agent-import/:id/part/:index", "agent.serve"),
            (Method::POST, "/api/backup/agent-import/:id/undo-part", "agent.serve"),
            (Method::POST, "/api/backup/agent-import/:id/undo-done", "agent.serve"),
            (Method::POST, "/api/backup/agent-import/:id/done", "agent.serve"),
        ];
        for (method, pattern, scope) in &routes {
            assert_eq!(route_class(method, pattern), Class::Scope(scope), "{method} {pattern}");
            assert!(pattern_existed_before(pattern), "{pattern}: the guard these routes were built with is the one `legacy` mode keeps");
            assert!(!ScopeSet::of(&[*scope]).has_dangerous(), "{scope}: no route of the backup is behind a dangerous scope");
            let cors = crate::auth::cors::decide(method, Some(pattern), Some("https://app.oaiy.com"), &paired, false);
            assert_eq!(cors.get("access-control-allow-origin"), Some("https://app.oaiy.com"), "{method} {pattern}: a paired page's preflight is answered");
            assert!(cors.get("access-control-allow-methods").is_some_and(|m| m.split(", ").any(|m| m == method.as_str())), "{method} {pattern}: {:?}", cors.get("access-control-allow-methods"));
            assert!(crate::auth::cors::decide(method, Some(pattern), Some("https://evil.example"), &paired, false).is_empty(), "{method} {pattern}: a page that has not paired gets nothing");
        }
        // Who holds them: the Agent's hand-over is for the Agent page and the owner alone, the status is for whoever may read the system.
        let holders = |scope: &str| ALL_PRESETS.iter().copied().filter(|p| p.scopes().contains(scope)).collect::<Vec<Preset>>();
        assert_eq!(holders("agent.serve"), [Preset::Owner, Preset::Agent]);
        assert_eq!(holders("system.read"), [Preset::Owner, Preset::Cli, Preset::CliAdmin, Preset::Readonly]);
        // The routes of the table are the routes of the module, both ways (a route added to `backup/routes.rs` needs its row and its line here).
        let ours: std::collections::BTreeSet<(String, String)> = routes.iter().map(|(m, p, _)| (m.as_str().to_string(), p.to_string())).collect();
        let module: std::collections::BTreeSet<(String, String)> = crate::backup::routes::ROUTES.iter().map(|(m, p)| (m.to_string(), p.to_string())).collect();
        assert_eq!(ours, module, "the routes `backup/routes.rs` lists are the ones this test holds to the table");
        let scanned = crate::auth::route_coverage::scan_main_router().found;
        let registered: std::collections::BTreeSet<(String, String)> = scanned.iter().filter(|f| f.file.starts_with("backup/routes.rs")).map(|f| (f.verb.as_str().to_string(), f.pattern.clone())).collect();
        assert_eq!(registered, ours, "the routes `backup/routes.rs` registers are the ones this test holds to the table");
    }

    /// Every HTTP method the dashboard's and the Agent's sources send to this API (`method: 'PATCH'`; tests excluded). Read at test time, so a
    /// verb added to a page later is held to the layer with no edit here.
    fn methods_the_pages_send() -> std::collections::BTreeMap<String, Vec<String>> {
        use std::path::{Path, PathBuf};
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(read) = std::fs::read_dir(dir) else { return };
            for entry in read.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().to_string();
                if path.is_dir() {
                    if !matches!(name.as_str(), "node_modules" | "dist" | "target" | "tests" | "__tests__" | "e2e") {
                        walk(&path, out);
                    }
                } else if (name.ends_with(".ts") || name.ends_with(".tsx")) && !name.contains(".test.") && !name.ends_with(".d.ts") {
                    out.push(path);
                }
            }
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join("..");
        let mut files = Vec::new();
        // (The dashboard and the Agent. Not the flow UI in `platform/ui`, whose `method` is a variable of a service it probes or a step it runs,
        // and whose `'HEAD'` is a comparison or a request to some other server.)
        for pages in ["platform/desktop/src", "app/src"] {
            walk(&root.join(pages), &mut files);
        }
        assert!(files.len() > 20, "the pages' sources were not found under {root:?}: {} files", files.len());
        let verb = regex::Regex::new(r#"method:\s*['"`](GET|POST|PUT|PATCH|DELETE|HEAD)['"`]"#).unwrap();
        let mut found: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap_or_default();
            for cap in verb.captures_iter(&text) {
                found.entry(cap[1].to_string()).or_default().push(file.strip_prefix(&root).unwrap_or(&file).display().to_string());
            }
        }
        found
    }

    /// A real preflight, as a browser sends it before a request from another origin, against the layer `serve` uses: what it answers is what
    /// the browser reads to decide whether the request may be made at all.
    #[tokio::test]
    async fn the_cors_layer_lets_every_method_the_pages_send_through_a_real_preflight() {
        use axum::body::Body;
        use axum::http::Request;
        use axum::{routing::any, Router};
        use tower::ServiceExt;
        let app = Router::new().fallback(any(|| async { "ok" })).layer(super::cors_layer());
        let found = methods_the_pages_send();
        assert!(found.contains_key("POST") && found.contains_key("DELETE"), "the walk found the pages' verbs: {:?}", found.keys().collect::<Vec<_>>());
        for origin in ["http://localhost:17973", "tauri://localhost", "http://tauri.localhost", "https://app.oaiy.com"] {
            for (verb, files) in &found {
                let request = Request::builder()
                    .method("OPTIONS")
                    .uri("/api/anything")
                    .header("Origin", origin)
                    .header("Access-Control-Request-Method", verb.as_str())
                    .header("Access-Control-Request-Headers", "content-type,authorization")
                    .body(Body::empty())
                    .unwrap();
                let response = app.clone().oneshot(request).await.unwrap();
                assert!(response.status().is_success(), "preflight of {verb} from {origin}: {}", response.status());
                let allowed = response.headers().get("access-control-allow-methods").and_then(|v| v.to_str().ok()).unwrap_or("");
                let allowed: Vec<&str> = allowed.split(',').map(str::trim).collect();
                assert!(
                    allowed.contains(&verb.as_str()) || allowed.contains(&"*"),
                    "{verb} is sent by {files:?} but the layer allows only {allowed:?}: the browser refuses the request after the preflight"
                );
                assert!(response.headers().get("access-control-allow-origin").is_some(), "{origin}");
            }
        }
    }

    #[test]
    fn a_page_on_loopback_is_trusted_with_a_privileged_route_by_a_debug_build_only_and_a_page_that_only_ends_like_ours_never() {
        // What ships (a release build) trusts OAIY's own window and oaiy.com; a debug build (the dev UI is served from a loopback port,
        // and the owner's `tauri dev` is one) trusts any loopback page too. The routes for the ring and the messages take that
        // gate like every route of their class, so this is what stands between them and a page the owner has open.
        for origin in ["http://localhost:3000", "http://127.0.0.1:5173"] {
            assert_eq!(is_allowed_origin_privileged(origin), cfg!(debug_assertions), "{origin}");
        }
        for origin in ["tauri://localhost", "https://oaiy.com", "https://app.oaiy.com"] {
            assert!(is_allowed_origin_privileged(origin), "{origin}");
        }
        for origin in ["https://evil.example", "null", "https://oaiy.com.evil.example", "https://evil.example/oaiy.com", "https://notoaiy.com", "http://oaiy.com", "tauri://localhost.evil.example"] {
            assert!(!is_allowed_origin_privileged(origin), "{origin}");
        }
    }

    #[test]
    fn companion_routes_are_privileged() {
        // These decide which phones may carry a live call's audio, and rotation
        // invalidates every existing pairing. A local web page must not reach
        // them just by being on loopback.
        // The GET is a RESTRICTED READ rather than privileged: it exposes the
        // roster and the desktop thumbprint, which an attacker would want in
        // order to imitate a pairing screen, but reading it changes nothing.
        assert!(is_restricted_read_path("/api/companion/aokie/pairing"));
        assert!(is_privileged_path(&Method::POST, "/api/companion/aokie/pairing/offers"));
        assert!(is_privileged_path(&Method::POST, "/api/companion/aokie/pairing/responses"));
        assert!(is_privileged_path(
            &Method::POST,
            "/api/companion/aokie/pairing/approvals/approval-1/approve"
        ));
        assert!(is_privileged_path(&Method::DELETE, "/api/companion/aokie/mobiles/abc"));
        assert!(is_privileged_path(&Method::POST, "/api/companion/aokie/identity/rotate"));
    }

    #[test]
    fn dead_letter_routes_are_gated_like_the_rest_of_the_bridge() {
        // Redrive re-dispatches a stored event through the trigger path and
        // RESERVES RUNS the worker executes — the same power as POST /runs, so
        // it belongs on the strict gate rather than the broad loopback
        // allow-list any local page can reach.
        assert!(is_privileged_path(&Method::POST, "/api/bridge/deadletters/dl_1/redrive"));
        // Deleting one destroys the only durable record that an event was lost.
        assert!(is_privileged_path(&Method::DELETE, "/api/bridge/deadletters/dl_1"));
        // And a dead letter stores the WHOLE envelope, which is exactly what
        // /api/bridge/events is already restricted for.
        assert!(is_restricted_read_path("/api/bridge/deadletters"));
    }

    #[test]
    fn a_download_request_actually_reads_its_checksum_field() {
        // Regression: the struct had no camelCase rename, so `expectedSha256`
        // deserialised to None and the download ran UNVERIFIED while the caller
        // believed it had asked for verification. Caught only by watching a
        // deliberately-wrong digest complete successfully against a live app.
        let digest = "0000000000000000000000000000000000000000000000000000000000000000";
        let req: ModelDownloadRequest = serde_json::from_str(&format!(
            r#"{{"url":"https://huggingface.co/x/resolve/main/m.gguf","expectedSha256":"{digest}"}}"#
        ))
        .expect("a well-formed body parses");
        assert_eq!(req.expected_sha256.as_deref(), Some(digest));
    }

    #[test]
    fn a_misspelled_checksum_field_is_refused_rather_than_ignored() {
        // The failure mode this whole feature exists to prevent is believing a
        // check happened when it did not — so an unknown field is a 400.
        assert!(serde_json::from_str::<ModelDownloadRequest>(
            r#"{"url":"https://huggingface.co/x/resolve/main/m.gguf","expected_sha256":"abc"}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ModelDownloadRequest>(
            r#"{"url":"https://huggingface.co/x/resolve/main/m.gguf","sha256":"abc"}"#
        )
        .is_err());
    }

    #[test]
    fn the_ordinary_download_body_still_parses() {
        // The rename must not break what the app itself sends.
        let req: ModelDownloadRequest = serde_json::from_str(
            r#"{"url":"https://huggingface.co/x/resolve/main/m.gguf","filename":"m.gguf","subdir":"qwen"}"#,
        )
        .expect("the app's own body parses");
        assert_eq!(req.filename.as_deref(), Some("m.gguf"));
        assert_eq!(req.subdir.as_deref(), Some("qwen"));
        assert_eq!(req.expected_sha256, None);
    }

    #[test]
    fn bridge_exec_routes_are_privileged() {
        // The blocker: the whole bridge exec surface was on the broad
        // loopback-permissive allow-list, so any local web page could PUT a flow
        // and POST /runs to execute it (RCE), or POST a connector command to send
        // an SMS. These must take the strict origin check.
        for (m, path) in [
            (Method::POST, "/api/bridge/runs"),
            (Method::POST, "/api/bridge/runs/run_1/claim"),
            (Method::POST, "/api/bridge/runs/run_1/finish"),
            (Method::POST, "/api/bridge/runs/run_1/cancel"),
            (Method::PUT, "/api/bridge/flows/pwn"),
            (Method::DELETE, "/api/bridge/flows/pwn"),
            (Method::POST, "/api/bridge/triggers"),
            (Method::DELETE, "/api/bridge/triggers/b1"),
            (Method::POST, "/api/bridge/connectors/aokie/request"),
            (Method::POST, "/api/plugins/aokie/start"),
            (Method::POST, "/api/plugins/aokie/stop"),
            (Method::POST, "/api/plugins/aokie/enabled"),
            // Trusting a package nobody signed is a trust act, like approving a pairing.
            (Method::POST, "/api/plugins/aokie/trust"),
            // Installing/removing native code is the most dangerous of the lot.
            (Method::POST, "/api/plugins/install"),
            (Method::DELETE, "/api/plugins/aokie"),
        ] {
            assert!(
                is_privileged_path(&m, path),
                "{m} {path} must be privileged (exec/side-effect surface)"
            );
        }
    }

    #[test]
    fn pairing_bootstrap_is_open_but_granting_is_privileged() {
        use axum::http::Method;
        // Raising and polling a pairing request must be OPEN — their whole job is
        // to let an untrusted consumer bootstrap, and neither grants anything.
        assert!(!is_privileged_path(&Method::POST, "/api/bridge/pairing"), "raising a request is open");
        assert!(!is_restricted_read_path("/api/bridge/pairing/pair_abc"), "polling is open");
        // Granting/denying/revoking IS the trust act — privileged.
        assert!(is_privileged_path(&Method::POST, "/api/bridge/pairing/pair_abc/approve"));
        assert!(is_privileged_path(&Method::POST, "/api/bridge/pairing/pair_abc/deny"));
        assert!(is_privileged_path(&Method::DELETE, "/api/bridge/pairings/tok_abc"), "revoke is privileged");
        // Seeing WHO is asking, and which apps are paired, is the UI's alone.
        assert!(is_restricted_read_path("/api/bridge/pairing"), "the pending list is gated");
        assert!(is_restricted_read_path("/api/bridge/pairings"), "the paired list is gated");
    }

    #[test]
    fn ai_gateway_is_gated_like_the_exec_surface() {
        use axum::http::Method;
        // Provider config + credential admin AND the credential-spending gateway
        // are all privileged for mutations — an anonymous local page must not
        // reconfigure a provider, plant a key, or spend the user's API credits.
        for (m, path) in [
            (Method::POST, "/api/ai/providers"),
            (Method::DELETE, "/api/ai/providers/openai"),
            (Method::POST, "/api/ai/providers/openai/key"),
            (Method::POST, "/api/ai/providers/openai/test"),
            (Method::POST, "/api/ai/v1/chat/completions"),
            (Method::POST, "/api/ai/providers/openai/v1/chat/completions"),
            (Method::POST, "/api/plugins/probe/ai/complete"),
            (Method::POST, "/api/plugins/%70robe/ai/complete"),
            (Method::POST, "/api/plugins/probe/ai/cancel"),
            (Method::POST, "/api/plugins/%70robe/voice/transcribe"),
            (Method::POST, "/api/plugins/probe/voice/speak"),
            (Method::POST, "/api/plugins/probe/voice/cancel"),
            (Method::POST, "/api/plugins/probe/voice/open"),
            (Method::POST, "/api/plugins/probe/voice/close"),
        ] {
            assert!(is_privileged_path(&m, path), "{m} {path} must be privileged");
        }
        // The reads (sources union, provider listing, models) are restricted —
        // gated against an arbitrary remote page, open to a paired token / trusted origin.
        for path in [
            "/api/ai/sources",
            "/api/ai/providers",
            "/api/ai/v1/models",
            "/api/ai/providers/openai/v1/models",
            "/api/plugins/probe/ai/sources",
            "/api/plugins/%70robe/ai/sources",
            "/api/plugins/%70robe/voice/status",
        ] {
            assert!(is_restricted_read_path(path), "{path} must be a restricted read");
        }
    }

    #[tokio::test]
    async fn scoped_plugin_ai_rejects_missing_or_remote_origins_including_encoded_ids() {
        use axum::{body::Body, http::Request, middleware, routing::any, Router};
        use tower::ServiceExt;
        let app = Router::new().fallback(any(|| async { "called" })).layer(middleware::from_fn_with_state(
            AuthConfig { token:Some("test-owner".into()), gui_mode:true, pairing:None }, super::origin_guard));
        for (method,path) in [("POST","/api/plugins/%70robe/ai/complete"),("POST","/api/plugins/probe/ai/cancel"),
            ("GET","/api/plugins/%70robe/ai/sources"),("HEAD","/api/plugins/probe/ai/sources"),
            ("GET","/api/plugins/%70robe/voice/status"),("HEAD","/api/plugins/probe/voice/status"),
            ("POST","/api/plugins/probe/voice/open"),("POST","/api/plugins/%70robe/voice/transcribe"),
            ("POST","/api/plugins/probe/voice/speak"),("POST","/api/plugins/probe/voice/cancel"),("POST","/api/plugins/probe/voice/close")] {
            for origin in [None,Some("https://untrusted.example"),Some("http://untrusted.example:3000")] {
                let mut req = Request::builder().method(method).uri(path);
                if let Some(origin) = origin { req = req.header("origin",origin); }
                assert_eq!(app.clone().oneshot(req.body(Body::empty()).unwrap()).await.unwrap().status(),403,"{method} {path} {origin:?}");
            }
            let req = Request::builder().method(method).uri(path).header("origin","http://localhost:3000").body(Body::empty()).unwrap();
            assert_eq!(app.clone().oneshot(req).await.unwrap().status().as_u16(),if cfg!(debug_assertions) {200} else {403},"the existing debug UI exception is preserved");
            for auth in [false,true] {
                let mut req = Request::builder().method(method).uri(path);
                req = if auth { req.header("authorization","Bearer test-owner") } else { req.header("origin","http://tauri.localhost") };
                assert_eq!(app.clone().oneshot(req.body(Body::empty()).unwrap()).await.unwrap().status(),200);
            }
        }
    }

    #[test]
    fn a_paired_token_satisfies_the_guard_like_the_configured_one() {
        use crate::bridge::pairing::PairingManager;
        let mgr = std::sync::Arc::new(std::sync::Mutex::new(PairingManager::new()));
        let token = {
            let mut m = mgr.lock().unwrap();
            let id = m.request("formlogic", None, None).pairing_id;
            m.approve(&id).unwrap().token.unwrap()
        };
        let auth = AuthConfig { token: None, gui_mode: false, pairing: Some(mgr) };
        assert!(auth.token_matches(&token), "a paired token authenticates");
        assert!(!auth.token_matches("oaiypat_wrong"), "a non-granted token does not");
        // And a configured token still works alongside pairing.
        let auth2 = AuthConfig {
            token: Some("cfg".into()),
            gui_mode: false,
            pairing: Some(std::sync::Arc::new(std::sync::Mutex::new(PairingManager::new()))),
        };
        assert!(auth2.token_matches("cfg"));
        assert!(!auth2.token_matches("nope"));
    }

    #[test]
    fn discovery_stays_open_but_reads_are_gated() {
        // capabilities + health are the discovery handshake the protocol commits
        // to keeping open. Everything else under the bridge/plugin prefixes
        // carries data (phone numbers, flow io, the OS username) and must be a
        // restricted read.
        assert!(!is_restricted_read_path("/api/health"));
        assert!(!is_restricted_read_path("/api/bridge/capabilities"));
        for path in [
            "/api/plugins",
            // Which plugin provides the phone and the calendar: the same tier as the plugins.
            "/api/modules",
            "/api/modules/events",
            "/api/plugins/aokie/logs",
            "/api/bridge/events",
            "/api/bridge/runs",
            "/api/bridge/runs/run_1",
            "/api/bridge/flows",
            "/api/bridge/triggers",
            // The setup wizard's record: how far setup got and what was accepted.
            "/api/setup",
            "/api/setup/catalog",
            "/api/setup/plugins/aokie",
            // The engines' catalog, the models chosen in Engines, the downloads.
            "/api/engines",
            "/api/engines/catalog",
            "/api/engines/downloads",
        ] {
            assert!(
                is_restricted_read_path(path),
                "{path} leaks data and must be a restricted read"
            );
        }
        assert!(!is_restricted_read_path("/api/setupx"));
    }

    #[test]
    fn setup_changes_and_engine_downloads_are_privileged() {
        // Accepting a plugin's capabilities is a trust act, and a check runs one
        // of the plugin's commands: a local page must reach neither.
        for (m, path) in [
            (Method::PUT, "/api/setup"),
            (Method::POST, "/api/setup/plugins/aokie/steps/permissions"),
            (Method::POST, "/api/setup/plugins/aokie/finish"),
            (Method::POST, "/api/setup/plugins/aokie/check/pair"),
            // Gigabytes into the engines' folder.
            (Method::POST, "/api/engines/downloads"),
        ] {
            assert!(is_privileged_path(&m, path), "{m} {path} must be privileged");
        }
        assert!(!is_privileged_path(&Method::GET, "/api/setup"));
    }

    #[test]
    fn checking_for_an_update_is_privileged_and_its_status_is_open_on_the_headless_server_only() {
        // The check makes OAIY phone GitHub: the dashboard's own window or a token, never a local page.
        assert!(is_privileged_path(&Method::POST, "/api/update/check"));
        assert!(is_privileged_path(&Method::POST, "/api/update/agent-flushed"));
        // There is no route that downloads or installs: those are commands of the dashboard's window.
        assert!(!is_privileged_path(&Method::GET, "/api/update/status"));
        // The status is open like health where nothing private is in it (the headless server); on the desktop's own
        // listener origin_guard makes it a restricted read, because it says whether a call is live. That is
        // asserted through the guard itself, in the update routes' guard test.
        assert!(!is_restricted_read_path("/api/update/status"));
        assert_eq!(super::UPDATE_STATUS_PATH, "/api/update/status");
    }

    #[test]
    fn the_agents_model_and_the_hardware_recommendation_are_restricted() {
        // Reading them: OAIY's own window or a token, never a remote page.
        assert!(is_restricted_read_path("/api/agent/preferences"));
        assert!(is_restricted_read_path("/api/engines/recommendation"));
        // Choosing the Agent's model decides which account its conversations
        // spend: the privileged gate, like setup.
        assert!(is_privileged_path(&Method::PUT, "/api/agent/preferences"));
    }

    #[test]
    fn the_control_api_is_restricted_and_its_changes_privileged() {
        // The MCP server's tools reach everything the other privileged routes
        // do, so it takes their gate: OAIY's own window or a token holder.
        for (m, path) in [
            (Method::POST, "/api/mcp"),
            (Method::PUT, "/api/control/settings"),
            (Method::POST, "/api/control/anything"),
            (Method::DELETE, "/api/control/log"),
            // Choosing the engines' models, starting and stopping their language model.
            (Method::PUT, "/api/engines/defaults"),
            (Method::POST, "/api/engines/llm/start"),
            (Method::POST, "/api/engines/llm/stop"),
            (Method::POST, "/api/engines/llm/restart"),
        ] {
            assert!(is_privileged_path(&m, path), "{m} {path} must be privileged");
        }
        for path in [
            "/api/mcp",
            "/api/control/settings",
            "/api/control/log",
            "/api/control/desktop-log",
            "/api/engines/defaults",
            "/api/engines/logs",
        ] {
            assert!(is_restricted_read_path(path), "{path} must be a restricted read");
        }
        assert!(!is_restricted_read_path("/api/mcpx"));
        assert!(!is_restricted_read_path("/api/controls"));
    }

    #[test]
    fn the_backup_routes_take_the_strict_gates_and_only_the_status_is_a_plain_read() {
        // Everything that carries a session's bytes is a privileged change.
        for path in [
            "/api/backup/agent/abc/part",
            "/api/backup/agent/abc/done",
            "/api/backup/agent-import/abc/undo-part",
            "/api/backup/agent-import/abc/undo-done",
            "/api/backup/agent-import/abc/done",
        ] {
            assert!(is_privileged_path(&Method::POST, path), "POST {path} must be privileged");
        }
        // The status is a restricted read; what carries the Agent's storage is an export read
        // (its own window's origin or a token, never a loopback page in a release build).
        assert!(is_restricted_read_path("/api/backup/status") && !super::is_export_path("/api/backup/status"));
        for path in ["/api/backup/agent-import", "/api/backup/agent-import/abc/part/0"] {
            assert!(super::is_export_path(path), "GET {path} must be an export read");
        }
        assert!(!super::is_export_path("/api/backupx") && !is_restricted_read_path("/api/backupx/status"));
        // What the vault design reserved under `/api/backup` is not this module's: its paths are not claimed by any of the three gates
        // (the design's rows in the access table judge them when they exist).
        for path in ["/api/backup", "/api/backup/catalog", "/api/backup/jobs/abc", "/api/backup/config", "/api/backup/run", "/api/backup/restore", "/api/backup/rollback", "/api/backup/verify", "/api/backup/identity", "/api/backup/create", "/api/backup/webview-import", "/api/backup/webview/abc/part"] {
            assert!(!crate::backup::routes::is_backup_path(path), "{path} is not a route of the backup module");
            assert!(!is_privileged_path(&Method::POST, path) && !super::is_export_path(path) && !is_restricted_read_path(path), "{path} is not claimed by the backup's gates");
        }
    }

    #[tokio::test]
    async fn a_stranger_cannot_reach_the_backup_routes_and_the_agents_page_can() {
        let dir = crate::secret_file::testing::TempDir::new("backup-guard");
        let app = super::guarded_for_tests(crate::backup::routes::router(dir.0.clone()), Some("desk-token".into()), true);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        for origin in [Some("null"), Some("http://evil.example"), None] {
            let mut post = client.post(format!("{base}/api/backup/agent/abc/part?seq=0")).body("x");
            let mut import = client.get(format!("{base}/api/backup/agent-import"));
            let mut status = client.get(format!("{base}/api/backup/status"));
            if let Some(origin) = origin {
                post = post.header("Origin", origin);
                import = import.header("Origin", origin);
                status = status.header("Origin", origin);
            }
            assert_eq!(post.send().await.unwrap().status(), 403, "POST {origin:?}");
            assert_eq!(import.send().await.unwrap().status(), 403, "GET import {origin:?}");
            assert_eq!(status.send().await.unwrap().status(), 403, "GET status {origin:?}");
        }
        // The Agent's page (its own scheme) passes the gate; the session does not exist, so the answer is the route's own.
        let own = client.post(format!("{base}/api/backup/agent/abc/part?seq=0")).header("Origin", "http://oaiy.localhost").body("x").send().await.unwrap();
        assert_eq!(own.status(), 404);
        // ... but the description of an import needs the secret the desktop gave the Agent's window as well: the origin alone gets 403.
        let import = client.get(format!("{base}/api/backup/agent-import")).header("Origin", "http://oaiy.localhost").send().await.unwrap();
        assert_eq!(import.status(), 403, "an origin is not enough");
        let import = client.get(format!("{base}/api/backup/agent-import")).header("Origin", "http://oaiy.localhost").header("X-Backup-Token", crate::backup::agent::page_token()).send().await.unwrap();
        assert_eq!(import.status(), 200);
        // A paired or configured token, and the linked provider's own site, pass the guard but are not the Agent's page.
        let token_only = client.get(format!("{base}/api/backup/agent-import")).bearer_auth("desk-token").send().await.unwrap();
        assert_eq!(token_only.status(), 403, "a token with no origin");
        let token_post = client.post(format!("{base}/api/backup/agent/abc/part?seq=0")).bearer_auth("desk-token").body("x").send().await.unwrap();
        assert_eq!(token_post.status(), 403);
        let provider = client.get(format!("{base}/api/backup/agent-import")).header("Origin", "https://oaiy.com").send().await.unwrap();
        assert_eq!(provider.status(), 403, "the provider's site");
        // The dashboard reads the status.
        let status = client.get(format!("{base}/api/backup/status")).header("Origin", "http://tauri.localhost").send().await.unwrap();
        assert_eq!(status.status(), 200);
        // A token holder gets in too.
        let token = client.get(format!("{base}/api/backup/status")).bearer_auth("desk-token").send().await.unwrap();
        assert_eq!(token.status(), 200);
        server.abort();
    }

    #[tokio::test]
    async fn the_mcp_endpoint_is_closed_to_a_stranger_and_open_to_the_token() {
        use axum::{middleware, routing::post, Router};
        let app = Router::new()
            .route("/api/mcp", post(|| async { "answered" }))
            .layer(middleware::from_fn_with_state(
                AuthConfig { token: Some("desk-token".into()), gui_mode: true, pairing: None },
                super::origin_guard,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let client = reqwest::Client::new();
        // A local page (any loopback origin in a release build) or a plugin's
        // sandboxed frame (origin null) is refused; so is a native caller
        // with no credential.
        for origin in [Some("null"), Some("http://evil.example"), None] {
            let mut request = client.post(format!("{base}/api/mcp")).body("{}");
            if let Some(origin) = origin {
                request = request.header("Origin", origin);
            }
            let status = request.send().await.unwrap().status();
            assert_eq!(status, 403, "{origin:?}");
        }
        // The Agent's page (its own scheme) and a token holder get in.
        let own = client.post(format!("{base}/api/mcp")).header("Origin", "http://oaiy.localhost").body("{}").send().await.unwrap();
        assert_eq!(own.status(), 200);
        let token = client.post(format!("{base}/api/mcp")).bearer_auth("desk-token").body("{}").send().await.unwrap();
        assert_eq!(token.status(), 200);
        server.abort();
    }

    /// A stand-in for the engines' control port: their catalog with one model
    /// downloading, a discovery document, and a download route that records
    /// what it was asked.
    async fn fake_studio(chosen: &'static str) -> (String, std::sync::Arc<std::sync::Mutex<Vec<serde_json::Value>>>, tokio::task::JoinHandle<()>) {
        use axum::routing::get;
        let asked = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let state = || {
            serde_json::json!({
                "dir": "D:/models", "free": 123_000_000_000u64, "token": false,
                "groups": [{ "id": "llm", "name": "Chat", "about": "Language models." }],
                "models": [
                    { "id": "qwen3.5-9b", "group": "llm", "name": "Qwen3.5 9B", "about": "A strong all-round chat model.", "license": "Apache-2.0",
                      "size_gb": 5.7, "vram_gb": 8, "recommended": true, "agent_tools": true, "installed": false, "partial": false,
                      "download": { "id": "qwen3.5-9b", "dir": "D:/models", "status": "downloading", "done": 1024, "total": 4096, "file": "Qwen3.5-9B-Q4_K_M.gguf",
                                    "files_done": 0, "files_total": 1, "speed": 2.5, "error": null, "added": [] } },
                    { "id": "qwen3-4b", "group": "llm", "name": "Qwen3 4B", "size_gb": 2.5, "vram_gb": 4, "installed": true, "partial": false, "download": null }
                ]
            })
        };
        let recorder = asked.clone();
        let app = axum::Router::new()
            .route("/api/downloads", get(move || async move { axum::Json(state()) }).post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let recorder = recorder.clone();
                async move {
                    recorder.lock().unwrap().push(body.clone());
                    if body["id"] == "nope" {
                        return (axum::http::StatusCode::BAD_REQUEST, axum::Json(serde_json::json!({ "error": "no model nope in the catalog" })));
                    }
                    (axum::http::StatusCode::OK, axum::Json(state()))
                }
            }))
            .route("/api/discovery", get(move || async move { axum::Json(serde_json::json!({ "defaults": { "llm": chosen, "image": "" } })) }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let ui = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (ui, asked, server)
    }

    #[tokio::test]
    async fn the_engines_catalog_is_relayed_with_the_model_chosen_in_engines() {
        let (ui, _, server) = fake_studio("Qwen3.8-Flash-Next").await;
        let v = super::engines_catalog_at(Some(ui)).await;
        assert_eq!(v["running"], true);
        // The model chosen in Engines, as its discovery document says: never one of ours.
        assert_eq!(v["defaults"]["llm"], "Qwen3.8-Flash-Next");
        assert_eq!(v["defaults"]["image"], serde_json::Value::Null, "an empty default is no model");
        let first = &v["models"][0];
        assert_eq!(first["id"], "qwen3.5-9b");
        assert_eq!(first["recommended"], true);
        // Whether the Agent can use its tools with it; a model that does not say chats only.
        assert_eq!((first["agentTools"].as_bool(), v["models"][1]["agentTools"].as_bool()), (Some(true), Some(false)));
        assert_eq!((first["sizeGb"].as_f64(), first["vramGb"].as_i64()), (Some(5.7), Some(8)));
        assert_eq!(first["download"]["filesTotal"], 1);
        assert_eq!(first["download"]["done"], 1024);
        assert_eq!(v["models"][1]["installed"], true);
        assert_eq!(v["models"][1]["download"], serde_json::Value::Null);
        server.abort();
    }

    #[tokio::test]
    async fn with_no_model_chosen_the_default_is_null() {
        let (ui, _, server) = fake_studio("").await;
        let v = super::engines_catalog_at(Some(ui)).await;
        assert_eq!(v["defaults"]["llm"], serde_json::Value::Null);
        server.abort();
    }

    #[tokio::test]
    async fn downloads_are_relayed_and_only_a_catalog_id_is_passed_on() {
        let (ui, asked, server) = fake_studio("").await;
        let v = super::engines_downloads_at(Some(ui.clone())).await;
        assert_eq!(v["running"], true);
        assert_eq!(v["downloads"].as_array().unwrap().len(), 1, "only the models downloading");
        assert_eq!(v["downloads"][0]["status"], "downloading");

        let v = super::engines_download_at(Some(ui.clone()), "qwen3.5-9b").await.unwrap();
        assert_eq!(v["downloads"][0]["id"], "qwen3.5-9b");
        // The engines' own folder: no folder is ever passed on.
        assert_eq!(asked.lock().unwrap().last().unwrap(), &serde_json::json!({ "id": "qwen3.5-9b" }));

        let refused = super::engines_download_at(Some(ui.clone()), "nope").await.unwrap_err();
        assert_eq!(refused, (400, "no model nope in the catalog".to_string()));
        let before = asked.lock().unwrap().len();
        for bad in ["", "../x", "a b", "x/y"] {
            assert_eq!(super::engines_download_at(Some(ui.clone()), bad).await.unwrap_err().0, 400, "{bad:?}");
        }
        assert_eq!(asked.lock().unwrap().len(), before, "a bad id never reaches the engines");
        server.abort();
    }

    #[tokio::test]
    async fn with_no_engines_the_relay_says_so() {
        assert_eq!(super::engines_catalog_at(None).await["running"], false);
        assert_eq!(super::engines_downloads_at(None).await["downloads"], serde_json::json!([]));
        assert_eq!(super::engines_download_at(None, "qwen3.5-9b").await.unwrap_err().0, 409);
        // Engines that went away since: not running, and why.
        let v = super::engines_catalog_at(Some("http://127.0.0.1:9".into())).await;
        assert_eq!(v["running"], false);
        assert!(v["error"].as_str().unwrap().contains("did not answer"));
    }

    #[test]
    fn a_release_build_refuses_a_random_localhost_page_on_the_exec_surface() {
        // The concrete blocker check: in a release build, a page served from an
        // arbitrary localhost port must NOT satisfy the privileged origin gate,
        // so PUT /api/bridge/flows from evil-on-localhost is refused even in GUI
        // mode with no token. (In debug the dev UI needs loopback, hence the
        // cfg.)
        let origin_priv_ok = is_allowed_origin_privileged("http://localhost:6006");
        #[cfg(not(debug_assertions))]
        {
            assert!(!origin_priv_ok, "a release build must not trust a random localhost page");
            assert!(
                !privileged_allowed(false, true, false, origin_priv_ok),
                "GUI + no token + random localhost origin must be refused on exec routes"
            );
        }
        // oaiy.com and the tauri webview always pass — the legit callers.
        assert!(is_allowed_origin_privileged("https://oaiy.com"));
        assert!(is_allowed_origin_privileged("tauri://localhost"));
        let _ = origin_priv_ok;
    }

    #[test]
    fn privileged_auth_matrix() {
        // Headless server (gui_mode=false) WITH a token: token is the only key;
        // a trusted/ spoofed Origin must NOT substitute.
        assert!(privileged_allowed(true, false, true, false), "valid token passes");
        assert!(!privileged_allowed(false, false, true, true), "origin can't bypass a set token");
        // GUI companion (gui_mode=true) WITH a token: token OR webview origin.
        assert!(privileged_allowed(true, true, true, false), "companion: token passes");
        assert!(privileged_allowed(false, true, true, true), "companion: webview origin passes");
        assert!(!privileged_allowed(false, true, true, false), "companion: neither → denied");
        // Headless (gui_mode=false) with NO token: privileged routes are CLOSED —
        // any local process can forge the Origin on a headless server, so a trusted
        // Origin must NOT substitute for a token. The operator must set a token.
        assert!(!privileged_allowed(false, false, false, true), "headless no-token: forged Origin does NOT pass");
        assert!(!privileged_allowed(false, false, false, false), "headless no-token: bad origin denied");
        // GUI companion with NO token: its real (unspoofable) webview origin admins.
        assert!(privileged_allowed(false, true, false, true), "GUI no-token: webview origin passes");
    }
}
