//! Localhost HTTP API the oaiy-web flow editor talks to.
//!
//! Phase 1 surface:
//!   GET /api/health    → { status, product, protocol, version }
//!
//! Phase 2 surface (this file):
//!   GET    /api/services                  → list registered + running services
//!   POST   /api/services/:id/start        → spawn the service process
//!   POST   /api/services/:id/stop         → terminate it
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
    /// The GGUF a single-model server (llama.cpp) is set to load, if the user
    /// picked one (else none — there is no implicit default). Shown in the
    /// service's Model picker.
    pub llama_model: Option<String>,
    /// The multimodal projector loaded beside `llama_model`, if the user picked
    /// one. Surfaced so the picker can show the current choice.
    pub llama_mmproj: Option<String>,
    /// The model NAME a multi-model server (Ollama) is set to use, if the user
    /// picked one (else the pre-pulled default). Shown in its Model picker.
    pub ollama_model: Option<String>,
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

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        product: PRODUCT_ID,
        companion: PRODUCT_ID,
        protocol: BRIDGE_PROTOCOL,
        version: env!("CARGO_PKG_VERSION"),
        api_version: API_VERSION,
        plugin_api_version: *crate::plugins::manifest::SUPPORTED_PLUGIN_API.end(),
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
/// user can clean-reinstall — e.g. swap an old llama.cpp build for a new one.
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

fn is_allowed_origin(origin: &str) -> bool {
    // Dev + locally-served oaiy-web (any loopback port).
    if is_loopback_origin(origin) {
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
            ) || (path.starts_with("/api/services/") && path.ends_with("/uninstall"))
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
        }
        // PUT is only used by the bridge (flow documents). A flow doc is
        // executable code the worker hands to the CLI, so it is exec surface.
        Method::PUT => is_bridge_exec_path(path),
        Method::DELETE => {
            path.starts_with("/api/services/")
                || path.starts_with("/api/models/")
                || path.starts_with("/api/python/venvs/")
                // Uninstalling a plugin removes native code from disk.
                || path.starts_with("/api/plugins/")
                || is_bridge_exec_path(path)
                || is_ai_exec_path(path)
        }
        _ => false,
    }
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
                || path.ends_with("/enabled")))
}

/// The AI gateway surface: provider CRUD + credential admin AND the chat/models
/// proxy that spends the stored key. ALL of it takes the exec-surface gate
/// (trusted origin OR bearer/paired token, fail-closed on a missing Origin) — an
/// anonymous local page must not reconfigure a provider, plant a key, or spend
/// the user's API credits. The path prefix covers `/api/ai/providers*` (config)
/// and `/api/ai/{v1,providers/:id/v1}/*` (the gateway). GET reads are handled by
/// `is_restricted_read_path`, not here.
fn is_ai_exec_path(path: &str) -> bool {
    path.starts_with("/api/ai/")
}

/// GET /api/services/:id/export returns the FULL ServiceTemplate — including `run.env` (which a
/// user-authored service may hold an API key in) and the verbatim install/helper script bodies.
/// It's the read-twin of the privileged `add_service` POST, so it's gated like a privileged read
/// (trusted origin or token) rather than left on the open GET surface.
fn is_export_path(path: &str) -> bool {
    path.starts_with("/api/services/") && path.ends_with("/export")
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
        // Bridge/plugin reads carry real data an arbitrary remote page must not
        // scrape cross-origin: events hold plugin-supplied payloads (for Aokie,
        // caller phone numbers and message bodies), runs hold flow inputs and
        // outputs, and the plugins listing exposes each plugin's absolute `dir`
        // — the OS username, the exact leak `/api/config` is already gated for.
        // `capabilities` + `health` stay open (discovery); everything else under
        // these prefixes is gated to a non-remote origin.
        || path == "/api/plugins"
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
        || path.starts_with("/api/ai/")
}

/// Stricter allow-list for privileged endpoints: OAIY Desktop's OWN webview and
/// oaiy.com only — never an arbitrary localhost page. Loopback origins are
/// allowed in debug builds (the dev UI is served from a localhost port) but NOT
/// in a release build, which is what ships.
fn is_allowed_origin_privileged(origin: &str) -> bool {
    if origin == "tauri://localhost"
        || origin == "http://tauri.localhost"
        || origin == "https://tauri.localhost"
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
    /// Does `presented` match the configured token OR a live paired token?
    fn token_matches(&self, presented: &str) -> bool {
        if let Some(want) = self.token.as_deref() {
            if token_eq(want, presented) {
                return true;
            }
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
            // A configured token must gate EVERY mutation on a headless box — a
            // forged Origin (any non-browser caller can set one) must not substitute
            // for it. Mirrors privileged_allowed: pass on a matching token, OR when
            // there's no real lockdown (GUI, or no token configured) AND the origin
            // is browser-acceptable (loopback/tauri/oaiy.com) or absent (native CLI).
            // So headless+token now requires the token even with a spoofed Origin,
            // while GUI mode and the no-token default keep their broad behavior.
            let origin_ok = match origin.as_deref() {
                Some(o) => is_allowed_origin(o),
                None => true, // native/CLI caller: no browser Origin to check
            };
            token_ok || ((auth.gui_mode || auth.token.is_none()) && origin_ok)
        };
        if !allowed {
            return (
                StatusCode::FORBIDDEN,
                Json(serde_json::json!({ "error": "origin not allowed" })),
            )
                .into_response();
        }
    } else if m == Method::GET {
        // The GET surface is otherwise ungated and served with CORS Any, so a page the user
        // visits could read it cross-origin. Gate the SENSITIVE reads (the rest — health, model /
        // catalog / service listings — carry no secrets and stay open). A native / no-Origin
        // caller is allowed: it has direct filesystem access anyway, and this keeps the CLI working.
        let path = req.uri().path();
        let export_read = is_export_path(path);
        let restricted_read = is_restricted_read_path(path);
        if export_read || restricted_read {
            let origin = req
                .headers()
                .get(ORIGIN)
                .and_then(|o| o.to_str().ok())
                .map(str::to_owned);
            let token_ok = bearer_token(&req).is_some_and(|got| auth.token_matches(&got));
            let allowed = match origin.as_deref() {
                None => true,
                Some(o) => {
                    if export_read {
                        token_ok || (auth.gui_mode && is_allowed_origin_privileged(o))
                    } else {
                        token_ok || is_allowed_origin(o)
                    }
                }
            };
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

pub async fn serve(
    port: u16,
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
) -> Result<(), BoxError> {
    // CORS stays permissive so a hosted oaiy-web at any domain can READ the
    // API (the localhost bind keeps non-local processes out). State-changing
    // and exec endpoints are additionally gated by `origin_guard` below, so a
    // random web page the user has open can't issue drive-by POST/DELETE
    // requests against the loopback API.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers(Any);

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
    let link_routes = crate::link::routes::router(link);
    let bridge_routes = crate::bridge::bridge_router(bridge);

    // The AI gateway is its own sub-router with its own state (provider store +
    // registry clone), merged INSIDE the guard layers like the bridge — provider
    // CRUD and the credential-injecting chat proxy need the same origin/token gate.
    let ai_routes = crate::ai::ai_router(crate::ai::AiState {
        providers: ai_providers,
        registry: registry_for_ai,
        codex: ai_codex,
    });

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
        .merge(companion_routes)
        .merge(link_routes)
        .merge(ai_routes)
        .layer(middleware::from_fn_with_state(
            AuthConfig { token: auth_token, gui_mode, pairing: Some(pairing_for_auth) },
            origin_guard,
        ))
        .layer(cors)
        // OUTSIDE the CORS layer so it runs after it and can add to the
        // preflight response CORS produced.
        .layer(axum::middleware::from_fn(allow_private_network));

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    log::info!("OAIY API listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        is_allowed_origin, is_allowed_origin_privileged, is_privileged_path, is_restricted_read_path,
        HealthResponse, API_VERSION, BRIDGE_PROTOCOL, PRODUCT_ID,
        privileged_allowed, AuthConfig, ModelDownloadRequest,
    };
    use axum::http::Method;

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

        crate::link::set_linked_origin_for_tests(None);
        assert!(!is_allowed_origin("http://formlogic.local"), "unlinking withdraws it");
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
        assert!(is_privileged_path(&Method::POST, "/api/bridge/runs"));
        // Reading history stays a restricted read, not a privileged one.
        assert!(is_restricted_read_path("/api/bridge/runs"));
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
        ] {
            assert!(is_restricted_read_path(path), "{path} must be a restricted read");
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
            "/api/plugins/aokie/logs",
            "/api/bridge/events",
            "/api/bridge/runs",
            "/api/bridge/runs/run_1",
            "/api/bridge/flows",
            "/api/bridge/triggers",
        ] {
            assert!(
                is_restricted_read_path(path),
                "{path} leaks data and must be a restricted read"
            );
        }
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
