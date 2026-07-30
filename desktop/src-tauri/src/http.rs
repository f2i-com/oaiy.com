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
struct HealthResponse {
    status: &'static str,
    /// Stable machine identity. Never localise or re-word this.
    product: &'static str,
    /// Bridge Protocol the rest of this API speaks.
    protocol: &'static str,
    version: &'static str,
}

/// Wire identity. Deliberately NOT derived from the crate name — renaming the
/// binary must not silently change what clients match on.
pub const PRODUCT_ID: &str = "oaiy-desktop";
/// Bump only on a breaking wire change; add fields freely without touching it.
pub const BRIDGE_PROTOCOL: &str = "oaiy-bridge/1";

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        product: PRODUCT_ID,
        protocol: BRIDGE_PROTOCOL,
        version: env!("CARGO_PKG_VERSION"),
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
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
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
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err400(&e),
    }
}

/// Install via `Runner` (same pipe + log machinery used for service
/// processes) so the existing /logs endpoint streams progress in real
/// time — no extra plumbing on the UI side.
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

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    match state.downloads.list_models() {
        Ok(models) => (StatusCode::OK, Json(models)).into_response(),
        Err(e) => err500(&e),
    }
}

#[derive(Deserialize)]
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
}

async fn start_model_download(
    State(state): State<AppState>,
    Json(req): Json<ModelDownloadRequest>,
) -> impl IntoResponse {
    match state
        .downloads
        .start(&req.url, req.filename.as_deref(), req.subdir.as_deref())
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
            ) || (path.starts_with("/api/services/") && path.ends_with("/uninstall"))
        }
        Method::DELETE => {
            path.starts_with("/api/services/")
                || path.starts_with("/api/models/")
                || path.starts_with("/api/python/venvs/")
        }
        _ => false,
    }
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
        || (path.starts_with("/api/services/") && path.ends_with("/logs"))
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
        let token_ok = matches!(
            (auth.token.as_deref(), bearer_token(&req)),
            (Some(want), Some(got)) if token_eq(want, &got)
        );
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
            let token_ok = matches!(
                (auth.token.as_deref(), bearer_token(&req)),
                (Some(want), Some(got)) if token_eq(want, &got)
            );
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

    let state = AppState {
        config,
        registry,
        downloads,
        python,
        catalog,
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/config", get(get_config))
        // services
        .route("/api/services", get(list_services).post(add_service))
        .route("/api/services/ensure-by-port", post(ensure_service_by_port))
        .route("/api/services/:id", delete(delete_service))
        .route("/api/services/:id/start", post(start_service))
        .route("/api/services/:id/stop", post(stop_service))
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
        .route("/api/python", get(python_status))
        .route("/api/python/install", post(install_python))
        .route("/api/python/logs", get(python_logs))
        .route("/api/python/venvs", post(create_venv))
        .route("/api/python/venvs/:name", delete(delete_venv))
        .with_state(state)
        .layer(middleware::from_fn_with_state(
            AuthConfig { token: auth_token, gui_mode },
            origin_guard,
        ))
        .layer(cors);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    log::info!("OAIY API listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::privileged_allowed;

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
