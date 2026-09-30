//! The console (design 4.7.9): what `oaiy-server auth ...` reaches on a running server.
//!
//! The running server is the only writer of its auth files, and memory is authoritative, so the console never
//! edits a file under a running server. It reads `console.json` and `console.token` and calls these routes, on
//! the loopback port, with the **console credential** (`oaiycon_...`): made at every start, held in memory, and
//! written with the port (0600, atomically) once the listener has bound; a clean exit removes both files, and a
//! stale file holds a dead credential. The credential holds every scope, is accepted only from a direct loopback
//! peer with no forwarded header and no `Origin` (4.5.3), and is `{"kind":"con","label":"console"}` in the audit
//! log. Anyone who can read `console.token` is the data folder's owner or root: the same trust as editing the files.
//!
//! The five console-only routes (class `Console` in the table): `POST /api/auth/console/reset-password` (also the way
//! `auth init` creates the first owner), `setup-code`, `session-link`, `sessions/revoke-all` and
//! `GET /api/auth/console/status`.

use std::io;
use std::path::Path;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::audit::Context as AuditContext;
use super::guard::{Denial, RequestInfo};
use super::login::{read_json, LoginState, Secret};
use super::principal::{Actor, Principal};
use super::scopes::ScopeSet;
use super::scrub::scrub_line;
use super::session::LoginFacts;
use super::setup::{MakeError, VALID_MS};
use super::store::MintSpec;
use super::token::Kind;

/// The file the console credential is in.
pub const TOKEN_FILE: &str = "console.token";
/// The file the port is in.
pub const INFO_FILE: &str = "console.json";
/// A console credential outlives any server it was made for: it is memory-only, so this is only the bound.
const CONSOLE_TTL_MS: u64 = 365 * 24 * 3_600_000;

/// The console's routes, merged into the login's router.
pub fn routes() -> Router<Arc<LoginState>> {
    Router::new()
        .route("/api/auth/console/reset-password", post(reset_password))
        .route("/api/auth/console/setup-code", post(setup_code))
        .route("/api/auth/console/session-link", post(session_link))
        .route("/api/auth/console/sessions/revoke-all", post(revoke_all))
        .route("/api/auth/console/status", get(status))
}

#[derive(Deserialize)]
struct ResetBody {
    password: Secret,
}

#[derive(Deserialize, Default)]
struct RevokeAllBody {
    #[serde(default)]
    devices: bool,
}

fn console_actor(principal: Option<Extension<Principal>>) -> Result<Actor, Denial> {
    match principal {
        Some(Extension(p)) => Ok(p.actor()),
        None => Err(Denial::auth_required()),
    }
}

fn ctx(info: &Option<Extension<RequestInfo>>) -> AuditContext<'_> {
    match info {
        Some(Extension(i)) => AuditContext {
            ip: Some(&i.client_ip),
            host: Some(&i.host),
            ua: None,
        },
        None => AuditContext::default(),
    }
}

/// The audit line every console command leaves: the command's name, never its arguments.
fn command(st: &LoginState, actor: &Actor, info: &Option<Extension<RequestInfo>>, name: &str) {
    st.critical(
        "console.command",
        Some(actor),
        &ctx(info),
        json!({ "command": name }),
    );
}

fn failed(e: Denial) -> Response {
    e.into_response()
}

async fn reset_password(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let actor = match console_actor(principal) {
        Ok(a) => a,
        Err(d) => return failed(d),
    };
    let req: ResetBody = match read_json(&headers, body).await {
        Ok(r) => r,
        Err(d) => return failed(d),
    };
    command(&st, &actor, &info, "reset-password");
    match st
        .console_set_password(&req.password.0, &actor, &ctx(&info))
        .await
    {
        Ok(created) => Json(json!({ "ok": true, "created": created })).into_response(),
        Err(d) => failed(d),
    }
}

async fn setup_code(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
) -> Response {
    let actor = match console_actor(principal) {
        Ok(a) => a,
        Err(d) => return failed(d),
    };
    command(&st, &actor, &info, "setup-code");
    if st.owner_configured() {
        return failed(Denial::new(
            StatusCode::CONFLICT,
            "already_configured",
            "An owner login already exists: a setup code is only for the first one. Use `auth reset-password`.",
        ));
    }
    match st.setup.make(&st.random) {
        Ok(code) => Json(json!({
            "code": code,
            "expiresMs": st.clock.now_ms().saturating_add(VALID_MS),
            "attemptsLeft": super::setup::WRONG_GUESSES,
        }))
        .into_response(),
        Err(e) => failed(match e {
            MakeError::Random(_) => Denial::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "store_unavailable",
                "The operating system gave no randomness.",
            ),
            MakeError::Io(_) => Denial::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "store_unavailable",
                "The setup code cannot be written.",
            ),
        }),
    }
}

async fn session_link(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
) -> Response {
    let actor = match console_actor(principal) {
        Ok(a) => a,
        Err(d) => return failed(d),
    };
    command(&st, &actor, &info, "session-link");
    if !st.owner_configured() {
        return failed(Denial::new(
            StatusCode::CONFLICT,
            "setup_required",
            "There is no owner yet: a session link is a way in for one that exists.",
        ));
    }
    match st.make_link() {
        Ok((code, url)) => {
            st.critical("link.issued", Some(&actor), &ctx(&info), json!({}));
            Json(json!({
                "code": code,
                "url": url,
                "expiresInSeconds": super::login::LINK_VALID_MS / 1000,
            }))
            .into_response()
        }
        Err(_) => failed(Denial::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "store_unavailable",
            "The operating system gave no randomness.",
        )),
    }
}

async fn revoke_all(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let actor = match console_actor(principal) {
        Ok(a) => a,
        Err(d) => return failed(d),
    };
    let req: RevokeAllBody = match read_json(&headers, body).await {
        Ok(r) => r,
        Err(d) => return failed(d),
    };
    command(&st, &actor, &info, "sessions-revoke-all");
    let (sessions, devices) = st.console_revoke_all(req.devices, &actor);
    Json(json!({ "ok": true, "sessions": sessions, "devices": devices })).into_response()
}

async fn status(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
) -> Response {
    let actor = match console_actor(principal) {
        Ok(a) => a,
        Err(d) => return failed(d),
    };
    command(&st, &actor, &info, "status");
    Json(st.status_json()).into_response()
}

// ---- the credential and its files ----------------------------------------------------------------

/// Why the console credential could not be published.
#[derive(Debug)]
pub enum PublishError {
    Mint(String),
    Write {
        file: &'static str,
        source: io::Error,
    },
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::Mint(e) => write!(f, "the console credential could not be made: {e}"),
            PublishError::Write { file, source } => {
                write!(f, "{file} could not be written: {source}")
            }
        }
    }
}

impl std::error::Error for PublishError {}

/// Make the console credential and write `console.token` and `console.json` (0600, atomically), once the listener has
/// bound and `port` is known. The credential is in memory (and in the file) and nowhere else.
pub fn publish(state: &LoginState, port: u16) -> Result<(), PublishError> {
    let mut spec = MintSpec::new(Kind::Con, "console", ScopeSet::all(), CONSOLE_TTL_MS);
    spec.created_by = None;
    let minted = state
        .store()
        .mint(spec)
        .map_err(|e| PublishError::Mint(e.to_string()))?;
    // The token first: a reader that finds the port finds a credential beside it.
    let mut text = minted.token.clone();
    text.push('\n');
    state
        .write_auth_file(TOKEN_FILE, text.as_bytes())
        .map_err(|source| PublishError::Write {
            file: TOKEN_FILE,
            source,
        })?;
    let info =
        json!({ "port": port, "pid": std::process::id(), "started_ms": state.clock.now_ms() });
    state
        .write_auth_file(INFO_FILE, format!("{info}\n").as_bytes())
        .map_err(|source| PublishError::Write {
            file: INFO_FILE,
            source,
        })
}

/// Remove both files (a clean exit): what is left after a crash holds a credential that is already dead.
pub fn remove_files(auth_dir: &Path) {
    for name in [TOKEN_FILE, INFO_FILE] {
        match std::fs::remove_file(auth_dir.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => log::warn!(
                "{}",
                scrub_line(&format!("auth: {name} could not be removed: {e}"))
            ),
        }
    }
}

/// The value of a status answer that a person reads, without the noise: for `auth status`.
pub fn status_lines(status: &Value) -> Vec<String> {
    let s = |k: &str| status[k].as_str().unwrap_or("?").to_string();
    let n = |k: &str| status[k].as_u64().unwrap_or(0);
    let b = |k: &str| status[k].as_bool().unwrap_or(false);
    let mut lines = vec![
        format!("exposure: {}", s("exposure")),
        format!("access mode: {}", s("accessMode")),
        format!(
            "login configured: {}",
            if b("loginConfigured") { "yes" } else { "no" }
        ),
        format!("setup code: {}", s("setupCode")),
        format!(
            "sessions: {}, devices: {}, paired tokens: {}",
            n("sessions"),
            n("devices"),
            n("tokens")
        ),
        format!(
            "slow mode: {} ({} failures in the last hour)",
            if b("slowMode") { "ON" } else { "off" },
            n("recentFailures")
        ),
        format!("storage: {}", s("storage")),
    ];
    for (key, label) in [
        ("blockedAddresses", "blocked address"),
        ("blockedDevices", "blocked device"),
    ] {
        if let Some(list) = status[key].as_array() {
            for entry in list {
                lines.push(format!(
                    "{label}: {} for {} more seconds",
                    entry["key"].as_str().unwrap_or("?"),
                    entry["retryAfterSeconds"].as_u64().unwrap_or(0)
                ));
            }
        }
    }
    if let Some(reason) = status["underAttack"].as_str() {
        lines.push(format!("LOGIN ATTACK: {reason}"));
    }
    lines
}
