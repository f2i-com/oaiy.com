//! The routes of the access model that exist now: `GET /api/auth/info`, `GET /api/auth/whoami` and
//! `POST /api/auth/derive`.
//!
//! (`loginConfigured` and `setupCode` are the web login's to say, when there is one.)
//!
//! They are rows of the table with `since: 2`, so the new guard judges them in every access mode:
//! `info` is public (and, like the login routes, gets no CORS headers: it is same-origin by
//! construction), `whoami` and `derive` need a credential, and `derive` adds its own rules (a `desk`,
//! `pat` or `static` parent only, never a session, never a derived credential).

use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Extension, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::audit::Context as AuditContext;
use super::guard::{Denial, Guard, RequestInfo};
use super::host::{Channel, HostClass};
use super::principal::{Principal, PrincipalKind};
use super::scopes::ScopeSet;
use super::store::{DeriveError, DeriveRequest, MintFailure};

/// Auth routes take at most this many bytes of body.
pub const MAX_BODY: usize = 16 * 1024;
/// The most scopes one request can name.
pub const MAX_SCOPES: usize = 64;
/// The longest label.
pub const MAX_LABEL: usize = 80;

/// The routes, with the guard they read the store and the settings from.
pub fn router(guard: Arc<Guard>) -> Router {
    Router::new()
        .route("/api/auth/info", get(info))
        .route("/api/auth/whoami", get(whoami))
        .route("/api/auth/derive", post(derive))
        .with_state(guard)
}

/// `GET /api/auth/info`: what a page needs to know before it has a credential: the request as the server
/// understood it (`seen`), so that a proxy mistake shows before it becomes a lockout. It says nothing of
/// the install's host names or its exposure: those came from a cross-origin reader in the first draft.
async fn info(State(guard): State<Arc<Guard>>, info: Option<Extension<RequestInfo>>) -> Response {
    let (seen, secure, app) = match info {
        Some(Extension(i)) => {
            let app = match i.host_class {
                Some(HostClass::LoopbackApp(a)) | Some(HostClass::Public(a)) => a.name(),
                _ => "dash",
            };
            (
                json!({ "clientIp": i.client_ip, "proto": i.proto, "host": i.host, "viaTrustedProxy": i.via_trusted_proxy }),
                i.channel == Channel::Secure,
                app,
            )
        }
        None => (Value::Null, false, "dash"),
    };
    Json(json!({
        "scheme": "oaiy-auth/1",
        "app": app,
        "apiVersion": crate::http::API_VERSION,
        "loginConfigured": guard.login_configured(),
        "setupCode": guard.setup_code_status(),
        "secureChannel": secure,
        "factors": ["password"],
        "seen": seen,
    }))
    .into_response()
}
fn kind_name(k: PrincipalKind) -> &'static str {
    match k {
        PrincipalKind::Session => "session",
        PrincipalKind::Desk => "desk",
        PrincipalKind::Pat => "pat",
        PrincipalKind::Run => "run",
        PrincipalKind::Console => "console",
        PrincipalKind::Static => "static",
        PrincipalKind::Legacy => "legacy",
    }
}

/// `GET /api/auth/whoami`: who the credential is and what it can do, so a client can learn its scopes and
/// when to pair again.
async fn whoami(principal: Option<Extension<Principal>>) -> Response {
    let Some(Extension(p)) = principal else {
        return Denial::auth_required().into_response();
    };
    let level = match p.control_level() {
        super::principal::ControlLevel::None => "none",
        super::principal::ControlLevel::Read => "read",
        super::principal::ControlLevel::Project => "project",
    };
    Json(json!({
        "id": p.id,
        "kind": kind_name(p.kind),
        "label": p.label,
        "scopes": p.scopes.names(),
        "expiresMs": p.expires_ms,
        "sessionExpiresMs": if p.kind == PrincipalKind::Session { json!(p.expires_ms) } else { Value::Null },
        "origin": p.origins.first(),
        "elevated": p.elevated,
        "controlLevel": level,
        "persisted": p.persisted,
    }))
    .into_response()
}

#[derive(Deserialize)]
struct DeriveBody {
    scopes: Vec<String>,
    #[serde(default, rename = "ttlSeconds")]
    ttl_seconds: Option<u64>,
    #[serde(default)]
    label: Option<String>,
}

/// A `Content-Type` of `application/json` (with or without parameters).
fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            v.split(';')
                .next()
                .is_some_and(|t| t.trim().eq_ignore_ascii_case("application/json"))
        })
}

/// `POST /api/auth/derive`: a child credential of at most what the caller holds. Only a `desk`, `pat` or
/// `static` credential may derive; a session cannot (an `HttpOnly` cookie must not become a portable
/// bearer) and a derived credential cannot derive.
async fn derive(
    State(guard): State<Arc<Guard>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let Some(Extension(parent)) = principal else {
        return Denial::auth_required().into_response();
    };
    if !is_json(&headers) {
        return Denial::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Send Content-Type: application/json.",
        )
        .into_response();
    }
    let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
        Ok(b) => b,
        Err(_) => {
            return Denial::bad_request("The body is over 16 KiB or could not be read.")
                .into_response()
        }
    };
    let parsed: DeriveBody = match serde_json::from_slice(&bytes) {
        Ok(b) => b,
        Err(e) => {
            return Denial::bad_request(format!(
                "The body is not the JSON of a derive request: {e}"
            ))
            .into_response()
        }
    };
    if parsed.scopes.is_empty() || parsed.scopes.len() > MAX_SCOPES {
        return Denial::invalid_request(format!("Name between 1 and {MAX_SCOPES} scopes."))
            .into_response();
    }
    let scopes = match ScopeSet::parse(parsed.scopes.iter().map(String::as_str)) {
        Ok(s) => s,
        Err(e) => return Denial::invalid_request(e.to_string()).into_response(),
    };
    let label = parsed.label.unwrap_or_else(|| "derived".into());
    if label.is_empty() || label.chars().count() > MAX_LABEL || label.chars().any(char::is_control)
    {
        return Denial::invalid_request(format!(
            "The label is 1 to {MAX_LABEL} characters with no control characters."
        ))
        .into_response();
    }
    let ttl_ms = match parsed.ttl_seconds {
        None => None,
        Some(0) => return Denial::invalid_request("ttlSeconds is at least 1.").into_response(),
        Some(s) => Some(s.saturating_mul(1000)),
    };
    match guard.store().derive(
        &parent,
        DeriveRequest {
            scopes,
            ttl_ms,
            label: label.clone(),
        },
    ) {
        Ok(minted) => {
            if let Some(audit) = guard.audit() {
                let ip = info.as_ref().map(|Extension(i)| i.client_ip.clone());
                // One line for the first derive of a minute and a count for the rest of it (a busy parent
                // must not wash the audit log out).
                audit.derived(&parent.actor(), &AuditContext { ip: ip.as_deref(), host: None, ua: None }, json!({ "kind": "run", "derived": true, "label": label, "scopes": minted.scopes.len(), "id": minted.id }));
            }
            (StatusCode::CREATED, Json(json!({ "id": minted.id, "token": minted.token, "scopes": minted.scopes.names(), "expiresMs": minted.expires_ms }))).into_response()
        }
        Err(e) => derive_error(e).into_response(),
    }
}

fn derive_error(e: DeriveError) -> Denial {
    match e {
        DeriveError::Refused(why) => {
            Denial::new(StatusCode::FORBIDDEN, "derive_refused", why).counted_denied()
        }
        DeriveError::RateLimited { retry_after_s } => Denial::rate_limited(retry_after_s),
        DeriveError::Mint(m) => match m {
            MintFailure::TooManyCredentials | MintFailure::TooManyChildren => {
                Denial::new(StatusCode::CONFLICT, "too_many_credentials", m.to_string())
            }
            MintFailure::ScopeNotGrantable(why) => {
                Denial::new(StatusCode::BAD_REQUEST, "scope_not_grantable", why)
            }
            MintFailure::Invalid(why) => Denial::invalid_request(why),
            MintFailure::TtlTooLong { .. } => Denial::invalid_request(m.to_string()),
            // The parent went away between the guard and here.
            MintFailure::ParentInvalid => {
                Denial::from_auth_error(&super::store::AuthError::Revoked {
                    reason: "parent_ended",
                })
            }
            MintFailure::StoreUnavailable(_) => Denial::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "store_unavailable",
                "The credential store cannot write.",
            ),
            MintFailure::Token(_) => Denial::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "store_unavailable",
                "No credential can be made right now.",
            ),
        },
    }
}

impl Denial {
    fn counted_denied(mut self) -> Denial {
        self.noise = Some("auth.denied");
        self
    }
}
