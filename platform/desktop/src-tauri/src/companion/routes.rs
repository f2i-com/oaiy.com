//! HTTP surface for companion device trust.
//!
//! Every route is scoped to a plugin id and refuses unless that plugin declares
//! `oaiy.companion.admission`. The implementation this is ported from hardcodes
//! one plugin and leaves a TODO for the second; keying on the capability is the
//! same gate expressed in the model OAIY already has, and it means a second
//! broker plugin needs no code here.
//!
//! These are PRIVILEGED routes — see `is_privileged_path` in `http.rs`. They
//! decide which devices may take a live call's audio, so a local web page must
//! not be able to reach them just by being on loopback.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::identity::{EndpointIdentity, EndpointIdentityHandle, RosterError};
use super::pairing::MobilePairingResponse;
use super::upstream::{UpstreamConfig, UpstreamHandle};
use crate::plugins::registry::PluginRegistryHandle;

/// One endpoint identity per broker plugin, created on first use.
///
/// Separate identities rather than one shared key: revoking the phones trusted
/// by one plugin must not disturb another's, and a key is the thing revocation
/// is expressed against.
pub struct CompanionRegistry {
    data_dir: PathBuf,
    plugins: PluginRegistryHandle,
    identities: Mutex<HashMap<String, EndpointIdentityHandle>>,
}

pub type CompanionHandle = Arc<CompanionRegistry>;

pub fn new_handle(data_dir: PathBuf, plugins: PluginRegistryHandle) -> CompanionHandle {
    Arc::new(CompanionRegistry {
        data_dir,
        plugins,
        identities: Mutex::new(HashMap::new()),
    })
}

impl CompanionRegistry {
    /// The identity for `plugin_id`, if that plugin may broker companions.
    ///
    /// Checked per request rather than cached: a plugin can be uninstalled or
    /// replaced by one with a different manifest while the app runs, and a
    /// stale yes here would keep serving a permission that no longer exists.
    pub fn identity_for(&self, plugin_id: &str) -> Result<EndpointIdentityHandle, (StatusCode, String)> {
        if plugin_id.is_empty()
            || plugin_id.len() > 64
            || !plugin_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err((StatusCode::BAD_REQUEST, "invalid plugin id".into()));
        }

        let declared = {
            let mut reg = self
                .plugins
                .lock()
                .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "plugin registry lock poisoned".to_string()))?;
            if reg.get(plugin_id).is_none() {
                reg.scan();
            }
            reg.get(plugin_id)
                .and_then(|r| r.manifest.as_ref().map(|m| m.resolved_capabilities()))
                .unwrap_or_default()
        };
        if !declared.contains("oaiy.companion.admission") {
            return Err((
                StatusCode::FORBIDDEN,
                format!("plugin {plugin_id:?} does not declare oaiy.companion.admission"),
            ));
        }

        let mut map = self
            .identities
            .lock()
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, "companion lock poisoned".to_string()))?;
        Ok(map
            .entry(plugin_id.to_string())
            .or_insert_with(|| EndpointIdentity::open(self.data_dir.clone(), plugin_id))
            .clone())
    }
}

fn fail(status: StatusCode, message: String) -> axum::response::Response {
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// A change to the roster that was not made: 404 for what the caller named, 500 for a roster that could not be saved, which
/// the caller must not take for done (a revocation that is not on disk is undone by the next start).
fn not_changed(error: RosterError) -> axum::response::Response {
    match error {
        RosterError::Refused(why) => fail(StatusCode::NOT_FOUND, why),
        RosterError::NotSaved(why) => fail(StatusCode::INTERNAL_SERVER_ERROR, why),
    }
}

fn resolve(
    state: &CompanionHandle,
    plugin_id: &str,
) -> Result<EndpointIdentityHandle, axum::response::Response> {
    state
        .identity_for(plugin_id)
        .map_err(|(status, message)| fail(status, message))
}

async fn status(
    State(state): State<CompanionHandle>,
    Path(plugin_id): Path<String>,
) -> axum::response::Response {
    match resolve(&state, &plugin_id) {
        Ok(id) => (StatusCode::OK, Json(id.status())).into_response(),
        Err(r) => r,
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct OfferBody {
    #[serde(default)]
    app_id: Option<String>,
    #[serde(default)]
    workspace_id: Option<String>,
    #[serde(default)]
    desktop_connection_id: Option<String>,
}

async fn create_offer(
    State(state): State<CompanionHandle>,
    Path(plugin_id): Path<String>,
    body: Option<Json<OfferBody>>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    let body = body.map(|Json(b)| b);
    // The pairing screen leaves these blank until FormLogic is linked, so they
    // fall back to the plugin's own identity rather than refusing — an offer
    // that cannot be produced is a screen the user cannot get past.
    let app_id = body
        .as_ref()
        .and_then(|b| b.app_id.clone())
        .unwrap_or_else(|| plugin_id.clone());
    let connection = body
        .as_ref()
        .and_then(|b| b.desktop_connection_id.clone())
        .unwrap_or_else(|| "local".to_string());
    let workspace = body.as_ref().and_then(|b| b.workspace_id.clone());

    match id.create_offer(&app_id, workspace.as_deref(), &connection) {
        Ok(offer) => (StatusCode::CREATED, Json(offer)).into_response(),
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

async fn receive_response(
    State(state): State<CompanionHandle>,
    Path(plugin_id): Path<String>,
    Json(response): Json<MobilePairingResponse>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    match id.receive_response(&response) {
        Ok(pending) => (StatusCode::OK, Json(pending)).into_response(),
        // 400, not 500: every rejection here is a statement about the message
        // the caller supplied — expired, replayed, or for another desktop.
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

async fn approve(
    State(state): State<CompanionHandle>,
    Path((plugin_id, approval_id)): Path<(String, String)>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    match id.approve(&approval_id) {
        Ok(m) => (StatusCode::OK, Json(m)).into_response(),
        Err(e) => not_changed(e),
    }
}

async fn deny(
    State(state): State<CompanionHandle>,
    Path((plugin_id, approval_id)): Path<(String, String)>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    match id.deny(&approval_id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => fail(StatusCode::NOT_FOUND, e),
    }
}

async fn revoke(
    State(state): State<CompanionHandle>,
    Path((plugin_id, thumbprint)): Path<(String, String)>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    match id.revoke(&thumbprint) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => not_changed(e),
    }
}

async fn rotate(
    State(state): State<CompanionHandle>,
    Path(plugin_id): Path<String>,
) -> axum::response::Response {
    let id = match resolve(&state, &plugin_id) {
        Ok(id) => id,
        Err(r) => return r,
    };
    match id.rotate() {
        Ok(s) => (StatusCode::OK, Json(s)).into_response(),
        Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// The relay this desktop brokers admissions through.
///
/// Not per-plugin: it is the user's account with a relay deployment, and two
/// broker plugins on one machine reach the same one. Their DEVICE TRUST stays
/// separate — that is the part revocation is expressed against.
async fn upstream_status(State(state): State<RouterState>) -> axum::response::Response {
    (StatusCode::OK, Json(state.upstream.status())).into_response()
}

async fn set_upstream(
    State(state): State<RouterState>,
    Json(config): Json<UpstreamConfig>,
) -> axum::response::Response {
    match state.upstream.set(config) {
        Ok(status) => (StatusCode::OK, Json(status)).into_response(),
        // 400: every rejection is about the URL or token the caller supplied.
        Err(e) => fail(StatusCode::BAD_REQUEST, e),
    }
}

async fn clear_upstream(State(state): State<RouterState>) -> axum::response::Response {
    (StatusCode::OK, Json(state.upstream.clear())).into_response()
}

/// State for the relay routes only.
///
/// Holds no [`CompanionHandle`] on purpose: the relay is per-MACHINE, not
/// per-plugin — it is the user's account with a relay deployment, and two
/// broker plugins reach the same one. Device trust stays per-plugin, which is
/// what the `/:plugin/` routes carry.
#[derive(Clone)]
pub struct RouterState {
    upstream: UpstreamHandle,
}

pub fn router(state: CompanionHandle, upstream: UpstreamHandle) -> Router {
    let relay = Router::new()
        .route(
            "/api/companion/relay",
            get(upstream_status)
                .post(set_upstream)
                .delete(clear_upstream),
        )
        .with_state(RouterState { upstream });
    Router::new()
        .route("/api/companion/:plugin/pairing", get(status))
        .route("/api/companion/:plugin/pairing/offers", post(create_offer))
        .route("/api/companion/:plugin/pairing/responses", post(receive_response))
        .route("/api/companion/:plugin/pairing/approvals/:id/approve", post(approve))
        .route("/api/companion/:plugin/pairing/approvals/:id/deny", post(deny))
        .route(
            "/api/companion/:plugin/mobiles/:thumbprint",
            axum::routing::delete(revoke),
        )
        .route("/api/companion/:plugin/identity/rotate", post(rotate))
        .with_state(state)
        .merge(relay)
}

#[cfg(test)]
mod tests {
    use super::super::identity::testing::phone_key;
    use super::*;

    /// A failed save answered 204 here, and the phone was approved again at the next start. A thumbprint nobody knows stays a 404.
    #[tokio::test]
    async fn a_revocation_that_could_not_be_saved_is_not_answered_as_done() {
        let dir = std::env::temp_dir().join(format!("oaiy-companion-routes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let plugin = dir.join("plugins").join("aokie");
        std::fs::create_dir_all(&plugin).unwrap();
        let manifest = serde_json::json!({
            "schemaVersion": 3, "id": "aokie", "name": "Aokie", "version": "0.1.0", "pluginApiVersion": 1,
            "entry": { "kind": "process", "command": "plugin.exe" },
            "capabilities": ["oaiy.companion.admission"],
        });
        std::fs::write(plugin.join("manifest.json"), manifest.to_string()).unwrap();
        std::fs::write(plugin.join("plugin.exe"), b"stub").unwrap();
        let state = new_handle(dir.clone(), crate::plugins::registry::new_handle(dir.join("plugins")));
        let identity = state.identity_for("aokie").map_err(|(_, why)| why).unwrap();
        let phone = phone_key(1);
        identity.approve_for_tests(&phone);

        let roster = dir.join("companion").join("aokie").join("roster.json");
        std::fs::remove_file(&roster).unwrap();
        std::fs::create_dir_all(roster.join("in-the-way")).unwrap();
        let asked = |thumbprint: &str| revoke(State(state.clone()), Path(("aokie".to_string(), thumbprint.to_string())));
        assert_eq!(asked(&phone.thumbprint).await.status(), StatusCode::INTERNAL_SERVER_ERROR);

        std::fs::remove_dir_all(&roster).unwrap();
        assert_eq!(asked("nobody-knows-this-one").await.status(), StatusCode::NOT_FOUND);
        assert_eq!(asked(&phone.thumbprint).await.status(), StatusCode::NO_CONTENT);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
