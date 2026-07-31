//! What a linkable provider looks like, as DATA.
//!
//! The point of this file is that no code below it names FormLogic. A provider
//! is a descriptor — endpoints, a client id, scopes, and where to find the
//! credential in the token response — so adding a second provider is a JSON
//! file, not a branch in the link flow.
//!
//! Descriptors load from two places, exactly like service templates: built-ins
//! compiled into the binary, and user files under `<data>/connectors/*.json`
//! which override a built-in of the same id.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Descriptors shipped with the app. One today; the list is the only place a
/// specific provider is named.
const BUILTIN: &[&str] = &[include_str!("../../resources/connectors/formlogic.json")];

/// How a provider authenticates a desktop.
///
/// An enum with one variant today, tagged in JSON, so a provider using a
/// different ceremony (device code, paste-a-key) is a new variant rather than a
/// reinterpretation of these fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum AuthSpec {
    /// OAuth 2.1 authorization code + PKCE S256, with the redirect landing on a
    /// loopback listener this app binds for the duration of the ceremony.
    Oauth2Pkce(Oauth2Pkce),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Oauth2Pkce {
    /// Sent as `client_id`. A public client — there is no secret, because a
    /// secret shipped in a desktop binary is not a secret.
    pub client_id: String,
    /// Joined to the base URL the user supplies. Relative so one descriptor
    /// serves every deployment of that provider.
    pub authorize_path: String,
    pub token_path: String,
    pub scopes: Vec<String>,
    /// Path on our loopback listener. The provider must have registered a
    /// matching loopback redirect.
    #[serde(default = "default_callback_path")]
    pub callback_path: String,
    /// Query parameter carrying a human label for this machine, if the provider
    /// shows one on its consent screen. Omitted when it does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_param: Option<String>,
    /// Anything else the provider requires on the authorize URL.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_authorize_params: BTreeMap<String, String>,
    /// Where the interesting values live in the token response.
    pub token_response: TokenResponseSpec,
}

fn default_callback_path() -> String {
    "/callback".to_string()
}

/// Which fields of the token response mean what.
///
/// Providers disagree here — one returns `access_token`, another a
/// product-specific key name — and that disagreement is data, not a reason for
/// the exchange code to know about any of them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TokenResponseSpec {
    /// The credential to store. Tried in order; the first present non-empty
    /// string wins.
    pub credential_fields: Vec<String>,
    /// A stable id for the connection the provider created, if it makes one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id_field: Option<String>,
    /// A human label to show ("Reception PC").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_name_field: Option<String>,
    /// The scopes actually granted, which may be narrower than requested.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_field: Option<String>,
}

/// A provider this desktop can link to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConnectorDescriptor {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub docs_url: Option<String>,
    /// Prefilled in the UI. The user can still point at their own deployment,
    /// which is why every path above is relative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_base_url: Option<String>,
    pub auth: AuthSpec,
    /// Optional GET that proves the stored credential still works.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub health_path: Option<String>,
    /// How this provider is told the desktop is still reachable. Omitted for a
    /// provider that tracks presence some other way, or not at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heartbeat: Option<HeartbeatSpec>,
    /// How this provider queues remote-control commands for the desktop.
    /// Omitted for a provider that has no such lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay: Option<RelaySpec>,
    /// How this provider tunnels end-to-end sealed AI turns to the desktop.
    /// Omitted for a provider whose web app has no such feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desktop_ai: Option<DesktopAiSpec>,
}

/// The sealed AI lane: the provider's web app relays a chat turn the backend
/// cannot read, and this desktop answers it with the account's own model.
///
/// Separate from [`RelaySpec`] because it is a different lane with a different
/// queue and different semantics — a long chat turn must not block service
/// control, and the provider treats the two independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DesktopAiSpec {
    /// Where this desktop publishes the public half of its sealing key. Until
    /// it does, the provider's web app cannot encrypt anything to it and says
    /// so — which is the whole visible symptom of this lane being absent.
    pub pubkey_path: String,
    /// Long-poll for sealed turns addressed to this instance.
    pub pending_path: String,
    /// Take one turn, exactly-once. `{id}` is substituted.
    pub claim_path: String,
    /// Append one sealed frame. `{id}` is substituted.
    pub frames_path: String,
    /// Report the terminal status. `{id}` is substituted.
    pub complete_path: String,
    #[serde(default = "default_relay_wait")]
    pub wait_seconds: u64,
    #[serde(default = "default_ai_batch")]
    pub batch_limit: u32,
    #[serde(default = "default_relay_backoff")]
    pub error_backoff_seconds: u64,
}

fn default_ai_batch() -> u32 {
    8
}

/// The long-poll / claim / complete lane a provider's web app uses to act on
/// this desktop from anywhere.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RelaySpec {
    /// Long-poll for work addressed to this instance.
    pub pending_path: String,
    /// Take one command, exactly-once. `{id}` is substituted.
    pub claim_path: String,
    /// Report the outcome. `{id}` is substituted.
    pub complete_path: String,
    /// How long the server may hold the poll open.
    #[serde(default = "default_relay_wait")]
    pub wait_seconds: u64,
    #[serde(default = "default_relay_batch")]
    pub batch_limit: u32,
    /// Pause after a failed poll, so a provider that is down does not become a
    /// hot loop against it.
    #[serde(default = "default_relay_backoff")]
    pub error_backoff_seconds: u64,
}

fn default_relay_wait() -> u64 {
    25
}

fn default_relay_batch() -> u32 {
    20
}

fn default_relay_backoff() -> u64 {
    10
}

/// The periodic ping that keeps a desktop looking online.
///
/// Providers decide reachability from how recently the desktop last spoke, so
/// the interval has to sit comfortably inside their window — FormLogic's is 90
/// seconds and its own desktop beats every 45.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeartbeatSpec {
    pub path: String,
    #[serde(default = "default_heartbeat_interval")]
    pub interval_seconds: u64,
    /// Body field carrying this install's stable id.
    pub instance_id_field: String,
    /// Body field carrying a human label, when the provider shows one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name_field: Option<String>,
}

fn default_heartbeat_interval() -> u64 {
    45
}

impl ConnectorDescriptor {
    fn validate(&self) -> Result<(), String> {
        if self.id.is_empty()
            || self.id.len() > 64
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
        {
            return Err(format!("connector id {:?} is not a safe identifier", self.id));
        }
        if self.name.trim().is_empty() {
            return Err(format!("connector {:?} has no name", self.id));
        }
        let AuthSpec::Oauth2Pkce(o) = &self.auth;
        if o.client_id.trim().is_empty() {
            return Err(format!("connector {:?} has no clientId", self.id));
        }
        for (label, path) in [
            ("authorizePath", &o.authorize_path),
            ("tokenPath", &o.token_path),
            ("callbackPath", &o.callback_path),
        ] {
            if !path.starts_with('/') {
                return Err(format!(
                    "connector {:?} {label} must be a path beginning with '/', got {path:?}",
                    self.id
                ));
            }
        }
        if o.scopes.is_empty() {
            return Err(format!("connector {:?} requests no scopes", self.id));
        }
        if let Some(h) = &self.heartbeat {
            if !h.path.starts_with('/') {
                return Err(format!(
                    "connector {:?} heartbeat path must begin with '/', got {:?}",
                    self.id, h.path
                ));
            }
            if h.instance_id_field.trim().is_empty() {
                return Err(format!("connector {:?} heartbeat names no instance id field", self.id));
            }
            // A zero interval would spin; one longer than any plausible presence
            // window would beat too rarely to keep the desktop online at all.
            if h.interval_seconds == 0 || h.interval_seconds > 3600 {
                return Err(format!(
                    "connector {:?} heartbeat interval {} is out of range (1..3600s)",
                    self.id, h.interval_seconds
                ));
            }
        }
        if let Some(r) = &self.relay {
            for (label, path) in [
                ("pendingPath", &r.pending_path),
                ("claimPath", &r.claim_path),
                ("completePath", &r.complete_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} relay {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            // Without the placeholder every claim would hit one fixed URL and
            // quietly do nothing useful.
            for (label, path) in [("claimPath", &r.claim_path), ("completePath", &r.complete_path)] {
                if !path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} relay {label} must contain the {{id}} placeholder",
                        self.id
                    ));
                }
            }
            if r.wait_seconds == 0 || r.wait_seconds > 300 {
                return Err(format!(
                    "connector {:?} relay wait {} is out of range (1..300s)",
                    self.id, r.wait_seconds
                ));
            }
        }
        if let Some(a) = &self.desktop_ai {
            for (label, path) in [
                ("pubkeyPath", &a.pubkey_path),
                ("pendingPath", &a.pending_path),
                ("claimPath", &a.claim_path),
                ("framesPath", &a.frames_path),
                ("completePath", &a.complete_path),
            ] {
                if !path.starts_with('/') {
                    return Err(format!(
                        "connector {:?} desktopAi {label} must begin with '/', got {path:?}",
                        self.id
                    ));
                }
            }
            for (label, path) in [
                ("claimPath", &a.claim_path),
                ("framesPath", &a.frames_path),
                ("completePath", &a.complete_path),
            ] {
                if !path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} desktopAi {label} must contain the {{id}} placeholder",
                        self.id
                    ));
                }
            }
            // The publish and the poll are not per-turn, so a placeholder there
            // would be sent to the provider literally.
            for (label, path) in [("pubkeyPath", &a.pubkey_path), ("pendingPath", &a.pending_path)] {
                if path.contains("{id}") {
                    return Err(format!(
                        "connector {:?} desktopAi {label} takes no {{id}} placeholder",
                        self.id
                    ));
                }
            }
            if a.wait_seconds == 0 || a.wait_seconds > 300 {
                return Err(format!(
                    "connector {:?} desktopAi wait {} is out of range (1..300s)",
                    self.id, a.wait_seconds
                ));
            }
        }
        if o.token_response.credential_fields.is_empty() {
            return Err(format!(
                "connector {:?} names no credential field, so a successful \
                 exchange could not be read",
                self.id
            ));
        }
        Ok(())
    }
}

/// Every descriptor available, built-ins first, user files overriding by id.
///
/// A malformed user file is skipped with a log rather than failing the load —
/// one bad file must not take away every provider, including the working ones.
pub fn load_all(data_dir: &Path) -> Vec<ConnectorDescriptor> {
    let mut out: Vec<ConnectorDescriptor> = Vec::new();
    for raw in BUILTIN {
        match serde_json::from_str::<ConnectorDescriptor>(raw) {
            Ok(d) => match d.validate() {
                Ok(()) => out.push(d),
                // A broken built-in is our bug, not the user's; loud in dev.
                Err(e) => debug_assert!(false, "built-in connector is invalid: {e}"),
            },
            Err(e) => debug_assert!(false, "built-in connector does not parse: {e}"),
        }
    }

    let dir = data_dir.join("connectors");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        // Notepad and PowerShell's `-Encoding utf8` both prepend a byte order
        // mark, and serde stops at it with "expected value at line 1 column 1".
        // Since the file is then skipped with only a log line, the user is told
        // to drop a descriptor in a folder and gets no provider and no reason.
        match serde_json::from_str::<ConnectorDescriptor>(raw.trim_start_matches('\u{feff}')) {
            Ok(d) => match d.validate() {
                Ok(()) => {
                    if let Some(slot) = out.iter_mut().find(|e| e.id == d.id) {
                        *slot = d;
                    } else {
                        out.push(d);
                    }
                }
                Err(e) => log::warn!("ignoring connector {}: {e}", path.display()),
            },
            Err(e) => log::warn!("ignoring connector {}: {e}", path.display()),
        }
    }
    out
}

pub fn find(data_dir: &Path, id: &str) -> Option<ConnectorDescriptor> {
    load_all(data_dir).into_iter().find(|d| d.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("oaiy-conn-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(p.join("connectors")).unwrap();
        p
    }

    #[test]
    fn the_builtin_descriptor_parses_and_validates() {
        // It is compiled in, so a mistake here ships. The debug_assert in
        // load_all would fire in dev, but this says so with a name.
        let all = load_all(std::path::Path::new("/nonexistent"));
        assert_eq!(all.len(), 1, "expected exactly the shipped built-in");
        let d = &all[0];
        assert_eq!(d.id, "formlogic");
        d.validate().unwrap();
        let AuthSpec::Oauth2Pkce(o) = &d.auth;
        assert!(!o.scopes.is_empty());
        assert!(
            o.token_response.credential_fields.iter().any(|f| f == "formlogic_api_key"),
            "the provider's own key field must be first-class"
        );
    }

    #[test]
    fn a_user_file_overrides_a_builtin_of_the_same_id() {
        // How someone points at a fork or a staging deployment without waiting
        // for a release.
        let d = dir("override");
        let mut custom: ConnectorDescriptor =
            serde_json::from_str(BUILTIN[0]).unwrap();
        custom.name = "My FormLogic".into();
        std::fs::write(
            d.join("connectors").join("formlogic.json"),
            serde_json::to_string(&custom).unwrap(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1, "an override must replace, not duplicate");
        assert_eq!(all[0].name, "My FormLogic");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_descriptor_saved_by_a_windows_editor_still_loads() {
        // Found the hard way: PowerShell's `-Encoding utf8` and Notepad both
        // write a BOM, serde refuses it at column 1, and the file is skipped
        // with nothing but a log line. The panel tells people to drop a file in
        // this folder, so the most likely way to write one must work.
        let d = dir("bom");
        let mut custom: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        custom.name = "BOM'd FormLogic".into();
        std::fs::write(
            d.join("connectors").join("formlogic.json"),
            format!("\u{feff}{}", serde_json::to_string(&custom).unwrap()),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "BOM'd FormLogic", "the BOM must not hide the file");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_new_provider_is_a_file_not_a_code_change() {
        // The whole point of the descriptor. If this test needs code to change,
        // the design has failed.
        let d = dir("newprovider");
        let other = serde_json::json!({
            "id": "acme",
            "name": "Acme Cloud",
            "auth": {
                "kind": "oauth2_pkce",
                "clientId": "acme-desktop",
                "authorizePath": "/authorize",
                "tokenPath": "/oauth/token",
                "scopes": ["data:read"],
                "tokenResponse": { "credentialFields": ["access_token"] }
            }
        });
        std::fs::write(
            d.join("connectors").join("acme.json"),
            serde_json::to_string(&other).unwrap(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 2);
        let acme = find(&d, "acme").expect("the new provider is available");
        assert_eq!(acme.name, "Acme Cloud");
        let AuthSpec::Oauth2Pkce(o) = &acme.auth;
        assert_eq!(o.callback_path, "/callback", "defaulted, not required");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_malformed_user_file_does_not_remove_the_working_providers() {
        // One bad file must cost only itself.
        let d = dir("malformed");
        std::fs::write(d.join("connectors").join("broken.json"), "{ not json").unwrap();
        std::fs::write(
            d.join("connectors").join("nocreds.json"),
            serde_json::json!({
                "id": "nocreds", "name": "No Creds",
                "auth": {
                    "kind": "oauth2_pkce", "clientId": "x",
                    "authorizePath": "/a", "tokenPath": "/t", "scopes": ["s"],
                    "tokenResponse": { "credentialFields": [] }
                }
            })
            .to_string(),
        )
        .unwrap();

        let all = load_all(&d);
        assert_eq!(all.len(), 1, "the built-in must survive its neighbours");
        assert_eq!(all[0].id, "formlogic");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn paths_must_be_paths_so_a_descriptor_cannot_redirect_the_ceremony() {
        // An absolute URL in authorizePath would send the user — and the
        // authorization code — to a host the base URL never named.
        let mut d: ConnectorDescriptor = serde_json::from_str(BUILTIN[0]).unwrap();
        let AuthSpec::Oauth2Pkce(o) = &mut d.auth;
        o.authorize_path = "https://evil.example/authorize".into();
        let err = d.validate().unwrap_err();
        assert!(err.contains("authorizePath"), "{err}");
    }
}
