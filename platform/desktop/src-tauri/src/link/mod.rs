//! Linking this desktop to ONE account on a remote provider.
//!
//! Two rules shape this module:
//!
//! 1. **One connection.** A desktop is a machine belonging to somebody; letting
//!    it hold several accounts at once would make "which account did that flow
//!    run under" a question with no answer.
//! 2. **The provider is data.** Every endpoint, scope and field name comes from
//!    a [`descriptor::ConnectorDescriptor`], so nothing below names a product.
//!    A second provider is a JSON file.
//!
//! Not to be confused with the PLUGIN connectors elsewhere in this crate
//! (`connector.aokie.sms.send`), which are command namespaces a plugin exposes.
//! This is an account link — outbound, one per machine.

pub mod descriptor;
pub mod oauth;
pub mod routes;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub use descriptor::ConnectorDescriptor;

/// The stored link. The credential is in here and never leaves the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LinkedAccount {
    /// Which descriptor this was made with.
    pub connector_id: String,
    pub base_url: String,
    /// Never serialized outward — [`LinkStatus`] reports only that one exists.
    pub credential: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub granted_scopes: Option<String>,
    pub linked_at: chrono::DateTime<chrono::Utc>,
}

/// Where a link attempt has got to.
///
/// Reported by polling rather than pushed: the ceremony happens in the user's
/// browser, so the UI has to ask anyway, and a poll needs no event plumbing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "phase")]
pub enum LinkPhase {
    Idle,
    /// The browser has been opened; we are holding the loopback port.
    AwaitingBrowser { authorize_url: String },
    Exchanging,
    Linked,
    Failed { message: String },
}

/// What a UI may know. No credential, ever.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkStatus {
    pub linked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_scopes: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linked_at: Option<chrono::DateTime<chrono::Utc>>,
    /// The in-flight attempt, if any.
    pub attempt: LinkPhase,
    /// Every provider this build can link to.
    pub available: Vec<AvailableConnector>,
}

/// A provider offered in the UI, without its machinery.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AvailableConnector {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_base_url: Option<String>,
    pub scopes: Vec<String>,
}

struct Inner {
    account: Option<LinkedAccount>,
    attempt: LinkPhase,
    /// Set while a ceremony is running, so a second Link click cannot open a
    /// second browser tab racing the first for the same one-use code.
    in_flight: bool,
}

pub struct LinkStore {
    path: PathBuf,
    data_dir: PathBuf,
    inner: Mutex<Inner>,
}

pub type LinkHandle = Arc<LinkStore>;

pub fn open_handle(data_dir: PathBuf) -> LinkHandle {
    let path = data_dir.join("link").join("account.json");
    let account = std::fs::read_to_string(&path)
        .ok()
        .and_then(|raw| serde_json::from_str::<LinkedAccount>(&raw).ok());
    Arc::new(LinkStore {
        path,
        data_dir,
        inner: Mutex::new(Inner {
            account,
            attempt: LinkPhase::Idle,
            in_flight: false,
        }),
    })
}

impl LinkStore {
    pub fn status(&self) -> LinkStatus {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let descriptors = descriptor::load_all(&self.data_dir);
        let available = descriptors
            .iter()
            .map(|d| {
                let descriptor::AuthSpec::Oauth2Pkce(o) = &d.auth;
                AvailableConnector {
                    id: d.id.clone(),
                    name: d.name.clone(),
                    description: d.description.clone(),
                    docs_url: d.docs_url.clone(),
                    default_base_url: d.default_base_url.clone(),
                    scopes: o.scopes.clone(),
                }
            })
            .collect();
        match inner.account.as_ref() {
            None => LinkStatus {
                linked: false,
                connector_id: None,
                connector_name: None,
                base_url: None,
                account_name: None,
                account_id: None,
                granted_scopes: None,
                linked_at: None,
                attempt: inner.attempt.clone(),
                available,
            },
            Some(a) => LinkStatus {
                linked: true,
                connector_name: descriptors
                    .iter()
                    .find(|d| d.id == a.connector_id)
                    .map(|d| d.name.clone()),
                connector_id: Some(a.connector_id.clone()),
                base_url: Some(a.base_url.clone()),
                account_name: a.account_name.clone(),
                account_id: a.account_id.clone(),
                granted_scopes: a.granted_scopes.clone(),
                linked_at: Some(a.linked_at),
                attempt: inner.attempt.clone(),
                available,
            },
        }
    }

    /// The stored credential, for whoever needs to call the provider.
    pub fn account(&self) -> Option<LinkedAccount> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .account
            .clone()
    }

    fn persist(&self, account: &LinkedAccount) -> Result<(), String> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("could not create the link directory: {e}"))?;
        }
        let raw = serde_json::to_string_pretty(account)
            .map_err(|e| format!("could not encode the link: {e}"))?;
        std::fs::write(&self.path, raw)
            .map_err(|e| format!("could not save the link: {e}"))?;
        restrict_to_owner(&self.path);
        Ok(())
    }

    /// Forget the link. Local only — see the note on the route.
    pub fn unlink(&self) -> LinkStatus {
        let _ = std::fs::remove_file(&self.path);
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.account = None;
            inner.attempt = LinkPhase::Idle;
        }
        self.status()
    }

    fn set_phase(&self, phase: LinkPhase) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).attempt = phase;
    }

    /// Claim the single in-flight slot.
    fn begin(&self) -> Result<(), String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.in_flight {
            return Err("a link attempt is already in progress".into());
        }
        inner.in_flight = true;
        inner.attempt = LinkPhase::Idle;
        Ok(())
    }

    fn end(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).in_flight = false;
    }
}

/// Keep the credential out of reach of other accounts on a shared machine.
fn restrict_to_owner(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(windows))]
    let _ = path;
}

/// Normalise a user-typed base URL, refusing one we should not send a
/// credential to.
pub fn normalize_base(raw: &str) -> Result<String, String> {
    let base = raw.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return Err("enter the provider's address".into());
    }
    if base.len() > 512 {
        return Err("that address is too long".into());
    }
    let is_loopback =
        base.starts_with("http://127.0.0.1") || base.starts_with("http://localhost");
    if !base.starts_with("https://") && !is_loopback {
        return Err("the address must be https (or loopback for local testing)".into());
    }
    if base.contains(['?', '#', ' ']) {
        return Err("the address must be a plain origin, with no query or fragment".into());
    }
    Ok(base)
}

/// This machine's name, for the provider's consent screen.
fn device_label() -> Option<String> {
    for key in ["COMPUTERNAME", "HOSTNAME"] {
        if let Ok(v) = std::env::var(key) {
            let v = v.trim().to_string();
            if !v.is_empty() {
                return Some(v.chars().take(100).collect());
            }
        }
    }
    None
}

/// Run the whole ceremony on a worker thread.
///
/// Returns immediately with the URL to open; the caller polls `status()`.
/// Blocking work — a loopback wait bounded by a human, then an HTTP exchange —
/// has no business on the request path.
pub fn start_link(
    store: LinkHandle,
    connector_id: &str,
    base_url: &str,
    open_browser: impl Fn(&str) + Send + 'static,
) -> Result<String, String> {
    let descriptor = descriptor::find(&store.data_dir, connector_id)
        .ok_or_else(|| format!("no connector named {connector_id:?}"))?;
    let base = normalize_base(base_url)?;
    let descriptor::AuthSpec::Oauth2Pkce(spec) = &descriptor.auth;
    let callback_path = spec.callback_path.clone();

    store.begin()?;
    // From here every exit path must clear the slot, or Link is dead until
    // restart.
    let result = (|| -> Result<(oauth::Loopback, oauth::Pkce, String, String), String> {
        let loopback = oauth::Loopback::bind(&callback_path)?;
        let pkce = oauth::generate_pkce()?;
        let state = oauth::random_token(16)?;
        let url = oauth::authorize_url(
            &descriptor,
            &base,
            &loopback.redirect_uri,
            &pkce,
            &state,
            device_label().as_deref(),
        );
        Ok((loopback, pkce, state, url))
    })();
    let (loopback, pkce, state, url) = match result {
        Ok(v) => v,
        Err(e) => {
            store.end();
            store.set_phase(LinkPhase::Failed { message: e.clone() });
            return Err(e);
        }
    };

    store.set_phase(LinkPhase::AwaitingBrowser {
        authorize_url: url.clone(),
    });

    let for_thread = store.clone();
    let url_for_thread = url.clone();
    std::thread::spawn(move || {
        open_browser(&url_for_thread);
        let deadline = Instant::now() + oauth::LINK_TIMEOUT;
        let outcome = (|| -> Result<LinkedAccount, String> {
            let params = loopback.wait(&callback_path, deadline)?;
            if let Some(err) = params.get("error") {
                let detail = params
                    .get("error_description")
                    .map(|d| format!(": {d}"))
                    .unwrap_or_default();
                return Err(format!("the provider refused the link ({err}){detail}"));
            }
            // Compared before the code is spent: a mismatch means this redirect
            // answers a ceremony we did not start.
            let got_state = params.get("state").map(String::as_str).unwrap_or("");
            if got_state != state {
                return Err("security check failed (state mismatch)".into());
            }
            let code = params
                .get("code")
                .filter(|c| !c.is_empty())
                .ok_or("the provider returned no authorization code")?;

            for_thread.set_phase(LinkPhase::Exchanging);
            let creds = oauth::exchange_code(
                &descriptor,
                &base,
                &loopback.redirect_uri,
                code,
                &pkce.verifier,
            )?;
            Ok(LinkedAccount {
                connector_id: descriptor.id.clone(),
                base_url: base.clone(),
                credential: creds.credential,
                account_id: creds.account_id,
                account_name: creds.account_name,
                granted_scopes: creds.granted_scopes,
                linked_at: chrono::Utc::now(),
            })
        })();

        match outcome {
            Ok(account) => match for_thread.persist(&account) {
                Ok(()) => {
                    let mut inner = for_thread.inner.lock().unwrap_or_else(|e| e.into_inner());
                    inner.account = Some(account);
                    inner.attempt = LinkPhase::Linked;
                }
                Err(e) => for_thread.set_phase(LinkPhase::Failed { message: e }),
            },
            Err(e) => for_thread.set_phase(LinkPhase::Failed { message: e }),
        }
        for_thread.end();
    });

    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(tag: &str) -> (std::path::PathBuf, LinkHandle) {
        let p = std::env::temp_dir().join(format!("oaiy-link-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        let h = open_handle(p.clone());
        (p, h)
    }

    fn account() -> LinkedAccount {
        LinkedAccount {
            connector_id: "formlogic".into(),
            base_url: "https://formlogic.com".into(),
            credential: "flk_supersecret".into(),
            account_id: Some("conn_1".into()),
            account_name: Some("Reception PC".into()),
            granted_scopes: Some("flows:read flows:write".into()),
            linked_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn the_status_never_carries_the_credential() {
        // It is read by the UI and by anything with the bearer token; the whole
        // point of holding it in the host is that it does not travel.
        let (dir, s) = store("secret");
        s.persist(&account()).unwrap();
        s.inner.lock().unwrap().account = Some(account());

        let status = s.status();
        assert!(status.linked);
        assert_eq!(status.account_name.as_deref(), Some("Reception PC"));
        let raw = serde_json::to_string(&status).unwrap();
        assert!(!raw.contains("flk_supersecret"), "{raw}");
        assert!(!raw.contains("credential"), "{raw}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_link_survives_a_restart_and_unlink_forgets_it() {
        let (dir, s) = store("persist");
        s.persist(&account()).unwrap();

        let reopened = open_handle(dir.clone());
        let a = reopened.account().expect("the link must survive");
        assert_eq!(a.credential, "flk_supersecret");
        assert_eq!(a.connector_id, "formlogic");

        assert!(!reopened.unlink().linked);
        assert!(open_handle(dir.clone()).account().is_none(), "unlink must be durable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_one_attempt_may_be_in_flight() {
        // Two browser tabs would race for one single-use code, and the loser's
        // failure would look like a bug in the provider.
        let (dir, s) = store("inflight");
        s.begin().unwrap();
        let err = s.begin().unwrap_err();
        assert!(err.contains("already in progress"), "{err}");
        s.end();
        s.begin().expect("the slot must be reusable once the attempt ends");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_start_releases_the_slot_rather_than_wedging_link_forever() {
        // The bug this guards: an early return between begin() and the worker
        // thread leaves in_flight set, and Link stays dead until restart.
        let (dir, s) = store("release");
        let err = start_link(s.clone(), "formlogic", "not-a-url", |_| {}).unwrap_err();
        assert!(err.contains("https"), "{err}");
        s.begin().expect("the slot must have been released");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_connector_is_refused_before_anything_is_opened() {
        let (dir, s) = store("unknown");
        let err = start_link(s.clone(), "nosuch", "https://x.example", |_| {
            panic!("must not open a browser for an unknown connector")
        })
        .unwrap_err();
        assert!(err.contains("nosuch"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_status_lists_every_available_connector_so_the_ui_needs_no_hardcoded_list() {
        let (dir, s) = store("available");
        let status = s.status();
        assert!(!status.linked);
        assert_eq!(status.available.len(), 1);
        assert_eq!(status.available[0].id, "formlogic");
        assert!(!status.available[0].scopes.is_empty(), "the UI shows what is granted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plain_http_to_a_remote_provider_is_refused() {
        assert!(normalize_base("http://formlogic.example").is_err());
        assert!(normalize_base("https://formlogic.com/").unwrap() == "https://formlogic.com");
        assert!(normalize_base("http://127.0.0.1:8080").is_ok(), "local dev stays possible");
        assert!(normalize_base("  ").is_err());
        // A query or fragment would survive into every joined path.
        assert!(normalize_base("https://x.example/?a=1").is_err());
    }
}
