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

pub mod condition;
pub mod data_node;
pub mod descriptor;
pub mod flow_runner;
pub mod flows;
pub mod heartbeat;
pub mod oauth;
pub mod ops;
pub mod relay;
pub mod routes;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub use descriptor::ConnectorDescriptor;

/// Origin of the provider this desktop is linked to, if any.
///
/// A process-wide cell rather than a parameter because the origin allow-list in
/// `http.rs` is a free function consulted per request, long before any handle is
/// in scope. Written whenever the link changes.
///
/// Why trust it at all: linking is the user explicitly approving that provider,
/// and its web app is the surface they then expect to manage this desktop from.
/// Deriving the trusted origin FROM the link means no address is ever hardcoded
/// and unlinking withdraws the trust in the same action.
static LINKED_ORIGIN: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// The linked provider's origin, for the origin allow-list.
pub fn linked_origin() -> Option<String> {
    LINKED_ORIGIN.read().ok().and_then(|g| g.clone())
}

/// Set the trusted origin directly, for tests in other modules.
#[cfg(test)]
pub fn set_linked_origin_for_tests(base_url: Option<&str>) {
    set_linked_origin(base_url);
}

fn set_linked_origin(base_url: Option<&str>) {
    if let Ok(mut guard) = LINKED_ORIGIN.write() {
        // Store the ORIGIN, not the base: a provider served under a path still
        // sends `Origin: scheme://host[:port]`.
        *guard = base_url.and_then(origin_of);
    }
}

/// `scheme://host[:port]` from a base URL, or `None` if it is not one we would
/// have accepted in the first place.
fn origin_of(base_url: &str) -> Option<String> {
    let (scheme, rest) = if let Some(r) = base_url.strip_prefix("https://") {
        ("https", r)
    } else if let Some(r) = base_url.strip_prefix("http://") {
        ("http", r)
    } else {
        return None;
    };
    let authority = rest.split(['/', '?', '#']).next().filter(|a| !a.is_empty())?;
    if authority.contains('@') {
        return None;
    }
    Some(format!("{scheme}://{authority}"))
}

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
    /// Stable id for THIS install, sent with every heartbeat.
    ///
    /// Optional so a link stored before heartbeats existed still loads; one is
    /// minted on first use. Persisted because a changing id reads as a new
    /// desktop each launch and leaves ghost rows behind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
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
    /// The user stopped it from this app. Distinct from `Failed` because
    /// nothing went wrong and an error banner would be a lie.
    Cancelled,
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
    /// When the provider was last told this desktop is here, and why the last
    /// attempt failed if it did. A link that looks fine but shows offline at the
    /// provider is otherwise unexplainable from this side.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heartbeat_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heartbeat_error: Option<String>,
    /// When the command lane last polled cleanly, and why it stopped if it did.
    ///
    /// Surfaced for the same reason as the heartbeat, and it is the half that
    /// bites harder: a relay that is failing shows up ONLY on the provider's
    /// website, as "no desktop picked it up in time" — which reads as a broken
    /// connection and sends the user looking in the wrong place entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_relay_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relay_error: Option<String>,
    /// When this desktop last looked at the account's queued flow runs, and why
    /// it stopped if it did.
    ///
    /// The same invisible failure as the relay's, one step worse: a run this
    /// desktop CLAIMED and could not report on is marked running at the
    /// provider, so no other runtime will take it either. That has to be
    /// readable from here, because it is not readable from anywhere else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_flow_run_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flow_run_error: Option<String>,
    /// Whether the linked connector declares each lane at all.
    ///
    /// Taken from the descriptor, not from whether anything has happened yet: a
    /// lane a provider simply does not have must read as absent, never as
    /// pending. Without these, a connector with no relay would sit under a
    /// "connecting…" that is never going to resolve.
    pub heartbeat_supported: bool,
    pub relay_supported: bool,
    /// Whether the connector declares a flow lane this desktop can claim FROM,
    /// as opposed to one it can only reserve INTO. Distinguishing the two is
    /// what keeps "this provider runs its own flows" from reading as "flow
    /// execution is broken here".
    pub flow_runs_supported: bool,
    /// This desktop's storage-node enrolment, once the provider has answered.
    /// The fingerprint here is what the owner compares against their browser
    /// before approving — the whole ceremony rests on the two matching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_node: Option<data_node::DataNodeStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_node_error: Option<String>,
    pub data_node_supported: bool,
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
    /// Raised to stop the current attempt. A fresh flag PER ATTEMPT, so a
    /// cancel can never carry over and kill the next one.
    cancel: Arc<AtomicBool>,
    last_heartbeat_at: Option<chrono::DateTime<chrono::Utc>>,
    heartbeat_error: Option<String>,
    last_relay_at: Option<chrono::DateTime<chrono::Utc>>,
    relay_error: Option<String>,
    last_flow_run_at: Option<chrono::DateTime<chrono::Utc>>,
    flow_run_error: Option<String>,
    data_node: Option<data_node::DataNodeStatus>,
    data_node_error: Option<String>,
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
    let store = Arc::new(LinkStore {
        path,
        data_dir,
        inner: Mutex::new(Inner {
            account,
            attempt: LinkPhase::Idle,
            in_flight: false,
            cancel: Arc::new(AtomicBool::new(false)),
            last_heartbeat_at: None,
            heartbeat_error: None,
            last_relay_at: None,
            relay_error: None,
            last_flow_run_at: None,
            flow_run_error: None,
            data_node: None,
            data_node_error: None,
        }),
    });
    // One worker for the process lifetime. It asks the store what to do each
    // tick, so linking and unlinking need not start or stop anything.
    heartbeat::spawn(store.clone());
    // Enrolment as a storage node the owner can approve. Separate from the
    // heartbeat: presence is per-minute, enrolment is per-hour and its answer
    // is a state the user acts on rather than a liveness signal.
    data_node::spawn(store.clone());
    set_linked_origin(store.account().as_ref().map(|a| a.base_url.as_str()));
    store
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
                last_heartbeat_at: None,
                heartbeat_error: None,
                last_relay_at: None,
                relay_error: None,
                last_flow_run_at: None,
                flow_run_error: None,
                heartbeat_supported: false,
                relay_supported: false,
                flow_runs_supported: false,
                data_node: None,
                data_node_error: None,
                data_node_supported: false,
                attempt: inner.attempt.clone(),
                available,
            },
            Some(a) => {
                // The descriptor says which lanes this provider has at all, so
                // the panel can distinguish a lane that is broken from one that
                // was never there.
                let d = descriptors.iter().find(|d| d.id == a.connector_id);
                LinkStatus {
                    linked: true,
                    connector_name: d.map(|d| d.name.clone()),
                    connector_id: Some(a.connector_id.clone()),
                    base_url: Some(a.base_url.clone()),
                    account_name: a.account_name.clone(),
                    account_id: a.account_id.clone(),
                    granted_scopes: a.granted_scopes.clone(),
                    linked_at: Some(a.linked_at),
                    last_heartbeat_at: inner.last_heartbeat_at,
                    heartbeat_error: inner.heartbeat_error.clone(),
                    last_relay_at: inner.last_relay_at,
                    relay_error: inner.relay_error.clone(),
                    last_flow_run_at: inner.last_flow_run_at,
                    flow_run_error: inner.flow_run_error.clone(),
                    heartbeat_supported: d.is_some_and(|d| d.heartbeat.is_some()),
                    relay_supported: d.is_some_and(|d| d.relay.is_some()),
                    // Every path of the claim lane, not just some: a provider
                    // missing one of them cannot have its runs executed here.
                    flow_runs_supported: d.and_then(|d| d.flows.as_ref()).is_some_and(|f| {
                        f.queued_path.is_some()
                            && f.claim_path.is_some()
                            && f.complete_path.is_some()
                            && f.graph_path.is_some()
                    }),
                    data_node: inner.data_node.clone(),
                    data_node_error: inner.data_node_error.clone(),
                    data_node_supported: d.is_some_and(|d| d.data_node.is_some()),
                    attempt: inner.attempt.clone(),
                    available,
                }
            }
        }
    }

    pub fn data_dir(&self) -> &std::path::Path {
        &self.data_dir
    }

    /// This install's stable id, minted and persisted on first use.
    ///
    /// Lazily rather than at link time so a link stored before heartbeats
    /// existed gets one too, instead of beating with an empty id the provider
    /// would reject.
    pub fn instance_id(&self) -> String {
        {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(id) = inner.account.as_ref().and_then(|a| a.instance_id.clone()) {
                return id;
            }
        }
        let minted = heartbeat::new_instance_id();
        let to_save = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            match inner.account.as_mut() {
                // Another caller may have won the race; keep theirs so the id
                // stays stable.
                Some(a) => {
                    let id = a.instance_id.get_or_insert(minted).clone();
                    a.instance_id = Some(id.clone());
                    Some((a.clone(), id))
                }
                None => None,
            }
        };
        match to_save {
            Some((account, id)) => {
                let _ = self.persist(&account);
                id
            }
            None => heartbeat::new_instance_id(),
        }
    }

    /// Record the outcome of a relay poll.
    ///
    /// The timestamp is what makes "no error" mean something: the lane is a long
    /// poll, so a clean return every few seconds is the only evidence it is
    /// alive. Without it, "never started" and "running fine" look identical.
    pub fn note_relay(&self, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if error.is_none() {
            inner.last_relay_at = Some(chrono::Utc::now());
        }
        inner.relay_error = error;
    }

    /// Record the outcome of one look at the account's queued flow runs.
    ///
    /// The timestamp is what makes "no error" mean something: without it a lane
    /// that never started and one that is claiming and running flows every few
    /// seconds look identical from the panel.
    pub fn note_flow_run(&self, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if error.is_none() {
            inner.last_flow_run_at = Some(chrono::Utc::now());
        }
        inner.flow_run_error = error;
    }

    /// Record the outcome of a node registration.
    pub fn note_data_node(&self, outcome: Result<data_node::DataNodeStatus, String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match outcome {
            Ok(status) => {
                inner.data_node = Some(status);
                inner.data_node_error = None;
            }
            // The last known record is KEPT: a failed refresh does not undo an
            // approval, and blanking the fingerprint mid-ceremony would leave
            // the owner with nothing to compare against.
            Err(e) => inner.data_node_error = Some(e),
        }
    }

    /// Record the outcome of a heartbeat.
    pub fn note_heartbeat(&self, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if error.is_none() {
            inner.last_heartbeat_at = Some(chrono::Utc::now());
        }
        inner.heartbeat_error = error;
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
            inner.last_heartbeat_at = None;
            inner.heartbeat_error = None;
            // Both lanes, or a failure from the provider just disconnected
            // would be shown against the next one that is linked.
            inner.last_relay_at = None;
            inner.relay_error = None;
            inner.last_flow_run_at = None;
            inner.flow_run_error = None;
            // The IDENTITY stays on disk: the same machine relinking is the
            // same node, and re-minting would ask the owner to approve a device
            // they already approved. Only the provider's answer is forgotten.
            inner.data_node = None;
            inner.data_node_error = None;
        }
        // Withdrawn in the same action that forgot the link: a provider we are
        // no longer linked to must not keep reaching this machine.
        set_linked_origin(None);
        self.status()
    }

    fn set_phase(&self, phase: LinkPhase) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).attempt = phase;
    }

    /// Claim the single in-flight slot, returning this attempt's cancel flag.
    fn begin(&self) -> Result<Arc<AtomicBool>, String> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.in_flight {
            return Err("a link attempt is already in progress".into());
        }
        inner.in_flight = true;
        inner.attempt = LinkPhase::Idle;
        // A NEW flag, not a reset of the old one: a stale handle raising the
        // previous attempt's flag must not reach into this one.
        inner.cancel = Arc::new(AtomicBool::new(false));
        Ok(inner.cancel.clone())
    }

    /// Stop the attempt in flight, if there is one.
    ///
    /// Idempotent and safe when nothing is running — the button is allowed to
    /// be clicked twice, and the answer is the same either way.
    pub fn cancel(&self) -> LinkStatus {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.in_flight {
                inner.cancel.store(true, Ordering::Relaxed);
                // Reported immediately rather than waiting for the worker to
                // notice: the user pressed a button and deserves to see it take
                // effect. The worker clears in_flight when it unwinds.
                inner.attempt = LinkPhase::Cancelled;
            } else if matches!(inner.attempt, LinkPhase::Failed { .. }) {
                // Dismisses a stale error too, so the panel can return to a
                // clean state without a second control.
                inner.attempt = LinkPhase::Idle;
            }
        }
        self.status()
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
    // Everywhere the block above did not consume it. `not(windows)` would leave
    // it unused on Windows, which is the one platform this ships on.
    #[cfg(not(unix))]
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
    if !crate::origin::may_carry_credential(&base) {
        return Err(crate::origin::INSECURE_ADDRESS_HELP.into());
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

    let cancel = store.begin()?;
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
            let params = loopback.wait(&callback_path, deadline, &cancel)?;
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
                instance_id: Some(heartbeat::new_instance_id()),
            })
        })();

        match outcome {
            Ok(account) => match for_thread.persist(&account) {
                Ok(()) => {
                    set_linked_origin(Some(account.base_url.as_str()));
                    let mut inner = for_thread.inner.lock().unwrap_or_else(|e| e.into_inner());
                    inner.account = Some(account);
                    inner.attempt = LinkPhase::Linked;
                }
                Err(e) => for_thread.set_phase(LinkPhase::Failed { message: e }),
            },
            // A cancel is not a failure: the user asked for this, so the
            // panel must not accuse them of an error.
            Err(e) if e == "cancelled" => for_thread.set_phase(LinkPhase::Cancelled),
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
            instance_id: None,
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
    fn an_instance_id_is_minted_once_and_then_kept() {
        // A changing id reads as a NEW desktop on every launch, so the provider
        // accumulates ghost rows and ambiguity checks start refusing to route.
        let (dir, s) = store("instance");
        let mut a = account();
        a.instance_id = None;
        s.persist(&a).unwrap();
        s.inner.lock().unwrap().account = Some(a);

        let first = s.instance_id();
        assert!(first.starts_with("oaiy-"), "{first}");
        assert_eq!(s.instance_id(), first, "the same id within a session");
        // …and across a restart, which is the half that matters.
        assert_eq!(open_handle(dir.clone()).instance_id(), first);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_link_stored_before_heartbeats_existed_still_loads() {
        // The stored file has no instanceId field. deny_unknown_fields makes
        // the reverse fatal, so this is the direction worth pinning.
        let (dir, s) = store("legacy");
        let legacy = serde_json::json!({
            "connectorId": "formlogic",
            "baseUrl": "https://formlogic.com",
            "credential": "flk_old",
            "linkedAt": chrono::Utc::now(),
        });
        std::fs::create_dir_all(dir.join("link")).unwrap();
        std::fs::write(dir.join("link").join("account.json"), legacy.to_string()).unwrap();

        let reopened = open_handle(dir.clone());
        let a = reopened.account().expect("a pre-heartbeat link must still load");
        assert_eq!(a.credential, "flk_old");
        assert!(a.instance_id.is_none());
        assert!(reopened.instance_id().starts_with("oaiy-"), "one is minted on demand");
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_status_reports_why_a_heartbeat_failed() {
        // A link that looks fine but shows offline at the provider is otherwise
        // unexplainable from this side — which is exactly the "No Desktop"
        // confusion this whole mechanism exists to prevent.
        let (dir, s) = store("hb-status");
        s.persist(&account()).unwrap();
        s.inner.lock().unwrap().account = Some(account());

        s.note_heartbeat(Some("the provider no longer accepts this desktop's key".into()));
        let status = s.status();
        assert!(status.linked);
        assert!(status.last_heartbeat_at.is_none());
        assert!(status.heartbeat_error.unwrap().contains("no longer accepts"));

        s.note_heartbeat(None);
        let ok = s.status();
        assert!(ok.last_heartbeat_at.is_some());
        assert!(ok.heartbeat_error.is_none(), "success must clear the error");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_status_reports_why_the_command_lane_stopped() {
        // The heartbeat's twin, and the one that fails more confusingly: a
        // broken relay is visible ONLY on the provider's website, as "no desktop
        // picked it up in time" — which reads as a connection problem and sends
        // the user looking at their network instead of at the reason.
        let (dir, s) = store("relay-status");
        s.persist(&account()).unwrap();
        s.inner.lock().unwrap().account = Some(account());

        // Before anything has happened: no error, and nothing claiming success.
        let fresh = s.status();
        assert!(fresh.relay_error.is_none() && fresh.last_relay_at.is_none());

        s.note_relay(Some("claim refused: HTTP 500".into()));
        let bad = s.status();
        assert!(bad.relay_error.unwrap().contains("500"));
        assert!(bad.last_relay_at.is_none(), "a failed poll is not a poll");

        s.note_relay(None);
        let good = s.status();
        assert!(good.relay_error.is_none(), "success must clear the error");
        // The timestamp is what makes "no error" mean something: on a long poll,
        // a clean return is the only evidence the lane is alive at all.
        assert!(good.last_relay_at.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_status_says_which_lanes_the_connector_actually_has() {
        // Without this the panel cannot tell a lane that is broken from one the
        // provider never had, and a connector with no relay would sit forever
        // under a "connecting…" that is never going to resolve.
        let (dir, s) = store("lanes");
        s.persist(&account()).unwrap();
        s.inner.lock().unwrap().account = Some(account());

        let status = s.status();
        assert!(status.heartbeat_supported, "the built-in connector declares one");
        assert!(status.relay_supported, "…and a relay");

        // Unlinked, nothing is claimed either way.
        let after = s.unlink();
        assert!(!after.heartbeat_supported && !after.relay_supported);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unlinking_forgets_the_relays_failure_along_with_the_key() {
        // Otherwise the next provider linked on this machine inherits the last
        // one's error and looks broken from the moment it connects.
        let (dir, s) = store("relay-unlink");
        s.persist(&account()).unwrap();
        s.inner.lock().unwrap().account = Some(account());
        // A healthy poll first, so there is a timestamp to leak, then a failure.
        s.note_relay(None);
        s.note_relay(Some("the provider no longer accepts this desktop's key".into()));
        s.note_heartbeat(Some("boom".into()));
        assert!(s.status().last_relay_at.is_some(), "the setup must have left one");

        let after = s.unlink();
        assert!(after.relay_error.is_none(), "a stale error must not outlive the link");
        assert!(after.last_relay_at.is_none());
        assert!(after.heartbeat_error.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_builtin_connector_beats_inside_the_providers_presence_window() {
        // FormLogic marks a desktop offline after 90s without contact. An
        // interval at or above that would leave it flickering offline no matter
        // how healthy the link is.
        let d = descriptor::find(std::path::Path::new("/nonexistent"), "formlogic").unwrap();
        let h = d.heartbeat.expect("the connector must declare a heartbeat");
        assert!(h.interval_seconds <= 45, "got {}", h.interval_seconds);
        assert_eq!(h.instance_id_field, "desktopInstanceId");
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
    fn cancelling_stops_the_attempt_and_frees_the_slot() {
        let (dir, s) = store("cancel");
        let cancel = s.begin().unwrap();
        assert!(!cancel.load(Ordering::Relaxed));

        let status = s.cancel();
        assert!(cancel.load(Ordering::Relaxed), "the worker must see the flag");
        // Reported straight away rather than waiting for the worker to notice:
        // the user pressed a button and deserves to see it take effect.
        assert_eq!(status.attempt, LinkPhase::Cancelled);

        // The worker unwinds and releases the slot; then Link works again.
        s.end();
        s.begin().expect("a cancelled attempt must not wedge the next one");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cancel_cannot_leak_into_the_next_attempt() {
        // The bug a shared flag would cause: cancel once, and every future
        // attempt dies instantly with no explanation. Each attempt gets its own.
        let (dir, s) = store("cancel-leak");
        let first = s.begin().unwrap();
        s.cancel();
        s.end();

        let second = s.begin().unwrap();
        assert!(first.load(Ordering::Relaxed), "the old attempt stays cancelled");
        assert!(
            !second.load(Ordering::Relaxed),
            "a fresh attempt must start uncancelled"
        );
        // …and raising the STALE handle must not reach the live attempt.
        first.store(true, Ordering::Relaxed);
        assert!(!second.load(Ordering::Relaxed));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelling_when_nothing_is_running_is_harmless() {
        // The button is allowed to be clicked twice.
        let (dir, s) = store("cancel-idle");
        assert_eq!(s.cancel().attempt, LinkPhase::Idle);
        assert_eq!(s.cancel().attempt, LinkPhase::Idle);
        s.begin().expect("an idle cancel must not consume the slot");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancel_also_dismisses_a_stale_failure() {
        // Otherwise the panel keeps an old error banner with no way to clear it
        // short of a successful link.
        let (dir, s) = store("cancel-dismiss");
        s.set_phase(LinkPhase::Failed { message: "boom".into() });
        assert_eq!(s.cancel().attempt, LinkPhase::Idle);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cancelled_attempt_is_not_reported_as_a_failure() {
        // The user asked for it; an error banner would accuse them of a fault.
        let (dir, s) = store("cancel-not-failure");
        s.begin().unwrap();
        let status = s.cancel();
        assert_ne!(
            std::mem::discriminant(&status.attempt),
            std::mem::discriminant(&LinkPhase::Failed { message: String::new() }),
        );
        assert_eq!(status.attempt, LinkPhase::Cancelled);
        // And it leaves no account behind.
        assert!(!status.linked);
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
    fn a_local_deployment_can_be_linked_over_plain_http() {
        // The real case: FormLogic served by WAMP as http://formlogic.local.
        // Requiring a certificate for that would mean the only way to use a
        // local install is to turn the check off entirely.
        assert_eq!(
            normalize_base("http://formlogic.local/").unwrap(),
            "http://formlogic.local"
        );
        assert!(normalize_base("http://formlogic.local:8080").is_ok());
        assert!(normalize_base("http://api.formlogic.local").is_ok());
        // …but the exception is on the HOST, so it cannot be smuggled in a path.
        assert!(normalize_base("http://evil.example/formlogic.local").is_err());
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
