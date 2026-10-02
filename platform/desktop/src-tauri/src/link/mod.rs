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

pub mod app_logic;
pub mod condition;
pub mod creds;
pub mod data_node;
pub mod descriptor;
pub mod flow_runner;
pub mod sealed_flows;
pub mod flows;
pub mod heartbeat;
pub mod net;
pub mod oauth;
pub mod ops;
pub mod outbox;
pub mod policy;
pub mod relay;
pub mod result_actions;
pub mod routes;
pub mod script_profile;
#[cfg(test)]
pub(crate) mod testkit;

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

/// A stored file of the link that this build could not use, said in plain words.
///
/// The file is KEPT as it is. It is not read as "not linked": an `account.json` that fails to
/// parse used to be dropped by `.ok()` and the desktop went on as if nobody had linked it, so a
/// build rolled back past a newer file shape, a half-restored folder or a file another program held
/// for a moment unlinked the desktop without a word. Now the status carries this, nothing writes
/// over the file until it has been put aside as `<name>.corrupt` (see [`LinkStore::make_way`]), and
/// forgetting a link that cannot be removed says so instead of going on as if it had been.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LinkError {
    /// The file, as a path under the data folder: `link/account.json`.
    pub file: String,
    /// What is wrong with it and what has been done about it. Never holds what is in the file.
    pub message: String,
    /// The file could not be read at all, which is what another program holding it open looks like: it is read
    /// again later (see [`reread`]), and a file that was read and was not a link is not.
    #[serde(skip)]
    pub(crate) busy: bool,
}

/// The one file of the provider link, under the data folder.
const ACCOUNT_FILE: &str = "link/account.json";

/// Read a JSON file of the link's (the provider's `account.json`, or one of the relay's sibling files,
/// [`crate::relay::link_store`]). No file is `Ok(None)`. One that is there and cannot be used is an
/// error and is left exactly as it is: it may be perfectly good (another program has it open) or the
/// only copy of a link that a newer build wrote. A file that cannot be read is tried again a few times in the
/// first second (the pauses of [`crate::secret_file::Patience`]), and after that by [`reread`].
pub(crate) fn read_stored<T: serde::de::DeserializeOwned>(path: &std::path::Path, shown_as: &str) -> Result<Option<T>, LinkError> {
    read_stored_after(path, shown_as, &crate::secret_file::Patience::default().start, true)
}

/// [`read_stored`], waiting for a file that cannot be read as `pauses` say, and saying so in the log when `log` is set.
fn read_stored_after<T: serde::de::DeserializeOwned>(path: &std::path::Path, shown_as: &str, pauses: &[std::time::Duration], log: bool) -> Result<Option<T>, LinkError> {
    use crate::secret_file::{read_text_patiently, Text};
    let unusable = |why: String, busy: bool| {
        let error = LinkError { file: shown_as.to_string(), message: format!("{shown_as} {why}. It has not been changed."), busy };
        // The log says what the status says and no more, so what a test finds in one is all there is in the other.
        if log {
            log::warn!("link: {}", error.message);
        }
        error
    };
    match read_text_patiently(path, pauses) {
        Text::Missing => Ok(None),
        Text::Text(text) => serde_json::from_str(&text).map(Some).map_err(|e| unusable(format!("is not a link this version of OAIY understands ({})", describe(&e)), false)),
        Text::Undecodable(why) => Err(unusable(format!("is not text ({why})"), false)),
        Text::Unreadable(e) => Err(unusable(format!("could not be read ({e}): another program may have it open and it is read again"), true)),
    }
}

/// What reading a file again found.
pub(crate) enum Reread<T> {
    /// It is not time, or it is still held: nothing is different.
    Still,
    /// It was read: the file's content, or nothing if there is no file now.
    Read(Option<T>),
    /// It is there and cannot be used for another reason than being held.
    Unusable(LinkError),
}

/// Read a file of the link once more, with nothing kept between one look and the next: what it holds, nothing if
/// it is gone, why it cannot be used, or [`Reread::Still`] if another program has it. It never waits, and it never
/// moves, writes or removes the file.
pub(crate) fn read_again<T: serde::de::DeserializeOwned>(path: &std::path::Path, shown_as: &str) -> Reread<T> {
    match read_stored_after(path, shown_as, &[], false) {
        Ok(read) => Reread::Read(read),
        Err(e) if e.busy => Reread::Still,
        Err(e) => {
            log::warn!("link: {}", e.message);
            Reread::Unusable(e)
        }
    }
}

/// [`read_again`] for a file whose timer its owner keeps next to it, behind a lock the owner holds for the whole
/// call (the relay store's files): at the times `retry` says, and at once when `force` is set (a write is about to
/// decide what to do with it). A session used to stay without its link for as long as it ran after three tries in
/// the first moments.
pub(crate) fn reread<T: serde::de::DeserializeOwned>(path: &std::path::Path, shown_as: &str, retry: &mut Option<crate::secret_file::Retry>, force: bool) -> Reread<T> {
    if !force && !retry.as_ref().is_some_and(|r| r.due()) {
        return Reread::Still;
    }
    match read_again(path, shown_as) {
        Reread::Read(read) => {
            if retry.take().is_some() {
                log::info!("link: {shown_as} could be read again");
            }
            Reread::Read(read)
        }
        Reread::Still => {
            match retry {
                Some(r) => r.failed(),
                None => *retry = Some(crate::secret_file::Retry::began(&crate::secret_file::Patience::default())),
            }
            Reread::Still
        }
        Reread::Unusable(e) => {
            *retry = None;
            Reread::Unusable(e)
        }
    }
}

/// Remove what was put aside from `path` (`<name>.corrupt`, `<name>.corrupt.1`, ...): a link that is forgotten
/// takes with it the copies of itself that were kept when it could not be read, which can hold the key it had.
pub(crate) fn purge_asides(path: &std::path::Path) -> Result<(), String> {
    let (Some(dir), Some(name)) = (path.parent(), path.file_name().map(|n| n.to_string_lossy().into_owned())) else {
        return Ok(());
    };
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(()) };
    let stem = format!("{name}.corrupt");
    let mut failed = Vec::new();
    for entry in entries.flatten() {
        let found = entry.file_name().to_string_lossy().into_owned();
        let numbered = found.strip_prefix(&format!("{stem}.")).is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
        if (found == stem || numbered) && entry.path().is_file() {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                failed.push(format!("{found} ({e})"));
            }
        }
    }
    if failed.is_empty() { Ok(()) } else { Err(failed.join(", ")) }
}

/// Why a file failed to parse, without a word of what is in it: serde quotes the value in `invalid type:
/// string "flk_..."`, the name in `unknown variant` and `unknown field`, and any of them can be a key
/// pasted into the wrong place. Only a field the program itself names (`missing field`) is quoted.
fn describe(e: &serde_json::Error) -> String {
    use serde_json::error::Category;
    let at = format!("line {}, column {}", e.line(), e.column());
    match e.classify() {
        Category::Data => {
            let said = e.to_string();
            if said.starts_with("missing field") {
                said
            } else if said.starts_with("unknown field") {
                format!("an unknown field, {at}")
            } else {
                format!("a value has the wrong form, {at}")
            }
        }
        Category::Syntax | Category::Eof => format!("it is not valid JSON, {at}"),
        Category::Io => "it could not be read".to_string(),
    }
}

/// Put `path`, a file of the link that could not be used, aside as `<name>.corrupt` so that the
/// store can write a new one: what was in it is never thrown away. A file that is no longer there
/// needs nothing. An error says the file is still where it was, and must stop the write.
pub(crate) fn put_aside(path: &std::path::Path, shown_as: &str) -> Result<(), String> {
    match crate::secret_file::keep_aside(path) {
        Ok(aside) => {
            log::warn!("link: {shown_as} could not be used; it is kept as {}", aside.display());
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{shown_as} could not be used and could not be put aside ({e}); it has not been changed and nothing was written")),
    }
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
    pub sealed_flows_supported: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_sealed_flow_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_flow_error: Option<String>,
    /// This desktop's storage-node enrolment, once the provider has answered.
    /// The fingerprint here is what the owner compares against their browser
    /// before approving — the whole ceremony rests on the two matching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_node: Option<data_node::DataNodeStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_node_error: Option<String>,
    pub data_node_supported: bool,
    /// Plugin events kept for the account until FormLogic can take them.
    pub outbox: outbox::Status,
    /// A stored file of the link that could not be used, or a link that could not be forgotten. An
    /// unlinked desktop with this is NOT one nobody linked: its file is there and has been left alone.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link_error: Option<LinkError>,
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
    /// Why `account.json` was not read, or why the link could not be forgotten (see [`LinkError`]).
    error: Option<LinkError>,
    /// When `account.json` is read again, if the first reads found another program holding it (see [`reread`]).
    retry: Option<crate::secret_file::Retry>,
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
    last_sealed_flow_at: Option<chrono::DateTime<chrono::Utc>>,
    sealed_flow_error: Option<String>,
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
    let store = load_store(data_dir);
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

/// What `open_handle` returns, without the two workers it starts. A worker lives as
/// long as the process and asks its store what to do every tick, so a test that
/// holds a store with the fixture account of `formlogic.com` would send that
/// provider a heartbeat and an enrolment, with a made-up key, from then until the
/// last test ends. A test opens no connection beyond this machine. The trusted
/// origin is set as `open_handle` sets it.
pub(crate) fn open_handle_without_workers(data_dir: PathBuf) -> LinkHandle {
    let store = load_store(data_dir);
    set_linked_origin(store.account().as_ref().map(|a| a.base_url.as_str()));
    store
}

fn load_store(data_dir: PathBuf) -> LinkHandle {
    let path = data_dir.join("link").join("account.json");
    // A file that is there and cannot be used is reported and left alone, never read as no link.
    let (account, error) = match read_stored::<LinkedAccount>(&path, ACCOUNT_FILE) {
        Ok(account) => (account, None),
        Err(e) => (None, Some(e)),
    };
    // A file another program holds is read again later, by whoever asks next (`reread`).
    let retry = error.as_ref().filter(|e| e.busy).map(|_| crate::secret_file::Retry::began(&crate::secret_file::Patience::default()));
    let store = Arc::new(LinkStore {
        path,
        data_dir,
        inner: Mutex::new(Inner {
            account,
            error,
            retry,
            attempt: LinkPhase::Idle,
            in_flight: false,
            cancel: Arc::new(AtomicBool::new(false)),
            last_heartbeat_at: None,
            heartbeat_error: None,
            last_relay_at: None,
            relay_error: None,
            last_flow_run_at: None,
            flow_run_error: None,
            last_sealed_flow_at: None,
            sealed_flow_error: None,
            data_node: None,
            data_node_error: None,
        }),
    });
    store
}

#[cfg(test)]
impl LinkStore {
    /// Leave the store with no account, and nothing else changed: the loop of a lane
    /// a test started has nothing to do from then on. Not `unlink`, which also
    /// clears the trusted origin and the provider's prelude that other tests read.
    pub(crate) fn drop_account_for_tests(&self) {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).account = None;
    }
}

/// A store holding `account`, for the tests of a lane's loop: it needs a link to
/// read and nothing else, so this starts none of the workers `open_handle` starts
/// and does not touch the trusted origin every other test reads.
#[cfg(test)]
pub(crate) fn store_for_tests(data_dir: PathBuf, account: Option<LinkedAccount>) -> LinkHandle {
    Arc::new(LinkStore {
        path: data_dir.join("link").join("account.json"),
        data_dir,
        inner: Mutex::new(Inner {
            account,
            error: None,
            retry: None,
            attempt: LinkPhase::Idle,
            in_flight: false,
            cancel: Arc::new(AtomicBool::new(false)),
            last_heartbeat_at: None,
            heartbeat_error: None,
            last_relay_at: None,
            relay_error: None,
            last_flow_run_at: None,
            flow_run_error: None,
            last_sealed_flow_at: None,
            sealed_flow_error: None,
            data_node: None,
            data_node_error: None,
        }),
    })
}

impl LinkStore {
    pub fn status(&self) -> LinkStatus {
        self.reread(false);
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
                sealed_flows_supported: false,
                last_sealed_flow_at: None,
                sealed_flow_error: None,
                data_node: None,
                data_node_error: None,
                data_node_supported: false,
                outbox: outbox::current_status(),
                link_error: inner.error.clone(),
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
                    sealed_flows_supported: d.is_some_and(|d| d.desktop_flows.is_some()),
                    last_sealed_flow_at: inner.last_sealed_flow_at,
                    sealed_flow_error: inner.sealed_flow_error.clone(),
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
                    outbox: outbox::current_status(),
                    link_error: inner.error.clone(),
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

    pub fn note_sealed_flow(&self, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if error.is_none() { inner.last_sealed_flow_at = Some(chrono::Utc::now()); }
        inner.sealed_flow_error = error;
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

    /// The provider does not offer data nodes: nothing to show, and no error.
    pub fn note_data_node_off(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.data_node = None;
        inner.data_node_error = None;
    }

    /// Record the outcome of a heartbeat.
    pub fn note_heartbeat(&self, error: Option<String>) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if error.is_none() {
            inner.last_heartbeat_at = Some(chrono::Utc::now());
        }
        inner.heartbeat_error = error;
    }

    /// Read `account.json` again if it could not be read when the store opened, because another program
    /// had it: when it is time, or at once when `force` is set. A scanner's hold in the first moments used to leave
    /// the desktop without its link for as long as it ran. A file that reads as a link is the link from then on;
    /// one that cannot be used is reported; one that is still held waits for the next time. It only ever reads:
    /// a link that is perfectly good is not moved aside, written over or removed by it.
    fn reread(&self, force: bool) {
        // Decided under the lock, and the timer left where it is: the lanes that ask for the account call this at the
        // top of their loops with no pause before it, so a second caller meets the first one's read in progress, and a
        // timer taken out for the length of a read and put back after it was lost to whoever took `None` in between.
        {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if inner.account.is_some() || inner.error.is_none() {
                return;
            }
            if !force && !inner.retry.as_ref().is_some_and(|r| r.due()) {
                return;
            }
        }
        // Without the lock: the file is read, and the lanes that ask for the account meanwhile are not kept waiting.
        let read = read_again::<LinkedAccount>(&self.path, ACCOUNT_FILE);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.account.is_some() || inner.error.is_none() {
            return;
        }
        match read {
            // Still held: the next time is later, and the timer is changed in place.
            Reread::Still => match inner.retry.as_mut() {
                Some(r) => r.failed(),
                None => inner.retry = Some(crate::secret_file::Retry::began(&crate::secret_file::Patience::default())),
            },
            Reread::Read(account) => {
                if inner.retry.take().is_some() {
                    log::info!("link: {ACCOUNT_FILE} could be read again");
                }
                inner.error = None;
                if let Some(account) = account {
                    set_linked_origin(Some(account.base_url.as_str()));
                    inner.account = Some(account);
                }
            }
            Reread::Unusable(e) => {
                inner.error = Some(e);
                inner.retry = None;
            }
        }
    }

    /// The stored credential, for whoever needs to call the provider.
    pub fn account(&self) -> Option<LinkedAccount> {
        self.reread(false);
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .account
            .clone()
    }

    fn persist(&self, account: &LinkedAccount) -> Result<(), String> {
        let raw = serde_json::to_string_pretty(account)
            .map_err(|e| format!("could not encode the link: {e}"))?;
        self.make_way()?;
        // The account's key is in this file: owner-only from its first byte, and
        // replaced whole so a reader on another thread never meets half of it.
        crate::secret_file::write(&self.path, raw)
            .map_err(|e| format!("could not save the link: {e}"))
    }

    /// A file that could not be used is never written over: it is put aside as
    /// `account.json.corrupt` first, and a failure to do that stops the write. Nothing to do
    /// when the file was fine.
    fn make_way(&self) -> Result<(), String> {
        // It may have become readable since (another program let go of it): then it is a link, and no file is
        // put aside for a write that replaces it.
        self.reread(true);
        let unusable = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.account.is_none() && inner.error.is_some()
        };
        if !unusable {
            return Ok(());
        }
        put_aside(&self.path, ACCOUNT_FILE)?;
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).error = None;
        Ok(())
    }

    /// Take the stored link away: `Err` says why it is still there, and `Ok(true)` that the file could not be
    /// used and was put aside rather than deleted (what is in it may be a link a newer build can read). A link
    /// that is not there to remove is forgotten all the same.
    fn forget_stored(&self) -> Result<bool, String> {
        self.reread(true);
        let unusable = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.account.is_none() && inner.error.is_some()
        };
        if unusable {
            return self.make_way().map(|()| true);
        }
        match std::fs::remove_file(&self.path) {
            Ok(()) => Ok(false),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => {
                log::warn!("link: {ACCOUNT_FILE} could not be removed: {e}");
                Err(format!("the link could not be forgotten: {ACCOUNT_FILE} could not be removed ({e}). It is still linked here, and would be again at the next start. Close whatever has it open and try again."))
            }
        }
    }

    /// Forget the link. Local only — see the note on the route.
    ///
    /// The file is removed FIRST, and a link that cannot be removed stays linked and says so
    /// ([`LinkStatus::link_error`]): going on as if it had been forgotten would leave the key on
    /// disk to link the desktop again at the next start, and the one who asked would never know.
    pub fn unlink(&self) -> LinkStatus {
        let put_aside = match self.forget_stored() {
            Ok(put_aside) => put_aside,
            Err(message) => {
                self.inner.lock().unwrap_or_else(|e| e.into_inner()).error = Some(LinkError { file: ACCOUNT_FILE.to_string(), message, busy: false });
                return self.status();
            }
        };
        // The copies kept when the link could not be read can hold the key it had: forgetting the link takes them
        // too. Not when this very call put an unreadable file aside, which is the one thing kept.
        let left_behind = if put_aside { Ok(()) } else { purge_asides(&self.path) };
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.account = None;
            inner.error = left_behind.err().map(|left| LinkError {
                file: ACCOUNT_FILE.to_string(),
                message: format!("the link was forgotten, but {left} could not be removed: a copy that was kept of it can hold the key it had"),
                busy: false,
            });
            inner.retry = None;
            inner.attempt = LinkPhase::Idle;
            inner.last_heartbeat_at = None;
            inner.heartbeat_error = None;
            // Both lanes, or a failure from the provider just disconnected
            // would be shown against the next one that is linked.
            inner.last_relay_at = None;
            inner.relay_error = None;
            inner.last_flow_run_at = None;
            inner.flow_run_error = None;
            inner.last_sealed_flow_at = None;
            inner.sealed_flow_error = None;
            // The IDENTITY stays on disk: the same machine relinking is the
            // same node, and re-minting would ask the owner to approve a device
            // they already approved. Only the provider's answer is forgotten.
            inner.data_node = None;
            inner.data_node_error = None;
        }
        // Withdrawn in the same action that forgot the link: a provider we are
        // no longer linked to must not keep reaching this machine.
        set_linked_origin(None);
        // And the provider's PRELUDE goes with it. The next link may be another
        // account on another deployment, whose standard library is its own —
        // keeping this one would run one account's helpers inside another's
        // scripts, which is worse than having none.
        script_profile::ProfileCache::global().invalidate(&self.data_dir);
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
                    inner.error = None;
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
        let h = open_handle_without_workers(p.clone());
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
    fn the_stored_link_is_owner_only_and_a_replacement_stays_so() {
        use crate::secret_file::testing::{assert_private, assert_private_dir};
        let (dir, s) = store("private");
        s.persist(&account()).unwrap();
        let file = dir.join("link").join("account.json");
        assert_private(&file);
        assert_private_dir(file.parent().unwrap());

        let mut renewed = account();
        renewed.credential = "flk_renewed".into();
        s.persist(&renewed).unwrap();
        assert_private(&file);
        assert_eq!(open_handle_without_workers(dir.clone()).account().unwrap().credential, "flk_renewed");
        let staged: Vec<_> = std::fs::read_dir(file.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(staged.is_empty(), "no copy of the key is left behind: {staged:?}");
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
        assert_eq!(open_handle_without_workers(dir.clone()).instance_id(), first);
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

        let reopened = open_handle_without_workers(dir.clone());
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

        let reopened = open_handle_without_workers(dir.clone());
        let a = reopened.account().expect("the link must survive");
        assert_eq!(a.credential, "flk_supersecret");
        assert_eq!(a.connector_id, "formlogic");

        assert!(!reopened.unlink().linked);
        assert!(open_handle_without_workers(dir.clone()).account().is_none(), "unlink must be durable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `LinkedAccount` as a build from before the relay's files holds it: it refuses a field it does
    /// not know, so a field added to `account.json` would unlink that build (and, until `LinkError`,
    /// without a word).
    #[derive(Debug, serde::Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    #[allow(dead_code)]
    struct LinkedAccountBeforeTheRelay {
        connector_id: String,
        base_url: String,
        credential: String,
        #[serde(default)]
        account_id: Option<String>,
        #[serde(default)]
        account_name: Option<String>,
        #[serde(default)]
        granted_scopes: Option<String>,
        linked_at: chrono::DateTime<chrono::Utc>,
        #[serde(default)]
        instance_id: Option<String>,
    }

    #[test]
    fn a_build_from_before_the_relay_still_reads_the_link_and_finds_nothing_new_in_its_folder() {
        // The relay's files are beside the provider's (design 4.16.2), so `account.json` is written exactly
        // as it always was and a rolled-back build reads it, with the relay linked or not.
        let (dir, s) = store("downgrade");
        s.persist(&account()).unwrap();
        let file = dir.join("link").join("account.json");
        let before = std::fs::read(&file).unwrap();

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
        relay.set_routes(crate::relay::link_store::Routes { commands: crate::relay::link_store::Route::Relay, ..Default::default() }).unwrap();
        relay.set_pins(Default::default()).unwrap();

        assert_eq!(std::fs::read(&file).unwrap(), before, "the relay's files do not touch account.json");
        let old: LinkedAccountBeforeTheRelay = serde_json::from_slice(&before).expect("the old struct reads what this build writes");
        assert_eq!((old.credential.as_str(), old.connector_id.as_str()), ("flk_supersecret", "formlogic"));
        // And a link this build rewrites later (a renewed key, a new instance id) is still that shape.
        let mut renewed = account();
        renewed.credential = "flk_renewed".into();
        s.persist(&renewed).unwrap();
        serde_json::from_slice::<LinkedAccountBeforeTheRelay>(&std::fs::read(&file).unwrap()).expect("and so is the next one");

        let names = |folder: &std::path::Path| -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(folder).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
            names.sort();
            names
        };
        assert_eq!(names(&dir.join("link")), ["account.json"], "nothing new in the provider's folder");
        assert_eq!(names(&dir.join("relay")), ["providers.json", "relay.json", "routes.json"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- a link that cannot be used is kept and reported, never read as no link ---------------

    /// An empty data folder with its `link` folder made.
    fn data_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oaiy-link-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("link")).unwrap();
        dir
    }

    /// A store over a data folder whose `link/account.json` holds `body`. It is made with
    /// `load_store`, not `open_handle_without_workers`: that one also sets the trusted origin every
    /// other test reads.
    fn store_over(tag: &str, body: &[u8]) -> (std::path::PathBuf, LinkHandle) {
        let dir = data_dir(tag);
        std::fs::write(dir.join("link").join("account.json"), body).unwrap();
        (dir.clone(), load_store(dir))
    }

    #[test]
    fn an_account_file_that_cannot_be_understood_is_kept_reported_and_not_read_as_no_link() {
        // `.ok()` used to turn each of these into "nobody has linked this desktop": a build rolled back past
        // a newer shape of the file, a restore that cut it, a file that is not text. The link was gone from
        // the screen and the key sat on the disk, with nothing said.
        let newer = br#"{"connectorId":"formlogic","baseUrl":"https://formlogic.com","credential":"flk_a","linkedAt":"2026-09-01T00:00:00Z","futureField":1}"#;
        for (what, body, says) in [
            ("a newer shape", &newer[..], "an unknown field, line 1"),
            ("a cut file", &br#"{"connectorId":"formlogic","baseUrl":"#[..], "not valid JSON"),
            ("not text", &[0xC3, 0x28, 0xA0][..], "is not text"),
        ] {
            let (dir, store) = store_over("kept", body);
            let status = store.status();
            assert!(!status.linked && store.account().is_none(), "{what}");
            let error = status.link_error.unwrap_or_else(|| panic!("{what}: nothing was said"));
            assert_eq!(error.file, "link/account.json", "{what}");
            assert!(error.message.contains(says), "{what}: {}", error.message);
            assert!(error.message.contains("It has not been changed"), "{what}: {}", error.message);
            assert_eq!(std::fs::read(dir.join("link").join("account.json")).unwrap(), body, "{what}: the file is as it was");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn what_serde_quotes_back_never_reaches_the_status_or_the_log() {
        // serde's own message quotes the value (`invalid type: string "flk_..."`), the name of an unknown
        // variant, the name of an unknown field and a number: any of them can be a key pasted into the wrong
        // place. What is said of a file is `describe`'s and nothing of serde's, and the log line is the status's.
        use crate::relay::link_store::{ProviderPins, Routes};
        const SECRET: &str = "flk_TOPSECRET";
        let file = |tail: &str| format!(r#"{{"connectorId":"formlogic","baseUrl":"https://formlogic.com","credential":"{SECRET}","linkedAt":"2026-09-01T00:00:00Z"{tail}}}"#);
        fn from<T: serde::de::DeserializeOwned>(raw: &str) -> serde_json::Error {
            serde_json::from_str::<T>(raw).map(|_| ()).unwrap_err()
        }
        // What goes wrong, serde's error for it, what `describe` must say, and what serde itself quotes of it.
        let cases: Vec<(&str, serde_json::Error, &str, &str)> = vec![
            ("an unknown field", from::<LinkedAccount>(&file(&format!(r#","{SECRET}":1"#))), "an unknown field", "TOPSECRET"),
            ("a string where a number belongs", from::<ProviderPins>(&format!(r#"{{"pins":[{{"providerId":"p","ed25519":"e","x25519":"x","thumbprint":"t","serial":"{SECRET}","pinnedAt":"2026-09-01T00:00:00Z"}}]}}"#)), "a value has the wrong form", "TOPSECRET"),
            ("an unknown variant", from::<Routes>(&format!(r#"{{"commands":"{SECRET}"}}"#)), "a value has the wrong form", "TOPSECRET"),
            ("a number where a string belongs", from::<LinkedAccount>(&file(r#","accountId":13371337"#)), "a value has the wrong form", "13371337"),
        ];
        for (what, error, says, quoted) in cases {
            let said = describe(&error);
            assert!(said.contains(says), "{what}: {said}");
            assert!(!said.contains("TOPSECRET") && !said.contains("13371337"), "{what}: {said}");
            assert!(error.to_string().contains(quoted), "{what}: this case means nothing unless serde quotes it, and it said {error}");
        }
        // And through the whole path: the status, what the screen is told, and the line that goes to the log.
        for body in [file(&format!(r#","{SECRET}":1"#)), file(r#","accountId":13371337"#)] {
            let (dir, store) = store_over("quote-path", body.as_bytes());
            let status = store.status();
            let shown = serde_json::to_string(&status).unwrap();
            assert!(!shown.contains("TOPSECRET") && !shown.contains("13371337"), "{shown}");
            let from_file: Result<Option<LinkedAccount>, LinkError> = read_stored(&dir.join("link").join("account.json"), "link/account.json");
            let error = from_file.unwrap_err();
            assert_eq!(Some(&error), status.link_error.as_ref(), "the status and the log line are one message");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn a_file_another_program_holds_is_reported_too_and_not_taken_for_no_link() {
        // A folder where the file belongs cannot be read, as a file another program has open cannot.
        let dir = data_dir("held");
        std::fs::create_dir(dir.join("link").join("account.json")).unwrap();
        let store = load_store(dir.clone());
        let error = store.status().link_error.expect("it could not be read, and that is said");
        assert!(error.message.contains("could not be read"), "{}", error.message);
        assert!(dir.join("link").join("account.json").is_dir(), "and left where it was");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn what_is_reported_never_quotes_what_is_in_the_file() {
        // serde's own message for a value of the wrong type quotes the value, and a status is read by the
        // dashboard and the Agent: a key in the wrong field must not reach them.
        let body = br#"{"connectorId":"formlogic","baseUrl":"https://formlogic.com","credential":"flk_TOPSECRET","linkedAt":"flk_TOPSECRET"}"#;
        let (dir, store) = store_over("quote", body);
        let status = store.status();
        let error = status.link_error.clone().expect("the date is not one");
        assert!(!error.message.contains("TOPSECRET"), "{}", error.message);
        assert!(!serde_json::to_string(&status).unwrap().contains("TOPSECRET"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_link_puts_a_file_that_could_not_be_used_aside_and_never_writes_over_it() {
        let newer = br#"{"connectorId":"formlogic","futureField":1}"#;
        let (dir, store) = store_over("aside", newer);
        assert!(store.status().link_error.is_some());

        store.persist(&account()).unwrap();
        let link = dir.join("link");
        assert_eq!(std::fs::read(link.join("account.json.corrupt")).unwrap(), newer, "what was there is kept");
        assert_eq!(load_store(dir.clone()).account().unwrap().credential, "flk_supersecret", "and the new link is the file now");
        assert!(store.status().link_error.is_none(), "it is dealt with");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_that_cannot_be_put_aside_stops_the_write_and_one_that_is_gone_needs_nothing() {
        // The write is refused, so what is in the file is never lost for want of a place to keep it.
        let refused = put_aside(std::path::Path::new("/"), "link/account.json").unwrap_err();
        assert!(refused.contains("nothing was written"), "{refused}");
        // The file went away since it was found unusable: nothing to keep, and the write goes on.
        let dir = data_dir("aside-gone");
        put_aside(&dir.join("link").join("account.json"), "link/account.json").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Say when the account file is read again: at once.
    fn read_again_now(store: &LinkStore) {
        store.inner.lock().unwrap().retry = Some(crate::secret_file::Retry::began(&crate::secret_file::Patience {
            start: vec![],
            later: vec![std::time::Duration::ZERO],
            window: std::time::Duration::ZERO,
        }));
    }

    #[test]
    fn a_file_another_program_held_at_start_is_read_again_and_the_link_comes_back() {
        // Three tries in the first moments and never again left the session without its link for as long as it ran.
        // A folder where the file belongs cannot be read, as a file a scanner holds cannot.
        let dir = data_dir("held-then-let-go");
        let file = dir.join("link").join("account.json");
        std::fs::create_dir(&file).unwrap();
        let store = load_store(dir.clone());
        assert!(store.account().is_none() && store.status().link_error.is_some(), "held: no link, and why");
        assert!(file.is_dir(), "and nothing is done to it");

        std::fs::remove_dir(&file).unwrap();
        std::fs::write(&file, serde_json::to_string(&account()).unwrap()).unwrap();
        assert!(store.account().is_none(), "it is not time to read it again");
        read_again_now(&store);
        let status = store.status();
        assert!(status.linked && status.link_error.is_none(), "{status:?}");
        assert_eq!(store.account().unwrap().credential, "flk_supersecret");
        assert!(!dir.join("link").join("account.json.corrupt").exists(), "a good file was read, not moved aside");
        assert!(store.inner.lock().unwrap().retry.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_held_file_is_read_again_however_many_lanes_ask_at_once() {
        // The lanes that read the account at the top of their loops (the command lane, the flow runner, the sealed flows, the
        // AI tunnel) have no pause before it, and the screen asks for the status as well. Eight threads in a tight loop meet
        // each other's read inside microseconds. A timer taken out of the lock for the length of a read and put back after it
        // was lost to whoever took `None` in between, and the link then never came back for as long as the session ran.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let dir = data_dir("held-race");
        let file = dir.join("link").join("account.json");
        std::fs::write(&file, serde_json::to_string(&account()).unwrap()).unwrap();
        // A good file that cannot be read for now, and that is let go of in one step (so that there is no moment when it is missing).
        let Some(held) = crate::secret_file::testing::make_unreadable(&file) else { return };
        let store = load_store(dir.clone());
        read_again_now(&store);
        let (stop, linked) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicUsize::new(0)));
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let (store, stop, linked) = (store.clone(), stop.clone(), linked.clone());
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let back = if i == 7 { store.status().linked } else { store.account().is_some() };
                        if back {
                            linked.fetch_add(1, Ordering::SeqCst);
                            return;
                        }
                    }
                })
            })
            .collect();

        // They meet each other against the held file for a while; then it is let go of.
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(linked.load(Ordering::SeqCst), 0, "held: nobody has a link yet");
        drop(held);

        // Every one of them gets it, soon.
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while linked.load(Ordering::SeqCst) < 8 && Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        stop.store(true, Ordering::Relaxed);
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(linked.load(Ordering::SeqCst), 8, "the link came back to every lane that asked: the timer was not lost");
        assert!(store.inner.lock().unwrap().retry.is_none(), "and the timer is done with");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_new_link_does_not_put_aside_a_file_that_can_be_read_by_then_and_a_forgotten_one_goes() {
        let dir = data_dir("held-then-linked");
        let file = dir.join("link").join("account.json");
        std::fs::create_dir(&file).unwrap();
        let store = load_store(dir.clone());
        assert!(store.status().link_error.is_some());

        // Let go of, and not yet read again: a link made now replaces the file; it is not a bad file to keep.
        std::fs::remove_dir(&file).unwrap();
        std::fs::write(&file, serde_json::to_string(&account()).unwrap()).unwrap();
        let mut renewed = account();
        renewed.credential = "flk_renewed".into();
        store.persist(&renewed).unwrap();
        assert!(!dir.join("link").join("account.json.corrupt").exists(), "nothing readable was put aside");
        assert_eq!(load_store(dir.clone()).account().unwrap().credential, "flk_renewed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forgetting_a_link_takes_the_copies_kept_of_earlier_files_and_no_other_file() {
        // The dialog says the key is forgotten, and a copy kept of a file that could not be read can hold the one it had.
        let dir = data_dir("forget-copies");
        let link = dir.join("link");
        let store = load_store(dir.clone());
        store.persist(&account()).unwrap();
        store.inner.lock().unwrap().account = Some(account());
        for kept in ["account.json.corrupt", "account.json.corrupt.1", "account.json.corrupt.30", "account.json.corrupt.x", "account.json.bak-baseurl", "app-logic-storage.json"] {
            std::fs::write(link.join(kept), r#"{"credential":"flk_THE_PREVIOUS_KEY"}"#).unwrap();
        }
        let after = store.unlink();
        assert!(!after.linked && after.link_error.is_none(), "{after:?}");
        let mut left: Vec<String> = std::fs::read_dir(&link).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left, ["account.json.bak-baseurl", "account.json.corrupt.x", "app-logic-storage.json"]);

        // A folder named like a copy is not a file of ours and is left alone; the copies that are files go.
        let stuck = link.join("account.json.corrupt.2");
        std::fs::create_dir_all(stuck.join("held")).unwrap();
        std::fs::write(link.join("account.json.corrupt.3"), "x").unwrap();
        store.persist(&account()).unwrap();
        store.inner.lock().unwrap().account = Some(account());
        let after = store.unlink();
        assert!(!after.linked && after.link_error.is_none(), "{after:?}");
        assert!(stuck.is_dir() && !link.join("account.json.corrupt.3").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn forgetting_a_file_that_could_not_be_used_keeps_it_aside() {
        let newer = br#"{"connectorId":"formlogic","futureField":1}"#;
        let (dir, store) = store_over("forget-bad", newer);
        let after = store.unlink();
        assert!(!after.linked && after.link_error.is_none());
        let link = dir.join("link");
        assert!(!link.join("account.json").exists(), "it is not the link now");
        assert_eq!(std::fs::read(link.join("account.json.corrupt")).unwrap(), newer, "and it is not thrown away");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_link_that_cannot_be_removed_stays_linked_and_says_so() {
        // `let _ = remove_file(..)` forgot the link in memory whatever happened to the file: the key stayed
        // on the disk, the screen said unlinked, and the next start linked the desktop again.
        let dir = data_dir("stuck");
        let store = load_store(dir.clone());
        let file = dir.join("link").join("account.json");
        // A file that is a folder with something in it cannot be removed, on any system.
        std::fs::create_dir_all(file.join("held")).unwrap();
        store.inner.lock().unwrap().account = Some(account());
        store.inner.lock().unwrap().error = None;

        let after = store.unlink();
        assert!(after.linked, "it is still linked, and the screen says so");
        let error = after.link_error.expect("and why");
        assert!(error.message.contains("could not be forgotten"), "{}", error.message);
        assert!(store.account().is_some(), "the lanes go on with a link that is still stored");

        // Once the file can be removed the next try forgets it, and the error goes with the link.
        std::fs::remove_dir_all(&file).unwrap();
        let after = store.unlink();
        assert!(!after.linked && after.link_error.is_none());
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
