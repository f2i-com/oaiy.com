//! Pairing: how an untrusted-origin consumer earns a bearer token.
//!
//! OAIY's exec routes are privileged, and a loopback port is not a trust
//! boundary. A browser served by oaiy.com or OAIY's own webview is trusted by
//! Origin; anyone else — FormLogic Web, a co-located script, a coding agent —
//! presents a bearer token. This module is how they get one, WITHOUT OAIY ever
//! trusting the requester's word for who they are:
//!
//! ```text
//!   consumer ── POST /api/bridge/pairing {product,label} ──►  a PENDING request
//!                                                              + a short code
//!        │                                                          │
//!        │  the user sees "FormLogic wants to connect · code 7QK2M9"│ in OAIY
//!        │  Desktop and clicks Approve (the physical trust act) ────┘
//!        ▼
//!   consumer ── GET /api/bridge/pairing/{id} (poll) ──►  {approved, token}
//! ```
//!
//! # The user's click is the whole security model
//!
//! The request route is unauthenticated on purpose — its only power is to raise
//! a prompt. Nothing is granted until a human approves in the OAIY UI, which is
//! the one thing a malicious local page cannot forge. The **code** is shown in
//! both places so the user confirms the prompt matches the app they meant to
//! pair; approving a hidden request whose code you never saw is the attack, and
//! the code is what defeats it.
//!
//! Requests are bounded and expire, so a page that spams pairing can annoy but
//! not exhaust. Approved **tokens are persisted** (a paired consumer stays paired
//! across restarts); pending requests are not (they are cheap to re-raise).
//!
//! # What a token reaches
//!
//! A request says what it asks for ([`PairingAccess`]): an app's (running flows,
//! plugins' commands, the AI gateway: FormLogic, a script), or the OAIY Agent's
//! in a browser tab (the phone, the calendar and contacts too, and the Agent's
//! tools for changing OAIY). The prompt says which, and the token reaches only
//! that (`http::paired_reaches`), only from the page it was granted to (a
//! program's, which sends no Origin, from no page), and it is forgotten after
//! [`IDLE_EXPIRY_MS`] without use. Pairing itself (approving, denying, revoking,
//! the lists) is never a paired app's.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// A pending request expires if not approved within this window — a prompt the
/// user ignored should not sit around to be approved by accident later.
const REQUEST_TTL_MS: u64 = 5 * 60 * 1000;
/// Cap on simultaneous pending requests, so a page spamming pairing cannot grow
/// the list without bound. Past it, room is made of a request the person turned
/// down, then of the oldest request of the origin that holds the most of them;
/// never of an origin's only request, and never of an approved one, which waits
/// for its requester to collect the token ([`PairingManager::request_for`]).
const MAX_PENDING: usize = 12;
/// Cap on the pending requests of one origin (a program's: those with none), so
/// one page cannot push every other app's prompt out. Its own oldest goes first.
const MAX_PENDING_PER_ORIGIN: usize = 3;
/// The longest `product` and `label` a request may give: they are shown in the
/// prompt, and nothing real is longer.
pub const MAX_PRODUCT_CHARS: usize = 64;
pub const MAX_LABEL_CHARS: usize = 200;
/// A paired token that has not been used for this long is forgotten: the app
/// pairs again. FormLogic keeps its token for the signed-in account, across
/// tabs and restarts of the browser (only a page with no sign-in keeps it for
/// its tab alone), so this, and not the end of a browser session, is what ends
/// a grant nobody revoked.
pub const IDLE_EXPIRY_MS: u64 = 90 * 24 * 60 * 60 * 1000;
/// How stale the stored last use may get before a use writes the file again.
const USE_PERSIST_MS: u64 = 24 * 60 * 60 * 1000;

/// What a paired app may reach: what its request asked for, so what the person
/// approved in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairingAccess {
    /// Run (and keep) flows, call plugins' commands, start and stop plugins and
    /// services, use the AI gateway: what FormLogic and scripts use. A grant made before
    /// requests said what they asked for is this one.
    #[default]
    Apps,
    /// The OAIY Agent in a browser tab: an app's, and the phone, the calendar
    /// and the contacts, the engines' model menu, and the Agent's own tools for
    /// changing OAIY (as far as its switch in Settings lets them).
    Agent,
}

impl PairingAccess {
    pub fn by_name(name: &str) -> Option<PairingAccess> {
        match name {
            "apps" => Some(PairingAccess::Apps),
            "agent" => Some(PairingAccess::Agent),
            _ => None,
        }
    }

    /// Whether a token of this kind reaches a route of the access table's
    /// `scope` (`auth/routes.rs`). No `auth.*` scope is in either: pairing,
    /// approving and revoking are the person's, in OAIY's own window.
    pub fn reaches(self, scope: &str) -> bool {
        // (FormLogic manages the flows and triggers it runs, docs/ecosystem/OAIY_PLATFORM.md: a flow can run a program,
        // which is why the prompt says so)
        const APPS: &[&str] = &[
            "flows.read",
            "flows.write",
            "runs.read",
            "runs.write",
            "connectors.use",
            "events.read",
            "plugins.read",
            "plugins.control",
            "services.read",
            "services.control",
            "ai.read",
            "ai.use",
            "speech.use",
        ];
        match self {
            PairingAccess::Apps => APPS.contains(&scope),
            // The design's preset for the Agent's page, and what the page in a tab asks of the engines (which
            // models there are, and the one its menu chooses).
            PairingAccess::Agent => {
                APPS.contains(&scope)
                    || matches!(scope, "models.read" | "models.write")
                    || crate::auth::presets::Preset::Agent.scopes().contains(scope)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairingStatus {
    Pending,
    Approved,
    Denied,
    /// Aged out before anyone approved it.
    Expired,
}

/// A pending pairing request, shown to the user for approval.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairingRequest {
    pub pairing_id: String,
    /// Consuming product, e.g. `formlogic`. Displayed; never trusted as identity.
    pub product: String,
    /// Human label for the prompt, e.g. "Receptionist · Acme Dental".
    pub label: Option<String>,
    /// The browser Origin that asked, when present — the most trustworthy signal
    /// of who is really asking (a real browser cannot forge it).
    pub origin: Option<String>,
    /// Short code shown here AND to the consumer, so the user confirms they match.
    pub code: String,
    /// What the request asks for: the prompt says it, and the token reaches that.
    pub access: PairingAccess,
    pub status: PairingStatus,
    pub created_at_ms: u64,
    /// When the person approved it: its requester has a full [`REQUEST_TTL_MS`] from then to
    /// collect the token, however late in the request's own time that was.
    #[serde(skip)]
    pub approved_at_ms: Option<u64>,
    /// The minted token — only ever present in the approve response / poll to the
    /// requester, never in the pending-list shown to the UI.
    #[serde(skip)]
    pub token: Option<String>,
}

/// A granted token, persisted so a paired consumer survives a restart.
///
/// Serializes fully (INCLUDING the secret) — that is exactly what persistence
/// needs, and the file lives in the user's own data dir. It must NEVER be handed
/// to an API response directly; use [`PairedToken::public`] for anything a
/// caller sees, which drops the secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedToken {
    /// A short, stable public id — never the secret. Safe to show in a "paired
    /// apps" list and to revoke by.
    pub id: String,
    pub product: String,
    pub label: Option<String>,
    /// The browser origin this token was granted TO, captured from the request
    /// the user actually approved. Kept because `product` and `label` are both
    /// supplied by the consumer and neither is unique: two different sites can
    /// each call themselves "formlogic", and without the origin a user
    /// reviewing granted access cannot tell which one they are revoking.
    /// `None` for a native caller, which sends no Origin.
    #[serde(default)]
    pub origin: Option<String>,
    /// What the person approved (`apps` for a grant from before there was a choice).
    #[serde(default)]
    pub access: PairingAccess,
    pub created_at_ms: u64,
    /// When the token was last presented (as of the last time the file was written).
    #[serde(default)]
    pub last_used_ms: Option<u64>,
    /// The secret bearer value.
    pub secret: String,
}

/// The secret-free view of a granted token, for API responses.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedTokenPublic {
    pub id: String,
    pub product: String,
    pub label: Option<String>,
    pub origin: Option<String>,
    pub access: PairingAccess,
    pub created_at_ms: u64,
    pub last_used_ms: Option<u64>,
}

impl PairedToken {
    pub fn public(&self) -> PairedTokenPublic {
        PairedTokenPublic {
            id: self.id.clone(),
            product: self.product.clone(),
            label: self.label.clone(),
            origin: self.origin.clone(),
            access: self.access,
            created_at_ms: self.created_at_ms,
            last_used_ms: self.last_used_ms,
        }
    }
}

/// Who a presented bearer is, as far as pairing knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grant {
    /// No paired token is this one.
    Unknown,
    /// A paired token, presented from a page it was not granted to (or, granted
    /// to a program, from a page at all): nobody's here.
    WrongOrigin,
    /// A paired token, from where it was granted to: it reaches what `access` does.
    Paired(PairingAccess),
}

pub struct PairingManager {
    pending: Vec<PairingRequest>,
    tokens: Vec<PairedToken>,
    path: Option<PathBuf>,
}

impl PairingManager {
    pub fn new() -> Self {
        Self { pending: Vec::new(), tokens: Vec::new(), path: None }
    }

    /// Load persisted tokens from `path`, creating a durable manager.
    pub fn open(path: PathBuf) -> Self {
        let tokens = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<PersistShape>(&s).ok())
            .map(|p| p.tokens)
            .unwrap_or_default();
        Self { pending: Vec::new(), tokens, path: Some(path) }
    }

    /// Raise a request for an app's access (the tests' shorthand for [`Self::request_for`]).
    #[cfg(test)]
    pub fn request(
        &mut self,
        product: &str,
        label: Option<String>,
        origin: Option<String>,
    ) -> PairingRequest {
        self.request_for(product, label, origin, PairingAccess::Apps).expect("room for a request")
    }

    /// Raise a pairing request. Returns the id + the code to show the consumer.
    ///
    /// Unauthenticated by design — see the module docs. Expired requests are
    /// swept first, and the list is capped: an origin's own oldest pending
    /// request makes room for its next. In a full list room is then made of a
    /// request the person turned down (nobody waits on it), then of the oldest
    /// request of the origin that holds the most ([`Self::most_crowded`]).
    ///
    /// Another origin's only request is never put out: the oldest request of
    /// all used to be, so a page with four origins (three requests each fill
    /// the list) removed every other app's prompt with its next request.
    /// Refused while there is no such room: every request is approved and
    /// waiting to be collected, or the only one of its origin (a few minutes
    /// at most, until they are answered or run out).
    pub fn request_for(
        &mut self,
        product: &str,
        label: Option<String>,
        origin: Option<String>,
        access: PairingAccess,
    ) -> Result<PairingRequest, String> {
        self.sweep_expired();
        let same_origin = |r: &PairingRequest| r.status == PairingStatus::Pending && r.origin == origin;
        while self.pending.iter().filter(|r| same_origin(r)).count() >= MAX_PENDING_PER_ORIGIN {
            let Some(oldest) = self.pending.iter().position(|r| same_origin(r)) else { break };
            self.pending.remove(oldest);
        }
        while self.pending.len() >= MAX_PENDING {
            let turned_down = self.pending.iter().position(|r| !matches!(r.status, PairingStatus::Pending | PairingStatus::Approved));
            let Some(room) = turned_down.or_else(|| self.most_crowded(&origin)) else {
                return Err(if self.pending.iter().all(|r| r.status == PairingStatus::Approved) {
                    "too many pairing requests were approved in the last few minutes: try again shortly".into()
                } else {
                    "too many apps are asking to pair at once: answer the requests in OAIY, or try again in a few minutes".into()
                });
            };
            self.pending.remove(room);
        }
        let req = PairingRequest {
            pairing_id: format!("pair_{}", token_hex(16)),
            product: product.to_string(),
            label,
            origin,
            code: human_code(),
            access,
            status: PairingStatus::Pending,
            created_at_ms: now_ms(),
            approved_at_ms: None,
            token: None,
        };
        self.pending.push(req.clone());
        Ok(req)
    }

    /// Where room is made in a full list for a request from `asking`: the oldest
    /// waiting request of the origin that holds the most of them (the asker's
    /// own, where it holds as many as any other). `None` where no origin holds
    /// more than one: an origin always keeps its newest request, whoever asks.
    fn most_crowded(&self, asking: &Option<String>) -> Option<usize> {
        let waiting = |r: &&PairingRequest| r.status == PairingStatus::Pending;
        let held_by = |origin: &Option<String>| self.pending.iter().filter(waiting).filter(|r| &r.origin == origin).count();
        let mut most: Option<(&Option<String>, usize)> = None;
        for r in self.pending.iter().filter(waiting) {
            let held = held_by(&r.origin);
            if most.is_none_or(|(of, n)| held > n || (held == n && of != asking && &r.origin == asking)) {
                most = Some((&r.origin, held));
            }
        }
        let (origin, held) = most?;
        if held < 2 {
            return None;
        }
        self.pending.iter().position(|r| r.status == PairingStatus::Pending && &r.origin == origin)
    }

    /// The consumer polls this. Returns the request WITH its token once approved.
    /// The token stays re-fetchable by whoever holds the (unguessable) pairingId
    /// until the request ages out — deliberately NOT single-use, so a transport
    /// blip mid-collection can retry and still land the token. The exposure is
    /// bounded by the 128-bit id (only the requester knows it) and the TTL sweep.
    pub fn poll(&mut self, pairing_id: &str) -> Option<PairingRequest> {
        self.sweep_expired();
        self.pending.iter().find(|r| r.pairing_id == pairing_id).cloned()
    }

    /// Pending requests for the approval UI. Never carries a token.
    pub fn pending(&mut self) -> Vec<PairingRequest> {
        self.sweep_expired();
        self.pending
            .iter()
            .filter(|r| r.status == PairingStatus::Pending)
            .cloned()
            .collect()
    }

    /// Approve a request (the user's trust act) and mint a token. Returns the
    /// request carrying the new secret, which the poller then collects.
    pub fn approve(&mut self, pairing_id: &str) -> Result<PairingRequest, String> {
        self.sweep_expired();
        let Some(req) = self.pending.iter_mut().find(|r| r.pairing_id == pairing_id) else {
            return Err(format!("no pending pairing {pairing_id}"));
        };
        if req.status != PairingStatus::Pending {
            return Err(format!("pairing {pairing_id} is already {:?}", req.status));
        }
        let secret = format!("oaiypat_{}", token_hex(32));
        let paired = PairedToken {
            id: format!("tok_{}", token_hex(8)),
            product: req.product.clone(),
            label: req.label.clone(),
            origin: req.origin.clone(),
            access: req.access,
            created_at_ms: now_ms(),
            last_used_ms: None,
            secret: secret.clone(),
        };
        req.status = PairingStatus::Approved;
        req.approved_at_ms = Some(now_ms());
        req.token = Some(secret);
        let out = req.clone();
        self.tokens.push(paired);
        self.persist();
        Ok(out)
    }

    pub fn deny(&mut self, pairing_id: &str) -> Result<(), String> {
        let Some(req) = self.pending.iter_mut().find(|r| r.pairing_id == pairing_id) else {
            return Err(format!("no pending pairing {pairing_id}"));
        };
        // (an approved one has its token granted already: that is revoked from the paired apps, not denied here, where
        // it would leave the grant working and only stop its requester collecting it)
        if req.status != PairingStatus::Pending {
            return Err(format!("pairing {pairing_id} is already {:?}", req.status));
        }
        req.status = PairingStatus::Denied;
        req.token = None;
        Ok(())
    }

    /// Is `token` a currently-granted secret? Constant-time-ish per candidate.
    /// (The old guard's question; the guard asks [`Self::grant_for`] first, which
    /// also forgets tokens left unused past [`IDLE_EXPIRY_MS`].)
    pub fn is_valid_token(&self, token: &str) -> bool {
        self.tokens.iter().any(|t| ct_eq(&t.secret, token))
    }

    /// Who `token` is, presented with `origin` (the request's `Origin`, if any).
    /// A token granted to a page works from that page (or from no page: a
    /// program can say any Origin it likes, so refusing it there buys nothing);
    /// one granted to a program works from no page. A use is noted, and written
    /// down at most once a day.
    pub fn grant_for(&mut self, token: &str, origin: Option<&str>) -> Grant {
        if token.is_empty() {
            return Grant::Unknown;
        }
        let now = now_ms();
        self.forget_idle(now);
        let Some(t) = self.tokens.iter_mut().find(|t| ct_eq(&t.secret, token)) else {
            return Grant::Unknown;
        };
        // (`null` is no page's own address: a grant written down for it, from before a request from such a page was
        // refused, is not every sandboxed frame's and every opened file's to use)
        if origin.is_some() && (origin != t.origin.as_deref() || origin == Some("null")) {
            return Grant::WrongOrigin;
        }
        let access = t.access;
        let noted = t.last_used_ms.unwrap_or(t.created_at_ms);
        t.last_used_ms = Some(now);
        if now.saturating_sub(noted) > USE_PERSIST_MS {
            self.persist();
        }
        Grant::Paired(access)
    }

    /// Forget the tokens nobody has presented for [`IDLE_EXPIRY_MS`].
    fn forget_idle(&mut self, now: u64) {
        let before = self.tokens.len();
        self.tokens
            .retain(|t| now.saturating_sub(t.last_used_ms.unwrap_or(t.created_at_ms)) <= IDLE_EXPIRY_MS);
        if self.tokens.len() != before {
            self.persist();
        }
    }

    /// The paired apps, for a "connected apps" list. Secret-free by construction.
    pub fn paired(&mut self) -> Vec<PairedTokenPublic> {
        self.forget_idle(now_ms());
        self.tokens.iter().map(PairedToken::public).collect()
    }

    /// Revoke a granted token by its public id.
    pub fn revoke(&mut self, id: &str) -> bool {
        let before = self.tokens.len();
        self.tokens.retain(|t| t.id != id);
        let removed = self.tokens.len() != before;
        if removed {
            self.persist();
        }
        removed
    }

    fn sweep_expired(&mut self) {
        let now = now_ms();
        for r in self.pending.iter_mut() {
            if r.status == PairingStatus::Pending && now.saturating_sub(r.created_at_ms) > REQUEST_TTL_MS {
                r.status = PairingStatus::Expired;
                r.token = None;
            }
        }
        // Drop anything terminal AND old, so the list does not accrete. An approved request's age is
        // from its approval: one approved in its last moment would otherwise go at the next poll, with
        // the token granted and listed and its app told it failed.
        self.pending
            .retain(|r| r.status == PairingStatus::Pending || now.saturating_sub(r.approved_at_ms.unwrap_or(r.created_at_ms)) <= REQUEST_TTL_MS);
    }

    /// The file holds every granted token in the clear, so it is written owner-only
    /// and replaced atomically ([`crate::secret_file::write`]). A failed save does not
    /// stop the grant — the consumer is paired for this session — but it is said.
    fn persist(&self) {
        let Some(path) = &self.path else { return };
        let shape = PersistShape { tokens: self.tokens.clone() };
        let Ok(body) = serde_json::to_string_pretty(&shape) else { return };
        if let Err(e) = crate::secret_file::write(path, body) {
            log::warn!("paired-app tokens could not be saved to {}: {e}", path.display());
        }
    }
}

impl Default for PairingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// On-disk shape. Separate so the persisted schema is explicit and versionable.
/// `PairedToken` serializes its secret, so this round-trips the full grant.
#[derive(Serialize, Deserialize)]
struct PersistShape {
    tokens: Vec<PairedToken>,
}

/// Who a pairing request's `Origin` says is asking: a program on this computer (no header: `None`), or a page, by the
/// address a browser gives it (`scheme://host[:port]`, as a browser writes one: lower case, nothing after the port).
///
/// The origin is the one thing in the prompt the asker did not write, and the grant works from that page alone
/// ([`PairingManager::grant_for`]), so it has to name one page. `null` names none: it is what every sandboxed frame,
/// `data:` page and file opened from the disk sends, so a grant to it would be a grant to all of them, shown to the
/// person as the word "null". It is refused, and so is anything else that is not a page's address.
pub fn asking_origin(header: Option<&str>) -> Result<Option<String>, &'static str> {
    let Some(origin) = header else { return Ok(None) };
    if origin == "null" {
        return Err("a page with no address of its own (a sandboxed frame, a data: page, a file opened from the disk) cannot be paired: open the app at its own address");
    }
    if !is_page_origin(origin) {
        return Err("the Origin of the request is not a page's address (scheme://host or scheme://host:port)");
    }
    Ok(Some(origin.to_string()))
}

/// Whether `origin` is `scheme://host[:port]` as a browser serializes it, and nothing more: no path, no query, no
/// name and password, no space, no upper case (a browser lowers the scheme and the host, and writes a name in
/// another script as its ASCII form).
fn is_page_origin(origin: &str) -> bool {
    let Some((scheme, rest)) = origin.split_once("://") else { return false };
    if !scheme.starts_with(|c: char| c.is_ascii_lowercase()) || !scheme.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '+' | '-' | '.')) {
        return false;
    }
    let (host_is_one, port) = match rest.strip_prefix('[') {
        // (an IPv6 address, in its brackets)
        Some(inner) => {
            let Some((address, after)) = inner.split_once(']') else { return false };
            let is_one = address.parse::<std::net::Ipv6Addr>().is_ok() && !address.chars().any(|c| c.is_ascii_uppercase());
            match after.strip_prefix(':') {
                Some(port) => (is_one, Some(port)),
                None => (is_one && after.is_empty(), None),
            }
        }
        None => {
            let (host, port) = match rest.split_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (rest, None),
            };
            let named = host.chars().any(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                && host.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '-' | '_'));
            (named, port)
        }
    };
    host_is_one && port.is_none_or(|p| !p.is_empty() && p.len() <= 5 && p.chars().all(|c| c.is_ascii_digit()) && p.parse::<u16>().is_ok())
}

/// Whether `product` is a product's short name: lower-case letters, digits, `.`, `_` and `-` (`formlogic`,
/// `oaiy-app`). The prompt shows it beside the origin, and it is the asker's own word: with a space, a `·` or a `/` in
/// it, it could be written to read as an address of its own.
pub fn is_product_name(product: &str) -> bool {
    !product.is_empty()
        && product.chars().count() <= MAX_PRODUCT_CHARS
        && product.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

/// A request's `label` as the prompt may show it: without the characters a person cannot see (controls, among them
/// a line break; zero-width characters; the ones that turn the direction of the text around, with which a label
/// reads as something it does not say). `None` where nothing is left.
pub fn shown_label(label: Option<String>) -> Option<String> {
    let label = label?;
    let shown: String = label.chars().filter(|c| !c.is_control() && !crate::backup::parts::is_invisible(*c)).collect();
    let shown = shown.trim();
    (!shown.is_empty()).then(|| shown.to_string())
}

pub type PairingHandle = Arc<Mutex<PairingManager>>;

pub fn new_handle() -> PairingHandle {
    Arc::new(Mutex::new(PairingManager::new()))
}

pub fn open_handle(path: PathBuf) -> PairingHandle {
    Arc::new(Mutex::new(PairingManager::open(path)))
}

// --- helpers --------------------------------------------------------------

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Hex string of `bytes` random bytes, via uuid's CSPRNG-backed v4.
fn token_hex(bytes: usize) -> String {
    let mut out = String::with_capacity(bytes * 2);
    while out.len() < bytes * 2 {
        out.push_str(&uuid::Uuid::new_v4().simple().to_string());
    }
    out.truncate(bytes * 2);
    out
}

/// A short human-readable pairing code: 6 chars from an unambiguous alphabet (no
/// 0/O, 1/I/L), so a user reading it aloud or matching it by eye cannot confuse
/// characters.
fn human_code() -> String {
    const ALPHABET: &[u8] = b"23456789ABCDEFGHJKMNPQRSTUVWXYZ";
    let raw = uuid::Uuid::new_v4();
    raw.as_bytes()[..6]
        .iter()
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Length-independent-ish equality: no early return on the first differing byte,
/// so a token cannot be recovered by timing.
fn ct_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_is_pending_with_a_code() {
        let mut m = PairingManager::new();
        let req = m.request("formlogic", Some("Acme".into()), Some("https://formlogic.com".into()));
        assert_eq!(req.status, PairingStatus::Pending);
        assert_eq!(req.code.len(), 6);
        assert!(req.token.is_none(), "no token before approval");
        // Visible to the approval UI, without a token.
        let pending = m.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].product, "formlogic");
    }

    #[test]
    fn nothing_is_granted_until_approval() {
        let mut m = PairingManager::new();
        let req = m.request("formlogic", None, None);
        // Polling a pending request yields no token.
        assert!(m.poll(&req.pairing_id).unwrap().token.is_none());
        assert!(m.paired().is_empty());
    }

    #[test]
    fn approval_mints_a_token_the_poller_collects() {
        let mut m = PairingManager::new();
        let req = m.request("formlogic", None, None);
        let approved = m.approve(&req.pairing_id).unwrap();
        let token = approved.token.expect("approval mints a token");
        assert!(token.starts_with("oaiypat_"));
        // The poller sees it.
        let polled = m.poll(&req.pairing_id).unwrap();
        assert_eq!(polled.status, PairingStatus::Approved);
        assert_eq!(polled.token.as_deref(), Some(token.as_str()));
        // And it validates.
        assert!(m.is_valid_token(&token));
    }

    #[test]
    fn a_granted_token_remembers_which_origin_it_was_granted_to() {
        // `product` and `label` both come from the consumer and neither is
        // unique — two different sites can each call themselves "formlogic".
        // The prompt showed the user an origin; if the grant forgets it, the
        // Connections list cannot tell them apart and a user revoking access
        // is guessing which one they are cutting off.
        let mut m = PairingManager::new();
        let a = m.request("formlogic", Some("FormLogic Web".into()), Some("http://formlogic.local".into()));
        let b = m.request("formlogic", Some("FormLogic Web".into()), Some("https://evil.example".into()));
        m.approve(&a.pairing_id).unwrap();
        m.approve(&b.pairing_id).unwrap();

        let origins: Vec<Option<String>> = m.paired().iter().map(|t| t.origin.clone()).collect();
        assert!(origins.contains(&Some("http://formlogic.local".into())));
        assert!(origins.contains(&Some("https://evil.example".into())));
    }

    #[test]
    fn a_native_caller_pairs_with_no_origin_rather_than_a_fake_one() {
        // A CLI sends no Origin. Recording a placeholder would be worse than
        // recording nothing — it would look like a site.
        let mut m = PairingManager::new();
        let req = m.request("oaiy-cli", None, None);
        m.approve(&req.pairing_id).unwrap();
        assert_eq!(m.paired()[0].origin, None);
    }

    #[test]
    fn the_pending_list_never_leaks_a_token() {
        let mut m = PairingManager::new();
        let req = m.request("formlogic", None, None);
        m.approve(&req.pairing_id).unwrap();
        // An approved request is no longer "pending"; and even the serialized
        // pending shape skips the token field.
        for r in m.pending() {
            let v = serde_json::to_value(&r).unwrap();
            assert!(v.get("token").is_none(), "token must never serialize into a list");
        }
    }

    #[test]
    fn a_random_token_does_not_validate() {
        let mut m = PairingManager::new();
        let id = m.request("x", None, None).pairing_id;
        m.approve(&id).unwrap();
        assert!(!m.is_valid_token("oaiypat_deadbeef"));
        assert!(!m.is_valid_token(""));
    }

    #[test]
    fn denied_pairing_grants_nothing() {
        let mut m = PairingManager::new();
        let req = m.request("evil", None, None);
        m.deny(&req.pairing_id).unwrap();
        assert!(m.approve(&req.pairing_id).is_err(), "a denied request cannot be approved");
        assert!(m.paired().is_empty());
    }

    #[test]
    fn double_approval_is_refused() {
        let mut m = PairingManager::new();
        let req = m.request("x", None, None);
        m.approve(&req.pairing_id).unwrap();
        assert!(m.approve(&req.pairing_id).is_err(), "already approved");
        // And it did not mint a second token.
        assert_eq!(m.paired().len(), 1);
    }

    #[test]
    fn revoke_invalidates_the_token() {
        let mut m = PairingManager::new();
        let id = m.request("x", None, None).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        let id = m.paired()[0].id.clone();
        assert!(m.is_valid_token(&token));
        assert!(m.revoke(&id));
        assert!(!m.is_valid_token(&token), "a revoked token stops working");
        assert!(!m.revoke(&id), "revoking twice is a no-op");
    }

    #[test]
    fn pending_requests_are_bounded() {
        let mut m = PairingManager::new();
        for i in 0..(MAX_PENDING + 10) {
            m.request(&format!("p{i}"), None, None);
        }
        assert!(m.pending().len() <= MAX_PENDING, "a spamming page cannot grow the list without bound");
    }

    #[test]
    fn distinct_requests_get_distinct_codes_and_ids() {
        let mut m = PairingManager::new();
        let a = m.request("x", None, None);
        let b = m.request("y", None, None);
        assert_ne!(a.pairing_id, b.pairing_id);
        // Codes could theoretically collide (6 chars) but ids never do.
        assert_ne!(a.code.chars().count(), 0);
    }

    #[test]
    fn tokens_persist_across_reopen() {
        let dir = std::env::temp_dir().join(format!("oaiy-pair-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("pairings.json");
        let token;
        {
            let mut m = PairingManager::open(path.clone());
            let id = m
                .request("formlogic", Some("Acme".into()), Some("http://formlogic.local".into()))
                .pairing_id;
            token = m.approve(&id).unwrap().token.unwrap();
        }
        // A fresh manager on the same file still honours the token — a paired
        // consumer stays paired across a restart.
        let mut m2 = PairingManager::open(path.clone());
        assert!(m2.is_valid_token(&token), "the granted token survived the reopen");
        assert_eq!(m2.paired().len(), 1);
        assert_eq!(m2.paired()[0].product, "formlogic");
        assert_eq!(m2.paired()[0].origin.as_deref(), Some("http://formlogic.local"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_tokens_are_stored_owner_only_and_revoking_keeps_them_so() {
        use crate::secret_file::testing::{assert_private, assert_private_dir, TempDir};
        let dir = TempDir::new("pairings");
        let path = dir.0.join("bridge").join("pairings.json");
        let mut m = PairingManager::open(path.clone());
        let id = m.request("formlogic", None, None).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        assert_private(&path);
        assert_private_dir(path.parent().unwrap());
        assert!(std::fs::read_to_string(&path).unwrap().contains(&token), "the grant is what the file is for");

        let granted = m.paired()[0].id.clone();
        assert!(m.revoke(&granted));
        assert_private(&path);
        assert!(!std::fs::read_to_string(&path).unwrap().contains(&token), "a revoked token is gone from the file");
    }

    #[test]
    fn a_token_works_from_the_page_it_was_granted_to_and_from_no_page() {
        let mut m = PairingManager::new();
        let id = m.request("formlogic", None, Some("https://formlogic.com".into())).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        assert_eq!(m.grant_for(&token, Some("https://formlogic.com")), Grant::Paired(PairingAccess::Apps));
        // (a program can say any Origin it likes: refusing it where it says none buys nothing)
        assert_eq!(m.grant_for(&token, None), Grant::Paired(PairingAccess::Apps));
        for elsewhere in ["https://evil.example", "https://formlogic.com.evil.example", "null", "http://formlogic.com"] {
            assert_eq!(m.grant_for(&token, Some(elsewhere)), Grant::WrongOrigin, "{elsewhere}");
        }
        assert_eq!(m.grant_for("oaiypat_nope", Some("https://formlogic.com")), Grant::Unknown);
        assert_eq!(m.grant_for("", None), Grant::Unknown);
    }

    #[test]
    fn a_program_s_token_works_from_no_page() {
        let mut m = PairingManager::new();
        let id = m.request("script", None, None).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        assert_eq!(m.grant_for(&token, None), Grant::Paired(PairingAccess::Apps));
        assert_eq!(m.grant_for(&token, Some("https://evil.example")), Grant::WrongOrigin, "a page holding a program's token is not that program");
    }

    #[test]
    fn the_token_carries_the_access_its_request_asked_for() {
        let mut m = PairingManager::new();
        let id = m.request_for("oaiy-agent", None, Some("https://oaiy.com".into()), PairingAccess::Agent).unwrap().pairing_id;
        assert_eq!(m.pending()[0].access, PairingAccess::Agent, "the prompt can say what it asks for");
        let token = m.approve(&id).unwrap().token.unwrap();
        assert_eq!(m.grant_for(&token, Some("https://oaiy.com")), Grant::Paired(PairingAccess::Agent));
        assert_eq!(m.paired()[0].access, PairingAccess::Agent);
    }

    #[test]
    fn what_each_kind_of_grant_reaches() {
        for scope in ["flows.read", "flows.write", "runs.write", "connectors.use", "events.read", "plugins.control", "ai.use"] {
            assert!(PairingAccess::Apps.reaches(scope) && PairingAccess::Agent.reaches(scope), "{scope}");
        }
        for scope in ["calls.read", "contacts.write", "calendar.write", "control.read", "agent.serve", "models.write"] {
            assert!(!PairingAccess::Apps.reaches(scope), "an app's grant does not reach {scope}");
            assert!(PairingAccess::Agent.reaches(scope), "the Agent's does reach {scope}");
        }
        // Neither: what defines code or trust (services, plugins, runtimes), pairing and OAIY Network's trust, setup, links,
        // the owner's settings and secrets.
        for scope in [
            "services.define", "plugins.install", "runtimes.install", "auth.read", "auth.manage", "auth.revoke", "setup.write",
            "link.manage", "companion.manage", "ai.admin", "secrets.write", "system.read", "logs.read", "control.admin",
            "calls.settings", "agent.settings", "vault.admin",
        ] {
            assert!(!PairingAccess::Apps.reaches(scope) && !PairingAccess::Agent.reaches(scope), "{scope}");
        }
    }

    #[test]
    fn an_old_grant_is_an_apps_grant_and_a_use_is_noted() {
        let dir = crate::secret_file::testing::TempDir::new("pairings-old");
        let path = dir.0.join("pairings.json");
        std::fs::write(&path, format!(r#"{{"tokens":[{{"id":"tok_1","product":"formlogic","label":null,"origin":null,"createdAtMs":{},"secret":"oaiypat_old"}}]}}"#, now_ms())).unwrap();
        let mut m = PairingManager::open(path.clone());
        assert_eq!(m.grant_for("oaiypat_old", None), Grant::Paired(PairingAccess::Apps));
        assert!(m.paired()[0].last_used_ms.is_some());
    }

    #[test]
    fn a_token_left_unused_is_forgotten() {
        let mut m = PairingManager::new();
        let id = m.request("formlogic", None, None).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        m.tokens[0].created_at_ms = now_ms() - IDLE_EXPIRY_MS - 1000;
        assert_eq!(m.grant_for(&token, None), Grant::Unknown);
        assert!(m.paired().is_empty());
        // One used within the time stays.
        let id = m.request("formlogic", None, None).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        m.tokens[0].created_at_ms = now_ms() - IDLE_EXPIRY_MS - 1000;
        m.tokens[0].last_used_ms = Some(now_ms() - 1000);
        assert_eq!(m.grant_for(&token, None), Grant::Paired(PairingAccess::Apps));
    }

    #[test]
    fn one_origin_cannot_push_the_others_out() {
        let mut m = PairingManager::new();
        let real = m.request("formlogic", None, Some("https://formlogic.com".into())).pairing_id;
        for i in 0..50 {
            m.request(&format!("spam{i}"), None, Some("https://evil.example".into()));
        }
        let pending = m.pending();
        assert!(pending.iter().any(|r| r.pairing_id == real), "the real request is still there");
        assert_eq!(pending.iter().filter(|r| r.origin.as_deref() == Some("https://evil.example")).count(), MAX_PENDING_PER_ORIGIN);

        // Nor can a page with several origins. Four of them fill the list with three requests each, and the next
        // request used to put out the oldest request of all: the real one's, which the person never saw.
        let mut m = PairingManager::new();
        let real = m.request("formlogic", None, Some("https://formlogic.com".into())).pairing_id;
        let script = m.request("oaiy-cli", None, None).pairing_id;
        for round in 0..10 {
            for site in 0..6 {
                let asked = m.request_for(&format!("spam{round}"), None, Some(format!("https://evil{site}.example")), PairingAccess::Apps);
                assert!(asked.is_ok(), "an origin that crowds the list makes room out of its own");
            }
        }
        let pending = m.pending();
        assert_eq!(pending.len(), MAX_PENDING);
        assert!(pending.iter().any(|r| r.pairing_id == real), "the real request is still there");
        assert!(pending.iter().any(|r| r.pairing_id == script), "and so is a program's");
        // An app that comes late still gets its prompt: room is made of the origin that holds the most.
        let late = m.request_for("late", None, Some("https://late.example".into()), PairingAccess::Apps).expect("room for it").pairing_id;
        let pending = m.pending();
        assert_eq!(pending.len(), MAX_PENDING);
        for id in [&real, &script, &late] {
            assert!(pending.iter().any(|r| &r.pairing_id == id), "{id}");
        }
    }

    #[test]
    fn nobodys_only_request_is_put_out_for_another() {
        // Every origin holds one: there is no room that is not somebody's only prompt, so the newcomer is told to wait
        // (the route answers 429), however many origins a page has.
        let mut m = PairingManager::new();
        let first: Vec<String> = (0..MAX_PENDING).map(|i| m.request("app", None, Some(format!("https://app{i}.example"))).pairing_id).collect();
        let refused = m.request_for("late", None, Some("https://late.example".into()), PairingAccess::Apps);
        assert!(refused.unwrap_err().contains("too many apps are asking"));
        assert_eq!(m.pending().iter().map(|r| r.pairing_id.clone()).collect::<Vec<_>>(), first, "all twelve are as they were");
        // One of the twelve asking again has no room either: it holds one, and room is made only of an origin's older request.
        let again = m.request_for("app", None, Some("https://app3.example".into()), PairingAccess::Apps);
        assert!(again.unwrap_err().contains("too many apps are asking"));
        // A request the person turned down waits for nobody: it is the room.
        m.deny(&first[5]).unwrap();
        let late = m.request_for("late", None, Some("https://late.example".into()), PairingAccess::Apps).expect("room where a request was denied").pairing_id;
        let pending = m.pending();
        assert_eq!(pending.len(), MAX_PENDING);
        assert!(pending.iter().any(|r| r.pairing_id == late));
        assert!(first.iter().enumerate().all(|(i, id)| i == 5 || pending.iter().any(|r| &r.pairing_id == id)), "the other eleven are still asked about");
        // An origin with two keeps its newest: where two hold as many, the asker's own older request goes first.
        let mut m = PairingManager::new();
        let theirs_old = m.request("theirs", None, Some("https://theirs.example".into())).pairing_id;
        let theirs_new = m.request("theirs", None, Some("https://theirs.example".into())).pairing_id;
        let mine_old = m.request("mine", None, Some("https://mine.example".into())).pairing_id;
        let mine_new = m.request("mine", None, Some("https://mine.example".into())).pairing_id;
        for i in 0..MAX_PENDING - 4 {
            m.request("app", None, Some(format!("https://app{i}.example")));
        }
        m.request("mine", None, Some("https://mine.example".into()));
        let ids: Vec<String> = m.pending().into_iter().map(|r| r.pairing_id).collect();
        assert!(!ids.contains(&mine_old) && ids.contains(&mine_new) && ids.contains(&theirs_old) && ids.contains(&theirs_new));
    }

    #[test]
    fn a_request_names_the_page_that_asks_or_a_program_and_null_names_nobody() {
        // A program on this computer sends no Origin, and is asked about as one.
        assert_eq!(asking_origin(None), Ok(None));
        // A page, by the address a browser gives it.
        for page in [
            "https://formlogic.com",
            "https://app.formlogic.com:8443",
            "http://formlogic.local",
            "http://localhost:5173",
            "http://127.0.0.1:17973",
            "http://[::1]:8080",
            "http://[::1]",
            "http://my_site.local",
            "https://xn--mnchen-3ya.example",
            "tauri://localhost",
            "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
        ] {
            assert_eq!(asking_origin(Some(page)), Ok(Some(page.to_string())), "{page}");
        }
        // A sandboxed frame, a data: page, a file opened from the disk: every one of them is "null", so it names none.
        assert!(asking_origin(Some("null")).is_err());
        // Not an address of a page and nothing more: these would be shown in the prompt as who asks.
        for not_one in [
            "",
            "Null",
            "NULL",
            "formlogic.com",
            "https://",
            "https:///",
            "https://formlogic.com/",
            "https://formlogic.com/path",
            "https://formlogic.com?x=1",
            "https://formlogic.com#x",
            "https://user@formlogic.com",
            "https://user:pw@formlogic.com",
            "https://formlogic.com:",
            "https://formlogic.com:port",
            "https://formlogic.com:99999",
            "https://formlogic.com:80:80",
            "https://formlogic.com evil.example",
            " https://formlogic.com",
            "https://FormLogic.com",
            "HTTPS://formlogic.com",
            "https://formlogic.com\u{202e}moc.live",
            "https://formlogіc.com",
            "https://formlogic.com · approved",
            "https://...",
            "http://[::1",
            "http://[::1]x",
            "http://[not-an-address]",
            "1https://formlogic.com",
            "javascript:alert(1)",
            "data:text/html,x",
            "file:///C:/page.html",
        ] {
            assert!(asking_origin(Some(not_one)).is_err(), "{not_one:?}");
        }
    }

    #[test]
    fn a_product_is_a_short_name_and_a_label_is_shown_without_what_a_person_cannot_see() {
        let (longest, too_long) = ("a".repeat(MAX_PRODUCT_CHARS), "a".repeat(MAX_PRODUCT_CHARS + 1));
        // What FormLogic and the Agent in a browser tab send, and what a script may call itself.
        for product in ["formlogic", "oaiy-app", "oaiy-cli", "my_script.v2", "x", longest.as_str()] {
            assert!(is_product_name(product), "{product}");
        }
        // Shown beside the address the request really came from: it cannot be written to read as one.
        for product in ["", " ", "formlogic · https://formlogic.com", "https://formlogic.com", "FormLogic", "form logic", "formlogic\n", "formlogic\u{202e}", "formlogіc", too_long.as_str()] {
            assert!(!is_product_name(product), "{product:?}");
        }
        // FormLogic's label is its own address: it is kept as it is.
        assert_eq!(shown_label(Some("https://formlogic.com".into())).as_deref(), Some("https://formlogic.com"));
        assert_eq!(shown_label(Some("Receptionist · Acme Dental".into())).as_deref(), Some("Receptionist · Acme Dental"));
        assert_eq!(shown_label(Some("Zoë’s café 日本語".into())).as_deref(), Some("Zoë’s café 日本語"));
        // Controls (a line break too), zero-width characters and the ones that turn the text's direction around.
        assert_eq!(shown_label(Some("Acme\u{202e}moc.cigolmrof//:sptth".into())).as_deref(), Some("Acmemoc.cigolmrof//:sptth"));
        assert_eq!(shown_label(Some("A\u{0}c\u{7}m\u{1b}e\r\n\tDental\u{200b}\u{2066}\u{2069}\u{feff}\u{e0041}".into())).as_deref(), Some("AcmeDental"));
        assert_eq!(shown_label(Some("  Acme  ".into())).as_deref(), Some("Acme"));
        // Nothing left is no label, as none sent is.
        assert_eq!(shown_label(Some("\u{200b}\u{202e} \n".into())), None);
        assert_eq!(shown_label(Some(String::new())), None);
        assert_eq!(shown_label(None), None);
    }

    #[test]
    fn a_grant_written_down_for_null_is_not_every_opaque_pages_to_use() {
        // A file from before a request with `Origin: null` was refused: the grant was made to "null", which every
        // sandboxed frame and every file opened from the disk says it is.
        let dir = std::env::temp_dir().join(format!("oaiy-pairing-null-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pairings.json");
        std::fs::write(&path, format!(r#"{{"tokens":[{{"id":"tok_1","product":"formlogic","label":null,"origin":"null","createdAtMs":{},"secret":"oaiypat_null"}}]}}"#, now_ms())).unwrap();
        let mut m = PairingManager::open(path);
        assert_eq!(m.grant_for("oaiypat_null", Some("null")), Grant::WrongOrigin);
        assert_eq!(m.grant_for("oaiypat_null", Some("https://evil.example")), Grant::WrongOrigin);
        // It is still listed, with the origin it was made to, so the person can see it and revoke it.
        assert_eq!(m.paired()[0].origin.as_deref(), Some("null"));
        // A grant to a page is as it was: that page's, and no opaque page's.
        let id = m.request("formlogic", None, Some("https://formlogic.com".into())).pairing_id;
        let token = m.approve(&id).unwrap().token.unwrap();
        assert_eq!(m.grant_for(&token, Some("https://formlogic.com")), Grant::Paired(PairingAccess::Apps));
        assert_eq!(m.grant_for(&token, Some("null")), Grant::WrongOrigin);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_approved_request_waiting_to_be_collected_is_never_evicted() {
        let mut m = PairingManager::new();
        let approved: Vec<String> = (0..MAX_PENDING)
            .map(|i| {
                let id = m.request(&format!("app{i}"), None, Some(format!("https://app{i}.example"))).pairing_id;
                m.approve(&id).unwrap();
                id
            })
            .collect();
        let refused = m.request_for("late", None, Some("https://late.example".into()), PairingAccess::Apps);
        assert!(refused.is_err(), "no room is made by throwing an approved token away");
        for id in &approved {
            assert!(m.poll(id).unwrap().token.is_some(), "{id} can still be collected");
        }
        assert!(m.deny(&approved[0]).is_err(), "an approved request is not denied after the fact (its grant is revoked instead)");
    }

    /// Approved at 4 minutes 59 of a request's five, and swept a moment later by its age from when it was raised: the app
    /// polled 404, and showed as failed, with a token granted that it never received.
    #[test]
    fn a_request_approved_late_stays_a_full_ttl_for_its_token_to_be_collected() {
        let mut m = PairingManager::new();
        let id = m.request("late", None, None).pairing_id;
        m.pending[0].created_at_ms -= REQUEST_TTL_MS - 1_000;
        m.approve(&id).unwrap();
        m.pending[0].created_at_ms -= 2_000;
        assert!(m.poll(&id).expect("still there").token.is_some(), "the token is still to be collected");
        // A full TTL from the approval, and no longer.
        m.pending[0].approved_at_ms = Some(now_ms() - REQUEST_TTL_MS - 1_000);
        assert!(m.poll(&id).is_none(), "a token nobody came for does not stay in the list");
    }

    #[test]
    fn the_code_alphabet_avoids_ambiguous_characters() {
        let mut m = PairingManager::new();
        for _ in 0..50 {
            let code = m.request("x", None, None).code;
            assert!(!code.contains(['0', 'O', '1', 'I', 'L']), "ambiguous char in {code}");
        }
    }
}
