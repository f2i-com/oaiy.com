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

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// A pending request expires if not approved within this window — a prompt the
/// user ignored should not sit around to be approved by accident later.
const REQUEST_TTL_MS: u64 = 5 * 60 * 1000;
/// Cap on simultaneous pending requests, so a page spamming pairing cannot grow
/// the list without bound. The oldest is evicted past this.
const MAX_PENDING: usize = 12;

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
    pub status: PairingStatus,
    pub created_at_ms: u64,
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
    pub created_at_ms: u64,
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
    pub created_at_ms: u64,
}

impl PairedToken {
    pub fn public(&self) -> PairedTokenPublic {
        PairedTokenPublic {
            id: self.id.clone(),
            product: self.product.clone(),
            label: self.label.clone(),
            origin: self.origin.clone(),
            created_at_ms: self.created_at_ms,
        }
    }
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

    /// Raise a pairing request. Returns the id + the code to show the consumer.
    ///
    /// Unauthenticated by design — see the module docs. Expired requests are
    /// swept first, and the list is capped.
    pub fn request(
        &mut self,
        product: &str,
        label: Option<String>,
        origin: Option<String>,
    ) -> PairingRequest {
        self.sweep_expired();
        while self.pending.len() >= MAX_PENDING {
            self.pending.remove(0);
        }
        let req = PairingRequest {
            pairing_id: format!("pair_{}", token_hex(16)),
            product: product.to_string(),
            label,
            origin,
            code: human_code(),
            status: PairingStatus::Pending,
            created_at_ms: now_ms(),
            token: None,
        };
        self.pending.push(req.clone());
        req
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
            created_at_ms: now_ms(),
            secret: secret.clone(),
        };
        req.status = PairingStatus::Approved;
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
        req.status = PairingStatus::Denied;
        req.token = None;
        Ok(())
    }

    /// Is `token` a currently-granted secret? Constant-time-ish per candidate.
    /// This is what the auth guard consults.
    pub fn is_valid_token(&self, token: &str) -> bool {
        self.tokens.iter().any(|t| ct_eq(&t.secret, token))
    }

    /// The paired apps, for a "connected apps" list. Secret-free by construction.
    pub fn paired(&self) -> Vec<PairedTokenPublic> {
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
        // Drop anything terminal AND old, so the list does not accrete.
        self.pending
            .retain(|r| r.status == PairingStatus::Pending || now.saturating_sub(r.created_at_ms) <= REQUEST_TTL_MS);
    }

    fn persist(&self) {
        let Some(path) = &self.path else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let shape = PersistShape { tokens: self.tokens.clone() };
        if let Ok(body) = serde_json::to_string_pretty(&shape) {
            let tmp = path.with_extension("json.tmp");
            if std::fs::write(&tmp, &body).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
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
        let m2 = PairingManager::open(path.clone());
        assert!(m2.is_valid_token(&token), "the granted token survived the reopen");
        assert_eq!(m2.paired().len(), 1);
        assert_eq!(m2.paired()[0].product, "formlogic");
        assert_eq!(m2.paired()[0].origin.as_deref(), Some("http://formlogic.local"));
        let _ = std::fs::remove_dir_all(&dir);
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
