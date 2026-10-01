//! The credential store: `<data>/auth/credentials.json` in memory, and the rules for what may be in it.
//!
//! - **The running process is the only writer** and memory is authoritative. Nothing here re-reads a
//!   file while running (no mtime watch, no `SIGHUP`), so nothing written to disk, a flow's output
//!   included, can change a live credential. A second process on the data folder is refused by the
//!   lock (`lock.rs`).
//! - Writes are atomic (temp file in the same directory, `fsync`, rename) and private (`0600` in a
//!   `0700` folder on Unix), through `secret_file::write`. `last_used_*` are kept in memory and
//!   flushed at most once a minute and at shutdown; a revocation is written at once.
//! - Secrets are stored as SHA-256 hashes and compared in constant time. An unknown id and a wrong
//!   secret are the same `401`, and take the same work.
//! - Readers keep unknown fields (`#[serde(flatten)]`), so an additive field never needs a new `v`.
//!   A file with an unknown `v` is an error for the server and an empty, memory-only, never-written
//!   store for the desktop. An unparsable `credentials.json` is moved aside and the store starts
//!   empty; an unparsable `owner.json` is a startup error; only `ENOENT` means "not there yet".
//! - Effective scopes are computed at each use from the running binary's tables, not frozen at
//!   mint, for sessions and desk credentials; a `pat` keeps exactly the scopes it was approved with.
//! - Limits: 512 persisted credentials, 64 live children per parent, 32 live sessions (the oldest idle
//!   one is revoked to make room) and one child session per `(parent, app)`.
//! - Signing in never needs a write: a session made while the disk is full lives in memory, marked
//!   not persisted. Every other writer (a pairing, a token) fails with `store_unavailable`.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::audit::{AuditLog, Context as AuditContext};
use super::chain::{self, Broken, STATIC_PARENT};
pub use super::clock::{Clock, ManualClock, SystemClock};
use super::lock::{AuthLock, LockError};
use super::presets::{control_project_needs_read, App, Preset};
use super::principal::{Principal, PrincipalKind};
use super::scopes::{
    is_auth_scope, is_dangerous, is_reserved, is_resource_scope, never_on_a_token, ScopeSet,
    NATIVE_TOKEN_DANGEROUS,
};
use super::token::{self, Kind, MintError};

/// The `v` this build reads and writes.
pub const FILE_VERSION: u64 = 1;
pub const MAX_PERSISTED: usize = 512;
pub const MAX_CHILDREN: usize = 64;
pub const MAX_SESSIONS: usize = 32;
/// Derived credentials one credential may make in a minute.
pub const DERIVE_PER_MINUTE: usize = 30;
/// A record expired or revoked for longer than this is dropped.
pub const PURGE_AFTER_MS: u64 = 7 * DAY;
/// A `run` credential (a derived or per-run one: memory-only, made in bulk) is dropped this long after it
/// ended, not seven days: long enough that a client that presents a stale one is told it expired and not that
/// it is unknown, short enough that thirty a minute do not pile up (10,800 in six hours).
pub const RUN_PURGE_AFTER_MS: u64 = 10 * MINUTE;
pub const FLUSH_EVERY_MS: u64 = 60_000;
pub const PURGE_EVERY_MS: u64 = HOUR;
/// Exit code of a refused configuration (`EX_CONFIG`): the unit does not restart it.
pub const EX_CONFIG: i32 = 78;
/// Below this much free space the store says `low`.
pub const LOW_SPACE_BYTES: u64 = 64 * 1024 * 1024;

const MINUTE: u64 = 60_000;
const HOUR: u64 = 60 * MINUTE;
const DAY: u64 = 24 * HOUR;
/// A hash no token has: compared against when the id is unknown, so an unknown id and a wrong
/// secret cost the same.
const DUMMY_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

// ---- writers ------------------------------------------------------------------------------------

/// How the store puts bytes on disk; the tests replace it to fail.
pub trait FileWriter: Send + Sync {
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()>;
}

/// Atomic, private: `secret_file::write`.
pub struct SecureWriter;

impl FileWriter for SecureWriter {
    fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        crate::secret_file::write(path, bytes)
    }
}

/// Whether a write failed because the disk is full or read-only.
pub fn is_storage_error(e: &io::Error) -> bool {
    #[cfg(unix)]
    const CODES: &[i32] = &[28, 30, 122]; // ENOSPC, EROFS, EDQUOT
    #[cfg(windows)]
    const CODES: &[i32] = &[112, 39, 19]; // ERROR_DISK_FULL, ERROR_HANDLE_DISK_FULL, ERROR_WRITE_PROTECT
    #[cfg(not(any(unix, windows)))]
    const CODES: &[i32] = &[];
    e.raw_os_error().is_some_and(|c| CODES.contains(&c))
}

// ---- records ----------------------------------------------------------------------------------

fn is_zero(n: &u64) -> bool {
    *n == 0
}

/// One credential as `credentials.json` holds it. Nothing here is the secret: `hash` is the SHA-256
/// of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub kind: Kind,
    pub hash: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub product: Option<String>,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default)]
    pub preset_ver: Option<u32>,
    /// The scopes it was made with. A session's or desk's effective scopes are recomputed from
    /// `(app, preset)`; a `pat`'s are exactly these.
    #[serde(default)]
    pub scopes: ScopeSet,
    #[serde(default)]
    pub origin: Option<String>,
    /// Further origins a desk credential is bound to (the dashboard has three).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origin_alt: Vec<String>,
    #[serde(default)]
    pub app: Option<App>,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub created_ms: u64,
    /// Absolute expiry. Zero (a missing field) is expired.
    #[serde(default)]
    pub expires_ms: u64,
    /// The idle timeout of a session, in milliseconds.
    #[serde(default)]
    pub idle_ms: Option<u64>,
    #[serde(default)]
    pub last_used_ms: Option<u64>,
    #[serde(default)]
    pub last_used_ip: Option<String>,
    #[serde(default)]
    pub revoked_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_reason: Option<String>,
    /// Imported from a plaintext pairing file; looked up by the hash of the whole token.
    #[serde(default)]
    pub legacy: bool,
    /// Reserved for a sender-constrained mode; must be null in this version.
    #[serde(default)]
    pub cnf: Option<Value>,
    #[serde(default)]
    pub max_uses: Option<u32>,
    #[serde(default)]
    pub uses: u32,
    #[serde(default)]
    pub created_by: Option<String>,
    /// Sessions only: the owner's `min_session_epoch` when it was made.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub epoch: u64,
    /// Made while the disk would not take a write: lives in memory, is never written.
    #[serde(skip)]
    pub memory_only: bool,
    /// Fields a newer OAIY wrote, kept and written back.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Record {
    #[cfg(test)]
    pub(crate) fn blank(id: &str, kind: Kind) -> Record {
        Record {
            id: id.to_string(),
            kind,
            hash: DUMMY_HASH.to_string(),
            label: String::new(),
            product: None,
            preset: None,
            preset_ver: None,
            scopes: ScopeSet::empty(),
            origin: None,
            origin_alt: Vec::new(),
            app: None,
            parent: None,
            created_ms: 0,
            expires_ms: 0,
            idle_ms: None,
            last_used_ms: None,
            last_used_ip: None,
            revoked_ms: None,
            revoked_reason: None,
            legacy: false,
            cnf: None,
            max_uses: None,
            uses: 0,
            created_by: None,
            epoch: 0,
            memory_only: false,
            extra: Map::new(),
        }
    }

    /// The origins the credential is bound to.
    pub fn origins(&self) -> Vec<String> {
        self.origin
            .iter()
            .chain(self.origin_alt.iter())
            .cloned()
            .collect()
    }

    fn persisted_kind(&self) -> bool {
        persisted_kind(self.kind)
    }

    /// Whether it is written to disk.
    fn on_disk(&self) -> bool {
        self.persisted_kind() && !self.memory_only
    }

    /// Alive: not revoked and not past its absolute expiry (idle time and ancestors are not looked at).
    fn alive(&self, now: u64) -> bool {
        self.revoked_ms.is_none() && now < self.expires_ms
    }
}

/// Sessions and paired tokens are written; the rest live only in memory.
fn persisted_kind(kind: Kind) -> bool {
    matches!(kind, Kind::Pat | Kind::Ses)
}

// ---- errors -----------------------------------------------------------------------------------

/// Why a store could not be opened.
#[derive(Debug)]
pub enum StoreError {
    Lock(LockError),
    /// A read error other than "not there": never "corrupt" and never "setup-only".
    Unreadable {
        file: PathBuf,
        source: io::Error,
    },
    /// A file written by a newer OAIY (the server only; the desktop starts memory-only).
    UnknownVersion {
        file: PathBuf,
        found: u64,
        supported: u64,
    },
    /// `owner.json` that does not parse: a mangled owner file must not reopen setup.
    OwnerUnparsable {
        file: PathBuf,
        detail: String,
    },
}

impl StoreError {
    /// The exit code the server uses: `78`, which the shipped unit does not restart.
    pub fn exit_code(&self) -> i32 {
        EX_CONFIG
    }
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Lock(e) => write!(f, "{e}"),
            StoreError::Unreadable { file, source } => write!(f, "cannot read {}: {source}", file.display()),
            StoreError::UnknownVersion { file, found, supported } => {
                write!(f, "{} was written by a newer OAIY (version {found}; this one reads version {supported}): update OAIY", file.display())
            }
            StoreError::OwnerUnparsable { file, detail } => write!(f, "{} cannot be read ({detail}); it is not replaced so that setup cannot be reopened: restore it or run `oaiy-server auth init --force`", file.display()),
        }
    }
}

impl std::error::Error for StoreError {}

/// Why a credential was not made.
#[derive(Debug)]
pub enum MintFailure {
    /// No randomness, or a kind that is not made here.
    Token(MintError),
    /// The 513th persisted credential. `409 too_many_credentials`.
    TooManyCredentials,
    /// The 65th live child of one parent. `409 too_many_credentials`.
    TooManyChildren,
    /// A scope the kind may not hold (`400 scope_not_grantable`).
    ScopeNotGrantable(String),
    /// A lifetime over the kind's maximum.
    TtlTooLong { max_ms: u64 },
    /// A request that is not well formed (`400 invalid_request`).
    Invalid(&'static str),
    /// The parent is not valid (revoked, expired, gone).
    ParentInvalid,
    /// The store could not write. `503 store_unavailable`.
    StoreUnavailable(io::Error),
}

impl std::fmt::Display for MintFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MintFailure::Token(e) => write!(f, "{e}"),
            MintFailure::TooManyCredentials => write!(f, "too many credentials"),
            MintFailure::TooManyChildren => {
                write!(f, "too many credentials derived from one parent")
            }
            MintFailure::ScopeNotGrantable(s) => write!(f, "{s}"),
            MintFailure::TtlTooLong { max_ms } => write!(
                f,
                "the lifetime is over the maximum of {} seconds",
                max_ms / 1000
            ),
            MintFailure::Invalid(why) => write!(f, "{why}"),
            MintFailure::ParentInvalid => write!(f, "the parent credential is not valid"),
            MintFailure::StoreUnavailable(e) => write!(f, "the credential store cannot write: {e}"),
        }
    }
}

impl std::error::Error for MintFailure {}

impl From<MintError> for MintFailure {
    fn from(e: MintError) -> Self {
        MintFailure::Token(e)
    }
}

/// Why a presented token was not accepted. Every variant is a `401`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthError {
    /// Unknown id, wrong secret, wrong kind, malformed: one answer for all of them.
    Invalid,
    /// Past its expiry (not a session).
    Expired,
    /// Revoked, or a parent ended (not a session).
    Revoked { reason: &'static str },
    /// A session that ended.
    SessionEnded { reason: &'static str },
}

impl AuthError {
    /// The error code of the response body (design 4.4).
    pub fn code(&self) -> &'static str {
        match self {
            AuthError::Invalid => "token_invalid",
            AuthError::Expired => "token_expired",
            AuthError::Revoked { .. } => "token_revoked",
            AuthError::SessionEnded { .. } => "session_expired",
        }
    }

    /// The `reason` the body adds: `idle`, `absolute`, `logged_out`, `password_changed`, `revoked`,
    /// `upgrade` or `parent_ended`.
    pub fn reason(&self) -> Option<&'static str> {
        match self {
            AuthError::Revoked { reason } | AuthError::SessionEnded { reason } => Some(reason),
            _ => None,
        }
    }
}

fn revoke_reason(stored: Option<&str>) -> &'static str {
    match stored {
        Some("logged_out") => "logged_out",
        Some("password_changed") => "password_changed",
        Some("upgrade") => "upgrade",
        _ => "revoked",
    }
}

/// Why a derived credential was not made.
#[derive(Debug)]
pub enum DeriveError {
    /// `403 derive_refused`: a parent that may not derive, or more than the parent holds.
    Refused(&'static str),
    /// `429`: too many in a minute.
    RateLimited {
        retry_after_s: u64,
    },
    Mint(MintFailure),
}

impl std::fmt::Display for DeriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeriveError::Refused(why) => write!(f, "{why}"),
            DeriveError::RateLimited { retry_after_s } => write!(
                f,
                "too many derived credentials; try again in {retry_after_s} seconds"
            ),
            DeriveError::Mint(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DeriveError {}

// ---- what a mint asks for ---------------------------------------------------------------------

/// What to make. Build it with [`MintSpec::new`] and set what differs.
#[derive(Clone, Debug)]
pub struct MintSpec {
    pub kind: Kind,
    pub label: String,
    pub product: Option<String>,
    pub preset: Option<Preset>,
    pub scopes: ScopeSet,
    /// The origins it is bound to. Never `null`; a `pat` has at most one.
    pub origins: Vec<String>,
    pub app: Option<App>,
    pub parent: Option<String>,
    pub ttl_ms: u64,
    pub idle_ms: Option<u64>,
    pub created_by: Option<String>,
    pub max_uses: Option<u32>,
    /// Fields written into the record beside its own: what the login keeps about a session (where it was made
    /// and by which browser). They are kept in `credentials.json` and never read by the store.
    pub extra: Map<String, Value>,
}

impl MintSpec {
    pub fn new(kind: Kind, label: &str, scopes: ScopeSet, ttl_ms: u64) -> MintSpec {
        MintSpec {
            kind,
            label: label.to_string(),
            product: None,
            preset: None,
            scopes,
            origins: Vec::new(),
            app: None,
            parent: None,
            ttl_ms,
            idle_ms: None,
            created_by: None,
            max_uses: None,
            extra: Map::new(),
        }
    }
}

/// A credential just made. The `Debug` form never shows the token.
#[derive(Clone)]
pub struct Minted {
    /// The secret, shown once.
    pub token: String,
    pub id: String,
    pub expires_ms: u64,
    pub scopes: ScopeSet,
}

impl std::fmt::Debug for Minted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Minted")
            .field("id", &self.id)
            .field("token", &"[redacted]")
            .field("expires_ms", &self.expires_ms)
            .finish()
    }
}

/// A request to derive a credential from the caller's own.
#[derive(Clone, Debug)]
pub struct DeriveRequest {
    pub scopes: ScopeSet,
    /// Wanted lifetime; the default is an hour.
    pub ttl_ms: Option<u64>,
    pub label: String,
}

pub const DEFAULT_DERIVE_TTL_MS: u64 = HOUR;
pub const MAX_DERIVE_TTL_MS: u64 = DAY;

// ---- the grant rules of 4.2.3 -----------------------------------------------------------------

const PAT_BROWSER_MAX: u64 = 90 * DAY;
const PAT_NATIVE_MAX: u64 = 365 * DAY;
const PAT_DANGEROUS_MAX: u64 = DAY;
const CEREMONY_MAX: u64 = 5 * MINUTE;
const SESSION_MAX: u64 = 30 * DAY;
const DESK_MAX: u64 = 365 * DAY;
pub const DEFAULT_SESSION_IDLE_MS: u64 = 8 * HOUR;

/// A well-formed origin, lowercased: `scheme://host[:port]` with no path, query, fragment or
/// userinfo, and never `null`.
pub fn canonical_origin(origin: &str) -> Option<String> {
    let o = origin.trim();
    if o != origin
        || o.is_empty()
        || o.eq_ignore_ascii_case("null")
        || !o.is_ascii()
        || o.bytes().any(|b| b.is_ascii_control() || b == b' ')
    {
        return None;
    }
    let (scheme, rest) = o.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    if !matches!(
        scheme.as_str(),
        "http" | "https" | "tauri" | "oaiy" | "oaiyflows"
    ) {
        return None;
    }
    if rest.is_empty() || rest.contains(['/', '?', '#', '@', '\\']) {
        return None;
    }
    Some(format!("{scheme}://{}", rest.to_ascii_lowercase()))
}

/// Whether a paired token may hold `name`: the grant rules of `check_grant`, applied again at use, so
/// that an edited file cannot widen a token. `alone` is whether it holds nothing else.
fn pat_may_hold(name: &str, browser: bool, alone: bool) -> bool {
    if name == "vault.kt" {
        // The ceremony token: browser-bound, and nothing else.
        return browser && alone;
    }
    if never_on_a_token(name) || is_reserved(name) {
        return false;
    }
    !is_dangerous(name) || (!browser && NATIVE_TOKEN_DANGEROUS.contains(&name))
}

/// The rules for what a kind may carry (design 4.2.3), checked at every mint.
fn check_grant(spec: &MintSpec) -> Result<(), MintFailure> {
    let scopes = &spec.scopes;
    if !control_project_needs_read(scopes) {
        return Err(MintFailure::ScopeNotGrantable(
            "control.project always travels with control.read".into(),
        ));
    }
    for name in scopes.names() {
        let core = super::scopes::is_known(&name);
        if !core && !is_resource_scope(&name) {
            return Err(MintFailure::ScopeNotGrantable(format!(
                "{name:?} is not a scope"
            )));
        }
        if !core && spec.kind != Kind::Pat {
            return Err(MintFailure::ScopeNotGrantable(format!(
                "{name} can be held only by a paired token"
            )));
        }
    }
    for origin in &spec.origins {
        if canonical_origin(origin).as_deref() != Some(origin.as_str()) {
            return Err(MintFailure::Invalid(
                "an origin must be scheme://host[:port], lowercase, and never null",
            ));
        }
    }
    let ttl_max = |max: u64| {
        if spec.ttl_ms > max {
            Err(MintFailure::TtlTooLong { max_ms: max })
        } else {
            Ok(())
        }
    };
    if spec.ttl_ms == 0 {
        return Err(MintFailure::Invalid("a credential needs a lifetime"));
    }
    match spec.kind {
        Kind::Dev => Err(MintFailure::Token(MintError::KindNotMintable(Kind::Dev))),
        Kind::Pat => {
            if spec.origins.len() > 1 {
                return Err(MintFailure::Invalid(
                    "a paired token is bound to one origin",
                ));
            }
            let browser = !spec.origins.is_empty();
            let names = scopes.core_names();
            // The vault ceremony token: browser-bound, one scope, five minutes, one use.
            if names.contains(&"vault.kt") {
                if names != ["vault.kt"]
                    || scopes.len() != 1
                    || !browser
                    || spec.max_uses != Some(1)
                {
                    return Err(MintFailure::ScopeNotGrantable(
                        "vault.kt is held alone, by a browser-bound token that is used once".into(),
                    ));
                }
                return ttl_max(CEREMONY_MAX);
            }
            for name in &names {
                if is_reserved(name) {
                    return Err(MintFailure::ScopeNotGrantable(format!(
                        "{name} cannot be held by a token"
                    )));
                }
                if never_on_a_token(name) {
                    return Err(MintFailure::ScopeNotGrantable(format!(
                        "{name} can never be held by a token"
                    )));
                }
                if is_dangerous(name) {
                    if browser {
                        return Err(MintFailure::ScopeNotGrantable(format!(
                            "{name} cannot be held by a token that belongs to a browser"
                        )));
                    }
                    if !NATIVE_TOKEN_DANGEROUS.contains(name) {
                        return Err(MintFailure::ScopeNotGrantable(format!(
                            "{name} cannot be held by a token"
                        )));
                    }
                }
            }
            if browser {
                ttl_max(PAT_BROWSER_MAX)
            } else if scopes.has_dangerous() {
                ttl_max(PAT_DANGEROUS_MAX)
            } else {
                ttl_max(PAT_NATIVE_MAX)
            }
        }
        Kind::Ses => {
            let app = spec
                .app
                .ok_or(MintFailure::Invalid("a session belongs to an app"))?;
            if !scopes.is_subset_of(&app.session_ceiling().scopes()) {
                return Err(MintFailure::ScopeNotGrantable(format!(
                    "a session on the {} host cannot hold more than the {} preset",
                    app.name(),
                    app.session_ceiling().name()
                )));
            }
            if !spec.origins.is_empty() {
                return Err(MintFailure::Invalid(
                    "a session is bound to its host by its cookie, not by an origin",
                ));
            }
            ttl_max(SESSION_MAX)
        }
        Kind::Dsk => {
            let app = spec.app.ok_or(MintFailure::Invalid(
                "a desk credential belongs to a webview",
            ))?;
            if spec.origins.is_empty() {
                return Err(MintFailure::Invalid(
                    "a desk credential is bound to its webview's origins",
                ));
            }
            if !scopes.is_subset_of(&app.desk_ceiling().scopes()) {
                return Err(MintFailure::ScopeNotGrantable(format!(
                    "a {} webview cannot hold more than the {} preset",
                    app.name(),
                    app.desk_ceiling().name()
                )));
            }
            ttl_max(DESK_MAX)
        }
        Kind::Run => {
            for name in scopes.names() {
                if is_dangerous(&name) || is_auth_scope(&name) || is_reserved(&name) {
                    return Err(MintFailure::ScopeNotGrantable(format!(
                        "{name} cannot be held by a per-run or derived credential"
                    )));
                }
            }
            ttl_max(DAY)
        }
        Kind::Con => {
            if !spec.origins.is_empty() {
                return Err(MintFailure::Invalid("the console credential has no origin"));
            }
            ttl_max(DESK_MAX)
        }
    }
}

// ---- the store --------------------------------------------------------------------------------

/// Which process is opening the store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Host {
    /// `oaiy-server`: a file it cannot understand stops it.
    Server,
    /// The desktop: a file from a newer OAIY leaves it with a memory-only store and a banner.
    Gui,
}

/// `ok`, `low` or `full`: what `GET /api/health` says of the store's disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    Ok,
    Low,
    Full,
}

impl Storage {
    pub fn name(self) -> &'static str {
        match self {
            Storage::Ok => "ok",
            Storage::Low => "low",
            Storage::Full => "full",
        }
    }
}

/// What `owner.json` says that the store needs. The rest of the file is the login's (kept as read).
#[derive(Clone, Debug)]
pub struct OwnerFile {
    pub min_session_epoch: u64,
    pub doc: Value,
}

struct Inner {
    /// `<data>/auth`; none for a memory-only store.
    dir: Option<PathBuf>,
    /// Never write (the desktop met a newer file).
    read_only: bool,
    records: HashMap<String, Record>,
    /// Hash of the whole legacy token -> id.
    legacy: HashMap<String, String>,
    /// Top-level fields of the file this build does not know, written back.
    file_extra: Map<String, Value>,
    /// Records of a shape this build cannot read (a `kind` or an `app` a newer build made): kept as they
    /// were written and written back with the rest, never looked up and never honoured (design 4.1 rule 4).
    unknown: Vec<Value>,
    /// The ids of the records made under each parent: what the cap on children and the sessions' rotation
    /// look at, instead of every record there is.
    children: HashMap<String, HashSet<String>>,
    /// When each `run` credential ended (its expiry, and again its revocation), earliest first: what
    /// [`Inner::purge_runs`] pops. An entry the record has outlived is dropped when it comes up.
    run_ends: BinaryHeap<Reverse<(u64, String)>>,
    owner: Option<OwnerFile>,
    min_epoch: u64,
    static_present: bool,
    /// When each session's elevation ends, on the elevation clock (the store's own, or a monotonic one: see
    /// [`AuthStore::use_monotonic_elevation`]).
    elevated: HashMap<String, u64>,
    derive_times: HashMap<String, VecDeque<u64>>,
    dirty_touch: bool,
    last_flush_ms: u64,
    last_purge_ms: u64,
    storage: Storage,
    notices: Vec<String>,
}

/// A source of random bytes for the tokens the store makes; the operating system's unless a test says.
pub type Random = Arc<dyn Fn(&mut [u8]) -> Result<(), MintError> + Send + Sync>;

fn default_random() -> Random {
    Arc::new(token::os_random)
}

/// The store. Cheap to share: every method takes `&self`.
pub struct AuthStore {
    inner: Mutex<Inner>,
    clock: Arc<dyn Clock>,
    /// The clock an elevation window is measured on, when it is not the store's: a monotonic one, so that a jump of the
    /// wall clock can neither extend a window nor end it (design 6, "clock skew").
    elevation_clock: Mutex<Option<Arc<dyn Clock>>>,
    writer: Arc<dyn FileWriter>,
    random: Mutex<Random>,
    _lock: Option<AuthLock>,
}

/// Read a file: `Ok(None)` for "not there", and every other failure an error naming the file.
fn read_optional(path: &Path) -> Result<Option<Vec<u8>>, StoreError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(StoreError::Unreadable {
            file: path.to_path_buf(),
            source,
        }),
    }
}

/// Read `<dir>/owner.json` as the store does at open: `Ok(None)` for a file that is not there, and an error naming it
/// for every other thing (`auth::exposure::inspect_owner_file` asks the same, so that the start and `check` agree).
pub(crate) fn read_owner(dir: &Path) -> Result<Option<OwnerFile>, StoreError> {
    let path = dir.join("owner.json");
    let Some(bytes) = read_optional(&path)? else {
        return Ok(None);
    };
    let unparsable = |detail: String| StoreError::OwnerUnparsable {
        file: path.clone(),
        detail,
    };
    let doc: Value = serde_json::from_slice(&bytes).map_err(|e| unparsable(e.to_string()))?;
    let v = doc
        .get("v")
        .and_then(Value::as_u64)
        .ok_or_else(|| unparsable("no version".into()))?;
    if v != FILE_VERSION {
        return Err(StoreError::UnknownVersion {
            file: path,
            found: v,
            supported: FILE_VERSION,
        });
    }
    let min_session_epoch = doc
        .get("min_session_epoch")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(Some(OwnerFile {
        min_session_epoch,
        doc,
    }))
}

impl AuthStore {
    /// A store that touches no disk: no folder, no lock, no reads, no writes. What the `legacy`
    /// access mode uses, so that mode changes nothing on the owner's machine.
    pub fn memory(clock: Arc<dyn Clock>) -> AuthStore {
        AuthStore {
            inner: Mutex::new(Inner::new(None)),
            clock,
            elevation_clock: Mutex::new(None),
            writer: Arc::new(SecureWriter),
            random: Mutex::new(default_random()),
            _lock: None,
        }
    }

    /// Open `<data>/auth` (`dir`): lock it, read `owner.json` and `credentials.json` by the rules of
    /// the module documentation, drop what expired long ago.
    pub fn open(
        dir: &Path,
        host: Host,
        clock: Arc<dyn Clock>,
        writer: Arc<dyn FileWriter>,
        audit: Option<Arc<AuditLog>>,
    ) -> Result<AuthStore, StoreError> {
        let lock = AuthLock::acquire(dir).map_err(StoreError::Lock)?;
        let mut inner = Inner::new(Some(dir.to_path_buf()));
        let now = clock.now_ms();
        inner.owner = read_owner(dir)?;
        inner.min_epoch = inner.owner.as_ref().map_or(0, |o| o.min_session_epoch);
        let path = dir.join("credentials.json");
        if let Some(bytes) = read_optional(&path)? {
            match parse_credentials(&bytes) {
                Ok((records, unknown, extra)) => {
                    inner.file_extra = extra;
                    if !unknown.is_empty() {
                        log::warn!(
                            "auth: {} holds {} credential(s) of a kind this build does not know: they are kept in the file and not used",
                            path.display(),
                            unknown.len()
                        );
                        inner.notices.push(format!(
                            "{} credential(s) were made by a newer OAIY: they are kept in the file and are not used by this version",
                            unknown.len()
                        ));
                    }
                    inner.unknown = unknown;
                    for r in records {
                        inner.insert_loaded(r);
                    }
                }
                Err(Unparsable::UnknownVersion(found)) => match host {
                    Host::Server => {
                        return Err(StoreError::UnknownVersion {
                            file: path,
                            found,
                            supported: FILE_VERSION,
                        })
                    }
                    Host::Gui => {
                        inner.read_only = true;
                        inner.notices.push("credentials were made by a newer OAIY; paired apps will not work until you update".into());
                        log::warn!("auth: {} was written by a newer OAIY (version {found}); starting with an empty store and leaving the file alone", path.display());
                    }
                },
                Err(Unparsable::Corrupt(detail)) => {
                    let aside = dir.join(format!("credentials.json.corrupt-{now}"));
                    let moved = std::fs::rename(&path, &aside);
                    log::error!(
                        "auth: {} cannot be read ({detail}); {}; starting with an empty store",
                        path.display(),
                        match &moved {
                            Ok(()) => format!("moved to {}", aside.display()),
                            Err(e) => format!("and could not be moved aside: {e}"),
                        }
                    );
                    inner.notices.push(format!("credentials.json could not be read and was moved aside; paired apps must pair again ({detail})"));
                    if let Some(audit) = &audit {
                        audit.critical("credentials.corrupt", None, &AuditContext::default(), serde_json::json!({ "detail": detail, "moved_to": moved.is_ok().then(|| aside.display().to_string()) }));
                    }
                }
            }
        }
        inner.last_flush_ms = now;
        let store = AuthStore {
            inner: Mutex::new(inner),
            clock,
            elevation_clock: Mutex::new(None),
            writer,
            random: Mutex::new(default_random()),
            _lock: Some(lock),
        };
        store.maintain();
        Ok(store)
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Replace the randomness (the tests make it fail).
    pub fn set_random(&self, random: Random) {
        *self.random.lock().unwrap_or_else(|e| e.into_inner()) = random;
    }

    /// Whether the operator's environment token is configured: a derived credential whose parent is
    /// the static token is valid only while it is.
    pub fn set_static_present(&self, present: bool) {
        self.lock().static_present = present;
    }

    /// Whether this store reads and writes a folder.
    pub fn is_persistent(&self) -> bool {
        let g = self.lock();
        g.dir.is_some() && !g.read_only
    }

    pub fn storage(&self) -> Storage {
        self.lock().storage
    }

    /// Banners the dashboard should show (a newer file, a corrupt file moved aside).
    pub fn notices(&self) -> Vec<String> {
        self.lock().notices.clone()
    }

    pub fn owner(&self) -> Option<OwnerFile> {
        self.lock().owner.clone()
    }

    pub fn min_session_epoch(&self) -> u64 {
        self.lock().min_epoch
    }

    /// End every session made before `epoch` (in memory; `owner.json` is the login's to write).
    pub fn set_min_session_epoch(&self, epoch: u64) {
        self.lock().min_epoch = epoch;
    }

    /// The record with this id (for lists and tests). Never the secret: there is none.
    pub fn record(&self, id: &str) -> Option<Record> {
        self.lock().records.get(id).cloned()
    }

    /// Put a record in as it would come from a file: the tests build the odd ones.
    #[cfg(test)]
    pub(crate) fn insert_for_tests(&self, r: Record) {
        self.lock().insert_loaded(r);
    }

    /// How many credentials of `kind` the store holds in memory, alive or not: what upkeep keeps small.
    pub fn held_count(&self, kind: Kind) -> usize {
        self.lock()
            .records
            .values()
            .filter(|r| r.kind == kind)
            .count()
    }

    /// How many ids the by-parent index holds (the tests hold it to the records there are).
    #[cfg(test)]
    pub(crate) fn indexed_children(&self) -> usize {
        self.lock().children.values().map(HashSet::len).sum()
    }

    /// How many credentials of `kind` are alive (not revoked, not expired).
    pub fn live_count(&self, kind: Kind) -> usize {
        let now = self.clock.now_ms();
        self.lock()
            .records
            .values()
            .filter(|r| r.kind == kind && r.alive(now))
            .count()
    }

    /// The origins of every live credential that has one: the CORS allow-list.
    pub fn allowed_origins(&self) -> BTreeSet<String> {
        let now = self.clock.now_ms();
        self.lock()
            .records
            .values()
            .filter(|r| r.alive(now))
            .flat_map(|r| r.origins())
            .collect()
    }

    // ---- making credentials --------------------------------------------------------------

    /// Make a credential of `spec.kind`. The secret is in the answer once and nowhere else.
    pub fn mint(&self, spec: MintSpec) -> Result<Minted, MintFailure> {
        check_grant(&spec)?;
        let now = self.clock.now_ms();
        let expires_ms = now.saturating_add(spec.ttl_ms);
        let mut g = self.lock();
        let persisted = persisted_kind(spec.kind);
        // What ended a while ago goes first, so that what follows looks at what is there and not at
        // everything there has been.
        g.purge_runs(now);

        if persisted
            && g.records
                .values()
                .filter(|r| r.on_disk() && r.alive(now))
                .count()
                >= MAX_PERSISTED
        {
            return Err(MintFailure::TooManyCredentials);
        }
        if let Some(parent) = &spec.parent {
            g.check_parent(parent, now)?;
            let children = g.live_children(parent, now);
            // A session's rotation frees its slot below; count only what would remain.
            let rotating = spec.kind == Kind::Ses
                && g.children.get(parent.as_str()).is_some_and(|ids| {
                    ids.iter()
                        .filter_map(|id| g.records.get(id))
                        .any(|r| r.kind == Kind::Ses && r.app == spec.app && r.alive(now))
                });
            if children - usize::from(rotating) >= MAX_CHILDREN {
                return Err(MintFailure::TooManyChildren);
            }
        }
        if spec.kind == Kind::Ses {
            // One child session per (parent, app): a new handoff rotates and revokes the previous one.
            if let Some(parent) = &spec.parent {
                let previous: Vec<String> = g
                    .children
                    .get(parent.as_str())
                    .into_iter()
                    .flatten()
                    .filter_map(|id| g.records.get(id))
                    .filter(|r| r.kind == Kind::Ses && r.app == spec.app && r.alive(now))
                    .map(|r| r.id.clone())
                    .collect();
                for id in previous {
                    g.revoke_in_memory(&id, "revoked", now);
                }
            }
            // At most 32 live sessions: the one used longest ago makes room.
            while g
                .records
                .values()
                .filter(|r| r.kind == Kind::Ses && r.alive(now))
                .count()
                >= MAX_SESSIONS
            {
                let oldest = g
                    .records
                    .values()
                    .filter(|r| r.kind == Kind::Ses && r.alive(now))
                    .min_by_key(|r| (r.last_used_ms.unwrap_or(r.created_ms), r.created_ms))
                    .map(|r| r.id.clone());
                match oldest {
                    Some(id) => g.revoke_in_memory(&id, "revoked", now),
                    None => break,
                }
            }
        }

        let random = self
            .random
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let mut new = None;
        for _ in 0..16 {
            let candidate = token::mint_with(spec.kind, &mut |buf: &mut [u8]| random(buf))?;
            if !g.records.contains_key(&candidate.id) {
                new = Some(candidate);
                break;
            }
        }
        let new = new.ok_or(MintFailure::Token(MintError::NoRandomness))?;

        let mut origins = spec.origins.iter();
        let record = Record {
            id: new.id.clone(),
            kind: spec.kind,
            hash: new.hash.clone(),
            label: spec.label.clone(),
            product: spec.product.clone(),
            preset: spec.preset.map(|p| p.name().to_string()),
            preset_ver: spec.preset.map(|_| Preset::VERSION),
            scopes: spec.scopes.clone(),
            origin: origins.next().cloned(),
            origin_alt: origins.cloned().collect(),
            app: spec.app,
            parent: spec.parent.clone(),
            created_ms: now,
            expires_ms,
            idle_ms: match spec.kind {
                Kind::Ses => Some(spec.idle_ms.unwrap_or(DEFAULT_SESSION_IDLE_MS)),
                _ => spec.idle_ms,
            },
            last_used_ms: None,
            last_used_ip: None,
            revoked_ms: None,
            revoked_reason: None,
            legacy: false,
            cnf: None,
            max_uses: spec.max_uses,
            uses: 0,
            created_by: spec.created_by.clone(),
            epoch: if spec.kind == Kind::Ses {
                g.min_epoch
            } else {
                0
            },
            memory_only: false,
            extra: spec.extra.clone(),
        };
        let id = record.id.clone();
        g.insert(record);
        if persisted {
            if let Err(e) = g.flush(self.writer.as_ref(), now) {
                if spec.kind == Kind::Ses {
                    // Sign-in never needs a write: the session lives in memory and says so.
                    if let Some(r) = g.records.get_mut(&id) {
                        r.memory_only = true;
                    }
                    log::warn!(
                        "auth: a session could not be written and lives in memory only: {e}"
                    );
                } else {
                    g.remove(&id);
                    return Err(MintFailure::StoreUnavailable(e));
                }
            }
        }
        Ok(Minted {
            token: new.token,
            id,
            expires_ms,
            scopes: spec.scopes,
        })
    }

    /// Derive a credential from the caller's own (design 4.6): only from a `desk`, `pat` or `static`
    /// credential, never from a session or from a derived credential; at most what the parent holds,
    /// no dangerous scope and no `auth.*`; a life of at most an hour by default, a day at most and never
    /// beyond the parent's own end; 64 live per parent and 30 a minute. The child is a memory-only `run`
    /// credential bound to the parent's origins and dies with its parent.
    pub fn derive(&self, parent: &Principal, req: DeriveRequest) -> Result<Minted, DeriveError> {
        if !matches!(
            parent.kind,
            PrincipalKind::Desk | PrincipalKind::Pat | PrincipalKind::Static
        ) {
            return Err(DeriveError::Refused(
                "only a desk, paired or static credential can derive one",
            ));
        }
        if req.scopes.is_empty() {
            return Err(DeriveError::Mint(MintFailure::Invalid(
                "a derived credential needs at least one scope",
            )));
        }
        if !req.scopes.is_subset_of(&parent.scopes) {
            return Err(DeriveError::Refused(
                "a derived credential cannot hold more than its parent",
            ));
        }
        for name in req.scopes.names() {
            if is_dangerous(&name)
                || is_auth_scope(&name)
                || is_reserved(&name)
                || is_resource_scope(&name)
            {
                return Err(DeriveError::Refused(
                    "a derived credential holds no dangerous, auth.* or connector scope",
                ));
            }
        }
        let now = self.clock.now_ms();
        let mut ttl = req
            .ttl_ms
            .unwrap_or(DEFAULT_DERIVE_TTL_MS)
            .min(MAX_DERIVE_TTL_MS);
        if let Some(end) = parent.expires_ms {
            ttl = ttl.min(end.saturating_sub(now));
        }
        if ttl == 0 {
            return Err(DeriveError::Refused(
                "the parent credential has no life left",
            ));
        }
        {
            let mut g = self.lock();
            let times = g.derive_times.entry(parent.id.clone()).or_default();
            while times
                .front()
                .is_some_and(|t| now.saturating_sub(*t) >= MINUTE)
            {
                times.pop_front();
            }
            if times.len() >= DERIVE_PER_MINUTE {
                let wait = times.front().map_or(1, |t| {
                    (MINUTE - now.saturating_sub(*t)).div_ceil(1000).max(1)
                });
                return Err(DeriveError::RateLimited {
                    retry_after_s: wait,
                });
            }
            times.push_back(now);
        }
        let mut spec = MintSpec::new(Kind::Run, &req.label, req.scopes, ttl);
        spec.parent = Some(parent.id.clone());
        spec.origins = parent.origins.clone();
        spec.app = parent.app;
        spec.created_by = Some(parent.id.clone());
        self.mint(spec).map_err(DeriveError::Mint)
    }

    // ---- using them ----------------------------------------------------------------------

    /// Who a presented token is, if it is valid: known, the right kind, not revoked, not expired, not
    /// idle, and every ancestor the same. An unknown id and a wrong secret are the same answer and take
    /// the same work. `ip` is recorded as where it was last used from.
    pub fn authenticate(&self, presented: &str, ip: Option<&str>) -> Result<Principal, AuthError> {
        let now = self.clock.now_ms();
        let mut g = self.lock();
        let (id, kind, hash, legacy) = if let Some(p) = token::parse(presented) {
            (
                Some(p.id.to_string()),
                Some(p.kind),
                token::secret_hash(p.secret),
                false,
            )
        } else if token::is_legacy_shape(presented) {
            let hash = token::legacy_hash(presented);
            (g.legacy.get(&hash).cloned(), Some(Kind::Pat), hash, true)
        } else {
            return Err(AuthError::Invalid);
        };
        let record = id.as_deref().and_then(|id| g.records.get(id));
        let expected = record.map_or(DUMMY_HASH, |r| r.hash.as_str());
        let hash_ok = token::hashes_equal(&hash, expected);
        let record = match record {
            Some(r)
                if hash_ok && Some(r.kind) == kind && r.legacy == legacy && r.kind != Kind::Dev =>
            {
                r.clone()
            }
            _ => return Err(AuthError::Invalid),
        };
        let ctx = g.chain_ctx(now);
        if let Err((failed, broken)) = chain::check_chain(&g.records, &record, &ctx) {
            return Err(auth_error(&record, &failed, broken));
        }
        // Used: this credential now, and the sessions above it (a child in use keeps its parent from
        // idling out; absolute times are never extended).
        let ancestors = chain::ancestors(&g.records, &record);
        if let Some(r) = g.records.get_mut(&record.id) {
            r.last_used_ms = Some(now);
            if let Some(ip) = ip {
                r.last_used_ip = Some(ip.chars().take(64).collect());
            }
        }
        for a in ancestors {
            if let Some(r) = g.records.get_mut(&a) {
                if r.kind == Kind::Ses && now.saturating_sub(r.last_used_ms.unwrap_or(0)) >= MINUTE
                {
                    r.last_used_ms = Some(now);
                }
            }
        }
        g.dirty_touch = true;
        let rec = g.records.get(&record.id).cloned().unwrap_or(record);
        let elevation_now = self.elevation_now(now);
        Ok(g.principal_for(&rec, elevation_now))
    }

    /// Count one use of a credential made with `max_uses` (the vault ceremony token) and revoke it at
    /// the limit. `false` when there is nothing left to spend.
    pub fn spend(&self, id: &str) -> bool {
        let now = self.clock.now_ms();
        let mut g = self.lock();
        let Some(r) = g.records.get_mut(id) else {
            return false;
        };
        if r.revoked_ms.is_some() {
            return false;
        }
        r.uses = r.uses.saturating_add(1);
        let exhausted = r.max_uses.is_some_and(|m| r.uses >= m);
        let over = r.max_uses.is_some_and(|m| r.uses > m);
        if exhausted {
            g.revoke_in_memory(id, "revoked", now);
        }
        g.persist_quietly(self.writer.as_ref(), now);
        !over
    }

    /// Revoke a credential. Its children end with it (their chain is checked at every use). A
    /// revocation is written at once, and never resurrected by a later flush.
    pub fn revoke(&self, id: &str, reason: &str) -> bool {
        let now = self.clock.now_ms();
        let mut g = self.lock();
        if !g.records.get(id).is_some_and(|r| r.revoked_ms.is_none()) {
            return false;
        }
        g.revoke_in_memory(id, reason, now);
        g.persist_quietly(self.writer.as_ref(), now);
        true
    }

    /// Revoke, with `reason`, every credential that is not revoked yet and for which `keep_out` says so: one
    /// write for all of them, and the ids that were revoked. What a password change, a logout everywhere and
    /// the console do, so that no later flush can bring one back.
    pub fn revoke_where(&self, reason: &str, keep_out: &dyn Fn(&Record) -> bool) -> Vec<String> {
        let now = self.clock.now_ms();
        let mut g = self.lock();
        let ids: Vec<String> = g
            .records
            .values()
            .filter(|r| r.revoked_ms.is_none() && keep_out(r))
            .map(|r| r.id.clone())
            .collect();
        for id in &ids {
            g.revoke_in_memory(id, reason, now);
        }
        if !ids.is_empty() {
            g.persist_quietly(self.writer.as_ref(), now);
        }
        ids
    }

    /// Measure elevation windows on `clock` from now on (a monotonic clock): what `set_elevated_until` and
    /// `elevated_until` say is a time on it. Without this the store's own clock is used. Windows already given are
    /// forgotten (they were on another clock); call it before any is.
    pub fn use_monotonic_elevation(&self, clock: Arc<dyn Clock>) {
        *self
            .elevation_clock
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(clock);
        self.lock().elevated.clear();
    }

    /// The time on the clock elevation windows are measured on.
    pub fn elevation_now_ms(&self) -> u64 {
        self.elevation_now(self.clock.now_ms())
    }

    fn elevation_now(&self, wall_now: u64) -> u64 {
        match &*self
            .elevation_clock
            .lock()
            .unwrap_or_else(|e| e.into_inner())
        {
            Some(clock) => clock.now_ms(),
            None => wall_now,
        }
    }

    /// Give a session an elevation until `until_ms` on the elevation clock (in memory only; a restart drops it).
    pub fn set_elevated_until(&self, id: &str, until_ms: u64) {
        self.lock().elevated.insert(id.to_string(), until_ms);
    }

    /// When a session's elevation ends on the elevation clock, if it has one (the time it was given, past or not).
    pub fn elevated_until(&self, id: &str) -> Option<u64> {
        self.lock().elevated.get(id).copied()
    }

    /// A copy of every record (for lists and for the console's counts). Never the secret: there is none.
    pub fn records(&self) -> Vec<Record> {
        self.lock().records.values().cloned().collect()
    }

    // ---- keeping it -----------------------------------------------------------------------

    /// Write the file now (at shutdown, before the process exits).
    pub fn flush(&self) -> io::Result<()> {
        let now = self.clock.now_ms();
        self.lock().flush(self.writer.as_ref(), now)
    }

    /// Periodic upkeep: clamp last uses in the future, write the last-used times if a minute has passed,
    /// and drop what expired more than seven days ago (hourly).
    pub fn maintain(&self) {
        let now = self.clock.now_ms();
        let elevation_now = self.elevation_now(now);
        let mut g = self.lock();
        g.clamp_future_use(now);
        // Derived credentials are memory-only: nothing to write when they go.
        g.purge_runs(now);
        let mut changed = false;
        if now.saturating_sub(g.last_purge_ms) >= PURGE_EVERY_MS || g.last_purge_ms == 0 {
            changed = g.purge(now, elevation_now);
            g.last_purge_ms = now;
        }
        if changed || (g.dirty_touch && now.saturating_sub(g.last_flush_ms) >= FLUSH_EVERY_MS) {
            g.persist_quietly(self.writer.as_ref(), now);
        }
    }
}

fn auth_error(record: &Record, failed: &str, broken: Broken) -> AuthError {
    let own = failed == record.id;
    let session = record.kind == Kind::Ses;
    match (broken, own) {
        (Broken::Unusable, _) => AuthError::Invalid,
        (Broken::Revoked { reason }, true) if session => AuthError::SessionEnded {
            reason: revoke_reason(reason.as_deref()),
        },
        (Broken::Revoked { reason }, true) => AuthError::Revoked {
            reason: revoke_reason(reason.as_deref()),
        },
        (Broken::Expired, true) if session => AuthError::SessionEnded { reason: "absolute" },
        (Broken::Expired, true) => AuthError::Expired,
        (Broken::Idle, true) => AuthError::SessionEnded { reason: "idle" },
        (Broken::Epoch, _) => AuthError::SessionEnded { reason: "upgrade" },
        // An ancestor ended (or is gone).
        (_, _) if session => AuthError::SessionEnded {
            reason: "parent_ended",
        },
        (_, _) => AuthError::Revoked {
            reason: "parent_ended",
        },
    }
}

impl Inner {
    fn new(dir: Option<PathBuf>) -> Inner {
        Inner {
            dir,
            read_only: false,
            records: HashMap::new(),
            legacy: HashMap::new(),
            file_extra: Map::new(),
            unknown: Vec::new(),
            children: HashMap::new(),
            run_ends: BinaryHeap::new(),
            owner: None,
            min_epoch: 0,
            static_present: false,
            elevated: HashMap::new(),
            derive_times: HashMap::new(),
            dirty_touch: false,
            last_flush_ms: 0,
            last_purge_ms: 0,
            storage: Storage::Ok,
            notices: Vec::new(),
        }
    }

    fn insert_loaded(&mut self, r: Record) {
        if r.legacy {
            self.legacy.insert(r.hash.clone(), r.id.clone());
        }
        self.insert(r);
    }

    /// Put a record in, and in the indexes that find it by its parent and by when it ends.
    fn insert(&mut self, r: Record) {
        if let Some(parent) = &r.parent {
            self.children
                .entry(parent.clone())
                .or_default()
                .insert(r.id.clone());
        }
        if r.kind == Kind::Run {
            self.run_ends.push(Reverse((r.expires_ms, r.id.clone())));
        }
        self.records.insert(r.id.clone(), r);
    }

    /// Take a record out of the store and the indexes.
    fn remove(&mut self, id: &str) {
        if let Some(r) = self.records.remove(id) {
            if let Some(ids) = r.parent.as_ref().and_then(|p| self.children.get_mut(p)) {
                ids.remove(id);
            }
        }
    }

    /// How many of the records made under `parent` are alive.
    fn live_children(&self, parent: &str, now: u64) -> usize {
        self.children.get(parent).map_or(0, |ids| {
            ids.iter()
                .filter(|id| self.records.get(*id).is_some_and(|r| r.alive(now)))
                .count()
        })
    }

    /// Drop the `run` credentials that ended more than [`RUN_PURGE_AFTER_MS`] ago (expired, or revoked): the
    /// ones at the front of the queue and only those, so what a derive costs does not grow with what has
    /// been derived. Whether anything went.
    fn purge_runs(&mut self, now: u64) -> bool {
        let mut removed = false;
        while let Some(Reverse((ended, id))) = self.run_ends.peek().cloned() {
            if ended.saturating_add(RUN_PURGE_AFTER_MS) > now {
                break;
            }
            self.run_ends.pop();
            // The entry may be old: the record may be gone, or have ended earlier (revoked) and been queued
            // again for then. What counts is when it ended by now.
            let gone = self.records.get(&id).is_some_and(|r| {
                r.kind == Kind::Run
                    && r.revoked_ms
                        .into_iter()
                        .chain(std::iter::once(r.expires_ms))
                        .min()
                        .unwrap_or(0)
                        .saturating_add(RUN_PURGE_AFTER_MS)
                        <= now
            });
            if gone {
                self.remove(&id);
                self.derive_times.remove(&id);
                self.elevated.remove(&id);
                removed = true;
            }
        }
        removed
    }

    fn chain_ctx(&self, now: u64) -> chain::Ctx {
        chain::Ctx {
            now_ms: now,
            min_epoch: self.min_epoch,
            static_present: self.static_present,
        }
    }

    /// A parent for a new child must be valid now.
    fn check_parent(&self, parent: &str, now: u64) -> Result<(), MintFailure> {
        if parent == STATIC_PARENT {
            return if self.static_present {
                Ok(())
            } else {
                Err(MintFailure::ParentInvalid)
            };
        }
        let rec = self.records.get(parent).ok_or(MintFailure::ParentInvalid)?;
        chain::check_chain(&self.records, rec, &self.chain_ctx(now))
            .map_err(|_| MintFailure::ParentInvalid)
    }

    fn revoke_in_memory(&mut self, id: &str, reason: &str, now: u64) {
        if let Some(r) = self.records.get_mut(id) {
            if r.revoked_ms.is_none() {
                r.revoked_ms = Some(now);
                r.revoked_reason = Some(reason.to_string());
                if r.kind == Kind::Run {
                    // Ended now, not at its expiry: queued again for then.
                    self.run_ends.push(Reverse((now, id.to_string())));
                }
            }
        }
        self.elevated.remove(id);
    }

    /// The effective scopes of a record (design 4.1 rule 5): sessions and desk credentials from the
    /// running binary's tables, clamped to the app's ceiling and never adding a dangerous scope they
    /// were not made with; a paired token exactly as approved; the rest as made.
    fn effective_scopes(&self, rec: &Record) -> ScopeSet {
        match rec.kind {
            Kind::Ses | Kind::Dsk => {
                let ceiling = match (rec.kind, rec.app) {
                    (Kind::Ses, Some(app)) => app.session_ceiling().scopes(),
                    (_, Some(app)) => app.desk_ceiling().scopes(),
                    // No app: nothing to recompute against, so nothing can be added.
                    (_, None) => rec.scopes.clone(),
                };
                let base = rec
                    .preset
                    .as_deref()
                    .and_then(Preset::by_name)
                    .map_or_else(|| rec.scopes.clone(), |p| p.scopes());
                base.intersection(&ceiling)
                    .filtered(|n| !is_dangerous(n) || rec.scopes.contains(n))
            }
            // Defence in depth against an edited file: a token never holds what the grant rules refuse.
            Kind::Pat => {
                let browser = rec.origin.is_some();
                rec.scopes
                    .filtered(|n| pat_may_hold(n, browser, rec.scopes.len() == 1))
            }
            Kind::Run => rec
                .scopes
                .filtered(|n| !is_dangerous(n) && !is_auth_scope(n) && !is_reserved(n)),
            Kind::Con | Kind::Dev => rec.scopes.clone(),
        }
    }

    fn principal_for(&self, rec: &Record, elevation_now: u64) -> Principal {
        let kind = match rec.kind {
            Kind::Pat => PrincipalKind::Pat,
            Kind::Ses => PrincipalKind::Session,
            Kind::Dsk => PrincipalKind::Desk,
            Kind::Run | Kind::Dev => PrincipalKind::Run,
            Kind::Con => PrincipalKind::Console,
        };
        let elevated = match rec.kind {
            // The OS user at the machine is the owner: the dashboard's desk credential is always elevated.
            Kind::Dsk => rec.app == Some(App::Dash),
            Kind::Con => true,
            Kind::Ses => self
                .elevated
                .get(&rec.id)
                .is_some_and(|until| *until > elevation_now),
            _ => false,
        };
        Principal {
            id: rec.id.clone(),
            kind,
            label: rec.label.clone(),
            scopes: self.effective_scopes(rec),
            origins: rec.origins(),
            app: rec.app,
            elevated,
            chain: chain::ancestors(&self.records, rec),
            persisted: rec.on_disk() && self.dir.is_some() && !self.read_only,
            expires_ms: Some(rec.expires_ms),
            preset: rec.preset.clone(),
            legacy_import: rec.legacy,
        }
    }

    /// A last use in the future (the clock went back) is set to now, so the idle timeout runs from here.
    fn clamp_future_use(&mut self, now: u64) {
        for r in self.records.values_mut() {
            if r.last_used_ms.is_some_and(|t| t > now) {
                r.last_used_ms = Some(now);
                self.dirty_touch = true;
            }
        }
    }

    /// Drop what ended more than seven days ago. Whether anything went.
    fn purge(&mut self, now: u64, elevation_now: u64) -> bool {
        let before = self.records.len();
        self.records.retain(|_, r| {
            let ended = r
                .revoked_ms
                .into_iter()
                .chain(std::iter::once(r.expires_ms))
                .min()
                .unwrap_or(0);
            now.saturating_sub(ended) < PURGE_AFTER_MS
        });
        self.legacy.retain(|_, id| self.records.contains_key(id));
        self.children.retain(|_, ids| {
            ids.retain(|id| self.records.contains_key(id));
            !ids.is_empty()
        });
        self.derive_times
            .retain(|id, _| self.records.contains_key(id) || id == STATIC_PARENT);
        self.elevated
            .retain(|id, until| self.records.contains_key(id) && *until > elevation_now);
        self.records.len() != before
    }

    /// Write the file if there is a folder to write to. Memory-only kinds and sessions that could not
    /// be written are left out.
    fn flush(&mut self, writer: &dyn FileWriter, now: u64) -> io::Result<()> {
        let Some(dir) = self.dir.clone() else {
            return Ok(());
        };
        if self.read_only {
            return Ok(());
        }
        // Sessions that lived in memory only because an earlier write failed are offered to this one.
        let retrying: Vec<String> = self
            .records
            .values()
            .filter(|r| r.memory_only && r.persisted_kind())
            .map(|r| r.id.clone())
            .collect();
        for id in &retrying {
            if let Some(r) = self.records.get_mut(id) {
                r.memory_only = false;
            }
        }
        let text = {
            let mut rows: Vec<&Record> = self.records.values().filter(|r| r.on_disk()).collect();
            rows.sort_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)));
            let mut doc = self.file_extra.clone();
            doc.insert("v".into(), Value::from(FILE_VERSION));
            let mut written = serde_json::to_value(&rows).map_err(io::Error::other)?;
            if let Value::Array(list) = &mut written {
                // What a newer build wrote and this one cannot read goes back as it came.
                list.extend(self.unknown.iter().cloned());
            }
            doc.insert("credentials".into(), written);
            let mut text =
                serde_json::to_string_pretty(&Value::Object(doc)).map_err(io::Error::other)?;
            text.push('\n');
            text
        };
        match writer.write(&dir.join("credentials.json"), text.as_bytes()) {
            Ok(()) => {
                self.dirty_touch = false;
                self.last_flush_ms = now;
                self.storage = match fs2::available_space(&dir) {
                    Ok(free) if free < LOW_SPACE_BYTES => Storage::Low,
                    _ => Storage::Ok,
                };
                Ok(())
            }
            Err(e) => {
                for id in &retrying {
                    if let Some(r) = self.records.get_mut(id) {
                        r.memory_only = true;
                    }
                }
                self.storage = Storage::Full;
                Err(e)
            }
        }
    }
    /// [`Inner::flush`], with a failure logged and nothing else: for the writes that follow a change
    /// already made in memory (memory is authoritative).
    fn persist_quietly(&mut self, writer: &dyn FileWriter, now: u64) {
        if let Err(e) = self.flush(writer, now) {
            log::warn!("auth: the credential store could not be written: {e}");
        }
    }
}

/// Why `credentials.json` was not used.
enum Unparsable {
    UnknownVersion(u64),
    Corrupt(String),
}

/// Whether `row` is a record of a newer build: sound in every way this build can check, except that its
/// `kind` or its `app` is a name this build does not have. (Such a record is kept, not honoured; a record
/// that is unsound in any other way makes the file unparsable, as before.)
fn newer_shape(row: &Value) -> bool {
    let Value::Object(fields) = row else {
        return false;
    };
    let mut probe = fields.clone();
    let mut replaced = false;
    if let Some(Value::String(kind)) = fields.get("kind") {
        if serde_json::from_value::<Kind>(Value::String(kind.clone())).is_err() {
            probe.insert("kind".into(), Value::from("pat"));
            replaced = true;
        }
    }
    if let Some(Value::String(app)) = fields.get("app") {
        if serde_json::from_value::<App>(Value::String(app.clone())).is_err() {
            probe.insert("app".into(), Value::Null);
            replaced = true;
        }
    }
    replaced && serde_json::from_value::<Record>(Value::Object(probe)).is_ok()
}

/// The records this build reads, the ones it keeps without reading (a newer build's), and the top-level
/// fields it does not know.
#[allow(clippy::type_complexity)]
fn parse_credentials(
    bytes: &[u8],
) -> Result<(Vec<Record>, Vec<Value>, Map<String, Value>), Unparsable> {
    let doc: Value =
        serde_json::from_slice(bytes).map_err(|e| Unparsable::Corrupt(e.to_string()))?;
    let Value::Object(mut doc) = doc else {
        return Err(Unparsable::Corrupt("not an object".into()));
    };
    let v = doc
        .remove("v")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Unparsable::Corrupt("no version".into()))?;
    if v != FILE_VERSION {
        return Err(Unparsable::UnknownVersion(v));
    }
    let rows = doc
        .remove("credentials")
        .ok_or_else(|| Unparsable::Corrupt("no credentials".into()))?;
    let Value::Array(rows) = rows else {
        return Err(Unparsable::Corrupt("credentials is not a list".into()));
    };
    let mut records: Vec<Record> = Vec::new();
    let mut unknown: Vec<Value> = Vec::new();
    for row in rows {
        match serde_json::from_value::<Record>(row.clone()) {
            Ok(record) => records.push(record),
            Err(_) if newer_shape(&row) => unknown.push(row),
            Err(e) => return Err(Unparsable::Corrupt(e.to_string())),
        }
    }
    let mut seen = std::collections::HashSet::new();
    let unknown_ids = unknown
        .iter()
        .filter_map(|r| r.get("id").and_then(Value::as_str));
    for id in records.iter().map(|r| r.id.as_str()).chain(unknown_ids) {
        if !seen.insert(id) {
            return Err(Unparsable::Corrupt(format!("the id {id} appears twice")));
        }
    }
    Ok((records, unknown, doc))
}
