//! The web login of `oaiy-server` (design 4.7): the endpoints, and the state they share.
//!
//! One owner, a password checked with Argon2id, server-side sessions in a cookie per app host, and a throttle that
//! cannot keep the owner out. The parts are in their own files (`password`, `policy`, `throttle`, `lanes`, `setup`,
//! `device`, `owner`, `session`, `cookie`); this file is what puts a request through them:
//!
//! - `POST /api/auth/login`: the dashboard host only (`404` elsewhere), a secure channel, no cross-origin browser,
//!   the optional address allow-list, then the lane: a valid `dev` cookie is the known-device lane (the reserved
//!   slot, a counter of its own), anything else the anonymous lane (a queue, a block per address, slow mode).
//!   **Every** answer that is not a success is the same `401 invalid_credentials`, and every one of them ran one
//!   verification: a wrong password, an unusable stored hash (verified against a hash of ours), a device cookie
//!   that is unknown or wrong (which is simply the anonymous lane).
//! - `POST /api/auth/setup`, `POST /api/auth/link`: the same prelude and the address throttle; a wrong code counts as
//!   a failure of the address and starts no hash.
//! - `GET /api/auth/session` (public, reads the cookie), `POST /api/auth/logout`, `elevate`, `password`, and the
//!   list and revocation of sessions and devices.
//!
//! Signing in never needs a write: a session made while the disk is full lives in memory, marked not persisted.
//! Setup, a password change and a reset do need one, and answer `503 store_unavailable` without changing anything.
//! Secrets are never logged and never in `Debug`: a password is held in a type that wipes itself, though the
//! request body buffers and `serde_json`'s intermediate strings are not under this code's control.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::{Extension, Path as UrlPath, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};
use zeroize::{Zeroize, Zeroizing};

use super::api::{is_json, MAX_BODY};
use super::audit::Context as AuditContext;
use super::clientip::Cidr;
use super::clock::{Clock, SystemClock};
use super::cookie::{self, Lookup, DEVICE_MAX_AGE, REMEMBER_MAX_AGE};
use super::device;
use super::guard::{Denial, Guard, RequestInfo};
use super::lanes::{
    self, AnonLane, AnonPermit, Hasher, Reject, ReservedLane, SessionGate, SessionLane,
};
use super::mode::{AccessMode, ConfigRefusal};
use super::owner::{self, CreateError, OwnerDoc};
use super::password::{Argon2Engine, HashError, PasswordEngine, Verdict};
use super::policy::{self, Reason};
use super::presets::App;
use super::principal::{Actor, Principal, PrincipalKind};
use super::scrub::scrub_line;
use super::session::{self, CookieHost, LoginFacts};
use super::setup::{Check, MakeError, SetupCode};
use super::store::{
    is_storage_error, AuthStore, FileWriter, MintFailure, Random, SecureWriter, StoreError,
};
use super::throttle::{Gate, LoginThrottle, FLUSH_EVERY_MS};
use super::token::{self, Kind};
use tokio::sync::OwnedSemaphorePermit;

/// Session-link codes kept at once (the oldest is dropped for a new one).
pub const MAX_LINKS: usize = 8;
/// A session link is valid this long, once.
pub const LINK_VALID_MS: u64 = 5 * 60_000;
/// The owner's devices are written at most this often when only a use changed them.
pub const OWNER_FLUSH_MS: u64 = 60_000;

// ---- the pieces a test replaces -----------------------------------------------------------------

/// A clock that only goes forward: what a link code's five minutes are measured on, so that a wall-clock jump
/// cannot extend one (design 6, "clock skew").
pub struct MonotonicClock {
    start: std::time::Instant,
}

impl MonotonicClock {
    pub fn new() -> MonotonicClock {
        MonotonicClock {
            start: std::time::Instant::now(),
        }
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for MonotonicClock {
    fn now_ms(&self) -> u64 {
        // One is added so that "zero" can never be a moment.
        self.start.elapsed().as_millis() as u64 + 1
    }
}

/// What the login is built from. [`LoginOptions::production`] is what `oaiy-server` runs; the tests replace the
/// clock, the engine, the randomness and the writer.
pub struct LoginOptions {
    /// The port the listener is on: the loopback dashboard's origin has it.
    pub port: u16,
    /// The dashboard's origin, for links (`https://dash.example.com`, or the loopback name).
    pub dash_origin: String,
    /// The host names of the install: the password estimate treats them as guessable.
    pub hosts: Vec<String>,
    /// `OAIY_LOGIN_ALLOW`: when not empty, only these networks may reach `login`, `setup` and `link`.
    pub login_allow: Vec<Cidr>,
    pub engine: Arc<dyn PasswordEngine>,
    pub random: Random,
    pub writer: Arc<dyn FileWriter>,
    /// Wall-clock time (sessions, the throttle); the store and the audit log read the same.
    pub clock: Arc<dyn Clock>,
    /// Monotonic time (link codes).
    pub mono: Arc<dyn Clock>,
}

impl LoginOptions {
    /// The real thing, from the environment: `OAIY_PUBLIC_URL` and `OAIY_LOGIN_ALLOW`.
    pub fn production(env: &dyn Fn(&str) -> Option<String>, port: u16) -> LoginOptions {
        let get = |name: &str| {
            env(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let public = get("OAIY_PUBLIC_URL");
        let dash_origin = public
            .as_deref()
            .map(|u| u.trim_end_matches('/').to_string())
            .unwrap_or_else(|| format!("http://dash.oaiy.localhost:{port}"));
        let mut hosts = Vec::new();
        for var in ["OAIY_PUBLIC_URL", "OAIY_AGENT_URL", "OAIY_FLOWS_URL"] {
            if let Some(url) = get(var) {
                if let Some(rest) = url.strip_prefix("https://") {
                    hosts.push(rest.trim_end_matches('/').to_string());
                }
            }
        }
        let mut login_allow = Vec::new();
        if let Some(list) = get("OAIY_LOGIN_ALLOW") {
            for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
                match Cidr::parse(entry) {
                    Some(c) => login_allow.push(c),
                    None => log::warn!(
                        "{}",
                        scrub_line(&format!(
                            "OAIY_LOGIN_ALLOW: {entry:?} is not an address or a network: ignored"
                        ))
                    ),
                }
            }
        }
        LoginOptions {
            port,
            dash_origin,
            hosts,
            login_allow,
            engine: Arc::new(Argon2Engine::production()),
            random: Arc::new(token::os_random),
            writer: Arc::new(SecureWriter),
            clock: Arc::new(SystemClock),
            mono: Arc::new(MonotonicClock::new()),
        }
    }
}

// ---- the state ----------------------------------------------------------------------------------

/// A session link the console made: only its hash, and when it ends (on the monotonic clock).
struct LinkCode {
    hash: String,
    ends_mono_ms: u64,
}

/// The state of the login, shared by its routes.
pub struct LoginState {
    pub(crate) guard: Arc<Guard>,
    pub(crate) clock: Arc<dyn Clock>,
    mono: Arc<dyn Clock>,
    pub(crate) auth_dir: PathBuf,
    writer: Arc<dyn FileWriter>,
    pub(crate) random: Random,
    pub(crate) port: u16,
    pub(crate) dash_origin: String,
    login_allow: Vec<Cidr>,
    /// Memory is authoritative: read once at start, written by the login and nothing else.
    owner: Mutex<Option<OwnerDoc>>,
    owner_dirty: AtomicBool,
    last_owner_flush: Mutex<u64>,
    /// Serialises what changes the owner (setup, a password change, a reset).
    pub(crate) owner_gate: tokio::sync::Mutex<()>,
    /// One write of `owner.json` at a time, from the owner as it is when the write starts: a slow write of an older
    /// owner can never land after a newer one.
    owner_write: Mutex<()>,
    /// Which password the owner has: one more with every password that is set (a setup, a change, a reset). A
    /// verification is made against a snapshot and carries its generation to the point where it would make a session.
    generation: AtomicU64,
    pub(crate) setup: SetupCode,
    pub(crate) throttle: Arc<LoginThrottle>,
    anon: Arc<AnonLane>,
    pub(crate) reserved: Arc<ReservedLane>,
    session_lane: SessionLane,
    pub(crate) hasher: Arc<Hasher>,
    links: Mutex<Vec<LinkCode>>,
    pub(crate) extra_inputs: Vec<String>,
}

/// Put the web login on `guard`: read `owner.json` (through the store), the setup code and `throttle.json`, and
/// install the login into the guard (so that it reads the session cookie and knows whether an owner exists).
/// Refuses, as the store does, what it cannot read: a mangled or unreadable file is a startup error and never
/// "setup-only".
pub fn enable(
    guard: &Arc<Guard>,
    auth_dir: &Path,
    opts: LoginOptions,
) -> Result<Arc<LoginState>, StoreError> {
    let owner =
        match guard.store().owner() {
            None => None,
            Some(o) => Some(OwnerDoc::from_value(&o.doc).map_err(|detail| {
                StoreError::OwnerUnparsable {
                    file: owner::path(auth_dir),
                    detail,
                }
            })?),
        };
    let setup = SetupCode::open(auth_dir, opts.clock.clone(), opts.writer.clone())?;
    // The throttle file is the guard's (it holds the failed-bearer throttle's state and writes the file): what it
    // kept for this login is in it under `login`. A file that cannot be read is an empty state, as it is for the
    // bearer throttle: this is a memory of failures, not a credential.
    let throttle = Arc::new(LoginThrottle::new(opts.clock.clone()));
    if let Some(saved) = guard.saved_throttle_state("login") {
        throttle.restore(&saved);
    }
    let anon = AnonLane::new(opts.clock.clone(), throttle.clone());
    let state = Arc::new(LoginState {
        guard: guard.clone(),
        clock: opts.clock.clone(),
        mono: opts.mono.clone(),
        auth_dir: auth_dir.to_path_buf(),
        writer: opts.writer.clone(),
        random: opts.random.clone(),
        port: opts.port,
        dash_origin: opts.dash_origin.clone(),
        login_allow: opts.login_allow.clone(),
        owner: Mutex::new(owner),
        owner_dirty: AtomicBool::new(false),
        last_owner_flush: Mutex::new(opts.clock.now_ms()),
        owner_gate: tokio::sync::Mutex::new(()),
        owner_write: Mutex::new(()),
        generation: AtomicU64::new(0),
        setup,
        throttle,
        anon,
        reserved: ReservedLane::new(),
        session_lane: SessionLane::new(opts.clock.clone()),
        hasher: Hasher::new(opts.engine.clone()),
        links: Mutex::new(Vec::new()),
        extra_inputs: policy::extra_inputs(opts.hosts.iter().map(String::as_str)),
    });
    let weak: std::sync::Weak<dyn LoginFacts> =
        Arc::downgrade(&state) as std::sync::Weak<dyn LoginFacts>;
    guard.install_login(weak);
    Ok(state)
}

impl LoginFacts for LoginState {
    fn owner_configured(&self) -> bool {
        self.owner_lock().is_some()
    }

    fn setup_code_status(&self) -> &'static str {
        self.setup.status().name()
    }

    fn under_attack(&self) -> bool {
        self.throttle.under_attack().is_some()
    }

    fn flush(&self) {
        self.flush_throttle();
        self.flush_owner();
    }
}

/// A password from a request body: wiped when dropped, and never shown.
pub(crate) struct Secret(pub String);

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret([redacted])")
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d).map(Secret)
    }
}

#[derive(Deserialize)]
struct LoginBody {
    password: Secret,
    #[serde(default)]
    remember: bool,
}

#[derive(Deserialize)]
struct SetupBody {
    code: Secret,
    password: Secret,
}

#[derive(Deserialize)]
struct ElevateBody {
    password: Secret,
    #[serde(default)]
    method: Option<String>,
}

#[derive(Deserialize)]
struct PasswordBody {
    current: Secret,
    next: Secret,
    #[serde(default, rename = "revokeTokens")]
    revoke_tokens: bool,
}

#[derive(Deserialize)]
struct LinkBody {
    code: Secret,
}

// ---- answers --------------------------------------------------------------------------------------

fn denial(status: StatusCode, code: &'static str, message: &str) -> Denial {
    Denial::new(status, code, message)
}

/// The one answer to every failed login, elevation and password check.
pub(crate) fn invalid_credentials() -> Denial {
    denial(
        StatusCode::UNAUTHORIZED,
        "invalid_credentials",
        "The password is not right.",
    )
}

fn too_many(retry_after_s: u64, message: &str) -> Denial {
    let mut d = denial(StatusCode::TOO_MANY_REQUESTS, "rate_limited", message);
    d.extra.push(("retryAfterSeconds", json!(retry_after_s)));
    d.headers.push(("retry-after", retry_after_s.to_string()));
    d
}

fn store_unavailable(why: &str) -> Denial {
    denial(StatusCode::SERVICE_UNAVAILABLE, "store_unavailable", why)
}

fn mint_denial(e: &MintFailure) -> Denial {
    match e {
        MintFailure::StoreUnavailable(_) => store_unavailable("The credential store cannot write."),
        MintFailure::Token(_) => store_unavailable(
            "The operating system gave no randomness, so no session can be made right now.",
        ),
        other => store_unavailable(&other.to_string()),
    }
}

fn weak_password(reasons: &[Reason]) -> Denial {
    let mut d = denial(
        StatusCode::BAD_REQUEST,
        "weak_password",
        "That password is not accepted.",
    );
    d.extra.push((
        "reasons",
        json!(reasons.iter().map(|r| r.code()).collect::<Vec<_>>()),
    ));
    d
}

fn not_found() -> Denial {
    Denial::not_found()
}

pub(crate) fn json_reply(status: StatusCode, body: Value, cookies: &[String]) -> Response {
    let mut response = (status, Json(body)).into_response();
    for c in cookies {
        if let Ok(v) = HeaderValue::from_str(c) {
            response.headers_mut().append(header::SET_COOKIE, v);
        }
    }
    response
}

fn no_content(cookies: &[String]) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    for c in cookies {
        if let Ok(v) = HeaderValue::from_str(c) {
            response.headers_mut().append(header::SET_COOKIE, v);
        }
    }
    response
}

/// The body of a route that takes JSON: `application/json`, at most 16 KiB, and the shape asked for. What is
/// wrong is never echoed back (a body may hold a password).
pub(crate) async fn read_json<T: DeserializeOwned>(
    headers: &HeaderMap,
    body: Body,
) -> Result<T, Denial> {
    if !is_json(headers) {
        return Err(denial(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "unsupported_media_type",
            "Send Content-Type: application/json.",
        ));
    }
    let bytes = axum::body::to_bytes(body, MAX_BODY)
        .await
        .map_err(|_| Denial::bad_request("The body is over 16 KiB or could not be read."))?;
    serde_json::from_slice(&bytes)
        .map_err(|_| Denial::bad_request("The body is not the JSON this route takes."))
}

// ---- the request prelude ----------------------------------------------------------------------------

/// What every public login route establishes first.
pub(crate) struct Prelude {
    pub info: RequestInfo,
    pub ch: CookieHost,
    pub ua: String,
}

impl Prelude {
    pub(crate) fn ctx(&self) -> AuditContext<'_> {
        AuditContext {
            ip: Some(&self.info.client_ip),
            host: Some(&self.info.host),
            ua: Some(&self.ua),
        }
    }
}

fn user_agent(headers: &HeaderMap) -> String {
    headers
        .get(header::USER_AGENT)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .chars()
        .take(session::MAX_UA)
        .collect()
}

impl LoginState {
    fn owner_lock(&self) -> std::sync::MutexGuard<'_, Option<OwnerDoc>> {
        self.owner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn store(&self) -> &Arc<AuthStore> {
        self.guard.store()
    }

    /// The order of the checks of a public login route (4.7.3): the dashboard host only, a secure channel, no
    /// cross-origin browser, the address allow-list.
    pub(crate) fn prelude(
        &self,
        info: Option<Extension<RequestInfo>>,
        headers: &HeaderMap,
    ) -> Result<Prelude, Denial> {
        let Some(Extension(info)) = info else {
            return Err(not_found());
        };
        let Some(ch) = CookieHost::of(&info) else {
            return Err(not_found());
        };
        if ch.app != App::Dash {
            return Err(not_found());
        }
        if !ch.secure() {
            return Err(denial(
                StatusCode::FORBIDDEN,
                "secure_channel_required",
                "Sign in over https, or from the machine itself.",
            ));
        }
        if let Err(fail) = cookie::check_same_origin_browser(headers, &ch.own_origin) {
            let mut d = denial(StatusCode::FORBIDDEN, "csrf", fail.message());
            d.noise = Some("auth.denied");
            return Err(d);
        }
        if !self.login_allow.is_empty() {
            let allowed = info
                .client_ip
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| self.login_allow.iter().any(|c| c.contains(ip)));
            if !allowed {
                self.noise("auth.denied", &info);
                return Err(denial(
                    StatusCode::FORBIDDEN,
                    "login_not_allowed",
                    "Sign-in is not allowed from this address.",
                ));
            }
        }
        Ok(Prelude {
            ua: user_agent(headers),
            info,
            ch,
        })
    }

    pub(crate) fn noise(&self, event: &str, info: &RequestInfo) {
        if let Some(log) = self.guard.audit() {
            log.noise(event, &info.client_ip, &info.host);
        }
    }

    pub(crate) fn critical(
        &self,
        event: &str,
        actor: Option<&Actor>,
        ctx: &AuditContext<'_>,
        detail: Value,
    ) {
        if let Some(log) = self.guard.audit() {
            log.critical(event, actor, ctx, detail);
        }
    }

    fn now(&self) -> u64 {
        self.clock.now_ms()
    }

    // ---- the owner file --------------------------------------------------------------------

    /// The stored password, the known devices and which password it is, if an owner exists. Taken as one, under the
    /// owner's lock, so that the hash and its generation cannot disagree.
    fn owner_snapshot(&self) -> Option<OwnerView> {
        let owner = self.owner_lock();
        owner.as_ref().map(|o| OwnerView {
            hash: o.password.clone(),
            devices: o.devices.clone(),
            generation: self.generation.load(Ordering::SeqCst),
        })
    }

    /// Take the owner gate if the password is still the one that was verified (`generation`, from the snapshot the
    /// verification was made against), and refuse as a wrong password does if it is not: a password changed while a
    /// login was verifying the old one leaves the login nothing to make. What the login makes (a session, a device)
    /// is made while the gate is held, so a change cannot fall between the check and the making: it either comes
    /// first and this is refused, or comes after and revokes what was made.
    async fn unchanged(&self, generation: u64) -> Result<Fresh<'_>, Denial> {
        let gate = self.owner_gate.lock().await;
        if self.generation.load(Ordering::SeqCst) != generation {
            return Err(invalid_credentials());
        }
        Ok(Fresh(gate))
    }

    /// Write the owner as it is in memory, for what may fail without failing the request (a device's use, a
    /// re-hash). A failure is logged, marked for a retry, and audited when the disk is the reason.
    fn persist_owner_best_effort(&self) {
        // The owner is copied and written under the write lock, so a write that started earlier and is slow is
        // finished before this one copies anything.
        let _one_at_a_time = self.owner_write.lock().unwrap_or_else(|e| e.into_inner());
        let Some(doc) = self.owner_lock().clone() else {
            return;
        };
        match owner::write(self.writer.as_ref(), &self.auth_dir, &doc) {
            Ok(()) => {
                self.owner_dirty.store(false, Ordering::SeqCst);
                *self
                    .last_owner_flush
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = self.now();
            }
            Err(e) => {
                self.owner_dirty.store(true, Ordering::SeqCst);
                log::warn!(
                    "{}",
                    scrub_line(&format!("auth: owner.json could not be written: {e}"))
                );
                if is_storage_error(&e) {
                    self.critical(
                        "disk.full",
                        None,
                        &AuditContext::default(),
                        json!({ "file": "owner.json" }),
                    );
                }
            }
        }
    }

    /// Replace the owner by `doc` (a changed password): the new file first, and only when it is written the new
    /// owner in memory, so that a disk that will not take it leaves the old one in force. Under the write lock, as
    /// every write of the file is.
    fn replace_owner(&self, doc: OwnerDoc) -> std::io::Result<()> {
        let _one_at_a_time = self.owner_write.lock().unwrap_or_else(|e| e.into_inner());
        owner::write(self.writer.as_ref(), &self.auth_dir, &doc)?;
        self.put_owner(doc);
        Ok(())
    }

    /// Make the first owner: the file, only if there is none, and then the owner in memory, under the write lock.
    fn create_owner(&self, doc: OwnerDoc) -> Result<(), CreateError> {
        let _one_at_a_time = self.owner_write.lock().unwrap_or_else(|e| e.into_inner());
        owner::create_exclusive(self.writer.as_ref(), &self.auth_dir, &doc)?;
        self.put_owner(doc);
        Ok(())
    }

    /// The owner in memory is this one, and it is another password than the one before: what was verified against
    /// the old one (`generation`) is no longer the owner's password.
    fn put_owner(&self, doc: OwnerDoc) {
        let mut owner = self.owner_lock();
        *owner = Some(doc);
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Write the owner if a use changed it and a minute has passed (or `force`).
    fn flush_owner(&self) {
        if self.owner_dirty.load(Ordering::SeqCst) {
            self.persist_owner_best_effort();
        }
    }

    /// Write a file of the auth folder, atomically and privately: the console credential's files.
    pub(crate) fn write_auth_file(&self, name: &str, bytes: &[u8]) -> std::io::Result<()> {
        self.writer.write(&self.auth_dir.join(name), bytes)
    }

    // ---- the throttle file -----------------------------------------------------------------

    /// Hand the guard the login throttle's state: it writes `throttle.json` with its own (the file has one writer).
    fn flush_throttle(&self) {
        self.guard
            .keep_throttle_state("login", self.throttle.snapshot());
    }

    /// The periodic upkeep: the throttle's file when it changed and five seconds have passed, the alert, the owner's
    /// devices once a minute, and link codes that ended.
    pub fn tick(&self) {
        if self.throttle.flush_due() {
            self.flush_throttle();
        }
        self.raise_alert();
        let due = {
            let last = *self
                .last_owner_flush
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            self.now().saturating_sub(last) >= OWNER_FLUSH_MS
        };
        if due {
            self.flush_owner();
        }
        self.purge_links();
    }

    fn raise_alert(&self) {
        if let Some(attack) = self.throttle.take_alert() {
            self.critical(
                "login.attack",
                None,
                &AuditContext::default(),
                json!({ "reason": attack.name(), "recentFailures": self.throttle.recent_failures() }),
            );
        }
    }

    // ---- lanes -----------------------------------------------------------------------------

    /// The lane a login is in, and its gate.
    async fn enter(&self, lane: &Lane, info: &RequestInfo) -> Result<Permit, Denial> {
        match lane {
            Lane::Device(id) => {
                // A blocked request writes nothing (the log records where a block starts, not every request it refuses).
                if let Gate::Blocked { retry_after_s } = self.throttle.device_gate(id) {
                    return Err(too_many(
                        retry_after_s,
                        "Too many wrong passwords from this device.",
                    ));
                }
                match self.reserved.acquire().await {
                    Ok(p) => Ok(Permit::Reserved(p)),
                    Err(r) => Err(self.rejected(r, info)),
                }
            }
            Lane::Anonymous(key) => {
                if let Gate::Blocked { retry_after_s } = self.throttle.address_gate(key) {
                    return Err(too_many(
                        retry_after_s,
                        "Too many failed attempts from this address.",
                    ));
                }
                if self.throttle.is_slow() {
                    self.noise("login.slow", info);
                }
                match self.anon.acquire(key).await {
                    Ok(p) => Ok(Permit::Anon(p)),
                    Err(r) => Err(self.rejected(r, info)),
                }
            }
        }
    }

    fn rejected(&self, r: Reject, info: &RequestInfo) -> Denial {
        self.noise("login.slow", info);
        too_many(
            r.retry_after_s(self.throttle.is_slow()),
            "Sign-in is delayed by attempts from elsewhere. Try again in a moment.",
        )
    }

    /// One verification, run to its end whatever becomes of the request that asked for it. A client that hangs up
    /// drops the request's future, but a pass of Argon2 that has begun runs on, holding its 64 MiB: so the pass keeps
    /// its place in the lane (`permit`) until it returns, and what it found is counted (`count`) by the task that ran
    /// it, not by a request that may be gone. Hanging up neither frees a place early nor escapes being counted. The
    /// request waits for the answer, and gets its place back with it.
    async fn verified(
        self: &Arc<Self>,
        permit: Permit,
        password: Zeroizing<String>,
        stored: String,
        count: impl FnOnce(&LoginState, &Verdict) + Send + 'static,
    ) -> (Verdict, Option<Permit>) {
        let state = self.clone();
        let task = tokio::spawn(async move {
            let verdict = state.hasher.verify(password, stored).await;
            count(&state, &verdict);
            (verdict, Some(permit))
        });
        // A task that panicked has no answer, and its place went with it.
        task.await.unwrap_or((Verdict::Mismatch, None))
    }

    /// A wrong password on `lane`.
    fn failed(&self, lane: &Lane, info: &RequestInfo) {
        let began = match lane {
            Lane::Device(id) => self.throttle.device_failed(id).blocked,
            Lane::Anonymous(key) => self.throttle.address_failed(key).blocked,
        };
        self.noise("login.fail", info);
        if began {
            self.noise("login.blocked", info);
        }
        self.raise_alert();
    }

    fn succeeded(&self, lane: &Lane) {
        match lane {
            Lane::Device(id) => self.throttle.device_succeeded(id),
            Lane::Anonymous(key) => self.throttle.address_succeeded(key),
        }
    }

    /// A failure that is not a password: a wrong setup code or link code counts against the address, and no
    /// hash is started.
    fn code_failed(&self, event: &str, pre: &Prelude) {
        let began = self.throttle.address_failed(&pre.info.client_key).blocked;
        self.noise(event, &pre.info);
        if began {
            self.noise("login.blocked", &pre.info);
        }
        self.raise_alert();
    }

    fn address_gate(&self, pre: &Prelude) -> Result<(), Denial> {
        if let Gate::Blocked { retry_after_s } = self.throttle.address_gate(&pre.info.client_key) {
            return Err(too_many(
                retry_after_s,
                "Too many failed attempts from this address.",
            ));
        }
        Ok(())
    }

    // ---- what a good login makes -------------------------------------------------------------

    /// The session of a login (a setup is one): elevated, made only with the proof that the password it verified is
    /// still the owner's.
    fn open_login_session(
        &self,
        _fresh: &Fresh<'_>,
        pre: &Prelude,
        remember: bool,
        device_cookie: Option<String>,
    ) -> Result<(Value, Vec<String>, Actor), Denial> {
        self.open_session(pre, remember, true, device_cookie)
    }

    /// A session for the owner, elevated, with its cookie; the body, the cookies to set and the actor.
    fn open_session(
        &self,
        pre: &Prelude,
        remember: bool,
        elevated: bool,
        device_cookie: Option<String>,
    ) -> Result<(Value, Vec<String>, Actor), Denial> {
        let minted = session::mint_session(self.store(), remember, &pre.info.client_ip, &pre.ua)
            .map_err(|e| mint_denial(&e))?;
        let now = self.now();
        if elevated {
            session::elevate(self.store(), &minted.id, now);
        }
        let principal = self
            .store()
            .authenticate(&minted.token, Some(&pre.info.client_ip))
            .map_err(|_| store_unavailable("The session could not be read back."))?;
        let secret = token::parse(&minted.token)
            .map(|p| p.secret.to_string())
            .unwrap_or_default();
        let csrf = token::csrf_value(&secret).unwrap_or_default();
        let body = session::session_json(self.store(), &principal, &csrf, now, true);
        let name = pre.ch.session_cookie();
        let mut cookies =
            vec![pre
                .ch
                .style
                .set(&name, &minted.token, remember.then_some(REMEMBER_MAX_AGE))];
        if let Some(d) = device_cookie {
            cookies.push(
                pre.ch
                    .style
                    .set(&pre.ch.device_cookie(), &d, Some(DEVICE_MAX_AGE)),
            );
        }
        Ok((body, cookies, principal.actor()))
    }

    /// A device for the browser that just proved the password: kept in the owner file when it can be written and in
    /// memory always. The cookie value.
    fn new_device(&self, _fresh: &Fresh<'_>, ip: &str) -> Option<String> {
        let mut fill = |buf: &mut [u8]| (self.random)(buf);
        let made = device::make(self.now(), ip, &mut fill).ok()?;
        {
            let mut owner = self.owner_lock();
            let doc = owner.as_mut()?;
            for gone in device::add(&mut doc.devices, made.device.clone()) {
                self.throttle.forget_device(&gone);
            }
        }
        self.persist_owner_best_effort();
        Some(made.token)
    }

    fn touch_device(&self, _fresh: &Fresh<'_>, id: &str, ip: &str) {
        let now = self.now();
        if let Some(doc) = self.owner_lock().as_mut() {
            device::touch(&mut doc.devices, id, now, ip);
        }
        // Only a use: written at the next upkeep, not now.
        self.owner_dirty.store(true, Ordering::SeqCst);
    }

    /// Hash the password again at the current cost, after a good login with a hash that was cheaper. A compare and
    /// set: the new hash replaces the stored one only if that is still the hash the login verified. A
    /// password changed while this hashed is the owner's, and is not undone (a changed password is always another
    /// string: it has its own salt).
    async fn rehash(&self, password: &Zeroizing<String>, verified: &str) {
        let Ok(new_hash) = self.hasher.hash(password.clone()).await else {
            return;
        };
        {
            // Under the owner's lock: the check and the set are one step, whatever a change does around them.
            let mut owner = self.owner_lock();
            let Some(doc) = owner.as_mut() else { return };
            if !token::hashes_equal(&doc.password, verified) {
                return;
            }
            doc.password = new_hash;
        }
        self.persist_owner_best_effort();
    }

    // ---- the routes --------------------------------------------------------------------------

    async fn do_login(
        self: &Arc<Self>,
        info: Option<Extension<RequestInfo>>,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response, Denial> {
        let pre = self.prelude(info, &headers)?;
        let req: LoginBody = read_json(&headers, body).await?;
        let Some(view) = self.owner_snapshot() else {
            return Err(denial(
                StatusCode::CONFLICT,
                "setup_required",
                "No owner login exists yet: make a setup code with `oaiy-server auth setup-code`.",
            ));
        };
        // The lane: a valid device cookie is the known-device lane; anything else (no cookie, an unknown id, a
        // wrong secret, a cookie of another kind) is the anonymous lane, all alike.
        let device_id = match cookie::find(&headers, &pre.ch.device_cookie()) {
            Lookup::One(v) => device::verify(&view.devices, &v).map(|d| d.id.clone()),
            _ => None,
        };
        let lane = match &device_id {
            Some(id) => Lane::Device(id.clone()),
            None => Lane::Anonymous(pre.info.client_key.clone()),
        };
        let permit = self.enter(&lane, &pre.info).await?;
        let password = policy::normalise(&req.password.0);
        // One verification for every path that reaches here, counted by the pass itself (a client that hangs up
        // during it is counted too, and its place is held until the pass ends).
        let (verdict, permit) = {
            let (lane, info) = (lane.clone(), pre.info.clone());
            self.verified(
                permit,
                password.clone(),
                view.hash.clone(),
                move |st, verdict| match verdict {
                    Verdict::Mismatch => st.failed(&lane, &info),
                    Verdict::Match { .. } => st.succeeded(&lane),
                },
            )
            .await
        };
        drop(permit);
        match verdict {
            Verdict::Mismatch => Err(invalid_credentials()),
            Verdict::Match { rehash } => {
                // The session and the device are made only if the password is still the one that was verified, and
                // while the gate is held: a password changed since is refused as a wrong one.
                let (body, cookies, actor) = {
                    let fresh = self.unchanged(view.generation).await?;
                    let device_cookie = match &lane {
                        Lane::Device(id) => {
                            self.touch_device(&fresh, id, &pre.info.client_ip);
                            None
                        }
                        Lane::Anonymous(_) => self.new_device(&fresh, &pre.info.client_ip),
                    };
                    self.open_login_session(&fresh, &pre, req.remember, device_cookie)?
                };
                if rehash {
                    self.rehash(&password, &view.hash).await;
                }
                self.critical(
                    "login.ok",
                    Some(&actor),
                    &pre.ctx(),
                    json!({ "device": device_id.is_some() }),
                );
                Ok(json_reply(StatusCode::OK, body, &cookies))
            }
        }
    }

    async fn do_setup(
        &self,
        info: Option<Extension<RequestInfo>>,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response, Denial> {
        let pre = self.prelude(info, &headers)?;
        let req: SetupBody = read_json(&headers, body).await?;
        if self.owner_lock().is_some() {
            return Err(already_configured());
        }
        self.address_gate(&pre)?;
        // The code first, and cheaply: nobody without it can make the server judge or hash a password.
        match self.setup.check(&req.code.0) {
            Check::Valid => {}
            Check::Expired => {
                return Err(denial(
                    StatusCode::GONE,
                    "setup_code_expired",
                    "The setup code has expired: make another with `oaiy-server auth setup-code`.",
                ))
            }
            Check::NoCode => {
                self.code_failed("setup.fail", &pre);
                let mut d = denial(
                    StatusCode::UNAUTHORIZED,
                    "invalid_setup_code",
                    "There is no setup code: make one with `oaiy-server auth setup-code`.",
                );
                d.extra.push(("attemptsLeft", json!(0)));
                return Err(d);
            }
            Check::Wrong { attempts_left } => {
                self.code_failed("setup.fail", &pre);
                let mut d = denial(
                    StatusCode::UNAUTHORIZED,
                    "invalid_setup_code",
                    "That is not the setup code.",
                );
                d.extra.push(("attemptsLeft", json!(attempts_left)));
                return Err(d);
            }
        }
        let password = policy::judge(&req.password.0, &self.extra_inputs)
            .map_err(|reasons| weak_password(&reasons))?;
        // The hash is the memory a verification takes: it goes through the anonymous lane like one.
        let permit = self
            .anon
            .acquire(&pre.info.client_key)
            .await
            .map_err(|r| self.rejected(r, &pre.info))?;
        let phc = self.hasher.hash(password).await;
        drop(permit);
        let phc = phc.map_err(hash_denial)?;

        let fresh = Fresh(self.owner_gate.lock().await);
        if self.owner_lock().is_some() {
            return Err(already_configured());
        }
        let doc = OwnerDoc::new(self.now(), phc);
        match self.create_owner(doc) {
            Ok(()) => {}
            Err(CreateError::Exists) => return Err(already_configured()),
            Err(CreateError::Io(e)) => {
                if is_storage_error(&e) {
                    self.critical(
                        "disk.full",
                        None,
                        &AuditContext::default(),
                        json!({ "file": "owner.json" }),
                    );
                }
                return Err(store_unavailable(
                    "The owner file cannot be written: setup needs a disk that takes a write.",
                ));
            }
        }
        self.setup.consume();
        self.throttle.address_succeeded(&pre.info.client_key);
        // Setup is a login: a session, elevated, and the browser's device.
        let device_cookie = self.new_device(&fresh, &pre.info.client_ip);
        let (body, cookies, actor) = self.open_login_session(&fresh, &pre, false, device_cookie)?;
        self.critical("setup.ok", Some(&actor), &pre.ctx(), json!({}));
        Ok(json_reply(StatusCode::CREATED, body, &cookies))
    }

    async fn do_link(
        &self,
        info: Option<Extension<RequestInfo>>,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response, Denial> {
        let pre = self.prelude(info, &headers)?;
        let req: LinkBody = read_json(&headers, body).await?;
        if self.owner_lock().is_none() {
            return Err(denial(
                StatusCode::CONFLICT,
                "setup_required",
                "No owner login exists yet.",
            ));
        }
        self.address_gate(&pre)?;
        if !self.take_link(&req.code.0) {
            self.code_failed("link.fail", &pre);
            return Err(Denial::bad_request(
                "This sign-in link is not valid or has expired.",
            ));
        }
        self.throttle.address_succeeded(&pre.info.client_key);
        // Neither remembered nor elevated, and no device cookie: this is the incident lifeline, not a login.
        let (body, cookies, actor) = self.open_session(&pre, false, false, None)?;
        self.critical("link.used", Some(&actor), &pre.ctx(), json!({}));
        Ok(json_reply(StatusCode::OK, body, &cookies))
    }

    fn do_session(&self, info: Option<Extension<RequestInfo>>, headers: HeaderMap) -> Response {
        let Some(Extension(info)) = info else {
            return not_found().into_response();
        };
        let ch = CookieHost::of(&info);
        let app = ch.as_ref().map_or("dash", |c| c.app.name());
        if let Some((principal, csrf)) = session::peek_session(&self.guard, &headers, &info) {
            let mut body =
                session::session_json(self.store(), &principal, &csrf, self.now(), false);
            if app == "dash" && self.throttle.under_attack().is_some() {
                body["loginAttack"] = json!(true);
            }
            return Json(body).into_response();
        }
        let login_url = match app {
            "dash" => Value::Null,
            other => json!(format!("{}/apps/{other}", self.dash_origin)),
        };
        Json(json!({ "authenticated": false, "app": app, "loginUrl": login_url })).into_response()
    }

    fn do_logout(&self, principal: &Principal, info: &RequestInfo) -> Response {
        self.store().revoke(&principal.id, "logged_out");
        self.session_lane.forget(&principal.id);
        self.critical(
            "logout",
            Some(&principal.actor()),
            &AuditContext {
                ip: Some(&info.client_ip),
                host: Some(&info.host),
                ua: None,
            },
            json!({}),
        );
        let clear = CookieHost::of(info).map(|c| c.clear_session());
        no_content(&clear.into_iter().collect::<Vec<_>>())
    }

    /// The session lane's front: the caller must be a dashboard session; it is counted, and given the reserved slot.
    async fn session_lane_enter(
        &self,
        principal: &Principal,
    ) -> Result<OwnedSemaphorePermit, Denial> {
        if principal.kind != PrincipalKind::Session || principal.app != Some(App::Dash) {
            return Err(denial(
                StatusCode::FORBIDDEN,
                "insufficient_scope",
                "Only a dashboard session can do this.",
            ));
        }
        if let SessionGate::Limited { retry_after_s } = self.session_lane.begin(&principal.id) {
            return Err(too_many(
                retry_after_s,
                "Too many password checks for this session in the last hour.",
            ));
        }
        self.reserved.acquire().await.map_err(|r| {
            too_many(
                r.retry_after_s(false),
                "The password check is busy: try again.",
            )
        })
    }

    /// A wrong answer in the session lane: the fifth in a row revokes the session.
    fn session_wrong(&self, principal: &Principal, info: &RequestInfo, what: &str) {
        let revoked = self.session_lane.wrong(&principal.id);
        let ctx = AuditContext {
            ip: Some(&info.client_ip),
            host: Some(&info.host),
            ua: None,
        };
        self.critical(
            &format!("{what}.fail"),
            Some(&principal.actor()),
            &ctx,
            json!({ "sessionRevoked": revoked }),
        );
        if revoked {
            self.store().revoke(&principal.id, "revoked");
            self.critical(
                "session.revoked",
                Some(&principal.actor()),
                &ctx,
                json!({ "why": "five wrong passwords" }),
            );
        }
    }

    async fn do_elevate(
        self: &Arc<Self>,
        principal: Principal,
        info: RequestInfo,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response, Denial> {
        let req: ElevateBody = read_json(&headers, body).await?;
        if req.method.as_deref().is_some_and(|m| m != "password") {
            return Err(Denial::invalid_request(
                "Only method \"password\" is supported.",
            ));
        }
        let permit = self.session_lane_enter(&principal).await?;
        let Some(view) = self.owner_snapshot() else {
            return Err(denial(
                StatusCode::CONFLICT,
                "setup_required",
                "No owner login exists.",
            ));
        };
        // Counted by the pass itself, as a login is (a client that hangs up is counted too).
        let (verdict, permit) = {
            let (who, info) = (principal.clone(), info.clone());
            self.verified(
                Permit::Reserved(permit),
                policy::normalise(&req.password.0),
                view.hash,
                move |st, verdict| match verdict {
                    Verdict::Mismatch => st.session_wrong(&who, &info, "elevate"),
                    Verdict::Match { .. } => st.session_lane.right(&who.id),
                },
            )
            .await
        };
        drop(permit);
        match verdict {
            Verdict::Mismatch => Err(invalid_credentials()),
            Verdict::Match { .. } => {
                let until = session::elevate(self.store(), &principal.id, self.now());
                self.critical(
                    "elevate.ok",
                    Some(&principal.actor()),
                    &AuditContext {
                        ip: Some(&info.client_ip),
                        host: Some(&info.host),
                        ua: None,
                    },
                    json!({}),
                );
                Ok(Json(json!({ "elevatedUntilMs": until })).into_response())
            }
        }
    }

    async fn do_password(
        self: &Arc<Self>,
        principal: Principal,
        info: RequestInfo,
        headers: HeaderMap,
        body: Body,
    ) -> Result<Response, Denial> {
        let req: PasswordBody = read_json(&headers, body).await?;
        let permit = self.session_lane_enter(&principal).await?;
        let Some(view) = self.owner_snapshot() else {
            return Err(denial(
                StatusCode::CONFLICT,
                "setup_required",
                "No owner login exists.",
            ));
        };
        // Counted by the pass itself, as a login is (a client that hangs up is counted too).
        let (verdict, permit) = {
            let (who, info) = (principal.clone(), info.clone());
            self.verified(
                Permit::Reserved(permit),
                policy::normalise(&req.current.0),
                view.hash,
                move |st, verdict| match verdict {
                    Verdict::Mismatch => st.session_wrong(&who, &info, "password"),
                    Verdict::Match { .. } => st.session_lane.right(&who.id),
                },
            )
            .await
        };
        if verdict == Verdict::Mismatch {
            drop(permit);
            return Err(invalid_credentials());
        }
        let next = policy::judge(&req.next.0, &self.extra_inputs)
            .map_err(|reasons| weak_password(&reasons))?;
        let phc = self.hasher.hash(next).await.map_err(hash_denial)?;
        drop(permit);
        // The change is made only if the password is still the one that was verified (another change may have been
        // made while this one was hashing), and while the gate is held.
        let _gate = self.unchanged(view.generation).await?;
        // Written before anything changes: a disk that will not take it leaves the old password in place.
        let mut doc = self
            .owner_lock()
            .clone()
            .ok_or_else(|| store_unavailable("The owner is gone."))?;
        let devices: Vec<String> = doc.devices.iter().map(|d| d.id.clone()).collect();
        doc.password = phc;
        doc.password_changed_ms = self.now();
        doc.devices.clear();
        if let Err(e) = self.replace_owner(doc) {
            if is_storage_error(&e) {
                self.critical(
                    "disk.full",
                    None,
                    &AuditContext::default(),
                    json!({ "file": "owner.json" }),
                );
            }
            return Err(store_unavailable(
                "The new password could not be written: the old one is still in force.",
            ));
        }
        for id in &devices {
            self.throttle.forget_device(id);
        }
        // Every other session, every derived credential and, on request, the paired tokens.
        let keep = principal.id.clone();
        let tokens = req.revoke_tokens;
        let revoked = self.store().revoke_where("password_changed", &|r| {
            (r.kind == Kind::Ses && r.id != keep)
                || r.kind == Kind::Run
                || (tokens && r.kind == Kind::Pat)
        });
        self.critical(
            "password.changed",
            Some(&principal.actor()),
            &AuditContext {
                ip: Some(&info.client_ip),
                host: Some(&info.host),
                ua: None,
            },
            json!({ "revoked": revoked.len(), "devices": devices.len(), "tokens": tokens }),
        ); // The browser's device is among those revoked: its cookie is no use, so it is dropped.
        let clear = CookieHost::of(&info).map(|c| c.clear_device());
        Ok(no_content(&clear.into_iter().collect::<Vec<_>>()))
    }

    // ---- sessions and devices --------------------------------------------------------------

    fn do_sessions(&self, principal: &Principal) -> Response {
        let now = self.now();
        let mut sessions: Vec<Value> = self
            .store()
            .records()
            .into_iter()
            .filter(|r| r.kind == Kind::Ses && r.revoked_ms.is_none() && now < r.expires_ms)
            .map(|r| {
                let made = r.extra.get("login").cloned().unwrap_or(Value::Null);
                json!({
                    "id": r.id,
                    "app": r.app.map_or("dash", |a| a.name()),
                    "createdMs": r.created_ms,
                    "lastUsedMs": r.last_used_ms.unwrap_or(r.created_ms),
                    "ip": r.last_used_ip.clone().or_else(|| made["ip"].as_str().map(str::to_owned)),
                    "userAgent": made["ua"].as_str().unwrap_or("").chars().take(session::MAX_UA).collect::<String>(),
                    "current": r.id == principal.id,
                })
            })
            .collect();
        sessions.sort_by_key(|s| std::cmp::Reverse(s["createdMs"].as_u64().unwrap_or(0)));
        let devices: Vec<Value> = self
            .owner_lock()
            .as_ref()
            .map(|o| {
                o.devices
                    .iter()
                    .map(|d| json!({ "id": d.id, "createdMs": d.created_ms, "lastMs": d.last_ms, "ip": d.ip }))
                    .collect()
            })
            .unwrap_or_default();
        Json(json!({ "sessions": sessions, "devices": devices })).into_response()
    }

    /// Revoke a session or a known device by id: the answer to a lost laptop. No step-up: revoking never needs the
    /// password, so that an owner under attack can cut access without typing one into a throttled queue.
    fn do_revoke(
        &self,
        principal: &Principal,
        info: &RequestInfo,
        id: &str,
    ) -> Result<Response, Denial> {
        let ctx = AuditContext {
            ip: Some(&info.client_ip),
            host: Some(&info.host),
            ua: None,
        };
        let is_session = self
            .store()
            .record(id)
            .is_some_and(|r| r.kind == Kind::Ses && r.revoked_ms.is_none());
        if is_session {
            self.store().revoke(id, "revoked");
            self.session_lane.forget(id);
            self.critical(
                "session.revoked",
                Some(&principal.actor()),
                &ctx,
                json!({ "id": id }),
            );
            return Ok(no_content(&[]));
        }
        let removed = {
            let mut owner = self.owner_lock();
            owner
                .as_mut()
                .is_some_and(|o| device::remove(&mut o.devices, id))
        };
        if removed {
            self.throttle.forget_device(id);
            self.persist_owner_best_effort();
            self.critical(
                "device.revoked",
                Some(&principal.actor()),
                &ctx,
                json!({ "id": id }),
            );
            return Ok(no_content(&[]));
        }
        Err(not_found())
    }

    fn do_revoke_others(&self, principal: &Principal, info: &RequestInfo) -> Response {
        let keep = (principal.kind == PrincipalKind::Session).then(|| principal.id.clone());
        let revoked = self.store().revoke_where("revoked", &|r| {
            r.kind == Kind::Ses && Some(&r.id) != keep.as_ref()
        });
        for id in &revoked {
            self.session_lane.forget(id);
        }
        self.critical(
            "session.revoked",
            Some(&principal.actor()),
            &AuditContext {
                ip: Some(&info.client_ip),
                host: Some(&info.host),
                ua: None,
            },
            json!({ "others": revoked.len() }),
        );
        no_content(&[])
    }

    // ---- session links -------------------------------------------------------------------------

    /// Make a session link (the console asks): 32 random bytes as 43 base64url characters, kept only as a hash and
    /// valid five minutes, once. The code, and the URL that holds it in the fragment.
    pub fn make_link(&self) -> Result<(String, String), MakeError> {
        let mut bytes = [0u8; 32];
        (self.random)(&mut bytes).map_err(MakeError::Random)?;
        let code = URL_SAFE_NO_PAD.encode(bytes);
        {
            let mut links = self.links.lock().unwrap_or_else(|e| e.into_inner());
            links.push(LinkCode {
                hash: token::secret_hash(&code),
                ends_mono_ms: self.mono.now_ms().saturating_add(LINK_VALID_MS),
            });
            while links.len() > MAX_LINKS {
                links.remove(0);
            }
        }
        let url = format!("{}/auth/link#{code}", self.dash_origin);
        Ok((code, url))
    }

    /// Spend a link code: `true` once for a live one. Every outstanding hash is compared in constant time.
    fn take_link(&self, code: &str) -> bool {
        // Not 43 characters of the alphabet: it cannot be one, and no hash of it means anything.
        if code.len() != 43
            || !code
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return false;
        }
        let presented = token::secret_hash(code);
        let now = self.mono.now_ms();
        let mut links = self.links.lock().unwrap_or_else(|e| e.into_inner());
        let mut found = None;
        for (i, link) in links.iter().enumerate() {
            let same = token::hashes_equal(&presented, &link.hash);
            if same && now < link.ends_mono_ms {
                found = Some(i);
            }
        }
        match found {
            Some(i) => {
                links.remove(i);
                true
            }
            None => false,
        }
    }

    fn purge_links(&self) {
        let now = self.mono.now_ms();
        self.links
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|l| now < l.ends_mono_ms);
    }

    /// How many session links are outstanding (for the tests and the status).
    pub fn links_outstanding(&self) -> usize {
        self.purge_links();
        self.links.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// The banner printed at start when there is no owner: no secret in it (design 4.7.1).
    pub fn banner(&self) -> Option<String> {
        if self.owner_lock().is_some() {
            return None;
        }
        Some(format!(
            "oaiy-server: no owner login yet. Run 'sudo -u oaiy oaiy-server auth setup-code' and open {}/setup, or run 'sudo -u oaiy oaiy-server auth init' to set the password on the console.",
            self.dash_origin
        ))
    }

    /// What the console's `status` says about the login: no secret in it.
    pub(crate) fn status_json(&self) -> Value {
        let now = self.now();
        let records = self.store().records();
        let live = |kind: Kind| {
            records
                .iter()
                .filter(|r| r.kind == kind && r.revoked_ms.is_none() && now < r.expires_ms)
                .count()
        };
        let devices = self.owner_lock().as_ref().map_or(0, |o| o.devices.len());
        let blocked = |list: Vec<(String, u64)>| -> Vec<Value> {
            list.into_iter()
                .map(|(k, s)| json!({ "key": k, "retryAfterSeconds": s }))
                .collect()
        };
        let gauge = self.hasher.gauge();
        json!({
            "exposure": self.guard.config().exposure.name(),
            "accessMode": self.guard.mode().name(),
            "loginConfigured": self.owner_lock().is_some(),
            "setupCode": self.setup.status().name(),
            "setupCodeAttemptsLeft": self.setup.attempts_left(),
            "sessions": live(Kind::Ses),
            "tokens": live(Kind::Pat),
            "devices": devices,
            "slowMode": self.throttle.is_slow(),
            "recentFailures": self.throttle.recent_failures(),
            "blockedAddresses": blocked(self.throttle.blocked_addresses()),
            "blockedDevices": blocked(self.throttle.blocked_devices()),
            "underAttack": self.throttle.under_attack().map(|a| a.name()),
            "storage": self.store().storage().name(),
            "persistent": self.store().is_persistent(),
            "linksOutstanding": self.links_outstanding(),
            "verifications": {
                "started": gauge.verifications,
                "running": gauge.running,
                "mostAtOnce": gauge.peak,
                "bound": lanes::FENCE,
            },
            "port": self.port,
        })
    }

    // ---- what the console does to the owner -------------------------------------------------------

    /// Set the password from the console: a new owner when there is none (and the setup code is spent), else a
    /// changed password that revokes every session and device. `Ok(true)` when an owner was created.
    pub(crate) async fn console_set_password(
        &self,
        password: &str,
        actor: &Actor,
        by_console_only_ctx: &AuditContext<'_>,
    ) -> Result<bool, Denial> {
        let normalised = policy::judge(password, &self.extra_inputs)
            .map_err(|reasons| weak_password(&reasons))?;
        let permit = self
            .reserved
            .acquire()
            .await
            .map_err(|r| too_many(r.retry_after_s(false), "The password hasher is busy."))?;
        let phc = self.hasher.hash(normalised).await.map_err(hash_denial)?;
        drop(permit);
        let _gate = self.owner_gate.lock().await;
        let existing = self.owner_lock().clone();
        let (doc, created) = match existing {
            None => (OwnerDoc::new(self.now(), phc), true),
            Some(mut doc) => {
                doc.password = phc;
                doc.password_changed_ms = self.now();
                doc.devices.clear();
                (doc, false)
            }
        };
        let old_devices: Vec<String> = self
            .owner_lock()
            .as_ref()
            .map(|o| o.devices.iter().map(|d| d.id.clone()).collect())
            .unwrap_or_default();
        // The file, and then the owner in memory, under the write lock (as every write of the file is).
        let written = if created {
            self.create_owner(doc).map_err(|e| match e {
                CreateError::Exists => already_configured(),
                CreateError::Io(_) => store_unavailable("The owner file cannot be written."),
            })
        } else {
            self.replace_owner(doc)
                .map_err(|_| store_unavailable("The owner file cannot be written."))
        };
        written?;
        for d in &old_devices {
            self.throttle.forget_device(d);
        }
        let revoked = self
            .store()
            .revoke_where("password_changed", &|r| r.kind == Kind::Ses);
        for id in &revoked {
            self.session_lane.forget(id);
        }
        if created {
            self.setup.consume();
        }
        self.critical(
            if created {
                "setup.ok"
            } else {
                "password.changed"
            },
            Some(actor),
            by_console_only_ctx,
            json!({ "by": "console", "revoked": revoked.len() }),
        );
        Ok(created)
    }

    /// Revoke every session, and every device when asked (the console's `sessions revoke-all`).
    pub(crate) fn console_revoke_all(&self, devices: bool, actor: &Actor) -> (usize, usize) {
        let revoked = self
            .store()
            .revoke_where("revoked", &|r| r.kind == Kind::Ses);
        for id in &revoked {
            self.session_lane.forget(id);
        }
        let mut removed = 0;
        if devices {
            let ids: Vec<String> = {
                let mut owner = self.owner_lock();
                owner
                    .as_mut()
                    .map(|o| {
                        let ids = o.devices.iter().map(|d| d.id.clone()).collect();
                        o.devices.clear();
                        ids
                    })
                    .unwrap_or_default()
            };
            removed = ids.len();
            for id in &ids {
                self.throttle.forget_device(id);
            }
            if removed > 0 {
                self.persist_owner_best_effort();
            }
        }
        self.critical(
            "session.revoked",
            Some(actor),
            &AuditContext::default(),
            json!({ "by": "console", "sessions": revoked.len(), "devices": removed }),
        );
        (revoked.len(), removed)
    }
}

fn already_configured() -> Denial {
    denial(
        StatusCode::CONFLICT,
        "already_configured",
        "An owner login already exists.",
    )
}

fn hash_denial(e: HashError) -> Denial {
    match e {
        HashError::NoRandomness => store_unavailable(
            "The operating system gave no randomness, so a password cannot be hashed right now.",
        ),
        HashError::Argon2(_) => store_unavailable("The password could not be hashed."),
    }
}

/// The proof that the owner gate is held and that the password is the one that was verified: what a login makes (a
/// device, a session) can be made only with it, so that the making cannot be moved out from under the gate.
struct Fresh<'a>(#[allow(dead_code)] tokio::sync::MutexGuard<'a, ()>);

/// What a verification is made against: the stored hash, the known devices, and which password it is.
struct OwnerView {
    hash: String,
    devices: Vec<owner::Device>,
    generation: u64,
}

/// The lane of a login.
#[derive(Clone)]
enum Lane {
    Device(String),
    Anonymous(String),
}

/// A place in one of the lanes, held while a password is verified (it frees the place when dropped).
#[allow(dead_code)]
enum Permit {
    Anon(AnonPermit),
    Reserved(OwnedSemaphorePermit),
}

// ---- the routes ------------------------------------------------------------------------------------

/// The login's routes. The console's are in `console.rs`; both are merged into the server's router behind the guard.
pub fn router(state: Arc<LoginState>) -> Router {
    Router::new()
        .route("/api/auth/login", post(login))
        .route("/api/auth/setup", post(setup))
        .route("/api/auth/link", post(link))
        .route("/api/auth/session", get(session_route))
        .route("/api/auth/logout", post(logout))
        .route("/api/auth/elevate", post(elevate))
        .route("/api/auth/password", post(password))
        .route("/api/auth/sessions", get(sessions))
        .route("/api/auth/sessions/revoke-others", post(revoke_others))
        .route("/api/auth/sessions/:id", delete(revoke))
        .merge(super::console::routes())
        .with_state(state)
}

async fn login(
    State(st): State<Arc<LoginState>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    st.do_login(info, headers, body)
        .await
        .unwrap_or_else(Denial::into_response)
}

async fn setup(
    State(st): State<Arc<LoginState>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    st.do_setup(info, headers, body)
        .await
        .unwrap_or_else(Denial::into_response)
}

async fn link(
    State(st): State<Arc<LoginState>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    st.do_link(info, headers, body)
        .await
        .unwrap_or_else(Denial::into_response)
}

async fn session_route(
    State(st): State<Arc<LoginState>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
) -> Response {
    st.do_session(info, headers)
}

async fn logout(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
) -> Response {
    let (Some(Extension(p)), Some(Extension(i))) = (principal, info) else {
        return Denial::auth_required().into_response();
    };
    st.do_logout(&p, &i)
}

async fn elevate(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let (Some(Extension(p)), Some(Extension(i))) = (principal, info) else {
        return Denial::auth_required().into_response();
    };
    st.do_elevate(p, i, headers, body)
        .await
        .unwrap_or_else(Denial::into_response)
}

async fn password(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let (Some(Extension(p)), Some(Extension(i))) = (principal, info) else {
        return Denial::auth_required().into_response();
    };
    st.do_password(p, i, headers, body)
        .await
        .unwrap_or_else(Denial::into_response)
}

async fn sessions(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
) -> Response {
    let Some(Extension(p)) = principal else {
        return Denial::auth_required().into_response();
    };
    st.do_sessions(&p)
}

async fn revoke(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
    UrlPath(id): UrlPath<String>,
) -> Response {
    let (Some(Extension(p)), Some(Extension(i))) = (principal, info) else {
        return Denial::auth_required().into_response();
    };
    st.do_revoke(&p, &i, &id)
        .unwrap_or_else(Denial::into_response)
}

async fn revoke_others(
    State(st): State<Arc<LoginState>>,
    principal: Option<Extension<Principal>>,
    info: Option<Extension<RequestInfo>>,
) -> Response {
    let (Some(Extension(p)), Some(Extension(i))) = (principal, info) else {
        return Denial::auth_required().into_response();
    };
    st.do_revoke_others(&p, &i)
}

/// The access mode of an `oaiy-server` built with the web login: `scoped` unless it is told another, and `legacy`
/// refused. The login needs the store on disk, which `legacy` never opens (it changes nothing under the data
/// folder), and `legacy` trusts Origins: a server with a login has moved past that (design 4.5.5 rule 6).
pub fn server_mode(value: Option<&str>) -> Result<AccessMode, ConfigRefusal> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(AccessMode::Scoped),
        Some(_) => match AccessMode::from_env(value)? {
            AccessMode::Legacy => Err(ConfigRefusal(
                "OAIY_ACCESS_MODE=legacy is refused by a server with the web login: use scoped (the default)"
                    .into(),
            )),
            mode => Ok(mode),
        },
    }
}

/// The period of the upkeep task: the throttle is written at most every five seconds when it changed.
pub const TICK_MS: u64 = FLUSH_EVERY_MS;

/// Upkeep for a running server: see [`LoginState::tick`]. Runs until the process ends.
pub async fn maintain_forever(state: Arc<LoginState>) {
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(TICK_MS)).await;
        state.tick();
    }
}

/// Whether the store of this guard can hold a login (it reads and writes `<data>/auth`).
pub fn can_host(guard: &Guard) -> bool {
    guard.store().is_persistent()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_server_with_the_web_login_defaults_to_scoped_and_refuses_legacy() {
        assert_eq!(server_mode(None), Ok(AccessMode::Scoped));
        assert_eq!(server_mode(Some("")), Ok(AccessMode::Scoped));
        assert_eq!(server_mode(Some("  ")), Ok(AccessMode::Scoped));
        assert_eq!(server_mode(Some("scoped")), Ok(AccessMode::Scoped));
        assert_eq!(server_mode(Some("SHADOW")), Ok(AccessMode::Shadow));
        let refused = server_mode(Some("legacy")).unwrap_err();
        assert!(refused.to_string().contains("legacy"), "{refused}");
        assert_eq!(refused.exit_code(), 78);
        // A value that is not a mode is refused as it always was.
        assert!(server_mode(Some("scopd"))
            .unwrap_err()
            .to_string()
            .contains("scopd"));
    }

    #[test]
    fn a_secret_is_never_shown() {
        let s = Secret("hunter2 hunter2 hunter2".into());
        assert!(!format!("{s:?}").contains("hunter2"));
    }

    #[test]
    fn the_monotonic_clock_never_reads_zero_and_never_goes_back() {
        let c = MonotonicClock::new();
        let a = c.now_ms();
        let b = c.now_ms();
        assert!(a >= 1 && b >= a);
    }
}
