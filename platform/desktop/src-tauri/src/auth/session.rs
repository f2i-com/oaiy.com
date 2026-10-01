//! Sessions, from the request's side and from the login's.
//!
//! **The cookie path of the guard** (design 3.5 steps 7 to 9, 4.5.2 and 4.7.5). A request with no `Authorization`
//! header is authenticated by the session cookie of the app its `Host` maps to, and only that one: a dashboard
//! cookie presented at the Agent host is not read, and the Agent's is a different cookie name. Only the hosts that
//! serve an app have cookies (a configured public host behind the proxy, or one of the three loopback app names);
//! `localhost`, an address and an extra host are for bearers. In order:
//!
//! 1. the cookie is looked up by name; the same name twice is `401 token_invalid` and the name is cleared
//!    (a sibling that set one of the same name); a cookie on a channel that is not secure is
//!    `403 secure_channel_required`;
//! 2. the value must have the grammar of a session token (`oaiyses_...`), else `401 token_invalid`;
//! 3. the checks of 4.5.2 (`Sec-Fetch-Site`, the exact `Origin`, `X-OAIY-CSRF` derived from the token) are made
//!    **before** anything is looked up, so a request from another site cannot even keep a session alive: `403 csrf`;
//! 4. the store verifies the token and its whole chain (idle, absolute, revoked, parent), and the session must be
//!    for this host's app.
//!
//! A bearer wins: with an `Authorization` header the cookie is not looked at (a page cannot be tricked into acting
//! with the session by adding a header, and a stolen cookie gains nothing from a bearer). The public routes that
//! read a cookie (`GET /api/auth/session`) see the anonymous answer instead of an error whenever any of this fails.
//!
//! **What the login makes of a session**: the lifetimes (idle 8 hours and absolute 24 hours, or remembered: idle 14
//! days and absolute 30 days), the fields kept about where and by which browser it was made, the elevation window
//! of 10 minutes, and the body a page is given about its own session.

use axum::http::{HeaderMap, Method, StatusCode};
use serde_json::{json, Map, Value};

use super::cookie::{self, Lookup, Style};
use super::guard::{Denial, Guard, RequestInfo};
use super::host::{expected_origin, Channel, HostName};
use super::presets::{App, Preset};
use super::principal::{ControlLevel, Principal, PrincipalKind};
use super::store::{AuthError, AuthStore, MintFailure, MintSpec, Minted, Record};
use super::token::{self, Kind};

/// The idle timeout of an ordinary session (sliding), and its absolute life.
pub const IDLE_MS: u64 = 8 * 3_600_000;
pub const ABSOLUTE_MS: u64 = 24 * 3_600_000;
/// The same when "remember this device" was ticked.
pub const REMEMBER_IDLE_MS: u64 = 14 * 24 * 3_600_000;
pub const REMEMBER_ABSOLUTE_MS: u64 = 30 * 24 * 3_600_000;
/// How long an elevation lasts.
pub const ELEVATION_MS: u64 = 10 * 60_000;
/// The user agent is cut to this many characters.
pub const MAX_UA: usize = 120;

/// What the guard and the public handlers ask of the login: they are the login's, but the guard is built before it.
pub trait LoginFacts: Send + Sync {
    /// Whether an owner exists (`owner.json`): without one the server is in setup-only mode.
    fn owner_configured(&self) -> bool;
    /// `active`, `expired` or `none`.
    fn setup_code_status(&self) -> &'static str;
    /// Whether `login.attack` holds now (the banner).
    fn under_attack(&self) -> bool;
    /// Write what is held in memory to disk (at shutdown).
    fn flush(&self);
}

// ---- what a request's host says about cookies -------------------------------------------------

/// The cookies of a request's host.
#[derive(Clone, Debug)]
pub struct CookieHost {
    pub style: Style,
    pub app: App,
    pub host: HostName,
    pub channel: Channel,
    /// The origin a page served for this host has: what `Origin` must equal.
    pub own_origin: String,
}

impl CookieHost {
    /// `None` for a host that does not serve an app.
    pub fn of(info: &RequestInfo) -> Option<CookieHost> {
        let host = HostName::parse(&info.host)?;
        let class = info.host_class.as_ref()?;
        let (style, app) = Style::of_host(class, &host)?;
        Some(CookieHost {
            style,
            app,
            channel: info.channel,
            own_origin: expected_origin(&host, info.channel),
            host,
        })
    }

    pub fn session_cookie(&self) -> String {
        self.style.session_name(self.app)
    }

    pub fn device_cookie(&self) -> String {
        self.style.device_name()
    }

    /// The `Set-Cookie` that ends this host's session cookie.
    pub fn clear_session(&self) -> String {
        self.style.clear(&self.session_cookie())
    }

    /// The `Set-Cookie` that ends the device cookie.
    pub fn clear_device(&self) -> String {
        self.style.clear(&self.device_cookie())
    }

    pub fn secure(&self) -> bool {
        self.channel == Channel::Secure
    }
}

// ---- the guard's cookie path ------------------------------------------------------------------

fn refuse(status: StatusCode, code: &'static str, message: &str, clear: Option<String>) -> Denial {
    let mut d = Denial::new(status, code, message);
    d.noise = Some("auth.denied");
    if let Some(c) = clear {
        d.headers.push(("set-cookie", c));
    }
    d
}

fn ended(e: &AuthError, ch: &CookieHost) -> Denial {
    let mut d = Denial::from_auth_error(e);
    // The browser is told to forget a cookie that is no use to it.
    d.headers.push(("set-cookie", ch.clear_session()));
    d
}

/// What reading the session cookie found.
enum Read {
    /// No cookie for this host's app (or cookies are not enabled here).
    None,
    /// A cookie: its value.
    Cookie(String),
    /// The name is there twice.
    Duplicate,
}

fn read_cookie(guard: &Guard, headers: &HeaderMap, ch: &CookieHost) -> Read {
    if guard.login().is_none() {
        return Read::None;
    }
    match cookie::find(headers, &ch.session_cookie()) {
        Lookup::Absent => Read::None,
        Lookup::One(v) => Read::Cookie(v),
        Lookup::Duplicate => Read::Duplicate,
    }
}

/// Steps 7 to 9 for a request with no `Authorization` header. `Ok(None)` is an anonymous request.
pub fn authenticate_cookie(
    guard: &Guard,
    headers: &HeaderMap,
    method: &Method,
    info: &RequestInfo,
) -> Result<Option<Principal>, Denial> {
    let Some(ch) = CookieHost::of(info) else {
        // A host that serves no app has no cookies to read. On a connection that cannot carry one (plain HTTP
        // from the network: a lan listener) a request that presents a session cookie of ours is refused and
        // told why, not quietly treated as anonymous (design 4.5.4).
        if info.channel == Channel::Insecure
            && guard.login().is_some()
            && cookie::carries_a_session_cookie(headers)
        {
            return Err(refuse(
                StatusCode::FORBIDDEN,
                "secure_channel_required",
                "A cookie is accepted only over https, or from the machine itself.",
                None,
            ));
        }
        return Ok(None);
    };
    let value = match read_cookie(guard, headers, &ch) {
        Read::None => return Ok(None),
        Read::Duplicate => {
            return Err(ended(&AuthError::Invalid, &ch));
        }
        Read::Cookie(v) => v,
    };
    if !ch.secure() {
        return Err(refuse(
            StatusCode::FORBIDDEN,
            "secure_channel_required",
            "A cookie is accepted only over https, or from the machine itself.",
            None,
        ));
    }
    let parsed = match token::parse(&value) {
        Some(p) if p.kind == Kind::Ses => p,
        _ => return Err(ended(&AuthError::Invalid, &ch)),
    };
    // Before anything is looked up: a request from another site does not even keep a session alive.
    let csrf = token::csrf_value(parsed.secret).unwrap_or_default();
    if let Err(fail) = cookie::check_session_request(method, headers, &ch.own_origin, &csrf) {
        return Err(refuse(StatusCode::FORBIDDEN, "csrf", fail.message(), None));
    }
    match guard.store().authenticate(&value, Some(&info.client_ip)) {
        Ok(p) if p.kind == PrincipalKind::Session && p.app == Some(ch.app) => Ok(Some(p)),
        // A session of another app in this app's cookie: not this host's session.
        Ok(_) => Err(ended(&AuthError::Invalid, &ch)),
        Err(e) => Err(ended(&e, &ch)),
    }
}

/// The session of a request to a public route that reads one (`GET /api/auth/session`): the anonymous answer
/// for everything that would be an error on another route, and for a request from another site.
pub fn peek_session(
    guard: &Guard,
    headers: &HeaderMap,
    info: &RequestInfo,
) -> Option<(Principal, String)> {
    let ch = CookieHost::of(info)?;
    let Read::Cookie(value) = read_cookie(guard, headers, &ch) else {
        return None;
    };
    if !ch.secure() || !cookie::fetch_site_ok(headers, &Method::GET) {
        return None;
    }
    let parsed = token::parse(&value).filter(|p| p.kind == Kind::Ses)?;
    let csrf = token::csrf_value(parsed.secret)?;
    match guard.store().authenticate(&value, Some(&info.client_ip)) {
        Ok(p) if p.kind == PrincipalKind::Session && p.app == Some(ch.app) => Some((p, csrf)),
        _ => None,
    }
}

// ---- what the login makes of a session --------------------------------------------------------

/// Make a dashboard session: `owner` preset, the lifetimes of the standard or the remembered kind, and where and by
/// which browser it was made. The store limits the number and never needs a write to succeed (a session made on a
/// full disk lives in memory and says so).
pub fn mint_session(
    store: &AuthStore,
    remember: bool,
    ip: &str,
    user_agent: &str,
) -> Result<Minted, MintFailure> {
    let (ttl, idle) = if remember {
        (REMEMBER_ABSOLUTE_MS, REMEMBER_IDLE_MS)
    } else {
        (ABSOLUTE_MS, IDLE_MS)
    };
    let mut spec = MintSpec::new(Kind::Ses, "dashboard", Preset::Owner.scopes(), ttl);
    spec.app = Some(App::Dash);
    spec.preset = Some(Preset::Owner);
    spec.idle_ms = Some(idle);
    let mut made = Map::new();
    made.insert("ip".into(), json!(ip.chars().take(64).collect::<String>()));
    made.insert(
        "ua".into(),
        json!(user_agent.chars().take(MAX_UA).collect::<String>()),
    );
    made.insert("remember".into(), json!(remember));
    spec.extra.insert("login".into(), Value::Object(made));
    store.mint(spec)
}

/// When a session goes idle, if it stays unused: its last use (or its creation) and its idle timeout.
pub fn idle_expires_ms(rec: &Record) -> u64 {
    rec.last_used_ms
        .unwrap_or(rec.created_ms)
        .saturating_add(rec.idle_ms.unwrap_or(IDLE_MS))
}

/// The end of the session's elevation if it is still running, else zero, as a wall-clock time (`now_ms` and what is
/// left of the window on the elevation clock: a step of the wall clock moves the answer, not the window).
pub fn elevated_until_ms(store: &AuthStore, id: &str, now_ms: u64) -> u64 {
    store
        .elevated_until(id)
        .and_then(|until| until.checked_sub(store.elevation_now_ms()))
        .filter(|left| *left > 0)
        .map_or(0, |left| now_ms.saturating_add(left))
}

/// Elevate a session for ten minutes from now: the window runs on the store's elevation clock (a monotonic one on a
/// server). The wall-clock time it ends at, from `now_ms`, for the answer.
pub fn elevate(store: &AuthStore, id: &str, now_ms: u64) -> u64 {
    store.set_elevated_until(id, store.elevation_now_ms().saturating_add(ELEVATION_MS));
    now_ms.saturating_add(ELEVATION_MS)
}

fn control_level_name(p: &Principal) -> &'static str {
    match p.control_level() {
        ControlLevel::None => "none",
        ControlLevel::Read => "read",
        ControlLevel::Project => "project",
    }
}

/// What a signed-in page is told about its own session: `csrf` is the value it sends in `X-OAIY-CSRF`.
pub fn session_json(
    store: &AuthStore,
    p: &Principal,
    csrf: &str,
    now_ms: u64,
    with_ok: bool,
) -> Value {
    let rec = store.record(&p.id);
    let app = p.app.map_or("dash", |a| a.name());
    let mut body = Map::new();
    if with_ok {
        body.insert("ok".into(), json!(true));
    } else {
        body.insert("authenticated".into(), json!(true));
    }
    body.insert("app".into(), json!(app));
    body.insert("csrf".into(), json!(csrf));
    body.insert("expiresMs".into(), json!(p.expires_ms.unwrap_or(0)));
    body.insert(
        "idleExpiresMs".into(),
        json!(rec.as_ref().map_or(0, idle_expires_ms)),
    );
    body.insert(
        "elevatedUntilMs".into(),
        json!(elevated_until_ms(store, &p.id, now_ms)),
    );
    body.insert("scopes".into(), json!(p.scopes.names()));
    if with_ok {
        body.insert("persisted".into(), json!(p.persisted));
        body.insert("next".into(), Value::Null);
    } else {
        body.insert("controlLevel".into(), json!(control_level_name(p)));
        body.insert("persisted".into(), json!(p.persisted));
    }
    Value::Object(body)
}
