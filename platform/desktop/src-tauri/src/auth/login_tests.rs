#![cfg(feature = "web")]
//! The web login, end to end in process: the real guard and the real login routes over a real credential store on
//! a temporary data folder, with a fake clock and a cheap (but real) Argon2id.
//!
//! Every normative rule of 4.7 has a test named after it here or in the file of the part that holds it
//! (`password`, `policy`, `throttle`, `lanes`, `setup`, `device`, `cookie`, `cookie_guard_tests`): the cookies and the
//! CSRF vector, the login and the logout cascade, the password change and its revocations, the elevation window, the
//! lanes (T9, T43), the setup code (T10), the answers that must not differ, the disk-full login (T27), the memory
//! bound, and the console's routes (T42). The console's command line, against a real server and a stopped one, is in
//! `tests/login_e2e.rs`.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, Method, Request};
use axum::middleware;
use axum::routing::any;
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::api;
use super::audit::{AuditLog, LogFile};
use super::bearer_throttle::ThrottleFile;
use super::clock::{Clock, ManualClock};
use super::console;
use super::guard::{scoped_cors, scoped_guard, Guard, GuardConfig};
use super::login::{self, LoginOptions, LoginState};
use super::mode::AccessMode;
use super::password::{Argon2Engine, Cost, HashError, PasswordEngine, Verdict};
use super::routes::ROUTES;
use super::session::LoginFacts;
use super::store::{AuthStore, FileWriter, Host, MintSpec, SecureWriter};
use super::token::{self, Kind};
use crate::secret_file::testing::TempDir;

const T0: u64 = 1_790_000_000_000;
const MIN: u64 = 60_000;
const HOUR: u64 = 60 * MIN;
const PASSWORD: &str = "k7Qz!mV3#pW9xLd2 rn8Tb";
const NEW_PASSWORD: &str = "Hv4$wN6@cJ1&zX8 qs5Fe";

#[cfg(unix)]
const ENOSPC: i32 = 28;
#[cfg(windows)]
const ENOSPC: i32 = 112;

/// The routes the login builds: everything else in the table is a stub.
const REAL: [&str; 16] = [
    "/api/auth/info",
    "/api/auth/whoami",
    "/api/auth/derive",
    "/api/auth/login",
    "/api/auth/setup",
    "/api/auth/link",
    "/api/auth/session",
    "/api/auth/logout",
    "/api/auth/elevate",
    "/api/auth/password",
    "/api/auth/sessions",
    "/api/auth/sessions/revoke-others",
    "/api/auth/sessions/:id",
    "/api/auth/console/reset-password",
    "/api/auth/console/setup-code",
    "/api/auth/console/session-link",
];
const REAL_MORE: [&str; 2] = [
    "/api/auth/console/sessions/revoke-all",
    "/api/auth/console/status",
];

/// A disk that can be made full, and slow for what holds a marker.
struct Disk {
    full: AtomicBool,
    writes: AtomicUsize,
    /// A write of bytes that hold this marker sleeps this long before it writes.
    slow: std::sync::Mutex<Option<(Vec<u8>, Duration)>>,
}

impl FileWriter for Disk {
    fn write(&self, path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        let slow = self.slow.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((marker, wait)) = slow {
            if bytes.windows(marker.len()).any(|w| w == marker.as_slice()) {
                // On a worker of a multi-threaded runtime the wait hands the worker (and the task that its
                // wake-ups have queued behind this one) to another thread: a busy disk does not stop the others.
                match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
                    Ok(tokio::runtime::RuntimeFlavor::MultiThread) => {
                        tokio::task::block_in_place(|| std::thread::sleep(wait))
                    }
                    _ => std::thread::sleep(wait),
                }
            }
        }
        if self.full.load(Ordering::SeqCst) {
            return Err(std::io::Error::from_raw_os_error(ENOSPC));
        }
        SecureWriter.write(path, bytes)
    }
}

/// Argon2id at the cheapest cost the verifier accepts, counting what it is asked.
struct Engine {
    inner: Argon2Engine,
    verifies: AtomicUsize,
    hashes: AtomicUsize,
    /// How many run at once, at most (for the memory bound).
    running: AtomicUsize,
    most: AtomicUsize,
    /// Held for this long inside each call, to make them overlap.
    hold: Duration,
}

impl Engine {
    fn new(hold: Duration) -> Arc<Engine> {
        Arc::new(Engine {
            inner: Argon2Engine::with(Cost::CHEAPEST, Arc::new(token::os_random)),
            verifies: AtomicUsize::new(0),
            hashes: AtomicUsize::new(0),
            running: AtomicUsize::new(0),
            most: AtomicUsize::new(0),
            hold,
        })
    }

    fn enter(&self) {
        let n = self.running.fetch_add(1, Ordering::SeqCst) + 1;
        self.most.fetch_max(n, Ordering::SeqCst);
        if !self.hold.is_zero() {
            std::thread::sleep(self.hold);
        }
    }

    fn leave(&self) {
        self.running.fetch_sub(1, Ordering::SeqCst);
    }
}

impl PasswordEngine for Engine {
    fn hash(&self, password: &[u8]) -> Result<String, HashError> {
        self.hashes.fetch_add(1, Ordering::SeqCst);
        self.enter();
        let r = self.inner.hash(password);
        self.leave();
        r
    }

    fn verify(&self, password: &[u8], stored: &str) -> Verdict {
        self.verifies.fetch_add(1, Ordering::SeqCst);
        self.enter();
        let r = self.inner.verify(password, stored);
        self.leave();
        r
    }
}

struct Env {
    app: Router,
    state: Arc<LoginState>,
    guard: Arc<Guard>,
    store: Arc<AuthStore>,
    clock: Arc<ManualClock>,
    mono: Arc<ManualClock>,
    engine: Arc<Engine>,
    disk: Arc<Disk>,
    dir: TempDir,
    proxied: bool,
}

struct Build {
    proxied: bool,
    login_allow: Option<&'static str>,
    hold: Duration,
    /// Reuse a data folder (a restart).
    dir: Option<TempDir>,
    /// A time to start the clock at (a restart later).
    at: u64,
    static_token: Option<&'static str>,
}

impl Default for Build {
    fn default() -> Self {
        Build {
            proxied: true,
            login_allow: None,
            hold: Duration::ZERO,
            dir: None,
            at: T0,
            static_token: None,
        }
    }
}

fn build(b: Build) -> Env {
    let dir = b.dir.unwrap_or_else(|| TempDir::new("login-tests"));
    let clock = Arc::new(ManualClock::new(b.at));
    let mono = Arc::new(ManualClock::new(1_000));
    let disk = Arc::new(Disk {
        full: AtomicBool::new(false),
        writes: AtomicUsize::new(0),
        slow: std::sync::Mutex::new(None),
    });
    let mut vars: std::collections::BTreeMap<String, String> = Default::default();
    if b.proxied {
        vars.insert("OAIY_PUBLIC_URL".into(), "https://dash.example.com".into());
        vars.insert("OAIY_AGENT_URL".into(), "https://agent.example.com".into());
        vars.insert("OAIY_FLOWS_URL".into(), "https://flows.example.com".into());
    }
    if let Some(list) = b.login_allow {
        vars.insert("OAIY_LOGIN_ALLOW".into(), list.into());
    }
    let env_vars = vars.clone();
    let (config, warnings) =
        GuardConfig::from_env(&move |n| vars.get(n).cloned(), false, false, 41000);
    assert!(warnings.is_empty(), "{warnings:?}");
    let auth = dir.0.join("auth");
    let audit = Arc::new(AuditLog::open(&auth, clock.clone(), false));
    let store = Arc::new(
        AuthStore::open(
            &auth,
            Host::Server,
            clock.clone(),
            disk.clone(),
            Some(audit.clone()),
        )
        .expect("the store opens"),
    );
    let guard = Arc::new(Guard::new(
        AccessMode::Scoped,
        config,
        store.clone(),
        b.static_token.map(str::to_owned),
        Some(audit),
        clock.clone(),
    ));
    // What the runtime does when it builds a guard that keeps its data in a folder.
    let (throttle_file, saved) = ThrottleFile::open(&auth.join("throttle.json"));
    guard.keep_throttle_in(throttle_file, saved.as_ref());
    let engine = Engine::new(b.hold);
    let mut opts =
        LoginOptions::production(&move |n| env_vars.get(n).cloned(), 41000).expect("the options");
    opts.engine = engine.clone();
    opts.writer = disk.clone();
    opts.clock = clock.clone();
    opts.mono = mono.clone();
    let state = login::enable(&guard, &auth, opts).expect("the login is enabled");
    let mut patterns: Vec<&str> = ROUTES
        .iter()
        .map(|r| r.pattern)
        .filter(|p| !REAL.contains(p) && !REAL_MORE.contains(p))
        .collect();
    patterns.sort_unstable();
    patterns.dedup();
    let mut app = Router::new();
    for p in patterns {
        app = app.route(p, any(|| async { "ok" }));
    }
    let app = app
        .merge(api::router(guard.clone()))
        .merge(login::router(state.clone()))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_guard))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_cors));
    Env {
        app,
        state,
        guard,
        store,
        clock,
        mono,
        engine,
        disk,
        dir,
        proxied: b.proxied,
    }
}

fn env() -> Env {
    build(Build::default())
}

/// Stop the server and start it again on the same data folder, later_ms later: everything held in memory is gone.
fn restart(e: Env, later_ms: u64) -> Env {
    let at = e.clock.now_ms() + later_ms;
    let proxied = e.proxied;
    let Env {
        app,
        state,
        guard,
        store,
        dir,
        ..
    } = e;
    // The lock on the folder is held by the store: it goes with the last of these.
    drop((app, state, guard, store));
    build(Build {
        dir: Some(dir),
        at,
        proxied,
        ..Build::default()
    })
}

// ---- requests ------------------------------------------------------------------------------------

/// Where a request comes from: the dashboard host as a browser reaches it, from a client address.
#[derive(Clone)]
struct Req {
    method: Method,
    path: String,
    host: String,
    /// The client address the proxy reports (proxied) or the socket's peer (loopback).
    from: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    proxied: bool,
}

fn req(env: &Env, method: Method, path: &str) -> Req {
    Req {
        method,
        path: path.to_string(),
        host: if env.proxied {
            "dash.example.com".into()
        } else {
            "dash.oaiy.localhost:41000".into()
        },
        from: if env.proxied {
            "203.0.113.9".into()
        } else {
            "127.0.0.1".into()
        },
        headers: Vec::new(),
        body: Vec::new(),
        proxied: env.proxied,
    }
}

impl Req {
    fn host(mut self, host: &str) -> Req {
        self.host = host.into();
        self
    }

    fn from(mut self, ip: &str) -> Req {
        self.from = ip.into();
        self
    }

    fn h(mut self, name: &str, value: &str) -> Req {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.headers.push((name.into(), value.into()));
        self
    }

    fn drop_h(mut self, name: &str) -> Req {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self
    }

    fn json(mut self, body: Value) -> Req {
        self.body = body.to_string().into_bytes();
        self.h("content-type", "application/json")
    }

    fn raw(mut self, body: &[u8]) -> Req {
        self.body = body.to_vec();
        self
    }

    fn cookie(self, pair: &str) -> Req {
        let existing = self
            .headers
            .iter()
            .find(|(n, _)| n == "cookie")
            .map(|(_, v)| format!("{v}; "))
            .unwrap_or_default();
        self.h("cookie", &format!("{existing}{pair}"))
    }

    /// What the page at the host sends with a mutation it makes.
    fn page(self, csrf: &str) -> Req {
        let origin = self.origin();
        self.h("origin", &origin)
            .h("sec-fetch-site", "same-origin")
            .h("x-oaiy-csrf", csrf)
    }

    fn origin(&self) -> String {
        if self.proxied {
            format!("https://{}", self.host)
        } else {
            format!("http://{}", self.host)
        }
    }

    /// A browser's form of the request (Origin and Fetch Metadata) without a session.
    fn browser(self) -> Req {
        let origin = self.origin();
        self.h("origin", &origin).h("sec-fetch-site", "same-origin")
    }

    /// As the console sends it: a bearer, from the machine itself, no Origin.
    fn bearer(mut self, token: &str) -> Req {
        self.proxied = false;
        self.host("127.0.0.1:41000")
            .h("authorization", &format!("Bearer {token}"))
    }
}

struct Reply {
    status: u16,
    headers: HeaderMap,
    text: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap_or(Value::Null)
    }

    fn code(&self) -> Option<String> {
        self.json()["error"]["code"].as_str().map(str::to_owned)
    }

    fn cookies(&self) -> Vec<String> {
        self.headers
            .get_all("set-cookie")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect()
    }

    /// The value of the cookie called `name` that this reply sets.
    fn cookie(&self, name: &str) -> Option<String> {
        self.cookies().into_iter().find_map(|c| {
            c.strip_prefix(&format!("{name}="))
                .map(|rest| rest.split(';').next().unwrap_or("").to_string())
        })
    }
}

async fn go(env: &Env, r: Req) -> Reply {
    let mut builder = Request::builder().method(r.method.clone()).uri(&r.path);
    builder = builder.header("host", &r.host);
    let peer: SocketAddr = if r.proxied {
        builder = builder
            .header("x-forwarded-for", &r.from)
            .header("x-forwarded-proto", "https");
        "127.0.0.1:50000".parse().unwrap()
    } else if r.host.starts_with("127.0.0.1") {
        // The console: the machine itself, no forwarded header.
        "127.0.0.1:50000".parse().unwrap()
    } else if r.from.contains(':') {
        format!("[{}]:50000", r.from).parse().unwrap()
    } else {
        format!("{}:50000", r.from).parse().unwrap()
    };
    for (n, v) in &r.headers {
        builder = builder.header(n.as_str(), v.as_str());
    }
    let mut request = builder.body(Body::from(r.body.clone())).unwrap();
    request.extensions_mut().insert(ConnectInfo(peer));
    let response = env.app.clone().oneshot(request).await.unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let text = String::from_utf8_lossy(
        &axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap(),
    )
    .into_owned();
    Reply {
        status,
        headers,
        text,
    }
}

fn session_cookie_name(env: &Env) -> &'static str {
    if env.proxied {
        "__Host-oaiy_dash"
    } else {
        "oaiy_dash_41000"
    }
}

fn device_cookie_name(env: &Env) -> &'static str {
    if env.proxied {
        "__Host-oaiy_dev"
    } else {
        "oaiy_dev_41000"
    }
}

// ---- helpers that put the install in a state -------------------------------------------------------

/// Make an owner through the real setup route (a console-made code, then the browser's `POST /api/auth/setup`).
async fn make_owner(env: &Env, password: &str) -> Reply {
    let code = env.state.setup.make(&env.state.random).unwrap();
    let r = go(
        env,
        req(env, Method::POST, "/api/auth/setup")
            .browser()
            .json(json!({ "code": code, "password": password })),
    )
    .await;
    assert_eq!(r.status, 201, "{}", r.text);
    r
}

/// A signed-in browser: its session cookie value, the csrf value and the device cookie.
struct Browser {
    session: String,
    csrf: String,
    device: Option<String>,
}

fn browser_from(env: &Env, r: &Reply) -> Browser {
    let session = r
        .cookie(session_cookie_name(env))
        .expect("a session cookie");
    Browser {
        csrf: r.json()["csrf"].as_str().expect("a csrf value").to_string(),
        device: r.cookie(device_cookie_name(env)),
        session,
    }
}

async fn login_as(env: &Env, password: &str, from: &str) -> Reply {
    go(
        env,
        req(env, Method::POST, "/api/auth/login")
            .from(from)
            .browser()
            .json(json!({ "password": password })),
    )
    .await
}

/// A request a signed-in page makes.
fn as_page(env: &Env, b: &Browser, method: Method, path: &str) -> Req {
    req(env, method, path)
        .cookie(&format!("{}={}", session_cookie_name(env), b.session))
        .page(&b.csrf)
}

/// A wrong password from `from`.
async fn wrong(env: &Env, from: &str) -> Reply {
    login_as(env, "not the password at all", from).await
}

/// The device cookie as a request header pair.
fn device_pair(env: &Env, b: &Browser) -> String {
    format!(
        "{}={}",
        device_cookie_name(env),
        b.device.as_ref().expect("a device cookie")
    )
}

fn audit_events(env: &Env) -> Vec<Value> {
    // Newest first.
    let log = AuditLog::open(&env.dir.0.join("auth"), env.clock.clone(), false);
    log.read(LogFile::Audit, 1000, None, None)
}

fn has_event(env: &Env, name: &str) -> bool {
    audit_events(env).iter().any(|e| e["event"] == name)
}

// ==================================== login ==========================================================

#[tokio::test]
async fn a_login_makes_a_session_with_the_exact_cookies_and_body_behind_a_proxy() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    e.clock.advance(HOUR);
    let r = login_as(&e, PASSWORD, "203.0.113.9").await;
    assert_eq!(r.status, 200, "{}", r.text);
    let cookies = r.cookies();
    assert_eq!(cookies.len(), 2, "{cookies:?}");
    let session = r.cookie("__Host-oaiy_dash").unwrap();
    // The exact attributes, in order: a session cookie has no Max-Age.
    assert_eq!(
        cookies[0],
        format!("__Host-oaiy_dash={session}; Path=/; Secure; HttpOnly; SameSite=Strict")
    );
    // The device cookie is set because this browser had none: 180 days.
    let device = r.cookie("__Host-oaiy_dev").unwrap();
    assert_eq!(
        cookies[1],
        format!(
            "__Host-oaiy_dev={device}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=15552000"
        )
    );
    assert!(session.starts_with("oaiyses_") && device.starts_with("oaiydev_"));
    assert_eq!((session.len(), device.len()), (68, 68));
    let body = r.json();
    assert_eq!(body["ok"], true);
    assert_eq!(body["app"], "dash");
    assert_eq!(body["next"], Value::Null);
    assert_eq!(body["persisted"], true);
    assert_eq!(body["scopes"].as_array().unwrap().len(), 54);
    // The csrf value is the one derived from the token secret (the vector's rule), and it works.
    let secret = token::parse(&session).unwrap().secret;
    assert_eq!(body["csrf"], token::csrf_value(secret).unwrap());
    // Standard lifetimes: idle 8 hours, absolute 24 hours, from now; a login counts as an elevation.
    let now = e.clock.now_ms();
    assert_eq!(body["expiresMs"], now + 24 * HOUR);
    assert_eq!(body["idleExpiresMs"], now + 8 * HOUR);
    assert_eq!(body["elevatedUntilMs"], now + 10 * MIN);
    // Cache-Control and nosniff on the answer.
    assert_eq!(r.headers.get("cache-control").unwrap(), "no-store");
    // No secret in the answer beyond the two it must carry.
    assert!(!r.text.contains(PASSWORD));
}

#[tokio::test]
async fn a_login_on_a_loopback_server_uses_the_port_in_the_cookie_name_and_no_secure() {
    let e = build(Build {
        proxied: false,
        ..Build::default()
    });
    make_owner(&e, PASSWORD).await;
    let r = login_as(&e, PASSWORD, "127.0.0.1").await;
    assert_eq!(r.status, 200, "{}", r.text);
    let cookies = r.cookies();
    let session = r.cookie("oaiy_dash_41000").unwrap();
    assert_eq!(
        cookies[0],
        format!("oaiy_dash_41000={session}; Path=/; HttpOnly; SameSite=Strict")
    );
    let device = r.cookie("oaiy_dev_41000").unwrap();
    assert_eq!(
        cookies[1],
        format!("oaiy_dev_41000={device}; Path=/; HttpOnly; SameSite=Strict; Max-Age=15552000")
    );
    assert!(cookies.iter().all(|c| !c.contains("Secure")));
}

#[tokio::test]
async fn remember_this_device_is_thirty_days_and_idle_fourteen() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .browser()
            .json(json!({ "password": PASSWORD, "remember": true })),
    )
    .await;
    assert_eq!(r.status, 200);
    let session = r.cookie("__Host-oaiy_dash").unwrap();
    assert_eq!(
        r.cookies()[0],
        format!("__Host-oaiy_dash={session}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=2592000")
    );
    let now = e.clock.now_ms();
    assert_eq!(r.json()["expiresMs"], now + 30 * 24 * HOUR);
    assert_eq!(r.json()["idleExpiresMs"], now + 14 * 24 * HOUR);
    let id = token::parse(&session).unwrap().id;
    let rec = e.store.record(id).unwrap();
    assert_eq!(
        (rec.idle_ms, rec.expires_ms - rec.created_ms),
        (Some(14 * 24 * HOUR), 30 * 24 * HOUR)
    );
}

#[tokio::test]
async fn login_is_for_the_dashboard_host_only_over_a_secure_channel_from_the_same_origin() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let ok = json!({ "password": PASSWORD });
    // Another app's host and a host that serves no app: 404, and nothing was verified.
    let before = e.engine.verifies.load(Ordering::SeqCst);
    for host in ["agent.example.com", "flows.example.com"] {
        let r = go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .host(host)
                .browser()
                .json(ok.clone()),
        )
        .await;
        assert_eq!(r.status, 404, "{host}");
    }
    assert_eq!(e.engine.verifies.load(Ordering::SeqCst), before);
    // A cross-origin browser: another origin, a sibling, null; Fetch Metadata that says another site.
    for origin in [
        "https://evil.example",
        "https://agent.example.com",
        "null",
        "http://dash.example.com",
    ] {
        let r = go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .h("origin", origin)
                .json(ok.clone()),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("csrf")),
            "Origin {origin}"
        );
    }
    for site in ["same-site", "cross-site", "none"] {
        let r = go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .h("sec-fetch-site", site)
                .json(ok.clone()),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("csrf")),
            "{site}"
        );
    }
    assert_eq!(
        e.engine.verifies.load(Ordering::SeqCst),
        before,
        "refused before any hash"
    );
    // A client that sends neither (curl) is let through: login is testable with curl.
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login").json(ok.clone()),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    // A channel that is not secure: the proxy did not say https.
    let mut plain = req(&e, Method::POST, "/api/auth/login").json(ok.clone());
    plain.proxied = false;
    plain.from = "203.0.113.9".into();
    let r = go(&e, plain).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("secure_channel_required"))
    );
}

#[tokio::test]
async fn login_takes_json_of_at_most_sixteen_kib_and_says_nothing_of_what_was_wrong() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let post = |e: &Env| req(e, Method::POST, "/api/auth/login").browser();
    let r = go(&e, post(&e).raw(br#"{"password":"x"}"#)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (415, Some("unsupported_media_type"))
    );
    let r = go(
        &e,
        post(&e)
            .h("content-type", "text/plain")
            .raw(br#"{"password":"x"}"#),
    )
    .await;
    assert_eq!(r.status, 415);
    for body in [
        &b"not json"[..],
        b"{}",
        b"[]",
        b"null",
        br#"{"password":12345}"#,
        br#"{"password":null}"#,
        b"",
    ] {
        let r = go(&e, post(&e).h("content-type", "application/json").raw(body)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (400, Some("bad_request")),
            "{body:?}"
        );
        assert!(
            !r.text.contains("12345"),
            "the body is not echoed: {}",
            r.text
        );
    }
    // 16 KiB is the most: one byte over is refused; nothing was verified.
    let before = e.engine.verifies.load(Ordering::SeqCst);
    let big = json!({ "password": "p".repeat(20_000) });
    let r = go(&e, post(&e).json(big)).await;
    assert_eq!((r.status, r.code().as_deref()), (400, Some("bad_request")));
    assert_eq!(e.engine.verifies.load(Ordering::SeqCst), before);
    // Application/json with parameters is fine.
    let r = go(
        &e,
        post(&e)
            .h("content-type", "application/json; charset=utf-8")
            .raw(json!({ "password": PASSWORD }).to_string().as_bytes()),
    )
    .await;
    assert_eq!(r.status, 200);
}

#[tokio::test]
async fn with_no_owner_a_login_is_409_setup_required_and_nothing_else_says_so() {
    let e = env();
    let r = login_as(&e, PASSWORD, "203.0.113.9").await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (409, Some("setup_required"))
    );
    assert_eq!(e.engine.verifies.load(Ordering::SeqCst), 0);
    // A route that needs a login says 401 setup_required (4.7.1).
    let r = go(&e, req(&e, Method::GET, "/api/config")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("setup_required"))
    );
    // The session and info answers are public and say the truth, without the code.
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(
        (
            info["loginConfigured"].as_bool(),
            info["setupCode"].as_str()
        ),
        (Some(false), Some("none"))
    );
    assert_eq!(info["scheme"], "oaiy-auth/1");
    assert_eq!(info["secureChannel"], true);
}

// ==================================== no difference in the answers ===================================

/// Everything a client can see of a reply, but the cookies it sets.
fn shape(r: &Reply) -> (u16, Value, Vec<(String, String)>) {
    let mut headers: Vec<(String, String)> = r
        .headers
        .iter()
        .filter(|(n, _)| *n != "set-cookie" && *n != "date")
        .map(|(n, v)| (n.to_string(), v.to_str().unwrap().to_string()))
        .collect();
    headers.sort();
    (r.status, r.json(), headers)
}

#[tokio::test]
async fn every_failed_login_is_the_same_401_and_runs_one_verification() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let dev = b.device.clone().unwrap();
    let parsed = token::parse(&dev).unwrap();
    // The cases: a wrong password from a browser with no device; the same with a device cookie whose id is
    // unknown, whose secret is wrong, that is a session token, that is junk; and a wrong password on the device
    // lane. An empty password and one of 5,000 characters are wrong passwords too.
    let unknown_id = format!("oaiydev_{}_{}", "e".repeat(16), parsed.secret);
    let wrong_secret = format!("oaiydev_{}_{}", parsed.id, "A".repeat(43));
    let as_session = format!("oaiyses_{}_{}", parsed.id, parsed.secret);
    let cases: Vec<(&str, Option<String>, String)> = vec![
        ("wrong password", None, "not the password at all".into()),
        (
            "unknown device id",
            Some(unknown_id),
            "not the password at all".into(),
        ),
        (
            "wrong device secret",
            Some(wrong_secret),
            "not the password at all".into(),
        ),
        (
            "a session token as a device",
            Some(as_session),
            "not the password at all".into(),
        ),
        (
            "junk device cookie",
            Some("garbage".into()),
            "not the password at all".into(),
        ),
        (
            "wrong password on the device lane",
            Some(dev.clone()),
            "not the password at all".into(),
        ),
        ("an empty password", None, String::new()),
        ("a very long password", None, "x".repeat(5_000)),
    ];
    let mut shapes = Vec::new();
    for (i, (what, device, password)) in cases.iter().enumerate() {
        // Each from its own address, so that no block is in the way.
        let from = format!("198.51.100.{}", i + 1);
        let mut r = req(&e, Method::POST, "/api/auth/login")
            .from(&from)
            .browser();
        if let Some(d) = device {
            r = r.cookie(&format!("{}={d}", device_cookie_name(&e)));
        }
        let before = e.engine.verifies.load(Ordering::SeqCst);
        let reply = go(&e, r.json(json!({ "password": password }))).await;
        assert_eq!(
            e.engine.verifies.load(Ordering::SeqCst),
            before + 1,
            "{what}: one verification, as for a wrong password"
        );
        assert_eq!(
            (reply.status, reply.code().as_deref()),
            (401, Some("invalid_credentials")),
            "{what}"
        );
        assert!(
            reply.cookies().is_empty(),
            "{what}: no cookie is set by a failure"
        );
        shapes.push((what.to_string(), shape(&reply)));
    }
    // The answers are byte for byte the same: status, body and headers.
    for (what, s) in &shapes[1..] {
        assert_eq!(*s, shapes[0].1, "{what} differs from a wrong password");
    }
}

#[tokio::test]
async fn an_unusable_stored_hash_answers_as_a_wrong_password_and_still_verifies_once() {
    // An owner file whose password hash is garbage, and one whose hash asks for far too much memory.
    for stored in [
        "not a hash at all".to_string(),
        "$argon2id$v=19$m=4294967295,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4".to_string(),
        "$argon2i$v=19$m=65536,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4".to_string(),
    ] {
        let dir = TempDir::new("login-bad-hash");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let doc = json!({ "v": 1, "created_ms": T0, "password_changed_ms": T0, "password": stored });
        std::fs::write(auth.join("owner.json"), doc.to_string()).unwrap();
        let e = build(Build {
            dir: Some(dir),
            ..Build::default()
        });
        assert!(e.state.owner_configured());
        let r = login_as(&e, PASSWORD, "203.0.113.9").await;
        assert_eq!((r.status, r.code().as_deref()), (401, Some("invalid_credentials")), "{stored}");
        assert_eq!(e.engine.verifies.load(Ordering::SeqCst), 1, "{stored}");
        let wrong = login_as(&e, "some other password", "203.0.113.10").await;
        assert_eq!(shape(&r), shape(&wrong));
    }
}

// ==================================== the throttle over HTTP (T9) ====================================

#[tokio::test]
async fn t9_five_failures_block_an_address_for_fifteen_minutes_and_a_blocked_request_hashes_nothing(
) {
    let e = env();
    make_owner(&e, PASSWORD).await;
    for i in 1..=5 {
        let r = wrong(&e, "203.0.113.50").await;
        assert_eq!(r.status, 401, "attempt {i}");
    }
    let verifies = e.engine.verifies.load(Ordering::SeqCst);
    // The sixth is blocked: even the right password, at once, with no verification.
    let r = login_as(&e, PASSWORD, "203.0.113.50").await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    assert_eq!(r.headers.get("retry-after").unwrap(), "900");
    assert_eq!(r.json()["retryAfterSeconds"], 900);
    assert_eq!(
        e.engine.verifies.load(Ordering::SeqCst),
        verifies,
        "no hashing happens for a block"
    );
    // Another address is untouched, and so is the owner's device (none here).
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.51").await.status, 200);
    // The block ends at fifteen minutes.
    e.clock.advance(15 * MIN - 1);
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.50").await.status, 429);
    e.clock.advance(1);
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.50").await.status, 200);
    // And a success cleared the counters: four wrong ones are not a block.
    for _ in 0..4 {
        assert_eq!(wrong(&e, "203.0.113.50").await.status, 401);
    }
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.50").await.status, 200);
}

#[tokio::test]
async fn an_ipv6_client_is_throttled_by_its_slash_64_and_not_by_its_address() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    for n in 1..=5u16 {
        wrong(&e, &format!("2001:db8:aaaa:bbbb:{n}::1")).await;
    }
    let r = login_as(&e, PASSWORD, "2001:db8:aaaa:bbbb:ffff::9").await;
    assert_eq!(r.status, 429, "the same /64");
    let r = login_as(&e, PASSWORD, "2001:db8:aaaa:cccc::9").await;
    assert_eq!(r.status, 200, "another /64");
}

#[tokio::test]
async fn the_throttle_state_survives_a_restart() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    for _ in 0..5 {
        wrong(&e, "203.0.113.60").await;
    }
    for i in 0..15 {
        wrong(&e, &format!("198.51.100.{}", 100 + i)).await;
    }
    assert!(e.state.throttle.is_slow(), "20 failures");
    e.state.flush();
    assert!(e.dir.0.join("auth").join("throttle.json").exists());
    // Restart: the same folder, two minutes later.
    let e2 = restart(e, 2 * MIN);
    assert!(
        e2.state.throttle.is_slow(),
        "slow mode is not cleared by a restart"
    );
    let r = login_as(&e2, PASSWORD, "203.0.113.60").await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (429, Some("rate_limited")),
        "nor is a block"
    );
}

// ==================================== T43: the owner's lanes =========================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t43_two_hundred_rotating_slash_64s_saturate_slow_mode_while_a_device_login_still_succeeds()
{
    let e = Arc::new(env());
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    // 200 failures from 200 different /64s (an attacker with a routed /56): every one is a wrong password. Slow mode
    // turns on at the 20th, and from then on each attempt has to wait its five seconds: the fake clock is moved on
    // as they are made, so this takes no real time.
    let mut attacker_ok = 0;
    for n in 0..200u32 {
        let from = format!("2001:db8:{:x}:{:x}::1", 0x1000 + n, n);
        let e2 = e.clone();
        let task = tokio::spawn(async move {
            login_as(&e2, "a guess of the wrong password", &from)
                .await
                .status
        });
        // The anonymous lane spaces starts by five seconds once slow mode is on: one move of five seconds per attempt.
        let mut moved = false;
        let mut spins = 0;
        while !task.is_finished() && spins < 4000 {
            tokio::time::sleep(Duration::from_millis(2)).await;
            if !moved && e.state.throttle.is_slow() {
                e.clock.advance(5_000);
                moved = true;
            }
            spins += 1;
        }
        if task.await.unwrap() == 401 {
            attacker_ok += 1;
        }
    }
    assert_eq!(attacker_ok, 200);
    assert!(e.state.throttle.is_slow(), "slow mode is on");
    assert_eq!(
        e.state.throttle.recent_failures(),
        200,
        "all 200 are inside the hour"
    );
    // Now three anonymous attackers queue behind one another in slow mode (nothing moves the clock), and the
    // owner's device-cookie login is answered at once and correctly.
    let mut queued = Vec::new();
    for n in 0..3u32 {
        let e2 = e.clone();
        let from = format!("2001:db8:{:x}::1", 0x5000 + n);
        queued.push(tokio::spawn(async move {
            login_as(&e2, "still the wrong password", &from)
                .await
                .status
        }));
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = std::time::Instant::now();
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .from("203.0.113.77")
            .browser()
            .cookie(&device_pair(&e, &b))
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "not behind the anonymous queue"
    );
    assert!(
        r.cookie(device_cookie_name(&e)).is_none(),
        "a known device keeps its cookie"
    );
    // The same owner without the device cookie is on the anonymous lane, in slow mode with three in front of it: it
    // waits (here, is not answered while the clock stands still).
    let e3 = e.clone();
    let anonymous =
        tokio::spawn(async move { login_as(&e3, PASSWORD, "203.0.113.78").await.status });
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !anonymous.is_finished(),
        "the anonymous lane is in slow mode"
    );
    // Let everything go.
    for _ in 0..40 {
        e.clock.advance(5_000);
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    for t in queued {
        assert_eq!(t.await.unwrap(), 401);
    }
    assert_eq!(anonymous.await.unwrap(), 200);
}

#[tokio::test]
async fn the_device_lane_has_its_own_counter_and_never_touches_an_address_or_slow_mode() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let with_device = |e: &Env, from: &str, password: &str| {
        req(e, Method::POST, "/api/auth/login")
            .from(from)
            .browser()
            .cookie(&device_pair(e, &b))
            .json(json!({ "password": password }))
    };
    // Four wrong answers on the device lane, then the right one: cleared.
    for _ in 0..4 {
        assert_eq!(
            go(&e, with_device(&e, "203.0.113.90", "wrong wrong wrong"))
                .await
                .status,
            401
        );
    }
    assert_eq!(
        go(&e, with_device(&e, "203.0.113.90", PASSWORD))
            .await
            .status,
        200
    );
    for _ in 0..4 {
        assert_eq!(
            go(&e, with_device(&e, "203.0.113.90", "wrong wrong wrong"))
                .await
                .status,
            401
        );
    }
    // The fifth blocks the DEVICE, for fifteen minutes, from any address.
    assert_eq!(
        go(&e, with_device(&e, "203.0.113.91", "wrong wrong wrong"))
            .await
            .status,
        401
    );
    let r = go(&e, with_device(&e, "203.0.113.92", PASSWORD)).await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    assert_eq!(r.headers.get("retry-after").unwrap(), "900");
    // The address itself was never counted: without the cookie it logs in, and slow mode is off.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.90").await.status, 200);
    assert!(!e.state.throttle.is_slow());
    assert_eq!(
        e.state.throttle.recent_failures(),
        0,
        "no failure of the device lane is an anonymous one"
    );
    // After the block the device works again.
    e.clock.advance(15 * MIN);
    assert_eq!(
        go(&e, with_device(&e, "203.0.113.92", PASSWORD))
            .await
            .status,
        200
    );
}

#[tokio::test]
async fn a_successful_device_login_does_not_clear_the_block_of_an_address() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    for _ in 0..5 {
        wrong(&e, "203.0.113.120").await;
    }
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .from("203.0.113.120")
            .browser()
            .cookie(&device_pair(&e, &b))
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 200, "the device lane skips the address's block");
    assert_eq!(
        login_as(&e, PASSWORD, "203.0.113.120").await.status,
        429,
        "which is still there"
    );
}

#[tokio::test]
async fn a_revoked_device_is_the_anonymous_lane_again() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let id = token::parse(b.device.as_ref().unwrap())
        .unwrap()
        .id
        .to_string();
    // Revoke it from the signed-in browser.
    let r = go(
        &e,
        as_page(&e, &b, Method::DELETE, &format!("/api/auth/sessions/{id}")),
    )
    .await;
    assert_eq!(r.status, 204, "{}", r.text);
    for _ in 0..5 {
        go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .from("203.0.113.130")
                .browser()
                .cookie(&device_pair(&e, &b))
                .json(json!({ "password": "wrong wrong wrong" })),
        )
        .await;
    }
    // Five failures with the dead cookie blocked the ADDRESS (it was the anonymous lane), not a device.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.130").await.status, 429);
    assert!(has_event(&e, "device.revoked"));
}

// ==================================== setup (T10) ====================================================

fn console_token(e: &Env) -> String {
    e.store
        .mint(MintSpec::new(
            Kind::Con,
            "console",
            super::scopes::ScopeSet::all(),
            365 * 24 * HOUR,
        ))
        .unwrap()
        .token
}

/// A setup code made the way the console makes it: through the running server's console route.
async fn console_setup_code(e: &Env) -> String {
    let con = console_token(e);
    let r = go(
        e,
        req(e, Method::POST, "/api/auth/console/setup-code")
            .bearer(&con)
            .json(json!({})),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    r.json()["code"].as_str().unwrap().to_string()
}

fn setup_req(e: &Env, code: &str, password: &str, from: &str) -> Req {
    req(e, Method::POST, "/api/auth/setup")
        .from(from)
        .browser()
        .json(json!({ "code": code, "password": password }))
}

#[tokio::test]
async fn setup_with_the_console_made_code_makes_the_owner_once_and_signs_in() {
    let e = env();
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(info["setupCode"], "none");
    let code = console_setup_code(&e).await;
    assert_eq!(
        code.len(),
        14,
        "three groups of four and two hyphens: {code}"
    );
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(
        (
            info["setupCode"].as_str(),
            info["loginConfigured"].as_bool()
        ),
        (Some("active"), Some(false))
    );
    assert!(
        !info.to_string().contains(&code),
        "info never says the code"
    );
    // Typed the way a person types it: lower case, spaces for hyphens.
    let typed = code.to_lowercase().replace('-', " ");
    let r = go(&e, setup_req(&e, &typed, PASSWORD, "203.0.113.9")).await;
    assert_eq!(r.status, 201, "{}", r.text);
    let b = browser_from(&e, &r);
    assert!(
        b.device.is_some(),
        "setup is a login: the browser is a known device"
    );
    assert_eq!(r.json()["ok"], true);
    assert_eq!(r.json()["elevatedUntilMs"], e.clock.now_ms() + 10 * MIN);
    // The owner file exists, private, holds a hash and no password; the code file is gone.
    let auth = e.dir.0.join("auth");
    let owner = std::fs::read_to_string(auth.join("owner.json")).unwrap();
    assert!(owner.contains("$argon2id$") && !owner.contains(PASSWORD));
    assert!(!auth.join("setup-code.json").exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(auth.join("owner.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(
        (
            info["loginConfigured"].as_bool(),
            info["setupCode"].as_str()
        ),
        (Some(true), Some("none"))
    );
    // The session works.
    let read = go(&e, as_page(&e, &b, Method::GET, "/api/config")).await;
    assert_eq!(read.status, 200);
    // A second setup, with a new code or the old, is 409 already_configured.
    let again = go(&e, setup_req(&e, &code, NEW_PASSWORD, "203.0.113.9")).await;
    assert_eq!(
        (again.status, again.code().as_deref()),
        (409, Some("already_configured"))
    );
    assert!(has_event(&e, "setup.ok"));
    // And the password is the one set.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.10").await.status, 200);
    assert_eq!(login_as(&e, NEW_PASSWORD, "203.0.113.11").await.status, 401);
}

#[tokio::test]
async fn a_weak_password_is_refused_with_its_reasons_and_the_code_stays_good() {
    let e = env();
    let code = console_setup_code(&e).await;
    let short = go(&e, setup_req(&e, &code, "too short one", "203.0.113.20")).await;
    assert_eq!(
        (short.status, short.code().as_deref()),
        (400, Some("weak_password"))
    );
    assert_eq!(short.json()["reasons"], json!(["too_short"]));
    let guessable = go(&e, setup_req(&e, &code, "passwordpassword", "203.0.113.20")).await;
    assert_eq!(guessable.json()["reasons"], json!(["too_guessable"]));
    let long = go(
        &e,
        setup_req(&e, &code, &"a1b2c3d4e5f6g7h8".repeat(9), "203.0.113.20"),
    )
    .await;
    assert_eq!(long.json()["reasons"], json!(["too_long"]));
    assert!(!e.state.owner_configured());
    assert_eq!(
        e.engine.hashes.load(Ordering::SeqCst),
        0,
        "nothing was hashed for a password that was refused"
    );
    // The code was not spent, and the failures are not failures of the address.
    let ok = go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.20")).await;
    assert_eq!(ok.status, 201, "{}", ok.text);
}

#[tokio::test]
async fn a_wrong_setup_code_counts_down_and_five_from_one_address_block_it() {
    let e = env();
    let code = console_setup_code(&e).await;
    for left in (95..=99).rev() {
        let r = go(
            &e,
            setup_req(&e, "AAAA-AAAA-AAAA", PASSWORD, "203.0.113.30"),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("invalid_setup_code"))
        );
        assert_eq!(r.json()["attemptsLeft"], left);
    }
    assert_eq!(
        e.engine.hashes.load(Ordering::SeqCst),
        0,
        "a wrong code starts no hash"
    );
    // The address is blocked now: even the right code is 429.
    let r = go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.30")).await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    // Another address with the right code is fine.
    assert_eq!(
        go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.31"))
            .await
            .status,
        201
    );
}

#[tokio::test]
async fn t10_the_hundred_and_first_guess_finds_a_burned_code_and_a_restart_gives_nothing_back() {
    let e = env();
    let code = console_setup_code(&e).await;
    // 40 wrong guesses from 40 different addresses, then a restart.
    for i in 0..40 {
        let r = go(
            &e,
            setup_req(
                &e,
                "AAAA-AAAA-AAAA",
                PASSWORD,
                &format!("198.51.100.{}", 1 + i),
            ),
        )
        .await;
        assert_eq!(r.status, 401);
    }
    let e = restart(e, 5 * MIN);
    let r = go(
        &e,
        setup_req(&e, "AAAA-AAAA-AAAA", PASSWORD, "198.51.100.200"),
    )
    .await;
    assert_eq!(r.json()["attemptsLeft"], 59, "the count is in the file");
    for i in 0..58 {
        go(
            &e,
            setup_req(
                &e,
                "AAAA-AAAA-AAAA",
                PASSWORD,
                &format!("192.0.2.{}", 1 + i),
            ),
        )
        .await;
    }
    let last = go(&e, setup_req(&e, "AAAA-AAAA-AAAA", PASSWORD, "192.0.2.100")).await;
    assert_eq!(last.json()["attemptsLeft"], 0, "the hundredth wrong guess");
    // Burned: even the right code is not accepted, and nothing makes another.
    let r = go(&e, setup_req(&e, &code, PASSWORD, "192.0.2.101")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("invalid_setup_code"))
    );
    assert_eq!(r.json()["attemptsLeft"], 0);
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(info["setupCode"], "none");
    e.clock.advance(1000 * HOUR);
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(info["setupCode"], "none", "never regenerated on its own");
    // The console makes another, which works.
    let fresh = console_setup_code(&e).await;
    assert_eq!(
        go(&e, setup_req(&e, &fresh, PASSWORD, "192.0.2.102"))
            .await
            .status,
        201
    );
}

#[tokio::test]
async fn t10_a_code_expires_after_a_day_and_a_second_code_replaces_the_first() {
    let e = env();
    let first = console_setup_code(&e).await;
    let second = console_setup_code(&e).await;
    assert_ne!(first, second);
    let r = go(&e, setup_req(&e, &first, PASSWORD, "203.0.113.40")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("invalid_setup_code")),
        "the first is dead"
    );
    e.clock.advance(24 * HOUR);
    let r = go(&e, setup_req(&e, &second, PASSWORD, "203.0.113.41")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (410, Some("setup_code_expired"))
    );
    let info = go(&e, req(&e, Method::GET, "/api/auth/info")).await.json();
    assert_eq!(info["setupCode"], "expired");
    assert!(!e.state.owner_configured());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_valid_setups_at_once_make_one_owner_and_the_other_gets_409() {
    let e = Arc::new(env());
    let code = console_setup_code(&e).await;
    let (a, b) = {
        let (e1, e2) = (e.clone(), e.clone());
        let (c1, c2) = (code.clone(), code.clone());
        let a =
            tokio::spawn(
                async move { go(&e1, setup_req(&e1, &c1, PASSWORD, "203.0.113.50")).await },
            );
        let b = tokio::spawn(async move {
            go(&e2, setup_req(&e2, &c2, NEW_PASSWORD, "203.0.113.51")).await
        });
        (a.await.unwrap(), b.await.unwrap())
    };
    let mut statuses = [a.status, b.status];
    statuses.sort_unstable();
    assert_eq!(statuses, [201, 409], "{} / {}", a.text, b.text);
    // Exactly one password is the owner's.
    let first_ok = login_as(&e, PASSWORD, "203.0.113.60").await.status == 200;
    let second_ok = login_as(&e, NEW_PASSWORD, "203.0.113.61").await.status == 200;
    assert!(first_ok ^ second_ok);
}

#[tokio::test]
async fn setup_is_for_the_dashboard_host_over_a_secure_channel_like_login() {
    let e = env();
    let code = console_setup_code(&e).await;
    let r = go(
        &e,
        setup_req(&e, &code, PASSWORD, "203.0.113.9").host("agent.example.com"),
    )
    .await;
    assert_eq!(r.status, 404);
    let r = go(
        &e,
        setup_req(&e, &code, PASSWORD, "203.0.113.9").h("origin", "https://evil.example"),
    )
    .await;
    assert_eq!((r.status, r.code().as_deref()), (403, Some("csrf")));
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), 0);
}

// ==================================== logout, elevation, sessions =====================================

/// An app-host session of `app`, made from the dashboard session `parent` the way a handoff makes one.
fn child_session(e: &Env, parent: &Browser, app: super::presets::App) -> (String, String) {
    let preset = super::presets::Preset::Agent;
    let mut spec = MintSpec::new(Kind::Ses, "agent", preset.scopes(), 24 * HOUR);
    spec.app = Some(app);
    spec.preset = Some(preset);
    spec.parent = Some(token::parse(&parent.session).unwrap().id.to_string());
    spec.idle_ms = Some(8 * HOUR);
    let child = e.store.mint(spec).unwrap();
    let csrf = token::csrf_value(token::parse(&child.token).unwrap().secret).unwrap();
    (child.token, csrf)
}

#[tokio::test]
async fn logout_revokes_the_session_and_what_derives_from_it_and_keeps_the_device_cookie() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let (child, child_csrf) = child_session(&e, &b, super::presets::App::Agent);
    let at_agent = |method: Method, path: &str| {
        req(&e, method, path)
            .host("agent.example.com")
            .cookie(&format!("__Host-oaiy_agent={child}"))
            .page(&child_csrf)
    };
    assert_eq!(
        go(&e, at_agent(Method::GET, "/api/agent/preferences"))
            .await
            .status,
        200
    );

    // Logout needs the session, its CSRF and the same origin.
    let no_csrf = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/logout").drop_h("x-oaiy-csrf"),
    )
    .await;
    assert_eq!(
        (no_csrf.status, no_csrf.code().as_deref()),
        (403, Some("csrf"))
    );
    let anon = go(&e, req(&e, Method::POST, "/api/auth/logout")).await;
    assert_eq!(anon.status, 401);

    let r = go(&e, as_page(&e, &b, Method::POST, "/api/auth/logout")).await;
    assert_eq!(r.status, 204, "{}", r.text);
    // Only the session cookie is told to go; the device cookie stays.
    assert_eq!(
        r.cookies(),
        vec!["__Host-oaiy_dash=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0".to_string()]
    );
    // The session is dead at once, with the reason; its child with it.
    let dead = go(&e, as_page(&e, &b, Method::GET, "/api/config")).await;
    assert_eq!(
        (dead.status, dead.code().as_deref()),
        (401, Some("session_expired"))
    );
    assert_eq!(dead.json()["reason"], "logged_out");
    let orphan = go(&e, at_agent(Method::GET, "/api/agent/preferences")).await;
    assert_eq!(
        (orphan.status, orphan.code().as_deref()),
        (401, Some("session_expired"))
    );
    assert_eq!(orphan.json()["reason"], "parent_ended");
    // The device cookie still works for a login from that browser.
    let again = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .browser()
            .cookie(&device_pair(&e, &b))
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!(again.status, 200);
    assert!(
        again.cookie(device_cookie_name(&e)).is_none(),
        "and it is still that device"
    );
    assert!(has_event(&e, "logout"));
}

#[tokio::test]
async fn the_elevation_of_a_login_is_ten_minutes_and_elevate_gives_another_with_the_password() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let dangerous = |e: &Env, b: &Browser| as_page(e, b, Method::POST, "/api/services");
    assert_eq!(
        go(&e, dangerous(&e, &b)).await.status,
        200,
        "a login is an elevation"
    );
    e.clock.advance(10 * MIN);
    let r = go(&e, dangerous(&e, &b)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("elevation_required"))
    );
    // The session says so.
    let s = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(s["elevatedUntilMs"], 0);
    // Confirm with the password: ten minutes from now.
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/elevate")
            .json(json!({ "password": PASSWORD, "method": "password" })),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.json()["elevatedUntilMs"], e.clock.now_ms() + 10 * MIN);
    assert_eq!(go(&e, dangerous(&e, &b)).await.status, 200);
    let s = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(s["elevatedUntilMs"], e.clock.now_ms() + 10 * MIN);
    // A restart drops it (it is in memory only), and the session, which was written, survives.
    let e = restart(e, MIN);
    let after = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session")).await;
    assert_eq!(after.status, 200, "{}", after.text);
    assert_eq!(after.json()["elevatedUntilMs"], 0);
    // A method that does not exist yet is refused.
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/elevate")
            .json(json!({ "password": PASSWORD, "method": "totp" })),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (400, Some("invalid_request"))
    );
}

#[tokio::test]
async fn an_elevation_is_a_session_lane_five_wrong_answers_revoke_the_session_and_they_are_shared_with_password(
) {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    // Three wrong elevations and two wrong `current` passwords: five in a row.
    for _ in 0..3 {
        let r = go(
            &e,
            as_page(&e, &b, Method::POST, "/api/auth/elevate")
                .json(json!({ "password": "wrong wrong wrong" })),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("invalid_credentials"))
        );
    }
    let change = |e: &Env| {
        as_page(e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": "wrong wrong wrong", "next": NEW_PASSWORD }))
    };
    assert_eq!(go(&e, change(&e)).await.status, 401);
    assert_eq!(
        go(&e, change(&e)).await.status,
        401,
        "the fifth answer is a 401 too"
    );
    // The session is gone.
    let r = go(&e, as_page(&e, &b, Method::GET, "/api/config")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("session_expired"))
    );
    assert!(has_event(&e, "elevate.fail") && has_event(&e, "password.fail"));
    // The password was not changed, and nothing counted against the address or slow mode.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.9").await.status, 200);
    assert!(!e.state.throttle.is_slow());
}

#[tokio::test]
async fn a_session_may_check_the_password_ten_times_an_hour_and_never_waits_in_the_anonymous_queue()
{
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let elevate = |e: &Env| {
        as_page(e, &b, Method::POST, "/api/auth/elevate").json(json!({ "password": PASSWORD }))
    };
    for i in 0..10 {
        let r = go(&e, elevate(&e)).await;
        assert_eq!(r.status, 200, "check {i}: {}", r.text);
    }
    let r = go(&e, elevate(&e)).await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    assert!(r.headers.get("retry-after").is_some());
    e.clock.advance(HOUR);
    assert_eq!(go(&e, elevate(&e)).await.status, 200);
}

#[tokio::test]
async fn only_a_dashboard_session_can_elevate_or_change_the_password() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let (child, csrf) = child_session(&e, &b, super::presets::App::Agent);
    for path in ["/api/auth/elevate", "/api/auth/password"] {
        let r = go(
            &e,
            req(&e, Method::POST, path)
                .host("agent.example.com")
                .cookie(&format!("__Host-oaiy_agent={child}"))
                .page(&csrf)
                .json(json!({ "password": PASSWORD, "current": PASSWORD, "next": NEW_PASSWORD })),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("insufficient_scope")),
            "{path}"
        );
    }
}

// ==================================== password change =================================================

fn a_pat(e: &Env) -> super::store::Minted {
    e.store
        .mint(MintSpec::new(
            Kind::Pat,
            "tool",
            super::scopes::ScopeSet::of(&["system.read"]),
            30 * 24 * HOUR,
        ))
        .unwrap()
}

#[tokio::test]
async fn a_password_change_revokes_every_other_session_every_device_and_every_derived_credential() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let current = browser_from(&e, &owner);
    // Another browser (a second device and session), and a third.
    let second = browser_from(&e, &login_as(&e, PASSWORD, "203.0.113.70").await);
    let third = browser_from(&e, &login_as(&e, PASSWORD, "203.0.113.71").await);
    // A paired token and a derived credential.
    let pat = a_pat(&e);
    let parent = e.store.authenticate(&pat.token, None).unwrap();
    let derived = e
        .store
        .derive(
            &parent,
            super::store::DeriveRequest {
                scopes: super::scopes::ScopeSet::of(&["system.read"]),
                ttl_ms: Some(HOUR),
                label: "child".into(),
            },
        )
        .unwrap();
    assert!(e.store.authenticate(&derived.token, None).is_ok());

    let r = go(
        &e,
        as_page(&e, &current, Method::POST, "/api/auth/password")
            .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 204, "{}", r.text);
    // The browser's own device cookie is dropped in the answer (it is revoked with the others).
    assert_eq!(
        r.cookies(),
        vec!["__Host-oaiy_dev=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0".to_string()]
    );
    // The current session stays; the others are dead with the reason; the derived credential is dead; the paired
    // token (not asked for) is alive.
    assert_eq!(
        go(&e, as_page(&e, &current, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
    for other in [&second, &third] {
        let r = go(&e, as_page(&e, other, Method::GET, "/api/config")).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("session_expired"))
        );
        assert_eq!(r.json()["reason"], "password_changed");
    }
    assert!(
        e.store.authenticate(&derived.token, None).is_err(),
        "derived credentials are revoked"
    );
    assert!(e.store.authenticate(&pat.token, None).is_ok());
    // The old password is gone and the new one works; no device cookie of before routes to the device lane.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.80").await.status, 401);
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .from("203.0.113.81")
            .browser()
            .cookie(&device_pair(&e, &second))
            .json(json!({ "password": NEW_PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 200);
    assert!(
        r.cookie(device_cookie_name(&e)).is_some(),
        "the old device is not known: a new one is made"
    );
    assert!(has_event(&e, "password.changed"));
    // The password is stored as a new hash on disk.
    let owner_file = std::fs::read_to_string(e.dir.0.join("auth").join("owner.json")).unwrap();
    assert!(!owner_file.contains(PASSWORD) && !owner_file.contains(NEW_PASSWORD));
    let devices = serde_json::from_str::<Value>(&owner_file).unwrap()["devices"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(
        devices, 1,
        "only the device made by the login after the change"
    );
}

#[tokio::test]
async fn a_password_change_can_revoke_the_paired_tokens_too_and_refuses_a_weak_or_wrong_one() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let pat = a_pat(&e);
    // A weak next password: 400 with the reasons, and nothing changed.
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": PASSWORD, "next": "short" })),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (400, Some("weak_password"))
    );
    assert_eq!(r.json()["reasons"], json!(["too_short"]));
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.90").await.status, 200);
    // The wrong current password: 401, same as a login.
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": "wrong wrong wrong", "next": NEW_PASSWORD })),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("invalid_credentials"))
    );
    // With revokeTokens the persisted paired tokens go too.
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD, "revokeTokens": true })),
    )
    .await;
    assert_eq!(r.status, 204);
    let err = e.store.authenticate(&pat.token, None).unwrap_err();
    assert_eq!(err.reason(), Some("password_changed"));
}

#[tokio::test]
async fn a_disk_that_will_not_take_the_new_password_leaves_the_old_one_in_force() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    e.disk.full.store(true, Ordering::SeqCst);
    let r = go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (503, Some("store_unavailable"))
    );
    assert!(
        has_event(&e, "disk.full"),
        "a full disk is told to the audit log"
    );
    e.disk.full.store(false, Ordering::SeqCst);
    assert_eq!(
        login_as(&e, PASSWORD, "203.0.113.95").await.status,
        200,
        "the old password still works"
    );
    assert_eq!(login_as(&e, NEW_PASSWORD, "203.0.113.96").await.status, 401);
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/config"))
            .await
            .status,
        200,
        "and nothing was revoked"
    );
}

// ==================================== the disk (T27) =================================================

#[tokio::test]
async fn t27_a_login_on_a_full_disk_still_succeeds_and_says_it_is_not_persisted() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    e.disk.full.store(true, Ordering::SeqCst);
    let r = login_as(&e, PASSWORD, "203.0.113.100").await;
    assert_eq!(r.status, 200, "sign-in never needs a write: {}", r.text);
    assert_eq!(r.json()["persisted"], false);
    let b = browser_from(&e, &r);
    let s = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(
        (s["authenticated"].as_bool(), s["persisted"].as_bool()),
        (Some(true), Some(false))
    );
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
    assert_eq!(e.store.storage().name(), "full");
    // A second login on the same full disk is fine too, and lives beside the first.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.101").await.status, 200);
    // When the disk has room again, the next write puts the memory-only sessions on disk.
    e.disk.full.store(false, Ordering::SeqCst);
    let id = token::parse(&b.session).unwrap().id.to_string();
    e.store.flush().unwrap();
    let file = std::fs::read_to_string(e.dir.0.join("auth").join("credentials.json")).unwrap();
    assert!(file.contains(&id), "written once the disk took it");
}

#[tokio::test]
async fn t27_setup_on_a_full_disk_is_503_and_makes_no_owner_and_keeps_the_code() {
    let e = env();
    let code = console_setup_code(&e).await;
    e.disk.full.store(true, Ordering::SeqCst);
    let r = go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.110")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (503, Some("store_unavailable"))
    );
    assert!(!e.state.owner_configured());
    assert!(!e.dir.0.join("auth").join("owner.json").exists());
    assert!(
        has_event(&e, "disk.full"),
        "a full disk is told to the audit log"
    );
    e.disk.full.store(false, Ordering::SeqCst);
    assert_eq!(
        go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.110"))
            .await
            .status,
        201,
        "the code was not spent"
    );
}

// ==================================== the session endpoint ===========================================

#[tokio::test]
async fn the_session_endpoint_is_public_says_who_you_are_and_never_carries_cors() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    // Anonymous, at each host.
    let dash = go(&e, req(&e, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(
        dash,
        json!({ "authenticated": false, "app": "dash", "loginUrl": null })
    );
    let agent = go(
        &e,
        req(&e, Method::GET, "/api/auth/session").host("agent.example.com"),
    )
    .await
    .json();
    assert_eq!(
        agent,
        json!({ "authenticated": false, "app": "agent", "loginUrl": "https://dash.example.com/apps/agent" })
    );
    // Signed in: the csrf value, the times, the scopes, the level.
    let cookie = format!("__Host-oaiy_dash={}", b.session);
    let s = go(
        &e,
        req(&e, Method::GET, "/api/auth/session").cookie(&cookie),
    )
    .await
    .json();
    assert_eq!(s["authenticated"], true);
    assert_eq!(s["csrf"], b.csrf.as_str());
    assert_eq!(s["controlLevel"], "project");
    assert_eq!(s["scopes"].as_array().unwrap().len(), 54);
    assert_eq!(s["persisted"], true);
    // A request from another site sees the anonymous answer (the cookie is ignored).
    let cross = go(
        &e,
        req(&e, Method::GET, "/api/auth/session")
            .cookie(&cookie)
            .h("sec-fetch-site", "cross-site"),
    )
    .await
    .json();
    assert_eq!(cross["authenticated"], false);
    // A wrong cookie is anonymous too, not an error.
    let junk = go(
        &e,
        req(&e, Method::GET, "/api/auth/session").cookie("__Host-oaiy_dash=garbage"),
    )
    .await;
    assert_eq!(
        (junk.status, junk.json()["authenticated"].as_bool()),
        (200, Some(false))
    );
    // No CORS headers on it, whatever the Origin: it is same-origin by construction.
    for origin in ["https://evil.example", "https://dash.example.com", "null"] {
        let r = go(
            &e,
            req(&e, Method::GET, "/api/auth/session").h("origin", origin),
        )
        .await;
        assert!(
            r.headers.get("access-control-allow-origin").is_none(),
            "{origin}"
        );
    }
    let r = go(
        &e,
        req(&e, Method::GET, "/api/auth/info").h("origin", "https://evil.example"),
    )
    .await;
    assert!(r.headers.get("access-control-allow-origin").is_none());
}

// ==================================== sessions and devices ===========================================

#[tokio::test]
async fn the_sessions_and_devices_are_listed_and_revoked_without_a_step_up() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let a = browser_from(&e, &owner);
    let r2 = login_as(&e, PASSWORD, "198.51.100.5").await;
    let b = browser_from(&e, &r2);
    // Not elevated (ten minutes on): listing and revoking need none.
    e.clock.advance(11 * MIN);
    let list = go(&e, as_page(&e, &a, Method::GET, "/api/auth/sessions")).await;
    assert_eq!(list.status, 200, "{}", list.text);
    let list = list.json();
    let sessions = list["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 2);
    let current: Vec<&Value> = sessions.iter().filter(|s| s["current"] == true).collect();
    assert_eq!(current.len(), 1);
    assert_eq!(current[0]["id"], token::parse(&a.session).unwrap().id);
    assert_eq!(current[0]["app"], "dash");
    for s in sessions {
        for key in [
            "id",
            "app",
            "createdMs",
            "lastUsedMs",
            "ip",
            "userAgent",
            "current",
        ] {
            assert!(s.get(key).is_some(), "{key} in {s}");
        }
    }
    assert_eq!(list["devices"].as_array().unwrap().len(), 2);
    assert!(!list.to_string().contains(&a.session) && !list.to_string().contains(&a.csrf));
    // Revoke the other session by id; then a device by id; an unknown id is 404.
    let bid = token::parse(&b.session).unwrap().id.to_string();
    assert_eq!(
        go(
            &e,
            as_page(&e, &a, Method::DELETE, &format!("/api/auth/sessions/{bid}"))
        )
        .await
        .status,
        204
    );
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/config"))
            .await
            .status,
        401
    );
    let devid = token::parse(b.device.as_ref().unwrap())
        .unwrap()
        .id
        .to_string();
    assert_eq!(
        go(
            &e,
            as_page(
                &e,
                &a,
                Method::DELETE,
                &format!("/api/auth/sessions/{devid}")
            )
        )
        .await
        .status,
        204
    );
    assert_eq!(
        go(
            &e,
            as_page(
                &e,
                &a,
                Method::DELETE,
                "/api/auth/sessions/0123456789abcdef"
            )
        )
        .await
        .status,
        404
    );
    let list = go(&e, as_page(&e, &a, Method::GET, "/api/auth/sessions"))
        .await
        .json();
    assert_eq!(list["sessions"].as_array().unwrap().len(), 1);
    assert_eq!(list["devices"].as_array().unwrap().len(), 1);
    // revoke-others keeps the caller.
    let c = browser_from(&e, &login_as(&e, PASSWORD, "198.51.100.6").await);
    let r = go(
        &e,
        as_page(&e, &a, Method::POST, "/api/auth/sessions/revoke-others"),
    )
    .await;
    assert_eq!(r.status, 204);
    assert_eq!(
        go(&e, as_page(&e, &a, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
    assert_eq!(
        go(&e, as_page(&e, &c, Method::GET, "/api/config"))
            .await
            .status,
        401
    );
}

// ==================================== the link (4.7.9) ===============================================

async fn console_link(e: &Env) -> (String, String) {
    let con = console_token(e);
    let r = go(
        e,
        req(e, Method::POST, "/api/auth/console/session-link")
            .bearer(&con)
            .json(json!({})),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let j = r.json();
    (
        j["code"].as_str().unwrap().to_string(),
        j["url"].as_str().unwrap().to_string(),
    )
}

fn link_req(e: &Env, code: &str, n: u32) -> Req {
    req(e, Method::POST, "/api/auth/link")
        .from(&format!("203.0.113.{n}"))
        .browser()
        .json(json!({ "code": code }))
}

#[tokio::test]
async fn a_session_link_makes_a_plain_session_once_within_five_minutes() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let (code, url) = console_link(&e).await;
    assert_eq!(code.len(), 43);
    assert_eq!(url, format!("https://dash.example.com/auth/link#{code}"));
    let r = go(&e, link_req(&e, &code, 120)).await;
    assert_eq!(r.status, 200, "{}", r.text);
    // Neither remembered nor elevated, and it does not make a device.
    assert_eq!(r.json()["elevatedUntilMs"], 0);
    let cookies = r.cookies();
    assert_eq!(cookies.len(), 1, "{cookies:?}");
    assert!(!cookies[0].contains("Max-Age"), "{cookies:?}");
    let b = browser_from(&e, &r);
    // It can list and revoke (that is what it is for) but not do what needs the elevation.
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/auth/sessions"))
            .await
            .status,
        200
    );
    let r = go(&e, as_page(&e, &b, Method::POST, "/api/services")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("elevation_required"))
    );
    // Once: the same code again is 400.
    let again = go(&e, link_req(&e, &code, 121)).await;
    assert_eq!(
        (again.status, again.code().as_deref()),
        (400, Some("bad_request"))
    );
    assert!(has_event(&e, "link.issued") && has_event(&e, "link.used"));
}

#[tokio::test]
async fn a_session_link_ends_after_five_minutes_on_the_monotonic_clock_whatever_the_wall_clock_does(
) {
    let e = env();
    make_owner(&e, PASSWORD).await;
    // The wall clock goes back a day: it does not extend the link.
    let (code, _) = console_link(&e).await;
    e.mono.advance(5 * MIN - 1);
    e.clock.rewind(24 * HOUR);
    assert_eq!(
        go(&e, link_req(&e, &code, 130)).await.status,
        200,
        "still inside five minutes"
    );
    let (code, _) = console_link(&e).await;
    e.mono.advance(5 * MIN);
    let r = go(&e, link_req(&e, &code, 131)).await;
    assert_eq!(
        r.status, 400,
        "five minutes on the monotonic clock is the end"
    );
    // A link that never existed, and one of the wrong length or alphabet, are 400 as well and count as failures.
    for bad in ["x", &"A".repeat(43), &"!".repeat(43), &"A".repeat(44)] {
        let r = go(&e, link_req(&e, bad, 132)).await;
        assert_eq!(r.status, 400, "{bad}");
    }
    // The fifth failure from an address blocks it: guessing links is throttled like guessing passwords.
    assert_eq!(
        go(&e, link_req(&e, "z", 132)).await.status,
        400,
        "the fifth"
    );
    assert_eq!(go(&e, link_req(&e, "z", 132)).await.status, 429);
}

#[tokio::test]
async fn at_most_eight_links_are_outstanding_and_each_is_spent_by_its_own_code() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let mut codes = Vec::new();
    for _ in 0..10 {
        codes.push(console_link(&e).await.0);
    }
    assert_eq!(e.state.links_outstanding(), 8);
    // The oldest two were dropped; the newest still works, once, and not by another's code.
    assert_eq!(go(&e, link_req(&e, &codes[0], 140)).await.status, 400);
    assert_eq!(go(&e, link_req(&e, &codes[9], 141)).await.status, 200);
    assert_eq!(go(&e, link_req(&e, &codes[9], 142)).await.status, 400);
    assert_eq!(go(&e, link_req(&e, &codes[5], 143)).await.status, 200);
    assert_eq!(e.state.links_outstanding(), 6);
}

// ==================================== the console's routes (T42) =====================================

#[tokio::test]
async fn the_console_routes_need_the_console_credential_and_nothing_else_will_do() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let pat = a_pat(&e).token;
    let con = console_token(&e);
    let routes = [
        (Method::POST, "/api/auth/console/reset-password"),
        (Method::POST, "/api/auth/console/setup-code"),
        (Method::POST, "/api/auth/console/session-link"),
        (Method::POST, "/api/auth/console/sessions/revoke-all"),
        (Method::GET, "/api/auth/console/status"),
    ];
    for (m, path) in routes {
        // Anonymous, a paired token, a session (with everything a session sends).
        let anon = go(&e, req(&e, m.clone(), path)).await;
        assert_eq!(anon.status, 401, "{path}");
        let with_pat = go(&e, req(&e, m.clone(), path).bearer(&pat).json(json!({}))).await;
        assert_eq!(
            (with_pat.status, with_pat.code().as_deref()),
            (403, Some("insufficient_scope")),
            "{path}"
        );
        let with_session = go(&e, as_page(&e, &b, m.clone(), path).json(json!({}))).await;
        assert_eq!(
            (with_session.status, with_session.code().as_deref()),
            (403, Some("insufficient_scope")),
            "{path}"
        );
        // The console credential through a proxy (a forwarded header) is worthless.
        let mut proxied = req(&e, m.clone(), path)
            .h("authorization", &format!("Bearer {con}"))
            .json(json!({}));
        proxied.proxied = true;
        let refused = go(&e, proxied).await;
        assert_eq!(
            (refused.status, refused.code().as_deref()),
            (403, Some("origin_mismatch")),
            "{path}"
        );
        // And with an Origin (a browser page holding the credential): refused.
        let with_origin = go(
            &e,
            req(&e, m.clone(), path)
                .bearer(&con)
                .h("origin", "https://dash.example.com")
                .json(json!({})),
        )
        .await;
        assert_eq!(
            (with_origin.status, with_origin.code().as_deref()),
            (403, Some("origin_mismatch")),
            "{path}"
        );
    }
}

#[tokio::test]
async fn t42_the_console_resets_the_password_without_a_restart_and_revokes_every_session_and_device(
) {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let a = browser_from(&e, &owner);
    let b = browser_from(&e, &login_as(&e, PASSWORD, "203.0.113.150").await);
    let con = console_token(&e);
    let reset = |e: &Env, password: &str| {
        req(e, Method::POST, "/api/auth/console/reset-password")
            .bearer(&con)
            .json(json!({ "password": password }))
    };
    let weak = go(&e, reset(&e, "short")).await;
    assert_eq!(
        (weak.status, weak.code().as_deref()),
        (400, Some("weak_password"))
    );
    let r = go(&e, reset(&e, NEW_PASSWORD)).await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(r.json()["created"], false);
    // Every session and device is gone at once (no restart); the old password fails and the new one works.
    for x in [&a, &b] {
        let r = go(&e, as_page(&e, x, Method::GET, "/api/config")).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("session_expired"))
        );
        assert_eq!(r.json()["reason"], "password_changed");
    }
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.151").await.status, 401);
    let fresh = login_as(&e, NEW_PASSWORD, "203.0.113.152").await;
    assert_eq!(fresh.status, 200);
    let owner_file = std::fs::read_to_string(e.dir.0.join("auth").join("owner.json")).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&owner_file).unwrap()["devices"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    // The command is in the audit log by name, and never with the password.
    let audit = std::fs::read_to_string(e.dir.0.join("auth").join("audit.jsonl")).unwrap();
    assert!(audit.contains("console.command") && audit.contains("reset-password"));
    assert!(!audit.contains(NEW_PASSWORD) && !audit.contains(PASSWORD));
}

#[tokio::test]
async fn the_console_makes_the_first_owner_when_there_is_none_and_spends_the_setup_code() {
    let e = env();
    console_setup_code(&e).await;
    let con = console_token(&e);
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/console/reset-password")
            .bearer(&con)
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!((r.status, r.json()["created"].as_bool()), (200, Some(true)));
    assert!(e.state.owner_configured());
    assert!(!e.dir.0.join("auth").join("setup-code.json").exists());
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.160").await.status, 200);
    // And a setup code cannot be made for an install that has an owner.
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/console/setup-code")
            .bearer(&con)
            .json(json!({})),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (409, Some("already_configured"))
    );
}

#[tokio::test]
async fn revoke_all_from_the_console_takes_the_sessions_and_the_devices_when_asked() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let a = browser_from(&e, &owner);
    let con = console_token(&e);
    let revoke_all = |e: &Env, body: Value| {
        req(e, Method::POST, "/api/auth/console/sessions/revoke-all")
            .bearer(&con)
            .json(body)
    };
    let r = go(&e, revoke_all(&e, json!({}))).await;
    assert_eq!(
        (
            r.status,
            r.json()["sessions"].as_u64(),
            r.json()["devices"].as_u64()
        ),
        (200, Some(1), Some(0))
    );
    assert_eq!(
        go(&e, as_page(&e, &a, Method::GET, "/api/config"))
            .await
            .status,
        401
    );
    // The device survives without the flag and is used.
    let login_with_device = |e: &Env| {
        req(e, Method::POST, "/api/auth/login")
            .browser()
            .cookie(&device_pair(e, &a))
            .json(json!({ "password": PASSWORD }))
    };
    let login = go(&e, login_with_device(&e)).await;
    assert_eq!(login.status, 200);
    assert!(
        login.cookie(device_cookie_name(&e)).is_none(),
        "a known device"
    );
    let r = go(&e, revoke_all(&e, json!({ "devices": true }))).await;
    assert_eq!(
        (r.json()["sessions"].as_u64(), r.json()["devices"].as_u64()),
        (Some(1), Some(1))
    );
    let login = go(&e, login_with_device(&e)).await;
    assert!(
        login.cookie(device_cookie_name(&e)).is_some(),
        "the device is gone: a new one is made"
    );
}

#[tokio::test]
async fn the_status_says_what_is_going_on_and_holds_no_secret() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    for _ in 0..5 {
        wrong(&e, "203.0.113.170").await;
    }
    let con = console_token(&e);
    let (code, _) = console_link(&e).await;
    let r = go(
        &e,
        req(&e, Method::GET, "/api/auth/console/status").bearer(&con),
    )
    .await;
    assert_eq!(r.status, 200);
    let s = r.json();
    assert_eq!(s["loginConfigured"], true);
    assert_eq!(s["exposure"], "proxied");
    assert_eq!(s["sessions"], 1);
    assert_eq!(s["devices"], 1);
    assert_eq!(s["blockedAddresses"][0]["key"], "203.0.113.170");
    assert_eq!(s["storage"], "ok");
    assert_eq!(s["linksOutstanding"], 1);
    // The verifications: five wrong passwords were checked, never more than the bound at once, none running now.
    let v = &s["verifications"];
    assert_eq!(
        (v["bound"].as_u64(), v["running"].as_u64()),
        (Some(3), Some(0))
    );
    assert_eq!(v["started"].as_u64(), Some(5), "{v}");
    assert!((1..=3).contains(&v["mostAtOnce"].as_u64().unwrap()), "{v}");
    let text = r.text.clone();
    for secret in [
        b.session.as_str(),
        b.csrf.as_str(),
        b.device.as_deref().unwrap(),
        code.as_str(),
        con.as_str(),
        PASSWORD,
    ] {
        assert!(!text.contains(secret), "the status holds a secret: {text}");
    }
    assert!(!text.contains("argon2"), "nor the password hash");
}

// ==================================== the memory bound and the allow-list =============================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_flood_of_logins_runs_at_most_three_verifications_at_once() {
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(30),
        ..Build::default()
    }));
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    e.engine.most.store(0, Ordering::SeqCst);
    let mut tasks = Vec::new();
    // 40 wrong logins from 40 addresses, 15 device-lane logins (some wrong), 10 elevations: all at once.
    for i in 0..40u32 {
        let e2 = e.clone();
        tasks.push(tokio::spawn(async move {
            login_as(&e2, "wrong wrong wrong", &format!("198.51.100.{}", 10 + i))
                .await
                .status
        }));
    }
    for i in 0..15u32 {
        let e2 = e.clone();
        let pair = device_pair(&e, &b);
        tasks.push(tokio::spawn(async move {
            go(
                &e2,
                req(&e2, Method::POST, "/api/auth/login")
                    .from(&format!("192.0.2.{}", 10 + i))
                    .browser()
                    .cookie(&pair)
                    .json(json!({ "password": if i % 3 == 0 { "wrong wrong wrong" } else { PASSWORD } })),
            )
            .await
            .status
        }));
    }
    for _ in 0..10 {
        let e2 = e.clone();
        let (session, csrf) = (b.session.clone(), b.csrf.clone());
        tasks.push(tokio::spawn(async move {
            let browser = Browser {
                session,
                csrf,
                device: None,
            };
            go(
                &e2,
                as_page(&e2, &browser, Method::POST, "/api/auth/elevate")
                    .json(json!({ "password": PASSWORD })),
            )
            .await
            .status
        }));
    }
    let mut statuses = Vec::new();
    for t in tasks {
        statuses.push(t.await.unwrap());
    }
    let most = e.engine.most.load(Ordering::SeqCst);
    assert!(most <= 3, "{most} verifications ran at once");
    assert!(most >= 2, "they did overlap ({most})");
    assert!(
        statuses.contains(&429),
        "and some of the flood was turned away at once"
    );
    assert!(
        statuses.iter().all(|s| matches!(s, 200 | 401 | 429)),
        "{statuses:?}"
    );
}

// A client that hangs up: the request's future is dropped at its next await point, but a pass of Argon2 that has
// started runs to its end on its blocking thread. The bound (2 + 1 passes at once, 192 MiB) and the counting of the
// failure must hold whatever becomes of the client.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_hangs_up_does_not_free_the_verification_bound_before_the_pass_ends() {
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(400),
        ..Build::default()
    }));
    make_owner(&e, PASSWORD).await;
    e.engine.most.store(0, Ordering::SeqCst);
    let mut hung_up = 0;
    for i in 0..24u32 {
        // Each request comes from its own address and is dropped 25 ms after it starts: long after its pass began,
        // long before it ends.
        let r = tokio::time::timeout(
            Duration::from_millis(25),
            login_as(&e, "wrong wrong wrong", &format!("198.51.100.{}", 10 + i)),
        )
        .await;
        if r.is_err() {
            hung_up += 1;
        }
    }
    // Let the passes that are still running end.
    tokio::time::sleep(Duration::from_millis(1300)).await;
    let most = e.engine.most.load(Ordering::SeqCst);
    assert!(
        hung_up >= 20,
        "the requests were really dropped ({hung_up})"
    );
    assert!(
        most <= 3,
        "{most} passes ran at once: dropping the request freed its place while its pass ran on"
    );
    assert!(most >= 2, "and they did overlap ({most})");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_hangs_up_is_counted_and_keeps_its_place_until_its_pass_ends() {
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(400),
        ..Build::default()
    }));
    make_owner(&e, PASSWORD).await;
    // Two attempts from one address, each dropped 25 ms in (its pass is running).
    for _ in 0..2 {
        let r = tokio::time::timeout(
            Duration::from_millis(25),
            login_as(&e, "wrong wrong wrong", "203.0.113.50"),
        )
        .await;
        assert!(r.is_err(), "the request was dropped");
    }
    // Their passes run on, in the places of the address (two): a third attempt is turned away at once.
    let third = login_as(&e, PASSWORD, "203.0.113.50").await;
    assert_eq!(
        (third.status, third.code().as_deref()),
        (429, Some("rate_limited")),
        "{}",
        third.text
    );
    // When they end, both were counted, though nobody was left to be told.
    tokio::time::sleep(Duration::from_millis(1000)).await;
    assert_eq!(e.state.throttle.recent_failures(), 2);
    // And the places are free again.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.50").await.status, 200);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wrong_passwords_of_a_client_that_hangs_up_still_revoke_a_session_in_the_session_lane() {
    for (path, body) in [
        (
            "/api/auth/elevate",
            json!({ "password": "wrong wrong wrong" }),
        ),
        (
            "/api/auth/password",
            json!({ "current": "wrong wrong wrong", "next": NEW_PASSWORD }),
        ),
    ] {
        let e = Arc::new(build(Build {
            hold: Duration::from_millis(100),
            ..Build::default()
        }));
        let owner = make_owner(&e, PASSWORD).await;
        let b = browser_from(&e, &owner);
        for _ in 0..5 {
            let r = tokio::time::timeout(
                Duration::from_millis(25),
                go(&e, as_page(&e, &b, Method::POST, path).json(body.clone())),
            )
            .await;
            assert!(r.is_err(), "{path}: the request was dropped");
            // Its pass ends (100 ms) before the next request is sent.
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let r = go(&e, as_page(&e, &b, Method::GET, "/api/config")).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("session_expired")),
            "{path}: five wrong answers, though every client hung up, revoke the session"
        );
    }
}

#[test]
fn a_login_allow_list_with_an_entry_that_is_not_an_address_is_refused_and_not_trimmed() {
    let with = |list: &'static str| {
        move |name: &str| (name == "OAIY_LOGIN_ALLOW").then(|| list.to_string())
    };
    // A typo must not turn the restriction off (a list of what parsed, or an empty one, would let everyone in).
    for bad in [
        "203.0.113.0/24x",
        "203.0.113.0/24, oops",
        "oops",
        "203.0.113.0/33",
        "203.0.113.0/24;198.51.100.0/24",
        ",",
    ] {
        let Err(refusal) = LoginOptions::production(&with(bad), 41000) else {
            panic!("{bad:?} was accepted");
        };
        assert!(
            refusal.to_string().contains("OAIY_LOGIN_ALLOW"),
            "{bad:?}: {refusal}"
        );
    }
    let Err(refusal) = LoginOptions::production(&with("203.0.113.0/24, oops, 9.9.9.9/40"), 41000)
    else {
        panic!("accepted");
    };
    assert!(
        refusal.to_string().contains("oops") && refusal.to_string().contains("9.9.9.9/40"),
        "every bad entry is named: {refusal}"
    );
    // What is right is taken, and nothing at all is no restriction.
    let ok = LoginOptions::production(&with(" 203.0.113.0/24, 2001:db8::/32,"), 41000).unwrap();
    assert_eq!(ok.login_allow.len(), 2);
    assert!(LoginOptions::production(&with(""), 41000)
        .unwrap()
        .login_allow
        .is_empty());
    assert!(LoginOptions::production(&|_| None, 41000)
        .unwrap()
        .login_allow
        .is_empty());
}

#[tokio::test]
async fn the_login_allow_list_refuses_other_addresses_before_any_hashing() {
    let e = build(Build {
        login_allow: Some("203.0.113.0/24, 2001:db8::/32"),
        ..Build::default()
    });
    let code = console_setup_code(&e).await;
    let outside = "198.51.100.7";
    let r = go(&e, setup_req(&e, &code, PASSWORD, outside)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("login_not_allowed"))
    );
    let r = go(&e, link_req(&e, "x", 7).from(outside)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("login_not_allowed"))
    );
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), 0);
    // From inside it works, v4 and v6.
    assert_eq!(
        go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.5"))
            .await
            .status,
        201
    );
    let hashed = e.engine.hashes.load(Ordering::SeqCst);
    let r = login_as(&e, PASSWORD, outside).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("login_not_allowed"))
    );
    assert_eq!(
        e.engine.verifies.load(Ordering::SeqCst),
        0,
        "before any hashing"
    );
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), hashed);
    assert_eq!(login_as(&e, PASSWORD, "2001:db8:1::5").await.status, 200);
}

// ==================================== nothing secret is written ======================================

#[tokio::test]
async fn t32_no_secret_is_in_the_audit_the_noise_or_the_files_of_the_data_folder_except_where_it_belongs(
) {
    let e = env();
    let code = console_setup_code(&e).await;
    let r = go(&e, setup_req(&e, &code, PASSWORD, "203.0.113.180")).await;
    let b = browser_from(&e, &r);
    let mut secrets: Vec<String> = vec![
        PASSWORD.into(),
        NEW_PASSWORD.into(),
        "wrong wrong wrong".into(),
        code.clone(),
        code.replace('-', ""),
        b.session.clone(),
        token::parse(&b.session).unwrap().secret.into(),
        b.csrf.clone(),
        b.device.clone().unwrap(),
        token::parse(b.device.as_ref().unwrap())
            .unwrap()
            .secret
            .into(),
    ];
    // A busy day: failures, a link, an elevation, a password change, a logout.
    for i in 0..7 {
        login_as(&e, "wrong wrong wrong", &format!("198.51.100.{}", 30 + i)).await;
    }
    let (link, _) = console_link(&e).await;
    go(&e, link_req(&e, &link, 181)).await;
    secrets.push(link);
    go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/elevate").json(json!({ "password": PASSWORD })),
    )
    .await;
    go(
        &e,
        as_page(&e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
    )
    .await;
    go(&e, as_page(&e, &b, Method::POST, "/api/auth/logout")).await;
    e.state.flush();
    e.guard.audit().unwrap().flush_noise();
    let auth = e.dir.0.join("auth");
    for file in [
        "audit.jsonl",
        "noise.jsonl",
        "owner.json",
        "credentials.json",
        "throttle.json",
    ] {
        let text = std::fs::read_to_string(auth.join(file)).unwrap_or_default();
        for s in &secrets {
            assert!(!text.contains(s.as_str()), "{file} holds a secret: {s}");
        }
    }
    assert!(!auth.join("setup-code.json").exists());
    // And what was written says what happened.
    for event in [
        "setup.ok",
        "link.issued",
        "link.used",
        "elevate.ok",
        "password.changed",
        "logout",
    ] {
        assert!(has_event(&e, event), "{event}");
    }
}

#[tokio::test]
async fn the_audit_names_who_and_from_where_for_a_login() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .from("203.0.113.190")
            .h("user-agent", "Mozilla/5.0 (X11; Linux x86_64) test")
            .browser()
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 200);
    let events = audit_events(&e);
    let login = events
        .iter()
        .find(|e| e["event"] == "login.ok" && e["ip"] == "203.0.113.190")
        .expect("a login.ok");
    assert_eq!(login["principal"]["kind"], "session");
    assert_eq!(login["principal"]["label"], "dashboard");
    assert_eq!(login["host"], "dash.example.com");
    assert_eq!(login["ua"], "Mozilla/5.0 (X11; Linux x86_64) test");
    assert_eq!(login["detail"]["device"], false);
    // The list of sessions shows the browser it was made by.
    let b = browser_from(&e, &r);
    let list = go(&e, as_page(&e, &b, Method::GET, "/api/auth/sessions"))
        .await
        .json();
    let mine = list["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["current"] == true)
        .cloned()
        .unwrap();
    assert_eq!(mine["userAgent"], "Mozilla/5.0 (X11; Linux x86_64) test");
}

// ==================================== fuzzing =========================================================

#[tokio::test]
async fn fuzzing_the_login_routes_with_odd_bodies_and_headers_never_panics_and_never_says_500() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let mut state = 0x5eed_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 24
    };
    let routes = [
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/link",
        "/api/auth/session",
    ];
    let types = [
        "application/json",
        "text/plain",
        "application/x-www-form-urlencoded",
        "application/json; charset=utf-8",
        "",
        "APPLICATION/JSON",
    ];
    let bodies: [&[u8]; 12] = [
        br#"{"password":"x"}"#,
        br#"{"password":["a"]}"#,
        br#"{"password":"a","remember":"yes"}"#,
        br#"{"code":1}"#,
        br#"{"code":"AAAA-AAAA-AAAA","password":"x"}"#,
        b"\xff\xfe\x00",
        b"{",
        b"{\"password\":\"\\ud800\"}",
        b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[",
        b"{\"a\":{\"a\":{\"a\":{\"a\":{\"a\":{\"a\":{\"a\":1}}}}}}}",
        b"",
        b"null",
    ];
    let cookies = [
        "",
        "__Host-oaiy_dev=garbage",
        "__Host-oaiy_dev=a; __Host-oaiy_dev=b",
        "__Host-oaiy_dash=oaiyses_x",
        "\t;;;=;==",
    ];
    for round in 0..300 {
        let path = routes[(next() % 4) as usize];
        let method = if path == "/api/auth/session" {
            Method::GET
        } else {
            Method::POST
        };
        let from = format!("198.51.{}.{}", next() % 250, next() % 250);
        let mut r = req(&e, method, path).from(&from);
        let t = types[(next() % types.len() as u64) as usize];
        if !t.is_empty() {
            r = r.h("content-type", t);
        }
        let c = cookies[(next() % cookies.len() as u64) as usize];
        if !c.is_empty() {
            r = r.h("cookie", c);
        }
        if next() % 3 == 0 {
            r = r.browser();
        }
        if next() % 5 == 0 {
            r = r.h(
                "origin",
                ["null", "https://x.example", "", "https://dash.example.com"]
                    [(next() % 4) as usize],
            );
        }
        r = r.raw(bodies[(next() % bodies.len() as u64) as usize]);
        let reply = go(&e, r).await;
        assert!(
            reply.status < 500 || reply.status == 503,
            "round {round}: {} {}",
            reply.status,
            reply.text
        );
        assert!(
            serde_json::from_str::<Value>(&reply.text).is_ok() || reply.text.is_empty(),
            "round {round}: {}",
            reply.text
        );
        assert!(!reply.text.contains(PASSWORD));
    }
}

// ==================================== the log and the alert (T44) =====================================

/// The counts the noise log holds for one address, by event.
fn noise_counts(e: &Env, ip: &str) -> std::collections::BTreeMap<String, u64> {
    e.guard.audit().unwrap().flush_noise();
    let text =
        std::fs::read_to_string(e.dir.0.join("auth").join("noise.jsonl")).unwrap_or_default();
    let mut counts = std::collections::BTreeMap::new();
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).unwrap();
        if v["ip"] == ip {
            *counts
                .entry(v["event"].as_str().unwrap().to_string())
                .or_insert(0) += v["count"].as_u64().unwrap();
        }
    }
    counts
}

#[tokio::test]
async fn a_block_writes_its_start_once_and_the_requests_it_refuses_write_nothing() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    for _ in 0..5 {
        wrong(&e, "203.0.113.200").await;
    }
    for _ in 0..10 {
        assert_eq!(login_as(&e, PASSWORD, "203.0.113.200").await.status, 429);
    }
    let counts = noise_counts(&e, "203.0.113.200");
    assert_eq!(counts.get("login.fail"), Some(&5), "{counts:?}");
    assert_eq!(
        counts.get("login.blocked"),
        Some(&1),
        "the start of the block, and none for what it refused: {counts:?}"
    );
}

#[tokio::test]
async fn a_wrong_setup_code_that_begins_a_block_writes_its_start_too() {
    let e = env();
    console_setup_code(&e).await;
    for _ in 0..5 {
        go(
            &e,
            setup_req(&e, "AAAA-AAAA-AAAA", PASSWORD, "203.0.113.201"),
        )
        .await;
    }
    let counts = noise_counts(&e, "203.0.113.201");
    assert_eq!(counts.get("setup.fail"), Some(&5), "{counts:?}");
    assert_eq!(counts.get("login.blocked"), Some(&1), "{counts:?}");
}

#[tokio::test]
async fn t44_slow_mode_for_an_hour_raises_login_attack_once_and_the_page_and_the_console_say_so() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let fail = |e: &Env, n: usize| {
        for i in 0..n {
            e.state
                .throttle
                .address_failed(&format!("2001:db8:{i:x}::"));
        }
    };
    fail(&e, 20);
    e.state.tick();
    assert!(!has_event(&e, "login.attack"), "not yet");
    // Slow mode stays on: twenty fresh failures every half hour, for an hour.
    for _ in 0..2 {
        e.clock.advance(30 * MIN);
        fail(&e, 20);
        e.state.tick();
    }
    let attacks = |e: &Env| {
        audit_events(e)
            .iter()
            .filter(|x| x["event"] == "login.attack")
            .count()
    };
    assert_eq!(attacks(&e), 1, "raised once");
    let event = audit_events(&e)
        .into_iter()
        .find(|x| x["event"] == "login.attack")
        .unwrap();
    assert_eq!(event["detail"]["reason"], "slow_mode_60_minutes");
    // Not again within the hour, however often the upkeep runs.
    for _ in 0..5 {
        e.state.tick();
    }
    assert_eq!(attacks(&e), 1);
    // The dashboard's banner and the console's status say so.
    let s = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(s["loginAttack"], true);
    let con = console_token(&e);
    let status = go(
        &e,
        req(&e, Method::GET, "/api/auth/console/status").bearer(&con),
    )
    .await
    .json();
    assert_eq!(status["underAttack"], "slow_mode_60_minutes");
    assert_eq!(status["slowMode"], true);
    // An hour later, with the failures aged out, the banner is gone.
    e.clock.advance(2 * HOUR);
    let s = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert!(s.get("loginAttack").is_none(), "{s}");
}

// ==================================== the routes carry no CORS =======================================

#[tokio::test]
async fn a_preflight_for_a_login_route_from_another_origin_is_answered_without_cors_headers() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    for (method, path) in [
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/setup"),
        ("POST", "/api/auth/link"),
        ("GET", "/api/auth/session"),
        ("GET", "/api/auth/info"),
    ] {
        let r = go(
            &e,
            req(&e, Method::OPTIONS, path)
                .h("origin", "https://evil.example")
                .h("access-control-request-method", method)
                .h("access-control-request-headers", "content-type"),
        )
        .await;
        assert_eq!(r.status, 204, "{path}");
        assert!(
            r.headers
                .keys()
                .all(|k| !k.as_str().starts_with("access-control-")),
            "{path}: {:?}",
            r.headers
        );
    }
}

// ==================================== rules the mutation check found unwatched ========================

/// The owner file as written on disk, as JSON.
fn owner_file(e: &Env) -> Value {
    serde_json::from_str(&std::fs::read_to_string(e.dir.0.join("auth").join("owner.json")).unwrap())
        .unwrap()
}

#[tokio::test]
async fn the_session_endpoint_does_not_read_a_cookie_that_came_over_a_channel_that_is_not_secure() {
    // A loopback server and its dashboard name, from a peer that is not this machine: no cookie travels there.
    let e = build(Build {
        proxied: false,
        ..Build::default()
    });
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let cookie = format!("{}={}", session_cookie_name(&e), b.session);
    let near = go(
        &e,
        req(&e, Method::GET, "/api/auth/session").cookie(&cookie),
    )
    .await
    .json();
    assert_eq!(near["authenticated"], true);
    let far = go(
        &e,
        req(&e, Method::GET, "/api/auth/session")
            .from("192.0.2.10")
            .cookie(&cookie),
    )
    .await
    .json();
    assert_eq!(far["authenticated"], false, "{far}");
}

#[tokio::test]
async fn a_throttle_file_that_cannot_be_read_starts_empty_throttles_and_the_server_starts() {
    // Not JSON, and then a folder where the file belongs: what it held was a memory of failures, not a credential,
    // so the server starts with no blocks (for the login's throttle as for the failed-bearer throttle's).
    for as_folder in [false, true] {
        let e = env();
        make_owner(&e, PASSWORD).await;
        for _ in 0..5 {
            wrong(&e, "203.0.113.61").await;
        }
        e.state.flush();
        let path = e.dir.0.join("auth").join("throttle.json");
        std::fs::remove_file(&path).unwrap();
        if as_folder {
            std::fs::create_dir(&path).unwrap();
        } else {
            std::fs::write(&path, "{ not json").unwrap();
        }
        let e2 = restart(e, MIN);
        assert_eq!(e2.state.throttle.tracked(), 0);
        assert_eq!(login_as(&e2, PASSWORD, "203.0.113.61").await.status, 200);
    }
}

#[tokio::test]
async fn the_login_and_the_bearer_throttles_share_one_file_and_neither_writes_over_the_other() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let file = |e: &Env| -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(e.dir.0.join("auth").join("throttle.json")).unwrap(),
        )
        .unwrap()
    };
    // A block of the login's, written by the login ...
    for _ in 0..5 {
        wrong(&e, "203.0.113.61").await;
    }
    e.state.flush();
    let doc = file(&e);
    assert_eq!(doc["v"], 1);
    assert!(
        doc["login"]["addresses"].get("203.0.113.61").is_some(),
        "{doc}"
    );
    // ... then a failure of the failed-bearer throttle's, written by the guard: the login's block is still in the file.
    e.guard.throttle().record_failure("198.51.100.5");
    e.guard.flush_throttle();
    let doc = file(&e);
    assert!(doc["bearer"].get("198.51.100.5").is_some(), "{doc}");
    assert!(
        doc["login"]["addresses"].get("203.0.113.61").is_some(),
        "the guard's write keeps the login's state: {doc}"
    );
    // And the login's next write keeps the bearer's.
    for _ in 0..5 {
        wrong(&e, "203.0.113.62").await;
    }
    e.state.flush();
    let doc = file(&e);
    assert!(doc["bearer"].get("198.51.100.5").is_some(), "{doc}");
    assert!(
        doc["login"]["addresses"].get("203.0.113.62").is_some(),
        "{doc}"
    );
}

#[tokio::test]
async fn the_bearer_throttle_is_saved_with_the_login_throttle_and_a_restart_keeps_its_block() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    let key = "203.0.113.77";
    let mut began = false;
    for _ in 0..20 {
        began = e.guard.throttle().record_failure(key);
    }
    assert!(began && e.guard.throttle().blocked_for(key).is_some());
    e.state.flush();
    let e2 = restart(e, 2 * MIN);
    let left = e2
        .guard
        .throttle()
        .blocked_for(key)
        .expect("still blocked after a restart");
    assert!(
        (779..=780).contains(&left),
        "fifteen minutes less the two that passed: {left}"
    );
}

#[tokio::test]
async fn a_good_login_with_a_hash_cheaper_than_the_current_cost_hashes_the_password_again_and_a_current_one_is_left_alone(
) {
    // The tests' own hashes are at the cheapest cost the verifier accepts: cheaper than the current one.
    let e = env();
    make_owner(&e, PASSWORD).await;
    let stored = |e: &Env| owner_file(e)["password"].as_str().unwrap().to_string();
    let before = stored(&e);
    let hashes = e.engine.hashes.load(Ordering::SeqCst);
    // A wrong password shows nothing about the hash: nothing is made again.
    assert_eq!(wrong(&e, "203.0.113.31").await.status, 401);
    assert_eq!(stored(&e), before);
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), hashes);
    // The right one: hashed again, written, and the new string is what is checked from then on.
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.32").await.status, 200);
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), hashes + 1);
    let after = stored(&e);
    assert_ne!(after, before, "a new salt makes a new string");
    assert!(after.starts_with("$argon2id$"));
    assert_eq!(login_as(&e, PASSWORD, "203.0.113.33").await.status, 200);
    // The design's own string is at the current cost: a right password with it is not hashed again.
    const CURRENT: &str = "$argon2id$v=19$m=65536,t=3,p=1$AAECAwQFBgcICQoLDA0ODw$DRo8ZSPI8G5OCvnFFapbVEjP69aDjy1Sw9i2743cPC4";
    let dir = TempDir::new("login-current-cost");
    let auth = dir.0.join("auth");
    std::fs::create_dir_all(&auth).unwrap();
    let doc = json!({ "v": 1, "created_ms": T0, "password_changed_ms": T0, "password": CURRENT });
    std::fs::write(auth.join("owner.json"), doc.to_string()).unwrap();
    let e = build(Build {
        dir: Some(dir),
        ..Build::default()
    });
    let r = login_as(&e, "correct horse battery staple", "203.0.113.34").await;
    assert_eq!(r.status, 200, "{}", r.text);
    assert_eq!(e.engine.hashes.load(Ordering::SeqCst), 0);
    assert_eq!(stored(&e), CURRENT);
}

#[tokio::test]
async fn a_login_from_a_known_device_records_its_use_and_the_upkeep_writes_it() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    e.clock.advance(3 * HOUR);
    let at = e.clock.now_ms();
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/login")
            .from("198.51.100.77")
            .browser()
            .cookie(&device_pair(&e, &b))
            .json(json!({ "password": PASSWORD })),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let signed_in = browser_from(&e, &r);
    let list = go(
        &e,
        as_page(&e, &signed_in, Method::GET, "/api/auth/sessions"),
    )
    .await
    .json();
    let device = &list["devices"][0];
    assert_eq!(
        (device["lastMs"].as_u64(), device["ip"].as_str()),
        (Some(at), Some("198.51.100.77")),
        "{list}"
    );
    // A use is written at the next upkeep, not at the login.
    e.state.flush();
    assert_eq!(owner_file(&e)["devices"][0]["last_ms"].as_u64(), Some(at));
}

#[tokio::test]
async fn the_seventeenth_device_pushes_out_the_oldest_and_its_ladder_with_it() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let first = browser_from(&e, &owner);
    // The first device is blocked on its own ladder.
    for _ in 0..5 {
        let r = go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .from("203.0.113.120")
                .browser()
                .cookie(&device_pair(&e, &first))
                .json(json!({ "password": "wrong wrong wrong" })),
        )
        .await;
        assert_eq!(r.status, 401);
    }
    assert_eq!(e.state.throttle.blocked_devices().len(), 1);
    // Sixteen more browsers sign in: the list holds sixteen and the first is gone.
    for i in 0..16 {
        let r = login_as(&e, PASSWORD, &format!("198.51.100.{}", 10 + i)).await;
        assert_eq!(r.status, 200, "{}", r.text);
    }
    assert_eq!(owner_file(&e)["devices"].as_array().unwrap().len(), 16);
    assert!(
        e.state.throttle.blocked_devices().is_empty(),
        "the ladder of a device that is gone is gone"
    );
}

#[tokio::test]
async fn a_session_link_is_for_an_owner_that_exists_and_a_setup_code_for_one_that_does_not() {
    let e = env();
    let con = console_token(&e);
    let link = |e: &Env| {
        req(e, Method::POST, "/api/auth/console/session-link")
            .bearer(&con)
            .json(json!({}))
    };
    let r = go(&e, link(&e)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (409, Some("setup_required"))
    );
    assert_eq!(e.state.links_outstanding(), 0);
    make_owner(&e, PASSWORD).await;
    assert_eq!(go(&e, link(&e)).await.status, 200);
    let r = go(
        &e,
        req(&e, Method::POST, "/api/auth/console/setup-code")
            .bearer(&con)
            .json(json!({})),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (409, Some("already_configured"))
    );
}

#[tokio::test]
async fn the_console_files_are_written_once_the_port_is_known_and_a_clean_exit_removes_them() {
    let e = env();
    let auth = e.dir.0.join("auth");
    assert!(!auth.join(console::TOKEN_FILE).exists());
    console::publish(&e.state, 41000).unwrap();
    let credential = std::fs::read_to_string(auth.join(console::TOKEN_FILE)).unwrap();
    assert!(credential.starts_with("oaiycon_") && credential.ends_with('\n'));
    let info: Value =
        serde_json::from_str(&std::fs::read_to_string(auth.join(console::INFO_FILE)).unwrap())
            .unwrap();
    assert_eq!(info["port"], 41000);
    // The credential in the file is the one the running server accepts.
    let r = go(
        &e,
        req(&e, Method::GET, "/api/auth/console/status").bearer(credential.trim()),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    console::remove_files(&auth);
    assert!(!auth.join(console::TOKEN_FILE).exists() && !auth.join(console::INFO_FILE).exists());
    // Nothing to remove is not an error, and says nothing.
    console::remove_files(&auth);
}

#[tokio::test]
async fn a_right_password_ends_the_run_of_wrong_ones_in_the_session_lane() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let elevate = |e: &Env, password: &str| {
        as_page(e, &b, Method::POST, "/api/auth/elevate").json(json!({ "password": password }))
    };
    let change = |e: &Env, current: &str| {
        as_page(e, &b, Method::POST, "/api/auth/password")
            .json(json!({ "current": current, "next": "short" }))
    };
    // Four wrong, one right, four wrong: never five in a row, so the session lives.
    for _ in 0..4 {
        assert_eq!(go(&e, elevate(&e, "wrong wrong wrong")).await.status, 401);
    }
    assert_eq!(go(&e, elevate(&e, PASSWORD)).await.status, 200);
    for _ in 0..4 {
        assert_eq!(go(&e, elevate(&e, "wrong wrong wrong")).await.status, 401);
    }
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
    // The same for the password change, whose right answer is the one whose new password is refused. The run of
    // four wrong ones the elevations left is ended by the first; then four wrong, a right one and three wrong.
    e.clock.advance(HOUR);
    let right_and_weak = |r: &Reply| (r.status, r.code());
    assert_eq!(
        right_and_weak(&go(&e, change(&e, PASSWORD)).await),
        (400, Some("weak_password".to_string()))
    );
    for _ in 0..4 {
        assert_eq!(go(&e, change(&e, "wrong wrong wrong")).await.status, 401);
    }
    assert_eq!(
        right_and_weak(&go(&e, change(&e, PASSWORD)).await),
        (400, Some("weak_password".to_string()))
    );
    for _ in 0..3 {
        assert_eq!(go(&e, change(&e, "wrong wrong wrong")).await.status, 401);
    }
    assert_eq!(
        go(&e, as_page(&e, &b, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
}

async fn console_status(e: &Env, con: &str) -> Value {
    go(
        e,
        req(e, Method::GET, "/api/auth/console/status").bearer(con),
    )
    .await
    .json()
}

#[tokio::test]
async fn the_list_holds_live_sessions_only_and_the_revoking_routes_touch_nothing_else() {
    let e = env();
    make_owner(&e, PASSWORD).await;
    // The caller is remembered (fourteen idle days), so it outlives the plain session of the setup (a day).
    let caller = browser_from(
        &e,
        &go(
            &e,
            req(&e, Method::POST, "/api/auth/login")
                .from("198.51.100.20")
                .browser()
                .json(json!({ "password": PASSWORD, "remember": true })),
        )
        .await,
    );
    let pat = a_pat(&e);
    let con = console_token(&e);
    e.clock.advance(25 * HOUR);
    let other = browser_from(&e, &login_as(&e, PASSWORD, "198.51.100.21").await);
    // The setup's session has ended by age, and a paired token and the console credential are not sessions.
    let list = go(&e, as_page(&e, &caller, Method::GET, "/api/auth/sessions"))
        .await
        .json();
    let sessions = list["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 2, "{list}");
    assert_eq!(sessions.iter().filter(|s| s["current"] == true).count(), 1);
    // By id, a paired token or the console credential is nothing to the route that revokes a session or a device.
    let con_id = token::parse(&con).unwrap().id.to_string();
    for id in [pat.id.clone(), con_id] {
        let r = go(
            &e,
            as_page(
                &e,
                &caller,
                Method::DELETE,
                &format!("/api/auth/sessions/{id}"),
            ),
        )
        .await;
        assert_eq!(r.status, 404, "{id}");
    }
    assert!(e.store.authenticate(&pat.token, None).is_ok());
    assert!(e.store.authenticate(&con, None).is_ok());
    // The status counts the live ones.
    let s = console_status(&e, &con).await;
    assert_eq!(
        (s["sessions"].as_u64(), s["tokens"].as_u64()),
        (Some(2), Some(1))
    );
    // Revoking the others takes the other session and only that.
    assert_eq!(
        go(
            &e,
            as_page(
                &e,
                &caller,
                Method::POST,
                "/api/auth/sessions/revoke-others"
            )
        )
        .await
        .status,
        204
    );
    assert_eq!(
        go(&e, as_page(&e, &other, Method::GET, "/api/config"))
            .await
            .status,
        401
    );
    assert_eq!(
        go(&e, as_page(&e, &caller, Method::GET, "/api/config"))
            .await
            .status,
        200
    );
    assert!(e.store.authenticate(&pat.token, None).is_ok());
    assert!(e.store.authenticate(&con, None).is_ok());
    let s = console_status(&e, &con).await;
    assert_eq!(
        (s["sessions"].as_u64(), s["tokens"].as_u64()),
        (Some(1), Some(1)),
        "a revoked session is not counted"
    );
}

#[tokio::test]
async fn the_banner_is_for_an_install_with_no_owner_and_holds_no_secret() {
    let e = env();
    let banner = e.state.banner().expect("no owner yet: a banner");
    assert!(
        banner.contains("auth setup-code") && banner.contains("https://dash.example.com/setup"),
        "{banner}"
    );
    assert!(
        !banner.contains("oaiy_") && !banner.contains("oaiycon_"),
        "{banner}"
    );
    make_owner(&e, PASSWORD).await;
    assert_eq!(e.state.banner(), None);
}

#[tokio::test]
async fn the_login_attack_flag_is_the_dashboards_and_no_other_apps() {
    let e = env();
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let (child, _) = child_session(&e, &b, super::presets::App::Agent);
    for round in 0..3 {
        if round > 0 {
            e.clock.advance(30 * MIN);
        }
        for i in 0..20 {
            e.state
                .throttle
                .address_failed(&format!("2001:db8:{i:x}::"));
        }
        e.state.tick();
    }
    let dash = go(&e, as_page(&e, &b, Method::GET, "/api/auth/session"))
        .await
        .json();
    assert_eq!(dash["loginAttack"], true, "{dash}");
    let agent = go(
        &e,
        req(&e, Method::GET, "/api/auth/session")
            .host("agent.example.com")
            .cookie(&format!("__Host-oaiy_agent={child}")),
    )
    .await
    .json();
    assert_eq!(agent["authenticated"], true, "{agent}");
    assert!(agent.get("loginAttack").is_none(), "{agent}");
}

// ==================================== the owner file has one writer at a time ==========================

/// The password hash in `owner.json`, as a restart would read it.
fn hash_on_disk(e: &Env) -> String {
    owner_file(e)["password"].as_str().unwrap().to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_write_of_the_old_owner_file_does_not_undo_a_password_change() {
    // Two ways to change the password: the owner's own, and the console's.
    for by_console in [false, true] {
        let e = Arc::new(build(Build::default()));
        let owner = make_owner(&e, PASSWORD).await;
        let b = browser_from(&e, &owner);
        let old_hash = hash_on_disk(&e);
        // A write of the owner file with the old hash in it is slow (a busy disk).
        *e.disk.slow.lock().unwrap() =
            Some((old_hash.clone().into_bytes(), Duration::from_millis(500)));
        // Revoking a device writes the owner file from the owner as it is in memory, and here it is slow.
        let device = token::parse(b.device.as_ref().unwrap())
            .unwrap()
            .id
            .to_string();
        let (e2, b2) = (e.clone(), (b.session.clone(), b.csrf.clone()));
        let slow = tokio::spawn(async move {
            let browser = Browser {
                session: b2.0,
                csrf: b2.1,
                device: None,
            };
            go(
                &e2,
                as_page(
                    &e2,
                    &browser,
                    Method::DELETE,
                    &format!("/api/auth/sessions/{device}"),
                ),
            )
            .await
            .status
        });
        tokio::time::sleep(Duration::from_millis(150)).await;
        // The password change, while that write is still in flight.
        let status = if by_console {
            let con = console_token(&e);
            go(
                &e,
                req(&e, Method::POST, "/api/auth/console/reset-password")
                    .bearer(&con)
                    .json(json!({ "password": NEW_PASSWORD })),
            )
            .await
            .status
        } else {
            go(
                &e,
                as_page(&e, &b, Method::POST, "/api/auth/password")
                    .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
            )
            .await
            .status
        };
        assert!(matches!(status, 200 | 204), "{by_console}: {status}");
        assert_eq!(slow.await.unwrap(), 204, "the device was revoked");
        assert_ne!(
            hash_on_disk(&e),
            old_hash,
            "{by_console}: owner.json holds the OLD password hash again after the change: a restart would accept the old password"
        );
        // And a restart reads the new password.
        let e3 = restart(
            Arc::try_unwrap(e)
                .ok()
                .expect("nothing else holds the server"),
            MIN,
        );
        assert_eq!(login_as(&e3, PASSWORD, "203.0.113.9").await.status, 401);
        assert_eq!(
            login_as(&e3, NEW_PASSWORD, "203.0.113.10").await.status,
            200
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_password_changes_at_once_from_the_same_password_make_one_change_and_refuse_the_other()
{
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(150),
        ..Build::default()
    }));
    let owner = make_owner(&e, PASSWORD).await;
    let first = browser_from(&e, &owner);
    let second = browser_from(&e, &login_as(&e, PASSWORD, "198.51.100.30").await);
    // Writing the owner file is slow, so that the second change has time to read the owner while the first is
    // still writing its own: only the gate stands between them.
    *e.disk.slow.lock().unwrap() =
        Some((b"password_changed_ms".to_vec(), Duration::from_millis(300)));
    let change = |e: Arc<Env>, b: Browser, next: &'static str| {
        tokio::spawn(async move {
            go(
                &e,
                as_page(&e, &b, Method::POST, "/api/auth/password")
                    .json(json!({ "current": PASSWORD, "next": next })),
            )
            .await
            .status
        })
    };
    let (a, b) = (
        change(e.clone(), first, NEW_PASSWORD),
        change(e.clone(), second, "Qm7&rT2!vX9# kd4Lp"),
    );
    let mut statuses = [a.await.unwrap(), b.await.unwrap()];
    statuses.sort_unstable();
    assert_eq!(
        statuses,
        [204, 401],
        "the second change asked with a password that was no longer the owner's"
    );
    // One of the two new passwords is the owner's now, and the old one is not.
    assert_eq!(login_as(&e, PASSWORD, "198.51.100.31").await.status, 401);
    let mut works = 0;
    for (i, pw) in [NEW_PASSWORD, "Qm7&rT2!vX9# kd4Lp"].iter().enumerate() {
        works += usize::from(
            login_as(&e, pw, &format!("198.51.100.{}", 32 + i))
                .await
                .status
                == 200,
        );
    }
    assert_eq!(works, 1);
}

// ==================================== a password change is not outlived by a login in flight ==========

/// A login with `password` from `from`, with a device cookie of no browser: the anonymous lane.
async fn late_login(e: &Env, password: &str, from: &str) -> Reply {
    login_as(e, password, from).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_login_that_verified_the_old_password_does_not_outlive_the_password_change() {
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(300),
        ..Build::default()
    }));
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let (e1, session, csrf) = (e.clone(), b.session.clone(), b.csrf.clone());
    let change = tokio::spawn(async move {
        let browser = Browser {
            session,
            csrf,
            device: None,
        };
        go(
            &e1,
            as_page(&e1, &browser, Method::POST, "/api/auth/password")
                .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
        )
        .await
        .status
    });
    // The change verifies the current password and then hashes the new one (300 ms each) and is written at about
    // 600 ms; a login with the old password starts at 450 ms, reads the old hash and verifies it (300 ms), so it
    // finishes at 750 ms, after the change is written.
    tokio::time::sleep(Duration::from_millis(450)).await;
    let e2 = e.clone();
    let late = tokio::spawn(async move { late_login(&e2, PASSWORD, "198.51.100.44").await });
    assert_eq!(change.await.unwrap(), 204);
    let late = late.await.unwrap();
    assert_eq!(
        (late.status, late.code().as_deref()),
        (401, Some("invalid_credentials")),
        "a login that verified the old password, and finished after the change, is refused like a wrong password: {}",
        late.text
    );
    assert!(
        late.cookies().is_empty(),
        "no session and no device cookie: {:?}",
        late.cookies()
    );
    assert_eq!(
        owner_file(&e)["devices"].as_array().unwrap().len(),
        0,
        "and no device was made in the owner file"
    );
    // Which password does the owner have, when everything has settled?
    assert_eq!(login_as(&e, PASSWORD, "198.51.100.45").await.status, 401);
    assert_eq!(
        login_as(&e, NEW_PASSWORD, "198.51.100.46").await.status,
        200
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_login_that_verified_the_old_password_does_not_outlive_a_console_reset() {
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(300),
        ..Build::default()
    }));
    make_owner(&e, PASSWORD).await;
    let con = console_token(&e);
    let (e1, con1) = (e.clone(), con.clone());
    let reset = tokio::spawn(async move {
        go(
            &e1,
            req(&e1, Method::POST, "/api/auth/console/reset-password")
                .bearer(&con1)
                .json(json!({ "password": NEW_PASSWORD })),
        )
        .await
        .status
    });
    // The reset hashes the new password (300 ms) and is written at about 310 ms; the login starts at 200 ms, reads
    // the old hash, and finishes verifying it (300 ms) at 500 ms, after the reset is written.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let late = late_login(&e, PASSWORD, "198.51.100.47").await;
    assert_eq!(reset.await.unwrap(), 200);
    assert_eq!(
        (late.status, late.code().as_deref()),
        (401, Some("invalid_credentials")),
        "{}",
        late.text
    );
    assert_eq!(login_as(&e, PASSWORD, "198.51.100.48").await.status, 401);
    assert_eq!(
        login_as(&e, NEW_PASSWORD, "198.51.100.49").await.status,
        200
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_re_hash_does_not_write_the_old_password_over_one_that_was_changed_while_it_hashed() {
    // The stored hash of the tests is cheaper than the current cost, so every good login hashes the password again.
    let e = Arc::new(build(Build {
        hold: Duration::from_millis(300),
        ..Build::default()
    }));
    let owner = make_owner(&e, PASSWORD).await;
    let b = browser_from(&e, &owner);
    let (e1, session, csrf) = (e.clone(), b.session.clone(), b.csrf.clone());
    let change = tokio::spawn(async move {
        let browser = Browser {
            session,
            csrf,
            device: None,
        };
        go(
            &e1,
            as_page(&e1, &browser, Method::POST, "/api/auth/password")
                .json(json!({ "current": PASSWORD, "next": NEW_PASSWORD })),
        )
        .await
        .status
    });
    // A login with the old password starts 150 ms into the change: it verifies (300 ms) before the change is written
    // (at about 610 ms), so it is a good login; its re-hash (300 ms more) ends at 750 ms, after the change is written.
    tokio::time::sleep(Duration::from_millis(150)).await;
    let late = late_login(&e, PASSWORD, "198.51.100.50").await;
    assert_eq!(change.await.unwrap(), 204);
    assert_eq!(
        late.status, 200,
        "the login was good when it verified: {}",
        late.text
    );
    // The change came after the login made its session, so it revoked it.
    let cookie = format!(
        "{}={}",
        session_cookie_name(&e),
        late.cookie(session_cookie_name(&e)).unwrap()
    );
    let who = go(&e, req(&e, Method::GET, "/api/config").cookie(&cookie)).await;
    assert_eq!(
        who.status, 401,
        "the session made before the change is revoked by it: {}",
        who.text
    );
    // Which password does the owner have? The re-hash did not undo the change.
    assert_eq!(
        login_as(&e, PASSWORD, "198.51.100.51").await.status,
        401,
        "the old password works again: the re-hash of the login wrote over the changed password"
    );
    assert_eq!(
        login_as(&e, NEW_PASSWORD, "198.51.100.52").await.status,
        200
    );
    // And the file, as a restart would read it.
    let e3 = restart(
        Arc::try_unwrap(e)
            .ok()
            .expect("nothing else holds the server"),
        MIN,
    );
    assert_eq!(login_as(&e3, PASSWORD, "198.51.100.53").await.status, 401);
    assert_eq!(
        login_as(&e3, NEW_PASSWORD, "198.51.100.54").await.status,
        200
    );
}
