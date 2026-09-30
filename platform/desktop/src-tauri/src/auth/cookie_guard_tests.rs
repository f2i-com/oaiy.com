#![cfg(feature = "web")]
//! The guard's cookie path, end to end in process: the session cookie of the host's app, the checks of 4.5.2, the
//! two shapes of cookie (behind a proxy, and on a loopback server), setup-only mode, and that an install with no
//! web login reads no cookie at all.
//!
//! A stub router answers `200 ok` for every row of the route table behind the real guard, as `guard_tests.rs` does.
//! The tests are the design's T6 (a cookie request from another origin, a sibling, another port, no `Origin`,
//! `null`, no or a wrong `X-OAIY-CSRF`, `Sec-Fetch-Site: same-site`, the same cookie twice), T7 (the dashboard cookie
//! at the Agent host) and the rules of 4.7.1 for the setup-only answer.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{HeaderMap, Method, Request};
use axum::middleware;
use axum::routing::any;
use axum::Router;
use serde_json::Value;
use tower::ServiceExt;

use super::api;
use super::audit::AuditLog;
use super::clock::{Clock, ManualClock};
use super::guard::{scoped_cors, scoped_guard, Guard, GuardConfig};
use super::mode::AccessMode;
use super::presets::App;
use super::routes::ROUTES;
use super::session::{self, LoginFacts};
use super::store::{AuthStore, MintSpec};
use super::token::{self, Kind};
use crate::secret_file::testing::TempDir;

const T0: u64 = 1_790_000_000_000;
const HOUR: u64 = 3_600_000;
const REAL: [&str; 3] = ["/api/auth/info", "/api/auth/whoami", "/api/auth/derive"];

struct Facts {
    owner: AtomicBool,
}

impl LoginFacts for Facts {
    fn owner_configured(&self) -> bool {
        self.owner.load(Ordering::SeqCst)
    }
    fn setup_code_status(&self) -> &'static str {
        "none"
    }
    fn under_attack(&self) -> bool {
        false
    }
    fn flush(&self) {}
}

struct Env {
    store: Arc<AuthStore>,
    clock: Arc<ManualClock>,
    guard: Arc<Guard>,
    facts: Arc<Facts>,
    app: Router,
    _dir: TempDir,
}

/// `proxied`: three public hosts behind a proxy that is the loopback peer; else a loopback server.
fn env(proxied: bool, install_login: bool) -> Env {
    let dir = TempDir::new("cookie-guard");
    let clock = Arc::new(ManualClock::new(T0));
    let vars: Vec<(&str, &str)> = if proxied {
        vec![
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_AGENT_URL", "https://agent.example.com"),
            ("OAIY_FLOWS_URL", "https://flows.example.com"),
        ]
    } else {
        vec![]
    };
    let map: std::collections::BTreeMap<String, String> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let (config, warnings) =
        GuardConfig::from_env(&move |n| map.get(n).cloned(), false, false, 41000);
    assert!(warnings.is_empty(), "{warnings:?}");
    let store = Arc::new(AuthStore::memory(clock.clone()));
    let audit = Arc::new(AuditLog::open(&dir.0.join("auth"), clock.clone(), false));
    let guard = Arc::new(Guard::new(
        AccessMode::Scoped,
        config,
        store.clone(),
        None,
        Some(audit),
        clock.clone(),
    ));
    let facts = Arc::new(Facts {
        owner: AtomicBool::new(true),
    });
    if install_login {
        let weak: std::sync::Weak<dyn LoginFacts> =
            Arc::downgrade(&facts) as std::sync::Weak<dyn LoginFacts>;
        guard.install_login(weak);
    }
    let mut patterns: Vec<&str> = ROUTES
        .iter()
        .map(|r| r.pattern)
        .filter(|p| !REAL.contains(p))
        .collect();
    patterns.sort_unstable();
    patterns.dedup();
    let mut app = Router::new();
    for p in patterns {
        app = app.route(p, any(|| async { "ok" }));
    }
    let app = app
        .merge(api::router(guard.clone()))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_guard))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_cors));
    Env {
        store,
        clock,
        guard,
        facts,
        app,
        _dir: dir,
    }
}

/// Where a request is from: the host a browser used, the peer, and the headers a proxy adds.
#[derive(Clone)]
struct Site {
    host: &'static str,
    origin: &'static str,
    forwarded: bool,
    peer: &'static str,
}

const DASH_PROXIED: Site = Site {
    host: "dash.example.com",
    origin: "https://dash.example.com",
    forwarded: true,
    peer: "127.0.0.1:50000",
};
const AGENT_PROXIED: Site = Site {
    host: "agent.example.com",
    origin: "https://agent.example.com",
    forwarded: true,
    peer: "127.0.0.1:50000",
};
const DASH_LOOPBACK: Site = Site {
    host: "dash.oaiy.localhost:41000",
    origin: "http://dash.oaiy.localhost:41000",
    forwarded: false,
    peer: "127.0.0.1:50000",
};

fn dash_cookie_name(site: &Site) -> String {
    if site.forwarded {
        "__Host-oaiy_dash".into()
    } else {
        "oaiy_dash_41000".into()
    }
}

struct Reply {
    status: u16,
    headers: HeaderMap,
    text: String,
}

impl Reply {
    fn code(&self) -> Option<String> {
        serde_json::from_str::<Value>(&self.text).ok()?["error"]["code"]
            .as_str()
            .map(str::to_owned)
    }

    fn reason(&self) -> Option<String> {
        serde_json::from_str::<Value>(&self.text).ok()?["reason"]
            .as_str()
            .map(str::to_owned)
    }

    fn set_cookies(&self) -> Vec<String> {
        self.headers
            .get_all("set-cookie")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect()
    }
}

struct Send {
    method: Method,
    path: String,
    site: Site,
    headers: Vec<(String, String)>,
}

fn send(site: &Site, method: Method, path: &str) -> Send {
    Send {
        method,
        path: path.to_string(),
        site: site.clone(),
        headers: Vec::new(),
    }
}

impl Send {
    fn h(mut self, name: &str, value: &str) -> Send {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    fn drop_h(mut self, name: &str) -> Send {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self
    }

    fn cookie(self, name: &str, value: &str) -> Send {
        let existing = self
            .headers
            .iter()
            .find(|(n, _)| n == "cookie")
            .map(|(_, v)| format!("{v}; "))
            .unwrap_or_default();
        self.h("cookie", &format!("{existing}{name}={value}"))
    }

    /// What a browser at the site sends for a mutation that its own page made.
    fn page(self, csrf: &str) -> Send {
        let origin = self.site.origin;
        self.h("origin", origin)
            .h("sec-fetch-site", "same-origin")
            .h("x-oaiy-csrf", csrf)
    }
}

async fn go(env: &Env, s: Send) -> Reply {
    let mut req = Request::builder().method(s.method).uri(&s.path);
    req = req.header("host", s.site.host);
    if s.site.forwarded {
        req = req
            .header("x-forwarded-for", "203.0.113.9")
            .header("x-forwarded-proto", "https");
    }
    for (n, v) in &s.headers {
        req = req.header(n.as_str(), v.as_str());
    }
    let mut req = req.body(Body::empty()).unwrap();
    let peer: SocketAddr = s.site.peer.parse().unwrap();
    req.extensions_mut().insert(ConnectInfo(peer));
    let response = env.app.clone().oneshot(req).await.unwrap();
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

/// A dashboard session; its token, and the value a page derives from it.
fn session_of(env: &Env, app: App) -> (String, String) {
    let minted = if app == App::Dash {
        session::mint_session(&env.store, false, "203.0.113.9", "test").unwrap()
    } else {
        let mut spec = MintSpec::new(
            Kind::Ses,
            "app",
            super::presets::Preset::Agent.scopes(),
            24 * HOUR,
        );
        spec.app = Some(app);
        spec.preset = Some(super::presets::Preset::Agent);
        spec.idle_ms = Some(8 * HOUR);
        env.store.mint(spec).unwrap()
    };
    let secret = token::parse(&minted.token).unwrap().secret.to_string();
    let csrf = token::csrf_value(&secret).unwrap();
    (minted.token, csrf)
}

const LOGOUT: &str = "/api/auth/logout";

/// A well-formed token that no store has made.
fn bogus_pat() -> String {
    format!("oaiypat_0123456789abcdef_{}", "A".repeat(43))
}

// ==================================== T6: a cookie request ==========================================

async fn t6(site: Site, other_site: &'static str, other_port_origin: &'static str) {
    let e = env(site.forwarded, true);
    let (token, csrf) = session_of(&e, App::Dash);
    let name = dash_cookie_name(&site);
    let post = |e: &Env| {
        let _ = e;
        send(&site, Method::POST, LOGOUT).cookie(&name, &token)
    };

    // The right request passes.
    let ok = go(&e, post(&e).page(&csrf)).await;
    assert_eq!((ok.status, ok.text.as_str()), (200, "ok"), "{}", ok.text);

    // Another origin: a sibling on the same registrable domain (or another localhost name), another port, another
    // scheme, `null`, none at all. Each is `403 csrf`, and nothing was looked up.
    for bad in [
        other_site,
        other_port_origin,
        "null",
        "http://evil.example",
        "",
    ] {
        let r = go(&e, post(&e).page(&csrf).h("origin", bad)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("csrf")),
            "Origin {bad:?}"
        );
    }
    let r = go(&e, post(&e).page(&csrf).drop_h("origin")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("csrf")),
        "no Origin"
    );

    // No X-OAIY-CSRF, a wrong one, one of the right length that is wrong, and a valid session's value for
    // another session.
    let (_, other_csrf) = session_of(&e, App::Dash);
    for bad in [
        None,
        Some("x".repeat(43)),
        Some(csrf[..42].to_string()),
        Some(other_csrf),
    ] {
        let mut s = post(&e).page(&csrf).drop_h("x-oaiy-csrf");
        if let Some(b) = &bad {
            s = s.h("x-oaiy-csrf", b);
        }
        let r = go(&e, s).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("csrf")),
            "header {bad:?}"
        );
    }

    // Fetch Metadata: same-site and cross-site.
    for site_value in ["same-site", "cross-site", "none"] {
        let r = go(&e, post(&e).page(&csrf).h("sec-fetch-site", site_value)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("csrf")),
            "{site_value}"
        );
    }
    // A client that sends no Fetch Metadata (older browsers) is judged by the Origin and the header.
    let r = go(&e, post(&e).page(&csrf).drop_h("sec-fetch-site")).await;
    assert_eq!(r.status, 200);

    // The same cookie name twice: refused, and the browser is told to delete it.
    let dup = go(
        &e,
        post(&e).page(&csrf).cookie(
            &name,
            "oaiyses_ffffffffffffffff_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        ),
    )
    .await;
    assert_eq!(
        (dup.status, dup.code().as_deref()),
        (401, Some("token_invalid"))
    );
    let cleared = dup.set_cookies();
    assert_eq!(cleared.len(), 1, "{cleared:?}");
    assert!(
        cleared[0].starts_with(&format!("{name}=;")) && cleared[0].contains("Max-Age=0"),
        "{cleared:?}"
    );
}

#[tokio::test]
async fn t6_proxied_a_cookie_request_is_judged_by_the_exact_origin_the_fetch_metadata_and_the_derived_header(
) {
    t6(
        DASH_PROXIED,
        "https://flows.example.com",
        "https://dash.example.com:8443",
    )
    .await;
}

#[tokio::test]
async fn t6_loopback_a_cookie_request_is_judged_by_the_exact_origin_the_fetch_metadata_and_the_derived_header(
) {
    t6(
        DASH_LOOPBACK,
        "http://flows.oaiy.localhost:41000",
        "http://dash.oaiy.localhost:41001",
    )
    .await;
}

#[tokio::test]
async fn a_cookie_read_needs_only_fetch_metadata_and_a_page_reads_without_origin_or_header() {
    for site in [DASH_PROXIED, DASH_LOOPBACK] {
        let e = env(site.forwarded, true);
        let (token, _) = session_of(&e, App::Dash);
        let name = dash_cookie_name(&site);
        let read = |value: Option<&str>| {
            let mut s = send(&site, Method::GET, "/api/config").cookie(&name, &token);
            if let Some(v) = value {
                s = s.h("sec-fetch-site", v);
            }
            s
        };
        assert_eq!(go(&e, read(None)).await.status, 200);
        assert_eq!(go(&e, read(Some("same-origin"))).await.status, 200);
        assert_eq!(
            go(&e, read(Some("none"))).await.status,
            200,
            "a typed URL or a link"
        );
        for bad in ["same-site", "cross-site", "bogus"] {
            let r = go(&e, read(Some(bad))).await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (403, Some("csrf")),
                "{bad}"
            );
        }
    }
}

// ==================================== T7: another app's cookie ======================================

#[tokio::test]
async fn t7_the_dashboard_cookie_is_not_read_at_the_agent_host_and_the_agents_is_its_own() {
    let e = env(true, true);
    let (dash_token, dash_csrf) = session_of(&e, App::Dash);
    let (agent_token, agent_csrf) = session_of(&e, App::Agent);

    // The dashboard cookie, by its own name, presented at the Agent host: another name, so not read.
    let r = go(
        &e,
        send(&AGENT_PROXIED, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_dash", &dash_token)
            .page(&dash_csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );

    // The dashboard's token in the Agent's cookie name (what a sibling that toss a cookie would try): the session
    // is not the Agent host's.
    let r = go(
        &e,
        send(&AGENT_PROXIED, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_agent", &dash_token)
            .page(&dash_csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid"))
    );
    assert!(r.set_cookies()[0].starts_with("__Host-oaiy_agent=;"));

    // The Agent's own session at its own host is accepted; at the dashboard host it is nothing.
    let r = go(
        &e,
        send(&AGENT_PROXIED, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_agent", &agent_token)
            .page(&agent_csrf),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_dash", &agent_token)
            .page(&agent_csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid"))
    );
}

#[tokio::test]
async fn a_cookie_only_reaches_the_routes_its_apps_preset_holds() {
    // The Agent host's session is the `agent` preset: no `system.read`.
    let e = env(true, true);
    let (agent_token, _) = session_of(&e, App::Agent);
    let r = go(
        &e,
        send(&AGENT_PROXIED, Method::GET, "/api/config").cookie("__Host-oaiy_agent", &agent_token),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("insufficient_scope"))
    );
    let r = go(
        &e,
        send(&AGENT_PROXIED, Method::GET, "/api/agent/preferences")
            .cookie("__Host-oaiy_agent", &agent_token),
    )
    .await;
    assert_eq!(r.status, 200);
}

// ==================================== the rest of the rules of a cookie ==============================

#[tokio::test]
async fn a_bearer_wins_over_a_cookie_and_the_cookie_is_not_looked_at() {
    let e = env(true, true);
    let (cookie_token, csrf) = session_of(&e, App::Dash);
    let pat = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "tool",
            super::scopes::ScopeSet::of(&["system.read"]),
            24 * HOUR,
        ))
        .unwrap()
        .token;
    // A good bearer with a cookie of a session that would be refused for lack of the checks: the bearer decides.
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::GET, "/api/config")
            .cookie("__Host-oaiy_dash", &cookie_token)
            .h("authorization", &format!("Bearer {pat}")),
    )
    .await;
    assert_eq!(r.status, 200);
    // The bearer is a token: it is judged as one, with the cookie beside it doing nothing (no csrf needed).
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::POST, "/api/services")
            .cookie("__Host-oaiy_dash", &cookie_token)
            .page(&csrf)
            .h("authorization", &format!("Bearer {pat}")),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch")),
        "an unbound bearer with an Origin is a browser using a token: refused as ever"
    );
    // A wrong bearer is a wrong bearer, however good the cookie.
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::GET, "/api/config")
            .cookie("__Host-oaiy_dash", &cookie_token)
            .h("authorization", &format!("Bearer {}", bogus_pat())),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid"))
    );
    // A session token as a bearer is refused before any lookup.
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::GET, "/api/config")
            .h("authorization", &format!("Bearer {cookie_token}")),
    )
    .await;
    assert_eq!((r.status, r.code().as_deref()), (400, Some("bad_request")));
}

#[tokio::test]
async fn a_cookie_over_a_channel_that_is_not_secure_is_refused_and_never_read() {
    let e = env(true, true);
    let (token, csrf) = session_of(&e, App::Dash);
    // The proxy did not say https.
    let mut insecure = DASH_PROXIED;
    insecure.forwarded = false;
    insecure.peer = "127.0.0.1:50000";
    // No forwarded headers at all from a loopback peer to a public host name: not a channel that carries a cookie.
    let r = go(
        &e,
        send(&insecure, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_dash", &token)
            .page(&csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("secure_channel_required"))
    );
    // A loopback server's app name from a peer that is not this machine.
    let e = env(false, true);
    let (token, csrf) = session_of(&e, App::Dash);
    let mut far = DASH_LOOPBACK;
    far.peer = "192.0.2.10:50000";
    let r = go(
        &e,
        send(&far, Method::POST, LOGOUT)
            .cookie("oaiy_dash_41000", &token)
            .page(&csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("secure_channel_required"))
    );
}

#[tokio::test]
async fn an_authorization_header_that_is_not_a_bearer_is_refused_and_a_cookie_beside_it_is_not_tried(
) {
    let e = env(true, true);
    let (token, _) = session_of(&e, App::Dash);
    let get = |authorization: Option<&str>| {
        let s = send(&DASH_PROXIED, Method::GET, "/api/config").cookie("__Host-oaiy_dash", &token);
        match authorization {
            Some(v) => s.h("authorization", v),
            None => s,
        }
    };
    // Only no header at all leaves the cookie to be read.
    assert_eq!(go(&e, get(None)).await.status, 200);
    for value in [
        "Basic dXNlcjpwYXNz",
        "Bearer",
        "Token abc",
        "bearer lower",
        "",
    ] {
        let r = go(&e, get(Some(value))).await;
        assert_eq!(r.status, 400, "{value:?}: {}", r.text);
    }
}

#[tokio::test]
async fn hosts_that_do_not_serve_an_app_have_no_cookies() {
    let e = env(false, true);
    let (token, csrf) = session_of(&e, App::Dash);
    // `localhost` and an address are for bearers: the cookie, even by the right name for that port, is nothing.
    let localhost = Site {
        host: "localhost:41000",
        origin: "http://localhost:41000",
        forwarded: false,
        peer: "127.0.0.1:50000",
    };
    let r = go(
        &e,
        send(&localhost, Method::POST, LOGOUT)
            .cookie("oaiy_dash_41000", &token)
            .page(&csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );
}

#[tokio::test]
async fn a_value_that_is_not_a_session_token_is_token_invalid_and_cleared() {
    let e = env(true, true);
    let (token, csrf) = session_of(&e, App::Dash);
    let pat = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "tool",
            super::scopes::ScopeSet::of(&["system.read"]),
            24 * HOUR,
        ))
        .unwrap()
        .token;
    for bad in [
        pat.as_str(),
        "garbage",
        "",
        &token[..67],
        &format!("{token}A"),
        &token.replace("oaiyses_", "oaiydev_"),
    ] {
        let r = go(
            &e,
            send(&DASH_PROXIED, Method::GET, "/api/config").cookie("__Host-oaiy_dash", bad),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("token_invalid")),
            "{bad:?}"
        );
        assert!(r.set_cookies()[0].contains("Max-Age=0"), "{bad:?}");
    }
    // What is not a session is refused as one before the request is judged as a page's: a mutation with no Origin
    // and no header is `token_invalid`, not `csrf`.
    for bad in [pat.as_str(), "garbage"] {
        let r = go(
            &e,
            send(&DASH_PROXIED, Method::POST, LOGOUT).cookie("__Host-oaiy_dash", bad),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("token_invalid")),
            "{bad:?}"
        );
        assert!(r.set_cookies()[0].contains("Max-Age=0"), "{bad:?}");
    }
    let _ = csrf;
}

#[tokio::test]
async fn a_session_that_ended_says_why_and_the_browser_is_told_to_drop_the_cookie() {
    let e = env(true, true);
    let get = |token: String| {
        send(&DASH_PROXIED, Method::GET, "/api/config").cookie("__Host-oaiy_dash", &token)
    };

    // Logged out.
    let (t1, _) = session_of(&e, App::Dash);
    let id1 = token::parse(&t1).unwrap().id.to_string();
    assert_eq!(go(&e, get(t1.clone())).await.status, 200);
    e.store.revoke(&id1, "logged_out");
    let r = go(&e, get(t1)).await;
    assert_eq!(
        (r.status, r.code().as_deref(), r.reason().as_deref()),
        (401, Some("session_expired"), Some("logged_out"))
    );
    assert!(r.set_cookies()[0].starts_with("__Host-oaiy_dash=;"));

    // Idle: 8 hours unused.
    let (t2, _) = session_of(&e, App::Dash);
    e.clock.advance(8 * HOUR + 1);
    let r = go(&e, get(t2)).await;
    assert_eq!(
        (r.status, r.code().as_deref(), r.reason().as_deref()),
        (401, Some("session_expired"), Some("idle"))
    );

    // Absolute: 24 hours, however busy (kept alive by use until then).
    let (t3, _) = session_of(&e, App::Dash);
    for _ in 0..4 {
        e.clock.advance(6 * HOUR - 1000);
        assert_eq!(go(&e, get(t3.clone())).await.status, 200);
    }
    // 24 hours less four seconds have passed since it was made: one more second is not yet the end, five is.
    e.clock.advance(1000);
    assert_eq!(go(&e, get(t3.clone())).await.status, 200);
    e.clock.advance(4000);
    let r = go(&e, get(t3)).await;
    assert_eq!(
        (r.status, r.code().as_deref(), r.reason().as_deref()),
        (401, Some("session_expired"), Some("absolute"))
    );

    // A password change.
    let (t4, _) = session_of(&e, App::Dash);
    e.store
        .revoke_where("password_changed", &|r| r.kind == Kind::Ses);
    let r = go(&e, get(t4)).await;
    assert_eq!(r.reason().as_deref(), Some("password_changed"));
}

#[tokio::test]
async fn the_elevation_of_a_session_is_ten_minutes_and_a_dangerous_route_needs_it() {
    let e = env(true, true);
    let (token, csrf) = session_of(&e, App::Dash);
    let id = token::parse(&token).unwrap().id.to_string();
    let call = || {
        send(&DASH_PROXIED, Method::POST, "/api/services")
            .cookie("__Host-oaiy_dash", &token)
            .page(&csrf)
    };
    let r = go(&e, call()).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("elevation_required"))
    );
    let body: Value = serde_json::from_str(&r.text).unwrap();
    assert_eq!(body["elevate"], "/api/auth/elevate");
    // A non-dangerous route needs none.
    assert_eq!(
        go(
            &e,
            send(&DASH_PROXIED, Method::PUT, "/api/services/x/gpu")
                .cookie("__Host-oaiy_dash", &token)
                .page(&csrf)
        )
        .await
        .status,
        200
    );
    // Elevated for ten minutes from now.
    let until = session::elevate(&e.store, &id, e.clock.now_ms());
    assert_eq!(until, T0 + 600_000);
    assert_eq!(go(&e, call()).await.status, 200);
    e.clock.set(T0 + 600_000 - 1);
    assert_eq!(go(&e, call()).await.status, 200);
    e.clock.set(T0 + 600_000);
    let r = go(&e, call()).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("elevation_required")),
        "not extended by use"
    );
}

// ==================================== setup-only mode ================================================

#[tokio::test]
async fn with_no_owner_an_anonymous_call_is_setup_required_and_a_bearer_is_unaffected() {
    let e = env(true, true);
    e.facts.owner.store(false, Ordering::SeqCst);
    for (m, p) in [
        (Method::GET, "/api/config"),
        (Method::POST, "/api/services"),
        (Method::POST, LOGOUT),
        (Method::GET, "/api/bridge/pairing"),
        (Method::GET, "/api/no/such/route"),
    ] {
        let r = go(&e, send(&DASH_PROXIED, m.clone(), p)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("setup_required")),
            "{m} {p}"
        );
        assert_eq!(
            r.headers.get("www-authenticate").unwrap(),
            "Bearer realm=\"oaiy\""
        );
    }
    // The short list stays open (design 4.7.1): health and what says whether there is a login. The public routes of the
    // model that are not on it, the bridge's, are closed until there is an owner.
    assert_eq!(
        go(&e, send(&DASH_PROXIED, Method::GET, "/api/health"))
            .await
            .status,
        200
    );
    for (m, p) in [
        (Method::GET, "/api/bridge/capabilities"),
        (Method::GET, "/api/bridge/pairing/x"),
        (Method::POST, "/api/bridge/pairing"),
    ] {
        let r = go(&e, send(&DASH_PROXIED, m.clone(), p)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (401, Some("setup_required")),
            "{m} {p}"
        );
    }
    // A bearer credential: as ever. A wrong one is a wrong one, not "setup required".
    let pat = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "tool",
            super::scopes::ScopeSet::of(&["system.read"]),
            24 * HOUR,
        ))
        .unwrap()
        .token;
    assert_eq!(
        go(
            &e,
            send(&DASH_PROXIED, Method::GET, "/api/config")
                .h("authorization", &format!("Bearer {pat}"))
        )
        .await
        .status,
        200
    );
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::GET, "/api/config")
            .h("authorization", &format!("Bearer {}", bogus_pat())),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid"))
    );
    // With an owner it is the ordinary answer.
    e.facts.owner.store(true, Ordering::SeqCst);
    let r = go(&e, send(&DASH_PROXIED, Method::GET, "/api/config")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );
}

#[tokio::test]
async fn info_says_whether_a_login_exists_and_an_install_without_one_says_no() {
    for (install, owner, expect) in [
        (true, true, true),
        (true, false, false),
        (false, true, false),
    ] {
        let e = env(true, install);
        e.facts.owner.store(owner, Ordering::SeqCst);
        let r = go(&e, send(&DASH_PROXIED, Method::GET, "/api/auth/info")).await;
        let body: Value = serde_json::from_str(&r.text).unwrap();
        assert_eq!(
            body["loginConfigured"], expect,
            "install {install}, owner {owner}"
        );
        assert_eq!(body["setupCode"], "none");
        assert_eq!(e.guard.login_configured(), expect);
    }
}

// ==================================== an install with no web login ===================================

#[tokio::test]
async fn an_install_with_no_web_login_reads_no_cookie_and_answers_as_it_always_did() {
    let e = env(true, false);
    let (token, csrf) = session_of(&e, App::Dash);
    // A valid session token in the right cookie: nothing reads it.
    let r = go(
        &e,
        send(&DASH_PROXIED, Method::POST, LOGOUT)
            .cookie("__Host-oaiy_dash", &token)
            .page(&csrf),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );
    // And there is no setup-only mode: an anonymous caller is asked for a credential.
    e.facts.owner.store(false, Ordering::SeqCst);
    let r = go(&e, send(&DASH_PROXIED, Method::GET, "/api/config")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );
}
