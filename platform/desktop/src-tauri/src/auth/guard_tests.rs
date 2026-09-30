//! The new guard, end to end, in process: the pipeline of design 3.5 over a stub router built from the route
//! table (every row answers `200 ok` when the guard lets it through), with real credentials from a real store.
//!
//! The rules it holds are the design's: the guard matrix (T1), no Origin trust (T2, T3), the Host allow-list
//! (T5), bearer fuzzing (T11), origin binding (T15), CORS (T16), the failed-bearer throttle (T53), the modes
//! (T37), the derive rules (T12) and the exposure checks (T45). The differential test of `legacy` mode is in
//! `http/legacy_neutrality.rs`.

use std::net::SocketAddr;
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
use super::audit::{AuditLog, LogFile};
use super::clock::ManualClock;
use super::guard::{scoped_cors, scoped_guard, Guard, GuardConfig};
use super::mode::AccessMode;
use super::presets::{App, Preset, ALL_PRESETS};
use super::routes::{Class, DeskRole, Only, ROUTES};
use super::scopes::ScopeSet;
use super::store::{AuthStore, MintSpec};
use super::token::Kind;
use crate::secret_file::testing::TempDir;

const T0: u64 = 1_790_000_000_000;
const DAY: u64 = 86_400_000;
const STATIC_TOKEN: &str = "abcdefghijklmnopqrstuvwxyz0123456789ABCD";
const DESK_ORIGIN: &str = "tauri://localhost";

/// The routes of the table that are real handlers (`api.rs`): the stub leaves them to it.
const REAL: [&str; 3] = ["/api/auth/info", "/api/auth/whoami", "/api/auth/derive"];

/// A route that exists and has no row: added without one.
const UNCLASSIFIED: &str = "/api/zz/added-without-a-row";

struct Env {
    guard: Arc<Guard>,
    store: Arc<AuthStore>,
    clock: Arc<ManualClock>,
    app: Router,
    audit: Arc<AuditLog>,
    _dir: TempDir,
}

fn env_with(
    mode: AccessMode,
    vars: &[(&str, &str)],
    bind_all: bool,
    gui: bool,
    static_token: Option<&str>,
) -> Env {
    env_at_port(mode, vars, bind_all, gui, static_token, 17972)
}

/// [`env_with`] for a listener on `port`.
fn env_at_port(
    mode: AccessMode,
    vars: &[(&str, &str)],
    bind_all: bool,
    gui: bool,
    static_token: Option<&str>,
    port: u16,
) -> Env {
    let dir = TempDir::new("guard-tests");
    let clock = Arc::new(ManualClock::new(T0));
    let map: std::collections::BTreeMap<String, String> = vars
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let (config, warnings) =
        GuardConfig::from_env(&move |n| map.get(n).cloned(), bind_all, gui, port);
    assert!(warnings.is_empty(), "{warnings:?}");
    let store = Arc::new(AuthStore::memory(clock.clone()));
    let audit = Arc::new(AuditLog::open(&dir.0.join("auth"), clock.clone(), false));
    let guard = Arc::new(Guard::new(
        mode,
        config,
        store.clone(),
        static_token.map(str::to_owned),
        Some(audit.clone()),
        clock.clone(),
    ));
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
        .route(UNCLASSIFIED, any(|| async { "ok" }))
        .merge(api::router(guard.clone()))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_guard))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_cors));
    Env {
        guard,
        store,
        clock,
        app,
        audit,
        _dir: dir,
    }
}

/// A desktop-shaped install: loopback, the desktop's own windows, the static token configured.
fn env(mode: AccessMode) -> Env {
    env_with(mode, &[], false, true, Some(STATIC_TOKEN))
}

fn concrete(pattern: &str) -> String {
    pattern
        .split('/')
        .map(|s| {
            if s.starts_with(':') || s.starts_with('*') {
                "x"
            } else {
                s
            }
        })
        .collect::<Vec<_>>()
        .join("/")
}

#[derive(Clone)]
struct Send {
    method: Method,
    path: String,
    headers: Vec<(String, String)>,
    peer: Option<SocketAddr>,
    body: Option<String>,
}

fn send(method: Method, path: &str) -> Send {
    Send {
        method,
        path: path.to_string(),
        headers: vec![("host".into(), "localhost:17972".into())],
        peer: Some("127.0.0.1:50000".parse().unwrap()),
        body: None,
    }
}

impl Send {
    fn h(mut self, name: &str, value: &str) -> Send {
        self.headers.retain(|(n, _)| !n.eq_ignore_ascii_case(name));
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    fn add(mut self, name: &str, value: &str) -> Send {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    fn bearer(self, token: &str) -> Send {
        self.h("authorization", &format!("Bearer {token}"))
    }

    fn peer(mut self, addr: &str) -> Send {
        self.peer = Some(addr.parse().unwrap());
        self
    }

    fn json(mut self, body: &str) -> Send {
        self.body = Some(body.to_string());
        self.h("content-type", "application/json")
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
}

async fn go(env: &Env, s: Send) -> Reply {
    let mut req = Request::builder().method(s.method).uri(&s.path);
    for (n, v) in &s.headers {
        req = req.header(n.as_str(), v.as_str());
    }
    let mut req = req.body(Body::from(s.body.unwrap_or_default())).unwrap();
    if let Some(peer) = s.peer {
        req.extensions_mut().insert(ConnectInfo(peer));
    }
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

fn native_pat(env: &Env, scopes: ScopeSet, ttl: u64) -> String {
    env.store
        .mint(MintSpec::new(Kind::Pat, "test token", scopes, ttl))
        .unwrap()
        .token
}

fn browser_pat(env: &Env, origin: &str, scopes: &[&str]) -> String {
    let mut s = MintSpec::new(Kind::Pat, "test app", ScopeSet::of(scopes), 30 * DAY);
    s.origins = vec![origin.into()];
    env.store.mint(s).unwrap().token
}

fn desk(env: &Env, app: App, preset: Preset, origins: &[&str]) -> String {
    let mut s = MintSpec::new(Kind::Dsk, "webview", preset.scopes(), DAY);
    s.app = Some(app);
    s.preset = Some(preset);
    s.origins = origins.iter().map(|o| o.to_string()).collect();
    env.store.mint(s).unwrap().token
}

/// The methods a row is asked with.
fn methods(verb: super::routes::Verb) -> Vec<Method> {
    use super::routes::Verb;
    match verb {
        Verb::Get => vec![Method::GET, Method::HEAD],
        Verb::Post => vec![Method::POST],
        Verb::Put => vec![Method::PUT],
        Verb::Patch => vec![Method::PATCH],
        Verb::Delete => vec![Method::DELETE],
        Verb::Any => vec![
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
        ],
    }
}

/// Every `(method, path, class)` of the table that the stub router serves.
fn table() -> Vec<(Method, String, Class, &'static str)> {
    let mut out = Vec::new();
    for r in ROUTES.iter().filter(|r| !REAL.contains(&r.pattern)) {
        for m in methods(r.method) {
            out.push((m, concrete(r.pattern), r.class, r.pattern));
        }
    }
    out
}

// ==================================== T1: the guard matrix =======================================

#[tokio::test]
async fn t1_no_credential_gets_401_on_every_route_that_is_not_public_and_the_public_ones_answer() {
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        let mut refused = 0;
        for (m, p, class, pattern) in table() {
            let r = go(&e, send(m.clone(), &p)).await;
            if class == Class::Public {
                assert_eq!(r.status, 200, "{mode:?} {m} {pattern}");
            } else {
                assert_eq!(
                    (r.status, r.code().as_deref()),
                    (401, Some("auth_required")),
                    "{mode:?} {m} {pattern}: {}",
                    r.text
                );
                assert_eq!(
                    r.headers.get("www-authenticate").unwrap(),
                    "Bearer realm=\"oaiy\""
                );
                refused += 1;
            }
        }
        assert!(refused > 240, "{refused} routes refused");
    }
}

#[tokio::test]
async fn t1_a_malformed_an_unknown_a_wrong_an_expired_and_a_revoked_credential_are_told_apart_exactly(
) {
    let e = env(AccessMode::Scoped);
    let good = native_pat(&e, ScopeSet::of(&["system.read"]), DAY);
    let parsed = super::token::parse(&good).unwrap();
    let wrong_secret = format!("oaiypat_{}_{}", parsed.id, "A".repeat(43));
    let unknown_id = format!("oaiypat_{}_{}", "f".repeat(16), parsed.secret);
    let expired = native_pat(&e, ScopeSet::of(&["system.read"]), 1000);
    let revoked = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "r",
            ScopeSet::of(&["system.read"]),
            DAY,
        ))
        .unwrap();
    e.store.revoke(&revoked.id, "revoked");
    e.clock.advance(2000);
    let mut checked = 0;
    for (m, p, class, pattern) in table()
        .into_iter()
        .filter(|(_, _, c, _)| *c != Class::Public)
    {
        let expect = |token: &str, status: u16, code: &str| {
            let (e, m, p, token, code) = (
                &e,
                m.clone(),
                p.clone(),
                token.to_string(),
                code.to_string(),
            );
            async move {
                let r = go(e, send(m.clone(), &p).bearer(&token)).await;
                assert_eq!(
                    (r.status, r.code().unwrap_or_default()),
                    (status, code.clone()),
                    "{m} {pattern}: {}",
                    r.text
                );
                r
            }
        };
        // Right id with the wrong secret and an id nobody has are the same answer, byte for byte.
        let a = expect(&wrong_secret, 401, "token_invalid").await;
        let b = expect(&unknown_id, 401, "token_invalid").await;
        assert_eq!(
            a.text, b.text,
            "an unknown id and a wrong secret are indistinguishable"
        );
        let r = expect(&expired, 401, "token_expired").await;
        assert_eq!(
            r.headers.get("www-authenticate").unwrap(),
            "Bearer realm=\"oaiy\", error=\"invalid_token\""
        );
        expect(&revoked.token, 401, "token_revoked").await;
        assert_eq!(
            go(&e, send(m.clone(), &p).bearer("!!bad")).await.status,
            400
        );
        let _ = class;
        checked += 1;
        if checked > 60 {
            break;
        }
    }
}

/// The scopes a preset holds, as the route rows see them.
fn expected_status(
    class: &Class,
    preset: Preset,
    kind_desk: Option<App>,
) -> (u16, Option<&'static str>) {
    match class {
        Class::Public | Class::AnyCredential => (200, None),
        Class::Console | Class::Session { .. } => (403, Some("insufficient_scope")),
        Class::Desk { roles } => match kind_desk {
            Some(app)
                if roles.contains(&match app {
                    App::Dash => DeskRole::Dashboard,
                    App::Agent => DeskRole::Agent,
                    App::Flows => DeskRole::Flows,
                }) =>
            {
                (200, None)
            }
            _ => (403, Some("insufficient_scope")),
        },
        Class::Scope(s) => {
            if preset.scopes().contains(s) {
                (200, None)
            } else {
                (403, Some("insufficient_scope"))
            }
        }
        Class::Unclassified => (403, Some("unclassified_route")),
    }
}

#[tokio::test]
async fn t1_exactly_the_routes_of_a_preset_succeed_for_a_credential_with_that_preset() {
    let e = env(AccessMode::Scoped);
    let mut total = 0usize;
    for preset in ALL_PRESETS {
        // Every preset but `owner` is a paired token (a native one: the dangerous scopes of `cli-admin` for a
        // day); `owner` is the dashboard's desk credential, from loopback with its own origin. The ceremony
        // token is browser-bound and used once.
        let (token, origin, desk_app) = match preset {
            Preset::Owner => (
                desk(&e, App::Dash, Preset::Owner, &[DESK_ORIGIN]),
                Some(DESK_ORIGIN),
                Some(App::Dash),
            ),
            Preset::Ceremony => {
                let mut s = MintSpec::new(Kind::Pat, "ceremony", preset.scopes(), 5 * 60_000);
                s.origins = vec!["https://app.example".into()];
                s.max_uses = Some(1);
                (
                    e.store.mint(s).unwrap().token,
                    Some("https://app.example"),
                    None,
                )
            }
            _ => (native_pat(&e, preset.scopes(), DAY), None, None),
        };
        for (m, p, class, pattern) in table() {
            let mut s = send(m.clone(), &p).bearer(&token);
            if let Some(o) = origin {
                s = s.h("origin", o);
            }
            let r = go(&e, s).await;
            let (status, code) = expected_status(&class, preset, desk_app);
            assert_eq!(
                (r.status, r.code().as_deref()),
                (status, code),
                "{} {m} {pattern}: {}",
                preset.name(),
                r.text
            );
            if let Class::Scope(scope) = class {
                if status == 403 {
                    assert_eq!(
                        r.json()["required"],
                        scope,
                        "{} {m} {pattern}",
                        preset.name()
                    );
                    assert!(r
                        .headers
                        .get("www-authenticate")
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .contains(&format!("scope=\"{scope}\"")));
                }
            }
            total += 1;
        }
    }
    assert!(total > 3000, "{total} requests");
}

#[tokio::test]
async fn t1_the_environment_token_is_the_cli_preset_on_every_install_and_never_more() {
    let e = env(AccessMode::Scoped);
    for (m, p, class, pattern) in table() {
        let r = go(&e, send(m.clone(), &p).bearer(STATIC_TOKEN)).await;
        let (status, code) = expected_status(&class, Preset::Cli, None);
        assert_eq!(
            (r.status, r.code().as_deref()),
            (status, code),
            "{m} {pattern}"
        );
    }
    // In particular it never reaches a dangerous route or anything under auth.
    for (m, p) in [
        (Method::POST, "/api/services"),
        (Method::POST, "/api/plugins/install"),
        (Method::GET, "/api/bridge/pairing"),
        (Method::POST, "/api/bridge/pairing/x/approve"),
        (Method::PUT, "/api/secrets/hf-token"),
    ] {
        assert_eq!(go(&e, send(m, p).bearer(STATIC_TOKEN)).await.status, 403);
    }
    // A wrong static-looking token is a token that is not valid.
    assert_eq!(
        go(
            &e,
            send(Method::GET, "/api/config").bearer("abcdefghijklmnopqrstuvwxyz0123456789ABCE")
        )
        .await
        .code()
        .as_deref(),
        Some("token_invalid")
    );
}

#[tokio::test]
async fn t1_a_route_that_exists_without_a_row_is_refused_for_everyone_and_a_route_that_is_not_there_is_401_or_404(
) {
    let e = env(AccessMode::Scoped);
    let owner = desk(&e, App::Dash, Preset::Owner, &[DESK_ORIGIN]);
    for token in [None, Some(STATIC_TOKEN.to_string()), Some(owner)] {
        let mut s = send(Method::GET, UNCLASSIFIED).h("origin", DESK_ORIGIN);
        if let Some(t) = &token {
            s = s.bearer(t);
        }
        let r = go(&e, s.clone()).await;
        // The desk credential needs its origin, the environment token refuses one: judge the row, not the binding.
        if token.as_deref() == Some(STATIC_TOKEN) {
            assert_eq!(
                go(&e, send(Method::GET, UNCLASSIFIED).bearer(STATIC_TOKEN))
                    .await
                    .code()
                    .as_deref(),
                Some("unclassified_route")
            );
        } else {
            assert_eq!(
                (r.status, r.code().as_deref()),
                (403, Some("unclassified_route")),
                "{:?}",
                token.is_some()
            );
        }
    }
    // A path with no route at all: 401 for a stranger, 404 for someone who is who they say they are.
    let r = go(&e, send(Method::GET, "/api/no/such/route")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("auth_required"))
    );
    let r = go(
        &e,
        send(Method::GET, "/api/no/such/route").bearer(STATIC_TOKEN),
    )
    .await;
    assert_eq!((r.status, r.code().as_deref()), (404, Some("not_found")));
    let r = go(&e, send(Method::POST, "/api/no/such/route").bearer("!!")).await;
    assert_eq!(r.status, 400);
    let r = go(
        &e,
        send(Method::GET, "/api/no/such/route")
            .bearer("oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid")),
        "a route that is not there does not tell a stranger it is not there"
    );
}

#[tokio::test]
async fn a_path_outside_the_api_that_no_route_answers_is_not_under_the_guard() {
    let e = env(AccessMode::Scoped);
    assert_eq!(go(&e, send(Method::GET, "/")).await.status, 404);
    assert_eq!(go(&e, send(Method::GET, "/favicon.ico")).await.status, 404);
    // And it gets no `no-store`: only the API does.
    assert!(go(&e, send(Method::GET, "/"))
        .await
        .headers
        .get("cache-control")
        .is_none());
}

#[tokio::test]
async fn every_answer_of_the_api_is_never_cached_and_never_sniffed() {
    let e = env(AccessMode::Scoped);
    for (m, p) in [
        (Method::GET, "/api/health"),
        (Method::GET, "/api/config"),
        (Method::GET, "/api/no/such"),
    ] {
        let r = go(&e, send(m, p)).await;
        assert_eq!(r.headers.get("cache-control").unwrap(), "no-store", "{p}");
        assert_eq!(
            r.headers.get("x-content-type-options").unwrap(),
            "nosniff",
            "{p}"
        );
    }
    let r = go(&e, send(Method::GET, "/api/config").bearer(STATIC_TOKEN)).await;
    assert_eq!(r.status, 200);
    assert_eq!(r.headers.get("cache-control").unwrap(), "no-store");
}

#[tokio::test]
async fn a_method_the_api_does_not_serve_is_405() {
    let e = env(AccessMode::Scoped);
    for m in ["TRACE", "CONNECT", "BREW", "PROPFIND"] {
        let r = go(
            &e,
            send(Method::from_bytes(m.as_bytes()).unwrap(), "/api/config").bearer(STATIC_TOKEN),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (405, Some("method_not_allowed")),
            "{m}"
        );
    }
}

// ============================= T2, T3: Origin is never a credential ==============================

#[tokio::test]
async fn t2_no_former_trusted_origin_is_a_credential_for_a_read_or_a_write() {
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        for origin in [
            "https://oaiy.com",
            "https://x.oaiy.com",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://tauri.localhost",
            "tauri://localhost",
            "http://oaiy.localhost",
            "oaiy://localhost",
            "http://oaiyflows.localhost",
            "http://formlogic.local",
            "null",
            "http://evil.example",
        ] {
            for (m, p, class, pattern) in table() {
                if class == Class::Public {
                    continue;
                }
                let r = go(&e, send(m.clone(), &p).h("origin", origin)).await;
                assert_eq!(
                    (r.status, r.code().as_deref()),
                    (401, Some("auth_required")),
                    "{mode:?} origin {origin} {m} {pattern}"
                );
            }
        }
    }
}

#[tokio::test]
async fn t3_a_mutation_with_a_forged_origin_and_no_credential_is_401_where_the_old_guard_passed_it()
{
    let e = env(AccessMode::Scoped);
    // The old guard, in a GUI, let a non-privileged mutation with an allowed Origin through with no credential.
    for (m, p) in [
        (Method::POST, "/api/services/x/start"),
        (Method::POST, "/api/bridge/leases/answer-calls"),
        (Method::POST, "/api/services/ensure-by-port"),
        (Method::POST, "/api/models/downloads/x/pause"),
    ] {
        for origin in [
            "http://localhost:3000",
            "https://x.oaiy.com",
            "tauri://localhost",
        ] {
            let r = go(&e, send(m.clone(), p).h("origin", origin).json("{}")).await;
            assert_eq!(r.status, 401, "{m} {p} {origin}");
        }
    }
}

// =============================================== T5: Host ========================================

#[tokio::test]
async fn t5_the_host_allow_list_stops_rebinding_and_accepts_loopback_names_on_any_port() {
    let e = env(AccessMode::Scoped);
    let with_host = |host: &str| {
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .h("host", host)
    };
    // A rebound page carries its own name: 421, from a loopback peer.
    for host in [
        "evil.example:17972",
        "evil.example",
        "127.0.0.1.evil.example:17972",
        "localhost.evil.example",
    ] {
        let r = go(&e, with_host(host)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (421, Some("misdirected_host")),
            "{host}"
        );
    }
    // Loopback names are accepted on any port: an SSH tunnel, a Docker port mapping.
    for host in [
        "localhost:17972",
        "localhost:9999",
        "127.0.0.1:2222",
        "127.0.0.1",
        "[::1]:17972",
        "LOCALHOST:8080",
    ] {
        assert_eq!(go(&e, with_host(host)).await.status, 200, "{host}");
    }
    // No Host at all, and junk.
    let mut no_host = with_host("x");
    no_host.headers.retain(|(n, _)| n != "host");
    assert_eq!(go(&e, no_host).await.status, 421);
    assert_eq!(go(&e, with_host("a b")).await.status, 421);
    // Health is exempt: a probe's Host is a pod address.
    let probe = send(Method::GET, "/api/health")
        .h("host", "10.42.0.7:17972")
        .peer("10.42.0.1:40000");
    assert_eq!(go(&e, probe.clone()).await.status, 200);
    assert_eq!(
        go(
            &e,
            Send {
                method: Method::HEAD,
                ..probe
            }
        )
        .await
        .status,
        200
    );
    // Nothing else is: the same address on any other route is misdirected.
    let r = go(
        &e,
        send(Method::GET, "/api/bridge/capabilities")
            .h("host", "10.42.0.7:17972")
            .peer("10.42.0.1:40000"),
    )
    .await;
    assert_eq!(r.status, 421);
}

// ================================= T11: bearer fuzzing, no lookup ================================

#[tokio::test]
async fn t11_a_hostile_authorization_header_is_a_400_before_any_lookup_and_is_never_counted_as_a_guess(
) {
    // A proxied install so that the throttle applies to this peer, and any lookup that fails would count.
    let e = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        None,
    );
    let secure = |m: Method, p: &str| {
        send(m, p)
            .peer("203.0.113.9:40000")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https")
    };
    let good = "oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    let cases: Vec<(&str, Vec<(String, String)>)> = vec![
        (
            "two headers",
            vec![
                ("authorization".into(), format!("Bearer {good}")),
                ("authorization".into(), "Bearer x".into()),
            ],
        ),
        (
            "129 bytes",
            vec![(
                "authorization".into(),
                format!("Bearer {}", "a".repeat(129)),
            )],
        ),
        (
            "a space inside",
            vec![("authorization".into(), "Bearer abc def".into())],
        ),
        (
            "wrong-case scheme",
            vec![("authorization".into(), format!("bearer {good}"))],
        ),
        (
            "Basic",
            vec![("authorization".into(), "Basic dXNlcjpwYXNz".into())],
        ),
        (
            "a session token as a bearer",
            vec![(
                "authorization".into(),
                "Bearer oaiyses_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
                    .into(),
            )],
        ),
        (
            "a device token as a bearer",
            vec![(
                "authorization".into(),
                "Bearer oaiydev_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
                    .into(),
            )],
        ),
        ("no token", vec![("authorization".into(), "Bearer ".into())]),
    ];
    for (name, headers) in cases {
        for _ in 0..30 {
            let mut s = secure(Method::GET, "/api/config");
            for (n, v) in &headers {
                s = s.add(n, v);
            }
            let r = go(&e, s).await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (400, Some("bad_request")),
                "{name}: {}",
                r.text
            );
        }
    }
    // 240 refused headers later the address is not blocked: a 400 is no guess.
    let r = go(
        &e,
        secure(Method::GET, "/api/config")
            .bearer("oaiypat_ffffffffffffffff_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (401, Some("token_invalid"))
    );
    // A control character cannot even be put in a header by a client; the parser refuses it anyway.
    assert!(super::token::bearer_from_headers(&[b"Bearer abc\x01def"]).is_err());
}

// ====================================== T15: origin binding ======================================

#[tokio::test]
async fn t15_a_bound_credential_is_used_only_from_its_origin_and_an_unbound_one_only_without_an_origin(
) {
    let e = env_with(AccessMode::Scoped, &[], false, true, None);
    let app_token = browser_pat(&e, "https://formlogic.example", &["services.read"]);
    let native = native_pat(&e, ScopeSet::of(&["services.read"]), DAY);
    let get = || send(Method::GET, "/api/services");
    // Bound, right origin: yes. Another origin, no origin, `null`: no.
    assert_eq!(
        go(
            &e,
            get()
                .bearer(&app_token)
                .h("origin", "https://formlogic.example")
        )
        .await
        .status,
        200
    );
    for origin in [
        "https://evil.example",
        "https://formlogic.example.evil.example",
        "http://formlogic.example",
        "https://FORMLOGIC.example",
        "null",
        "https://formlogic.example/",
    ] {
        let r = go(&e, get().bearer(&app_token).h("origin", origin)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("origin_mismatch")),
            "{origin}"
        );
    }
    let r = go(&e, get().bearer(&app_token)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch")),
        "a bound token with no origin"
    );
    // The same-origin GET carries no Origin and says so with Fetch Metadata: allowed only at the bound origin's own Host.
    let e2 = env_with(
        AccessMode::Scoped,
        &[("OAIY_ALLOWED_HOSTS", "formlogic.example:17972")],
        false,
        true,
        None,
    );
    let bound_to_own_host = browser_pat(&e2, "http://formlogic.example:17972", &["services.read"]);
    let ok = go(
        &e2,
        get()
            .bearer(&bound_to_own_host)
            .h("host", "formlogic.example:17972")
            .h("sec-fetch-site", "same-origin"),
    )
    .await;
    assert_eq!(ok.status, 200, "{}", ok.text);
    let cross = go(
        &e2,
        get()
            .bearer(&bound_to_own_host)
            .h("host", "formlogic.example:17972")
            .h("sec-fetch-site", "cross-site"),
    )
    .await;
    assert_eq!(cross.status, 403);
    let none = go(
        &e2,
        get()
            .bearer(&bound_to_own_host)
            .h("host", "formlogic.example:17972"),
    )
    .await;
    assert_eq!(none.status, 403);
    // Unbound: an Origin header means a browser, and a browser may not use it.
    assert_eq!(go(&e, get().bearer(&native)).await.status, 200);
    for origin in [
        "https://formlogic.example",
        "http://localhost:3000",
        "null",
        "tauri://localhost",
    ] {
        let r = go(&e, get().bearer(&native).h("origin", origin)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("origin_mismatch")),
            "{origin}"
        );
    }
    // The environment token is unbound too.
    let e3 = env(AccessMode::Scoped);
    let r = go(
        &e3,
        send(Method::GET, "/api/services")
            .bearer(STATIC_TOKEN)
            .h("origin", "https://formlogic.example"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch"))
    );
}

#[tokio::test]
async fn t15_a_desk_credential_works_only_from_a_direct_loopback_peer_with_its_own_origin() {
    let e = env(AccessMode::Scoped);
    let dash = desk(
        &e,
        App::Dash,
        Preset::Owner,
        &[
            "tauri://localhost",
            "http://tauri.localhost",
            "https://tauri.localhost",
        ],
    );
    let get = || send(Method::GET, "/api/services").bearer(&dash);
    for origin in [
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    ] {
        assert_eq!(
            go(&e, get().h("origin", origin)).await.status,
            200,
            "{origin}"
        );
    }
    // Another origin: the credential is not the origin's to use.
    assert_eq!(
        go(&e, get().h("origin", "http://oaiy.localhost"))
            .await
            .code()
            .as_deref(),
        Some("origin_mismatch")
    );
    assert_eq!(
        go(&e, get()).await.code().as_deref(),
        Some("origin_mismatch"),
        "no Origin: a webview always sends one"
    );
    // From a peer that is not loopback, or in process: worthless (copied off the machine).
    let r = go(
        &e,
        get()
            .h("origin", "tauri://localhost")
            .peer("192.168.1.9:40000")
            .h("host", "localhost:17972"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch"))
    );
    let mut in_process = get().h("origin", "tauri://localhost");
    in_process.peer = None;
    in_process.headers.retain(|(n, _)| n != "host");
    assert_eq!(
        go(&e, in_process).await.code().as_deref(),
        Some("origin_mismatch")
    );
}

#[tokio::test]
async fn t15_a_desk_or_run_credential_that_came_through_a_proxy_is_refused_however_the_proxy_is_set_up(
) {
    // A proxied install trusting the loopback proxy: the peer is loopback and a forwarded header is present.
    let e = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        Some(STATIC_TOKEN),
    );
    let dash = desk(&e, App::Dash, Preset::Owner, &["https://dash.example.com"]);
    let via_proxy = |token: &str| {
        send(Method::GET, "/api/services")
            .bearer(token)
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https")
            .h("origin", "https://dash.example.com")
    };
    let r = go(&e, via_proxy(&dash)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch")),
        "{}",
        r.text
    );
    // A derived credential inherits both rules from its parent.
    let parent = e.store.authenticate(&dash, None).unwrap();
    let child = e
        .store
        .derive(
            &parent,
            super::store::DeriveRequest {
                scopes: ScopeSet::of(&["services.read"]),
                ttl_ms: None,
                label: "c".into(),
            },
        )
        .unwrap();
    let r = go(&e, via_proxy(&child.token)).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch"))
    );
    // The console and the per-run credential too.
    let run = e
        .store
        .mint(MintSpec::new(
            Kind::Run,
            "run",
            ScopeSet::of(&["services.read"]),
            60_000,
        ))
        .unwrap();
    assert_eq!(
        go(&e, via_proxy(&run.token)).await.code().as_deref(),
        Some("origin_mismatch")
    );
    let con = e
        .store
        .mint(MintSpec::new(Kind::Con, "console", ScopeSet::all(), DAY))
        .unwrap();
    assert_eq!(
        go(&e, via_proxy(&con.token)).await.code().as_deref(),
        Some("origin_mismatch")
    );
    // Direct from loopback (the CLI on the server, straight to the port): the run credential works, unbound.
    let direct = go(&e, send(Method::GET, "/api/services").bearer(&run.token)).await;
    assert_eq!(direct.status, 200, "{}", direct.text);
    // And from a public address it does not, whatever it says.
    let r = go(
        &e,
        send(Method::GET, "/api/services")
            .bearer(&run.token)
            .peer("203.0.113.50:5000")
            .h("host", "dash.example.com"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("origin_mismatch"))
    );
}

// ================================= exposure: proxies, LAN, forwarded ==============================

#[tokio::test]
async fn t45_a_forwarded_header_on_a_local_install_is_421_and_a_proxied_install_believes_only_its_proxy(
) {
    let e = env(AccessMode::Scoped);
    for name in [
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "forwarded",
        "via",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
    ] {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .add(name, "203.0.113.9"),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (421, Some("proxy_detected")),
            "{name}"
        );
    }
    // Health is not exempt from this one (only from the Host check).
    let r = go(
        &e,
        send(Method::GET, "/api/health").add("x-forwarded-for", "203.0.113.9"),
    )
    .await;
    assert_eq!(r.status, 421);
    // A proxied install: the same header from its proxy is fine, and `info` says what the server saw.
    let p = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        None,
    );
    let r = go(
        &p,
        send(Method::GET, "/api/auth/info")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https"),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    let seen = &r.json()["seen"];
    assert_eq!(
        (
            seen["clientIp"].as_str(),
            seen["proto"].as_str(),
            seen["host"].as_str(),
            seen["viaTrustedProxy"].as_bool()
        ),
        (
            Some("203.0.113.9"),
            Some("https"),
            Some("dash.example.com"),
            Some(true)
        )
    );
    assert_eq!(r.json()["secureChannel"], true);
    // The same header from a peer that is not the proxy is ignored, not believed.
    let r = go(
        &p,
        send(Method::GET, "/api/auth/info")
            .h("host", "dash.example.com")
            .peer("198.51.100.7:4000")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https"),
    )
    .await;
    assert_eq!(r.json()["seen"]["clientIp"], "198.51.100.7");
    assert_eq!(
        r.json()["secureChannel"],
        false,
        "an untrusted peer's word does not make a channel secure"
    );
    // A proxy that says http for the public host has a broken configuration.
    let r = go(
        &p,
        send(Method::GET, "/api/auth/info")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "http"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (400, Some("proxy_misconfigured"))
    );
    // nginx's default `Host: 127.0.0.1:17972` through the proxy is misdirected.
    let r = go(
        &p,
        send(Method::GET, "/api/config")
            .bearer("x")
            .h("host", "127.0.0.1:17972")
            .add("x-forwarded-for", "203.0.113.9"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (421, Some("misdirected_host"))
    );
}

#[tokio::test]
async fn t45_a_bearer_from_a_public_address_on_a_lan_listener_is_refused_unless_the_operator_says_otherwise(
) {
    let lan = env_with(AccessMode::Scoped, &[], true, false, Some(STATIC_TOKEN));
    let from = |peer: &str, host: &str| {
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer(peer)
            .h("host", host)
    };
    // Private and loopback peers are fine; a public one is not.
    for peer in [
        "192.168.1.9:4000",
        "10.0.0.7:4000",
        "100.64.1.1:4000",
        "[fd12::5]:4000",
        "127.0.0.1:4000",
    ] {
        assert_eq!(
            go(&lan, from(peer, "192.168.1.5:17972")).await.status,
            200,
            "{peer}"
        );
    }
    let r = go(&lan, from("203.0.113.50:4000", "192.168.1.5:17972")).await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("plaintext_from_public_address"))
    );
    // No Authorization from a public address: that is just an anonymous request.
    let r = go(
        &lan,
        send(Method::GET, "/api/config")
            .peer("203.0.113.50:4000")
            .h("host", "192.168.1.5:17972"),
    )
    .await;
    assert_eq!(r.code().as_deref(), Some("auth_required"));
    // The wrong port, or a name, is not this listener.
    assert_eq!(
        go(&lan, from("192.168.1.9:4000", "192.168.1.5:9999"))
            .await
            .status,
        421
    );
    assert_eq!(
        go(&lan, from("192.168.1.9:4000", "nas.local:17972"))
            .await
            .status,
        421
    );
    let allowed = env_with(
        AccessMode::Scoped,
        &[("OAIY_ALLOW_PUBLIC_PLAINTEXT", "1")],
        true,
        false,
        Some(STATIC_TOKEN),
    );
    assert_eq!(
        go(&allowed, from("203.0.113.50:4000", "192.168.1.5:17972"))
            .await
            .status,
        200
    );
}

// ================================ T53: the failed-bearer throttle ================================

#[tokio::test]
async fn t53_twenty_guesses_block_an_address_and_nothing_else_is_affected() {
    let e = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        Some(STATIC_TOKEN),
    );
    let from = |ip: &str| {
        send(Method::GET, "/api/config")
            .peer("127.0.0.1:5000")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", ip)
            .add("x-forwarded-proto", "https")
    };
    let guess = |n: u32| format!("oaiypat_{n:016x}_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8");
    for n in 0..20 {
        let r = go(&e, from("203.0.113.9").bearer(&guess(n))).await;
        assert_eq!(r.code().as_deref(), Some("token_invalid"), "guess {n}");
    }
    // The 21st is refused before any lookup: 429, with Retry-After.
    let r = go(&e, from("203.0.113.9").bearer(&guess(99))).await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    assert_eq!(r.headers.get("retry-after").unwrap(), "900");
    assert_eq!(r.json()["retryAfterSeconds"], 900);
    // Even the right credential is refused from a blocked address (the block is before the lookup).
    assert_eq!(
        go(&e, from("203.0.113.9").bearer(STATIC_TOKEN))
            .await
            .status,
        429
    );
    // A request with no Authorization from that address is not throttled: a public route stays reachable.
    assert_eq!(
        go(&e, from("203.0.113.9")).await.code().as_deref(),
        Some("auth_required")
    );
    assert_eq!(
        go(
            &e,
            Send {
                path: "/api/health".into(),
                ..from("203.0.113.9")
            }
        )
        .await
        .status,
        200,
        "health is still answered from the blocked address"
    );
    // Another address, and another /64 of the same v6 range, are unaffected.
    assert_eq!(
        go(&e, from("203.0.113.10").bearer(STATIC_TOKEN))
            .await
            .status,
        200
    );
    // The block ends after 15 minutes.
    e.clock.advance(15 * 60_000);
    assert_eq!(
        go(&e, from("203.0.113.9").bearer(STATIC_TOKEN))
            .await
            .status,
        200
    );
    // And is counted in the noise log, not the audit log.
    e.audit.flush_noise();
    let noise = e.audit.read(LogFile::Noise, 50, None, None);
    assert!(
        noise
            .iter()
            .any(|l| l["event"] == "bearer.failed" && l["ip"] == "203.0.113.9" && l["count"] == 20),
        "{noise:?}"
    );
    assert!(noise.iter().any(|l| l["event"] == "bearer.blocked"));
    assert!(
        e.audit.read(LogFile::Audit, 50, None, None).is_empty(),
        "a guess is anonymous traffic: it never reaches the audit file"
    );
}

#[tokio::test]
async fn t53_an_expired_or_revoked_credential_is_not_a_guess_and_a_local_loopback_peer_is_never_blocked(
) {
    let e = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        None,
    );
    let from_public = |token: &str| {
        send(Method::GET, "/api/config")
            .peer("127.0.0.1:5000")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https")
            .bearer(token)
    };
    let short = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "short",
            ScopeSet::of(&["system.read"]),
            1000,
        ))
        .unwrap();
    let revoked = e
        .store
        .mint(MintSpec::new(
            Kind::Pat,
            "revoked",
            ScopeSet::of(&["system.read"]),
            DAY,
        ))
        .unwrap();
    e.store.revoke(&revoked.id, "revoked");
    e.clock.advance(5000);
    for _ in 0..40 {
        assert_eq!(
            go(&e, from_public(&short.token)).await.code().as_deref(),
            Some("token_expired")
        );
        assert_eq!(
            go(&e, from_public(&revoked.token)).await.code().as_deref(),
            Some("token_revoked")
        );
    }
    assert_ne!(
        go(&e, from_public(&short.token)).await.status,
        429,
        "a stale paired app is not an attacker"
    );
    // A local install: the desktop's own windows and a local script cannot be blocked by it.
    let local = env(AccessMode::Scoped);
    for n in 0..60u32 {
        let bad = format!("oaiypat_{n:016x}_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8");
        assert_eq!(
            go(&local, send(Method::GET, "/api/config").bearer(&bad))
                .await
                .code()
                .as_deref(),
            Some("token_invalid")
        );
    }
    assert_eq!(
        go(
            &local,
            send(Method::GET, "/api/config").bearer(STATIC_TOKEN)
        )
        .await
        .status,
        200
    );
    // But a LAN listener's private peer is throttled: only a local install exempts loopback.
    let lan = env_with(AccessMode::Scoped, &[], true, false, Some(STATIC_TOKEN));
    let lan_from = |token: &str| {
        send(Method::GET, "/api/config")
            .peer("192.168.1.9:4000")
            .h("host", "192.168.1.5:17972")
            .bearer(token)
    };
    for n in 0..20u32 {
        go(
            &lan,
            lan_from(&format!(
                "oaiypat_{n:016x}_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
            )),
        )
        .await;
    }
    assert_eq!(go(&lan, lan_from(STATIC_TOKEN)).await.status, 429);
}

// ======================================= T37: the modes ==========================================

#[tokio::test]
async fn t37_every_route_the_model_adds_refuses_a_stranger_in_every_mode() {
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        for r in ROUTES.iter().filter(|r| r.since == 2) {
            for m in methods(r.method) {
                let path = concrete(r.pattern);
                let reply = go(&e, send(m.clone(), &path)).await;
                if r.class == Class::Public {
                    // The public ones that exist answer; the ones another step builds are the stub's `ok`.
                    assert!(
                        reply.status == 200 || reply.status == 400 || reply.status == 415,
                        "{mode:?} {m} {path}: {}",
                        reply.status
                    );
                } else {
                    assert!(
                        matches!(reply.status, 401 | 403 | 404),
                        "{mode:?} {m} {path} answered {}",
                        reply.status
                    );
                    assert_ne!(reply.status, 200);
                }
            }
        }
    }
}

#[tokio::test]
async fn t37_shadow_lets_only_a_non_dangerous_scope_mismatch_through_and_writes_it_to_the_noise_log(
) {
    let e = env(AccessMode::Shadow);
    let readonly = native_pat(&e, Preset::Readonly.scopes(), DAY);
    // Allowed and logged: services.control is a scope the token lacks and is not dangerous.
    let r = go(
        &e,
        send(Method::POST, "/api/services/x/start")
            .bearer(&readonly)
            .json("{}"),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    // Refused exactly as in scoped: a dangerous scope, a human-only route.
    for (m, p) in [
        (Method::POST, "/api/plugins/install"),
        (Method::POST, "/api/services"),
        (Method::DELETE, "/api/plugins/x"),
        (Method::PUT, "/api/secrets/hf-token"),
        (Method::POST, "/api/bridge/pairing/x/approve"),
        (Method::POST, "/api/python/install"),
    ] {
        let r = go(&e, send(m.clone(), p).bearer(&readonly).json("{}")).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("insufficient_scope")),
            "{m} {p}"
        );
    }
    // Never an authentication failure: no credential, a bad one, an expired one, a revoked one.
    assert_eq!(
        go(&e, send(Method::POST, "/api/services/x/start"))
            .await
            .status,
        401
    );
    assert_eq!(
        go(
            &e,
            send(Method::POST, "/api/services/x/start").bearer("nonsense")
        )
        .await
        .status,
        401
    );
    let expired = native_pat(&e, ScopeSet::of(&["system.read"]), 1000);
    e.clock.advance(5000);
    assert_eq!(
        go(
            &e,
            send(Method::POST, "/api/services/x/start").bearer(&expired)
        )
        .await
        .code()
        .as_deref(),
        Some("token_expired")
    );
    // Still no Origin trust, and the row rule holds: an unclassified route is refused.
    assert_eq!(
        go(
            &e,
            send(Method::POST, "/api/services/x/start").h("origin", "https://oaiy.com")
        )
        .await
        .status,
        401
    );
    assert_eq!(
        go(&e, send(Method::GET, UNCLASSIFIED).bearer(&readonly))
            .await
            .code()
            .as_deref(),
        Some("unclassified_route")
    );
    // The mismatch that was allowed is in the noise log as `auth.shadow_denied`.
    e.audit.flush_noise();
    let noise = e.audit.read(LogFile::Noise, 50, None, None);
    assert!(
        noise
            .iter()
            .any(|l| l["event"] == "auth.shadow_denied" && l["count"] == 1),
        "{noise:?}"
    );
    // In scoped the same request is refused.
    let scoped = env(AccessMode::Scoped);
    let readonly = native_pat(&scoped, Preset::Readonly.scopes(), DAY);
    let r = go(
        &scoped,
        send(Method::POST, "/api/services/x/start")
            .bearer(&readonly)
            .json("{}"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("insufficient_scope"))
    );
}

// ======================================== CORS (T16) =============================================

fn preflight(path: &str, origin: &str, method: &str, asks_headers: &str) -> Send {
    send(Method::OPTIONS, path)
        .h("origin", origin)
        .h("access-control-request-method", method)
        .h("access-control-request-headers", asks_headers)
}

#[tokio::test]
async fn t16_cors_answers_a_paired_origin_a_public_route_and_nobody_else() {
    let e = env(AccessMode::Scoped);
    let _paired = browser_pat(
        &e,
        "https://formlogic.example",
        &["services.read", "calendar.write"],
    );
    // An unpaired origin: the preflight is 204 with no CORS headers, so the browser blocks the real request.
    for (path, method) in [
        ("/api/services", "GET"),
        ("/api/bridge/runs", "POST"),
        ("/api/config", "GET"),
    ] {
        let r = go(
            &e,
            preflight(path, "https://evil.example", method, "authorization"),
        )
        .await;
        assert_eq!(r.status, 204);
        assert!(
            r.headers
                .keys()
                .all(|k| !k.as_str().starts_with("access-control-")),
            "{path}: {:?}",
            r.headers
        );
    }
    // A paired origin: its own origin echoed, PATCH allowed, `authorization` listed by name.
    let r = go(
        &e,
        preflight(
            "/api/calendar/appointments/:id",
            "https://formlogic.example",
            "PATCH",
            "authorization, content-type",
        ),
    )
    .await;
    assert_eq!(r.status, 204);
    assert_eq!(
        r.headers.get("access-control-allow-origin").unwrap(),
        "https://formlogic.example"
    );
    assert_eq!(r.headers.get("vary").unwrap(), "Origin");
    let methods = r
        .headers
        .get("access-control-allow-methods")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(methods.split(", ").any(|m| m == "PATCH"), "{methods}");
    let allowed = r
        .headers
        .get("access-control-allow-headers")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        allowed.split(", ").any(|h| h == "authorization"),
        "{allowed}"
    );
    assert!(r.headers.get("access-control-allow-credentials").is_none());
    // A public route answers any origin with `*`; the pairing bootstrap included.
    let r = go(
        &e,
        preflight(
            "/api/bridge/pairing",
            "https://never-paired.example",
            "POST",
            "content-type",
        ),
    )
    .await;
    assert_eq!(
        (
            r.status,
            r.headers
                .get("access-control-allow-origin")
                .unwrap()
                .to_str()
                .unwrap()
        ),
        (204, "*")
    );
    // `Origin: null` never matches anything but the public rows.
    let r = go(
        &e,
        preflight("/api/services", "null", "GET", "authorization"),
    )
    .await;
    assert!(r.headers.get("access-control-allow-origin").is_none());
    let r = go(&e, preflight("/api/health", "null", "GET", "content-type")).await;
    assert_eq!(r.headers.get("access-control-allow-origin").unwrap(), "*");
    // The real request carries the headers too, so the app can read a refusal.
    let r = go(
        &e,
        send(Method::GET, "/api/services").h("origin", "https://formlogic.example"),
    )
    .await;
    assert_eq!(r.status, 401);
    assert_eq!(
        r.headers.get("access-control-allow-origin").unwrap(),
        "https://formlogic.example"
    );
    assert!(r
        .headers
        .get("access-control-expose-headers")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("www-authenticate"));
    let r = go(
        &e,
        send(Method::GET, "/api/services").h("origin", "https://evil.example"),
    )
    .await;
    assert!(r.headers.get("access-control-allow-origin").is_none());
    // A path with no route, and a preflight that names no method: nothing.
    assert!(go(
        &e,
        preflight("/api/no/such/route", "https://formlogic.example", "GET", "")
    )
    .await
    .headers
    .get("access-control-allow-origin")
    .is_none());
    // A bare OPTIONS (no requested method) is not a preflight: it is answered `204` by the guard, the same for
    // every path, and gets no CORS headers (what they would say depends on the route: F2).
    for path in ["/api/services", "/api/health", "/api/no/such/route"] {
        let bare = go(
            &e,
            send(Method::OPTIONS, path).h("origin", "https://formlogic.example"),
        )
        .await;
        assert!(
            bare.headers.get("access-control-allow-origin").is_none()
                && bare.headers.get("access-control-allow-headers").is_none(),
            "{path}: {:?}",
            bare.headers
        );
    }
}

#[tokio::test]
async fn t16_private_network_access_is_answered_only_to_an_origin_that_may_have_it() {
    let e = env(AccessMode::Scoped);
    browser_pat(&e, "https://formlogic.example", &["services.read"]);
    let pna = |origin: &str, path: &str| {
        preflight(path, origin, "GET", "authorization")
            .h("access-control-request-private-network", "true")
    };
    let r = go(&e, pna("https://formlogic.example", "/api/services")).await;
    assert_eq!(
        r.headers
            .get("access-control-allow-private-network")
            .unwrap(),
        "true"
    );
    let r = go(&e, pna("https://evil.example", "/api/services")).await;
    assert!(
        r.headers
            .get("access-control-allow-private-network")
            .is_none(),
        "a hostile page must not be told it may reach the port"
    );
    let r = go(&e, pna("https://evil.example", "/api/health")).await;
    assert_eq!(
        r.headers
            .get("access-control-allow-private-network")
            .unwrap(),
        "true",
        "the public row is answered for anyone"
    );
    // And not unless asked.
    let r = go(
        &e,
        preflight(
            "/api/services",
            "https://formlogic.example",
            "GET",
            "authorization",
        ),
    )
    .await;
    assert!(r
        .headers
        .get("access-control-allow-private-network")
        .is_none());
}

#[tokio::test]
async fn t16_the_six_same_origin_routes_get_no_cors_headers_from_anyone() {
    let e = env(AccessMode::Scoped);
    browser_pat(&e, "https://formlogic.example", &["services.read"]);
    for (path, method) in [
        ("/api/auth/info", "GET"),
        ("/api/auth/session", "GET"),
        ("/api/auth/login", "POST"),
        ("/api/auth/setup", "POST"),
        ("/api/auth/link", "POST"),
        ("/api/auth/callback", "POST"),
    ] {
        for origin in ["https://formlogic.example", "https://evil.example"] {
            let r = go(&e, preflight(path, origin, method, "content-type")).await;
            assert!(
                r.headers
                    .keys()
                    .all(|k| !k.as_str().starts_with("access-control-")),
                "{path} {origin}"
            );
            let real = go(
                &e,
                send(Method::from_bytes(method.as_bytes()).unwrap(), path)
                    .h("origin", origin)
                    .json("{}"),
            )
            .await;
            assert!(
                real.headers
                    .keys()
                    .all(|k| !k.as_str().starts_with("access-control-")),
                "{path} {origin}"
            );
        }
    }
}

// ================================= the routes: whoami, derive, info ==============================

#[tokio::test]
async fn whoami_says_who_and_what() {
    let e = env(AccessMode::Scoped);
    let r = go(
        &e,
        send(Method::GET, "/api/auth/whoami").bearer(STATIC_TOKEN),
    )
    .await;
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(
        (
            v["kind"].as_str(),
            v["id"].as_str(),
            v["elevated"].as_bool(),
            v["controlLevel"].as_str(),
            v["persisted"].as_bool()
        ),
        (
            Some("static"),
            Some("static"),
            Some(false),
            Some("none"),
            Some(false)
        )
    );
    assert_eq!(v["scopes"].as_array().unwrap().len(), 15);
    assert!(v["origin"].is_null() && v["sessionExpiresMs"].is_null());
    // A paired app: its own scopes, origin and expiry, and no secret in the answer.
    let pat = browser_pat(
        &e,
        "https://formlogic.example",
        &["control.read", "control.project", "ai.read"],
    );
    let r = go(
        &e,
        send(Method::GET, "/api/auth/whoami")
            .bearer(&pat)
            .h("origin", "https://formlogic.example"),
    )
    .await;
    let v = r.json();
    assert_eq!(
        (
            v["kind"].as_str(),
            v["origin"].as_str(),
            v["controlLevel"].as_str()
        ),
        (
            Some("pat"),
            Some("https://formlogic.example"),
            Some("project")
        )
    );
    assert_eq!(v["expiresMs"], T0 + 30 * DAY);
    assert!(
        !r.text.contains(&pat) && !r.text.contains("AAEC"),
        "no secret in whoami"
    );
    // The desk credential of the dashboard is elevated.
    let d = desk(&e, App::Dash, Preset::Owner, &["tauri://localhost"]);
    let v = go(
        &e,
        send(Method::GET, "/api/auth/whoami")
            .bearer(&d)
            .h("origin", "tauri://localhost"),
    )
    .await
    .json();
    assert_eq!(
        (v["kind"].as_str(), v["elevated"].as_bool()),
        (Some("desk"), Some(true))
    );
    assert_eq!(v["scopes"].as_array().unwrap().len(), 54);
}

#[tokio::test]
async fn derive_makes_a_child_of_at_most_what_the_caller_holds() {
    let e = env(AccessMode::Scoped);
    let d = |body: &str| {
        send(Method::POST, "/api/auth/derive")
            .bearer(STATIC_TOKEN)
            .json(body)
    };
    let r = go(
        &e,
        d(r#"{"scopes":["ai.read","ai.use"],"ttlSeconds":600,"label":"chatgpt"}"#),
    )
    .await;
    assert_eq!(r.status, 201, "{}", r.text);
    let v = r.json();
    assert_eq!(v["scopes"], serde_json::json!(["ai.read", "ai.use"]));
    assert_eq!(v["expiresMs"], T0 + 600_000);
    let token = v["token"].as_str().unwrap().to_string();
    assert!(super::token::parse(&token).is_some_and(|p| p.kind == Kind::Run));
    // The child works, is a `run` credential and holds only what it was given.
    let w = go(&e, send(Method::GET, "/api/auth/whoami").bearer(&token))
        .await
        .json();
    assert_eq!(
        (
            w["kind"].as_str(),
            w["label"].as_str(),
            w["scopes"].as_array().map(Vec::len)
        ),
        (Some("run"), Some("chatgpt"), Some(2))
    );
    assert_eq!(
        go(
            &e,
            send(Method::POST, "/api/ai/v1/chat/completions")
                .bearer(&token)
                .json("{}")
        )
        .await
        .status,
        200
    );
    assert_eq!(
        go(&e, send(Method::GET, "/api/config").bearer(&token))
            .await
            .code()
            .as_deref(),
        Some("insufficient_scope")
    );
    // A derived credential cannot derive.
    let r = go(
        &e,
        send(Method::POST, "/api/auth/derive")
            .bearer(&token)
            .json(r#"{"scopes":["ai.read"]}"#),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("derive_refused"))
    );
    // More than the parent holds, a dangerous scope, an auth scope: refused.
    for scopes in [
        r#"["flows.approve"]"#,
        r#"["ai.read","auth.read"]"#,
        r#"["services.define"]"#,
        r#"["models.write","services.define"]"#,
    ] {
        let r = go(&e, d(&format!(r#"{{"scopes":{scopes}}}"#))).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("derive_refused")),
            "{scopes}"
        );
    }
    // The static token has no `control.project`: cannot hand it on.
    assert_eq!(
        go(&e, d(r#"{"scopes":["control.project","control.read"]}"#))
            .await
            .status,
        403
    );
}

#[tokio::test]
async fn derive_checks_its_request_before_it_makes_anything() {
    let e = env(AccessMode::Scoped);
    let d = |body: &str| {
        send(Method::POST, "/api/auth/derive")
            .bearer(STATIC_TOKEN)
            .json(body)
    };
    for (body, status, code) in [
        ("not json", 400, "bad_request"),
        (r#"{"scopes":"ai.read"}"#, 400, "bad_request"),
        (r#"{}"#, 400, "bad_request"),
        (r#"{"scopes":[]}"#, 400, "invalid_request"),
        (r#"{"scopes":["nonsense.scope"]}"#, 400, "invalid_request"),
        (r#"{"scopes":["ai.*"]}"#, 400, "invalid_request"),
        (
            r#"{"scopes":["ai.read"],"ttlSeconds":0}"#,
            400,
            "invalid_request",
        ),
        (
            r#"{"scopes":["ai.read"],"ttlSeconds":-5}"#,
            400,
            "bad_request",
        ),
        (
            r#"{"scopes":["ai.read"],"label":""}"#,
            400,
            "invalid_request",
        ),
        (
            r#"{"scopes":["ai.read"],"label":"bad\u0007label"}"#,
            400,
            "invalid_request",
        ),
    ] {
        let r = go(&e, d(body)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (status, Some(code)),
            "{body}: {}",
            r.text
        );
    }
    // The label is at most 80 characters, and the scopes at most 64.
    assert_eq!(
        go(
            &e,
            d(&format!(
                r#"{{"scopes":["ai.read"],"label":"{}"}}"#,
                "x".repeat(81)
            ))
        )
        .await
        .status,
        400
    );
    assert_eq!(
        go(
            &e,
            d(&format!(
                r#"{{"scopes":["ai.read"],"label":"{}"}}"#,
                "x".repeat(80)
            ))
        )
        .await
        .status,
        201
    );
    let many = vec!["ai.read"; 65].join("\",\"");
    assert_eq!(
        go(&e, d(&format!(r#"{{"scopes":["{many}"]}}"#)))
            .await
            .status,
        400
    );
    // Content-Type must be JSON: a form post is the classic cross-site shape.
    let r = go(
        &e,
        send(Method::POST, "/api/auth/derive")
            .bearer(STATIC_TOKEN)
            .h("content-type", "text/plain")
            .json_body_only(r#"{"scopes":["ai.read"]}"#),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (415, Some("unsupported_media_type"))
    );
    let r = go(
        &e,
        send(Method::POST, "/api/auth/derive").bearer(STATIC_TOKEN),
    )
    .await;
    assert_eq!(r.status, 415);
    // The body is at most 16 KiB.
    let big = format!(
        r#"{{"scopes":["ai.read"],"label":"ok","pad":"{}"}}"#,
        "x".repeat(20_000)
    );
    assert_eq!(go(&e, d(&big)).await.status, 400);
    // A `ttlSeconds` a day or more is a day.
    let r = go(&e, d(r#"{"scopes":["ai.read"],"ttlSeconds":999999999}"#)).await;
    assert_eq!(r.json()["expiresMs"], T0 + DAY);
    // Nothing was made by the refusals: only the accepted ones exist.
    assert_eq!(e.store.live_count(Kind::Run), 2);
}

impl Send {
    /// Set the body without changing the content type.
    fn json_body_only(mut self, body: &str) -> Send {
        self.body = Some(body.to_string());
        self
    }
}

#[tokio::test]
async fn derive_from_a_paired_token_inherits_its_origin_binding_and_at_most_thirty_a_minute_are_made(
) {
    let e = env(AccessMode::Scoped);
    let pat = browser_pat(&e, "https://formlogic.example", &["ai.read", "ai.use"]);
    let ask = |body: &str| {
        send(Method::POST, "/api/auth/derive")
            .bearer(&pat)
            .h("origin", "https://formlogic.example")
            .json(body)
    };
    let r = go(&e, ask(r#"{"scopes":["ai.read"]}"#)).await;
    assert_eq!(r.status, 201, "{}", r.text);
    let child = r.json()["token"].as_str().unwrap().to_string();
    // The child keeps its parent's binding and works only from the machine: the bound origin, from loopback.
    let get = || send(Method::GET, "/api/ai/sources").bearer(&child);
    assert_eq!(
        go(&e, get().h("origin", "https://formlogic.example"))
            .await
            .status,
        200
    );
    for (name, s) in [
        ("another origin", get().h("origin", "https://evil.example")),
        ("no origin", get()),
        (
            "not from loopback",
            get()
                .h("origin", "https://formlogic.example")
                .peer("203.0.113.9:4000")
                .h("host", "localhost:17972"),
        ),
    ] {
        let r = go(&e, s).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("origin_mismatch")),
            "{name}"
        );
    }
    for _ in 0..29 {
        assert_eq!(go(&e, ask(r#"{"scopes":["ai.read"]}"#)).await.status, 201);
    }
    let r = go(&e, ask(r#"{"scopes":["ai.read"]}"#)).await;
    assert_eq!((r.status, r.code().as_deref()), (429, Some("rate_limited")));
    assert!(r.headers.get("retry-after").is_some());
    e.clock.advance(60_000);
    assert_eq!(go(&e, ask(r#"{"scopes":["ai.read"]}"#)).await.status, 201);
}

#[tokio::test]
async fn info_is_public_says_what_the_server_saw_and_reveals_no_install_detail() {
    let e = env(AccessMode::Scoped);
    let r = go(&e, send(Method::GET, "/api/auth/info")).await;
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(v["scheme"], "oaiy-auth/1");
    assert_eq!(v["apiVersion"], crate::http::API_VERSION);
    assert_eq!(
        (
            v["loginConfigured"].as_bool(),
            v["setupCode"].as_str(),
            v["secureChannel"].as_bool()
        ),
        (Some(false), Some("none"), Some(true))
    );
    assert_eq!(v["factors"], serde_json::json!(["password"]));
    assert_eq!(v["seen"]["clientIp"], "127.0.0.1");
    assert_eq!(v["app"], "dash");
    let text = r.text.to_lowercase();
    for hidden in [
        "exposure",
        "proxied",
        "trustedproxies",
        "allowed_hosts",
        "public_url",
        "oaiy_",
        "data",
    ] {
        assert!(
            !text.contains(hidden),
            "info must not say `{hidden}`: {text}"
        );
    }
    // A credential, a wrong one included, changes nothing: it is a public route.
    assert_eq!(
        go(&e, send(Method::GET, "/api/auth/info").bearer("garbage"))
            .await
            .status,
        200
    );
    // No CORS: same-origin by construction.
    let r = go(
        &e,
        send(Method::GET, "/api/auth/info").h("origin", "https://evil.example"),
    )
    .await;
    assert!(r.headers.get("access-control-allow-origin").is_none());
}

#[tokio::test]
async fn the_new_routes_leave_a_mark_in_the_audit_log_that_holds_no_secret() {
    let e = env(AccessMode::Scoped);
    let r = go(
        &e,
        send(Method::POST, "/api/auth/derive")
            .bearer(STATIC_TOKEN)
            .json(r#"{"scopes":["ai.read"],"label":"chatgpt"}"#),
    )
    .await;
    let token = r.json()["token"].as_str().unwrap().to_string();
    let audit = e.audit.read(LogFile::Audit, 10, None, None);
    assert_eq!(audit.len(), 1, "{audit:?}");
    assert_eq!(audit[0]["event"], "credential.created");
    assert_eq!(audit[0]["principal"]["kind"], "static");
    assert_eq!(audit[0]["detail"]["derived"], true);
    assert!(
        !audit[0].to_string().contains(&token),
        "the token is shown once and never logged"
    );
    assert!(!audit[0].to_string().contains(STATIC_TOKEN));
}

#[tokio::test]
async fn refusals_are_counted_as_noise_and_never_as_audit() {
    let e = env_with(AccessMode::Scoped, &[], false, true, Some(STATIC_TOKEN));
    let readonly = native_pat(&e, Preset::Readonly.scopes(), DAY);
    for _ in 0..3 {
        assert_eq!(
            go(
                &e,
                send(Method::GET, "/api/services/x/logs").bearer(&readonly)
            )
            .await
            .status,
            403
        );
    }
    assert_eq!(
        go(
            &e,
            send(Method::GET, "/api/config").h("host", "evil.example")
        )
        .await
        .status,
        421
    );
    e.audit.flush_noise();
    let noise = e.audit.read(LogFile::Noise, 20, None, None);
    let denied: u64 = noise
        .iter()
        .filter(|l| l["event"] == "auth.denied")
        .map(|l| l["count"].as_u64().unwrap())
        .sum();
    assert_eq!(denied, 4, "{noise:?}");
    assert!(e.audit.read(LogFile::Audit, 20, None, None).is_empty());
    // A guess of the noise: the lines carry an address and a host, never a token.
    let text = serde_json::to_string(&noise).unwrap();
    assert!(!text.contains(&readonly) && !text.contains(STATIC_TOKEN));
}

// ============================== which requests the guard claims (legacy) ==========================

/// A router that reports, for each request, whether the guard would claim it.
fn claims_router(mode: AccessMode) -> (Router, Arc<Guard>) {
    let e = env_with(mode, &[], false, true, Some(STATIC_TOKEN));
    let guard = e.guard.clone();
    let mut patterns: Vec<&str> = ROUTES.iter().map(|r| r.pattern).collect();
    patterns.sort_unstable();
    patterns.dedup();
    let mut app = Router::new();
    for p in patterns {
        app = app.route(p, any(|| async { "ok" }));
    }
    let app = app.layer(middleware::from_fn_with_state(
        guard.clone(),
        |axum::extract::State(g): axum::extract::State<Arc<Guard>>,
         req: axum::extract::Request,
         next: axum::middleware::Next| async move {
            let claims = g.claims(&req);
            let mut response = next.run(req).await;
            response.headers_mut().insert(
                "x-claims",
                axum::http::HeaderValue::from_static(if claims { "1" } else { "0" }),
            );
            response
        },
    ));
    (app, guard)
}

#[tokio::test]
async fn in_legacy_mode_the_new_guard_claims_exactly_the_routes_the_model_adds() {
    let (app, _guard) = claims_router(AccessMode::Legacy);
    let (mut claimed, mut left_alone) = (0, 0);
    // The patterns that existed before the model, worked out here from the rows and not from the helper the
    // guard uses.
    let old_patterns: std::collections::BTreeSet<&str> = ROUTES
        .iter()
        .filter(|r| r.since == 1)
        .map(|r| r.pattern)
        .collect();
    for r in ROUTES {
        for m in methods(r.method) {
            let req = Request::builder()
                .method(m.clone())
                .uri(concrete(r.pattern))
                .body(Body::empty())
                .unwrap();
            let response = app.clone().oneshot(req).await.unwrap();
            let claims = response.headers().get("x-claims").unwrap() == "1";
            assert_eq!(
                claims,
                r.since == 2 && !old_patterns.contains(r.pattern),
                "{m} {}: since {}",
                r.pattern,
                r.since
            );
            if claims {
                claimed += 1
            } else {
                left_alone += 1
            }
        }
    }
    assert!(
        claimed > 90 && left_alone > 200,
        "{claimed} claimed, {left_alone} left to the old guard"
    );
    // A path with no route, a method with no row and OPTIONS are the old guard's.
    for (m, p) in [
        (Method::GET, "/api/no/such/route"),
        (Method::GET, "/"),
        (Method::TRACE, "/api/config"),
        (Method::OPTIONS, "/api/auth/info"),
    ] {
        let req = Request::builder()
            .method(m.clone())
            .uri(p)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(response.headers().get("x-claims").unwrap(), "0", "{m} {p}");
    }
}

#[tokio::test]
async fn f1_in_legacy_mode_a_method_the_table_adds_to_an_old_route_is_the_old_guards_not_the_new_ones(
) {
    // `DELETE /api/bridge/pairing` is a row with `since: 2` on the pattern of the old `GET` and `POST`
    // `/api/bridge/pairing`: every method of an old route keeps the guard the route always had.
    let (app, _guard) = claims_router(AccessMode::Legacy);
    for m in [
        Method::GET,
        Method::HEAD,
        Method::POST,
        Method::PUT,
        Method::PATCH,
        Method::DELETE,
        Method::OPTIONS,
    ] {
        let req = Request::builder()
            .method(m.clone())
            .uri("/api/bridge/pairing")
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            response.headers().get("x-claims").unwrap(),
            "0",
            "{m} /api/bridge/pairing is an old route"
        );
    }
    // Every row of that kind is on an old pattern, and there is at least the one the reviewer found.
    let on_old_pattern: Vec<String> = ROUTES
        .iter()
        .filter(|r| {
            r.since == 2
                && ROUTES
                    .iter()
                    .any(|o| o.since == 1 && o.pattern == r.pattern)
        })
        .map(|r| r.key())
        .collect();
    assert!(
        on_old_pattern.contains(&"DELETE /api/bridge/pairing".to_string()),
        "{on_old_pattern:?}"
    );
    // In scoped mode the same request is the new guard's.
    let (scoped, _g) = claims_router(AccessMode::Scoped);
    let req = Request::builder()
        .method(Method::DELETE)
        .uri("/api/bridge/pairing")
        .body(Body::empty())
        .unwrap();
    let response = scoped.oneshot(req).await.unwrap();
    assert_eq!(response.headers().get("x-claims").unwrap(), "1");
}

#[tokio::test]
async fn in_scoped_and_shadow_mode_the_new_guard_claims_every_request() {
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let (app, _guard) = claims_router(mode);
        for (m, p) in [
            (Method::GET, "/api/config"),
            (Method::GET, "/api/no/such/route"),
            (Method::POST, "/api/services"),
            (Method::GET, "/api/auth/info"),
            (Method::GET, "/"),
        ] {
            let req = Request::builder()
                .method(m.clone())
                .uri(p)
                .body(Body::empty())
                .unwrap();
            let response = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                response.headers().get("x-claims").unwrap(),
                "1",
                "{mode:?} {m} {p}"
            );
        }
    }
}

// ================================== F5: two branches the first tests did not reach =====================

/// A request whose header values are raw bytes: a header that is not text.
async fn go_raw(env: &Env, method: Method, path: &str, headers: &[(&str, &[u8])]) -> Reply {
    let mut req = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "localhost:17972");
    for (name, value) in headers {
        req = req.header(
            *name,
            axum::http::HeaderValue::from_bytes(value).expect("a header value of bytes"),
        );
    }
    let mut req = req.body(Body::empty()).unwrap();
    req.extensions_mut().insert(ConnectInfo(
        "127.0.0.1:50000".parse::<SocketAddr>().unwrap(),
    ));
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

#[tokio::test]
async fn y03_an_origin_that_is_not_text_is_no_origin_a_bound_credential_is_bound_to_and_a_native_one_takes_none(
) {
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        let bound = format!(
            "Bearer {}",
            browser_pat(&e, "https://app.example", &["system.read"])
        );
        let native = format!(
            "Bearer {}",
            native_pat(&e, ScopeSet::of(&["system.read"]), DAY)
        );
        // Sanity: the same requests with a text `Origin` go the ways the rules say.
        let ok = go(
            &e,
            send(Method::GET, "/api/config")
                .h("authorization", &bound)
                .h("origin", "https://app.example"),
        )
        .await;
        assert_eq!(ok.status, 200, "{mode:?}: {}", ok.text);
        let native_ok = go(
            &e,
            send(Method::GET, "/api/config").h("authorization", &native),
        )
        .await;
        assert_eq!(native_ok.status, 200, "{mode:?}");
        for origin in [
            &b"\xff\xfe"[..],
            b"https://app.example\xff",
            b"\x80",
            b"h\xe9llo",
        ] {
            let r = go_raw(
                &e,
                Method::GET,
                "/api/config",
                &[("authorization", bound.as_bytes()), ("origin", origin)],
            )
            .await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (403, Some("origin_mismatch")),
                "{mode:?} {origin:?}: {}",
                r.text
            );
            assert!(r.text.contains("bound to another origin"), "{}", r.text);
            let r = go_raw(
                &e,
                Method::GET,
                "/api/config",
                &[("authorization", native.as_bytes()), ("origin", origin)],
            )
            .await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (403, Some("origin_mismatch")),
                "{mode:?} {origin:?}"
            );
            assert!(r.text.contains("native client"), "{}", r.text);
        }
    }
}

#[tokio::test]
async fn y21_the_health_probe_is_exempt_from_the_host_check_for_get_and_head_of_that_one_path_only()
{
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        let foreign = "10.1.2.3:8080";
        for m in [Method::GET, Method::HEAD] {
            let r = go(&e, send(m.clone(), "/api/health").h("host", foreign)).await;
            assert_eq!(r.status, 200, "{mode:?} {m}: {}", r.text);
            // A probe with no Host at all, too.
            let r = go(&e, send(m.clone(), "/api/health").h("host", "")).await;
            assert_eq!(r.status, 200, "{mode:?} {m} with no Host");
        }
        for (m, p) in [
            (Method::POST, "/api/health"),
            (Method::PUT, "/api/health"),
            (Method::PATCH, "/api/health"),
            (Method::DELETE, "/api/health"),
            (Method::OPTIONS, "/api/health"),
            (Method::GET, "/api/health/x"),
            (Method::GET, "/api/healthz"),
            (Method::GET, "/api/config"),
            (Method::HEAD, "/api/config"),
        ] {
            for host in [foreign, ""] {
                let r = go(&e, send(m.clone(), p).h("host", host)).await;
                assert_eq!(
                    (r.status, r.code().as_deref()),
                    (421, Some("misdirected_host")),
                    "{mode:?} {m} {p} Host {host:?}: {}",
                    r.text
                );
            }
        }
    }
}

// ============================= F4: the static token in the shape the design gives it ==============

/// Tokens of the static token's shape (`[\x21-\x7e]{32,256}`, 16 different characters) that the strict bearer
/// rule (`[A-Za-z0-9._~+/=-]`, 128 bytes) alone would refuse.
fn wide_static_tokens() -> Vec<(&'static str, String)> {
    let printable = |len: usize| -> String { ('!'..='~').cycle().take(len).collect() };
    vec![
        (
            "36 characters with a dollar sign and an exclamation mark",
            "Sup3r$ecret!Zq7kLm9VbNw2XyHdFg5!".to_string(),
        ),
        ("129 characters", printable(129)),
        ("256 characters", printable(256)),
    ]
}

#[tokio::test]
async fn f4_a_static_token_of_the_designs_shape_is_a_bearer_in_every_mode_and_a_hostile_one_is_still_400(
) {
    for (what, token) in wide_static_tokens() {
        for mode in [AccessMode::Scoped, AccessMode::Shadow] {
            let e = env_with(mode, &[], false, true, Some(&token));
            // The configured token: the `cli` preset, as for any other shape.
            let r = go(&e, send(Method::GET, "/api/config").bearer(&token)).await;
            assert_eq!(r.status, 200, "{mode:?} {what}: {}", r.text);
            let me = go(&e, send(Method::GET, "/api/auth/whoami").bearer(&token)).await;
            assert_eq!(me.status, 200, "{mode:?} {what}");
            assert_eq!(me.json()["kind"], "static");
            assert_eq!(me.json()["scopes"].as_array().map(Vec::len), Some(15));
            // Another token of the same shape is not the operator's, so it is held to the strict rule.
            let mut wrong = token.clone();
            wrong.replace_range(..1, if token.starts_with('Z') { "Y" } else { "Z" });
            let r = go(&e, send(Method::GET, "/api/config").bearer(&wrong)).await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (400, Some("bad_request")),
                "{mode:?} {what}"
            );
            // A bearer that is not the token stays under the strict rule: 129 bytes, a space, a comma.
            for hostile in ["a".repeat(129), "a b".to_string(), "abc,def".to_string()] {
                let r = go(
                    &e,
                    send(Method::GET, "/api/config")
                        .h("authorization", &format!("Bearer {hostile}")),
                )
                .await;
                assert_eq!(
                    (r.status, r.code().as_deref()),
                    (400, Some("bad_request")),
                    "{mode:?} {what}: {hostile:?}"
                );
            }
        }
    }
}

#[tokio::test]
async fn f4_in_legacy_mode_the_new_routes_take_a_static_token_of_the_designs_shape_too() {
    // `whoami` is judged by the new guard even in `legacy` (it is a route the model adds); the old guard's
    // routes take any token, as they always did.
    for (what, token) in wide_static_tokens() {
        let e = env_with(AccessMode::Legacy, &[], false, false, Some(&token));
        let r = go(&e, send(Method::GET, "/api/auth/whoami").bearer(&token)).await;
        assert_eq!(r.status, 200, "{what}: {}", r.text);
        assert_eq!(r.json()["kind"], "static");
    }
}

#[tokio::test]
async fn f4_a_static_token_that_fails_the_shape_rule_is_ignored_where_the_new_guard_judges_every_route_and_kept_in_legacy(
) {
    // ACC-14 flips what this test pinned (it used to say that such a token is still accepted "until the startup
    // rule exists"). The rule of design 4.1 is a startup refusal in `oaiy-server` (`auth::exposure`, exit 78) and
    // "ignored with a warning" in every other embedding: the desktop's guard in `scoped` and `shadow` drops the
    // token. In `legacy`, which changes nothing for the routes that existed before the model, the token stays
    // what it was (a bearer; the new routes take it as the `cli` preset), and a line says what will change.
    for token in ["short", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "change-me"] {
        assert!(
            super::token::check_static_token_shape(token).is_err(),
            "{token} fails the shape rule"
        );
        for mode in [AccessMode::Scoped, AccessMode::Shadow] {
            let e = env_with(mode, &[], false, true, Some(token));
            // Not the operator's token any more: a bearer of the strict charset that names nobody.
            for path in ["/api/auth/whoami", "/api/config"] {
                let r = go(&e, send(Method::GET, path).bearer(token)).await;
                assert_eq!(
                    (r.status, r.code().as_deref()),
                    (401, Some("token_invalid")),
                    "{mode:?} {token} {path}: {}",
                    r.text
                );
            }
        }
        let e = env_with(AccessMode::Legacy, &[], false, true, Some(token));
        let r = go(&e, send(Method::GET, "/api/auth/whoami").bearer(token)).await;
        assert_eq!(r.status, 200, "legacy {token}: {}", r.text);
        assert_eq!(r.json()["kind"], "static");
    }
    // A short token outside the strict charset is refused by neither rule in either way: it is a 400 as before.
    let e = env_with(AccessMode::Scoped, &[], false, true, Some("ab$cd"));
    let r = go(&e, send(Method::GET, "/api/config").bearer("ab$cd")).await;
    assert_eq!(r.status, 400);
    // A token of the right shape is kept in every mode (the test above holds it to that too).
    for mode in [AccessMode::Scoped, AccessMode::Shadow, AccessMode::Legacy] {
        let e = env_with(mode, &[], false, true, Some(STATIC_TOKEN));
        let r = go(
            &e,
            send(Method::GET, "/api/auth/whoami").bearer(STATIC_TOKEN),
        )
        .await;
        assert_eq!(r.status, 200, "{mode:?}");
    }
}

// ======================= F9: what the guard says of the address it sees, and what it keeps ================

fn proxied_env() -> Env {
    env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        Some(STATIC_TOKEN),
    )
}

fn events_named(e: &Env, name: &str) -> Vec<Value> {
    e.audit
        .read(LogFile::Audit, 1000, None, None)
        .into_iter()
        .filter(|l| l["event"] == name)
        .collect()
}

#[tokio::test]
async fn f9_a_forwarded_header_on_a_local_install_is_audited_and_logged_once_a_minute() {
    let e = env(AccessMode::Scoped);
    async fn ask(e: &Env, header: &str) -> Reply {
        go(
            e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .h(header, "203.0.113.9"),
        )
        .await
    }
    for _ in 0..6 {
        let r = ask(&e, "x-forwarded-for").await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (421, Some("proxy_detected"))
        );
    }
    let lines = events_named(&e, "proxy.detected");
    assert_eq!(lines.len(), 1, "six requests, one audit line: {lines:?}");
    assert_eq!(lines[0]["detail"]["header"], "x-forwarded-for");
    assert_eq!(lines[0]["ip"], "127.0.0.1");
    assert_eq!(e.guard.note_counts().0, 1, "and one log line");
    // Still the same minute a little later; a minute on, one more (and it names the header it saw).
    e.clock.advance(59_000);
    ask(&e, "via").await;
    assert_eq!(events_named(&e, "proxy.detected").len(), 1);
    e.clock.advance(1_000);
    ask(&e, "via").await;
    let lines = events_named(&e, "proxy.detected");
    assert_eq!(lines.len(), 2);
    assert_eq!(e.guard.note_counts().0, 2);
    assert!(lines.iter().any(|l| l["detail"]["header"] == "via"));
}

#[tokio::test]
async fn f9_a_proxy_that_says_http_for_an_https_host_is_audited_once() {
    let e = proxied_env();
    for _ in 0..4 {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .h("host", "dash.example.com")
                .h("x-forwarded-for", "203.0.113.9")
                .h("x-forwarded-proto", "http"),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (400, Some("proxy_misconfigured"))
        );
    }
    assert_eq!(events_named(&e, "proxy.misconfigured").len(), 1);
    // Once: a minute later it is still the one.
    e.clock.advance(3_600_000);
    go(
        &e,
        send(Method::GET, "/api/config")
            .h("host", "dash.example.com")
            .h("x-forwarded-for", "203.0.113.9")
            .h("x-forwarded-proto", "http"),
    )
    .await;
    assert_eq!(events_named(&e, "proxy.misconfigured").len(), 1);
}

#[tokio::test]
async fn f9_a_trusted_proxys_unusable_forwarded_for_is_logged_once_a_minute_and_a_usable_one_is_not(
) {
    let e = proxied_env();
    async fn ask(e: &Env, xff: &str) -> Reply {
        go(
            e,
            send(Method::GET, "/api/config")
                .h("host", "dash.example.com")
                .h("x-forwarded-for", xff)
                .h("x-forwarded-proto", "https"),
        )
        .await
    }
    // A usable header: nothing to say.
    ask(&e, "203.0.113.9").await;
    assert_eq!(e.guard.note_counts().1, 0);
    // An entry that is not an address, and a header of nothing but trusted proxies: the proxy stands for everyone.
    for xff in ["not-an-ip", "203.0.113.9, garbage", "127.0.0.1", "::1"] {
        for _ in 0..3 {
            ask(&e, xff).await;
        }
    }
    assert_eq!(e.guard.note_counts().1, 1, "twelve requests, one line");
    e.clock.advance(60_000);
    ask(&e, "not-an-ip").await;
    assert_eq!(e.guard.note_counts().1, 2);
}

#[tokio::test]
async fn f9_the_throttle_is_saved_when_it_changed_and_a_new_guard_starts_from_the_file() {
    use super::bearer_throttle::ThrottleFile;
    let dir = TempDir::new("guard-throttle");
    let path = dir.0.join("throttle.json");
    // A LAN listener: the throttle applies to every peer there, a loopback one too.
    let wrong = format!("oaiypat_{}_{}", "0123456789abcdef", "A".repeat(43));
    async fn ask(e: &Env, peer: &str, wrong: &str) -> Reply {
        go(e, send(Method::GET, "/api/config").peer(peer).bearer(wrong)).await
    }
    let e = env_at_port(
        AccessMode::Scoped,
        &[],
        true,
        false,
        Some(STATIC_TOKEN),
        17972,
    );
    let (file, saved) = ThrottleFile::open(&path);
    assert!(saved.is_none());
    e.guard.keep_throttle_in(file, saved.as_ref());
    e.guard.flush_throttle();
    assert!(!path.exists(), "nothing changed, nothing written");
    for _ in 0..20 {
        assert_eq!(ask(&e, "192.168.1.20:5000", &wrong).await.status, 401);
    }
    assert_eq!(
        ask(&e, "192.168.1.20:5000", &wrong).await.status,
        429,
        "blocked"
    );
    e.guard.flush_throttle();
    assert!(path.exists(), "changed, so written");
    let saved_text = std::fs::read_to_string(&path).unwrap();
    // Written when it changed, not on every round of upkeep.
    std::fs::remove_file(&path).unwrap();
    e.guard.flush_throttle();
    assert!(!path.exists(), "not written again while nothing changes");
    // A guard that starts later from the file knows the block, and only that one.
    std::fs::write(&path, &saved_text).unwrap();
    let again = env_at_port(
        AccessMode::Scoped,
        &[],
        true,
        false,
        Some(STATIC_TOKEN),
        17972,
    );
    let (file, saved) = ThrottleFile::open(&path);
    again.guard.keep_throttle_in(file, saved.as_ref());
    assert_eq!(
        ask(&again, "192.168.1.20:5000", &wrong).await.status,
        429,
        "the block survived"
    );
    assert_eq!(
        ask(&again, "192.168.1.21:5000", &wrong).await.status,
        401,
        "and nobody else is blocked"
    );
    // Without the file it is gone (this is what a restart used to do).
    let fresh = env_at_port(
        AccessMode::Scoped,
        &[],
        true,
        false,
        Some(STATIC_TOKEN),
        17972,
    );
    assert_eq!(ask(&fresh, "192.168.1.20:5000", &wrong).await.status, 401);
}

// ================================ F8: a LAN listener on a port the schemes use ========================

#[tokio::test]
async fn f8_a_lan_listener_on_port_80_or_443_or_another_answers_the_address_it_was_reached_at() {
    for (bound, hosts, refused) in [
        (
            80u16,
            vec!["192.168.1.5", "192.168.1.5:80"],
            vec!["192.168.1.5:17972"],
        ),
        (
            443,
            vec!["192.168.1.5", "192.168.1.5:443"],
            vec!["192.168.1.5:17972"],
        ),
        (
            17972,
            vec!["192.168.1.5:17972"],
            vec!["192.168.1.5", "192.168.1.5:80", "192.168.1.5:443"],
        ),
        (8080, vec!["192.168.1.5:8080"], vec!["192.168.1.5"]),
    ] {
        let e = env_at_port(
            AccessMode::Scoped,
            &[],
            true,
            false,
            Some(STATIC_TOKEN),
            bound,
        );
        for host in hosts {
            // A peer on the LAN, with the token: the request is for this server.
            let r = go(
                &e,
                send(Method::GET, "/api/config")
                    .h("host", host)
                    .peer("192.168.1.20:50000")
                    .bearer(STATIC_TOKEN),
            )
            .await;
            assert_eq!(r.status, 200, "bound {bound}, Host {host}: {}", r.text);
        }
        for host in refused {
            let r = go(
                &e,
                send(Method::GET, "/api/config")
                    .h("host", host)
                    .peer("192.168.1.20:50000")
                    .bearer(STATIC_TOKEN),
            )
            .await;
            assert_eq!(
                (r.status, r.code().as_deref()),
                (421, Some("misdirected_host")),
                "bound {bound}, Host {host}"
            );
        }
    }
}

// ============================ F7: a busy parent does not wash out the audit log ======================

#[tokio::test]
async fn f7_a_parent_that_derives_twelve_in_a_minute_writes_one_audit_line_and_a_count() {
    let e = env(AccessMode::Scoped);
    for i in 0..12 {
        let r = go(
            &e,
            send(Method::POST, "/api/auth/derive")
                .bearer(STATIC_TOKEN)
                .json(r#"{"scopes":["system.read"],"ttlSeconds":60,"label":"x"}"#),
        )
        .await;
        assert_eq!(r.status, 201, "#{i}: {}", r.text);
    }
    e.audit.flush_noise();
    let lines = e.audit.read(LogFile::Audit, 100, None, None);
    let created = lines
        .iter()
        .filter(|l| l["event"] == "credential.created")
        .count();
    let more: Vec<&Value> = lines
        .iter()
        .filter(|l| l["event"] == "credential.derived_more")
        .collect();
    assert_eq!(created, 1, "one in full");
    assert_eq!(more.len(), 1, "one count");
    assert_eq!(more[0]["detail"]["count"], 11);
}

// ================================= F2: OPTIONS that is not a preflight ================================

/// A router with a probe on the engine gateway (an `any` route: its handler runs for every method, and
/// finds the engine), a public route, and the guard and CORS layers the listener puts in front.
fn options_probe(guard: &Arc<Guard>, hits: &Arc<std::sync::atomic::AtomicUsize>) -> Router {
    use std::sync::atomic::Ordering;
    let probe = {
        let hits = hits.clone();
        move || {
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                (
                    axum::http::StatusCode::SERVICE_UNAVAILABLE,
                    "engine_unavailable",
                )
            }
        }
    };
    Router::new()
        .route("/api/ai/engine/gateway/*path", any(probe))
        // A route with handlers for some methods only: the router adds an `Allow` naming them to what a
        // method it has no handler for gets, unless the answer already has one.
        .route(
            "/api/services",
            axum::routing::get(|| async { "ok" }).post(|| async { "ok" }),
        )
        .route("/api/health", axum::routing::get(|| async { "ok" }))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_guard))
        .layer(middleware::from_fn_with_state(guard.clone(), scoped_cors))
}

async fn options_answer(
    app: &Router,
    path: &str,
    headers: &[(&str, &str)],
) -> (u16, Vec<(String, String)>, String) {
    let mut req = Request::builder()
        .method(Method::OPTIONS)
        .uri(path)
        .header("host", "localhost:17972");
    for (n, v) in headers {
        req = req.header(*n, *v);
    }
    let mut req = req.body(Body::empty()).unwrap();
    req.extensions_mut().insert(ConnectInfo(
        "127.0.0.1:50000".parse::<SocketAddr>().unwrap(),
    ));
    let response = app.clone().oneshot(req).await.unwrap();
    let status = response.status().as_u16();
    let mut h: Vec<(String, String)> = response
        .headers()
        .iter()
        .filter(|(n, _)| n.as_str() != "date")
        .map(|(n, v)| (n.as_str().to_owned(), v.to_str().unwrap_or("?").to_owned()))
        .collect();
    h.sort();
    let body = String::from_utf8_lossy(
        &axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap(),
    )
    .into_owned();
    (status, h, body)
}

#[tokio::test]
async fn f2_a_bare_options_never_reaches_a_handler_and_says_nothing_about_which_paths_exist() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for mode in [AccessMode::Scoped, AccessMode::Shadow] {
        let e = env(mode);
        let hits = Arc::new(AtomicUsize::new(0));
        let app = options_probe(&e.guard, &hits);
        let token = native_pat(&e, ScopeSet::of(&["ai.use"]), DAY);
        let bearer = format!("Bearer {token}");
        for extra in [
            vec![],
            vec![("authorization", bearer.as_str())],
            // An Origin with no requested method is not a preflight either.
            vec![("origin", "https://formlogic.example")],
        ] {
            // A route with a handler for every method, one that has no route at all, and a path no route
            // is under: the same answer to all three, and the handler is never run.
            let gateway = options_answer(&app, "/api/ai/engine/gateway/x", &extra).await;
            let absent = options_answer(&app, "/api/no/such/route", &extra).await;
            let deep = options_answer(&app, "/api/ai/engine/gateway", &extra).await;
            let limited = options_answer(&app, "/api/services", &extra).await;
            assert_eq!(gateway.0, 204, "{mode:?} {extra:?}: {gateway:?}");
            assert_eq!(gateway.2, "", "a 204 has no body");
            assert_eq!(
                gateway, absent,
                "{mode:?} {extra:?}: a path that exists and one that does not answer alike"
            );
            assert_eq!(gateway, deep, "{mode:?} {extra:?}");
            assert_eq!(
                gateway, limited,
                "{mode:?} {extra:?}: the router's own Allow (GET, HEAD, POST) must not show through"
            );
            assert!(
                gateway.1.iter().all(|(n, v)| n != "allow" || v.is_empty()),
                "no Allow header names the methods of a route: {gateway:?}"
            );
            assert_eq!(
                hits.load(Ordering::SeqCst),
                0,
                "{mode:?} {extra:?}: a handler ran for an OPTIONS"
            );
        }
        // A public route is passed on: what its router says is public already (405 and its Allow header).
        let health = options_answer(&app, "/api/health", &[]).await;
        assert_eq!(health.0, 405, "{mode:?}: {health:?}");
        assert!(health.1.iter().any(|(n, _)| n == "allow"), "{health:?}");
        // The request that is not an OPTIONS still reaches the handler (a credential with the scope).
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/api/ai/engine/gateway/x")
            .header("host", "localhost:17972")
            .header("authorization", &bearer)
            .body(Body::empty())
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:50000".parse::<SocketAddr>().unwrap(),
        ));
        let r = app.clone().oneshot(req).await.unwrap();
        assert_eq!(r.status().as_u16(), 503);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn f2_a_real_preflight_is_still_answered_by_the_cors_layer_with_its_headers() {
    use std::sync::atomic::AtomicUsize;
    let e = env(AccessMode::Scoped);
    let hits = Arc::new(AtomicUsize::new(0));
    let app = options_probe(&e.guard, &hits);
    let _pat = browser_pat(&e, "https://formlogic.example", &["ai.use"]);
    let (status, headers, body) = options_answer(
        &app,
        "/api/ai/engine/gateway/x",
        &[
            ("origin", "https://formlogic.example"),
            ("access-control-request-method", "POST"),
        ],
    )
    .await;
    assert_eq!((status, body.as_str()), (204, ""));
    assert!(
        headers
            .iter()
            .any(|(n, v)| n == "access-control-allow-origin" && v == "https://formlogic.example"),
        "{headers:?}"
    );
    assert_eq!(hits.load(std::sync::atomic::Ordering::SeqCst), 0);
}

// ========================================= startup rules ========================================

#[test]
fn the_guard_reports_its_mode_and_the_storage_state_for_health() {
    let e = env(AccessMode::Shadow);
    let x = e.guard.health_extras();
    assert_eq!((x.access, x.storage), ("shadow", "ok"));
    assert_eq!(e.guard.mode(), AccessMode::Shadow);
    let _ = (Only::Everywhere, e.guard.config().port);
}

// ============== ACC-14: proxy-only, forged forwarded headers, the lan listener, the docker shape ======

/// A container pair (design 4.14): bound beyond loopback behind a public URL, the proxy's network named.
fn proxy_only_env() -> Env {
    env_with(
        AccessMode::Scoped,
        &[
            ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ("OAIY_TRUSTED_PROXIES", "172.30.0.0/24"),
        ],
        true,
        false,
        Some(STATIC_TOKEN),
    )
}

#[tokio::test]
async fn t45_a_proxy_only_install_answers_its_proxy_and_this_machine_and_nothing_else() {
    let e = proxy_only_env();
    assert!(e.guard.config().proxy_only);
    let through_proxy = |peer: &str| {
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer(peer)
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.9")
            .add("x-forwarded-proto", "https")
    };
    // The proxy, in its own network (and as a mapped IPv6 address, which is the same peer).
    for peer in [
        "172.30.0.3:5000",
        "172.30.0.254:5000",
        "[::ffff:172.30.0.3]:5000",
    ] {
        let r = go(&e, through_proxy(peer)).await;
        assert_eq!(r.status, 200, "{peer}: {}", r.text);
    }
    // Everyone else who reaches the port directly, with or without the headers a proxy would add: refused, and
    // nothing they send in a header changes it.
    for peer in [
        "203.0.113.9:5000",
        "[2001:db8::9]:5000",
        "192.168.1.9:5000",
        "10.0.0.7:5000",
        "172.30.1.3:5000",
        "172.31.0.3:5000",
        "172.29.255.255:5000",
        "[::ffff:172.31.0.3]:5000",
    ] {
        let r = go(&e, through_proxy(peer)).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("direct_access_refused")),
            "{peer} with the proxy's headers"
        );
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .peer(peer)
                .h("host", "10.9.9.9:17972"),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("direct_access_refused")),
            "{peer} bare"
        );
        // Not a credential in the world changes it, and no request is judged before it: a request with nothing
        // (an anonymous one) is refused the same way, not with a `401` that says a credential would do.
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .peer(peer)
                .h("host", "dash.example.com"),
        )
        .await;
        assert_eq!(
            r.code().as_deref(),
            Some("direct_access_refused"),
            "{peer} anonymous"
        );
    }
    // Every path, routed or not: the pages a later step serves are behind the same door as the API.
    for path in ["/", "/index.html", "/assets/app.js", "/api/no/such/route"] {
        let r = go(
            &e,
            send(Method::GET, path)
                .peer("203.0.113.9:5000")
                .h("host", "dash.example.com"),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("direct_access_refused")),
            "{path}"
        );
        let r = go(
            &e,
            send(Method::GET, path)
                .peer("172.30.0.3:5000")
                .h("host", "dash.example.com"),
        )
        .await;
        assert_ne!(r.code().as_deref(), Some("direct_access_refused"), "{path}");
    }
    // This machine, straight to the port (the CLI on the server, `oaiy-server auth ...`, a check inside the
    // container): answered, with no forwarded header. With one it is not a client of the port but a proxy that
    // was never named.
    for (peer, host) in [
        ("127.0.0.1:5000", "127.0.0.1:17972"),
        ("[::1]:5000", "[::1]:17972"),
        ("127.0.0.1:5000", "localhost:17972"),
    ] {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .peer(peer)
                .h("host", host),
        )
        .await;
        assert_eq!(r.status, 200, "{peer} {host}: {}", r.text);
    }
    let r = go(
        &e,
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer("127.0.0.1:5000")
            .h("host", "127.0.0.1:17972")
            .add("x-forwarded-for", "203.0.113.9"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (403, Some("direct_access_refused"))
    );
}

#[tokio::test]
async fn t45_a_probe_of_health_from_a_pod_address_is_answered_by_a_proxy_only_install_and_nothing_else_is(
) {
    let e = proxy_only_env();
    for m in [Method::GET, Method::HEAD] {
        for peer in ["10.244.1.7:5000", "203.0.113.9:5000", "172.31.9.9:5000"] {
            let r = go(
                &e,
                send(m.clone(), "/api/health")
                    .peer(peer)
                    .h("host", "10.244.1.7:17972"),
            )
            .await;
            assert_eq!(r.status, 200, "{m} {peer}");
        }
    }
    // Only `GET` and `HEAD` of that one path.
    let r = go(
        &e,
        send(Method::POST, "/api/health")
            .peer("10.244.1.7:5000")
            .h("host", "10.244.1.7:17972"),
    )
    .await;
    assert_eq!(r.code().as_deref(), Some("direct_access_refused"));
    let r = go(
        &e,
        send(Method::GET, "/api/auth/info")
            .peer("10.244.1.7:5000")
            .h("host", "10.244.1.7:17972"),
    )
    .await;
    assert_eq!(r.code().as_deref(), Some("direct_access_refused"));
}

#[tokio::test]
async fn t45_an_install_that_is_not_proxy_only_does_not_refuse_direct_peers() {
    // Bound to loopback behind a public URL (Caddy on this machine), a lan listener, and a local install: nobody is
    // refused for not being the proxy. (The rest of the pipeline judges them as it always did.)
    for e in [
        proxied_env(),
        env_with(AccessMode::Scoped, &[], true, false, Some(STATIC_TOKEN)),
        env(AccessMode::Scoped),
    ] {
        assert!(!e.guard.config().proxy_only);
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .peer("192.168.1.9:5000")
                .h("host", "192.168.1.5:17972"),
        )
        .await;
        assert_ne!(r.code().as_deref(), Some("direct_access_refused"));
    }
}

#[tokio::test]
async fn t30_a_forged_forwarded_for_from_an_untrusted_peer_is_ignored_and_dodges_no_throttle() {
    let e = proxied_env();
    // Twenty wrong bearers from one untrusted address, each with another forged client address: they are all that
    // address, and it is blocked at the twentieth.
    let wrong = format!("oaiypat_0123456789abcdef_{}", "A".repeat(43));
    for i in 0..20 {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(&wrong)
                .peer("198.51.100.7:4000")
                .h("host", "dash.example.com")
                .add("x-forwarded-for", &format!("203.0.113.{}", i + 1))
                .add("x-forwarded-proto", "https"),
        )
        .await;
        assert_eq!(r.status, 401, "guess {i}: {}", r.text);
    }
    let r = go(
        &e,
        send(Method::GET, "/api/config")
            .bearer(&wrong)
            .peer("198.51.100.7:4000")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.200")
            .add("x-forwarded-proto", "https"),
    )
    .await;
    assert_eq!(
        (r.status, r.code().as_deref()),
        (429, Some("rate_limited")),
        "the forged addresses did not spread the guesses"
    );
    // Nor did the forgery blame a client for what the peer did: 203.0.113.1 is not blocked, as a client of the
    // real proxy.
    let r = go(
        &e,
        send(Method::GET, "/api/config")
            .bearer(&wrong)
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "203.0.113.1")
            .add("x-forwarded-proto", "https"),
    )
    .await;
    assert_eq!(r.status, 401);
    // What the server understood of a forged header: the peer, on an insecure channel.
    let r = go(
        &e,
        send(Method::GET, "/api/auth/info")
            .peer("198.51.100.8:4000")
            .h("host", "dash.example.com")
            .add("x-forwarded-for", "10.0.0.1")
            .add("x-forwarded-host", "dash.example.com")
            .add("x-forwarded-proto", "https"),
    )
    .await;
    let seen = &r.json()["seen"];
    assert_eq!(
        (
            seen["clientIp"].as_str(),
            seen["viaTrustedProxy"].as_bool(),
            seen["proto"].as_str()
        ),
        (Some("198.51.100.8"), Some(false), Some("http"))
    );
    assert_eq!(r.json()["secureChannel"], false);
}

#[tokio::test]
async fn t30_only_x_forwarded_for_names_a_client_and_the_other_forwarded_headers_are_never_believed(
) {
    let e = proxied_env();
    let info = |extra: &[(&str, &str)]| {
        let mut s = send(Method::GET, "/api/auth/info")
            .h("host", "dash.example.com")
            .add("x-forwarded-proto", "https");
        for (n, v) in extra {
            s = s.add(n, v);
        }
        s
    };
    // A trusted proxy (the default: this machine) that names a client only in another header: the client is the
    // proxy, and it is not a client the server can tell from its neighbours (a warning is logged once).
    for header in [
        ("forwarded", "for=203.0.113.9;proto=https"),
        ("forwarded", "for=\"[2001:db8::9]\""),
        ("x-real-ip", "203.0.113.9"),
        ("cf-connecting-ip", "203.0.113.9"),
        ("true-client-ip", "203.0.113.9"),
        ("x-client-ip", "203.0.113.9"),
        ("x-forwarded-host", "203.0.113.9"),
        ("via", "1.1 203.0.113.9"),
    ] {
        let r = go(&e, info(&[header])).await;
        assert_eq!(
            r.json()["seen"]["clientIp"],
            "127.0.0.1",
            "{}: {}",
            header.0,
            r.text
        );
    }
    // With `X-Forwarded-For` beside them, that one is the answer, whatever the others say.
    let r = go(
        &e,
        info(&[
            ("x-forwarded-for", "203.0.113.9"),
            ("x-real-ip", "198.51.100.1"),
            ("cf-connecting-ip", "198.51.100.2"),
            ("forwarded", "for=198.51.100.3"),
        ]),
    )
    .await;
    assert_eq!(r.json()["seen"]["clientIp"], "203.0.113.9");
    // Several `X-Forwarded-For` lines are one list, walked from the right: the left ones are the client's own.
    let r = go(
        &e,
        info(&[
            ("x-forwarded-for", "198.51.100.66"),
            ("x-forwarded-for", "2001:db8::66, 203.0.113.9"),
            ("x-forwarded-for", "127.0.0.1"),
        ]),
    )
    .await;
    assert_eq!(r.json()["seen"]["clientIp"], "203.0.113.9");
    // One entry that is not an address, anywhere: none of it is believed.
    for bad in [
        "203.0.113.9, unknown",
        "unknown, 203.0.113.9",
        "203.0.113.9:443",
        "[2001:db8::9]",
    ] {
        let r = go(&e, info(&[("x-forwarded-for", bad)])).await;
        assert_eq!(r.json()["seen"]["clientIp"], "127.0.0.1", "{bad}");
    }
}

#[tokio::test]
async fn t30_a_trusted_proxy_that_names_no_client_is_warned_of_once_a_minute_and_the_loopback_cli_is_not(
) {
    let e = proxied_env();
    assert_eq!(e.guard.no_forwarded_for_lines(), 0);
    // The CLI on this machine (a trusted peer by default) reaches the port by a loopback name and sends no header:
    // that is not a proxy that forgot one.
    for _ in 0..3 {
        let r = go(&e, send(Method::GET, "/api/auth/info")).await;
        assert_eq!(r.json()["seen"]["clientIp"], "127.0.0.1");
    }
    assert_eq!(
        e.guard.no_forwarded_for_lines(),
        0,
        "the loopback CLI is not a proxy"
    );
    // A proxy: the public host, and no `X-Forwarded-For`. Every client is the proxy's address; one line a minute.
    let via_proxy = || {
        send(Method::GET, "/api/auth/info")
            .h("host", "dash.example.com")
            .add("x-forwarded-proto", "https")
    };
    for _ in 0..5 {
        let r = go(&e, via_proxy()).await;
        assert_eq!(
            (
                r.json()["seen"]["clientIp"].as_str(),
                r.json()["seen"]["viaTrustedProxy"].as_bool()
            ),
            (Some("127.0.0.1"), Some(true))
        );
    }
    assert_eq!(e.guard.no_forwarded_for_lines(), 1, "logged once");
    e.clock.advance(61_000);
    go(&e, via_proxy()).await;
    assert_eq!(
        e.guard.no_forwarded_for_lines(),
        2,
        "and again a minute later"
    );
    // With a client named, or from a peer that is not trusted, there is nothing to say.
    go(&e, via_proxy().add("x-forwarded-for", "203.0.113.9")).await;
    go(&e, via_proxy().peer("198.51.100.7:4000")).await;
    e.clock.advance(61_000);
    go(&e, via_proxy().add("x-forwarded-for", "203.0.113.9")).await;
    go(&e, via_proxy().peer("198.51.100.7:4000")).await;
    assert_eq!(e.guard.no_forwarded_for_lines(), 2);
}

#[tokio::test]
async fn t30_a_forwarded_for_line_that_is_not_text_makes_the_whole_header_unusable_not_absent() {
    let e = proxied_env();
    let ask = |lines: &[&[u8]]| {
        let mut req = Request::builder()
            .method(Method::GET)
            .uri("/api/auth/info")
            .header("host", "dash.example.com")
            .header("x-forwarded-proto", "https");
        for l in lines {
            req = req.header(
                "x-forwarded-for",
                axum::http::HeaderValue::from_bytes(l).unwrap(),
            );
        }
        let mut req = req.body(Body::empty()).unwrap();
        req.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:50000".parse::<SocketAddr>().unwrap(),
        ));
        req
    };
    // Bytes that are not visible ASCII in a line of the header: the line is an entry that is not an address, so none of
    // the header is believed (the readable line beside it is the client's to write as much as the unreadable one).
    for lines in [
        vec![&b"198.51.100.66\xff"[..], &b"203.0.113.9"[..]],
        vec![&b"203.0.113.9"[..], &b"\xe9"[..]],
        vec![&b"\x80"[..]],
    ] {
        let response = e.app.clone().oneshot(ask(&lines)).await.unwrap();
        let body: Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["seen"]["clientIp"], "127.0.0.1", "{lines:?}");
    }
    let response = e.app.clone().oneshot(ask(&[b"203.0.113.9"])).await.unwrap();
    let body: Value = serde_json::from_slice(
        &axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(body["seen"]["clientIp"], "203.0.113.9");
}

#[tokio::test]
async fn t45_a_docker_bridge_peer_is_a_client_until_the_operator_names_its_network() {
    // A proxy container's peer is a bridge address, and the loopback-only default does not trust it: the address the
    // server sees is the proxy's, for everyone (`seen` says so). With the compose network named it is the client's.
    let default = env_with(
        AccessMode::Scoped,
        &[("OAIY_PUBLIC_URL", "https://dash.example.com")],
        false,
        false,
        Some(STATIC_TOKEN),
    );
    let named = proxy_only_env();
    async fn ask(e: &Env) -> Reply {
        go(
            e,
            send(Method::GET, "/api/auth/info")
                .peer("172.30.0.3:5000")
                .h("host", "dash.example.com")
                .add("x-forwarded-for", "203.0.113.9")
                .add("x-forwarded-proto", "https"),
        )
        .await
    }
    let r = ask(&default).await;
    assert_eq!(
        (
            r.json()["seen"]["clientIp"].as_str(),
            r.json()["secureChannel"].as_bool()
        ),
        (Some("172.30.0.3"), Some(false)),
        "not trusted: its word is worth nothing"
    );
    let r = ask(&named).await;
    assert_eq!(
        (
            r.json()["seen"]["clientIp"].as_str(),
            r.json()["seen"]["viaTrustedProxy"].as_bool(),
            r.json()["secureChannel"].as_bool()
        ),
        (Some("203.0.113.9"), Some(true), Some(true))
    );
}

#[tokio::test]
async fn t45_a_bearer_from_a_public_peer_on_a_lan_listener_is_refused_in_every_way_an_address_can_be_written(
) {
    let lan = env_with(AccessMode::Scoped, &[], true, false, Some(STATIC_TOKEN));
    let ask = |peer: &str| {
        go(
            &lan,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .peer(peer)
                .h("host", "192.168.1.5:17972"),
        )
    };
    for peer in [
        "203.0.113.50:4000",
        "8.8.8.8:4000",
        "172.32.0.1:4000",
        "172.15.255.255:4000",
        "100.128.0.1:4000",
        "100.63.255.255:4000",
        "192.169.0.1:4000",
        "[2001:db8::9]:4000",
        "[2606:4700::1]:4000",
        "[fb00::1]:4000",
        "[::ffff:203.0.113.50]:4000",
        "[::ffff:8.8.8.8]:4000",
    ] {
        let r = ask(peer).await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (403, Some("plaintext_from_public_address")),
            "{peer}"
        );
    }
    for peer in [
        "10.0.0.1:4000",
        "172.16.0.1:4000",
        "172.31.255.255:4000",
        "192.168.255.1:4000",
        "100.64.0.1:4000",
        "100.127.255.255:4000",
        "169.254.1.1:4000",
        "127.0.0.1:4000",
        "[::1]:4000",
        "[fd12:3456::1]:4000",
        "[fc00::1]:4000",
        "[fe80::1]:4000",
        "[::ffff:192.168.1.9]:4000",
        "[::ffff:127.0.0.1]:4000",
    ] {
        let r = ask(peer).await;
        assert_eq!(r.status, 200, "{peer}: {}", r.text);
    }
    // The header a client would forge to look private changes nothing: it is not read on a lan listener
    // (no proxy is trusted there), and the peer is what is judged.
    let r = go(
        &lan,
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer("203.0.113.50:4000")
            .h("host", "192.168.1.5:17972")
            .add("x-forwarded-for", "192.168.1.9"),
    )
    .await;
    assert_eq!(r.code().as_deref(), Some("plaintext_from_public_address"));
}

#[tokio::test]
async fn t45_a_lan_listener_with_a_trusted_proxy_judges_the_bearer_by_the_peer_and_not_by_the_forwarded_client(
) {
    // A proxy on the private network in front of a lan listener: the plaintext rule is about the connection the
    // bearer travels on, which is the proxy's, not the client's.
    let e = env_with(
        AccessMode::Scoped,
        &[("OAIY_TRUSTED_PROXIES", "10.0.0.0/8")],
        true,
        false,
        Some(STATIC_TOKEN),
    );
    let r = go(
        &e,
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer("10.1.1.1:4000")
            .h("host", "192.168.1.5:17972")
            .add("x-forwarded-for", "203.0.113.9"),
    )
    .await;
    assert_eq!(r.status, 200, "{}", r.text);
    // A public peer that says it is that proxy is a public peer.
    let r = go(
        &e,
        send(Method::GET, "/api/config")
            .bearer(STATIC_TOKEN)
            .peer("203.0.113.9:4000")
            .h("host", "192.168.1.5:17972")
            .add("x-forwarded-for", "10.1.1.1"),
    )
    .await;
    assert_eq!(r.code().as_deref(), Some("plaintext_from_public_address"));
}

#[test]
fn a_desktop_ignores_the_settings_of_a_proxy_and_is_never_a_proxied_install() {
    // Design 3.4: the desktop's API is on loopback only. Its environment can name a proxy all it likes.
    let vars = |name: &str| match name {
        "OAIY_PUBLIC_URL" => Some("https://dash.example.com".to_string()),
        "OAIY_AGENT_URL" => Some("https://agent.example.com".to_string()),
        "OAIY_TRUSTED_PROXIES" => Some("10.0.0.0/8".to_string()),
        "OAIY_ALLOW_PUBLIC_PLAINTEXT" => Some("1".to_string()),
        "OAIY_ALLOWED_HOSTS" => Some("nas.example:9000".to_string()),
        _ => None,
    };
    let (c, warnings) = GuardConfig::from_env(&vars, false, true, 17972);
    assert_eq!(c.exposure, super::mode::Exposure::Local);
    assert!(c.trusted.is_empty() && !c.allow_public_plaintext && !c.proxy_only);
    assert_eq!(warnings.len(), 4, "{warnings:?}");
    assert!(warnings.iter().all(|w| w.contains("ignored")));
    // A server reads them.
    let (c, _) = GuardConfig::from_env(&vars, false, false, 17972);
    assert_eq!(c.exposure, super::mode::Exposure::Proxied);
    assert!(!c.trusted.is_empty());
}

#[tokio::test]
async fn t45_the_desktops_host_allow_list_is_the_loopback_names_on_any_port_and_nothing_else() {
    let e = env(AccessMode::Scoped);
    for host in [
        "localhost:17972",
        "127.0.0.1:9999",
        "[::1]:17972",
        "LOCALHOST",
        "127.0.0.1",
    ] {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .h("host", host),
        )
        .await;
        assert_eq!(r.status, 200, "{host}");
    }
    for host in [
        "dash.oaiy.localhost:17972",
        "oaiy.localhost",
        "evil.example:17972",
        "127.0.0.1.evil.example:17972",
        "192.168.1.5:17972",
        "",
    ] {
        let r = go(
            &e,
            send(Method::GET, "/api/config")
                .bearer(STATIC_TOKEN)
                .h("host", host),
        )
        .await;
        assert_eq!(
            (r.status, r.code().as_deref()),
            (421, Some("misdirected_host")),
            "{host:?}"
        );
    }
}
