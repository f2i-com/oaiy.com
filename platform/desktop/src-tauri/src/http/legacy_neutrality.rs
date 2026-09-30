//! The differential test: in `legacy` mode (the default), every route that existed before the access model is
//! answered exactly as the guard of commit 2ea1ee8 answered it (the frozen copy holds four more lines, for the
//! routes of the receptionist's transfers and messages, which that guard did not know: see `frozen_guard.rs`).
//!
//! Two routers are built from the same stub routes: the routes of the real router (the scan of the source that
//! `route_coverage` holds to the table) that existed before the model, each answering `200 ok` to exactly the
//! methods the real router registers on it. One has the live guard of the listener in front of it:
//! `access_guard` with the access mode `legacy`, as `serve` builds it. The other has the frozen copy of the old
//! guard (`frozen_guard.rs`). Every request of a large matrix goes to both, and status, every header but `date`
//! and the body must be identical. The matrix is generated from the route table:
//!
//! - every standard method (`GET`, `HEAD`, `POST`, `PUT`, `PATCH`, `DELETE`, `OPTIONS`) to every path the
//!   table knows, so that the methods a route has, the methods it does not have (`405` after the guard) and the
//!   methods the table adds to a route that existed before (`DELETE /api/bridge/pairing`) are all asked, with
//!   concrete path parameters (the old predicates look at prefixes and suffixes);
//! - credentials: none, the configured token, a wrong one, the process-internal token, a real paired token, a
//!   wrong paired-looking one, a token of the new grammar, an empty bearer, a lowercase scheme and a doubled
//!   space;
//! - `Origin`: none, the desktop's own webviews and pages, `https://oaiy.com` and a subdomain, loopback pages,
//!   `null` and strangers;
//! - the desktop's mode (`gui_mode`) and the headless server's, with a token configured and without.
//!
//! Routes the model adds are not compared (the new guard judges them, by design); unknown paths are.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Method, Request};
use axum::middleware;
use axum::routing::{any, MethodRouter};
use axum::Router;
use tower::ServiceExt;

use super::frozen_guard;
use super::{access_guard, AccessState, AuthConfig};
use crate::auth::routes::{Verb, ROUTES};
use crate::auth::{AccessMode, AccessSettings};
use crate::bridge::pairing::PairingManager;

/// `/api/services/:id/start` as a request path.
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

async fn ok() -> &'static str {
    "ok"
}

/// The routes of the real router (the scan of the source, the same one `route_coverage` holds to the table)
/// that existed before the access model, each answering `200 ok` to exactly the methods the real router
/// registers on it and `405` to the others, as the real router does after the guard. A route that only the
/// model adds is left out: the new guard judges those, by design, and a request for one is then a path that
/// does not exist, which both guards must treat alike.
fn stub_routes() -> Router {
    static STUBS: std::sync::OnceLock<Router> = std::sync::OnceLock::new();
    STUBS.get_or_init(build_stub_routes).clone()
}

fn build_stub_routes() -> Router {
    let mut methods = std::collections::BTreeMap::<String, std::collections::BTreeSet<Verb>>::new();
    for f in crate::auth::route_coverage::scan_main_router().found {
        if crate::auth::routes::pattern_existed_before(&f.pattern) {
            methods.entry(f.pattern).or_default().insert(f.verb);
        }
    }
    let mut app = Router::new();
    for (pattern, verbs) in methods {
        let mut method_router = if verbs.contains(&Verb::Any) {
            any(ok)
        } else {
            MethodRouter::new()
        };
        for v in verbs {
            method_router = match v {
                Verb::Get => method_router.get(ok),
                Verb::Post => method_router.post(ok),
                Verb::Put => method_router.put(ok),
                Verb::Patch => method_router.patch(ok),
                Verb::Delete => method_router.delete(ok),
                Verb::Any => method_router,
            };
        }
        app = app.route(&pattern, method_router);
    }
    app
}

/// Every path pattern the table knows, old or new, as a request path.
fn all_patterns() -> Vec<String> {
    let mut patterns: Vec<String> = ROUTES.iter().map(|r| concrete(r.pattern)).collect();
    patterns.sort_unstable();
    patterns.dedup();
    patterns
}

/// Where a request is sent: every standard method to every path the table knows (so the methods a route has,
/// the methods it does not have, and the methods the table adds to an old route are all asked), and paths that
/// do not exist.
fn requests() -> Vec<(Method, String)> {
    let mut out = Vec::new();
    for path in all_patterns() {
        for m in [
            Method::GET,
            Method::HEAD,
            Method::POST,
            Method::PUT,
            Method::PATCH,
            Method::DELETE,
            Method::OPTIONS,
        ] {
            out.push((m, path.clone()));
        }
    }
    // The methods a route does not have (the router answers 405 after the guard), and paths that do not exist.
    for (m, p) in [
        (Method::DELETE, "/api/health"),
        (Method::PUT, "/api/config"),
        (Method::POST, "/api/config"),
        (Method::GET, "/api/no/such/route"),
        (Method::POST, "/api/no/such/route"),
        (Method::DELETE, "/api/services/x/nope"),
        (Method::GET, "/nope"),
        (Method::GET, "/"),
        (Method::OPTIONS, "/api/config"),
        (Method::OPTIONS, "/api/services"),
        (Method::OPTIONS, "/api/no/such/route"),
        (Method::GET, "/api/bridge/pairing/x/extra"),
        (Method::GET, "/api/update/status/x"),
        (Method::POST, "/api/bridge/pairing/x/approve/y"),
        // A route the model adds that does not exist yet: the router has none, the old guard answers.
        (Method::GET, "/api/system/gpus"),
        (Method::PUT, "/api/services/x/gpu"),
        (Method::POST, "/api/auth/console/status"),
    ] {
        out.push((m, p.to_string()));
    }
    out.sort_by(|a, b| (a.1.as_str(), a.0.as_str()).cmp(&(b.1.as_str(), b.0.as_str())));
    out.dedup();
    out
}

const STATIC_TOKEN: &str = "desk-token";

/// The live listener's guard in `legacy` mode, over the stub routes.
fn live(gui: bool, token: Option<&str>, pairing: crate::bridge::PairingHandle) -> Router {
    let unused = std::env::temp_dir().join("oaiy-legacy-neutrality-nothing-is-made-here");
    let guard = crate::auth::build_guard(
        &AccessSettings::legacy(),
        &unused,
        17972,
        false,
        gui,
        token.map(str::to_owned),
        &|_| None,
    )
    .expect("a legacy guard");
    assert_eq!(guard.mode(), AccessMode::Legacy);
    let state = AccessState {
        legacy: AuthConfig {
            token: token.map(str::to_owned),
            gui_mode: gui,
            pairing: Some(pairing),
        },
        guard,
    };
    stub_routes().layer(middleware::from_fn_with_state(state, access_guard))
}

/// The frozen guard, over the same routes.
fn frozen(gui: bool, token: Option<&str>, pairing: crate::bridge::PairingHandle) -> Router {
    let state = frozen_guard::AuthConfig {
        token: token.map(str::to_owned),
        gui_mode: gui,
        pairing: Some(pairing),
    };
    stub_routes().layer(middleware::from_fn_with_state(
        state,
        frozen_guard::origin_guard,
    ))
}

/// What an answer is, for comparing: the status, every header but `date` (sorted, so an added or a changed
/// header is a difference) and the body.
#[derive(Debug, PartialEq, Eq)]
struct Answer {
    status: u16,
    content_type: Option<String>,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

async fn ask(
    app: &Router,
    method: &Method,
    path: &str,
    authorization: Option<&str>,
    origin: Option<&str>,
) -> Answer {
    let mut req = Request::builder().method(method.clone()).uri(path);
    if let Some(a) = authorization {
        req = req.header("authorization", a);
    }
    if let Some(o) = origin {
        req = req.header("origin", o);
    }
    let response = app
        .clone()
        .oneshot(req.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let mut headers: Vec<(String, String)> = response
        .headers()
        .iter()
        .filter(|(n, _)| n.as_str() != "date")
        .map(|(n, v)| {
            (
                n.as_str().to_owned(),
                String::from_utf8_lossy(v.as_bytes()).into_owned(),
            )
        })
        .collect();
    headers.sort();
    let body = axum::body::to_bytes(response.into_body(), 1 << 20)
        .await
        .unwrap()
        .to_vec();
    Answer {
        status,
        content_type,
        headers,
        body,
    }
}

fn a_paired_token() -> (crate::bridge::PairingHandle, String) {
    let pairing: crate::bridge::PairingHandle = Arc::new(Mutex::new(PairingManager::new()));
    let token = {
        let mut m = pairing.lock().unwrap();
        let id = m.request("formlogic", None, None).pairing_id;
        m.approve(&id).unwrap().token.unwrap()
    };
    (pairing, token)
}

const ORIGINS: [Option<&str>; 12] = [
    None,
    Some("tauri://localhost"),
    Some("http://tauri.localhost"),
    Some("http://oaiy.localhost"),
    Some("oaiy://localhost"),
    Some("https://oaiy.com"),
    Some("https://x.oaiy.com"),
    Some("http://localhost:3000"),
    Some("http://127.0.0.1:5173"),
    Some("null"),
    Some("http://evil.example"),
    Some("http://formlogic.local"),
];

#[tokio::test(flavor = "multi_thread")]
async fn in_legacy_mode_every_route_that_existed_before_the_access_model_is_answered_as_the_old_guard_answered_it(
) {
    let (pairing, paired) = a_paired_token();
    let internal = crate::internal_token().to_string();
    assert!(!internal.is_empty(), "the process has an internal token");
    let credentials: Arc<Vec<Option<String>>> = Arc::new(vec![
        None,
        Some(format!("Bearer {STATIC_TOKEN}")),
        Some("Bearer wrong-token".into()),
        Some(format!("Bearer {internal}")),
        Some(format!("Bearer {paired}")),
        Some("Bearer oaiypat_wrong".into()),
        Some("Bearer oaiypat_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8".into()),
        Some("Bearer ".into()),
        Some(format!("bearer {STATIC_TOKEN}")),
        Some(format!("Bearer  {STATIC_TOKEN}")),
    ]);
    let requests = Arc::new(requests());
    assert!(
        requests.len() > 1400,
        "{} requests per combination",
        requests.len()
    );
    // The methods the table adds to a route that existed before are asked: the class of mistake that a table
    // row with `since: 2` on an old pattern makes (`DELETE /api/bridge/pairing` next to the old `GET`/`POST`).
    for r in ROUTES
        .iter()
        .filter(|r| r.since == 2 && crate::auth::routes::pattern_existed_before(r.pattern))
    {
        assert!(
            requests.contains(&(
                Method::from_bytes(r.method.as_str().as_bytes()).unwrap_or(Method::GET),
                concrete(r.pattern)
            )),
            "{} is not asked",
            r.key()
        );
    }

    let mut tasks = Vec::new();
    for gui in [true, false] {
        for token in [Some(STATIC_TOKEN), None] {
            let (new, old) = (
                live(gui, token, pairing.clone()),
                frozen(gui, token, pairing.clone()),
            );
            let (requests, credentials) = (requests.clone(), credentials.clone());
            tasks.push(tokio::spawn(async move {
                let mut compared = 0usize;
                let mut statuses = std::collections::BTreeMap::<u16, usize>::new();
                let mut differences = Vec::new();
                for (method, path) in requests.iter() {
                    for credential in credentials.iter() {
                        for origin in ORIGINS {
                            let a = ask(&new, method, path, credential.as_deref(), origin).await;
                            let b = ask(&old, method, path, credential.as_deref(), origin).await;
                            compared += 1;
                            *statuses.entry(b.status).or_default() += 1;
                            if a != b && differences.len() < 20 {
                                differences.push(format!("gui={gui} token={token:?} {method} {path} auth={credential:?} origin={origin:?}: live {} {:?} {:?} / old {} {:?} {:?}", a.status, a.headers, String::from_utf8_lossy(&a.body), b.status, b.headers, String::from_utf8_lossy(&b.body)));
                            }
                        }
                    }
                }
                (compared, statuses, differences)
            }));
        }
    }
    let mut compared = 0usize;
    let mut statuses = std::collections::BTreeMap::<u16, usize>::new();
    let mut differences = Vec::new();
    for task in tasks {
        let (c, s, d) = task.await.unwrap();
        compared += c;
        for (status, n) in s {
            *statuses.entry(status).or_default() += n;
        }
        differences.extend(d);
    }
    assert!(
        differences.is_empty(),
        "{} of {compared} answers differ; the first:\n{}",
        differences.len(),
        differences.join("\n")
    );
    // The matrix reached every family of answer: it is not comparing two refusals.
    assert!(compared > 600_000, "{compared} answers compared");
    assert!(
        statuses.get(&200).copied().unwrap_or(0) > 20_000,
        "{statuses:?}"
    );
    assert!(
        statuses.get(&403).copied().unwrap_or(0) > 20_000,
        "{statuses:?}"
    );
    assert!(statuses.contains_key(&404), "{statuses:?}");
    assert!(
        statuses.get(&405).copied().unwrap_or(0) > 20_000,
        "the methods a route does not have are asked: {statuses:?}"
    );
}

#[tokio::test]
async fn the_matrix_would_notice_a_guard_that_answered_differently() {
    // The differential test is only worth something if it fails when the live guard drifts. A guard that lets
    // `Origin: https://oaiy.com` reach a privileged route (as the old one did in a GUI, and refuses now) is
    // caught: here the two are given different modes on purpose.
    let (pairing, _) = a_paired_token();
    let gui = live(true, Some(STATIC_TOKEN), pairing.clone());
    let headless = frozen(false, Some(STATIC_TOKEN), pairing);
    let (method, path) = (Method::POST, "/api/services");
    let a = ask(&gui, &method, path, None, Some("tauri://localhost")).await;
    let b = ask(&headless, &method, path, None, Some("tauri://localhost")).await;
    assert_ne!(
        a, b,
        "a guard in another mode answers differently, and the comparison sees it"
    );
}

/// The routes of the receptionist's transfers and messages: `(method, pattern)`, each with a `since: 1` row in the table.
const RECEPTIONISTS_ROUTES: [(&str, &str); 11] = [
    ("GET", "/api/messages"),
    ("GET", "/api/messages/:id"),
    ("PATCH", "/api/messages/:id"),
    ("DELETE", "/api/messages/:id"),
    ("GET", "/api/ring/settings"),
    ("PUT", "/api/ring/settings"),
    ("GET", "/api/ring/preview"),
    ("GET", "/api/ring/active"),
    ("POST", "/api/ring/active/:id/respond"),
    ("POST", "/api/ring/notices/:id/dismiss"),
    ("POST", "/api/voice/calls/:id/message"),
];

/// The routes the receptionist's transfers and messages add were built before the access model merged, with the guard of their kind (a
/// read is a restricted read, a change is privileged: `is_personal_path`). They have `since: 1` rows, so `legacy` mode keeps that guard for
/// them and the differential compares it with the frozen one, which knows the same four lines: they are in the comparison (the stubs
/// answer each of them), a page that is not OAIY's window gets nothing from either guard, the configured token gets everything and the
/// desktop's own window is let in.
#[tokio::test]
async fn the_routes_of_the_receptionists_transfers_and_messages_are_in_the_comparison_and_are_held_as_personal_routes() {
    let (pairing, _) = a_paired_token();
    let asked = requests();
    let bare = stub_routes();
    for (method, pattern) in RECEPTIONISTS_ROUTES {
        let method = Method::from_bytes(method.as_bytes()).unwrap();
        let path = concrete(pattern);
        assert!(asked.contains(&(method.clone(), path.clone())), "{method} {path} is not asked");
        let stub = ask(&bare, &method, &path, None, None).await;
        assert_eq!(stub.status, 200, "{method} {pattern} is not among the routes compared (its row is not `since: 1`?): {stub:?}");
        for (name, guarded) in [("live", live(true, Some(STATIC_TOKEN), pairing.clone())), ("frozen", frozen(true, Some(STATIC_TOKEN), pairing.clone()))] {
            let stranger = ask(&guarded, &method, &path, None, Some("http://evil.example")).await;
            assert_eq!(stranger.status, 403, "{name}: {method} {path} from a page that is not the window");
            let window = ask(&guarded, &method, &path, None, Some("http://oaiy.localhost")).await;
            assert_eq!(window.status, 200, "{name}: {method} {path} from the desktop's own window");
            let token = ask(&guarded, &method, &path, Some(&format!("Bearer {STATIC_TOKEN}")), None).await;
            assert_eq!(token.status, 200, "{name}: {method} {path} with the configured token");
        }
    }
}

#[tokio::test]
async fn the_old_guards_bodies_and_statuses_are_still_what_they_were() {
    // A few of the old answers pinned by value, so that a change to both guards cannot pass unseen.
    let (pairing, _) = a_paired_token();
    let headless = live(false, Some(STATIC_TOKEN), pairing.clone());
    let a = ask(&headless, &Method::GET, "/api/config", None, None).await;
    assert_eq!(
        (a.status, String::from_utf8_lossy(&a.body).into_owned()),
        (403, r#"{"error":"authentication required"}"#.to_string())
    );
    let a = ask(&headless, &Method::GET, "/api/health", None, None).await;
    assert_eq!((a.status, a.body.as_slice()), (200, b"ok".as_slice()));
    let a = ask(
        &headless,
        &Method::GET,
        "/api/config",
        Some("Bearer desk-token"),
        None,
    )
    .await;
    assert_eq!(a.status, 200);
    let gui = live(true, None, pairing);
    let a = ask(
        &gui,
        &Method::POST,
        "/api/services",
        None,
        Some("https://evil.example"),
    )
    .await;
    assert_eq!(
        (a.status, String::from_utf8_lossy(&a.body).into_owned()),
        (403, r#"{"error":"origin not allowed"}"#.to_string())
    );
    let a = ask(
        &gui,
        &Method::POST,
        "/api/services",
        None,
        Some("tauri://localhost"),
    )
    .await;
    assert_eq!(
        a.status, 200,
        "the desktop's own webview passes with no credential, as it always did"
    );
    let a = ask(&gui, &Method::GET, "/api/services", None, None).await;
    assert_eq!(
        a.status, 200,
        "an unrestricted read needs nothing on the desktop"
    );
    let a = ask(
        &gui,
        &Method::GET,
        "/api/config",
        None,
        Some("http://localhost:3000"),
    )
    .await;
    assert_eq!(
        a.status, 200,
        "and a loopback page reads what the old guard let it read"
    );
}

#[tokio::test]
async fn a_handler_that_reads_the_principal_gets_the_legacy_owner_in_legacy_mode() {
    use crate::auth::principal::{Principal, PrincipalKind};
    use axum::extract::Extension;
    let (pairing, _) = a_paired_token();
    let unused = std::env::temp_dir().join("oaiy-legacy-neutrality-nothing-is-made-here");
    let guard = crate::auth::build_guard(
        &AccessSettings::legacy(),
        &unused,
        17972,
        false,
        false,
        Some(STATIC_TOKEN.into()),
        &|_| None,
    )
    .unwrap();
    let state = AccessState {
        legacy: AuthConfig {
            token: Some(STATIC_TOKEN.into()),
            gui_mode: false,
            pairing: Some(pairing),
        },
        guard,
    };
    let app = Router::new()
        .route(
            "/api/config",
            axum::routing::get(|p: Option<Extension<Principal>>| async move {
                p.map(|Extension(p)| format!("{:?}:{}", p.kind, p.scopes.len()))
                    .unwrap_or_else(|| "none".into())
            }),
        )
        .layer(middleware::from_fn_with_state(state, access_guard));
    let a = ask(
        &app,
        &Method::GET,
        "/api/config",
        Some("Bearer desk-token"),
        None,
    )
    .await;
    assert_eq!(
        String::from_utf8_lossy(&a.body),
        // (The legacy owner holds every scope there is: 55 now, and never a number written here.)
        format!("{:?}:{}", PrincipalKind::Legacy, crate::auth::scopes::SCOPES.len())
    );
    // Refused requests never reach the handler, so nothing is handed to a caller who did not pass.
    let a = ask(&app, &Method::GET, "/api/config", Some("Bearer nope"), None).await;
    assert_eq!(a.status, 403);
}

#[tokio::test]
async fn health_gains_access_and_storage_only_with_the_guard_in_front_and_nothing_else_changes() {
    let (pairing, _) = a_paired_token();
    let unused = std::env::temp_dir().join("oaiy-legacy-neutrality-nothing-is-made-here");
    let guard = crate::auth::build_guard(
        &AccessSettings::legacy(),
        &unused,
        17972,
        false,
        true,
        None,
        &|_| None,
    )
    .unwrap();
    let state = AccessState {
        legacy: AuthConfig {
            token: None,
            gui_mode: true,
            pairing: Some(pairing),
        },
        guard,
    };
    let with = Router::new()
        .route("/api/health", axum::routing::get(super::health))
        .layer(middleware::from_fn_with_state(state, access_guard));
    let without = Router::new().route("/api/health", axum::routing::get(super::health));
    let a: serde_json::Value = serde_json::from_slice(
        &ask(&with, &Method::GET, "/api/health", None, None)
            .await
            .body,
    )
    .unwrap();
    let b: serde_json::Value = serde_json::from_slice(
        &ask(&without, &Method::GET, "/api/health", None, None)
            .await
            .body,
    )
    .unwrap();
    // Without the guard: exactly what health always said.
    let names = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        names(&b),
        [
            "apiVersion",
            "companion",
            "pluginApiVersion",
            "product",
            "protocol",
            "status",
            "version"
        ]
    );
    // With it: the same fields, and two more.
    assert_eq!(a["access"], "legacy");
    assert_eq!(a["storage"], "ok");
    let mut without_new = a.clone();
    without_new.as_object_mut().unwrap().remove("access");
    without_new.as_object_mut().unwrap().remove("storage");
    assert_eq!(
        without_new, b,
        "every field health had is unchanged, and the version is still 1: {a}"
    );
    assert_eq!(
        a["apiVersion"], 1,
        "apiVersion 2 announces pairing v2 and cookie login, which do not exist yet"
    );
    // The bytes keep their order: the fixed fields first, in the order they were declared.
    let text = String::from_utf8(
        ask(&without, &Method::GET, "/api/health", None, None)
            .await
            .body,
    )
    .unwrap();
    assert!(text.starts_with(r#"{"status":"ok","product":"oaiy-desktop","companion":"oaiy-desktop","protocol":"oaiy-bridge/1","version":"#), "{text}");
}

#[tokio::test]
async fn a_route_the_model_adds_is_judged_by_the_new_guard_even_in_legacy_mode() {
    use crate::auth::api;
    let (pairing, paired) = a_paired_token();
    let unused = std::env::temp_dir().join("oaiy-legacy-neutrality-nothing-is-made-here");
    for gui in [true, false] {
        let guard = crate::auth::build_guard(
            &AccessSettings::legacy(),
            &unused,
            17972,
            false,
            gui,
            Some(STATIC_TOKEN.into()),
            &|_| None,
        )
        .unwrap();
        let state = AccessState {
            legacy: AuthConfig {
                token: Some(STATIC_TOKEN.into()),
                gui_mode: gui,
                pairing: Some(pairing.clone()),
            },
            guard: guard.clone(),
        };
        let app = stub_routes()
            .merge(api::router(guard))
            .layer(middleware::from_fn_with_state(state, access_guard));
        // No credential: 401, not the old guard's pass (a GUI read with no Origin passes there) and not its 403.
        for (m, p) in [
            (Method::GET, "/api/auth/whoami"),
            (Method::POST, "/api/auth/derive"),
        ] {
            for origin in ORIGINS {
                let a = ask(&app, &m, p, None, origin).await;
                assert_eq!(
                    a.status,
                    401,
                    "gui={gui} {m} {p} origin={origin:?}: {}",
                    String::from_utf8_lossy(&a.body)
                );
            }
        }
        // Credentials that are not credentials here fail closed: the paired token, the internal token, a guess.
        for bad in [
            format!("Bearer {paired}"),
            format!("Bearer {}", crate::internal_token()),
            "Bearer nonsense".to_string(),
        ] {
            let a = ask(&app, &Method::GET, "/api/auth/whoami", Some(&bad), None).await;
            assert_eq!(
                a.status,
                401,
                "gui={gui}: {}",
                String::from_utf8_lossy(&a.body)
            );
            assert!(String::from_utf8_lossy(&a.body).contains("token_invalid"));
        }
        // The configured token authenticates as the cli preset: never more.
        let a = ask(
            &app,
            &Method::GET,
            "/api/auth/whoami",
            Some("Bearer desk-token"),
            None,
        )
        .await;
        assert_eq!(a.status, 200, "gui={gui}");
        let body: serde_json::Value = serde_json::from_slice(&a.body).unwrap();
        assert_eq!(
            (
                body["kind"].as_str(),
                body["scopes"].as_array().map(Vec::len)
            ),
            (Some("static"), Some(15))
        );
        // The public one is public.
        assert_eq!(
            ask(&app, &Method::GET, "/api/auth/info", None, None)
                .await
                .status,
            200
        );
    }
}
