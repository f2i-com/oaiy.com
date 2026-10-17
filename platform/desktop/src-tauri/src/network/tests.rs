//! Two OAIY computers in one process: a worker and a primary paired over loopback, the primary using the worker's
//! engines (a fake engine host here: its discovery, a picture, a file, a streamed chat and its resources).

use std::path::PathBuf;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use super::store::{Role, Settings};
use super::tls::{self, Pin};
use super::*;

const PNG: &[u8] = b"\x89PNG\r\n\x1a\nnot really";

/// A fake engine host (the Studio): its control port's state and resources, and its gateway under `/gw`.
async fn fake_engines() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let gw = format!("{base}/gw");
    let file = format!("{gw}/files/images/2026-10-18/x/image-0001.png");
    let app = Router::new()
        .route("/api/state", get({
            let gw = gw.clone();
            move || async move { axum::Json(json!({ "gateway_url": gw })) }
        }))
        .route("/api/resources", get(|| async {
            axum::Json(json!({
                "devices": [{"id": "egpu", "name": "NV card (eGPU)", "unified": false, "memory_total_mb": 24564, "memory_used_mb": 9000}],
                "models": [{"model": "tiny-image", "kind": "image", "device": "egpu", "state": "kept", "pinned": false}],
                "queue": {"running": null, "queued": 0},
                "events": [{"event": "load", "model": "tiny-image", "why": "a picture"}],
            }))
        }))
        // (pins: what reached the engines echoed back, and one they cannot keep refused with their reason)
        .route("/api/resources/pins", post(|body: String| async move { pinned(&body, true) }))
        .route("/api/resources/unpin", post(|body: String| async move { pinned(&body, false) }))
        .route("/gw/v1/discovery", get(|| async {
            axum::Json(json!({
                "base_url": "http://127.0.0.1:8080",
                "endpoints": [
                    {"name": "chat", "path": "/v1/chat/completions", "spec": "openai", "url": "http://127.0.0.1:8080/v1/chat/completions"},
                    {"name": "images", "path": "/v1/images/generations", "spec": "openai"},
                    {"name": "files", "path": "/files", "spec": "openai"},
                ],
                "models": {
                    "llm": [{"id": "tiny-llm", "default": true, "files_present": true}],
                    "image": [{"id": "tiny-image", "default": true, "files_present": true}, {"id": "gone", "files_present": false, "missing_files": ["/Users/someone/secret/model.gguf"]}],
                },
                "defaults": {"llm": "tiny-llm", "image": "tiny-image"},
            }))
        }))
        .route("/gw/v1/images/generations", post({
            let file = file.clone();
            move |body: String| async move {
                let asked: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
                axum::Json(json!({ "data": [{ "b64_json": "aGk=", "oaiy": { "file": file, "seed": 7, "model": asked["model"] } }] }))
            }
        }))
        .route("/gw/files/images/2026-10-18/x/image-0001.png", get(|| async { ([("content-type", "image/png")], PNG) }))
        .route("/gw/v1/chat/completions", post(|body: String| async move {
            let asked: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
            assert!(asked.get("model").is_none(), "the worker sends its engine no model: it answers with its chosen one");
            let chunks = ["Hel", "lo"].iter().map(|t| format!("data: {}\n\n", json!({"choices": [{"delta": {"content": t}}]}))).collect::<String>() + "data: [DONE]\n\n";
            ([("content-type", "text/event-stream")], chunks)
        }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    base
}

/// The fake engines' answer to a pin or an unpin: refused for `too-big`, else the resources with what they were asked.
fn pinned(body: &str, pin: bool) -> axum::response::Response {
    use axum::response::IntoResponse;
    let asked: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if asked["model"] == "too-big" {
        return (StatusCode::CONFLICT, axum::Json(json!({ "error": "too-big is not kept beside tiny-image, which is pinned: unpin it first" }))).into_response();
    }
    axum::Json(json!({
        "models": [{"model": "tiny-image", "kind": "image", "pinned": pin}],
        "pins": if pin { json!([{"model": asked["model"], "state": "warming"}]) } else { json!([]) },
        "asked": { "model": asked["model"], "kind": asked["kind"], "pin": pin },
        "events": [{"event": "pin"}],
    }))
    .into_response()
}

fn temp(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oaiy-network-{tag}-{}-{}", std::process::id(), uuid::Uuid::new_v4().simple()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn ai(engines: Option<String>) -> crate::ai::AiState {
    let dir = temp("ai");
    let registry = std::sync::Arc::new(std::sync::Mutex::new(crate::services::registry::Registry::empty(dir.join("data"), dir.join("models"))));
    crate::ai::AiState::new(crate::ai::providers::new_handle(), registry, crate::ai::codex::absent_for_tests()).with_engines_at(engines)
}

async fn node(tag: &str, role: Role, engines: Option<String>) -> Network {
    let net = Network::open_with(&temp(tag), ai(engines), false);
    net.set(Settings { enabled: true, role, name: format!("the {tag}"), port: 0, bind: "127.0.0.1".into() }).await.unwrap();
    net
}

async fn call(app: &Router, method: &str, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(path);
    if body.is_some() {
        req = req.header("content-type", "application/json");
    }
    let resp = app.clone().oneshot(req.body(body.map_or_else(Body::empty, |b| Body::from(b.to_string()))).unwrap()).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 24).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap_or_else(|_| json!(String::from_utf8_lossy(&bytes))))
}

/// Pair `primary` with `worker`, the worker's person approving.
async fn pair(primary: &Network, worker: &Network) -> String {
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let address = worker.listening().unwrap().to_string();
    let (status, started) = call(&p, "POST", "/api/network/pairings/start", Some(json!({ "address": address }))).await;
    assert_eq!(status, StatusCode::OK, "{started}");
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    let request = &waiting["requests"][0];
    assert_eq!(request["code"], started["code"], "both screens show the same code");
    assert!(request["name"].as_str().unwrap().starts_with("the primary"), "{request}");
    let pairing = json!({ "pairingId": started["pairingId"] });
    let (_, before) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing.clone())).await;
    assert_eq!(before["status"], "pending", "nothing is paired before the worker's person approves");
    let (status, approved) = call(&w, "POST", &format!("/api/network/requests/{}/approve", request["id"].as_str().unwrap()), None).await;
    assert_eq!((status, approved["status"].as_str()), (StatusCode::OK, Some("approved")));
    let (_, there) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing.clone())).await;
    assert_eq!((there["status"].as_str(), &there["code"]), (Some("approved_there"), &started["code"]), "{there}");
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert!(nodes["nodes"].as_array().unwrap().iter().all(|n| n["nodeId"] != started["worker"]["nodeId"]), "not paired here before this side says the codes matched");
    let (_, confirmed) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": true }))).await;
    assert_eq!(confirmed["status"], "approved", "{confirmed}");
    confirmed["node"]["nodeId"].as_str().unwrap().to_string()
}

/// Approved on the worker, the pairing waits for this side's person to say the codes matched; saying they did not,
/// nothing is paired here and the worker forgets this computer.
#[tokio::test]
async fn a_pairing_needs_both_screens_and_a_no_here_undoes_the_workers_yes() {
    let worker = node("worker-both", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-both", Role::Primary, None).await;
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let (_, started) = call(&p, "POST", "/api/network/pairings/start", Some(json!({ "address": worker.listening().unwrap().to_string() }))).await;
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    call(&w, "POST", &format!("/api/network/requests/{}/approve", waiting["requests"][0]["id"].as_str().unwrap()), None).await;
    let (_, worker_side) = call(&w, "GET", "/api/network/nodes", None).await;
    assert_eq!(worker_side["nodes"].as_array().unwrap().len(), 1, "the worker's yes");
    let pairing = json!({ "pairingId": started["pairingId"] });
    let (_, there) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing.clone())).await;
    assert_eq!(there["status"], "approved_there");
    let (_, again) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing)).await;
    assert_eq!(again["status"], "approved_there", "asked again, it still waits for this side");
    let (_, no) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": false }))).await;
    assert_eq!(no["status"], "declined");
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert!(nodes["nodes"].as_array().unwrap().is_empty(), "nothing paired here");
    let (_, worker_side) = call(&w, "GET", "/api/network/nodes", None).await;
    assert!(worker_side["nodes"].as_array().unwrap().is_empty(), "the worker forgot this computer: {worker_side}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

#[tokio::test]
async fn a_primary_pairs_with_a_worker_and_uses_its_engines_through_its_own_routes() {
    let engines = fake_engines().await;
    let worker = node("worker", Role::Worker, Some(engines)).await;
    let primary = node("primary", Role::Primary, None).await;
    let p = router(primary.clone());
    let worker_id = pair(&primary, &worker).await;
    assert_eq!(worker_id, worker.identity().unwrap().node_id);

    // the worker as the primary sees it: online, its GPU, its models (the one whose files are gone left out)
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    let n = &nodes["nodes"][0];
    assert_eq!((n["online"].as_bool(), n["role"].as_str(), n["name"].as_str()), (Some(true), Some("worker"), Some("the worker")));
    assert_eq!(n["devices"][0]["name"], "NV card (eGPU)");
    let models: Vec<&str> = n["models"].as_array().unwrap().iter().map(|m| m["model"].as_str().unwrap()).collect();
    assert_eq!(models, ["tiny-llm", "tiny-image"]);
    assert_eq!(n["models"][1]["state"], "kept");
    assert!(!nodes.to_string().contains("/Users/someone"), "no private path crosses the network");

    // its models in the catalogue: node-scoped ids, never a default, their calls through this desktop
    let services = primary::remote_services_of(&primary);
    let image = services.iter().find(|s| s["kind"] == "image").unwrap();
    assert_eq!(image["id"], format!("engine:image:node.{worker_id}/tiny-image"));
    assert_eq!(image["default"], false);
    assert_eq!(image["ready"], true);
    assert_eq!(image["model"], "tiny-image", "the worker's own name for it is what its engine is sent");
    assert_eq!(image["endpoint"], format!("/api/network/nodes/{worker_id}/gateway/v1/images/generations"));
    assert_eq!(image["filesUrl"], format!("/api/network/nodes/{worker_id}/gateway/files"));
    assert_eq!(image["name"], "OAIY engine · Image · tiny-image · the worker");
    let llm = services.iter().find(|s| s["kind"] == "llm").unwrap();
    assert_eq!(llm["endpoint"], format!("/api/network/nodes/{worker_id}/v1/chat/completions"));
    assert_eq!(llm["id"], format!("engine:llm:node.{worker_id}/tiny-llm"));

    // a picture made on the worker, through the primary
    let (status, made) = call(&p, "POST", image["endpoint"].as_str().unwrap(), Some(json!({ "model": "tiny-image", "prompt": "a fox" }))).await;
    assert_eq!(status, StatusCode::OK, "{made}");
    assert_eq!(made["data"][0]["oaiy"]["model"], "tiny-image");
    // its file, through the primary too
    let resp = p.clone().oneshot(Request::get(format!("/api/network/nodes/{worker_id}/gateway/files/images/2026-10-18/x/image-0001.png")).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()["content-type"], "image/png");
    assert_eq!(&axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap()[..], PNG);
    // a chat, streamed
    let resp = p
        .clone()
        .oneshot(Request::post(llm["endpoint"].as_str().unwrap()).header("content-type", "application/json").body(Body::from(json!({ "model": "x", "stream": true, "messages": [{"role": "user", "content": "hi"}] }).to_string())).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let text = String::from_utf8(axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
    assert!(text.contains("Hel") && text.contains("[DONE]"), "{text}");
    // only the engine's media routes are reached: not its control port, not a path outside them
    let (status, _) = call(&p, "GET", &format!("/api/network/nodes/{worker_id}/gateway/api/config"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    worker.stop();
    primary.stop();
}

#[tokio::test]
async fn a_stranger_reaches_only_health_and_an_impostor_is_not_trusted() {
    let worker = node("worker2", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary2", Role::Primary, None).await;
    let address = worker.listening().unwrap();
    let stranger = super::identity::Identity::generate().unwrap();
    let anyone = tls::client(&stranger, Pin::Record(Default::default())).unwrap();
    let health: Value = anyone.get(format!("https://{address}/worker/v1/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!((health["service"].as_str(), health["paired"].as_bool()), (Some("oaiy-worker"), Some(false)));
    for path in ["/worker/v1/capabilities", "/worker/v1/resources"] {
        assert_eq!(anyone.get(format!("https://{address}{path}")).send().await.unwrap().status(), reqwest::StatusCode::FORBIDDEN, "{path}");
    }
    let r = anyone.post(format!("https://{address}/worker/v1/gateway/v1/images/generations")).body("{}").send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);
    let r = anyone.post(format!("https://{address}/worker/v1/chat")).body("{}").send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);

    let worker_id = pair(&primary, &worker).await;
    // the stranger is still a stranger after another computer paired
    assert_eq!(anyone.get(format!("https://{address}/worker/v1/capabilities")).send().await.unwrap().status(), reqwest::StatusCode::FORBIDDEN);
    // a computer at the worker's address with another certificate is not the worker: the pinned client refuses it
    let impostor = node("impostor", Role::Worker, None).await;
    let pinned = tls::client(&primary.identity().unwrap(), Pin::Fingerprint(worker.identity().unwrap().fingerprint)).unwrap();
    assert!(pinned.get(format!("https://{}/worker/v1/health", impostor.listening().unwrap())).send().await.is_err());
    assert!(pinned.get(format!("https://{address}/worker/v1/health")).send().await.is_ok());

    // unpaired at the worker: the primary is refused there and shows the worker offline, saying why
    let w = router(worker.clone());
    let primary_id = primary.identity().unwrap().node_id;
    let resp = w.clone().oneshot(Request::delete(format!("/api/network/nodes/{primary_id}")).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    let peer = primary.inner().peers().get(&worker_id).cloned().unwrap();
    primary::check(&primary, &peer).await;
    let (_, nodes) = call(&router(primary.clone()), "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["online"], false);
    assert!(nodes["nodes"][0]["lastError"].as_str().unwrap().contains("no longer accepts"), "{nodes}");
    // a request for its model now says plainly which worker did not answer
    let (status, body) = call(&router(primary.clone()), "POST", &format!("/api/network/nodes/{worker_id}/gateway/v1/images/generations"), Some(json!({}))).await;
    assert_eq!(status, StatusCode::FORBIDDEN, "the worker's own refusal is passed on: {body}");

    for n in [&worker, &primary, &impostor] {
        n.stop();
    }
}

/// A worker's model pinned from the primary, by its id here or its name there: the worker's engines asked to keep it;
/// one they cannot keep refused with their reason; unpinned again; never a path, another worker's model, or a stranger's.
#[tokio::test]
async fn a_primary_pins_a_workers_model_and_unpins_it() {
    let worker = node("worker-pins", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-pins", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let pins = format!("/api/network/nodes/{node_id}/pins");
    let (status, v) = call(&p, "PUT", &pins, Some(json!({ "id": format!("engine:image:node.{node_id}/tiny-image") }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["asked"], json!({ "model": "tiny-image", "kind": "image", "pin": true }));
    assert_eq!(v["pins"][0]["state"], "warming");
    assert!(v.get("events").is_none(), "the worker's event log stays there");
    let (status, v) = call(&p, "PUT", &pins, Some(json!({ "model": "too-big" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{v}");
    assert!(v.to_string().contains("not kept beside tiny-image"), "the engines' reason: {v}");
    let (status, v) = call(&p, "DELETE", &format!("{pins}/tiny-image"), None).await;
    assert_eq!((status, &v["asked"]), (StatusCode::OK, &json!({ "model": "tiny-image", "kind": "image", "pin": false })), "{v}");
    for bad in [json!({ "model": "../../etc/passwd" }), json!({ "model": "/Users/x/model.gguf" }), json!({ "id": "engine:image:node.someoneelse/tiny-image" })] {
        let (status, v) = call(&p, "PUT", &pins, Some(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {v}");
    }
    let (status, _) = call(&p, "PUT", "/api/network/nodes/nobody/pins", Some(json!({ "model": "tiny-image" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // a client of the worker's media routes (ai.use) cannot step past them to its pins or unpairing
    for path in ["gateway/../pins", "gateway/%2e%2e/unpair", "gateway/v1/images/generations/../../../unpair"] {
        let (status, v) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/{path}"), Some(json!({ "model": "tiny-image" }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {v}");
    }
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"].as_array().unwrap().len(), 1, "still paired");
    // a stranger cannot pin on the worker
    let stranger = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let r = stranger.post(format!("https://{}/worker/v1/pins", worker.listening().unwrap())).json(&json!({ "model": "tiny-image" })).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);
    for n in [&worker, &primary] {
        n.stop();
    }
}

#[tokio::test]
async fn unpairing_at_the_primary_tells_the_worker_and_forgets_its_models() {
    let worker = node("worker3", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary3", Role::Primary, None).await;
    let worker_id = pair(&primary, &worker).await;
    assert!(!primary::remote_services_of(&primary).is_empty());
    let p = router(primary.clone());
    let resp = p.clone().oneshot(Request::delete(format!("/api/network/nodes/{worker_id}")).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    assert!(primary::remote_services_of(&primary).is_empty());
    assert!(worker.inner().peers().list().is_empty(), "the worker was told, and no longer accepts the primary");
    let (status, body) = call(&p, "POST", &format!("/api/network/nodes/{worker_id}/v1/chat/completions"), Some(json!({}))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (StatusCode::NOT_FOUND, Some("node_unknown")));
    worker.stop();
    primary.stop();
}

#[tokio::test]
async fn off_nothing_listens_and_a_worker_is_never_reached_off_the_lan() {
    let net = Network::open_with(&temp("off"), ai(None), false);
    net.apply().await.unwrap();
    assert!(net.listening().is_none());
    assert!(primary::remote_services_of(&net).is_empty());
    let (_, status) = call(&router(net.clone()), "GET", "/api/network/status", None).await;
    assert_eq!((status["enabled"].as_bool(), status["role"].as_str()), (Some(false), Some("standalone")));
    assert!(status["nodeId"].is_null(), "no identity is made while Network is off");

    let primary = node("primary4", Role::Primary, None).await;
    let (status, body) = call(&router(primary.clone()), "POST", "/api/network/pairings/start", Some(json!({ "address": "8.8.8.8:17975" }))).await;
    assert_eq!((status, body["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("invalid_address")), "{body}");
    // a worker binding a public address is refused too
    let worker = Network::open_with(&temp("bind"), ai(None), false);
    let err = worker.set(Settings { enabled: true, role: Role::Worker, name: String::new(), port: 0, bind: "8.8.8.8".into() }).await.unwrap_err();
    assert!(err.contains("not a local network address"), "{err}");
    primary.stop();
}

/// What a worker shows to be typed on the primary: its LAN addresses each once and IPv4 first, without loopback, the
/// IPv6 link-local addresses every interface has, or a public address.
#[test]
fn the_addresses_shown_are_the_ones_a_person_can_type() {
    let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
    let got = typeable(vec![
        ip("fe80::1"),
        ip("fd85:2144:420d::7ebf"),
        ip("192.168.2.147"),
        ip("127.0.0.1"),
        ip("fe80::bcc6:4eff:febc:20b2"),
        ip("192.168.2.147"),
        ip("8.8.8.8"),
        ip("10.0.0.4"),
    ]);
    assert_eq!(got, vec![ip("192.168.2.147"), ip("10.0.0.4"), ip("fd85:2144:420d::7ebf")]);
}
