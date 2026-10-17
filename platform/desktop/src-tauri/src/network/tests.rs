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
                "policy": {"llm_pinned": true},
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

/// Given up here before the worker's person answered, a pairing is withdrawn there (they are no longer asked); one
/// that ran out after the worker's yes, with no answer here, is undone there (it was paired on that side alone).
#[tokio::test]
async fn a_pairing_given_up_or_run_out_here_is_undone_on_the_worker() {
    let worker = node("worker-lapse", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-lapse", Role::Primary, None).await;
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let address = json!({ "address": worker.listening().unwrap().to_string() });
    let (_, started) = call(&p, "POST", "/api/network/pairings/start", Some(address.clone())).await;
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    assert_eq!(waiting["requests"].as_array().unwrap().len(), 1);
    let (_, stopped) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": false }))).await;
    assert_eq!(stopped["status"], "declined");
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    assert!(waiting["requests"].as_array().unwrap().is_empty(), "withdrawn there: {waiting}");

    let (_, started) = call(&p, "POST", "/api/network/pairings/start", Some(address)).await;
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    call(&w, "POST", &format!("/api/network/requests/{}/approve", waiting["requests"][0]["id"].as_str().unwrap()), None).await;
    let (_, there) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"] }))).await;
    assert_eq!(there["status"], "approved_there");
    let (_, worker_side) = call(&w, "GET", "/api/network/nodes", None).await;
    assert_eq!(worker_side["nodes"].as_array().unwrap().len(), 1, "paired on the worker's side alone");
    for o in lock(&primary.inner().outgoing).iter_mut() {
        o.at = std::time::Instant::now().checked_sub(std::time::Duration::from_secs(6 * 60)).unwrap();
    }
    primary::forget_lapsed(&primary).await;
    let (_, worker_side) = call(&w, "GET", "/api/network/nodes", None).await;
    assert!(worker_side["nodes"].as_array().unwrap().is_empty(), "undone there: {worker_side}");
    let (late, _) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": true }))).await;
    assert_eq!(late, StatusCode::NOT_FOUND, "too late to say yes here");

    // paired, then asked again and given up: the pairing there stands
    pair(&primary, &worker).await;
    let (_, again) = call(&p, "POST", "/api/network/pairings/start", Some(json!({ "address": worker.listening().unwrap().to_string() }))).await;
    call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": again["pairingId"], "accept": false }))).await;
    let (_, worker_side) = call(&w, "GET", "/api/network/nodes", None).await;
    assert_eq!(worker_side["nodes"].as_array().unwrap().len(), 1, "still paired there: {worker_side}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// The code is made of both sides' random values, the worker's committed to first: the worker asks its person only once
/// the primary has told its own, tells its value back once, held to its commitment, and both screens show one code.
#[tokio::test]
async fn the_pairing_code_comes_from_a_value_committed_to_first() {
    use super::identity::{commitment, nonce, pairing_code, Identity};
    let worker = node("worker-commit", Role::Worker, None).await;
    let w = router(worker.clone());
    let me = Identity::generate().unwrap();
    let asking = tls::client(&me, Pin::Record(Default::default())).unwrap();
    let at = |path: &str| format!("https://{}{path}", worker.listening().unwrap());
    let asked: Value = asking.post(at("/worker/v1/pair")).json(&json!({ "nodeId": me.node_id, "name": "a primary" })).send().await.unwrap().json().await.unwrap();
    let committed = asked["commitment"].as_str().unwrap().to_string();
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    assert!(waiting["requests"].as_array().unwrap().is_empty(), "not asked before the primary's value is told: {waiting}");
    let reveal = at(&format!("/worker/v1/pair/{}", asked["requestId"].as_str().unwrap()));
    let r = asking.post(&reveal).json(&json!({ "nonce": "not hex" })).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    let mine = nonce().unwrap();
    let told: Value = asking.post(&reveal).json(&json!({ "nonce": mine })).send().await.unwrap().json().await.unwrap();
    let theirs = told["nonce"].as_str().unwrap();
    let worker_fp = worker.identity().unwrap().fingerprint;
    assert_eq!(commitment(&me.fingerprint, &worker_fp, theirs), committed, "the value told back is the one committed to");
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    assert_eq!(waiting["requests"][0]["code"], pairing_code(&me.fingerprint, &worker_fp, &mine, theirs), "{waiting}");
    let again = asking.post(&reveal).json(&json!({ "nonce": nonce().unwrap() })).send().await.unwrap();
    assert_eq!(again.status(), reqwest::StatusCode::CONFLICT, "told once: no second try at a code of the caller's choosing");
    // (another certificate cannot tell it for the one that asked)
    let other = tls::client(&Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let r = other.post(&reveal).json(&json!({ "nonce": nonce().unwrap() })).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::NOT_FOUND);
    worker.stop();
}

/// A worker unpaired at the primary while it was off is told once it answers again, and stops accepting the primary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_unpaired_while_off_is_told_when_it_is_back() {
    let worker = node("worker-untold", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-untold", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let port = worker.listening().unwrap().port();
    worker.stop();
    let (status, _) = call(&p, "DELETE", &format!("/api/network/nodes/{node_id}"), None).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    let (_, still) = call(&w, "GET", "/api/network/nodes", None).await;
    assert_eq!(still["nodes"].as_array().unwrap().len(), 1, "it did not hear, being off");
    worker.set(Settings { enabled: true, role: Role::Worker, name: "the worker-untold".into(), port, bind: "127.0.0.1".into() }).await.unwrap();
    primary::tell_untold(&primary).await;
    let (_, told) = call(&w, "GET", "/api/network/nodes", None).await;
    assert!(told["nodes"].as_array().unwrap().is_empty(), "told once back: {told}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A worker saved again while it listens (renamed) keeps its port: the listener before it has let the port go first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_saved_again_listens_on_the_same_port() {
    let worker = node("worker-again", Role::Worker, None).await;
    let port = worker.listening().unwrap().port();
    for name in ["the worker renamed", "and again"] {
        worker.set(Settings { enabled: true, role: Role::Worker, name: name.into(), port, bind: "127.0.0.1".into() }).await.unwrap();
        assert_eq!(worker.listening().map(|a| a.port()), Some(port));
    }
    let stranger = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let health: Value = stranger.get(format!("https://127.0.0.1:{port}/worker/v1/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(health["name"], "and again");
    worker.stop();
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
    let resp = p
        .clone()
        .oneshot(Request::post(image["endpoint"].as_str().unwrap()).header("host", "127.0.0.1:17972").header("content-type", "application/json").body(Body::from(json!({ "model": "tiny-image", "prompt": "a fox" }).to_string())).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let made: Value = serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(made["data"][0]["oaiy"]["model"], "tiny-image");
    // (its file named where the caller reaches it: this desktop's route for the worker, not the worker's own loopback)
    assert_eq!(made["data"][0]["oaiy"]["file"], format!("http://127.0.0.1:17972/api/network/nodes/{worker_id}/gateway/files/images/2026-10-18/x/image-0001.png"), "{made}");
    // its file, through the primary too
    let resp = p.clone().oneshot(Request::get(format!("/api/network/nodes/{worker_id}/gateway/files/images/2026-10-18/x/image-0001.png")).body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()["content-type"], "image/png");
    assert_eq!(&axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap()[..], PNG);
    // a chat, streamed
    let resp = p
        .clone()
        .oneshot(Request::post(llm["endpoint"].as_str().unwrap()).header("content-type", "application/json").body(Body::from(json!({ "model": "tiny-llm", "stream": true, "messages": [{"role": "user", "content": "hi"}] }).to_string())).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.headers()["content-type"], "text/event-stream");
    let text = String::from_utf8(axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
    assert!(text.contains("Hel") && text.contains("[DONE]"), "{text}");
    // (a model the worker does not run now is refused, not answered by the one it does)
    let (status, v) = call(&p, "POST", llm["endpoint"].as_str().unwrap(), Some(json!({ "model": "another-llm", "messages": [{"role": "user", "content": "hi"}] }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{v}");
    assert!(v.to_string().contains("runs tiny-llm now"), "{v}");
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
    // (the language model pinned there, not running: shown pinned, and unpinned as the language model, not a picture)
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    let llm = nodes["nodes"][0]["models"].as_array().unwrap().iter().find(|m| m["model"] == "tiny-llm").cloned().unwrap();
    assert_eq!((&llm["state"], &llm["pinned"]), (&json!("not loaded"), &json!(true)), "{llm}");
    let (status, v) = call(&p, "DELETE", &format!("{pins}/tiny-llm"), None).await;
    assert_eq!((status, &v["asked"]), (StatusCode::OK, &json!({ "model": "tiny-llm", "kind": "llm", "pin": false })), "{v}");
    for bad in [json!({ "model": "../../etc/passwd" }), json!({ "model": "/Users/x/model.gguf" }), json!({ "id": "engine:image:node.someoneelse/tiny-image" })] {
        let (status, v) = call(&p, "PUT", &pins, Some(bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {v}");
    }
    let (status, _) = call(&p, "PUT", "/api/network/nodes/nobody/pins", Some(json!({ "model": "tiny-image" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // a client of the worker's media routes (ai.use) cannot step past them to its pins or unpairing
    // (nor with a tab or a newline the URL parser drops, `.%0A.` being `..` once parsed, nor a slash still encoded)
    for path in [
        "gateway/../pins",
        "gateway/%2e%2e/unpair",
        "gateway/v1/images/generations/../../../unpair",
        "gateway/v1/images/.%0A./.%0A./.%0A./pins",
        "gateway/v1/images/.%09./.%09./.%09./unpair",
        "gateway/v1/images/.%0D./.%0D./.%0D./unpair",
        "gateway/v1/images/a%252f..%252f..%252f..%252fpins",
    ] {
        let (status, v) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/{path}"), Some(json!({ "model": "tiny-image" }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {v}");
    }
    // (encoded twice, it passes here as it is a plain segment, and the worker, decoding it again, refuses it)
    let (status, v) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/gateway/v1/images/.%250A./.%250A./.%250A./pins"), Some(json!({ "model": "tiny-image" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{v}");
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
