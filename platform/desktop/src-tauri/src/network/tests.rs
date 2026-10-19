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
        .route("/api/system", get(|| async {
            axum::Json(json!({ "gpus": [{"name": "NV card (eGPU)", "memory_total_mb": 24564, "temperature": 58}], "ram": {"total_mb": 65536, "available_mb": 40000} }))
        }))
        // (a picture being made and one waiting, each with what it was asked to make: never shared)
        .route("/api/jobs", get(|| async {
            axum::Json(json!({ "jobs": [
                {"id": "j1", "kind": "image", "model": "tiny-image", "status": "in_progress", "progress": 40, "prompt": "a private prompt", "files": ["/Users/someone/out.png"]},
                {"id": "j2", "kind": "image", "model": "tiny-image", "status": "queued", "prompt": "another private prompt"},
                {"id": "j0", "kind": "image", "model": "tiny-image", "status": "completed", "prompt": "done"},
            ]}))
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
                "llm": {"state": "ready", "context_tokens": 32768},
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
    net.set(Settings { enabled: true, uses: None, lends: None, agent: false, share_models: None, role, name: format!("the {tag}"), port: 0, bind: "127.0.0.1".into() }).await.unwrap();
    net
}

/// The primary's heartbeat stopped, so what a test sets of what it heard of its workers stays as set.
fn quiet(primary: &Network) {
    if let Some(stop) = lock(&primary.inner().running).heartbeat.take() {
        let _ = stop.send(true);
    }
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
    pair_with(primary, worker, false).await
}

/// As [`pair`]; `both_ways`: the primary's person lets the worker use it too.
async fn pair_with(primary: &Network, worker: &Network, both_ways: bool) -> String {
    pair_answer(primary, worker, both_ways).await["node"]["nodeId"].as_str().unwrap().to_string()
}

/// As [`pair_with`], the primary's answer to its person's yes.
async fn pair_answer(primary: &Network, worker: &Network, both_ways: bool) -> Value {
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let address = worker.listening().unwrap().to_string();
    let (status, started) = call(&p, "POST", "/api/network/pairings/start", Some(json!({ "address": address }))).await;
    assert_eq!(status, StatusCode::OK, "{started}");
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    let request = &waiting["requests"][0];
    assert_eq!(request["code"], started["code"], "both screens show the same code");
    assert_eq!(request["name"], primary.display_name(), "{request}");
    let pairing = json!({ "pairingId": started["pairingId"] });
    let (_, before) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing.clone())).await;
    assert_eq!(before["status"], "pending", "nothing is paired before the worker's person approves");
    let (status, approved) = call(&w, "POST", &format!("/api/network/requests/{}/approve", request["id"].as_str().unwrap()), None).await;
    assert_eq!((status, approved["status"].as_str()), (StatusCode::OK, Some("approved")));
    let (_, there) = call(&p, "POST", "/api/network/pairings/confirm", Some(pairing.clone())).await;
    assert_eq!((there["status"].as_str(), &there["code"]), (Some("approved_there"), &started["code"]), "{there}");
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert!(nodes["nodes"].as_array().unwrap().iter().all(|n| n["nodeId"] != started["worker"]["nodeId"]), "not paired here before this side says the codes matched");
    let (_, confirmed) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": true, "bothWays": both_ways }))).await;
    assert_eq!(confirmed["status"], "approved", "{confirmed}");
    confirmed
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

/// Each time a primary tells its random value a new code is made. Asked again and again (a computer in the middle trying
/// for the code the other screen shows), one address is held to a few codes a minute and a few more within a request's
/// time, and every address together to a few more, whatever addresses they claim. A request replaced by a new one
/// from the same computer counts, and the person sees how many tries its address had.
#[tokio::test]
async fn pairing_codes_are_held_to_a_few_tries() {
    use super::identity::{nonce, Identity};
    let worker = node("worker-tries", Role::Worker, None).await;
    let w = router(worker.clone());
    let me = Identity::generate().unwrap();
    let asking = tls::client(&me, Pin::Record(Default::default())).unwrap();
    let at = |path: &str| format!("https://{}{path}", worker.listening().unwrap());
    let try_once = || async {
        let asked: Value = asking.post(at("/worker/v1/pair")).json(&json!({ "nodeId": me.node_id, "name": "a primary" })).send().await.unwrap().json().await.unwrap();
        asking.post(at(&format!("/worker/v1/pair/{}", asked["requestId"].as_str().unwrap()))).json(&json!({ "nonce": nonce().unwrap() })).send().await.unwrap().status()
    };
    // (the tries made so far, made older)
    let age = |secs: u64| {
        for t in lock(&worker.inner().pairing_tries).iter_mut() {
            t.0 = t.0.checked_sub(std::time::Duration::from_secs(secs)).unwrap();
        }
    };
    for _ in 0..5 {
        assert!(try_once().await.is_success());
    }
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    assert_eq!(waiting["requests"].as_array().unwrap().len(), 1, "each new request replaced the one before: {waiting}");
    assert_eq!(waiting["requests"][0]["tries"], 5, "and the person is told how many codes its address had: {waiting}");
    assert_eq!(try_once().await, reqwest::StatusCode::TOO_MANY_REQUESTS, "a sixth within the minute");
    // a minute on, a few more, up to ten for one address within a request's time
    age(90);
    for _ in 0..5 {
        assert!(try_once().await.is_success());
    }
    age(90);
    assert_eq!(try_once().await, reqwest::StatusCode::TOO_MANY_REQUESTS, "ten for one address");
    // (a primary asking from there now is told why, and to wait)
    let primary = node("primary-tries", Role::Primary, None).await;
    let (status, refused) = call(&router(primary.clone()), "POST", "/api/network/pairings/start", Some(json!({ "address": worker.listening().unwrap().to_string() }))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY, "{refused}");
    assert!(refused["error"]["message"].as_str().unwrap_or_default().contains("wait a few minutes"), "{refused}");
    primary.stop();
    // every address together: ten a minute, and twenty within a request's time
    let claimed = |i: u8| std::net::IpAddr::from([192, 168, 7, i]);
    for i in 0..10 {
        assert_eq!(worker::take_try(&worker, claimed(i)), Some(1), "{i}");
    }
    assert_eq!(worker::take_try(&worker, claimed(100)), None, "ten in all within the minute, from whatever addresses");
    age(61);
    assert_eq!(worker::take_try(&worker, claimed(200)), None, "twenty in all within a request's time");
    // a request's time on, they no longer count
    age(5 * 60);
    assert_eq!(worker::take_try(&worker, claimed(1)), Some(1));
    worker.stop();
}

/// A computer not paired here sends its request's body whole, soon and small: one that trickles in is answered and let
/// go within moments, and a large one is refused before any route reads it (a voice copy's route takes 32 MB from the
/// paired primary, read before the route says the caller is not it).
#[tokio::test]
async fn a_strangers_request_body_is_small_and_soon() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let worker = node("worker-body", Role::Worker, None).await;
    let address = worker.listening().unwrap();
    let id = super::identity::Identity::generate().unwrap();
    let stranger = tls::client(&id, Pin::Record(Default::default())).unwrap();
    let r = stranger.put(format!("https://{address}/worker/v1/voices/x")).json(&json!({ "clip": "A".repeat(1 << 20) })).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::PAYLOAD_TOO_LARGE);
    // (a small one reaches its route as before)
    let r = stranger.post(format!("https://{address}/worker/v1/pair")).json(&json!({ "nodeId": "ab", "name": "x" })).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    // a body that never comes: answered, and the connection let go, within the stranger's time
    let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(tls::client_config(&id, Pin::Record(Default::default())).unwrap()));
    let tcp = tokio::net::TcpStream::connect(address).await.unwrap();
    let mut conn = connector.connect(rustls::pki_types::ServerName::try_from("oaiy-node.invalid").unwrap(), tcp).await.unwrap();
    conn.write_all(b"POST /worker/v1/pair HTTP/1.1\r\nhost: worker\r\ncontent-type: application/json\r\ncontent-length: 100\r\n\r\n{").await.unwrap();
    let started = std::time::Instant::now();
    let mut answer = Vec::new();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(20), conn.read_to_end(&mut answer)).await;
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.starts_with("HTTP/1.1 408"), "{answer}");
    assert!(started.elapsed() < std::time::Duration::from_secs(10), "let go in {:?}", started.elapsed());
    worker.stop();
}

/// A primary's voices are copied to a worker under its certificate (sixteen hex digits of it), not its node id, which
/// is its own say: another primary cannot name itself to share their names, replace them, or have them removed with it.
/// They are removed when it is unpaired.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_primarys_voice_copies_are_kept_under_its_certificate() {
    // (two certificates whose fingerprints share their first six digits: their copies never meet)
    let (a, b) = (format!("ab12cd{}", "0".repeat(58)), format!("ab12cd{}", "1".repeat(58)));
    let (for_a, for_b) = (worker::copied_voice_name(&a, "front-desk"), worker::copied_voice_name(&b, "front-desk"));
    assert_ne!(for_a, for_b);
    assert!(!for_a.starts_with(&worker::copied_voice_name(&b, "")) && crate::voice::voices::is_copy(&for_a), "{for_a}");
    // (within a voice's 48 characters: a long name cut, with a tag of its whole name)
    let long = worker::copied_voice_name(&a, &"x".repeat(40));
    assert!(long.len() <= 48 && crate::voice::voices::clean_name(&long).as_deref() == Ok(long.as_str()), "{long}");
    assert_ne!(long, worker::copied_voice_name(&a, &format!("{}y", "x".repeat(39))));
    // a paired primary's copy, named so, and removed when it is unpaired
    crate::voice::voices::init(&temp("voices-copies"));
    let worker = node("worker-copies", Role::Worker, None).await;
    let primary = node("primary-copies", Role::Primary, None).await;
    pair(&primary, &worker).await;
    let me = primary.identity().unwrap();
    let pinned = tls::client(&me, Pin::Record(Default::default())).unwrap();
    let clip = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, vec![7u8; 2048]);
    let r: Value = pinned.put(format!("https://{}/worker/v1/voices/copy-test", worker.listening().unwrap())).json(&json!({ "clip": clip, "extension": "wav" })).send().await.unwrap().json().await.unwrap();
    let kept = format!("from-{}-copy-test", &me.fingerprint[..16]);
    assert_eq!(r["voice"], kept, "{r}");
    let of_it = || crate::voice::voices::copies(&worker::copied_voice_name(&me.fingerprint, "")).iter().any(|c| c.name == kept);
    assert!(of_it());
    call(&router(worker.clone()), "DELETE", &format!("/api/network/nodes/{}", me.node_id), None).await;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while of_it() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(!of_it(), "its copies go with it");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A computer found on the LAN is paired at an address that reaches it: IPv4 first (the addresses come in no order),
/// and never an IPv6 link-local one, which needs its interface named.
#[test]
fn a_found_computer_is_paired_at_an_address_that_reaches_it() {
    let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
    let found = discovery::reachable([ip("fe80::bcc6:4eff:febc:20b2"), ip("fd85:2144:420d::7ebf"), ip("192.168.2.147"), ip("8.8.8.8"), ip("10.0.0.4")].into_iter(), 17975);
    assert_eq!(found, ["192.168.2.147:17975", "10.0.0.4:17975", "[fd85:2144:420d::7ebf]:17975"]);
    assert!(discovery::reachable([ip("fe80::1")].into_iter(), 17975).is_empty(), "link-local alone: not kept");
}

/// A port another program holds is refused before anything is kept, as the listener would find it: on Windows too,
/// where a probe that let the address be reused would bind it beside a program that lets it be reused as well (and the
/// listener, which does not, would then fail).
#[tokio::test]
async fn a_port_another_program_holds_is_refused_before_anything_is_kept() {
    for reusable in [true, false] {
        let held = tokio::net::TcpSocket::new_v4().unwrap();
        held.set_reuseaddr(reusable).unwrap();
        held.bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let held = held.listen(8).unwrap();
        let port = held.local_addr().unwrap().port();
        let net = Network::open_with(&temp("held"), ai(None), false);
        let err = net.set(Settings { enabled: true, uses: None, lends: None, agent: false, share_models: None, role: Role::Worker, name: String::new(), port, bind: "127.0.0.1".into() }).await.unwrap_err();
        assert!(err.contains("cannot lend itself"), "held reusable {reusable}: {err}");
        assert!(!net.settings().enabled && net.listening().is_none(), "nothing kept or started");
        drop(held);
    }
}

/// This computer's name, where none is set, is the system's own, as `hostname` gives it (asked with no console window,
/// which Windows would otherwise open for it at every launch).
#[cfg(windows)]
#[test]
fn this_computers_name_is_the_systems_own() {
    let name = system_name().map(|n| n.trim().to_string()).unwrap_or_default();
    assert!(!name.is_empty() && !name.contains('\n'), "{name:?}");
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
    worker.set(Settings { enabled: true, uses: None, lends: None, agent: false, share_models: None, role: Role::Worker, name: "the worker-untold".into(), port, bind: "127.0.0.1".into() }).await.unwrap();
    primary::tell_untold(&primary).await;
    let (_, told) = call(&w, "GET", "/api/network/nodes", None).await;
    assert!(told["nodes"].as_array().unwrap().is_empty(), "told once back: {told}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A worker's model chosen as this computer's default of its kind: the catalogue marks it its kind's default (this
/// computer's own not), the engine's chats go to the worker for a language model, one chosen in Engines since takes
/// its place again, and unpairing the worker lets it go.
#[tokio::test]
async fn a_workers_model_can_be_this_computers_default() {
    let worker = node("worker-default", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-default", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let image = format!("engine:image:node.{node_id}/tiny-image");
    let set = |defaults: Value| call(&p, "PUT", "/api/network/settings", Some(json!({ "defaults": defaults })));
    let (status, v) = set(json!({ "image": image })).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!((&v["defaults"]["image"]["model"], &v["defaults"]["image"]["nodeName"]), (&json!("tiny-image"), &json!("the worker-default")), "{v}");
    assert!(primary.listening().is_none() && v["enabled"] == true, "only the defaults changed: {v}");
    // this computer's own picture model (its engines answering now) and the worker's, as the catalogue lists them
    let catalogue = || {
        let mut services = vec![json!({ "id": "engine:image:qwen", "kind": "image", "model": "qwen", "default": true }), json!({ "id": "engine:llm:flash", "kind": "llm", "model": "flash", "default": true })];
        services.extend(primary::remote_services_of(&primary));
        primary::apply_defaults_of(&primary, &mut services);
        services
    };
    let services = catalogue();
    let default_of = |kind: &str| services.iter().filter(|s| s["kind"] == kind && s["default"] == true).map(|s| s["id"].as_str().unwrap().to_string()).collect::<Vec<_>>();
    assert_eq!(default_of("image"), vec![image.clone()], "the worker's picture model is the default, this computer's not");
    assert_eq!(default_of("llm"), vec!["engine:llm:flash".to_string()], "a kind with no default chosen on a worker keeps its own");
    // (this computer's default then is kept beside it, learnt once its engines answered)
    assert_eq!(primary.inner().defaults().get("image").unwrap().instead_of.as_deref(), Some("qwen"));
    for bad in [json!({ "image": format!("engine:llm:node.{node_id}/tiny-llm") }), json!({ "image": "engine:image:node.nobody/tiny-image" }), json!({ "image": format!("engine:image:node.{node_id}/gone") })] {
        let (status, v) = set(bad.clone()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}: {v}");
    }
    // the language model: the worker's chosen one answers the engine's chats
    let (status, _) = set(json!({ "llm": format!("engine:llm:node.{node_id}/tiny-llm") })).await;
    assert_eq!(status, StatusCode::OK);
    let entry = primary::default_chat_entry_of(&primary, Some("flash")).unwrap();
    assert_eq!(entry["nodeId"], json!(node_id));
    let answer = primary::chat_on(&primary, &node_id, axum::body::Bytes::from(json!({ "stream": true, "messages": [{"role": "user", "content": "hi"}] }).to_string())).await;
    let text = String::from_utf8(axum::body::to_bytes(answer.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap();
    assert!(text.contains("Hel"), "{text}");
    // another language model chosen here since, in Engines: this computer's own again
    assert!(primary::default_chat_entry_of(&primary, Some("big")).is_none());
    assert!(primary.inner().defaults().get("llm").is_none());
    // chosen as Engines and the Agent's model menu choose (`PUT /api/engines/defaults`): a worker's, checked as the settings
    // route checks it, then this computer's own again even where it is the one the worker's stood in for
    let llm = format!("engine:llm:node.{node_id}/tiny-llm");
    primary::choose_default_of(&primary, "llm", Some(&llm)).await.unwrap();
    assert_eq!(primary.inner().defaults().get("llm").unwrap().id, llm);
    assert!(primary::choose_default_of(&primary, "llm", Some("engine:llm:node.nobody/tiny-llm")).await.is_err());
    assert_eq!(primary.inner().defaults().get("llm").unwrap().id, llm, "a refused choice changes nothing");
    primary::choose_default_of(&primary, "llm", None).await.unwrap();
    assert!(primary.inner().defaults().get("llm").is_none() && primary::default_chat_entry_of(&primary, Some("flash")).is_none());
    // let go by hand, and with the worker unpaired
    let (_, v) = set(json!({ "image": null })).await;
    assert!(v["defaults"].get("image").is_none(), "{v}");
    set(json!({ "image": image })).await;
    call(&p, "DELETE", &format!("/api/network/nodes/{node_id}"), None).await;
    assert!(primary.inner().defaults().all().is_empty(), "unpaired: none left on it");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// The flows a worker shares ("Let apps run it" there) are its primary's to list and run there, by its own runner: one
/// it does not share is not; a run is followed and stopped from the primary, its pictures named at the primary's route
/// for the worker; another computer's runs are not the primary's to read.
#[tokio::test]
async fn a_primary_runs_the_flows_a_worker_shares_there() {
    let worker = node("worker-flows", Role::Worker, Some(fake_engines().await)).await;
    let flows = std::sync::Arc::new(crate::bridge::worker::FlowStore::new(temp("flows")));
    let shared = json!({ "name": "Make a logo", "oaiyRemote": true, "nodes": [{ "id": "n1", "type": "input_text", "data": { "label": "Prompt", "value": "a fox" } }] });
    flows.put("logo", &shared.to_string()).unwrap();
    flows.put("private", &json!({ "name": "Mine", "nodes": [] }).to_string()).unwrap();
    let ledger = crate::bridge::ledger::new_handle();
    worker.set_bridge(ledger.clone(), flows);
    let primary = node("primary-flows", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let (status, listed) = call(&p, "GET", &format!("/api/network/nodes/{node_id}/flows"), None).await;
    assert_eq!(status, StatusCode::OK, "{listed}");
    assert_eq!(listed["flows"].as_array().unwrap().len(), 1, "its shared flows alone: {listed}");
    assert_eq!((&listed["flows"][0]["flowId"], &listed["flows"][0]["inputs"][0]["name"]), (&json!("logo"), &json!("Prompt")));
    // (and with its node, as the heartbeat heard them)
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["flows"][0]["name"], "Make a logo", "{nodes}");
    let (status, refused) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/flows/private/runs"), Some(json!({}))).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{refused}");
    assert!(refused.to_string().contains("Let apps run it"), "{refused}");
    let (status, run) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/flows/logo/runs"), Some(json!({ "input": { "Prompt": "a lighthouse" } }))).await;
    assert_eq!((status, &run["status"]), (StatusCode::CREATED, &json!("queued")), "{run}");
    let run_id = run["runId"].as_str().unwrap().to_string();
    let kept = ledger.lock().unwrap().get(&run_id).unwrap();
    assert_eq!((kept.flow_id.as_deref(), kept.input.clone()), (Some("logo"), Some(json!({ "Prompt": "a lighthouse" }))));
    // finished there with a picture of its engines': named at this computer's route for the worker
    {
        let mut l = ledger.lock().unwrap();
        l.claim(&run_id, crate::bridge::ledger::Runtime::Desktop, "test");
        let picture = format!("http://127.0.0.1:{}/api/ai/engine/gateway/files/images/x.png", crate::api_port());
        l.finish(&run_id, crate::bridge::ledger::RunStatus::Succeeded, Some(json!({ "output": { "image": picture } })), None).unwrap();
    }
    let resp = p.clone().oneshot(Request::get(format!("/api/network/nodes/{node_id}/runs/{run_id}")).header("host", "127.0.0.1:17972").body(Body::empty()).unwrap()).await.unwrap();
    let done: Value = serde_json::from_slice(&axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(done["status"], "succeeded");
    assert_eq!(done["output"]["output"]["image"], format!("http://127.0.0.1:17972/api/network/nodes/{node_id}/gateway/files/images/x.png"), "{done}");
    // a run is stopped from here; one the worker keeps for someone else is not this computer's to read
    let (_, second) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/flows/logo/runs"), Some(json!({}))).await;
    let (status, stopped) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/runs/{}/cancel", second["runId"].as_str().unwrap()), None).await;
    assert_eq!((status, &stopped["status"]), (StatusCode::OK, &json!("cancelled")), "{stopped}");
    let other = {
        let request = crate::bridge::ledger::RunRequest {
            caller_product: "formlogic".into(),
            flow_id: Some("logo".into()),
            inline_graph: false,
            input: None,
            timeout_ms: None,
            mode: "queued".into(),
            correlation_id: "c".into(),
            idempotency_key: "k".into(),
            lineage: Default::default(),
            trigger_event: None,
            caller_label: None,
        };
        match ledger.lock().unwrap().reserve(&request) {
            crate::bridge::ledger::ReserveOutcome::Reserved(r) => r.run_id,
            _ => unreachable!(),
        }
    };
    let (status, _) = call(&p, "GET", &format!("/api/network/nodes/{node_id}/runs/{other}"), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "another caller's run");

    // what it has and its queue, as the heartbeat hears them: its processor, memory and GPUs; the pictures it makes
    // and has waiting (their kind and model, never what they were asked); its flow runs, another computer's unnamed
    let peer = primary.inner().peers().get(&node_id, Role::Worker).cloned().unwrap();
    primary::check(&primary, &peer).await;
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    let n = &nodes["nodes"][0];
    assert!(n["machine"]["cpu"]["cores"].as_u64().unwrap() >= 1, "{n}");
    assert_eq!((&n["machine"]["ram"]["total_mb"], &n["machine"]["gpus"][0]["temperature"]), (&json!(65536), &json!(58)), "{n}");
    let jobs = n["jobs"].as_array().unwrap();
    assert_eq!(
        jobs.iter().map(|j| (j["kind"].as_str().unwrap(), j["status"].as_str().unwrap())).collect::<Vec<_>>(),
        [("image", "running"), ("image", "waiting"), ("flow", "waiting")],
        "{n}"
    );
    assert_eq!(jobs[0]["progress"], 40);
    assert_eq!((&jobs[2]["mine"], &jobs[2]["flow"], &jobs[2]["runId"]), (&json!(false), &Value::Null, &Value::Null), "another caller's run, unnamed");
    let said = nodes.to_string();
    assert!(!said.contains("private prompt") && !said.contains("/Users/someone"), "{said}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A computer's engines that keep one model file (by its SHA-256 and size): where it is, said to its own desktop alone.
async fn engines_keeping(path: PathBuf, sha: String, size: u64) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().route(
        "/api/library/find",
        post(move |axum::Json(b): axum::Json<Value>| {
            let (path, sha) = (path.clone(), sha.clone());
            async move {
                let files: Vec<Value> = b["files"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|f| {
                        let mine = f["sha256"] == json!(sha) && f["size"] == json!(size);
                        json!({ "sha256": f["sha256"], "size": f["size"], "path": if mine { json!(path) } else { Value::Null }, "checked": mine })
                    })
                    .collect();
                axum::Json(json!({ "files": files }))
            }
        }),
    );
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    base
}

/// A model file a computer lends is fetched from it by the computer using it, by its SHA-256 alone, carried on from a
/// partial file; one it does not have is not, nor by a computer not paired with it, nor once its person stops sharing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_model_file_a_paired_computer_has_is_fetched_from_it() {
    use sha2::{Digest, Sha256};
    let dir = temp("lent-files");
    std::fs::create_dir_all(&dir).unwrap();
    let weights: Vec<u8> = (0..3_000_000u32).map(|i| (i * 31 % 251) as u8).collect();
    let path = dir.join("model.safetensors");
    std::fs::write(&path, &weights).unwrap();
    let sha: String = Sha256::digest(&weights).iter().map(|b| format!("{b:02x}")).collect();
    let size = weights.len() as u64;
    let worker = node("worker-files", Role::Worker, Some(engines_keeping(path.clone(), sha.clone(), size).await)).await;
    let primary = node("primary-files", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let peer = primary.inner().peers().get(&node_id, Role::Worker).cloned().unwrap();
    primary::check(&primary, &peer).await;

    // who has it: the worker, for this file alone
    let other = "ab".repeat(32);
    let who = files::who_has(&primary, &[(sha.clone(), size), (other.clone(), size)]).await;
    assert_eq!(who.iter().map(|p| p.as_ref().map(|p| p.node_id.clone())).collect::<Vec<_>>(), vec![Some(node_id.clone()), None]);
    // fetched, carried on from half of it here already
    let part = dir.join("got/model.safetensors.part");
    std::fs::create_dir_all(part.parent().unwrap()).unwrap();
    std::fs::write(&part, &weights[..1_000_000]).unwrap();
    let seen = std::sync::Mutex::new(0u64);
    let progress = |n: u64| *seen.lock().unwrap() = n;
    let from = files::fetch(&primary, &sha, size, &part, &progress, &|| false).await.unwrap();
    assert_eq!(from.as_deref(), Some("the worker-files"));
    assert_eq!(std::fs::read(&part).unwrap(), weights, "the same bytes, the first million kept");
    assert_eq!(*seen.lock().unwrap(), size);
    // one it does not have: none fetched
    assert_eq!(files::fetch(&primary, &other, size, &dir.join("none.part"), &|_| {}, &|| false).await.unwrap(), None);

    // a computer not paired with it is refused, by its path or not
    let stranger = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let address = worker.listening().unwrap();
    let r = stranger.get(format!("https://{address}/worker/v1/models/blob/{sha}?size={size}")).send().await.unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);
    // its person stops sharing model files: refused, and not said to have it
    worker.set(Settings { share_models: Some(false), ..worker.settings() }).await.unwrap();
    let peer = primary.inner().peers().get(&node_id, Role::Worker).cloned().unwrap();
    primary::check(&primary, &peer).await;
    assert_eq!(files::who_has(&primary, &[(sha.clone(), size)]).await, vec![None]);
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A stand-in for OAIY Voice: its models, a transcription, and speech (24 kHz PCM) in a voice it has, the copies of a
/// primary's voices (`from-…`) among them.
#[cfg(unix)]
const FAKE_VOICE: &str = r#"#!/usr/bin/env python3
import http.server, json, sys
port = int(sys.argv[1])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def whole(self, code, body, ctype="application/json", extra=()):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        for k, v in extra: self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == "/v1/models": self.whole(200, json.dumps({"data": [{"id": "parakeet"}]}).encode())
        else: self.whole(404, b"{}")
    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/audio/transcriptions": self.whole(200, json.dumps({"text": "hello from %d" % port}).encode())
        elif self.path == "/v1/audio/speech":
            if asked.get("voice") and not asked["voice"].startswith("from-"): self.whole(400, b'{"error": "no such voice"}')
            else: self.whole(200, b"\x01\x00" * 4800, "audio/pcm", [("X-Sample-Rate", "24000")])
        else: self.whole(404, b"{}")
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"#;

/// OAIY Voice set up on a worker, used from the primary in place of its own (Services → Other computers): the primary's
/// speech-to-text and speech go to the worker's, started there when asked, the primary's voice copied there first; the
/// worker lends that service's speech routes alone; that computer not answering, this computer's own OAIY Voice answers
/// a call meanwhile, where it has one; unpairing gives it back.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_primary_uses_oaiy_voice_on_a_worker() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp("fake-voice");
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("fake-oaiy-voice");
    std::fs::write(&script, FAKE_VOICE).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    // (a stand-in on a port of its own, in a computer's services)
    let set_up = |net: &Network| {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let template: crate::services::template::ServiceTemplate = serde_json::from_value(json!({
            "id": "oaiy-voice", "name": "OAIY Voice", "description": "a stand-in", "category": "Speech", "defaultPort": port,
            "autostart": false, "run": { "command": script.to_string_lossy(), "args": ["${port}"], "env": {} },
            "health": { "url": "http://127.0.0.1:${port}/v1/models", "timeoutSecs": 20 },
        }))
        .unwrap();
        let mut registry = net.inner().ai.registry.lock().unwrap();
        std::fs::create_dir_all(registry.data_dir().join("templates")).unwrap();
        registry.add_template(template).unwrap();
        port
    };
    crate::voice::voices::init(&temp("voices-data"));
    let worker = node("worker-voice", Role::Worker, Some(fake_engines().await)).await;
    let port = set_up(&worker);
    let primary = node("primary-voice", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let (status, v) = call(&p, "PUT", "/api/network/settings", Some(json!({ "services": { "oaiy-voice": node_id } }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(v["services"]["oaiy-voice"]["nodeName"], "the worker-voice", "{v}");
    let engines = crate::voice::engines::Engines::via(primary::service_target_of(&primary, "oaiy-voice").unwrap());
    let heard = engines.transcribe_wav(vec![0u8; 3200]).await.unwrap();
    assert_eq!(heard, format!("hello from {port}"), "heard on the worker");
    let voice = crate::voice::voices::chosen().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    engines.speak("Good morning.", Some(&voice), &tx).await.unwrap();
    drop(tx);
    let mut samples = 0;
    while let Some(chunk) = rx.recv().await {
        samples += chunk.len();
    }
    assert!(samples >= 4000, "spoken on the worker, in a copy of this computer's voice: {samples} samples");
    assert!(crate::voice::voices::copies("from-").iter().any(|c| c.name.ends_with(&voice)), "the voice copied there");
    assert!(!crate::voice::voices::list().iter().any(|c| c.name.starts_with("from-")), "kept apart from the worker's own voices");
    // the worker lends that service's speech routes, and nothing else of it or of another
    let lent = primary::service_target_of(&primary, "oaiy-voice").unwrap();
    for (path, code) in [("/worker/v1/services/oaiy-voice/v1/config", 404), ("/worker/v1/services/ollama/v1/models", 404)] {
        let r = lent.client.get(format!("{}{path}", lent.root)).send().await.unwrap();
        assert_eq!(r.status().as_u16(), code, "{path}");
    }
    // that computer not answering (the heartbeat found it so, or asked now), this computer's own answers meanwhile
    let own = set_up(&primary);
    quiet(&primary);
    lock(&primary.inner().nodes).get_mut(&node_id).unwrap().online = false;
    let offline = primary::service_target_of(&primary, "oaiy-voice").unwrap();
    assert!(offline.offline);
    let here = crate::voice::engines::Engines::via_or_here(offline, primary.inner().ai.registry.clone());
    assert_eq!(here.transcribe_wav(vec![0u8; 3200]).await.unwrap(), format!("hello from {own}"), "heard here");
    lock(&primary.inner().nodes).get_mut(&node_id).unwrap().online = true;
    let answering = crate::voice::engines::Engines::via_or_here(primary::service_target_of(&primary, "oaiy-voice").unwrap(), primary.inner().ai.registry.clone());
    assert_eq!(answering.transcribe_wav(vec![0u8; 3200]).await.unwrap(), format!("hello from {port}"), "on the worker again once it answers");
    worker.stop();
    assert_eq!(answering.transcribe_wav(vec![0u8; 3200]).await.unwrap(), format!("hello from {own}"), "asked now and not answering: heard here");
    // a service the worker has not set up is not chosen; unpairing gives this computer's own back
    let (status, _) = call(&p, "PUT", "/api/network/settings", Some(json!({ "services": { "oaiy-voice": "nobody" } }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    call(&p, "DELETE", &format!("/api/network/nodes/{node_id}"), None).await;
    assert!(primary.inner().services().all().is_empty());
    for n in [&worker, &primary] {
        let _ = n.inner().ai.registry.lock().unwrap().stop("oaiy-voice");
    }
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A computer can use others and lend itself at once: two such computers paired both ways in one pairing (the one
/// asking lets the other use it too), each then using the other, one card a computer; unpairing forgets both ways on
/// both.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_computers_can_use_each_other() {
    let a = node("a-both", Role::Primary, Some(fake_engines().await)).await;
    let b = node("b-both", Role::Worker, Some(fake_engines().await)).await;
    for n in [&a, &b] {
        n.set(Settings { uses: Some(true), lends: Some(true), ..n.settings() }).await.unwrap();
        assert_eq!(n.settings().active_role(), Role::Both);
    }
    assert!(a.listening().is_some() && b.listening().is_some(), "both lend themselves");
    let answer = pair_answer(&a, &b, true).await;
    assert_eq!((&answer["bothWays"], &answer["bothWaysError"]), (&json!(true), &Value::Null), "{answer}");
    let b_id = answer["node"]["nodeId"].as_str().unwrap().to_string();
    let a_id = a.identity().unwrap().node_id;
    assert!(a.inner().peers().get(&b_id, Role::Worker).is_some() && a.inner().peers().get(&b_id, Role::Primary).is_some(), "a uses b, and b may use a");
    let a_on_b = b.inner().peers().get(&a_id, Role::Worker).cloned().expect("b keeps a as a computer it uses");
    assert_eq!(a_on_b.address, a.listening().unwrap().to_string(), "where a listens");
    // b uses a, through its own routes
    primary::check(&b, &a_on_b).await;
    let (_, nodes) = call(&router(b.clone()), "GET", "/api/network/nodes", None).await;
    let seen = &nodes["nodes"][0];
    assert_eq!(nodes["nodes"].as_array().unwrap().len(), 1, "one card a computer: {nodes}");
    assert_eq!((&seen["uses"], &seen["usesThis"], &seen["online"]), (&json!(true), &json!(true), &json!(true)), "{seen}");
    assert!(seen["models"].as_array().unwrap().iter().any(|m| m["model"] == "tiny-image"), "{seen}");
    // one computer, paired both ways: counted once, as one each uses and one that uses it
    let (_, status) = call(&router(a.clone()), "GET", "/api/network/status", None).await;
    assert_eq!((&status["paired"], &status["using"], &status["usedBy"]), (&json!(1), &json!(1), &json!(1)), "{status}");
    // b makes a's picture model its default (forgotten when they unpair, from either side)
    let image = format!("engine:image:node.{a_id}/tiny-image");
    let (code, set) = call(&router(b.clone()), "PUT", "/api/network/settings", Some(json!({ "defaults": { "image": image } }))).await;
    assert_eq!((code, &set["defaults"]["image"]["id"]), (StatusCode::OK, &json!(image)), "{set}");
    // a computer that names itself as b (b's node id, its own certificate) is not let ask a in b's place
    let impostor = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let asked = impostor
        .post(format!("https://{}/worker/v1/pair", a.listening().unwrap()))
        .json(&json!({ "nodeId": b_id, "name": "the b-both" }))
        .send()
        .await
        .unwrap();
    assert_eq!(asked.status(), reqwest::StatusCode::CONFLICT, "{}", asked.text().await.unwrap());
    let stand_in = store::Peer { fingerprint: "ff".repeat(32), role: Role::Primary, ..a.inner().peers().get(&b_id, Role::Worker).cloned().unwrap() };
    assert_eq!(a.inner().peers().add(stand_in), Err(store::TAKEN.to_string()), "nor kept in its place");
    // a only uses (not both ways): b keeps nothing of a to use
    let c = node("c-uses", Role::Primary, None).await;
    let b_for_c = pair(&c, &b).await;
    assert_eq!(b_for_c, b_id);
    let c_id = c.identity().unwrap().node_id;
    assert!(b.inner().peers().get(&c_id, Role::Primary).is_some() && b.inner().peers().get(&c_id, Role::Worker).is_none());
    // unpaired at a: forgotten both ways, on both
    call(&router(a.clone()), "DELETE", &format!("/api/network/nodes/{b_id}"), None).await;
    assert!(!a.inner().peers().knows(&b_id) && !b.inner().peers().knows(&a_id), "forgotten both ways on both");
    assert!(b.inner().peers().knows(&c_id), "and nothing else");
    assert!(b.inner().defaults().get("image").is_none(), "b's default on a forgotten with a, told by a");
    for n in [&a, &b, &c] {
        n.stop();
    }
}

/// A computer that asks to pair by itself (its person lets its Agent set up OAIY Network) is approved at once only
/// while the other computer's window to pair by itself is open (its person or Agent opened it, minutes long): outside
/// it, its person is asked as for any other. The first such pairing closes the window, and is told on the other's
/// page with its code; the computer that asked pairs on its side only once its person says the codes match. Past a few
/// computers so paired, the person is asked again even with the window open.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_computer_pairs_by_itself_only_while_the_others_window_is_open() {
    let a = node("a-auto", Role::Primary, None).await;
    let b = node("b-auto", Role::Worker, Some(fake_engines().await)).await;
    a.set(Settings { lends: Some(true), agent: true, ..a.settings() }).await.unwrap();
    // b's Agent may set it up, but no window is open: b's person is asked as before (its say-so alone is anyone's)
    b.set(Settings { agent: true, ..b.settings() }).await.unwrap();
    let (pa, pb) = (router(a.clone()), router(b.clone()));
    let at = json!({ "address": b.listening().unwrap().to_string() });
    let (_, started) = call(&pa, "POST", "/api/network/pairings/start", Some(at.clone())).await;
    assert_eq!(started["status"], "pending", "{started}");
    let (_, waiting) = call(&pb, "GET", "/api/network/requests", None).await;
    assert_eq!(waiting["requests"].as_array().unwrap().len(), 1);
    call(&pa, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": false }))).await;
    // a window more than ten minutes long, or opened on a computer that does not lend itself, is refused
    let (code, _) = call(&pb, "PUT", "/api/network/settings", Some(json!({ "openPairingMinutes": 11 }))).await;
    assert_eq!(code, StatusCode::BAD_REQUEST);
    let (code, _) = call(&pa, "PUT", "/api/network/settings", Some(json!({ "lends": false, "openPairingMinutes": 5 }))).await;
    assert_eq!(code, StatusCode::CONFLICT);
    assert!(a.settings().lends(), "nothing changed by a refused request");
    // b's person opens the window: approved there at once, and the window closed by it
    let (_, opened) = call(&pb, "PUT", "/api/network/settings", Some(json!({ "openPairingMinutes": 5 }))).await;
    assert!(opened["pairingOpenUntilMs"].as_u64().unwrap() > store::now_ms() + 4 * 60_000, "{opened}");
    let at = json!({ "address": b.listening().unwrap().to_string() });
    let (status, started) = call(&pa, "POST", "/api/network/pairings/start", Some(at)).await;
    assert_eq!((status, &started["status"]), (StatusCode::OK, &json!("approved_there")), "{started}");
    let b_id = started["worker"]["nodeId"].as_str().unwrap().to_string();
    let a_id = a.identity().unwrap().node_id;
    assert!(b.inner().peers().get(&a_id, Role::Primary).is_some(), "paired on b's side by itself");
    let (_, waiting) = call(&pb, "GET", "/api/network/requests", None).await;
    assert!(waiting["requests"].as_array().unwrap().is_empty(), "no person asked there: {waiting}");
    let (_, told) = call(&pb, "GET", "/api/network/status", None).await;
    assert!(told["pairingOpenUntilMs"].is_null(), "closed by the pairing: {told}");
    assert_eq!(told["pairedByItself"][0]["name"], a.display_name(), "{told}");
    // (b shows the code with the computer it paired by itself, for a's person to compare)
    assert_eq!(told["pairedByItself"][0]["code"], started["code"], "{told}");
    // a's side pairs only on its person's word that b shows the same code, whatever b answered: anyone answering for b
    // could say it approved
    assert!(a.inner().peers().get(&b_id, Role::Worker).is_none(), "not paired on a's side before its person says the codes match");
    let (_, asked) = call(&pa, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"] }))).await;
    assert_eq!(asked["status"], "approved_there", "{asked}");
    let (_, paired) = call(&pa, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"], "accept": true, "bothWays": true }))).await;
    assert_eq!(paired["status"], "approved", "{paired}");
    assert!(a.inner().peers().get(&b_id, Role::Worker).is_some() && a.inner().peers().get(&b_id, Role::Primary).is_some());
    assert!(b.inner().peers().get(&a_id, Role::Primary).is_some() && b.inner().peers().get(&a_id, Role::Worker).is_some());
    // b says it was paired by itself (it compared no code); a compared it
    assert!(b.inner().peers().get(&a_id, Role::Primary).is_some_and(|p| p.by_itself));
    assert!(a.inner().peers().get(&b_id, Role::Worker).is_some_and(|p| !p.by_itself));
    let (_, nodes) = call(&pb, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["pairedByItself"], true, "{nodes}");
    // the window closed: the next computer asking so waits for the person
    let c = node("c-auto", Role::Primary, None).await;
    c.set(Settings { agent: true, ..c.settings() }).await.unwrap();
    let ask_c = || async { call(&router(c.clone()), "POST", "/api/network/pairings/start", Some(json!({ "address": b.listening().unwrap().to_string() }))).await.1 };
    assert_eq!(ask_c().await["status"], "pending");
    // past a few computers so paired (computers, not pairings), its person is asked even with the window open
    let peer = b.inner().peers().get(&a_id, Role::Primary).cloned().unwrap();
    for i in 0..8 {
        let made = store::Peer { node_id: format!("{i:032x}"), fingerprint: format!("{i:064}"), ..peer.clone() };
        b.inner().peers().add(made).unwrap();
    }
    call(&pb, "PUT", "/api/network/settings", Some(json!({ "openPairingMinutes": 2 }))).await;
    assert_eq!(ask_c().await["status"], "pending");
    let (_, still) = call(&pb, "GET", "/api/network/status", None).await;
    assert!(still["pairingOpenUntilMs"].is_u64(), "not taken by a request it did not approve: {still}");
    for n in [&a, &b, &c] {
        n.stop();
    }
}

/// A computer approved to use this one says it lends itself too (`/lends`) as they pair alone: not one approved long
/// ago, which would make this computer use it (and ask it first for model files) by saying so; once said, it stands.
/// A node id not as OAIY makes them (what names a computer's voice copies) is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_computer_lends_itself_back_only_as_they_pair() {
    let worker = node("worker-lends", Role::Worker, None).await;
    let primary = node("primary-lends", Role::Primary, None).await;
    pair(&primary, &worker).await;
    let primary_id = primary.identity().unwrap().node_id;
    let pinned = tls::client(&primary.identity().unwrap(), Pin::Record(Default::default())).unwrap();
    let address = worker.listening().unwrap();
    let lends = || pinned.post(format!("https://{address}/worker/v1/lends")).json(&json!({ "port": 40001, "name": "the primary-lends" })).send();
    // approved eleven minutes ago: refused
    let approved = worker.inner().peers().get(&primary_id, Role::Primary).cloned().unwrap();
    worker.inner().peers().add(store::Peer { paired_at_ms: store::now_ms() - 11 * 60_000, ..approved.clone() }).unwrap();
    assert_eq!(lends().await.unwrap().status(), reqwest::StatusCode::FORBIDDEN);
    assert!(worker.inner().peers().get(&primary_id, Role::Worker).is_none());
    // as they pair: kept, and said again, it changes nothing
    worker.inner().peers().add(store::Peer { paired_at_ms: store::now_ms(), ..approved }).unwrap();
    assert_eq!(lends().await.unwrap().status(), reqwest::StatusCode::NO_CONTENT);
    assert!(worker.inner().peers().get(&primary_id, Role::Worker).is_some_and(|w| w.address.ends_with(":40001")));
    assert_eq!(lends().await.unwrap().status(), reqwest::StatusCode::NO_CONTENT);
    // a node id as OAIY does not make them
    let stranger = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let asked = stranger.post(format!("https://{address}/worker/v1/pair")).json(&json!({ "nodeId": "ab", "name": "x" })).send().await.unwrap();
    assert_eq!(asked.status(), reqwest::StatusCode::BAD_REQUEST);
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// The window to pair by itself goes with what allows it: unticking the Agent's setting, or no longer lending, closes
/// it; and a pairing that gives back a window taken in it does not reopen one closed since.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pairing_window_closes_with_the_agent_setting_and_stays_closed() {
    let w = node("worker-window", Role::Worker, None).await;
    w.set(Settings { agent: true, ..w.settings() }).await.unwrap();
    assert!(w.open_pairing(5).is_some());
    w.keep(Settings { agent: false, ..w.settings() }).unwrap();
    assert_eq!(w.pairing_open_until_ms(), None, "unticked: closed");
    w.keep(Settings { agent: true, ..w.settings() }).unwrap();
    assert!(w.open_pairing(5).is_some());
    w.set(Settings { lends: Some(false), uses: Some(true), ..w.settings() }).await.unwrap();
    assert_eq!(w.pairing_open_until_ms(), None, "no longer lending: closed");
    // taken by a pairing that then does not go through, closed by the person meanwhile: given back, it stays closed
    w.set(Settings { lends: Some(true), ..w.settings() }).await.unwrap();
    let until = w.open_pairing(5);
    let turn = w.take_pairing_window().expect("open");
    w.open_pairing(0);
    w.give_back_pairing_window(until, turn);
    assert_eq!(w.pairing_open_until_ms(), None);
    // not closed meanwhile: given back
    let until = w.open_pairing(5);
    let turn = w.take_pairing_window().expect("open");
    w.give_back_pairing_window(until, turn);
    assert!(w.pairing_open_until_ms().is_some());
    w.stop();
}

/// A worker saved again while it listens (renamed) keeps its port: the listener before it has let the port go first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_worker_saved_again_listens_on_the_same_port() {
    let worker = node("worker-again", Role::Worker, None).await;
    let port = worker.listening().unwrap().port();
    for name in ["the worker renamed", "and again"] {
        worker.set(Settings { enabled: true, uses: None, lends: None, agent: false, share_models: None, role: Role::Worker, name: name.into(), port, bind: "127.0.0.1".into() }).await.unwrap();
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
    // (the window its language model runs with there, and whether the computer itself answers)
    assert_eq!((&llm["contextTokens"], &llm["nodeOnline"]), (&json!(32768), &json!(true)), "{llm}");
    assert!(image.get("contextTokens").is_none() && image["nodeOnline"] == true, "{image}");

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

    // stopped there: its models listed still, not ready, the computer itself not answering (`nodeOnline`), and why
    // said with where it was asked
    let peer = primary.inner().peers().get(&worker_id, Role::Worker).cloned().unwrap();
    let port = worker.listening().unwrap().port();
    worker.stop();
    primary::check(&primary, &peer).await;
    let llm = primary::remote_services_of(&primary).into_iter().find(|s| s["kind"] == "llm").unwrap();
    assert_eq!((&llm["ready"], &llm["nodeOnline"]), (&json!(false), &json!(false)), "{llm}");
    let (status, v) = call(&p, "GET", &format!("/api/network/nodes/{worker_id}/resources"), None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains(&format!("the worker did not answer: connection refused at 127.0.0.1:{port}")), "{v}");
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
    let peer = primary.inner().peers().get(&worker_id, Role::Worker).cloned().unwrap();
    primary::check(&primary, &peer).await;
    let (_, nodes) = call(&router(primary.clone()), "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["online"], false);
    assert!(nodes["nodes"][0]["lastError"].as_str().unwrap().contains("no longer lends itself"), "{nodes}");
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
    let err = worker.set(Settings { enabled: true, uses: None, lends: None, agent: false, share_models: None, role: Role::Worker, name: String::new(), port: 0, bind: "8.8.8.8".into() }).await.unwrap_err();
    assert!(err.contains("not a local network address"), "{err}");
    primary.stop();
}

/// What a worker shows to be typed on the primary: its LAN addresses each once and IPv4 first, without loopback, the
/// IPv6 link-local addresses every interface has, or a public address.
#[test]
fn the_addresses_shown_are_the_ones_a_person_can_type() {
    let ip = |s: &str| s.parse::<std::net::IpAddr>().unwrap();
    let ips = vec![
        ip("fe80::1"),
        ip("fd85:2144:420d::7ebf"),
        ip("192.168.2.147"),
        ip("127.0.0.1"),
        ip("fe80::bcc6:4eff:febc:20b2"),
        ip("192.168.2.147"),
        ip("8.8.8.8"),
        ip("10.0.0.4"),
    ];
    assert_eq!(typeable(ips.clone(), None), vec![ip("192.168.2.147"), ip("10.0.0.4"), ip("fd85:2144:420d::7ebf")]);
    // (the address the default route leaves by first: a virtual adapter's, listed before it, after)
    assert_eq!(typeable(ips.clone(), Some(ip("10.0.0.4"))), vec![ip("10.0.0.4"), ip("192.168.2.147"), ip("fd85:2144:420d::7ebf")]);
    assert_eq!(typeable(ips, Some(ip("8.8.8.8"))), vec![ip("192.168.2.147"), ip("10.0.0.4"), ip("fd85:2144:420d::7ebf")], "one not typeable is not put first");
}

/// A connection a paired primary keeps open to a worker (its streamed chat, a call's speech, a run followed): one
/// request on it, answered whole (`GET` with no body), its status line.
async fn ask_on<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(conn: &mut S, path: &str) -> std::io::Result<String> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    conn.write_all(format!("GET {path} HTTP/1.1\r\nhost: worker\r\n\r\n").as_bytes()).await?;
    let mut got = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let n = tokio::time::timeout(std::time::Duration::from_secs(5), conn.read(&mut chunk)).await.map_err(|_| std::io::Error::from(std::io::ErrorKind::TimedOut))??;
        if n == 0 {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        got.extend_from_slice(&chunk[..n]);
        let text = String::from_utf8_lossy(&got).to_string();
        let Some((head, body)) = text.split_once("\r\n\r\n") else { continue };
        let length = head.lines().find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap_or(0))).unwrap_or(0);
        if body.len() >= length {
            return Ok(head.lines().next().unwrap_or_default().to_string());
        }
    }
}

/// Saving a lending computer's settings as the Networking page does (its name and port each time) restarts its
/// listener only where it is to listen elsewhere: renamed, or using other computers too, it keeps the connections of
/// the computers using it (and says its new name); moved to another port, it listens there instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saving_settings_keeps_the_connections_of_the_computers_using_it() {
    let worker = node("worker-keeps", Role::Worker, None).await;
    let primary = node("primary-keeps", Role::Primary, None).await;
    pair(&primary, &worker).await;
    let address = worker.listening().unwrap();
    let w = router(worker.clone());
    let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(tls::client_config(&primary.identity().unwrap(), Pin::Fingerprint(worker.identity().unwrap().fingerprint)).unwrap()));
    let tcp = tokio::net::TcpStream::connect(address).await.unwrap();
    let mut conn = connector.connect(rustls::pki_types::ServerName::try_from("oaiy-node.invalid").unwrap(), tcp).await.unwrap();
    assert!(ask_on(&mut conn, "/worker/v1/health").await.unwrap().contains("200"));
    for change in [json!({ "name": "the worker renamed", "port": address.port() }), json!({ "uses": true, "name": "the worker renamed", "port": address.port() })] {
        let (status, v) = call(&w, "PUT", "/api/network/settings", Some(change.clone())).await;
        assert_eq!(status, StatusCode::OK, "{v}");
        assert_eq!(worker.listening(), Some(address), "{change}");
        assert!(ask_on(&mut conn, "/worker/v1/health").await.unwrap().contains("200"), "the same connection answers after {change}");
    }
    let stranger = tls::client(&super::identity::Identity::generate().unwrap(), Pin::Record(Default::default())).unwrap();
    let health: Value = stranger.get(format!("https://{address}/worker/v1/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(health["name"], "the worker renamed");
    // (another port: listened on there, the connection to the old one ended)
    let free = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (status, v) = call(&w, "PUT", "/api/network/settings", Some(json!({ "port": free }))).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(worker.listening().map(|a| a.port()), Some(free));
    assert!(ask_on(&mut conn, "/worker/v1/health").await.is_err(), "the old listener's connection ended");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A worker's language model as this computer's default is whichever it runs: a chat naming the one it ran before
/// (the Agent reads its model once a minute) is answered by the one it runs now. Not the default, the chat is refused,
/// saying what that computer runs now and how to follow it.
#[tokio::test]
async fn a_chat_on_the_default_follows_the_model_the_worker_runs_now() {
    let worker = node("worker-follows", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-follows", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let chat = format!("/api/network/nodes/{node_id}/v1/chat/completions");
    let asked = json!({ "model": "an-older-llm", "stream": true, "messages": [{"role": "user", "content": "hi"}] });
    let (status, v) = call(&p, "POST", &chat, Some(asked.clone())).await;
    assert_eq!(status, StatusCode::CONFLICT, "{v}");
    let said = v["error"]["message"].as_str().unwrap();
    assert!(said.contains("the worker-follows runs tiny-llm now, not an-older-llm") && said.contains("this computer's default"), "{said}");
    let (status, _) = call(&p, "PUT", "/api/network/settings", Some(json!({ "defaults": { "llm": format!("engine:llm:node.{node_id}/tiny-llm") } }))).await;
    assert_eq!(status, StatusCode::OK);
    let (status, v) = call(&p, "POST", &chat, Some(asked)).await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(v.as_str().unwrap_or_default().contains("Hel"), "answered by the model it runs now: {v}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A default on a worker says whether it is in effect, and why not where it is not: that computer not heard from since
/// this one started (it was asleep), its engines not answering since, or the model no longer one of its.
#[tokio::test]
async fn a_default_on_a_worker_says_whether_it_is_in_effect_and_why_not() {
    let worker = node("worker-waits", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-waits", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let (_, v) = call(&p, "PUT", "/api/network/settings", Some(json!({ "defaults": { "image": format!("engine:image:node.{node_id}/tiny-image") } }))).await;
    assert_eq!((&v["defaults"]["image"]["inEffect"], &v["defaults"]["image"]["waiting"]), (&json!(true), &Value::Null), "{v}");
    quiet(&primary);
    let heard = lock(&primary.inner().nodes).get(&node_id).cloned().unwrap();
    let waiting = || async { call(&p, "GET", "/api/network/status", None).await.1["defaults"]["image"].clone() };
    // (this computer restarted, that one asleep: nothing heard of it yet)
    lock(&primary.inner().nodes).clear();
    let d = waiting().await;
    assert_eq!((&d["inEffect"], &d["waiting"], &d["nodeName"]), (&json!(false), &json!("not_heard"), &json!("the worker-waits")), "{d}");
    // (heard, its engines not answering since)
    let mut down = heard.clone();
    down.capabilities.as_mut().unwrap()["discovery"] = Value::Null;
    lock(&primary.inner().nodes).insert(node_id.clone(), down);
    assert_eq!(waiting().await["waiting"], "engines_down");
    // (heard, the model gone from it)
    let mut gone = heard.clone();
    gone.capabilities.as_mut().unwrap()["discovery"]["models"]["image"] = json!([]);
    lock(&primary.inner().nodes).insert(node_id.clone(), gone);
    assert_eq!(waiting().await["waiting"], "model_gone");
    lock(&primary.inner().nodes).insert(node_id.clone(), heard);
    assert_eq!(waiting().await["inEffect"], true);
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A service used on a worker (OAIY Voice) stays listed, in the status and with the computer, while that computer has
/// not been heard from since this one started (`heard: false`), so its page can offer this computer's own again.
#[tokio::test]
async fn a_service_in_use_is_listed_while_its_computer_is_not_heard_from() {
    let worker = node("worker-unheard", Role::Worker, None).await;
    let primary = node("primary-unheard", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    primary.inner().services().set("oaiy-voice", Some(&node_id)).unwrap();
    quiet(&primary);
    let p = router(primary.clone());
    let (_, status) = call(&p, "GET", "/api/network/status", None).await;
    assert_eq!((&status["services"]["oaiy-voice"]["heard"], &status["services"]["oaiy-voice"]["online"]), (&json!(true), &json!(true)), "{status}");
    // (heard of, its OAIY Voice not set up there now: said so)
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["services"], json!([{ "id": "oaiy-voice", "name": "OAIY Voice", "installed": false, "status": null, "heard": true }]), "{nodes}");
    // this computer restarted, that one asleep
    lock(&primary.inner().nodes).clear();
    let (_, status) = call(&p, "GET", "/api/network/status", None).await;
    assert_eq!((&status["services"]["oaiy-voice"]["nodeName"], &status["services"]["oaiy-voice"]["heard"]), (&json!("the worker-unheard"), &json!(false)), "{status}");
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["services"], json!([{ "id": "oaiy-voice", "name": "OAIY Voice", "installed": true, "status": null, "heard": false }]), "{nodes}");
    // (and given back, nothing heard)
    let (status, v) = call(&p, "PUT", "/api/network/settings", Some(json!({ "services": { "oaiy-voice": null } }))).await;
    assert_eq!((status, &v["services"]), (StatusCode::OK, &json!({})), "{v}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A pairing approved on the worker whose primary never says yes on its side (it restarted first, its pairing gone
/// with it) is undone there once the pairing's time is up; one whose primary did say yes (its heartbeat came) stands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_approval_never_confirmed_by_the_primary_is_undone_on_the_worker() {
    let worker = node("worker-unconfirmed", Role::Worker, Some(fake_engines().await)).await;
    let primary = node("primary-unconfirmed", Role::Primary, None).await;
    let (p, w) = (router(primary.clone()), router(worker.clone()));
    let (_, started) = call(&p, "POST", "/api/network/pairings/start", Some(json!({ "address": worker.listening().unwrap().to_string() }))).await;
    let (_, waiting) = call(&w, "GET", "/api/network/requests", None).await;
    call(&w, "POST", &format!("/api/network/requests/{}/approve", waiting["requests"][0]["id"].as_str().unwrap()), None).await;
    let (_, there) = call(&p, "POST", "/api/network/pairings/confirm", Some(json!({ "pairingId": started["pairingId"] }))).await;
    assert_eq!(there["status"], "approved_there");
    // (the primary restarts: its pairing in progress is gone, and it never asks again)
    lock(&primary.inner().outgoing).clear();
    let age = |n: &Network| {
        for (_, at) in lock(&n.inner().unconfirmed).values_mut() {
            *at = at.checked_sub(worker::UNCONFIRMED + std::time::Duration::from_secs(1)).unwrap();
        }
    };
    worker::drop_unconfirmed(&worker);
    assert_eq!(call(&w, "GET", "/api/network/nodes", None).await.1["nodes"].as_array().unwrap().len(), 1, "within its time it stands");
    age(&worker);
    worker::drop_unconfirmed(&worker);
    let (_, nodes) = call(&w, "GET", "/api/network/nodes", None).await;
    assert!(nodes["nodes"].as_array().unwrap().is_empty(), "undone there: {nodes}");
    // confirmed on the primary (its heartbeat heard at once): it stands past that time
    pair(&primary, &worker).await;
    age(&worker);
    worker::drop_unconfirmed(&worker);
    assert_eq!(call(&w, "GET", "/api/network/nodes", None).await.1["nodes"].as_array().unwrap().len(), 1, "a primary heard from stands");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// Paired both ways where this computer does not lend itself after all: paired one way, the answer saying why.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_ways_not_done_is_said() {
    let worker = node("worker-oneway", Role::Worker, None).await;
    let primary = node("primary-oneway", Role::Primary, None).await;
    let answer = pair_answer(&primary, &worker, true).await;
    assert_eq!(answer["bothWays"], false, "{answer}");
    assert!(answer["bothWaysError"].as_str().unwrap_or_default().contains("not lending itself"), "{answer}");
    let node_id = answer["node"]["nodeId"].as_str().unwrap();
    assert!(primary.inner().peers().get(node_id, Role::Worker).is_some() && primary.inner().peers().get(node_id, Role::Primary).is_none());
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// What a worker's engines say reaches the primary as that computer's, by its name, not as the primary's own.
#[tokio::test]
async fn a_workers_errors_name_it() {
    let worker = node("worker-named", Role::Worker, None).await;
    let primary = node("primary-named", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let (status, v) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/v1/chat/completions"), Some(json!({ "messages": [] }))).await;
    assert_eq!((status, v["error"]["message"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("the worker-named's engines are not running")), "{v}");
    let (status, v) = call(&p, "POST", &format!("/api/network/nodes/{node_id}/gateway/v1/images/generations"), Some(json!({}))).await;
    assert_eq!((status, v["error"]["message"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("the worker-named's engines are not running")), "{v}");
    let (_, nodes) = call(&p, "GET", "/api/network/nodes", None).await;
    assert_eq!(nodes["nodes"][0]["enginesError"], "the worker-named's engines are not running", "{nodes}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A computer that never answers is said to be one that could not be connected to, not one that is slow; one that
/// answers slowly is.
#[test]
fn a_computer_not_reached_is_not_called_slow() {
    let at = Some("192.168.50.20:17975");
    assert_eq!(primary::failure_words(true, true, "timed out", at), "could not connect to 192.168.50.20:17975 (is that computer on, awake and on this network?)");
    assert_eq!(primary::failure_words(true, false, "No route to host (os error 65)", at), "could not connect to 192.168.50.20:17975 (is that computer on, awake and on this network?)");
    assert!(primary::failure_words(true, false, "Connection refused (os error 61)", at).starts_with("connection refused at 192.168.50.20:17975"));
    assert_eq!(primary::failure_words(false, true, "timed out", at), "it took too long to answer");
    assert_eq!(primary::failure_words(true, false, "the computer answering is not the one paired (its certificate differs)", at), "the computer answering is not the one paired (its certificate differs)");
}

/// Engines that take their time to answer (a busy computer): the worker's heartbeat answer waits for them a few
/// seconds at most, so the primary sees it online, its engines said to be slow, rather than offline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_busy_workers_heartbeat_answers_in_time() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let slow = format!("http://{}", listener.local_addr().unwrap());
    let app = Router::new().fallback(|| async {
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        axum::Json(json!({}))
    });
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let worker = node("worker-busy", Role::Worker, Some(slow)).await;
    let primary = node("primary-busy", Role::Primary, None).await;
    let started = std::time::Instant::now();
    let node_id = pair(&primary, &worker).await;
    assert!(started.elapsed() < std::time::Duration::from_secs(3), "answered within its wait: {:?}", started.elapsed());
    let (_, nodes) = call(&router(primary.clone()), "GET", "/api/network/nodes", None).await;
    let n = nodes["nodes"].as_array().unwrap().iter().find(|n| n["nodeId"] == json!(node_id)).cloned().unwrap();
    assert_eq!(n["online"], true, "{n}");
    assert!(n["enginesError"].as_str().unwrap_or_default().contains("did not answer within"), "{n}");
    for n in [&worker, &primary] {
        n.stop();
    }
}

/// A run on a worker waited for from here keeps being followed through a few polls that go unanswered (that computer
/// gone for a moment); given up after more, the error keeps its id and key. Asked again with its key, the same run is
/// answered, never another started.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_run_waited_for_on_a_worker_is_not_lost_or_run_twice() {
    let worker = node("worker-runs", Role::Worker, Some(fake_engines().await)).await;
    let flows = std::sync::Arc::new(crate::bridge::worker::FlowStore::new(temp("flows-runs")));
    flows.put("logo", &json!({ "name": "Make a logo", "oaiyRemote": true, "nodes": [] }).to_string()).unwrap();
    let ledger = crate::bridge::ledger::new_handle();
    worker.set_bridge(ledger.clone(), flows);
    let primary = node("primary-runs", Role::Primary, None).await;
    let node_id = pair(&primary, &worker).await;
    let p = router(primary.clone());
    let runs = format!("/api/network/nodes/{node_id}/flows/logo/runs");
    // the same key twice: one run
    let (_, first) = call(&p, "POST", &runs, Some(json!({ "key": "k-1" }))).await;
    let (status, again) = call(&p, "POST", &runs, Some(json!({ "key": "k-1" }))).await;
    assert_eq!((status, &again["runId"]), (StatusCode::OK, &first["runId"]), "{again}");
    let (_, other) = call(&p, "POST", &runs, Some(json!({}))).await;
    assert_ne!(other["runId"], first["runId"]);
    // waited for: that computer goes for a moment, the run finishes there meanwhile, and it is answered once back
    let port = worker.listening().unwrap().port();
    let waiting = {
        let p = p.clone();
        let runs = runs.clone();
        tokio::spawn(async move { call(&p, "POST", &runs, Some(json!({ "key": "k-2", "wait": true, "timeoutMs": 60000 }))).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    worker.stop();
    let run_id = ledger.lock().unwrap().recent(64, &[crate::bridge::ledger::RunStatus::Queued]).iter().find(|r| r.idempotency_key.ends_with(":k-2")).map(|r| r.run_id.clone()).unwrap();
    {
        let mut l = ledger.lock().unwrap();
        l.claim(&run_id, crate::bridge::ledger::Runtime::Desktop, "test");
        l.finish(&run_id, crate::bridge::ledger::RunStatus::Succeeded, Some(json!({ "output": { "text": "done" } })), None).unwrap();
    }
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    worker.set(Settings { port, ..worker.settings() }).await.unwrap();
    let (status, done) = waiting.await.unwrap();
    assert_eq!((status, &done["status"], &done["runId"]), (StatusCode::OK, &json!("succeeded"), &json!(run_id)), "{done}");
    // gone for good: given up after a few misses, the error keeping the run's id and its key
    let waiting = {
        let p = p.clone();
        let runs = runs.clone();
        tokio::spawn(async move { call(&p, "POST", &runs, Some(json!({ "key": "k-3", "wait": true, "timeoutMs": 60000 }))).await })
    };
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;
    worker.stop();
    let (status, lost) = waiting.await.unwrap();
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{lost}");
    assert!(lost["runId"].is_string() && lost["key"] == "k-3" && lost["run"]["status"] == "queued", "{lost}");
    assert!(lost["error"]["message"].as_str().unwrap().contains("may still be running there"), "{lost}");
    // back, and asked again with its key: that run, not another
    worker.set(Settings { port, ..worker.settings() }).await.unwrap();
    let (_, joined) = call(&p, "POST", &runs, Some(json!({ "key": "k-3" }))).await;
    assert_eq!(joined["runId"], lost["runId"], "{joined}");
    for n in [&worker, &primary] {
        n.stop();
    }
}
