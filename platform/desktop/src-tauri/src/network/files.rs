//! Model files lent between paired computers: a computer getting a model (its engines' Get models, Setup, the Agent)
//! fetches a file from a computer it uses that has it, over the LAN, before Hugging Face is asked. Files are named by
//! their SHA-256 alone (as Hugging Face publishes it), never by a path: a computer serves only a model file of its
//! engines' (where downloads go, the folders of the models in its settings) whose SHA-256 it has read to be the one
//! asked for, and the computer fetching it checks it again (docs/NETWORK.md).
//!
//! The computer that lends: `POST /worker/v1/models/have` (which of these it has, from what it knows already, and
//! whether that is checked) and `GET /worker/v1/models/blob/:sha256?size=` (the file, from a byte on with `Range`).
//! Its person can switch it off (`shareModels`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::{Extension, Path as UrlPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::primary::{client_for, url};
use super::store::{Peer, Role};
use super::worker::{primary_of, refuse, Caller};
use super::{lock, Network};

/// Files asked about at once.
const MAX_ASKED: usize = 256;
/// How long the computers lending are asked which files they have, all at once, before what they said is used.
const ASK_BUDGET: Duration = Duration::from_secs(4);
/// A file's bytes not arriving for this long, once they started: the lender is given up on.
const STALLED: Duration = Duration::from_secs(60);
/// How often a transfer looks whether it is to stop, while it waits on the lender.
const STOP_POLL: Duration = Duration::from_millis(300);
/// Files sent at once to one computer.
const SENDS_PER_COMPUTER: usize = 2;

#[derive(Deserialize)]
pub(super) struct Have {
    files: Vec<Wanted>,
}

#[derive(Clone, Deserialize)]
struct Wanted {
    sha256: String,
    size: u64,
}

fn sha_ok(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether this computer lends its model files to `caller` (a computer paired to use it, its person sharing them).
fn lends_files_to(net: &Network, caller: &Caller) -> Result<Peer, Response> {
    let Some(p) = primary_of(net, caller) else {
        return Err(refuse(StatusCode::FORBIDDEN, "not_paired", "this computer is not paired with the caller"));
    };
    if !net.settings().shares_models() {
        return Err(refuse(StatusCode::FORBIDDEN, "not_shared", "this computer does not share its model files (Share model files is off there)"));
    }
    Ok(p)
}

/// Where this computer's engines keep the files, by SHA-256 and size: `check` reads a file's SHA-256 where it is not
/// known yet. `(path, checked)` for each file, matched by its SHA-256 and size; `None` where the engines did not answer
/// (in time: reading a large file's SHA-256 takes a while).
async fn find(net: &Network, files: &[Wanted], check: bool) -> Option<Vec<Option<(PathBuf, bool)>>> {
    let body = json!({ "files": files.iter().map(|f| json!({ "sha256": f.sha256, "size": f.size })).collect::<Vec<_>>(), "check": check });
    let wait = if check { Duration::from_secs(600) } else { Duration::from_secs(20) };
    let found = super::machine::engines_post(net, "/api/library/find", &body, wait).await;
    let listed = found.get("files").and_then(Value::as_array)?.clone();
    Some(
        files
            .iter()
            .map(|f| {
                listed
                    .iter()
                    .find(|x| x["sha256"].as_str() == Some(f.sha256.as_str()) && x["size"].as_u64() == Some(f.size))
                    .and_then(|x| Some((PathBuf::from(x["path"].as_str()?), x["checked"].as_bool() == Some(true))))
            })
            .collect(),
    )
}

/// `POST /worker/v1/models/have {files: [{sha256, size}]}`: which of them this computer has, from what it knows
/// already, and whether that is checked (its SHA-256 read; else a file as large, read when it is asked for),
/// `{files: [{sha256, size, have, checked}]}`. No path is said.
pub(super) async fn have(State(net): State<Network>, Extension(caller): Extension<Caller>, Json(asked): Json<Have>) -> Response {
    if let Err(r) = lends_files_to(&net, &caller) {
        return r;
    }
    let files: Vec<Wanted> = asked.files.into_iter().take(MAX_ASKED).filter(|f| sha_ok(&f.sha256)).map(|f| Wanted { sha256: f.sha256.to_ascii_lowercase(), size: f.size }).collect();
    let Some(found) = find(&net, &files, false).await else {
        return refuse(StatusCode::SERVICE_UNAVAILABLE, "engines_unavailable", "this computer's engines did not answer");
    };
    Json(json!({
        "files": files.iter().zip(found).map(|(f, at)| json!({ "sha256": f.sha256, "size": f.size, "have": at.is_some(), "checked": at.is_some_and(|(_, c)| c) })).collect::<Vec<_>>()
    }))
    .into_response()
}

#[derive(Deserialize)]
pub(super) struct BlobQuery {
    size: u64,
}

/// One reading of a file's SHA-256 at a time, by SHA-256: a second ask waits for the first and finds it known.
fn checking(sha: &str) -> Arc<tokio::sync::Mutex<()>> {
    static ONE: Mutex<Option<HashMap<String, Arc<tokio::sync::Mutex<()>>>>> = Mutex::new(None);
    let mut all = lock(&ONE);
    let map = all.get_or_insert_with(HashMap::new);
    // (none kept for long: those no one holds now are let go)
    map.retain(|_, m| Arc::strong_count(m) > 1);
    map.entry(sha.to_string()).or_default().clone()
}

/// Files sent at once to the computer with this certificate.
fn sends_to(fingerprint: &str) -> Arc<tokio::sync::Semaphore> {
    static SENDS: Mutex<Option<HashMap<String, Arc<tokio::sync::Semaphore>>>> = Mutex::new(None);
    let mut all = lock(&SENDS);
    all.get_or_insert_with(HashMap::new).entry(fingerprint.to_string()).or_insert_with(|| Arc::new(tokio::sync::Semaphore::new(SENDS_PER_COMPUTER))).clone()
}

/// `GET /worker/v1/models/blob/:sha256?size=`: the model file whose SHA-256 that is (read here before it is sent),
/// from the byte `Range: bytes=N-` names on.
pub(super) async fn blob(State(net): State<Network>, Extension(caller): Extension<Caller>, UrlPath(sha): UrlPath<String>, Query(q): Query<BlobQuery>, headers: HeaderMap) -> Response {
    let p = match lends_files_to(&net, &caller) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let sha = sha.to_ascii_lowercase();
    if !sha_ok(&sha) {
        return refuse(StatusCode::BAD_REQUEST, "invalid_request", "a file is asked for by its SHA-256");
    }
    let Ok(sending) = sends_to(&p.fingerprint).try_acquire_owned() else {
        return refuse(StatusCode::TOO_MANY_REQUESTS, "busy", "this computer is sending that computer as many files as it sends at once");
    };
    let found = {
        let one = checking(&sha);
        let _reading = one.lock().await;
        find(&net, &[Wanted { sha256: sha.clone(), size: q.size }], true).await
    };
    let path = match found.map(|f| f.into_iter().next().flatten()) {
        Some(Some((path, true))) => path,
        Some(_) => return refuse(StatusCode::NOT_FOUND, "not_found", "this computer has no such model file"),
        None => return refuse(StatusCode::SERVICE_UNAVAILABLE, "still_reading", "this computer's engines did not answer in time (a large file's SHA-256 is still being read): ask again shortly"),
    };
    let from = headers
        .get("range")
        .and_then(|v| v.to_str().ok())
        .and_then(|r| r.strip_prefix("bytes="))
        .and_then(|r| r.strip_suffix('-'))
        .and_then(|n| n.parse::<u64>().ok())
        .unwrap_or(0);
    let Ok(mut file) = tokio::fs::File::open(&path).await else {
        return refuse(StatusCode::NOT_FOUND, "not_found", "the file is gone");
    };
    let size = file.metadata().await.map(|m| m.len()).unwrap_or(0);
    if size != q.size || (from > 0 && from >= size) {
        return refuse(StatusCode::RANGE_NOT_SATISFIABLE, "range", "no such bytes of that file");
    }
    if from > 0 {
        use tokio::io::AsyncSeekExt;
        if file.seek(std::io::SeekFrom::Start(from)).await.is_err() {
            return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "the file could not be read");
        }
    }
    log::info!("network: lending {} its model file {}… ({} bytes from {from})", p.name, &sha[..12], size);
    // (the send counted until its last byte has gone)
    let stream = futures_util::stream::unfold((file, sending), |(mut file, sending)| async move {
        use tokio::io::AsyncReadExt;
        let mut buf = vec![0u8; 1 << 20];
        match file.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok::<_, std::io::Error>(axum::body::Bytes::from(buf)), (file, sending)))
            }
            Err(e) => Some((Err(e), (file, sending))),
        }
    });
    let mut r = axum::body::Body::from_stream(stream).into_response();
    *r.status_mut() = if from > 0 { StatusCode::PARTIAL_CONTENT } else { StatusCode::OK };
    let h = r.headers_mut();
    h.insert("content-type", "application/octet-stream".parse().expect("a header"));
    h.insert("content-length", (size - from).to_string().parse().expect("a header"));
    if from > 0 {
        if let Ok(v) = format!("bytes {from}-{}/{size}", size - 1).parse() {
            h.insert("content-range", v);
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------------------------
// The computer getting a model: who has a file, and fetching it from them
// ---------------------------------------------------------------------------------------------------------------

/// The computers this one uses that answer now (none while it does not use others), in the order they were paired.
fn lenders(net: &Network) -> Vec<Peer> {
    if !net.settings().uses() {
        return Vec::new();
    }
    let online: Vec<String> = lock(&net.inner().nodes).iter().filter(|(_, s)| s.online).map(|(id, _)| id.clone()).collect();
    net.inner().peers().list().iter().filter(|p| p.role == Role::Worker && online.contains(&p.node_id)).cloned().collect()
}

/// For each of `files` (SHA-256, bytes), the computers this one uses that say they have it, those that checked it
/// first: all asked at once, what answered within a few seconds kept.
async fn holders(net: &Network, files: &[(String, u64)]) -> Vec<Vec<Peer>> {
    let mut out: Vec<Vec<(Peer, bool)>> = vec![Vec::new(); files.len()];
    let lenders = lenders(net);
    if lenders.is_empty() || files.is_empty() {
        return vec![Vec::new(); files.len()];
    }
    let Ok(id) = net.identity() else { return vec![Vec::new(); files.len()] };
    let body = json!({ "files": files.iter().map(|(s, n)| json!({ "sha256": s.to_ascii_lowercase(), "size": n })).collect::<Vec<_>>() });
    let asks = lenders.into_iter().map(|p| {
        let (id, body) = (id.clone(), body.clone());
        async move {
            let client = client_for(&id, &p.fingerprint).ok()?;
            let r = client.post(url(&p.address, "/worker/v1/models/have")).json(&body).timeout(ASK_BUDGET).send().await.ok()?;
            let v = r.json::<Value>().await.ok()?;
            Some((p, v))
        }
    });
    let answers = tokio::time::timeout(ASK_BUDGET + Duration::from_millis(500), futures_util::future::join_all(asks)).await.unwrap_or_default();
    for (p, v) in answers.into_iter().flatten() {
        for f in v.get("files").and_then(Value::as_array).into_iter().flatten().filter(|f| f["have"] == json!(true)) {
            // (matched by SHA-256 and size, never by where in the answer it is)
            let at = files.iter().position(|(s, n)| f["sha256"].as_str().is_some_and(|x| x.eq_ignore_ascii_case(s)) && f["size"].as_u64() == Some(*n));
            if let Some(i) = at {
                out[i].push((p.clone(), f["checked"] == json!(true)));
            }
        }
    }
    out.into_iter()
        .map(|mut l| {
            l.sort_by_key(|(_, checked)| !checked);
            l.into_iter().map(|(p, _)| p).collect()
        })
        .collect()
}

/// For each of `files` (SHA-256, bytes), a computer this one uses that has it (one that checked it before one that
/// has a file as large not read yet).
pub(crate) async fn who_has(net: &Network, files: &[(String, u64)]) -> Vec<Option<Peer>> {
    holders(net, files).await.into_iter().map(|l| l.into_iter().next()).collect()
}

/// What `fut` gives, or `None` once `stop` says to stop (asked every few hundred milliseconds), or `Err` past `idle`.
async fn waiting<T>(fut: impl std::future::Future<Output = T>, stop: &(dyn Fn() -> bool + Sync), idle: Option<Duration>) -> Result<Option<T>, ()> {
    tokio::pin!(fut);
    let started = tokio::time::Instant::now();
    loop {
        tokio::select! {
            out = &mut fut => return Ok(Some(out)),
            _ = tokio::time::sleep(STOP_POLL) => {
                if stop() {
                    return Ok(None);
                }
                if idle.is_some_and(|i| started.elapsed() > i) {
                    return Err(());
                }
            }
        }
    }
}

/// Fetch file `sha256` (`size` bytes) into `part`, carrying on from what it holds, from a computer this one uses that
/// has it (the next one where one sends nothing, or not the file): its name once `part` is whole, `None` where none has
/// it (or `stop` said to). `progress`: the bytes in `part` as it grows. The caller checks the whole file's SHA-256.
pub(crate) async fn fetch(net: &Network, sha256: &str, size: u64, part: &Path, progress: &(dyn Fn(u64) + Sync), stop: &(dyn Fn() -> bool + Sync)) -> Result<Option<String>, String> {
    let lenders = holders(net, &[(sha256.to_string(), size)]).await.into_iter().next().unwrap_or_default();
    if lenders.is_empty() {
        return Ok(None);
    }
    let id = net.identity()?;
    let mut last = None;
    for peer in lenders {
        match fetch_from(&id, &peer, sha256, size, part, progress, stop).await {
            Ok(true) => return Ok(Some(peer.name)),
            Ok(false) => return Ok(None),
            Err(e) => {
                log::warn!("network: model file {}… not from {}: {e}", &sha256[..12.min(sha256.len())], peer.name);
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| "no computer sent it".into()))
}

/// [`fetch`] from one computer: `true` once `part` is whole, `false` where `stop` said to stop.
async fn fetch_from(id: &super::identity::Identity, peer: &Peer, sha256: &str, size: u64, part: &Path, progress: &(dyn Fn(u64) + Sync), stop: &(dyn Fn() -> bool + Sync)) -> Result<bool, String> {
    use tokio::io::AsyncWriteExt;
    let client = client_for(id, &peer.fingerprint)?;
    let mut have = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
    if have > size {
        let _ = tokio::fs::remove_file(part).await;
        have = 0;
    }
    if have == size {
        return Ok(true);
    }
    let mut ask = client.get(url(&peer.address, &format!("/worker/v1/models/blob/{sha256}?size={size}")));
    if have > 0 {
        ask = ask.header("range", format!("bytes={have}-"));
    }
    // (the lender reads the file's SHA-256 before it answers: minutes for a large one, stopped meanwhile if asked)
    let Some(answer) = waiting(ask.send(), stop, None).await.map_err(|_| "it stalled".to_string())? else { return Ok(false) };
    let mut r = answer.map_err(|e| format!("{}: {e}", peer.name))?;
    if !r.status().is_success() {
        return Err(format!("{} answered HTTP {}", peer.name, r.status()));
    }
    // (what it sends is the rest of the file asked for, or the whole of it: anything else is not taken)
    let header = |k: &str| r.headers().get(k).and_then(|v| v.to_str().ok()).map(str::to_owned);
    if r.status() == reqwest::StatusCode::PARTIAL_CONTENT {
        let range = header("content-range").unwrap_or_default();
        if have == 0 || range != format!("bytes {have}-{}/{size}", size - 1) {
            return Err(format!("{} sent other bytes than asked for ({range})", peer.name));
        }
    } else {
        have = 0;
    }
    if header("content-length").and_then(|n| n.parse::<u64>().ok()).is_some_and(|n| n != size - have) {
        return Err(format!("{} said it sends another length than the file has", peer.name));
    }
    if let Some(parent) = part.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
    }
    let mut out = tokio::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(part).await.map_err(|e| format!("{}: {e}", part.display()))?;
    loop {
        let chunk = match waiting(r.chunk(), stop, Some(STALLED)).await {
            Ok(Some(c)) => c.map_err(|e| format!("{}: {e}", peer.name))?,
            Ok(None) => {
                out.flush().await.map_err(|e| e.to_string())?;
                return Ok(false);
            }
            Err(()) => return Err(format!("{} stopped sending for a minute", peer.name)),
        };
        let Some(chunk) = chunk else { break };
        if have + chunk.len() as u64 > size {
            // (more than the file has: not the file; what it sent is not kept)
            drop(out);
            let _ = tokio::fs::remove_file(part).await;
            return Err(format!("{} sent more than the file has", peer.name));
        }
        out.write_all(&chunk).await.map_err(|e| format!("{}: {e}", part.display()))?;
        have += chunk.len() as u64;
        progress(have);
    }
    out.flush().await.map_err(|e| e.to_string())?;
    if have != size {
        return Err(format!("{} sent {have} bytes of {size}", peer.name));
    }
    Ok(true)
}

/// The engines' downloads, getting model files from the computers this one uses first (the desktop's own engines).
#[cfg(feature = "gui")]
pub fn lend_to_engines(net: Network) {
    struct Lan {
        net: Network,
        rt: tokio::runtime::Handle,
    }
    impl oaiy_studio::downloads::PeerFiles for Lan {
        fn have(&self, files: &[(String, u64)]) -> Vec<Option<String>> {
            self.rt.block_on(who_has(&self.net, files)).into_iter().map(|p| p.map(|p| p.name)).collect()
        }
        fn fetch(&self, sha256: &str, size: u64, part: &Path, progress: &dyn Fn(u64), stop: &dyn Fn() -> bool) -> Result<Option<String>, String> {
            // (the engines' own thread: its callbacks stay on it, the transfer runs on the desktop's runtime)
            let (tx, rx) = std::sync::mpsc::channel::<u64>();
            let stopping = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (net, sha, part_owned, flag) = (self.net.clone(), sha256.to_string(), part.to_path_buf(), stopping.clone());
            let task = self.rt.spawn(async move {
                let progress = move |n: u64| {
                    let _ = tx.send(n);
                };
                let stop = move || flag.load(std::sync::atomic::Ordering::SeqCst);
                fetch(&net, &sha, size, &part_owned, &progress, &stop).await
            });
            let mut stop_asked: Option<std::time::Instant> = None;
            loop {
                match rx.recv_timeout(Duration::from_millis(300)) {
                    Ok(n) => progress(n),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if stop() && stop_asked.is_none() {
                    stopping.store(true, std::sync::atomic::Ordering::SeqCst);
                    stop_asked = Some(std::time::Instant::now());
                }
                // (a transfer that does not stop within a few seconds of being told is ended)
                if stop_asked.is_some_and(|at| at.elapsed() > Duration::from_secs(3)) {
                    task.abort();
                    return Ok(None);
                }
            }
            self.rt.block_on(task).map_err(|e| e.to_string())?
        }
    }
    let Ok(rt) = tokio::runtime::Handle::try_current() else { return };
    oaiy_studio::downloads::set_peer_files(Some(std::sync::Arc::new(Lan { net, rt })));
}
