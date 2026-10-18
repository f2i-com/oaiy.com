//! Model files lent between paired computers: a computer getting a model (its engines' Get models, Setup, the Agent)
//! fetches a file from a computer it uses that has it, over the LAN, before Hugging Face is asked. Files are named by
//! their SHA-256 alone (as Hugging Face publishes it), never by a path: a computer serves only a model file of its
//! engines' (where downloads go, the folders of the models in its settings) whose SHA-256 it has read to be the one
//! asked for, and the computer fetching it checks it again (docs/NETWORK.md).
//!
//! The computer that lends: `POST /worker/v1/models/have` (which of these it has, from what it knows already) and
//! `GET /worker/v1/models/blob/:sha256?size=` (the file, from a byte on with `Range`). Its person can switch it off
//! (`shareModels`).

use std::path::{Path, PathBuf};
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
/// known yet. `(sha256, size) -> path, checked`.
async fn find(net: &Network, files: &[Wanted], check: bool) -> Vec<Option<(PathBuf, bool)>> {
    let body = json!({ "files": files.iter().map(|f| json!({ "sha256": f.sha256, "size": f.size })).collect::<Vec<_>>(), "check": check });
    // (reading a large file's SHA-256 takes a while: tens of seconds for a model's weights)
    let wait = if check { Duration::from_secs(600) } else { Duration::from_secs(20) };
    let found = super::machine::engines_post(net, "/api/library/find", &body, wait).await;
    let listed = found.get("files").and_then(Value::as_array).cloned().unwrap_or_default();
    files
        .iter()
        .map(|f| {
            listed
                .iter()
                .find(|x| x["sha256"].as_str() == Some(f.sha256.as_str()) && x["size"].as_u64() == Some(f.size))
                .and_then(|x| Some((PathBuf::from(x["path"].as_str()?), x["checked"].as_bool() == Some(true))))
        })
        .collect()
}

/// `POST /worker/v1/models/have {files: [{sha256, size}]}`: which of them this computer has (from what it knows
/// already: checked when one is fetched), `{files: [{sha256, size, have}]}`. No path is said.
pub(super) async fn have(State(net): State<Network>, Extension(caller): Extension<Caller>, Json(asked): Json<Have>) -> Response {
    if let Err(r) = lends_files_to(&net, &caller) {
        return r;
    }
    let files: Vec<Wanted> = asked.files.into_iter().take(MAX_ASKED).filter(|f| sha_ok(&f.sha256)).map(|f| Wanted { sha256: f.sha256.to_ascii_lowercase(), size: f.size }).collect();
    let found = find(&net, &files, false).await;
    Json(json!({ "files": files.iter().zip(found).map(|(f, at)| json!({ "sha256": f.sha256, "size": f.size, "have": at.is_some() })).collect::<Vec<_>>() })).into_response()
}

#[derive(Deserialize)]
pub(super) struct BlobQuery {
    size: u64,
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
    let Some(Some((path, true))) = find(&net, &[Wanted { sha256: sha.clone(), size: q.size }], true).await.into_iter().next() else {
        return refuse(StatusCode::NOT_FOUND, "not_found", "this computer has no such model file");
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
    if size != q.size || from > size {
        return refuse(StatusCode::RANGE_NOT_SATISFIABLE, "range", "no such bytes of that file");
    }
    if from > 0 {
        use tokio::io::AsyncSeekExt;
        if file.seek(std::io::SeekFrom::Start(from)).await.is_err() {
            return refuse(StatusCode::INTERNAL_SERVER_ERROR, "internal", "the file could not be read");
        }
    }
    log::info!("network: lending {} its model file {}… ({} bytes from {from})", p.name, &sha[..12], size);
    let stream = futures_util::stream::unfold(file, |mut file| async move {
        use tokio::io::AsyncReadExt;
        let mut buf = vec![0u8; 1 << 20];
        match file.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok::<_, std::io::Error>(axum::body::Bytes::from(buf)), file))
            }
            Err(e) => Some((Err(e), file)),
        }
    });
    let mut r = axum::body::Body::from_stream(stream).into_response();
    *r.status_mut() = if from > 0 { StatusCode::PARTIAL_CONTENT } else { StatusCode::OK };
    let h = r.headers_mut();
    h.insert("content-type", "application/octet-stream".parse().expect("a header"));
    h.insert("content-length", (size - from).to_string().parse().expect("a header"));
    if from > 0 {
        if let Ok(v) = format!("bytes {from}-{}/{size}", size.saturating_sub(1)).parse() {
            h.insert("content-range", v);
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------------------------
// The computer getting a model: who has a file, and fetching it from them
// ---------------------------------------------------------------------------------------------------------------

/// The computers this one uses that answer now, nearest asked first (none while it does not use others).
fn lenders(net: &Network) -> Vec<Peer> {
    if !net.settings().uses() {
        return Vec::new();
    }
    let online: Vec<String> = lock(&net.inner().nodes).iter().filter(|(_, s)| s.online).map(|(id, _)| id.clone()).collect();
    net.inner().peers().list().iter().filter(|p| p.role == Role::Worker && online.contains(&p.node_id)).cloned().collect()
}

/// For each of `files` (SHA-256, bytes), a computer this one uses that has it, from what each knows already.
pub(crate) async fn who_has(net: &Network, files: &[(String, u64)]) -> Vec<Option<Peer>> {
    let mut out: Vec<Option<Peer>> = vec![None; files.len()];
    let Ok(id) = net.identity() else { return out };
    let body = json!({ "files": files.iter().map(|(s, n)| json!({ "sha256": s, "size": n })).collect::<Vec<_>>() });
    for p in lenders(net) {
        if out.iter().all(Option::is_some) {
            break;
        }
        let Ok(client) = client_for(&id, &p.fingerprint) else { continue };
        let Ok(r) = client.post(url(&p.address, "/worker/v1/models/have")).json(&body).timeout(Duration::from_secs(30)).send().await else { continue };
        let Ok(v) = r.json::<Value>().await else { continue };
        for (i, f) in v.get("files").and_then(Value::as_array).into_iter().flatten().enumerate().take(files.len()) {
            if out[i].is_none() && f["have"] == json!(true) && f["sha256"].as_str() == Some(files[i].0.as_str()) {
                out[i] = Some(p.clone());
            }
        }
    }
    out
}

/// Fetch file `sha256` (`size` bytes) into `part`, carrying on from what it holds, from a computer this one uses that
/// has it: its name once `part` is whole, `None` where none has it (or `stop` said to). `progress`: the bytes in
/// `part` as it grows. The caller checks the whole file's SHA-256.
pub(crate) async fn fetch(net: &Network, sha256: &str, size: u64, part: &Path, progress: &(dyn Fn(u64) + Sync), stop: &(dyn Fn() -> bool + Sync)) -> Result<Option<String>, String> {
    use tokio::io::AsyncWriteExt;
    let Some(Some(peer)) = who_has(net, &[(sha256.to_string(), size)]).await.into_iter().next() else { return Ok(None) };
    let id = net.identity()?;
    let client = client_for(&id, &peer.fingerprint)?;
    let mut have = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
    if have > size {
        let _ = tokio::fs::remove_file(part).await;
        have = 0;
    }
    if have < size {
        let mut ask = client.get(url(&peer.address, &format!("/worker/v1/models/blob/{sha256}?size={size}")));
        if have > 0 {
            ask = ask.header("range", format!("bytes={have}-"));
        }
        let mut r = ask.send().await.map_err(|e| format!("{}: {e}", peer.name))?;
        if !r.status().is_success() {
            return Err(format!("{} answered HTTP {}", peer.name, r.status()));
        }
        // (a computer that sent the whole file in place of the rest: started again)
        if have > 0 && r.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            have = 0;
        }
        if let Some(parent) = part.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| e.to_string())?;
        }
        let mut out = tokio::fs::OpenOptions::new().create(true).write(true).append(have > 0).truncate(have == 0).open(part).await.map_err(|e| format!("{}: {e}", part.display()))?;
        while let Some(chunk) = r.chunk().await.map_err(|e| format!("{}: {e}", peer.name))? {
            out.write_all(&chunk).await.map_err(|e| format!("{}: {e}", part.display()))?;
            have += chunk.len() as u64;
            progress(have);
            if stop() {
                out.flush().await.map_err(|e| e.to_string())?;
                return Ok(None);
            }
        }
        out.flush().await.map_err(|e| e.to_string())?;
    }
    if have != size {
        return Err(format!("{} sent {have} bytes of {size}", peer.name));
    }
    Ok(Some(peer.name))
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
            loop {
                match rx.recv_timeout(Duration::from_millis(400)) {
                    Ok(n) => progress(n),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if stop() {
                    stopping.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }
            self.rt.block_on(task).map_err(|e| e.to_string())?
        }
    }
    let Ok(rt) = tokio::runtime::Handle::try_current() else { return };
    oaiy_studio::downloads::set_peer_files(Some(std::sync::Arc::new(Lan { net, rt })));
}
