//! The Playground's requests for another OAIY computer's models (OAIY Network): `/api/play/node/<id>/<target>[/rest]`
//! on the control port, for this computer's own window alone. Each is sent to that computer through the desktop that
//! runs the studio ([`PeerModels`]), and answered as that computer answers it: a chat's reply streamed as it comes, a
//! job's polls and its content, and the files it made, whose addresses that computer names at this route.

use crate::api::gateway::{incognito_mode, json_reply};
use crate::models::downloads::{PeerModel, PeerModels, PeerReply, PeerRequest};
use crate::util::{error_json, str_or};
use crate::Studio;
use oaiy_engine::http::{respond_head, Request, Stream};
use oaiy_engine::json::Json;
use std::io::{self, Read, Write};
use std::net::TcpStream;

/// The Playground's targets that reach a paired computer, each with the kind of model it is for (none: its files).
/// An edit is a picture request with its pictures in it, which that computer's picture route takes.
const TARGETS: [(&str, &str); 9] = [
    ("chat", "llm"),
    ("images", "image"),
    ("edits", "image"),
    ("videos", "video"),
    ("speech", "speech"),
    ("music", "music"),
    ("sound", "sound"),
    ("model3d", "model3d"),
    ("files", ""),
];

/// The headers of that computer's answer passed on with a body of known length (its type and length are said anyway).
const KEPT: [&str; 5] = ["Content-Range", "Accept-Ranges", "Content-Disposition", "X-Sample-Rate", "Cache-Control"];

fn err(w: &mut TcpStream, status: u16, message: &str) -> io::Result<bool> {
    let (kind, code) = match status {
        404 => ("invalid_request_error", "not_found"),
        409 => ("invalid_request_error", "conflict"),
        s if s >= 500 => ("server_error", "unavailable"),
        _ => ("invalid_request_error", "invalid_request"),
    };
    json_reply(w, status, &error_json(message, kind, code))
}

/// A computer's id as OAIY Network names it.
fn node_id_ok(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// Whether `rest`, the part of a route below a target (`/<job>/content`), names a route below it and nothing else: no
/// empty, `.` or `..` segment (nor one written `%2e`), no slash or backslash written as an escape, no backslash or
/// control character.
fn plain_rest(rest: &str) -> bool {
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix('/') else { return false };
    let lower = rest.to_ascii_lowercase();
    !rest.contains('\\') && !rest.chars().any(char::is_control) && !lower.contains("%2f") && !lower.contains("%5c")
        && rest.split('/').all(|seg| {
            let seg = seg.to_ascii_lowercase().replace("%2e", ".");
            !seg.is_empty() && seg != "." && seg != ".."
        })
}

/// The route on `node`'s gateway for `target`: its models' own, as that computer said them, else this studio's default
/// route for the target (which a computer whose routes were not changed has too).
fn route_on(models: &[PeerModel], node: &str, target: &str, kind: &str) -> String {
    let said = models.iter().filter(|m| m.node_id == node && !kind.is_empty() && m.kind == kind).map(|m| m.path.trim_end_matches('/')).find(|p| p.starts_with('/'));
    if let Some(path) = said {
        return path.to_string();
    }
    // (an edit is a picture request there, as its picture route takes one with its pictures)
    let target = if target == "edits" { "images" } else { target };
    let routes = crate::config::default_json();
    routes
        .get("gateway")
        .and_then(|g| g.get("routes"))
        .and_then(Json::as_array)
        .and_then(|r| r.iter().find(|r| str_or(r, "target", "") == target))
        .map(|r| str_or(r, "path", "").to_string())
        .unwrap_or_default()
}

/// `/api/play/node/<rest>` (`rest` after that), its request asked for by this computer's own window (the caller
/// refuses any other). `peers`: where the paired computers are reached (none: the studio runs without OAIY's desktop).
pub fn handle(studio: &Studio, req: &Request, w: &mut TcpStream, rest: &str, peers: Option<&dyn PeerModels>) -> io::Result<bool> {
    let (node, rest) = rest.split_once('/').unwrap_or((rest, ""));
    let (target, sub) = rest.find('/').map_or((rest, ""), |at| (&rest[..at], &rest[at..]));
    if !node_id_ok(node) {
        return err(w, 404, "not a computer of OAIY Network");
    }
    let Some(&(target, kind)) = TARGETS.iter().find(|(t, _)| *t == target) else {
        return err(w, 404, "unknown playground target");
    };
    if !plain_rest(sub) {
        return err(w, 400, "not one of that computer's routes");
    }
    let method = req.method.as_str();
    if !matches!(method, "GET" | "POST" | "DELETE") {
        return err(w, 405, &format!("{method} is not sent to another computer"));
    }
    // (what that computer keeps of a request is its own settings' to say: incognito here is not asked of it)
    let incognito = req.header("x-oaiy-incognito").is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"));
    if incognito_mode(studio) || incognito {
        return err(w, 409, "Incognito is on: another computer's models are not used from here, as that computer keeps what it makes by its own settings. Pick a model of this computer, or switch incognito off.");
    }
    let Some(peers) = peers else {
        return err(w, 503, "Other computers' models are reached through OAIY's desktop, which does not run these engines");
    };
    let models = peers.models();
    let Some(name) = models.iter().find(|m| m.node_id == node).map(|m| m.node_name.clone()) else {
        return err(w, 404, "That computer is not one this computer uses now, or it lists no models (Setup → Networking)");
    };
    let path = format!("{}{sub}", route_on(&models, node, target, kind));
    if !path.starts_with('/') {
        return err(w, 404, &format!("{name} has no route for {target}"));
    }
    let query = req.path.split_once('?').map(|(_, q)| format!("?{q}")).unwrap_or_default();
    // (the files that computer made are named at this route for it, as this window reached it: fetched through here)
    let base = req.header("host").filter(|h| oaiy_engine::http::loopback_name(h)).map(|h| format!("http://{h}/api/play/node/{node}"));
    let asked = PeerRequest {
        method,
        path: &path,
        query: &query,
        content_type: req.header("content-type"),
        range: req.header("range"),
        base: base.as_deref(),
        body: req.body.clone(),
    };
    match peers.forward(node, asked) {
        Ok(reply) => relay(w, reply),
        Err(e) => err(w, 502, &format!("{name} was not reached: {e}")),
    }
}

/// That computer's answer passed on: a body of known length as it is, with what says what it is (a range of a video
/// as a range); an event stream, or a body of no stated length, chunk by chunk as it comes. A client that leaves makes
/// a write fail, and the body is dropped with this, which ends the request on that computer.
fn relay(w: &mut TcpStream, reply: PeerReply) -> io::Result<bool> {
    let PeerReply { status, content_type, headers, mut body } = reply;
    let header = |name: &str| headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str());
    let length = header("content-length").and_then(|n| n.trim().parse::<u64>().ok());
    match length.filter(|_| !content_type.contains("event-stream")) {
        Some(len) => {
            let extra: Vec<(&str, &str)> = KEPT.iter().filter_map(|k| header(k).map(|v| (*k, v))).collect();
            respond_head(w, status, &content_type, &extra, len, true)?;
            let sent = io::copy(&mut body.by_ref().take(len), w)?;
            w.flush()?;
            if sent < len {
                // (the connection closed under a body shorter than it said: not taken for whole)
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the other computer's answer ended before its length"));
            }
            Ok(true)
        }
        None => {
            let mut out = Stream::start_status(w, status, &content_type)?;
            let mut piece = vec![0u8; 64 << 10];
            loop {
                // (a stream that broke there is broken here too: the connection closes without the chunked end)
                let n = body.read(&mut piece)?;
                if n == 0 {
                    break;
                }
                out.send(&piece[..n])?;
            }
            out.finish()?;
            Ok(true)
        }
    }
}

/// The paired computers' models for the page (`Studio::state`'s `peers`): each with its computer, its kind, whether
/// it answers now, and whether it is OAIY's default of its kind in place of this computer's own.
pub fn models_json(peers: Option<&dyn PeerModels>) -> Json {
    let list = peers.map(|p| p.models()).unwrap_or_default();
    Json::Arr(
        list.iter()
            .map(|m| {
                Json::obj([
                    ("node_id", Json::str(&m.node_id)),
                    ("node_name", Json::str(&m.node_name)),
                    ("kind", Json::str(&m.kind)),
                    ("model", Json::str(&m.model)),
                    ("ready", Json::Bool(m.ready)),
                    ("node_online", Json::Bool(m.node_online)),
                    ("context_tokens", m.context_tokens.map_or(Json::Null, Json::Int)),
                    ("default", Json::Bool(m.default)),
                ])
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// A request a paired computer was sent.
    #[derive(Clone)]
    struct Sent {
        node: String,
        method: String,
        path: String,
        query: String,
        content_type: Option<String>,
        base: Option<String>,
        body: Vec<u8>,
    }

    /// A paired computer that answers what it is given: the requests it was sent, kept for the test to read.
    struct Fake {
        sent: Mutex<Vec<Sent>>,
        answer: Mutex<Option<PeerReply>>,
    }

    impl PeerModels for Fake {
        fn models(&self) -> Vec<PeerModel> {
            let model = |kind: &str, model: &str, path: &str, default: bool| PeerModel {
                node_id: "abc123".into(),
                node_name: "Studio PC".into(),
                kind: kind.into(),
                model: model.into(),
                path: path.into(),
                ready: true,
                node_online: true,
                context_tokens: (kind == "llm").then_some(65536),
                default,
            };
            vec![model("llm", "qwen3.8-27b", "/v1/chat/completions", true), model("image", "flux-3", "/v1/images/generations", false), model("video", "ltx", "/v2/videos", false)]
        }

        fn forward(&self, node: &str, req: PeerRequest) -> Result<PeerReply, String> {
            self.sent.lock().unwrap().push(Sent {
                node: node.into(),
                method: req.method.into(),
                path: req.path.into(),
                query: req.query.into(),
                content_type: req.content_type.map(str::to_owned),
                base: req.base.map(str::to_owned),
                body: req.body,
            });
            self.answer.lock().unwrap().take().ok_or_else(|| "nothing to answer".to_string())
        }
    }

    fn fake(answer: PeerReply) -> Fake {
        Fake { sent: Mutex::new(Vec::new()), answer: Mutex::new(Some(answer)) }
    }

    fn json_reply_of(body: &str) -> PeerReply {
        PeerReply { status: 200, content_type: "application/json".into(), headers: vec![("content-length".into(), body.len().to_string())], body: Box::new(io::Cursor::new(body.as_bytes().to_vec())) }
    }

    /// One request served on a real connection: what the client read, whole.
    fn served(studio: &Studio, req: Request, rest: &str, peers: &dyn PeerModels) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let mut c = TcpStream::connect(addr).unwrap();
            c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut all = Vec::new();
            let _ = c.read_to_end(&mut all);
            String::from_utf8_lossy(&all).into_owned()
        });
        let (mut w, _) = listener.accept().unwrap();
        let _ = handle(studio, &req, &mut w, rest, Some(peers));
        drop(w);
        client.join().unwrap()
    }

    fn studio(name: &str) -> Studio {
        let root = std::env::temp_dir().join(format!("oaiy-studio-peers-{name}-{}", std::process::id()));
        Studio::for_test(&root, crate::config::default_json())
    }

    fn request(method: &str, path: &str, body: &str) -> Request {
        Request::new(method, path, vec![("Host".into(), "127.0.0.1:7860".into()), ("Content-Type".into(), "application/json".into())], body.as_bytes().to_vec())
    }

    /// A request for a paired computer's model goes to that computer on its own route for the kind (what it said, else
    /// the usual one), with the query, the body and this route as where its files are named; its answer comes back as
    /// it was.
    #[test]
    fn a_paired_computer_s_model_is_asked_on_that_computer() {
        let s = studio("forward");
        let peer = fake(json_reply_of(r#"{"data":[{"url":"http://127.0.0.1:7860/api/play/node/abc123/files/images/a.png"}]}"#));
        let out = served(&s, request("POST", "/api/play/node/abc123/images?x=1", r#"{"model":"flux-3","prompt":"a fox"}"#), "abc123/images", &peer);
        assert!(out.starts_with("HTTP/1.1 200"), "{out}");
        assert!(out.contains("/api/play/node/abc123/files/images/a.png"), "{out}");
        let sent = peer.sent.lock().unwrap()[0].clone();
        assert_eq!((sent.node.as_str(), sent.method.as_str(), sent.path.as_str(), sent.query.as_str()), ("abc123", "POST", "/v1/images/generations", "?x=1"));
        assert_eq!(sent.content_type.as_deref(), Some("application/json"));
        assert_eq!(sent.base.as_deref(), Some("http://127.0.0.1:7860/api/play/node/abc123"));
        assert_eq!(sent.body, br#"{"model":"flux-3","prompt":"a fox"}"#);
        // A job's poll and its content below the computer's own route (its videos are at /v2/videos there); its files at
        // the usual route; an edit on its picture route.
        for (rest, want) in [("abc123/videos/vid_1/content", "/v2/videos/vid_1/content"), ("abc123/files/images/a.png", "/files/images/a.png"), ("abc123/edits", "/v1/images/generations"), ("abc123/music/j1", "/v1/audio/music/j1")] {
            let peer = fake(json_reply_of("{}"));
            served(&s, request("GET", &format!("/api/play/node/{rest}"), ""), rest, &peer);
            assert_eq!(peer.sent.lock().unwrap()[0].path, want, "{rest}");
        }
    }

    /// What is not sent: another computer than those paired, a route that climbs out of the target's, another target,
    /// and anything while incognito is on (that computer keeps what it makes by its own settings).
    #[test]
    fn only_a_paired_computer_s_own_routes_are_asked_and_not_in_incognito() {
        let s = studio("refused");
        for (rest, status) in [("zzz999/images", "404"), ("abc123/images/../../pins", "400"), ("abc123/images/%2e%2e/x", "400"), ("abc123/resources", "404"), ("../x/images", "404")] {
            let peer = fake(json_reply_of("{}"));
            let out = served(&s, request("POST", &format!("/api/play/node/{rest}"), "{}"), rest, &peer);
            assert!(out.starts_with(&format!("HTTP/1.1 {status}")), "{rest}: {out}");
            assert!(peer.sent.lock().unwrap().is_empty(), "{rest} was sent");
        }
        let peer = fake(json_reply_of("{}"));
        let asked = Request::new("POST", "/api/play/node/abc123/chat", vec![("Host".into(), "127.0.0.1:7860".into()), ("X-OAIY-Incognito".into(), "1".into())], b"{}".to_vec());
        let out = served(&s, asked, "abc123/chat", &peer);
        assert!(out.starts_with("HTTP/1.1 409") && out.contains("Incognito is on"), "{out}");
        assert!(peer.sent.lock().unwrap().is_empty());
    }

    /// A body that comes in pieces: each piece passed on as it comes, not the whole kept back first.
    struct Pieces {
        left: Vec<&'static [u8]>,
        gone: Arc<AtomicBool>,
    }

    impl Read for Pieces {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.left.is_empty() {
                // (a stream that waits on the model's next word, for longer than the test)
                std::thread::sleep(Duration::from_millis(50));
                return Ok(0);
            }
            std::thread::sleep(Duration::from_millis(30));
            let piece = self.left.remove(0);
            out[..piece.len()].copy_from_slice(piece);
            Ok(piece.len())
        }
    }

    impl Drop for Pieces {
        fn drop(&mut self) {
            self.gone.store(true, Ordering::SeqCst);
        }
    }

    /// A chat's reply is an event stream: passed on chunk by chunk, each as it comes, and ended as it ends.
    #[test]
    fn a_chat_s_reply_is_streamed_as_it_comes() {
        let s = studio("stream");
        let gone = Arc::new(AtomicBool::new(false));
        let pieces: Vec<&'static [u8]> = vec![b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n", b"data: [DONE]\n\n"];
        let peer = fake(PeerReply { status: 200, content_type: "text/event-stream".into(), headers: Vec::new(), body: Box::new(Pieces { left: pieces, gone: Arc::clone(&gone) }) });
        let out = served(&s, request("POST", "/api/play/node/abc123/chat", r#"{"model":"qwen3.8-27b","stream":true}"#), "abc123/chat", &peer);
        assert!(out.contains("Transfer-Encoding: chunked") && out.contains("text/event-stream"), "{out}");
        let hi = out.find("\"Hi\"").expect("the first piece");
        assert!(out.find("[DONE]").is_some_and(|done| done > hi), "{out}");
        assert!(out.ends_with("0\r\n\r\n"), "ended as a whole stream: {out}");
        assert_eq!(peer.sent.lock().unwrap()[0].path, "/v1/chat/completions");
        assert!(gone.load(Ordering::SeqCst), "the body is let go once passed on");
    }

    /// A client that leaves in the middle of a reply: the body is dropped at the next piece, which ends the request on
    /// the other computer (the desktop's reader closes its stream).
    #[test]
    fn a_client_that_leaves_ends_the_reply_there() {
        let s = studio("leaves");
        let gone = Arc::new(AtomicBool::new(false));
        /// Pieces for as long as the reader is read.
        struct Endless(Arc<AtomicBool>);
        impl Read for Endless {
            fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
                std::thread::sleep(Duration::from_millis(20));
                let piece = b"data: {\"choices\":[{\"delta\":{\"content\":\"word \"}}]}\n\n";
                out[..piece.len()].copy_from_slice(piece);
                Ok(piece.len())
            }
        }
        impl Drop for Endless {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let peer = fake(PeerReply { status: 200, content_type: "text/event-stream".into(), headers: Vec::new(), body: Box::new(Endless(Arc::clone(&gone))) });
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
            let c = TcpStream::connect(addr).unwrap();
            let mut r = BufReader::new(c);
            let mut line = String::new();
            // (the head and a piece or two, then gone)
            for _ in 0..12 {
                line.clear();
                r.read_line(&mut line).unwrap();
            }
        });
        let (mut w, _) = listener.accept().unwrap();
        let req = request("POST", "/api/play/node/abc123/chat", "{}");
        let answered = std::thread::spawn(move || {
            let r = handle(&s, &req, &mut w, "abc123/chat", Some(&peer));
            (r.is_err(), peer)
        });
        client.join().unwrap();
        let (failed, peer) = answered.join().unwrap();
        assert!(failed, "the write to a client that left fails");
        drop(peer);
        assert!(gone.load(Ordering::SeqCst), "the reply's body is let go");
    }

    /// A file of known length is passed on whole with its length and its range (a video is seeked in by ranges).
    #[test]
    fn a_file_keeps_its_length_and_range() {
        let s = studio("file");
        let peer = fake(PeerReply {
            status: 206,
            content_type: "video/mp4".into(),
            headers: vec![("content-length".into(), "4".into()), ("content-range".into(), "bytes 2-5/10".into()), ("accept-ranges".into(), "bytes".into())],
            body: Box::new(io::Cursor::new(b"abcd".to_vec())),
        });
        let out = served(&s, request("GET", "/api/play/node/abc123/videos/v1/content", ""), "abc123/videos/v1/content", &peer);
        assert!(out.starts_with("HTTP/1.1 206"), "{out}");
        assert!(out.contains("Content-Length: 4") && out.contains("Content-Range: bytes 2-5/10") && out.contains("Accept-Ranges: bytes"), "{out}");
        assert!(out.ends_with("\r\n\r\nabcd"), "{out}");
    }

    /// The paired computers' models are the page's alone: never in the discovery document, from which the desktop makes
    /// its catalogue (they would be there twice) and which a computer lending its engines passes on (they would be
    /// lent on to a third).
    #[test]
    fn paired_models_are_never_in_the_discovery_document() {
        let s = studio("discovery");
        crate::downloads::set_peer_models(Some(Arc::new(fake(json_reply_of("{}")))));
        let doc = crate::api::discovery::document(&s, "http://127.0.0.1:8080", true).to_json();
        let state = s.state();
        crate::downloads::set_peer_models(None);
        for name in ["qwen3.8-27b", "flux-3", "Studio PC", "abc123"] {
            assert!(!doc.contains(name), "{name} is in the discovery document: {doc}");
        }
        assert_eq!(state.get("peers").and_then(Json::as_array).map(<[Json]>::len), Some(3), "the page is told them");
    }

    /// The page's list: each model with its computer, whether it answers, and whether it is OAIY's default.
    #[test]
    fn the_page_is_told_each_paired_model() {
        let peer = fake(json_reply_of("{}"));
        let list = models_json(Some(&peer));
        let first = &list.as_array().unwrap()[0];
        assert_eq!(first.get("node_name").and_then(Json::as_str), Some("Studio PC"));
        assert_eq!(first.get("default"), Some(&Json::Bool(true)));
        assert_eq!(first.get("context_tokens").and_then(Json::as_i64), Some(65536));
        assert_eq!(models_json(None), Json::Arr(Vec::new()));
    }
}
