//! A stand-in provider for the tests of what the lanes do with their connections.
//!
//! The stubs beside each lane's tests answer `Connection: close`, which suits what
//! they are about (the sequence of a poll, a claim and a report) and cannot show a
//! connection being reused. This one keeps a connection open for as many requests
//! as come down it, counts the connections it was given, and remembers every
//! request with its headers, so a test can say "two polls, one connection" and
//! "this lane's bearer, and no other, reached this server".

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One request as the server saw it.
#[derive(Clone, Debug)]
pub struct Seen {
    pub method: String,
    pub target: String,
    /// Header names in lower case.
    pub headers: HashMap<String, String>,
    pub body: String,
    /// When it arrived.
    pub at: Instant,
}

impl Seen {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(String::as_str)
    }

    /// `GET /path?query`, as a request line without the version.
    pub fn line(&self) -> String {
        format!("{} {}", self.method, self.target)
    }
}

/// What the server answers a request with.
pub struct Reply {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Reply {
    pub fn ok(body: &str) -> Reply {
        Reply::status(200, body)
    }

    pub fn status(status: u16, body: &str) -> Reply {
        Reply { status, headers: Vec::new(), body: body.to_string() }
    }

    pub fn redirect(status: u16, to: &str) -> Reply {
        Reply { status, headers: vec![("Location".into(), to.into())], body: String::new() }
    }
}

type Handler = dyn Fn(&Seen) -> Reply + Send + Sync;

/// A server on a port of its own that keeps its connections alive.
pub struct Provider {
    /// `http://127.0.0.1:<port>`
    pub base: String,
    connections: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Seen>>>,
    open: Arc<Mutex<Vec<TcpStream>>>,
    stop: Arc<AtomicBool>,
    addr: SocketAddr,
    accepting: Option<std::thread::JoinHandle<()>>,
}

impl Provider {
    /// Serve until dropped, answering each request with what `handler` says.
    pub fn start(handler: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Provider {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a test port");
        let addr = listener.local_addr().unwrap();
        let handler: Arc<Handler> = Arc::new(handler);
        let connections = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let open = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let accepting = {
            let (connections, seen, open, stop) = (connections.clone(), seen.clone(), open.clone(), stop.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(stream) = stream else { return };
                    connections.fetch_add(1, Ordering::SeqCst);
                    if let Ok(clone) = stream.try_clone() {
                        open.lock().unwrap().push(clone);
                    }
                    let (handler, seen) = (handler.clone(), seen.clone());
                    std::thread::spawn(move || serve_connection(stream, handler, seen));
                }
            })
        };
        Provider { base: format!("http://{addr}"), connections, seen, open, stop, addr, accepting: Some(accepting) }
    }

    /// How many connections have been made to this server.
    pub fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    /// Every request so far, in the order they arrived.
    pub fn requests(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }

    /// The request lines so far (`GET /path`).
    pub fn lines(&self) -> Vec<String> {
        self.requests().iter().map(Seen::line).collect()
    }

    /// Wait until `count` requests to a path beginning `prefix` have arrived, and
    /// return them: for a lane's loop, which is left to run on its own thread.
    pub fn wait_for(&self, prefix: &str, count: usize, within: Duration) -> Vec<Seen> {
        let deadline = Instant::now() + within;
        loop {
            let seen: Vec<Seen> = self.requests().into_iter().filter(|r| r.target.starts_with(prefix)).collect();
            if seen.len() >= count {
                return seen;
            }
            assert!(
                Instant::now() < deadline,
                "only {} of {count} requests to {prefix} in {within:?}: {:?}",
                seen.len(),
                self.lines()
            );
            std::thread::sleep(Duration::from_millis(25));
        }
    }
}

/// A store linked to `base` with the shipped connector, changed by `edit`, in a
/// data folder of its own: what a lane's loop reads to know where to poll and how
/// (a user's connector file replaces the built-in one of the same id).
pub fn linked_to(
    base: &str,
    tag: &str,
    edit: impl FnOnce(&mut super::ConnectorDescriptor),
) -> (super::LinkHandle, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("oaiy-lane-{tag}-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(dir.join("connectors")).unwrap();
    let mut descriptor = super::descriptor::builtin().remove(0);
    edit(&mut descriptor);
    std::fs::write(dir.join("connectors").join("formlogic.json"), serde_json::to_string(&descriptor).unwrap()).unwrap();
    let account = super::LinkedAccount {
        connector_id: "formlogic".into(),
        base_url: base.into(),
        credential: "flk_lane".into(),
        account_id: None,
        account_name: None,
        granted_scopes: None,
        linked_at: chrono::Utc::now(),
        instance_id: Some("oaiy-test".into()),
    };
    (super::store_for_tests(dir.clone(), Some(account)), dir)
}

impl Drop for Provider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accepting thread out of its wait, then close what is open so
        // that no connection thread outlives the test.
        let _ = TcpStream::connect(self.addr);
        if let Some(t) = self.accepting.take() {
            let _ = t.join();
        }
        for stream in self.open.lock().unwrap().drain(..) {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        }
    }
}

fn serve_connection(stream: TcpStream, handler: Arc<Handler>, seen: Arc<Mutex<Vec<Seen>>>) {
    let Ok(read_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read_half);
    let mut out = stream;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let mut parts = line.split_whitespace();
        let (Some(method), Some(target)) = (parts.next(), parts.next()) else { return };
        let (method, target) = (method.to_string(), target.to_string());
        let mut headers = HashMap::new();
        loop {
            let mut header = String::new();
            match reader.read_line(&mut header) {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let length: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut body = vec![0u8; length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        let request = Seen {
            method,
            target,
            headers,
            body: String::from_utf8_lossy(&body).into_owned(),
            at: Instant::now(),
        };
        seen.lock().unwrap().push(request.clone());

        let reply = handler(&request);
        let mut head = format!(
            "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
            reply.status,
            reply.body.len()
        );
        for (name, value) in &reply.headers {
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        head.push_str("\r\n");
        if out.write_all(head.as_bytes()).and_then(|()| out.write_all(reply.body.as_bytes())).is_err() {
            return;
        }
        let _ = out.flush();
    }
}
