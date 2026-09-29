//! A stand-in provider for the tests of what the lanes do with their connections.
//!
//! The stubs beside each lane's tests answer `Connection: close`, which suits what
//! they are about (the sequence of a poll, a claim and a report) and cannot show a
//! connection being reused. This one keeps a connection open for as many requests
//! as come down it, counts the connections it was given, and remembers every
//! request with its headers and the connection it came down, so a test can say
//! "two polls, one connection" and "this lane's bearer, and no other, reached this
//! server". It also notes when each connection was closed by the other end, so a
//! test can say how long a client held one open after its reply.

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
    /// Which connection it came down: 0 for the first this server was given, 1 for
    /// the next, and so on.
    pub conn: usize,
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

    /// Too many requests (429): asked to leave the server alone for `seconds`, with
    /// `message` as the provider's own words.
    pub fn too_many(seconds: u64, message: &str) -> Reply {
        Reply {
            status: 429,
            headers: vec![("Retry-After".into(), seconds.to_string())],
            body: serde_json::json!({ "message": message }).to_string(),
        }
    }
}

/// What a connection has been through, as far as the server can tell.
#[derive(Clone, Copy, Debug, Default)]
struct Life {
    /// When the server last answered a request on it.
    last_reply: Option<Instant>,
    /// When the other end closed it.
    closed: Option<Instant>,
}

type Handler = dyn Fn(&Seen) -> Reply + Send + Sync;

/// A server on a port of its own that keeps its connections alive.
pub struct Provider {
    /// `http://127.0.0.1:<port>`
    pub base: String,
    connections: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Seen>>>,
    lives: Arc<Mutex<Vec<Life>>>,
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
        let lives = Arc::new(Mutex::new(Vec::new()));
        let open = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let accepting = {
            let (connections, seen, lives, open, stop) =
                (connections.clone(), seen.clone(), lives.clone(), open.clone(), stop.clone());
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let Ok(stream) = stream else { return };
                    // Numbered as it is accepted, so the number is the same one
                    // `connections()` counted and a request is tagged with.
                    let index = {
                        let mut lives = lives.lock().unwrap();
                        lives.push(Life::default());
                        lives.len() - 1
                    };
                    connections.fetch_add(1, Ordering::SeqCst);
                    if let Ok(clone) = stream.try_clone() {
                        open.lock().unwrap().push(clone);
                    }
                    let (handler, seen, lives) = (handler.clone(), seen.clone(), lives.clone());
                    std::thread::spawn(move || {
                        serve_connection(stream, index, handler, seen, &lives);
                        lives.lock().unwrap()[index].closed = Some(Instant::now());
                    });
                }
            })
        };
        Provider { base: format!("http://{addr}"), connections, seen, lives, open, stop, addr, accepting: Some(accepting) }
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

    /// How long the client kept connection `conn` open after the server's last
    /// reply on it, once the client closed it: how long a lane held a connection
    /// that nothing was using. Waits up to `within` for the close, and fails the
    /// test if it does not come (the client is holding it still).
    pub fn closed_after_reply(&self, conn: usize, within: Duration) -> Duration {
        self.try_closed_after_reply(conn, within).unwrap_or_else(|| {
            panic!("connection {conn} was still open {within:?} after this test began waiting for it to close")
        })
    }

    /// [`Provider::closed_after_reply`] that says `None` instead of failing when the
    /// connection is still open after `within`: for a test that has a lane's loop to
    /// stop before it may fail.
    pub fn try_closed_after_reply(&self, conn: usize, within: Duration) -> Option<Duration> {
        let deadline = Instant::now() + within;
        loop {
            let life = self.lives.lock().unwrap().get(conn).copied();
            if let Some(Life { last_reply: Some(replied), closed: Some(closed) }) = life {
                return Some(closed.saturating_duration_since(replied));
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
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

/// Stop a lane's loop that a test started, as far as a loop that never ends can be
/// stopped: the store loses its account, and the loop, which had nothing left to
/// wait for, has nothing to do from its next time round.
///
/// A loop left running on a link goes on polling a port nobody listens on once its
/// test is over, and starts its client afresh at every failure: on the client
/// other tests count connections on, and, for the queue check, with the provider's
/// prelude the other tests' cache holds. Call it once the requests a test wants
/// have been seen and before its provider goes.
pub fn stop_lane(store: &super::LinkHandle) {
    store.drop_account_for_tests();
    // The last poll it made was answered at once, so this is time enough for the
    // loop to be out of its request and into its wait.
    std::thread::sleep(Duration::from_millis(300));
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

fn serve_connection(
    stream: TcpStream,
    index: usize,
    handler: Arc<Handler>,
    seen: Arc<Mutex<Vec<Seen>>>,
    lives: &Mutex<Vec<Life>>,
) {
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
            conn: index,
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
        lives.lock().unwrap()[index].last_reply = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_says_which_connection_it_came_down_and_a_close_is_timed_from_the_last_reply() {
        // No body, so a reply is the one write of its head and one read takes it all.
        let server = Provider::start(|_| Reply::ok(""));
        let ask = |stream: &mut TcpStream, target: &str| {
            write!(stream, "GET {target} HTTP/1.1\r\nHost: test\r\n\r\n").unwrap();
            let mut buf = [0u8; 256];
            let n = stream.read(&mut buf).unwrap();
            assert!(String::from_utf8_lossy(&buf[..n]).starts_with("HTTP/1.1 200"));
        };
        let addr = server.base.trim_start_matches("http://").to_string();
        let mut first = TcpStream::connect(&addr).unwrap();
        ask(&mut first, "/one");
        ask(&mut first, "/two");
        let mut second = TcpStream::connect(&addr).unwrap();
        ask(&mut second, "/three");

        assert_eq!(server.connections(), 2);
        let conns: Vec<(String, usize)> = server.requests().iter().map(|r| (r.line(), r.conn)).collect();
        assert_eq!(
            conns,
            [("GET /one".to_string(), 0), ("GET /two".to_string(), 0), ("GET /three".to_string(), 1)]
        );

        // Held for a while after its last reply, then closed by this end.
        std::thread::sleep(Duration::from_millis(300));
        drop(first);
        let held = server.closed_after_reply(0, Duration::from_secs(5));
        assert!(held >= Duration::from_millis(250) && held < Duration::from_secs(3), "{held:?}");
        // The other is closed as soon as it has been answered.
        drop(second);
        assert!(server.closed_after_reply(1, Duration::from_secs(5)) < Duration::from_secs(2));
    }
}
