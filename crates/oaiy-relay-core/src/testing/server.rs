//! The stub relay behind a loopback TCP socket (std only), so that the loopback HTTP client and a whole client process can talk to it as they would to a relay: one thread per
//! connection (a held poll does not block the others), `Connection: close`, `Content-Length` bodies.

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::client::http::{Cancel, HttpRequest, Method};

use super::stub::StubRelay;

/// A running server. Dropping it stops it.
pub struct StubServer {
    /// The address it listens on (`127.0.0.1` and a free port).
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl StubServer {
    /// `http://127.0.0.1:<port>`.
    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Stops accepting and waits for the accept thread.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for StubServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        410 => "Gone",
        422 => "Unprocessable Content",
        426 => "Upgrade Required",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

fn serve_one(stub: &StubRelay, mut stream: TcpStream, addr: SocketAddr) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if buf.len() > 64 * 1024 {
            return;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end - 4]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or("/"));
    let headers: Vec<(String, String)> = lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
    let length: usize = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.parse().ok()).unwrap_or(0);
    while buf.len() < head_end + length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let body = buf[head_end..head_end + length].to_vec();
    let request = HttpRequest {
        method: if method == "POST" { Method::Post } else { Method::Get },
        url: format!("http://{addr}{target}"),
        headers,
        body: if method == "POST" { Some(body) } else { None },
        timeout: Duration::from_secs(60),
        max_response_bytes: usize::MAX / 2,
        cancel: Cancel::new(),
    };
    let Ok(response) = stub.handle(&request) else {
        // No response: the connection is closed as it was dropped.
        return;
    };
    let mut out = format!("HTTP/1.1 {} {}\r\n", response.status, reason(response.status));
    for (k, v) in &response.headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", response.body.len()));
    let _ = stream.write_all(out.as_bytes());
    let _ = stream.write_all(&response.body);
    let _ = stream.flush();
}

impl StubRelay {
    /// Serves the stub on `127.0.0.1` at a port the operating system chooses, and sets its public URL to it. Only in a build with the `loopback-http` feature, and it makes this
    /// program read plain `http` on loopback ([`crate::url::allow_loopback_http`]): the stub's own public URL is one.
    #[cfg(feature = "loopback-http")]
    pub fn serve_loopback(&self) -> std::io::Result<StubServer> {
        crate::url::allow_loopback_http(true);
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        self.set_public_url(&format!("http://{addr}"));
        let stop = Arc::new(AtomicBool::new(false));
        let (stub, flag) = (self.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !flag.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let stub = stub.clone();
                        std::thread::spawn(move || serve_one(&stub, stream, addr));
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(5)),
                    Err(_) => break,
                }
            }
        });
        Ok(StubServer { addr, stop, thread: Some(thread) })
    }
}
