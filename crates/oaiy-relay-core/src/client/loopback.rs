//! A blocking HTTP/1.1 client over `std::net::TcpStream` that talks to **loopback only** (feature `loopback-http`).
//!
//! For this crate's tests (the in-process stub relay served on a socket, the real PHP relay under `php -S`), the end-to-end harness and nothing else: it has no TLS, no
//! connection reuse and no redirects (every request is `Connection: close`, and a `3xx` is returned as it is), and it refuses any host that is not `127.0.0.1` or `localhost` before it
//! opens a socket, so that no build with the feature can send a credential across a network. It reads a body that has a `Content-Length`, a chunked body, or one that ends with the
//! connection, never more than the request's `max_response_bytes`, and it ends the exchange at the request's `timeout` or when its `cancel` is set (it reads in slices of 25 ms to
//! notice).

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use super::http::{HttpClient, HttpRequest, HttpResponse, TransportError};

/// The client.
#[derive(Debug, Default, Clone, Copy)]
pub struct LoopbackHttp;

/// `host[:port]` and the path of an `http://` URL on loopback.
fn split(url: &str) -> Result<(String, SocketAddr, String), TransportError> {
    let rest = url.strip_prefix("http://").ok_or_else(|| TransportError::Other("loopback client: http only".into()))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| TransportError::Other("loopback client: port".into()))?),
        None => (authority, 80),
    };
    if host != "127.0.0.1" && host != "localhost" {
        return Err(TransportError::Other("loopback client: not a loopback host".into()));
    }
    // The path goes into the request line as it is: a space, a line break or any other control or non-ASCII byte in it would split or smuggle a request.
    if path.bytes().any(|b| b <= b' ' || b >= 0x7f) {
        return Err(TransportError::Other("loopback client: a path with a space, a control character or a byte above 0x7e".into()));
    }
    let addr = (if host == "localhost" { "127.0.0.1" } else { host }, port)
        .to_socket_addrs()
        .map_err(|_| TransportError::Dns)?
        .next()
        .ok_or(TransportError::Dns)?;
    Ok((authority.to_string(), addr, path.to_string()))
}

fn map_io(e: &std::io::Error) -> TransportError {
    match e.kind() {
        ErrorKind::ConnectionRefused => TransportError::Refused,
        ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted | ErrorKind::BrokenPipe | ErrorKind::UnexpectedEof => TransportError::Reset,
        ErrorKind::TimedOut | ErrorKind::WouldBlock => TransportError::Timeout,
        _ => TransportError::Other(format!("{:?}", e.kind())),
    }
}

/// Reads until `done` says the buffer holds what is needed, ending at the deadline or the cancel.
fn fill(
    stream: &mut TcpStream,
    buf: &mut Vec<u8>,
    req: &HttpRequest,
    deadline: Instant,
    mut done: impl FnMut(&[u8]) -> bool,
) -> Result<bool, TransportError> {
    let mut chunk = [0u8; 16 * 1024];
    loop {
        if done(buf) {
            return Ok(true);
        }
        if req.cancel.is_cancelled() {
            return Err(TransportError::Cancelled);
        }
        if Instant::now() >= deadline {
            return Err(TransportError::Timeout);
        }
        match stream.read(&mut chunk) {
            Ok(0) => return Ok(false),
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                // A header block or a body that has grown past what the caller allows is refused before it is stored any further.
                if buf.len() > req.max_response_bytes + 64 * 1024 {
                    return Err(TransportError::BodyTooLarge);
                }
            }
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(e) => return Err(map_io(&e)),
        }
    }
}

impl HttpClient for LoopbackHttp {
    fn send(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let deadline = Instant::now() + req.timeout;
        let (authority, addr, path) = split(&req.url)?;
        if req.cancel.is_cancelled() {
            return Err(TransportError::Cancelled);
        }
        let mut stream = TcpStream::connect_timeout(&addr, req.timeout.min(Duration::from_secs(5))).map_err(|e| map_io(&e))?;
        stream.set_read_timeout(Some(Duration::from_millis(25))).map_err(|e| map_io(&e))?;
        stream.set_write_timeout(Some(req.timeout.min(Duration::from_secs(10)))).map_err(|e| map_io(&e))?;
        let mut head = format!("{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n", req.method.as_str(), path, authority);
        for (k, v) in &req.headers {
            if k.eq_ignore_ascii_case("host") || k.eq_ignore_ascii_case("content-length") || k.eq_ignore_ascii_case("connection") {
                continue;
            }
            if k.bytes().any(|b| b <= b' ' || b == b':') || v.bytes().any(|b| b == b'\r' || b == b'\n') {
                return Err(TransportError::Other("loopback client: a header with a line break".into()));
            }
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        if let Some(b) = &req.body {
            head.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
        head.push_str("\r\n");
        stream.write_all(head.as_bytes()).map_err(|e| map_io(&e))?;
        if let Some(b) = &req.body {
            stream.write_all(b).map_err(|e| map_io(&e))?;
        }
        stream.flush().map_err(|e| map_io(&e))?;

        let mut buf = Vec::new();
        let find_end = |b: &[u8]| b.windows(4).any(|w| w == b"\r\n\r\n");
        if !fill(&mut stream, &mut buf, req, deadline, find_end)? {
            return Err(if buf.is_empty() { TransportError::Reset } else { TransportError::ShortBody });
        }
        let end = buf.windows(4).position(|w| w == b"\r\n\r\n").ok_or(TransportError::ShortBody)? + 4;
        let head_text = String::from_utf8_lossy(&buf[..end - 4]).into_owned();
        let mut lines = head_text.split("\r\n");
        let status_line = lines.next().unwrap_or("");
        let status: u16 = status_line
            .split(' ')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| TransportError::Other("loopback client: a status line".into()))?;
        let mut headers = Vec::new();
        for l in lines {
            if let Some((k, v)) = l.split_once(':') {
                headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
            }
        }
        let header = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
        let chunked = header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
        let length: Option<usize> = header("content-length").and_then(|v| v.parse().ok());
        let mut body = buf[end..].to_vec();
        let bodyless = status == 204 || status == 304 || (100..200).contains(&status);
        if bodyless {
            body.clear();
        } else if chunked {
            let mut decoded = Vec::new();
            let mut raw = body;
            loop {
                // A chunk: `<hex>[;ext]\r\n<data>\r\n`; the last is `0\r\n\r\n` (trailers are ignored).
                let mut size_end = None;
                let ok = fill(&mut stream, &mut raw, req, deadline, |b| {
                    size_end = b.windows(2).position(|w| w == b"\r\n");
                    size_end.is_some()
                })?;
                if !ok {
                    return Err(TransportError::ShortBody);
                }
                let i = size_end.ok_or(TransportError::ShortBody)?;
                let size_text = String::from_utf8_lossy(&raw[..i]).to_string();
                let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
                    .map_err(|_| TransportError::Other("loopback client: a chunk size".into()))?;
                if size == 0 {
                    break;
                }
                // `size` is a number the server chose: nothing is added to it before it is compared with what is allowed (a size near usize::MAX is not an overflow).
                if size > req.max_response_bytes.saturating_sub(decoded.len()) {
                    return Err(TransportError::BodyTooLarge);
                }
                let need = i + 2 + size + 2;
                if !fill(&mut stream, &mut raw, req, deadline, |b| b.len() >= need)? {
                    return Err(TransportError::ShortBody);
                }
                decoded.extend_from_slice(&raw[i + 2..i + 2 + size]);
                raw.drain(..need);
            }
            body = decoded;
        } else if let Some(n) = length {
            if n > req.max_response_bytes {
                return Err(TransportError::BodyTooLarge);
            }
            if !fill(&mut stream, &mut body, req, deadline, |b| b.len() >= n)? {
                return Err(TransportError::ShortBody);
            }
            body.truncate(n);
        } else {
            // Ends with the connection.
            while fill(&mut stream, &mut body, req, deadline, |_| false)? {}
        }
        if body.len() > req.max_response_bytes {
            return Err(TransportError::BodyTooLarge);
        }
        Ok(HttpResponse { status, headers, body })
    }
}
