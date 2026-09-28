//! Just enough HTTP/1.1 for a JSON API: requests with a `Content-Length` or
//! chunked body, `Expect: 100-continue`, keep-alive, and responses that are
//! either whole (`Content-Length`) or streamed (chunked, for server-sent
//! events). A matching client ([`fetch`]) reads the same subset back, so a
//! host can proxy one of these servers without buffering its event streams.
//!
//! Shared by `oaiy-llm-server` and `oaiy-studio`; std-only like the rest of the core.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Largest request body accepted (images arrive inline as base64).
pub const MAX_BODY: usize = 64 << 20;

pub struct Request {
    pub method: String,
    pub path: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// A request built in memory (tests, and hosts that dispatch internally).
    pub fn new(method: &str, path: &str, headers: Vec<(String, String)>, body: Vec<u8>) -> Request {
        Request { method: method.into(), path: path.into(), headers, body }
    }

    /// A header's value (case-insensitive name).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// The path without its query string.
    pub fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or("")
    }

    /// A query-string parameter, percent-decoded.
    pub fn query(&self, name: &str) -> Option<String> {
        let query = self.path.split_once('?')?.1;
        query.split('&').find_map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (percent_decode(k) == name).then(|| percent_decode(v))
        })
    }

    fn keep_alive(&self) -> bool {
        !self.header("connection").is_some_and(|v| v.eq_ignore_ascii_case("close"))
    }
}

/// `%XX` and `+` decoding for query strings; invalid escapes are kept as-is.
pub fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => {
                let hex = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok());
                match hex {
                    Some(v) => {
                        out.push(v);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

pub enum ReadError {
    /// The client closed the connection between requests.
    Closed,
    /// A malformed request: answer 400 and close.
    Bad(String),
    Io,
}

impl From<io::Error> for ReadError {
    fn from(_: io::Error) -> Self {
        ReadError::Io
    }
}

fn line(r: &mut impl BufRead) -> Result<String, ReadError> {
    let mut buf = Vec::new();
    let n = r.take(64 * 1024).read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Err(ReadError::Closed);
    }
    if buf.last() != Some(&b'\n') {
        return Err(ReadError::Bad("header line too long".into()));
    }
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
    String::from_utf8(buf).map_err(|_| ReadError::Bad("header is not UTF-8".into()))
}

fn headers(r: &mut impl BufRead) -> Result<Vec<(String, String)>, ReadError> {
    let mut headers = Vec::new();
    loop {
        let h = line(r).map_err(|e| match e {
            ReadError::Closed => ReadError::Bad("connection closed inside the headers".into()),
            e => e,
        })?;
        if h.is_empty() {
            return Ok(headers);
        }
        let (k, v) = h.split_once(':').ok_or_else(|| ReadError::Bad(format!("bad header {h:?}")))?;
        headers.push((k.trim().to_string(), v.trim().to_string()));
        if headers.len() > 200 {
            return Err(ReadError::Bad("too many headers".into()));
        }
    }
}

fn chunked_body(r: &mut impl BufRead, limit: usize) -> Result<Vec<u8>, ReadError> {
    let mut body = Vec::new();
    loop {
        let size_line = line(r)?;
        let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| ReadError::Bad("bad chunk size".into()))?;
        if size == 0 {
            // trailers, then the blank line
            while !line(r)?.is_empty() {}
            return Ok(body);
        }
        // `size` is attacker-chosen: compare without adding, which could wrap.
        if size > limit.saturating_sub(body.len()) {
            return Err(ReadError::Bad("body too large".into()));
        }
        let at = body.len();
        body.resize(at + size, 0);
        r.read_exact(&mut body[at..])?;
        line(r)?; // the CRLF after the chunk
    }
}

/// Read one request. `writer` receives the `100 Continue` interim response
/// when the client asks for it.
pub fn read_request(r: &mut BufReader<TcpStream>, writer: &mut TcpStream) -> Result<Request, ReadError> {
    let mut first = line(r)?;
    while first.is_empty() {
        first = line(r)?; // tolerate stray blank lines between requests
    }
    let mut parts = first.split_whitespace();
    let (method, path) = match (parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(p), Some(v)) if v.starts_with("HTTP/1.") => (m.to_string(), p.to_string()),
        _ => return Err(ReadError::Bad(format!("bad request line {first:?}"))),
    };
    let headers = headers(r)?;
    let mut req = Request { method, path, headers, body: Vec::new() };
    if req.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
        writer.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    if req.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        req.body = chunked_body(r, MAX_BODY).map_err(|e| match e {
            ReadError::Bad(_) => ReadError::Bad("request body too large".into()),
            e => e,
        })?;
    } else if let Some(len) = req.header("content-length") {
        let len: usize = len.parse().map_err(|_| ReadError::Bad("bad Content-Length".into()))?;
        if len > MAX_BODY {
            return Err(ReadError::Bad("request body too large".into()));
        }
        req.body.resize(len, 0);
        r.read_exact(&mut req.body)?;
    }
    Ok(req)
}

pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        206 => "Partial Content",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        416 => "Range Not Satisfiable",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

// `Authorization` is named: the `*` wildcard never covers it, so browser
// clients sending an API key would fail their preflight. Chrome also asks a
// loopback server before a page from the internet may reach it (Private
// Network Access); which origins are served is still the gateway's decision.
const COMMON: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type, *\r\nAccess-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\nAccess-Control-Allow-Private-Network: true\r\n";

/// Only the head of a response whose `len`-byte body the caller writes next
/// (a file copied in pieces rather than read whole).
pub fn respond_head(w: &mut impl Write, status: u16, content_type: &str, extra: &[(&str, &str)], len: u64, keep_alive: bool) -> io::Result<()> {
    let mut head = format!("HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\n{COMMON}", reason(status));
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str(if keep_alive { "Connection: keep-alive\r\n\r\n" } else { "Connection: close\r\n\r\n" });
    w.write_all(head.as_bytes())
}

/// A whole response.
pub fn respond(w: &mut impl Write, status: u16, content_type: &str, body: &[u8], keep_alive: bool) -> io::Result<()> {
    respond_with(w, status, content_type, &[], body, keep_alive)
}

/// A whole response with extra headers (`Content-Disposition`, `Cache-Control`, ...).
pub fn respond_with(
    w: &mut impl Write,
    status: u16,
    content_type: &str,
    extra: &[(&str, &str)],
    body: &[u8],
    keep_alive: bool,
) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{COMMON}",
        reason(status),
        body.len(),
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str(if keep_alive { "Connection: keep-alive\r\n\r\n" } else { "Connection: close\r\n\r\n" });
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// A streamed response: chunked transfer encoding, each `send` one chunk.
pub struct Stream<'a> {
    w: &'a mut TcpStream,
}

impl<'a> Stream<'a> {
    pub fn start(w: &'a mut TcpStream, content_type: &str) -> io::Result<Stream<'a>> {
        Self::start_status(w, 200, content_type)
    }

    /// A stream with a status other than 200 (a proxied error that streams, say).
    pub fn start_status(w: &'a mut TcpStream, status: u16, content_type: &str) -> io::Result<Stream<'a>> {
        let head = format!(
            "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{COMMON}Connection: keep-alive\r\n\r\n",
            reason(status)
        );
        w.write_all(head.as_bytes())?;
        w.flush()?;
        Ok(Stream { w })
    }

    pub fn send(&mut self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        write!(self.w, "{:x}\r\n", data.len())?;
        self.w.write_all(data)?;
        self.w.write_all(b"\r\n")?;
        self.w.flush()
    }

    pub fn finish(self) -> io::Result<()> {
        self.w.write_all(b"0\r\n\r\n")?;
        self.w.flush()
    }
}

/// Serve one connection: read requests and hand each to `handle` until the
/// client closes or asks to. `handle` returns whether the connection may be
/// kept open.
/// How long a connection may sit idle between requests, and how long a write
/// may wait on a client that stopped reading (a stalled event-stream reader
/// must not hold its request -- and whatever that request holds -- forever).
const IDLE: std::time::Duration = std::time::Duration::from_secs(300);
const WRITE_STALL: std::time::Duration = std::time::Duration::from_secs(120);

pub fn serve(stream: TcpStream, mut handle: impl FnMut(&Request, &mut TcpStream) -> io::Result<bool>) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(IDLE));
    let _ = stream.set_write_timeout(Some(WRITE_STALL));
    let Ok(mut writer) = stream.try_clone() else { return };
    let mut reader = BufReader::new(stream);
    loop {
        match read_request(&mut reader, &mut writer) {
            Ok(req) => {
                let keep = req.keep_alive();
                match handle(&req, &mut writer) {
                    Ok(true) if keep => continue,
                    _ => return,
                }
            }
            Err(ReadError::Bad(m)) => {
                let body = format!("{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\"}}}}", crate::json::Json::str(m).to_json());
                let _ = respond(&mut writer, 400, "application/json", body.as_bytes(), false);
                return;
            }
            Err(_) => return,
        }
    }
}

/// A response being read by [`fetch`]: the status and headers are in; the body
/// is read on demand, whole or chunk by chunk.
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    reader: BufReader<TcpStream>,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn chunked(&self) -> bool {
        self.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    }

    /// The whole body (at most `limit` bytes).
    pub fn body(mut self, limit: usize) -> io::Result<Vec<u8>> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if self.chunked() {
            return chunked_body(&mut self.reader, limit).map_err(|e| match e {
                ReadError::Bad(m) => bad(&m),
                _ => bad("connection closed inside the body"),
            });
        }
        match self.header("content-length").map(str::parse::<usize>) {
            Some(Ok(n)) if n > limit => Err(bad("response body too large")),
            Some(Ok(n)) => {
                let mut body = vec![0; n];
                self.reader.read_exact(&mut body)?;
                Ok(body)
            }
            Some(Err(_)) => Err(bad("bad Content-Length")),
            None => {
                let mut body = Vec::new();
                self.reader.take(limit as u64 + 1).read_to_end(&mut body)?;
                if body.len() > limit {
                    return Err(bad("response body too large"));
                }
                Ok(body)
            }
        }
    }

    /// Pass the body on as it arrives: `each` gets every chunk (or read, for a
    /// body that is not chunked) and may stop the transfer by returning an error.
    pub fn for_each_chunk(mut self, mut each: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if self.chunked() {
            loop {
                let size_line = line(&mut self.reader).map_err(|_| bad("chunk header"))?;
                let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
                    .map_err(|_| bad("bad chunk size"))?;
                if size == 0 {
                    while !line(&mut self.reader).map_err(|_| bad("trailer"))?.is_empty() {}
                    return Ok(());
                }
                if size > 64 << 20 {
                    return Err(bad("chunk larger than 64 MiB"));
                }
                let mut chunk = vec![0; size];
                self.reader.read_exact(&mut chunk)?;
                line(&mut self.reader).map_err(|_| bad("chunk end"))?;
                each(&chunk)?;
            }
        }
        let mut left = match self.header("content-length").map(str::parse::<u64>) {
            Some(Ok(n)) => Some(n),
            Some(Err(_)) => return Err(bad("bad Content-Length")),
            None => None,
        };
        let mut buf = vec![0; 64 * 1024];
        loop {
            let want = left.map_or(buf.len(), |n| n.min(buf.len() as u64) as usize);
            if want == 0 {
                return Ok(());
            }
            let n = self.reader.read(&mut buf[..want])?;
            if n == 0 {
                return if left.is_some_and(|n| n > 0) { Err(bad("connection closed inside the body")) } else { Ok(()) };
            }
            if let Some(l) = left.as_mut() {
                *l -= n as u64;
            }
            each(&buf[..n])?;
        }
    }
}

/// One request to `addr` (`host:port`) on a fresh connection. `timeout` bounds
/// connecting and each read (a streamed reply may run far longer in total).
pub fn fetch(
    addr: &str,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    timeout: Duration,
) -> io::Result<Response> {
    let target: SocketAddr = addr
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, format!("no address for {addr}")))?;
    let mut stream = TcpStream::connect_timeout(&target, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_nodelay(true)?;
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\nContent-Length: {}\r\n", body.len());
    for (k, v) in headers {
        if !k.eq_ignore_ascii_case("host") && !k.eq_ignore_ascii_case("content-length") && !k.eq_ignore_ascii_case("connection") {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    let mut reader = BufReader::new(stream);
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
    let status_line = loop {
        let l = line(&mut reader).map_err(|_| bad("no response".into()))?;
        if l.is_empty() {
            continue;
        }
        let status: u16 = l.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or_else(|| bad(format!("bad status line {l:?}")))?;
        // Skip interim responses (100 Continue) and their headers.
        if (100..200).contains(&status) {
            headers_or(&mut reader)?;
            continue;
        }
        break status;
    };
    let headers = headers_or(&mut reader)?;
    Ok(Response { status: status_line, headers, reader })
}

fn headers_or(r: &mut BufReader<TcpStream>) -> io::Result<Vec<(String, String)>> {
    headers(r).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            match e {
                ReadError::Bad(m) => m,
                _ => "connection closed inside the headers".into(),
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn a_huge_chunk_size_is_refused_rather_than_wrapping() {
        // 5 bytes, then a size that would wrap `len + size` past the limit.
        let body = b"5\r\nhello\r\nfffffffffffffffd\r\nxx\r\n0\r\n\r\n";
        let mut r = BufReader::new(&body[..]);
        assert!(matches!(chunked_body(&mut r, 1 << 20), Err(ReadError::Bad(_))));
    }

    #[test]
    fn query_parameters_are_decoded() {
        let r = Request::new("GET", "/files/a?variant=thumb%20nail&x=1+2&bad=%zz", Vec::new(), Vec::new());
        assert_eq!(r.route(), "/files/a");
        assert_eq!(r.query("variant").as_deref(), Some("thumb nail"));
        assert_eq!(r.query("x").as_deref(), Some("1 2"));
        assert_eq!(r.query("bad").as_deref(), Some("%zz"));
        assert_eq!(r.query("missing"), None);
    }

    #[test]
    fn client_reads_whole_and_streamed_replies_from_the_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                serve(stream, |req, w| {
                    if req.route() == "/stream" {
                        let mut s = Stream::start(w, "text/event-stream")?;
                        s.send(b"data: 1\n\n")?;
                        s.send(b"data: 2\n\n")?;
                        s.finish()?;
                    } else {
                        let mut body = req.method.clone().into_bytes();
                        body.extend_from_slice(&req.body);
                        respond_with(w, 201, "text/plain", &[("X-Test", "yes")], &body, true)?;
                    }
                    Ok(false)
                });
            }
        });
        let t = Duration::from_secs(5);
        let whole = fetch(&addr, "POST", "/echo", &[("Content-Type", "text/plain")], b"hello", t).unwrap();
        assert_eq!((whole.status, whole.header("x-test")), (201, Some("yes")));
        assert_eq!(whole.body(1024).unwrap(), b"POSThello");
        let mut chunks = Vec::new();
        fetch(&addr, "GET", "/stream", &[], b"", t).unwrap().for_each_chunk(|c| {
            chunks.push(c.to_vec());
            Ok(())
        }).unwrap();
        assert_eq!(chunks, vec![b"data: 1\n\n".to_vec(), b"data: 2\n\n".to_vec()]);
        server.join().unwrap();
    }
}
