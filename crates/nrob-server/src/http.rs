//! Just enough HTTP/1.1 for a JSON API: requests with a `Content-Length` or
//! chunked body, `Expect: 100-continue`, keep-alive, and responses that are
//! either whole (`Content-Length`) or streamed (chunked, for server-sent
//! events).

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;

/// Largest request body accepted (images arrive inline as base64).
pub const MAX_BODY: usize = 64 << 20;

pub struct Request {
    pub method: String,
    pub path: String,
    headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// A header's value (case-insensitive name).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn keep_alive(&self) -> bool {
        !self.header("connection").is_some_and(|v| v.eq_ignore_ascii_case("close"))
    }
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
    let mut headers = Vec::new();
    loop {
        let h = line(r).map_err(|e| match e {
            ReadError::Closed => ReadError::Bad("connection closed inside the headers".into()),
            e => e,
        })?;
        if h.is_empty() {
            break;
        }
        let (k, v) = h.split_once(':').ok_or_else(|| ReadError::Bad(format!("bad header {h:?}")))?;
        headers.push((k.trim().to_string(), v.trim().to_string()));
        if headers.len() > 200 {
            return Err(ReadError::Bad("too many headers".into()));
        }
    }
    let mut req = Request { method, path, headers, body: Vec::new() };
    if req.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue")) {
        writer.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    if req.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        loop {
            let size_line = line(r)?;
            let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| ReadError::Bad("bad chunk size".into()))?;
            if size == 0 {
                // trailers, then the blank line
                while !line(r)?.is_empty() {}
                break;
            }
            if req.body.len() + size > MAX_BODY {
                return Err(ReadError::Bad("request body too large".into()));
            }
            let at = req.body.len();
            req.body.resize(at + size, 0);
            r.read_exact(&mut req.body[at..])?;
            line(r)?; // the CRLF after the chunk
        }
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

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Unknown",
    }
}

const COMMON: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

/// A whole response.
pub fn respond(w: &mut impl Write, status: u16, content_type: &str, body: &[u8], keep_alive: bool) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{COMMON}Connection: {}\r\n\r\n",
        reason(status),
        body.len(),
        if keep_alive { "keep-alive" } else { "close" }
    );
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
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n{COMMON}Connection: keep-alive\r\n\r\n"
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
pub fn serve(stream: TcpStream, mut handle: impl FnMut(&Request, &mut TcpStream) -> io::Result<bool>) {
    let _ = stream.set_nodelay(true);
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
                let body = format!("{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\"}}}}", nrob::json::Json::str(m).to_json());
                let _ = respond(&mut writer, 400, "application/json", body.as_bytes(), false);
                return;
            }
            Err(_) => return,
        }
    }
}
