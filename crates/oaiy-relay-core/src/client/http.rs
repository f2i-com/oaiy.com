//! The HTTP seam: one trait the platform implements, and the request and response it exchanges.
//!
//! The crate has no HTTP client and no TLS of its own. The desktop implements [`HttpClient`] over `reqwest` (rustls, redirects off, one connection pool per origin) and
//! the phone over its own stack (rustls with the platform verifier); the tests use the in-process stub relay or the loopback client of the `loopback-http` feature. What the
//! trait asks of an implementation is the part of README section 1 that a transport can break:
//!
//! - **no redirect is ever followed** (the relay does not redirect, and a bearer must not leave its origin): a `3xx` is returned as it is and the client treats it as a
//!   failure (P2);
//! - the response body is read **at most `max_response_bytes`**, and more is [`TransportError::BodyTooLarge`] (the poll's `maxBytes` is 1 MiB: a relay that sends more is
//!   hostile or broken), and a body that ends before its `Content-Length` is [`TransportError::ShortBody`];
//! - the request is abandoned when `timeout` has passed ([`TransportError::Timeout`]) or when `cancel` is set ([`TransportError::Cancelled`]), whichever is first: this
//!   is how a poll that a newer one replaces (P1) is ended, and how a loop is stopped;
//! - every header the server sent is returned, repeated ones included, in the order received (the names in lower case if the adapter can, and in any case the client reads
//!   them without regard to case: [`HttpResponse::normalised`]).

use core::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// An HTTP method of the protocol (README section 1: GET, POST and OPTIONS are the methods first-party clients use; a client never needs OPTIONS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// GET.
    Get,
    /// POST.
    Post,
}

impl Method {
    /// `GET` or `POST`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
        }
    }
}

/// A flag that is set to end the requests that carry it. Cloning it shares the flag.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    /// A flag that is not set.
    pub fn new() -> Cancel {
        Cancel::default()
    }

    /// Sets the flag: every request that carries it ends with [`TransportError::Cancelled`] as soon as the transport notices.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// True once set.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// A request.
#[derive(Clone)]
pub struct HttpRequest {
    /// GET or POST.
    pub method: Method,
    /// The whole URL: `scheme://host[:port]/v1/...`.
    pub url: String,
    /// The headers to send, as `(name, value)`. A value may be a credential: this type prints no value.
    pub headers: Vec<(String, String)>,
    /// The body, for a POST.
    pub body: Option<Vec<u8>>,
    /// The longest the whole exchange may take.
    pub timeout: Duration,
    /// The most bytes of the response body to read.
    pub max_response_bytes: usize,
    /// Set to abandon the request.
    pub cancel: Cancel,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(k, _)| k.as_str()).collect();
        write!(f, "HttpRequest({} {} headers {:?}, {} body bytes)", self.method.as_str(), self.url, names, self.body.as_ref().map_or(0, Vec::len))
    }
}

impl HttpRequest {
    /// The value of the request header `name` (case-insensitive), for a test that looks at what was sent.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// A response: a status and what came with it.
///
/// The client does not trust an adapter to have lower-cased the header names (an OkHttp adapter returns them as the server sent them): it normalises them itself, once, as the
/// response comes in ([`HttpResponse::normalised`]), and [`HttpResponse::header`] matches without regard to case in any case.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// The status code.
    pub status: u16,
    /// The headers, names in lower case once the response has been through [`HttpResponse::normalised`].
    pub headers: Vec<(String, String)>,
    /// The body. A body can carry a credential (the answers of `/v1/enroll` and `/v1/tokens/rotate` do): `Debug` prints its length and nothing of it.
    pub body: Vec<u8>,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(k, _)| k.as_str()).collect();
        write!(f, "HttpResponse({} headers {:?}, {} body bytes)", self.status, names, self.body.len())
    }
}

impl HttpResponse {
    /// A response with its header names in lower case.
    pub fn new(status: u16, headers: Vec<(String, String)>, body: Vec<u8>) -> HttpResponse {
        HttpResponse { status, headers, body }.normalised()
    }

    /// This response with every header name in ASCII lower case (values and order untouched).
    pub fn normalised(mut self) -> HttpResponse {
        for (k, _) in self.headers.iter_mut() {
            k.make_ascii_lowercase();
        }
        self
    }

    /// The first header called `name`, whatever the case of either.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

/// Why there is no response (P2: "no HTTP response").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransportError {
    /// The connection was refused.
    Refused,
    /// The connection was reset or closed.
    Reset,
    /// The name did not resolve.
    Dns,
    /// The TLS handshake or the certificate failed.
    Tls,
    /// The request's `timeout` passed.
    Timeout,
    /// The request's `cancel` was set.
    Cancelled,
    /// The body was longer than `max_response_bytes`.
    BodyTooLarge,
    /// The body ended before it was complete.
    ShortBody,
    /// Anything else, in words that carry no secret.
    Other(String),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::Refused => f.write_str("connection refused"),
            TransportError::Reset => f.write_str("connection reset"),
            TransportError::Dns => f.write_str("name did not resolve"),
            TransportError::Tls => f.write_str("TLS failure"),
            TransportError::Timeout => f.write_str("timed out"),
            TransportError::Cancelled => f.write_str("cancelled"),
            TransportError::BodyTooLarge => f.write_str("response body too large"),
            TransportError::ShortBody => f.write_str("response body ended early"),
            TransportError::Other(w) => write!(f, "{w}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// The HTTP client of the platform.
pub trait HttpClient: Send + Sync {
    /// Sends `request` and returns the response, or says why there is none. See the module for what an implementation promises.
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError>;
}

impl<T: HttpClient + ?Sized> HttpClient for Arc<T> {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        (**self).send(request)
    }
}

impl<T: HttpClient + ?Sized> HttpClient for Box<T> {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        (**self).send(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_names_are_read_without_regard_to_case_and_the_response_normalises_them() {
        let r = HttpResponse::new(
            200,
            vec![
                ("X-OAIY-Proof".into(), "p".into()),
                ("Retry-After".into(), "7".into()),
                ("retry-after".into(), "9".into()),
                ("X-OAIY-Time".into(), "5".into()),
            ],
            b"{}".to_vec(),
        );
        assert_eq!(r.headers.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), ["x-oaiy-proof", "retry-after", "retry-after", "x-oaiy-time"]);
        assert_eq!(r.header("X-OAIY-Proof"), Some("p"));
        assert_eq!(r.header("x-oaiy-proof"), Some("p"));
        assert_eq!(r.header("RETRY-AFTER"), Some("7"), "the first of a repeated name");
        // A literal built by hand, as an adapter might, is read the same way, and `normalised` does not touch values or order.
        let raw = HttpResponse { status: 200, headers: vec![("Retry-After".into(), "Mixed-Case Value".into())], body: Vec::new() };
        assert_eq!(raw.header("retry-after"), Some("Mixed-Case Value"));
        assert_eq!(raw.normalised().headers, vec![("retry-after".to_string(), "Mixed-Case Value".to_string())]);
    }

    #[test]
    fn debug_of_a_response_prints_no_body() {
        let r = HttpResponse::new(
            201,
            vec![("content-type".into(), "application/json".into())],
            br#"{"token":"oaiyrt1.AAAAAAAAAAA.SECRETSECRETSECRET"}"#.to_vec(),
        );
        let shown = format!("{r:?}");
        assert!(!shown.contains("SECRET") && !shown.contains("83, 69, 67"), "{shown}");
        assert!(shown.contains("201") && shown.contains(&r.body.len().to_string()), "{shown}");
    }
}
