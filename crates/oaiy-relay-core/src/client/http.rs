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
//! - header names of the response are returned in lower case, and every header the server sent is returned, repeated ones included, in the order received.

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// The status code.
    pub status: u16,
    /// The headers, names in lower case.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// The first header called `name` (give it in lower case).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
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
