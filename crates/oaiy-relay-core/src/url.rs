//! A relay's base URL (`common#publicUrl`): `https://host[:port]`, no path, no userinfo, no query, no fragment, no trailing slash.
//!
//! `http` is a different type of value in a test build: only with the `loopback-http` feature, and only for a loopback address (`127.0.0.1`, `localhost`), is plain `http`
//! read at all (README section 1: "https only; plain http is accepted by a client only for loopback in test builds. There is no `.local` http exception for relay
//! credentials"). Without the feature the scheme is not recognised, so no configuration, URI or offer can make a product build send a credential in the clear.

use core::fmt;

use crate::error::{Error, Result};

/// A validated relay origin.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RelayUrl {
    https: bool,
    host: String,
    port: Option<u16>,
}

fn host_ok(host: &str) -> bool {
    !host.is_empty() && host.len() <= 253 && host.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost")
}

impl RelayUrl {
    /// Parses and normalises (scheme and host in lower case, the port without leading zeros). `https://host` or `https://host:port`; anything else is refused, `http`
    /// included unless the `loopback-http` feature is on and the host is loopback.
    pub fn parse(text: &str) -> Result<RelayUrl> {
        RelayUrl::parse_with(text, cfg!(feature = "loopback-http"))
    }

    /// [`RelayUrl::parse`] with the choice made by the caller: `allow_loopback_http` false is what every build without the `loopback-http` feature does (and what
    /// this crate's tests of that refusal call, because its own tests are built with the feature on).
    pub fn parse_with(text: &str, allow_loopback_http: bool) -> Result<RelayUrl> {
        let (https, rest) = if let Some(r) = strip_prefix_ci(text, "https://") {
            (true, r)
        } else if allow_loopback_http {
            match strip_prefix_ci(text, "http://") {
                Some(r) => (false, r),
                None => return Err(Error::Uri("relay url: https is required")),
            }
        } else {
            return Err(Error::Uri("relay url: https is required"));
        };
        let (host, port) = match rest.split_once(':') {
            Some((h, p)) => {
                if p.is_empty() || p.len() > 5 || !p.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(Error::Uri("relay url: port"));
                }
                let n: u32 = p.parse().map_err(|_| Error::Uri("relay url: port"))?;
                if n == 0 || n > 65535 {
                    return Err(Error::Uri("relay url: port"));
                }
                (h, Some(n as u16))
            }
            None => (rest, None),
        };
        if !host_ok(host) {
            return Err(Error::Uri("relay url: host, or a path, userinfo, query or fragment that a relay url has none of"));
        }
        let host = host.to_ascii_lowercase();
        if !https && !is_loopback(&host) {
            return Err(Error::Uri("relay url: plain http is for loopback only"));
        }
        Ok(RelayUrl { https, host, port })
    }

    /// True for `https`.
    pub fn is_https(&self) -> bool {
        self.https
    }

    /// The host, in lower case.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// The port, or the scheme's default.
    pub fn port(&self) -> u16 {
        self.port.unwrap_or(if self.https { 443 } else { 80 })
    }

    /// `scheme://host[:port]`, as written on the wire (the form `offer.relay.url` and the pairing key's `u` have).
    pub fn origin(&self) -> String {
        let mut s = String::from(if self.https { "https://" } else { "http://" });
        s.push_str(&self.host);
        if let Some(p) = self.port {
            s.push(':');
            s.push_str(&p.to_string());
        }
        s
    }

    /// The URL of `path` (which begins with `/v1/`).
    pub fn join(&self, path: &str) -> String {
        format!("{}{}", self.origin(), path)
    }
}

fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text.len() >= prefix.len() && text.is_char_boundary(prefix.len()) && text[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&text[prefix.len()..])
    } else {
        None
    }
}

impl fmt::Debug for RelayUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelayUrl({})", self.origin())
    }
}

impl fmt::Display for RelayUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.origin())
    }
}

/// Percent-encodes `text` for a query value (RFC 3986 unreserved characters are kept).
pub fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 3);
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Percent-decodes a query value. A `%` that is not followed by two hex digits is refused, `+` is a plus (not a space), and the result must be UTF-8.
pub fn percent_decode(text: &str) -> Result<String> {
    let b = text.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = b.get(i + 1..i + 3).ok_or(Error::Uri("percent escape"))?;
            let hi = (hex[0] as char).to_digit(16).ok_or(Error::Uri("percent escape"))?;
            let lo = (hex[1] as char).to_digit(16).ok_or(Error::Uri("percent escape"))?;
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| Error::Uri("percent escape: not UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_origins_are_accepted_and_normalised() {
        let u = RelayUrl::parse("https://Relay.Example.com").unwrap();
        assert_eq!(u.origin(), "https://relay.example.com");
        assert_eq!(u.port(), 443);
        assert_eq!(RelayUrl::parse("HTTPS://relay.example.com:8443").unwrap().origin(), "https://relay.example.com:8443");
        assert_eq!(RelayUrl::parse("https://relay.example.com:0443").unwrap().origin(), "https://relay.example.com:443");
        assert_eq!(u.join("/v1/info"), "https://relay.example.com/v1/info");
    }

    #[test]
    fn everything_that_is_not_a_bare_origin_is_refused() {
        for bad in [
            "",
            "relay.example.com",
            "ftp://relay.example.com",
            "https://",
            "https://relay.example.com/",
            "https://relay.example.com/path",
            "https://relay.example.com?x=1",
            "https://relay.example.com#f",
            "https://user@relay.example.com",
            "https://user:pw@relay.example.com",
            "https://relay.example.com:",
            "https://relay.example.com:0",
            "https://relay.example.com:65536",
            "https://relay.example.com:80a",
            "https://[::1]",
            "https://relay example.com",
            "https://relay.example.com\u{0}",
            "https://relay_example.com",
        ] {
            assert!(RelayUrl::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn plain_http_is_not_recognised_in_a_build_without_the_test_feature() {
        // `parse` is `parse_with(text, cfg!(feature = "loopback-http"))`: this is every product build.
        assert!(RelayUrl::parse_with("http://127.0.0.1:8080", false).is_err());
        assert!(RelayUrl::parse_with("http://relay.example.com", false).is_err());
        assert!(RelayUrl::parse_with("HTTP://localhost", false).is_err());
    }

    #[test]
    fn plain_http_is_for_loopback_only_even_with_the_test_feature() {
        assert_eq!(RelayUrl::parse_with("http://127.0.0.1:8080", true).unwrap().origin(), "http://127.0.0.1:8080");
        assert!(!RelayUrl::parse_with("http://localhost", true).unwrap().is_https());
        assert!(RelayUrl::parse_with("http://relay.example.com", true).is_err());
        assert!(RelayUrl::parse_with("http://10.0.0.1", true).is_err());
        assert!(RelayUrl::parse_with("http://127.0.0.1.evil.test", true).is_err());
        assert!(RelayUrl::parse_with("http://127.0.0.1/path", true).is_err());
    }

    #[test]
    fn percent_coding() {
        assert_eq!(percent_encode("https://relay.example.com"), "https%3A%2F%2Frelay.example.com");
        assert_eq!(percent_decode("https%3A%2F%2Frelay.example.com").unwrap(), "https://relay.example.com");
        assert_eq!(percent_decode("a+b").unwrap(), "a+b");
        assert!(percent_decode("%").is_err() && percent_decode("%4").is_err() && percent_decode("%zz").is_err() && percent_decode("%ff").is_err());
    }
}
