//! A relay's base URL (`common#publicUrl`): `https://host[:port]`, no path, no userinfo, no query, no fragment, no trailing slash.
//!
//! **Plain `http` is a decision of the program, not of the build.** Only a program that calls [`allow_loopback_http`] (a function that exists only in a build with the
//! `loopback-http` feature, which a product never enables) makes [`RelayUrl::parse`] read `http://127.0.0.1` or `http://localhost` at all (README section 1: "https only; plain
//! http is accepted by a client only for loopback in test builds. There is no `.local` http exception for relay credentials"). Cargo unifies features across a whole build, so a
//! feature alone could be switched on by any crate in it; a flag that has to be set at run time cannot be set by compiling. Without the call, and in every build without the
//! feature, the scheme is not recognised: no configuration, URI or offer can make a program send a credential in the clear.
//!
//! The origin is normalised: scheme and host in lower case, a port that is the scheme's default dropped (`https://h:443` is `https://h`), and a host that is a name of labels (no
//! empty label, so no `https://.`, `a..b` or trailing dot).

use core::fmt;

use crate::error::{Error, Result};

#[cfg(feature = "loopback-http")]
static ALLOW_LOOPBACK_HTTP: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Lets [`RelayUrl::parse`] read plain `http` URLs on loopback (and only those), for this program. Off until it is called. For tests and the end-to-end harness: a product
/// build has no such function.
#[cfg(feature = "loopback-http")]
pub fn allow_loopback_http(allow: bool) {
    ALLOW_LOOPBACK_HTTP.store(allow, core::sync::atomic::Ordering::SeqCst);
}

/// Whether the program asked for plain `http` on loopback: always false in a build without the `loopback-http` feature.
pub fn loopback_http_allowed() -> bool {
    #[cfg(feature = "loopback-http")]
    {
        ALLOW_LOOPBACK_HTTP.load(core::sync::atomic::Ordering::SeqCst)
    }
    #[cfg(not(feature = "loopback-http"))]
    {
        false
    }
}

/// A validated relay origin.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RelayUrl {
    https: bool,
    host: String,
    /// `None` for the scheme's default port.
    port: Option<u16>,
}

/// A host name of labels: 1 to 63 characters each of letters, digits and hyphens, none starting or ending with a hyphen, no empty label, at most 253 characters in all.
fn host_ok(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 253
        && host.split('.').all(|label| {
            (1..=63).contains(&label.len())
                && label.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                && !label.starts_with('-')
                && !label.ends_with('-')
        })
}

fn is_loopback(host: &str) -> bool {
    matches!(host, "127.0.0.1" | "localhost")
}

impl RelayUrl {
    /// Parses and normalises (scheme and host in lower case, the default port dropped). `https://host` or `https://host:port`; anything else is refused, `http` included unless
    /// the program called [`allow_loopback_http`] and the host is loopback.
    pub fn parse(text: &str) -> Result<RelayUrl> {
        RelayUrl::parse_with(text, loopback_http_allowed())
    }

    /// [`RelayUrl::parse`] with the choice made by the caller: `allow_loopback_http` false is what every program does that has not asked (and what this crate's tests of that
    /// refusal call).
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
        // The scheme's default port is the same origin as no port.
        let port = port.filter(|p| *p != if https { 443 } else { 80 });
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

    /// `scheme://host[:port]`, as written on the wire (the form `offer.relay.url` and the pairing key's `u` have; the default port is not written).
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
        assert_eq!(
            RelayUrl::parse("https://relay.example.com:0443").unwrap().origin(),
            "https://relay.example.com",
            "the default port is the same origin as none"
        );
        assert_eq!(RelayUrl::parse("https://relay.example.com:443").unwrap(), RelayUrl::parse("https://relay.example.com").unwrap());
        assert_eq!(RelayUrl::parse("https://relay.example.com:444").unwrap().port(), 444);
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
            "https://.",
            "https://a..b",
            "https://a.b.",
            "https://.a.b",
            "https://-a.example.com",
            "https://a-.example.com",
            "https://a.example.com.:443",
            "https://a.b..:443",
            "https://:443",
        ] {
            assert!(RelayUrl::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn plain_http_is_not_recognised_in_a_build_without_the_test_feature() {
        // `parse` is `parse_with(text, loopback_http_allowed())` and nothing but a call of `allow_loopback_http` (which a product build does not have) turns that on.
        assert!(RelayUrl::parse_with("http://127.0.0.1:8080", false).is_err());
        assert!(RelayUrl::parse_with("http://relay.example.com", false).is_err());
        assert!(RelayUrl::parse_with("HTTP://localhost", false).is_err());
    }

    #[test]
    fn plain_http_is_read_only_after_the_program_asks_for_it() {
        // Nothing but this call (which a build without the `loopback-http` feature does not have) makes `parse` read `http`, and it is off again when the program says so.
        assert!(!loopback_http_allowed(), "off until it is called");
        assert!(RelayUrl::parse("http://127.0.0.1:8080").is_err());
        allow_loopback_http(true);
        assert!(loopback_http_allowed());
        assert_eq!(RelayUrl::parse("http://127.0.0.1:8080").unwrap().origin(), "http://127.0.0.1:8080");
        assert_eq!(RelayUrl::parse("http://127.0.0.1:80").unwrap().origin(), "http://127.0.0.1", "the default port of http is dropped too");
        assert!(RelayUrl::parse("http://relay.example.com").is_err(), "loopback only, even when asked");
        allow_loopback_http(false);
        assert!(RelayUrl::parse("http://127.0.0.1:8080").is_err());
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
