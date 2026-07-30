//! Egress safety for the AI proxy.
//!
//! Provider base URLs are configured by the desktop user (management-plane), but
//! the proxy still forwards outbound with a stored key, so a mistyped or hostile
//! base URL must not become an SSRF into the local network. Adapted from
//! [`crate::services::downloads`]'s `is_disallowed_ip` + URL validation (NOT the
//! heavier DNS-pinning egress module): require `https`, reject embedded
//! credentials, and reject a host that IS or RESOLVES TO an internal address.
//! Redirects are disabled on the client, so a 3xx to a fresh host is surfaced,
//! never followed.
//!
//! `allow_local` relaxes this for a user-declared on-device endpoint (an ollama /
//! LM Studio server): `http` + loopback/private become reachable, but link-local
//! (169.254.x, incl. the 169.254.169.254 cloud-metadata address) stays blocked.

use std::net::{IpAddr, ToSocketAddrs};
use std::time::Duration;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Whole-request cap for buffered (non-streaming) calls. Streaming omits it.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
/// Idle cap: fires when NO bytes arrive within the window (unlike the whole-request
/// timeout). This is what reclaims a stalled streaming upstream — a misbehaving
/// provider that returns 200 then goes silent can't pin the connection forever —
/// without cutting off a legitimately long but active stream.
const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Reject any address a proxy must never reach (identical predicate to
/// `downloads::is_disallowed_ip`): loopback, RFC1918 private, link-local incl.
/// 169.254.169.254 metadata, CGNAT 100.64/10, unspecified, unique-local v6, etc.
fn is_disallowed_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || o[0] == 0
                || (o[0] == 100 && (o[1] & 0xc0) == 0x40) // 100.64.0.0/10 CGNAT
        }
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_disallowed_ip(IpAddr::V4(v4));
            }
            let seg0 = v6.segments()[0];
            (seg0 & 0xfe00) == 0xfc00 // unique-local fc00::/7
                || (seg0 & 0xffc0) == 0xfe80 // link-local fe80::/10
        }
    }
}

/// Addresses that stay blocked even for a user-declared local endpoint: cloud
/// metadata (169.254.169.254) and any link-local, which a loopback dev server
/// never legitimately uses.
fn is_link_local_or_metadata(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_link_local(),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return v4.is_link_local();
            }
            (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

fn ip_blocked(ip: IpAddr, allow_local: bool) -> bool {
    if allow_local {
        is_link_local_or_metadata(ip)
    } else {
        is_disallowed_ip(ip)
    }
}

fn check_scheme_and_creds(url: &reqwest::Url, allow_local: bool) -> Result<(), String> {
    let ok_scheme = url.scheme() == "https" || (allow_local && url.scheme() == "http");
    if !ok_scheme {
        return Err(if allow_local {
            "provider base URL must use http or https".into()
        } else {
            "provider base URL must use https".into()
        });
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err("provider URL must not embed credentials".into());
    }
    if url.host_str().is_none() {
        return Err("provider URL has no host".into());
    }
    Ok(())
}

/// Offline-safe syntactic check for config time (upsert): parse + scheme + creds
/// + host present, with NO DNS resolution (so configuring a provider works while
/// offline). The full resolve-and-check runs per-request in [`validate`].
pub fn check_base_url_syntax(base_url: &str, allow_local: bool) -> Result<(), String> {
    if base_url.trim().is_empty() {
        return Err("provider base URL is required".into());
    }
    if base_url.chars().any(|c| c.is_control() || c == ' ') {
        return Err("provider base URL contains invalid characters".into());
    }
    let url = reqwest::Url::parse(base_url).map_err(|_| "provider base URL is invalid".to_string())?;
    check_scheme_and_creds(&url, allow_local)?;
    // A bare-IP base URL is still checked syntactically for the obvious internal
    // literals, so an obviously-bad config fails fast even offline.
    if let Some(host) = url.host_str() {
        if let Ok(ip) = host.parse::<IpAddr>() {
            if ip_blocked(ip, allow_local) {
                return Err(format!("provider URL points at a disallowed address ({ip})"));
            }
        }
    }
    Ok(())
}

/// Full request-time validation: join `base_url` + `path` (path can't escape the
/// base), require https (or http when `allow_local`), reject creds, and reject a
/// host that is / resolves to an internal address. Returns the outbound URL.
pub fn validate(base_url: &str, path: &str, allow_local: bool) -> Result<reqwest::Url, String> {
    if base_url.len() > 4096 || base_url.chars().any(|c| c.is_control() || c == ' ') {
        return Err("provider base URL is invalid".into());
    }
    let base = base_url.trim_end_matches('/');
    let joined = if path.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{}", path.trim_start_matches('/'))
    };
    let url = reqwest::Url::parse(&joined).map_err(|_| "provider URL is invalid".to_string())?;
    check_scheme_and_creds(&url, allow_local)?;
    let host = url.host_str().unwrap(); // checked above

    // Bare IP literal — check directly, no DNS.
    if let Ok(ip) = host.parse::<IpAddr>() {
        if ip_blocked(ip, allow_local) {
            return Err(format!("provider URL points at a disallowed address ({ip})"));
        }
        return Ok(url);
    }

    // Hostname — resolve and reject if ANY resolved address is internal (a name
    // resolving to several addresses can't smuggle a private one through).
    let port = url.port_or_known_default().unwrap_or(443);
    let mut saw_any = false;
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("cannot resolve provider host '{host}': {e}"))?;
    for addr in addrs {
        saw_any = true;
        if ip_blocked(addr.ip(), allow_local) {
            return Err(format!(
                "provider host '{host}' resolves to a disallowed address ({})",
                addr.ip()
            ));
        }
    }
    if !saw_any {
        return Err(format!("provider host '{host}' did not resolve"));
    }
    Ok(url)
}

/// A reqwest client for a provider call: redirects DISABLED (a 3xx must never be
/// followed to an unvalidated host). Streaming omits the whole-request timeout so
/// a long SSE response isn't cut off; non-streaming keeps the 120s cap.
pub fn client(streaming: bool) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder()
        .user_agent(concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION")))
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(CONNECT_TIMEOUT)
        // Idle-read guard on BOTH paths: a stalled upstream is reclaimed even on
        // the streaming path, which omits the whole-request timeout below.
        .read_timeout(READ_IDLE_TIMEOUT);
    if !streaming {
        builder = builder.timeout(REQUEST_TIMEOUT);
    }
    builder.build().map_err(|e| format!("http client build failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_non_https_and_internal_targets() {
        assert!(validate("http://api.example.com", "/v1/models", false).is_err(), "http rejected");
        assert!(validate("https://127.0.0.1", "/v1/models", false).is_err(), "loopback rejected");
        assert!(validate("https://10.0.0.5", "/v1/chat", false).is_err(), "private rejected");
        assert!(validate("https://169.254.169.254", "/v1/models", false).is_err(), "metadata rejected");
        assert!(validate("https://user:pw@api.example.com", "/v1", false).is_err(), "creds rejected");
    }

    #[test]
    fn allow_local_permits_loopback_but_not_metadata() {
        // Loopback http is allowed for a declared local endpoint...
        assert!(validate("http://127.0.0.1:11434", "/v1/models", true).is_ok(), "loopback ok when allow_local");
        // ...but the metadata / link-local address never is.
        assert!(validate("http://169.254.169.254", "/v1/models", true).is_err(), "metadata still blocked");
    }

    #[test]
    fn path_joins_without_escaping_base() {
        let u = validate("https://api.openai.com/", "/v1/chat/completions", false).unwrap();
        assert_eq!(u.as_str(), "https://api.openai.com/v1/chat/completions");
    }

    #[test]
    fn syntax_check_is_offline_and_matches_scheme_rules() {
        assert!(check_base_url_syntax("https://api.openai.com", false).is_ok());
        assert!(check_base_url_syntax("http://localhost:11434", false).is_err());
        assert!(check_base_url_syntax("http://localhost:11434", true).is_ok());
        assert!(check_base_url_syntax("https://10.0.0.1", false).is_err());
    }
}
