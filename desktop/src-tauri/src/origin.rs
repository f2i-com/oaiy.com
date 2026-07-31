//! Which addresses may carry a credential over plain `http`.
//!
//! Everything remote must be `https` — a bearer token on the open internet in
//! clear is not a trade worth making. But two kinds of address are legitimately
//! served without TLS and are where people actually develop:
//!
//! * **loopback** — `localhost`, `127.0.0.0/8`, `::1`. Never leaves the machine.
//! * **`.local`** — the mDNS/LAN namespace (RFC 6762). A WAMP or docker
//!   deployment reachable as `formlogic.local` is the normal shape of a local
//!   install, and demanding a certificate for it would mean the only way to use
//!   one is to disable the check entirely.
//!
//! `.local` is a deliberate, narrower concession than loopback: those requests
//! DO cross the local network, so the credential is visible to anything else on
//! that segment. That is acceptable for a machine you administer on a network
//! you control, and it is the user's call to make — but it is why this is a
//! named exception rather than "allow http when it looks internal".
//!
//! Matching is on the parsed HOST, never a substring. `https://evil.example/.local`
//! and `http://evil.example/?x=formlogic.local` must not pass.

/// The host of a URL, lowercased, with any port and userinfo removed.
///
/// `None` when the URL carries userinfo — `http://a@evil.example` is a shape
/// whose host humans routinely misread, so it is refused rather than parsed.
pub fn host_of(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))?;
    // Authority ends at the first path, query or fragment delimiter.
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .filter(|a| !a.is_empty())?;
    if authority.contains('@') {
        return None;
    }
    // Strip the port. An IPv6 literal is bracketed, so only look after ']'.
    let host = match authority.rfind(']') {
        Some(close) => &authority[..=close],
        None => authority.split(':').next().unwrap_or(authority),
    };
    Some(host.to_ascii_lowercase())
}

/// Does this host resolve only to this machine?
pub fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = bare.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
}

/// Is this an mDNS/LAN name (`something.local`)?
///
/// Requires a label before the suffix: a bare `.local` is not a host, and
/// `local` alone is an ordinary single-label name that could be anything.
pub fn is_mdns_host(host: &str) -> bool {
    match host.strip_suffix(".local") {
        Some(label) => !label.is_empty() && !label.ends_with('.'),
        None => false,
    }
}

/// May we send a credential to this URL?
///
/// `https` anywhere, or `http` to loopback / `.local`.
pub fn may_carry_credential(url: &str) -> bool {
    if url.starts_with("https://") {
        return host_of(url).is_some();
    }
    if !url.starts_with("http://") {
        return false;
    }
    match host_of(url) {
        Some(host) => is_loopback_host(&host) || is_mdns_host(&host),
        None => false,
    }
}

/// The message to show when [`may_carry_credential`] refuses.
///
/// One sentence naming every accepted shape, because "must be https" alone left
/// a user with a working `.local` deployment guessing.
pub const INSECURE_ADDRESS_HELP: &str =
    "the address must be https, or plain http only for localhost, an IP on this \
     machine, or a .local name";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_is_always_acceptable() {
        assert!(may_carry_credential("https://formlogic.com"));
        assert!(may_carry_credential("https://formlogic.com:8443/api/v1"));
        assert!(may_carry_credential("https://formlogic.local"));
    }

    #[test]
    fn plain_http_to_a_local_name_is_accepted() {
        // The case this exists for: a WAMP install reachable as formlogic.local.
        assert!(may_carry_credential("http://formlogic.local"));
        assert!(may_carry_credential("http://formlogic.local/api/v1"));
        assert!(may_carry_credential("http://formlogic.local:8080"));
        assert!(may_carry_credential("http://FormLogic.LOCAL"), "host is case-insensitive");
        assert!(may_carry_credential("http://api.formlogic.local"));
    }

    #[test]
    fn plain_http_to_loopback_is_accepted() {
        assert!(may_carry_credential("http://localhost:17972"));
        assert!(may_carry_credential("http://127.0.0.1"));
        assert!(may_carry_credential("http://127.0.0.53:8080"), "all of 127/8 is loopback");
        assert!(may_carry_credential("http://[::1]:8080"));
        assert!(may_carry_credential("http://app.localhost"));
    }

    #[test]
    fn plain_http_to_anywhere_else_is_refused() {
        assert!(!may_carry_credential("http://formlogic.com"));
        assert!(!may_carry_credential("http://192.168.1.10"), "a LAN IP is not a .local name");
        assert!(!may_carry_credential("http://10.0.0.5:8080"));
    }

    #[test]
    fn the_suffix_is_matched_on_the_host_not_the_string() {
        // The attack a `contains(".local")` check would wave through: the token
        // would go to evil.example in clear.
        assert!(!may_carry_credential("http://evil.example/.local"));
        assert!(!may_carry_credential("http://evil.example/?next=formlogic.local"));
        assert!(!may_carry_credential("http://evil.example#formlogic.local"));
        assert!(!may_carry_credential("http://evil.example/a/formlogic.local"));
        // …and a host that merely ENDS in the letters, without the dot.
        assert!(!may_carry_credential("http://notlocal"));
        assert!(!may_carry_credential("http://evillocal"));
    }

    #[test]
    fn userinfo_is_refused_rather_than_parsed() {
        // `http://formlogic.local@evil.example` reads as the .local host to a
        // human and resolves to evil.example. Refusing beats guessing.
        assert!(!may_carry_credential("http://formlogic.local@evil.example"));
        assert!(!may_carry_credential("https://user:pass@formlogic.com"));
        assert!(host_of("http://a@b.local").is_none());
    }

    #[test]
    fn a_bare_local_suffix_is_not_a_host() {
        assert!(!is_mdns_host(".local"));
        assert!(!is_mdns_host("local"));
        assert!(!is_mdns_host("..local"));
        assert!(is_mdns_host("a.local"));
    }

    #[test]
    fn host_parsing_strips_ports_and_paths() {
        assert_eq!(host_of("https://a.example:8443/x?y#z").as_deref(), Some("a.example"));
        assert_eq!(host_of("http://[::1]:9/x").as_deref(), Some("[::1]"));
        assert_eq!(host_of("http://[::1]").as_deref(), Some("[::1]"));
        assert_eq!(host_of("ftp://a.example"), None, "only http(s)");
        assert_eq!(host_of("http:///nohost"), None);
    }
}
