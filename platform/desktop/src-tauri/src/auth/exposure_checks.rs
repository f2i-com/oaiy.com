//! Checks that do not depend on how an install was configured (design 4.5.5), because configuration can
//! be wrong: a proxy that rewrites `Host` and adds no forwarded header looks like a local client.
//!
//! - On a **local** install a request carrying any forwarded header is refused (`421 proxy_detected`): a
//!   local client, an SSH tunnel and a browser add none of them.
//! - On a **lan** listener a request that carries `Authorization` from a public peer address is refused
//!   (`403 plaintext_from_public_address`) unless `OAIY_ALLOW_PUBLIC_PLAINTEXT=1`: a bearer over plain
//!   HTTP from the internet is sniffable.

use std::net::IpAddr;

use axum::http::HeaderMap;

use super::clientip::unmap;

/// The headers a proxy adds.
pub const FORWARDED_HEADERS: [&str; 8] = [
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "forwarded",
    "via",
    "x-real-ip",
    "cf-connecting-ip",
    "true-client-ip",
];

/// The first forwarded header the request carries, if any.
pub fn forwarded_header(headers: &HeaderMap) -> Option<&'static str> {
    FORWARDED_HEADERS
        .iter()
        .copied()
        .find(|name| headers.contains_key(*name))
}

/// Whether an address is one that only this machine or its network can be: loopback, RFC 1918,
/// link-local, `100.64.0.0/10` (carrier-grade NAT) and `fc00::/7`. Anything else is public.
pub fn is_private_address(ip: IpAddr) -> bool {
    match unmap(ip) {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || (o[0] == 100 && (64..=127).contains(&o[1]))
        }
        IpAddr::V6(v6) => {
            let first = v6.segments()[0];
            v6.is_loopback() || (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80
        }
    }
}

pub fn is_public_address(ip: IpAddr) -> bool {
    !is_private_address(ip)
}

/// Whether the peer is this machine.
pub fn is_loopback(ip: IpAddr) -> bool {
    unmap(ip).is_loopback()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn every_forwarded_header_is_found_whatever_its_case() {
        for name in FORWARDED_HEADERS {
            let mut h = HeaderMap::new();
            h.insert(
                HeaderName::from_bytes(name.to_ascii_uppercase().as_bytes()).unwrap(),
                HeaderValue::from_static("x"),
            );
            assert_eq!(forwarded_header(&h), Some(name), "{name}");
        }
        let mut plain = HeaderMap::new();
        plain.insert("host", HeaderValue::from_static("localhost"));
        plain.insert("authorization", HeaderValue::from_static("Bearer x"));
        plain.insert("origin", HeaderValue::from_static("http://localhost"));
        plain.insert("x-oaiy-session", HeaderValue::from_static("project"));
        assert_eq!(forwarded_header(&plain), None);
        assert_eq!(forwarded_header(&HeaderMap::new()), None);
    }

    #[test]
    fn a_private_address_is_loopback_rfc1918_link_local_cgnat_or_unique_local() {
        for private in [
            "127.0.0.1",
            "127.9.9.9",
            "::1",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "100.127.255.255",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "::ffff:192.168.1.1",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_private_address(ip(private)), "{private}");
            assert!(!is_public_address(ip(private)));
        }
        for public in [
            "203.0.113.9",
            "8.8.8.8",
            "172.32.0.1",
            "172.15.255.255",
            "100.63.255.255",
            "100.128.0.1",
            "192.169.0.1",
            "2001:db8::1",
            "2606:4700::1",
            "fb00::1",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public_address(ip(public)), "{public}");
        }
    }

    #[test]
    fn loopback_means_this_machine_including_a_mapped_address() {
        assert!(
            is_loopback(ip("127.0.0.1"))
                && is_loopback(ip("::1"))
                && is_loopback(ip("::ffff:127.0.0.1"))
                && is_loopback(ip("127.5.5.5"))
        );
        assert!(
            !is_loopback(ip("192.168.1.1"))
                && !is_loopback(ip("203.0.113.9"))
                && !is_loopback(ip("::"))
        );
    }
}
