//! Who is asking, by address (design 4.5.4).
//!
//! **Trusted proxies.** `OAIY_TRUSTED_PROXIES` is a list of IPs or CIDRs whose `X-Forwarded-For` is
//! believed. Default: `127.0.0.1/32,::1/128` when `OAIY_PUBLIC_URL` is set, none otherwise. IPv4-mapped
//! IPv6 peers (`::ffff:a.b.c.d`) are unmapped before matching.
//!
//! **Effective client address.** If the peer is not trusted it is the peer and every forwarded header is
//! ignored. If the peer is trusted, all `X-Forwarded-For` lines are joined with `,` and walked from the
//! right: the first entry that is not a trusted proxy is the client. One unparsable entry, a trusted peer
//! with no header, or a header made only of trusted proxies falls back to the peer (fail safe: all clients
//! then share the proxy's limits). IPv6 clients are keyed by their /64.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// An IPv4-mapped IPv6 address as the IPv4 address it stands for.
pub fn unmap(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        v4 => v4,
    }
}

/// A network: an address and a prefix length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cidr {
    net: IpAddr,
    prefix: u8,
}

impl Cidr {
    /// `203.0.113.0/24`, `2001:db8::/32`, or a bare address (a host route).
    pub fn parse(text: &str) -> Option<Cidr> {
        let text = text.trim();
        let (addr, prefix) = match text.split_once('/') {
            Some((a, p)) => (a, Some(p)),
            None => (text, None),
        };
        let ip = unmap(addr.trim().parse::<IpAddr>().ok()?);
        let max = if ip.is_ipv4() { 32 } else { 128 };
        let prefix = match prefix {
            // A mapped address written with a /prefix is a v6 prefix that means a v4 one 96 bits shorter.
            Some(p) => {
                let n: u8 = p.trim().parse().ok()?;
                if addr.contains(':') && ip.is_ipv4() && n >= 96 {
                    n - 96
                } else {
                    n
                }
            }
            None => max,
        };
        (prefix <= max).then(|| Cidr {
            net: mask(ip, prefix),
            prefix,
        })
    }

    pub fn contains(&self, ip: IpAddr) -> bool {
        let ip = unmap(ip);
        ip.is_ipv4() == self.net.is_ipv4() && mask(ip, self.prefix) == self.net
    }

    /// `203.0.113.0/24`: the network as `parse` reads it back.
    pub fn describe(&self) -> String {
        format!("{}/{}", self.net, self.prefix)
    }
}

/// `ip` with everything past `prefix` bits cleared.
fn mask(ip: IpAddr, prefix: u8) -> IpAddr {
    match ip {
        IpAddr::V4(v4) => {
            let bits = u32::from(v4);
            let m = if prefix == 0 {
                0
            } else {
                u32::MAX << (32 - u32::from(prefix.min(32)))
            };
            IpAddr::V4(Ipv4Addr::from(bits & m))
        }
        IpAddr::V6(v6) => {
            let bits = u128::from(v6);
            let m = if prefix == 0 {
                0
            } else {
                u128::MAX << (128 - u32::from(prefix.min(128)))
            };
            IpAddr::V6(Ipv6Addr::from(bits & m))
        }
    }
}

/// The peers whose `X-Forwarded-*` headers are believed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrustedProxies(Vec<Cidr>);

impl TrustedProxies {
    /// No proxy is trusted.
    pub fn none() -> Self {
        Self::default()
    }

    /// `127.0.0.1/32` and `::1/128`: a proxy on the same machine.
    pub fn loopback() -> Self {
        TrustedProxies(vec![
            Cidr::parse("127.0.0.1/32").unwrap(),
            Cidr::parse("::1/128").unwrap(),
        ])
    }

    /// A comma list of addresses and networks, and the entries that could not be read.
    pub fn parse_list(text: &str) -> (Self, Vec<String>) {
        let mut cidrs = Vec::new();
        let mut rejected = Vec::new();
        for entry in text.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match Cidr::parse(entry) {
                Some(c) => cidrs.push(c),
                None => rejected.push(entry.to_string()),
            }
        }
        (TrustedProxies(cidrs), rejected)
    }

    pub fn contains(&self, ip: IpAddr) -> bool {
        self.0.iter().any(|c| c.contains(ip))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The networks, as text: what the startup audit event says the install trusts.
    pub fn describe(&self) -> Vec<String> {
        self.0.iter().map(Cidr::describe).collect()
    }
}

/// The effective client of a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientIp {
    pub ip: IpAddr,
    /// The throttle's key: the address, or for IPv6 the /64 (`20010db800000000`).
    pub key: String,
    /// The address came from `X-Forwarded-For` (the peer is a trusted proxy that named a client).
    pub via_proxy: bool,
    /// The peer is a trusted proxy but its header could not be used, so the peer stands for everyone.
    pub fell_back: bool,
}

/// The throttle key of an address: IPv4 as it is, IPv6 by its first 64 bits in hex.
pub fn bucket_key(ip: IpAddr) -> String {
    match unmap(ip) {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => {
            let bits = (u128::from(v6) >> 64) as u64;
            format!("{bits:016x}")
        }
    }
}

/// The effective client for a request from `peer` carrying these `X-Forwarded-For` header lines.
pub fn client_ip(peer: IpAddr, forwarded_for: &[&str], trusted: &TrustedProxies) -> ClientIp {
    let peer = unmap(peer);
    let direct = |fell_back: bool| ClientIp {
        ip: peer,
        key: bucket_key(peer),
        via_proxy: false,
        fell_back,
    };
    if !trusted.contains(peer) {
        return direct(false);
    }
    // Every header line, joined with commas, then each entry.
    let entries: Vec<&str> = forwarded_for
        .iter()
        .flat_map(|line| line.split(','))
        .map(str::trim)
        .collect();
    if entries.is_empty() || entries.iter().all(|e| e.is_empty()) {
        return direct(false);
    }
    let mut parsed = Vec::with_capacity(entries.len());
    for entry in entries {
        match entry.parse::<IpAddr>() {
            Ok(ip) => parsed.push(unmap(ip)),
            // One unparsable entry anywhere: nothing in the header can be believed.
            Err(_) => return direct(true),
        }
    }
    match parsed.iter().rev().find(|ip| !trusted.contains(**ip)) {
        Some(client) => ClientIp {
            ip: *client,
            key: bucket_key(*client),
            via_proxy: true,
            fell_back: false,
        },
        // A header made only of trusted proxies.
        None => direct(true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    fn effective(peer: &str, xff: &[&str], trusted: &str) -> (String, String) {
        let (t, rejected) = TrustedProxies::parse_list(trusted);
        assert!(rejected.is_empty(), "{rejected:?}");
        let c = client_ip(ip(peer), xff, &t);
        (c.ip.to_string(), c.key)
    }

    #[test]
    fn f9_the_trusted_proxies_are_described_as_networks_that_read_back() {
        let (t, rejected) =
            TrustedProxies::parse_list("10.0.0.5/8, 192.0.2.7, 2001:db8:1::/32, ::1");
        assert!(rejected.is_empty());
        assert_eq!(
            t.describe(),
            ["10.0.0.0/8", "192.0.2.7/32", "2001:db8::/32", "::1/128"]
        );
        // What it says reads back as the same set.
        let (again, _) = TrustedProxies::parse_list(&t.describe().join(","));
        assert_eq!(again, t);
        assert!(TrustedProxies::none().describe().is_empty());
        assert_eq!(
            TrustedProxies::loopback().describe(),
            ["127.0.0.1/32", "::1/128"]
        );
    }

    /// The design's reference table (4.5.4, vectors2.mjs and vectors4.mjs): 11 rows.
    #[test]
    fn the_eleven_reference_vectors_of_the_design() {
        let loopback = "127.0.0.1/32, ::1/128";
        for (peer, xff, trusted, want) in [
            ("127.0.0.1", vec!["203.0.113.9"], loopback, "203.0.113.9"),
            (
                "127.0.0.1",
                vec!["198.51.100.7, 203.0.113.9"],
                loopback,
                "203.0.113.9",
            ),
            (
                "127.0.0.1",
                vec!["203.0.113.9, 127.0.0.1"],
                loopback,
                "203.0.113.9",
            ),
            (
                "::ffff:127.0.0.1",
                vec!["203.0.113.9"],
                loopback,
                "203.0.113.9",
            ),
            ("127.0.0.1", vec![], loopback, "127.0.0.1"),
            ("127.0.0.1", vec!["not-an-ip"], loopback, "127.0.0.1"),
            ("203.0.113.50", vec!["10.0.0.1"], loopback, "203.0.113.50"),
            (
                "10.0.0.5",
                vec!["203.0.113.9, 10.0.0.7"],
                "10.0.0.0/24",
                "203.0.113.9",
            ),
            (
                "172.30.0.3",
                vec!["203.0.113.9"],
                "172.30.0.0/24",
                "203.0.113.9",
            ),
            ("172.30.0.3", vec!["203.0.113.9"], loopback, "172.30.0.3"),
            ("::1", vec!["2001:db8::5"], loopback, "2001:db8::5"),
        ] {
            let (got, key) = effective(peer, &xff, trusted);
            assert_eq!(got, want, "peer {peer} xff {xff:?} trusted {trusted}");
            if want == "2001:db8::5" {
                assert_eq!(
                    key, "20010db800000000",
                    "an IPv6 client is keyed by its /64"
                );
            }
        }
    }

    #[test]
    fn an_untrusted_peer_is_itself_whatever_it_says() {
        for xff in [vec!["10.0.0.1"], vec!["1.2.3.4, 5.6.7.8"], vec![]] {
            let c = client_ip(ip("203.0.113.50"), &xff, &TrustedProxies::loopback());
            assert_eq!(
                (c.ip, c.via_proxy, c.fell_back),
                (ip("203.0.113.50"), false, false)
            );
        }
        // With no proxy trusted at all, even loopback is just a peer.
        let c = client_ip(ip("127.0.0.1"), &["203.0.113.9"], &TrustedProxies::none());
        assert_eq!(c.ip, ip("127.0.0.1"));
    }

    #[test]
    fn the_client_is_the_first_entry_from_the_right_that_is_not_a_trusted_proxy() {
        let t = TrustedProxies::parse_list("10.0.0.0/8").0;
        // The left entries are client-supplied and never used while a right one is not a proxy.
        let c = client_ip(
            ip("10.1.1.1"),
            &["6.6.6.6, 203.0.113.9, 10.2.2.2, 10.3.3.3"],
            &t,
        );
        assert_eq!((c.ip, c.via_proxy), (ip("203.0.113.9"), true));
        // Several header lines are one list.
        let c = client_ip(ip("10.1.1.1"), &["6.6.6.6", "203.0.113.9", "10.2.2.2"], &t);
        assert_eq!(c.ip, ip("203.0.113.9"));
        // Spaces and empty entries between commas are tolerated only if the rest parses.
        let c = client_ip(ip("10.1.1.1"), &["  203.0.113.9  ,10.2.2.2"], &t);
        assert_eq!(c.ip, ip("203.0.113.9"));
    }

    #[test]
    fn a_header_that_cannot_be_used_falls_back_to_the_peer_and_says_so() {
        let t = TrustedProxies::loopback();
        for xff in [
            vec!["garbage"],
            vec!["203.0.113.9, garbage"],
            vec!["garbage, 203.0.113.9"],
            vec!["127.0.0.1"],
            vec!["127.0.0.1, ::1"],
            vec!["203.0.113.9:8080"],
            vec!["203.0.113.9,,"],
        ] {
            let c = client_ip(ip("127.0.0.1"), &xff, &t);
            assert_eq!(c.ip, ip("127.0.0.1"), "{xff:?}");
            assert!(!c.via_proxy, "{xff:?}");
            assert!(c.fell_back, "{xff:?}");
        }
        // No header from a trusted peer: the peer itself, and nothing to warn about.
        let c = client_ip(ip("127.0.0.1"), &[], &t);
        assert!(!c.fell_back && !c.via_proxy);
        let c = client_ip(ip("127.0.0.1"), &[""], &t);
        assert!(!c.fell_back && !c.via_proxy);
    }

    #[test]
    fn an_ipv6_client_is_keyed_by_its_slash_64_and_an_ipv4_one_by_itself() {
        assert_eq!(
            bucket_key(ip("2001:db8:0:1:aaaa:bbbb:cccc:dddd")),
            "20010db800000001"
        );
        assert_eq!(
            bucket_key(ip("2001:db8:0:1::1")),
            bucket_key(ip("2001:db8:0:1:ffff::2"))
        );
        assert_ne!(
            bucket_key(ip("2001:db8:0:1::1")),
            bucket_key(ip("2001:db8:0:2::1"))
        );
        assert_eq!(bucket_key(ip("203.0.113.9")), "203.0.113.9");
        assert_eq!(
            bucket_key(ip("::ffff:203.0.113.9")),
            "203.0.113.9",
            "a mapped address is the v4 address"
        );
    }

    #[test]
    fn cidrs_match_by_prefix_and_family() {
        let c = |s: &str| Cidr::parse(s).unwrap();
        assert!(
            c("10.0.0.0/24").contains(ip("10.0.0.255"))
                && !c("10.0.0.0/24").contains(ip("10.0.1.0"))
        );
        assert!(c("0.0.0.0/0").contains(ip("203.0.113.9")) && !c("0.0.0.0/0").contains(ip("::1")));
        assert!(
            c("127.0.0.1").contains(ip("127.0.0.1")) && !c("127.0.0.1").contains(ip("127.0.0.2"))
        );
        assert!(c("::1/128").contains(ip("::1")) && !c("::1/128").contains(ip("::2")));
        assert!(
            c("2001:db8::/32").contains(ip("2001:db8:ffff::1"))
                && !c("2001:db8::/32").contains(ip("2001:db9::1"))
        );
        assert!(
            c("127.0.0.1/32").contains(ip("::ffff:127.0.0.1")),
            "a mapped peer matches a v4 network"
        );
        assert!(
            !c("10.0.0.0/8").contains(ip("2001:db8::1")),
            "a v4 network never contains a v6 address"
        );
        // Host bits in the network are ignored, as an operator would expect.
        assert!(c("10.0.0.7/24").contains(ip("10.0.0.99")));
        for bad in [
            "",
            "10.0.0.0/33",
            "::1/129",
            "10.0.0/8",
            "not-an-ip",
            "10.0.0.0/-1",
            "10.0.0.0/x",
            "1.2.3.4/",
        ] {
            assert_eq!(Cidr::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_list_reports_what_it_could_not_read_and_keeps_the_rest() {
        let (t, rejected) =
            TrustedProxies::parse_list("127.0.0.1, bogus, 10.0.0.0/8 ,, 10.0.0.0/99");
        assert_eq!(rejected, ["bogus", "10.0.0.0/99"]);
        assert!(
            t.contains(ip("127.0.0.1")) && t.contains(ip("10.9.9.9")) && !t.contains(ip("8.8.8.8"))
        );
        assert!(TrustedProxies::parse_list("").0.is_empty());
        assert!(TrustedProxies::none().is_empty() && !TrustedProxies::loopback().is_empty());
        assert!(TrustedProxies::loopback().contains(ip("::ffff:127.0.0.1")));
    }
}
