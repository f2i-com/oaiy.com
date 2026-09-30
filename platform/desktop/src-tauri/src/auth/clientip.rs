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
    /// `203.0.113.0/24`, `2001:db8::/32`, or a bare address (a host route). Nothing is guessed: the address and
    /// the prefix are read as they are written, with no space inside the entry, and a prefix is digits only
    /// (Rust's own integer parser would take `+8`). An address written in the IPv4-mapped form
    /// (`::ffff:10.0.0.0/104`) is the IPv4 network with the prefix 96 bits shorter, and with a prefix below 96 it
    /// is refused: that would name every mapped address and much else, which nobody who wrote it means.
    pub fn parse(text: &str) -> Option<Cidr> {
        let text = text.trim();
        let (addr, prefix) = match text.split_once('/') {
            Some((a, p)) => (a, Some(p)),
            None => (text, None),
        };
        let written = addr.parse::<IpAddr>().ok()?;
        let ip = unmap(written);
        let mapped = written != ip;
        let prefix = match prefix {
            None => {
                if ip.is_ipv4() {
                    32
                } else {
                    128
                }
            }
            Some(p) => {
                if p.is_empty() || p.len() > 3 || !p.bytes().all(|b| b.is_ascii_digit()) {
                    return None;
                }
                let n: u16 = p.parse().ok()?;
                let prefix = if mapped { n.checked_sub(96)? } else { n };
                let max = if ip.is_ipv4() { 32 } else { 128 };
                if prefix > max {
                    return None;
                }
                prefix as u8
            }
        };
        Some(Cidr {
            net: mask(ip, prefix),
            prefix,
        })
    }

    /// What the network is, for the cross-check against another implementation: whether it is IPv4, its
    /// address as an integer and its prefix length.
    #[cfg(test)]
    pub(crate) fn parts(&self) -> (bool, u128, u8) {
        match self.net {
            IpAddr::V4(v4) => (true, u128::from(u32::from(v4)), self.prefix),
            IpAddr::V6(v6) => (false, u128::from(v6), self.prefix),
        }
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

/// One entry of `X-Forwarded-For` (already trimmed of its spaces) as an address: an IPv4 or IPv6 literal exactly
/// as the standard library reads one (no port, no brackets, no zone, no leading zeros in an IPv4 address), an
/// IPv4-mapped address as the IPv4 address it stands for. Anything else is `None`, and one such entry makes the
/// whole header unusable.
pub fn parse_forwarded_ip(entry: &str) -> Option<IpAddr> {
    entry.parse::<IpAddr>().ok().map(unmap)
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
        match parse_forwarded_ip(entry) {
            Some(ip) => parsed.push(ip),
            // One unparsable entry anywhere: nothing in the header can be believed.
            None => return direct(true),
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

    // ---- ACC-14: the parsers are read as written, and nothing that is not an address is one ---------

    #[test]
    fn a_network_is_read_as_written_with_no_leniency_in_the_prefix_or_the_spaces() {
        // Rust's integer parser takes a leading `+`, and the old code trimmed inside the entry: neither is a network.
        for bad in [
            "10.0.0.0/+8",
            "10.0.0.0/ 8",
            "10.0.0.0 /8",
            "10.0.0.0/8 /8",
            "10.0.0.0/8/8",
            "10.0.0.0/0x8",
            "10.0.0.0/0008",
            "10.0.0.0/",
            "/8",
            "::1/+128",
            "::1/ 128",
            "fe80::1%eth0/64",
            "[::1]/128",
            "10.0.0.5, 10.0.0.6",
            "10.0.0.0/8\u{0}",
            "\u{ff11}\u{ff10}.0.0.0/8",
        ] {
            assert_eq!(Cidr::parse(bad), None, "{bad:?}");
        }
        // What is fine: the whole entry may have spaces around it, and a prefix may have a leading zero.
        assert!(Cidr::parse("  10.0.0.0/8\t").is_some());
        assert_eq!(Cidr::parse("10.0.0.0/08"), Cidr::parse("10.0.0.0/8"));
        assert_eq!(Cidr::parse("10.0.0.0/0"), Cidr::parse("0.0.0.0/0"));
    }

    #[test]
    fn a_network_written_in_the_mapped_form_is_the_ipv4_network_and_below_96_bits_it_is_refused() {
        // `::ffff:10.0.0.0/104` is 10.0.0.0/8: the first 96 bits are the mapping.
        assert_eq!(
            Cidr::parse("::ffff:10.0.0.0/104"),
            Cidr::parse("10.0.0.0/8")
        );
        assert_eq!(Cidr::parse("::ffff:10.0.0.5"), Cidr::parse("10.0.0.5/32"));
        assert_eq!(
            Cidr::parse("::ffff:10.0.0.5/128"),
            Cidr::parse("10.0.0.5/32")
        );
        assert_eq!(Cidr::parse("::ffff:0:0/96"), Cidr::parse("0.0.0.0/0"));
        // Below 96 it would name every mapped address and much else (`::/8` is not `10.0.0.0/8`): refused, not
        // guessed at (the earlier code read it as the IPv4 network with that prefix, which is neither).
        for bad in [
            "::ffff:10.0.0.0/8",
            "::ffff:10.0.0.0/0",
            "::ffff:10.0.0.0/95",
            "::ffff:10.0.0.0/129",
        ] {
            assert_eq!(Cidr::parse(bad), None, "{bad:?}");
        }
        // The mapped peer of a v4 network, and a v4 network that is not a v6 one.
        let n = Cidr::parse("::ffff:10.0.0.0/104").unwrap();
        assert!(n.contains(ip("10.9.9.9")) && n.contains(ip("::ffff:10.9.9.9")));
        assert!(!n.contains(ip("11.0.0.1")) && !n.contains(ip("::1")));
        // `::/0` is every IPv6 address; a mapped address is an IPv4 one, so it is not in it.
        let all6 = Cidr::parse("::/0").unwrap();
        assert!(all6.contains(ip("2001:db8::1")) && !all6.contains(ip("::ffff:1.2.3.4")));
    }

    #[test]
    fn a_forwarded_entry_is_an_address_as_the_standard_library_reads_one_and_nothing_else() {
        for good in [
            "203.0.113.9",
            "2001:db8::5",
            "2001:DB8:0:0:0:0:0:5",
            "::1",
            "::ffff:203.0.113.9",
            "::ffff:cb00:7109",
            "0:0:0:0:0:ffff:203.0.113.9",
        ] {
            assert!(parse_forwarded_ip(good).is_some(), "{good}");
        }
        assert_eq!(
            parse_forwarded_ip("::FFFF:203.0.113.9"),
            Some(ip("203.0.113.9")),
            "a mapped address is the IPv4 address"
        );
        for bad in [
            "",
            " ",
            "unknown",
            "_hidden",
            "for=203.0.113.9",
            "\"203.0.113.9\"",
            "203.0.113.9:80",
            "[2001:db8::5]",
            "[2001:db8::5]:80",
            "203.0.113",
            "203.0.113.9.1",
            "203.0.113.09",
            "203.0.113.256",
            "0x7f.0.0.1",
            "2130706433",
            "127.1",
            "fe80::1%eth0",
            "203.0.113.9/32",
            "203.0.113.9\u{0}",
            " 203.0.113.9",
            "203.0.113.9 ",
            "\u{ff12}\u{ff10}\u{ff13}.0.113.9",
        ] {
            assert_eq!(parse_forwarded_ip(bad), None, "{bad:?}");
        }
    }

    /// A small deterministic generator: the tests here run the same way every time, and need no new crate.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
            &items[self.below(items.len() as u64) as usize]
        }
    }

    fn random_ip(r: &mut Xorshift) -> IpAddr {
        match r.below(4) {
            0 => IpAddr::V4(Ipv4Addr::from(r.next() as u32)),
            1 => IpAddr::V6(Ipv6Addr::from(
                ((r.next() as u128) << 64) | r.next() as u128,
            )),
            2 => IpAddr::V6(Ipv4Addr::from(r.next() as u32).to_ipv6_mapped()),
            _ => {
                // Near the networks the trusted lists name, so that trusted and untrusted addresses both come up.
                let base = *r.pick(&[
                    [10u8, 0, 0, 0],
                    [172, 30, 0, 0],
                    [127, 0, 0, 0],
                    [203, 0, 113, 0],
                ]);
                let mut o = base;
                o[3] = r.below(256) as u8;
                o[2] |= (r.below(3)) as u8;
                IpAddr::V4(Ipv4Addr::from(o))
            }
        }
    }

    fn random_trusted(r: &mut Xorshift) -> TrustedProxies {
        let pool = [
            "127.0.0.1/32",
            "::1/128",
            "10.0.0.0/8",
            "172.30.0.0/24",
            "203.0.113.0/25",
            "2001:db8::/32",
            "::ffff:10.0.0.0/104",
            "192.168.0.0/16",
        ];
        let n = 1 + r.below(3);
        let list: Vec<&str> = (0..n).map(|_| *r.pick(&pool)).collect();
        TrustedProxies::parse_list(&list.join(",")).0
    }

    /// The rules that must hold whatever the request says (design 4.5.4): a fuzz over peers, lists and headers,
    /// with garbage among the entries.
    #[test]
    fn whatever_the_header_says_the_derivation_holds_the_rules_of_the_design() {
        let mut r = Xorshift(0x9E37_79B9_7F4A_7C15);
        let junk = [
            "",
            " ",
            "unknown",
            "_x",
            "1.2.3",
            "1.2.3.4:80",
            "::g",
            "[::1]",
            "fe80::1%1",
            "01.2.3.4",
            "1.2.3.4/24",
            "for=1.2.3.4",
            "\"1.2.3.4\"",
        ];
        for round in 0..60_000 {
            let trusted = random_trusted(&mut r);
            let peer = random_ip(&mut r);
            let mut entries: Vec<String> = Vec::new();
            for _ in 0..r.below(5) {
                entries.push(if r.below(6) == 0 {
                    (*r.pick(&junk)).to_string()
                } else {
                    random_ip(&mut r).to_string()
                });
            }
            let one_line = entries.join(", ");
            let lines: Vec<&str> = if entries.is_empty() {
                vec![]
            } else if r.below(2) == 0 {
                vec![one_line.as_str()]
            } else {
                entries.iter().map(String::as_str).collect()
            };
            let c = client_ip(peer, &lines, &trusted);
            let peer_trusted = trusted.contains(peer);
            // 1. An untrusted peer is itself, whatever it sends.
            if !peer_trusted {
                assert_eq!(
                    (c.ip, c.via_proxy, c.fell_back),
                    (unmap(peer), false, false),
                    "round {round}: {peer} {lines:?}"
                );
            }
            // 2. A believed address came from a trusted peer, is not a trusted proxy, and is in the header.
            if c.via_proxy {
                assert!(peer_trusted && !trusted.contains(c.ip), "round {round}");
                let listed = lines
                    .iter()
                    .flat_map(|l| l.split(','))
                    .filter_map(|e| parse_forwarded_ip(e.trim()))
                    .any(|ip| ip == c.ip);
                assert!(listed, "round {round}: {} not in {lines:?}", c.ip);
            }
            // 3. A fall back is the trusted peer itself, and says so.
            if c.fell_back {
                assert!(
                    peer_trusted && !c.via_proxy && c.ip == unmap(peer),
                    "round {round}"
                );
            }
            // 4. What is left is the peer, with nothing to warn of.
            if !c.via_proxy && !c.fell_back {
                assert_eq!(c.ip, unmap(peer), "round {round}");
            }
            // 5. The key is the address's own bucket.
            assert_eq!(c.key, bucket_key(c.ip), "round {round}");
            // 6. What a client controls is the left of the header: once a client has been found, addresses put in
            // front of the header (another line before the proxy's, or more entries on the left of it) never
            // change who it is.
            if c.via_proxy {
                let mut longer = vec!["198.51.100.66, 2001:db8:dead::1".to_string()];
                longer.extend(lines.iter().map(|l| l.to_string()));
                let refs: Vec<&str> = longer.iter().map(String::as_str).collect();
                let again = client_ip(peer, &refs, &trusted);
                assert_eq!(
                    (again.ip, again.via_proxy, again.fell_back),
                    (c.ip, c.via_proxy, c.fell_back),
                    "round {round}: {peer} {lines:?}"
                );
            }
            // 7. The other headers a proxy or a client may add are never read.
        }
    }

    #[test]
    fn the_parsers_take_any_text_without_panicking() {
        let mut r = Xorshift(0xDEAD_BEEF_CAFE_F00D);
        let alphabet: Vec<char> =
            "0123456789abcdefABCDEF.:/[]%,;+- \t\r\n\u{0}\u{e9}\u{ff11}xz_@\\?#"
                .chars()
                .collect();
        for _ in 0..80_000 {
            let len = r.below(48) as usize;
            let text: String = (0..len).map(|_| *r.pick(&alphabet)).collect();
            let _ = Cidr::parse(&text);
            let _ = parse_forwarded_ip(&text);
            let (t, _) = TrustedProxies::parse_list(&text);
            let _ = t.contains(ip("10.1.2.3"));
            let _ = crate::auth::host::HostName::parse(&text);
            let lines = [text.as_str()];
            let _ = client_ip(ip("127.0.0.1"), &lines, &TrustedProxies::loopback());
        }
    }
}
