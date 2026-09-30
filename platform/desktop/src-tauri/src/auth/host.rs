//! The Host allow-list and the channel (design 4.5.4).
//!
//! `Host` is what the proxy forwards; `X-Forwarded-Host` is not used. A `Host` that is not ours is
//! `421 misdirected_host`: that is also what stops DNS rebinding (a rebound page carries its own name in
//! `Host`, never a loopback name).
//!
//! | Install | Allowed `Host` values |
//! |---|---|
//! | Every install | `localhost`, `127.0.0.1` and `[::1]` on **any port**, and `OAIY_ALLOWED_HOSTS` |
//! | Loopback server | the above plus `dash.oaiy.localhost`, `agent.oaiy.localhost`, `flows.oaiy.localhost` |
//! | Proxied | the hosts of `OAIY_PUBLIC_URL`, `OAIY_AGENT_URL`, `OAIY_FLOWS_URL`; the loopback names only from a direct loopback peer |
//! | LAN | any IP-literal `Host` with the bound port |
//! | Desktop | the first row only |
//!
//! Exempt: `GET` and `HEAD /api/health`, so a probe (its `Host` is a pod address) is not `421`.

use std::collections::{BTreeMap, BTreeSet};

use super::mode::Exposure;
use super::presets::App;

/// A `Host` (or a configured host) as `host` and optional `port`, lowercased, with the port of the
/// scheme's default (`:443`, `:80`) removed: `dash.example.com`, not `dash.example.com:443`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HostName {
    /// Lowercase; an IPv6 literal keeps its brackets.
    pub host: String,
    pub port: Option<u16>,
}

impl HostName {
    /// `None` for an empty value, a value with userinfo, a path, a space or other junk, a bad port, or a
    /// bracket that does not close.
    pub fn parse(value: &str) -> Option<HostName> {
        let v = value.trim();
        if v.is_empty()
            || v.len() > 255
            || v.bytes().any(|b| {
                b.is_ascii_control()
                    || b == b' '
                    || b == b'/'
                    || b == b'@'
                    || b == b'\\'
                    || b == b'?'
                    || b == b'#'
            })
            || !v.is_ascii()
        {
            return None;
        }
        let v = v.to_ascii_lowercase();
        let (host, port) = if let Some(rest) = v.strip_prefix('[') {
            let (inside, after) = rest.split_once(']')?;
            if inside.is_empty()
                || !inside
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
            {
                return None;
            }
            let port = match after {
                "" => None,
                a => Some(a.strip_prefix(':')?),
            };
            (format!("[{inside}]"), port)
        } else {
            match v.rsplit_once(':') {
                Some((h, p)) => {
                    if h.contains(':') {
                        // An unbracketed IPv6 literal is not a Host value.
                        return None;
                    }
                    (h.to_string(), Some(p))
                }
                None => (v.clone(), None),
            }
        };
        if host.is_empty() || host.starts_with('.') || host.ends_with("..") || host == "[" {
            return None;
        }
        if !host.starts_with('[')
            && !host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.' || b == b'_')
        {
            return None;
        }
        let port = match port {
            None => None,
            Some(p) => {
                let n: u16 = p.parse().ok().filter(|n| *n != 0)?;
                (n != 80 && n != 443).then_some(n)
            }
        };
        Some(HostName { host, port })
    }

    /// `localhost`, `127.0.0.1` or `[::1]`, on any port.
    pub fn is_loopback_name(&self) -> bool {
        matches!(self.host.as_str(), "localhost" | "127.0.0.1" | "[::1]")
    }

    /// An IP literal (v4 or bracketed v6).
    pub fn is_ip_literal(&self) -> bool {
        self.host.starts_with('[') || self.host.parse::<std::net::Ipv4Addr>().is_ok()
    }

    /// `host` or `host:port`, the port omitted when it is a default.
    pub fn display(&self) -> String {
        match self.port {
            Some(p) => format!("{}:{p}", self.host),
            None => self.host.clone(),
        }
    }
}

/// What a `Host` turned out to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostClass {
    /// `localhost`, `127.0.0.1`, `[::1]`.
    Loopback,
    /// One of the three app names on loopback (`agent.oaiy.localhost`).
    LoopbackApp(App),
    /// One of the configured public hosts.
    Public(App),
    /// An IP literal on a LAN listener.
    LanAddress,
    /// Named in `OAIY_ALLOWED_HOSTS`.
    Extra,
}

/// Why a request is not for this server.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Misdirected;

/// The set of names this install answers to.
#[derive(Clone, Debug)]
pub struct HostPolicy {
    exposure: Exposure,
    /// The listener's port, for a LAN listener's IP-literal hosts.
    port: u16,
    extra: BTreeSet<HostName>,
    public: BTreeMap<HostName, App>,
    /// `*.oaiy.localhost` names (the loopback server, not the desktop).
    loopback_apps: bool,
}

/// `dash.oaiy.localhost` and its two siblings; none is `oaiy.localhost`, which the Agent uses to detect
/// "OAIY's own window".
pub fn loopback_app_of(host: &str) -> Option<App> {
    match host {
        "dash.oaiy.localhost" => Some(App::Dash),
        "agent.oaiy.localhost" => Some(App::Agent),
        "flows.oaiy.localhost" => Some(App::Flows),
        _ => None,
    }
}

impl HostPolicy {
    pub fn new(
        exposure: Exposure,
        port: u16,
        extra: BTreeSet<HostName>,
        public: BTreeMap<HostName, App>,
        loopback_apps: bool,
    ) -> HostPolicy {
        HostPolicy {
            exposure,
            port,
            extra,
            public,
            loopback_apps,
        }
    }

    /// The hosts an install with these settings answers to; `direct_loopback` is whether the peer is this
    /// machine with no forwarded header (the only peer a proxied install answers loopback names to).
    pub fn classify(
        &self,
        host: &HostName,
        direct_loopback: bool,
    ) -> Result<HostClass, Misdirected> {
        if self.extra.contains(host) {
            return Ok(HostClass::Extra);
        }
        if host.is_loopback_name() {
            // A proxied install answers loopback names only to a direct loopback peer (the CLI on the
            // server); through the proxy they would be a way around the public host names.
            return if self.exposure != Exposure::Proxied || direct_loopback {
                Ok(HostClass::Loopback)
            } else {
                Err(Misdirected)
            };
        }
        if self.loopback_apps {
            if let Some(app) = loopback_app_of(&host.host) {
                return Ok(HostClass::LoopbackApp(app));
            }
        }
        if let Some(app) = self.public.get(host) {
            return Ok(HostClass::Public(*app));
        }
        if self.exposure == Exposure::Lan && host.is_ip_literal() && self.is_bound_port(host.port) {
            return Ok(HostClass::LanAddress);
        }
        Err(Misdirected)
    }

    /// The names this install answers to, as text: what the startup audit event says. The loopback names on
    /// any port, the three app names on a loopback server, the configured public hosts, the extra ones, and
    /// on a LAN listener the IP literals of its port.
    pub fn describe(&self) -> Vec<String> {
        let mut out: Vec<String> = ["localhost", "127.0.0.1", "[::1]"]
            .iter()
            .map(|h| format!("{h} (any port)"))
            .collect();
        if self.loopback_apps {
            out.extend(
                [
                    "dash.oaiy.localhost",
                    "agent.oaiy.localhost",
                    "flows.oaiy.localhost",
                ]
                .iter()
                .map(|h| h.to_string()),
            );
        }
        out.extend(self.public.keys().map(HostName::display));
        out.extend(self.extra.iter().map(HostName::display));
        if self.exposure == Exposure::Lan {
            out.push(format!("any IP literal on port {}", self.port));
        }
        out
    }

    /// Whether a `Host` with this port (as [`HostName`] keeps it: `None` for no port and for `:80` and `:443`,
    /// which it drops as the schemes' defaults) is for the port the listener is bound to: the port itself, and
    /// no port at all when the listener is on 80 or 443, the ports a client uses when it names none.
    fn is_bound_port(&self, port: Option<u16>) -> bool {
        match port {
            Some(p) => p == self.port,
            None => self.port == 80 || self.port == 443,
        }
    }

    /// Whether `host` is one of the configured public hosts.
    pub fn is_public(&self, host: &HostName) -> bool {
        self.public.contains_key(host)
    }

    /// The exposure this policy was made for.
    pub fn exposure(&self) -> Exposure {
        self.exposure
    }
}

/// Whether the request's connection is one a cookie or a login may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    Secure,
    Insecure,
}

/// A trusted proxy said `http` for a public https host: its configuration is wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProxyMisconfigured;

/// The channel (design 4.5.4): secure when (a) the peer is a trusted proxy, `X-Forwarded-Proto` is `https`
/// and `Host` is a configured public host, or (b) `Host` is a loopback name, the peer is loopback and no
/// forwarded header is present. Anything else is insecure. A trusted proxy that says `http` for a public
/// host is a `400 proxy_misconfigured`.
pub fn channel(
    policy: &HostPolicy,
    host: &HostName,
    peer_is_trusted_proxy: bool,
    forwarded_proto: Option<&str>,
    peer_is_loopback: bool,
    forwarded_present: bool,
) -> Result<Channel, ProxyMisconfigured> {
    if peer_is_trusted_proxy && policy.is_public(host) {
        return match forwarded_proto
            .map(|p| p.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("https") => Ok(Channel::Secure),
            Some("http") => Err(ProxyMisconfigured),
            _ => Ok(Channel::Insecure),
        };
    }
    // The three app names of a loopback server are loopback names too: they resolve to this machine, and a
    // loopback server serves its login on them (the cookies of 4.7.5 are named for exactly these hosts).
    let is_loopback =
        host.is_loopback_name() || (policy.loopback_apps && loopback_app_of(&host.host).is_some());
    if is_loopback && peer_is_loopback && !forwarded_present {
        return Ok(Channel::Secure);
    }
    Ok(Channel::Insecure)
}

/// The origin a browser page served for this `Host` has, as the same-origin checks need it: `https://<host>`
/// on a secure channel behind a proxy, `http://<host>:<port>` for the loopback names.
pub fn expected_origin(host: &HostName, channel: Channel) -> String {
    let scheme = if channel == Channel::Secure
        && !host.is_loopback_name()
        && !host.host.ends_with(".oaiy.localhost")
    {
        "https"
    } else {
        "http"
    };
    format!("{scheme}://{}", host.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> HostName {
        HostName::parse(s).unwrap_or_else(|| panic!("{s} does not parse"))
    }

    fn policy(exposure: Exposure, port: u16, loopback_apps: bool) -> HostPolicy {
        let mut public = BTreeMap::new();
        if exposure == Exposure::Proxied {
            public.insert(h("dash.example.com"), App::Dash);
            public.insert(h("agent.example.com"), App::Agent);
            public.insert(h("flows.example.com:8443"), App::Flows);
        }
        HostPolicy::new(exposure, port, BTreeSet::new(), public, loopback_apps)
    }

    #[test]
    fn a_host_is_lowercase_with_its_default_port_removed() {
        let long = "a".repeat(300);
        assert_eq!(
            h("Dash.Example.COM:443"),
            HostName {
                host: "dash.example.com".into(),
                port: None
            }
        );
        assert_eq!(h("dash.example.com:80").port, None);
        assert_eq!(h("dash.example.com:8443").port, Some(8443));
        assert_eq!(h("localhost:17972").display(), "localhost:17972");
        assert_eq!(
            h("[::1]:17972"),
            HostName {
                host: "[::1]".into(),
                port: Some(17972)
            }
        );
        assert_eq!(h("[::1]").host, "[::1]");
        assert_eq!(h("127.0.0.1").display(), "127.0.0.1");
        for bad in [
            "",
            " ",
            "a b",
            "a/b",
            "user@host",
            "host:0",
            "host:99999",
            "host:abc",
            "[::1",
            "[]",
            "[::1]x",
            "::1",
            "host:8080:9090",
            ".hidden",
            "ho\nst",
            "h\u{e9}",
            "host?x",
            "host#x",
            "a\\b",
            long.as_str(),
        ] {
            assert_eq!(HostName::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_loopback_names_are_accepted_on_any_port_on_every_install() {
        // T5: `localhost:<any port>` is accepted: an SSH tunnel, a Docker port mapping, a moved port.
        for exposure in [Exposure::Local, Exposure::Lan] {
            let p = policy(exposure, 17972, false);
            for host in [
                "localhost",
                "localhost:17972",
                "localhost:9999",
                "127.0.0.1:2222",
                "127.0.0.1",
                "[::1]:17972",
                "LOCALHOST:8080",
            ] {
                assert_eq!(
                    p.classify(&h(host), true),
                    Ok(HostClass::Loopback),
                    "{host}"
                );
            }
        }
    }

    #[test]
    fn a_rebound_name_is_misdirected_on_every_install() {
        // DNS rebinding: the page carries its own name, never a loopback one.
        for exposure in [Exposure::Local, Exposure::Lan, Exposure::Proxied] {
            let p = policy(exposure, 17972, true);
            for host in [
                "evil.example:17972",
                "evil.example",
                "127.0.0.1.evil.example",
                "localhost.evil.example:17972",
                "oaiy.localhost",
                "attacker.oaiy.localhost",
                "x.dash.oaiy.localhost",
            ] {
                assert_eq!(
                    p.classify(&h(host), true),
                    Err(Misdirected),
                    "{exposure:?} {host}"
                );
            }
        }
    }

    #[test]
    fn the_loopback_server_also_answers_its_three_app_names_and_the_desktop_does_not() {
        let server = policy(Exposure::Local, 17972, true);
        assert_eq!(
            server.classify(&h("dash.oaiy.localhost:17972"), true),
            Ok(HostClass::LoopbackApp(App::Dash))
        );
        assert_eq!(
            server.classify(&h("agent.oaiy.localhost:41000"), true),
            Ok(HostClass::LoopbackApp(App::Agent))
        );
        assert_eq!(
            server.classify(&h("flows.oaiy.localhost"), true),
            Ok(HostClass::LoopbackApp(App::Flows))
        );
        // `oaiy.localhost` is the Agent's "OAIY's own window" name: none of the three is it.
        assert_eq!(
            server.classify(&h("oaiy.localhost"), true),
            Err(Misdirected)
        );
        let desktop = policy(Exposure::Local, 17972, false);
        assert_eq!(
            desktop.classify(&h("dash.oaiy.localhost:17972"), true),
            Err(Misdirected)
        );
        assert_eq!(
            desktop.classify(&h("localhost:17972"), true),
            Ok(HostClass::Loopback)
        );
    }

    #[test]
    fn a_proxied_install_answers_its_public_hosts_and_loopback_only_to_a_direct_loopback_peer() {
        let p = policy(Exposure::Proxied, 17972, false);
        assert_eq!(
            p.classify(&h("dash.example.com"), false),
            Ok(HostClass::Public(App::Dash))
        );
        assert_eq!(
            p.classify(&h("dash.example.com:443"), false),
            Ok(HostClass::Public(App::Dash))
        );
        assert_eq!(
            p.classify(&h("agent.example.com"), false),
            Ok(HostClass::Public(App::Agent))
        );
        assert_eq!(
            p.classify(&h("flows.example.com:8443"), false),
            Ok(HostClass::Public(App::Flows))
        );
        assert_eq!(
            p.classify(&h("flows.example.com"), false),
            Err(Misdirected),
            "the port is part of the name when it is not a default"
        );
        assert_eq!(p.classify(&h("other.example.com"), false), Err(Misdirected));
        // The default nginx block forwards `Host: 127.0.0.1:17972`: through the proxy that is refused (T45).
        assert_eq!(p.classify(&h("127.0.0.1:17972"), false), Err(Misdirected));
        assert_eq!(p.classify(&h("localhost:17972"), false), Err(Misdirected));
        // The CLI on the server, straight to the port, is answered.
        assert_eq!(
            p.classify(&h("127.0.0.1:17972"), true),
            Ok(HostClass::Loopback)
        );
    }

    #[test]
    fn f9_a_policy_says_the_names_it_answers_to_for_the_startup_event() {
        // A desktop: the loopback names only.
        let desktop = policy(Exposure::Local, 17972, false).describe();
        assert_eq!(desktop.len(), 3);
        assert!(desktop.iter().all(|h| h.ends_with("(any port)")));
        // A loopback server: and its three app names.
        let server = policy(Exposure::Local, 17972, true).describe();
        for name in [
            "dash.oaiy.localhost",
            "agent.oaiy.localhost",
            "flows.oaiy.localhost",
        ] {
            assert!(server.contains(&name.to_string()), "{name}");
        }
        // A proxied install: the public hosts, with their ports when they have one.
        let proxied = policy(Exposure::Proxied, 17972, false).describe();
        for name in [
            "dash.example.com",
            "agent.example.com",
            "flows.example.com:8443",
        ] {
            assert!(proxied.contains(&name.to_string()), "{name} in {proxied:?}");
        }
        // A LAN listener: the IP literals of its port.
        let lan = policy(Exposure::Lan, 8080, false).describe();
        assert!(
            lan.contains(&"any IP literal on port 8080".to_string()),
            "{lan:?}"
        );
        // The extra hosts.
        let mut extra = BTreeSet::new();
        extra.insert(h("nas.example:9000"));
        let with_extra =
            HostPolicy::new(Exposure::Local, 17972, extra, BTreeMap::new(), false).describe();
        assert!(
            with_extra.contains(&"nas.example:9000".to_string()),
            "{with_extra:?}"
        );
    }

    #[test]
    fn f8_a_lan_listener_on_port_80_or_443_answers_an_ip_literal_with_no_port_and_with_its_own() {
        // A client that names no port uses the scheme's: 80 for http, 443 for https. `HostName` drops both
        // from a `Host`, so an address with no port is what `:80` and `:443` come to; a listener on either
        // answers it (the address is one of the machine's own, so nothing is opened to a rebinding name).
        for bound in [80u16, 443] {
            let p = policy(Exposure::Lan, bound, false);
            for host in [
                "192.168.1.5",
                "192.168.1.5:80",
                "192.168.1.5:443",
                "10.0.0.7",
                "[fd12::5]",
                "[fd12::5]:80",
                "[fd12::5]:443",
            ] {
                assert_eq!(
                    p.classify(&h(host), false),
                    Ok(HostClass::LanAddress),
                    "bound {bound}, Host {host}"
                );
            }
            // Another port, a name, and a public name are not this server's.
            for host in [
                "192.168.1.5:17972",
                "192.168.1.5:8080",
                "[fd12::5]:17972",
                "nas.local",
                "nas.local:80",
                "evil.example",
            ] {
                assert_eq!(
                    p.classify(&h(host), false),
                    Err(Misdirected),
                    "bound {bound}, Host {host}"
                );
            }
        }
    }

    #[test]
    fn f8_a_lan_listener_on_another_port_answers_that_port_only() {
        for bound in [17972u16, 8080, 8443, 1] {
            let p = policy(Exposure::Lan, bound, false);
            assert_eq!(
                p.classify(&h(&format!("192.168.1.5:{bound}")), false),
                Ok(HostClass::LanAddress),
                "bound {bound}"
            );
            for host in [
                "192.168.1.5",
                "192.168.1.5:80",
                "192.168.1.5:443",
                "192.168.1.5:9",
                "[fd12::5]",
            ] {
                assert_eq!(
                    p.classify(&h(host), false),
                    Err(Misdirected),
                    "bound {bound}, Host {host}"
                );
            }
        }
    }

    #[test]
    fn a_lan_listener_answers_an_ip_literal_with_its_bound_port_and_no_name() {
        let p = policy(Exposure::Lan, 17972, false);
        assert_eq!(
            p.classify(&h("192.168.1.5:17972"), false),
            Ok(HostClass::LanAddress)
        );
        assert_eq!(
            p.classify(&h("10.0.0.7:17972"), false),
            Ok(HostClass::LanAddress)
        );
        assert_eq!(
            p.classify(&h("[fd12::5]:17972"), false),
            Ok(HostClass::LanAddress)
        );
        assert_eq!(
            p.classify(&h("192.168.1.5:9999"), false),
            Err(Misdirected),
            "the wrong port"
        );
        assert_eq!(
            p.classify(&h("192.168.1.5"), false),
            Err(Misdirected),
            "no port is port 80"
        );
        assert_eq!(
            p.classify(&h("nas.local:17972"), false),
            Err(Misdirected),
            "a name is not an address"
        );
        // A local install has no such rule.
        assert_eq!(
            policy(Exposure::Local, 17972, false).classify(&h("192.168.1.5:17972"), false),
            Err(Misdirected)
        );
    }

    #[test]
    fn extra_hosts_are_answered_on_every_install() {
        let mut extra = BTreeSet::new();
        extra.insert(h("oaiy.internal:17972"));
        extra.insert(h("proxy.internal"));
        for exposure in [Exposure::Local, Exposure::Lan, Exposure::Proxied] {
            let p = HostPolicy::new(exposure, 17972, extra.clone(), BTreeMap::new(), false);
            assert_eq!(
                p.classify(&h("oaiy.internal:17972"), false),
                Ok(HostClass::Extra)
            );
            assert_eq!(
                p.classify(&h("proxy.internal"), false),
                Ok(HostClass::Extra)
            );
            assert_eq!(
                p.classify(&h("oaiy.internal"), false),
                Err(Misdirected),
                "host:port is exact"
            );
        }
    }

    #[test]
    fn the_channel_is_secure_behind_a_trusted_proxy_on_https_or_on_loopback_direct() {
        let p = policy(Exposure::Proxied, 17972, false);
        let dash = h("dash.example.com");
        assert_eq!(
            channel(&p, &dash, true, Some("https"), false, true),
            Ok(Channel::Secure)
        );
        assert_eq!(
            channel(&p, &dash, true, Some("HTTPS"), false, true),
            Ok(Channel::Secure)
        );
        // A trusted proxy that says http for a public https host has a broken configuration (T30).
        assert_eq!(
            channel(&p, &dash, true, Some("http"), false, true),
            Err(ProxyMisconfigured)
        );
        // No X-Forwarded-Proto at all: not a channel that can carry a cookie.
        assert_eq!(
            channel(&p, &dash, true, None, false, true),
            Ok(Channel::Insecure)
        );
        assert_eq!(
            channel(&p, &dash, true, Some("gopher"), false, true),
            Ok(Channel::Insecure)
        );
        // An untrusted peer's word is worth nothing.
        assert_eq!(
            channel(&p, &dash, false, Some("https"), false, true),
            Ok(Channel::Insecure)
        );
        // Loopback direct: secure.
        let local = h("localhost:17972");
        assert_eq!(
            channel(&p, &local, false, None, true, false),
            Ok(Channel::Secure)
        );
        // With a forwarded header a loopback name is not a direct client.
        assert_eq!(
            channel(&p, &local, false, None, true, true),
            Ok(Channel::Insecure)
        );
        // A non-loopback peer using a loopback name is not secure either.
        assert_eq!(
            channel(&p, &local, false, None, false, false),
            Ok(Channel::Insecure)
        );
        // A public host from a trusted proxy is judged by its proto, not by being loopback.
        assert_eq!(
            channel(
                &p,
                &h("other.example.com"),
                true,
                Some("https"),
                false,
                true
            ),
            Ok(Channel::Insecure)
        );
    }

    /// Channel classification (design 4.5.4), over every combination of what it reads, against the two rules of the
    /// design written out on their own: (a) the peer is a trusted proxy, `X-Forwarded-Proto` is `https` and `Host`
    /// is a configured public host; (b) `Host` is a loopback name, the peer is loopback and no forwarded header is
    /// present. Anything else is insecure; a trusted proxy that says `http` for a public host is misconfigured.
    #[test]
    fn t30_the_channel_is_the_two_rules_of_the_design_over_every_combination_of_its_inputs() {
        let hosts = [
            "dash.example.com",
            "agent.example.com",
            "other.example.com",
            "localhost:17972",
            "127.0.0.1:17972",
            "[::1]:17972",
            "dash.oaiy.localhost:17972",
            "192.168.1.5:17972",
            "oaiy.localhost",
        ];
        let protos = [
            None,
            Some("https"),
            Some("HTTPS"),
            Some(" https "),
            Some("http"),
            Some("HTTP"),
            Some("gopher"),
            Some(""),
            Some("https, http"),
        ];
        let mut secure = 0;
        for loopback_apps in [false, true] {
            let p = policy(
                if loopback_apps {
                    Exposure::Local
                } else {
                    Exposure::Proxied
                },
                17972,
                loopback_apps,
            );
            // Which of those are configured public hosts on this install: none on a loopback server.
            let public = |host: &str| p.is_public(&h(host));
            for host in hosts {
                for proto in protos {
                    for trusted in [false, true] {
                        for peer_loopback in [false, true] {
                            for forwarded in [false, true] {
                                let name = h(host);
                                let want: Result<Channel, ProxyMisconfigured> = if trusted
                                    && public(host)
                                {
                                    match proto.map(|x| x.trim().to_ascii_lowercase()).as_deref() {
                                        Some("https") => Ok(Channel::Secure),
                                        Some("http") => Err(ProxyMisconfigured),
                                        _ => Ok(Channel::Insecure),
                                    }
                                } else {
                                    let loopbackish = matches!(
                                        name.host.as_str(),
                                        "localhost" | "127.0.0.1" | "[::1]"
                                    ) || (loopback_apps
                                        && name.host == "dash.oaiy.localhost");
                                    if loopbackish && peer_loopback && !forwarded {
                                        Ok(Channel::Secure)
                                    } else {
                                        Ok(Channel::Insecure)
                                    }
                                };
                                let got =
                                    channel(&p, &name, trusted, proto, peer_loopback, forwarded);
                                assert_eq!(
                                    got, want,
                                    "apps {loopback_apps} host {host} proto {proto:?} trusted {trusted} loopback {peer_loopback} forwarded {forwarded}"
                                );
                                if got == Ok(Channel::Secure) {
                                    secure += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(secure > 20, "the table has {secure} secure rows");
    }

    #[test]
    fn t30_a_secure_channel_needs_the_trusted_proxy_to_say_https_and_nothing_a_client_says_counts()
    {
        let p = policy(Exposure::Proxied, 17972, false);
        let dash = h("dash.example.com");
        // The Caddy double: a trusted peer, `X-Forwarded-Proto: https`, the public Host.
        assert_eq!(
            channel(&p, &dash, true, Some("https"), false, true),
            Ok(Channel::Secure)
        );
        // A client that sends the same headers itself is not a trusted proxy.
        assert_eq!(
            channel(&p, &dash, false, Some("https"), false, true),
            Ok(Channel::Insecure)
        );
        // Even from this machine, a public host is not a loopback name.
        assert_eq!(
            channel(&p, &dash, false, Some("https"), true, true),
            Ok(Channel::Insecure)
        );
        assert_eq!(
            channel(&p, &dash, false, None, true, false),
            Ok(Channel::Insecure)
        );
        // A wrong `X-Forwarded-Proto` from the proxy, and a missing one.
        assert_eq!(
            channel(&p, &dash, true, Some("http"), false, true),
            Err(ProxyMisconfigured)
        );
        assert_eq!(
            channel(&p, &dash, true, None, false, true),
            Ok(Channel::Insecure)
        );
    }

    #[test]
    fn the_app_names_of_a_loopback_server_are_a_secure_channel_from_the_machine_itself_and_only_there(
    ) {
        // Without this the login of a loopback server (`http://dash.oaiy.localhost:<port>`) is refused as
        // `secure_channel_required` before it is read.
        let p = policy(Exposure::Local, 41000, true);
        for name in [
            "dash.oaiy.localhost:41000",
            "agent.oaiy.localhost:41000",
            "flows.oaiy.localhost",
        ] {
            let host = h(name);
            assert_eq!(
                channel(&p, &host, false, None, true, false),
                Ok(Channel::Secure),
                "{name} from a loopback peer"
            );
            // The same rules as `localhost`: a forwarded header, or a peer that is not this machine.
            assert_eq!(
                channel(&p, &host, false, None, true, true),
                Ok(Channel::Insecure),
                "{name} with a forwarded header"
            );
            assert_eq!(
                channel(&p, &host, false, None, false, false),
                Ok(Channel::Insecure),
                "{name} from another machine"
            );
        }
        // A install that does not serve the app names (the desktop) has no such names.
        let desktop = policy(Exposure::Local, 41000, false);
        assert_eq!(
            channel(
                &desktop,
                &h("dash.oaiy.localhost:41000"),
                false,
                None,
                true,
                false
            ),
            Ok(Channel::Insecure)
        );
        // `oaiy.localhost` and look-alikes are not app names.
        for name in [
            "oaiy.localhost",
            "x.dash.oaiy.localhost",
            "dash.oaiy.localhost.evil.example",
        ] {
            assert_eq!(
                channel(&p, &h(name), false, None, true, false),
                Ok(Channel::Insecure),
                "{name}"
            );
        }
    }

    #[test]
    fn the_expected_origin_is_what_a_page_at_that_host_sends() {
        assert_eq!(
            expected_origin(&h("dash.example.com"), Channel::Secure),
            "https://dash.example.com"
        );
        assert_eq!(
            expected_origin(&h("flows.example.com:8443"), Channel::Secure),
            "https://flows.example.com:8443"
        );
        assert_eq!(
            expected_origin(&h("localhost:17972"), Channel::Secure),
            "http://localhost:17972"
        );
        assert_eq!(
            expected_origin(&h("127.0.0.1:41000"), Channel::Secure),
            "http://127.0.0.1:41000"
        );
        assert_eq!(
            expected_origin(&h("dash.oaiy.localhost:41000"), Channel::Secure),
            "http://dash.oaiy.localhost:41000"
        );
        assert_eq!(
            expected_origin(&h("192.168.1.5:17972"), Channel::Insecure),
            "http://192.168.1.5:17972"
        );
    }
}
