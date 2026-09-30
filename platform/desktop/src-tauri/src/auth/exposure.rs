//! Exposure and the startup rules (design 4.5.4, 4.5.5 and the variables of 4.13).
//!
//! [`evaluate`] reads the environment of `oaiy-server` and says, all at once, what is wrong with it; it reads
//! nothing else (whether `<data>/auth/owner.json` exists is a [`Facts`] the caller looks up, so that every rule
//! is a pure function of two values and can be tested without a disk). [`validate_config`] is the same with the
//! first violation as its error: the one line `oaiy-server` prints before it exits 78 (`EX_CONFIG`, which the
//! shipped unit does not restart). `oaiy-server check` prints every violation.
//!
//! The rules of 4.5.5, by number:
//!
//! 1. `OAIY_SERVER_BIND` is `loopback`, `lan` (an alias of `0.0.0.0`) or an IP literal.
//! 2. A **lan** install (a bind beyond loopback and no `OAIY_PUBLIC_URL`) needs `<data>/auth/owner.json`. A
//!    **proxied** install may start without one, in setup-only mode.
//! 3. `OAIY_PUBLIC_URL`, `OAIY_AGENT_URL` and `OAIY_FLOWS_URL` are each `https://<host>[:<port>]` with no path,
//!    query or fragment, none has a loopback host, and no two share a host. An app whose URL is unset is not
//!    served; the dashboard's URL is required by the other two.
//! 4. A bind beyond loopback with `OAIY_PUBLIC_URL` needs `OAIY_TRUSTED_PROXIES` ("name your proxy"): that is the
//!    proxy-only shape, in which a direct peer that is not a trusted proxy is `403 direct_access_refused`.
//! 5. `OAIY_SERVER_TOKEN` has the shape of design 4.1 (32 to 256 printable ASCII characters, 16 different).
//! 6. The access mode (`mode::validate_mode`): `shadow` on a proxied or lan install, `legacy` where the web login
//!    is built in, where an owner exists or where the exposure is not local.
//!
//! Rule 7 (the data folder) is the credential store's, at open, and `check` reads the files the same way. Rule 8
//! (the desktop's `lanAccess`) is not a refusal: [`desktop_bind`] ignores it.
//!
//! Beyond the design, each of these is a violation too, because ignoring it would be the unsafe direction or a
//! surprise the operator would find out about on the internet: a port that is not a port, an entry of
//! `OAIY_TRUSTED_PROXIES`, `OAIY_ALLOWED_HOSTS` or `OAIY_LOGIN_ALLOW` that is not what it should be (an
//! allow-list with nothing usable in it would be no list at all).

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, Ipv4Addr};

use super::clientip::{unmap, Cidr, TrustedProxies};
use super::host::HostName;
use super::mode::{validate_mode, AccessMode, Exposure};
use super::presets::App;
use super::token::check_static_token_shape;

/// What the environment cannot say: it is looked up by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// `<data>/auth/owner.json` exists (or could not be read: the store says why when it opens).
    pub owner_exists: bool,
    /// This build has the web login (the `web` feature): its default mode is `scoped` and it refuses `legacy`.
    pub web_login: bool,
}

/// Which rule a violation is of: the numbers are those of design 4.5.5; the rest are this step's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// 1
    Bind,
    /// 2
    LanNeedsOwner,
    /// 3
    PublicUrl,
    /// 4
    ProxyOnlyNeedsProxies,
    /// 5
    StaticToken,
    /// 6
    Mode,
    Port,
    TrustedProxies,
    AllowedHosts,
    LoginAllow,
}

impl Rule {
    /// The number of design 4.5.5 this is, if it is one of its rules.
    pub fn number(self) -> Option<u8> {
        match self {
            Rule::Bind => Some(1),
            Rule::LanNeedsOwner => Some(2),
            Rule::PublicUrl => Some(3),
            Rule::ProxyOnlyNeedsProxies => Some(4),
            Rule::StaticToken => Some(5),
            Rule::Mode => Some(6),
            _ => None,
        }
    }
}

/// One thing that stops the server starting: one line, saying what to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub rule: Rule,
    pub message: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Where `oaiy-server` listens (`OAIY_SERVER_BIND`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    /// `127.0.0.1`: the default.
    Loopback,
    /// `0.0.0.0`.
    Lan,
    /// An address, named.
    Addr(IpAddr),
}

impl Bind {
    /// Unset and blank are the default; `loopback` and `lan` are matched without regard to case; anything else
    /// must be an IP literal (an IPv6 one may be in brackets). A typo is an error: it used to keep loopback
    /// quietly, and now says so.
    pub fn parse(value: Option<&str>) -> Result<Bind, String> {
        let Some(v) = value.map(str::trim).filter(|v| !v.is_empty()) else {
            return Ok(Bind::Loopback);
        };
        if v.eq_ignore_ascii_case("loopback") {
            return Ok(Bind::Loopback);
        }
        if v.eq_ignore_ascii_case("lan") {
            return Ok(Bind::Lan);
        }
        let bare = v
            .strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .unwrap_or(v);
        bare.parse::<IpAddr>().map(Bind::Addr).map_err(|_| {
            format!(
                "OAIY_SERVER_BIND={v:?} is not loopback, lan or an IP address: use loopback (the default), lan (every interface) or an address such as 192.168.1.5"
            )
        })
    }

    /// The address to bind.
    pub fn addr(self) -> IpAddr {
        match self {
            Bind::Loopback => IpAddr::V4(Ipv4Addr::LOCALHOST),
            Bind::Lan => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            Bind::Addr(a) => a,
        }
    }

    /// Whether only this machine can reach the listener.
    pub fn is_loopback(self) -> bool {
        unmap(self.addr()).is_loopback()
    }
}

/// A configured public host: `https://<host>[:<port>]`. `Err` says what to change.
pub fn parse_https_origin(var: &str, text: &str) -> Result<(HostName, String), String> {
    let bad = |why: &str| {
        Err(format!(
            "{var}={text:?} {why}: write it as https://<host>[:<port>] with no path, query or fragment"
        ))
    };
    let Some(rest) = text.strip_prefix("https://") else {
        return bad("is not an https URL");
    };
    if rest.is_empty() {
        return bad("has no host");
    }
    if rest.contains(['/', '?', '#']) {
        return bad("has a path, a query or a fragment");
    }
    if rest.contains('@') {
        return bad("has a user name");
    }
    let Some(host) = HostName::parse(rest) else {
        return bad("is not a host name and an optional port");
    };
    // `HostName` drops `:80` and `:443` as the schemes' defaults; `:80` on an https URL is not a default of
    // https, and the Origin a browser sends would carry it.
    if rest
        .rsplit_once(':')
        .is_some_and(|(h, p)| !h.ends_with(':') && p == "80")
    {
        return bad("names port 80, which is not an https port");
    }
    if is_loopback_host(&host) {
        return bad("is a loopback host: a public URL names the host the proxy answers to");
    }
    let origin = match host.port {
        Some(p) => format!("https://{}:{p}", host.host),
        None => format!("https://{}", host.host),
    };
    Ok((host, origin))
}

/// `localhost` and `*.localhost` (which resolve to this machine, RFC 6761), and an address that is loopback or
/// unspecified.
fn is_loopback_host(host: &HostName) -> bool {
    let name = host.host.as_str();
    if name == "localhost" || name.ends_with(".localhost") {
        return true;
    }
    let bare = name
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .unwrap_or(name);
    bare.parse::<IpAddr>()
        .is_ok_and(|ip| unmap(ip).is_loopback() || ip.is_unspecified())
}

/// The three URL variables, in the order their apps are named.
const URL_VARS: [(&str, App); 3] = [
    ("OAIY_PUBLIC_URL", App::Dash),
    ("OAIY_AGENT_URL", App::Agent),
    ("OAIY_FLOWS_URL", App::Flows),
];

/// A configuration that passed every rule.
#[derive(Clone, Debug)]
pub struct Config {
    pub bind: Bind,
    pub port: u16,
    pub exposure: Exposure,
    pub mode: AccessMode,
    /// The hosts of the configured URLs and the app each serves.
    pub public: BTreeMap<HostName, App>,
    /// The URLs, as origins (`https://dash.example.com`).
    pub public_origins: BTreeMap<App, String>,
    /// `OAIY_TRUSTED_PROXIES`, or its default: loopback when `OAIY_PUBLIC_URL` is set, none otherwise.
    pub trusted: TrustedProxies,
    /// `OAIY_TRUSTED_PROXIES` was set (it replaces the default, it does not add to it).
    pub trusted_explicit: bool,
    pub allowed_hosts: BTreeSet<HostName>,
    /// `OAIY_LOGIN_ALLOW`: empty means anywhere.
    pub login_allow: Vec<Cidr>,
    /// `OAIY_ALLOW_PUBLIC_PLAINTEXT=1`.
    pub allow_public_plaintext: bool,
    /// `OAIY_SERVER_TOKEN`, trimmed, of the shape of design 4.1.
    pub static_token: Option<String>,
    /// `<data>/auth/owner.json` exists: with none the browser side is in setup-only mode.
    pub owner_exists: bool,
    /// Things that are ignored or have no effect here: worth a line, not a refusal.
    pub warnings: Vec<String>,
}

impl Config {
    /// The **proxy-only** shape (rule 4): a bind beyond loopback behind a named proxy. A direct peer that is
    /// neither one of those proxies nor this machine is `403 direct_access_refused`.
    pub fn proxy_only(&self) -> bool {
        self.exposure == Exposure::Proxied && !self.bind.is_loopback()
    }

    /// No owner login yet: the browser side is setup-only until the console makes one (design 4.7.1).
    pub fn setup_only_at_start(&self) -> bool {
        !self.owner_exists
    }

    /// The lines printed at start: what this install is and what it will and will not do. They hold no secret.
    pub fn banner_lines(&self) -> Vec<String> {
        let listen = format!("{}:{}", fmt_ip(self.bind.addr()), self.port);
        let mut out = Vec::new();
        match self.exposure {
            Exposure::Local => out.push(format!(
                "oaiy-server: exposure local: listening on {listen}, this machine only; no proxy is expected (a forwarded header is refused)"
            )),
            Exposure::Lan => out.push(format!(
                "oaiy-server: exposure lan: listening on {listen}, reachable from the network over plain HTTP: bearer tokens only, no cookies, no sign-in page and no UI here (they need https: put a reverse proxy in front and set OAIY_PUBLIC_URL)"
            )),
            Exposure::Proxied => {
                let urls: Vec<String> = URL_VARS
                    .iter()
                    .filter_map(|(_, app)| {
                        self.public_origins
                            .get(app)
                            .map(|o| format!("{} {o}", app.name()))
                    })
                    .collect();
                out.push(format!(
                    "oaiy-server: exposure proxied: listening on {listen} behind a reverse proxy; serving {}",
                    urls.join(", ")
                ));
                out.push(format!(
                    "oaiy-server: trusted proxies {}{}",
                    if self.trusted.is_empty() {
                        "none".to_string()
                    } else {
                        self.trusted.describe().join(", ")
                    },
                    if self.trusted_explicit {
                        ""
                    } else {
                        " (the default: a proxy on this machine)"
                    }
                ));
                if self.proxy_only() {
                    out.push(
                        "oaiy-server: proxy-only: a connection that is neither from a trusted proxy nor from this machine is refused (403 direct_access_refused)"
                            .to_string(),
                    );
                }
            }
        }
        out
    }
}

fn fmt_ip(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => format!("[{v6}]"),
    }
}

/// What [`evaluate`] found.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Present when there is no violation.
    pub config: Option<Config>,
    /// Every violation, in the order of the rules.
    pub violations: Vec<Violation>,
    pub warnings: Vec<String>,
}

fn nonblank(env: &dyn Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    env(name)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// The port, `OAIY_SERVER_PORT`: unset or blank is the desktop's; anything else must be 1 to 65535.
pub fn parse_port(value: Option<&str>) -> Result<u16, String> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(crate::DESKTOP_PORT),
        Some(v) => v.parse::<u16>().ok().filter(|p| *p != 0).ok_or_else(|| {
            format!("OAIY_SERVER_PORT={v:?} is not a port: use a number from 1 to 65535")
        }),
    }
}

/// The access mode of a server: `scoped` by default where the web login is built in (it needs the store on disk,
/// which `legacy` never opens) and `legacy` where it is not, exactly as `OAIY_ACCESS_MODE` says otherwise;
/// `legacy` is refused where the web login is built in.
pub fn mode_from_env(value: Option<&str>, web_login: bool) -> Result<AccessMode, String> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(if web_login {
            AccessMode::Scoped
        } else {
            AccessMode::Legacy
        }),
        Some(_) => match AccessMode::from_env(value).map_err(|r| r.to_string())? {
            AccessMode::Legacy if web_login => Err(
                "OAIY_ACCESS_MODE=legacy is refused by a server with the web login: use scoped (the default)"
                    .to_string(),
            ),
            mode => Ok(mode),
        },
    }
}

/// Read the settings of 4.13 from `env`, apply every rule of 4.5.5 and say everything that is wrong.
pub fn evaluate(env: &dyn Fn(&str) -> Option<String>, facts: &Facts) -> Evaluation {
    let mut violations: Vec<Violation> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    macro_rules! violate {
        ($rule:expr, $message:expr $(,)?) => {
            violations.push(Violation {
                rule: $rule,
                message: $message,
            })
        };
    }

    // The port.
    let port = match parse_port(env("OAIY_SERVER_PORT").as_deref()) {
        Ok(p) => p,
        Err(m) => {
            violate!(Rule::Port, m);
            crate::DESKTOP_PORT
        }
    };

    // 1. the bind.
    let bind = match Bind::parse(env("OAIY_SERVER_BIND").as_deref()) {
        Ok(b) => b,
        Err(m) => {
            violate!(Rule::Bind, m);
            // Nothing after this can be judged by the bind: judge the rest as if it were the default.
            Bind::Loopback
        }
    };

    // 3. the public URLs.
    let mut public: BTreeMap<HostName, App> = BTreeMap::new();
    let mut origins: BTreeMap<App, String> = BTreeMap::new();
    let mut by_hostname: BTreeMap<String, &str> = BTreeMap::new();
    let public_url_set = nonblank(env, "OAIY_PUBLIC_URL").is_some();
    for (var, app) in URL_VARS {
        let Some(text) = nonblank(env, var) else {
            continue;
        };
        match parse_https_origin(var, &text) {
            Err(m) => violate!(Rule::PublicUrl, m),
            Ok((host, origin)) => {
                // Cookies do not separate ports: two apps on one host name could overwrite each other's.
                if let Some(other) = by_hostname.insert(host.host.clone(), var) {
                    violate!(
                        Rule::PublicUrl,
                        format!(
                            "{var} and {other} name the same host ({}): each app needs a host of its own",
                            host.host
                        ),
                    );
                } else {
                    public.insert(host, app);
                    origins.insert(app, origin);
                }
            }
        }
    }
    if !public_url_set
        && (nonblank(env, "OAIY_AGENT_URL").is_some() || nonblank(env, "OAIY_FLOWS_URL").is_some())
    {
        violate!(
            Rule::PublicUrl,
            "OAIY_AGENT_URL and OAIY_FLOWS_URL need OAIY_PUBLIC_URL, the dashboard's: set it too"
                .to_string(),
        );
    }

    let exposure = Exposure::compute(!bind.is_loopback(), public_url_set);

    // The trusted proxies.
    let trusted_text = nonblank(env, "OAIY_TRUSTED_PROXIES");
    let trusted_explicit = trusted_text.is_some();
    let trusted = match &trusted_text {
        Some(list) => {
            let (t, rejected) = TrustedProxies::parse_list(list);
            if !rejected.is_empty() {
                violate!(
                    Rule::TrustedProxies,
                    format!(
                        "OAIY_TRUSTED_PROXIES: {} is not an address or a network (write 10.0.0.5 or 172.30.0.0/24, separated by commas)",
                        rejected
                            .iter()
                            .map(|r| format!("{r:?}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            }
            t
        }
        None if public_url_set => TrustedProxies::loopback(),
        None => TrustedProxies::none(),
    };
    if trusted_explicit && exposure == Exposure::Local {
        warnings.push(
            "OAIY_TRUSTED_PROXIES has no effect on a local install (a forwarded header is refused); set OAIY_PUBLIC_URL if a proxy is in front".to_string(),
        );
    }
    for net in trusted.describe() {
        if net.ends_with("/0") {
            warnings.push(format!(
                "OAIY_TRUSTED_PROXIES trusts {net}, which is every address: any client could forge X-Forwarded-For"
            ));
        }
    }

    // 4. behind a proxy, off this machine: the proxy must be named (the default, a proxy on this machine, is no
    // answer for a listener other machines can reach).
    if public_url_set && !bind.is_loopback() && (!trusted_explicit || trusted.is_empty()) {
        violate!(
            Rule::ProxyOnlyNeedsProxies,
            "OAIY_SERVER_BIND is not loopback and OAIY_PUBLIC_URL is set, so OAIY_TRUSTED_PROXIES must name the proxy (for example 172.30.0.0/24 for a compose network): direct connections are refused, and with no proxy named nothing could reach this server".to_string(),
        );
    }

    // The extra hosts.
    let mut allowed_hosts = BTreeSet::new();
    if let Some(list) = nonblank(env, "OAIY_ALLOWED_HOSTS") {
        for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match HostName::parse(entry) {
                Some(h) => {
                    allowed_hosts.insert(h);
                }
                None => violate!(
                    Rule::AllowedHosts,
                    format!("OAIY_ALLOWED_HOSTS: {entry:?} is not a host name with an optional port (write nas.example or nas.example:9000)"),
                ),
            }
        }
    }

    // The login's address allow-list: an entry that is not read would leave a list with nothing in it, and an
    // empty list lets everyone in.
    let mut login_allow: Vec<Cidr> = Vec::new();
    if let Some(list) = nonblank(env, "OAIY_LOGIN_ALLOW") {
        for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            match Cidr::parse(entry) {
                Some(c) => login_allow.push(c),
                None => violate!(
                    Rule::LoginAllow,
                    format!("OAIY_LOGIN_ALLOW: {entry:?} is not an address or a network (write 203.0.113.7 or 203.0.113.0/24, separated by commas): the list would let everyone in"),
                ),
            }
        }
        if login_allow.is_empty() && !violations.iter().any(|v| v.rule == Rule::LoginAllow) {
            violate!(
                Rule::LoginAllow,
                "OAIY_LOGIN_ALLOW names no address or network: unset it, or list where the owner signs in from".to_string(),
            );
        }
        if exposure == Exposure::Lan {
            warnings.push(
                "OAIY_LOGIN_ALLOW has no effect on a lan install (it has no sign-in: that needs https)"
                    .to_string(),
            );
        }
    }

    // The plaintext override: exactly `1`.
    let allow_public_plaintext = match nonblank(env, "OAIY_ALLOW_PUBLIC_PLAINTEXT").as_deref() {
        None => false,
        Some("1") => {
            if exposure != Exposure::Lan {
                warnings.push(
                    "OAIY_ALLOW_PUBLIC_PLAINTEXT=1 has no effect: it is for a lan install"
                        .to_string(),
                );
            }
            true
        }
        Some(other) => {
            warnings.push(format!(
                "OAIY_ALLOW_PUBLIC_PLAINTEXT={other:?} is not 1: bearers from public addresses stay refused"
            ));
            false
        }
    };

    // 5. the static token.
    let static_token = match nonblank(env, "OAIY_SERVER_TOKEN") {
        None => None,
        Some(t) => match check_static_token_shape(&t) {
            Ok(()) => Some(t),
            Err(shape) => {
                violate!(
                    Rule::StaticToken,
                    format!(
                        "{}: use a random value such as `openssl rand -base64 32` (random hex has too few different characters), or a token made on the console (`oaiy-server auth token create`)",
                        shape.message()
                    ),
                );
                None
            }
        },
    };

    // 2. a lan install needs an owner.
    if exposure == Exposure::Lan && !facts.owner_exists {
        violate!(
            Rule::LanNeedsOwner,
            format!(
                "OAIY_SERVER_BIND={} reaches the network, which needs an owner login first: run `oaiy-server auth init`, then start the server again",
                env("OAIY_SERVER_BIND")
                    .map(|b| b.trim().to_string())
                    .filter(|b| !b.is_empty())
                    .unwrap_or_else(|| "lan".into())
            ),
        );
    }

    // 6. the mode.
    let mode = match mode_from_env(env("OAIY_ACCESS_MODE").as_deref(), facts.web_login) {
        Ok(mode) => {
            if let Err(refusal) = validate_mode(mode, exposure, facts.owner_exists) {
                violate!(Rule::Mode, refusal.to_string());
            }
            mode
        }
        Err(m) => {
            violate!(Rule::Mode, m);
            AccessMode::Scoped
        }
    };

    violations.sort_by_key(|v| v.rule);
    let config = violations.is_empty().then(|| Config {
        bind,
        port,
        exposure,
        mode,
        public,
        public_origins: origins,
        trusted,
        trusted_explicit,
        allowed_hosts,
        login_allow,
        allow_public_plaintext,
        static_token,
        owner_exists: facts.owner_exists,
        warnings: warnings.clone(),
    });
    Evaluation {
        config,
        violations,
        warnings,
    }
}

/// [`evaluate`], with the first violation as the error: the one line the server prints before it exits 78.
pub fn validate_config(
    env: &dyn Fn(&str) -> Option<String>,
    facts: &Facts,
) -> Result<Config, String> {
    let evaluation = evaluate(env, facts);
    match evaluation.config {
        Some(config) => Ok(config),
        None => Err(evaluation
            .violations
            .first()
            .map(|v| v.message.clone())
            .unwrap_or_else(|| "the configuration is refused".to_string())),
    }
}

// ---- the data folder and `check` -----------------------------------------------------------------

/// Where the server keeps its data: `OAIY_DATA_DIR`, else `.oaiy-server` in the home folder.
pub fn data_dir_from_env(env: &dyn Fn(&str) -> Option<String>) -> Option<std::path::PathBuf> {
    if let Some(d) = env("OAIY_DATA_DIR").filter(|d| !d.trim().is_empty()) {
        return Some(std::path::PathBuf::from(d));
    }
    let home = env("HOME")
        .filter(|h| !h.is_empty())
        .or_else(|| env("USERPROFILE").filter(|h| !h.is_empty()))?;
    Some(std::path::PathBuf::from(home).join(".oaiy-server"))
}

/// Whether `<data>/auth/owner.json` is there. A file that cannot be looked at counts as there: the store says
/// why when it opens, and rule 2 must not send the operator to `auth init` over a file that exists.
pub fn owner_file_present(auth_dir: &std::path::Path) -> bool {
    match std::fs::metadata(auth_dir.join("owner.json")) {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// What `oaiy-server check` prints, and the exit code: 0 with the exposure and `ok`, 78 with every violation.
pub fn print_check(
    violations: &[String],
    warnings: &[String],
    config: Option<&Config>,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
) -> i32 {
    for w in warnings {
        let _ = writeln!(err, "oaiy-server check: warning: {w}");
    }
    if violations.is_empty() {
        if let Some(c) = config {
            for line in c.banner_lines() {
                let _ = writeln!(out, "{line}");
            }
        }
        let _ = writeln!(out, "oaiy-server check: ok");
        return 0;
    }
    for v in violations {
        let _ = writeln!(err, "oaiy-server check: {v}");
    }
    super::mode::EX_CONFIG
}

/// `oaiy-server check` for a build without the web login (no console, so no files to read but the owner's): the
/// rules of 4.5.5 over the environment.
pub fn check_headless(
    env: &dyn Fn(&str) -> Option<String>,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
) -> i32 {
    let owner_exists = data_dir_from_env(env).is_some_and(|d| owner_file_present(&d.join("auth")));
    let evaluation = evaluate(
        env,
        &Facts {
            owner_exists,
            web_login: false,
        },
    );
    let violations: Vec<String> = evaluation
        .violations
        .iter()
        .map(|v| v.message.clone())
        .collect();
    print_check(
        &violations,
        &evaluation.warnings,
        evaluation.config.as_ref(),
        out,
        err,
    )
}

// ---- the desktop (rule 8) -----------------------------------------------------------------------

/// Where the desktop's API listens: always loopback. The config key `lanAccess` is ignored, with one line
/// saying so (design 4.5.5 rule 8, 10.3): reaching a desktop from another device is the relay's job.
pub fn desktop_bind(lan_access: Option<&str>) -> (IpAddr, Option<String>) {
    let asked = lan_access.is_some_and(|v| v.trim() == "true");
    (
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        asked.then(|| {
            "lanAccess is on, and is ignored: the desktop's API listens on this machine only (reach it from another device through the relay)".to_string()
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const OWNER: Facts = Facts {
        owner_exists: true,
        web_login: true,
    };
    const NO_OWNER: Facts = Facts {
        owner_exists: false,
        web_login: true,
    };
    /// A build without the web login.
    const HEADLESS: Facts = Facts {
        owner_exists: false,
        web_login: false,
    };
    const GOOD_TOKEN: &str = "abcdefghijklmnopqrstuvwxyz0123456789ABCD";

    fn vars(pairs: &[(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        let map: BTreeMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |n| map.get(n).cloned()
    }

    fn eval(pairs: &[(&'static str, &'static str)], facts: Facts) -> Evaluation {
        evaluate(&vars(pairs), &facts)
    }

    /// The rules the environment breaks.
    fn broken(pairs: &[(&'static str, &'static str)], facts: Facts) -> Vec<Rule> {
        eval(pairs, facts)
            .violations
            .iter()
            .map(|v| v.rule)
            .collect()
    }

    fn ok(pairs: &[(&'static str, &'static str)], facts: Facts) -> Config {
        let e = eval(pairs, facts);
        assert!(e.violations.is_empty(), "{pairs:?}: {:?}", e.violations);
        e.config.expect("a config")
    }

    // ---- a server with no new variables is a local install ---------------------------------------

    #[test]
    fn a_server_with_no_new_variables_is_a_local_install_with_nothing_trusted() {
        for facts in [NO_OWNER, OWNER, HEADLESS] {
            let c = ok(&[], facts);
            assert_eq!(
                (c.exposure, c.bind, c.port),
                (Exposure::Local, Bind::Loopback, 17972)
            );
            assert!(c.trusted.is_empty() && c.public.is_empty() && c.allowed_hosts.is_empty());
            assert!(
                c.login_allow.is_empty() && !c.allow_public_plaintext && c.static_token.is_none()
            );
            assert!(!c.proxy_only() && c.warnings.is_empty());
        }
        // A blank variable is an unset one, as everywhere else.
        let c = ok(
            &[
                ("OAIY_SERVER_BIND", "  "),
                ("OAIY_PUBLIC_URL", ""),
                ("OAIY_TRUSTED_PROXIES", " "),
                ("OAIY_SERVER_TOKEN", "  "),
                ("OAIY_SERVER_PORT", " "),
            ],
            NO_OWNER,
        );
        assert_eq!(c.exposure, Exposure::Local);
        assert_eq!(c.static_token, None);
        // The default mode: scoped where the web login is built in, legacy where it is not.
        assert_eq!(ok(&[], NO_OWNER).mode, AccessMode::Scoped);
        assert_eq!(ok(&[], HEADLESS).mode, AccessMode::Legacy);
        assert_eq!(
            validate_config(&vars(&[]), &NO_OWNER)
                .unwrap()
                .banner_lines()
                .len(),
            1
        );
    }

    // ---- rule 1: OAIY_SERVER_BIND ------------------------------------------------------------------

    #[test]
    fn rule_1_the_bind_is_loopback_lan_or_an_ip_literal() {
        for (text, addr, loopback) in [
            ("loopback", "127.0.0.1", true),
            ("LOOPBACK", "127.0.0.1", true),
            ("lan", "0.0.0.0", false),
            (" Lan ", "0.0.0.0", false),
            ("LAN", "0.0.0.0", false),
            ("0.0.0.0", "0.0.0.0", false),
            ("::", "::", false),
            ("127.0.0.1", "127.0.0.1", true),
            ("127.0.0.2", "127.0.0.2", true),
            ("::1", "::1", true),
            ("[::1]", "::1", true),
            ("[::]", "::", false),
            ("::ffff:127.0.0.1", "::ffff:127.0.0.1", true),
            ("192.168.1.5", "192.168.1.5", false),
            ("10.0.0.7", "10.0.0.7", false),
            ("2001:db8::1", "2001:db8::1", false),
        ] {
            let b = Bind::parse(Some(text)).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(b.addr().to_string(), addr, "{text}");
            assert_eq!(b.is_loopback(), loopback, "{text}");
        }
        assert_eq!(Bind::parse(None), Ok(Bind::Loopback));
        assert_eq!(Bind::parse(Some("")), Ok(Bind::Loopback));
        assert_eq!(Bind::parse(Some("  ")), Ok(Bind::Loopback));
    }

    #[test]
    fn rule_1_a_bind_that_is_none_of_them_is_refused_and_says_what_to_use() {
        for bad in [
            "bogus",
            "true",
            "1",
            "any",
            "all",
            "public",
            "localhost",
            "127.0.0.1:8080",
            "0.0.0.0/0",
            "192.168.1",
            "1.2.3.4.5",
            "lan;",
            "lan lan",
            "::1%lo",
            "[::1",
            "::1]",
            "0x7f.1",
            "256.0.0.1",
            "https://0.0.0.0",
        ] {
            let e = eval(&[("OAIY_SERVER_BIND", bad)], NO_OWNER);
            assert!(
                e.violations.iter().any(|v| v.rule == Rule::Bind),
                "{bad:?} should be refused: {:?}",
                e.violations
            );
            assert!(e.config.is_none());
            let msg = &e
                .violations
                .iter()
                .find(|v| v.rule == Rule::Bind)
                .unwrap()
                .message;
            assert!(
                msg.contains("OAIY_SERVER_BIND") && msg.contains("loopback") && msg.contains("lan"),
                "{msg}"
            );
            assert!(!msg.contains('\n'), "one line: {msg}");
        }
    }

    // ---- rule 2: a lan install needs an owner ---------------------------------------------------

    #[test]
    fn rule_2_a_lan_bind_with_no_owner_is_refused_and_names_the_console_command() {
        for bind in ["lan", "0.0.0.0", "192.168.1.5", "::", "2001:db8::1"] {
            let e = eval(&[("OAIY_SERVER_BIND", bind)], NO_OWNER);
            let v = e
                .violations
                .iter()
                .find(|v| v.rule == Rule::LanNeedsOwner)
                .unwrap_or_else(|| panic!("{bind}: {:?}", e.violations));
            assert!(v.message.contains("oaiy-server auth init"), "{}", v.message);
            assert!(v.message.contains(bind), "{}", v.message);
            assert!(e.config.is_none());
            // Static token or not, it is no stand-in for a login.
            assert_eq!(
                broken(
                    &[
                        ("OAIY_SERVER_BIND", bind),
                        ("OAIY_SERVER_TOKEN", GOOD_TOKEN)
                    ],
                    NO_OWNER
                ),
                [Rule::LanNeedsOwner],
                "{bind}"
            );
        }
    }

    #[test]
    fn rule_2_a_lan_bind_with_an_owner_starts_and_says_it_is_lan() {
        let c = ok(&[("OAIY_SERVER_BIND", "lan")], OWNER);
        assert_eq!(c.exposure, Exposure::Lan);
        assert!(!c.proxy_only());
        assert!(
            c.banner_lines()[0].contains("exposure lan"),
            "{:?}",
            c.banner_lines()
        );
        assert!(c.banner_lines()[0].contains("0.0.0.0:17972"));
        // A lan bind reaches the network; a specific address is the same exposure.
        assert_eq!(
            ok(&[("OAIY_SERVER_BIND", "192.168.1.5")], OWNER).exposure,
            Exposure::Lan
        );
    }

    #[test]
    fn rule_2_needs_no_owner_where_the_bind_is_loopback_or_a_proxy_is_named() {
        // Loopback, no proxy: setup-only until the console makes the owner.
        let c = ok(&[], NO_OWNER);
        assert!(c.setup_only_at_start());
        assert_eq!(ok(&[], OWNER).setup_only_at_start(), false);
        // Loopback with a proxy on this machine: setup-only.
        let c = ok(&[("OAIY_PUBLIC_URL", "https://dash.example.com")], NO_OWNER);
        assert_eq!(c.exposure, Exposure::Proxied);
        assert!(c.setup_only_at_start());
        // A container pair: bound beyond loopback, the proxy named. A proxied install, setup-only.
        let c = ok(
            &[
                ("OAIY_SERVER_BIND", "0.0.0.0"),
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", "172.30.0.0/24"),
            ],
            NO_OWNER,
        );
        assert_eq!(c.exposure, Exposure::Proxied);
        assert!(c.proxy_only() && c.setup_only_at_start());
    }

    // ---- rule 3: the public URLs ------------------------------------------------------------------

    #[test]
    fn rule_3_a_public_url_is_an_https_origin_and_nothing_else() {
        for bad in [
            "https://dash.example.com/some/path",
            "https://dash.example.com/",
            "https://dash.example.com?x=1",
            "https://dash.example.com#frag",
            "http://dash.example.com",
            "HTTPS://dash.example.com",
            "dash.example.com",
            "https://",
            "https://user@dash.example.com",
            "https://user:pw@dash.example.com",
            "https://dash example.com",
            "https://dash.example.com:0",
            "https://dash.example.com:99999",
            "https://dash.example.com:abc",
            "https://dash.example.com:80",
            "https://dash.example.com:8443:1",
            "ftp://dash.example.com",
            "//dash.example.com",
        ] {
            assert_eq!(
                broken(&[("OAIY_PUBLIC_URL", bad)], NO_OWNER),
                [Rule::PublicUrl],
                "{bad:?}"
            );
        }
    }

    #[test]
    fn rule_3_each_fault_of_a_public_url_is_named_so_that_the_operator_knows_what_to_change() {
        for (bad, says) in [
            (
                "https://dash.example.com/some/path",
                "a path, a query or a fragment",
            ),
            (
                "https://dash.example.com?x=1",
                "a path, a query or a fragment",
            ),
            (
                "https://dash.example.com#top",
                "a path, a query or a fragment",
            ),
            ("http://dash.example.com", "not an https URL"),
            ("dash.example.com", "not an https URL"),
            ("https://", "no host"),
            ("https://user@dash.example.com", "a user name"),
            ("https://dash.example.com:80", "port 80"),
            ("https://localhost:8443", "a loopback host"),
            ("https://dash example.com", "not a host name"),
        ] {
            let e = eval(&[("OAIY_PUBLIC_URL", bad)], NO_OWNER);
            let message = &e.violations[0].message;
            assert!(
                message.contains(says)
                    && message.contains("OAIY_PUBLIC_URL")
                    && message.contains("https://<host>[:<port>]"),
                "{bad}: {message}"
            );
        }
    }

    #[test]
    fn rule_3_a_public_url_that_names_this_machine_is_refused() {
        for bad in [
            "https://localhost",
            "https://LOCALHOST",
            "https://localhost:8443",
            "https://dash.localhost",
            "https://dash.oaiy.localhost",
            "https://127.0.0.1",
            "https://127.0.0.2:8443",
            "https://[::1]",
            "https://[::ffff:127.0.0.1]",
            "https://0.0.0.0",
            "https://[::]",
        ] {
            assert_eq!(
                broken(&[("OAIY_PUBLIC_URL", bad)], NO_OWNER),
                [Rule::PublicUrl],
                "{bad:?}"
            );
        }
    }

    #[test]
    fn rule_3_two_apps_may_not_share_a_host_and_the_others_need_the_dashboards_url() {
        let dash = "https://dash.example.com";
        assert_eq!(
            broken(
                &[
                    ("OAIY_PUBLIC_URL", dash),
                    ("OAIY_AGENT_URL", "https://dash.example.com")
                ],
                NO_OWNER
            ),
            [Rule::PublicUrl]
        );
        // Cookies do not separate ports: the same host name on another port is the same host.
        assert_eq!(
            broken(
                &[
                    ("OAIY_PUBLIC_URL", dash),
                    ("OAIY_FLOWS_URL", "https://DASH.example.com:8443")
                ],
                NO_OWNER
            ),
            [Rule::PublicUrl]
        );
        assert_eq!(
            broken(
                &[
                    ("OAIY_PUBLIC_URL", dash),
                    ("OAIY_AGENT_URL", "https://apps.example.com"),
                    ("OAIY_FLOWS_URL", "https://apps.example.com:9443"),
                ],
                NO_OWNER
            ),
            [Rule::PublicUrl]
        );
        // An app with no dashboard is not served: the dashboard's URL is required for a proxied install.
        for only in [
            ("OAIY_AGENT_URL", "https://agent.example.com"),
            ("OAIY_FLOWS_URL", "https://flows.example.com"),
        ] {
            let e = eval(&[only], NO_OWNER);
            assert_eq!(
                e.violations.iter().map(|v| v.rule).collect::<Vec<_>>(),
                [Rule::PublicUrl]
            );
            assert!(e.violations[0].message.contains("OAIY_PUBLIC_URL"));
        }
        // Every problem at once, not the first only.
        let e = eval(
            &[
                ("OAIY_PUBLIC_URL", "http://dash.example.com"),
                ("OAIY_AGENT_URL", "https://localhost"),
                ("OAIY_FLOWS_URL", "https://flows.example.com/x"),
            ],
            NO_OWNER,
        );
        assert_eq!(e.violations.len(), 3, "{:?}", e.violations);
    }

    #[test]
    fn rule_3_good_public_urls_are_read_to_hosts_and_origins() {
        let c = ok(
            &[
                ("OAIY_PUBLIC_URL", "https://Dash.Example.com"),
                ("OAIY_AGENT_URL", "https://agent.example.com:8443"),
                ("OAIY_FLOWS_URL", "https://flows.example.com:443"),
            ],
            NO_OWNER,
        );
        assert_eq!(c.exposure, Exposure::Proxied);
        let host = |s: &str| HostName::parse(s).unwrap();
        assert_eq!(c.public.get(&host("dash.example.com")), Some(&App::Dash));
        assert_eq!(
            c.public.get(&host("agent.example.com:8443")),
            Some(&App::Agent)
        );
        assert_eq!(c.public.get(&host("flows.example.com")), Some(&App::Flows));
        assert_eq!(c.public_origins[&App::Dash], "https://dash.example.com");
        assert_eq!(
            c.public_origins[&App::Agent],
            "https://agent.example.com:8443"
        );
        assert_eq!(c.public_origins[&App::Flows], "https://flows.example.com");
        // The dashboard alone is fine: the others are simply not served.
        let c = ok(&[("OAIY_PUBLIC_URL", "https://dash.example.com")], NO_OWNER);
        assert_eq!(c.public.len(), 1);
        // An address as a host, and an IPv6 one.
        ok(&[("OAIY_PUBLIC_URL", "https://203.0.113.9:8443")], NO_OWNER);
        ok(&[("OAIY_PUBLIC_URL", "https://[2001:db8::9]")], NO_OWNER);
    }

    // ---- rule 4: name your proxy ------------------------------------------------------------------

    #[test]
    fn rule_4_a_network_bind_behind_a_public_url_must_name_its_proxy() {
        for bind in ["lan", "0.0.0.0", "10.0.0.7", "::"] {
            let pairs = [
                ("OAIY_SERVER_BIND", bind),
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
            ];
            assert_eq!(
                broken(&pairs, NO_OWNER),
                [Rule::ProxyOnlyNeedsProxies],
                "{bind}"
            );
            assert_eq!(
                broken(&pairs, OWNER),
                [Rule::ProxyOnlyNeedsProxies],
                "{bind}"
            );
            let e = eval(&pairs, OWNER);
            assert!(e.violations[0].message.contains("OAIY_TRUSTED_PROXIES"));
            // Blank, or nothing but separators, names nobody.
            for blank in ["", "  ", ",", " , ,"] {
                let with = [
                    ("OAIY_SERVER_BIND", bind),
                    ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                    ("OAIY_TRUSTED_PROXIES", blank),
                ];
                assert_eq!(
                    broken(&with, OWNER),
                    [Rule::ProxyOnlyNeedsProxies],
                    "{bind} {blank:?}"
                );
            }
        }
    }

    #[test]
    fn rule_4_a_named_proxy_or_a_loopback_bind_is_not_refused() {
        let named = ok(
            &[
                ("OAIY_SERVER_BIND", "0.0.0.0"),
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", "172.30.0.0/24, 10.0.0.5"),
            ],
            OWNER,
        );
        assert!(named.proxy_only() && named.trusted_explicit);
        assert_eq!(named.trusted.describe(), ["172.30.0.0/24", "10.0.0.5/32"]);
        assert!(named
            .banner_lines()
            .iter()
            .any(|l| l.contains("direct_access_refused")));
        // Loopback bind: the default proxy is the one on this machine, and the shape is not proxy-only.
        let local_proxy = ok(&[("OAIY_PUBLIC_URL", "https://dash.example.com")], OWNER);
        assert!(!local_proxy.proxy_only() && !local_proxy.trusted_explicit);
        assert_eq!(local_proxy.trusted.describe(), ["127.0.0.1/32", "::1/128"]);
        // Without a public URL a network bind is a lan install: rule 4 is not what judges it.
        assert!(
            !broken(&[("OAIY_SERVER_BIND", "lan")], OWNER).contains(&Rule::ProxyOnlyNeedsProxies)
        );
        // A named list replaces the default, it does not add to it.
        let c = ok(
            &[
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", "192.0.2.7"),
            ],
            OWNER,
        );
        assert_eq!(c.trusted.describe(), ["192.0.2.7/32"]);
    }

    #[test]
    fn a_trusted_proxy_that_is_not_an_address_is_refused_and_a_broad_one_is_warned_of() {
        for bad in [
            "nginx",
            "10.0.0.0/33",
            "10.0.0/8",
            "172.30.0.0/24;10.0.0.5",
            "::1/129",
            "1.2.3.4/x",
        ] {
            let pairs = [
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", bad),
            ];
            assert_eq!(broken(&pairs, OWNER), [Rule::TrustedProxies], "{bad:?}");
        }
        // One good entry does not excuse the other.
        assert_eq!(
            broken(
                &[
                    ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                    ("OAIY_TRUSTED_PROXIES", "10.0.0.5, nginx"),
                ],
                OWNER
            ),
            [Rule::TrustedProxies]
        );
        let c = ok(
            &[
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", "0.0.0.0/0"),
            ],
            OWNER,
        );
        assert!(
            c.warnings.iter().any(|w| w.contains("every address")),
            "{:?}",
            c.warnings
        );
        // On a local install the setting does nothing, and says so.
        let c = ok(&[("OAIY_TRUSTED_PROXIES", "10.0.0.5")], OWNER);
        assert!(
            c.warnings.iter().any(|w| w.contains("no effect")),
            "{:?}",
            c.warnings
        );
    }

    // ---- rule 5: the static token ---------------------------------------------------------------

    #[test]
    fn rule_5_a_static_token_of_the_wrong_shape_is_refused() {
        let too_long = "a1".repeat(129);
        let few_distinct = "ab".repeat(20);
        let fifteen_distinct = format!("{}{}", "abcdefghijklmno", "a".repeat(20));
        for (what, bad) in [
            ("short", "short".to_string()),
            (
                "31 characters",
                "abcdefghijklmnopqrstuvwxyz01234".to_string(),
            ),
            (
                "257 characters",
                format!("{}x", "abcdefghij".repeat(25) + "abcdef"),
            ),
            (
                "a space inside",
                "abcdefghijklmnopqrstuvwxyz 0123456789ABCD".to_string(),
            ),
            (
                "a tab inside",
                "abcdefghijklmnopqrstuvwxyz\t0123456789ABCD".to_string(),
            ),
            (
                "a control character",
                "abcdefghijklmnopqrstuvwxyz\u{1}0123456789ABCD".to_string(),
            ),
            (
                "a non-ASCII character",
                "abcdefghijklmnopqrstuvwxyz0123456789ABC\u{e9}".to_string(),
            ),
            ("too long", too_long.clone()),
            ("two characters repeated", few_distinct),
            ("15 distinct characters", fifteen_distinct),
            ("change-me", "change-me".to_string()),
        ] {
            let e = evaluate(
                &|n| (n == "OAIY_SERVER_TOKEN").then(|| bad.clone()),
                &NO_OWNER,
            );
            assert_eq!(
                e.violations.iter().map(|v| v.rule).collect::<Vec<_>>(),
                [Rule::StaticToken],
                "{what}"
            );
            let msg = &e.violations[0].message;
            assert!(
                msg.contains("OAIY_SERVER_TOKEN") && msg.contains("auth token create"),
                "{msg}"
            );
            // The token is never echoed into the message (it is a credential, even a weak one).
            assert!(!msg.contains(&bad) || bad.len() < 12, "{what}: {msg}");
        }
    }

    #[test]
    fn rule_5_a_static_token_of_the_right_shape_is_kept_trimmed() {
        let sixteen_of_thirty_two = "abcdefghijklmnop".repeat(2);
        let printable_256: String = ('!'..='~').cycle().take(256).collect();
        for good in [
            GOOD_TOKEN.to_string(),
            "Sup3r$ecret!Zq7kLm9VbNw2XyHdFg5!".to_string(),
            sixteen_of_thirty_two,
            printable_256,
        ] {
            let padded = format!("  {good}\n");
            let c = evaluate(
                &|n| (n == "OAIY_SERVER_TOKEN").then(|| padded.clone()),
                &NO_OWNER,
            )
            .config
            .unwrap_or_else(|| panic!("{good} was refused"));
            assert_eq!(c.static_token.as_deref(), Some(good.as_str()));
        }
        assert_eq!(ok(&[], NO_OWNER).static_token, None);
    }

    // ---- rule 6: the mode -----------------------------------------------------------------------

    #[test]
    fn rule_6_shadow_is_refused_on_a_proxied_or_lan_install_and_allowed_on_a_local_one() {
        assert_eq!(
            broken(
                &[
                    ("OAIY_ACCESS_MODE", "shadow"),
                    ("OAIY_PUBLIC_URL", "https://dash.example.com")
                ],
                NO_OWNER
            ),
            [Rule::Mode]
        );
        assert_eq!(
            broken(
                &[("OAIY_ACCESS_MODE", "shadow"), ("OAIY_SERVER_BIND", "lan")],
                OWNER
            ),
            [Rule::Mode]
        );
        assert_eq!(
            ok(&[("OAIY_ACCESS_MODE", "shadow")], NO_OWNER).mode,
            AccessMode::Shadow
        );
        assert_eq!(
            ok(
                &[("OAIY_ACCESS_MODE", "scoped"), ("OAIY_SERVER_BIND", "lan")],
                OWNER
            )
            .mode,
            AccessMode::Scoped
        );
    }

    #[test]
    fn rule_6_legacy_is_refused_with_the_web_login_with_an_owner_and_off_a_local_install() {
        // With the web login built in.
        assert_eq!(
            broken(&[("OAIY_ACCESS_MODE", "legacy")], NO_OWNER),
            [Rule::Mode]
        );
        // Without it, on a local install with no owner: as it has always been.
        assert_eq!(
            ok(&[("OAIY_ACCESS_MODE", "legacy")], HEADLESS).mode,
            AccessMode::Legacy
        );
        // An owner login exists.
        assert_eq!(
            broken(
                &[("OAIY_ACCESS_MODE", "legacy")],
                Facts {
                    owner_exists: true,
                    web_login: false
                }
            ),
            [Rule::Mode]
        );
        // Off a local install: proxied and lan.
        assert_eq!(
            broken(
                &[
                    ("OAIY_ACCESS_MODE", "legacy"),
                    ("OAIY_PUBLIC_URL", "https://dash.example.com")
                ],
                HEADLESS
            ),
            [Rule::Mode]
        );
        assert_eq!(
            broken(
                &[("OAIY_ACCESS_MODE", "legacy"), ("OAIY_SERVER_BIND", "lan")],
                Facts {
                    owner_exists: true,
                    web_login: false
                }
            ),
            [Rule::Mode]
        );
        // The default of a build without the login is legacy, so a lan bind with no mode named is refused too.
        assert_eq!(
            broken(
                &[("OAIY_SERVER_BIND", "lan")],
                Facts {
                    owner_exists: true,
                    web_login: false
                }
            ),
            [Rule::Mode]
        );
        // A value that is not a mode.
        let e = eval(&[("OAIY_ACCESS_MODE", "scopd")], NO_OWNER);
        assert_eq!(e.violations[0].rule, Rule::Mode);
        assert!(e.violations[0].message.contains("scopd"));
    }

    // ---- the rest -----------------------------------------------------------------------------------

    #[test]
    fn a_port_is_a_number_from_1_to_65535() {
        assert_eq!(parse_port(None), Ok(17972));
        assert_eq!(parse_port(Some(" 8080 ")), Ok(8080));
        assert_eq!(parse_port(Some("65535")), Ok(65535));
        for bad in [
            "0",
            "65536",
            "-1",
            "80a",
            "http",
            "1e3",
            "0x50",
            "8080 8081",
        ] {
            assert!(parse_port(Some(bad)).is_err(), "{bad:?}");
            assert_eq!(
                broken(&[("OAIY_SERVER_PORT", bad)], NO_OWNER),
                [Rule::Port],
                "{bad:?}"
            );
        }
    }

    #[test]
    fn extra_hosts_and_the_login_allow_list_are_read_strictly() {
        let c = ok(
            &[
                (
                    "OAIY_ALLOWED_HOSTS",
                    "nas.example, nas.example:9000 ,[::1]:8080",
                ),
                (
                    "OAIY_LOGIN_ALLOW",
                    "203.0.113.0/24, 2001:db8::/32, 198.51.100.7",
                ),
            ],
            OWNER,
        );
        assert_eq!(c.allowed_hosts.len(), 3);
        assert_eq!(c.login_allow.len(), 3);
        assert_eq!(
            broken(&[("OAIY_ALLOWED_HOSTS", "nas example")], OWNER),
            [Rule::AllowedHosts]
        );
        assert_eq!(
            broken(&[("OAIY_ALLOWED_HOSTS", "https://nas.example")], OWNER),
            [Rule::AllowedHosts]
        );
        // An allow-list of nothing usable would let everyone in: every way to write one is a violation.
        for bad in [
            "nonsense",
            "203.0.113.0/33",
            "203.0.113.0/24;198.51.100.0/24",
            "203.0.113.0/24, nonsense",
        ] {
            assert_eq!(
                broken(&[("OAIY_LOGIN_ALLOW", bad)], OWNER),
                [Rule::LoginAllow],
                "{bad:?}"
            );
        }
        assert_eq!(
            broken(&[("OAIY_LOGIN_ALLOW", ",")], OWNER),
            [Rule::LoginAllow]
        );
    }

    #[test]
    fn the_plaintext_override_is_exactly_1_and_only_for_a_lan_install() {
        assert!(
            ok(
                &[
                    ("OAIY_SERVER_BIND", "lan"),
                    ("OAIY_ALLOW_PUBLIC_PLAINTEXT", "1")
                ],
                OWNER
            )
            .allow_public_plaintext
        );
        for other in ["true", "yes", "on", "0", "2", " 1x"] {
            let c = ok(
                &[
                    ("OAIY_SERVER_BIND", "lan"),
                    ("OAIY_ALLOW_PUBLIC_PLAINTEXT", other),
                ],
                OWNER,
            );
            assert!(!c.allow_public_plaintext, "{other:?}");
            assert!(
                c.warnings.iter().any(|w| w.contains("is not 1")),
                "{other:?}"
            );
        }
        let c = ok(&[("OAIY_ALLOW_PUBLIC_PLAINTEXT", "1")], OWNER);
        assert!(
            c.warnings.iter().any(|w| w.contains("no effect")),
            "{:?}",
            c.warnings
        );
    }

    #[test]
    fn every_violation_is_listed_at_once_in_the_order_of_the_rules_and_the_first_is_the_error() {
        let pairs = [
            ("OAIY_SERVER_BIND", "bogus"),
            ("OAIY_PUBLIC_URL", "https://dash.example.com/x"),
            ("OAIY_SERVER_TOKEN", "short"),
            ("OAIY_ACCESS_MODE", "scopd"),
            ("OAIY_SERVER_PORT", "0"),
        ];
        let e = eval(&pairs, NO_OWNER);
        let rules: Vec<Rule> = e.violations.iter().map(|v| v.rule).collect();
        assert_eq!(
            rules,
            [
                Rule::Bind,
                Rule::PublicUrl,
                Rule::StaticToken,
                Rule::Mode,
                Rule::Port
            ]
        );
        assert!(rules.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(
            rules.iter().filter_map(|r| r.number()).collect::<Vec<_>>(),
            [1, 3, 5, 6]
        );
        let one = validate_config(&vars(&pairs), &NO_OWNER).unwrap_err();
        assert_eq!(one, e.violations[0].message);
        assert!(!one.contains('\n'));
        assert!(e.config.is_none());
        // The rules all at once for a lan install with a bad token and no owner.
        assert_eq!(
            broken(
                &[("OAIY_SERVER_BIND", "lan"), ("OAIY_SERVER_TOKEN", "x")],
                NO_OWNER
            ),
            [Rule::LanNeedsOwner, Rule::StaticToken]
        );
    }

    #[test]
    fn the_desktop_never_binds_beyond_loopback_and_says_so_when_lan_access_is_on() {
        for value in [None, Some("false"), Some(""), Some("no")] {
            let (addr, note) = desktop_bind(value);
            assert_eq!(addr, IpAddr::V4(Ipv4Addr::LOCALHOST));
            assert_eq!(note, None, "{value:?}");
        }
        let (addr, note) = desktop_bind(Some(" true "));
        assert_eq!(addr, IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert!(note.unwrap().contains("lanAccess"));
    }

    #[test]
    fn the_banner_says_what_the_install_is_and_holds_no_secret() {
        let c = ok(
            &[
                ("OAIY_SERVER_BIND", "0.0.0.0"),
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_AGENT_URL", "https://agent.example.com"),
                ("OAIY_TRUSTED_PROXIES", "172.30.0.0/24"),
                ("OAIY_SERVER_TOKEN", GOOD_TOKEN),
            ],
            NO_OWNER,
        );
        let text = c.banner_lines().join("\n");
        for want in [
            "exposure proxied",
            "dash https://dash.example.com",
            "agent https://agent.example.com",
            "172.30.0.0/24",
            "direct_access_refused",
        ] {
            assert!(text.contains(want), "{want} in {text}");
        }
        assert!(!text.contains(GOOD_TOKEN));
        let local = ok(&[], OWNER).banner_lines().join("\n");
        assert!(
            local.contains("exposure local") && local.contains("127.0.0.1:17972"),
            "{local}"
        );
    }
}
