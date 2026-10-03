//! Exposure and the startup rules (design 4.5.4, 4.5.5 and the variables of 4.13).
//!
//! [`evaluate`] reads the environment of `oaiy-server` and says, all at once, what is wrong with it; it reads
//! nothing else (whether `<data>/auth/owner.json` exists is a [`Facts`] the caller looks up, so that every rule
//! is a pure function of two values and can be tested without a disk). [`validate_config`] is the same with the
//! first violation as its error: the one line `oaiy-server` prints before it exits 78 (`EX_CONFIG`, which the
//! shipped unit does not restart: `RestartPreventExitStatus=78`). `oaiy-server check` prints every violation, the
//! owner file's first (`inspect_owner_file`: read as the store reads it, which is what the start does).
//!
//! The rules of 4.5.5, by number:
//!
//! 1. `OAIY_SERVER_BIND` is `loopback`, `lan` (an alias of `0.0.0.0`) or an IP literal.
//! 2. A **lan** install (a bind beyond loopback and no `OAIY_PUBLIC_URL`) needs `<data>/auth/owner.json`, a file
//!    this server reads as an owner (`inspect_owner_file`: as the store reads it; an empty, cut-off or newer file
//!    is a refusal named by the file, in the start and in `check`). The shape is all it reads: a file of the full
//!    shape satisfies the rule even when its hash matches no password (`the_owner_file_is_read_as_the_server_reads_it`
//!    holds that), which fails closed: nobody can sign in until `auth reset-password`. A **proxied** install may
//!    start without one, in setup-only mode.
//! 3. `OAIY_PUBLIC_URL`, `OAIY_AGENT_URL` and `OAIY_FLOWS_URL` are each `https://<host>[:<port>]` with no path,
//!    query or fragment, none has a loopback host, and no two share a host. An app whose URL is unset is not
//!    served; the dashboard's URL is required by the other two.
//! 4. A bind beyond loopback with `OAIY_PUBLIC_URL` needs `OAIY_TRUSTED_PROXIES` ("name your proxy"): that is the
//!    proxy-only shape, in which a direct peer that is not a trusted proxy is `403 direct_access_refused`.
//! 5. `OAIY_SERVER_TOKEN` has no common pattern (`token::check_static_token_shape`: 32 to 256 printable ASCII
//!    characters, an estimated 128 bits for the alphabet they are written in, at least 8 different, no placeholder word, no
//!    run that counts or follows the keyboard, no piece of 8 characters twice in either case, not a phrase of common words,
//!    not the digest of a password everyone tries). A guard against the obvious, not a strength meter. Design 4.1 gave it 16
//!    different characters, which refused `openssl rand -hex 24` about one time in two.
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
use super::host::{parse_port_digits, HostName};
use super::mode::{validate_mode, AccessMode, Exposure};
use super::presets::App;
use super::token::static_token_refusal;

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

/// What follows `https://`, the scheme in any case (RFC 3986: a scheme is case-insensitive, so `HTTPS://DASH.EXAMPLE.TEST`
/// is the URL of `https://dash.example.test`), or `None` for a text that is not an https URL.
pub fn strip_https_scheme(text: &str) -> Option<&str> {
    let (scheme, rest) = (text.get(..8)?, text.get(8..)?);
    scheme.eq_ignore_ascii_case("https://").then_some(rest)
}

/// A configured public host: `https://<host>[:<port>]`. `Err` says what to change. The origin it makes is
/// lowercase, whatever the case of the text.
pub fn parse_https_origin(var: &str, text: &str) -> Result<(HostName, String), String> {
    let bad = |why: &str| {
        Err(format!(
            "{var}={text:?} {why}: write it as https://<host>[:<port>] with no path, query or fragment"
        ))
    };
    let Some(rest) = strip_https_scheme(text) else {
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
        Some(v) => parse_port_digits(v).ok_or_else(|| {
            format!("OAIY_SERVER_PORT={v:?} is not a port: use a number from 1 to 65535, with digits only")
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

/// The routes that existed before the access model, which a server without the web login let a token reach and which the
/// `cli` preset (what `OAIY_SERVER_TOKEN` is on the web build) cannot: `403 insufficient_scope`. The README lists them
/// (The headless server on the web build), `auth::login_tests` pins this number and these scopes to the route table.
pub const TOKEN_ONLY_LOSES_ROUTES: usize = 113;
/// The scopes those routes ask for, in order.
pub const TOKEN_ONLY_LOSES_SCOPES: [&str; 29] = [
    "agent.read",
    "agent.serve",
    "agent.settings",
    "agent.tasks",
    "ai.admin",
    "auth.manage",
    "auth.read",
    "auth.revoke",
    "calendar.read",
    "calendar.write",
    "calls.manage",
    "calls.read",
    "calls.settings",
    "calls.write",
    "companion.manage",
    "companion.read",
    "connectors.use",
    "contacts.read",
    "contacts.write",
    "control.admin",
    "control.read",
    "link.manage",
    "link.read",
    "plugins.install",
    "runtimes.install",
    "services.define",
    "setup.read",
    "setup.write",
    "speech.use",
];

/// One line for `check` and the start's log: what `OAIY_SERVER_TOKEN` is on the web build, and what it is not.
fn token_only_warning() -> String {
    format!(
        "OAIY_SERVER_TOKEN is the `cli` preset on this build (it has the web login, whose access mode is `scoped`), which is less than a server without the web login let a token reach: {TOKEN_ONLY_LOSES_ROUTES} routes answer it `403 insufficient_scope` (voice calls and voices, the calendar, contacts, the Agent's tasks and preferences, setup, the control settings and log, the account link, pairings, the companion relay, AI provider keys, and defining services or installing runtimes and plugins). There is no setting that brings the old behaviour back: run `oaiy-server auth init` and make the tokens those routes need with `oaiy-server auth token create --preset cli --scope <scope>` (`--preset cli-admin` adds the installs). Until the owner exists the server is in setup-only mode: a caller with no credential is told `setup_required` everywhere but health and the login routes, and your token is judged as it is everywhere else"
    )
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
        Some(t) => match static_token_refusal(&t) {
            None => Some(t),
            Some(why) => {
                violate!(
                    Rule::StaticToken,
                    format!(
                        "{why}: use a random value such as `openssl rand -base64 32`, or a token made on the console (`oaiy-server auth token create`)"
                    ),
                );
                None
            }
        },
    };

    // What the operator's token is on the web build, told where it is read (`check` and the start's log), not found out
    // by a 403: a server without the web login let a token reach every route, and this one holds it to the `cli` preset.
    if static_token.is_some() && facts.web_login {
        warnings.push(token_only_warning());
    }

    // 2. a lan install needs an owner.
    if exposure == Exposure::Lan && !facts.owner_exists {
        let bind = env("OAIY_SERVER_BIND")
            .map(|b| b.trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "lan".into());
        violate!(
            Rule::LanNeedsOwner,
            if facts.web_login {
                format!(
                    "OAIY_SERVER_BIND={bind} reaches the network, which needs an owner login first: run `oaiy-server auth init`, then start the server again"
                )
            } else {
                // No console in this build: `auth init` answers "no console", and is not what to run.
                format!(
                    "OAIY_SERVER_BIND={bind} reaches the network, which needs an owner login first, and this build has no web login to make one (it was built without the `web` feature): use the web build of oaiy-server, run `oaiy-server auth init` with it, then start the server again"
                )
            },
        );
    }

    // 6. the mode.
    let mode = match mode_from_env(env("OAIY_ACCESS_MODE").as_deref(), facts.web_login) {
        Ok(mode) => {
            if let Err(refusal) = validate_mode(mode, exposure, facts.owner_exists) {
                let defaulted = nonblank(env, "OAIY_ACCESS_MODE").is_none();
                if defaulted && !facts.web_login {
                    // The operator named no mode: the `legacy` that a build without the web login runs unless told
                    // otherwise is what is refused, and the line says so (not "OAIY_ACCESS_MODE=legacy is refused").
                    let why = if facts.owner_exists {
                        "an owner login exists (<data>/auth/owner.json)".to_string()
                    } else {
                        format!("this is a {} install", exposure.name())
                    };
                    violate!(
                        Rule::Mode,
                        format!(
                            "OAIY_ACCESS_MODE is not set, and a build without the web login (this one was built without the `web` feature) then runs legacy, which is refused because {why}: use the web build of oaiy-server, or set OAIY_ACCESS_MODE=scoped"
                        ),
                    );
                } else {
                    violate!(Rule::Mode, refusal.to_string());
                }
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

/// What `<data>/auth/owner.json` is to a server that opens it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnerState {
    /// No file: no owner login yet.
    Absent,
    /// A file that this server reads as an owner.
    Usable,
    /// A file that is there and that this server would refuse to start with (one line: which file and why, as the
    /// store says it). It is an owner file all the same: rule 2 must not send the operator to `auth init` over it.
    Refused(String),
}

impl OwnerState {
    /// Whether there is a file, usable or not.
    pub fn exists(&self) -> bool {
        !matches!(self, OwnerState::Absent)
    }

    /// The line that says why this server would not start with it.
    pub fn refusal(&self) -> Option<&str> {
        match self {
            OwnerState::Refused(line) => Some(line),
            _ => None,
        }
    }
}

/// Read `<data>/auth/owner.json` as the server does at start, so that the start (rule 2 and the mode it allows) and
/// `oaiy-server check` say the same: the store's own reader (a file that is not JSON, has no version or has another
/// one is refused, and never taken for "no owner yet"), and in a build with the web login the owner document that
/// login reads from it. Existence alone is not it: an empty file used to satisfy rule 2 and stop the server one step
/// later.
pub fn inspect_owner_file(auth_dir: &std::path::Path) -> OwnerState {
    match super::store::read_owner(auth_dir) {
        Ok(None) => OwnerState::Absent,
        Err(e) => OwnerState::Refused(e.to_string()),
        #[cfg(feature = "web")]
        Ok(Some(owner)) => match super::owner::OwnerDoc::from_value(&owner.doc) {
            Ok(_) => OwnerState::Usable,
            Err(detail) => OwnerState::Refused(
                super::store::StoreError::OwnerUnparsable {
                    file: super::owner::path(auth_dir),
                    detail,
                }
                .to_string(),
            ),
        },
        #[cfg(not(feature = "web"))]
        Ok(Some(_)) => OwnerState::Usable,
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
    let owner =
        data_dir_from_env(env).map_or(OwnerState::Absent, |d| inspect_owner_file(&d.join("auth")));
    let evaluation = evaluate(
        env,
        &Facts {
            owner_exists: owner.exists(),
            web_login: false,
        },
    );
    // The file first, as the start says it, then the rules: what stops the start is what `check` lists.
    let violations: Vec<String> = owner
        .refusal()
        .map(str::to_string)
        .into_iter()
        .chain(evaluation.violations.iter().map(|v| v.message.clone()))
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
    const GOOD_TOKEN: &str = "Vl0JTnJtFseAe9ePKCDhuBymfRXQ8osZ-QMlM86leCU";

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
            "HTTP://dash.example.com",
            // (`HTTPS://dash.example.com` was in this list: a scheme is case-insensitive, RFC 3986, and it is read.)
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
        let one_run = format!("{}{}", "abcdefghijklmno", "a".repeat(20));
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
            ("a run of one character", one_run),
            ("change-me", "change-me".to_string()),
            // The old shape took this one: 40 different characters, that count.
            (
                "a run that counts",
                "abcdefghijklmnopqrstuvwxyz0123456789ABCD".to_string(),
            ),
            (
                "a word from an example",
                "verysecrettokenverysecrettoken1234".to_string(),
            ),
            (
                "a run along the keyboard",
                "qwertyuiopasdfghjklzxcvbnmqwertyui".to_string(),
            ),
            (
                "a piece twice",
                "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1k7Qz!mV3".to_string(),
            ),
            (
                "32 decimal digits, 106 bits",
                "67374834151859291927224391593075".to_string(),
            ),
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
                msg.contains("OAIY_SERVER_TOKEN")
                    && msg.contains("openssl rand -base64 32")
                    && msg.contains("auth token create"),
                "{msg}"
            );
            assert!(!msg.contains("random hex"), "hex is fine now: {msg}");
            // The token is never echoed into the message (it is a credential, even a weak one).
            assert!(!msg.contains(&bad) || bad.len() < 12, "{what}: {msg}");
        }
    }

    /// The line names what was matched (a random hex token with `0000` in it was refused with a line that did not list
    /// `0000`), says that a random token has one by chance, and never says what the token says.
    #[test]
    fn rule_5_the_line_names_what_matched_and_never_the_token() {
        for (token, names) in [
            // A random-looking token with `0000` in it: the review's case.
            ("f450474c461a0000acde4cc71a4d10a9", "\"0000\""),
            (
                "d812d74ba4c163bbf58d6fa2605bxxxx96946c9d50880c35500",
                "\"xxxx\"",
            ),
            (
                "d812d74ba4c163bbf58d6fa2605bchangeme46c9d50880c35500",
                "\"changeme\"",
            ),
            (
                "k7Qz!mV3#pW9xLd2rn8TbHv4$wN6@cJ1abcdefgh",
                "characters 33 to 40",
            ),
            ("correct-horse-battery-staple-oaiy-2026", "common words"),
            ("5f4dcc3b5aa765d61d8327deb882cf99", "MD5, SHA-1 or SHA-256"),
        ] {
            let e = evaluate(
                &|n| (n == "OAIY_SERVER_TOKEN").then(|| token.to_string()),
                &NO_OWNER,
            );
            assert_eq!(
                e.violations.iter().map(|v| v.rule).collect::<Vec<_>>(),
                [Rule::StaticToken],
                "{token}"
            );
            let line = &e.violations[0].message;
            assert!(line.contains(names), "{token}: {line}");
            assert!(
                line.contains("openssl rand -base64 32") && line.contains("auth token create"),
                "{line}"
            );
            assert!(
                !line.contains(token),
                "the whole token is never in the line: {line}"
            );
        }
        // A token that is refused by chance says to make another.
        let e = evaluate(
            &|n| (n == "OAIY_SERVER_TOKEN").then(|| "f450474c461a0000acde4cc71a4d10a9".to_string()),
            &NO_OWNER,
        );
        assert!(e.violations[0].message.contains("generate another"));
    }

    /// N1: what `OAIY_SERVER_TOKEN` is on the web build is said where an operator reads it (`check` and the start's log),
    /// with the count of routes that the `cli` preset does not reach, for every web-build install that sets a good
    /// token (with an owner or without) and for no other.
    #[test]
    fn a_token_on_the_web_build_is_told_what_it_is_and_no_other_install_is() {
        let said = |e: &Evaluation| -> Vec<String> {
            e.warnings
                .iter()
                .filter(|w| w.contains("is the `cli` preset"))
                .cloned()
                .collect()
        };
        for facts in [NO_OWNER, OWNER] {
            let e = eval(&[("OAIY_SERVER_TOKEN", GOOD_TOKEN)], facts);
            assert!(e.violations.is_empty(), "{:?}", e.violations);
            let lines = said(&e);
            assert_eq!(lines.len(), 1, "{:?}", e.warnings);
            for must in [
                format!("{TOKEN_ONLY_LOSES_ROUTES} routes"),
                "403 insufficient_scope".to_string(),
                "There is no setting that brings the old behaviour back".to_string(),
                "oaiy-server auth init".to_string(),
                "--preset cli --scope".to_string(),
                "setup-only mode".to_string(),
            ] {
                assert!(lines[0].contains(&must), "{must}: {}", lines[0]);
            }
            assert!(!lines[0].contains(GOOD_TOKEN), "the warning never says the token");
        }
        // No token: there is nothing to say. Not the web build: a token there is what it always was.
        assert!(said(&eval(&[], NO_OWNER)).is_empty());
        assert!(said(&eval(&[("OAIY_SERVER_TOKEN", GOOD_TOKEN)], HEADLESS)).is_empty());
        // A token the rule refuses is a refusal, and not a description of what it would be.
        let e = eval(
            &[("OAIY_SERVER_TOKEN", "change-me-change-me-change-me-change-me")],
            NO_OWNER,
        );
        assert_eq!(e.violations.len(), 1);
        assert!(said(&e).is_empty(), "{:?}", e.warnings);
    }

    #[test]
    fn rule_5_a_static_token_of_the_right_shape_is_kept_trimmed() {
        // 256 printable characters that are no pattern: a xorshift stream, seeded.
        let mut x: u64 = 88_172_645_463_325_252;
        let printable_256: String = (0..256)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (0x21 + (x % 94) as u8) as char
            })
            .collect();
        for good in [
            GOOD_TOKEN.to_string(),
            "Sup3r$ecret!Zq7kLm9VbNw2XyHdFg5!".to_string(),
            // Random hex, of the lengths people use, and one of them with only 15 of the 16 digits.
            "f450474c461f635bacde4cc71a4d10a9".to_string(),
            "d812d74ba4c163bbf58d6fa2605b596946c9d50880c35500".to_string(),
            "zkwdexEjP8A727+Q2moKCafq4F2IVlZbLJOseRU8Ric=".to_string(),
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
        // The default of a build without the login is legacy, so a lan bind with no mode named is refused too (and
        // the line says that it is the default that is refused: see the test of the messages below).
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

    /// A build without the web login has no console: rule 2 used to send the operator to `oaiy-server auth init`, which
    /// answers "no console" there, and the `legacy` that such a build defaults to was refused as if the operator had
    /// set it. Each line now says what is so.
    #[test]
    fn a_build_without_the_web_login_does_not_send_the_operator_to_a_console_it_has_not_got() {
        let lan_scoped = [("OAIY_SERVER_BIND", "lan"), ("OAIY_ACCESS_MODE", "scoped")];
        // Rule 2.
        let e = eval(&lan_scoped, HEADLESS);
        assert_eq!(
            e.violations.iter().map(|v| v.rule).collect::<Vec<_>>(),
            [Rule::LanNeedsOwner]
        );
        let line = &e.violations[0].message;
        assert!(
            line.contains("this build has no web login")
                && line.contains("web build of oaiy-server")
                && line.contains("`web` feature"),
            "{line}"
        );
        // The web build says what to run, and does not say it has no login.
        let e = eval(&lan_scoped, NO_OWNER);
        let line = &e.violations[0].message;
        assert!(
            line.contains("run `oaiy-server auth init`, then start the server again")
                && !line.contains("no web login"),
            "{line}"
        );
        // Rule 6: the operator set no mode, and the line says that it is the build's default that is refused.
        for (pairs, facts, rules, why) in [
            // No owner on a lan install is rule 2's as well, and the mode is named for what it is.
            (
                vec![("OAIY_SERVER_BIND", "lan")],
                HEADLESS,
                vec![Rule::LanNeedsOwner, Rule::Mode],
                "this is a lan install",
            ),
            (
                vec![("OAIY_SERVER_BIND", "lan")],
                Facts {
                    owner_exists: true,
                    web_login: false,
                },
                vec![Rule::Mode],
                "an owner login exists",
            ),
            (
                vec![("OAIY_PUBLIC_URL", "https://dash.example.com")],
                HEADLESS,
                vec![Rule::Mode],
                "this is a proxied install",
            ),
            (
                vec![],
                Facts {
                    owner_exists: true,
                    web_login: false,
                },
                vec![Rule::Mode],
                "an owner login exists",
            ),
        ] {
            let e = eval(&pairs, facts);
            assert_eq!(
                e.violations.iter().map(|v| v.rule).collect::<Vec<_>>(),
                rules,
                "{pairs:?}"
            );
            let line = &e.violations.last().unwrap().message;
            assert!(
                line.contains("OAIY_ACCESS_MODE is not set")
                    && line.contains(why)
                    && line.contains(
                        "use the web build of oaiy-server, or set OAIY_ACCESS_MODE=scoped"
                    ),
                "{pairs:?}: {line}"
            );
        }
        // What the operator did set is refused as it is named, there and in the web build.
        for facts in [
            HEADLESS,
            Facts {
                owner_exists: true,
                web_login: false,
            },
        ] {
            let e = eval(
                &[
                    ("OAIY_ACCESS_MODE", "legacy"),
                    ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ],
                facts,
            );
            assert!(
                e.violations[0]
                    .message
                    .starts_with("OAIY_ACCESS_MODE=legacy is refused"),
                "{}",
                e.violations[0].message
            );
        }
        // And the install that a build without the login can serve is served, as it was: local, and scoped named.
        assert!(eval(&[], HEADLESS).violations.is_empty());
        assert!(eval(
            &lan_scoped,
            Facts {
                owner_exists: true,
                web_login: false
            }
        )
        .violations
        .is_empty());
    }

    /// The owner file is read as the server reads it, by the start and by `check`: existence was all that rule 2
    /// asked, and a file that the store refused satisfied it and stopped the server a step later.
    #[test]
    fn the_owner_file_is_read_as_the_server_reads_it() {
        use crate::secret_file::testing::TempDir;
        let dir = TempDir::new("owner-state");
        let auth = dir.0.join("auth");
        assert_eq!(inspect_owner_file(&auth), OwnerState::Absent, "no folder");
        std::fs::create_dir_all(&auth).unwrap();
        assert_eq!(inspect_owner_file(&auth), OwnerState::Absent, "no file");
        assert!(!OwnerState::Absent.exists());
        let put = |text: &str| std::fs::write(auth.join("owner.json"), text).unwrap();
        // What the store refuses: each is a file that is there, and a line that names it and says why.
        for (what, text, says) in [
            ("an empty file", "", "cannot be read"),
            ("text", "not json", "cannot be read"),
            ("a cut-off file", r#"{"v":1,"password":"#, "cannot be read"),
            ("a list", "[]", "no version"),
            ("no version", r#"{"password":"x"}"#, "no version"),
            ("a version that is text", r#"{"v":"1"}"#, "no version"),
            ("a newer OAIY's", r#"{"v":2}"#, "newer OAIY"),
        ] {
            put(text);
            let state = inspect_owner_file(&auth);
            assert!(
                state.exists(),
                "{what}: a file that is there is an owner file, not setup"
            );
            let line = state
                .refusal()
                .unwrap_or_else(|| panic!("{what}: {state:?}"));
            assert!(
                line.contains("owner.json") && line.contains(says),
                "{what}: {line}"
            );
        }
        // A folder where the file belongs cannot be read either.
        std::fs::remove_file(auth.join("owner.json")).unwrap();
        std::fs::create_dir(auth.join("owner.json")).unwrap();
        let state = inspect_owner_file(&auth);
        assert!(
            state.exists() && state.refusal().is_some_and(|l| l.contains("cannot read")),
            "{state:?}"
        );
        std::fs::remove_dir(auth.join("owner.json")).unwrap();
        // A file that is an owner: with the web login, the document it reads (the times and the password hash).
        put(
            r#"{"v":1,"created_ms":1,"password_changed_ms":1,"min_session_epoch":3,"password":"$argon2id$x"}"#,
        );
        assert_eq!(inspect_owner_file(&auth), OwnerState::Usable);
        // N6: the rule reads the shape and not the password: the hash above is one no password matches (the login
        // verifies nothing against it, so nobody signs in: it fails closed), and the file still satisfies rule 2.
        #[cfg(feature = "web")]
        assert!(!crate::auth::password::is_usable("$argon2id$x"));
        #[cfg(feature = "web")]
        {
            put(r#"{"v":1}"#);
            let state = inspect_owner_file(&auth);
            assert!(
                state.refusal().is_some_and(|l| l.contains("created_ms")),
                "the login cannot read it, and the start refuses what it cannot read: {state:?}"
            );
        }
        #[cfg(not(feature = "web"))]
        {
            put(r#"{"v":1}"#);
            assert_eq!(inspect_owner_file(&auth), OwnerState::Usable);
        }
    }

    #[test]
    fn check_of_a_build_without_the_web_login_lists_an_owner_file_it_cannot_use_first() {
        use crate::secret_file::testing::TempDir;
        let dir = TempDir::new("check-headless-owner");
        let auth = dir.0.join("auth");
        std::fs::create_dir_all(&auth).unwrap();
        let data = dir.0.display().to_string();
        let env = move |n: &str| match n {
            "OAIY_DATA_DIR" => Some(data.clone()),
            "OAIY_SERVER_TOKEN" => Some("short".to_string()),
            "OAIY_ACCESS_MODE" => Some("scoped".to_string()),
            _ => None,
        };
        for text in ["", "not json", r#"{"v":2}"#] {
            std::fs::write(auth.join("owner.json"), text).unwrap();
            let (mut out, mut err) = (Vec::new(), Vec::new());
            let code = check_headless(&env, &mut out, &mut err);
            let err = String::from_utf8(err).unwrap();
            assert_eq!(code, 78, "{text:?}: {err}");
            assert!(out.is_empty());
            // The file first, then the token: each on a line of its own, as the start would have said the first.
            let lines: Vec<&str> = err.lines().collect();
            assert_eq!(lines.len(), 2, "{text:?}: {err}");
            assert!(lines[0].contains("owner.json"), "{text:?}: {err}");
            assert!(lines[1].contains("OAIY_SERVER_TOKEN"), "{text:?}: {err}");
        }
        std::fs::write(
            auth.join("owner.json"),
            r#"{"v":1,"created_ms":1,"password_changed_ms":1,"password":"x"}"#,
        )
        .unwrap();
        let env = |n: &str| (n == "OAIY_DATA_DIR").then(|| dir.0.display().to_string());
        // (No mode is named, so the build's default, legacy, meets an owner login: that is refused, as it was.)
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = check_headless(&env, &mut out, &mut err);
        let err = String::from_utf8(err).unwrap();
        assert_eq!(code, 78, "{err}");
        assert!(
            err.contains("an owner login exists") && !err.contains("owner.json cannot"),
            "{err}"
        );
    }

    // ---- the rest -----------------------------------------------------------------------------------

    #[test]
    fn a_port_is_a_number_from_1_to_65535() {
        assert_eq!(parse_port(None), Ok(17972));
        assert_eq!(parse_port(Some(" 8080 ")), Ok(8080));
        assert_eq!(parse_port(Some("65535")), Ok(65535));
        assert_eq!(parse_port(Some("1")), Ok(1));
        for bad in [
            "0",
            "65536",
            "-1",
            "80a",
            "http",
            "1e3",
            "0x50",
            "8080 8081",
            // What `u16::from_str` takes and no one writes: a sign, a leading zero, digits of another script.
            "+8080",
            "08080",
            "00",
            "\u{ff18}\u{ff10}\u{ff18}\u{ff10}",
            "8_080",
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
    fn rule_3_the_scheme_of_a_public_url_is_read_in_any_case_and_the_origin_is_lowercase() {
        // RFC 3986: a scheme is case-insensitive. This one was refused as "not an https URL".
        for (text, origin) in [
            ("HTTPS://DASH.EXAMPLE.TEST", "https://dash.example.test"),
            (
                "Https://Dash.Example.Test:8443",
                "https://dash.example.test:8443",
            ),
            ("hTTps://dash.example.test", "https://dash.example.test"),
            ("  HTTPS://dash.example.test  ", "https://dash.example.test"),
        ] {
            let c = ok(&[("OAIY_PUBLIC_URL", text)], NO_OWNER);
            assert_eq!(c.public_origins[&App::Dash], origin, "{text:?}");
            assert_eq!(c.exposure, Exposure::Proxied);
        }
        // A scheme that is not https in any case is still refused, and a text that is too short to have one, or has
        // characters that are not a scheme's, does not panic.
        for bad in [
            "http://dash.example.test",
            "HTTP://dash.example.test",
            "httpss://dash.example.test",
            "https:/dash.example.test",
            "https//dash.example.test",
            "https",
            "https:/",
            "\u{ff48}ttps://dash.example.test",
            "\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}",
        ] {
            assert_eq!(
                broken(&[("OAIY_PUBLIC_URL", bad)], NO_OWNER),
                [Rule::PublicUrl],
                "{bad:?}"
            );
        }
        // The port is read as strictly as a `Host`'s.
        for bad in [
            "https://dash.example.test:+8443",
            "https://dash.example.test:08443",
            "https://dash.example.test:0",
            "https://dash.example.test:65536",
            "https://dash.example.test:",
        ] {
            assert_eq!(
                broken(&[("OAIY_PUBLIC_URL", bad)], NO_OWNER),
                [Rule::PublicUrl],
                "{bad:?}"
            );
        }
        // The desktop's reader, which does not judge, reads the scheme in the same way.
        assert!(
            strip_https_scheme("HTTPS://x") == Some("x")
                && strip_https_scheme("http://x").is_none()
        );
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
