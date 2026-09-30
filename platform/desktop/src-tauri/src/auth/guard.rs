//! Guard v2: the request pipeline of the design (3.5), keyed on `(Method, MatchedPath)`.
//!
//! One guard decides every request in one of two ways, by the access mode:
//!
//! - `scoped` and `shadow`: the new pipeline for every route.
//! - `legacy`: the new pipeline for the routes the access model adds (the rows of the table with
//!   `since: 2`), and today's `origin_guard`, unchanged, for every other request, including a request
//!   for a route that does not exist. See [`Guard::claims`]. Nothing about a route that existed before
//!   the model changes in this mode: not its status, not its body, not its headers.
//!
//! The pipeline, in order (an in-process request, one made by code in this process with no socket, has
//! no `Host`, no peer and no `Origin`, so the steps that read them are skipped for it):
//!
//! 1. a method other than `GET HEAD POST PUT PATCH DELETE OPTIONS`: `405`;
//! 2. `Host` not in the allow-list: `421 misdirected_host` (except `GET`/`HEAD /api/health`);
//! 3. the effective client address (trusted proxies);
//! 4. a forwarded header on a local install: `421 proxy_detected`; a bearer from a public address on a
//!    LAN listener: `403 plaintext_from_public_address`; the channel;
//! 5. `OPTIONS`: public (the CORS layer answers it);
//! 6. classify `(Method, MatchedPath)`; a route with no row is `403 unclassified_route`;
//! 7. the credential: `Authorization: Bearer`, strictly parsed (`400 bad_request`);
//! 8. the failed-bearer throttle (`429`), before any lookup;
//! 9. verify: the environment token (the `cli` preset), or a token of the grammar from the store; the
//!    kind rules (`desk`, `run` and `con` only from a direct loopback peer) and the origin binding;
//! 10. authorize by class and mode (`403 insufficient_scope`, `403 elevation_required`, ...);
//! 11. the principal goes on the request as an extension and the handler runs;
//! 12. `Cache-Control: no-store` and `X-Content-Type-Options: nosniff` on the answer.
//!
//! Credentials that are not credentials here, and fail closed: the process-wide internal token and a
//! legacy `oaiypat_<64 hex>` pairing token do not authenticate (`401 token_invalid`); the migration of the
//! latter into the store is a later step, and the former is deleted by another.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, MatchedPath, Request};
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};

use super::audit::AuditLog;
use super::bearer_throttle::BearerThrottle;
use super::clientip::{client_ip, ClientIp, TrustedProxies};
use super::clock::Clock;
use super::exposure_checks::{forwarded_header, is_loopback, is_public_address};
use super::host::{
    channel, expected_origin, Channel, HostClass, HostName, HostPolicy, ProxyMisconfigured,
};
use super::mode::{AccessMode, Exposure};
use super::presets::App;
use super::principal::{Principal, PrincipalKind};
use super::routes::{lookup, pattern_existed_before, route_class, Class, DeskRole, Verb};
use super::scopes::is_dangerous;
use super::store::{AuthError, AuthStore};
use super::token;

/// The methods the API serves.
fn is_served_method(m: &Method) -> bool {
    matches!(
        *m,
        Method::GET
            | Method::HEAD
            | Method::POST
            | Method::PUT
            | Method::PATCH
            | Method::DELETE
            | Method::OPTIONS
    )
}

/// Who a request came from at the socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Peer {
    Socket(SocketAddr),
    /// Made by code in this process with no socket (the control tools' calls through the router).
    InProcess,
}

impl Peer {
    pub fn ip(self) -> Option<IpAddr> {
        match self {
            Peer::Socket(a) => Some(a.ip()),
            Peer::InProcess => None,
        }
    }

    pub fn is_loopback(self) -> bool {
        self.ip().is_some_and(is_loopback)
    }
}

/// What the guard learned about a request, put on it for the handlers (`GET /api/auth/info` says it back).
#[derive(Clone, Debug)]
pub struct RequestInfo {
    pub peer: Peer,
    /// The effective client address, as text.
    pub client_ip: String,
    /// The throttle's key for it.
    pub client_key: String,
    pub via_trusted_proxy: bool,
    /// `https` when a trusted proxy said so, else `http`.
    pub proto: &'static str,
    /// The `Host` as the server understood it.
    pub host: String,
    pub host_class: Option<HostClass>,
    pub channel: Channel,
}

/// What `GET /api/health` adds to its answer (`access` and `storage`), put on every request by the guard.
#[derive(Clone, Copy, Debug)]
pub struct HealthExtras {
    pub access: &'static str,
    pub storage: &'static str,
}

// ---- the answer to a refused request ----------------------------------------------------------

/// A request refused: `{"error":{"code","message"}}` and, next to it, whatever the code adds.
#[derive(Clone, Debug)]
pub struct Denial {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
    pub extra: Vec<(&'static str, Value)>,
    pub headers: Vec<(&'static str, String)>,
    /// The noise event this refusal is counted as, if it is counted.
    pub noise: Option<&'static str>,
}

impl Denial {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Denial {
        Denial {
            status,
            code,
            message: message.into(),
            extra: Vec::new(),
            headers: Vec::new(),
            noise: None,
        }
    }

    fn with(mut self, key: &'static str, value: Value) -> Denial {
        self.extra.push((key, value));
        self
    }

    fn header(mut self, name: &'static str, value: String) -> Denial {
        self.headers.push((name, value));
        self
    }

    fn counted(mut self, event: &'static str) -> Denial {
        self.noise = Some(event);
        self
    }

    pub fn bad_request(message: impl Into<String>) -> Denial {
        Denial::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    pub fn invalid_request(message: impl Into<String>) -> Denial {
        Denial::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    pub fn auth_required() -> Denial {
        Denial::new(
            StatusCode::UNAUTHORIZED,
            "auth_required",
            "This route needs a credential.",
        )
        .header("www-authenticate", "Bearer realm=\"oaiy\"".into())
    }

    /// From a credential that was presented and was not accepted.
    pub fn from_auth_error(e: &AuthError) -> Denial {
        let status = StatusCode::UNAUTHORIZED;
        let message = match e {
            AuthError::Invalid => "The credential is not valid.",
            AuthError::Expired => "The credential has expired.",
            AuthError::Revoked { .. } => "The credential has been revoked.",
            AuthError::SessionEnded { .. } => "The session has ended.",
        };
        let mut d = Denial::new(status, e.code(), message).header(
            "www-authenticate",
            "Bearer realm=\"oaiy\", error=\"invalid_token\"".into(),
        );
        if let Some(reason) = e.reason() {
            d = d.with("reason", json!(reason));
        }
        d
    }

    pub fn insufficient_scope(required: &str) -> Denial {
        Denial::new(
            StatusCode::FORBIDDEN,
            "insufficient_scope",
            format!("This credential does not hold {required}."),
        )
        .with("required", json!(required))
        .header(
            "www-authenticate",
            format!("Bearer realm=\"oaiy\", error=\"insufficient_scope\", scope=\"{required}\""),
        )
    }

    pub fn elevation_required() -> Denial {
        Denial::new(
            StatusCode::FORBIDDEN,
            "elevation_required",
            "Confirm with your password first.",
        )
        .with("elevate", json!("/api/auth/elevate"))
    }

    pub fn origin_mismatch(message: impl Into<String>) -> Denial {
        Denial::new(StatusCode::FORBIDDEN, "origin_mismatch", message).counted("auth.denied")
    }

    pub fn unclassified_route() -> Denial {
        Denial::new(
            StatusCode::FORBIDDEN,
            "unclassified_route",
            "This route has no access rule, so it is refused.",
        )
        .counted("auth.denied")
    }

    pub fn misdirected_host() -> Denial {
        Denial::new(
            StatusCode::MISDIRECTED_REQUEST,
            "misdirected_host",
            "This server does not answer to that Host.",
        )
        .counted("auth.denied")
    }

    pub fn proxy_detected() -> Denial {
        Denial::new(StatusCode::MISDIRECTED_REQUEST, "proxy_detected", "A forwarded header reached an install that is not behind a proxy: set OAIY_PUBLIC_URL.").counted("auth.denied")
    }

    pub fn plaintext_from_public_address() -> Denial {
        Denial::new(
            StatusCode::FORBIDDEN,
            "plaintext_from_public_address",
            "A bearer token over plain HTTP from a public address is refused.",
        )
        .counted("auth.denied")
    }

    pub fn rate_limited(retry_after_s: u64) -> Denial {
        Denial::new(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limited",
            "Too many failed credentials from this address.",
        )
        .with("retryAfterSeconds", json!(retry_after_s))
        .header("retry-after", retry_after_s.to_string())
    }

    pub fn not_found() -> Denial {
        Denial::new(
            StatusCode::NOT_FOUND,
            "not_found",
            "There is no such route.",
        )
    }

    pub fn method_not_allowed() -> Denial {
        Denial::new(
            StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "That method is not served.",
        )
    }

    pub fn proxy_misconfigured() -> Denial {
        Denial::new(
            StatusCode::BAD_REQUEST,
            "proxy_misconfigured",
            "The proxy says http for an https host: fix its X-Forwarded-Proto.",
        )
        .counted("auth.denied")
    }

    pub fn into_response(self) -> Response {
        let body = {
            let mut o = serde_json::Map::new();
            o.insert(
                "error".into(),
                json!({ "code": self.code, "message": self.message }),
            );
            for (k, v) in self.extra {
                o.insert(k.into(), v);
            }
            Value::Object(o)
        };
        let mut response = (self.status, axum::Json(body)).into_response();
        let h = response.headers_mut();
        for (name, value) in self.headers {
            if let (Ok(n), Ok(v)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(&value),
            ) {
                h.insert(n, v);
            }
        }
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        h.insert(
            "x-content-type-options",
            HeaderValue::from_static("nosniff"),
        );
        response
    }
}

// ---- authorization: the class of the route against the principal ------------------------------

/// What the guard concludes for a route's class and a principal.
#[derive(Debug)]
pub enum Verdict {
    Allow,
    /// `shadow` mode let a non-dangerous scope mismatch through: the scope it lacked.
    ShadowAllow {
        scope: &'static str,
    },
    Deny(Denial),
}

fn desk_role(app: Option<App>) -> Option<DeskRole> {
    match app? {
        App::Dash => Some(DeskRole::Dashboard),
        App::Agent => Some(DeskRole::Agent),
        App::Flows => Some(DeskRole::Flows),
    }
}

/// Decide by the route's class (design 4.3). `principal` is `None` for an anonymous request.
pub fn authorize(class: &Class, principal: Option<&Principal>, mode: AccessMode) -> Verdict {
    match class {
        Class::Public => return Verdict::Allow,
        Class::Unclassified => return Verdict::Deny(Denial::unclassified_route()),
        _ => {}
    }
    let Some(p) = principal else {
        return Verdict::Deny(Denial::auth_required());
    };
    match class {
        Class::Public | Class::Unclassified => unreachable!("handled above"),
        Class::AnyCredential => Verdict::Allow,
        Class::Console => {
            if p.kind == PrincipalKind::Console {
                Verdict::Allow
            } else {
                Verdict::Deny(Denial::insufficient_scope("console").counted("auth.denied"))
            }
        }
        Class::Session { elevate } => {
            if p.kind != PrincipalKind::Session {
                Verdict::Deny(Denial::insufficient_scope("session").counted("auth.denied"))
            } else if *elevate && !p.elevated {
                Verdict::Deny(Denial::elevation_required())
            } else {
                Verdict::Allow
            }
        }
        Class::Desk { roles } => {
            if p.kind == PrincipalKind::Desk && desk_role(p.app).is_some_and(|r| roles.contains(&r))
            {
                Verdict::Allow
            } else {
                Verdict::Deny(Denial::insufficient_scope("desk").counted("auth.denied"))
            }
        }
        Class::Scope(scope) => {
            if p.has(scope) {
                // A dangerous scope on a cookie session needs a step-up; every other kind of credential
                // holds it only because it was made with one.
                if is_dangerous(scope) && p.kind == PrincipalKind::Session && !p.elevated {
                    Verdict::Deny(Denial::elevation_required())
                } else {
                    Verdict::Allow
                }
            } else if mode == AccessMode::Shadow && !is_dangerous(scope) {
                Verdict::ShadowAllow { scope }
            } else {
                Verdict::Deny(Denial::insufficient_scope(scope).counted("auth.denied"))
            }
        }
    }
}

// ---- configuration ----------------------------------------------------------------------------

/// What the guard needs to know about the install.
#[derive(Clone, Debug)]
pub struct GuardConfig {
    pub exposure: Exposure,
    /// The desktop (a window, no server UI) rather than `oaiy-server`.
    pub gui: bool,
    pub port: u16,
    pub trusted: TrustedProxies,
    pub hosts: HostPolicy,
    /// `OAIY_ALLOW_PUBLIC_PLAINTEXT=1`.
    pub allow_public_plaintext: bool,
}

/// `https://host[:port]` with no path, query or fragment, as a host.
fn https_origin_host(url: &str) -> Option<HostName> {
    let rest = url.trim().strip_prefix("https://")?;
    if rest.is_empty() || rest.contains(['/', '?', '#', '@']) {
        return None;
    }
    HostName::parse(rest)
}

impl GuardConfig {
    /// Read the settings of 4.13 from the environment (`env`), leniently: what cannot be read is
    /// ignored and named in the warnings (the strict startup rules are a later step).
    pub fn from_env(
        env: &dyn Fn(&str) -> Option<String>,
        bind_all: bool,
        gui: bool,
        port: u16,
    ) -> (GuardConfig, Vec<String>) {
        let mut warnings = Vec::new();
        let get = |name: &str| {
            env(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let public_url = get("OAIY_PUBLIC_URL");
        let exposure = Exposure::compute(bind_all, public_url.is_some());
        let mut public: BTreeMap<HostName, App> = BTreeMap::new();
        for (var, app) in [
            ("OAIY_PUBLIC_URL", App::Dash),
            ("OAIY_AGENT_URL", App::Agent),
            ("OAIY_FLOWS_URL", App::Flows),
        ] {
            if let Some(url) = get(var) {
                match https_origin_host(&url) {
                    Some(h) if !h.is_loopback_name() => {
                        public.insert(h, app);
                    }
                    _ => warnings.push(format!("{var}={url:?} is not https://<host>[:port] with no path (or has a loopback host): ignored")),
                }
            }
        }
        let trusted = match get("OAIY_TRUSTED_PROXIES") {
            Some(list) => {
                let (t, rejected) = TrustedProxies::parse_list(&list);
                for r in rejected {
                    warnings.push(format!(
                        "OAIY_TRUSTED_PROXIES: {r:?} is not an address or a network: ignored"
                    ));
                }
                t
            }
            None if public_url.is_some() => TrustedProxies::loopback(),
            None => TrustedProxies::none(),
        };
        let mut extra = BTreeSet::new();
        if let Some(list) = get("OAIY_ALLOWED_HOSTS") {
            for entry in list.split(',').map(str::trim).filter(|e| !e.is_empty()) {
                match HostName::parse(entry) {
                    Some(h) => {
                        extra.insert(h);
                    }
                    None => warnings.push(format!(
                        "OAIY_ALLOWED_HOSTS: {entry:?} is not host[:port]: ignored"
                    )),
                }
            }
        }
        let loopback_apps = !gui && exposure == Exposure::Local;
        let hosts = HostPolicy::new(exposure, port, extra, public, loopback_apps);
        let allow_public_plaintext = get("OAIY_ALLOW_PUBLIC_PLAINTEXT").as_deref() == Some("1");
        (
            GuardConfig {
                exposure,
                gui,
                port,
                trusted,
                hosts,
                allow_public_plaintext,
            },
            warnings,
        )
    }
}

// ---- the guard --------------------------------------------------------------------------------

/// The guard.
pub struct Guard {
    mode: AccessMode,
    config: GuardConfig,
    store: Arc<AuthStore>,
    static_token: Option<String>,
    throttle: BearerThrottle,
    audit: Option<Arc<AuditLog>>,
}

/// A request refused, and where it came from (the address and host it is counted against).
struct Refusal {
    denial: Denial,
    ip: String,
    host: String,
}

/// What `admit` learned, for the handlers.
struct Admitted {
    principal: Option<Principal>,
    info: Option<RequestInfo>,
}

impl Guard {
    pub fn new(
        mode: AccessMode,
        config: GuardConfig,
        store: Arc<AuthStore>,
        static_token: Option<String>,
        audit: Option<Arc<AuditLog>>,
        clock: Arc<dyn Clock>,
    ) -> Guard {
        let static_token = static_token
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());
        store.set_static_present(static_token.is_some());
        Guard {
            mode,
            config,
            store,
            static_token,
            throttle: BearerThrottle::new(clock),
            audit,
        }
    }

    pub fn mode(&self) -> AccessMode {
        self.mode
    }

    pub fn config(&self) -> &GuardConfig {
        &self.config
    }

    pub fn store(&self) -> &Arc<AuthStore> {
        &self.store
    }

    pub fn audit(&self) -> Option<&Arc<AuditLog>> {
        self.audit.as_ref()
    }

    pub fn throttle(&self) -> &BearerThrottle {
        &self.throttle
    }

    pub fn health_extras(&self) -> HealthExtras {
        HealthExtras {
            access: self.mode.name(),
            storage: self.store.storage().name(),
        }
    }

    /// Whether this guard decides the request (see the module documentation): always in `scoped` and
    /// `shadow`; in `legacy` only for a route the access model adds. A request for a route that existed
    /// before (any method of it: a method the table adds to an old route is the old guard's too), and a
    /// request for a route that does not exist, are the old guard's.
    pub fn claims(&self, req: &Request) -> bool {
        if self.mode.is_enforcing() {
            return true;
        }
        let Some(matched) = req.extensions().get::<MatchedPath>() else {
            return false;
        };
        let Some(verb) = Verb::of_method(req.method()) else {
            return false;
        };
        lookup(verb, matched.as_str())
            .is_some_and(|row| row.since == 2 && !pattern_existed_before(matched.as_str()))
    }

    fn note(&self, event: &str, ip: &str, host: &str) {
        if let Some(audit) = &self.audit {
            audit.noise(event, ip, host);
        }
    }

    /// Decide `req`: refuse it, or put the principal and the request's facts on it and run the handler.
    pub async fn handle(&self, mut req: Request, next: Next) -> Response {
        let path = req.uri().path().to_owned();
        match self.admit(&mut req) {
            Ok(admitted) => {
                if let Some(info) = admitted.info {
                    req.extensions_mut().insert(info);
                }
                if let Some(p) = admitted.principal {
                    req.extensions_mut().insert(p);
                }
                let under_api = path.starts_with("/api/");
                let mut response = next.run(req).await;
                if under_api {
                    let h = response.headers_mut();
                    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
                    h.insert(
                        "x-content-type-options",
                        HeaderValue::from_static("nosniff"),
                    );
                }
                response
            }
            Err(refusal) => {
                if let Some(event) = refusal.denial.noise {
                    self.note(event, &refusal.ip, &refusal.host);
                }
                refusal.denial.into_response()
            }
        }
    }

    /// Steps 1 to 10. On a refusal, the client address and host to count it against.
    fn admit(&self, req: &mut Request) -> Result<Admitted, Box<Refusal>> {
        let peer = req
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map_or(Peer::InProcess, |c| Peer::Socket(c.0));
        let method = req.method().clone();
        let path = req.uri().path().to_owned();
        let matched = req
            .extensions()
            .get::<MatchedPath>()
            .map(|m| m.as_str().to_owned());
        let host_text = req
            .headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
            .or_else(|| req.uri().authority().map(|a| a.to_string()));
        let fail = |d: Denial, ip: &str| -> Box<Refusal> {
            Box::new(Refusal {
                denial: d,
                ip: ip.to_string(),
                host: host_text.clone().unwrap_or_default(),
            })
        };
        let peer_text = peer
            .ip()
            .map(|i| i.to_string())
            .unwrap_or_else(|| "in-process".into());

        // 1. methods
        if !is_served_method(&method) {
            return Err(fail(Denial::method_not_allowed(), &peer_text));
        }
        // A path outside the API that no route answers is not under the guard (there is nothing to guard).
        if matched.is_none() && !path.starts_with("/api/") && path != "/api" {
            return Ok(Admitted {
                principal: None,
                info: None,
            });
        }
        let headers = req.headers().clone();
        let forwarded = forwarded_header(&headers);
        let direct_loopback = peer.is_loopback() && forwarded.is_none();
        let is_health_probe = matches!(method, Method::GET | Method::HEAD) && path == "/api/health";

        // 2-4. the address side of the request (skipped for a request with no socket)
        let mut info = RequestInfo {
            peer,
            client_ip: peer_text.clone(),
            client_key: peer_text.clone(),
            via_trusted_proxy: false,
            proto: "http",
            host: host_text.clone().unwrap_or_default(),
            host_class: None,
            channel: Channel::Insecure,
        };
        if let Some(peer_ip) = peer.ip() {
            let host = host_text.as_deref().and_then(HostName::parse);
            match (&host, is_health_probe) {
                (Some(h), _) => match self.config.hosts.classify(h, direct_loopback) {
                    Ok(class) => info.host_class = Some(class),
                    Err(_) if is_health_probe => {}
                    Err(_) => return Err(fail(Denial::misdirected_host(), &peer_text)),
                },
                (None, true) => {}
                (None, false) => return Err(fail(Denial::misdirected_host(), &peer_text)),
            }
            let xff: Vec<&str> = headers
                .get_all("x-forwarded-for")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .collect();
            let client: ClientIp = client_ip(peer_ip, &xff, &self.config.trusted);
            info.client_ip = client.ip.to_string();
            info.client_key = client.key.clone();
            info.via_trusted_proxy = client.via_proxy || self.config.trusted.contains(peer_ip);
            if self.config.exposure == Exposure::Local && forwarded.is_some() {
                return Err(fail(Denial::proxy_detected(), &info.client_ip));
            }
            if self.config.exposure == Exposure::Lan
                && headers.contains_key(header::AUTHORIZATION)
                && is_public_address(peer_ip)
                && !self.config.allow_public_plaintext
            {
                return Err(fail(
                    Denial::plaintext_from_public_address(),
                    &info.client_ip,
                ));
            }
            let xfp = headers
                .get("x-forwarded-proto")
                .and_then(|v| v.to_str().ok());
            if let Some(h) = &host {
                match channel(
                    &self.config.hosts,
                    h,
                    self.config.trusted.contains(peer_ip),
                    xfp,
                    peer.is_loopback(),
                    forwarded.is_some(),
                ) {
                    Ok(c) => info.channel = c,
                    Err(ProxyMisconfigured) => {
                        return Err(fail(Denial::proxy_misconfigured(), &info.client_ip))
                    }
                }
            }
            if self.config.trusted.contains(peer_ip)
                && xfp
                    .map(|p| p.trim().eq_ignore_ascii_case("https"))
                    .unwrap_or(false)
            {
                info.proto = "https";
            }
        }
        let (ip_for_noise, host_for_noise) = (info.client_ip.clone(), info.host.clone());
        let fail = move |d: Denial| -> Box<Refusal> {
            Box::new(Refusal {
                denial: d,
                ip: ip_for_noise.clone(),
                host: host_for_noise.clone(),
            })
        };

        // 5. OPTIONS is public: the CORS layer answers preflights, and a bare one reaches the router.
        if method == Method::OPTIONS {
            return Ok(Admitted {
                principal: None,
                info: Some(info),
            });
        }

        // 6. classify by the route axum matched
        let class = match &matched {
            Some(m) => route_class(&method, m),
            None => Class::Unclassified,
        };
        let no_route = matched.is_none();

        // 7-9. the credential, if there is one
        let principal = if class == Class::Public {
            None
        } else {
            self.authenticate(
                &headers,
                &info,
                direct_loopback,
                self.throttle_applies(peer),
            )
            .map_err(&fail)?
        };

        // A route that is not there: 404 for someone who is who they say they are, 401 for a stranger.
        if no_route {
            return Err(fail(if principal.is_some() {
                Denial::not_found()
            } else {
                Denial::auth_required()
            }));
        }

        // 10. authorize
        match authorize(&class, principal.as_ref(), self.mode) {
            Verdict::Allow => Ok(Admitted {
                principal,
                info: Some(info),
            }),
            Verdict::ShadowAllow { scope } => {
                self.note("auth.shadow_denied", &info.client_ip, &info.host);
                log::warn!("auth: shadow mode allowed {method} {path} without the scope {scope}");
                Ok(Admitted {
                    principal,
                    info: Some(info),
                })
            }
            Verdict::Deny(d) => Err(fail(d)),
        }
    }

    /// The failed-bearer throttle does not apply to loopback peers of a local install: a desktop's own
    /// windows and a local script cannot be blocked by it.
    fn throttle_applies(&self, peer: Peer) -> bool {
        !(self.config.exposure == Exposure::Local && peer.is_loopback()) && peer != Peer::InProcess
    }

    /// Steps 7 to 9: `Ok(None)` for an anonymous request.
    fn authenticate(
        &self,
        headers: &HeaderMap,
        info: &RequestInfo,
        direct_loopback: bool,
        throttled: bool,
    ) -> Result<Option<Principal>, Denial> {
        let values: Vec<&[u8]> = headers
            .get_all(header::AUTHORIZATION)
            .iter()
            .map(|v| v.as_bytes())
            .collect();
        let presented = match token::bearer_from_headers(&values) {
            Ok(None) => return Ok(None),
            Ok(Some(t)) => t,
            Err(e) => return Err(Denial::bad_request(e.message())),
        };
        // 8. a blocked address is refused before any lookup
        if throttled {
            if let Some(retry) = self.throttle.blocked_for(&info.client_key) {
                self.note("bearer.blocked", &info.client_ip, &info.host);
                return Err(Denial::rate_limited(retry));
            }
        }
        // 9. the environment token, then the store
        let principal = if self
            .static_token
            .as_deref()
            .is_some_and(|want| token::secrets_equal(want.as_bytes(), presented.as_bytes()))
        {
            Principal::static_token()
        } else {
            match self.store.authenticate(presented, Some(&info.client_ip)) {
                Ok(p) => p,
                Err(e) => {
                    if e == AuthError::Invalid && throttled {
                        self.throttle.record_failure(&info.client_key);
                        self.note("bearer.failed", &info.client_ip, &info.host);
                    }
                    return Err(Denial::from_auth_error(&e));
                }
            }
        };
        self.check_binding(&principal, headers, info, direct_loopback)?;
        Ok(Some(principal))
    }

    /// The kind rules and the origin binding (design 4.5.3).
    fn check_binding(
        &self,
        p: &Principal,
        headers: &HeaderMap,
        info: &RequestInfo,
        direct_loopback: bool,
    ) -> Result<(), Denial> {
        // `desk`, `run` and `con` credentials are worth something only on the machine they were made on.
        if p.kind.needs_direct_loopback() && !direct_loopback {
            return Err(Denial::origin_mismatch("This credential works only from the machine it was made on, with no proxy in between."));
        }
        let origin = headers.get(header::ORIGIN).map(|v| v.to_str().ok());
        if p.is_bound() {
            match origin {
                Some(Some(o)) => {
                    if !p.origins.iter().any(|b| b == o) {
                        return Err(Denial::origin_mismatch(
                            "This credential is bound to another origin.",
                        ));
                    }
                }
                // An Origin that is not text is not one of ours.
                Some(None) => {
                    return Err(Denial::origin_mismatch(
                        "This credential is bound to another origin.",
                    ))
                }
                // No Origin: only a same-origin GET carries none, and then it must be for the bound origin.
                None => {
                    let same_origin = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok())
                        == Some("same-origin");
                    let host = HostName::parse(&info.host);
                    let expected = host.map(|h| expected_origin(&h, info.channel));
                    if !(same_origin && expected.is_some_and(|e| p.origins.contains(&e))) {
                        return Err(Denial::origin_mismatch(
                            "This credential is bound to an origin, and the request names none.",
                        ));
                    }
                }
            }
        } else if origin.is_some() {
            // A token that leaked into a web page is useless from that page.
            return Err(Denial::origin_mismatch(
                "This credential belongs to a native client: a browser cannot use it.",
            ));
        }
        Ok(())
    }
}

/// A `Router` layer: `axum::middleware::from_fn_with_state(guard, scoped_guard)` makes the new guard the
/// only one (`scoped` and `shadow` modes).
pub async fn scoped_guard(
    axum::extract::State(guard): axum::extract::State<Arc<Guard>>,
    req: Request,
    next: Next,
) -> Response {
    guard.handle(req, next).await
}

/// CORS and Private Network Access for the `scoped` and `shadow` modes (`auth/cors.rs`), outside the guard so
/// that a paired app can read the refusal it gets. A preflight is answered here, without a credential, `204`;
/// a route with no row, and an origin no live credential is bound to, get no headers.
pub async fn scoped_cors(
    axum::extract::State(guard): axum::extract::State<Arc<Guard>>,
    req: Request,
    next: Next,
) -> Response {
    use super::cors;
    let method = req.method().clone();
    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|o| o.to_str().ok())
        .map(str::to_owned);
    let matched = req
        .extensions()
        .get::<MatchedPath>()
        .map(|m| m.as_str().to_owned());
    let asks_pna = cors::asks_private_network(req.headers());
    // The live credentials' origins are only looked up for a request that has an Origin.
    let allowed = |origin: &Option<String>| {
        if origin.is_some() {
            guard.store().allowed_origins()
        } else {
            Default::default()
        }
    };
    if cors::is_preflight(&method, req.headers()) {
        let decision = match cors::preflight_method(req.headers()) {
            Some(asked) => cors::decide(
                &asked,
                matched.as_deref(),
                origin.as_deref(),
                &allowed(&origin),
                asks_pna,
            ),
            None => cors::Cors::default(),
        };
        let mut response = StatusCode::NO_CONTENT.into_response();
        decision.apply(response.headers_mut());
        return response;
    }
    let decision = cors::decide(
        &method,
        matched.as_deref(),
        origin.as_deref(),
        &allowed(&origin),
        asks_pna,
    );
    let mut response = next.run(req).await;
    decision.apply(response.headers_mut());
    response
}

#[cfg(test)]
mod pure_tests {
    use super::*;
    use crate::auth::presets::Preset;
    use crate::auth::scopes::ScopeSet;

    fn principal(kind: PrincipalKind, scopes: ScopeSet) -> Principal {
        Principal {
            id: "0123456789abcdef".into(),
            kind,
            label: "t".into(),
            scopes,
            origins: vec![],
            app: None,
            elevated: false,
            chain: vec![],
            persisted: false,
            expires_ms: None,
            preset: None,
            legacy_import: false,
        }
    }

    fn denied(v: Verdict) -> Denial {
        match v {
            Verdict::Deny(d) => d,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_public_route_needs_nothing() {
        for mode in [AccessMode::Legacy, AccessMode::Scoped, AccessMode::Shadow] {
            assert!(matches!(
                authorize(&Class::Public, None, mode),
                Verdict::Allow
            ));
        }
    }

    #[test]
    fn a_route_with_no_row_is_refused_for_everyone_in_every_mode() {
        // Fail closed: even the owner's desk credential.
        let p = principal(PrincipalKind::Desk, ScopeSet::all());
        for mode in [AccessMode::Legacy, AccessMode::Scoped, AccessMode::Shadow] {
            for who in [None, Some(&p)] {
                let d = denied(authorize(&Class::Unclassified, who, mode));
                assert_eq!(
                    (d.status, d.code),
                    (StatusCode::FORBIDDEN, "unclassified_route")
                );
            }
        }
    }

    #[test]
    fn an_anonymous_request_to_a_route_that_needs_a_credential_is_401_auth_required() {
        for class in [
            Class::AnyCredential,
            Class::Console,
            Class::Session { elevate: false },
            Class::Desk {
                roles: &[DeskRole::Dashboard],
            },
            Class::Scope("system.read"),
            Class::Scope("plugins.install"),
        ] {
            for mode in [AccessMode::Legacy, AccessMode::Scoped, AccessMode::Shadow] {
                let d = denied(authorize(&class, None, mode));
                assert_eq!(
                    (d.status, d.code),
                    (StatusCode::UNAUTHORIZED, "auth_required"),
                    "{class:?} {mode:?}"
                );
                assert!(d
                    .headers
                    .iter()
                    .any(|(n, v)| *n == "www-authenticate" && v == "Bearer realm=\"oaiy\""));
            }
        }
    }

    #[test]
    fn a_scope_is_held_exactly_or_refused_with_the_scope_named() {
        let p = principal(PrincipalKind::Pat, ScopeSet::of(&["ai.read", "ai.use"]));
        assert!(matches!(
            authorize(&Class::Scope("ai.read"), Some(&p), AccessMode::Scoped),
            Verdict::Allow
        ));
        let d = denied(authorize(
            &Class::Scope("flows.write"),
            Some(&p),
            AccessMode::Scoped,
        ));
        assert_eq!(
            (d.status, d.code),
            (StatusCode::FORBIDDEN, "insufficient_scope")
        );
        assert!(d
            .extra
            .iter()
            .any(|(k, v)| *k == "required" && v == "flows.write"));
        assert!(d.headers.iter().any(|(n, v)| *n == "www-authenticate"
            && v == "Bearer realm=\"oaiy\", error=\"insufficient_scope\", scope=\"flows.write\""));
        // A scope is not a prefix: `ai` and `ai.*` are not `ai.read`.
        let narrow = principal(PrincipalKind::Pat, ScopeSet::of(&["ai.use"]));
        assert!(
            denied(authorize(
                &Class::Scope("ai.read"),
                Some(&narrow),
                AccessMode::Scoped
            ))
            .code
                == "insufficient_scope"
        );
    }

    #[test]
    fn shadow_lets_only_a_non_dangerous_scope_mismatch_through() {
        let p = principal(PrincipalKind::Pat, ScopeSet::of(&["system.read"]));
        // Allowed and logged: a scope that is not dangerous.
        assert!(matches!(
            authorize(&Class::Scope("flows.write"), Some(&p), AccessMode::Shadow),
            Verdict::ShadowAllow {
                scope: "flows.write"
            }
        ));
        // Refused exactly as in scoped: a dangerous scope.
        for dangerous in [
            "plugins.install",
            "services.define",
            "flows.approve",
            "auth.manage",
            "control.admin",
        ] {
            let d = denied(authorize(
                &Class::Scope(dangerous),
                Some(&p),
                AccessMode::Shadow,
            ));
            assert_eq!(d.code, "insufficient_scope", "{dangerous}");
        }
        // And scoped refuses both.
        assert_eq!(
            denied(authorize(
                &Class::Scope("flows.write"),
                Some(&p),
                AccessMode::Scoped
            ))
            .code,
            "insufficient_scope"
        );
        // Shadow never touches authentication: no credential is still 401.
        assert_eq!(
            denied(authorize(
                &Class::Scope("flows.write"),
                None,
                AccessMode::Shadow
            ))
            .code,
            "auth_required"
        );
        // A credential that holds the scope is simply allowed and not logged as a mismatch.
        let holds = principal(PrincipalKind::Pat, ScopeSet::of(&["flows.write"]));
        assert!(matches!(
            authorize(
                &Class::Scope("flows.write"),
                Some(&holds),
                AccessMode::Shadow
            ),
            Verdict::Allow
        ));
    }

    #[test]
    fn a_dangerous_scope_on_a_session_needs_a_step_up_and_on_nothing_else_does() {
        let mut session = principal(PrincipalKind::Session, Preset::Owner.scopes());
        let d = denied(authorize(
            &Class::Scope("plugins.install"),
            Some(&session),
            AccessMode::Scoped,
        ));
        assert_eq!(
            (d.status, d.code),
            (StatusCode::FORBIDDEN, "elevation_required")
        );
        assert!(d
            .extra
            .iter()
            .any(|(k, v)| *k == "elevate" && v == "/api/auth/elevate"));
        // Also in shadow: elevation is enforced exactly as in scoped.
        assert_eq!(
            denied(authorize(
                &Class::Scope("plugins.install"),
                Some(&session),
                AccessMode::Shadow
            ))
            .code,
            "elevation_required"
        );
        session.elevated = true;
        assert!(matches!(
            authorize(
                &Class::Scope("plugins.install"),
                Some(&session),
                AccessMode::Scoped
            ),
            Verdict::Allow
        ));
        // A non-dangerous scope needs none.
        session.elevated = false;
        assert!(matches!(
            authorize(
                &Class::Scope("flows.write"),
                Some(&session),
                AccessMode::Scoped
            ),
            Verdict::Allow
        ));
        // A desk credential and a native token that hold a dangerous scope were made with one.
        let desk = principal(PrincipalKind::Desk, Preset::Owner.scopes());
        assert!(matches!(
            authorize(
                &Class::Scope("plugins.install"),
                Some(&desk),
                AccessMode::Scoped
            ),
            Verdict::Allow
        ));
        let native = principal(PrincipalKind::Pat, ScopeSet::of(&["plugins.install"]));
        assert!(matches!(
            authorize(
                &Class::Scope("plugins.install"),
                Some(&native),
                AccessMode::Scoped
            ),
            Verdict::Allow
        ));
        // A session that lacks the scope is told so, not asked for a password.
        let agent_session = principal(PrincipalKind::Session, Preset::Agent.scopes());
        assert_eq!(
            denied(authorize(
                &Class::Scope("plugins.install"),
                Some(&agent_session),
                AccessMode::Scoped
            ))
            .code,
            "insufficient_scope"
        );
    }

    #[test]
    fn the_console_desk_and_session_classes_take_only_their_own_kind() {
        let console = principal(PrincipalKind::Console, ScopeSet::all());
        let pat = principal(PrincipalKind::Pat, ScopeSet::all());
        let session = principal(PrincipalKind::Session, ScopeSet::all());
        let mut desk = principal(PrincipalKind::Desk, ScopeSet::all());
        desk.app = Some(App::Agent);
        assert!(matches!(
            authorize(&Class::Console, Some(&console), AccessMode::Scoped),
            Verdict::Allow
        ));
        for other in [&pat, &session, &desk] {
            assert_eq!(
                denied(authorize(&Class::Console, Some(other), AccessMode::Scoped)).code,
                "insufficient_scope"
            );
        }
        assert!(matches!(
            authorize(
                &Class::Session { elevate: false },
                Some(&session),
                AccessMode::Scoped
            ),
            Verdict::Allow
        ));
        for other in [&pat, &console, &desk] {
            assert_eq!(
                denied(authorize(
                    &Class::Session { elevate: false },
                    Some(other),
                    AccessMode::Scoped
                ))
                .code,
                "insufficient_scope"
            );
        }
        assert_eq!(
            denied(authorize(
                &Class::Session { elevate: true },
                Some(&session),
                AccessMode::Scoped
            ))
            .code,
            "elevation_required"
        );
        let dashboard_only = Class::Desk {
            roles: &[DeskRole::Dashboard],
        };
        let any_desk = Class::Desk {
            roles: &[DeskRole::Dashboard, DeskRole::Agent, DeskRole::Flows],
        };
        assert!(matches!(
            authorize(&any_desk, Some(&desk), AccessMode::Scoped),
            Verdict::Allow
        ));
        assert_eq!(
            denied(authorize(&dashboard_only, Some(&desk), AccessMode::Scoped)).code,
            "insufficient_scope",
            "the Agent's desk is not the dashboard's"
        );
        assert_eq!(
            denied(authorize(&any_desk, Some(&pat), AccessMode::Scoped)).code,
            "insufficient_scope"
        );
        // Any credential means any.
        for p in [&pat, &console, &session, &desk] {
            assert!(matches!(
                authorize(&Class::AnyCredential, Some(p), AccessMode::Scoped),
                Verdict::Allow
            ));
        }
    }

    #[test]
    fn every_denial_has_the_status_and_code_of_the_design() {
        let table: Vec<(Denial, u16, &str)> = vec![
            (Denial::bad_request("x"), 400, "bad_request"),
            (Denial::invalid_request("x"), 400, "invalid_request"),
            (Denial::auth_required(), 401, "auth_required"),
            (
                Denial::from_auth_error(&AuthError::Invalid),
                401,
                "token_invalid",
            ),
            (
                Denial::from_auth_error(&AuthError::Expired),
                401,
                "token_expired",
            ),
            (
                Denial::from_auth_error(&AuthError::Revoked { reason: "revoked" }),
                401,
                "token_revoked",
            ),
            (
                Denial::from_auth_error(&AuthError::SessionEnded { reason: "idle" }),
                401,
                "session_expired",
            ),
            (Denial::insufficient_scope("x.y"), 403, "insufficient_scope"),
            (Denial::elevation_required(), 403, "elevation_required"),
            (Denial::origin_mismatch("x"), 403, "origin_mismatch"),
            (Denial::unclassified_route(), 403, "unclassified_route"),
            (
                Denial::plaintext_from_public_address(),
                403,
                "plaintext_from_public_address",
            ),
            (Denial::not_found(), 404, "not_found"),
            (Denial::method_not_allowed(), 405, "method_not_allowed"),
            (Denial::misdirected_host(), 421, "misdirected_host"),
            (Denial::proxy_detected(), 421, "proxy_detected"),
            (Denial::proxy_misconfigured(), 400, "proxy_misconfigured"),
            (Denial::rate_limited(7), 429, "rate_limited"),
        ];
        for (d, status, code) in table {
            assert_eq!((d.status.as_u16(), d.code), (status, code));
        }
    }

    #[tokio::test]
    async fn a_denial_is_the_error_body_with_the_extras_beside_it_and_never_cached() {
        let response = Denial::rate_limited(42).into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers().get("retry-after").unwrap(), "42");
        assert_eq!(response.headers().get("cache-control").unwrap(), "no-store");
        assert_eq!(
            response.headers().get("x-content-type-options").unwrap(),
            "nosniff"
        );
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "rate_limited");
        assert!(v["error"]["message"].as_str().unwrap().len() > 5);
        assert_eq!(v["retryAfterSeconds"], 42);
        // A token-invalid answer carries its reason and the challenge.
        let r = Denial::from_auth_error(&AuthError::SessionEnded {
            reason: "parent_ended",
        })
        .into_response();
        assert_eq!(
            r.headers().get("www-authenticate").unwrap(),
            "Bearer realm=\"oaiy\", error=\"invalid_token\""
        );
        let v: Value =
            serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 4096).await.unwrap())
                .unwrap();
        assert_eq!(
            (v["error"]["code"].as_str(), v["reason"].as_str()),
            (Some("session_expired"), Some("parent_ended"))
        );
    }

    #[test]
    fn a_denial_names_no_secret() {
        for d in [
            Denial::from_auth_error(&AuthError::Invalid),
            Denial::auth_required(),
            Denial::insufficient_scope("x.y"),
            Denial::origin_mismatch("x"),
        ] {
            assert!(
                !d.message.contains("oaiy") || d.message.contains("OAIY"),
                "{}",
                d.message
            );
        }
    }

    fn cfg(vars: &[(&str, &str)], bind_all: bool, gui: bool) -> (GuardConfig, Vec<String>) {
        let map: BTreeMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        GuardConfig::from_env(&move |name| map.get(name).cloned(), bind_all, gui, 17972)
    }

    #[test]
    fn the_settings_are_read_from_the_environment_of_4_13() {
        let (c, w) = cfg(&[], false, false);
        assert!(w.is_empty());
        assert_eq!(c.exposure, Exposure::Local);
        assert!(
            c.trusted.is_empty(),
            "no proxy is trusted by default on a local install"
        );
        assert!(!c.allow_public_plaintext);
        let (c, w) = cfg(
            &[
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_AGENT_URL", "https://agent.example.com:8443"),
                ("OAIY_FLOWS_URL", "https://flows.example.com"),
            ],
            false,
            false,
        );
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(c.exposure, Exposure::Proxied);
        assert!(
            c.trusted.contains("127.0.0.1".parse().unwrap()),
            "a proxied install trusts a proxy on loopback by default"
        );
        assert!(!c.trusted.contains("10.0.0.1".parse().unwrap()));
        let (c, _) = cfg(
            &[
                ("OAIY_PUBLIC_URL", "https://dash.example.com"),
                ("OAIY_TRUSTED_PROXIES", "10.0.0.0/8"),
            ],
            true,
            false,
        );
        assert!(
            c.trusted.contains("10.1.1.1".parse().unwrap())
                && !c.trusted.contains("127.0.0.1".parse().unwrap()),
            "the setting replaces the default"
        );
        let (c, _) = cfg(&[], true, false);
        assert_eq!(c.exposure, Exposure::Lan);
        let (c, _) = cfg(&[("OAIY_ALLOW_PUBLIC_PLAINTEXT", "1")], true, false);
        assert!(c.allow_public_plaintext);
        let (c, _) = cfg(&[("OAIY_ALLOW_PUBLIC_PLAINTEXT", "true")], true, false);
        assert!(!c.allow_public_plaintext, "only exactly 1");
    }

    #[test]
    fn a_setting_that_cannot_be_read_is_ignored_and_named() {
        let (c, w) = cfg(
            &[
                ("OAIY_PUBLIC_URL", "http://dash.example.com"),
                ("OAIY_AGENT_URL", "https://agent.example.com/path"),
                ("OAIY_FLOWS_URL", "https://localhost"),
                ("OAIY_TRUSTED_PROXIES", "10.0.0.0/8, nonsense"),
                ("OAIY_ALLOWED_HOSTS", "ok.example:9000, bad host"),
            ],
            false,
            false,
        );
        assert_eq!(w.len(), 5, "{w:?}");
        for named in [
            "OAIY_PUBLIC_URL",
            "OAIY_AGENT_URL",
            "OAIY_FLOWS_URL",
            "nonsense",
            "bad host",
        ] {
            assert!(w.iter().any(|x| x.contains(named)), "{named} not in {w:?}");
        }
        // A public URL that was set but unreadable still makes the install proxied (fail toward the stricter rules).
        assert_eq!(c.exposure, Exposure::Proxied);
    }
}
