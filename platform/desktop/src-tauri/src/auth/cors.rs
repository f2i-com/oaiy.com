//! CORS and Private Network Access (design 4.5.1), for the `scoped` and `shadow` modes.
//!
//! It replaces `CorsLayer::allow_origin(Any)` and the unconditional Private Network Access answer of the
//! `legacy` mode. Requests with no `Origin` header (native clients, same-origin GETs) get no CORS headers.
//! For a request or preflight with an `Origin`, by the route's class (a preflight is classified from the
//! route it names and its `Access-Control-Request-Method`):
//!
//! | Case | Response headers |
//! |---|---|
//! | Route class `Public` | `Access-Control-Allow-Origin: *`, methods `GET, HEAD, POST, OPTIONS`, headers `content-type`, `Max-Age: 600`. Never `Allow-Credentials`. Except the six routes that are same-origin by construction, which never get CORS headers |
//! | `Origin` is the origin of a live credential | that origin, `Vary: Origin`, every method, the headers `Authorization` and the OAIY ones (listed by name: a wildcard does not cover `Authorization`), `Expose-Headers`, `Max-Age: 600` |
//! | Anything else, `Origin: null` included | none: a preflight is answered `204` without them, so the browser blocks the request |
//! | PNA | only when the request asks (`Access-Control-Request-Private-Network: true`) and one of the rows above applied |
//!
//! `null` is never a member of the allowed set, so an `Origin: null` request never matches.

use std::collections::BTreeSet;

use axum::http::{HeaderMap, HeaderName, HeaderValue, Method};

use super::routes::{route_class, Class};

/// The routes that never get CORS headers: the login, setup, link and callback routes and the two GETs a
/// page reads about itself. Nobody else may read or drive them from a page.
pub const NEVER_CORS: [(&str, &str); 6] = [
    ("GET", "/api/auth/info"),
    ("GET", "/api/auth/session"),
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/setup"),
    ("POST", "/api/auth/link"),
    ("POST", "/api/auth/callback"),
];

const ALLOWED_HEADERS: &str = "authorization, content-type, x-oaiy-session, x-oaiy-csrf, x-oaiy-approve, idempotency-key, if-none-match";
const ALLOWED_METHODS: &str = "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS";
const EXPOSED_HEADERS: &str = "etag, retry-after, www-authenticate";
const PUBLIC_METHODS: &str = "GET, HEAD, POST, OPTIONS";
const MAX_AGE: &str = "600";

/// What the CORS layer adds to a response.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cors {
    pub headers: Vec<(&'static str, String)>,
}

impl Cors {
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Put the headers on a response.
    pub fn apply(&self, headers: &mut HeaderMap) {
        for (name, value) in &self.headers {
            if let (Ok(n), Ok(v)) = (
                HeaderName::from_bytes(name.as_bytes()),
                HeaderValue::from_str(value),
            ) {
                if *name == "vary" {
                    headers.append(n, v);
                } else {
                    headers.insert(n, v);
                }
            }
        }
    }
}

/// Whether the route is one of the six that never get CORS headers.
pub fn never_cors(method: &Method, matched: &str) -> bool {
    let m = if method == Method::HEAD {
        "GET"
    } else {
        method.as_str()
    };
    NEVER_CORS.iter().any(|(nm, np)| *nm == m && *np == matched)
}

/// The headers for a request (or, with `preflight`, the preflight of a request) to the route `matched`
/// with `method`, from `origin`, given the origins of the live credentials that have one.
///
/// `asks_private_network` is `Access-Control-Request-Private-Network: true` on the request.
pub fn decide(
    method: &Method,
    matched: Option<&str>,
    origin: Option<&str>,
    allowed: &BTreeSet<String>,
    asks_private_network: bool,
) -> Cors {
    let (Some(matched), Some(origin)) = (matched, origin) else {
        return Cors::default();
    };
    if never_cors(method, matched) {
        return Cors::default();
    }
    let mut headers: Vec<(&'static str, String)> = Vec::new();
    match route_class(method, matched) {
        // No row, no answer: the browser blocks it.
        Class::Unclassified => {}
        Class::Public => {
            headers.push(("access-control-allow-origin", "*".into()));
            headers.push(("access-control-allow-methods", PUBLIC_METHODS.into()));
            headers.push(("access-control-allow-headers", "content-type".into()));
            headers.push(("access-control-max-age", MAX_AGE.into()));
        }
        _ if origin != "null" && allowed.contains(origin) => {
            headers.push(("access-control-allow-origin", origin.to_string()));
            headers.push(("vary", "Origin".into()));
            headers.push(("access-control-allow-methods", ALLOWED_METHODS.into()));
            headers.push(("access-control-allow-headers", ALLOWED_HEADERS.into()));
            headers.push(("access-control-expose-headers", EXPOSED_HEADERS.into()));
            headers.push(("access-control-max-age", MAX_AGE.into()));
        }
        _ => {}
    }
    if !headers.is_empty() && asks_private_network {
        headers.push(("access-control-allow-private-network", "true".into()));
    }
    Cors { headers }
}
/// The method a preflight asks about: `Access-Control-Request-Method`, if it is a method.
pub fn preflight_method(headers: &HeaderMap) -> Option<Method> {
    headers
        .get("access-control-request-method")
        .and_then(|v| v.to_str().ok())
        .and_then(|m| Method::from_bytes(m.trim().as_bytes()).ok())
}

/// Whether the request is a CORS preflight: `OPTIONS` with an `Origin` and `Access-Control-Request-Method`.
pub fn is_preflight(method: &Method, headers: &HeaderMap) -> bool {
    method == Method::OPTIONS
        && headers.contains_key("origin")
        && headers.contains_key("access-control-request-method")
}

/// Whether the request asks for Private Network Access (`Access-Control-Request-Private-Network: true`).
pub fn asks_private_network(headers: &HeaderMap) -> bool {
    headers
        .get("access-control-request-private-network")
        .and_then(|v| v.to_str().ok())
        == Some("true")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(origins: &[&str]) -> BTreeSet<String> {
        origins.iter().map(|o| o.to_string()).collect()
    }

    fn d(method: Method, path: &str, origin: Option<&str>, set: &[&str], pna: bool) -> Cors {
        decide(&method, Some(path), origin, &allowed(set), pna)
    }

    #[test]
    fn a_request_with_no_origin_gets_no_cors_headers() {
        for (m, p) in [
            (Method::GET, "/api/health"),
            (Method::GET, "/api/config"),
            (Method::POST, "/api/bridge/pairing"),
        ] {
            assert!(d(m, p, None, &["https://a.example"], false).is_empty());
        }
    }

    #[test]
    fn a_public_route_answers_any_origin_and_never_with_credentials() {
        for origin in ["https://anyone.example", "http://localhost:3000", "null"] {
            let c = d(Method::GET, "/api/health", Some(origin), &[], false);
            assert_eq!(c.get("access-control-allow-origin"), Some("*"), "{origin}");
            assert_eq!(
                c.get("access-control-allow-methods"),
                Some("GET, HEAD, POST, OPTIONS")
            );
            assert_eq!(c.get("access-control-allow-headers"), Some("content-type"));
            assert_eq!(c.get("access-control-max-age"), Some("600"));
            assert_eq!(c.get("access-control-allow-credentials"), None);
            assert_eq!(c.get("vary"), None);
        }
        // The pairing bootstrap and its poll are public.
        assert_eq!(
            d(
                Method::POST,
                "/api/bridge/pairing",
                Some("https://app.example"),
                &[],
                false
            )
            .get("access-control-allow-origin"),
            Some("*")
        );
        assert_eq!(
            d(
                Method::GET,
                "/api/bridge/pairing/:id",
                Some("https://app.example"),
                &[],
                false
            )
            .get("access-control-allow-origin"),
            Some("*")
        );
        assert_eq!(
            d(
                Method::GET,
                "/api/bridge/capabilities",
                Some("https://app.example"),
                &[],
                false
            )
            .get("access-control-allow-origin"),
            Some("*")
        );
    }

    #[test]
    fn the_six_same_origin_routes_never_get_cors_headers_whoever_asks() {
        for (m, p) in [
            (Method::GET, "/api/auth/info"),
            (Method::GET, "/api/auth/session"),
            (Method::POST, "/api/auth/login"),
            (Method::POST, "/api/auth/setup"),
            (Method::POST, "/api/auth/link"),
            (Method::POST, "/api/auth/callback"),
        ] {
            for origin in ["https://anyone.example", "https://paired.example", "null"] {
                assert!(
                    d(
                        m.clone(),
                        p,
                        Some(origin),
                        &["https://paired.example"],
                        true
                    )
                    .is_empty(),
                    "{m} {p} {origin}"
                );
            }
        }
        assert!(
            never_cors(&Method::HEAD, "/api/auth/info"),
            "HEAD is the GET row"
        );
        assert!(!never_cors(&Method::POST, "/api/auth/info"));
        assert!(!never_cors(&Method::GET, "/api/auth/whoami"));
    }

    #[test]
    fn the_origin_of_a_live_credential_is_echoed_with_the_full_method_and_header_lists() {
        let set = ["https://formlogic.example"];
        let c = d(
            Method::GET,
            "/api/services",
            Some("https://formlogic.example"),
            &set,
            false,
        );
        assert_eq!(
            c.get("access-control-allow-origin"),
            Some("https://formlogic.example")
        );
        assert_eq!(c.get("vary"), Some("Origin"));
        assert_eq!(
            c.get("access-control-allow-methods"),
            Some("GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS"),
            "PATCH is there now"
        );
        let headers = c.get("access-control-allow-headers").unwrap();
        for name in [
            "authorization",
            "content-type",
            "x-oaiy-session",
            "x-oaiy-csrf",
            "x-oaiy-approve",
            "idempotency-key",
            "if-none-match",
        ] {
            assert!(
                headers.split(", ").any(|h| h == name),
                "{name} missing from {headers}"
            );
        }
        assert!(
            !headers.contains('*'),
            "a wildcard does not cover Authorization, so it is listed by name"
        );
        assert_eq!(
            c.get("access-control-expose-headers"),
            Some("etag, retry-after, www-authenticate")
        );
        assert_eq!(c.get("access-control-max-age"), Some("600"));
        assert_eq!(c.get("access-control-allow-credentials"), None);
        // A preflight for a change asks the same route with its own method.
        let c = d(
            Method::PATCH,
            "/api/calendar/appointments/:id",
            Some("https://formlogic.example"),
            &set,
            false,
        );
        assert_eq!(
            c.get("access-control-allow-origin"),
            Some("https://formlogic.example")
        );
    }

    #[test]
    fn an_origin_that_has_not_paired_gets_nothing_on_a_route_that_is_not_public() {
        let set = ["https://formlogic.example"];
        assert!(d(
            Method::GET,
            "/api/services",
            Some("https://evil.example"),
            &set,
            false
        )
        .is_empty());
        assert!(d(
            Method::POST,
            "/api/bridge/runs",
            Some("https://formlogic.example.evil.example"),
            &set,
            false
        )
        .is_empty());
        assert!(
            d(
                Method::GET,
                "/api/services",
                Some("http://formlogic.example"),
                &set,
                false
            )
            .is_empty(),
            "a different scheme is a different origin"
        );
        assert!(
            d(
                Method::GET,
                "/api/services",
                Some("https://FORMLOGIC.example"),
                &set,
                false
            )
            .is_empty(),
            "the comparison is exact"
        );
        // A hosted app that has not paired can reach exactly the public routes.
        assert!(d(
            Method::GET,
            "/api/config",
            Some("https://oaiy.com"),
            &set,
            false
        )
        .is_empty());
        assert!(d(
            Method::GET,
            "/api/config",
            Some("http://localhost:3000"),
            &set,
            false
        )
        .is_empty());
    }

    #[test]
    fn origin_null_is_never_matched_even_if_it_were_in_the_set() {
        assert!(
            d(Method::GET, "/api/services", Some("null"), &["null"], false).is_empty(),
            "null is never a member"
        );
        assert!(d(
            Method::POST,
            "/api/bridge/runs",
            Some("null"),
            &["null", "https://a.example"],
            true
        )
        .is_empty());
    }

    #[test]
    fn private_network_access_is_answered_only_when_asked_and_only_where_cors_applied() {
        let set = ["https://formlogic.example"];
        // Allowed origin, asked: yes.
        let c = d(
            Method::GET,
            "/api/services",
            Some("https://formlogic.example"),
            &set,
            true,
        );
        assert_eq!(c.get("access-control-allow-private-network"), Some("true"));
        // Allowed origin, not asked: no.
        assert_eq!(
            d(
                Method::GET,
                "/api/services",
                Some("https://formlogic.example"),
                &set,
                false
            )
            .get("access-control-allow-private-network"),
            None
        );
        // A public route, asked: yes.
        assert_eq!(
            d(
                Method::GET,
                "/api/health",
                Some("https://x.example"),
                &set,
                true
            )
            .get("access-control-allow-private-network"),
            Some("true")
        );
        // A disallowed origin, asked: no (it would let a hostile page reach the port).
        assert!(d(
            Method::GET,
            "/api/services",
            Some("https://evil.example"),
            &set,
            true
        )
        .is_empty());
        assert!(d(Method::GET, "/api/services", Some("null"), &set, true).is_empty());
        // A same-origin route, asked: no.
        assert!(d(
            Method::POST,
            "/api/auth/login",
            Some("https://formlogic.example"),
            &set,
            true
        )
        .is_empty());
    }

    #[test]
    fn a_route_that_is_not_in_the_table_gets_nothing() {
        let set = ["https://formlogic.example"];
        assert!(
            decide(
                &Method::GET,
                None,
                Some("https://formlogic.example"),
                &allowed(&set),
                true
            )
            .is_empty(),
            "no route, no headers"
        );
        assert!(d(
            Method::GET,
            "/api/no/such/route",
            Some("https://formlogic.example"),
            &set,
            false
        )
        .is_empty());
        assert!(d(
            Method::TRACE,
            "/api/services",
            Some("https://formlogic.example"),
            &set,
            false
        )
        .is_empty());
    }

    #[test]
    fn a_preflight_is_classified_by_the_method_it_asks_about() {
        let mut h = HeaderMap::new();
        h.insert(
            "origin",
            HeaderValue::from_static("https://formlogic.example"),
        );
        h.insert(
            "access-control-request-method",
            HeaderValue::from_static("PATCH"),
        );
        h.insert(
            "access-control-request-headers",
            HeaderValue::from_static("authorization"),
        );
        assert!(is_preflight(&Method::OPTIONS, &h));
        assert!(!is_preflight(&Method::GET, &h));
        assert_eq!(preflight_method(&h), Some(Method::PATCH));
        h.remove("access-control-request-method");
        assert!(
            !is_preflight(&Method::OPTIONS, &h),
            "no requested method is not a preflight"
        );
        h.insert(
            "access-control-request-method",
            HeaderValue::from_static("not a method"),
        );
        assert_eq!(preflight_method(&h), None);
        assert!(!asks_private_network(&h));
        h.insert(
            "access-control-request-private-network",
            HeaderValue::from_static("true"),
        );
        assert!(asks_private_network(&h));
        h.insert(
            "access-control-request-private-network",
            HeaderValue::from_static("false"),
        );
        assert!(!asks_private_network(&h));
    }

    #[test]
    fn the_headers_are_put_on_a_response_and_vary_is_added_not_replaced() {
        let c = d(
            Method::GET,
            "/api/services",
            Some("https://formlogic.example"),
            &["https://formlogic.example"],
            false,
        );
        let mut h = HeaderMap::new();
        h.insert("vary", HeaderValue::from_static("Accept-Encoding"));
        c.apply(&mut h);
        let vary: Vec<_> = h
            .get_all("vary")
            .iter()
            .map(|v| v.to_str().unwrap().to_string())
            .collect();
        assert_eq!(vary, ["Accept-Encoding", "Origin"]);
        assert_eq!(
            h.get("access-control-allow-origin").unwrap(),
            "https://formlogic.example"
        );
    }
}
