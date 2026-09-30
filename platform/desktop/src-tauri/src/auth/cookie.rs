//! Cookies (design 4.7.5) and the checks that make acting on one safe (4.5.2).
//!
//! **Names and attributes.** On an install behind a proxy (https) the cookies are `__Host-oaiy_dash`,
//! `__Host-oaiy_agent`, `__Host-oaiy_flows` (sessions) and `__Host-oaiy_dev` (the known device), with
//! `Path=/; Secure; HttpOnly; SameSite=Strict`, no `Domain`, and `Max-Age` only when asked for. The `__Host-`
//! prefix makes a browser refuse the cookie unless it is `Secure`, `Path=/` and has no `Domain`, and refuse to
//! let a sibling subdomain or a plain-HTTP page set or overwrite it. On a loopback server over plain HTTP the
//! names are `oaiy_dash_<port>`, `oaiy_agent_<port>`, `oaiy_flows_<port>`, `oaiy_dev_<port>` (cookies ignore
//! ports, so the port is in the name), with `Path=/; HttpOnly; SameSite=Strict` and no `Secure`.
//!
//! **Reading.** A `Cookie` header is read strictly and by name only: the same name twice is [`Lookup::Duplicate`]
//! (a sibling that set a cookie of the same name: the request is refused and the name cleared, 4.5.2 rule 5), a
//! header that is not visible ASCII, or that is over 16 KiB, is not read at all, and a name is compared exactly
//! (cookie names are case-sensitive). Nothing here allocates more than the value it returns.
//!
//! **The checks for a cookie** (4.5.2). Cookies do not separate ports, and every subdomain of one registrable
//! domain is same-site, so `SameSite=Strict` does not stop a script on a sibling host from causing a request that
//! carries the cookie. The boundary is an exact match of the `Origin`:
//!
//! 1. `Sec-Fetch-Site`, when present, must be `same-origin` (or `none` for `GET` and `HEAD`: a typed URL, a link);
//! 2. for `POST PUT PATCH DELETE` the `Origin` must be present and equal the expected origin of the request's
//!    `Host` (`null`, from a sandboxed frame, never matches);
//! 3. for those methods `X-OAIY-CSRF` must equal the value derived from the session token (constant-time).
//!
//! Steps 1 and 2 are the defence, step 3 the second lock (it also covers a client that sends no `Sec-Fetch-Site`).
//! A request that carries a bearer is not judged here: the bearer wins, and a browser never adds one by itself.

use axum::http::{header, HeaderMap, Method};

use super::host::{HostClass, HostName};
use super::presets::App;
use super::token;

/// `Max-Age` of a remembered session: 30 days.
pub const REMEMBER_MAX_AGE: u64 = 30 * 24 * 3600;
/// `Max-Age` of the device cookie: 180 days.
pub const DEVICE_MAX_AGE: u64 = 180 * 24 * 3600;
/// A `Cookie` header longer than this is not read.
pub const MAX_COOKIE_HEADER: usize = 16 * 1024;
/// The longest value looked at.
pub const MAX_VALUE: usize = 4096;
/// The header the page sends its derived value in.
pub const CSRF_HEADER: &str = "x-oaiy-csrf";

/// Which of the two shapes of cookie an install uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// Behind a proxy, over https: `__Host-` names, `Secure`.
    Host,
    /// A loopback server over plain HTTP: the port in the name, no `Secure`.
    Loopback { port: u16 },
}

impl Style {
    /// The style and the app of a host, when it is one that serves cookies: a configured public host (behind
    /// the proxy) or one of the loopback app names. Every other host (`localhost`, an address, an extra host)
    /// is for bearers only.
    pub fn of_host(class: &HostClass, host: &HostName) -> Option<(Style, App)> {
        match class {
            HostClass::Public(app) => Some((Style::Host, *app)),
            HostClass::LoopbackApp(app) => Some((
                Style::Loopback {
                    port: host.port.unwrap_or(80),
                },
                *app,
            )),
            _ => None,
        }
    }

    fn name(self, what: &str) -> String {
        match self {
            Style::Host => format!("__Host-oaiy_{what}"),
            Style::Loopback { port } => format!("oaiy_{what}_{port}"),
        }
    }

    /// The name of the session cookie of an app.
    pub fn session_name(self, app: App) -> String {
        self.name(app.name())
    }

    /// The name of the known-device cookie.
    pub fn device_name(self) -> String {
        self.name("dev")
    }

    fn secure(self) -> &'static str {
        match self {
            Style::Host => "; Secure",
            Style::Loopback { .. } => "",
        }
    }

    /// A `Set-Cookie` value: `NAME=VALUE; Path=/; [Secure; ]HttpOnly; SameSite=Strict[; Max-Age=N]`. A session
    /// cookie has no `Max-Age` unless the owner asked to be remembered.
    pub fn set(self, name: &str, value: &str, max_age: Option<u64>) -> String {
        debug_assert!(
            value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
            "a token needs no escaping"
        );
        let mut out = format!(
            "{name}={value}; Path=/{}; HttpOnly; SameSite=Strict",
            self.secure()
        );
        if let Some(seconds) = max_age {
            out.push_str(&format!("; Max-Age={seconds}"));
        }
        out
    }

    /// The same name and attributes with an empty value and `Max-Age=0`: the browser deletes the cookie.
    pub fn clear(self, name: &str) -> String {
        format!(
            "{name}=; Path=/{}; HttpOnly; SameSite=Strict; Max-Age=0",
            self.secure()
        )
    }
}

// ---- reading a Cookie header ------------------------------------------------------------------

/// What a `Cookie` header says about one name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The name is not there (or the header cannot be read).
    Absent,
    /// The name is there once, with this value.
    One(String),
    /// The name is there more than once: a sibling set one of the same name.
    Duplicate,
}

fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(
            b,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

/// The value of the cookie called `name` among every `Cookie` header of the request.
pub fn find(headers: &HeaderMap, name: &str) -> Lookup {
    let mut found: Option<String> = None;
    let mut total = 0usize;
    for value in headers.get_all(header::COOKIE) {
        total = total.saturating_add(value.len());
        if total > MAX_COOKIE_HEADER {
            return Lookup::Absent;
        }
        // Visible ASCII (and tab) only: `to_str` refuses anything else, and so does a header a browser
        // would not have sent.
        let Ok(text) = value.to_str() else {
            continue;
        };
        for pair in text.split(';') {
            let pair = pair.trim_matches(|c| c == ' ' || c == '\t');
            let Some((n, v)) = pair.split_once('=') else {
                continue;
            };
            if n.is_empty() || !n.bytes().all(is_tchar) || n != name {
                continue;
            }
            if found.is_some() {
                return Lookup::Duplicate;
            }
            let v: String = v.chars().take(MAX_VALUE).collect();
            found = Some(v);
        }
    }
    match found {
        Some(v) => Lookup::One(v),
        None => Lookup::Absent,
    }
}

/// Whether a request carries a cookie of this name at all (once or more).
pub fn carries(headers: &HeaderMap, name: &str) -> bool {
    !matches!(find(headers, name), Lookup::Absent)
}

// ---- the checks of 4.5.2 ----------------------------------------------------------------------

/// Why a cookie request was refused (all `403 csrf`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrfFail {
    /// `Sec-Fetch-Site` said the request is not from this origin.
    FetchSite,
    /// The `Origin` is absent, `null`, or not this host's.
    Origin,
    /// `X-OAIY-CSRF` is absent or wrong.
    Token,
}

impl CsrfFail {
    pub fn message(self) -> &'static str {
        match self {
            CsrfFail::FetchSite => "This request did not come from this site.",
            CsrfFail::Origin => "This request's Origin is not this site's.",
            CsrfFail::Token => "The X-OAIY-CSRF header is missing or wrong.",
        }
    }
}

/// `POST PUT PATCH DELETE`: the methods that change something.
pub fn is_unsafe(method: &Method) -> bool {
    matches!(
        *method,
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    )
}

/// The one value of a header, if it is there exactly once and is text.
fn single<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>, ()> {
    let mut all = headers.get_all(name).iter();
    match (all.next(), all.next()) {
        (None, _) => Ok(None),
        (Some(v), None) => v.to_str().map(Some).map_err(|_| ()),
        (Some(_), Some(_)) => Err(()),
    }
}

/// Step 1: `Sec-Fetch-Site`. Absent is allowed (a non-browser client); present, it must be `same-origin`, or
/// `none` for `GET`/`HEAD`. `same-site`, `cross-site`, anything else, a value that is not text and a second
/// header are all refused.
pub fn fetch_site_ok(headers: &HeaderMap, method: &Method) -> bool {
    match single(headers, "sec-fetch-site") {
        Ok(None) => true,
        Ok(Some("same-origin")) => true,
        Ok(Some("none")) => matches!(*method, Method::GET | Method::HEAD),
        _ => false,
    }
}

/// Step 2: the `Origin` is there once and equals `own_origin` (lowercase scheme and host, the port of the `Host`
/// header). `Origin: null` never does.
pub fn origin_ok(headers: &HeaderMap, own_origin: &str) -> bool {
    match single(headers, "origin") {
        Ok(Some(o)) => o.to_ascii_lowercase() == own_origin,
        _ => false,
    }
}

/// Step 3: `X-OAIY-CSRF` is there once and equals `expected`, compared in constant time.
pub fn csrf_header_ok(headers: &HeaderMap, expected: &str) -> bool {
    match single(headers, CSRF_HEADER) {
        Ok(Some(v)) => token::secrets_equal(v.as_bytes(), expected.as_bytes()),
        _ => false,
    }
}

/// All three, for a request authenticated by a session cookie. `own_origin` is the origin a page served for
/// the request's `Host` has, and `csrf_value` the value derived from the session token.
pub fn check_session_request(
    method: &Method,
    headers: &HeaderMap,
    own_origin: &str,
    csrf_value: &str,
) -> Result<(), CsrfFail> {
    if !fetch_site_ok(headers, method) {
        return Err(CsrfFail::FetchSite);
    }
    if is_unsafe(method) {
        if !origin_ok(headers, own_origin) {
            return Err(CsrfFail::Origin);
        }
        if !csrf_header_ok(headers, csrf_value) {
            return Err(CsrfFail::Token);
        }
    }
    Ok(())
}

/// The rule of `login`, `setup`, `link` and `callback` (4.7.3): they refuse a cross-origin browser request. An
/// `Origin` that is present must be this host's (a client that sends none, such as `curl`, is let through: these
/// routes are how a session is first made), and a `Sec-Fetch-Site` that is present must be `same-origin`.
pub fn check_same_origin_browser(headers: &HeaderMap, own_origin: &str) -> Result<(), CsrfFail> {
    match single(headers, "sec-fetch-site") {
        Ok(None) | Ok(Some("same-origin")) => {}
        _ => return Err(CsrfFail::FetchSite),
    }
    match single(headers, "origin") {
        Ok(None) => Ok(()),
        Ok(Some(o)) if o.to_ascii_lowercase() == own_origin => Ok(()),
        _ => Err(CsrfFail::Origin),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    const SECRET: &str = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    const CSRF: &str = "A-wCyU91xe9jXHeH2QyQhZgv00DTWUwk1H9x_7RSRmg";
    const TOKEN: &str = "oaiyses_0123456789abcdef_AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (n, v) in pairs {
            h.append(
                axum::http::HeaderName::from_bytes(n.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    // Attributes, exactly ------------------------------------------------------------------------

    #[test]
    fn the_names_of_a_proxied_install_carry_the_host_prefix() {
        let s = Style::Host;
        assert_eq!(s.session_name(App::Dash), "__Host-oaiy_dash");
        assert_eq!(s.session_name(App::Agent), "__Host-oaiy_agent");
        assert_eq!(s.session_name(App::Flows), "__Host-oaiy_flows");
        assert_eq!(s.device_name(), "__Host-oaiy_dev");
    }

    #[test]
    fn the_names_of_a_loopback_server_carry_the_port_and_no_prefix() {
        let s = Style::Loopback { port: 41000 };
        assert_eq!(s.session_name(App::Dash), "oaiy_dash_41000");
        assert_eq!(s.session_name(App::Agent), "oaiy_agent_41000");
        assert_eq!(s.session_name(App::Flows), "oaiy_flows_41000");
        assert_eq!(s.device_name(), "oaiy_dev_41000");
        // Another port is another cookie: two tunnelled servers do not overwrite each other's.
        assert_ne!(
            Style::Loopback { port: 41001 }.session_name(App::Dash),
            s.session_name(App::Dash)
        );
    }

    #[test]
    fn a_proxied_session_cookie_is_exactly_secure_httponly_strict_with_no_domain() {
        let s = Style::Host;
        assert_eq!(
            s.set("__Host-oaiy_dash", TOKEN, None),
            format!("__Host-oaiy_dash={TOKEN}; Path=/; Secure; HttpOnly; SameSite=Strict")
        );
        // Remember this device: 30 days.
        assert_eq!(
            s.set("__Host-oaiy_dash", TOKEN, Some(REMEMBER_MAX_AGE)),
            format!(
                "__Host-oaiy_dash={TOKEN}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=2592000"
            )
        );
        // The device cookie: 180 days.
        assert_eq!(
            s.set("__Host-oaiy_dev", TOKEN, Some(DEVICE_MAX_AGE)),
            format!(
                "__Host-oaiy_dev={TOKEN}; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=15552000"
            )
        );
        let text = s.set("__Host-oaiy_dash", TOKEN, Some(1));
        assert!(!text.to_ascii_lowercase().contains("domain"), "no Domain");
    }

    #[test]
    fn a_loopback_cookie_is_httponly_strict_and_has_no_secure() {
        let s = Style::Loopback { port: 41000 };
        assert_eq!(
            s.set("oaiy_dash_41000", TOKEN, None),
            format!("oaiy_dash_41000={TOKEN}; Path=/; HttpOnly; SameSite=Strict")
        );
        assert_eq!(
            s.set("oaiy_dash_41000", TOKEN, Some(REMEMBER_MAX_AGE)),
            format!("oaiy_dash_41000={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000")
        );
        assert_eq!(
            s.set("oaiy_dev_41000", TOKEN, Some(DEVICE_MAX_AGE)),
            format!("oaiy_dev_41000={TOKEN}; Path=/; HttpOnly; SameSite=Strict; Max-Age=15552000")
        );
        assert!(!s.set("oaiy_dash_41000", TOKEN, None).contains("Secure"));
    }

    #[test]
    fn deleting_is_the_same_name_and_attributes_with_max_age_zero_for_both_variants() {
        assert_eq!(
            Style::Host.clear("__Host-oaiy_dash"),
            "__Host-oaiy_dash=; Path=/; Secure; HttpOnly; SameSite=Strict; Max-Age=0"
        );
        assert_eq!(
            Style::Loopback { port: 80 }.clear("oaiy_agent_80"),
            "oaiy_agent_80=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
        );
    }

    #[test]
    fn only_a_host_that_serves_an_app_has_cookies() {
        let h = |s: &str| HostName::parse(s).unwrap();
        assert_eq!(
            Style::of_host(&HostClass::Public(App::Agent), &h("agent.example.com")),
            Some((Style::Host, App::Agent))
        );
        assert_eq!(
            Style::of_host(
                &HostClass::LoopbackApp(App::Dash),
                &h("dash.oaiy.localhost:41000")
            ),
            Some((Style::Loopback { port: 41000 }, App::Dash))
        );
        assert_eq!(
            Style::of_host(
                &HostClass::LoopbackApp(App::Flows),
                &h("flows.oaiy.localhost")
            ),
            Some((Style::Loopback { port: 80 }, App::Flows)),
            "no port in the Host is port 80"
        );
        for class in [HostClass::Loopback, HostClass::LanAddress, HostClass::Extra] {
            assert_eq!(
                Style::of_host(&class, &h("localhost:17972")),
                None,
                "{class:?}"
            );
        }
    }

    // Reading ------------------------------------------------------------------------------------

    #[test]
    fn a_cookie_is_found_by_its_exact_name() {
        let h = headers(&[(
            "cookie",
            "a=1; __Host-oaiy_dash=abc; oaiy_dash_41000=def;other=x",
        )]);
        assert_eq!(find(&h, "__Host-oaiy_dash"), Lookup::One("abc".into()));
        assert_eq!(find(&h, "oaiy_dash_41000"), Lookup::One("def".into()));
        assert_eq!(find(&h, "other"), Lookup::One("x".into()));
        assert_eq!(find(&h, "oaiy_dash"), Lookup::Absent);
        assert_eq!(
            find(&h, "__host-oaiy_dash"),
            Lookup::Absent,
            "names are case-sensitive"
        );
        assert_eq!(find(&HeaderMap::new(), "a"), Lookup::Absent);
        // A value that holds an equals sign keeps it; an empty value is a value.
        let h = headers(&[("cookie", "t=a=b; e=")]);
        assert_eq!(find(&h, "t"), Lookup::One("a=b".into()));
        assert_eq!(find(&h, "e"), Lookup::One(String::new()));
    }

    #[test]
    fn the_same_name_twice_is_a_duplicate_in_one_header_or_across_two_and_even_with_equal_values() {
        for h in [
            headers(&[("cookie", "n=1; n=2")]),
            headers(&[("cookie", "n=1"), ("cookie", "n=2")]),
            headers(&[("cookie", "n=1; x=y"), ("cookie", "z=1; n=1")]),
            headers(&[("cookie", "n=1;n=1")]),
        ] {
            assert_eq!(find(&h, "n"), Lookup::Duplicate);
        }
        // A different name that only shares a prefix is not the same name.
        assert_eq!(
            find(&headers(&[("cookie", "n=1; nn=2; n1=3")]), "n"),
            Lookup::One("1".into())
        );
    }

    #[test]
    fn segments_that_are_not_cookies_are_skipped_and_the_name_must_be_a_token() {
        let h = headers(&[("cookie", "junk; =novalue; a b=c; n=1;;; ;x")]);
        assert_eq!(find(&h, "n"), Lookup::One("1".into()));
        assert_eq!(
            find(&h, "a b"),
            Lookup::Absent,
            "a space is not a name character"
        );
        assert_eq!(find(&h, ""), Lookup::Absent);
    }

    #[test]
    fn a_header_that_is_not_visible_ascii_or_is_too_long_is_not_read() {
        let mut h = HeaderMap::new();
        h.append(
            header::COOKIE,
            HeaderValue::from_bytes(b"n=caf\xc3\xa9").unwrap(),
        );
        assert_eq!(find(&h, "n"), Lookup::Absent);
        let big = format!("n=1; pad={}", "x".repeat(MAX_COOKIE_HEADER));
        assert_eq!(find(&headers(&[("cookie", &big)]), "n"), Lookup::Absent);
        // The bound is on all the headers together.
        let half = "p=".to_string() + &"x".repeat(MAX_COOKIE_HEADER / 2 + 8);
        let h = headers(&[("cookie", &half), ("cookie", &half), ("cookie", "n=1")]);
        assert_eq!(find(&h, "n"), Lookup::Absent);
        // A very long value is cut, not copied whole.
        let h = headers(&[("cookie", &format!("n={}", "y".repeat(8000)))]);
        match find(&h, "n") {
            Lookup::One(v) => assert_eq!(v.len(), MAX_VALUE),
            other => panic!("{other:?}"),
        }
    }

    /// A tiny deterministic generator, so that the fuzzing is the same every run.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 24
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn fuzzing_the_cookie_parser_never_panics_and_holds_its_invariants() {
        let mut rng = Rng(0xC0FFEE);
        let alphabet: &[u8] = b"abcXYZ019_-=;, \t\"\\/%:.()<>@[]{}?*+!#$&'^`|~";
        let names = ["n", "m", "__Host-oaiy_dash", "oaiy_dev_1", "N"];
        // How many of each answer the run produced: the invariants below mean nothing if it produced one kind.
        let (mut ones, mut dups, mut absent, mut unreadable) = (0, 0, 0, 0);
        for round in 0..30_000 {
            // Part random bytes, part well-formed pairs, so both the junk paths and the matching paths run.
            let mut header = String::new();
            let pieces = rng.below(8);
            for _ in 0..pieces {
                if rng.below(2) == 0 {
                    header.push_str(names[rng.below(names.len() as u64) as usize]);
                    header.push('=');
                    for _ in 0..rng.below(12) {
                        header.push(alphabet[rng.below(alphabet.len() as u64) as usize] as char);
                    }
                } else {
                    for _ in 0..rng.below(20) {
                        header.push(alphabet[rng.below(alphabet.len() as u64) as usize] as char);
                    }
                }
                header.push_str(["; ", ";", " ; ", "\t;"][rng.below(4) as usize]);
            }
            let mut h = HeaderMap::new();
            match HeaderValue::from_str(&header) {
                Ok(v) => {
                    h.append(header::COOKIE, v);
                }
                Err(_) => {
                    unreadable += 1;
                    continue;
                }
            }
            for name in names {
                let got = find(&h, name);
                // The invariants: a value has no `;` (it ends a pair), and no value is longer than the bound;
                // a duplicate means the name really is there twice.
                match &got {
                    Lookup::One(v) => {
                        ones += 1;
                        assert!(
                            !v.contains(';') && v.len() <= MAX_VALUE,
                            "round {round}: {header:?}"
                        );
                        assert_eq!(count_pairs(&header, name), 1, "round {round}: {header:?}");
                    }
                    Lookup::Duplicate => {
                        dups += 1;
                        assert!(count_pairs(&header, name) >= 2, "round {round}: {header:?}")
                    }
                    Lookup::Absent => {
                        absent += 1;
                        assert_eq!(count_pairs(&header, name), 0, "round {round}: {header:?}")
                    }
                }
            }
        }
        assert!(
            ones > 1000 && dups > 100 && absent > 1000,
            "the run covered {ones} finds, {dups} duplicates, {absent} misses ({unreadable} unreadable)"
        );
    }

    /// How many `name=` pairs a header has, counted by the plainest means.
    fn count_pairs(header: &str, name: &str) -> usize {
        header
            .split(';')
            .map(|p| p.trim_matches(|c| c == ' ' || c == '\t'))
            .filter(|p| p.split_once('=').is_some_and(|(n, _)| n == name))
            .count()
    }

    #[test]
    fn fuzzing_the_header_checks_never_panics_and_only_the_exact_forms_pass() {
        let mut rng = Rng(0xBADC0DE);
        let charset: &[u8] = b"abcdefghijklmnopqrstuvwxyzHTPS:/.-0123456789 null,";
        for _ in 0..20_000 {
            let mut value = String::new();
            for _ in 0..rng.below(40) {
                value.push(charset[rng.below(charset.len() as u64) as usize] as char);
            }
            let Ok(v) = HeaderValue::from_str(&value) else {
                continue;
            };
            let mut h = HeaderMap::new();
            h.append("origin", v.clone());
            h.append("sec-fetch-site", v.clone());
            h.append(CSRF_HEADER, v);
            // None of these random values is the exact form (the sets do not overlap enough to be it).
            let _ = fetch_site_ok(&h, &Method::POST);
            let _ = origin_ok(&h, "https://dash.example.com");
            assert_eq!(csrf_header_ok(&h, CSRF), value == CSRF);
            assert_eq!(
                origin_ok(&h, "https://dash.example.com"),
                value.eq_ignore_ascii_case("https://dash.example.com")
            );
            assert_eq!(fetch_site_ok(&h, &Method::POST), value == "same-origin");
            assert_eq!(
                fetch_site_ok(&h, &Method::GET),
                value == "same-origin" || value == "none"
            );
        }
    }

    // The checks of 4.5.2 ------------------------------------------------------------------------

    #[test]
    fn the_csrf_value_is_the_one_derived_from_the_token_secret() {
        // The known-answer vector of 4.1: the value the page holds and the server recomputes.
        assert_eq!(token::csrf_value(SECRET).as_deref(), Some(CSRF));
    }

    const ORIGIN: &str = "https://dash.example.com";

    fn good() -> HeaderMap {
        headers(&[
            ("origin", ORIGIN),
            ("sec-fetch-site", "same-origin"),
            (CSRF_HEADER, CSRF),
        ])
    }

    #[test]
    fn a_same_origin_request_with_the_header_passes_and_each_wrong_part_is_named() {
        for method in [Method::POST, Method::PUT, Method::PATCH, Method::DELETE] {
            assert_eq!(
                check_session_request(&method, &good(), ORIGIN, CSRF),
                Ok(())
            );
        }
        // Another origin: a sibling on the same registrable domain, another port, another scheme, null.
        for other in [
            "https://flows.example.com",
            "https://dash.example.com:8443",
            "http://dash.example.com",
            "null",
            "https://dash.example.com.evil.example",
            "",
        ] {
            let mut h = good();
            h.insert("origin", HeaderValue::from_str(other).unwrap());
            assert_eq!(
                check_session_request(&Method::POST, &h, ORIGIN, CSRF),
                Err(CsrfFail::Origin),
                "{other:?}"
            );
        }
        // No Origin at all on a mutation.
        let mut h = good();
        h.remove("origin");
        assert_eq!(
            check_session_request(&Method::POST, &h, ORIGIN, CSRF),
            Err(CsrfFail::Origin)
        );
        // The Origin's case does not matter (scheme and host are lowercased), and only that.
        let mut h = good();
        h.insert(
            "origin",
            HeaderValue::from_static("HTTPS://Dash.Example.COM"),
        );
        assert_eq!(
            check_session_request(&Method::POST, &h, ORIGIN, CSRF),
            Ok(())
        );
        // No header, a wrong one, one of another length, two of them.
        for bad in [
            None,
            Some("x".repeat(43)),
            Some(CSRF[..42].to_string()),
            Some(format!("{CSRF}A")),
        ] {
            let mut h = good();
            h.remove(CSRF_HEADER);
            if let Some(b) = &bad {
                h.insert(CSRF_HEADER, HeaderValue::from_str(b).unwrap());
            }
            assert_eq!(
                check_session_request(&Method::POST, &h, ORIGIN, CSRF),
                Err(CsrfFail::Token),
                "{bad:?}"
            );
        }
        let mut h = good();
        h.append(CSRF_HEADER, HeaderValue::from_static(CSRF));
        assert_eq!(
            check_session_request(&Method::POST, &h, ORIGIN, CSRF),
            Err(CsrfFail::Token)
        );
        let mut h = good();
        h.append("origin", HeaderValue::from_static(ORIGIN));
        assert_eq!(
            check_session_request(&Method::POST, &h, ORIGIN, CSRF),
            Err(CsrfFail::Origin)
        );
    }

    #[test]
    fn fetch_metadata_refuses_same_site_and_cross_site_and_none_only_for_reads() {
        for (site, method, ok) in [
            ("same-origin", Method::POST, true),
            ("same-origin", Method::GET, true),
            ("same-site", Method::POST, false),
            ("same-site", Method::GET, false),
            ("cross-site", Method::POST, false),
            ("cross-site", Method::GET, false),
            ("none", Method::GET, true),
            ("none", Method::HEAD, true),
            ("none", Method::POST, false),
            ("none", Method::DELETE, false),
            ("Same-Origin", Method::GET, false),
            ("", Method::GET, false),
            ("nonsense", Method::GET, false),
        ] {
            let h = headers(&[("sec-fetch-site", site)]);
            assert_eq!(fetch_site_ok(&h, &method), ok, "{site:?} {method}");
        }
        // Absent is a non-browser client: allowed.
        assert!(fetch_site_ok(&HeaderMap::new(), &Method::POST));
        // Two headers are ambiguous.
        let h = headers(&[
            ("sec-fetch-site", "same-origin"),
            ("sec-fetch-site", "same-origin"),
        ]);
        assert!(!fetch_site_ok(&h, &Method::GET));
        // A refused Fetch Metadata is reported before the Origin.
        let mut h = good();
        h.insert("sec-fetch-site", HeaderValue::from_static("cross-site"));
        assert_eq!(
            check_session_request(&Method::POST, &h, ORIGIN, CSRF),
            Err(CsrfFail::FetchSite)
        );
    }

    #[test]
    fn each_method_that_changes_something_needs_the_origin_the_header_and_a_browser_that_says_same_origin(
    ) {
        for method in [Method::POST, Method::PUT, Method::PATCH, Method::DELETE] {
            assert!(is_unsafe(&method), "{method}");
            // Nothing at all: the Origin is asked for first, then the header.
            assert_eq!(
                check_session_request(&method, &HeaderMap::new(), ORIGIN, CSRF),
                Err(CsrfFail::Origin),
                "{method}"
            );
            let only_origin = headers(&[("origin", ORIGIN)]);
            assert_eq!(
                check_session_request(&method, &only_origin, ORIGIN, CSRF),
                Err(CsrfFail::Token),
                "{method}"
            );
            // `none` (typed in the address bar) is for reads only.
            let mut typed = good();
            typed.insert("sec-fetch-site", HeaderValue::from_static("none"));
            assert_eq!(
                check_session_request(&method, &typed, ORIGIN, CSRF),
                Err(CsrfFail::FetchSite),
                "{method}"
            );
        }
        for method in [Method::GET, Method::HEAD, Method::OPTIONS] {
            assert!(!is_unsafe(&method), "{method}");
        }
    }

    #[test]
    fn a_read_needs_only_fetch_metadata_and_carries_no_origin_or_header() {
        for method in [Method::GET, Method::HEAD] {
            assert_eq!(
                check_session_request(&method, &HeaderMap::new(), ORIGIN, CSRF),
                Ok(())
            );
            let h = headers(&[("sec-fetch-site", "none")]);
            assert_eq!(check_session_request(&method, &h, ORIGIN, CSRF), Ok(()));
        }
    }

    #[test]
    fn login_setup_link_and_callback_refuse_a_cross_origin_browser_and_let_curl_through() {
        // No Origin, no Fetch Metadata: curl.
        assert_eq!(check_same_origin_browser(&HeaderMap::new(), ORIGIN), Ok(()));
        assert_eq!(
            check_same_origin_browser(&headers(&[("origin", ORIGIN)]), ORIGIN),
            Ok(())
        );
        assert_eq!(
            check_same_origin_browser(
                &headers(&[("origin", ORIGIN), ("sec-fetch-site", "same-origin")]),
                ORIGIN
            ),
            Ok(())
        );
        for (origin, site, fail) in [
            (Some("https://evil.example"), None, CsrfFail::Origin),
            (Some("null"), None, CsrfFail::Origin),
            (
                Some("https://flows.example.com"),
                Some("same-site"),
                CsrfFail::FetchSite,
            ),
            (None, Some("cross-site"), CsrfFail::FetchSite),
            (None, Some("none"), CsrfFail::FetchSite),
            (Some(ORIGIN), Some("same-site"), CsrfFail::FetchSite),
        ] {
            let mut pairs = Vec::new();
            if let Some(o) = origin {
                pairs.push(("origin", o));
            }
            if let Some(s) = site {
                pairs.push(("sec-fetch-site", s));
            }
            assert_eq!(
                check_same_origin_browser(&headers(&pairs), ORIGIN),
                Err(fail),
                "{origin:?} {site:?}"
            );
        }
    }
}
