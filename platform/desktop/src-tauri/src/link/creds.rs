//! The one place a credential is put on an outgoing request (design 4.16.3).
//!
//! Every credentialed request used to read `account.credential` where it was made and hand it to
//! `bearer_auth`: 32 places in 12 files, and nothing that chose a credential by where the request was
//! going. A change that sent the provider's key to another address was one wrong URL away. Here a
//! credential is registered for an ORIGIN (scheme, host and port, compared as the URL standard parses
//! them) and a KIND, and a request asks [`Creds::for_url`] for the one that belongs to where it is going:
//!
//! ```ignore
//! let creds = Creds::for_account(&account)?;
//! client.post(&url).json(&body).with_creds(&creds, CredKind::Provider)?.send()
//! ```
//!
//! A request whose origin has no credential of that kind is refused before anything is sent
//! ([`CredError::NoCredentialForOrigin`]). The call site names the kind, so the choice never rests on
//! the origin alone, and the rules of registration keep a relay's token and the provider's key apart:
//!
//! 1. one credential for each (origin, kind);
//! 2. a [`CredKind::Provider`] and a [`CredKind::Issuer`] credential may share an origin, because the
//!    legacy FormLogic issuer sits at the provider's own address, and each call then gets its own;
//! 3. a [`CredKind::Relay`] origin is unique among all the origins registered, in either order: a relay
//!    that shares an address with anything else is refused ("give the relay its own hostname");
//! 4. a relay's address has no path: a relay on a shared domain would share an origin with the owner's
//!    other pages.
//!
//! The clients that carry these credentials come from [`client_builder`]: redirects off, so a credential
//! cannot be taken anywhere by the server it was given to, a fixed user agent, and no cookie store.
//!
//! Only the heartbeat goes through here so far. The other lanes still call `bearer_auth` themselves, and
//! convert one lane group at a time (see `CONVERTED` in the tests, which fails when a converted lane
//! reads a credential again). Until they have, a new credentialed request is still one the guard does
//! not see: that guard, for every file, is the last step of the conversion.

use reqwest::blocking::RequestBuilder;
use reqwest::header::{HeaderValue, AUTHORIZATION};
use url::Url;

use super::net::Keep;
use super::LinkedAccount;

/// What a credential is for. The call site names it: where a request is going is not enough to say
/// which credential it carries, because two kinds may share an origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredKind {
    /// The linked provider's key, for the provider's own origin.
    Provider,
    /// This desktop's token at the owner's relay, for the relay's origin alone.
    Relay,
    /// The legacy companion issuer's bearer, which is at the provider's origin in the one deployment there is.
    Issuer,
}

impl CredKind {
    fn name(self) -> &'static str {
        match self {
            CredKind::Provider => "provider",
            CredKind::Relay => "relay",
            CredKind::Issuer => "issuer",
        }
    }
}

/// Why a credential was not put on a request, or not registered. Nothing in these says what a credential is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredError {
    /// No credential of this kind is registered for the origin the request is going to. This is the refusal
    /// that keeps one destination's credential from reaching another.
    NoCredentialForOrigin { kind: CredKind, origin: String },
    /// The URL is not an origin a credential can be bound to: it is not http or https, or it has no host.
    NotAnOrigin(String),
    /// The URL carries a user name or a password. `https://relay.example.com@evil.test/` reads as the relay to
    /// a person and goes to evil.test; none is accepted, whatever host it names.
    UserInfo,
    /// A credential of this kind is registered for this origin already.
    Duplicate { kind: CredKind, origin: String },
    /// A relay's origin is shared with another credential's, or one that would be.
    RelayOriginShared { origin: String },
    /// A relay's address has a path, a query or a fragment.
    RelayHasPath { origin: String },
    /// The secret is empty or cannot be the value of a header.
    Unusable,
    /// The request could not be built.
    BadRequest(String),
}

impl std::fmt::Display for CredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CredError::NoCredentialForOrigin { kind, origin } => {
                write!(f, "no {} credential is held for {origin}, so nothing was sent", kind.name())
            }
            CredError::NotAnOrigin(why) => write!(f, "that address is not one a credential can be sent to ({why})"),
            CredError::UserInfo => write!(f, "an address with a user name or a password in it is never sent a credential"),
            CredError::Duplicate { kind, origin } => write!(f, "a {} credential is held for {origin} already", kind.name()),
            CredError::RelayOriginShared { origin } => {
                write!(f, "{origin} is shared with another credential, and a relay's own hostname is never shared: give the relay a hostname of its own")
            }
            CredError::RelayHasPath { origin } => {
                write!(f, "a relay's address is its hostname alone, with no path ({origin} has one): a relay under a path shares its origin with the rest of that site")
            }
            CredError::Unusable => write!(f, "the credential is empty or has characters a request cannot carry"),
            CredError::BadRequest(why) => write!(f, "the request could not be built ({why})"),
        }
    }
}

impl std::error::Error for CredError {}

/// Where a credential may go: scheme, host and port as the URL standard reads them. The parser lowercases
/// the host, turns an internationalised one into its ASCII form, writes an IPv6 address one way and leaves
/// out a port that is the scheme's own, so two spellings of one origin are equal here and a host that
/// merely looks alike (a trailing dot, a longer name, a user name in front) is not.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Origin {
    scheme: String,
    host: String,
    port: u16,
}

impl Origin {
    fn of(url: &Url) -> Result<Origin, CredError> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(CredError::NotAnOrigin(format!("the scheme is {}", url.scheme())));
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err(CredError::UserInfo);
        }
        let host = url.host_str().filter(|h| !h.is_empty()).ok_or_else(|| CredError::NotAnOrigin("it has no host".into()))?;
        let port = url.port_or_known_default().ok_or_else(|| CredError::NotAnOrigin("it has no port".into()))?;
        Ok(Origin { scheme: url.scheme().to_string(), host: host.to_string(), port })
    }

    /// `https://host`, with the port only when it is not the scheme's own: what an error says.
    fn shown(&self) -> String {
        let default = if self.scheme == "https" { 443 } else { 80 };
        if self.port == default {
            format!("{}://{}", self.scheme, self.host)
        } else {
            format!("{}://{}:{}", self.scheme, self.host, self.port)
        }
    }
}

struct Entry {
    kind: CredKind,
    origin: Origin,
    /// `Bearer <secret>`, marked sensitive so that nothing prints it.
    bearer: HeaderValue,
}

/// The credentials a request may carry, by origin and kind. Built for the moment of a request from what
/// is linked ([`Creds::for_account`]), so a link that changes is read the next time.
#[derive(Default)]
pub struct Creds {
    entries: Vec<Entry>,
}

impl std::fmt::Debug for Creds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let held: Vec<String> = self.entries.iter().map(|e| format!("{} for {}", e.kind.name(), e.origin.shown())).collect();
        f.debug_struct("Creds").field("held", &held).finish()
    }
}

impl Creds {
    /// The credentials of the provider this desktop is linked to: the account's key, for the origin of its base address.
    pub fn for_account(account: &LinkedAccount) -> Result<Creds, CredError> {
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, &account.base_url, &account.credential)?;
        Ok(creds)
    }

    /// Bind `secret` to the origin of `base_url`, for `kind`, under the rules at the top of this file.
    pub fn register(&mut self, kind: CredKind, base_url: &str, secret: &str) -> Result<(), CredError> {
        let url = Url::parse(base_url).map_err(|e| CredError::NotAnOrigin(e.to_string()))?;
        let origin = Origin::of(&url)?;
        if kind == CredKind::Relay && (!matches!(url.path(), "" | "/") || url.query().is_some() || url.fragment().is_some()) {
            return Err(CredError::RelayHasPath { origin: origin.shown() });
        }
        if self.entries.iter().any(|e| e.kind == kind && e.origin == origin) {
            return Err(CredError::Duplicate { kind, origin: origin.shown() });
        }
        let shares_with_a_relay = self.entries.iter().any(|e| e.origin == origin && (e.kind == CredKind::Relay || kind == CredKind::Relay));
        if shares_with_a_relay {
            return Err(CredError::RelayOriginShared { origin: origin.shown() });
        }
        if secret.trim().is_empty() {
            return Err(CredError::Unusable);
        }
        let mut bearer = HeaderValue::from_str(&format!("Bearer {secret}")).map_err(|_| CredError::Unusable)?;
        bearer.set_sensitive(true);
        self.entries.push(Entry { kind, origin, bearer });
        Ok(())
    }

    /// The credential of `kind` for where `url` goes, or the reason there is none. The one function that
    /// chooses a credential.
    pub fn for_url(&self, url: &Url, kind: CredKind) -> Result<Authorization, CredError> {
        let origin = Origin::of(url)?;
        match self.entries.iter().find(|e| e.kind == kind && e.origin == origin) {
            Some(entry) => Ok(Authorization { bearer: entry.bearer.clone(), origin, kind }),
            None => Err(CredError::NoCredentialForOrigin { kind, origin: origin.shown() }),
        }
    }
}

/// A credential chosen for one origin. It can be put on a request, by [`AuthedRequest::apply`] and nowhere
/// else, and only on a request going to that origin; it prints as nothing.
#[derive(Clone)]
pub struct Authorization {
    bearer: HeaderValue,
    origin: Origin,
    kind: CredKind,
}

impl std::fmt::Debug for Authorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Authorization(for {})", self.origin.shown())
    }
}

/// A request that can carry a credential.
pub trait AuthedRequest: Sized {
    /// Put `auth` on the request, in place of any `Authorization` header it has. Refused for a request that
    /// is going to another origin than the one `auth` was chosen for.
    fn apply(self, auth: Authorization) -> Result<Self, CredError>;

    /// Choose the credential of `kind` for where the request is going ([`Creds::for_url`]) and put it on.
    /// Fails without anything being sent when there is none. The last step before `send`: the URL and the
    /// body are the request's own, and it is the URL that decides.
    fn with_creds(self, creds: &Creds, kind: CredKind) -> Result<Self, CredError>;
}

impl AuthedRequest for RequestBuilder {
    fn apply(self, auth: Authorization) -> Result<Self, CredError> {
        let (client, request) = self.build_split();
        let mut request = request.map_err(|e| CredError::BadRequest(e.without_url().to_string()))?;
        let going_to = Origin::of(request.url())?;
        if going_to != auth.origin {
            return Err(CredError::NoCredentialForOrigin { kind: auth.kind, origin: going_to.shown() });
        }
        request.headers_mut().insert(AUTHORIZATION, auth.bearer);
        Ok(RequestBuilder::from_parts(client, request))
    }

    fn with_creds(self, creds: &Creds, kind: CredKind) -> Result<Self, CredError> {
        let (client, request) = self.build_split();
        let mut request = request.map_err(|e| CredError::BadRequest(e.without_url().to_string()))?;
        let auth = creds.for_url(request.url(), kind)?;
        request.headers_mut().insert(AUTHORIZATION, auth.bearer);
        Ok(RequestBuilder::from_parts(client, request))
    }
}

/// How the desktop names itself to a provider.
const USER_AGENT: &str = concat!("oaiy-desktop/", env!("CARGO_PKG_VERSION"));

/// A blocking client as a credentialed lane keeps it: the settings every lane shares ([`super::net::blocking_builder`]),
/// and redirects OFF, so a server a credential was given to cannot send it on to another by a redirect (a client that
/// follows one drops the header only for another host or port; a change of scheme alone keeps it); a user agent that
/// is the same everywhere; and no cookie store (the client is built without that feature, and the tests look for the
/// header). The credential is not on the client: it goes on each request ([`AuthedRequest`]).
pub fn client_builder(keep: Keep) -> reqwest::blocking::ClientBuilder {
    super::net::blocking_builder(keep).redirect(reqwest::redirect::Policy::none()).user_agent(USER_AGENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::testkit::{Provider, Reply};

    fn url(text: &str) -> Url {
        Url::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    /// The origin `for_url` finds a credential for, or what it says instead.
    fn asked(creds: &Creds, target: &str, kind: CredKind) -> Result<(), CredError> {
        creds.for_url(&url(target), kind).map(|_| ())
    }

    fn refused(origin: &str, kind: CredKind) -> Result<(), CredError> {
        Err(CredError::NoCredentialForOrigin { kind, origin: origin.to_string() })
    }

    #[test]
    fn a_credential_goes_to_its_own_origin_and_to_no_look_alike_of_it() {
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "https://formlogic.com/api/v1", "flk_secret").unwrap();
        let p = CredKind::Provider;

        // The same origin, however it is spelled and whatever follows it: the path and the query are not part of it,
        // a port that is the scheme's own is the same as none, and the host is read as the URL standard reads it.
        for same in [
            "https://formlogic.com/api/v1/desktop-connections",
            "https://formlogic.com",
            "https://formlogic.com:443/x",
            "https://FormLogic.COM/x?y=1#z",
            "HTTPS://formlogic.com/x",
        ] {
            assert_eq!(asked(&creds, same, p), Ok(()), "{same}");
        }
        // Anything that is another origin, or that looks like this one and is not.
        for (other, said) in [
            ("http://formlogic.com/api/v1", "http://formlogic.com"),
            ("https://formlogic.com:8443/api/v1", "https://formlogic.com:8443"),
            ("https://api.formlogic.com/", "https://api.formlogic.com"),
            ("https://formlogic.com./", "https://formlogic.com."),
            ("https://formlogic.com.evil.test/", "https://formlogic.com.evil.test"),
            ("https://evilformlogic.com/", "https://evilformlogic.com"),
            ("https://formlogic.com%2eevil.test/", "https://formlogic.com.evil.test"),
            ("https://formlogic.co/", "https://formlogic.co"),
        ] {
            assert_eq!(asked(&creds, other, p), refused(said, p), "{other}");
        }
        // A user name or a password is refused whichever host it names, even this one: `formlogic.com@evil.test` reads as
        // formlogic.com to a person and goes to evil.test.
        for with_userinfo in ["https://user:pw@formlogic.com/", "https://user@formlogic.com/", "https://formlogic.com:pw@formlogic.com/", "https://formlogic.com@evil.test/"] {
            assert_eq!(asked(&creds, with_userinfo, p), Err(CredError::UserInfo), "{with_userinfo}");
        }
        // Not an origin a credential goes to at all.
        for not_http in ["ftp://formlogic.com/", "file:///c:/formlogic.com", "data:text/plain,formlogic.com", "mailto:a@formlogic.com", "wss://formlogic.com/"] {
            assert!(matches!(asked(&creds, not_http, p), Err(CredError::NotAnOrigin(_))), "{not_http}");
        }
    }

    #[test]
    fn an_internationalised_host_is_the_origin_its_ascii_form_is_and_no_other() {
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "https://b\u{fc}cher.example", "flk_secret").unwrap();
        let p = CredKind::Provider;
        for same in ["https://xn--bcher-kva.example/", "https://b\u{fc}cher.example/x", "https://B\u{dc}CHER.example/x"] {
            assert_eq!(asked(&creds, same, p), Ok(()), "{same}");
        }
        for other in ["https://bucher.example/", "https://xn--bcher-kva.example.evil.test/", "https://b\u{fc}cher.example.evil.test/"] {
            assert!(matches!(asked(&creds, other, p), Err(CredError::NoCredentialForOrigin { .. })), "{other}");
        }
    }

    #[test]
    fn an_ipv6_host_is_the_origin_its_address_is_and_no_other_form_of_it() {
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "http://[::1]:8080", "flk_secret").unwrap();
        let p = CredKind::Provider;
        for same in ["http://[::1]:8080/x", "http://[0:0:0:0:0:0:0:1]:8080/x", "http://[0000::0001]:8080/"] {
            assert_eq!(asked(&creds, same, p), Ok(()), "{same}");
        }
        for other in ["http://[::1]:8081/", "http://[::1]/", "http://[::2]:8080/", "http://[::ffff:127.0.0.1]:8080/", "http://127.0.0.1:8080/", "http://localhost:8080/", "https://[::1]:8080/"] {
            assert!(matches!(asked(&creds, other, p), Err(CredError::NoCredentialForOrigin { .. })), "{other}");
        }
    }

    #[test]
    fn the_kind_is_named_by_the_call_and_a_credential_of_another_kind_is_never_found() {
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "https://formlogic.com", "flk_provider").unwrap();
        creds.register(CredKind::Relay, "https://relay.example.com", "oaiyrt1.relay").unwrap();
        let (provider, relay) = ("https://formlogic.com/api/v1/x", "https://relay.example.com/v1/poll");
        assert_eq!(asked(&creds, provider, CredKind::Provider), Ok(()));
        assert_eq!(asked(&creds, relay, CredKind::Relay), Ok(()));
        // The provider's key is not the relay's to carry, and the relay's token is not the provider's.
        assert_eq!(asked(&creds, relay, CredKind::Provider), refused("https://relay.example.com", CredKind::Provider));
        assert_eq!(asked(&creds, provider, CredKind::Relay), refused("https://formlogic.com", CredKind::Relay));
        assert_eq!(asked(&creds, provider, CredKind::Issuer), refused("https://formlogic.com", CredKind::Issuer), "none registered for that kind");
    }

    #[test]
    fn a_provider_and_an_issuer_may_share_an_origin_and_each_call_gets_its_own() {
        // The legacy FormLogic issuer is at the provider's own address (`https://formlogic.com/api/v1`).
        let server = Provider::start(|_| Reply::ok("{}"));
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, &server.base, "flk_PROVIDER").unwrap();
        creds.register(CredKind::Issuer, &format!("{}/api/v1", server.base), "flk_ISSUER").unwrap();

        let http = client_builder(Keep::Between).build().unwrap();
        for (path, kind) in [("/api/v1/desktop-connections", CredKind::Provider), ("/api/v1/aokie-companion/admission", CredKind::Issuer), ("/api/v1/desktop-connections", CredKind::Provider)] {
            http.post(format!("{}{path}", server.base)).with_creds(&creds, kind).unwrap().send().unwrap();
        }
        let bearers: Vec<(String, String)> = server.requests().iter().map(|r| (r.target.clone(), r.header("authorization").unwrap_or("none").to_string())).collect();
        assert_eq!(
            bearers,
            [
                ("/api/v1/desktop-connections".to_string(), "Bearer flk_PROVIDER".to_string()),
                ("/api/v1/aokie-companion/admission".to_string(), "Bearer flk_ISSUER".to_string()),
                ("/api/v1/desktop-connections".to_string(), "Bearer flk_PROVIDER".to_string()),
            ]
        );
    }

    #[test]
    fn a_relay_origin_is_unique_among_all_the_origins_registered_in_either_order() {
        let relay = CredKind::Relay;
        // A relay at the provider's own origin, or the issuer's, whichever came first.
        for (first, second) in [(CredKind::Provider, relay), (CredKind::Issuer, relay), (relay, CredKind::Provider), (relay, CredKind::Issuer)] {
            let mut creds = Creds::default();
            creds.register(first, "https://formlogic.com", "a").unwrap();
            let shared = creds.register(second, "https://formlogic.com", "b").unwrap_err();
            assert_eq!(shared, CredError::RelayOriginShared { origin: "https://formlogic.com".into() }, "{first:?} then {second:?}");
        }
        // The same origin under another spelling is the same origin.
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "https://relay.example.com", "a").unwrap();
        for spelling in ["https://RELAY.example.com:443", "https://relay.example.com/"] {
            assert!(matches!(creds.register(relay, spelling, "b"), Err(CredError::RelayOriginShared { .. })), "{spelling}");
        }
        // Another origin is fine, and two relays are two origins.
        creds.register(relay, "https://relay.example.net", "b").unwrap();
        assert!(matches!(creds.register(relay, "https://relay.example.net:443", "c"), Err(CredError::Duplicate { kind: CredKind::Relay, .. })));
        creds.register(relay, "https://relay.example.net:8443", "c").unwrap();
        // One credential for each origin and kind.
        assert!(matches!(creds.register(CredKind::Provider, "https://relay.example.com", "d"), Err(CredError::Duplicate { kind: CredKind::Provider, .. })));
    }

    #[test]
    fn a_relays_address_is_its_hostname_alone() {
        let mut creds = Creds::default();
        for with_path in ["https://relay.example.com/relay", "https://relay.example.com/?x=1", "https://relay.example.com/#a", "https://relay.example.com/v1/"] {
            assert!(matches!(creds.register(CredKind::Relay, with_path, "t"), Err(CredError::RelayHasPath { .. })), "{with_path}");
        }
        creds.register(CredKind::Relay, "https://relay.example.com/", "t").unwrap();
        // A provider is served under a path often enough: that is not refused.
        creds.register(CredKind::Provider, "https://formlogic.com/api/v1", "t").unwrap();
    }

    #[test]
    fn a_secret_that_cannot_be_sent_is_refused_when_it_is_registered_and_none_is_ever_printed() {
        let mut creds = Creds::default();
        for bad in ["", "   ", "has\nnewline", "bell\u{7}"] {
            assert_eq!(creds.register(CredKind::Provider, "https://formlogic.com", bad), Err(CredError::Unusable), "{bad:?}");
        }
        creds.register(CredKind::Provider, "https://formlogic.com", "flk_TOPSECRET").unwrap();
        let auth = creds.for_url(&url("https://formlogic.com/x"), CredKind::Provider).unwrap();
        for printed in [format!("{creds:?}"), format!("{auth:?}"), format!("{:?}", auth.clone())] {
            assert!(!printed.contains("TOPSECRET"), "{printed}");
        }
        // And what is said of a refusal quotes no secret either, nor the address's user name.
        let said = creds.for_url(&url("https://evil.test/x"), CredKind::Provider).unwrap_err().to_string();
        assert!(!said.contains("TOPSECRET") && said.contains("https://evil.test"), "{said}");
        let said = creds.for_url(&url("https://user:hunter2@formlogic.com/"), CredKind::Provider).unwrap_err().to_string();
        assert!(!said.contains("hunter2"), "{said}");
    }

    #[test]
    fn a_request_to_an_origin_with_no_credential_is_never_sent() {
        let server = Provider::start(|_| Reply::ok("{}"));
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, "https://formlogic.com", "flk_secret").unwrap();
        let http = client_builder(Keep::Between).build().unwrap();
        let refused = http.get(format!("{}/x", server.base)).with_creds(&creds, CredKind::Provider).unwrap_err();
        assert!(matches!(refused, CredError::NoCredentialForOrigin { .. }), "{refused}");
        assert!(server.requests().is_empty(), "nothing reached the server: {:?}", server.lines());
    }

    #[test]
    fn a_credential_replaces_an_authorization_the_request_had_and_is_put_on_no_other_origin() {
        let (right, wrong) = (Provider::start(|_| Reply::ok("{}")), Provider::start(|_| Reply::ok("{}")));
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, &right.base, "flk_RIGHT").unwrap();
        let http = client_builder(Keep::Between).build().unwrap();

        http.get(format!("{}/x", right.base)).header("authorization", "Bearer someone-elses").with_creds(&creds, CredKind::Provider).unwrap().send().unwrap();
        assert_eq!(right.requests()[0].header("authorization"), Some("Bearer flk_RIGHT"));

        // A credential chosen for one origin is not put on a request to another.
        let chosen = creds.for_url(&url(&format!("{}/x", right.base)), CredKind::Provider).unwrap();
        let refused = http.get(format!("{}/x", wrong.base)).apply(chosen).unwrap_err();
        assert!(matches!(refused, CredError::NoCredentialForOrigin { .. }), "{refused}");
        assert!(wrong.requests().is_empty());
    }

    #[test]
    fn a_credentialed_client_follows_no_redirect_sends_one_user_agent_and_keeps_no_cookie() {
        // The credential went to the server the request was for; a server that answers with a redirect does not get
        // to move it. (A client that follows one drops the header for another host or port and keeps it for a change
        // of scheme alone; this one does not follow at all.)
        let elsewhere = Provider::start(|_| Reply::ok("{}"));
        let moved_to = format!("{}/moved", elsewhere.base);
        let home = Provider::start(move |req| {
            if req.target == "/cookie" {
                Reply { status: 200, headers: vec![("Set-Cookie".into(), "session=1".into())], body: "{}".into() }
            } else {
                Reply::redirect(307, &moved_to)
            }
        });
        let mut creds = Creds::default();
        creds.register(CredKind::Provider, &home.base, "flk_secret").unwrap();
        let http = client_builder(Keep::Between).build().unwrap();

        let reply = http.post(format!("{}/beat", home.base)).with_creds(&creds, CredKind::Provider).unwrap().send().unwrap();
        assert_eq!(reply.status().as_u16(), 307, "the redirect is the answer, not followed");
        assert!(elsewhere.requests().is_empty(), "nothing was sent where the redirect pointed: {:?}", elsewhere.lines());

        http.get(format!("{}/cookie", home.base)).send().unwrap();
        http.get(format!("{}/after", home.base)).send().unwrap();
        let seen = home.requests();
        assert!(seen.iter().all(|r| r.header("user-agent") == Some(USER_AGENT)), "{:?}", seen.iter().map(|r| r.header("user-agent")).collect::<Vec<_>>());
        assert!(seen.iter().all(|r| r.header("cookie").is_none()), "no cookie is kept for the next request");
    }

    /// The lanes that have been converted, with their source: a lane here applies no credential of its own.
    /// Converting a lane adds it to this list; the last step of the conversion replaces the list with a scan of
    /// every file of the crate (design 4.16.3).
    const CONVERTED: &[(&str, &str)] = &[("link/heartbeat.rs", include_str!("heartbeat.rs"))];

    #[test]
    fn a_converted_lane_reads_its_credential_nowhere_but_here() {
        for (name, source) in CONVERTED {
            // Everything before the module of tests: the code that runs.
            let source = source.replace("\r\n", "\n");
            let code = source.split("#[cfg(test)]\nmod tests").next().unwrap();
            assert!(code.len() < source.len(), "{name}: the module of tests was not found");
            for forbidden in ["bearer_auth", ".credential", "AUTHORIZATION", "\"Bearer", "Bearer {", "\"authorization\"", "basic_auth"] {
                assert!(!code.contains(forbidden), "{name} puts a credential on a request itself ({forbidden}): it goes through creds.rs");
            }
            assert!(code.contains("with_creds("), "{name} does not take its credential from creds.rs");
            assert!(code.contains("client_builder("), "{name} builds its client by hand: a credentialed client comes from creds::client_builder");
        }
    }
}
