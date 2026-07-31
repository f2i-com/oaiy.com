//! The authorization-code + PKCE ceremony, driven entirely by a descriptor.
//!
//! Nothing here names a provider. The base URL comes from the user, every path
//! and field name from [`super::descriptor::ConnectorDescriptor`], so a second
//! provider needs a JSON file and no code.
//!
//! Shape: bind a loopback listener on a kernel-assigned port, send the user's
//! browser to the provider's consent page with a PKCE challenge, and catch the
//! redirect ourselves. No secret ships in the binary (a public client), no code
//! is pasted by hand, and the authorization code is only usable by whoever
//! holds the verifier — this process.

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
use std::time::{Duration, Instant};

use super::descriptor::{AuthSpec, ConnectorDescriptor};

/// How long the user has to finish in the browser.
///
/// Long enough to log in and read the consent screen; short enough that a
/// listener left open by an abandoned attempt does not linger all session.
pub const LINK_TIMEOUT: Duration = Duration::from_secs(300);

/// A PKCE pair. The verifier never leaves this process until the exchange.
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

pub fn generate_pkce() -> Result<Pkce, String> {
    let verifier = random_token(32)?;
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    Ok(Pkce { verifier, challenge })
}

pub fn random_token(bytes: usize) -> Result<String, String> {
    let mut buf = vec![0u8; bytes];
    getrandom::getrandom(&mut buf).map_err(|e| format!("no system randomness: {e}"))?;
    Ok(URL_SAFE_NO_PAD.encode(buf))
}

/// Percent-encode for a query value. Unreserved set per RFC 3986.
fn q(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Join a user-supplied base to a descriptor path without letting either
/// smuggle in a different host.
pub fn join(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

/// The URL to open in the user's browser.
pub fn authorize_url(
    descriptor: &ConnectorDescriptor,
    base_url: &str,
    redirect_uri: &str,
    pkce: &Pkce,
    state: &str,
    device_label: Option<&str>,
) -> String {
    let AuthSpec::Oauth2Pkce(o) = &descriptor.auth;
    let mut url = join(base_url, &o.authorize_path);
    url.push('?');
    let mut parts: Vec<String> = Vec::new();
    // Descriptor extras first, so they can never overwrite the security
    // parameters below — a descriptor that set its own `code_challenge` would
    // otherwise silently disable PKCE.
    for (k, v) in &o.extra_authorize_params {
        parts.push(format!("{}={}", q(k), q(v)));
    }
    parts.push(format!("client_id={}", q(&o.client_id)));
    parts.push(format!("redirect_uri={}", q(redirect_uri)));
    parts.push(format!("scope={}", q(&o.scopes.join(" "))));
    parts.push(format!("code_challenge={}", q(&pkce.challenge)));
    parts.push("code_challenge_method=S256".to_string());
    parts.push(format!("state={}", q(state)));
    if let (Some(param), Some(label)) = (o.device_param.as_deref(), device_label) {
        if !label.trim().is_empty() {
            parts.push(format!("{}={}", q(param), q(label.trim())));
        }
    }
    url.push_str(&parts.join("&"));
    url
}

/// A bound loopback listener waiting for the provider's redirect.
pub struct Loopback {
    listener: TcpListener,
    pub redirect_uri: String,
}

impl Loopback {
    /// Bind 127.0.0.1 on a kernel-assigned port.
    ///
    /// Loopback only — never 0.0.0.0. The redirect carries an authorization
    /// code, and binding a routable interface would offer it to the network.
    pub fn bind(callback_path: &str) -> Result<Self, String> {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))
            .map_err(|e| format!("could not open a local callback port: {e}"))?;
        let port = listener
            .local_addr()
            .map_err(|e| format!("could not read the callback port: {e}"))?
            .port();
        Ok(Self {
            listener,
            redirect_uri: format!("http://127.0.0.1:{port}{callback_path}"),
        })
    }

    /// Wait for the browser to arrive, and answer it with a page the user sees.
    ///
    /// Returns the query parameters. Requests for anything other than the
    /// callback path are answered 404 and ignored rather than ending the wait:
    /// browsers speculatively fetch `/favicon.ico`, and treating that as the
    /// callback would abort a ceremony that had not happened yet.
    pub fn wait(&self, callback_path: &str, deadline: Instant) -> Result<HashMap<String, String>, String> {
        self.listener
            .set_nonblocking(true)
            .map_err(|e| format!("callback listener: {e}"))?;
        loop {
            if Instant::now() >= deadline {
                return Err("timed out waiting for the browser".into());
            }
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).ok();
                    match Self::serve(stream, callback_path) {
                        Ok(Some(params)) => return Ok(params),
                        // Not the callback — keep waiting.
                        Ok(None) => continue,
                        Err(_) => continue,
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(80));
                }
                Err(e) => return Err(format!("callback listener failed: {e}")),
            }
        }
    }

    fn serve(
        mut stream: std::net::TcpStream,
        callback_path: &str,
    ) -> Result<Option<HashMap<String, String>>, String> {
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .map_err(|e| format!("callback read failed: {e}"))?;

        // "GET /callback?code=..&state=.. HTTP/1.1"
        let target = request_line.split_whitespace().nth(1).unwrap_or("");
        let (path, query) = match target.split_once('?') {
            Some((p, q)) => (p, q),
            None => (target, ""),
        };
        if path != callback_path {
            let _ = stream.write_all(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            return Ok(None);
        }

        let params = parse_query(query);
        let (title, detail) = if params.contains_key("error") {
            (
                "Linking was refused",
                "You can close this tab and try again from OAIY Desktop.",
            )
        } else {
            ("You're linked", "You can close this tab and return to OAIY Desktop.")
        };
        let body = format!(
            "<!doctype html><meta charset=utf-8><title>{title}</title>\
             <body style=\"font:15px/1.5 system-ui,sans-serif;display:grid;place-items:center;\
             height:100vh;margin:0;background:#0b0f14;color:#e6edf3\">\
             <div style=\"text-align:center;max-width:28rem;padding:2rem\">\
             <h1 style=\"font-size:1.25rem;margin:0 0 .5rem\">{title}</h1>\
             <p style=\"margin:0;opacity:.75\">{detail}</p></div>"
        );
        let _ = stream.write_all(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        );
        let _ = stream.flush();
        Ok(Some(params))
    }
}

pub fn parse_query(query: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        out.insert(percent_decode(k), percent_decode(v));
    }
    out
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// What a completed link produced.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExchangedCredential {
    pub credential: String,
    pub account_id: Option<String>,
    pub account_name: Option<String>,
    pub granted_scopes: Option<String>,
}

/// Redeem the authorization code.
///
/// Blocking: the whole ceremony runs on its own thread because it waits on a
/// human. A blocking client keeps that one flow readable instead of colouring
/// the call stack async for a single request.
pub fn exchange_code(
    descriptor: &ConnectorDescriptor,
    base_url: &str,
    redirect_uri: &str,
    code: &str,
    verifier: &str,
) -> Result<ExchangedCredential, String> {
    let AuthSpec::Oauth2Pkce(o) = &descriptor.auth;
    let url = join(base_url, &o.token_path);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| format!("could not build the token client: {e}"))?;

    let form = [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("code_verifier", verifier),
        ("redirect_uri", redirect_uri),
        ("client_id", o.client_id.as_str()),
    ];
    let resp = client
        .post(&url)
        .form(&form)
        .send()
        .map_err(|e| format!("could not reach the token endpoint: {e}"))?;

    let status = resp.status();
    let value: serde_json::Value = resp
        .json()
        .map_err(|e| format!("the token endpoint returned a non-JSON response: {e}"))?;
    if !status.is_success() {
        // OAuth error bodies are `{error, error_description}`; surface the
        // description because "invalid_grant" alone tells the user nothing.
        let msg = value
            .get("error_description")
            .and_then(|v| v.as_str())
            .or_else(|| value.get("error").and_then(|v| v.as_str()))
            .unwrap_or("the provider refused the exchange");
        return Err(format!("HTTP {}: {msg}", status.as_u16()));
    }

    let spec = &o.token_response;
    let pick = |field: &Option<String>| -> Option<String> {
        field
            .as_ref()
            .and_then(|f| value.get(f))
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .filter(|s| !s.is_empty())
    };
    let credential = spec
        .credential_fields
        .iter()
        .find_map(|f| value.get(f).and_then(|v| v.as_str()).filter(|s| !s.is_empty()))
        .ok_or_else(|| {
            format!(
                "the exchange succeeded but carried none of the expected credential fields ({})",
                spec.credential_fields.join(", ")
            )
        })?
        .to_string();

    Ok(ExchangedCredential {
        credential,
        account_id: pick(&spec.account_id_field),
        account_name: pick(&spec.account_name_field),
        granted_scopes: pick(&spec.scope_field),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::descriptor;

    fn formlogic() -> ConnectorDescriptor {
        descriptor::load_all(std::path::Path::new("/nonexistent"))
            .into_iter()
            .next()
            .unwrap()
    }

    #[test]
    fn pkce_challenge_is_the_sha256_of_the_verifier() {
        // The provider recomputes exactly this. Getting it wrong fails at the
        // last step of a ceremony the user already completed by hand.
        let p = generate_pkce().unwrap();
        assert_eq!(p.verifier.len(), 43, "32 bytes base64url is 43 chars");
        let expect = URL_SAFE_NO_PAD.encode(Sha256::digest(p.verifier.as_bytes()));
        assert_eq!(p.challenge, expect);
        // Unreserved characters only, or it breaks in a query string.
        assert!(p
            .verifier
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~')));
    }

    #[test]
    fn two_ceremonies_never_share_a_verifier_or_state() {
        let a = generate_pkce().unwrap();
        let b = generate_pkce().unwrap();
        assert_ne!(a.verifier, b.verifier);
        assert_ne!(random_token(32).unwrap(), random_token(32).unwrap());
    }

    #[test]
    fn the_authorize_url_carries_every_parameter_the_provider_requires() {
        let d = formlogic();
        let p = Pkce { verifier: "v".into(), challenge: "chal".into() };
        let url = authorize_url(
            &d,
            "https://formlogic.com/",
            "http://127.0.0.1:5111/callback",
            &p,
            "st8",
            Some("  Reception PC  "),
        );
        assert!(url.starts_with("https://formlogic.com/oauth/authorize?"), "{url}");
        assert!(url.contains("client_id=formlogic-desktop"));
        assert!(url.contains("response_type=code"), "descriptor extras must appear");
        assert!(url.contains("code_challenge=chal"));
        assert!(url.contains("code_challenge_method=S256"));
        assert!(url.contains("state=st8"));
        // Encoded, not raw: a bare ':' or '/' in a query value is a bug.
        assert!(url.contains("redirect_uri=http%3A%2F%2F127.0.0.1%3A5111%2Fcallback"), "{url}");
        assert!(url.contains("scope=flows%3Aread%20flows%3Awrite"), "{url}");
        assert!(url.contains("device=Reception%20PC"), "trimmed and encoded: {url}");
        // The base's trailing slash must not produce a doubled one.
        assert!(!url.contains("com//oauth"), "{url}");
    }

    #[test]
    fn a_descriptor_cannot_disable_pkce_by_supplying_its_own_challenge() {
        // Extras are written first precisely so the security parameters win.
        // Providers see the LAST value for a repeated key in practice, and more
        // importantly ours is never dropped.
        let mut d = formlogic();
        let AuthSpec::Oauth2Pkce(o) = &mut d.auth;
        o.extra_authorize_params
            .insert("code_challenge".into(), "attacker".into());
        let p = Pkce { verifier: "v".into(), challenge: "real".into() };
        let url = authorize_url(&d, "https://x.example", "http://127.0.0.1:1/callback", &p, "s", None);
        let ours = url.find("code_challenge=real").expect("our challenge is present");
        let theirs = url.find("code_challenge=attacker").expect("theirs is present");
        assert!(ours > theirs, "ours must come last: {url}");
    }

    #[test]
    fn the_device_label_is_omitted_when_the_provider_does_not_ask_for_one() {
        let mut d = formlogic();
        let AuthSpec::Oauth2Pkce(o) = &mut d.auth;
        o.device_param = None;
        let p = Pkce { verifier: "v".into(), challenge: "c".into() };
        let url = authorize_url(&d, "https://x.example", "http://127.0.0.1:1/callback", &p, "s", Some("PC"));
        assert!(!url.contains("device="), "{url}");
    }

    #[test]
    fn the_callback_binds_loopback_only() {
        // The redirect carries an authorization code; a routable bind would
        // offer it to the network.
        let lb = Loopback::bind("/callback").unwrap();
        assert!(lb.redirect_uri.starts_with("http://127.0.0.1:"), "{}", lb.redirect_uri);
        assert!(lb.redirect_uri.ends_with("/callback"));
        assert!(lb.listener.local_addr().unwrap().ip().is_loopback());
    }

    #[test]
    fn query_parsing_decodes_and_survives_junk() {
        let p = parse_query("code=abc%2Fdef&state=a+b&empty=&flag");
        assert_eq!(p.get("code").map(String::as_str), Some("abc/def"));
        assert_eq!(p.get("state").map(String::as_str), Some("a b"));
        assert_eq!(p.get("empty").map(String::as_str), Some(""));
        assert_eq!(p.get("flag").map(String::as_str), Some(""));
        assert!(parse_query("").is_empty());
    }

    #[test]
    fn the_loopback_ignores_a_favicon_fetch_and_keeps_waiting() {
        // Browsers fetch /favicon.ico unprompted. Treating that as the callback
        // would abort the ceremony before the user had approved anything.
        let lb = Loopback::bind("/callback").unwrap();
        let uri = lb.redirect_uri.clone();
        let addr: String = uri.trim_start_matches("http://").split('/').next().unwrap().into();

        let noise = addr.clone();
        std::thread::spawn(move || {
            use std::io::Write as _;
            std::thread::sleep(Duration::from_millis(60));
            if let Ok(mut s) = std::net::TcpStream::connect(&noise) {
                let _ = s.write_all(b"GET /favicon.ico HTTP/1.1\r\nHost: x\r\n\r\n");
            }
            std::thread::sleep(Duration::from_millis(120));
            if let Ok(mut s) = std::net::TcpStream::connect(&noise) {
                let _ = s.write_all(b"GET /callback?code=THECODE&state=S HTTP/1.1\r\nHost: x\r\n\r\n");
            }
        });

        let params = lb
            .wait("/callback", Instant::now() + Duration::from_secs(10))
            .expect("the real callback should be caught");
        assert_eq!(params.get("code").map(String::as_str), Some("THECODE"));
        assert_eq!(params.get("state").map(String::as_str), Some("S"));
    }

    #[test]
    fn waiting_past_the_deadline_gives_up_rather_than_hanging() {
        let lb = Loopback::bind("/callback").unwrap();
        let err = lb
            .wait("/callback", Instant::now() + Duration::from_millis(200))
            .unwrap_err();
        assert!(err.contains("timed out"), "{err}");
    }
}
