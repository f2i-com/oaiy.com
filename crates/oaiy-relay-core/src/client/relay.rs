//! The relay client: the calls of README section 5 that a desktop and a phone make, over the traits of this module's siblings.
//!
//! **What it enforces, so that a caller cannot get it wrong:**
//!
//! - **No token without a proof (P9, README 8 rule 3 and 9.1 rule 7).** A request that carries a bearer is refused with [`ClientError::NotProved`] unless the relay's identity was
//!   proved against the pinned key within the last 600 seconds (the loop's schedule is 300), and with [`ClientError::Suspect`] after a proof that failed, until one succeeds.
//!   Nothing is sent. The enrolment and the pairing calls prove first, by themselves.
//! - **A bearer goes to the origin it was made for, and only there.** Every URL is built from the one [`RelayUrl`] the client was made with, and a `3xx` is a failure and is
//!   never followed (the [`HttpClient`] contract).
//! - **A response is read as bytes and judged by what it is**, not by what the relay says it is: an error answer's `message` is returned as data (and never shown as anything
//!   but "message from your relay (unverified)"), a `2xx` whose body is not what its schema says is [`ClientError::BadAnswer`].
//! - **The relay clock** is sampled from `X-OAIY-Time` on every response, when the response is received.
//!
//! All methods take `&self`: the state behind them (the proof, the clock offset) is a mutex that is never held across an HTTP call, so a poll loop on one thread and a post
//! from another share one client.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::admission::{MobileAdmission, MobileExpect, MobileRequest, PluginAdmission, PluginRequest};
use crate::b64;
use crate::enrol::{self, Enrolled, EnrolmentKey};
use crate::error::Error;
use crate::ids::{self, Token};
use crate::info::{self, Info, Proved};
use crate::json::{self, Json};
use crate::keys::{VerifyKey, X25519Public};
use crate::poll::{self, Answer};
use crate::url::RelayUrl;

use super::clock::{Clock, OffsetClock, Rng};
use super::http::{Cancel, HttpClient, HttpRequest, HttpResponse, Method, TransportError};
use super::status::Health;

/// The longest a proof may be before an authenticated call is refused: twice the loop's schedule (`PROOF_EVERY_S`), a safety net and not the schedule.
pub const PROOF_MAX_AGE_S: u64 = 2 * poll::PROOF_EVERY_S;
/// The most a response body may be: the poll's `maxBytes` ceiling (1 MiB) and room for the rest of the answer.
pub const MAX_RESPONSE_BYTES: usize = 1_048_576 + 65_536;

/// How the client presents itself and how long it waits.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// `X-OAIY-Client: <product>/<version>`: diagnostics only, never used for authorisation.
    pub product: String,
    /// `X-OAIY-Level`: the protocol level this client implements (1).
    pub level: u32,
    /// The timeout of every call but a poll (whose timeout is its `wait + 10`).
    pub request_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig { product: format!("oaiy-relay-core/{}", env!("CARGO_PKG_VERSION")), level: 1, request_timeout: Duration::from_secs(15) }
    }
}

/// An error answer of the relay (`error.schema.json`): what it said, as data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayError {
    /// The HTTP status.
    pub status: u16,
    /// `error.code`, when the body has one.
    pub code: Option<String>,
    /// `error.message`: one sentence the relay wrote. **Unverified**: show it as plain text under "message from your relay (unverified)" and act on nothing in it.
    pub message: Option<String>,
    /// The wait the relay asked for, read as P6 reads it.
    pub retry_after: Option<u64>,
    /// `error.rule` (`gap` or `in_flight`) of a refused consumer poll.
    pub rule: Option<String>,
}

/// What a call can fail with.
#[derive(Debug, Clone, PartialEq)]
pub enum ClientError {
    /// No response.
    Transport(TransportError),
    /// The relay answered with an error status.
    Relay(RelayError),
    /// A `2xx` whose body is not what its schema says.
    BadAnswer(&'static str),
    /// The relay's identity has not been proved (or not recently enough): no token was sent.
    NotProved,
    /// The relay's proof did not verify: "not who it was". No token is sent until a proof verifies.
    Suspect,
    /// No relay key is pinned yet (a typed pairing learns it from the offer).
    NoPin,
    /// A value this client was asked to send is not what the protocol allows.
    Request(Error),
    /// A value the relay sent failed a check of the protocol layer.
    Protocol(Error),
    /// The call was cancelled.
    Cancelled,
}

impl core::fmt::Display for ClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ClientError::Transport(e) => write!(f, "no response: {e}"),
            ClientError::Relay(e) => write!(f, "the relay answered {} {}", e.status, e.code.as_deref().unwrap_or("")),
            ClientError::BadAnswer(w) => write!(f, "the relay's answer is not what it should be: {w}"),
            ClientError::NotProved => f.write_str("the relay's identity has not been proved"),
            ClientError::Suspect => f.write_str("the relay is not who it was"),
            ClientError::NoPin => f.write_str("no relay key is pinned"),
            ClientError::Request(e) => write!(f, "invalid request: {e}"),
            ClientError::Protocol(e) => write!(f, "{e}"),
            ClientError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<TransportError> for ClientError {
    fn from(e: TransportError) -> Self {
        if e == TransportError::Cancelled {
            ClientError::Cancelled
        } else {
            ClientError::Transport(e)
        }
    }
}

impl ClientError {
    /// The relay's error answer, when this is one.
    pub fn relay(&self) -> Option<&RelayError> {
        if let ClientError::Relay(e) = self {
            Some(e)
        } else {
            None
        }
    }

    /// The relay's error code, when this is an error answer that has one.
    pub fn code(&self) -> Option<&str> {
        self.relay().and_then(|e| e.code.as_deref())
    }
}

/// Why a proof failed (P9): no answer is a failure like any other, an answer that does not verify is a stop.
#[derive(Debug, Clone, PartialEq)]
pub enum ProveError {
    /// No answer to the proof request, or an answer that is not a proof (a status other than `200`).
    NoAnswer(ClientError),
    /// An answer that does not verify: a replayed body and signature, another key, a proof over another time.
    Invalid(Error),
}

/// A consumer poll's parameters (README 5.1).
#[derive(Debug, Clone)]
pub struct PollRequest {
    /// `since`: the highest `seq` accepted, which acknowledges every item up to it.
    pub since: u64,
    /// `epoch`: the epoch the client last saw, sent back byte for byte; `None` is no check (the retry after a first `400`).
    pub epoch: Option<String>,
    /// `wait`: seconds the server may hold the request: `info.wait.default` for a poll that is held, 0 for a short poll.
    pub wait_s: u64,
    /// `limit`: 1 to 64 (32 by default).
    pub limit: u32,
}

/// The answer to a poll, as the rules read it. A transport failure is not an error here: it is an answer ("no response", P2).
#[derive(Debug, Clone)]
pub struct PollReply {
    /// The status, or `None` when no response came back.
    pub status: Option<u16>,
    /// The headers, names in lower case.
    pub headers: Vec<(String, String)>,
    /// The body parsed as JSON, or `None` when there was none or it was not JSON.
    pub body: Option<Json>,
    /// Why there is no response, when there is none.
    pub transport: Option<TransportError>,
}

impl PollReply {
    /// The view the rules take.
    pub fn answer(&self) -> Answer<'_> {
        Answer { status: self.status, headers: &self.headers, body: self.body.as_ref() }
    }
}

struct ProofRecord {
    at: Duration,
    info: Info,
}

#[derive(Default)]
struct Shared {
    pin: Option<String>,
    proof: Option<ProofRecord>,
    suspect: bool,
    clock: OffsetClock,
    info_read: Option<Info>,
}

/// The relay client.
pub struct RelayClient {
    url: RelayUrl,
    http: Arc<dyn HttpClient>,
    clock: Arc<dyn Clock>,
    rng: Mutex<Box<dyn Rng>>,
    config: ClientConfig,
    state: Mutex<Shared>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A poisoned mutex holds state that a panicking thread was in the middle of changing; the data are plain values and are still valid, so the lock is taken anyway.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl RelayClient {
    /// A client for `url`, which proves the relay against `pinned_thumbprint` (the `f` of the enrolment or pairing key, or the stored profile's `relay_thumbprint`; `None`
    /// until a typed pairing learns it from the offer: [`RelayClient::pin`]).
    pub fn new(
        url: RelayUrl,
        pinned_thumbprint: Option<String>,
        http: Arc<dyn HttpClient>,
        clock: Arc<dyn Clock>,
        rng: Box<dyn Rng>,
        config: ClientConfig,
    ) -> RelayClient {
        RelayClient { url, http, clock, rng: Mutex::new(rng), config, state: Mutex::new(Shared { pin: pinned_thumbprint, ..Default::default() }) }
    }

    /// The relay's URL.
    pub fn url(&self) -> &RelayUrl {
        &self.url
    }

    /// The clock the client was made with.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }

    /// The configuration.
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Pins the relay key's thumbprint if none is pinned yet; a different one than the pinned is an error (a pin is set once).
    pub fn pin(&self, thumbprint: &str) -> Result<(), ClientError> {
        if !ids::is_thumbprint(thumbprint) {
            return Err(ClientError::Request(Error::Invalid("pin")));
        }
        let mut s = lock(&self.state);
        match &s.pin {
            Some(p) if p != thumbprint => Err(ClientError::Request(Error::Mismatch("the relay key is already pinned to another"))),
            _ => {
                s.pin = Some(thumbprint.to_string());
                Ok(())
            }
        }
    }

    /// The pinned thumbprint.
    pub fn pinned(&self) -> Option<String> {
        lock(&self.state).pin.clone()
    }

    /// A random draw `u` for a jittered pause, from 0 up to but not including 1.
    pub fn jitter(&self) -> f64 {
        lock(&self.rng).unit()
    }

    /// Random bytes.
    pub fn random(&self, buf: &mut [u8]) {
        lock(&self.rng).fill(buf);
    }

    /// The `info` of the last good proof.
    pub fn info(&self) -> Option<Info> {
        lock(&self.state).proof.as_ref().map(|p| p.info.clone())
    }

    /// Seconds since the last good proof.
    pub fn proof_age_s(&self) -> Option<u64> {
        let now = self.clock.monotonic();
        lock(&self.state).proof.as_ref().map(|p| now.saturating_sub(p.at).as_secs())
    }

    /// True after a proof that did not verify and before one that does.
    pub fn is_suspect(&self) -> bool {
        lock(&self.state).suspect
    }

    /// The relay's clock now (`X-OAIY-Time` corrected as README section 1 says), or `None` before any sample.
    pub fn relay_now(&self) -> Option<i64> {
        let mono = self.clock.monotonic();
        lock(&self.state).clock.remote_now(self.clock.unix_now(), mono)
    }

    /// The relay's time if a sample was taken, else the local clock's: for a window that must be judged at all (a response that arrived judges a window with the time of
    /// the relay that sent it).
    pub fn relay_now_or_local(&self) -> i64 {
        self.relay_now().unwrap_or_else(|| self.clock.unix_now())
    }

    /// True when the relay's clock differs from ours by more than 60 seconds (a warning for the owner; nothing is judged by it).
    pub fn clock_mismatch(&self) -> bool {
        let mono = self.clock.monotonic();
        lock(&self.state).clock.mismatch(mono)
    }

    /// The relay offset in seconds, when sampled.
    pub fn relay_offset_s(&self) -> Option<f64> {
        let mono = self.clock.monotonic();
        lock(&self.state).clock.offset(mono)
    }

    fn sample_time(&self, response: &HttpResponse) {
        if let Some(t) = response.header("x-oaiy-time").and_then(|t| info::parse_time_header(t).ok()) {
            let (local, mono) = (self.clock.unix_now(), self.clock.monotonic());
            lock(&self.state).clock.observe(t, local, mono);
        }
    }

    /// One exchange, with the headers every request carries, and the relay clock sampled from the response.
    #[allow(clippy::too_many_arguments)]
    fn exchange(
        &self,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
        bearer: Option<&Token>,
        extra: &[(&str, String)],
        timeout: Duration,
        cancel: &Cancel,
    ) -> Result<HttpResponse, TransportError> {
        let mut headers = vec![
            ("Accept".to_string(), "application/json".to_string()),
            ("X-OAIY-Client".to_string(), self.config.product.clone()),
            ("X-OAIY-Level".to_string(), self.config.level.to_string()),
        ];
        if body.is_some() {
            headers.push(("Content-Type".to_string(), "application/json; charset=utf-8".to_string()));
        }
        if let Some(t) = bearer {
            headers.push(("Authorization".to_string(), format!("Bearer {}", t.expose())));
        }
        for (k, v) in extra {
            headers.push(((*k).to_string(), v.clone()));
        }
        let request =
            HttpRequest { method, url: self.url.join(path), headers, body, timeout, max_response_bytes: MAX_RESPONSE_BYTES, cancel: cancel.clone() };
        let response = self.http.send(&request)?;
        self.sample_time(&response);
        Ok(response)
    }

    fn gate(&self) -> Result<(), ClientError> {
        let now = self.clock.monotonic();
        let s = lock(&self.state);
        if s.suspect {
            return Err(ClientError::Suspect);
        }
        match &s.proof {
            Some(p) if now.saturating_sub(p.at).as_secs() <= PROOF_MAX_AGE_S => Ok(()),
            _ => Err(ClientError::NotProved),
        }
    }

    fn authed(
        &self,
        token: &Token,
        method: Method,
        path: &str,
        body: Option<Vec<u8>>,
        timeout: Option<Duration>,
        cancel: &Cancel,
    ) -> Result<HttpResponse, ClientError> {
        self.gate()?;
        Ok(self.exchange(method, path, body, Some(token), &[], timeout.unwrap_or(self.config.request_timeout), cancel)?)
    }

    /// The error answer of a response that is not a success: its status, and what its body said, as data.
    pub fn error_of(&self, response: &HttpResponse) -> RelayError {
        let body = json::parse(&response.body).ok();
        let headers = &response.headers;
        let answer = Answer { status: Some(response.status), headers, body: body.as_ref() };
        let err = body.as_ref().and_then(|b| b.get("error"));
        RelayError {
            status: response.status,
            code: err.and_then(|e| e.get_str("code")).map(str::to_string),
            message: err.and_then(|e| e.get_str("message")).map(|m| m.chars().take(200).collect()),
            retry_after: poll::retry_after(&answer, Some(self.clock.unix_now())),
            rule: err.and_then(|e| e.get_str("rule")).map(str::to_string),
        }
    }

    fn success_json(&self, response: &HttpResponse, ok: &[u16]) -> Result<Json, ClientError> {
        if !ok.contains(&response.status) {
            return Err(ClientError::Relay(self.error_of(response)));
        }
        json::parse(&response.body).map_err(|_| ClientError::BadAnswer("not JSON"))
    }

    // ------------------------------------------------------------------------------------------------------------ identity

    /// The identity proof (README 8, rule 2): `GET /v1/info` with a fresh nonce of 16 random bytes, verified against the pinned key. A proof that verifies sets the time of the
    /// proof, clears the suspicion, and returns the `info` (whose `wait` and `minClient` the loop then uses). **No token is sent by this call.**
    pub fn prove(&self, cancel: &Cancel) -> Result<Proved, ProveError> {
        let pin = lock(&self.state).pin.clone().ok_or(ProveError::NoAnswer(ClientError::NoPin))?;
        let mut nonce = [0u8; 16];
        self.random(&mut nonce);
        let response = self
            .exchange(Method::Get, "/v1/info", None, None, &[("X-OAIY-Nonce", b64::encode(&nonce))], self.config.request_timeout, cancel)
            .map_err(|e| ProveError::NoAnswer(e.into()))?;
        if response.status != 200 {
            return Err(ProveError::NoAnswer(ClientError::Relay(self.error_of(&response))));
        }
        let verdict = (|| {
            let proof = response.header("x-oaiy-proof").ok_or(Error::Invalid("no X-OAIY-Proof"))?;
            let time = response.header("x-oaiy-time").ok_or(Error::Invalid("no X-OAIY-Time"))?;
            info::verify_proof(&response.body, &nonce, time, proof, &pin)
        })();
        match verdict {
            Ok(proved) => {
                let mut s = lock(&self.state);
                s.suspect = false;
                s.proof = Some(ProofRecord { at: self.clock.monotonic(), info: proved.info.clone() });
                s.info_read = Some(proved.info.clone());
                Ok(proved)
            }
            Err(e) => {
                let mut s = lock(&self.state);
                s.suspect = true;
                s.proof = None;
                Err(ProveError::Invalid(e))
            }
        }
    }

    /// Proves the relay again when the last proof is older than the loop's schedule (300 seconds) or there is none: for a call made outside the poll loop.
    pub fn ensure_proved(&self, cancel: &Cancel) -> Result<(), ClientError> {
        match self.proof_age_s() {
            Some(age) if age < poll::PROOF_EVERY_S && !self.is_suspect() => Ok(()),
            _ => match self.prove(cancel) {
                Ok(_) => Ok(()),
                Err(ProveError::NoAnswer(e)) => Err(e),
                Err(ProveError::Invalid(_)) => Err(ClientError::Suspect),
            },
        }
    }

    /// `GET /v1/info` without a nonce (README 8.5: cached for ten minutes, and never what identity rests on): the document and, when the relay sent `X-OAIY-Sig`, its static
    /// signature checked against the pinned key. Used to re-read `minClient` after a `426`.
    pub fn read_info(&self, cancel: &Cancel) -> Result<Info, ClientError> {
        let pin = lock(&self.state).pin.clone().ok_or(ClientError::NoPin)?;
        let response = self.exchange(Method::Get, "/v1/info", None, None, &[], self.config.request_timeout, cancel)?;
        if response.status != 200 {
            return Err(ClientError::Relay(self.error_of(&response)));
        }
        let info = Info::parse(&response.body).map_err(ClientError::Protocol)?;
        if info.relay_key.thumbprint() != pin {
            return Err(ClientError::Suspect);
        }
        if let Some(sig) = response.header("x-oaiy-sig") {
            info::verify_static_signature(&info.relay_key, &response.body, sig).map_err(ClientError::Protocol)?;
        }
        lock(&self.state).info_read = Some(info.clone());
        Ok(info)
    }

    /// `GET /v1/health`: liveness, with no credential.
    pub fn health(&self, cancel: &Cancel) -> Result<Health, ClientError> {
        let response = self.exchange(Method::Get, "/v1/health", None, None, &[], self.config.request_timeout, cancel)?;
        if response.status != 200 {
            return Err(ClientError::Relay(self.error_of(&response)));
        }
        Health::parse(&response.body).map_err(ClientError::Protocol)
    }

    // ------------------------------------------------------------------------------------------------------------ enrolment

    /// Redeems an enrolment key (README 10.2): pins the key's `f`, proves the relay (**before anything is sent**), checks that `relayId` of the answer is the proved relay's, and
    /// returns the device id and token. The caller stores them (see [`crate::client::enrol_and_store`]).
    pub fn enroll(
        &self,
        key: &EnrolmentKey,
        name: &str,
        ed25519: &VerifyKey,
        x25519: &X25519Public,
        cancel: &Cancel,
    ) -> Result<Enrolled, ClientError> {
        if key.relay != self.url {
            return Err(ClientError::Request(Error::Mismatch("the enrolment key is for another relay")));
        }
        self.pin(&key.relay_thumbprint)?;
        let proved = match self.prove(cancel) {
            Ok(p) => p,
            Err(ProveError::NoAnswer(e)) => return Err(e),
            Err(ProveError::Invalid(_)) => return Err(ClientError::Suspect),
        };
        let mut nonce = [0u8; 16];
        self.random(&mut nonce);
        let request = enrol::build_request(key, name, ed25519, x25519, &nonce).map_err(ClientError::Request)?;
        let response = self.exchange(
            Method::Post,
            "/v1/enroll",
            Some(request.body.into_bytes()),
            None,
            &[("X-OAIY-Proof", request.proof)],
            self.config.request_timeout,
            cancel,
        )?;
        if response.status != 201 {
            return Err(ClientError::Relay(self.error_of(&response)));
        }
        let enrolled = Enrolled::parse(&response.body, key.role).map_err(ClientError::Protocol)?;
        if enrolled.relay_id != proved.info.relay_id {
            return Err(ClientError::BadAnswer("relayId is not the relay that was proved"));
        }
        Ok(enrolled)
    }

    // ------------------------------------------------------------------------------------------------------------ poll

    /// One consumer poll (`GET /v1/poll`), as the rules of P1 to P9 make it: with `wait` and `limit`, the stored `epoch` when there is one, and a timeout of `wait + 10`
    /// seconds (10 for a short poll). A poll that reaches its own timeout is "no response" like any other. `Err` only when the call cannot be made: no fresh proof.
    pub fn poll(&self, token: &Token, request: &PollRequest, cancel: &Cancel) -> Result<PollReply, ClientError> {
        let mut path = format!("/v1/poll?since={}", request.since);
        if let Some(e) = &request.epoch {
            path.push_str("&epoch=");
            path.push_str(e);
        }
        path.push_str(&format!("&wait={}&limit={}", request.wait_s, request.limit));
        let timeout = Duration::from_secs(request.wait_s + poll::POLL_TIMEOUT_EXTRA_S);
        self.gate()?;
        match self.exchange(Method::Get, &path, None, Some(token), &[], timeout, cancel) {
            Ok(r) => {
                let body = json::parse(&r.body).ok();
                Ok(PollReply { status: Some(r.status), headers: r.headers, body, transport: None })
            }
            Err(e) => Ok(PollReply { status: None, headers: Vec::new(), body: None, transport: Some(e) }),
        }
    }

    // ------------------------------------------------------------------------------------------------------------ items

    /// `POST /v1/items`: up to 64 items in one request. See [`PostItem`] for what is checked before anything is sent.
    pub fn post_items(&self, token: &Token, items: &[PostItem], cancel: &Cancel) -> Result<Vec<PostResult>, ClientError> {
        let info = self.info();
        let body = post_body(items, info.as_ref()).map_err(ClientError::Request)?;
        let response = self.authed(token, Method::Post, "/v1/items", Some(body.into_bytes()), None, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        parse_post_response(&doc, items)
    }

    /// `POST /v1/tokens/rotate`: a new token; the old one works for ten more minutes (`graceUntil`).
    pub fn rotate_token(&self, token: &Token, cancel: &Cancel) -> Result<(Token, u64), ClientError> {
        let response = self.authed(token, Method::Post, "/v1/tokens/rotate", Some(b"{}".to_vec()), None, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        let new = Token::parse(doc.get_str("token").ok_or(ClientError::BadAnswer("token"))?).map_err(ClientError::Protocol)?;
        let grace = doc.get_uint53("graceUntil").ok_or(ClientError::BadAnswer("graceUntil"))?;
        Ok((new, grace))
    }

    // ------------------------------------------------------------------------------------------------------------ admission

    /// `POST /v1/admission` with a phone's token: the phone's admission, read as the shipped phone reads it. `expect` is the phone's own session.
    pub fn admission_mobile(
        &self,
        token: &Token,
        request: &MobileRequest,
        expect: &MobileExpect<'_>,
        cancel: &Cancel,
    ) -> Result<MobileAdmission, ClientError> {
        let body = request.to_body().map_err(ClientError::Request)?;
        let response = self.authed(token, Method::Post, "/v1/admission", Some(body.into_bytes()), None, cancel)?;
        if response.status != 200 {
            return Err(ClientError::Relay(self.error_of(&response)));
        }
        MobileAdmission::parse(&response.body, expect, self.relay_now_or_local()).map_err(ClientError::Protocol)
    }

    /// `POST /v1/admission` with a desktop's token: the plugin's admission.
    pub fn admission_plugin(&self, token: &Token, request: &PluginRequest, cancel: &Cancel) -> Result<PluginAdmission, ClientError> {
        let body = request.to_body().map_err(ClientError::Request)?;
        let response = self.authed(token, Method::Post, "/v1/admission", Some(body.into_bytes()), None, cancel)?;
        if response.status != 200 {
            return Err(ClientError::Relay(self.error_of(&response)));
        }
        PluginAdmission::parse(&response.body, request, self.relay_now_or_local()).map_err(ClientError::Protocol)
    }
}

// ---------------------------------------------------------------------------------------------------------------- items

/// The `hdr` of an item (README 3, "hdr allow-list"): `re`, `ct`, `eph`, `kid`, `prio`, `n`, `sig`, and nothing else. Built with the methods and checked when the post is made.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Hdr(Vec<(&'static str, HdrValue)>);

/// A value of an `hdr` member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdrValue {
    /// A string.
    Str(String),
    /// A non-negative integer.
    Int(u64),
}

impl Hdr {
    /// An empty header (sent as no `hdr` member at all).
    pub fn new() -> Hdr {
        Hdr::default()
    }

    /// `re`: the id of the item this answers.
    pub fn re(mut self, id: impl Into<String>) -> Hdr {
        self.0.push(("re", HdrValue::Str(id.into())));
        self
    }

    /// `ct`: `text`, `json`, `sealed1`, `tunnel1` or `noise1`.
    pub fn ct(mut self, ct: &str) -> Hdr {
        self.0.push(("ct", HdrValue::Str(ct.to_string())));
        self
    }

    /// `kid`: an id of at most 64 characters.
    pub fn kid(mut self, kid: impl Into<String>) -> Hdr {
        self.0.push(("kid", HdrValue::Str(kid.into())));
        self
    }

    /// `prio`: 0 or 1.
    pub fn prio(mut self, prio: u64) -> Hdr {
        self.0.push(("prio", HdrValue::Int(prio)));
        self
    }

    /// `n`: a non-negative integer.
    pub fn n(mut self, n: u64) -> Hdr {
        self.0.push(("n", HdrValue::Int(n)));
        self
    }

    /// `sig`: the base64url of an Ed25519 signature (on a `ring`).
    pub fn sig(mut self, sig: impl Into<String>) -> Hdr {
        self.0.push(("sig", HdrValue::Str(sig.into())));
        self
    }

    fn get(&self, key: &str) -> Option<&HdrValue> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    fn to_json(&self) -> Result<Json, Error> {
        let mut members = Vec::new();
        for (k, v) in &self.0 {
            if self.0.iter().filter(|(o, _)| o == k).count() > 1 {
                return Err(Error::Invalid("hdr: a repeated member"));
            }
            let ok = match (*k, v) {
                ("re", HdrValue::Str(s)) => ids::is_item_id(s),
                ("ct", HdrValue::Str(s)) => ["text", "json", "sealed1", "tunnel1", "noise1"].contains(&s.as_str()),
                ("kid", HdrValue::Str(s)) => {
                    (1..=64).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
                }
                ("prio", HdrValue::Int(n)) => *n <= 1,
                ("n", HdrValue::Int(n)) => *n <= json::MAX_SAFE_INT,
                ("sig", HdrValue::Str(s)) => (1..=88).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
                _ => false,
            };
            if !ok {
                return Err(Error::Invalid("hdr: a member the relay would refuse"));
            }
            members.push((
                k.to_string(),
                match v {
                    HdrValue::Str(s) => Json::str(s.clone()),
                    HdrValue::Int(n) => Json::int(*n),
                },
            ));
        }
        let hdr = Json::Obj(members);
        if hdr.to_compact().len() > 512 {
            return Err(Error::Invalid("hdr: more than 512 bytes"));
        }
        Ok(hdr)
    }
}

/// An item to post (`post-request.schema.json`). What is checked before anything is sent: the address (`dev:<deviceId>` or `rbx:<rid>`), the lane (one a client may post to
/// through `POST /v1/items`), the id, the `ttl` (an integer from 1, within the lane's maximum when `info` says it), the `hdr` (the allow-list, 512 bytes), the body (UTF-8, within the
/// lane's cap), a `ring` with `hdr.sig`, and 1 to 64 items in at most 1 MiB. The relay checks them again; an error it can name is a mistake in the code that built the item.
#[derive(Debug, Clone)]
pub struct PostItem {
    /// `to`.
    pub to: String,
    /// `lane`: `cmd`, `res`, `ai.out`, `ctl`, `sync` or `ring`.
    pub lane: String,
    /// `id`, chosen by the sender, unique per (mailbox, lane, sender) while the relay remembers it: a retry uses the same id and the same body.
    pub id: String,
    /// `ttl` in seconds; `None` is the lane's default.
    pub ttl: Option<u64>,
    /// `hdr`.
    pub hdr: Hdr,
    /// `body`.
    pub body: String,
}

/// The lanes a client may post to through `POST /v1/items`.
pub const POSTABLE_LANES: [&str; 6] = ["cmd", "res", "ai.out", "ctl", "sync", "ring"];

fn post_body(items: &[PostItem], info: Option<&Info>) -> Result<String, Error> {
    if items.is_empty() || items.len() > 64 {
        return Err(Error::Invalid("post: 1 to 64 items"));
    }
    let mut list = Vec::new();
    for it in items {
        let to_ok = it.to.strip_prefix("dev:").is_some_and(|d| ids::is_device_id(d) || ids::is_provider_id(d))
            || it.to.strip_prefix("rbx:").is_some_and(ids::is_pid);
        if !to_ok || !POSTABLE_LANES.contains(&it.lane.as_str()) || !ids::is_item_id(&it.id) {
            return Err(Error::Invalid("post: to, lane or id"));
        }
        if let Some(lane) = info.and_then(|i| i.lane(&it.lane)) {
            if it.body.len() as u64 > lane.body || it.ttl.is_some_and(|t| t < lane.ttl_min || t > lane.ttl_max) {
                return Err(Error::Invalid("post: a body or ttl beyond what the relay serves for the lane"));
            }
        }
        if it.ttl == Some(0) {
            return Err(Error::Invalid("post: ttl"));
        }
        if it.lane == "ring" && it.hdr.get("sig").is_none() {
            return Err(Error::Invalid("post: a ring carries hdr.sig"));
        }
        let mut m = vec![
            ("to".to_string(), Json::str(it.to.clone())),
            ("lane".to_string(), Json::str(it.lane.clone())),
            ("id".to_string(), Json::str(it.id.clone())),
        ];
        if let Some(t) = it.ttl {
            m.push(("ttl".to_string(), Json::int(t)));
        }
        if !it.hdr.0.is_empty() {
            m.push(("hdr".to_string(), it.hdr.to_json()?));
        }
        m.push(("body".to_string(), Json::str(it.body.clone())));
        list.push(Json::Obj(m));
    }
    let text = Json::Obj(vec![("items".to_string(), Json::Arr(list))]).to_compact();
    if text.len() > 1_048_576 {
        return Err(Error::Invalid("post: more than 1 MiB"));
    }
    Ok(text)
}

/// What the relay did with one posted item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostStatus {
    /// Stored, with its `seq`.
    Queued,
    /// The same sender's repeat of an item with the same body: the original `seq`.
    Duplicate,
    /// Not stored: the code and message of the relay (unverified), and whether trying again can help.
    Rejected,
}

/// The result of one posted item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostResult {
    /// The item's id.
    pub id: String,
    /// What happened.
    pub status: PostStatus,
    /// The `seq` of a queued or duplicate item.
    pub seq: Option<u64>,
    /// The error code of a rejected item.
    pub code: Option<String>,
    /// The error message of a rejected item (the relay's words).
    pub message: Option<String>,
}

impl PostResult {
    /// True for a rejection whose code says to try again (`rate_limited`, `quota_exceeded`, `internal`, `unavailable`): a sender retries only these (README 5.2).
    pub fn retryable(&self) -> bool {
        self.status == PostStatus::Rejected && matches!(self.code.as_deref(), Some("rate_limited" | "quota_exceeded" | "internal" | "unavailable"))
    }
}

fn parse_post_response(doc: &Json, sent: &[PostItem]) -> Result<Vec<PostResult>, ClientError> {
    let bad = |w| ClientError::BadAnswer(w);
    if doc.get("v").and_then(Json::as_int) != Some(1) {
        return Err(bad("post: v"));
    }
    let results = doc.get("results").and_then(Json::as_array).filter(|r| r.len() == sent.len()).ok_or(bad("post: results"))?;
    let mut out = Vec::new();
    for (r, s) in results.iter().zip(sent) {
        if r.get_str("id") != Some(s.id.as_str()) {
            return Err(bad("post: a result is for another item"));
        }
        let seq = r.get_uint53("seq").filter(|n| *n >= 1);
        let status = match r.get_str("status") {
            Some("queued") if seq.is_some() => PostStatus::Queued,
            Some("duplicate") if seq.is_some() => PostStatus::Duplicate,
            Some("rejected") if r.get("error").is_some_and(Json::is_object) => PostStatus::Rejected,
            _ => return Err(bad("post: status")),
        };
        let error = r.get("error");
        out.push(PostResult {
            id: s.id.clone(),
            status,
            seq,
            code: error.and_then(|e| e.get_str("code")).map(str::to_string),
            message: error.and_then(|e| e.get_str("message")).map(|m| m.chars().take(200).collect()),
        });
    }
    Ok(out)
}
