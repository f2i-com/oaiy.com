//! The pairing routes of README 10.1, as calls of the client: the desktop's three (with its token) and the phone's two (with the `pid` as the capability).
//!
//! The phone's `GET /v1/pair/{pid}` needs no credential and no proof (what it sends is a capability the relay already has, and what comes back is checked by a MAC the relay cannot
//! make); its `POST .../response` carries the phone's keys and is **refused here unless the relay's identity has been proved** (README 8, rule 3: "before the phone's pairing").

use std::time::Duration;

use crate::error::Error;
use crate::ids::{self, Token};
use crate::json::Json;

use super::http::{Cancel, Method};
use super::relay::{ClientError, RelayClient};

/// What the desktop sends to open a rendezvous (`pairing-create-request.schema.json`).
#[derive(Debug, Clone)]
pub struct PairCreate {
    /// The `pid` in its 22-character text.
    pub pid: String,
    /// The offer text, exactly.
    pub offer: String,
    /// The offer's MAC.
    pub mac: String,
    /// The rendezvous' life in seconds (1 to 900); `None` is the relay's default of 600.
    pub ttl: Option<u64>,
    /// The offer's `appId`.
    pub app_id: String,
    /// The thumbprint of the desktop endpoint key.
    pub desktop_thumbprint: String,
}

impl PairCreate {
    /// The body, in the order the fixture records it: `pid`, `offer`, `mac`, `ttl`, `appId`, `desktopThumbprint`. The same request is sent again, byte for byte, to retry one whose
    /// answer was lost (Interpretation 51).
    pub fn to_body(&self) -> String {
        let mut m = vec![("pid", Json::str(self.pid.clone())), ("offer", Json::str(self.offer.clone())), ("mac", Json::str(self.mac.clone()))];
        if let Some(t) = self.ttl {
            m.push(("ttl", Json::int(t)));
        }
        m.push(("appId", Json::str(self.app_id.clone())));
        m.push(("desktopThumbprint", Json::str(self.desktop_thumbprint.clone())));
        Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()).to_compact()
    }
}

/// The `201` answer of `POST /v1/pair`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairCreated {
    /// The `pid`.
    pub pid: String,
    /// When the rendezvous expires, in relay time.
    pub exp: u64,
}

/// The relay's side of a rendezvous (`pairing-fetch-response.schema.json`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairState {
    /// Waiting for the phone's response.
    Open,
    /// The phone answered; waiting for the owner.
    Answered,
    /// The owner approved: the sealed token is there.
    Approved,
    /// The owner (or three wrong codes) denied.
    Denied,
}

impl PairState {
    /// The wire word.
    pub const fn as_str(self) -> &'static str {
        match self {
            PairState::Open => "open",
            PairState::Answered => "answered",
            PairState::Approved => "approved",
            PairState::Denied => "denied",
        }
    }
}

/// The approval receipt as the relay returns it: `{"issuedAt","signature","grants"}`. The schema of the answer now requires `grants` (the sorted set the desktop signed, so that a phone
/// can verify the receipt from what it reads alone); the member stays optional here so that a relay of the earlier shape, which returned none, is read and the caller's own list is used
/// (`PhonePairing::wait_outcome`), and a receipt with none and no list from the caller is refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptWire {
    /// `issuedAt`.
    pub issued_at: u64,
    /// `signature`, base64url.
    pub signature: String,
    /// `grants`, when present.
    pub grants: Option<Vec<String>>,
}

/// What `GET /v1/pair/{pid}` answered.
#[derive(Debug, Clone, PartialEq)]
pub struct PairFetch {
    /// The state.
    pub state: PairState,
    /// The offer text (open and answered).
    pub offer: Option<String>,
    /// The offer's MAC (open and answered).
    pub mac: Option<String>,
    /// The rendezvous' expiry (open and answered).
    pub exp: Option<u64>,
    /// The phone's device id on the relay (approved).
    pub device_id: Option<String>,
    /// The sealed token (approved).
    pub sealed_token: Option<String>,
    /// The receipt (approved).
    pub receipt: Option<ReceiptWire>,
    /// `hold.refused`'s `retryAfter`, when the relay refused the hold and answered at once.
    pub hold_refused_retry_after: Option<u64>,
    /// `hold.granted`: the relay says it held the request (for as long as it says, which the caller checks against its own clock).
    pub hold_granted: bool,
    /// `hold.superseded`: a newer request of the same party replaced this one, which was ended early.
    pub hold_superseded: bool,
    /// The relay's time.
    pub time: u64,
}

fn parse_fetch(doc: &Json) -> Result<PairFetch, ClientError> {
    let bad = |w: &'static str| ClientError::BadAnswer(w);
    if doc.get("v").and_then(Json::as_int) != Some(1) {
        return Err(bad("pair: v"));
    }
    let state = match doc.get_str("state") {
        Some("open") => PairState::Open,
        Some("answered") => PairState::Answered,
        Some("approved") => PairState::Approved,
        Some("denied") => PairState::Denied,
        _ => return Err(bad("pair: state")),
    };
    let time = doc.get_uint53("time").ok_or(bad("pair: time"))?;
    let text = |k: &str| doc.get_str(k).map(str::to_string);
    let mut fetch = PairFetch {
        state,
        offer: text("offer"),
        mac: text("mac"),
        exp: doc.get_uint53("exp"),
        device_id: text("deviceId"),
        sealed_token: text("sealedToken"),
        receipt: None,
        hold_refused_retry_after: None,
        hold_granted: false,
        hold_superseded: false,
        time,
    };
    if matches!(state, PairState::Open | PairState::Answered) && (fetch.offer.is_none() || fetch.mac.is_none() || fetch.exp.is_none()) {
        return Err(bad("pair: offer, mac and exp of an open rendezvous"));
    }
    if state == PairState::Approved {
        let receipt = doc.get("receipt").filter(|r| r.is_object()).ok_or(bad("pair: receipt"))?;
        let grants = match receipt.get("grants") {
            None => None,
            Some(g) => Some(
                g.as_array()
                    .ok_or(bad("pair: receipt.grants"))?
                    .iter()
                    .map(|x| x.as_str().filter(|s| ids::is_grant(s)).map(str::to_string).ok_or(bad("pair: receipt.grants")))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        fetch.receipt = Some(ReceiptWire {
            issued_at: receipt.get_uint53("issuedAt").ok_or(bad("pair: receipt.issuedAt"))?,
            signature: receipt.get_str("signature").ok_or(bad("pair: receipt.signature"))?.to_string(),
            grants,
        });
        if !fetch.device_id.as_deref().is_some_and(ids::is_device_id) || fetch.sealed_token.is_none() {
            return Err(bad("pair: deviceId and sealedToken of an approved rendezvous"));
        }
    }
    if let Some(h) = doc.get("hold").filter(|h| h.is_object()) {
        if h.get("refused").and_then(Json::as_bool) == Some(true) {
            fetch.hold_refused_retry_after = Some(h.get("retryAfter").and_then(Json::as_uint53).unwrap_or(2));
        }
        fetch.hold_granted = h.get("granted").and_then(Json::as_bool) == Some(true);
        fetch.hold_superseded = h.get("superseded").and_then(Json::as_bool) == Some(true);
    }
    Ok(fetch)
}

/// The answer to a decision, a reject or a burn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairAck {
    /// `approved`, `denied`, `open` or `expired`.
    pub state: String,
    /// The phone's device id on the relay, for an approval.
    pub device_id: Option<String>,
}

impl RelayClient {
    /// `POST /v1/pair` (desktop token): opens a rendezvous. The answer is `201`; the same request again (a retry of one whose answer was lost) gets the original `201`.
    pub fn pair_create(&self, token: &Token, create: &PairCreate, cancel: &Cancel) -> Result<PairCreated, ClientError> {
        if !ids::is_pid(&create.pid) {
            return Err(ClientError::Request(Error::Invalid("pid")));
        }
        let response = self.authed(token, Method::Post, "/v1/pair", Some(create.to_body().into_bytes()), None, cancel)?;
        let doc = self.success_json(&response, &[201])?;
        let pid = doc.get_str("pid").filter(|p| *p == create.pid).ok_or(ClientError::BadAnswer("pair: pid"))?;
        Ok(PairCreated { pid: pid.to_string(), exp: doc.get_uint53("exp").ok_or(ClientError::BadAnswer("pair: exp"))? })
    }

    /// `GET /v1/pair/{pid}[?wait=&state=]` (no credential): the offer and the state, or the outcome. `wait` is the seconds the relay may hold the request (clamped by it to
    /// `info.wait.max`), `state` the state the caller last saw: a wait is held only while the rendezvous is still in that state.
    pub fn pair_fetch(&self, pid: &str, wait: Option<u64>, seen: Option<PairState>, cancel: &Cancel) -> Result<PairFetch, ClientError> {
        if !ids::is_pid(pid) {
            return Err(ClientError::Request(Error::Invalid("pid")));
        }
        let mut path = format!("/v1/pair/{pid}");
        let mut sep = '?';
        if let Some(w) = wait {
            path.push_str(&format!("{sep}wait={w}"));
            sep = '&';
        }
        if let Some(s) = seen {
            path.push_str(&format!("{sep}state={}", s.as_str()));
        }
        let timeout = Duration::from_secs(wait.unwrap_or(0) + 10);
        let response = self.exchange(Method::Get, &path, None, None, &[], timeout, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        parse_fetch(&doc)
    }

    /// `POST /v1/pair/{pid}/response`: the phone's response. Refused here (nothing is sent) unless the relay has been proved. `202` is accepted; the identical response again (a retry
    /// of one whose `202` was lost) is `202` again; another response is `409 already_answered`.
    pub fn pair_respond(&self, pid: &str, response_text: &str, cancel: &Cancel) -> Result<(), ClientError> {
        if !ids::is_pid(pid) || response_text.is_empty() || response_text.len() > 8192 {
            return Err(ClientError::Request(Error::Invalid("response")));
        }
        self.gate()?;
        let body = Json::obj([("response", Json::str(response_text))]).to_compact();
        let response = self.exchange(
            Method::Post,
            &format!("/v1/pair/{pid}/response"),
            Some(body.into_bytes()),
            None,
            &[],
            self.config().request_timeout,
            cancel,
        )?;
        let doc = self.success_json(&response, &[202])?;
        if doc.get_str("state") != Some("answered") {
            return Err(ClientError::BadAnswer("pair: state"));
        }
        Ok(())
    }

    /// `POST /v1/pair/{pid}/decision` (desktop token) with a body the caller built (`{"approve":true,...}` or `{"approve":false}`).
    pub fn pair_decision(&self, token: &Token, pid: &str, body: &str, cancel: &Cancel) -> Result<PairAck, ClientError> {
        // The pid goes into the path of a request that carries the desktop's token: it is the relay's own spelling or nothing.
        if !ids::is_pid(pid) {
            return Err(ClientError::Request(Error::Invalid("pid")));
        }
        let response = self.authed(token, Method::Post, &format!("/v1/pair/{pid}/decision"), Some(body.as_bytes().to_vec()), None, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        let state = doc.get_str("state").filter(|s| matches!(*s, "approved" | "denied")).ok_or(ClientError::BadAnswer("pair: state"))?;
        let device_id = doc.get_str("deviceId").map(str::to_string);
        if (state == "approved") != device_id.as_deref().is_some_and(ids::is_device_id) {
            return Err(ClientError::BadAnswer("pair: deviceId"));
        }
        Ok(PairAck { state: state.to_string(), device_id })
    }

    /// `POST /v1/pair/{pid}/reject` (desktop token): returns an `answered` rendezvous to `open` (at most twice; the third ends it).
    pub fn pair_reject(&self, token: &Token, pid: &str, reason: &str, cancel: &Cancel) -> Result<PairAck, ClientError> {
        if !ids::is_pid(pid) {
            return Err(ClientError::Request(Error::Invalid("pid")));
        }
        let body = Json::obj([("reason", Json::str(reason.chars().take(200).collect::<String>()))]).to_compact();
        let response = self.authed(token, Method::Post, &format!("/v1/pair/{pid}/reject"), Some(body.into_bytes()), None, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        Ok(PairAck { state: doc.get_str("state").ok_or(ClientError::BadAnswer("pair: state"))?.to_string(), device_id: None })
    }

    /// `POST /v1/pair/{pid}/burn` (desktop token): ends the rendezvous now.
    pub fn pair_burn(&self, token: &Token, pid: &str, cancel: &Cancel) -> Result<PairAck, ClientError> {
        if !ids::is_pid(pid) {
            return Err(ClientError::Request(Error::Invalid("pid")));
        }
        let response = self.authed(token, Method::Post, &format!("/v1/pair/{pid}/burn"), Some(b"{}".to_vec()), None, cancel)?;
        let doc = self.success_json(&response, &[200])?;
        Ok(PairAck { state: doc.get_str("state").ok_or(ClientError::BadAnswer("pair: state"))?.to_string(), device_id: None })
    }
}
