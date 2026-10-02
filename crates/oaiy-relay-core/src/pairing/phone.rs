//! The phone's half of pairing v3 (README 10.1; package MOB-22a): read the pairing key or the typed code, prove the relay, fetch and verify the offer, answer it, show the short
//! authentication string, wait for the owner's decision, verify the receipt and open the sealed token.
//!
//! **What the phone trusts, and when.** Nothing about a relay is trusted until its identity is proved against a key that came **out of band**: the `f` of a scanned key, or, for a typed
//! code (which carries only the secret), the fingerprint inside an offer whose MAC verified under the secret (so the desktop wrote it) and whose `relay.url` is the host the owner typed.
//! The MAC is checked over the text exactly as received and before anything in it is read. The desktop's keys and the host identity come **only** from that MAC-verified offer and
//! are the phone's whole peer pin: the relay's answers never add to it. **The approval receipt is verified before a profile is made** (design 6.5: "no valid receipt, no profile"),
//! and the token is opened only then: a relay that says "approved" with no desktop decision behind it gets a phone that stores nothing.
//!
//! **The receipt and the grants** (a gap in the contract, see this crate's README): the desktop signs the grants it approved into the receipt, and the relay's answer to the phone
//! does not carry them. [`PhonePairing::wait_outcome`] therefore takes the grants from `receipt.grants` when the relay returns them (the schema allows extra members) and otherwise
//! from the caller, who must have them from somewhere the relay cannot forge; with neither it fails closed with [`PairingError::ReceiptGrantsUnknown`].

use std::sync::Arc;
use std::time::Duration;

use crate::client::http::Cancel;
use crate::client::pair::{PairFetch, PairState};
use crate::client::store::{PeerPin, ProfileKind, ProfileStore, RelayProfile, SecretStore, SECRET_TOKEN};
use crate::client::{ClientError, RelayClient};
use crate::ids::{self, Token};
use crate::keys::{Signer, X25519Secret};
use crate::pairing::math::{self, Derived, PairingKey, PairingSecret, Sas};
use crate::pairing::offer::Offer;
use crate::pairing::response::{Claims, Response, CLAIMS_LIFE_MAX_S};
use crate::pairing::PairingError;
use crate::sealed;
use crate::url::RelayUrl;

/// The phone's identity for one pairing.
pub struct PhoneIdentity {
    /// The phone's endpoint key: it signs the response and is pinned by the desktop.
    pub endpoint: Signer,
    /// The phone's X25519 key: the token is sealed to it.
    pub x25519: X25519Secret,
    /// The id the phone names itself in the claims (`dev-` and 22 characters; the relay makes its own).
    pub device_id: String,
    /// The name the owner will see on the desktop (cleaned to 60 characters).
    pub display_name: Option<String>,
}

/// What the owner gave: a scanned or pasted pairing key, or a typed code and the relay's host name.
#[derive(Debug, Clone, Copy)]
pub enum PairingInput<'a> {
    /// `oaiy://pair?v=3&u=...&s=...`.
    Key(&'a str),
    /// The 28-character code (separators and case free) and the host name of the relay (the code carries only the secret, so the relay's address is asked for).
    Typed {
        /// The code.
        code: &'a str,
        /// The host, as the owner typed it (`relay.example.com`).
        host: &'a str,
    },
}

/// Where the pairing is, from the input alone: made without any network call (a typo in a typed code costs nothing).
pub struct PairingTarget {
    /// The relay's base URL.
    pub relay: RelayUrl,
    /// The thumbprint the relay's key must have, when the input carried it (a key does, a typed code does not).
    pub fingerprint: Option<String>,
    secret: PairingSecret,
}

impl PairingTarget {
    /// Reads the input. A typed code is normalised and its check characters verified here; a key is parsed as README 10.1 says. `expires_at` of a key that has already passed is
    /// refused by the caller against relay time (the phone has none yet).
    pub fn from_input(input: PairingInput<'_>) -> Result<PairingTarget, PairingError> {
        match input {
            PairingInput::Key(uri) => {
                let k = PairingKey::parse(uri)?;
                Ok(PairingTarget { relay: k.relay, fingerprint: k.relay_thumbprint, secret: k.secret })
            }
            PairingInput::Typed { code, host } => {
                let secret = math::parse_typed_code(code)?;
                // The owner types a host name. A scheme is accepted only where `RelayUrl` accepts it: `http://127.0.0.1:port` in a build with the loopback test feature, and never otherwise.
                let host = host.trim().trim_end_matches('/');
                let relay = RelayUrl::parse(&if host.contains("://") { host.to_string() } else { format!("https://{host}") })?;
                Ok(PairingTarget { relay, fingerprint: None, secret })
            }
        }
    }

    /// The `pid` of this pairing.
    pub fn pid(&self) -> Result<String, PairingError> {
        Ok(math::pid_text(&self.secret.derive()?.pid))
    }
}

/// What the owner is shown before anything is posted (README 10.1 step 3): the relay's host and the desktop's name, from the verified offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSummary {
    /// The relay's host name.
    pub relay_host: String,
    /// The desktop's name (data written by the desktop: show it as text).
    pub desktop_name: String,
    /// The app.
    pub app_id: String,
    /// The thumbprint of the desktop endpoint key.
    pub desktop_thumbprint: String,
    /// When the offer expires, in relay time.
    pub expires_at: u64,
}

/// What the pairing came to.
#[allow(clippy::large_enum_variant)]
pub enum Outcome {
    /// Approved and verified: the token and the profile to keep (see [`store_paired`]).
    Paired(Paired),
    /// The owner denied it (or three wrong codes did).
    Denied,
    /// The rendezvous is over (expired, burned) or the relay no longer knows it.
    Expired,
    /// The desktop returned the rendezvous to `open`: it did not accept the phone's response (a failed check on its side). The phone may answer again.
    Rejected,
}

/// A verified pairing, ready to be stored.
pub struct Paired {
    /// The profile (no secret in it).
    pub profile: RelayProfile,
    /// The device token, a credential.
    pub token: Token,
}

enum State {
    Start,
    Offered,
    Answered,
    Done,
}

/// The phone's pairing.
pub struct PhonePairing {
    client: Arc<RelayClient>,
    target: PairingTarget,
    identity: PhoneIdentity,
    derived: Derived,
    pid: String,
    offer: Option<Offer>,
    response: Option<Response>,
    state: State,
}

impl PhonePairing {
    /// A pairing for `target` over `client` (made for `target.relay`, with the pin the target carries, if any).
    pub fn new(client: Arc<RelayClient>, target: PairingTarget, identity: PhoneIdentity) -> Result<PhonePairing, PairingError> {
        if *client.url() != target.relay {
            return Err(PairingError::Protocol(crate::Error::Mismatch("the client is for another relay than the pairing")));
        }
        let derived = target.secret.derive()?;
        let pid = math::pid_text(&derived.pid);
        if let Some(f) = &target.fingerprint {
            client.pin(f)?;
        }
        Ok(PhonePairing { client, target, identity, derived, pid, offer: None, response: None, state: State::Start })
    }

    /// The `pid`.
    pub fn pid(&self) -> &str {
        &self.pid
    }

    /// The thumbprint of the phone's endpoint key (the `holderKeyThumbprint` of its admissions).
    pub fn endpoint_thumbprint(&self) -> String {
        self.identity.endpoint.thumbprint()
    }

    /// The phone's id as it names itself in the claims.
    pub fn claimed_device_id(&self) -> &str {
        &self.identity.device_id
    }

    /// The verified offer, once fetched.
    pub fn offer(&self) -> Option<&Offer> {
        self.offer.as_ref()
    }

    fn prove(&self, cancel: &Cancel) -> Result<(), PairingError> {
        match self.client.prove(cancel) {
            Ok(_) => Ok(()),
            Err(crate::client::ProveError::NoAnswer(e)) => Err(e.into()),
            Err(crate::client::ProveError::Invalid(_)) => Err(PairingError::Suspect),
        }
    }

    /// Fetches the offer and verifies it. With a scanned key the relay is proved first (against the key's `f`); with a typed code the offer is fetched first (no credential, nothing
    /// secret is sent), its MAC is verified under the secret, and the relay is then proved against the fingerprint **inside** that offer, which must be the one the offer was MACed with.
    /// Then: the offer's `relay.url` is this relay, its window holds in relay-corrected time, and the key that proved itself is the offer's `relay.fingerprint` (and the key's `f`).
    pub fn fetch_offer(&mut self, cancel: &Cancel) -> Result<OfferSummary, PairingError> {
        if !matches!(self.state, State::Start) {
            return Err(PairingError::WrongState("the offer was fetched already"));
        }
        if self.target.fingerprint.is_some() {
            self.prove(cancel)?;
        }
        let fetch = self.client.pair_fetch(&self.pid, None, None, cancel).map_err(|e| match e.relay().map(|r| r.status) {
            Some(404 | 410) => PairingError::NotFound,
            _ => PairingError::Client(e),
        })?;
        match fetch.state {
            PairState::Open => {}
            PairState::Answered => return Err(PairingError::AlreadyAnswered),
            PairState::Approved | PairState::Denied => return Err(PairingError::WrongState("the rendezvous is over")),
        }
        let (text, mac) = (fetch.offer.as_deref().unwrap_or(""), fetch.mac.as_deref().unwrap_or(""));
        // The MAC first, over the text exactly as received; nothing in the text is read before it verifies.
        let offer = Offer::verify(text, mac, &self.derived.mac_key)?;
        if offer.relay != self.target.relay {
            return Err(PairingError::Protocol(crate::Error::Mismatch("the offer is for another relay than the one the owner named")));
        }
        match &self.target.fingerprint {
            Some(f) if *f != offer.relay_fingerprint => {
                return Err(PairingError::Protocol(crate::Error::Mismatch("the offer names another relay key than the pairing key")))
            }
            Some(_) => {}
            None => {
                self.client.pin(&offer.relay_fingerprint)?;
                self.prove(cancel)?;
            }
        }
        offer.check_window(self.client.relay_now_or_local())?;
        let summary = OfferSummary {
            relay_host: self.target.relay.host().to_string(),
            desktop_name: offer.desktop_name.clone(),
            app_id: offer.app_id.clone(),
            desktop_thumbprint: offer.desktop_endpoint.thumbprint(),
            expires_at: offer.expires_at,
        };
        self.offer = Some(offer);
        self.state = State::Offered;
        Ok(summary)
    }

    fn offer_ref(&self) -> Result<&Offer, PairingError> {
        self.offer.as_ref().ok_or(PairingError::WrongState("no offer was fetched"))
    }

    /// The short authentication string (the 13 characters the phone shows): from the verified offer and the phone's own key.
    pub fn sas(&self) -> Result<Sas, PairingError> {
        let offer = self.offer_ref()?;
        Ok(math::sas(&offer.desktop_endpoint.to_bytes(), &self.identity.endpoint.verify_key().to_bytes(), &offer.nonce, &self.derived.pid)?)
    }

    /// Answers the offer: the claims signed with the phone's endpoint key and MACed under the pairing secret, posted to the rendezvous. Call it once the owner has confirmed (the
    /// two confirmations are the app's). Built once and sent as it is: calling it again after a lost `202` sends the identical text, which the relay answers `202` again.
    /// Returns the SAS to show. A `409 already_answered` is [`PairingError::AlreadyAnswered`].
    pub fn respond(&mut self, cancel: &Cancel) -> Result<Sas, PairingError> {
        if !matches!(self.state, State::Offered | State::Answered) {
            return Err(PairingError::WrongState("no offer to answer"));
        }
        if self.response.is_none() {
            let offer = self.offer_ref()?.clone();
            let now = u64::try_from(self.client.relay_now_or_local()).map_err(|_| PairingError::Protocol(crate::Error::Invalid("time")))?;
            let claims = Claims {
                app_id: offer.app_id.clone(),
                desktop_connection_id: offer.desktop_connection_id.clone(),
                desktop_key_thumbprint: offer.desktop_endpoint.thumbprint(),
                device_id: self.identity.device_id.clone(),
                display_name: self.identity.display_name.as_deref().map(|n| ids::clean_name(n, 60)).filter(|n| !n.is_empty()),
                mobile_endpoint: self.identity.endpoint.verify_key(),
                mobile_x25519: self.identity.x25519.public_key(),
                pairing_nonce: offer.nonce,
                jti: offer.jti.clone(),
                issued_at: now,
                expires_at: now + CLAIMS_LIFE_MAX_S,
            };
            self.response = Some(Response::build(&self.identity.endpoint, &self.derived.mac_key, claims)?);
        }
        let text = self.response.as_ref().map(|r| r.text.clone()).unwrap_or_default();
        self.client.pair_respond(&self.pid, &text, cancel).map_err(|e| match e.code() {
            Some("already_answered") => PairingError::AlreadyAnswered,
            Some("not_found") | Some("expired") => PairingError::NotFound,
            _ => PairingError::Client(e),
        })?;
        self.state = State::Answered;
        self.sas()
    }

    /// Waits for the owner's decision with `GET /v1/pair/{pid}?wait=&state=answered`, paced as a poll is (a refused hold is a short poll at `retryAfter`, 2, 4, 5 seconds with
    /// jitter; a `429` waits its `Retry-After`; a failure backs off 1, 2, 4 ... 60 seconds), until the rendezvous is decided or over. On approval: the receipt is verified with the
    /// desktop key pinned from the offer, the token is opened, and a profile is made; **nothing is returned, and nothing is stored, if either fails.**
    ///
    /// `out_of_band_grants` are the grants the desktop approved when the relay's answer does not carry them (see the module): the receipt covers them, so a list that is not what was
    /// approved fails the receipt.
    pub fn wait_outcome(&mut self, out_of_band_grants: Option<&[String]>, cancel: &Cancel) -> Result<Outcome, PairingError> {
        if !matches!(self.state, State::Answered) {
            return Err(PairingError::WrongState("the offer was not answered"));
        }
        let wait = self.client.info().map_or(20, |i| i.wait.max.min(20));
        let (mut failures, mut refusals) = (0u32, 0u32);
        let offer_expires = self.offer_ref()?.expires_at as i64;
        loop {
            if cancel.is_cancelled() {
                return Err(PairingError::Cancelled);
            }
            if self.client.relay_now_or_local() > offer_expires + 900 {
                self.state = State::Done;
                return Ok(Outcome::Expired);
            }
            let reply = self.client.pair_fetch(&self.pid, Some(wait), Some(PairState::Answered), cancel);
            let pause = match reply {
                Ok(f) if f.hold_refused_retry_after.is_some() && f.state == PairState::Answered => {
                    let r = f.hold_refused_retry_after.unwrap_or(2).clamp(1, 120);
                    let info = self.client.info();
                    let fallback = info.map_or(5, |i| i.wait.fallback_s.clamp(1, 60));
                    let p = r.max(fallback.min(r.saturating_mul(1u64 << refusals.min(20))));
                    refusals += 1;
                    failures = 0;
                    p as f64
                }
                Ok(f) => {
                    failures = 0;
                    refusals = 0;
                    match f.state {
                        // Held for `wait` seconds and unchanged: ask again at once. A relay that does not hold (`wait` 0, or a request answered at once) is a short poll, paced at the
                        // poll gap and never faster than a second, so that an owner who takes minutes costs a request a second and not a spin.
                        PairState::Answered if wait > 0 => 0.0,
                        PairState::Answered => self.client.info().map_or(1.0, |i| (i.wait.poll_gap_ms as f64 / 1000.0).max(1.0)),
                        PairState::Open => {
                            self.state = State::Offered;
                            return Ok(Outcome::Rejected);
                        }
                        PairState::Denied => {
                            self.state = State::Done;
                            return Ok(Outcome::Denied);
                        }
                        PairState::Approved => {
                            let paired = self.accept(&f, out_of_band_grants)?;
                            self.state = State::Done;
                            return Ok(Outcome::Paired(paired));
                        }
                    }
                }
                Err(ClientError::Cancelled) => return Err(PairingError::Cancelled),
                Err(e) => match e.relay().map(|r| (r.status, r.retry_after)) {
                    Some((404 | 410, _)) => {
                        self.state = State::Done;
                        return Ok(Outcome::Expired);
                    }
                    Some((429, ask)) => {
                        failures = 0;
                        ask.unwrap_or(1).clamp(1, 120) as f64
                    }
                    Some((s, _)) if (400..500).contains(&s) && s != 408 => return Err(PairingError::Client(e)),
                    _ => {
                        failures += 1;
                        (1u64 << (failures - 1).min(6)).min(60) as f64
                    }
                },
            };
            if pause > 0.0 {
                let jittered = pause * (1.0 + crate::poll::JITTER * self.client.jitter());
                if !self.client.clock().sleep(Duration::from_secs_f64(jittered), cancel) {
                    return Err(PairingError::Cancelled);
                }
            }
        }
    }

    /// The approval, verified: receipt with the grants, then the token. Order matters: nothing about the token is looked at until the desktop's receipt verifies.
    fn accept(&self, f: &PairFetch, out_of_band: Option<&[String]>) -> Result<Paired, PairingError> {
        let offer = self.offer_ref()?;
        let receipt = f.receipt.as_ref().ok_or(PairingError::ReceiptInvalid)?;
        let grants: Vec<String> = match (&receipt.grants, out_of_band) {
            (Some(g), _) => g.clone(),
            (None, Some(g)) => g.to_vec(),
            (None, None) => return Err(PairingError::ReceiptGrantsUnknown),
        };
        if grants.len() > 16 || grants.iter().any(|g| !ids::is_known_grant(g)) {
            return Err(PairingError::ReceiptInvalid);
        }
        let now = self.client.relay_now_or_local();
        let asked = self.response.as_ref().map_or(0, |r| r.claims.issued_at) as i64;
        let issued = i64::try_from(receipt.issued_at).map_err(|_| PairingError::ReceiptInvalid)?;
        // A receipt dated before the phone asked, or in the future, is not an answer to this pairing.
        if issued < asked - crate::pairing::offer::SKEW_S || issued > now + crate::pairing::offer::SKEW_S {
            return Err(PairingError::ReceiptInvalid);
        }
        math::verify_receipt(
            &offer.desktop_endpoint,
            &offer.app_id,
            &grants,
            receipt.issued_at,
            &self.identity.endpoint.verify_key().thumbprint(),
            &self.pid,
            &receipt.signature,
        )
        .map_err(|_| PairingError::ReceiptInvalid)?;
        let token = sealed::open_token(&self.identity.x25519, f.sealed_token.as_deref().unwrap_or("")).map_err(|_| PairingError::TokenInvalid)?;
        let device_id = f.device_id.clone().filter(|d| ids::is_device_id(d)).ok_or(PairingError::TokenInvalid)?;
        let relay_id = self.client.info().map(|i| i.relay_id).ok_or(PairingError::Suspect)?;
        let profile = RelayProfile {
            kind: ProfileKind::Phone,
            relay: self.target.relay.clone(),
            relay_id,
            relay_thumbprint: offer.relay_fingerprint.clone(),
            device_id,
            name: self.identity.display_name.as_deref().map(|n| ids::clean_name(n, 60)).unwrap_or_default(),
            enrolled_at: now,
            app_id: Some(offer.app_id.clone()),
            grants,
            peer: Some(PeerPin {
                desktop_connection_id: offer.desktop_connection_id.clone(),
                desktop_name: offer.desktop_name.clone(),
                desktop_endpoint: offer.desktop_endpoint,
                desktop_x25519: offer.desktop_x25519,
                host_ed25519: offer.host_ed25519,
                host_x25519: offer.host_x25519,
            }),
        };
        Ok(Paired { profile, token })
    }
}

/// Keeps a verified pairing: the token goes to the secret store first, the profile after it, and a profile that cannot be stored takes the token back out, so that a half-made pairing
/// is never left behind. The phone never replaces an existing profile by itself: the caller asks the owner first (README 10.1: "it never replaces or activates an existing relay
/// profile without a second confirmation").
pub fn store_paired(paired: &Paired, secrets: &dyn SecretStore, profiles: &dyn ProfileStore) -> Result<(), PairingError> {
    secrets.put(SECRET_TOKEN, paired.token.expose().as_bytes()).map_err(|_| PairingError::Store)?;
    if profiles.save(&paired.profile).is_err() {
        let _ = secrets.delete(SECRET_TOKEN);
        return Err(PairingError::Store);
    }
    Ok(())
}

/// The phone's keys for a pairing, made fresh.
pub fn new_identity(display_name: Option<&str>, random: &mut dyn crate::client::Rng) -> Result<PhoneIdentity, PairingError> {
    let mut id = [0u8; 16];
    random.fill(&mut id);
    Ok(PhoneIdentity {
        endpoint: Signer::generate()?,
        x25519: X25519Secret::generate()?,
        device_id: format!("dev-{}", crate::b64::encode(&id)),
        display_name: display_name.map(str::to_string),
    })
}
