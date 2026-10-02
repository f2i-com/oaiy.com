//! The desktop's half of pairing v3 (README 10.1): create an offer, open the rendezvous, verify the phone's response, gate the approval behind the short authentication string,
//! sign the receipt and decide.
//!
//! **The gate is in the type of the thing, not in the UI.** An approval body can be made only for a pending pairing whose SAS the owner has typed correctly: [`DesktopPairing::approve_body`]
//! refuses anything else with [`PairingError::SasRequired`], and [`DesktopPairing::confirm_sas`] is the only path to a confirmed state. Three wrong entries (a complete 13-character entry
//! with a good check character that is not the code) deny the pairing and burn the rendezvous; an incomplete entry or a wrong check character is a mistyped character and costs
//! nothing (README 10.1 step 6).
//!
//! **What this keeps, and what the host keeps.** The party keeps what must not be forgotten between a response and a decision: the pending offers with their secrets, the nonces and
//! `jti`s that an approval has consumed (a response is used once), and the thumbprints of phone keys that were revoked (a response from one is refused). A host that restarts
//! between the two loses the pending offers (they live ten minutes and are made again); it persists the consumed set and the revoked keys itself if it wants them across restarts.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use oaiy_crypto::zeroize::Secret;

use crate::b64;
use crate::client::http::Cancel;
use crate::client::pair::PairCreate;
use crate::client::store::Item;
use crate::client::{RelayClient, Rng};
use crate::ids::{self, Token};
use crate::json::Json;
use crate::keys::{Signer, VerifyKey, X25519Public};
use crate::pairing::math::{self, PairingKey, PairingSecret, Sas, SasEntry};
use crate::pairing::offer::{Offer, OfferParams};
use crate::pairing::response::Response;
use crate::pairing::PairingError;
use crate::url::RelayUrl;

/// The attempts the owner has at the SAS before the pairing is denied.
pub const SAS_ATTEMPTS: u8 = 3;

/// The desktop's identity as pairing uses it.
pub struct DesktopIdentity {
    /// The desktop's relay device id (`desktopConnectionId`).
    pub device_id: String,
    /// The name the owner will see on the phone (cleaned to 60 characters).
    pub name: String,
    /// The endpoint key: it signs the approval receipt and the phone pins it.
    pub endpoint: Signer,
    /// The endpoint's X25519 key (`desktopX25519`).
    pub endpoint_x25519: X25519Public,
    /// The host identity's Ed25519 key, which the phone pins and which signs rings.
    pub host_ed25519: VerifyKey,
    /// The host identity's X25519 key, which the phone pins.
    pub host_x25519: X25519Public,
}

/// A new offer: what the owner is shown (the QR, the link, the typed code) and what is sent to the relay.
pub struct NewOffer {
    /// The `pid`, 22 characters.
    pub pid: String,
    /// The offer.
    pub offer: Offer,
    /// The offer's MAC.
    pub mac: String,
    /// `oaiy://pair?v=3&u=...&f=...&s=...&x=...`: the QR and the deep link. **It holds the secret**: it is shown on the owner's screen and nowhere else.
    pub pairing_uri: String,
    /// The typed code, seven groups of four. It holds the secret too.
    pub typed_code: String,
    /// The request that opens the rendezvous (`POST /v1/pair`), to be sent as it is and, if its answer is lost, sent again as it is.
    pub create: PairCreate,
}

/// What a response that arrived led to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairEvent {
    /// A good response: the owner must now type the phone's code.
    AwaitingSas {
        /// The `pid`.
        pid: String,
        /// The phone's display name (data from the phone: show it as text).
        phone_name: Option<String>,
        /// The thumbprint of the phone's endpoint key.
        phone_thumbprint: String,
    },
    /// A response that failed a check: the relay is told to return the rendezvous to `open` (`POST .../reject`).
    Rejected {
        /// The `pid`.
        pid: String,
        /// A fixed word: `malformed`, `replayed`, `revoked`, `window`, `binding`, `mac`, `signature` or `key`.
        reason: &'static str,
    },
    /// Not for any pending pairing of this desktop, or not in a state to take a response.
    Ignored(&'static str),
}

/// What the owner's entry of the SAS led to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SasOutcome {
    /// Fewer than 13 characters. Not an attempt.
    Incomplete,
    /// Not made of the alphabet, or too long. Not an attempt.
    Invalid,
    /// A wrong check character. Not an attempt.
    BadCheck,
    /// A wrong code: one attempt used.
    Wrong {
        /// How many attempts are left.
        attempts_left: u8,
    },
    /// Three wrong codes: the pairing was denied and the rendezvous burned.
    Denied,
    /// The right code: the pairing was approved.
    Approved {
        /// The phone's device id on the relay.
        device_id: String,
    },
}

struct Verified {
    claims: crate::pairing::response::Claims,
    sas: Sas,
    attempts: u8,
    confirmed: bool,
}

#[allow(clippy::large_enum_variant)]
enum Phase {
    AwaitingResponse,
    AwaitingSas(Verified),
    /// The approval is made and being sent: the body is kept so that a retry sends the same receipt.
    Approving {
        body: String,
    },
    Done,
}

struct Pending {
    mac_key: Secret<32>,
    pid_bytes: [u8; 16],
    offer: Offer,
    phase: Phase,
}

/// The desktop's pairings in flight.
pub struct DesktopPairing {
    identity: Arc<DesktopIdentity>,
    app_id: String,
    relay: RelayUrl,
    relay_fingerprint: String,
    pending: HashMap<String, Pending>,
    consumed: HashSet<(String, String)>,
    revoked: HashSet<String>,
    /// The ids of the pair items already judged: a poll delivers at least once, and a response that was judged and rejected must not be judged (and rejected) again, which would
    /// return a later, good response of the same rendezvous to open under the phone.
    seen: HashSet<String>,
}

/// The pid an item id names: `pid`, `pid.2` or `pid.3` (Interpretation 26: the second and third responses of a rendezvous that was reopened by a reject).
fn pid_of_item(id: &str) -> Option<&str> {
    let base = id.split_once('.').map_or(id, |(p, n)| if matches!(n, "2" | "3") { p } else { "" });
    ids::is_pid(base).then_some(base)
}

impl DesktopPairing {
    /// A party for `identity` pairing phones for `app_id` through the relay at `relay` (whose key has the thumbprint `relay_fingerprint`).
    pub fn new(identity: Arc<DesktopIdentity>, app_id: &str, relay: RelayUrl, relay_fingerprint: &str) -> DesktopPairing {
        DesktopPairing {
            identity,
            app_id: app_id.to_string(),
            relay,
            relay_fingerprint: relay_fingerprint.to_string(),
            pending: HashMap::new(),
            consumed: HashSet::new(),
            revoked: HashSet::new(),
            seen: HashSet::new(),
        }
    }

    /// Makes the party refuse a response from the phone key with this thumbprint (a key that was revoked).
    pub fn revoke_phone_key(&mut self, thumbprint: &str) {
        self.revoked.insert(thumbprint.to_string());
    }

    /// The nonces and `jti`s that approvals have consumed, for a host that keeps them across restarts.
    pub fn consumed(&self) -> Vec<(String, String)> {
        let mut v: Vec<_> = self.consumed.iter().cloned().collect();
        v.sort();
        v
    }

    /// Restores consumed nonces and `jti`s.
    pub fn restore_consumed(&mut self, consumed: impl IntoIterator<Item = (String, String)>) {
        self.consumed.extend(consumed);
    }

    /// The number of rendezvous this party has not finished.
    pub fn open_count(&self) -> usize {
        self.pending.values().filter(|p| !matches!(p.phase, Phase::Done)).count()
    }

    /// Makes an offer with a secret, a nonce and a `jti` drawn from `rng`, at `relay_now`.
    pub fn create_offer(&mut self, rng: &mut dyn Rng, relay_now: i64) -> Result<NewOffer, PairingError> {
        let mut secret = [0u8; 16];
        let mut nonce = [0u8; 32];
        let mut jti = [0u8; 12];
        rng.fill(&mut secret);
        rng.fill(&mut nonce);
        rng.fill(&mut jti);
        self.create_offer_with(secret, nonce, format!("pair-{}", b64::encode(&jti)), relay_now)
    }

    /// Makes an offer from given randomness (what the recorded ceremony and the vectors fix).
    pub fn create_offer_with(&mut self, secret: [u8; 16], nonce: [u8; 32], jti: String, relay_now: i64) -> Result<NewOffer, PairingError> {
        let secret = PairingSecret::new(secret);
        let derived = secret.derive()?;
        let id = &self.identity;
        let offer = Offer::build(&OfferParams {
            app_id: &self.app_id,
            desktop_connection_id: &id.device_id,
            desktop_name: &id.name,
            desktop_endpoint: &id.endpoint.verify_key(),
            desktop_x25519: &id.endpoint_x25519,
            host_ed25519: &id.host_ed25519,
            host_x25519: &id.host_x25519,
            nonce,
            jti: &jti,
            issued_at: u64::try_from(relay_now).map_err(|_| PairingError::Protocol(crate::Error::Invalid("time")))?,
            relay: &self.relay,
            relay_fingerprint: &self.relay_fingerprint,
        })?;
        let pid = math::pid_text(&derived.pid);
        let mac = offer.mac(&derived.mac_key)?;
        let create = PairCreate {
            pid: pid.clone(),
            offer: offer.text.clone(),
            mac: mac.clone(),
            ttl: Some(600),
            app_id: self.app_id.clone(),
            desktop_thumbprint: id.endpoint.thumbprint(),
        };
        let new = NewOffer {
            pid: pid.clone(),
            pairing_uri: PairingKey::to_uri(&self.relay, &self.relay_fingerprint, &secret, offer.expires_at),
            typed_code: secret.typed_code(),
            offer: offer.clone(),
            mac,
            create,
        };
        self.pending.insert(pid, Pending { mac_key: derived.mac_key, pid_bytes: derived.pid, offer, phase: Phase::AwaitingResponse });
        Ok(new)
    }

    /// Opens the rendezvous on the relay (`POST /v1/pair`). Call it again with the same offer to retry one whose answer was lost: the relay answers the original `201`.
    pub fn open(&self, client: &RelayClient, token: &Token, offer: &NewOffer, cancel: &Cancel) -> Result<crate::client::PairCreated, PairingError> {
        Ok(client.pair_create(token, &offer.create, cancel)?)
    }

    /// Judges a `pair` item that the poll delivered (item id `pid`, `pid.2` or `pid.3`, body the phone's response text, byte for byte), in the order the README gives: the pairing it
    /// belongs to and its state; the shape and identifiers; that the nonce and `jti` have not been consumed; that the phone's key is not revoked; the lifetime against relay time
    /// with 30 seconds of slack; the binding to the offer; the phone's key of 32 bytes and not of small order; the MAC in constant time; the signature last, strictly.
    pub fn receive_response(&mut self, item_id: &str, body: &str, relay_now: i64) -> PairEvent {
        let Some(pid) = pid_of_item(item_id).map(str::to_string) else {
            return PairEvent::Ignored("not a pairing item");
        };
        if !self.seen.insert(item_id.to_string()) {
            return PairEvent::Ignored("this item was judged already");
        }
        let Some(p) = self.pending.get_mut(&pid) else {
            return PairEvent::Ignored("no such pairing");
        };
        if !matches!(p.phase, Phase::AwaitingResponse) {
            return PairEvent::Ignored("this pairing is not waiting for a response");
        }
        let reject = |reason: &'static str| PairEvent::Rejected { pid: pid.clone(), reason };
        let response = match Response::parse(body) {
            Ok(r) => r,
            Err(crate::Error::Crypto(_)) => return reject("key"),
            Err(_) => return reject("malformed"),
        };
        let c = &response.claims;
        if self.consumed.contains(&(b64::encode(&c.pairing_nonce), c.jti.clone())) {
            return reject("replayed");
        }
        if self.revoked.contains(&c.mobile_endpoint.thumbprint()) {
            return reject("revoked");
        }
        match response.verify(&p.offer, &p.mac_key, relay_now) {
            Ok(()) => {}
            Err(crate::Error::OutsideWindow(_)) => return reject("window"),
            Err(crate::Error::Mismatch(_)) => return reject("binding"),
            Err(crate::Error::BadMac(_)) => return reject("mac"),
            Err(_) => return reject("signature"),
        }
        let sas = match math::sas(&p.offer.desktop_endpoint.to_bytes(), &c.mobile_endpoint.to_bytes(), &p.offer.nonce, &p.pid_bytes) {
            Ok(s) => s,
            Err(_) => return reject("malformed"),
        };
        let event = PairEvent::AwaitingSas { pid: pid.clone(), phone_name: c.display_name.clone(), phone_thumbprint: c.mobile_endpoint.thumbprint() };
        p.phase = Phase::AwaitingSas(Verified { claims: response.claims, sas, attempts: 0, confirmed: false });
        event
    }

    /// The SAS the phone shows, for a test or a host that compares them itself (the owner types the phone's code; the desktop never displays this).
    pub fn expected_sas(&self, pid: &str) -> Option<Sas> {
        match &self.pending.get(pid)?.phase {
            Phase::AwaitingSas(v) => Some(v.sas.clone()),
            _ => None,
        }
    }

    /// Handles a `pair` item end to end: judges it and, when it fails a check, tells the relay to return the rendezvous to `open`.
    pub fn on_pair_item(&mut self, client: &RelayClient, token: &Token, item: &Item, cancel: &Cancel) -> Result<PairEvent, PairingError> {
        if item.lane != "pair" || item.from != "relay" {
            return Ok(PairEvent::Ignored("not a pair item from the relay"));
        }
        let event = self.receive_response(&item.id, &item.body, client.relay_now_or_local());
        if let PairEvent::Rejected { pid, reason } = &event {
            // Best effort: a reject that fails leaves the rendezvous `answered`, and the owner can only wait it out or burn it.
            let _ = client.pair_reject(token, pid, reason, cancel);
        }
        Ok(event)
    }

    /// Judges what the owner typed. Pure (no I/O): `Confirmed` means the SAS was right and an approval may be made; `Exhausted` means three wrong entries and the caller must deny
    /// and burn (`confirm_sas` does).
    pub fn submit_sas(&mut self, pid: &str, typed: &str) -> Result<SasStep, PairingError> {
        let p = self.pending.get_mut(pid).ok_or(PairingError::UnknownPairing)?;
        let Phase::AwaitingSas(v) = &mut p.phase else {
            return Err(PairingError::WrongState("no response is waiting for a code"));
        };
        if v.confirmed {
            return Ok(SasStep::Confirmed);
        }
        Ok(match math::judge_sas_entry(&v.sas, typed) {
            SasEntry::Incomplete => SasStep::Incomplete,
            SasEntry::Invalid => SasStep::Invalid,
            SasEntry::BadCheck => SasStep::BadCheck,
            SasEntry::Wrong => {
                v.attempts += 1;
                if v.attempts >= SAS_ATTEMPTS {
                    p.phase = Phase::Done;
                    SasStep::Exhausted
                } else {
                    SasStep::Wrong { attempts_left: SAS_ATTEMPTS - v.attempts }
                }
            }
            SasEntry::Right => {
                v.confirmed = true;
                SasStep::Confirmed
            }
        })
    }

    /// The approval body for a pairing whose SAS was typed correctly: `{"approve":true,"phone":{...},"name","appId","grants","receipt":{"issuedAt","signature"}}` with the receipt signed by
    /// the desktop endpoint key over the canonical document. **Refused with [`PairingError::SasRequired`] for a pairing whose code has not been typed.** Consumes the response's nonce
    /// and `jti`. Calling it again for the same pairing returns the same body (a retry of a decision whose answer was lost sends the same receipt).
    pub fn approve_body(&mut self, pid: &str, grants: &[String], relay_now: i64) -> Result<String, PairingError> {
        let p = self.pending.get_mut(pid).ok_or(PairingError::UnknownPairing)?;
        if let Phase::Approving { body } = &p.phase {
            return Ok(body.clone());
        }
        let Phase::AwaitingSas(v) = &p.phase else {
            return Err(PairingError::WrongState("no response is waiting for a decision"));
        };
        if !v.confirmed {
            return Err(PairingError::SasRequired);
        }
        if grants.len() > 16 || grants.iter().any(|g| !ids::is_known_grant(g)) || grants.iter().enumerate().any(|(i, g)| grants[..i].contains(g)) {
            return Err(PairingError::Protocol(crate::Error::Invalid("grants")));
        }
        let c = &v.claims;
        let phone_thumb = c.mobile_endpoint.thumbprint();
        let issued_at = u64::try_from(relay_now).map_err(|_| PairingError::Protocol(crate::Error::Invalid("time")))?;
        let signature = math::sign_receipt(&self.identity.endpoint, &self.app_id, grants, issued_at, &phone_thumb, pid)?;
        let name = ids::clean_name(c.display_name.as_deref().unwrap_or("Phone"), 60);
        let body = Json::obj([
            ("approve", Json::Bool(true)),
            (
                "phone",
                Json::obj([
                    ("ed25519", Json::str(c.mobile_endpoint.to_b64u())),
                    ("x25519", Json::str(c.mobile_x25519.to_b64u())),
                    ("thumbprint", Json::str(phone_thumb)),
                ]),
            ),
            ("name", Json::str(if name.is_empty() { "Phone".to_string() } else { name })),
            ("appId", Json::str(self.app_id.clone())),
            ("grants", Json::Arr(grants.iter().map(|g| Json::str(g.clone())).collect())),
            ("receipt", Json::obj([("issuedAt", Json::int(issued_at)), ("signature", Json::str(signature))])),
        ])
        .to_compact();
        self.consumed.insert((b64::encode(&c.pairing_nonce), c.jti.clone()));
        p.phase = Phase::Approving { body: body.clone() };
        Ok(body)
    }

    /// Marks an approval as delivered.
    pub fn approved(&mut self, pid: &str) {
        if let Some(p) = self.pending.get_mut(pid) {
            p.phase = Phase::Done;
        }
    }

    /// The owner's decision and the I/O of it: judges the typed code and, according to it, tells the relay (`confirm_sas` is the whole gate: the right code approves, a third wrong
    /// one denies and burns, everything else is a mistyped character and does nothing).
    #[allow(clippy::too_many_arguments)]
    pub fn confirm_sas(
        &mut self,
        client: &RelayClient,
        token: &Token,
        pid: &str,
        typed: &str,
        grants: &[String],
        cancel: &Cancel,
    ) -> Result<SasOutcome, PairingError> {
        match self.submit_sas(pid, typed)? {
            SasStep::Incomplete => Ok(SasOutcome::Incomplete),
            SasStep::Invalid => Ok(SasOutcome::Invalid),
            SasStep::BadCheck => Ok(SasOutcome::BadCheck),
            SasStep::Wrong { attempts_left } => Ok(SasOutcome::Wrong { attempts_left }),
            SasStep::Exhausted => {
                let _ = client.pair_decision(token, pid, "{\"approve\":false}", cancel);
                let _ = client.pair_burn(token, pid, cancel);
                Ok(SasOutcome::Denied)
            }
            SasStep::Confirmed => {
                let body = self.approve_body(pid, grants, client.relay_now_or_local())?;
                let ack = client.pair_decision(token, pid, &body, cancel)?;
                self.approved(pid);
                Ok(SasOutcome::Approved {
                    device_id: ack.device_id.ok_or(PairingError::Protocol(crate::Error::Invalid("no device id in an approval")))?,
                })
            }
        }
    }

    /// The owner says no: tells the relay and forgets the pairing.
    pub fn deny(&mut self, client: &RelayClient, token: &Token, pid: &str, cancel: &Cancel) -> Result<(), PairingError> {
        if self.pending.remove(pid).is_none() {
            return Err(PairingError::UnknownPairing);
        }
        client.pair_decision(token, pid, "{\"approve\":false}", cancel)?;
        Ok(())
    }

    /// Ends a rendezvous now (the owner closes the dialog, or the offer expired).
    pub fn burn(&mut self, client: &RelayClient, token: &Token, pid: &str, cancel: &Cancel) -> Result<(), PairingError> {
        self.pending.remove(pid);
        client.pair_burn(token, pid, cancel)?;
        Ok(())
    }
}

/// What a typed code led to, before any I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SasStep {
    /// Fewer than 13 characters.
    Incomplete,
    /// Not the alphabet, or too long.
    Invalid,
    /// A wrong check character.
    BadCheck,
    /// A wrong code; attempts left.
    Wrong {
        /// Attempts left.
        attempts_left: u8,
    },
    /// Three wrong codes.
    Exhausted,
    /// The right code.
    Confirmed,
}
