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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use oaiy_crypto::zeroize::Secret;

use crate::b64;
use crate::client::http::Cancel;
use crate::client::pair::PairCreate;
use crate::client::store::Item;
use crate::client::RelayClient;
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
/// The most pairings in flight at once (an offer lives ten minutes; a desktop shows one at a time): a new offer is refused beyond it, after the finished and the expired ones are
/// dropped.
pub const MAX_PENDING: usize = 32;
/// The most nonces and `jti`s that approvals consumed that the party remembers (the oldest are forgotten first: an offer is good for ten minutes, so an old one cannot come back).
pub const MAX_CONSUMED: usize = 1024;

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

/// A new offer: what the owner is shown (the QR, the link, the typed code) and what is sent to the relay. The two texts that hold the secret are wiped when it is dropped.
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

impl Drop for NewOffer {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.pairing_uri.zeroize();
        self.typed_code.zeroize();
    }
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
    /// The approval is made and being sent: the body is kept so that a retry sends the same receipt, and the verified response with the SAS so that a retry is judged against the
    /// code the owner types again.
    Approving {
        v: Verified,
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

impl Pending {
    /// The pairing is over: its MAC key is wiped now and not at some later drop.
    fn finish(&mut self) {
        self.phase = Phase::Done;
        self.mac_key = Secret::new([0u8; 32]);
    }
}

/// An insertion-ordered set with a cap: when it is full the oldest member is forgotten.
struct Bounded<T: std::hash::Hash + Eq + Clone> {
    set: HashSet<T>,
    order: VecDeque<T>,
    cap: usize,
}

impl<T: std::hash::Hash + Eq + Clone> Bounded<T> {
    fn new(cap: usize) -> Self {
        Bounded { set: HashSet::new(), order: VecDeque::new(), cap }
    }

    fn contains(&self, v: &T) -> bool {
        self.set.contains(v)
    }

    /// True when it was not there.
    fn insert(&mut self, v: T) -> bool {
        if !self.set.insert(v.clone()) {
            return false;
        }
        self.order.push_back(v);
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        true
    }

    fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        self.order.retain(|v| keep(v));
        self.set.retain(|v| self.order.contains(v));
    }
}

/// The desktop's pairings in flight.
pub struct DesktopPairing {
    identity: Arc<DesktopIdentity>,
    app_id: String,
    relay: RelayUrl,
    relay_fingerprint: String,
    pending: HashMap<String, Pending>,
    consumed: Bounded<(String, String)>,
    revoked: HashSet<String>,
    /// The ids of the pair items already judged: a poll delivers at least once, and a response that was judged and rejected must not be judged (and rejected) again, which would
    /// return a later, good response of the same rendezvous to open under the phone. Only items of a pairing in flight are remembered, and they go with it.
    seen: Bounded<String>,
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
            consumed: Bounded::new(MAX_CONSUMED),
            revoked: HashSet::new(),
            seen: Bounded::new(4 * MAX_PENDING),
        }
    }

    /// Makes the party refuse a response from the phone key with this thumbprint (a key that was revoked).
    pub fn revoke_phone_key(&mut self, thumbprint: &str) {
        self.revoked.insert(thumbprint.to_string());
    }

    /// The nonces and `jti`s that approvals have consumed, for a host that keeps them across restarts.
    pub fn consumed(&self) -> Vec<(String, String)> {
        let mut v: Vec<_> = self.consumed.order.iter().cloned().collect();
        v.sort();
        v
    }

    /// Restores consumed nonces and `jti`s.
    pub fn restore_consumed(&mut self, consumed: impl IntoIterator<Item = (String, String)>) {
        for c in consumed {
            self.consumed.insert(c);
        }
    }

    /// How much this party remembers, as `(pairings, judged item ids, consumed nonces)`: each is bounded ([`MAX_PENDING`], four times that, [`MAX_CONSUMED`]), whatever a relay
    /// delivers.
    pub fn remembered(&self) -> (usize, usize, usize) {
        (self.pending.len(), self.seen.order.len(), self.consumed.order.len())
    }

    /// The number of rendezvous this party has not finished.
    pub fn open_count(&self) -> usize {
        self.pending.values().filter(|p| !matches!(p.phase, Phase::Done)).count()
    }

    /// Makes an offer at `relay_now`, with a secret, a nonce and a `jti` that the operating system's random generator draws **inside this crate**: the host passes no random
    /// source for them, so that no host can make the pairing secret predictable (a seeded generator in a test, a poor one on a platform).
    pub fn create_offer(&mut self, relay_now: i64) -> Result<NewOffer, PairingError> {
        let secret = Secret::<16>::random().map_err(crate::Error::from)?;
        let nonce = Secret::<32>::random().map_err(crate::Error::from)?;
        let jti = Secret::<12>::random().map_err(crate::Error::from)?;
        self.make_offer(*secret.expose(), *nonce.expose(), format!("pair-{}", b64::encode(jti.expose())), relay_now)
    }

    /// Makes an offer from given randomness: what the recorded ceremony and the vectors fix, and nothing else. **Only in a build with the `testing` feature**, which a product does
    /// not enable: it is the one way to make an offer whose secret a caller chose, and a secret a caller chose is a secret the caller can guess.
    #[cfg(any(test, feature = "testing"))]
    #[doc(hidden)]
    pub fn create_offer_with(&mut self, secret: [u8; 16], nonce: [u8; 32], jti: String, relay_now: i64) -> Result<NewOffer, PairingError> {
        self.make_offer(secret, nonce, jti, relay_now)
    }

    fn make_offer(&mut self, secret: [u8; 16], nonce: [u8; 32], jti: String, relay_now: i64) -> Result<NewOffer, PairingError> {
        let secret = PairingSecret::new(secret);
        let derived = secret.derive()?;
        let pid = math::pid_text(&derived.pid);
        // The finished and the expired pairings are dropped first; one that is still in flight under this pid is never replaced (that would hand it three new attempts at the SAS),
        // and there is a limit to how many are open at once.
        self.pending.retain(|_, p| !matches!(p.phase, Phase::Done) && (p.offer.expires_at as i64) + crate::pairing::offer::SKEW_S > relay_now);
        let pending = &self.pending;
        self.seen.retain(|item| pid_of_item(item).is_some_and(|p| pending.contains_key(p)));
        if self.pending.contains_key(&pid) {
            return Err(PairingError::WrongState("a pairing with this secret is in flight"));
        }
        if self.pending.len() >= MAX_PENDING {
            return Err(PairingError::WrongState("too many pairings are in flight"));
        }
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
        // An item is remembered only if it belongs to a pairing in flight (a relay cannot make the party remember anything by naming pairings that do not exist).
        let Some(p) = self.pending.get_mut(&pid) else {
            return PairEvent::Ignored("no such pairing");
        };
        if self.seen.contains(&item_id.to_string()) {
            return PairEvent::Ignored("this item was judged already");
        }
        if !matches!(p.phase, Phase::AwaitingResponse) {
            return PairEvent::Ignored("this pairing is not waiting for a response");
        }
        self.seen.insert(item_id.to_string());
        let reject = |reason: &'static str| PairEvent::Rejected { pid: pid.clone(), reason };
        // The offer's own window (README 10.1 step 5, the live challenge): a response is not taken to an offer that has expired, whatever the response's own window says.
        if p.offer.check_window(relay_now).is_err() {
            return reject("window");
        }
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
        // What the owner types is judged **every time**, also after the code was right once: a pairing that is confirmed and not yet approved (a grant list the host refused) or being
        // approved (a decision whose answer was lost) is approved only by the code, typed again, and never by whatever comes next.
        let (v, counting) = match &mut p.phase {
            Phase::AwaitingSas(v) => (v, true),
            Phase::Approving { v, .. } => (v, false),
            _ => return Err(PairingError::WrongState("no response is waiting for a code")),
        };
        let step = match math::judge_sas_entry(&v.sas, typed) {
            SasEntry::Incomplete => SasStep::Incomplete,
            SasEntry::Invalid => SasStep::Invalid,
            SasEntry::BadCheck => SasStep::BadCheck,
            SasEntry::Wrong if !counting => {
                // An approval has been sent: a wrong entry now costs nothing and never denies (the relay may have approved already, and a denial after it is a conflict).
                SasStep::Wrong { attempts_left: SAS_ATTEMPTS.saturating_sub(v.attempts) }
            }
            SasEntry::Wrong => {
                v.attempts += 1;
                if v.attempts >= SAS_ATTEMPTS {
                    p.finish();
                    return Ok(SasStep::Exhausted);
                }
                SasStep::Wrong { attempts_left: SAS_ATTEMPTS - v.attempts }
            }
            SasEntry::Right => {
                v.confirmed = true;
                SasStep::Confirmed
            }
        };
        Ok(step)
    }

    /// The approval body for a pairing whose SAS was typed correctly: `{"approve":true,"phone":{...},"name","appId","grants","receipt":{"issuedAt","signature"}}` with the receipt signed by
    /// the desktop endpoint key over the canonical document. **Refused with [`PairingError::SasRequired`] for a pairing whose code has not been typed.** Consumes the response's nonce
    /// and `jti`. Calling it again for the same pairing returns the same body (a retry of a decision whose answer was lost sends the same receipt).
    pub fn approve_body(&mut self, pid: &str, grants: &[String], relay_now: i64) -> Result<String, PairingError> {
        let p = self.pending.get_mut(pid).ok_or(PairingError::UnknownPairing)?;
        if let Phase::Approving { body, .. } = &p.phase {
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
        let Phase::AwaitingSas(v) = std::mem::replace(&mut p.phase, Phase::Done) else {
            return Err(PairingError::WrongState("no response is waiting for a decision"));
        };
        p.phase = Phase::Approving { v, body: body.clone() };
        Ok(body)
    }

    /// Marks an approval as delivered.
    pub fn approved(&mut self, pid: &str) {
        if let Some(p) = self.pending.get_mut(pid) {
            p.finish();
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
