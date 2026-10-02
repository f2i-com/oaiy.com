//! `GET /v1/info` and the relay's identity proof (README section 8).
//!
//! The document is static and signed (`X-OAIY-Sig`), which proves only that its bytes were once signed by the relay's key: anyone can copy the body and the signature.
//! **Identity** is the interactive proof: a request with `X-OAIY-Nonce` is answered with the same body and `X-OAIY-Proof`, an Ed25519 signature over
//! `"oaiy/relay/1/info-proof" || 0x00 || nonce bytes || SHA-256(body) || ASCII decimal of X-OAIY-Time`. [`verify_proof`] is the whole check a client makes before it sends
//! a bearer: the body is a valid `info`, `relayKey` is a good key whose thumbprint is the one pinned out of band, the proof verifies over the client's own nonce and over
//! the time of the same answer. A replayed body and signature with a new nonce fails (vector A6).
//!
//! It does not detect a live forwarding proxy that holds a valid certificate for the relay's name (there is no channel binding to TLS): README 8, rule 4.

use oaiy_crypto::kdf::sha256;
use oaiy_crypto::zeroize::ct_eq;

use crate::error::{Error, Result};
use crate::json::{self, Json};
use crate::keys::{SignDomain, VerifyKey};

/// The `wait` object: how long a poll may be held and how a client paces itself (README 5.1 and 5.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wait {
    /// `wait.default`: the `wait` of a poll that is to be held (20 unless the relay says less).
    pub default: u64,
    /// `wait.max`.
    pub max: u64,
    /// `wait.pollGapMs`: the pause after a poll that made no progress (250 by default).
    pub poll_gap_ms: u64,
    /// `wait.fallbackS`: the cap of the pause after a refused hold (5 by default).
    pub fallback_s: u64,
}

/// One lane's limits: `limits.lanes.<lane>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaneLimits {
    /// The most bytes of a body.
    pub body: u64,
    /// The TTL a post gets when it names none.
    pub ttl_default: u64,
    /// The smallest TTL.
    pub ttl_min: u64,
    /// The largest TTL.
    pub ttl_max: u64,
}

/// What a client reads of `limits`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// Items in one `POST /v1/items` (at most 64).
    pub batch_items: u64,
    /// Bytes of one `POST /v1/items` (at most 1 MiB).
    pub batch_bytes: u64,
    /// Live items in a mailbox.
    pub mailbox_items: u64,
    /// Live bytes in a mailbox.
    pub mailbox_bytes: u64,
    /// The cap of a serialised `hdr`.
    pub hdr_bytes: u64,
    /// The lanes this relay serves, with their limits.
    pub lanes: Vec<(String, LaneLimits)>,
}

/// A parsed `info` document.
#[derive(Debug, Clone)]
pub struct Info {
    /// The body exactly as received (the bytes the signature and the proof cover).
    pub raw: Vec<u8>,
    /// `minClient`: a client whose level is lower is answered `426`.
    pub min_client: u64,
    /// `relayId` (`rly-...`).
    pub relay_id: String,
    /// The relay's signing key (`relayKey.publicKey`).
    pub relay_key: VerifyKey,
    /// `software.version`.
    pub software_version: String,
    /// `features`: what the relay implements and has enabled. A client uses a feature only if it is listed.
    pub features: Vec<String>,
    /// `wait`.
    pub wait: Wait,
    /// `presenceWindow`.
    pub presence_window: u64,
    /// `limits`.
    pub limits: Limits,
}

fn uint(v: &Json, what: &'static str) -> Result<u64> {
    v.as_uint53().ok_or(Error::Invalid(what))
}

fn member<'a>(v: &'a Json, key: &str, what: &'static str) -> Result<&'a Json> {
    v.get(key).ok_or(Error::Invalid(what))
}

impl Info {
    /// Reads the document the schema (`info.schema.json`) describes: the members a client needs, strictly typed, and `relayKey.thumbprint` equal to the thumbprint of
    /// `relayKey.publicKey`. Unknown members are ignored (README section "Versioning"); a `time` member is refused (the body is static).
    pub fn parse(body: &[u8]) -> Result<Info> {
        let doc = json::parse(body)?;
        if !doc.is_object() {
            return Err(Error::Invalid("info: not an object"));
        }
        if doc.get("time").is_some() {
            return Err(Error::Invalid("info: a static document has no time"));
        }
        if doc.get_str("protocol") != Some("oaiy-relay/1") {
            return Err(Error::Invalid("info: protocol"));
        }
        let min_client = uint(member(&doc, "minClient", "info: minClient")?, "info: minClient")?;
        if min_client < 1 {
            return Err(Error::Invalid("info: minClient"));
        }
        let relay_id = doc.get_str("relayId").filter(|s| crate::ids::is_relay_id(s)).ok_or(Error::Invalid("info: relayId"))?.to_string();
        let rk = member(&doc, "relayKey", "info: relayKey")?;
        if rk.get_str("algorithm") != Some("ed25519") {
            return Err(Error::Invalid("info: relayKey.algorithm"));
        }
        let relay_key = VerifyKey::from_b64u(rk.get_str("publicKey").ok_or(Error::Invalid("info: relayKey.publicKey"))?)?;
        let stated = rk.get_str("thumbprint").ok_or(Error::Invalid("info: relayKey.thumbprint"))?;
        if !ct_eq(stated.as_bytes(), relay_key.thumbprint().as_bytes()) {
            return Err(Error::Mismatch("info: relayKey.thumbprint is not the thumbprint of relayKey.publicKey"));
        }
        let software = member(&doc, "software", "info: software")?;
        let software_name = software.get_str("name").ok_or(Error::Invalid("info: software.name"))?;
        let software_version = software.get_str("version").ok_or(Error::Invalid("info: software.version"))?;
        if software_name.len() > 64 || software_version.len() > 32 {
            return Err(Error::Invalid("info: software"));
        }
        let features = member(&doc, "features", "info: features")?
            .as_array()
            .ok_or(Error::Invalid("info: features"))?
            .iter()
            .map(|f| f.as_str().map(str::to_string).ok_or(Error::Invalid("info: features")))
            .collect::<Result<Vec<_>>>()?;
        let w = member(&doc, "wait", "info: wait")?;
        let wait = Wait {
            default: uint(member(w, "default", "info: wait.default")?, "info: wait.default")?,
            max: uint(member(w, "max", "info: wait.max")?, "info: wait.max")?,
            poll_gap_ms: uint(member(w, "pollGapMs", "info: wait.pollGapMs")?, "info: wait.pollGapMs")?,
            fallback_s: uint(member(w, "fallbackS", "info: wait.fallbackS")?, "info: wait.fallbackS")?,
        };
        if wait.default > 300 || wait.max > 300 || wait.fallback_s < 1 {
            return Err(Error::Invalid("info: wait"));
        }
        let presence_window = uint(member(&doc, "presenceWindow", "info: presenceWindow")?, "info: presenceWindow")?;
        let l = member(&doc, "limits", "info: limits")?;
        let mut lanes = Vec::new();
        for (name, lane) in member(l, "lanes", "info: limits.lanes")?.as_object().ok_or(Error::Invalid("info: limits.lanes"))? {
            let ttl = member(lane, "ttl", "info: limits.lanes.ttl")?;
            lanes.push((
                name.clone(),
                LaneLimits {
                    body: uint(member(lane, "body", "info: limits.lanes.body")?, "info: limits.lanes.body")?,
                    ttl_default: uint(member(ttl, "default", "info: ttl")?, "info: ttl")?,
                    ttl_min: uint(member(ttl, "min", "info: ttl")?, "info: ttl")?,
                    ttl_max: uint(member(ttl, "max", "info: ttl")?, "info: ttl")?,
                },
            ));
        }
        let limits = Limits {
            batch_items: uint(member(l, "batchItems", "info: limits.batchItems")?, "info: limits.batchItems")?,
            batch_bytes: uint(member(l, "batchBytes", "info: limits.batchBytes")?, "info: limits.batchBytes")?,
            mailbox_items: uint(member(l, "mailboxItems", "info: limits.mailboxItems")?, "info: limits.mailboxItems")?,
            mailbox_bytes: uint(member(l, "mailboxBytes", "info: limits.mailboxBytes")?, "info: limits.mailboxBytes")?,
            hdr_bytes: uint(member(l, "hdrBytes", "info: limits.hdrBytes")?, "info: limits.hdrBytes")?,
            lanes,
        };
        member(l, "held", "info: limits.held")?;
        Ok(Info {
            raw: body.to_vec(),
            min_client,
            relay_id,
            relay_key,
            software_version: software_version.to_string(),
            features,
            wait,
            presence_window,
            limits,
        })
    }

    /// True when the relay lists `feature`.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }

    /// The limits of `lane`, when the relay serves it.
    pub fn lane(&self, lane: &str) -> Option<&LaneLimits> {
        self.limits.lanes.iter().find(|(n, _)| n == lane).map(|(_, l)| l)
    }
}

/// `X-OAIY-Time` as the proof covers it: the canonical decimal spelling of a non-negative integer (no sign, no leading zero, no space), at most 16 digits.
pub fn parse_time_header(text: &str) -> Result<i64> {
    let ok = !text.is_empty() && text.len() <= 16 && text.bytes().all(|b| b.is_ascii_digit()) && (text == "0" || !text.starts_with('0'));
    if !ok {
        return Err(Error::Invalid("X-OAIY-Time"));
    }
    text.parse::<i64>().map_err(|_| Error::Invalid("X-OAIY-Time"))
}

/// The proof's own check, with the key already trusted: `proof` is the Ed25519 signature over `info-proof || 0x00 || nonce || SHA-256(body) || ASCII decimal of time`.
pub fn verify_proof_signature(key: &VerifyKey, body: &[u8], nonce: &[u8], time: i64, proof: &str) -> Result<()> {
    let digest = sha256(body);
    let time_text = time.to_string();
    key.verify_b64u(SignDomain::InfoProof, &[nonce, &digest, time_text.as_bytes()], proof)
}

/// The static signature's check: `X-OAIY-Sig` is over `info || 0x00 || body`. It proves the bytes were once signed by this key, nothing about who answers now.
pub fn verify_static_signature(key: &VerifyKey, body: &[u8], signature: &str) -> Result<()> {
    key.verify_b64u(SignDomain::Info, &[body], signature)
}

/// What a client makes of a good proof.
#[derive(Debug, Clone)]
pub struct Proved {
    /// The document, whose values the client then uses (`wait.default`, `pollGapMs`, `fallbackS`, `minClient`).
    pub info: Info,
    /// The relay's `X-OAIY-Time` of the same answer (a sample of the relay's clock).
    pub relay_time: i64,
}

/// The whole check of README 8, rule 2: `body` is a valid `info`; its `relayKey` hashes to `pinned_thumbprint` (the `f` of an enrolment or pairing key, or
/// `offer.relay.fingerprint` for a typed pairing); the proof verifies over `nonce` (the bytes the client sent, 16 to 32 of them) and over `time_header` of the same answer.
/// Any failure is an error and the caller sends no token: it reports the relay as "not who it was".
pub fn verify_proof(body: &[u8], nonce: &[u8], time_header: &str, proof_header: &str, pinned_thumbprint: &str) -> Result<Proved> {
    if !(16..=32).contains(&nonce.len()) {
        return Err(Error::Invalid("nonce: 16 to 32 bytes"));
    }
    let info = Info::parse(body)?;
    if !ct_eq(info.relay_key.thumbprint().as_bytes(), pinned_thumbprint.as_bytes()) {
        return Err(Error::Mismatch("the relay's key is not the pinned one"));
    }
    let relay_time = parse_time_header(time_header)?;
    verify_proof_signature(&info.relay_key, body, nonce, relay_time, proof_header)?;
    Ok(Proved { info, relay_time })
}
