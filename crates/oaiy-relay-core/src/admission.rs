//! The Aokie admission (README 10.6): the requests of the two roles, the bearer's shape, and the answers read as the shipped decoders read them.
//!
//! `aokie-adm-v2.` + lowercase hex of the claims JSON + `.` + lowercase hex of the relay's HMAC-SHA-256 of those exact bytes. A client cannot verify the HMAC (the relay
//! keeps the secret) and treats the whole token as opaque: it sends it back as `Authorization: Bearer` on the compatibility routes and reads only its shape, so that a
//! malformed answer is refused before it is used. The sizes of vectors A4 and A4b (888 characters for a phone, 964 and 92 more per phone for the plugin) are what
//! [`Bearer::len`] reports.

use zeroize::Zeroizing;

use crate::error::{Error, Result};
use crate::ids;
use crate::json::{self, Json};
use crate::keys::VerifyKey;

/// The bearer's prefix.
pub const PREFIX: &str = "aokie-adm-v2.";
/// The longest bearer `admission-mobile-response.schema.json` allows.
pub const MAX_LEN: usize = 8192;

/// An admission bearer: a credential for the compatibility routes, wiped when dropped and printed nowhere.
pub struct Bearer {
    text: Zeroizing<String>,
    claims: Json,
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    // (`len() % 2` and not `is_multiple_of`, which is newer than the toolchains this crate is built with elsewhere.)
    #[allow(clippy::manual_is_multiple_of)]
    let odd = text.len() % 2 != 0;
    if odd || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return None;
    }
    text.as_bytes().chunks(2).map(|p| u8::from_str_radix(core::str::from_utf8(p).ok()?, 16).ok()).collect()
}

impl Bearer {
    /// Reads the shape: the prefix, lowercase hex that decodes to a JSON object with an integer `exp`, a dot and exactly 64 lowercase hex characters.
    pub fn parse(text: &str) -> Result<Bearer> {
        if text.len() > MAX_LEN {
            return Err(Error::Invalid("bearer: too long"));
        }
        let rest = text.strip_prefix(PREFIX).ok_or(Error::Invalid("bearer: prefix"))?;
        let (claims_hex, mac_hex) = rest.split_once('.').ok_or(Error::Invalid("bearer: shape"))?;
        if mac_hex.len() != 64 || unhex(mac_hex).is_none() {
            return Err(Error::Invalid("bearer: mac"));
        }
        let claims_bytes = unhex(claims_hex).filter(|b| !b.is_empty()).ok_or(Error::Invalid("bearer: claims"))?;
        let claims = json::parse(&claims_bytes)?;
        if !claims.is_object() || claims.get("exp").and_then(Json::as_uint53).is_none() {
            return Err(Error::Invalid("bearer: claims"));
        }
        Ok(Bearer { text: Zeroizing::new(text.to_string()), claims })
    }

    /// The token, for `Authorization: Bearer`. Keep the borrow short.
    pub fn expose(&self) -> &str {
        &self.text
    }

    /// Its length in characters (888 for a phone in vector A4).
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Always false: a bearer has claims.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The claims, as the relay wrote them (advisory to a client: the relay is the party that reads them).
    pub fn claims(&self) -> &Json {
        &self.claims
    }

    /// `exp`, Unix seconds in relay time. The relay accepts the bearer up to and including `exp + 30` seconds.
    pub fn expires_at(&self) -> u64 {
        self.claims.get("exp").and_then(Json::as_uint53).unwrap_or(0)
    }
}

impl core::fmt::Debug for Bearer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Bearer({} characters, redacted)", self.text.len())
    }
}

/// The transports a carrier offers (`supportedTransports`): `relay` is the framed stream, `relay-poll` the poll mode (Interpretation 40).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// `relay`: the SSE-framed stream of the compatibility routes.
    Relay,
    /// `relay-poll`: `GET frames?since=&wait=` in a loop.
    RelayPoll,
}

impl Transport {
    /// The wire word.
    pub const fn as_str(self) -> &'static str {
        match self {
            Transport::Relay => "relay",
            Transport::RelayPoll => "relay-poll",
        }
    }
}

fn transports_json(transports: &[Transport]) -> Json {
    Json::Arr(transports.iter().map(|t| Json::str(t.as_str())).collect())
}

/// What a phone sends to `POST /v1/admission` with its token (`admission-mobile-request.schema.json`).
#[derive(Debug, Clone)]
pub struct MobileRequest {
    /// The app the phone was paired for.
    pub app_id: String,
    /// The phone's id as the relay knows it (`dev-...`).
    pub device_id: String,
    /// A display name (cleaned to 60 characters), optional.
    pub display_name: Option<String>,
    /// The thumbprint of the phone's endpoint key.
    pub holder_thumbprint: String,
    /// What the phone's carrier can open. Absent means `["relay"]` to the relay. The relay serves `relay` as the framed stream only when the host passed the streaming
    /// probe, and a list with neither is `422` (Interpretation 40): a phone that polls sends `RelayPoll`.
    pub transports: Option<Vec<Transport>>,
}

impl MobileRequest {
    /// The body text, in the order the shipped phone writes it: `appId`, `deviceId`, `displayName`, `holderKeyThumbprint`, `supportedTransports`.
    pub fn to_body(&self) -> Result<String> {
        if !ids::is_app_id(&self.app_id) || !ids::is_device_id(&self.device_id) || !ids::is_thumbprint(&self.holder_thumbprint) {
            return Err(Error::Invalid("mobile admission request"));
        }
        let mut m = vec![("appId", Json::str(self.app_id.clone())), ("deviceId", Json::str(self.device_id.clone()))];
        if let Some(n) = &self.display_name {
            m.push(("displayName", Json::str(ids::clean_name(n, 60))));
        }
        m.push(("holderKeyThumbprint", Json::str(self.holder_thumbprint.clone())));
        if let Some(t) = &self.transports {
            if t.is_empty() || t.len() > 2 || (t.len() == 2 && t[0] == t[1]) {
                return Err(Error::Invalid("supportedTransports"));
            }
            m.push(("supportedTransports", transports_json(t)));
        }
        Ok(Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()).to_compact())
    }
}

/// What the desktop's broker sends for the plugin (`admission-plugin-request.schema.json`).
#[derive(Debug, Clone)]
pub struct PluginRequest {
    /// The app.
    pub app_id: String,
    /// The plugin's id.
    pub plugin_id: String,
    /// A display name, optional.
    pub display_name: Option<String>,
    /// The plugin's endpoint key (the desktop endpoint key).
    pub endpoint: VerifyKey,
    /// The thumbprints of the phones the desktop approved, strictly ascending bytewise, none the plugin's own.
    pub approved_peers: Vec<String>,
    /// The roster revision, 1 to 2^53 - 1.
    pub revision: u64,
    /// What the plugin's carrier can open; absent is read as `["relay"]`.
    pub transports: Option<Vec<Transport>>,
}

impl PluginRequest {
    /// The body text, in the order the desktop's broker writes it, with the roster hash recomputed.
    pub fn to_body(&self) -> Result<String> {
        let own = self.endpoint.thumbprint();
        if !ids::is_app_id(&self.app_id) || !ids::is_app_id(&self.plugin_id) || self.approved_peers.is_empty() || self.revision == 0 {
            return Err(Error::Invalid("plugin admission request"));
        }
        crate::roster::check(&own, self.revision, &self.approved_peers)?;
        let mut m = vec![("appId", Json::str(self.app_id.clone())), ("pluginId", Json::str(self.plugin_id.clone()))];
        if let Some(n) = &self.display_name {
            m.push(("displayName", Json::str(ids::clean_name(n, 60))));
        }
        m.push((
            "endpointPublicKey",
            Json::obj([
                ("algorithm", Json::str("ed25519")),
                ("publicKey", Json::str(self.endpoint.to_b64u())),
                ("thumbprint", Json::str(own.clone())),
            ]),
        ));
        m.push(("holderKeyThumbprint", Json::str(own)));
        m.push(("approvedPeerKeyThumbprints", Json::Arr(self.approved_peers.iter().map(|t| Json::str(t.clone())).collect())));
        m.push(("peerRosterRevision", Json::int(self.revision)));
        m.push(("peerRosterHash", Json::str(crate::roster::hash(self.revision, &self.approved_peers)?)));
        if let Some(t) = &self.transports {
            if t.is_empty() || t.len() > 2 || (t.len() == 2 && t[0] == t[1]) {
                return Err(Error::Invalid("supportedTransports"));
            }
            m.push(("supportedTransports", transports_json(t)));
        }
        Ok(Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()).to_compact())
    }
}

/// One entry of `iceServers`, validated by the rules both shipped decoders apply (and `aokie-media`'s `IceServerConfig::validate_all`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IceServer {
    /// 1 to 8 URLs of at most 2,048 bytes beginning `stun:`, `stuns:`, `turn:` or `turns:`, none mixing STUN and TURN.
    pub urls: Vec<String>,
    /// A TURN entry's username (`<expiry>:<opaque id>`); empty for STUN.
    pub username: String,
    /// A TURN entry's credential; empty for STUN.
    pub credential: String,
    /// A TURN entry's expiry, 31 seconds to 24 hours ahead.
    pub expires_at: Option<u64>,
}

impl IceServer {
    /// True for a TURN entry (any URL that begins `turn:` or `turns:`).
    pub fn is_turn(&self) -> bool {
        self.urls.iter().any(|u| u.starts_with("turn:") || u.starts_with("turns:"))
    }
}

fn parse_ice(servers: &Json, relay_only: bool, now: i64) -> Result<(Vec<IceServer>, Option<u64>)> {
    let bad = Error::Invalid("iceServers");
    let list = servers.as_array().ok_or(bad.clone())?;
    if list.len() > 8 {
        return Err(bad);
    }
    let mut out = Vec::new();
    for s in list {
        let members = s.as_object().ok_or(bad.clone())?;
        let urls: Vec<String> = s
            .get("urls")
            .and_then(Json::as_array)
            .filter(|u| (1..=8).contains(&u.len()))
            .ok_or(bad.clone())?
            .iter()
            .map(|u| {
                u.as_str()
                    .filter(|t| {
                        t.len() <= 2048
                            && !t.chars().any(|c| c.is_control())
                            && ["stun:", "stuns:", "turn:", "turns:"].iter().any(|p| t.starts_with(p))
                    })
                    .map(str::to_string)
                    .ok_or(bad.clone())
            })
            .collect::<Result<_>>()?;
        let username = s.get_str("username").ok_or(bad.clone())?.to_string();
        let credential = s.get_str("credential").ok_or(bad.clone())?.to_string();
        let turn = urls.iter().any(|u| u.starts_with("turn"));
        let stun = urls.iter().any(|u| u.starts_with("stun"));
        if turn && stun {
            return Err(bad);
        }
        let expires_at = match s.get("expiresAt") {
            None => None,
            Some(e) => Some(e.as_uint53().ok_or(bad.clone())?),
        };
        if turn {
            let ahead = expires_at.map(|e| e as i64 - now);
            let ok = !username.is_empty()
                && username.len() <= 512
                && !credential.is_empty()
                && credential.len() <= 2048
                && ahead.is_some_and(|a| a > 30 && a <= 86_400)
                && members.len() == 4;
            if !ok {
                return Err(bad);
            }
        } else if !(username.is_empty() && credential.is_empty() && expires_at.is_none() && members.len() == 3) {
            return Err(bad);
        }
        out.push(IceServer { urls, username, credential, expires_at });
    }
    let earliest = out.iter().filter(|s| s.is_turn()).filter_map(|s| s.expires_at).min();
    if relay_only && earliest.is_none() {
        return Err(bad);
    }
    Ok((out, earliest))
}

/// The three URLs of the compatibility routes (byte-identical in every admission, with no per-admission query string), and whether the relay offers the poll mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayEndpoints {
    /// `GET .../relay/challenge`.
    pub challenge_url: String,
    /// `POST` and `GET .../relay/frames`.
    pub frames_url: String,
    /// `GET .../relay/stream`.
    pub stream_url: String,
    /// `mode: "poll"`: a poll-mode carrier.
    pub poll_mode: bool,
}

fn parse_endpoints(v: &Json) -> Result<RelayEndpoints> {
    let bad = Error::Invalid("admission: relay");
    let members = v.as_object().ok_or(bad.clone())?;
    if members.iter().any(|(k, _)| !matches!(k.as_str(), "challengeUrl" | "framesUrl" | "streamUrl" | "mode")) {
        return Err(bad);
    }
    let url = |k: &str| v.get_str(k).filter(|u| u.starts_with("https://")).map(str::to_string).ok_or(bad.clone());
    let poll_mode = match v.get("mode") {
        None => false,
        Some(m) if m.as_str() == Some("poll") => true,
        Some(_) => return Err(bad),
    };
    Ok(RelayEndpoints { challenge_url: url("challengeUrl")?, frames_url: url("framesUrl")?, stream_url: url("streamUrl")?, poll_mode })
}

fn gateway_ok(url: &str) -> bool {
    url.strip_prefix("wss://")
        .and_then(|r| r.strip_suffix("/v2/realtime"))
        .is_some_and(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-' | b':')))
}

fn members_are(v: &Json, allowed: &[&str], required: &[&str]) -> bool {
    v.as_object().is_some_and(|m| m.iter().all(|(k, _)| allowed.contains(&k.as_str())) && required.iter().all(|k| v.get(k).is_some()))
}

fn scopes_of(v: &Json) -> Result<Vec<String>> {
    let bad = Error::Invalid("admission: scopes");
    let list = v.as_array().filter(|l| (1..=16).contains(&l.len())).ok_or(bad.clone())?;
    let scopes: Vec<String> =
        list.iter().map(|s| s.as_str().filter(|g| ids::is_grant(g)).map(str::to_string).ok_or(bad.clone())).collect::<Result<_>>()?;
    let mut sorted = scopes.clone();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != scopes.len() {
        return Err(bad);
    }
    Ok(scopes)
}

/// What a phone checks an admission against: its own session.
#[derive(Debug, Clone, Copy)]
pub struct MobileExpect<'a> {
    /// The app the phone was paired for.
    pub app_id: &'a str,
    /// The phone's own device id.
    pub device_id: &'a str,
    /// The thumbprint of the phone's endpoint key.
    pub holder_thumbprint: &'a str,
}

/// The phone's admission (`admission-mobile-response.schema.json`), read as the shipped phone's `validate_admission` reads it: exactly the schema's members; a bearer; `expiresIn`
/// 1 to 300 and `expiresAt` in `(now, now + 300]`; `appId`, the device id and the holder thumbprint the phone's own; `role` `mobile`; scopes of known grants that include
/// `state_read`; an `expectedPeerKeyThumbprint` that is not the holder's; a `device` of exactly eight members whose grants are the scopes; ICE as the decoders read it.
///
/// **`expectedPeerKeyThumbprint`, `iceServers`, `scopes` and `device` are advisory**: the phone's desktop pin comes only from the MAC-verified offer, and an admission that names
/// another expected peer than the pinned desktop's is a hard error for the carrier (MOB-21b), not for this reader.
#[derive(Debug)]
pub struct MobileAdmission {
    /// The bearer for the compatibility routes.
    pub bearer: Bearer,
    /// `expiresIn`.
    pub expires_in: u64,
    /// `expiresAt`, relay time.
    pub expires_at: u64,
    /// `gatewayUrl`: never dialled.
    pub gateway_url: String,
    /// `appId`.
    pub app_id: String,
    /// `subjectId`: the phone's device id.
    pub subject_id: String,
    /// `scopes`.
    pub scopes: Vec<String>,
    /// `iceServers`.
    pub ice_servers: Vec<IceServer>,
    /// `relayOnly`.
    pub relay_only: bool,
    /// `turnCredentialExpiresAt`: the earliest TURN expiry, or none.
    pub turn_credential_expires_at: Option<u64>,
    /// `holderKeyThumbprint`.
    pub holder_thumbprint: String,
    /// `expectedPeerKeyThumbprint`: advisory.
    pub expected_peer_thumbprint: String,
    /// `relay`.
    pub relay: RelayEndpoints,
}

impl MobileAdmission {
    /// Reads and checks the answer, with `now` the relay-corrected time.
    pub fn parse(body: &[u8], expect: &MobileExpect<'_>, now: i64) -> Result<MobileAdmission> {
        const ALL: [&str; 16] = [
            "accessToken",
            "tokenType",
            "expiresIn",
            "expiresAt",
            "gatewayUrl",
            "appId",
            "scopes",
            "iceServers",
            "relayOnly",
            "turnCredentialExpiresAt",
            "holderKeyThumbprint",
            "relay",
            "subjectId",
            "role",
            "expectedPeerKeyThumbprint",
            "device",
        ];
        let doc = json::parse(body)?;
        if !members_are(&doc, &ALL, &ALL) {
            return Err(Error::Invalid("mobile admission: members"));
        }
        let bearer = Bearer::parse(doc.get_str("accessToken").ok_or(Error::Invalid("accessToken"))?)?;
        if bearer.len() < 16 || doc.get_str("tokenType") != Some("Bearer") || doc.get_str("role") != Some("mobile") {
            return Err(Error::Invalid("mobile admission: tokenType or role"));
        }
        let expires_in = doc.get_uint53("expiresIn").filter(|n| (1..=300).contains(n)).ok_or(Error::Invalid("expiresIn"))?;
        let expires_at = doc
            .get_uint53("expiresAt")
            .filter(|e| (*e as i64) > now && (*e as i64) <= now + 300)
            .ok_or(Error::OutsideWindow("admission expiresAt"))?;
        let gateway_url = doc.get_str("gatewayUrl").filter(|g| gateway_ok(g)).ok_or(Error::Invalid("gatewayUrl"))?.to_string();
        let app_id = doc.get_str("appId").filter(|a| *a == expect.app_id).ok_or(Error::Mismatch("admission: appId"))?.to_string();
        let subject_id = doc.get_str("subjectId").filter(|a| *a == expect.device_id).ok_or(Error::Mismatch("admission: subjectId"))?.to_string();
        let holder = doc
            .get_str("holderKeyThumbprint")
            .filter(|a| *a == expect.holder_thumbprint)
            .ok_or(Error::Mismatch("admission: holderKeyThumbprint"))?
            .to_string();
        let expected_peer = doc
            .get_str("expectedPeerKeyThumbprint")
            .filter(|t| ids::is_thumbprint(t) && *t != holder)
            .ok_or(Error::Invalid("expectedPeerKeyThumbprint"))?
            .to_string();
        let scopes = scopes_of(doc.get("scopes").ok_or(Error::Invalid("scopes"))?)?;
        if !scopes.iter().all(|s| ids::is_known_grant(s)) || !scopes.iter().any(|s| s == "state_read") {
            return Err(Error::Invalid("admission: scopes"));
        }
        let relay_only = doc.get("relayOnly").and_then(Json::as_bool).ok_or(Error::Invalid("relayOnly"))?;
        let (ice_servers, earliest) = parse_ice(doc.get("iceServers").ok_or(Error::Invalid("iceServers"))?, relay_only, now)?;
        let turn_credential_expires_at = match doc.get("turnCredentialExpiresAt") {
            Some(Json::Null) => None,
            Some(v) => Some(v.as_uint53().ok_or(Error::Invalid("turnCredentialExpiresAt"))?),
            None => return Err(Error::Invalid("turnCredentialExpiresAt")),
        };
        if turn_credential_expires_at != earliest {
            return Err(Error::Mismatch("turnCredentialExpiresAt is not the earliest TURN expiry"));
        }
        let device = doc.get("device").ok_or(Error::Invalid("device"))?;
        const DEVICE: [&str; 8] = ["id", "appId", "subjectId", "role", "displayName", "grants", "approvedAt", "lastSeenAt"];
        if !members_are(device, &DEVICE, &DEVICE)
            || device.get_str("role") != Some("mobile")
            || device.get_str("appId") != Some(&app_id)
            || device.get_str("subjectId") != Some(&subject_id)
        {
            return Err(Error::Invalid("admission: device"));
        }
        let name_len = device.get_str("displayName").map(str::len).unwrap_or(0);
        let iso =
            |k: &str| device.get_str(k).is_some_and(|t| t.len() >= 20 && t.ends_with('Z') && t.as_bytes()[4] == b'-' && t.as_bytes()[10] == b'T');
        if !(1..=120).contains(&name_len) || !iso("approvedAt") || !iso("lastSeenAt") {
            return Err(Error::Invalid("admission: device"));
        }
        let device_grants = scopes_of_device(device.get("grants").ok_or(Error::Invalid("device.grants"))?)?;
        if device_grants != scopes {
            return Err(Error::Mismatch("admission: device.grants are not the scopes"));
        }
        let relay = parse_endpoints(doc.get("relay").ok_or(Error::Invalid("relay"))?)?;
        Ok(MobileAdmission {
            bearer,
            expires_in,
            expires_at,
            gateway_url,
            app_id,
            subject_id,
            scopes,
            ice_servers,
            relay_only,
            turn_credential_expires_at,
            holder_thumbprint: holder,
            expected_peer_thumbprint: expected_peer,
            relay,
        })
    }
}

fn scopes_of_device(v: &Json) -> Result<Vec<String>> {
    let list = v.as_array().filter(|l| l.len() <= 16).ok_or(Error::Invalid("device.grants"))?;
    list.iter().map(|s| s.as_str().filter(|g| ids::is_grant(g)).map(str::to_string).ok_or(Error::Invalid("device.grants"))).collect()
}

/// The plugin's admission (`admission-plugin-response.schema.json`), read as the plugin's `AdmissionResponse::into_credentials` reads it: exactly the schema's members; `role`
/// `plugin`; `appId` and `subjectId` as sent; the five members the broker echoes equal to what it sent; `expiresIn` and `expiresAt - now` each above 10 and at most 300; ICE as
/// the decoders read it.
#[derive(Debug)]
pub struct PluginAdmission {
    /// The bearer for the compatibility routes.
    pub bearer: Bearer,
    /// `expiresIn`.
    pub expires_in: u64,
    /// `expiresAt`, relay time.
    pub expires_at: u64,
    /// `scopes`.
    pub scopes: Vec<String>,
    /// `iceServers`.
    pub ice_servers: Vec<IceServer>,
    /// `relayOnly`.
    pub relay_only: bool,
    /// `relay`.
    pub relay: RelayEndpoints,
}

impl PluginAdmission {
    /// Reads and checks the answer against what the broker sent.
    pub fn parse(body: &[u8], sent: &PluginRequest, now: i64) -> Result<PluginAdmission> {
        const ALL: [&str; 19] = [
            "accessToken",
            "tokenType",
            "expiresIn",
            "expiresAt",
            "gatewayUrl",
            "appId",
            "subjectId",
            "role",
            "scopes",
            "device",
            "iceServers",
            "relayOnly",
            "turnCredentialExpiresAt",
            "endpointPublicKey",
            "holderKeyThumbprint",
            "approvedPeerKeyThumbprints",
            "peerRosterRevision",
            "peerRosterHash",
            "relay",
        ];
        let doc = json::parse(body)?;
        if !members_are(&doc, &ALL, &ALL) {
            return Err(Error::Invalid("plugin admission: members"));
        }
        let bearer = Bearer::parse(doc.get_str("accessToken").ok_or(Error::Invalid("accessToken"))?)?;
        if doc.get_str("tokenType") != Some("Bearer") || doc.get_str("role") != Some("plugin") {
            return Err(Error::Invalid("plugin admission: tokenType or role"));
        }
        let expires_in = doc.get_uint53("expiresIn").filter(|n| (11..=300).contains(n)).ok_or(Error::Invalid("expiresIn"))?;
        let expires_at = doc
            .get_uint53("expiresAt")
            .filter(|e| (*e as i64) - now > 10 && (*e as i64) - now <= 300)
            .ok_or(Error::OutsideWindow("admission expiresAt"))?;
        if !doc.get_str("gatewayUrl").is_some_and(gateway_ok) {
            return Err(Error::Invalid("gatewayUrl"));
        }
        if doc.get_str("appId") != Some(&sent.app_id) || doc.get_str("subjectId") != Some(&sent.plugin_id) {
            return Err(Error::Mismatch("plugin admission: appId or subjectId"));
        }
        // The five members the broker echoes must be what it sent.
        let own = sent.endpoint.thumbprint();
        let endpoint = doc.get("endpointPublicKey").ok_or(Error::Invalid("endpointPublicKey"))?;
        let echoed_peers: Vec<&str> = doc
            .get("approvedPeerKeyThumbprints")
            .and_then(Json::as_array)
            .ok_or(Error::Invalid("approvedPeerKeyThumbprints"))?
            .iter()
            .filter_map(Json::as_str)
            .collect();
        let sent_peers: Vec<&str> = sent.approved_peers.iter().map(String::as_str).collect();
        if endpoint.get_str("publicKey") != Some(&sent.endpoint.to_b64u())
            || doc.get_str("holderKeyThumbprint") != Some(&own)
            || echoed_peers != sent_peers
            || doc.get_uint53("peerRosterRevision") != Some(sent.revision)
            || doc.get_str("peerRosterHash") != Some(&crate::roster::hash(sent.revision, &sent.approved_peers)?)
        {
            return Err(Error::Mismatch("plugin admission: the echoed roster is not what was sent"));
        }
        let scopes = scopes_of(doc.get("scopes").ok_or(Error::Invalid("scopes"))?)?;
        let relay_only = doc.get("relayOnly").and_then(Json::as_bool).ok_or(Error::Invalid("relayOnly"))?;
        let (ice_servers, earliest) = parse_ice(doc.get("iceServers").ok_or(Error::Invalid("iceServers"))?, relay_only, now)?;
        let turn_at = match doc.get("turnCredentialExpiresAt") {
            Some(Json::Null) => None,
            Some(v) => Some(v.as_uint53().ok_or(Error::Invalid("turnCredentialExpiresAt"))?),
            None => return Err(Error::Invalid("turnCredentialExpiresAt")),
        };
        if turn_at != earliest {
            return Err(Error::Mismatch("turnCredentialExpiresAt is not the earliest TURN expiry"));
        }
        let relay = parse_endpoints(doc.get("relay").ok_or(Error::Invalid("relay"))?)?;
        Ok(PluginAdmission { bearer, expires_in, expires_at, scopes, ice_servers, relay_only, relay })
    }
}
