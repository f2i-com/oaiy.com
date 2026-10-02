//! An in-process stub of the relay (feature `testing`): enough of README sections 5 to 8 for a client to be exercised without PHP, answering the way the real relay does where
//! the client depends on it.
//!
//! **What it implements**: `GET /v1/health`, `GET /v1/info` with the static signature and the identity proof, `POST /v1/enroll`, `GET /v1/poll` (the epoch and `reset`, the gap
//! rule, the bound of three, supersede, holds that really wait, refused holds on request, acknowledgement by `since`), `POST /v1/items` (the role matrix of README 4, idempotency per
//! sender, quotas), `POST /v1/tokens/rotate`, `POST /v1/admission` for both roles, the revocation of a device, and (in `pairing`) the rendezvous. **What it does not**: slots,
//! presence, reply boxes, tickets, the Aokie frames, calibration, rate-limit buckets other than the poll's. It is an implementation of the contract for a test to talk to, not a
//! reading of it to be tested against: the real relay is the reference (`tests/relay_php.rs` runs the same scenarios against it), and where the two differ the stub is wrong.
//!
//! **Faults**: [`StubRelay::set_down`] (every request is refused: an outage), [`StubRelay::fail_next`] (the next requests get a status, a dropped connection or a delay),
//! [`StubRelay::reset_epoch`] (a restore from a backup), [`StubRelay::revoke`] (a revoked device), and a log of every request ([`StubRelay::log`]) so that a test can assert that no
//! token was sent before the identity proof.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use oaiy_crypto::kdf::{hmac_sha256, sha256};
use oaiy_crypto::zeroize::{ct_eq, Secret};

use crate::admission::PREFIX as BEARER_PREFIX;
use crate::b64;
use crate::client::http::{HttpClient, HttpRequest, HttpResponse, TransportError};
use crate::enrol::Role;
use crate::ids;
use crate::json::{self, Json};
use crate::keys::{SignDomain, Signer, VerifyKey, X25519Public};
use crate::url::percent_encode;

use super::json_response;

/// The relay's settings (the parts of `config.json` that a client sees).
#[derive(Debug, Clone)]
pub struct StubConfig {
    /// `wait.default` of `info`.
    pub wait_default: u64,
    /// `wait.max`: the longest a poll is held (real seconds).
    pub wait_max: u64,
    /// `wait.pollGapMs`.
    pub poll_gap_ms: u64,
    /// `wait.fallbackS`.
    pub fallback_s: u64,
    /// `minClient`.
    pub min_client: u64,
    /// Call features: the admission routes answer, and `info` lists them.
    pub call_features: bool,
    /// Whether the streaming probe passed (`compat.sse-framed-poll`): without it an admission that offers only `relay` is `422`.
    pub stream_probe: bool,
    /// Whether `GET /v1/pair/{pid}` returns `grants` inside `receipt` once the owner has approved. **The shipped relay does not** (its receipt is `{issuedAt, signature}`), which is why
    /// a phone cannot verify the receipt it is given without the grants from somewhere else; a test turns this on to show what the one-line fix of the relay would do.
    pub receipt_includes_grants: bool,
    /// The relay's public URL, as written in an enrolment key.
    pub public_url: String,
    /// The relay's signing seed.
    pub relay_seed: [u8; 32],
    /// The relay id.
    pub relay_id: String,
}

impl Default for StubConfig {
    fn default() -> Self {
        StubConfig {
            wait_default: 1,
            wait_max: 1,
            poll_gap_ms: 250,
            fallback_s: 5,
            min_client: 1,
            call_features: true,
            stream_probe: false,
            receipt_includes_grants: false,
            public_url: "https://relay.stub.test".into(),
            // The seed and id of the protocol package's vectors (`keys.ed25519Seeds.relay`, `keys.ids.relay`): the thumbprint is `b7dKD2-DlMApkGljz-RJwDdNcyxwyFBpSmhz2cRBnaw`.
            relay_seed: [0x33; 32],
            relay_id: "rly-0NHS09TV1tfY2drb3N3e3w".into(),
        }
    }
}

/// A fault applied to the next requests.
#[derive(Debug, Clone)]
pub enum Fault {
    /// Answer with this status, headers and body.
    Respond(u16, Vec<(String, String)>, String),
    /// Drop the connection: no response.
    Drop,
    /// Wait this long (real time) and then answer normally.
    Delay(Duration),
}

/// What a test sees of a request the stub received.
#[derive(Debug, Clone)]
pub struct LoggedRequest {
    /// The method.
    pub method: String,
    /// The path and query.
    pub target: String,
    /// The `Authorization` header, as sent.
    pub authorization: Option<String>,
    /// The body.
    pub body: String,
    /// The status the stub answered, or 0 for no response.
    pub status: u16,
}

#[derive(Clone)]
pub(crate) struct Device {
    pub id: String,
    pub role: String,
    pub name: String,
    pub thumbprint: Option<String>,
    pub owner_desktop: Option<String>,
    pub app_id: Option<String>,
    pub grants: Vec<String>,
    /// The thumbprint of the desktop endpoint key a phone paired with (its pin; the admission's expectedPeerKeyThumbprint).
    pub peer_thumbprint: Option<String>,
    pub revoked: bool,
}

struct TokenRec {
    id: String,
    secret_sha: [u8; 32],
    device: String,
    not_after: Option<i64>,
}

#[derive(Clone)]
struct StoredItem {
    seq: u64,
    id: String,
    lane: String,
    from: String,
    at: i64,
    exp: i64,
    hdr: Json,
    body: String,
    body_sha: [u8; 32],
    acked: bool,
}

#[derive(Default)]
struct Mailbox {
    items: Vec<StoredItem>,
    counter: u64,
    last_poll_end: Option<Duration>,
    last_since: u64,
}

#[derive(Default)]
struct Hold {
    generation: u64,
    waiting: u32,
}

struct EnrolKey {
    kid: String,
    public: VerifyKey,
    role: Role,
    exp: i64,
    used: bool,
}

pub(crate) struct State {
    pub epoch: String,
    pub devices: Vec<Device>,
    tokens: Vec<TokenRec>,
    mailboxes: HashMap<String, Mailbox>,
    holds: HashMap<String, Hold>,
    enrol_keys: Vec<EnrolKey>,
    pub down: bool,
    faults: VecDeque<(Option<String>, Fault)>,
    refuse_holds: u32,
    pub log: Vec<LoggedRequest>,
    pub(crate) pairings: HashMap<String, super::stub_pairing::Pairing>,
    id_counter: u64,
    seed: u64,
    /// Pairing reads that are being held open (`GET /v1/pair/{pid}?wait=`), for `GET /v1/admin/status`.
    pub(crate) pair_waiting: u32,
}

pub(crate) struct Inner {
    pub cfg: Mutex<StubConfig>,
    pub state: Mutex<State>,
    pub cv: Condvar,
    now: Box<dyn Fn() -> i64 + Send + Sync>,
    mono: Box<dyn Fn() -> Duration + Send + Sync>,
    relay_key: Signer,
}

/// The stub relay. Cloning shares it.
#[derive(Clone)]
pub struct StubRelay(pub(crate) Arc<Inner>);

/// A response being built.
pub(crate) struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
    /// The value of X-OAIY-Time, when the handler needs it to be a value it has already used (the identity proof covers it).
    pub time: Option<i64>,
}

pub(crate) fn ok(body: Json) -> Resp {
    Resp { status: 200, headers: Vec::new(), body: body.to_compact(), time: None }
}

pub(crate) fn err(status: u16, code: &str, message: &str) -> Resp {
    Resp {
        status,
        headers: Vec::new(),
        body: Json::obj([("error", Json::obj([("code", Json::str(code)), ("message", Json::str(message))]))]).to_compact(),
        time: None,
    }
}

pub(crate) fn err_retry(status: u16, code: &str, message: &str, retry: u64, rule: Option<&str>) -> Resp {
    let mut e = vec![("code", Json::str(code)), ("message", Json::str(message)), ("retryAfter", Json::int(retry))];
    if let Some(r) = rule {
        e.push(("rule", Json::str(r)));
    }
    Resp {
        status,
        headers: vec![("retry-after".into(), retry.to_string())],
        body: Json::obj([("error", Json::Obj(e.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))]).to_compact(),
        time: None,
    }
}

/// The authenticated caller.
pub(crate) struct Principal {
    pub device: String,
    pub role: String,
}

pub(crate) fn query(target: &str) -> (String, HashMap<String, String>) {
    let (path, q) = target.split_once('?').unwrap_or((target, ""));
    let mut map = HashMap::new();
    for pair in q.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        map.insert(k.to_string(), v.to_string());
    }
    (path.to_string(), map)
}

fn is_query_uint(v: &str) -> bool {
    v == "0" || (!v.is_empty() && v.len() <= 16 && !v.starts_with('0') && v.bytes().all(|b| b.is_ascii_digit()))
}

impl StubRelay {
    /// A stub with `config`, whose clock is the machine's.
    pub fn new(config: StubConfig) -> StubRelay {
        StubRelay::with_clock(config, || std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64))
    }

    /// A stub whose clock (the relay's time, in `X-OAIY-Time` and in every `time`) is `now`, and whose intervals (the poll gap) are measured on the machine's monotonic clock.
    pub fn with_clock(config: StubConfig, now: impl Fn() -> i64 + Send + Sync + 'static) -> StubRelay {
        let start = Instant::now();
        StubRelay::with_clocks(config, now, move || start.elapsed())
    }

    /// A stub with both clocks given: the relay's time, and a monotonic time for the gap rule (a test that drives the client with a fake clock gives the fake clock's, so that a
    /// pause the client takes without waiting is a pause the relay sees).
    pub fn with_clocks(
        config: StubConfig,
        now: impl Fn() -> i64 + Send + Sync + 'static,
        mono: impl Fn() -> Duration + Send + Sync + 'static,
    ) -> StubRelay {
        let relay_key = Signer::from_seed(&Secret::new(config.relay_seed));
        let state = State {
            epoch: b64::encode(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]),
            devices: Vec::new(),
            tokens: Vec::new(),
            mailboxes: HashMap::new(),
            holds: HashMap::new(),
            enrol_keys: Vec::new(),
            down: false,
            faults: VecDeque::new(),
            refuse_holds: 0,
            log: Vec::new(),
            pairings: HashMap::new(),
            id_counter: 1,
            seed: 0x0a1b_2c3d_4e5f_6071,
            pair_waiting: 0,
        };
        StubRelay(Arc::new(Inner {
            cfg: Mutex::new(config),
            state: Mutex::new(state),
            cv: Condvar::new(),
            now: Box::new(now),
            mono: Box::new(mono),
            relay_key,
        }))
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, State> {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(crate) fn cfg(&self) -> StubConfig {
        self.0.cfg.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The relay's time.
    pub fn now(&self) -> i64 {
        (self.0.now)()
    }

    /// The thumbprint of the relay's key: what a client pins.
    pub fn relay_thumbprint(&self) -> String {
        self.0.relay_key.thumbprint()
    }

    /// The relay's id.
    pub fn relay_id(&self) -> String {
        self.cfg().relay_id
    }

    /// The public URL.
    pub fn public_url(&self) -> String {
        self.cfg().public_url
    }

    /// Changes the public URL (a server on a loopback port knows its address only once it is listening).
    pub fn set_public_url(&self, url: &str) {
        self.0.cfg.lock().unwrap_or_else(|e| e.into_inner()).public_url = url.to_string();
    }

    /// Changes a setting while the stub runs.
    pub fn configure(&self, f: impl FnOnce(&mut StubConfig)) {
        f(&mut self.0.cfg.lock().unwrap_or_else(|e| e.into_inner()));
    }

    /// While true every request is refused: an outage.
    pub fn set_down(&self, down: bool) {
        self.lock().down = down;
    }

    /// The next `n` requests get `fault`.
    pub fn fail_next(&self, n: usize, fault: Fault) {
        let mut st = self.lock();
        for _ in 0..n {
            st.faults.push_back((None, fault.clone()));
        }
    }

    /// The next `n` requests whose path begins with `prefix` get `fault` (the others are answered normally in the meantime): a fault for the poll and not for the identity
    /// proof that comes before it.
    pub fn fail_next_on(&self, prefix: &str, n: usize, fault: Fault) {
        let mut st = self.lock();
        for _ in 0..n {
            st.faults.push_back((Some(prefix.to_string()), fault.clone()));
        }
    }

    /// The next `n` polls that wait are answered at once as a short poll with `hold.refused` (a relay whose pool is nearly full).
    pub fn refuse_holds(&self, n: u32) {
        self.lock().refuse_holds = n;
    }

    /// A restore from a backup: the epoch changes, and every client's next poll answers `reset`.
    pub fn reset_epoch(&self) {
        let mut st = self.lock();
        let e = Self::random_bytes(&mut st, 8);
        st.epoch = b64::encode(&e);
        self.0.cv.notify_all();
    }

    /// The current epoch.
    pub fn epoch(&self) -> String {
        self.lock().epoch.clone()
    }

    /// Revokes a device: its tokens stop working (`401 revoked`), its inbox is purged, and a poll it holds ends with `401 revoked`.
    pub fn revoke(&self, device: &str) {
        let mut st = self.lock();
        for d in st.devices.iter_mut().filter(|d| d.id == device) {
            d.revoked = true;
        }
        st.mailboxes.remove(device);
        self.0.cv.notify_all();
    }

    /// Every request received, in order, with the status it was answered.
    pub fn log(&self) -> Vec<LoggedRequest> {
        self.lock().log.clone()
    }

    /// Forgets the request log.
    pub fn clear_log(&self) {
        self.lock().log.clear();
    }

    pub(crate) fn random_bytes(st: &mut State, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            st.seed = st.seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = st.seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            out.extend_from_slice(&(z ^ (z >> 31)).to_le_bytes());
        }
        out.truncate(n);
        out
    }

    fn new_id(st: &mut State, prefix: &str) -> String {
        st.id_counter += 1;
        let mut raw = Self::random_bytes(st, 16);
        raw[..8].copy_from_slice(&st.id_counter.to_be_bytes());
        format!("{prefix}-{}", b64::encode(&raw))
    }

    // ------------------------------------------------------------------------------------------------------------ setup

    /// Mints an enrolment key (`oaiy://enroll?...`) for a desktop or a provider, valid `ttl` seconds.
    pub fn mint_enrolment_key(&self, role: Role, ttl: i64) -> String {
        let cfg = self.cfg();
        let now = self.now();
        let mut st = self.lock();
        let mut secret = [0u8; 16];
        secret.copy_from_slice(&Self::random_bytes(&mut st, 16));
        let key = crate::enrol::EnrolmentKey::to_uri(
            &crate::url::RelayUrl::parse(&cfg.public_url).expect("the stub's public url"),
            &self.relay_thumbprint(),
            &secret,
            role,
            (now + ttl) as u64,
        )
        .expect("enrolment key");
        let parsed = crate::enrol::EnrolmentKey::parse(&key).expect("the key just made");
        let public = parsed.signer().expect("signer").verify_key();
        st.enrol_keys.push(EnrolKey { kid: parsed.kid.clone(), public, role, exp: now + ttl, used: false });
        key
    }

    /// Creates a device directly (not through enrolment or pairing) and returns its id and token: a phone that a desktop approved, or a provider.
    #[allow(clippy::too_many_arguments)]
    pub fn create_device(
        &self,
        role: &str,
        name: &str,
        owner_desktop: Option<&str>,
        app_id: Option<&str>,
        grants: &[&str],
        ed25519: Option<&VerifyKey>,
    ) -> (String, String) {
        let mut st = self.lock();
        self.create_device_locked(&mut st, role, name, owner_desktop, app_id, grants, ed25519)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_device_locked(
        &self,
        st: &mut State,
        role: &str,
        name: &str,
        owner_desktop: Option<&str>,
        app_id: Option<&str>,
        grants: &[&str],
        ed25519: Option<&VerifyKey>,
    ) -> (String, String) {
        let id = Self::new_id(st, if role == "provider" { "prov" } else { "dev" });
        let token_id = b64::encode(&Self::random_bytes(st, 8));
        let secret = Self::random_bytes(st, 32);
        st.devices.push(Device {
            id: id.clone(),
            role: role.to_string(),
            name: name.to_string(),
            thumbprint: ed25519.map(VerifyKey::thumbprint),
            owner_desktop: owner_desktop.map(str::to_string),
            app_id: app_id.map(str::to_string),
            grants: grants.iter().map(|g| g.to_string()).collect(),
            peer_thumbprint: None,
            revoked: false,
        });
        st.tokens.push(TokenRec { id: token_id.clone(), secret_sha: sha256(&secret), device: id.clone(), not_after: None });
        (id, format!("oaiyrt1.{token_id}.{}", b64::encode(&secret)))
    }

    /// Puts an item in a device's inbox as the relay itself (`from` is `relay`): a `ctl` notice, a `pair` item.
    pub fn post_as_relay(&self, to_device: &str, lane: &str, id: &str, hdr: Json, body: &str, ttl: i64) -> u64 {
        let now = self.now();
        let mut st = self.lock();
        let seq = Self::enqueue(&mut st, to_device, lane, id, "relay", hdr, body, now, now + ttl).0;
        self.0.cv.notify_all();
        seq
    }

    /// The number of live (not acknowledged) items in a device's inbox.
    pub fn live_items(&self, device: &str) -> usize {
        self.lock().mailboxes.get(device).map_or(0, |m| m.items.iter().filter(|i| !i.acked).count())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn enqueue(st: &mut State, to: &str, lane: &str, id: &str, from: &str, hdr: Json, body: &str, at: i64, exp: i64) -> (u64, bool, bool) {
        let m = st.mailboxes.entry(to.to_string()).or_default();
        let sha = sha256(body.as_bytes());
        if let Some(existing) = m.items.iter().find(|i| i.lane == lane && i.from == from && i.id == id) {
            return (existing.seq, true, existing.body_sha != sha);
        }
        m.counter += 1;
        let seq = m.counter;
        m.items.push(StoredItem {
            seq,
            id: id.to_string(),
            lane: lane.to_string(),
            from: from.to_string(),
            at,
            exp,
            hdr,
            body: body.to_string(),
            body_sha: sha,
            acked: false,
        });
        (seq, false, false)
    }

    // ------------------------------------------------------------------------------------------------------------ requests

    fn stamp(&self, mut r: Resp) -> HttpResponse {
        let mut extra: Vec<(&str, &str)> = Vec::new();
        for (k, v) in &r.headers {
            extra.push((k.as_str(), v.as_str()));
        }
        let body = std::mem::take(&mut r.body);
        json_response(r.status, &extra, &body, r.time.unwrap_or_else(|| self.now()))
    }

    pub(crate) fn principal(&self, st: &State, req: &HttpRequest) -> Result<Principal, Resp> {
        let unauthorized = || err(401, "unauthorized", "The credential is not accepted.");
        let header = req.header("authorization").ok_or_else(unauthorized)?;
        let token = header.strip_prefix("Bearer ").ok_or_else(unauthorized)?;
        let parsed = crate::ids::Token::parse(token).map_err(|_| unauthorized())?;
        let rec = st.tokens.iter().find(|t| t.id == parsed.id()).ok_or_else(unauthorized)?;
        let secret = b64::decode(token.rsplit('.').next().unwrap_or("")).map_err(|_| unauthorized())?;
        if !ct_eq(&sha256(&secret), &rec.secret_sha) {
            return Err(unauthorized());
        }
        if rec.not_after.is_some_and(|n| n < self.now()) {
            return Err(unauthorized());
        }
        let device = st.devices.iter().find(|d| d.id == rec.device).ok_or_else(unauthorized)?;
        if device.revoked {
            return Err(err(401, "revoked", "This device was revoked."));
        }
        Ok(Principal { device: device.id.clone(), role: device.role.clone() })
    }

    /// Handles one request. A hold really waits (in real time, at most `wait.max` seconds), so call this from a thread of its own for a poll.
    pub fn handle(&self, req: &HttpRequest) -> Result<HttpResponse, TransportError> {
        let path_of_request = req.url.splitn(4, '/').nth(3).map(|p| format!("/{p}")).unwrap_or_else(|| "/".into());
        let (down, fault) = {
            let mut st = self.lock();
            let at = st.faults.iter().position(|(prefix, _)| prefix.as_ref().is_none_or(|p| path_of_request.starts_with(p.as_str())));
            let fault = at.and_then(|i| st.faults.remove(i)).map(|(_, f)| f);
            (st.down, fault)
        };
        let record = |status: u16| {
            let mut st = self.lock();
            st.log.push(LoggedRequest {
                method: req.method.as_str().to_string(),
                target: req.url.splitn(4, '/').nth(3).map(|p| format!("/{p}")).unwrap_or_default(),
                authorization: req.header("authorization").map(str::to_string),
                body: req.body.as_ref().map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default(),
                status,
            });
        };
        if down {
            record(0);
            return Err(TransportError::Refused);
        }
        match fault {
            Some(Fault::Drop) => {
                record(0);
                return Err(TransportError::Reset);
            }
            Some(Fault::Respond(status, headers, body)) => {
                record(status);
                let extra: Vec<(&str, &str)> = headers.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                return Ok(json_response(status, &extra, &body, self.now()));
            }
            Some(Fault::Delay(d)) => std::thread::sleep(d),
            None => {}
        }
        let target = req.url.splitn(4, '/').nth(3).map(|p| format!("/{p}")).unwrap_or_else(|| "/".into());
        let (path, q) = query(&target);
        let method = req.method.as_str();
        // A client below minClient is refused everywhere but at info, which it must be able to read to learn it (the real relay hardcodes the floor at 1 and never refuses a level it
        // is given, so this is the stub's way to exercise the client's 426 rule).
        let level: u64 = req.header("x-oaiy-level").and_then(|l| l.parse().ok()).unwrap_or(1);
        if path != "/v1/info" && level < self.cfg().min_client {
            let http = self.stamp(err(426, "upgrade_required", "This client is too old for this relay."));
            record(426);
            return Ok(http);
        }
        let response = self.route(req, method, &path, &q);
        if response.status == 0 {
            record(0);
            return Err(TransportError::Cancelled);
        }
        let http = self.stamp(response);
        record(http.status);
        Ok(http)
    }

    fn route(&self, req: &HttpRequest, method: &str, path: &str, q: &HashMap<String, String>) -> Resp {
        match (method, path) {
            ("GET", "/v1/health") => ok(Json::obj([
                ("ok", Json::Bool(true)),
                ("time", Json::int(self.now())),
                ("authHeaderSeen", Json::Bool(req.header("authorization").is_some())),
            ])),
            ("GET", "/v1/info") => self.info(req),
            ("POST", "/v1/enroll") => self.enroll(req),
            ("GET", "/v1/poll") => self.poll(req, q),
            ("POST", "/v1/items") => self.post_items(req),
            ("POST", "/v1/tokens/rotate") => self.rotate(req),
            ("GET", "/v1/admin/status") => self.admin_status(req),
            ("POST", "/v1/admission") | ("POST", "/v1/aokie-companion/admission") => self.admission(req),
            (m, p) if p.starts_with("/v1/pair") => self.pairing_route(req, m, p, q),
            ("POST", p) if p.starts_with("/v1/devices/") && p.ends_with("/revoke") => self.revoke_route(req, p),
            _ => err(404, "not_found", "No such route."),
        }
    }

    // ------------------------------------------------------------------------------------------------------------ info

    /// The `info` document as the stub serves it (a static text, signed).
    pub fn info_text(&self) -> String {
        let cfg = self.cfg();
        let mut features = vec!["poll", "items", "presence", "pairing.v3", "methods.post-forms"];
        if cfg.call_features {
            features.extend(["call", "admission.aokie-adm-v2", "compat.aokie-companion-relay"]);
            if cfg.stream_probe {
                features.push("compat.sse-framed-poll");
            }
        }
        let lanes =
            [("cmd", 32768, 60, 300), ("res", 98304, 300, 3600), ("ctl", 4096, 3600, 86400), ("sync", 65536, 21600, 86400), ("ring", 4096, 30, 300)];
        let lane_json = Json::Obj(
            lanes
                .iter()
                .map(|(n, body, d, m)| {
                    (
                        n.to_string(),
                        Json::obj([
                            ("body", Json::int(*body)),
                            ("ttl", Json::obj([("default", Json::int(*d)), ("min", Json::int(1)), ("max", Json::int(*m))])),
                        ]),
                    )
                })
                .collect(),
        );
        let key = self.0.relay_key.verify_key();
        Json::obj([
            ("protocol", Json::str("oaiy-relay/1")),
            ("minClient", Json::int(cfg.min_client)),
            ("relayId", Json::str(cfg.relay_id.clone())),
            (
                "relayKey",
                Json::obj([
                    ("algorithm", Json::str("ed25519")),
                    ("publicKey", Json::str(key.to_b64u())),
                    ("thumbprint", Json::str(key.thumbprint())),
                ]),
            ),
            ("software", Json::obj([("name", Json::str("oaiy-relay-stub")), ("version", Json::str("0.1.0"))])),
            ("features", Json::Arr(features.into_iter().map(Json::str).collect())),
            (
                "wait",
                Json::obj([
                    ("default", Json::int(cfg.wait_default)),
                    ("max", Json::int(cfg.wait_max)),
                    ("pollGapMs", Json::int(cfg.poll_gap_ms)),
                    ("fallbackS", Json::int(cfg.fallback_s)),
                ]),
            ),
            ("presenceWindow", Json::int(60)),
            (
                "limits",
                Json::obj([
                    ("batchItems", Json::int(64)),
                    ("batchBytes", Json::int(1_048_576)),
                    ("mailboxItems", Json::int(512)),
                    ("mailboxBytes", Json::int(8_388_608)),
                    ("hdrBytes", Json::int(512)),
                    ("held", Json::obj([("soft", Json::int(3)), ("hard", Json::int(4)), ("measured", Json::Bool(false))])),
                    ("lanes", lane_json),
                ]),
            ),
        ])
        .to_compact()
    }

    fn info(&self, req: &HttpRequest) -> Resp {
        let body = self.info_text();
        let key = &self.0.relay_key;
        let time = self.now();
        let mut headers = vec![("x-oaiy-sig".to_string(), key.sign_b64u(SignDomain::Info, &[body.as_bytes()]))];
        if let Some(nonce_text) = req.header("x-oaiy-nonce") {
            let Ok(nonce) = b64::decode(nonce_text) else { return err(400, "invalid_request", "The nonce is not base64url.") };
            if !(16..=32).contains(&nonce.len()) {
                return err(400, "invalid_request", "The nonce is 16 to 32 bytes.");
            }
            let digest = sha256(body.as_bytes());
            let proof = key.sign_b64u(SignDomain::InfoProof, &[&nonce, &digest, time.to_string().as_bytes()]);
            headers.push(("x-oaiy-proof".to_string(), proof));
        }
        Resp { status: 200, headers, body, time: Some(time) }
    }

    // ------------------------------------------------------------------------------------------------------------ enrol

    fn enroll(&self, req: &HttpRequest) -> Resp {
        let bad = || err(401, "unauthorized", "The enrolment key is not accepted.");
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let (Some(kid), Some(role)) = (doc.get_str("kid"), doc.get_str("role")) else {
            return err(400, "invalid_request", "kid and role are required.");
        };
        let proof = req.header("x-oaiy-proof").unwrap_or("");
        let now = self.now();
        let mut st = self.lock();
        let Some(idx) = st.enrol_keys.iter().position(|k| k.kid == kid) else { return bad() };
        let key = &st.enrol_keys[idx];
        let role_ok = matches!((role, key.role), ("desktop", Role::Desktop) | ("provider", Role::Provider));
        if key.used || key.exp < now || !role_ok || key.public.verify_b64u(SignDomain::Enroll, &[body], proof).is_err() {
            return bad();
        }
        let Some(keys) = doc.get("keys") else { return err(400, "invalid_request", "keys are required.") };
        let (Some(ed), Some(x)) = (keys.get_str("ed25519"), keys.get_str("x25519")) else { return err(400, "invalid_request", "keys are required.") };
        let (Ok(ed), Ok(_x)) = (VerifyKey::from_b64u_registrable(ed), X25519Public::from_b64u(x)) else {
            return err(422, "unprocessable", "A key of small order.");
        };
        st.enrol_keys[idx].used = true;
        let name = ids::clean_name(doc.get_str("name").unwrap_or(""), 60);
        let (id, token) = self.create_device_locked(&mut st, role, &name, None, None, &[], Some(&ed));
        drop(st);
        let cfg = self.cfg();
        Resp {
            status: 201,
            headers: Vec::new(),
            body: Json::obj([
                ("deviceId", Json::str(id)),
                ("token", Json::str(token)),
                ("relayId", Json::str(cfg.relay_id)),
                ("time", Json::int(now)),
            ])
            .to_compact(),
            time: None,
        }
    }

    // ------------------------------------------------------------------------------------------------------------ poll

    fn poll(&self, req: &HttpRequest, q: &HashMap<String, String>) -> Resp {
        let cfg = self.cfg();
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let invalid = |what: &str| err(400, "invalid_request", what);
        let since = match q.get("since") {
            None => 0,
            Some(v) if is_query_uint(v) => v.parse().unwrap_or(0),
            Some(_) => return invalid("since"),
        };
        let wait = match q.get("wait") {
            None => 0,
            Some(v) if is_query_uint(v) => v.parse::<u64>().unwrap_or(0).min(cfg.wait_max),
            Some(_) => return invalid("wait"),
        };
        let limit = match q.get("limit") {
            None => 32,
            Some(v) if is_query_uint(v) && (1..=64).contains(&v.parse::<usize>().unwrap_or(0)) => v.parse().unwrap_or(32),
            Some(_) => return invalid("limit"),
        };
        let epoch_asked = q.get("epoch").cloned();
        let device = who.device.clone();
        let now = self.now();
        let epoch = st.epoch.clone();
        let highest = st.mailboxes.entry(device.clone()).or_default().counter;
        // A different epoch, or a cursor above anything ever issued: the mailbox is not the one the client knew.
        if epoch_asked.as_deref().is_some_and(|e| e != epoch) || since > highest {
            st.mailboxes.entry(device.clone()).or_default().last_poll_end = None;
            let mut m = vec![
                ("v", Json::int(1)),
                ("epoch", Json::str(epoch)),
                ("cursor", Json::int(highest)),
                ("items", Json::Arr(vec![])),
                ("more", Json::Bool(false)),
                ("time", Json::int(now)),
                ("reset", Json::Bool(true)),
            ];
            if wait > 0 {
                m.push(("hold", Json::obj([("granted", Json::Bool(true))])));
            }
            return ok(Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
        }
        // The gap rule: a poll that makes no progress sooner than `pollGapMs` after the last one ended.
        {
            let mb = st.mailboxes.entry(device.clone()).or_default();
            if let Some(end) = mb.last_poll_end {
                if since <= mb.last_since && (self.0.mono)().saturating_sub(end) < Duration::from_millis(cfg.poll_gap_ms) {
                    return err_retry(429, "rate_limited", "Polling too fast.", 1, Some("gap"));
                }
            }
        }
        let hold_refused = wait > 0 && st.refuse_holds > 0;
        if hold_refused {
            st.refuse_holds -= 1;
        }
        let hold_wanted = wait > 0 && !hold_refused;
        if hold_wanted && st.holds.entry(device.clone()).or_default().waiting >= 3 {
            return err_retry(429, "rate_limited", "Too many polls.", 1, Some("in_flight"));
        }
        // Acknowledge what `since` covers.
        for it in st.mailboxes.entry(device.clone()).or_default().items.iter_mut().filter(|i| i.seq <= since) {
            if !it.acked {
                it.acked = true;
                it.body.clear();
            }
        }
        let generation = {
            let h = st.holds.entry(device.clone()).or_default();
            // Only a poll that waits takes the place of an older one.
            if hold_wanted {
                h.generation += 1;
                h.waiting += 1;
            }
            h.generation
        };
        let deadline = Instant::now() + Duration::from_secs(if hold_wanted { wait } else { 0 });
        let mut superseded = false;
        let mut revoked = false;
        let mut cancelled = false;
        let mut picked: Vec<StoredItem> = Vec::new();
        let mut more = false;
        loop {
            if st.devices.iter().any(|d| d.id == device && d.revoked) {
                revoked = true;
                break;
            }
            if req.cancel.is_cancelled() {
                cancelled = true;
                break;
            }
            if st.holds.get(&device).is_none_or(|h| h.generation != generation) {
                superseded = true;
                break;
            }
            let now = self.now();
            let live: Vec<StoredItem> =
                st.mailboxes.entry(device.clone()).or_default().items.iter().filter(|i| !i.acked && i.seq > since && i.exp > now).cloned().collect();
            if !live.is_empty() || Instant::now() >= deadline {
                more = live.len() > limit;
                picked = live.into_iter().take(limit).collect();
                break;
            }
            let (guard, _) = self.0.cv.wait_timeout(st, Duration::from_millis(40)).unwrap_or_else(|e| e.into_inner());
            st = guard;
        }
        if hold_wanted {
            if let Some(h) = st.holds.get_mut(&device) {
                h.waiting = h.waiting.saturating_sub(1);
            }
        }
        if cancelled {
            return Resp { status: 0, headers: Vec::new(), body: String::new(), time: None };
        }
        if revoked {
            return err(401, "revoked", "This device was revoked.");
        }
        {
            let mb = st.mailboxes.entry(device.clone()).or_default();
            if picked.is_empty() {
                mb.last_poll_end = if superseded { None } else { Some((self.0.mono)()) };
                mb.last_since = since;
            } else {
                mb.last_poll_end = None;
            }
        }
        let items: Vec<Json> = picked
            .iter()
            .map(|i| {
                Json::obj([
                    ("seq", Json::int(i.seq)),
                    ("id", Json::str(i.id.clone())),
                    ("lane", Json::str(i.lane.clone())),
                    ("from", Json::str(i.from.clone())),
                    ("at", Json::int(i.at)),
                    ("exp", Json::int(i.exp)),
                    ("hdr", i.hdr.clone()),
                    ("body", Json::str(i.body.clone())),
                ])
            })
            .collect();
        let cursor = picked.last().map_or(since, |i| i.seq);
        let mut m = vec![
            ("v", Json::int(1)),
            ("epoch", Json::str(epoch)),
            ("cursor", Json::int(cursor)),
            ("items", Json::Arr(items)),
            ("more", Json::Bool(more)),
            ("time", Json::int(self.now())),
        ];
        let mut headers = Vec::new();
        if wait > 0 {
            let hold = if hold_refused {
                headers.push(("x-oaiy-hold".to_string(), "refused".to_string()));
                Json::obj([("refused", Json::Bool(true)), ("retryAfter", Json::int(cfg.fallback_s.min(2)))])
            } else if superseded {
                headers.push(("x-oaiy-hold".to_string(), "granted".to_string()));
                Json::obj([("granted", Json::Bool(true)), ("superseded", Json::Bool(true))])
            } else {
                headers.push(("x-oaiy-hold".to_string(), "granted".to_string()));
                Json::obj([("granted", Json::Bool(true))])
            };
            m.push(("hold", hold));
        }
        Resp { status: 200, headers, body: Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()).to_compact(), time: None }
    }

    // ------------------------------------------------------------------------------------------------------------ items

    fn post_items(&self, req: &HttpRequest) -> Resp {
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let Some(items) = doc.get("items").and_then(Json::as_array).filter(|i| (1..=64).contains(&i.len())) else {
            return err(400, "invalid_request", "1 to 64 items.");
        };
        let now = self.now();
        let mut results = Vec::new();
        for it in items {
            let id = it.get_str("id").unwrap_or("").to_string();
            let rejected = |code: &str, message: &str| {
                Json::obj([
                    ("id", Json::str(id.clone())),
                    ("status", Json::str("rejected")),
                    ("error", Json::obj([("code", Json::str(code)), ("message", Json::str(message))])),
                ])
            };
            let (Some(to), Some(lane), Some(text)) = (it.get_str("to"), it.get_str("lane"), it.get_str("body")) else {
                results.push(rejected("invalid_item", "to, lane and body are required."));
                continue;
            };
            if !ids::is_item_id(&id) || !["cmd", "res", "ctl", "sync", "ring", "ai.out"].contains(&lane) {
                results.push(rejected("invalid_item", "An id or a lane the relay does not accept."));
                continue;
            }
            let ttl = match it.get("ttl") {
                None => 60,
                Some(t) => match t.as_uint53() {
                    Some(n) if n >= 1 => n as i64,
                    _ => {
                        results.push(rejected("invalid_item", "ttl"));
                        continue;
                    }
                },
            };
            let Some(target) = to.strip_prefix("dev:").filter(|d| ids::is_principal_id(d)) else {
                results.push(rejected("not_found", "No such recipient."));
                continue;
            };
            let Some(recipient) = st.devices.iter().find(|d| d.id == target && !d.revoked).cloned() else {
                results.push(rejected("not_found", "No such recipient."));
                continue;
            };
            let allowed = match (who.role.as_str(), lane, recipient.role.as_str()) {
                ("provider", "cmd", "desktop") => true,
                ("desktop", "res", _) => true,
                ("desktop", "ring" | "sync", "phone") => recipient.owner_desktop.as_deref() == Some(who.device.as_str()),
                ("desktop", "ctl", _) => true,
                _ => false,
            };
            if !allowed {
                results.push(rejected("forbidden", "Not allowed."));
                continue;
            }
            if st.mailboxes.get(target).map_or(0, |m| m.items.iter().filter(|i| !i.acked).count()) >= 512 {
                results.push(rejected("quota_exceeded", "The mailbox is full."));
                continue;
            }
            let hdr = it.get("hdr").cloned().unwrap_or_else(|| Json::Obj(vec![]));
            let (seq, duplicate, conflict) = Self::enqueue(&mut st, target, lane, &id, &who.device, hdr, text, now, now + ttl);
            results.push(if conflict {
                rejected("conflict", "Another body under the same id.")
            } else {
                Json::obj([
                    ("id", Json::str(id.clone())),
                    ("status", Json::str(if duplicate { "duplicate" } else { "queued" })),
                    ("seq", Json::int(seq)),
                ])
            });
        }
        self.0.cv.notify_all();
        ok(Json::obj([("v", Json::int(1)), ("results", Json::Arr(results)), ("time", Json::int(now))]))
    }

    // ------------------------------------------------------------------------------------------------------------ tokens, revoke

    fn rotate(&self, req: &HttpRequest) -> Resp {
        let mut st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        let now = self.now();
        if st.tokens.iter().any(|t| t.device == who.device && t.not_after.is_some_and(|n| n > now)) {
            return err(409, "conflict", "A rotation is already in its grace period.");
        }
        let current = req.header("authorization").and_then(|h| h.strip_prefix("Bearer ")).and_then(|t| t.split('.').nth(1)).unwrap_or("").to_string();
        for t in st.tokens.iter_mut().filter(|t| t.device == who.device && t.id == current) {
            t.not_after = Some(now + 600);
        }
        let token_id = b64::encode(&Self::random_bytes(&mut st, 8));
        let secret = Self::random_bytes(&mut st, 32);
        st.tokens.push(TokenRec { id: token_id.clone(), secret_sha: sha256(&secret), device: who.device, not_after: None });
        ok(Json::obj([
            ("token", Json::str(format!("oaiyrt1.{token_id}.{}", b64::encode(&secret)))),
            ("graceUntil", Json::int(now + 600)),
            ("time", Json::int(now)),
        ]))
    }

    /// `GET /v1/admin/status` (the token of a desktop): the document of `admin-status.schema.json` with what a test of a client looks at filled in truly: **`holds`**, how many requests the
    /// relay is holding open at this moment (`live`) and of what kind (`byKind`: `poll`, `pair`), which is how a test sees that a client keeps one request open and not two (MOB-21a), the
    /// unrevoked devices, and the number of live items. What a stub has no way to measure (the PHP and database facts, the rejection counters) is present and empty, as the schema wants.
    fn admin_status(&self, req: &HttpRequest) -> Resp {
        let st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        if who.role != "desktop" {
            return err(403, "forbidden", "Only a desktop reads the status.");
        }
        let polls: u32 = st.holds.values().map(|h| h.waiting).sum();
        let pairs = st.pair_waiting;
        let devices: Vec<Json> = st
            .devices
            .iter()
            .filter(|d| !d.revoked)
            .map(|d| {
                Json::obj([
                    ("id", Json::str(d.id.clone())),
                    ("role", Json::str(d.role.clone())),
                    ("name", Json::str(d.name.clone())),
                    ("online", Json::Bool(false)),
                ])
            })
            .collect();
        let now = self.now();
        let live_items: usize = st.mailboxes.values().map(|m| m.items.iter().filter(|i| !i.acked && i.exp > now).count()).sum();
        let bytes: usize = st.mailboxes.values().flat_map(|m| m.items.iter().filter(|i| !i.acked && i.exp > now)).map(|i| i.body.len()).sum();
        ok(Json::obj([
            ("v", Json::int(1)),
            ("version", Json::str("oaiy-relay-stub 0.1.0")),
            ("php", Json::obj([("version", Json::str("0")), ("sapi", Json::str("stub")), ("extensions", Json::Arr(vec![]))])),
            ("db", Json::obj([("driver", Json::str("sqlite"))])),
            ("items", Json::obj([("live", Json::int(live_items as u64)), ("bytes", Json::int(bytes as u64)), ("oldestAgeS", Json::Null)])),
            ("devices", Json::Arr(devices)),
            (
                "holds",
                Json::obj([
                    ("soft", Json::int(3)),
                    ("hard", Json::int(4)),
                    ("measured", Json::Bool(false)),
                    ("byKind", Json::obj([("poll", Json::int(polls)), ("pair", Json::int(pairs))])),
                    ("live", Json::int(polls + pairs)),
                ]),
            ),
            ("rejected24h", Json::Obj(vec![])),
            ("noAuthHeader24h", Json::int(0)),
            ("tokensOlderThan90d", Json::Arr(vec![])),
            ("warnings", Json::Arr(vec![])),
            ("time", Json::int(now)),
        ]))
    }

    fn revoke_route(&self, req: &HttpRequest, path: &str) -> Resp {
        let st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        if who.role != "desktop" {
            return err(403, "forbidden", "Only a desktop revokes.");
        }
        let target = path.trim_start_matches("/v1/devices/").trim_end_matches("/revoke").to_string();
        let allowed = st.devices.iter().any(|d| d.id == target && (d.owner_desktop.as_deref() == Some(who.device.as_str()) || d.role == "provider"));
        drop(st);
        if !allowed {
            return err(404, "not_found", "No such device.");
        }
        self.revoke(&target);
        Resp { status: 204, headers: Vec::new(), body: String::new(), time: None }
    }

    // ------------------------------------------------------------------------------------------------------------ admission

    fn admission(&self, req: &HttpRequest) -> Resp {
        let cfg = self.cfg();
        let st = self.lock();
        let who = match self.principal(&st, req) {
            Ok(p) => p,
            Err(r) => return r,
        };
        if !cfg.call_features {
            return err(403, "feature_disabled", "Call features are off.");
        }
        let Some(body) = req.body.as_ref() else { return err(400, "invalid_request", "A body is required.") };
        let Ok(doc) = json::parse(body) else { return err(400, "invalid_request", "Not JSON.") };
        let now = self.now();
        let device = st.devices.iter().find(|d| d.id == who.device).cloned().expect("the principal's device");
        let transports: Vec<String> = match doc.get("supportedTransports") {
            None => vec!["relay".into()],
            Some(t) => match t.as_array() {
                Some(a) if (1..=2).contains(&a.len()) && a.iter().all(|x| matches!(x.as_str(), Some("relay" | "relay-poll"))) => {
                    a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()
                }
                _ => return err(400, "invalid_request", "supportedTransports"),
            },
        };
        let secret = [1u8; 32];
        let bearer = |claims: &str| {
            let mac = hmac_sha256(&secret, claims.as_bytes()).expect("hmac");
            let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
            format!("{BEARER_PREFIX}{}.{}", hex(claims.as_bytes()), hex(&mac))
        };
        let poll_mode = transports.iter().any(|t| t == "relay-poll");
        let stream = transports.iter().any(|t| t == "relay") && cfg.stream_probe;
        if !poll_mode && !stream {
            return err(422, "unprocessable", "This host serves no carrier the client can open.");
        }
        let endpoints = {
            let mut m = vec![
                ("challengeUrl", Json::str("https://relay.stub.test/v1/aokie-companion/relay/challenge")),
                ("framesUrl", Json::str("https://relay.stub.test/v1/aokie-companion/relay/frames")),
                ("streamUrl", Json::str("https://relay.stub.test/v1/aokie-companion/relay/stream")),
            ];
            if !stream {
                m.push(("mode", Json::str("poll")));
            }
            Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
        };
        let ice = Json::Arr(vec![Json::obj([
            ("urls", Json::Arr(vec![Json::str("stun:stun.stub.test:3478")])),
            ("username", Json::str("")),
            ("credential", Json::str("")),
        ])]);
        match who.role.as_str() {
            "phone" => {
                let (Some(app), Some(dev), Some(holder)) = (doc.get_str("appId"), doc.get_str("deviceId"), doc.get_str("holderKeyThumbprint")) else {
                    return err(400, "invalid_request", "appId, deviceId and holderKeyThumbprint are required.");
                };
                if dev != device.id || Some(holder) != device.thumbprint.as_deref() || Some(app) != device.app_id.as_deref() {
                    return err(403, "forbidden", "Not this phone.");
                }
                let Some(owner) = device.owner_desktop.as_ref().and_then(|o| st.devices.iter().find(|d| &d.id == o && !d.revoked)) else {
                    return err(403, "forbidden", "No desktop.");
                };
                if !device.grants.iter().any(|g| g == "state_read") {
                    return err(403, "forbidden", "No state_read.");
                }
                let peer = device.peer_thumbprint.clone().or_else(|| owner.thumbprint.clone()).unwrap_or_default();
                let claims = Json::obj([
                    ("aud", Json::str("aokie-v2-gateway")),
                    ("appId", Json::str(app)),
                    ("subjectId", Json::str(device.id.clone())),
                    ("role", Json::str("mobile")),
                    ("holderKeyThumbprint", Json::str(holder)),
                    ("expectedPeerKeyThumbprint", Json::str(peer.clone())),
                    ("scopes", Json::Arr(device.grants.iter().map(|g| Json::str(g.clone())).collect())),
                    ("dsk", Json::str(owner.id.clone())),
                    ("exp", Json::int(now + 90)),
                    ("jti", Json::str("adm_00000000000000000000000000000001")),
                ])
                .to_compact();
                let iso = "2026-09-21T14:13:20Z";
                ok(Json::obj([
                    ("accessToken", Json::str(bearer(&claims))),
                    ("tokenType", Json::str("Bearer")),
                    ("expiresIn", Json::int(90)),
                    ("expiresAt", Json::int(now + 90)),
                    ("gatewayUrl", Json::str("wss://relay.stub.test/v2/realtime")),
                    ("appId", Json::str(app)),
                    ("scopes", Json::Arr(device.grants.iter().map(|g| Json::str(g.clone())).collect())),
                    ("iceServers", ice),
                    ("relayOnly", Json::Bool(false)),
                    ("turnCredentialExpiresAt", Json::Null),
                    ("holderKeyThumbprint", Json::str(holder)),
                    ("relay", endpoints),
                    ("subjectId", Json::str(device.id.clone())),
                    ("role", Json::str("mobile")),
                    ("expectedPeerKeyThumbprint", Json::str(peer)),
                    (
                        "device",
                        Json::obj([
                            ("id", Json::str(device.id.clone())),
                            ("appId", Json::str(app)),
                            ("subjectId", Json::str(device.id.clone())),
                            ("role", Json::str("mobile")),
                            ("displayName", Json::str(if device.name.is_empty() { "Phone".to_string() } else { device.name.clone() })),
                            ("grants", Json::Arr(device.grants.iter().map(|g| Json::str(g.clone())).collect())),
                            ("approvedAt", Json::str(iso)),
                            ("lastSeenAt", Json::str(iso)),
                        ]),
                    ),
                ]))
            }
            "desktop" => {
                let (Some(app), Some(plugin)) = (doc.get_str("appId"), doc.get_str("pluginId")) else {
                    return err(400, "invalid_request", "appId and pluginId are required.");
                };
                let echoed: Vec<(&str, Json)> =
                    ["endpointPublicKey", "holderKeyThumbprint", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash"]
                        .into_iter()
                        .map(|k| doc.get(k).map(|v| (k, v.clone())))
                        .collect::<Option<_>>()
                        .unwrap_or_default();
                if echoed.len() != 5 {
                    return err(400, "invalid_request", "The roster members are required.");
                }
                let claims = Json::obj([
                    ("aud", Json::str("aokie-v2-gateway")),
                    ("appId", Json::str(app)),
                    ("subjectId", Json::str(plugin)),
                    ("role", Json::str("plugin")),
                    ("exp", Json::int(now + 90)),
                ])
                .to_compact();
                let mut m: Vec<(&str, Json)> = vec![
                    ("accessToken", Json::str(bearer(&claims))),
                    ("tokenType", Json::str("Bearer")),
                    ("expiresIn", Json::int(90)),
                    ("expiresAt", Json::int(now + 90)),
                    ("gatewayUrl", Json::str("wss://relay.stub.test/v2/realtime")),
                    ("appId", Json::str(app)),
                    ("subjectId", Json::str(plugin)),
                    ("role", Json::str("plugin")),
                    ("scopes", Json::Arr(vec![Json::str("state_read"), Json::str("rtc_signal")])),
                    (
                        "device",
                        Json::obj([
                            ("id", Json::str(plugin)),
                            ("appId", Json::str(app)),
                            ("subjectId", Json::str(plugin)),
                            ("role", Json::str("plugin")),
                        ]),
                    ),
                    ("iceServers", ice),
                    ("relayOnly", Json::Bool(false)),
                    ("turnCredentialExpiresAt", Json::Null),
                    ("relay", endpoints),
                ];
                m.extend(echoed);
                ok(Json::Obj(m.into_iter().map(|(k, v)| (k.to_string(), v)).collect()))
            }
            _ => err(403, "forbidden", "Not for this role."),
        }
    }

    /// The URL of an `oaiy://enroll` key made for this stub (`percent_encode` of the public URL), for a test that wants to break one.
    pub fn enrolment_uri_prefix(&self) -> String {
        format!("oaiy://enroll?v=1&u={}", percent_encode(&self.cfg().public_url))
    }
}

impl HttpClient for StubRelay {
    fn send(&self, request: &HttpRequest) -> Result<HttpResponse, TransportError> {
        if request.cancel.is_cancelled() {
            return Err(TransportError::Cancelled);
        }
        // A hold runs on the caller's thread and looks at the request's cancel every 40 ms.
        self.handle(request)
    }
}
