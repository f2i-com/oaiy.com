//! What a client keeps: secrets, the profile of the relay it belongs to, and the poll cursor with what it accepted.
//!
//! Three traits and the reference implementations a test and a small host need (in memory and on a file system). The platform supplies its own where it must: the desktop's
//! [`SecretStore`] is `oaiy-keystore` (feature `keystore`), the phone's is the Android Keystore. Every trait keeps the rule of the keystore: **`Ok(None)` is "never stored" and
//! an error is "could not read", and the two are never the same**: a caller that meets an error does not enrol a new identity and does not treat the name as a first run.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use zeroize::Zeroizing;

use crate::error::Error;
use crate::ids;
use crate::json::{self, Json};
use crate::keys::{VerifyKey, X25519Public};
use crate::url::RelayUrl;

/// The name the device token is stored under (the same as `oaiy_keystore::names::RELAY_TOKEN`).
pub const SECRET_TOKEN: &str = "relay.token";

/// Why a store could not do what it was asked: in words that carry no secret.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreError(pub String);

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "store: {}", self.0)
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError(format!("{:?}", e.kind()))
    }
}

impl From<Error> for StoreError {
    fn from(e: Error) -> Self {
        StoreError(e.to_string())
    }
}

/// Named secrets (a token, a seed).
pub trait SecretStore: Send + Sync {
    /// The secret stored under `name`: `Ok(None)` is "never stored", an error is "could not read".
    fn get(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>, StoreError>;
    /// Stores `value` under `name`, replacing any earlier one. `Err` means nothing changed.
    fn put(&self, name: &str, value: &[u8]) -> Result<(), StoreError>;
    /// Removes `name`; removing a name that is not there is not an error.
    fn delete(&self, name: &str) -> Result<(), StoreError>;
}

/// Secrets in memory: for tests and for a host that keeps its own copy elsewhere.
#[derive(Default)]
pub struct MemorySecretStore {
    values: Mutex<Vec<(String, Zeroizing<Vec<u8>>)>>,
    /// While true every operation fails: a store that cannot be read.
    pub broken: std::sync::atomic::AtomicBool,
}

impl MemorySecretStore {
    /// An empty store.
    pub fn new() -> MemorySecretStore {
        MemorySecretStore::default()
    }

    fn check(&self) -> Result<(), StoreError> {
        if self.broken.load(std::sync::atomic::Ordering::SeqCst) {
            Err(StoreError("the secret store cannot be read".into()))
        } else {
            Ok(())
        }
    }
}

impl SecretStore for MemorySecretStore {
    fn get(&self, name: &str) -> Result<Option<Zeroizing<Vec<u8>>>, StoreError> {
        self.check()?;
        let values = self.values.lock().map_err(|_| StoreError("poisoned".into()))?;
        Ok(values.iter().find(|(k, _)| k == name).map(|(_, v)| Zeroizing::new(v.to_vec())))
    }

    fn put(&self, name: &str, value: &[u8]) -> Result<(), StoreError> {
        self.check()?;
        let mut values = self.values.lock().map_err(|_| StoreError("poisoned".into()))?;
        values.retain(|(k, _)| k != name);
        values.push((name.to_string(), Zeroizing::new(value.to_vec())));
        Ok(())
    }

    fn delete(&self, name: &str) -> Result<(), StoreError> {
        self.check()?;
        self.values.lock().map_err(|_| StoreError("poisoned".into()))?.retain(|(k, _)| k != name);
        Ok(())
    }
}

/// Which product a profile is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileKind {
    /// A desktop (a `dev-` device of role `desktop`).
    Desktop,
    /// A phone (a `dev-` device of role `phone`, made by pairing).
    Phone,
}

/// What a phone pins from the MAC-verified offer: the **only** source of its peer pin (README 10.1), never the relay's word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerPin {
    /// The desktop's relay device id.
    pub desktop_connection_id: String,
    /// The desktop's name at pairing.
    pub desktop_name: String,
    /// The desktop's endpoint key (it signed the approval receipt).
    pub desktop_endpoint: VerifyKey,
    /// The desktop's endpoint X25519 key.
    pub desktop_x25519: X25519Public,
    /// The host identity's Ed25519 key (it signs rings).
    pub host_ed25519: VerifyKey,
    /// The host identity's X25519 key.
    pub host_x25519: X25519Public,
}

/// The relay a client belongs to, as it is kept between runs. No secret is in it (the token is a [`SecretStore`] entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayProfile {
    /// Desktop or phone.
    pub kind: ProfileKind,
    /// The relay's base URL.
    pub relay: RelayUrl,
    /// The relay's id (`rly-...`).
    pub relay_id: String,
    /// The thumbprint of the relay key the owner pinned (the `f` of the enrolment or pairing key, or the offer's fingerprint): the identity proof is checked against it.
    pub relay_thumbprint: String,
    /// This device's id on the relay.
    pub device_id: String,
    /// This device's name.
    pub name: String,
    /// When it was enrolled or paired, in relay time.
    pub enrolled_at: i64,
    /// A phone's app.
    pub app_id: Option<String>,
    /// A phone's grants as the desktop approved them (advisory until the receipt is verified).
    pub grants: Vec<String>,
    /// A phone's peer pin.
    pub peer: Option<PeerPin>,
}

fn key_json(k: &VerifyKey) -> Json {
    Json::str(k.to_b64u())
}

impl RelayProfile {
    /// The profile as JSON text (`{"v":1,...}`).
    pub fn to_json(&self) -> String {
        let mut m = vec![
            ("v".to_string(), Json::int(1)),
            ("kind".to_string(), Json::str(if self.kind == ProfileKind::Desktop { "desktop" } else { "phone" })),
            ("relay".to_string(), Json::str(self.relay.origin())),
            ("relayId".to_string(), Json::str(self.relay_id.clone())),
            ("relayThumbprint".to_string(), Json::str(self.relay_thumbprint.clone())),
            ("deviceId".to_string(), Json::str(self.device_id.clone())),
            ("name".to_string(), Json::str(self.name.clone())),
            ("enrolledAt".to_string(), Json::int(self.enrolled_at)),
        ];
        if let Some(a) = &self.app_id {
            m.push(("appId".into(), Json::str(a.clone())));
        }
        if self.kind == ProfileKind::Phone {
            m.push(("grants".into(), Json::Arr(self.grants.iter().map(|g| Json::str(g.clone())).collect())));
        }
        if let Some(p) = &self.peer {
            m.push((
                "peer".into(),
                Json::obj([
                    ("desktopConnectionId", Json::str(p.desktop_connection_id.clone())),
                    ("desktopName", Json::str(p.desktop_name.clone())),
                    ("desktopEndpoint", key_json(&p.desktop_endpoint)),
                    ("desktopX25519", Json::str(p.desktop_x25519.to_b64u())),
                    ("hostEd25519", key_json(&p.host_ed25519)),
                    ("hostX25519", Json::str(p.host_x25519.to_b64u())),
                ]),
            ));
        }
        Json::Obj(m).to_compact()
    }

    /// Reads what [`RelayProfile::to_json`] wrote. Strict about shape, so a damaged file is an error and never a half-built profile.
    pub fn from_json(text: &[u8]) -> Result<RelayProfile, StoreError> {
        let doc = json::parse(text)?;
        let bad = || StoreError("the relay profile is damaged".into());
        if doc.get("v").and_then(Json::as_int) != Some(1) {
            return Err(bad());
        }
        let kind = match doc.get_str("kind") {
            Some("desktop") => ProfileKind::Desktop,
            Some("phone") => ProfileKind::Phone,
            _ => return Err(bad()),
        };
        let relay = RelayUrl::parse(doc.get_str("relay").ok_or_else(bad)?)?;
        let relay_id = doc.get_str("relayId").filter(|s| ids::is_relay_id(s)).ok_or_else(bad)?.to_string();
        let relay_thumbprint = doc.get_str("relayThumbprint").filter(|s| ids::is_thumbprint(s)).ok_or_else(bad)?.to_string();
        let device_id = doc.get_str("deviceId").filter(|s| ids::is_device_id(s)).ok_or_else(bad)?.to_string();
        let name = doc.get_str("name").ok_or_else(bad)?.to_string();
        let enrolled_at = doc.get("enrolledAt").and_then(Json::as_int).and_then(|n| i64::try_from(n).ok()).ok_or_else(bad)?;
        let app_id = match doc.get("appId") {
            None => None,
            Some(a) => Some(a.as_str().filter(|s| ids::is_app_id(s)).ok_or_else(bad)?.to_string()),
        };
        let grants = match doc.get("grants") {
            None => Vec::new(),
            Some(g) => g
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|x| x.as_str().filter(|s| ids::is_grant(s)).map(str::to_string).ok_or_else(bad))
                .collect::<Result<_, _>>()?,
        };
        let peer = match doc.get("peer") {
            None => None,
            Some(p) => {
                let s = |k: &str| p.get_str(k).ok_or_else(bad);
                Some(PeerPin {
                    desktop_connection_id: s("desktopConnectionId")
                        .and_then(|v| if ids::is_device_id(v) { Ok(v.to_string()) } else { Err(bad()) })?,
                    desktop_name: s("desktopName")?.to_string(),
                    desktop_endpoint: VerifyKey::from_b64u(s("desktopEndpoint")?)?,
                    desktop_x25519: X25519Public::from_b64u(s("desktopX25519")?)?,
                    host_ed25519: VerifyKey::from_b64u(s("hostEd25519")?)?,
                    host_x25519: X25519Public::from_b64u(s("hostX25519")?)?,
                })
            }
        };
        Ok(RelayProfile { kind, relay, relay_id, relay_thumbprint, device_id, name, enrolled_at, app_id, grants, peer })
    }
}

/// The relay profile on disk or in memory.
pub trait ProfileStore: Send + Sync {
    /// The profile, `Ok(None)` when none was ever stored, an error when it cannot be read or is damaged (never treated as "not enrolled").
    fn load(&self) -> Result<Option<RelayProfile>, StoreError>;
    /// Stores the profile, atomically.
    fn save(&self, profile: &RelayProfile) -> Result<(), StoreError>;
    /// Forgets the profile (a revoked device, "forget this relay").
    fn clear(&self) -> Result<(), StoreError>;
}

/// A profile in memory.
#[derive(Default)]
pub struct MemoryProfileStore(Mutex<Option<RelayProfile>>);

impl MemoryProfileStore {
    /// An empty store.
    pub fn new() -> MemoryProfileStore {
        MemoryProfileStore::default()
    }
}

impl ProfileStore for MemoryProfileStore {
    fn load(&self) -> Result<Option<RelayProfile>, StoreError> {
        Ok(self.0.lock().map_err(|_| StoreError("poisoned".into()))?.clone())
    }

    fn save(&self, profile: &RelayProfile) -> Result<(), StoreError> {
        *self.0.lock().map_err(|_| StoreError("poisoned".into()))? = Some(profile.clone());
        Ok(())
    }

    fn clear(&self) -> Result<(), StoreError> {
        *self.0.lock().map_err(|_| StoreError("poisoned".into()))? = None;
        Ok(())
    }
}

/// Writes `bytes` to `path` so that a reader sees the old file or the new one and never half of either: a temporary file in the same directory, flushed to the device, then renamed
/// over the target.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let dir = path.parent().ok_or_else(|| StoreError("no parent directory".into()))?;
    fs::create_dir_all(dir)?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// The profile as one file (`relay.json`), written atomically.
pub struct FileProfileStore {
    path: PathBuf,
}

impl FileProfileStore {
    /// The profile at `path`.
    pub fn new(path: impl Into<PathBuf>) -> FileProfileStore {
        FileProfileStore { path: path.into() }
    }
}

impl ProfileStore for FileProfileStore {
    fn load(&self) -> Result<Option<RelayProfile>, StoreError> {
        match fs::read(&self.path) {
            Ok(bytes) => RelayProfile::from_json(&bytes).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn save(&self, profile: &RelayProfile) -> Result<(), StoreError> {
        write_atomic(&self.path, profile.to_json().as_bytes())
    }

    fn clear(&self) -> Result<(), StoreError> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

/// An item as the relay delivers it (`item.schema.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    /// `seq`: per mailbox.
    pub seq: u64,
    /// `id`, chosen by the sender.
    pub id: String,
    /// `lane`.
    pub lane: String,
    /// `from`: a hint, never an authority; a consumer de-duplicates on `(from, id)` and only after authenticating the sender.
    pub from: String,
    /// `at`: the relay's receive time.
    pub at: u64,
    /// `exp`: the relay-enforced expiry.
    pub exp: u64,
    /// `hdr`, an object (as compact JSON text).
    pub hdr: String,
    /// `body`: an opaque string, exactly as posted.
    pub body: String,
    /// `rp`: the reply box id, when there is one.
    pub rp: Option<String>,
}

impl Item {
    /// Reads an element of `items`: every required member of the schema, typed. `None` for one that is not an item (it is still **accepted** as far as the cursor goes,
    /// P2: a poison item cannot stall it, and the consumer counts it dropped).
    pub fn from_json(v: &Json) -> Option<Item> {
        let seq = v.get_uint53("seq")?;
        let id = v.get_str("id").filter(|s| ids::is_item_id(s))?;
        let lane = v.get_str("lane")?;
        let from = v.get_str("from")?;
        let hdr = v.get("hdr").filter(|h| h.is_object())?;
        let body = v.get_str("body")?;
        let rp = match v.get("rp") {
            None => None,
            Some(r) => Some(r.as_str().filter(|s| ids::is_pid(s))?.to_string()),
        };
        Some(Item {
            seq,
            id: id.to_string(),
            lane: lane.to_string(),
            from: from.to_string(),
            at: v.get_uint53("at")?,
            exp: v.get_uint53("exp")?,
            hdr: hdr.to_compact(),
            body: body.to_string(),
            rp,
        })
    }
}

/// An item that the rules accepted: its `seq`, the item when it has the shape of one, and what was received.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptedItem {
    /// `seq`.
    pub seq: u64,
    /// The item, or `None` for something that has a `seq` and is not an item.
    pub item: Option<Item>,
    /// The element as received.
    pub raw: Json,
}

/// The poll cursor: the highest `seq` accepted and the relay's epoch, as the relay gave it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PollCursor {
    /// `since`.
    pub since: u64,
    /// `epoch`, sent back byte for byte; `None` until the relay has told one (an omitted epoch is no check).
    pub epoch: Option<String>,
}

/// What a poll that made progress asks its store to write before the poll that acknowledges it is sent (P2, "to persist").
#[derive(Debug, Clone)]
pub struct PersistBatch<'a> {
    /// The accepted items, in ascending `seq`; none on a reset.
    pub items: &'a [AcceptedItem],
    /// The `since` to carry next: the highest `seq` of the items, or a reset's cursor.
    pub since: u64,
    /// The relay's epoch of this answer.
    pub epoch: &'a str,
    /// True when the answer was a reset: the cursor is the server's, it may be lower than the one stored, and "mailbox reset: in-flight items may be lost" is recorded.
    pub reset: bool,
}

/// Where the poll cursor and the accepted items go. **A write that succeeds is durable**: the relay deletes what the next poll acknowledges, so the items and the cursor
/// must be on the device before that poll is sent; and a failure moves nothing (the items come again, and the consumer de-duplicates on `(from, id)`).
pub trait PollStore: Send {
    /// The stored cursor (`since` 0 and no epoch when nothing was ever stored).
    fn load(&mut self) -> Result<PollCursor, StoreError>;
    /// Writes what was accepted and the cursor and epoch that go with it.
    fn persist(&mut self, batch: &PersistBatch<'_>) -> Result<(), StoreError>;
    /// Forgets the stored epoch and keeps the cursor (the action `clear_epoch`: the retry after a first `400` leaves the epoch out).
    fn clear_epoch(&mut self) -> Result<(), StoreError>;
}

/// A cursor and an inbox in memory, with a switch to make the next writes fail.
#[derive(Default)]
pub struct MemoryPollStore {
    cursor: PollCursor,
    /// Everything accepted, in order (the "ledger" a test looks at).
    pub accepted: Vec<AcceptedItem>,
    /// How many of the next writes fail.
    pub fail_writes: u32,
    /// How many times a reset was recorded.
    pub resets: u32,
}

impl MemoryPollStore {
    /// An empty store.
    pub fn new() -> MemoryPollStore {
        MemoryPollStore::default()
    }

    /// A store that starts at a given cursor.
    pub fn at(since: u64, epoch: Option<&str>) -> MemoryPollStore {
        MemoryPollStore { cursor: PollCursor { since, epoch: epoch.map(str::to_string) }, ..Default::default() }
    }

    /// The cursor as stored now.
    pub fn cursor(&self) -> &PollCursor {
        &self.cursor
    }
}

impl PollStore for MemoryPollStore {
    fn load(&mut self) -> Result<PollCursor, StoreError> {
        Ok(self.cursor.clone())
    }

    fn persist(&mut self, batch: &PersistBatch<'_>) -> Result<(), StoreError> {
        if self.fail_writes > 0 {
            self.fail_writes -= 1;
            return Err(StoreError("the disk is full".into()));
        }
        self.accepted.extend(batch.items.iter().cloned());
        self.cursor = PollCursor { since: batch.since, epoch: Some(batch.epoch.to_string()) };
        if batch.reset {
            self.resets += 1;
        }
        Ok(())
    }

    fn clear_epoch(&mut self) -> Result<(), StoreError> {
        self.cursor.epoch = None;
        Ok(())
    }
}

/// The cursor as `cursor.json` and the accepted items appended to `inbox.jsonl` (one JSON object per line: the item as received), in a directory. The items are written and
/// flushed **before** the cursor, so a crash between the two re-delivers and loses nothing (a consumer de-duplicates on `(from, id)`). A host that has a ledger of its own
/// implements [`PollStore`] over it instead.
pub struct FilePollStore {
    dir: PathBuf,
}

impl FilePollStore {
    /// A store in `dir` (created when first written).
    pub fn new(dir: impl Into<PathBuf>) -> FilePollStore {
        FilePollStore { dir: dir.into() }
    }

    fn cursor_path(&self) -> PathBuf {
        self.dir.join("cursor.json")
    }

    fn inbox_path(&self) -> PathBuf {
        self.dir.join("inbox.jsonl")
    }

    fn write_cursor(&self, cursor: &PollCursor, reset_note: bool) -> Result<(), StoreError> {
        let mut m = vec![("since".to_string(), Json::int(cursor.since))];
        if let Some(e) = &cursor.epoch {
            m.push(("epoch".to_string(), Json::str(e.clone())));
        }
        if reset_note {
            m.push(("note".to_string(), Json::str("mailbox reset: in-flight items may be lost")));
        }
        write_atomic(&self.cursor_path(), Json::Obj(m).to_compact().as_bytes())
    }

    /// The items appended so far, as received.
    pub fn read_inbox(&self) -> Result<Vec<Json>, StoreError> {
        match fs::read(self.inbox_path()) {
            Ok(bytes) => bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()).map(|l| json::parse(l).map_err(StoreError::from_json)).collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e.into()),
        }
    }
}

impl StoreError {
    fn from_json(e: json::JsonError) -> StoreError {
        StoreError(format!("damaged inbox: {e}"))
    }
}

impl From<json::JsonError> for StoreError {
    fn from(e: json::JsonError) -> Self {
        StoreError(format!("damaged file: {e}"))
    }
}

impl PollStore for FilePollStore {
    fn load(&mut self) -> Result<PollCursor, StoreError> {
        match fs::read(self.cursor_path()) {
            Ok(bytes) => {
                let doc = json::parse(&bytes)?;
                let since = doc.get_uint53("since").ok_or_else(|| StoreError("damaged cursor".into()))?;
                let epoch = match doc.get("epoch") {
                    None => None,
                    Some(e) => Some(e.as_str().filter(|s| ids::is_epoch(s)).ok_or_else(|| StoreError("damaged cursor".into()))?.to_string()),
                };
                Ok(PollCursor { since, epoch })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(PollCursor::default()),
            Err(e) => Err(e.into()),
        }
    }

    fn persist(&mut self, batch: &PersistBatch<'_>) -> Result<(), StoreError> {
        fs::create_dir_all(&self.dir)?;
        if !batch.items.is_empty() {
            let mut f = fs::OpenOptions::new().create(true).append(true).open(self.inbox_path())?;
            for it in batch.items {
                f.write_all(it.raw.to_compact().as_bytes())?;
                f.write_all(b"\n")?;
            }
            f.sync_all()?;
        }
        self.write_cursor(&PollCursor { since: batch.since, epoch: Some(batch.epoch.to_string()) }, batch.reset)
    }

    fn clear_epoch(&mut self) -> Result<(), StoreError> {
        let cursor = self.load()?;
        self.write_cursor(&PollCursor { since: cursor.since, epoch: None }, false)
    }
}
