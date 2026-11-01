//! What OAIY Network keeps in the data folder (`network/`): its settings (`settings.json`) and the paired computers
//! (`peers.json`, owner-only: a peer's certificate fingerprint is what lets it in).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::secret_file::{keep_aside, kept_aside, read_text_patiently, Patience, Retry, Text};

pub const SETTINGS_FILE: &str = "settings.json";
pub const PEERS_FILE: &str = "peers.json";
pub const DEFAULTS_FILE: &str = "defaults.json";
pub const SERVICES_FILE: &str = "remote-services.json";

/// What went wrong reading each file of this store.
struct Trouble {
    /// In words for the person, for the log and the Networking page.
    said: String,
    /// The file is still there and was not read: saving would write over what nobody has seen.
    keep_out: bool,
    /// Set while saving is kept out: when the file is read again.
    retry: Option<Retry>,
}

fn troubles() -> MutexGuard<'static, BTreeMap<PathBuf, Trouble>> {
    static TROUBLES: OnceLock<Mutex<BTreeMap<PathBuf, Trouble>>> = OnceLock::new();
    TROUBLES.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// A file of this store as it was left. Only a file that is not there is an empty store: paired computers read as none
/// would be written over at the next save. One that was read and is not what it should be (cut short, or not text)
/// is kept aside as `<name>.corrupt` (`.corrupt.1`, and so on: no earlier one is replaced), to repair; one that could
/// not be read may be perfectly good (a backup tool holding it), so it stays where it is, and neither it nor one that
/// would not move is saved over: each is read again ([`read_kept_again`]) until it can be.
fn read_kept<T: DeserializeOwned + Default>(path: &Path) -> T {
    troubles().remove(path);
    read_file(path, &Patience::default().start).unwrap_or_default()
}

/// A file that was left to be read yet ([`read_kept`]), if it is time to: what it holds once that is known, and it is
/// then no longer kept out.
fn read_kept_again<T: DeserializeOwned + Default>(path: &Path) -> Option<T> {
    if !troubles().get(path).is_some_and(|t| t.retry.as_ref().is_some_and(Retry::due)) {
        return None;
    }
    read_file(path, &[])
}

/// What the file at `path` holds, or `None` for one that is not known yet, with what went wrong noted down.
fn read_file<T: DeserializeOwned + Default>(path: &Path, pauses: &[Duration]) -> Option<T> {
    let why = match read_text_patiently(path, pauses) {
        Text::Missing => {
            note_copy_of(path);
            return Some(T::default());
        }
        Text::Text(text) => match serde_json::from_str(&text) {
            Ok(kept) => {
                if holds_nothing(text.as_bytes()) {
                    note_copy_of(path);
                } else {
                    troubles().remove(path);
                }
                return Some(kept);
            }
            Err(e) => e.to_string(),
        },
        Text::Undecodable(why) => why,
        Text::Unreadable(e) => {
            not_read(path, format!("{} could not be read ({e}), since another program may have it open or its permissions may not let OAIY read it: nothing will be saved over it, and it is read again until it can be", path.display()));
            return None;
        }
    };
    match keep_aside(path) {
        Ok(aside) => {
            let said = format!("{} could not be read ({why}): it is kept as {}, to repair or restore from a backup", path.display(), aside.display());
            log::error!("network: {said}");
            troubles().insert(path.to_path_buf(), Trouble { said, keep_out: false, retry: None });
            Some(T::default())
        }
        Err(moved) => {
            not_read(path, format!("{} could not be read ({why}) or moved aside ({moved}): nothing will be saved over it, so repair it or move it aside, and it is read again until that is done", path.display()));
            None
        }
    }
}

/// The file stays where it is, not known, and nothing is saved over it until it is.
fn not_read(path: &Path, said: String) {
    let mut all = troubles();
    match all.get_mut(path) {
        Some(Trouble { said: was, retry: Some(retry), .. }) => {
            retry.failed();
            *was = said;
        }
        _ => {
            log::error!("network: {said}");
            all.insert(path.to_path_buf(), Trouble { said, keep_out: true, retry: Some(Retry::began(&Patience::default())) });
        }
    }
}

/// A store with nothing in it (no file, or an empty one) beside a copy kept aside: that copy is why, and a page with no
/// computers on it says so for as long as that is the case, restarts included.
fn note_copy_of(path: &Path) {
    let said = kept_aside(path).pop().map(|aside| format!("{} could not be read when OAIY last tried: it is kept as {}, to repair or restore from a backup", path.display(), aside.display()));
    match said {
        Some(said) => troubles().insert(path.to_path_buf(), Trouble { said, keep_out: false, retry: None }),
        None => troubles().remove(path),
    };
}

/// Whether what a file says is nothing at all: `[]` or `{}`, as a list or a map with nothing in it is saved.
fn holds_nothing(saved: &[u8]) -> bool {
    matches!(String::from_utf8_lossy(saved).trim(), "[]" | "{}")
}

/// Save a file of this store, unless [`read_kept`] left it to be read yet.
fn write_kept(path: &Path, body: Vec<u8>) -> Result<(), String> {
    if let Some(trouble) = troubles().get(path).filter(|t| t.keep_out) {
        return Err(trouble.said.clone());
    }
    let something = !holds_nothing(&body);
    crate::secret_file::write(path, body).map_err(|e| e.to_string())?;
    if something {
        troubles().remove(path);
    }
    Ok(())
}

/// What went wrong with the files in `dir`, for the Networking page.
pub fn problems(dir: &Path) -> Vec<String> {
    troubles().iter().filter(|(path, _)| path.starts_with(dir)).map(|(_, t)| t.said.clone()).collect()
}

/// Whether a file in `dir` is still to be read.
pub fn unread(dir: &Path) -> bool {
    troubles().iter().any(|(path, t)| path.starts_with(dir) && t.retry.is_some())
}

/// Time to read the file at `path` again, now (a test has no use for the pauses it would wait out).
#[cfg(test)]
pub(crate) fn due_now(path: &Path) {
    let now = Patience { start: vec![], later: vec![Duration::ZERO], window: Duration::ZERO };
    troubles().get_mut(path).unwrap().retry = Some(Retry::began(&now));
}

/// The port a worker listens on unless set (the desktop's API is 17972, its voice gateway 17872).
pub const DEFAULT_PORT: u16 = 17975;

/// What a computer is on the network, and what a paired computer is to this one. A computer can use other computers
/// (a primary to them), lend itself (a worker to them), or both; a pairing is one way, the computer that uses the
/// other the primary of it, so two computers that use each other are paired twice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Not on it (as with Network switched off).
    #[default]
    Standalone,
    /// Uses paired workers' models, through its own API; a peer of this role uses this computer.
    Primary,
    /// Makes what a paired primary asks of it, on its own engines; a peer of this role is one this computer uses.
    Worker,
    /// Both: uses other computers and lends itself (a computer's own, never a peer's).
    Both,
}

/// Network's settings: off until the person switches it on, using other computers, lending this one, or both.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub enabled: bool,
    /// The one role chosen before a computer could both use others and lend itself: what `uses` and `lends` are where
    /// they are not set (a settings file of then).
    pub role: Role,
    /// It uses other computers (their models, flows and services).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uses: Option<bool>,
    /// It lends itself to the computers that pair with it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lends: Option<bool>,
    /// The person lets the Agent set up OAIY Network here (its tools pair, unpair and choose what is used), and this
    /// computer pairs by itself with another that has this on too, with no codes to compare (a network the person
    /// trusts: their home's or office's).
    pub agent: bool,
    /// Lending itself, it lends its model files too (by their SHA-256), so a computer using it need not download
    /// them again: on unless its person switches it off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_models: Option<bool>,
    /// The name other computers show for this one (the computer's own name where empty).
    pub name: String,
    /// The port a worker listens on.
    pub port: u16,
    /// The address a worker listens on (one of this computer's LAN addresses), or every interface where empty.
    pub bind: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { enabled: false, role: Role::Standalone, uses: None, lends: None, agent: false, share_models: None, name: String::new(), port: DEFAULT_PORT, bind: String::new() }
    }
}

impl Settings {
    pub fn load(dir: &Path) -> Settings {
        read_kept(&dir.join(SETTINGS_FILE))
    }

    /// The settings, once a file that could not be read can be.
    pub fn read_again(dir: &Path) -> Option<Settings> {
        read_kept_again(&dir.join(SETTINGS_FILE))
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
        write_kept(&dir.join(SETTINGS_FILE), serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?)
    }

    /// It uses other computers now (Network on, and set so).
    pub fn uses(&self) -> bool {
        self.enabled && self.uses.unwrap_or(matches!(self.role, Role::Primary | Role::Both))
    }

    /// It lends itself now (Network on, and set so): its listener is up, and computers may ask to pair.
    pub fn lends(&self) -> bool {
        self.enabled && self.lends.unwrap_or(matches!(self.role, Role::Worker | Role::Both))
    }

    /// It lends its model files now (lending itself, and not switched off).
    pub fn shares_models(&self) -> bool {
        self.lends() && self.share_models != Some(false)
    }

    /// What it is now: none while Network is off, else using others, lending itself, or both.
    pub fn active_role(&self) -> Role {
        match (self.uses(), self.lends()) {
            (true, true) => Role::Both,
            (true, false) => Role::Primary,
            (false, true) => Role::Worker,
            (false, false) => Role::Standalone,
        }
    }
}

/// A paired computer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Peer {
    pub node_id: String,
    pub name: String,
    /// What it is to this computer: a worker (to a primary) or the primary (to a worker).
    pub role: Role,
    /// Its certificate's SHA-256, pinned when the two were paired: the only thing that lets it in.
    pub fingerprint: String,
    /// Where it was last reached (`host:port`; a worker's), refreshed from discovery when it moves.
    #[serde(default)]
    pub address: String,
    pub paired_at_ms: u64,
    /// Paired by itself (both computers let their Agent set up OAIY Network): no person compared codes, so the
    /// Networking page says so.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub by_itself: bool,
}

/// The paired computers, kept in `peers.json`.
#[derive(Debug)]
pub struct Peers {
    path: PathBuf,
    list: Vec<Peer>,
}

impl Peers {
    pub fn open(dir: &Path) -> Peers {
        let path = dir.join(PEERS_FILE);
        let list = read_kept(&path);
        Peers { path, list }
    }

    /// Read a file that could not be read, if it is time to.
    pub fn read_again(&mut self) {
        if let Some(list) = read_kept_again(&self.path) {
            self.list = list;
        }
    }

    pub fn list(&self) -> &[Peer] {
        &self.list
    }

    /// The pairing with `node_id` of this role: a computer this one uses (`Worker`), or one that uses it (`Primary`).
    pub fn get(&self, node_id: &str, role: Role) -> Option<&Peer> {
        self.list.iter().find(|p| p.node_id == node_id && p.role == role)
    }

    /// Whether `node_id` is paired with this computer, either way.
    pub fn knows(&self, node_id: &str) -> bool {
        self.list.iter().any(|p| p.node_id == node_id)
    }

    /// The paired computer with this certificate (a connection's own), of this role.
    pub fn by_fingerprint(&self, fingerprint: &str, role: Role) -> Option<&Peer> {
        self.list.iter().find(|p| p.fingerprint == fingerprint && p.role == role)
    }

    /// Whether `node_id` is kept here with another certificate than `fingerprint`, either way: a computer that names
    /// itself as one paired with this computer, which it is not (a node id is a computer's own say; its certificate,
    /// proved on each connection, is what it is).
    pub fn taken(&self, node_id: &str, fingerprint: &str) -> bool {
        self.list.iter().any(|p| p.node_id == node_id && p.fingerprint != fingerprint)
    }

    /// Pair (or pair again) with `peer`: one entry per node id and way, and per certificate and way; refused for a node
    /// id kept here with another certificate ([`Peers::taken`]): it would stand in for that computer, its defaults and
    /// services. In effect once it is kept: a pairing that could not be written down lets nothing in.
    pub fn add(&mut self, peer: Peer) -> Result<(), String> {
        if self.taken(&peer.node_id, &peer.fingerprint) {
            return Err(TAKEN.into());
        }
        let mut next: Vec<Peer> = self.list.iter().filter(|p| p.role != peer.role || (p.node_id != peer.node_id && p.fingerprint != peer.fingerprint)).cloned().collect();
        next.push(peer);
        save(&self.path, &next)?;
        self.list = next;
        Ok(())
    }

    /// Unpair, both ways: its connections are refused from now on, even where that could not be written down (then
    /// said so).
    pub fn remove(&mut self, node_id: &str) -> Result<bool, String> {
        let before = self.list.len();
        self.list.retain(|p| p.node_id != node_id);
        let removed = self.list.len() != before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// Unpair one way only: `node_id` no longer `role` (a pairing both ways that could not be finished the other way).
    pub fn remove_way(&mut self, node_id: &str, role: Role) -> Result<(), String> {
        let before = self.list.len();
        self.list.retain(|p| p.node_id != node_id || p.role != role);
        if self.list.len() != before {
            self.save()?;
        }
        Ok(())
    }

    /// A worker's new address (it moved: another DHCP lease, or another port), kept for the next start.
    pub fn moved(&mut self, node_id: &str, address: &str) -> Result<(), String> {
        match self.list.iter_mut().find(|p| p.node_id == node_id && p.role == Role::Worker) {
            Some(p) if p.address != address => {
                p.address = address.to_string();
                self.save()
            }
            _ => Ok(()),
        }
    }

    fn save(&self) -> Result<(), String> {
        save(&self.path, &self.list)
    }
}

/// What [`Peers::add`] says for a node id paired here with another certificate.
pub const TAKEN: &str = "another computer is paired here under that node id: unpair it first";

fn save(path: &Path, list: &[Peer]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
    }
    write_kept(path, serde_json::to_vec_pretty(list).map_err(|e| e.to_string())?)
}

/// A paired worker's model chosen as this computer's engine default for its kind (a primary's), with this computer's
/// own default for the kind then: one chosen here since, in Engines, takes its place again.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerDefault {
    /// Its id in the engine catalogue (`engine:<kind>:node.<nodeId>/<model>`).
    pub id: String,
    #[serde(default)]
    pub instead_of: Option<String>,
}

impl WorkerDefault {
    /// The worker it is on.
    pub fn node_id(&self) -> &str {
        self.id.split_once(":node.").and_then(|(_, rest)| rest.split_once('/')).map_or("", |(node, _)| node)
    }
}

/// The engine defaults on a worker, by kind, kept in `defaults.json`.
#[derive(Debug)]
pub struct Defaults {
    path: PathBuf,
    map: std::collections::BTreeMap<String, WorkerDefault>,
}

impl Defaults {
    pub fn open(dir: &Path) -> Defaults {
        let path = dir.join(DEFAULTS_FILE);
        let map = read_kept(&path);
        Defaults { path, map }
    }

    /// Read a file that could not be read, if it is time to.
    pub fn read_again(&mut self) {
        if let Some(map) = read_kept_again(&self.path) {
            self.map = map;
        }
    }

    pub fn all(&self) -> &std::collections::BTreeMap<String, WorkerDefault> {
        &self.map
    }

    pub fn get(&self, kind: &str) -> Option<&WorkerDefault> {
        self.map.get(kind)
    }

    /// `kind`'s default on a worker (`None`: this computer's own again), in effect once it is kept.
    pub fn set(&mut self, kind: &str, chosen: Option<WorkerDefault>) -> Result<(), String> {
        let mut next = self.map.clone();
        match chosen {
            Some(d) => next.insert(kind.to_string(), d),
            None => next.remove(kind),
        };
        if next == self.map {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
        }
        write_kept(&self.path, serde_json::to_vec_pretty(&next).map_err(|e| e.to_string())?)?;
        self.map = next;
        Ok(())
    }

    /// The defaults on `node_id` let go (it was unpaired): this computer's own again.
    pub fn drop_node(&mut self, node_id: &str) -> Result<(), String> {
        let kinds: Vec<String> = self.map.iter().filter(|(_, d)| d.node_id() == node_id).map(|(k, _)| k.clone()).collect();
        for kind in kinds {
            self.set(&kind, None)?;
        }
        Ok(())
    }
}

/// The services a primary uses on a worker in place of its own (OAIY Voice), by service id: the worker's node id.
#[derive(Debug)]
pub struct ServiceChoices {
    path: PathBuf,
    map: std::collections::BTreeMap<String, String>,
}

impl ServiceChoices {
    pub fn open(dir: &Path) -> ServiceChoices {
        let path = dir.join(SERVICES_FILE);
        let map = read_kept(&path);
        ServiceChoices { path, map }
    }

    /// Read a file that could not be read, if it is time to.
    pub fn read_again(&mut self) {
        if let Some(map) = read_kept_again(&self.path) {
            self.map = map;
        }
    }

    pub fn all(&self) -> &std::collections::BTreeMap<String, String> {
        &self.map
    }

    pub fn get(&self, service: &str) -> Option<&str> {
        self.map.get(service).map(String::as_str)
    }

    /// `service` used on worker `node` (`None`: this computer's own again), in effect once it is kept.
    pub fn set(&mut self, service: &str, node: Option<&str>) -> Result<(), String> {
        let mut next = self.map.clone();
        match node {
            Some(n) => next.insert(service.to_string(), n.to_string()),
            None => next.remove(service),
        };
        if next == self.map {
            return Ok(());
        }
        if let Some(dir) = self.path.parent() {
            crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
        }
        write_kept(&self.path, serde_json::to_vec_pretty(&next).map_err(|e| e.to_string())?)?;
        self.map = next;
        Ok(())
    }

    /// The services used on `node` let go (it was unpaired): this computer's own again.
    pub fn drop_node(&mut self, node: &str) -> Result<(), String> {
        let services: Vec<String> = self.map.iter().filter(|(_, n)| *n == node).map(|(s, _)| s.clone()).collect();
        for service in services {
            self.set(&service, None)?;
        }
        Ok(())
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(id: &str, fp: &str, role: Role) -> Peer {
        Peer { node_id: id.into(), name: format!("{id}'s"), role, fingerprint: fp.into(), address: "192.168.1.5:17975".into(), paired_at_ms: 1, by_itself: false }
    }

    #[test]
    fn settings_are_off_until_switched_on() {
        let dir = std::env::temp_dir().join(format!("oaiy-network-settings-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let s = Settings::load(&dir);
        assert_eq!(s, Settings::default());
        assert_eq!(s.active_role(), Role::Standalone);
        let on = Settings { enabled: true, role: Role::Worker, name: "rig".into(), ..Settings::default() };
        on.save(&dir).unwrap();
        assert_eq!(Settings::load(&dir), on);
        assert_eq!(on.active_role(), Role::Worker);
        assert!(on.lends() && !on.uses(), "a worker's settings of before: it lends itself");
        assert_eq!(Settings { enabled: false, ..on.clone() }.active_role(), Role::Standalone, "a role chosen but Network off: standalone");
        // both: it uses others and lends itself
        let both = Settings { uses: Some(true), ..on };
        assert_eq!((both.active_role(), both.uses(), both.lends()), (Role::Both, true, true));
        let neither = Settings { uses: Some(false), lends: Some(false), ..both };
        assert_eq!(neither.active_role(), Role::Standalone);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A byte that is not text used to read as "no file": the paired computers were gone, and the next save wrote the
    /// store back without them.
    #[test]
    fn a_damaged_file_is_moved_aside_and_said_and_a_good_one_is_never_written_over() {
        let dir = crate::secret_file::testing::TempDir::new("network-damaged");
        for (file, open) in [
            (PEERS_FILE, &(|d: &Path| Peers::open(d).list().len()) as &dyn Fn(&Path) -> usize),
            (DEFAULTS_FILE, &|d: &Path| Defaults::open(d).all().len()),
            (SERVICES_FILE, &|d: &Path| ServiceChoices::open(d).all().len()),
        ] {
            let path = dir.0.join(file);
            std::fs::write(&path, b"[{\"nodeId\":\"caf\xE9\"}]").unwrap();
            assert_eq!(open(&dir.0), 0, "{file}");
            assert_eq!(std::fs::read(path.with_extension("json.corrupt")).unwrap(), b"[{\"nodeId\":\"caf\xE9\"}]", "{file}");
            assert!(!path.exists(), "{file}");
            assert!(problems(&dir.0).iter().any(|p| p.contains(file)), "{file}: {:?}", problems(&dir.0));
        }
        // Saved again, and the next start is a good one: nothing is said any more of a file that reads.
        let mut peers = Peers::open(&dir.0);
        peers.add(peer("a", "fa", Role::Worker)).unwrap();
        assert_eq!(Peers::open(&dir.0).list().len(), 1);
        assert!(!problems(&dir.0).iter().any(|p| p.contains(PEERS_FILE)));
    }

    /// A read that fails for another reason says nothing about the file: it is there, and may be perfectly good.
    #[test]
    fn a_file_that_cannot_be_read_is_not_saved_over() {
        let dir = crate::secret_file::testing::TempDir::new("network-unreadable");
        // (a folder where the file belongs: a read that fails as a held file's does, on every system)
        std::fs::create_dir(dir.0.join(PEERS_FILE)).unwrap();
        std::fs::write(dir.0.join(PEERS_FILE).join("still-here"), b"x").unwrap();
        let mut peers = Peers::open(&dir.0);
        assert!(peers.list().is_empty());
        let refused = peers.add(peer("a", "fa", Role::Worker)).unwrap_err();
        assert!(refused.contains("nothing will be saved over it"), "{refused}");
        assert!(peers.list().is_empty(), "it is not kept in memory as if it were saved");
        assert!(dir.0.join(PEERS_FILE).join("still-here").exists());
        assert_eq!(problems(&dir.0).len(), 1);
        // The same for the settings, which live in a plain value.
        std::fs::create_dir(dir.0.join(SETTINGS_FILE)).unwrap();
        assert_eq!(Settings::load(&dir.0), Settings::default());
        assert!(Settings::default().save(&dir.0).unwrap_err().contains("nothing will be saved over it"));
    }

    /// A file held at start (an antivirus scan, a backup) cost the paired computers and every save, until a restart.
    #[test]
    fn a_file_that_could_not_be_read_is_read_again_and_saving_comes_back() {
        let dir = crate::secret_file::testing::TempDir::new("network-later");
        let path = dir.0.join(PEERS_FILE);
        std::fs::create_dir(&path).unwrap();
        let mut peers = Peers::open(&dir.0);
        assert!(unread(&dir.0));
        // Not time yet: nothing is read, and nothing changes.
        peers.read_again();
        assert!(peers.add(peer("b", "fb", Role::Worker)).is_err());
        // Time, and still held.
        due_now(&path);
        peers.read_again();
        assert!(peers.add(peer("b", "fb", Role::Worker)).is_err());
        assert_eq!(problems(&dir.0).len(), 1, "said once");

        std::fs::remove_dir_all(&path).unwrap();
        std::fs::write(&path, serde_json::to_vec(&[peer("a", "fa", Role::Worker)]).unwrap()).unwrap();
        due_now(&path);
        peers.read_again();
        assert_eq!(peers.list().len(), 1, "they are back with nobody restarting");
        assert!(problems(&dir.0).is_empty() && !unread(&dir.0), "{:?}", problems(&dir.0));
        peers.add(peer("b", "fb", Role::Worker)).unwrap();
        assert_eq!(Peers::open(&dir.0).list().len(), 2, "and what was saved went beside them");
    }

    /// The other three, through the one read they share, and the settings, which come back as a value.
    #[test]
    fn every_file_of_the_store_is_read_again() {
        let dir = crate::secret_file::testing::TempDir::new("network-later-all");
        for file in [SETTINGS_FILE, DEFAULTS_FILE, SERVICES_FILE] {
            std::fs::create_dir(dir.0.join(file)).unwrap();
        }
        let mut defaults = Defaults::open(&dir.0);
        let mut services = ServiceChoices::open(&dir.0);
        let settings = Settings::load(&dir.0);
        assert_eq!(settings, Settings::default());
        assert_eq!(problems(&dir.0).len(), 3);

        let chosen = WorkerDefault { id: "engine:llm:node.a/m".into(), instead_of: None };
        for (file, body) in [
            (SETTINGS_FILE, serde_json::to_vec(&Settings { enabled: true, ..Settings::default() }).unwrap()),
            (DEFAULTS_FILE, serde_json::to_vec(&BTreeMap::from([("llm", &chosen)])).unwrap()),
            (SERVICES_FILE, serde_json::to_vec(&BTreeMap::from([("voice", "a")])).unwrap()),
        ] {
            let path = dir.0.join(file);
            std::fs::remove_dir_all(&path).unwrap();
            std::fs::write(&path, body).unwrap();
            due_now(&path);
        }
        assert!(Settings::read_again(&dir.0).is_some_and(|s| s.enabled));
        defaults.read_again();
        services.read_again();
        assert_eq!(defaults.get("llm"), Some(&chosen));
        assert_eq!(services.get("voice"), Some("a"));
        assert!(problems(&dir.0).is_empty() && !unread(&dir.0));
        assert!(Settings::read_again(&dir.0).is_none(), "nothing to read any more");
    }

    /// PowerShell 5.1's `Out-File` writes UTF-16, and some editors put a mark on UTF-8. Neither is damage.
    #[test]
    fn a_file_in_another_encoding_is_read_and_stays_where_it_is() {
        let dir = crate::secret_file::testing::TempDir::new("network-encodings");
        let path = dir.0.join(PEERS_FILE);
        let text = serde_json::to_string(&[peer("a", "fa", Role::Worker)]).unwrap();
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain(text.encode_utf16().flat_map(u16::to_le_bytes)).collect();
        for body in [[&[0xEF, 0xBB, 0xBF][..], text.as_bytes()].concat(), utf16] {
            std::fs::write(&path, body).unwrap();
            assert_eq!(Peers::open(&dir.0).list().len(), 1);
            assert!(problems(&dir.0).is_empty() && kept_aside(&path).is_empty(), "{:?}", problems(&dir.0));
        }
    }

    /// A second damaged file replaced the copy of the first, and a restart ended the only word of it.
    #[test]
    fn every_damaged_file_is_kept_and_the_notice_lasts_while_the_store_is_empty() {
        let dir = crate::secret_file::testing::TempDir::new("network-damaged-twice");
        let path = dir.0.join(PEERS_FILE);
        std::fs::write(&path, b"[first").unwrap();
        let mut peers = Peers::open(&dir.0);
        assert!(problems(&dir.0)[0].contains("peers.json.corrupt"));

        // A restart with nothing done: the same, with the copy named.
        drop(Peers::open(&dir.0));
        assert!(problems(&dir.0)[0].contains("peers.json.corrupt"), "{:?}", problems(&dir.0));

        // Paired again, and it is a store like any other.
        peers.add(peer("a", "fa", Role::Worker)).unwrap();
        assert!(problems(&dir.0).is_empty(), "{:?}", problems(&dir.0));

        // Unpaired, and the list is empty again with the copy still there.
        peers.remove("a").unwrap();
        drop(Peers::open(&dir.0));
        assert!(problems(&dir.0)[0].contains("peers.json.corrupt"), "{:?}", problems(&dir.0));

        std::fs::write(&path, b"[second").unwrap();
        drop(Peers::open(&dir.0));
        assert_eq!(std::fs::read(path.with_extension("json.corrupt")).unwrap(), b"[first", "the first copy stands");
        assert_eq!(std::fs::read(path.with_extension("json.corrupt.1")).unwrap(), b"[second");
        assert!(problems(&dir.0)[0].contains("peers.json.corrupt.1"), "the newest is named: {:?}", problems(&dir.0));

        // With the copies gone there is nothing to say.
        for aside in kept_aside(&path) {
            std::fs::remove_file(aside).unwrap();
        }
        drop(Peers::open(&dir.0));
        assert!(problems(&dir.0).is_empty(), "{:?}", problems(&dir.0));
    }

    #[test]
    fn peers_are_one_per_node_and_per_certificate_and_go_when_unpaired() {
        let dir = std::env::temp_dir().join(format!("oaiy-network-peers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut peers = Peers::open(&dir);
        peers.add(peer("a", "fa", Role::Worker)).unwrap();
        peers.add(peer("b", "fb", Role::Worker)).unwrap();
        // its node id under another certificate: another computer in its place, refused until it is unpaired; then
        // paired with the new one, the old one no longer lets it in
        assert_eq!(peers.add(peer("a", "fa2", Role::Worker)), Err(TAKEN.to_string()));
        assert_eq!(peers.add(peer("a", "fa2", Role::Primary)), Err(TAKEN.to_string()), "either way");
        peers.remove("a").unwrap();
        peers.add(peer("a", "fa2", Role::Worker)).unwrap();
        let peers = Peers::open(&dir);
        assert_eq!(peers.list().len(), 2);
        assert!(peers.by_fingerprint("fa", Role::Worker).is_none() && peers.by_fingerprint("fa2", Role::Worker).is_some());
        assert!(peers.by_fingerprint("fb", Role::Primary).is_none(), "a worker's certificate is not a primary's");
        let mut peers = peers;
        peers.moved("b", "192.168.1.9:17975").unwrap();
        assert_eq!(Peers::open(&dir).get("b", Role::Worker).unwrap().address, "192.168.1.9:17975");
        // paired the other way too (it uses this computer as well): one pairing each way
        peers.add(peer("a", "fa2", Role::Primary)).unwrap();
        assert!(peers.get("a", Role::Worker).is_some() && peers.get("a", Role::Primary).is_some() && peers.knows("a"));
        assert!(peers.remove("a").unwrap() && !peers.remove("a").unwrap() && !peers.knows("a"), "unpaired both ways");
        assert!(Peers::open(&dir).by_fingerprint("fa2", Role::Worker).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
