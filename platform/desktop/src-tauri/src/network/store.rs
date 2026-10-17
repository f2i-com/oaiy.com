//! What OAIY Network keeps in the data folder (`network/`): its settings (`settings.json`) and the paired computers
//! (`peers.json`, owner-only: a peer's certificate fingerprint is what lets it in).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const SETTINGS_FILE: &str = "settings.json";
pub const PEERS_FILE: &str = "peers.json";

/// The port a worker listens on unless set (the desktop's API is 17972, its voice gateway 17872).
pub const DEFAULT_PORT: u16 = 17975;

/// What this computer is on the network.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Not on it (as with Network switched off).
    #[default]
    Standalone,
    /// Uses paired workers' models, through its own API.
    Primary,
    /// Makes what a paired primary asks of it, on its own engines.
    Worker,
}

/// Network's settings: off, standalone, until the person switches it on and chooses a role.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub enabled: bool,
    pub role: Role,
    /// The name other computers show for this one (the computer's own name where empty).
    pub name: String,
    /// The port a worker listens on.
    pub port: u16,
    /// The address a worker listens on (one of this computer's LAN addresses), or every interface where empty.
    pub bind: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { enabled: false, role: Role::Standalone, name: String::new(), port: DEFAULT_PORT, bind: String::new() }
    }
}

impl Settings {
    pub fn load(dir: &Path) -> Settings {
        std::fs::read_to_string(dir.join(SETTINGS_FILE)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
        crate::secret_file::write(&dir.join(SETTINGS_FILE), serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
    }

    /// The role in effect: none while Network is off.
    pub fn active_role(&self) -> Role {
        if self.enabled { self.role } else { Role::Standalone }
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
        let list = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        Peers { path, list }
    }

    pub fn list(&self) -> &[Peer] {
        &self.list
    }

    pub fn get(&self, node_id: &str) -> Option<&Peer> {
        self.list.iter().find(|p| p.node_id == node_id)
    }

    /// The paired computer with this certificate (a connection's own), of this role.
    pub fn by_fingerprint(&self, fingerprint: &str, role: Role) -> Option<&Peer> {
        self.list.iter().find(|p| p.fingerprint == fingerprint && p.role == role)
    }

    /// Pair (or pair again) with `peer`: one entry per node id, and per certificate. In effect once it is kept: a pairing
    /// that could not be written down lets nothing in.
    pub fn add(&mut self, peer: Peer) -> Result<(), String> {
        let mut next: Vec<Peer> = self.list.iter().filter(|p| p.node_id != peer.node_id && p.fingerprint != peer.fingerprint).cloned().collect();
        next.push(peer);
        save(&self.path, &next)?;
        self.list = next;
        Ok(())
    }

    /// Unpair: its connections are refused from now on, even where that could not be written down (then said so).
    pub fn remove(&mut self, node_id: &str) -> Result<bool, String> {
        let before = self.list.len();
        self.list.retain(|p| p.node_id != node_id);
        let removed = self.list.len() != before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// A worker's new address (it moved: another DHCP lease), kept for the next start.
    pub fn moved(&mut self, node_id: &str, address: &str) -> Result<(), String> {
        match self.list.iter_mut().find(|p| p.node_id == node_id) {
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

fn save(path: &Path, list: &[Peer]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        crate::secret_file::create_private_dir(dir).map_err(|e| e.to_string())?;
    }
    crate::secret_file::write(path, serde_json::to_vec_pretty(list).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(id: &str, fp: &str, role: Role) -> Peer {
        Peer { node_id: id.into(), name: format!("{id}'s"), role, fingerprint: fp.into(), address: "192.168.1.5:17975".into(), paired_at_ms: 1 }
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
        assert_eq!(Settings { enabled: false, ..on }.active_role(), Role::Standalone, "a role chosen but Network off: standalone");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peers_are_one_per_node_and_per_certificate_and_go_when_unpaired() {
        let dir = std::env::temp_dir().join(format!("oaiy-network-peers-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut peers = Peers::open(&dir);
        peers.add(peer("a", "fa", Role::Worker)).unwrap();
        peers.add(peer("b", "fb", Role::Worker)).unwrap();
        // paired again with a new certificate: the old one no longer lets it in
        peers.add(peer("a", "fa2", Role::Worker)).unwrap();
        let peers = Peers::open(&dir);
        assert_eq!(peers.list().len(), 2);
        assert!(peers.by_fingerprint("fa", Role::Worker).is_none() && peers.by_fingerprint("fa2", Role::Worker).is_some());
        assert!(peers.by_fingerprint("fb", Role::Primary).is_none(), "a worker's certificate is not a primary's");
        let mut peers = peers;
        peers.moved("b", "192.168.1.9:17975").unwrap();
        assert_eq!(Peers::open(&dir).get("b").unwrap().address, "192.168.1.9:17975");
        assert!(peers.remove("a").unwrap() && !peers.remove("a").unwrap());
        assert!(Peers::open(&dir).by_fingerprint("fa2", Role::Worker).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
