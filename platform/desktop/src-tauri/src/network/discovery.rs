//! Workers on the LAN, by mDNS/DNS-SD: a worker advertises `_oaiy-worker._tcp.local.` with its node id, name,
//! protocol, port and in a line what it has (its GPUs, memory and cores: nothing private, no paths, models or keys),
//! and a primary browses for them. Finding a worker
//! grants nothing: it is paired, with the person's approval, before anything of it is used.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::identity::Identity;
use super::{tls, PROTOCOL};

pub const SERVICE: &str = "_oaiy-worker._tcp.local.";
/// Workers found on the LAN kept at once.
const MAX_FOUND: usize = 64;

/// A worker seen on the LAN.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Found {
    pub node_id: String,
    pub name: String,
    /// `ip:port` for each of its local addresses.
    pub addresses: Vec<String>,
    pub protocol: u32,
    /// What it has, as it says it: its GPUs, memory and cores, in a line (`None` from an older OAIY).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<String>,
    pub seen_ms: u64,
}

/// Advertise this worker on `daemon` (its port, id, name and what it has).
pub(crate) fn advertise(daemon: &mdns_sd::ServiceDaemon, id: &Identity, name: &str, port: u16, hardware: &str) -> Result<(), String> {
    let short: String = id.node_id.chars().take(12).collect();
    let host = format!("oaiy-{short}.local.");
    let instance: String = format!("{} ({short})", name.chars().filter(|c| !c.is_control() && *c != '.').take(48).collect::<String>());
    let protocol = PROTOCOL.to_string();
    let props = [("id", id.node_id.as_str()), ("name", name), ("proto", protocol.as_str()), ("ver", env!("CARGO_PKG_VERSION")), ("hw", hardware)];
    let info = mdns_sd::ServiceInfo::new(SERVICE, &instance, &host, "", port, &props[..]).map_err(|e| e.to_string())?.enable_addr_auto();
    daemon.register(info).map_err(|e| e.to_string())
}

/// Browse for workers on `daemon`, keeping what is found (and dropping what leaves) in `found`.
pub(crate) fn browse(daemon: &mdns_sd::ServiceDaemon, found: Arc<Mutex<HashMap<String, Found>>>) -> Result<(), String> {
    let events = daemon.browse(SERVICE).map_err(|e| e.to_string())?;
    std::thread::Builder::new()
        .name("oaiy-network-browse".into())
        .spawn(move || {
            // (names to node ids, for a worker that leaves: its removal names the instance alone)
            let mut by_name: HashMap<String, String> = HashMap::new();
            while let Ok(event) = events.recv() {
                match event {
                    mdns_sd::ServiceEvent::ServiceResolved(s) => {
                        let Some(node_id) = s.get_property_val_str("id").map(|v| v.chars().filter(|c| c.is_ascii_alphanumeric()).take(64).collect::<String>()) else { continue };
                        if node_id.is_empty() {
                            continue;
                        }
                        let port = s.get_port();
                        let addresses: Vec<String> = s
                            .get_addresses()
                            .iter()
                            .map(|a| a.to_ip_addr())
                            .filter(|ip| tls::local_address(*ip))
                            .map(|ip| std::net::SocketAddr::new(ip, port).to_string())
                            .collect();
                        if addresses.is_empty() {
                            continue;
                        }
                        let name: String = s.get_property_val_str("name").unwrap_or("").chars().filter(|c| !c.is_control()).take(80).collect();
                        let protocol = s.get_property_val_str("proto").and_then(|p| p.parse().ok()).unwrap_or(0);
                        let hardware = s.get_property_val_str("hw").map(|h| h.chars().filter(|c| !c.is_control()).take(160).collect::<String>()).filter(|h| !h.is_empty());
                        by_name.insert(s.get_fullname().to_string(), node_id.clone());
                        let entry = Found { node_id: node_id.clone(), name, addresses, protocol, hardware, seen_ms: super::store::now_ms() };
                        let mut found = found.lock().unwrap_or_else(|p| p.into_inner());
                        // (as many as a LAN has: a computer advertising thousands of made-up workers fills no memory)
                        if found.len() < MAX_FOUND || found.contains_key(&node_id) {
                            found.insert(node_id, entry);
                        }
                    }
                    mdns_sd::ServiceEvent::ServiceRemoved(_, fullname) => {
                        if let Some(id) = by_name.remove(&fullname) {
                            found.lock().unwrap_or_else(|p| p.into_inner()).remove(&id);
                        }
                    }
                    mdns_sd::ServiceEvent::SearchStopped(_) => break,
                    _ => {}
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}
