//! OAIY Network (docs/NETWORK.md): one OAIY (the primary) using the models of other OAIY installations on the same LAN
//! (its workers), each run by the worker's own engines and reached through the primary's own API.
//!
//! - [`identity`]: this computer's node id, key and self-signed certificate.
//! - [`store`]: the settings (off until switched on) and the paired computers, pinned by certificate.
//! - [`tls`]: mutual TLS with no authority, each side checking the other's pinned certificate.
//! - `worker`: a worker's listener on the LAN (`/worker/v1/*`): pairing, its capabilities, and its engines' routes
//!   for a paired primary alone.
//! - `primary`: the desktop's `/api/network/*` routes, the workers' heartbeat, the remote models in the engine
//!   catalogue, and the routes that carry their requests and files.
//! - `discovery`: workers advertised and found by mDNS (`_oaiy-worker._tcp.local.`), an address typed in otherwise.
//!
//! Off, nothing here listens, advertises or calls out, and the desktop behaves as it did before.

pub mod identity;
pub mod store;
pub mod tls;

mod discovery;
mod primary;
mod worker;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use serde_json::{json, Value};

use self::identity::Identity;
use self::store::{Peers, Role, Settings};

pub use primary::{remote_services, router};

/// The protocol a worker and a primary speak (`/worker/v1`).
pub const PROTOCOL: u32 = 2;

static SHARED: OnceLock<Network> = OnceLock::new();

/// The desktop's Network, where it has been opened (the engine catalogue asks it for the workers' models).
pub fn shared() -> Option<&'static Network> {
    SHARED.get()
}

/// The name of worker `node_id`, where it is paired with this computer as a primary.
pub fn paired_worker(node_id: &str) -> Option<String> {
    let net = shared()?;
    if net.settings().active_role() != store::Role::Primary {
        return None;
    }
    net.inner().peers().get(node_id).filter(|p| p.role == store::Role::Worker).map(|p| p.name.clone())
}

#[derive(Clone)]
pub struct Network {
    inner: Arc<Inner>,
}

pub(crate) struct Inner {
    dir: PathBuf,
    /// This computer's engines (a worker's are what it serves; a primary's are left as they are).
    ai: crate::ai::AiState,
    settings: Mutex<Settings>,
    identity: Mutex<Option<Identity>>,
    peers: Mutex<Peers>,
    /// A worker's pairing requests, waiting for the person to approve them here.
    incoming: Mutex<Vec<worker::Incoming>>,
    /// A primary's pairings in progress, waiting for the worker's approval.
    outgoing: Mutex<Vec<primary::Outgoing>>,
    /// What a primary last heard from each paired worker.
    nodes: Mutex<HashMap<String, primary::NodeState>>,
    /// Workers unpaired here that did not answer then: told when they do (while this computer runs).
    untold: Mutex<Vec<store::Peer>>,
    /// Workers advertised on the LAN (mDNS), by node id.
    found: Arc<Mutex<HashMap<String, discovery::Found>>>,
    running: Mutex<Running>,
    /// Settings applied one change at a time: two at once would each start a listener and an advertisement.
    applying: tokio::sync::Mutex<()>,
    /// mDNS off (tests): advertising and browsing are not started.
    mdns: bool,
}

/// What is running for the role in effect.
#[derive(Default)]
struct Running {
    /// A worker's listener: where it listens, the sender that stops it, and its task (ended: its port is free).
    listener: Option<(std::net::SocketAddr, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>)>,
    /// A primary's heartbeat: the sender that stops it.
    heartbeat: Option<tokio::sync::watch::Sender<bool>>,
    mdns: Option<mdns_sd::ServiceDaemon>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Network {
    /// Network as the data folder keeps it (`<data>/network`), nothing started yet ([`Network::apply`] does).
    /// OAIY Network of this desktop, with discovery on the LAN (mDNS) unless `OAIY_NETWORK_DISCOVERY=off`: a network
    /// where multicast is unwanted pairs by address.
    pub fn open(data_dir: &Path, ai: crate::ai::AiState) -> Network {
        let mdns = !matches!(std::env::var("OAIY_NETWORK_DISCOVERY").as_deref().map(str::trim), Ok("off" | "0" | "false"));
        Network::open_with(data_dir, ai, mdns)
    }

    pub(crate) fn open_with(data_dir: &Path, ai: crate::ai::AiState, mdns: bool) -> Network {
        let dir = data_dir.join("network");
        Network {
            inner: Arc::new(Inner {
                settings: Mutex::new(Settings::load(&dir)),
                peers: Mutex::new(Peers::open(&dir)),
                dir,
                ai,
                identity: Mutex::new(None),
                incoming: Mutex::new(Vec::new()),
                outgoing: Mutex::new(Vec::new()),
                nodes: Mutex::new(HashMap::new()),
                untold: Mutex::new(Vec::new()),
                found: Arc::new(Mutex::new(HashMap::new())),
                running: Mutex::new(Running::default()),
                applying: tokio::sync::Mutex::new(()),
                mdns,
            }),
        }
    }

    /// This Network as the desktop's ([`shared`]); the first opened is kept.
    pub fn share(&self) {
        let _ = SHARED.set(self.clone());
    }

    pub fn settings(&self) -> Settings {
        lock(&self.inner.settings).clone()
    }

    pub(crate) fn inner(&self) -> &Inner {
        &self.inner
    }

    /// This computer's identity, made the first time it is asked for (Network switched on).
    pub fn identity(&self) -> Result<Identity, String> {
        let mut id = lock(&self.inner.identity);
        if let Some(id) = id.as_ref() {
            return Ok(id.clone());
        }
        let made = Identity::load_or_create(&self.inner.dir)?;
        *id = Some(made.clone());
        Ok(made)
    }

    /// The name other computers show for this one.
    pub fn display_name(&self) -> String {
        let name = lock(&self.inner.settings).name.trim().to_string();
        if !name.is_empty() {
            return name;
        }
        hostname()
    }

    /// Take new settings: kept, and what runs changed to match (a worker's listener and advertisement, a primary's
    /// heartbeat and browsing).
    pub async fn set(&self, settings: Settings) -> Result<(), String> {
        // (an address a worker cannot listen on is refused before anything is kept or stopped)
        if settings.active_role() == Role::Worker {
            bind_address(&settings)?;
        }
        settings.save(&self.inner.dir)?;
        *lock(&self.inner.settings) = settings;
        self.apply().await
    }

    /// Start what the settings ask for, stopping what they no longer do.
    pub async fn apply(&self) -> Result<(), String> {
        let _one_at_a_time = self.inner.applying.lock().await;
        // (the listener stopped and its task ended, so its port is free to listen on again: a worker renamed keeps it)
        if let Some(ended) = self.stop_running() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ended).await;
        }
        let settings = self.settings();
        let role = settings.active_role();
        if role == Role::Standalone {
            return Ok(());
        }
        let id = self.identity()?;
        // (started once what it serves is: nothing is left running, or advertised, by a listener that failed)
        let daemon = || {
            if !self.inner.mdns {
                return None;
            }
            mdns_sd::ServiceDaemon::new()
                .map_err(|e| log::warn!("network: mDNS could not start ({e}); workers are added by their address"))
                .ok()
        };
        let mut started = None;
        match role {
            Role::Worker => {
                let bind = bind_address(&settings)?;
                let (addr, stop, task) = worker::listen(self.clone(), &id, bind).await?;
                log::info!("network: this computer is a worker, listening on {addr}");
                started = daemon();
                if let Some(d) = &started {
                    if let Err(e) = discovery::advertise(d, &id, &self.display_name(), addr.port()) {
                        log::warn!("network: not advertised on the LAN ({e}); a primary adds it by its address");
                    }
                }
                lock(&self.inner.running).listener = Some((addr, stop, task));
            }
            Role::Primary => {
                started = daemon();
                if let Some(d) = &started {
                    if let Err(e) = discovery::browse(d, self.inner.found.clone()) {
                        log::warn!("network: workers on the LAN are not looked for ({e}); add them by their address");
                    }
                }
                lock(&self.inner.running).heartbeat = Some(primary::heartbeat(self.clone()));
            }
            Role::Standalone => {}
        }
        lock(&self.inner.running).mdns = started;
        Ok(())
    }

    /// Stop everything running (Network off, or about to start again).
    pub fn stop(&self) {
        drop(self.stop_running());
    }

    /// As [`Network::stop`], answering the listener's task, which ends once its port is free.
    fn stop_running(&self) -> Option<tokio::task::JoinHandle<()>> {
        let mut running = lock(&self.inner.running);
        let mut task = None;
        if let Some((_, stop, t)) = running.listener.take() {
            let _ = stop.send(true);
            task = Some(t);
        }
        if let Some(stop) = running.heartbeat.take() {
            let _ = stop.send(true);
        }
        if let Some(d) = running.mdns.take() {
            let _ = d.shutdown();
        }
        lock(&self.inner.found).clear();
        task
    }

    /// Where a worker listens (none unless it is one, switched on).
    pub fn listening(&self) -> Option<std::net::SocketAddr> {
        lock(&self.inner.running).listener.as_ref().map(|(a, _, _)| *a)
    }

    /// What the Networking page shows of this computer.
    pub fn status(&self) -> Value {
        let settings = self.settings();
        let id = if settings.enabled { self.identity().ok() } else { lock(&self.inner.identity).clone() };
        let listening = self.listening();
        json!({
            "enabled": settings.enabled,
            "role": settings.active_role(),
            "settings": settings,
            "nodeId": id.as_ref().map(|i| i.node_id.clone()),
            "name": self.display_name(),
            "fingerprint": id.as_ref().map(|i| identity::short(&i.fingerprint)),
            "listening": listening.map(|a| a.to_string()),
            "addresses": listening.map(|a| lan_addresses(a.port())).unwrap_or_default(),
            "discovery": self.inner.mdns && settings.enabled,
            "protocol": PROTOCOL,
        })
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let running = self.running.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some((_, stop, _)) = running.listener.take() {
            let _ = stop.send(true);
        }
        if let Some(stop) = running.heartbeat.take() {
            let _ = stop.send(true);
        }
        if let Some(d) = running.mdns.take() {
            let _ = d.shutdown();
        }
    }
}

impl Inner {
    pub(crate) fn peers(&self) -> MutexGuard<'_, Peers> {
        lock(&self.peers)
    }
}

/// The address a worker listens on: the one chosen (one of this computer's own), else every interface.
fn bind_address(settings: &Settings) -> Result<std::net::SocketAddr, String> {
    let ip: std::net::IpAddr = if settings.bind.trim().is_empty() {
        std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED)
    } else {
        settings.bind.trim().parse().map_err(|_| format!("{} is not an address of this computer", settings.bind.trim()))?
    };
    if !ip.is_unspecified() && !tls::local_address(ip) {
        return Err(format!("{ip} is not a local network address: a worker listens on the LAN alone"));
    }
    Ok(std::net::SocketAddr::new(ip, settings.port))
}

/// This computer's LAN addresses, as `ip:port` (what a primary is told to type where discovery is blocked).
fn lan_addresses(port: u16) -> Vec<String> {
    let ips: Vec<std::net::IpAddr> = if_addrs::get_if_addrs().unwrap_or_default().into_iter().map(|i| i.ip()).collect();
    typeable(ips)
        .into_iter()
        .map(|ip| std::net::SocketAddr::new(ip, port).to_string())
        .collect()
}

/// The addresses a person can type on another computer, each once, IPv4 first: not loopback, and not an IPv6 link-local
/// address (one is on every interface, and it needs its interface named to be reached).
fn typeable(ips: Vec<std::net::IpAddr>) -> Vec<std::net::IpAddr> {
    let mut out: Vec<std::net::IpAddr> = Vec::new();
    for ip in ips {
        let link_local_v6 = matches!(ip, std::net::IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80);
        if ip.is_loopback() || link_local_v6 || !tls::local_address(ip) || out.contains(&ip) {
            continue;
        }
        out.push(ip);
    }
    out.sort_by_key(|ip| ip.is_ipv6());
    out
}

/// This computer's name, asked of the system once (a stranger's `health` must not start a process each time).
fn hostname() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("OAIY_NETWORK_NAME")
            .ok()
            .or_else(|| std::process::Command::new("hostname").output().ok().and_then(|o| String::from_utf8(o.stdout).ok()))
            .map(|s| s.trim().trim_end_matches(".local").to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "This computer".to_string())
    })
    .clone()
}
