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
pub(crate) mod files;
pub mod machine;
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

pub use primary::{apply_defaults, chat_on_worker, default_chat_entry, remote_services, router, service_target, ServiceTarget};

/// The protocol a worker and a primary speak (`/worker/v1`).
pub const PROTOCOL: u32 = 2;

static SHARED: OnceLock<Network> = OnceLock::new();

/// The desktop's Network, where it has been opened (the engine catalogue asks it for the workers' models).
pub fn shared() -> Option<&'static Network> {
    SHARED.get()
}

/// The name of worker `node_id`, where this computer uses it (it is paired as this computer's worker).
pub fn paired_worker(node_id: &str) -> Option<String> {
    let net = shared()?;
    if !net.settings().uses() {
        return None;
    }
    net.inner().peers().get(node_id, Role::Worker).map(|p| p.name.clone())
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
    /// A primary's engine defaults on its workers, by kind.
    defaults: Mutex<store::Defaults>,
    /// The services a primary uses on its workers (OAIY Voice), by service.
    services: Mutex<store::ServiceChoices>,
    /// This computer's flows and run ledger (a worker runs the flows it shares for its primary): none in some tests.
    bridge: Mutex<Option<Bridge>>,
    /// Workers advertised on the LAN (mDNS), by node id.
    found: Arc<Mutex<HashMap<String, discovery::Found>>>,
    running: Mutex<Running>,
    /// Until when this computer pairs by itself with a computer that asks so (its person or its Agent opened the
    /// window, for minutes; the first such pairing closes it): never saved, closed at a restart.
    pairing_open_until: Mutex<Option<std::time::Instant>>,
    /// Each time the window is opened, closed or taken: a pairing that gives it back does so only in its own turn.
    pairing_turn: std::sync::atomic::AtomicU64,
    /// The last computers paired by themselves here, `(name, when)`, for the Networking page to tell the person.
    paired_by_itself: Mutex<Vec<(String, u64)>>,
    /// Why this computer is not lending itself although set to (its listener could not start), for the page.
    listen_error: Mutex<Option<String>>,
    /// Settings applied one change at a time: two at once would each start a listener and an advertisement.
    applying: tokio::sync::Mutex<()>,
    /// mDNS off (tests): advertising and browsing are not started.
    mdns: bool,
}

/// This computer's flows and the ledger their runs are kept in.
#[derive(Clone)]
pub(crate) struct Bridge {
    pub ledger: crate::bridge::ledger::LedgerHandle,
    pub flows: Arc<crate::bridge::worker::FlowStore>,
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
                defaults: Mutex::new(store::Defaults::open(&dir)),
                services: Mutex::new(store::ServiceChoices::open(&dir)),
                bridge: Mutex::new(None),
                dir,
                ai,
                identity: Mutex::new(None),
                incoming: Mutex::new(Vec::new()),
                outgoing: Mutex::new(Vec::new()),
                nodes: Mutex::new(HashMap::new()),
                untold: Mutex::new(Vec::new()),
                found: Arc::new(Mutex::new(HashMap::new())),
                running: Mutex::new(Running::default()),
                pairing_open_until: Mutex::new(None),
                pairing_turn: std::sync::atomic::AtomicU64::new(0),
                paired_by_itself: Mutex::new(Vec::new()),
                listen_error: Mutex::new(None),
                applying: tokio::sync::Mutex::new(()),
                mdns,
            }),
        }
    }

    /// This computer's flows and run ledger, for the flows a worker shares with its primary.
    pub fn set_bridge(&self, ledger: crate::bridge::ledger::LedgerHandle, flows: Arc<crate::bridge::worker::FlowStore>) {
        *lock(&self.inner.bridge) = Some(Bridge { ledger, flows });
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

    /// Take new settings: kept, and what runs changed to match (lending: the listener and its advertisement; using
    /// others: the heartbeat to them and browsing for them; both, both).
    pub async fn set(&self, settings: Settings) -> Result<(), String> {
        // (an address it cannot listen on is refused before anything is kept or stopped: a port another program has is
        // tried now, the listener's own port left to it)
        if settings.lends() {
            let bind = bind_address(&settings)?;
            let mine = self.listening().is_some_and(|a| a.port() == bind.port() && bind.port() != 0);
            if !mine {
                // (tried as the listener binds it, the address reusable: a port it served a moment ago, waiting out its
                // closed connections, is not taken by another program)
                let probe = if bind.is_ipv6() { tokio::net::TcpSocket::new_v6() } else { tokio::net::TcpSocket::new_v4() };
                probe
                    .and_then(|s| {
                        s.set_reuseaddr(true)?;
                        s.bind(bind)?;
                        s.listen(1).map(drop)
                    })
                    .map_err(|e| format!("this computer cannot lend itself on {bind}: {e} (choose another port)"))?;
            }
        }
        settings.save(&self.inner.dir)?;
        // (pairing by itself is the Agent's, while this computer lends itself: a window open is closed with either)
        if !settings.agent || !settings.lends() {
            self.open_pairing(0);
        }
        *lock(&self.inner.settings) = settings;
        self.apply().await
    }

    /// Unpair `node_id`, both ways, and forget what was chosen of it: its models no longer this computer's defaults,
    /// its services no longer used here (this computer's own again), its last report gone.
    pub(crate) fn forget(&self, node_id: &str) -> Result<(), String> {
        // (all of it forgotten, the rest too where one part could not be written down: that one is said)
        let unpaired = self.inner.peers().remove(node_id);
        let _ = self.inner.defaults().drop_node(node_id);
        let _ = self.inner.services().drop_node(node_id);
        lock(&self.inner.nodes).remove(node_id);
        // (its voice copies removed off the async threads: files)
        let node = node_id.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(rt) => drop(rt.spawn_blocking(move || worker::forget_voices(&node))),
            Err(_) => worker::forget_voices(&node),
        }
        unpaired.map(|_| ())
    }

    /// Open (for `minutes`, at most ten) or close (`0`) the window in which this computer pairs by itself with a
    /// computer that asks so: until when it is open.
    pub(crate) fn open_pairing(&self, minutes: u64) -> Option<u64> {
        let mut open = lock(&self.inner.pairing_open_until);
        *open = (minutes > 0).then(|| std::time::Instant::now() + std::time::Duration::from_secs(minutes.min(10) * 60));
        self.inner.pairing_turn.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        drop(open);
        self.pairing_open_until_ms()
    }

    /// When the window to pair by itself closes (ms since the epoch), while it is open.
    pub(crate) fn pairing_open_until_ms(&self) -> Option<u64> {
        let open = (*lock(&self.inner.pairing_open_until))?;
        let left = open.checked_duration_since(std::time::Instant::now())?;
        Some(store::now_ms() + left.as_millis() as u64)
    }

    /// Take the window to pair by itself, where it is open: closed now (one computer pairs in it), and the turn it was
    /// taken in, to give it back by ([`Network::give_back_pairing_window`]).
    pub(crate) fn take_pairing_window(&self) -> Option<u64> {
        let mut open = lock(&self.inner.pairing_open_until);
        let was = open.is_some_and(|until| until > std::time::Instant::now());
        *open = None;
        let turn = self.inner.pairing_turn.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        was.then_some(turn)
    }

    /// Give the window back (a pairing taken in it that did not go through), unless it was opened or closed since.
    pub(crate) fn give_back_pairing_window(&self, until_ms: Option<u64>, turn: u64) {
        let mut open = lock(&self.inner.pairing_open_until);
        if self.inner.pairing_turn.load(std::sync::atomic::Ordering::SeqCst) != turn {
            return;
        }
        if let Some(ms) = until_ms {
            let left = ms.saturating_sub(store::now_ms());
            *open = Some(std::time::Instant::now() + std::time::Duration::from_millis(left));
        }
    }

    /// A computer paired by itself here: said on the Networking page (the last five).
    pub(crate) fn note_paired_by_itself(&self, name: &str) {
        let mut noted = lock(&self.inner.paired_by_itself);
        noted.push((name.to_string(), store::now_ms()));
        let extra = noted.len().saturating_sub(5);
        noted.drain(..extra);
    }

    /// Keep new settings that change nothing running (whether the Agent may set it up).
    pub(crate) fn keep(&self, settings: Settings) -> Result<(), String> {
        settings.save(&self.inner.dir)?;
        if !settings.agent || !settings.lends() {
            self.open_pairing(0);
        }
        *lock(&self.inner.settings) = settings;
        Ok(())
    }

    /// Start what the settings ask for, stopping what they no longer do.
    pub async fn apply(&self) -> Result<(), String> {
        let _one_at_a_time = self.inner.applying.lock().await;
        // (the listener stopped and its task ended, so its port is free to listen on again: a worker renamed keeps it)
        if let Some(ended) = self.stop_running() {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ended).await;
        }
        let settings = self.settings();
        if !settings.uses() && !settings.lends() {
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
        *lock(&self.inner.listen_error) = None;
        // (a listener that cannot start leaves this computer using others as set: said on the page, not a stop)
        let listening = if settings.lends() {
            let started = match bind_address(&settings) {
                Ok(bind) => worker::listen(self.clone(), &id, bind).await,
                Err(e) => Err(e),
            };
            match started {
                Ok(l) => Some(l),
                Err(e) => {
                    log::warn!("network: this computer is not lending itself: {e}");
                    *lock(&self.inner.listen_error) = Some(e);
                    None
                }
            }
        } else {
            None
        };
        if let Some((addr, stop, task)) = listening {
            log::info!("network: this computer lends itself, listening on {addr}");
            started = daemon();
            if let Some(d) = &started {
                // (what it has, said with its name: from its engines where they answer within a few seconds)
                let read = async { tokio::join!(machine::engines_get(self, "/api/system"), machine::engines_get(self, "/api/resources")) };
                let (system, resources) = tokio::time::timeout(std::time::Duration::from_secs(5), read).await.unwrap_or((Value::Null, Value::Null));
                let summary = tokio::task::spawn_blocking(move || machine::summary(&system, &resources)).await.unwrap_or_default();
                if let Err(e) = discovery::advertise(d, &id, &self.display_name(), addr.port(), &summary) {
                    log::warn!("network: not advertised on the LAN ({e}); another computer adds it by its address");
                }
            }
            lock(&self.inner.running).listener = Some((addr, stop, task));
        }
        if settings.uses() {
            if started.is_none() {
                started = daemon();
            }
            if let Some(d) = &started {
                if let Err(e) = discovery::browse(d, self.inner.found.clone()) {
                    log::warn!("network: computers on the LAN are not looked for ({e}); add them by their address");
                }
            }
            lock(&self.inner.running).heartbeat = Some(primary::heartbeat(self.clone()));
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
            "uses": settings.uses(),
            "lends": settings.lends(),
            "agent": settings.agent,
            "shareModels": settings.share_models != Some(false),
            "pairingOpenUntilMs": self.pairing_open_until_ms(),
            "pairedByItself": lock(&self.inner.paired_by_itself).iter().map(|(name, at)| json!({ "name": name, "at": at })).collect::<Vec<_>>(),
            "listenError": lock(&self.inner.listen_error).clone(),
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

    pub(crate) fn defaults(&self) -> MutexGuard<'_, store::Defaults> {
        lock(&self.defaults)
    }

    pub(crate) fn services(&self) -> MutexGuard<'_, store::ServiceChoices> {
        lock(&self.services)
    }

    pub(crate) fn bridge(&self) -> Option<Bridge> {
        lock(&self.bridge).clone()
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
