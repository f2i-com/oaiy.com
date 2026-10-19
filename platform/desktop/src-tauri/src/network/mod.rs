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

pub use primary::{apply_defaults, chat_on_worker, choose_default, default_chat_entry, remote_services, router, service_target, ServiceTarget};

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
    /// The pairing codes made here lately (a primary told its random value), when and for which address: a computer
    /// asking again and again for a code of its choosing is held to a few tries.
    pairing_tries: Mutex<Vec<(std::time::Instant, std::net::IpAddr)>>,
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
    /// The last computers paired by themselves here, `(name, when, code)`, for the Networking page to tell the person
    /// (with the code, which the other computer's person compares before it pairs on its side).
    paired_by_itself: Mutex<Vec<(String, u64, String)>>,
    /// Why this computer is not lending itself although set to (its listener could not start), for the page.
    listen_error: Mutex<Option<String>>,
    /// The primaries approved here whose own person has not said yes yet, by certificate: the node id and when. One
    /// not heard from (its heartbeat) within the pairing's time is unpaired again here (`worker::drop_unconfirmed`): it
    /// stopped or restarted before its person answered, and would otherwise stay paired on this side alone.
    pub(crate) unconfirmed: Mutex<HashMap<String, (String, std::time::Instant)>>,
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
    /// The address the listener was asked for (port 0: any): settings saved again with the same one keep it, and the
    /// computers using this one their streams, calls and runs.
    asked: Option<std::net::SocketAddr>,
    /// A primary's heartbeat: the sender that stops it.
    heartbeat: Option<tokio::sync::watch::Sender<bool>>,
    mdns: Option<mdns_sd::ServiceDaemon>,
    /// This computer's advertisement on the LAN: its full name there, and the name it says (renamed: said again).
    advertised: Option<(String, String)>,
    /// Looking for computers lending themselves on the LAN (this computer uses others).
    browsing: bool,
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
                pairing_tries: Mutex::new(Vec::new()),
                outgoing: Mutex::new(Vec::new()),
                nodes: Mutex::new(HashMap::new()),
                untold: Mutex::new(Vec::new()),
                found: Arc::new(Mutex::new(HashMap::new())),
                running: Mutex::new(Running::default()),
                pairing_open_until: Mutex::new(None),
                pairing_turn: std::sync::atomic::AtomicU64::new(0),
                paired_by_itself: Mutex::new(Vec::new()),
                listen_error: Mutex::new(None),
                unconfirmed: Mutex::new(HashMap::new()),
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
                // (tried as the listener binds it: the address reusable off Windows, so a port it served a moment ago,
                // waiting out its closed connections, is not taken by another program; never on Windows, where that
                // would bind a port another program holds that lets it be reused too, and the listener, which does not,
                // would then fail. Bound alone, never listening: the probe takes no connection meant for that program.)
                let probe = if bind.is_ipv6() { tokio::net::TcpSocket::new_v6() } else { tokio::net::TcpSocket::new_v4() };
                probe
                    .and_then(|s| {
                        if !cfg!(windows) {
                            s.set_reuseaddr(true)?;
                        }
                        s.bind(bind)
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
        // (its voice copies are kept under its certificate, as one that used this computer: read before it goes)
        let used_this = self.inner.peers().get(node_id, Role::Primary).map(|p| p.fingerprint.clone());
        let unpaired = self.inner.peers().remove(node_id);
        let _ = self.inner.defaults().drop_node(node_id);
        let _ = self.inner.services().drop_node(node_id);
        lock(&self.inner.nodes).remove(node_id);
        // (its voice copies removed off the async threads: files)
        if let Some(fingerprint) = used_this {
            match tokio::runtime::Handle::try_current() {
                Ok(rt) => drop(rt.spawn_blocking(move || worker::forget_voices(&fingerprint))),
                Err(_) => worker::forget_voices(&fingerprint),
            }
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

    /// A computer paired by itself here: said on the Networking page (the last five), with the code its person compares
    /// before that computer pairs on its side.
    pub(crate) fn note_paired_by_itself(&self, name: &str, code: &str) {
        let mut noted = lock(&self.inner.paired_by_itself);
        noted.push((name.to_string(), store::now_ms(), code.to_string()));
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

    /// Start what the settings ask for, stopping what they no longer do, and nothing else: the listener is started
    /// again only where it is to listen elsewhere (another port or address) or lending is switched on or off, so a
    /// computer renamed, or one that starts or stops using others, keeps the connections of the computers using it
    /// (their streamed chats, calls' speech and flow runs followed).
    pub async fn apply(&self) -> Result<(), String> {
        let _one_at_a_time = self.inner.applying.lock().await;
        let settings = self.settings();
        if !settings.uses() && !settings.lends() {
            // (the listener stopped and its task ended, so its port is free to listen on again)
            if let Some(ended) = self.stop_running() {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ended).await;
            }
            return Ok(());
        }
        let id = self.identity()?;
        let asked = settings.lends().then(|| bind_address(&settings));
        // (where it is asked to listen still: the same address, or the port it took where any was asked for before)
        let keep = {
            let mut running = lock(&self.inner.running);
            let at = running.listener.as_ref().map(|(a, _, _)| *a);
            let same = match (&asked, at, running.asked) {
                (Some(Ok(bind)), Some(at), Some(was)) => *bind == was || (bind.ip() == was.ip() && bind.port() == at.port()),
                _ => false,
            };
            if let (true, Some(Ok(bind))) = (same, &asked) {
                running.asked = Some(*bind);
            }
            same
        };
        let mut started_again = false;
        if !keep {
            if let Some(ended) = self.stop_listener() {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(5), ended).await;
            }
            *lock(&self.inner.listen_error) = None;
            // (a listener that cannot start leaves this computer using others as set: said on the page, not a stop)
            if let Some(asked) = asked {
                let started = match asked {
                    Ok(bind) => worker::listen(self.clone(), &id, bind).await.map(|l| (bind, l)),
                    Err(e) => Err(e),
                };
                match started {
                    Ok((bind, (addr, stop, task))) => {
                        log::info!("network: this computer lends itself, listening on {addr}");
                        let mut running = lock(&self.inner.running);
                        running.listener = Some((addr, stop, task));
                        running.asked = Some(bind);
                        started_again = true;
                    }
                    Err(e) => {
                        log::warn!("network: this computer is not lending itself: {e}");
                        *lock(&self.inner.listen_error) = Some(e);
                    }
                }
            }
        }
        // (the heartbeat to the computers it uses: with Use other computers alone)
        {
            let mut running = lock(&self.inner.running);
            match (settings.uses(), running.heartbeat.is_some()) {
                (true, false) => running.heartbeat = Some(primary::heartbeat(self.clone())),
                (false, true) => {
                    if let Some(stop) = running.heartbeat.take() {
                        let _ = stop.send(true);
                    }
                }
                _ => {}
            }
        }
        self.apply_discovery(&id, &settings, started_again).await;
        Ok(())
    }

    /// Advertising this computer and looking for others on the LAN (mDNS) as the settings ask: advertised while it
    /// lends itself, again where its listener started again or its name changed; looking while it uses others; the
    /// daemon stopped when neither is wanted.
    async fn apply_discovery(&self, id: &Identity, settings: &Settings, started_again: bool) {
        if !self.inner.mdns {
            return;
        }
        let listening = self.listening();
        if listening.is_none() && !settings.uses() {
            let mut running = lock(&self.inner.running);
            if let Some(d) = running.mdns.take() {
                let _ = d.shutdown();
            }
            running.advertised = None;
            running.browsing = false;
            drop(running);
            lock(&self.inner.found).clear();
            return;
        }
        let daemon = {
            let mut running = lock(&self.inner.running);
            if running.mdns.is_none() {
                running.mdns = mdns_sd::ServiceDaemon::new().map_err(|e| log::warn!("network: mDNS could not start ({e}); computers are added by their address")).ok();
                running.advertised = None;
                running.browsing = false;
            }
            running.mdns.clone()
        };
        let Some(d) = daemon else { return };
        let name = self.display_name();
        let (advertised, browsing) = {
            let running = lock(&self.inner.running);
            (running.advertised.clone(), running.browsing)
        };
        match listening {
            Some(addr) if started_again || advertised.as_ref().is_none_or(|(_, said)| *said != name) => {
                if let Some((full, _)) = &advertised {
                    let _ = d.unregister(full);
                }
                // (what it has, said with its name: from its engines where they answer within a few seconds)
                let read = async { tokio::join!(machine::engines_get(self, "/api/system"), machine::engines_get(self, "/api/resources")) };
                let (system, resources) = tokio::time::timeout(std::time::Duration::from_secs(5), read).await.unwrap_or((Value::Null, Value::Null));
                let summary = tokio::task::spawn_blocking(move || machine::summary(&system, &resources)).await.unwrap_or_default();
                let made = discovery::advertise(&d, id, &name, addr.port(), &summary)
                    .map_err(|e| log::warn!("network: not advertised on the LAN ({e}); another computer adds it by its address"))
                    .ok();
                lock(&self.inner.running).advertised = made.map(|full| (full, name));
            }
            Some(_) => {}
            None => {
                if let Some((full, _)) = &advertised {
                    let _ = d.unregister(full);
                }
                lock(&self.inner.running).advertised = None;
            }
        }
        match (settings.uses(), browsing) {
            (true, false) => match discovery::browse(&d, self.inner.found.clone()) {
                Ok(()) => lock(&self.inner.running).browsing = true,
                Err(e) => log::warn!("network: computers on the LAN are not looked for ({e}); add them by their address"),
            },
            (false, true) => {
                let _ = d.stop_browse(discovery::SERVICE);
                lock(&self.inner.running).browsing = false;
                lock(&self.inner.found).clear();
            }
            _ => {}
        }
    }

    /// Stop everything running (Network off, or about to start again).
    pub fn stop(&self) {
        drop(self.stop_running());
    }

    /// As [`Network::stop`], answering the listener's task, which ends once its port is free.
    fn stop_running(&self) -> Option<tokio::task::JoinHandle<()>> {
        let task = self.stop_listener();
        let mut running = lock(&self.inner.running);
        if let Some(stop) = running.heartbeat.take() {
            let _ = stop.send(true);
        }
        if let Some(d) = running.mdns.take() {
            let _ = d.shutdown();
        }
        running.advertised = None;
        running.browsing = false;
        drop(running);
        lock(&self.inner.found).clear();
        task
    }

    /// The listener stopped (its connections with it), answering its task, which ends once its port is free.
    fn stop_listener(&self) -> Option<tokio::task::JoinHandle<()>> {
        let mut running = lock(&self.inner.running);
        running.asked = None;
        let (_, stop, task) = running.listener.take()?;
        let _ = stop.send(true);
        Some(task)
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
            "pairedByItself": lock(&self.inner.paired_by_itself).iter().map(|(name, at, code)| json!({ "name": name, "at": at, "code": code })).collect::<Vec<_>>(),
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
        return Err(format!("{ip} is not a local network address: a computer lends itself on the LAN alone"));
    }
    Ok(std::net::SocketAddr::new(ip, settings.port))
}

/// This computer's LAN addresses, as `ip:port` (what a primary is told to type where discovery is blocked): its
/// adapters that are up (Windows lists those that are down, or not there, with their last address too), the one its
/// default route leaves by first, where the system says which.
fn lan_addresses(port: u16) -> Vec<String> {
    let adapters: Vec<Adapter> = if_addrs::get_if_addrs().unwrap_or_default().iter().map(|i| Adapter { ip: i.ip(), up: i.is_oper_up(), point_to_point: i.is_p2p() }).collect();
    offered(&adapters, default_route_address())
        .into_iter()
        .map(|ip| std::net::SocketAddr::new(ip, port).to_string())
        .collect()
}

/// One of this computer's addresses, as its adapter is: up, and a point-to-point link (a VPN's tunnel, `utun` on a Mac).
struct Adapter {
    ip: std::net::IpAddr,
    up: bool,
    point_to_point: bool,
}

/// The addresses offered of `adapters`, as [`typeable`] orders them (`first`, the default route's, first): those that
/// are up, where the system says so, and none of a point-to-point link's. A VPN that sends all traffic through it is
/// where the default route leaves by, and its address was offered first, though no computer on the LAN reaches it.
fn offered(adapters: &[Adapter], first: Option<std::net::IpAddr>) -> Vec<std::net::IpAddr> {
    let lan: Vec<&Adapter> = adapters.iter().filter(|a| !a.point_to_point).collect();
    let up: Vec<std::net::IpAddr> = lan.iter().filter(|a| a.up).map(|a| a.ip).collect();
    // (none said to be up: a system that does not say, and its addresses as they are)
    let ips = if up.is_empty() { lan.iter().map(|a| a.ip).collect() } else { up };
    typeable(ips, first)
}

/// The address this computer's default route leaves by, the one other computers on its network most likely reach
/// (rather than a virtual adapter's, a VM's or a VPN's, which come first in adapter order on Windows): what the system
/// chooses for a UDP socket aimed past the LAN, at a documentation address. A UDP socket connected sends nothing.
fn default_route_address() -> Option<std::net::IpAddr> {
    let socket = std::net::UdpSocket::bind((std::net::Ipv4Addr::UNSPECIFIED, 0)).ok()?;
    socket.connect((std::net::Ipv4Addr::new(192, 0, 2, 1), 9)).ok()?;
    socket.local_addr().ok().map(|a| a.ip()).filter(|ip| !ip.is_unspecified())
}

/// The addresses a person can type on another computer, each once, IPv4 first and `first` (the default route's)
/// before the rest: not loopback, and not an IPv6 link-local address (one is on every interface, and it needs its
/// interface named to be reached).
fn typeable(ips: Vec<std::net::IpAddr>, first: Option<std::net::IpAddr>) -> Vec<std::net::IpAddr> {
    let mut out: Vec<std::net::IpAddr> = Vec::new();
    for ip in ips {
        let link_local_v6 = matches!(ip, std::net::IpAddr::V6(v6) if (v6.segments()[0] & 0xffc0) == 0xfe80);
        if ip.is_loopback() || link_local_v6 || !tls::local_address(ip) || out.contains(&ip) {
            continue;
        }
        out.push(ip);
    }
    out.sort_by_key(|ip| (Some(*ip) != first, ip.is_ipv6()));
    out
}

/// This computer's name, asked of the system once (a stranger's `health` must not start a process each time).
fn hostname() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        std::env::var("OAIY_NETWORK_NAME")
            .ok()
            .or_else(system_name)
            .map(|s| s.trim().trim_end_matches(".local").to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "This computer".to_string())
    })
    .clone()
}

/// The name the system gives this computer, `hostname`'s answer, asked with no console window (the app's window has
/// no console, so Windows would flash a new one at every launch). Not Windows' `COMPUTERNAME`: that is the older
/// NetBIOS name, upper-case and cut to 15 characters, and computers would show another name than before.
fn system_name() -> Option<String> {
    use crate::HiddenCommand as _;
    std::process::Command::new("hostname").pipe_hidden().output().ok().and_then(|o| String::from_utf8(o.stdout).ok())
}
