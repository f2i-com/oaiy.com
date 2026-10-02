//! A client, a stub relay and a fake clock, wired together, and the small things the scenario tests keep needing.

#![allow(dead_code)]

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use oaiy_crypto::zeroize::Secret;
use oaiy_relay_core::client::{
    enrol_and_store, Cancel, ClientConfig, Clock, Event, LoopEnd, MemoryPollStore, MemoryProfileStore, MemorySecretStore, PollHandle, PollLoop,
    PollLoopConfig, RecordingSink, RelayClient, RelayProfile, SecretStore, SECRET_TOKEN,
};
use oaiy_relay_core::enrol::{EnrolmentKey, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::testing::stub::{StubConfig, StubRelay};
use oaiy_relay_core::testing::{FakeClock, SeededRng};
use oaiy_relay_core::url::RelayUrl;

pub const T0: i64 = 1_790_000_000;

/// A stub whose polls do not wait (`wait.default` 0), so that a loop's pauses are the only thing that paces it.
pub fn quick() -> StubConfig {
    StubConfig { wait_default: 0, wait_max: 0, ..Default::default() }
}

pub struct Env {
    pub clock: Arc<FakeClock>,
    pub stub: StubRelay,
    pub client: Arc<RelayClient>,
    pub secrets: Arc<MemorySecretStore>,
    pub profiles: Arc<MemoryProfileStore>,
}

pub fn env(cfg: StubConfig) -> Env {
    let clock = Arc::new(FakeClock::new(T0));
    let (c1, c2) = (clock.clone(), clock.clone());
    let url = cfg.public_url.clone();
    let stub = StubRelay::with_clocks(cfg, move || c1.unix_now(), move || c2.monotonic());
    let client = client_for(&stub, &clock, &url, 7);
    Env { clock, stub, client, secrets: Arc::new(MemorySecretStore::new()), profiles: Arc::new(MemoryProfileStore::new()) }
}

pub fn client_for(stub: &StubRelay, clock: &Arc<FakeClock>, url: &str, seed: u64) -> Arc<RelayClient> {
    Arc::new(RelayClient::new(
        RelayUrl::parse(url).unwrap(),
        Some(stub.relay_thumbprint()),
        Arc::new(stub.clone()),
        clock.clone(),
        Box::new(SeededRng::new(seed)),
        ClientConfig::default(),
    ))
}

impl Env {
    /// Enrols a desktop through the client, as DK-03 does: returns its token and profile.
    pub fn enrol_desktop(&self) -> (Token, RelayProfile) {
        let uri = self.stub.mint_enrolment_key(Role::Desktop, 3600);
        let key = EnrolmentKey::parse(&uri).unwrap();
        let ed = Signer::generate().unwrap().verify_key();
        let x = X25519Secret::generate().unwrap().public_key();
        let profile =
            enrol_and_store(&self.client, &key, "Front desk PC", &ed, &x, &*self.secrets, &*self.profiles, &Cancel::new()).expect("enrolment");
        let token = Token::parse(std::str::from_utf8(&self.secrets.get(SECRET_TOKEN).unwrap().unwrap()).unwrap()).unwrap();
        (token, profile)
    }

    /// A second client of the same relay (a provider, a phone), proved and ready.
    pub fn other_client(&self, seed: u64) -> Arc<RelayClient> {
        let c = client_for(&self.stub, &self.clock, &self.stub.public_url(), seed);
        c.prove(&Cancel::new()).expect("proof");
        c
    }

    /// A provider device and its token.
    pub fn provider(&self) -> (String, Token) {
        let (id, token) = self.stub.create_device("provider", "FormLogic", None, None, &[], None);
        (id, Token::parse(&token).unwrap())
    }
}

/// Runs a loop on a thread; `finish` stops it and returns how it ended and its store.
pub struct Running {
    pub handle: PollHandle,
    pub sink: Arc<RecordingSink>,
    thread: Option<JoinHandle<(LoopEnd, MemoryPollStore)>>,
}

pub fn run_loop(client: &Arc<RelayClient>, token: &Token, store: MemoryPollStore) -> Running {
    let sink = Arc::new(RecordingSink::new());
    let mut lp = PollLoop::new(client.clone(), token.clone(), store, sink.clone(), PollLoopConfig::default());
    let handle = lp.handle();
    let thread = std::thread::spawn(move || {
        let end = lp.run();
        (end, lp.into_store())
    });
    Running { handle, sink, thread: Some(thread) }
}

impl Running {
    /// Waits (up to 10 s of real time) until `cond` holds of the events so far.
    pub fn wait_for(&self, what: &str, cond: impl Fn(&[Event]) -> bool) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            if cond(&self.sink.events()) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        panic!("timed out waiting for {what}; events: {:#?}", self.sink.events());
    }

    /// Stops the loop and returns how it ended and what it stored.
    pub fn finish(mut self) -> (LoopEnd, MemoryPollStore) {
        self.handle.stop();
        self.thread.take().unwrap().join().expect("the loop thread")
    }

    /// Waits for the loop to end on its own.
    pub fn join(mut self) -> (LoopEnd, MemoryPollStore) {
        self.thread.take().unwrap().join().expect("the loop thread")
    }
}

/// Waits (up to 10 s) for `cond`.
pub fn wait_until(what: &str, cond: impl Fn() -> bool) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(10) {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("timed out waiting for {what}");
}

pub fn seeded_secret(b: u8) -> Secret<32> {
    Secret::new([b; 32])
}
