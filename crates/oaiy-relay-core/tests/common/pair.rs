//! A desktop, a phone, a relay and a clock for the pairing tests.

#![allow(dead_code)]

use std::sync::Arc;

use oaiy_relay_core::client::*;
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::{DesktopIdentity, DesktopPairing, NewOffer, PairEvent, PairingInput, PairingTarget, PhoneIdentity, PhonePairing};
use oaiy_relay_core::testing::stub::StubConfig;
use oaiy_relay_core::testing::SeededRng;

use super::env::{client_for, env, quick, Env};

pub const GRANTS: [&str; 6] = ["state_read", "caller_read", "captions_read", "assistance_read", "assistance_respond", "rtc_signal"];

pub fn grants() -> Vec<String> {
    GRANTS.iter().map(|g| g.to_string()).collect()
}

pub struct World {
    pub env: Env,
    pub token: Token,
    pub profile: RelayProfile,
    pub desktop: DesktopPairing,
    pub identity: Arc<DesktopIdentity>,
    pub rng: SeededRng,
}

pub fn world(cfg: StubConfig) -> World {
    let env = env(cfg);
    let (token, profile) = env.enrol_desktop();
    let identity = Arc::new(DesktopIdentity {
        device_id: profile.device_id.clone(),
        name: "Front desk PC".into(),
        endpoint: Signer::generate().unwrap(),
        endpoint_x25519: X25519Secret::generate().unwrap().public_key(),
        host_ed25519: Signer::generate().unwrap().verify_key(),
        host_x25519: X25519Secret::generate().unwrap().public_key(),
    });
    let desktop = DesktopPairing::new(identity.clone(), "aokie", profile.relay.clone(), &profile.relay_thumbprint);
    World { env, token, profile, desktop, identity, rng: SeededRng::new(99) }
}

pub fn default_world() -> World {
    world(quick())
}

impl World {
    /// Opens a rendezvous as the desktop.
    pub fn new_offer(&mut self) -> NewOffer {
        let offer = self.desktop.create_offer(self.env.client.relay_now_or_local()).unwrap();
        self.desktop.open(&self.env.client, &self.token, &offer, &Cancel::new()).unwrap();
        offer
    }

    /// A phone: its own client and keys.
    pub fn phone(&self, input: PairingInput<'_>, seed: u64) -> PhonePairing {
        let target = PairingTarget::from_input(input).unwrap();
        let client = client_for_target(&self.env, &target, seed);
        let identity = oaiy_relay_core::pairing::phone::new_identity(Some("Test phone")).unwrap();
        PhonePairing::new(client, target, identity).unwrap()
    }

    /// Polls the desktop's inbox once and gives every `pair` item to the party, as the poll loop's consumer does.
    pub fn deliver(&mut self) -> Vec<PairEvent> {
        // Let the relay's gap rule see time pass between two polls of one device.
        self.env.clock.advance(std::time::Duration::from_secs(1));
        let reply = self.env.client.poll(&self.token, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
        let body = reply.body.expect("a poll answer");
        let items = body.get("items").and_then(|i| i.as_array()).unwrap().to_vec();
        let mut events = Vec::new();
        for raw in items {
            if let Some(item) = Item::from_json(&raw) {
                events.push(self.desktop.on_pair_item(&self.env.client, &self.token, &item, &Cancel::new()).unwrap());
            }
        }
        events
    }
}

pub fn client_for_target(env: &Env, target: &PairingTarget, seed: u64) -> Arc<RelayClient> {
    let c = Arc::new(RelayClient::new(
        target.relay.clone(),
        None,
        Arc::new(env.stub.clone()),
        env.clock.clone(),
        Box::new(SeededRng::new(seed)),
        ClientConfig::default(),
    ));
    let _ = client_for;
    c
}

pub fn phone_identity(_seed: u64) -> PhoneIdentity {
    oaiy_relay_core::pairing::phone::new_identity(Some("Test phone")).unwrap()
}
