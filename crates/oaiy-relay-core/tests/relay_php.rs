//! Layer L4 against the **real PHP relay** (`php -S` on loopback in a scratch data directory; skipped, with a message, where PHP is not usable): the client's enrolment, its identity proof,
//! the poll loop with real holds, a reset, a revocation, an outage, the gap rule and the bound of three, posting between a provider and a desktop, and a whole pairing between this
//! crate's desktop party and its phone party, with the relay's own admission read by this crate's decoders. Nothing here touches the owner's live relay, ports, WAMP web root or
//! data (see `tests/common/php.rs`).

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::env::run_loop;
use common::php::PhpRelay;
use oaiy_relay_core::admission::{MobileExpect, MobileRequest, PluginAdmission, PluginRequest, Transport};
use oaiy_relay_core::client::*;
use oaiy_relay_core::enrol::{EnrolmentKey, Role};
use oaiy_relay_core::ids::Token;
use oaiy_relay_core::json::Json;
use oaiy_relay_core::keys::{Signer, X25519Secret};
use oaiy_relay_core::pairing::phone::{store_paired, Outcome};
use oaiy_relay_core::pairing::{DesktopIdentity, DesktopPairing, PairEvent, PairingError, PairingInput, PairingTarget, PhonePairing, SasOutcome};
use oaiy_relay_core::poll::{self, Action, Answer, Counters, DecideInput, Outcome as PollOutcome, PollInfo, Report};
use oaiy_relay_core::ring::{self, RingBody};
use oaiy_relay_core::url::RelayUrl;

/// A real clock that records the pauses it was asked for.
struct RecordingClock {
    inner: SystemClock,
    sleeps: Mutex<Vec<Duration>>,
}

impl RecordingClock {
    fn new() -> Arc<RecordingClock> {
        Arc::new(RecordingClock { inner: SystemClock::new(), sleeps: Mutex::new(Vec::new()) })
    }

    fn sleeps(&self) -> Vec<Duration> {
        self.sleeps.lock().unwrap().clone()
    }
}

impl Clock for RecordingClock {
    fn unix_now(&self) -> i64 {
        self.inner.unix_now()
    }

    fn monotonic(&self) -> Duration {
        self.inner.monotonic()
    }

    fn sleep(&self, d: Duration, cancel: &Cancel) -> bool {
        self.sleeps.lock().unwrap().push(d);
        self.inner.sleep(d, cancel)
    }
}

struct Real {
    relay: PhpRelay,
    clock: Arc<RecordingClock>,
    secrets: MemorySecretStore,
    profiles: MemoryProfileStore,
}

fn start(workers: usize, wait_max: u64) -> Option<Real> {
    let relay = PhpRelay::start(workers, wait_max, true)?;
    Some(Real { relay, clock: RecordingClock::new(), secrets: MemorySecretStore::new(), profiles: MemoryProfileStore::new() })
}

fn rng_for(seed: u64) -> Box<dyn Rng> {
    let _ = seed;
    Box::new(OsRng)
}

impl Real {
    fn client(&self, pin: Option<String>, seed: u64) -> Arc<RelayClient> {
        Arc::new(RelayClient::new(
            RelayUrl::parse(&self.relay.public_url()).unwrap(),
            pin,
            self.relay.http(),
            self.clock.clone(),
            rng_for(seed),
            ClientConfig::default(),
        ))
    }

    /// Enrols the desktop with the key the installer wrote, keeping the host identity it registered.
    fn enrol_desktop(&self) -> (Arc<RelayClient>, Token, RelayProfile, Signer, X25519Secret) {
        let key = EnrolmentKey::parse(&self.relay.first_key).unwrap();
        let client = self.client(None, 1);
        let (host, host_x) = (Signer::generate().unwrap(), X25519Secret::generate().unwrap());
        let profile =
            enrol_and_store(&client, &key, "Front desk PC", &host.verify_key(), &host_x.public_key(), &self.secrets, &self.profiles, &Cancel::new())
                .expect("enrolment against the real relay");
        let token = Token::parse(std::str::from_utf8(&self.secrets.get(SECRET_TOKEN).unwrap().unwrap()).unwrap()).unwrap();
        (client, token, profile, host, host_x)
    }

    fn enrol_provider(&self) -> (Arc<RelayClient>, Token, String) {
        let uri = self.relay.cli(&["key", "provider", "--print"]);
        let key = EnrolmentKey::parse(uri.lines().last().unwrap()).unwrap();
        assert_eq!(key.role, Role::Provider);
        let client = self.client(None, 2);
        let enrolled = client
            .enroll(&key, "FormLogic", &Signer::generate().unwrap().verify_key(), &X25519Secret::generate().unwrap().public_key(), &Cancel::new())
            .unwrap();
        (client, enrolled.token, enrolled.device_id)
    }
}

fn cmd(to: &str, id: &str, body: &str) -> PostItem {
    PostItem { to: format!("dev:{to}"), lane: "cmd".into(), id: id.into(), ttl: Some(60), hdr: Hdr::new().ct("sealed1"), body: body.into() }
}

#[test]
fn enrolment_the_proof_and_the_whole_loop_on_the_real_relay() {
    let Some(real) = start(4, 2) else { return };
    let (client, token, profile, _, _) = real.enrol_desktop();
    // `info` as the real relay writes it is read by this crate's reader, and the proof against the key of the enrolment key verifies.
    let info = client.info().expect("a proved info");
    assert_eq!(info.relay_id, profile.relay_id);
    assert!(info.has_feature("pairing.v3") && info.has_feature("poll") && info.has_feature("admission.aokie-adm-v2"));
    assert_eq!((info.wait.default, info.wait.max, info.wait.poll_gap_ms, info.wait.fallback_s), (2, 2, 250, 5));
    assert!(info.lane("cmd").is_some() && info.lane("ring").is_some());
    let (pclient, ptoken, provider_id) = real.enrol_provider();
    pclient.prove(&Cancel::new()).unwrap();

    // The loop, with real holds: a posted command is delivered while a poll is held.
    let running = run_loop(&client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|e| matches!(e, Event::State(ConnectionState::Connected))));
    std::thread::sleep(Duration::from_millis(400));
    let started = std::time::Instant::now();
    let posted = pclient.post_items(&ptoken, &[cmd(&profile.device_id, "c1", "first")], &Cancel::new()).unwrap();
    assert_eq!((posted[0].status.clone(), posted[0].seq), (PostStatus::Queued, Some(1)));
    running.wait_for("c1", |ev| ev.iter().any(|e| matches!(e, Event::Accepted { since: 1, .. })));
    assert!(started.elapsed() < Duration::from_millis(1800), "a held poll woke for a post: {:?}", started.elapsed());
    // The same item again is a duplicate with the original seq; another body under the id is a conflict.
    assert_eq!(pclient.post_items(&ptoken, &[cmd(&profile.device_id, "c1", "first")], &Cancel::new()).unwrap()[0].status, PostStatus::Duplicate);
    assert_eq!(pclient.post_items(&ptoken, &[cmd(&profile.device_id, "c1", "other")], &Cancel::new()).unwrap()[0].code.as_deref(), Some("conflict"));
    pclient.post_items(&ptoken, &[cmd(&profile.device_id, "c2", "second")], &Cancel::new()).unwrap();
    running.wait_for("c2", |ev| ev.iter().any(|e| matches!(e, Event::Accepted { since: 2, .. })));
    // A restore from a backup (a new epoch): the next poll answers `reset`, the loop adopts the cursor once and goes on.
    real.relay.cli(&["reset", "epoch"]);
    running.wait_for("the reset", |ev| ev.iter().any(|e| matches!(e, Event::MailboxReset)));
    pclient.post_items(&ptoken, &[cmd(&profile.device_id, "c3", "third")], &Cancel::new()).unwrap();
    running.wait_for("c3", |ev| ev.iter().any(|e| matches!(e, Event::Accepted { since: 3, .. })));
    // A revocation ends the held poll with `401 revoked` and the loop with `forget_credential`.
    real.relay.cli(&["revoke", &profile.device_id]);
    let sink = running.sink.clone();
    let (end, store) = running.join();
    assert!(matches!(&end, LoopEnd::Stopped { action: Action::ForgetCredential, status: Some(401), code: Some(c), .. } if c == "revoked"), "{end:?}");
    assert_eq!(sink.states().last(), Some(&ConnectionState::Revoked));
    assert_eq!(store.accepted.iter().map(|a| a.item.as_ref().unwrap().id.clone()).collect::<Vec<_>>(), vec!["c1", "c2", "c3"]);
    assert_eq!(store.accepted[0].item.as_ref().unwrap().from, provider_id, "from is the sender the relay names");
    assert_eq!(store.resets, 1);
    // No 429 along the way: the loop kept to the pauses of 5.1.1 against a relay that applies the gap rule.
    assert!(!sink.events().iter().any(|e| matches!(e, Event::Answer { outcome: PollOutcome::Flow, .. })), "{:?}", sink.events());
}

#[test]
fn an_outage_of_the_real_relay_is_paced_reported_and_recovered_from() {
    let Some(mut real) = start(2, 1) else { return };
    let (client, token, _, _, _) = real.enrol_desktop();
    let running = run_loop(&client, &token, MemoryPollStore::new());
    running.wait_for("connected", |ev| ev.iter().any(|e| matches!(e, Event::State(ConnectionState::Connected))));
    real.relay.stop_servers();
    running.wait_for("unreachable", |ev| ev.iter().any(|e| matches!(e, Event::Report(Report::Unreachable))));
    real.relay.start_servers();
    running.wait_for("connected again", |ev| ev.iter().filter(|e| matches!(e, Event::State(ConnectionState::Connected))).count() >= 2);
    let sink = running.sink.clone();
    let (end, _) = running.finish();
    assert_eq!(end, LoopEnd::Cancelled);
    // The failure pauses were 1, 2, ... seconds with up to 20 percent jitter.
    let long: Vec<f64> = real.clock.sleeps().iter().map(Duration::as_secs_f64).filter(|s| *s >= 1.0).collect();
    assert!(long[0] >= 1.0 && long[0] < 1.21, "{long:?}");
    assert!(long[1] >= 2.0 && long[1] < 2.41, "{long:?}");
    assert_eq!(sink.reports().iter().filter(|r| **r == Report::Unreachable).count(), 1);
}

#[test]
fn the_gap_rule_and_the_bound_of_three_are_what_the_rules_say_they_are() {
    let Some(real) = start(8, 2) else { return };
    let (client, token, _, _, _) = real.enrol_desktop();
    // The gap rule: a second empty short poll within 250 ms is a 429 with `rule: gap` and `Retry-After: 1`, and the rules read it as flow.
    let req = PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 };
    let first = client.poll(&token, &req, &Cancel::new()).unwrap();
    assert_eq!(first.status, Some(200));
    let second = client.poll(&token, &req, &Cancel::new()).unwrap();
    assert_eq!(second.status, Some(429), "{:?}", second.body);
    assert_eq!(second.body.as_ref().unwrap().get("error").unwrap().get_str("rule"), Some("gap"));
    assert_eq!(second.headers.iter().find(|(k, _)| k == "retry-after").map(|(_, v)| v.as_str()), Some("1"));
    let d = poll::decide(&DecideInput {
        counters: Counters::default(),
        info: PollInfo::default(),
        answer: second.answer(),
        since: 0,
        persisted: true,
        we_replaced: true,
        min_client_above_ours: false,
        now_epoch: None,
        u: 0.0,
    });
    assert_eq!((d.outcome, d.base_s), (PollOutcome::Flow, 1.0));
    // Once the gap has passed it is answered again, with the epoch the relay gave.
    std::thread::sleep(Duration::from_millis(300));
    let third = client.poll(&token, &req, &Cancel::new()).unwrap();
    assert_eq!(third.status, Some(200));
    let epoch = third.body.as_ref().unwrap().get_str("epoch").unwrap().to_string();
    assert_eq!(epoch.len(), 11);
    std::thread::sleep(Duration::from_millis(300)); // the gap rule again: the last answer was empty
                                                    // The bound of three polls that wait: six at once, and at least one is refused `429 rate_limited` (with the rule of the refusal), which the rules read as flow and, for
                                                    // `in_flight`, as `cancel_own_polls`.
    let replies: Vec<PollReply> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..6)
            .map(|_| {
                let (client, token) = (client.clone(), token.clone());
                s.spawn(move || client.poll(&token, &PollRequest { since: 0, epoch: None, wait_s: 2, limit: 32 }, &Cancel::new()).unwrap())
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let refused: Vec<&PollReply> = replies.iter().filter(|r| r.status == Some(429)).collect();
    assert!(!refused.is_empty(), "six polls at once: {:?}", replies.iter().map(|r| r.status).collect::<Vec<_>>());
    for r in &refused {
        let rule = r.body.as_ref().and_then(|b| b.get("error")).and_then(|e| e.get_str("rule")).map(str::to_string);
        assert!(matches!(rule.as_deref(), Some("gap" | "in_flight")), "{rule:?}");
        let d = poll::decide(&DecideInput {
            counters: Counters::default(),
            info: PollInfo::default(),
            answer: Answer { status: r.status, headers: &r.headers, body: r.body.as_ref() },
            since: 0,
            persisted: true,
            we_replaced: true,
            min_client_above_ours: false,
            now_epoch: None,
            u: 0.0,
        });
        assert_eq!(d.outcome, PollOutcome::Flow, "a 429 is never a failure");
        assert_eq!(d.action == Some(Action::CancelOwnPolls), rule.as_deref() == Some("in_flight"));
    }
    // And every poll that was answered says which way: held and ended by a newer one, or by the wait.
    assert!(replies.iter().any(|r| r.status == Some(200)));
}

#[test]
fn a_provider_and_a_desktop_exchange_a_command_and_its_result_through_the_real_relay() {
    let Some(real) = start(3, 1) else { return };
    let (dclient, dtoken, profile, _, _) = real.enrol_desktop();
    let (pclient, ptoken, provider_id) = real.enrol_provider();
    pclient.prove(&Cancel::new()).unwrap();
    assert_eq!(
        pclient.post_items(&ptoken, &[cmd(&profile.device_id, "cmd-1", "sealed container text")], &Cancel::new()).unwrap()[0].status,
        PostStatus::Queued
    );
    let reply = dclient.poll(&dtoken, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
    let item = Item::from_json(&reply.body.as_ref().unwrap().get("items").unwrap().as_array().unwrap()[0]).unwrap();
    assert_eq!(
        (item.lane.as_str(), item.id.as_str(), item.from.as_str(), item.body.as_str()),
        ("cmd", "cmd-1", provider_id.as_str(), "sealed container text")
    );
    assert_eq!(item.hdr, "{\"ct\":\"sealed1\"}");
    // The desktop answers with a `res` carrying `hdr.re`; the relay accepts it only for a command it still remembers, from a desktop, to the sender.
    let res = PostItem {
        to: format!("dev:{provider_id}"),
        lane: "res".into(),
        id: "res-cmd-1".into(),
        ttl: Some(300),
        hdr: Hdr::new().re("cmd-1").ct("sealed1"),
        body: "result".into(),
    };
    assert_eq!(dclient.post_items(&dtoken, &[res], &Cancel::new()).unwrap()[0].status, PostStatus::Queued);
    // A provider cannot post to a provider, and a desktop cannot post a command: the relay says forbidden for the item.
    let forbidden = dclient.post_items(&dtoken, &[PostItem { lane: "cmd".into(), ..cmd(&provider_id, "x", "y") }], &Cancel::new()).unwrap();
    assert_eq!(forbidden[0].code.as_deref(), Some("forbidden"));
    // A relay that is not the pinned one is refused before a token is sent: the same relay, a client pinned to another key.
    let impostor = real.client(Some("SWUejy55xcz8xgSi-GX15ERZzyuXnb6voSaopq2Jakw".into()), 9);
    assert!(matches!(impostor.prove(&Cancel::new()), Err(ProveError::Invalid(_))));
    assert_eq!(
        impostor.poll(&dtoken, &PollRequest { since: 0, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap_err(),
        ClientError::Suspect
    );
}

#[test]
fn a_second_process_with_the_same_credential_is_told_apart_from_an_outage() {
    let Some(real) = start(6, 1) else { return };
    let (client, token, profile, _, _) = real.enrol_desktop();
    let other = real.client(Some(profile.relay_thumbprint.clone()), 12);
    other.prove(&Cancel::new()).unwrap();
    let first = run_loop(&client, &token, MemoryPollStore::new());
    first.wait_for("connected", |ev| ev.iter().any(|e| matches!(e, Event::State(ConnectionState::Connected))));
    let second = run_loop(&other, &token, MemoryPollStore::new());
    // Two processes of one credential supersede each other: the real relay says so in the hold object (`superseded`), and the rules call it `duplicate_credential` once it repeats,
    // which is not `unreachable`.
    first.wait_for("duplicate_credential", |ev| ev.iter().any(|e| matches!(e, Event::Report(Report::DuplicateCredential))));
    let reports = first.sink.reports();
    assert!(!reports.contains(&Report::Unreachable), "{reports:?}");
    second.handle.stop();
    first.handle.stop();
    let _ = second.join();
    let _ = first.join();
}

fn pair_items(client: &RelayClient, token: &Token, since: &mut u64) -> Vec<Item> {
    std::thread::sleep(Duration::from_millis(300));
    let reply = client.poll(token, &PollRequest { since: *since, epoch: None, wait_s: 0, limit: 32 }, &Cancel::new()).unwrap();
    let body = reply.body.expect("a poll answer");
    let items = body.get("items").and_then(Json::as_array).unwrap_or(&[]).to_vec();
    let parsed: Vec<Item> = items.iter().filter_map(Item::from_json).collect();
    if let Some(last) = parsed.last() {
        *since = last.seq;
    }
    parsed
}

#[test]
fn a_whole_pairing_between_the_desktop_party_and_the_phone_party_on_the_real_relay() {
    let Some(real) = start(4, 2) else { return };
    let (dclient, dtoken, profile, host, host_x) = real.enrol_desktop();
    let identity = Arc::new(DesktopIdentity {
        device_id: profile.device_id.clone(),
        name: "Front desk PC".into(),
        endpoint: Signer::generate().unwrap(),
        endpoint_x25519: X25519Secret::generate().unwrap().public_key(),
        host_ed25519: host.verify_key(),
        host_x25519: host_x.public_key(),
    });
    let mut desktop = DesktopPairing::new(identity.clone(), "aokie", profile.relay.clone(), &profile.relay_thumbprint);
    let mut since = 0u64;

    // ---- by the scanned key
    let offer = desktop.create_offer(dclient.relay_now_or_local()).unwrap();
    desktop.open(&dclient, &dtoken, &offer, &Cancel::new()).expect("the real relay opens the rendezvous");
    let target = PairingTarget::from_input(PairingInput::Key(&offer.pairing_uri)).unwrap();
    let pclient = real.client(None, 3);
    let mut phone = PhonePairing::new(pclient.clone(), target, oaiy_relay_core::pairing::phone::new_identity(Some("Test phone")).unwrap()).unwrap();
    let summary = phone.fetch_offer(&Cancel::new()).expect("the offer the real relay serves verifies");
    assert_eq!((summary.desktop_name.as_str(), summary.app_id.as_str()), ("Front desk PC", "aokie"));
    let sas = phone.respond(&Cancel::new()).expect("the real relay takes the response");
    let events: Vec<PairEvent> =
        pair_items(&dclient, &dtoken, &mut since).iter().map(|i| desktop.on_pair_item(&dclient, &dtoken, i, &Cancel::new()).unwrap()).collect();
    assert!(matches!(events.as_slice(), [PairEvent::AwaitingSas { phone_name: Some(n), .. }] if n == "Test phone"), "{events:?}");
    // A mistyped code costs nothing; the right one approves, and the real relay accepts the decision with its receipt.
    let grants: Vec<String> =
        ["state_read", "caller_read", "captions_read", "assistance_read", "assistance_respond", "rtc_signal"].iter().map(|g| g.to_string()).collect();
    let mut typo = sas.display();
    typo.pop();
    assert_eq!(desktop.confirm_sas(&dclient, &dtoken, &offer.pid, &typo, &grants, &Cancel::new()).unwrap(), SasOutcome::Incomplete);
    let approved = desktop.confirm_sas(&dclient, &dtoken, &offer.pid, &sas.display(), &grants, &Cancel::new()).unwrap();
    let SasOutcome::Approved { device_id } = approved else { panic!("{approved:?}") };

    // The finding, on the real relay: the receipt covers the grants and the relay's answer to the phone does not carry them, so the phone cannot verify it by itself and fails closed.
    assert_eq!(
        phone.wait_outcome(None, &Cancel::new()).err(),
        Some(PairingError::ReceiptGrantsUnknown),
        "the real relay returns {{issuedAt, signature}} only"
    );
    // With the grants the desktop approved, from a source the relay cannot forge (here: the test), the receipt verifies and the real sealed token opens.
    let paired = match phone.wait_outcome(Some(&grants), &Cancel::new()).unwrap() {
        Outcome::Paired(p) => p,
        _ => panic!("not paired"),
    };
    assert_eq!(paired.profile.device_id, device_id);
    store_paired(&paired, &MemorySecretStore::new(), &MemoryProfileStore::new()).unwrap();

    // The phone's token works, and the real relay's admission is read as the shipped phone reads it.
    std::thread::sleep(Duration::from_millis(300));
    let holder = phone.endpoint_thumbprint();
    let request = MobileRequest {
        app_id: "aokie".into(),
        device_id: device_id.clone(),
        display_name: Some("Test phone".into()),
        holder_thumbprint: holder.clone(),
        transports: Some(vec![Transport::RelayPoll]),
    };
    let admission = pclient
        .admission_mobile(
            &paired.token,
            &request,
            &MobileExpect { app_id: "aokie", device_id: &device_id, holder_thumbprint: &holder },
            &Cancel::new(),
        )
        .expect("a mobile admission from the real relay");
    assert_eq!(admission.scopes, grants);
    assert!(admission.relay.as_ref().is_some_and(|r| r.poll_mode), "this host passed no streaming probe: the relay offers the poll mode");
    assert_eq!(admission.expected_peer_thumbprint, identity.endpoint.thumbprint(), "the relay's record of the desktop it was paired with");
    // The plugin's admission with the desktop's token, with the roster of the one phone.
    let plugin = PluginRequest {
        app_id: "aokie".into(),
        plugin_id: "aokie".into(),
        display_name: Some("Receptionist".into()),
        endpoint: identity.endpoint.verify_key(),
        approved_peers: vec![holder.clone()],
        revision: 1,
        transports: Some(vec![Transport::RelayPoll]),
    };
    let plugin_admission = dclient.admission_plugin(&dtoken, &plugin, &Cancel::new()).expect("a plugin admission from the real relay");
    assert!(plugin_admission.relay.as_ref().is_some_and(|r| r.poll_mode));
    let _: &PluginAdmission = &plugin_admission;

    // A ring: the desktop signs the body text with the host identity it registered, the relay verifies it, the phone verifies it again with the host key it pinned from the offer.
    let expires = pclient.relay_now_or_local() + 120;
    let body = format!("{{\"aokieClass\":\"voice_offer\",\"schemaVersion\":\"1\",\"eventId\":\"evt_1\",\"offerId\":\"offer_1\",\"appId\":\"aokie\",\"callId\":\"call_1\",\"callEpoch\":\"1\",\"ownerEpoch\":\"0\",\"expiresAt\":\"{expires}\"}}");
    let sig = ring::sign(&host, &body);
    let posted = dclient
        .post_items(
            &dtoken,
            &[PostItem {
                to: format!("dev:{device_id}"),
                lane: "ring".into(),
                id: "ring-1".into(),
                ttl: Some(60),
                hdr: Hdr::new().ct("json").sig(sig),
                body: body.clone(),
            }],
            &Cancel::new(),
        )
        .unwrap();
    assert_eq!(posted[0].status, PostStatus::Queued, "{posted:?}");
    let mut phone_since = 0;
    let rings = pair_items(&pclient, &paired.token, &mut phone_since);
    let ring_item = rings.iter().find(|i| i.lane == "ring").expect("the ring");
    let hdr = oaiy_relay_core::json::parse(ring_item.hdr.as_bytes()).unwrap();
    let peer = paired.profile.peer.as_ref().unwrap();
    ring::verify(&peer.host_ed25519, &ring_item.body, hdr.get_str("sig").unwrap()).expect("the phone verifies the ring with the host key it pinned");
    let parsed = RingBody::parse(&ring_item.body).unwrap();
    parsed.check_window(pclient.relay_now_or_local()).unwrap();
    // A ring signed by another key is refused by the relay itself.
    let forged = dclient
        .post_items(
            &dtoken,
            &[PostItem {
                to: format!("dev:{device_id}"),
                lane: "ring".into(),
                id: "ring-2".into(),
                ttl: Some(60),
                hdr: Hdr::new().sig(ring::sign(&Signer::generate().unwrap(), &body)),
                body,
            }],
            &Cancel::new(),
        )
        .unwrap();
    assert_eq!(forged[0].code.as_deref(), Some("invalid_item"));

    // ---- by the typed code, with the host as the owner types it
    let second = desktop.create_offer(dclient.relay_now_or_local()).unwrap();
    desktop.open(&dclient, &dtoken, &second, &Cancel::new()).unwrap();
    let target = PairingTarget::from_input(PairingInput::Typed { code: &second.typed_code, host: &real.relay.public_url() }).unwrap();
    let mut typed =
        PhonePairing::new(real.client(None, 4), target, oaiy_relay_core::pairing::phone::new_identity(Some("Second phone")).unwrap()).unwrap();
    typed.fetch_offer(&Cancel::new()).unwrap();
    let sas2 = typed.respond(&Cancel::new()).unwrap();
    for i in pair_items(&dclient, &dtoken, &mut since).iter().filter(|i| i.lane == "pair") {
        desktop.on_pair_item(&dclient, &dtoken, i, &Cancel::new()).unwrap();
    }
    assert!(matches!(
        desktop.confirm_sas(&dclient, &dtoken, &second.pid, &sas2.display(), &grants, &Cancel::new()).unwrap(),
        SasOutcome::Approved { .. }
    ));
    assert!(matches!(typed.wait_outcome(Some(&grants), &Cancel::new()).unwrap(), Outcome::Paired(_)));

    // A pairing is single use: the first offer's rendezvous reads as ended or already decided to anyone who asks again.
    let late = PairingTarget::from_input(PairingInput::Key(&offer.pairing_uri)).unwrap();
    let mut again = PhonePairing::new(real.client(None, 5), late, oaiy_relay_core::pairing::phone::new_identity(None).unwrap()).unwrap();
    assert!(again.fetch_offer(&Cancel::new()).is_err());
}
