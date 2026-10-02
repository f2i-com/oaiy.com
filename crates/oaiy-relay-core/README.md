# oaiy-relay-core

The shared **relay client core** of OAIY: one library, used by the desktop (which polls its mailbox and approves pairings) and by the phone (which pairs and polls), so that the
two ends of the relay protocol (`platform/protocol/relay/v1`, served by `platform/relay`) are written, tested and fixed once.

It has **no consumers yet**. Nothing in the desktop app or the mobile project depends on it; it only builds and tests. The desktop's relay client (DK-03) and the phone's pairing
and poll client (MOB-08, MOB-21a, MOB-22a) are the first two.

## What is in it

| Module | What it is |
|---|---|
| `b64`, `json` | Strict base64url (no padding, no unused bits, one spelling per byte string) and a strict JSON parser (no duplicate members, no lone surrogates, depth 64, integers told from other numbers by their spelling) with the canonical form (members sorted bytewise, integers -2^63 to 2^64-1 only) |
| `ids`, `url` | Identifier validators, the redacting `Token`, the relay URL (`https` only; plain `http` for loopback with the `loopback-http` feature) |
| `keys` | `Signer`, `VerifyKey`, `X25519Public`, `X25519Secret` over `oaiy-crypto`. A signature is made only under a `SignDomain` of the protocol; verification is strict; keys of small order cannot be built |
| `info`, `enrol`, `sealed`, `ring`, `ticket`, `rotation`, `roster`, `admission` | The protocol's signed and sealed objects, each read as the repository's readers read them: the identity proof of `GET /v1/info`, the enrolment key, `sealed1` (sealed token and signed container), rings, provider tickets, rotation statements, the peer-roster hash and the Aokie admissions |
| `pairing` | Pairing v3: `math` (secret, typed code, SAS, MACs, receipt), `offer`, `response`, and the two halves as state machines: `desktop::DesktopPairing` and `phone::PhonePairing` |
| `poll` | The poll rules of README 5.1.1 (P1 to P9) as **one deterministic function**: `assess` (what a 200 adopts) and `decide` (outcome, pause, counters, action, reports). No I/O, no clock, no randomness |
| `client` | The relay client behind traits: `RelayClient` (proof gate, enrolment, poll, post, rotation, admission, pairing routes), `PollLoop` (the blocking driver that persists before it acknowledges), stores, the status sink, and a loopback-only HTTP client |
| `testing` (feature) | An in-process stub relay (with a loopback server), a fake clock, a seeded random source and a scripted HTTP client, public so that a consumer's tests can drive the client without PHP |

There is one crate and not two because the desktop and the phone are the two ends of one protocol: the offer the desktop writes is the offer the phone reads, and a rule that
is fixed in one place is fixed for both. A future `oaiy-mobile-core` (JNI) would depend on this crate and add only the JNI surface.

### Features and dependencies

- Dependencies: `oaiy-crypto` (the repository's audited cryptography surface, by path), `zeroize`, and optionally `oaiy-keystore`. No `tokio`, no HTTP or TLS crate, no `serde`,
  no `unsafe` (`#![forbid(unsafe_code)]`). Dev-dependency: `serde_json` (already in `Cargo.lock`), for the differential test of the JSON parser.
- `loopback-http`: a blocking HTTP/1.1 client over `std::net::TcpStream` that talks to loopback only (no redirects, a body cap, a timeout, cancellation), and the plain `http://`
  relay URL for loopback. Test builds only: it has no TLS.
- `testing`: the stub relay and the fakes. `keystore`: `SecretStore` over `oaiy-keystore`.
- The crate is a workspace member and **not** a default member, so `cargo check` of the desktop does not build it.

## The traits a host implements

The product supplies the I/O; the crate supplies the behaviour.

| Trait | Supplies | Notes |
|---|---|---|
| `HttpClient` | `send(&HttpRequest) -> Result<HttpResponse, TransportError>` | **Must not follow redirects** (a redirected request would carry the token to another host), must enforce `max_response_bytes` and `timeout`, must honour `cancel`, and lower-cases header names. Desktop: the shared HTTPS client. Phone: OkHttp through JNI, or `ureq`/`reqwest` in Rust |
| `Clock` | `unix_now`, `monotonic`, `sleep(duration, &Cancel) -> bool` | The relay's own clock is estimated from `X-OAIY-Time` (median of 5, slewed at most 1 s a minute) and used for every window |
| `Rng` | `fill` | `OsRng` in a product |
| `SecretStore` | `get`/`put`/`delete` of named secrets | The device token and the endpoint keys. The `keystore` feature adapts `oaiy-keystore::KeyStore` |
| `ProfileStore` | `load`/`save`/`clear` of `RelayProfile` | Non-secret pins: relay, relay id and key thumbprint, device id, peer pin. A file implementation is provided |
| `PollStore` | `load`/`persist(batch)`/`clear_epoch` | **`persist` must be durable before it returns**: the loop sends the poll that acknowledges items only after it succeeds. `FilePollStore` writes `inbox.jsonl` and then `cursor.json`, so a crash redelivers and loses nothing |
| `StatusSink` | `event(&Event)` | Connection state (`Connected`, `Unreachable`, `Suspect`, `Revoked` ...), reports (`unreachable`, `duplicate_credential`, ...), accepted counts |

### How the desktop uses it (DK-03)

1. `enrol_and_store(&client, &enrolment_key, name, host_ed25519, host_x25519, &secrets, &profiles, &cancel)`: proves the relay against the key's pin, enrols, stores the token
   and the profile.
2. `PollLoop::new(client, token, poll_store, sink, PollLoopConfig::default())` and `run()` on a thread; `PollHandle::stop()` and `network_changed()` from the app. The loop
   proves the relay (before the first poll, after a pause of 60 s or more, after a network change, every 300 s), polls with `wait + 10` s timeouts, persists what it accepted,
   decides by `poll::decide`, pauses, and ends on `forget_credential`, `update_client`, `report_relay_changed` or a defect.
3. `DesktopPairing`: `create_offer`, `open` (the rendezvous), `on_pair_item` for every `pair` item the loop delivers, `confirm_sas` when the owner types the phone's code
   (three wrong entries deny and burn), which posts the decision with the signed receipt.
4. `RelayClient::post_items`, `admission_plugin`, `rotate_token`.

### How the phone uses it (MOB-08, MOB-21a, MOB-22a)

1. `PairingTarget::from_input(PairingInput::Key(uri))` (the scanned `oaiy-pair:` URI) or `PairingInput::Typed { code, host }`.
2. `PhonePairing::new(client, target, identity)`, `fetch_offer()` (proves the relay, verifies the offer's MAC first), `respond()` (returns the SAS to show),
   `wait_outcome(grants_from_the_desktop)` (verifies the receipt before a profile exists, opens the sealed token), then `store_paired`.
3. `PollLoop` with the paired token, and `RelayClient::admission_mobile` for the phone's admission.

## What the client refuses to do

- **No token before the relay is proved.** A bearer is sent only after the relay's identity was proved against the pinned thumbprint within the last 600 s; after a failed
  proof the client is `Suspect` and sends nothing until one succeeds.
- **No acknowledgement before the write.** An item is persisted before the poll that carries its `seq` is sent; a failed write is a failure, never `unreachable`.
- **No profile without a receipt.** The phone verifies the desktop's signed receipt before it makes a profile, and opens the sealed token only after that.
- **No signature outside a domain, no non-strict verification, no key of small order.**
- **No redirect, no plain `http` outside loopback, no secret in `Debug` output.**

## Tests, by layer

Run everything with `cargo test -p oaiy-relay-core` (features are on for this crate's own tests). `--test-threads 4` is plenty.

| Layer | Where | Tests | What it shows |
|---|---|---|---|
| L1 unit | `src/**` | 42 | Parsers, arithmetic, boundaries |
| L1 known answers | `tests/vectors.rs` | 30 | Every vector A0 to A12 of `vectors.json` (A5 is the TURN credential, not part of the client), the sealed-token and ceremony recordings |
| L2 fixtures | `tests/poll_fixture.rs`, `pairing_ceremony.rs`, `aokie_fixtures.rs` | 5 + 1 + 5 | All 135 cases of `poll-client.json`; the recorded pairing ceremony reproduced **byte for byte** by the two parties, ending with the recorded SAS and token; the recorded Aokie admissions accepted, about 40 damaged ones refused |
| differential | `tests/json_differential.rs`, `poll_differential.rs` | 3 + 2 | The JSON parser against `serde_json`; `decide` against the repository's Python reader on 18,000 random cases and its Node reader on 6,000 |
| L3 stub relay | `tests/client_stub.rs`, `pairing_stub.rs`, `stores.rs`, `loopback.rs` | 23 + 17 + 7 + 7 | The client against an in-process relay: outage pacing (1, 2, 4 ... 32 s), 429 rules, refused holds (2, 4, 5), reset, revoke, 426, a failing store, a duplicate credential, a woken hold, both pairings (key and typed code), the adversarial list, and the loopback transport against raw sockets that misbehave |
| **L4 real relay** | `tests/relay_php.rs` | 6 | The client against the **real PHP relay** (see below) |
| fuzz | `tests/fuzz.rs` | 9 | Every decoder on random bytes and on damaged copies of the protocol's corpus: no panic, no hang, and the round-trip and range properties of what is accepted |
| mutation | `mutation/` | 83 mutants | Each breakage must make the tests fail (see below) |

### L4: the real PHP relay

`tests/relay_php.rs` installs a relay with `platform/relay/bin/install.php` into a scratch directory under the cargo target directory and serves it with `php -S` on
`127.0.0.1` on ports the OS chose. It never touches the repository's own relay data, the WAMP web root, or any of the owner's ports. PHP is found at `OAIY_PHP`, then
`C:\wamp64\bin\php\php8.4.15\php.exe`, then `php` on the path, and must have `sodium` and `pdo_sqlite`; **otherwise the tests print `SKIPPED: <why>` and pass**, so read the output
of a CI run for that word.

`php -S` on Windows is single threaded, so a held poll would block every other request. The relay keeps its holds in the shared database, so the harness runs a small fleet of
servers on one data directory behind `FleetHttp` (it picks a server that is idle and rewrites the public port), which behaves as the workers of one host do.

It covers: enrolment and the identity proof; the poll loop with real holds and a post that wakes a hold; duplicate and conflicting posts; a reset by `bin/relay.php reset epoch`;
a revocation by `bin/relay.php revoke`; an outage (every server killed, then restarted) with the paced retries and the `unreachable` report; the gap rule and the bound of three
polls that wait; two processes with one credential; a command and its result between a provider and a desktop; a client pinned to another key refused before any token is sent;
two whole pairings (key and typed code) between the desktop party and the phone party, including the real relay's sealed token, the phone's and the plugin's admission, and a ring
signed with the host key that the relay verifies and the phone verifies again.

### Fuzzing

`OAIY_FUZZ_ROUNDS` (default 2000 per target) raises the rounds; 200,000 rounds in a release build and 100,000 in a debug build (which has overflow checks) pass. The fuzz test found
one defect, now fixed with its own test: a `Retry-After` date read against a client clock at `i64::MIN` overflowed.

### Mutation testing

`python crates/oaiy-relay-core/mutation/run.py` (Windows: it ends a hung run with `taskkill`) applies the 83 mutations of `mutation/mutations.py` one at a time (a changed constant, a removed check, an inverted comparison, a
reordered write), builds, and requires the tests to fail; a mutant that does not compile is counted apart, one that survives the fast layers is also run against the real relay.
It changes files in place and restores them; run it in a worktree nobody else builds in. The result of the last full run is `mutation/results.txt`: **83 mutants, 83 killed, none survived** (one reported a build failure that did not repeat and is killed on a rerun; the harness now retries a failed build once). The first run, of 79 mutants, left eight standing, which showed eight things the tests did not pin (the median over five clock samples, one spelling of a typed code, a proof for a nonce of another size, a ticket and a rotation statement one second too long, a wait of 301 s in an info, a raw 0x1f control character in a string, a non-canonical Ed25519 key); each has a test now. Four of the 83 change `oaiy-crypto` (small-order and canonical key checks) and are marked `dependency`.

## Not covered, and not verified

- **No TLS and no product HTTP client.** `LoopbackHttp` is for tests. The behaviour that matters for a real `HttpClient` (no redirects, the body cap, the timeout, cancellation) is
  specified on the trait and tested against the loopback implementation only.
- **No consumer.** Nothing here has run inside the desktop app or on a phone. The Android and wasm results below are compile checks, not runs.
- **The real relay only on plain HTTP over loopback**, served by `php -S` workers, not behind Apache or nginx with PHP-FPM; hold behaviour under FPM, a CDN or a proxy that buffers is
  not exercised. Mailbox quotas and limits (`413`, full mailbox), TTL expiry in real time, `re` and `peek` lookups, the streaming routes, the WebSocket gateway, TURN credentials,
  and provider key rotation are not exercised in L4 (rotation, tickets and the quotas are covered by vectors and the stub only).
- **Constant-time behaviour and zeroization are by construction** (`oaiy-crypto`'s `ct_eq`, `Secret`, `Zeroizing`), not measured here. A mutant that only changes timing cannot be
  killed by a functional test.
- **Time.** The 300-second proof schedule, the relay-clock slew over hours, and a phone that sleeps through a proof's age are tested with fake clocks, not in real time.
- **Platform behaviour**: doze, network changes (`PollHandle::network_changed` is exercised against the stub), the phone's keystore, and the desktop's keystore adapter on a real OS
  keystore are not tested here.
- **Mutation score is not proof.** A survivor list is shown, equivalent mutants may exist, and 79 hand-made mutants are a sample of the ways to be wrong.

## Findings about the contract

Gaps and disagreements found while building and testing this. The contract (`platform/protocol/relay/v1`, `platform/relay`) is not changed by this crate.

1. **The receipt covers the grants, and the relay does not return them to the phone (needs a decision).** The desktop signs `{appId, grants, issuedAt, phoneThumbprint, pid}` into
   the receipt, but `GET /v1/pair/{pid}` answers the phone with `receipt: {issuedAt, signature}` only (`platform/relay/src/Pairing.php`: `approval()` verifies the grants and keeps `issuedAt` and `signature`). The phone cannot verify the receipt without the grants, and it must not
   trust grants the relay tells it unverified. Confirmed against the real relay: `PhonePairing::wait_outcome(None)` fails closed with `ReceiptGrantsUnknown`; with the grants from
   elsewhere it pairs. The simplest fix is for the relay to return `receipt.grants` (the signature then protects them, so a relay that lies makes the receipt fail). Until then the phone
   needs the grants out of band (for example, shown in the pairing UI and entered, or fetched as `scopes` of the admission, which is after the token is stored).
2. **A relay installed on an `http://` base writes `ws://` and `http://` URLs that the schemas and the shipped readers refuse.** `gatewayUrl` is `ws://127.0.0.1:PORT/v2/realtime`
   and the compatibility endpoints are `http://...`, while `admission-*-response.schema.json` has `^wss://` and the release readers accept `wss` and `https` only. This crate reads the
   relaxed form only for a client of a loopback `http` relay in a build with `loopback-http` (`MobileAdmission::parse_with(.., lax)`), as a managed-beta build does for the endpoints.
   The README could say that a plain-`http` relay is outside the schema by design.
3. **The two poll-client readers disagree on whole-number floats.** `verify_poll_client.mjs` (Node) reads `1.0`, `1e2` and `5.0` as integers, because JavaScript cannot tell them
   from `1`, `100` and `5`; the Python reader and this crate do not. The 135 cases of the table do not contain such a value, so both pass. P2's wording ("an integer") should say that
   a number with a fraction or an exponent, and `-0`, is not one, and the Node reader should look at the lexeme.
4. **I found no pacing rule for `GET /v1/pair/{pid}` with `wait=0`.** A phone that polls an `answered` rendezvous without a hold would spin (a test of this crate did, until it was
   given one). This crate waits `max(pollGapMs, 1 s)` between such reads; the contract could say how fast a phone may ask.
5. **`php -S` on Windows cannot serve a hold and another request at once** (design 9.4), so any single-process development relay deadlocks a poll against a post. Documented
   in the design; noted here because the L4 harness needs a fleet to work around it.

## Android and wasm

**Android** (NDK 27.0.12077973; `cargo check` and `cargo build --lib`): `aarch64-linux-android` and `x86_64-linux-android` pass with the default features, with `keystore` (which
builds `oaiy-keystore` for Android) and with all features. These are compile checks: nothing was linked or run on a device or an emulator.

**wasm32-unknown-unknown**: the default build **fails in `getrandom` 0.3** (under `oaiy-crypto`'s OS random source), which has no backend for that target until the final application sets
`--cfg getrandom_backend="wasm_js"` and enables getrandom's `wasm_js` feature; `oaiy-crypto` was not changed. With a custom getrandom backend (compile check only) this crate checks
cleanly with the default features and with `testing` and `loopback-http`. `std::net` and `SystemTime` compile for that target and fail at run time, so a green check does not mean
`LoopbackHttp` or `SystemClock` work there: a web client brings its own `HttpClient` and `Clock`. The `keystore` feature does not build for wasm32 (`oaiy-keystore` has no platform
module for it), so a wasm build leaves it off. No feature gate was needed in this crate.

## Where the admission differs from the shipped decoders

`MobileAdmission` and `PluginAdmission` follow `aokie_decoders.py` (the Python port of the shipped readers), including that an unusable `relay` advertisement degrades to the WebSocket
gateway (`relay: None`) instead of failing the admission, and that a phone's `iceServers` may be absent. They check the shape of `gatewayUrl` (`wss`, `/v2/realtime`) but not that it
equals the gateway the phone holds from its signed discovery (a phone paired to a personal relay has none): the carrier compares it. `expectedPeerKeyThumbprint` is advisory here, as
the shipped phone's pin comes only from the MAC-verified offer.
