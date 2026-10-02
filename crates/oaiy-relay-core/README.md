# oaiy-relay-core

The shared **relay client core** of OAIY: one library, used by the desktop (which polls its mailbox and approves pairings) and by the phone (which pairs and polls), so that the
two ends of the relay protocol (`platform/protocol/relay/v1`, served by `platform/relay`) are written, tested and fixed once.

It has **no consumers yet**. Nothing in the desktop app or the mobile project depends on it; it only builds and tests. The desktop's relay client (DK-03) and the phone's pairing
and poll client (MOB-08, MOB-21a, MOB-22a) are the first two.

## What is in it

| Module | What it is |
|---|---|
| `b64`, `json` | Strict base64url (no padding, no unused bits, one spelling per byte string) and a strict JSON parser (no duplicate members, no lone surrogates, depth 64, integers told from other numbers by their spelling) with the canonical form (members sorted **by the UTF-8 bytes of their names**, integers -2^63 to 2^64-1 only; see "Canonical JSON is not RFC 8785" below) |
| `ids`, `url` | Identifier validators, the redacting `Token`, the relay URL (`https` only; no empty label, no trailing dot, the default port is no port; plain `http` on loopback only, in test builds, only after the program asks for it) |
| `keys` | `Signer`, `VerifyKey`, `X25519Public`, `X25519Secret` over `oaiy-crypto`. A signature is made only under a `SignDomain` of the protocol; verification is strict; keys of small order cannot be built |
| `info`, `enrol`, `sealed`, `ring`, `ticket`, `rotation`, `roster`, `admission` | The protocol's signed and sealed objects, each read as the repository's readers read them: the identity proof of `GET /v1/info`, the enrolment key, `sealed1` (sealed token and signed container), rings, provider tickets, rotation statements, the peer-roster hash and the Aokie admissions |
| `pairing` | Pairing v3: `math` (secret, typed code, SAS, MACs, receipt), `offer`, `response`, and the two halves as state machines: `desktop::DesktopPairing` and `phone::PhonePairing` |
| `poll` | The poll rules of README 5.1.1 (P1 to P9) as **one deterministic function**: `assess` (what a 200 adopts) and `decide` (outcome, pause, counters, action, reports). No I/O, no clock, no randomness |
| `client` | The relay client behind traits: `RelayClient` (proof gate, enrolment, poll, post, rotation, admission, pairing routes), `PollLoop` (the blocking driver that persists before it acknowledges), stores, the status sink, and a loopback-only HTTP client |
| `testing` (feature) | An in-process stub relay (with a loopback server and a `GET /v1/admin/status` that counts the requests it holds), a fake clock, a seeded random source and a scripted HTTP client, public so that a consumer's tests can drive the client without PHP |

There is one crate and not two because the desktop and the phone are the two ends of one protocol: the offer the desktop writes is the offer the phone reads, and a rule that
is fixed in one place is fixed for both. A future `oaiy-mobile-core` (JNI) would depend on this crate and add only the JNI surface.

### Features and dependencies

- Dependencies: `oaiy-crypto` (the repository's audited cryptography surface, by path), `zeroize`, and optionally `oaiy-keystore`. No `tokio`, no HTTP or TLS crate, no `serde`,
  no `unsafe` (`#![forbid(unsafe_code)]`). Dev-dependency: `serde_json` (already in `Cargo.lock`), for the differential tests.
- `loopback-http`: a blocking HTTP/1.1 client over `std::net::TcpStream` that talks to loopback only (no redirects, a body cap, a timeout, cancellation), and the plain `http://`
  relay URL for loopback. **Test builds only: it has no TLS.** Plain `http` is read only after the program calls `url::allow_loopback_http(true)` (a function that exists only with
  this feature); that the feature is switched on somewhere in the build (cargo unifies features) does not turn it on.
- `testing`: the stub relay and the fakes. `keystore`: `SecretStore` over `oaiy-keystore`.
- The crate is a workspace member and **not** a default member, so `cargo check` of the desktop does not build it.

**A relay installed on a plain `http://` base is outside the schemas by design.** It writes `ws://` and `http://` URLs (`gatewayUrl`, the compatibility endpoints) where
`admission-*-response.schema.json` has `^wss://` and the release readers accept `wss` and `https` only. This crate reads that form only for a client of a loopback `http` relay
(`MobileAdmission::parse_with(.., lax)`), which exists only in a test build; every other client reads exactly what the schemas say.

## The traits a host implements

The product supplies the I/O; the crate supplies the behaviour.

| Trait | Supplies | Notes |
|---|---|---|
| `HttpClient` | `send(&HttpRequest) -> Result<HttpResponse, TransportError>` | **Must not follow redirects** (a redirected request would carry the token to another host), must enforce `max_response_bytes` and `timeout`, and must honour `cancel`. Header names may come back in any case: the client normalises them as the response comes in, and every lookup of a header is case-insensitive. Desktop: the shared HTTPS client. Phone: OkHttp through JNI, or `ureq`/`reqwest` in Rust. **The adapter owns the copies it makes** of a request's header values and body and of a response's body: the crate wipes its own, not the adapter's (see "Secrets") |
| `Clock` | `unix_now`, `monotonic`, `sleep(duration, &Cancel) -> bool` | **The monotonic clock must keep counting while the device is suspended** (see "The clock contract") |
| `Rng` | `fill` | **Jitter of pauses only.** No secret and no protocol value is drawn from it: the pairing secret, the offer's nonce and `jti`, the proof nonces, the identity keys and the device ids come from the operating system's generator inside the crate. `OsRng` in a product |
| `SecretStore` | `get`/`put`/`delete` of named secrets | The device token and the endpoint keys. The `keystore` feature adapts `oaiy-keystore::KeyStore` |
| `ProfileStore` | `load`/`save`/`clear` of `RelayProfile` | Non-secret pins: relay, relay id and key thumbprint, device id, peer pin. A file implementation is provided |
| `PollStore` | `load`/`persist(batch)`/`clear_epoch` | **`persist` must be durable before it returns**: the loop sends the poll that acknowledges items only after it succeeds. `FilePollStore` is bounded (see "The poll store") |
| `StatusSink` | `event(&Event)` | Connection state (`Connected`, `Unreachable`, `Suspect`, `Revoked` ...), reports (`unreachable`, `duplicate_credential`, ...), accepted counts |

### The clock contract

Three promises, and what the crate does so that a host that breaks one of them cannot make it send a token to a relay it has not just proved:

1. **`monotonic()` counts the time the device is suspended.** The age of an identity proof, the relay's time, the schedule of proofs and the replacement gap of P1 are all measured
   with it, so a clock that stands still in sleep makes a proof of eight hours ago look a minute old. On Android that is `SystemClock.elapsedRealtime()` (not `uptimeMillis()`); on
   Linux `CLOCK_BOOTTIME` (not `CLOCK_MONOTONIC`); on macOS `mach_continuous_time` (not `mach_absolute_time`); on Windows a counter that includes sleep (`QueryPerformanceCounter`
   does; `QueryUnbiasedInterruptTime` does not). `SystemClock` uses `std::time::Instant`, which on Linux, Android and macOS is **not** one of these (the platform's own
   documentation: this crate did not measure it by suspending a machine): a host on those platforms passes its own `Clock`.
2. **The wall clock backs the monotonic one up.** A proof is aged by both clocks: a wall clock that moved forward by more than the proof's life (600 s), or back by more than 60 s,
   makes the proof stale whatever the monotonic clock says, and the next call proves the relay again before it sends a token.
3. **The relay's time is kept on the monotonic clock.** Each `X-OAIY-Time` sample is the relay's time minus the monotonic time at which it arrived; the median of five is the estimate and
   the applied value moves towards it by at most one second a minute. A step of the PC's wall clock therefore does not move the relay's time (it shows as a mismatch with the wall
   clock, which is warned about above 60 s). A sample more than a day from the wall clock is not believed, so a relay cannot move the time that windows are judged by by saying so in
   its first answer; the phone's key expiry (`x`) is judged against the same time.

### How the desktop uses it (DK-03)

1. `enrol_and_store(&client, &enrolment_key, name, host_ed25519, host_x25519, &secrets, &profiles, &cancel)`: proves the relay against the key's pin, enrols, stores the token
   and the profile.
2. `PollLoop::new(client, token, poll_store, sink, PollLoopConfig::default())` and `run()` on a thread; `PollHandle::stop()` and `network_changed()` from the app. The loop
   proves the relay (before the first poll, after a pause of 60 s or more, after a network change, every 300 s), polls with `wait + 10` s timeouts, persists what it accepted,
   decides by `poll::decide`, pauses, and ends on `forget_credential`, `update_client`, `report_relay_changed` or a defect. A burst of network changes is debounced into a few
   proofs; a poll that replaces another starts no sooner than 250 ms after the one it replaces (P1); a relay that advertises a poll gap below 250 ms is polled at 250 ms.
3. `DesktopPairing`: `create_offer`, `open` (the rendezvous), `on_pair_item` for every `pair` item the loop delivers, `confirm_sas` when the owner types the phone's code
   (three wrong entries deny and burn), which posts the decision with the signed receipt. It judges the offer's own window, never replaces a pairing in flight, and bounds what it
   remembers (32 pairings, 128 judged items, 1,024 consumed nonces).
4. `RelayClient::post_items`, `admission_plugin`, `rotate_token`.

### How the phone uses it (MOB-08, MOB-21a, MOB-22a)

1. `PairingTarget::from_input(PairingInput::Key(uri))` (the scanned `oaiy-pair:` URI) or `PairingInput::Typed { code, host }`.
2. `PhonePairing::new(client, target, identity)`, `fetch_offer()` (proves the relay, verifies the offer's MAC first, judges its window and the key's own expiry), `respond()`
   (returns the SAS to show), `wait_outcome(grants_from_the_desktop)` (verifies the receipt before a profile exists, opens the sealed token), then `store_paired`. A phone whose
   response the desktop rejected answers afresh (new claims, signature and MAC) and never resends the lapsed text.
3. `PollLoop` with the paired token, and `RelayClient::admission_for_profile` for the phone's admission: it is read as the shipped phone reads it **and** compared with what the owner
   approved (its scopes are the receipt's signed grants, its expected peer is the desktop that was pinned from the offer).

**The receipt's grants.** The desktop signs `{appId, grants, issuedAt, phoneThumbprint, pid}` into the receipt. The shipped relay returns `receipt: {issuedAt, signature}` only, so
the phone needs the grants from elsewhere (`wait_outcome(Some(grants))`); a relay that returns `receipt.grants` is believed over the caller's list, because the signature
protects it (a relay that lies makes the receipt fail). The relay is to be changed to return them (see "Findings"); until then `wait_outcome(None)` fails closed with
`ReceiptGrantsUnknown`, and the test that shows the phone pairing from what the real relay returns is `#[ignore]`d and marked TODO.

## What the client refuses to do

- **No token before the relay is proved.** A bearer is sent only after the relay's identity was proved against the pinned thumbprint within the last 600 s (by the monotonic and the
  wall clock); after a failed proof the client is `Suspect` and sends nothing until one succeeds. This holds for every call that carries a token (poll, post, rotation, both
  admissions, the four pairing calls of the desktop).
- **No acknowledgement before the write.** An item is persisted before the poll that carries its `seq` is sent; a failed write is a failure, never `unreachable`.
- **No profile without a receipt.** The phone verifies the desktop's signed receipt before it makes a profile, and opens the sealed token only after that.
- **No signature outside a domain, no non-strict verification, no key of small order.**
- **No redirect, no plain `http` outside loopback.** And no secret in `Debug` output (next section).

## Secrets: what is redacted and what is wiped

**`Debug` prints nothing of a secret** for: `Token`, `EnrolmentKey`, `Enrolled`, `Signer`, `X25519Secret`, `PairingSecret`, `PairingKey`, `PairingInput` (the typed code and the
scanned key), `Bearer`, `IceServer` (so also `MobileAdmission` and `PluginAdmission`, whose derived `Debug` goes through it), `HttpRequest` (header values and body), `HttpResponse`
(its length, not its body) and `Sas`. Each is a test (`rv_secret_debug`, `hardening`, the unit tests).

**Wiped when dropped** (`Drop` or `Zeroizing`; measured in freed memory by counting allocators, `tests/wipe_drops.rs` and the reviewer's `tests/rv_enc_wipe.rs`, for every scenario of
their table, with a positive and a negative control): the token text and the keys of `oaiy-crypto`; the six-bit digits and the output of the base64url decoder of a secret; the parsed
answers that hold a token (`json::parse_wiped`); the typed code and the pairing and enrolment keys as they are built and read; an `HttpRequest`'s header values and body; the
offer the owner is shown (`NewOffer`) and the pairing's MAC key when the pairing is over; a TURN credential; the `Sas`; the bodies of the enrolment and rotation answers.

**Not wiped, and not claimed:**

- the copies a **host** makes: the `HttpClient` adapter's request and response buffers, the strings a UI makes of the typed code or the SAS, whatever the OS keeps (socket and TLS
  buffers, swap, a crash dump);
- **stack copies**: a value moved or returned leaves its old bytes in a stack frame or a register; neither measurement sees the stack, and `Zeroizing` does not reach them;
- a `Json` tree that was parsed with `json::parse` and not `parse_wiped` (every document that is not known to hold a secret: items, `info`, the poll's answers);
- `Item` bodies (sealed content, opened elsewhere) and the poll store's files, which hold what the relay delivered;
- memory a buffer gave back by growing (`realloc`) outside the paths above.

## Pacing, with the numbers

The poll rules are README 5.1.1 (P1 to P9) and `poll::decide` is the table of them, tested against the repository's own fixture and two readers. Two things the table leaves out:

- **A floor for `pollGapMs`.** The relay may be configured with a gap of 0; the client never polls faster than **250 ms** apart (`MIN_POLL_GAP_MS`, P3's default). The floor is in
  `PollInfo::from_info`, where a client reads `info`, so that `decide` stays the table as written.
- **The phone's wait for the owner (`GET /v1/pair/{pid}`).** The relay's budgets are 30 requests a minute per address, **60 counted `GET`s per rendezvous** (about 10 s apart over its
  600 s) and **10 outcome reads a minute**. The phone asks again at once only when the relay says it held the request (`hold.granted`, not `superseded`) **and** the request took
  as long as a hold takes by the phone's own clock (at least half the wait, at most 10 s); every other answer that leaves the rendezvous as it was is followed by a pause of **10 s
  with up to 20 percent jitter**, so a relay or a proxy that answers at once and says it held never makes the phone spin. A refused hold is paced as a poll's (2, 4, 5 s); a `429`
  waits its `Retry-After`; a failure backs off 1, 2, 4 ... 60 s. (The contract does not say how fast a phone may ask for an `answered` rendezvous with `wait=0`; see "Findings".)
- **An integer is an integer literal** (Interpretation 23 of the poll rules): digits only, with an optional minus sign, no fraction, no exponent and not `-0`. `1.0`, `1e2` and `-0`
  are not integers, whatever a language's `Number` makes of them; a `seq` or `cursor` above 2^53 - 1 is not a uint53 and is dropped, not read as a smaller number. The crate follows
  this; the Node reader of the fixtures does not (see "Findings").

## The poll store

`FilePollStore` writes the accepted items to `inbox.jsonl` (one JSON object a line, as received) and flushes them **before** it writes the cursor (`cursor.json`, by write and rename,
with the directory flushed where the platform allows it; Windows has no directory flush), so a crash between the two redelivers and loses nothing. It is **bounded and survives a
crash in an append**: a last line with no newline is cut off before the next append (the cursor never moved past it, so its items come again) and is never read; the file is closed
into a segment of about 1 MiB (`inbox-NNNNNN.jsonl`) that the consumer reads and deletes (`closed_segments`, `read_segment`, `discard_segment`) without touching the file the loop
writes to; and a write that would take the inbox past 64 MiB fails (a paced storage failure: the relay keeps the items) and nothing is ever dropped to make room. A cursor that
cannot be read is an error and not a first run.

## Tests, by layer

Run everything with `cargo test -p oaiy-relay-core` (features are on for this crate's own tests). `--test-threads 4` is plenty. The counts are those of the last full run
(`2 October 2026`): **313 tests pass**, 14 are `#[ignore]`d on purpose (listed below), none are skipped quietly.

| Layer | Where | Tests | What it shows |
|---|---|---|---|
| L1 unit | `src/**` | 54 | Parsers, arithmetic, boundaries, the clock estimator, the poll table's helpers (a `Retry-After` trimmed of spaces and tabs only, a refused hold that never overflows) |
| L1 known answers | `tests/vectors.rs` | 35 | Every vector A0 to A12 of `vectors.json` (A5 is the TURN credential, not part of the client), the sealed-token and ceremony recordings, and the edges the vectors leave open (the info's thumbprint, a `time` member, a small-order R, an empty sealed box, a sync container's domain) |
| L2 fixtures | `tests/poll_fixture.rs`, `pairing_ceremony.rs`, `aokie_fixtures.rs` | 5 + 16 + 6 | All 135 cases of `poll-client.json`; the recorded pairing ceremony reproduced **byte for byte** by the two parties, ending with the recorded SAS and token, **and played again with one artifact damaged at a time** (an offer's MAC or text or window, the relay it names, a response's signature, MAC or window, the typed code, a receipt's signature, date or grants, the sealed token), each of which must be refused at its step with nothing sent that should not be; the recorded Aokie admissions accepted, about 40 damaged ones refused |
| differential, self-contained | `tests/json_differential.rs`, `poll_differential.rs` | 3 + 2 | The JSON parser against `serde_json`; `decide` against the repository's Python reader on 18,000 random cases and its Node reader on 6,000 (SKIPPED loudly without python and node, a failure where `OAIY_REQUIRE_TOOLS=1`) |
| L3 stub relay | `tests/client_stub.rs`, `pairing_stub.rs`, `stores.rs`, `loopback.rs`, `hardening.rs`, `wipe_drops.rs` | 32 + 21 + 11 + 7 + 9 + 1 | The client against an in-process relay: outage pacing (1, 2, 4 ... 32 s), 429 rules, refused holds (2, 4, 5), reset, revoke, 426, a failing store, a duplicate credential, a woken hold, the proof gate on every call that carries a token, **the relay's own count of held requests (one for a loop, one for a phone, never two)**, both pairings (key and typed code), a receipt dated ahead, the file store after a crash and when full, the loopback transport against raw sockets that misbehave, the hardening round (a 429 for the proof, header names, wall-clock steps, bounds), and the drop-time wipe |
| **L4 real relay** | `tests/relay_php.rs` | 7 (+1 ignored) | The client against the **real PHP relay** (see below) |
| fuzz | `tests/fuzz.rs` | 9 | Every decoder on random bytes and on damaged copies of the protocol's corpus: no panic, no hang, and the round-trip and range properties of what is accepted |
| the review | `tests/rv_*.rs` | 95 | The independent review's tests: 33 on the client's I/O, 41 on the two pairing parties, the secrets (what `Debug` prints, what freed memory holds, which generator they come from), encoding, allocation, the findings (each was `#[ignore]`d until its fix landed, and none is ignored now) |
| mutation | `mutation/` | 212 mutants | Each breakage must make the tests fail (see below) |

**Ignored on purpose:** the 13 drivers of the differential tests (`rv_enc_corpus` 7, `rv_ed25519_diff`, `rv_x25519_diff`, `rv_pairing_math_diff`, `rv_sealed_diff`'s driver,
`rv_poll_diff`, `rv_admission_diff`), which need generated input and fail loudly when they are run without it (they are run by `tools/run-differentials.ps1`, below), and
`relay_php::the_phone_pairs_from_the_grants_the_real_relay_returns_with_the_receipt`, a **TODO** for the relay change (the relay does not return `receipt.grants` yet).

### L4: the real PHP relay

`tests/relay_php.rs` installs a relay with `platform/relay/bin/install.php` into a scratch directory under the cargo target directory and serves it with `php -S` on
`127.0.0.1` on ports the OS chose. It never touches the repository's own relay data, the WAMP web root, or any of the owner's ports. PHP is found at `OAIY_PHP`, then
`C:\wamp64\bin\php\php8.4.15\php.exe`, then `php` on the path, and must have `sodium` and `pdo_sqlite`. **Where there is none the layer prints a banner (`SKIPPED: <why>`) and passes,
unless `OAIY_REQUIRE_PHP=1`, which makes a missing PHP a failure**: the Linux lane of the CI sets it (and installs PHP if the runner has none), the Windows lane runs the layer where its
runner has PHP and otherwise says so as a warning annotation. Read the output of a run for the word SKIPPED.

`php -S` on Windows is single threaded, so a held poll would block every other request. The relay keeps its holds in the shared database, so the harness runs a small fleet of
servers on one data directory behind `FleetHttp` (it picks a server that is idle and rewrites the public port), which behaves as the workers of one host do.

It covers: enrolment and the identity proof; the poll loop with real holds and a post that wakes a hold; duplicate and conflicting posts; a reset by `bin/relay.php reset epoch`;
a revocation by `bin/relay.php revoke`; an outage (every server killed, then restarted) with the paced retries and the `unreachable` report; the gap rule and the bound of three
polls that wait; two processes with one credential; a command and its result between a provider and a desktop; a client pinned to another key refused before any token is sent;
two whole pairings (key and typed code) between the desktop party and the phone party, including the real relay's sealed token, the phone's and the plugin's admission, a response
the desktop rejected and the phone's fresh answer under `pid.2`, and a ring signed with the host key that the relay verifies and the phone verifies again.

### Differentials against what the rest of the system uses

`powershell -File crates/oaiy-relay-core/tools/run-differentials.ps1` (options: `-Only ed25519,x25519,math,sealed,poll,admission,b64,json,text`, `-Keep`, `-AllowSkips`) generates
corpora in a scratch copy of the generators (the repository stays clean), runs the ignored drivers with them and asserts what must hold. It needs `php` with sodium and gmp,
`python` with `cryptography`, `node` and `cargo`; a pipeline whose tool is missing prints `SKIPPED` and makes the exit status 3. Results of the last run:

| Pipeline | Against | Result |
|---|---|---|
| ed25519 | libsodium (PHP), OpenSSL (Node, Python `cryptography`) on 249 hand-built edge cases | the crate accepts exactly what libsodium accepts (0 differences). Beside it: the crate's `VerifyKey::from_bytes` refuses what the relay's `Crypto::isValidEd25519Public` accepts in 2 cases (non-canonical encodings) and accepts what it refuses in 72 (keys with a component of small order that the relay will not enrol); signature verdicts are libsodium's in all of them |
| x25519 | libsodium, 3,054 u-coordinates (14 low-order encodings, the edge of the field) | refuses the 18 that libsodium refuses, equal shared secrets on the rest |
| math | an independent recomputation of the pairing arithmetic | equal (300 cases, 95 typed-code variants, 13 SAS entries) |
| sealed | libsodium's sealed boxes (430) and the crate's (12) | all open, both ways |
| poll | the reviewer's second implementation of the README, 20,000 cases | 0 disagreements, with the readings the README now fixes (dates added as they stand, a `seq` above 2^53 - 1 dropped, `-0` not an integer, no `Retry-After` for a failure the client caused) |
| admission | the Python port of the shipped decoders, 20,000 damaged copies of the recorded admissions | **1,005 differ, pinned**: the crate is stricter in 485 (the bearer and the scopes, and ICE servers, gateway URLs and the peer thumbprint where the damage is) and looser in 516 and 4 relay advertisements (see the last section) |
| b64 | PHP `B64.php`, a strict Python reference, Node, 1,010,908 inputs | 0 differences |
| json | PHP `json_decode` and the relay's `Json::decode`, Python, Node, `serde_json`, 200,000 inputs | the general reader accepts exactly what the strict Python reference accepts and writes the same text (0 differences); it is stricter than the others on purpose (a duplicate member, a lone surrogate, more than 64 levels) and reads `1e999` as Python does and `serde_json` does not |
| text | PHP `Ids.php`, `cleanName`, the typed-code and SAS rules, the schemas' patterns on relay URLs | 0 differences in 120,009 typed codes, 176,543 SAS entries, 144,144 ids and 40,018 names; 16 of 7,564 URLs differ from the schemas' pattern **on purpose** (an empty label, a trailing dot, a label that starts or ends with a hyphen, an over-long host, the default port) |

### Fuzzing

`OAIY_FUZZ_ROUNDS` (default 2000 per target) raises the rounds; 200,000 rounds in a release build (27 s) and 100,000 in a debug build (which has overflow checks; 70 s) pass. The fuzz test found one defect, now fixed with its own test: a `Retry-After` date read against a client
clock at `i64::MIN` overflowed.

### Mutation testing

`python crates/oaiy-relay-core/mutation/run.py` (Windows: it ends a hung run with `taskkill`; `--check` verifies every anchor without a build, `--only`, `--from`, `--no-php`, `--out`)
applies the 212 mutations of `mutation/mutations.py`, `review_mutations.py` and `hardening_mutations.py` one at a time (a changed constant, a removed check, an inverted
comparison, a reordered write), builds, and runs the tests in three tiers (the fast ones, every other test target of the crate, the real relay) until one fails. The outcomes are
**KILLED, SURVIVED, INVALID** (the anchor is not there exactly once, or the mutant does not compile, retried once) **and TIMEOUT** (a run that did not end in 240 s: reported apart and
**never counted as a kill**); the exit status is 0 only when every mutant is KILLED. A file with CRLF line ends is refused. It changes files in place and restores them: run it in a
worktree nobody else builds in, with a `CARGO_TARGET_DIR` of its own.

**Result of the last full run (commit `dca04510`, 2 October 2026, `mutation/results.txt`): 212 mutants, 212 killed, 0 survived, 0 invalid, 0 timed out.** The unit tests of the
library killed 57, the fast integration tests 121, the other test targets of the crate 34, and **none was left to the real relay** (the harness names only the first test that fails,
so a kill by a test that is sensitive to many things, such as the wipe table of `rv_enc_wipe`, is a kill by the harness's definition and not an attribution). The 212 are the 83 of the
implementer, 88 of the independent reviewer and 41 written for the second round; **two mutants are left out as equivalent, and the reasons are written beside them in the files**:
the reviewer's `K05` (an all-zero Diffie-Hellman result: no peer key that could produce one can be built, since all fourteen small-order encodings are refused first) and `H15` (the poll
loop's failure arm: nothing reaches it, since the loop drops an epoch that is not the relay's spelling and `RelayClient::poll` can fail with nothing else the loop does not handle above).
Six mutants change `oaiy-crypto` where the crate delegates (`dependency`).

How the number was reached, because "83 of 83" was true and incomplete: the implementer's 83 were all killed on LF checkouts, but the reviewer's run of a sample of 20 of them on a CRLF
checkout gave 18 killed and **2 INVALID** (an anchor that spans lines does not match CRLF text); the crate is LF in every checkout since (`.gitattributes`) and the harness refuses a CRLF
file. The reviewer's own 89, run on the crate as it was reviewed, gave 51 killed, 1 hang (reported apart and not a kill) and 37 survived; the 38 that were not killed, run again on the
tests of `c205f09b`, gave 20 killed, 17 survived and 1 timed out, and each now has a test. The first full run after the second round (`a54af7d6`, 213 mutants) left 5 standing and 2 that only the real relay killed; they got tests, and the run above
is the second. The ceremony tests alone (`tests/pairing_ceremony.rs`, which plays the recorded pairing byte for byte and then once for each damaged artifact) were checked against the
pairing mutants one at a time: they kill the SAS gate (an approval without the code, a wrong code not counted, no denial, no burn), the response's MAC and signature and window, the
receipt's check, the offer's window with and without a key expiry, the relay the offer names and the relay key it states.

## Not covered, and not verified

- **No TLS and no product HTTP client.** `LoopbackHttp` is for tests. The behaviour that matters for a real `HttpClient` (no redirects, the body cap, the timeout, cancellation) is
  specified on the trait and tested against the loopback implementation only.
- **No consumer.** Nothing here has run inside the desktop app or on a phone. The Android and wasm results below are compile checks, not runs.
- **The real relay only on plain HTTP over loopback**, served by `php -S` workers, not behind Apache or nginx with PHP-FPM; hold behaviour under FPM, a CDN or a proxy that buffers is
  not exercised. Mailbox quotas and limits (`413`, full mailbox), TTL expiry in real time, `re` and `peek` lookups, the streaming routes, the WebSocket gateway, TURN credentials,
  and provider key rotation are not exercised in L4 (rotation, tickets and the quotas are covered by vectors and the stub only).
- **Constant-time behaviour is by construction** (`oaiy-crypto`'s `ct_eq`), not measured. A mutant that only changes timing cannot be killed by a functional test. Zeroization is
  measured where "Secrets" says, and only there.
- **Suspend.** The crate's promise about a monotonic clock that counts sleep is a contract on the host, not something this crate tested by suspending a machine; the 300-second proof
  schedule, the relay-clock slew over hours, and a phone that sleeps through a proof's age are tested with fake clocks (including a wall clock that steps), not in real time.
- **The Linux CI lane was not run.** Everything here ran on Windows. The `cfg(unix)` code (the directory flush) is linted and compiled for the Android targets, not run; the lane installs PHP from
  the distribution if the runner has none (Ubuntu 22.04 has 8.1, and the relay supports 8.0 and later), which was not tried here (the real-relay layer ran on PHP 8.4.15).
- **Directory flushes on Windows.** The store flushes the directory after a rename where the platform has a way to (Unix); Windows has none, and this was not tested by a crash.
- **Platform behaviour**: doze, network changes (`PollHandle::network_changed` is exercised against the stub and the real relay), the phone's keystore, and the desktop's keystore
  adapter on a real OS keystore are not tested here.
- **Mutation score is not proof.** The survivors and equivalent mutants are listed above, and 212 hand-made mutants are a sample of the ways to be wrong.
- **Lows of the review not done:** `Sas` keeps its three fields public (its `Debug` prints nothing, its `Drop` wipes it and `display()` cannot panic on one built by hand).

## Findings about the contract

Gaps and disagreements found while building and testing this. The contract (`platform/protocol/relay/v1`, `platform/relay`) is not changed by this crate; what needs a change there is
marked **relay** or **protocol**.

1. **The receipt covers the grants, and the relay does not return them to the phone. (Decided: the relay returns `receipt.grants`. Relay change.)** The desktop signs the grants into
   the receipt, but `GET /v1/pair/{pid}` answers with `receipt: {issuedAt, signature}` only (`platform/relay/src/Pairing.php`: `approval()` verifies the grants and keeps `issuedAt` and
   `signature`). The change is small and safe, because the signature covers the grants and the schema allows extra members: **relay** `Pairing.php:343-345` (store the sorted grants in
   the receipt JSON), `pairing-fetch-response.schema.json` (the receipt object), README step 9 (lines 410 and 412), `tests/cases/pairing.php`, `pairing-ceremony.json` (line 163) and
   its two readers (`verify_fixtures.py` 264-276, `verify_fixtures.mjs` 262) and the pinned digests in `relay_conformance.py`. This crate already verifies from the grants the relay
   returns when it does and keeps the explicit-grants path; `relay_php.rs` has an `#[ignore]`d TODO test of the whole pairing from what the real relay returns, to be enabled with the change.
2. **A relay installed on an `http://` base writes `ws://` and `http://` URLs that the schemas and the shipped readers refuse. (Decided: outside the schemas by design; test builds only.)**
   See "Features and dependencies".
3. **An integer is an integer literal. (Decided. Protocol and relay, the fixtures' Node reader.)** The two poll-client readers disagreed on whole-number floats:
   `verify_poll_client.mjs` reads `1.0`, `1e2` and `5.0` as integers (JavaScript cannot tell them from `1`, `100` and `5`), the Python reader and this crate do not. The README rule is
   now "an integer literal: digits only, no fraction, exponent or `-0`" (Interpretation 23); **protocol:** write it into P2 and make the Node reader look at the lexeme. The 135
   cases of the table do not contain such a value, so both pass today.
4. **The pairing wait needs a pacing rule. (Decided: the numbers above. Protocol.)** The contract does not say how fast a phone may read an `answered` rendezvous with `wait=0`; a phone
   that did so spun (a test of this crate did, until it was given a pause). The relay's budgets are 30 requests a minute per address, 60 counted `GET`s per rendezvous and 10 outcome
   reads a minute; **protocol:** put these numbers and the 10 s pause into the protocol README.
5. **`php -S` on Windows cannot serve a hold and another request at once** (design 9.4), so any single-process development relay deadlocks a poll against a post. Documented in the
   design; noted here because the L4 harness needs a fleet to work around it.
6. **Canonical JSON is not RFC 8785.** The README (line 48) sorts members by the UTF-8 bytes of their names, as the relay does; RFC 8785 sorts by UTF-16 code units. They differ only for
   names with a character above U+FFFF beside one in U+E000 to U+FFFF (839 keys of the 200,000-input corpus; none of the protocol's own, which are ASCII). This crate follows the README
   and says so; a signer in another language that implements RFC 8785 would sign other bytes for such a name.
7. **Key validity differs between the relay and the crate** (see the Ed25519 row above): the relay's `isValidEd25519Public` refuses keys that the crate builds (those with a component of
   small order) and builds two non-canonical encodings that the crate refuses. Signature verdicts are libsodium's either way. **Relay or protocol** if the two rules are meant to be
   one: this crate would adopt the relay's.

## Android and wasm

**Android** (NDK 27.0.12077973; `cargo check` and `cargo build --lib`): `aarch64-linux-android` and `x86_64-linux-android` pass with the default features, with `keystore` (which
builds `oaiy-keystore` for Android) and with all features (re-run on 2 October 2026 after the last change to the library). These are compile checks: nothing was linked or run on a device or an emulator.

**wasm32-unknown-unknown**: the default build **fails in `getrandom` 0.3** (under `oaiy-crypto`'s OS random source), which has no backend for that target until the final application sets
`--cfg getrandom_backend="wasm_js"` and enables getrandom's `wasm_js` feature; `oaiy-crypto` was not changed. With a custom getrandom backend (compile check only) this crate checks
cleanly with the default features and with `testing` and `loopback-http`. `std::net` and `SystemTime` compile for that target and fail at run time, so a green check does not mean
`LoopbackHttp` or `SystemClock` work there: a web client brings its own `HttpClient` and `Clock`. The `keystore` feature does not build for wasm32 (`oaiy-keystore` has no platform
module for it), so a wasm build leaves it off. No feature gate was needed in this crate.

## Where the admission differs from the shipped decoders

`MobileAdmission` and `PluginAdmission` follow `aokie_decoders.py` (the Python port of the shipped readers), including that an unusable `relay` advertisement degrades to the WebSocket
gateway (`relay: None`) instead of failing the admission, and that a phone's `iceServers` may be absent. The differential above (1,005 of 20,000 damaged copies) shows where they
differ, in both directions.

**Stricter than the shipped decoders:** the bearer (its claims and their shape) and the scopes (they must be known grants, without a repeat), and, on the damaged copies, members of the
phone's device record, the shape of an ICE server (a TURN server without a `username` or `credential`), `gatewayUrl` and the peer thumbprint.

**Looser than the shipped decoders** (the crate reads these as the shape of the schema and leaves the rest to the carrier, which has the information to compare):

- the **echoed endpoint key of a plugin admission**: the crate does not compare the key the relay echoes (its `algorithm` and `thumbprint`) with the plugin's own;
- the **phone's device record**: the `id` and other members of it that the shipped decoder compares are only shape-checked;
- the **content of an ICE credential** (the TURN username and credential are any string);
- the **`gatewayUrl` and `relayOnly` values**: the crate checks the shape of `gatewayUrl` (`wss`, `/v2/realtime`) but not that it equals the gateway the phone holds from its signed discovery
  (a phone paired to a personal relay has none), and takes `relayOnly` as it is: the carrier compares them.

`expectedPeerKeyThumbprint` is advisory when the admission is read alone, because the shipped phone's pin comes only from the MAC-verified offer; `admission_for_profile` compares it
with the pin, and the scopes with the receipt's grants.
