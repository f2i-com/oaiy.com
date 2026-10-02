# Aokie compatibility fixtures

What the relay answers to the shipped Aokie plugin and phone, recorded from the real relay code with the keys and ids of the
protocol package's vectors, so that a contract test in the Aokie repository can run the plugin's and the phone's own decoders
against it, and so that anything else that claims to be an Aokie-compatible relay can be held to the same documents.

| File | What it holds |
|---|---|
| `admission.json` | nine admissions: the plugin's (stream, poll mode, a carrier that offers both, no ICE server, STUN only) and two phones' (stream, poll mode, STUN only, relay only, the longest TURN credential). Each has the request the desktop's broker or the phone sends and the relay's answer. |
| `challenge.json` | `GET .../relay/challenge` for the plugin, phone A and phone B, each with the bearer it was built from. |
| `frames.json` | a conversation through `POST` and `GET .../relay/frames`: three frames from a phone (an empty object, a 2^53+1 integer, a float `1.0`, Unicode, nested empties), the plugin reading them, finding the tail, answering, both phones reading. Request bodies are also kept as the exact text sent (`bodyText`). |
| `stream.json` | three bodies of `GET .../relay/stream` byte for byte: frames waiting, a resume with `since=2`, an idle stream with a keepalive. |
| `errors.json` | fourteen answers that are not success: the codes of section 10.6 in FormLogic's three-member shape, with the headers a client acts on. |
| `ice.json` | ICE and TURN minting for fixed inputs (fully deterministic), including FormLogic's own known answers. |
| `aokie_decoders.py` | the rules of the shipped decoders, transcribed (no relay code, no JSON Schema). |
| `verify_aokie_fixtures.py` | reads every file with those rules and refuses about 170 damaged copies of them. `python verify_aokie_fixtures.py [-v]`; the last line is `<n> checks, 0 mismatches`. The conformance suite runs it. |

`php tests/fixtures.php --write aokie` records them again (from `platform/relay`); `--check` verifies the committed files without
changing them. The relay clock of every recording is 1790000000, the admission secret is Appendix A4's test secret
(`relay.admissionSecretHex`, so a bearer can be verified), and the TURN secret is Appendix A5's. Random in a recording: the
`jti` of an admission and the `connectionId` and `challengeNonce` of a challenge; a rewrite changes those and nothing else.
The bearers are credentials of a relay that never existed. No device token is written anywhere.

## The rules, in one place

Read from the Aokie sources; `aokie_decoders.py` is the executable form and names the Rust function of each.

**The plugin's `AdmissionResponse`** (`aokie-plugin/src/companion_gateway/admission.rs`, `deny_unknown_fields`). Every member but
`relay` is required (`device` may be any JSON value and is read by nobody; `turnCredentialExpiresAt` is a member that is a number
or `null`). `into_credentials` then requires: `tokenType` `Bearer`, `role` `plugin`, `appId` and `subjectId` (the plugin id) as
sent, and the five members the desktop's broker echoes (`endpointPublicKey`, `holderKeyThumbprint`,
`approvedPeerKeyThumbprints`, `peerRosterRevision`, `peerRosterHash`) equal to what it sent; `appId` and `subjectId` of 1 to 200
bytes of `[A-Za-z0-9_.:-]`; an access token of 1 to 16,384 bytes with no control character; `expiresIn` and `expiresAt - now` each
above 10 and at most 300; the ICE rules below; a `gatewayUrl` that is `wss` with no credentials or fragment. A `relay` member that
is malformed, on `http`, or spread over more than one origin does not fail the admission: the carrier keeps the WebSocket
gateway. The three URLs are compared as a cursor domain across admissions, so they must be byte-identical in every admission.

**The phone's `AdmissionResponse`** (`apps/aokie-mobile/src-tauri/src/managed_auth.rs`, `deny_unknown_fields`). Required:
everything but `iceServers` and `relay`. `device` is exactly `{id, appId, subjectId, role, displayName, grants, approvedAt,
lastSeenAt}` and `device.grants` equals `scopes` as an ordered list. `validate_admission` requires a bearer of 16 to 16,384 bytes,
`expiresIn` 1 to 300, `expiresAt` in (now, now + 300], and against the phone's session: the same `gatewayUrl`, `appId` and
device id, `role` `mobile`, the holder thumbprint the phone made, `relayOnly` equal to the discovery document's, an
`expectedPeerKeyThumbprint` that differs from the holder, scopes that include `state_read` and are all among the fourteen names.

**ICE** (both decoders, and `aokie-media`'s `IceServerConfig::validate_all`). At most 8 servers of 1 to 8 urls; a url of at most
2,048 bytes with no control character, beginning `stun:`, `stuns:`, `turn:` or `turns:`; a username of at most 512 bytes and a
credential of at most 2,048. Any entry with a `turn:` or `turns:` url is TURN: a non-empty username and credential and an
`expiresAt` more than 30 seconds and at most 24 hours ahead. Any other entry is STUN: empty username and credential and no
`expiresAt`. The plugin requires the two members on a STUN entry (the phone defaults them); an `expiresAt` of `null` is refused by
the plugin. `relayOnly` needs a TURN entry, and `turnCredentialExpiresAt` equals the earliest TURN `expiresAt` (or `null` with none).

**`EndpointChallengeFrame`** (`aokie-protocol/src/v2.rs`, `deny_unknown_fields`, then `validate(now)`). `kind` `endpoint_challenge`,
`schemaVersion` 2, safe ids, a `role` of `mobile` or `plugin`, and the peer policy: a mobile carries `expectedPeerKeyThumbprint`
(not its own key) and NO roster member; a plugin carries a strictly ascending roster of 1 to 64 that does not hold its own key, a
revision of 1 or more and the hash that recomputes, and no `expectedPeerKeyThumbprint`. `expiresAt` must be after `now` and at
most 30 seconds ahead of it. **The relay's 25 second life therefore tolerates a phone clock that runs at most 5 seconds behind
the relay's and 24 seconds ahead of it**; see the design defects in the package README.

**The stream** (`SseParser::push` and `parse_sse_block`, in both carriers). Chunks are normalised from CRLF; an event ends at a
blank line; comment lines and unknown fields are skipped; `id` is a `u64`; `event: end` is an `End` with that id (or none);
`event: frame` needs JSON `data` with a `u64` `seq`, a string `from` and a `frame` member, and the plugin also reads `subjectId`
and `grants` (an unknown name, a duplicate, more than 16 or a non-list gives an empty authority set). A block that decodes to
nothing is ignored. The plugin resumes from the highest `seq` it saw (`?since=`), primes its cursor with `GET ...?since=N&wait=0`
reading `lastSeq`, and treats a body that ends without `end` as a failure.

**An error** (`MobileApiError`, `deny_unknown_fields`): exactly `error` (true), `code` (a safe id) and `message` (1 to 240 bytes,
no control character). Anything else is read as a bare HTTP status.

## What a Rust contract test in the Aokie repository needs

None of this was written in the Aokie repository from here: this package has no Cargo build of the Aokie workspace, and the Aokie repository was only read. **The phone side has since been run** by an Android emulator test of the shipped phone (outside this repository), which exercises the phone's real decoders and carrier against a loopback relay of this package and against these fixtures; the plugin side has not been run. The loopback relay is plain http, so its admission carries `ws://` and `http://` URLs, which these fixtures (a deployed relay's) do not (protocol `README.md`, Interpretation 61).

1. **The plugin.** A test in `crates/aokie-plugin/src/companion_gateway/tests.rs` (the types are `pub(super)`): for each 200 plugin
   case of `admission.json`, `serde_json::from_value::<AdmissionResponse>(body)` and `into_credentials(Some(app_id), plugin_id,
   authority)` with an `EndpointAuthority` built from the case's request (`endpointPublicKey`, the sorted roster, the revision and
   the hash). `into_credentials` and the ICE validation call `unix_now()` directly: either add a `now` seam, or shift the times of
   the fixture by `real_now - 1790000000` before decoding (`expiresAt`, `turnCredentialExpiresAt`, each TURN `expiresAt`; the
   plugin verifies neither the bearer nor the TURN credential). Assert `relay` is `Some` with the three URLs and `transport_label()`
   is `relay`, and that the `lifetime` is 80 seconds.
2. **The phone.** `validate_admission(session, &admission, holder)` in `managed_auth.rs` with a `ManagedSession` whose
   `gateway_url`, `app_id`, `device_id` and `discovery_relay_only` are the case's `gatewayUrl`, `request.appId`,
   `request.deviceId` and `relayOnly`, and the same time shift. **A phone paired to a personal relay has no signed discovery
   document**, so where those four values come from is undecided on the phone side (from the pairing offer's `relay` member is
   the natural choice); the test fixes what the relay needs of it.
3. **Challenges.** `EndpointChallengeFrame::validate(now_unix)` is pure: feed every case of `challenge.json` with `now = 1790000000`
   and check that a plugin's and a phone's shapes both pass and that `validate_peer_policy` accepts them (the plugin's roster hash
   recomputes with `peer_roster_hash`).
4. **The stream and the frames.** `SseParser::push` of both carriers is pure and `pub(crate)`: push every body of `stream.json`
   whole and cut at every byte offset, and assert the same `RelayStreamEvent`s, that `subject_id` and `grants` are those of
   `frames.json`, and that a `frame` is the semantic value posted (the plugin re-serialises through `serde_json::Value`, so
   member order is not compared). `tail_cursor_page` reads `lastSeq` of the tail step. `relay_post_body` must produce exactly
   the `bodyText` of the post steps from their frames.
5. **Errors.** `classify_admission_http_failure(status, is_json, bytes)` (phone) with the admission errors of `errors.json`: 403 and
   422 are `AdmissionFailure::Policy { code, message }`, 401 is `Unauthorized`; and `relay_status_error` for the codes of the
   routes: 401 rotates, **403, 404 and 503 make the plugin re-bootstrap**, 429 on a post is retried and then dropped, and every
    other status (a `500`) is a failure to reconnect that a frames post retries three times. That is why a database that is busy, gone or
    cannot be opened is **`500 internal` on these routes**, on all eight requests the plugin and the phone make (and `503` on the native ones):
    see Interpretation 59, which also says what the plugin's 10 second timeouts do to a long wait, and `tests/lib/AokiePlugin.php`
    in the relay package for the plugin's table with the lines of `companion_relay.rs` it comes from.
6. **ICE.** `validate_admission_ice_configuration` and `IceServerConfig::validate_all` over every `iceServers` of `admission.json`
   and `ice.json` (with the time shift), and `usable_relay_endpoints` over the `relay` members and the variants the verifier damages.
7. **An end to end test** that serves `admission.json`, `challenge.json`, `frames.json` and `stream.json` from a local HTTP stub and
   runs `RelayChannel` against it (the plugin's `connect`, `recv_text`, `send_text`) is the strongest form and needs a stub server
   in the test dependencies.
8. **`docs/contracts/aokie-companion-realtime.v2.schema.json` disagrees with its own decoder**: its `endpointChallenge` requires
   `approvedPeerKeyThumbprints` on every challenge (present and empty for a mobile), while `EndpointChallengeFrame` defaults the
   member and refuses a non-empty one, and FormLogic omits it. The relay follows the decoder and FormLogic; the schema should
   follow them too.
