# OAIY Relay Protocol v1

`oaiy-relay/1`: the wire contract between an OAIY relay (a small server the owner runs
themselves) and its clients: the OAIY desktop, a phone, a provider such as FormLogic, and
the Aokie plugin. The relay stores and forwards small opaque items in per-endpoint
mailboxes; clients fetch them with one multiplexed long poll of at most 20 seconds.

This folder is the contract, not an implementation. A relay written in any language is
conformant when it passes `../../tests/relay_conformance.py` (schemas and vectors) and,
later, the black-box suite that reuses those schemas against a running relay.

| File | What it is |
|---|---|
| `README.md` | this document: the normative rules |
| `*.schema.json` | JSON Schema 2020-12, ids `https://oaiy.com/schemas/relay/v1/<name>.schema.json` |
| `vectors.json` | known-answer vectors: inputs and expected outputs |
| `generate_vectors.py` | writes `vectors.json` (`--check` proves the file is current) |
| `verify_vectors.mjs` | an independent Node re-computation of every vector (`node:crypto` only) |

Test: `python ../../tests/relay_conformance.py` (from `platform/protocol/relay/v1`, or from anywhere:
`python platform/protocol/tests/relay_conformance.py`).

Key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## The one idea

The relay is transport, not authority. Devices authenticate to it with revocable
capability tokens. What a desktop will actually do is decided by signatures it verifies
against keys pinned out of band, judged against the signer's clock, never the relay's. A
hostile relay can drop, delay and replay what is not sealed; it cannot forge a command,
a pairing or a chat request, and it cannot read sealed lanes. Every `from`, `subjectId`
and grant that the relay writes into an envelope is a hint, never an authority.

## Versioning

- `protocol` is `oaiy-relay/<major>`; the major is also the URL prefix (`/v1/`). A client that does not know the major MUST refuse with a plain message and MUST NOT guess.
- Within a major the relay MAY add optional fields, features and lanes. It MUST NOT remove or repurpose them, add required request members, or narrow a limit below what an older client can send without answering `426 upgrade_required`. Anything else is `/v2`, served beside `/v1` while both are needed. Consumers MUST ignore unknown members unless a schema says `additionalProperties: false`.
- A client uses a feature only if `info.features` lists it, and ignores features it does not know.
- `minClient` is compared with the optional request header `X-OAIY-Level` (default 1); a lower level gets `426`. In v1 `minClient` is 1.

## 1. Conventions

- **Base URL.** `https://host[/path]`. Every route below is relative to it and begins `/v1/`. https only; plain http is accepted by a client only for loopback in test builds. There is no `.local` http exception for relay credentials. A relay's `public_url` is path-free and MUST NOT share an origin with the provider or any other registered credential.
- **Encodings.** `b64u` is base64url without padding (RFC 4648 section 5). Padding, whitespace and characters outside the alphabet are refused. Where a value has a fixed decoded length, a reader also refuses a spelling whose unused low bits are not zero (see Interpretations). Timestamps are Unix seconds (integers). Sizes are bytes of the UTF-8 text as sent.
- **Raw bytes into hashes, MACs, signatures and key derivations.** A value that has a text spelling (b64u, hex, Crockford base32) enters SHA-256, HMAC, an Ed25519 signature or HKDF as its RAW bytes, never as its text: `pid` and `s` are 16 bytes, a nonce, a public key and a secret are the bytes their b64u text decodes to, a digest is its 32 bytes. `||` joins byte strings; a quoted string in an expression is its ASCII bytes and `0x00` is one zero byte. The only texts that are hashed, MACed or signed as text are those the description calls a text: an offer text, a canonical JSON text, a body text "exactly as posted" or "as sent", and the ASCII of a decimal time or of the twelve SAS characters. A JSON document that carries a b64u string carries the text of it, because a JSON string is text (the `pid` member of the approval receipt is the 22-character spelling). Interpretation 21.
- **JSON.** UTF-8 (RFC 8259). Request bodies are at most 1 MiB. Unknown members in requests and responses are ignored unless a schema says `additionalProperties: false`. Signed and MACed material is always verified over the exact bytes received and only then parsed; nothing is re-serialised before verification. Integer members are written as integers: a spelling with a fraction or exponent (`60.0`, `6e1`), and the spelling `-0`, is not an integer for the relay even though JSON Schema's `integer` accepts the first two and JSON's grammar the third (Interpretations 2 and 23).
- **Canonical JSON** is required in exactly one family, pairing, because the phone's format dictates it: keys sorted bytewise (by their UTF-8 bytes), no whitespace, integers only (a float, `-0`, or an integer outside -2^63 to 2^64-1, is refused, so that every integer has exactly one spelling), strings escaped as JSON (quote, backslash and control characters escaped; `/` and non-ASCII left as they are, control escapes `\b \f \n \r \t` or `\u00xx` in lower case). Keys are ASCII everywhere, so byte order and code point order agree. Signer-shipped texts (commands, results, tickets, rotation statements, enrolment bodies, admission claims, info) are NOT canonical: the signer chooses the bytes once and ships them.
- **Relay time.** Every response carries the header `X-OAIY-Time` (the relay's Unix seconds when the response is produced) and every dynamic JSON success body also carries `"time"`; `GET /v1/info` is static and carries the header only. A client keeps a relay offset `time - localNowAtReceipt`, sampled when the response is received (a held poll's time is its end, so a request-midpoint sample would be biased by half the hold), takes the median of the last five samples and slews the offset by at most 1 second a minute. Relay time is used for relay-side expiry, pairing windows, ring `expiresAt` on the phone, and for warning the owner of a clock difference above 60 seconds. It is NOT used to judge signed commands or tickets (section 9.4).
- **Headers.** Requests: `Authorization: Bearer <credential>`; optional `X-OAIY-Client: <product>/<version>` (diagnostics only, never used for authorisation); optional `X-OAIY-Level: <int>`. Responses: `X-OAIY-Relay: oaiy-relay/1`, `X-OAIY-Time`, `Cache-Control: no-store` (except ETag-validated GETs, which use `private, no-cache`, and the static `info`), `Retry-After` on 429 and 503, `X-OAIY-Hold: granted|refused` on held routes.
- **Methods.** GET, POST and OPTIONS are the methods first-party clients use. Every mutating route has a POST form (section 5); PUT, PATCH and DELETE are accepted as aliases for clients that prefer them. Anything else is `405 method_not_allowed`, including HEAD. The reason is deployment: stock web application firewall rules commonly allow only GET, HEAD, POST and OPTIONS, and a revoke must never be blocked by one.
- **Content types.** `application/json; charset=utf-8` both ways (the media type is compared case-insensitively and the charset parameter may be omitted). A request that carries a body with any other content type is `415 unsupported_media_type`. A request without a body needs no content type.

## 2. Identifiers

| Name | Form | Schema definition |
|---|---|---|
| `relayId` | `rly-` + 22 characters b64u (128 random bits), generated at install | `common#relayId` |
| `deviceId` | `dev-` + 22 characters b64u, generated by the relay at enrolment; the only form a desktop or the provider may put into a URL or a capability | `common#deviceId` |
| `providerId` | `prov-` + 22 characters b64u, generated by the relay when a provider key is redeemed | `common#providerId` |
| item `id` | `[A-Za-z0-9._-]{1,128}` but never `.` or `..`, chosen by the sender; unique per (mailbox, lane, sender) while the relay remembers it | `common#itemId` |
| `pid` (pairing rendezvous) | 16 bytes from HKDF of the pairing secret, written as b64u (22 characters) in URLs and JSON; a hash or a KDF takes the 16 raw bytes, never the 22 characters (the SAS input) | `common#pid` |
| `rid` (reply box) | b64u of 16 bytes (22 characters), chosen by the browser | `common#rid` |
| token | `oaiyrt1.` + b64u(8 bytes) + `.` + b64u(32 bytes) = 63 characters | `common#token` |
| admin token | `oaiyadm1.` + b64u(8 bytes) + `.` + b64u(32 bytes) | `common#adminToken` |
| key thumbprint | b64u(SHA-256 of the canonical JWK `{"crv":"Ed25519","kty":"OKP","x":"<b64u>"}`), 43 characters; identical to the Aokie `endpoint_thumbprint` | `common#thumbprint` |
| `appId`, `pluginId` | `[A-Za-z0-9_.:-]{1,64}` (narrowed from Aokie's 200 so a party mailbox name fits a 255 byte indexed column); a longer id is `400 invalid_request` | `common#appId` |
| mailbox address | `dev:<deviceId>`, `rbx:<rid>`, `app:<appId>@<desktopDeviceId>/plugin`, `app:<appId>@<desktopDeviceId>/mobile:<thumbprint>`; the party forms exist only behind the compatibility routes (section 10.6) and are scoped to the desktop that paired the phone | `common#mailbox`, `common#inboxAddress` |

## 3. Mailboxes and items

A mailbox is an append-only queue with one consumer. Posts address `dev:<deviceId>` (a device
inbox; the id itself starts `dev-`, so an address reads `dev:dev-Q1w2E3r4T5y6U7i8O9p0aB`) or `rbx:<rid>` (a reply box).

**Delivered item** (`item.schema.json`, an element of `items` in a poll response):

```json
{"seq": 42, "id": "6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a", "lane": "cmd", "from": "prov-Q1w2E3r4T5y6U7i8O9p0aB",
 "at": 1790000000, "exp": 1790000060, "hdr": {"ct": "sealed1"}, "body": "<opaque string>", "rp": "Ry3kq0wEo2nq1c9h5c7Zab"}
```

| Field | Meaning |
|---|---|
| `seq` | unsigned integer below 2^53, per mailbox, allocated by the relay inside the post transaction from that mailbox's own counter, strictly increasing in commit order, first value 1, gaps allowed. It is not global, so a cursor reveals nothing about other mailboxes. |
| `id`, `lane` | as posted |
| `from` | the sender's device or provider id, `rbx:<rid>`, or `relay`. A hint for routing and logs. Never an authority. |
| `at`, `exp` | relay receive time and the relay-enforced expiry (`at + ttl`), relay time |
| `hdr` | small routing header, always an object, keys from the allow-list below only |
| `body` | an opaque UTF-8 string. The relay never parses, re-encodes or trims it. |
| `rp` | optional reply box id (lanes `ai`, `ai.in`) |

**Post form** (`post-request.schema.json`, an element of `items` in `POST /v1/items`):
`{"to": "...", "lane": "...", "id": "...", "ttl": 60, "hdr": {...}, "body": "..."}`. `to`, `lane`, `id`
and `body` are required; `hdr` is optional (absent means `{}`). `ttl` is an integer number of seconds from 1 to the lane's maximum (both published in `info.limits.lanes`); absent means the lane default; zero, negative, fractional, non-numeric, null or above the maximum is `400 invalid_item` (no silent clamping). It is a duration, not an absolute time, so a sender's clock cannot matter; the relay computes `exp`.

**`hdr` allow-list.** `re` (an item id: the item this answers), `ct` (one of `text`, `json`, `sealed1`, `tunnel1`, `noise1`), `eph` (for `tunnel1`: the browser's ephemeral public key in the tunnel's `ephPub` format, standard base64 with padding, 44 characters), `kid` (an id of at most 64 characters), `prio` (0 or 1), `n` (a non-negative integer), `sig` (b64u of an Ed25519 signature, at most 88 characters; used on `ring` items). Any other key is `400 invalid_item`. The serialised header (compact JSON with slashes and non-ASCII unescaped, as UTF-8) is at most 512 bytes; the largest header the allow-list can express is 405 bytes, so the cap is defence in depth, but a relay MUST still count it. `ct` values: `sealed1` is a signed container sealed with `crypto_box_seal` (section 10.3); `tunnel1` is the existing NaCl `crypto_box` envelope, standard base64 of `nonce(24) || box`; `noise1` is reserved. The relay does not require a particular `ct` on a lane; the receiver checks it.

**Item states.** `queued` (stored, never returned), `delivered` (returned by at least one poll), `acked` (a later poll passed `since >= seq`), `expired`. On ack or expiry the relay deletes the `body` immediately and keeps the metadata (`id`, lane, sender, times, state, SHA-256 of the body) for 10 minutes so that duplicates are recognised and the sender can ask for the state.

**Idempotency.** `(mailbox, lane, sender, id)` is unique while metadata is retained: an id belongs to the device that chose it, so another sender's item of the same id is a different item and neither sees the other's `seq` or conflicts with it. A repeat by the same sender with an identical body hash is answered `duplicate` with the original `seq`; a repeat by the same sender with a different body is `conflict`. Delivery to a consumer is at least once; consumers de-duplicate on `id`.

**Epoch and reset.** The relay keeps a random 8 byte `epoch` (b64u, 11 characters), created at install and regenerated by a restore. Every poll response carries it; a client echoes the epoch it last saw. A poll whose `epoch` differs, or whose `since` is above the highest `seq` ever issued for the mailbox, is answered `reset: true` with the new epoch and the mailbox's highest `seq` as `cursor`, and no items. This catches a restored older backup even when new posts have already pushed a mailbox's counter past a client's saved cursor.

**Quotas.** Per inbox (`dev:` and `rbx:`): at most 512 live items and 8 MiB of live bodies. Of these, the bulk lanes (`ai`, `ai.in`, `ai.out`, `sync`) may hold at most 75 percent of the items and bytes (384 items, 6 MiB); at least 25 percent is reserved for `cmd`, `res`, `pair`, `ring` and `ctl`. Items the relay itself creates (`pair`, `ctl`) bypass the quotas. Per reply box: at most 2 `ai` items, 32 `ai.in` items and 2 MiB posted by the holder over its life. Per Aokie party mailbox: at most 1,024 live frames and 8 MiB, and no single sender may hold more than a quarter of either (256 frames, 2 MiB). Beyond a limit a post is `429 quota_exceeded` with `Retry-After: 5` (`429 relay_backpressure` on the compatibility routes). This is typed backpressure, never a silent drop.

## 4. Lanes, limits and who may post

`info.limits.lanes.<lane>` publishes `{"body": N, "ttl": {"default": D, "min": 1, "max": M}}`; the served `body` is the smaller of the configured cap and the largest body the calibration measured through the host's own stack. `body` counts bytes of the UTF-8 text. A relay MUST NOT widen a number past the maximum below; it may narrow one it can prove it must.

| Lane | Sender role to recipient | Max `body` | TTL default / max (s) | Body |
|---|---|---|---|---|
| `cmd` | `provider` (and, later, a `phone` holding the `cmd` flag) to `desktop` | 32,768 | 60 / 300 | `sealed1` container |
| `res` | `desktop` to the device that sent the `cmd` named in `hdr.re` | 98,304 | 300 / 3,600 | `sealed1` container |
| `ai` | reply box holder to `desktop` | 393,216 | 300 / 600 | `tunnel1` |
| `ai.in` | reply box holder to `desktop` | 65,536 | 300 / 600 | `tunnel1` |
| `ai.out` | `desktop` to the reply box | 393,216 | 360 / 900 | `tunnel1` |
| `pair` | the relay (on a phone's rendezvous response) to `desktop` | 16,384 | 900 / 900 | JSON text |
| `ring` | `desktop` to `phone`; call features | 4,096 | 30 / 300 | JSON text, signed in `hdr.sig` |
| `ctl` | `relay` or `desktop` to any device | 4,096 | 3,600 / 86,400 | JSON text |
| `sync` | `desktop` to `phone` | 65,536 | 21,600 / 86,400 | `json` or `sealed1` |
| `sig` | the Aokie plugin and phone, only through the compatibility routes; call features | 196,608 | 120 / 300 | `json` (later `noise1`) |

`flow`, `flow.in` and `flow.out` are reserved and not served in v1; a post naming them is `400 invalid_item`. A relay lists in `info.limits.lanes` every lane a client may post to through `POST /v1/items` that it serves (`cmd`, `res`, `ctl`, `sync`, and `ring` when call features are on); lanes reached only through reply boxes or the compatibility routes are listed when those routes are served.

**Who may post what** (enforced by the relay, `403 forbidden` otherwise):

1. `cmd`: a `provider` device to any `desktop` device of this relay. The relay does not track which provider is linked to which desktop: a personal relay has one owner, and what makes a desktop act is its own pinned provider keys, not the relay's table. A phone may post `cmd` only if the desktop set `canCmd` on that phone (default off).
2. `res`: only a `desktop`, only to the sender of the `cmd` whose id equals `hdr.re`, and only while the `cmd` metadata is retained (10 minutes after ack or expiry). A `res` without `hdr.re` is `400 invalid_item`.
3. `ai`, `ai.in`: only through `POST /v1/rbx/{rid}/items` with that reply box's secret; the recipient is the reply box's desktop. Through `POST /v1/items` they are `403 forbidden`.
4. `ai.out`: only the `desktop` named by the reply box.
5. `ring`, `sync`: only a `desktop` to a `phone` that this desktop approved (`ownerDesktop`). A `ring` MUST carry `hdr.sig`; the relay verifies it against the desktop's registered host identity key over `"oaiy/relay/1/ring" || 0x00 || body text exactly as posted` and answers `400 invalid_item` if it does not verify (the phone verifies again; the relay is not trusted). With call features off a `ring` is `403 feature_disabled`.
6. `ctl`: the relay itself, or a `desktop` to devices it approved.
7. `pair`: never by a client. The relay creates it.
8. A `provider` can never post to a `phone`, and a `phone` never to a `provider`.
9. `sig`: never through `POST /v1/items`; only through the compatibility routes.

**`ctl` messages** (`ctl.schema.json`; JSON text, at most 4,096 bytes, hints only; none grants authority and none is shown as trusted): `{"t":"provider.updated","id":"prov-..."}`, `{"t":"provider.rotated","b":"<b64u statement bytes>","s":"<b64u signature>"}`, `{"t":"device.revoked","id":"dev-..."}`, `{"t":"token.age","days":95}`, `{"t":"relay.notice","level":"info|warn","message":"<=200 chars"}`. A receiver ignores a `t` it does not know. A client renders `relay.notice` as plain text under the label "message from your relay (unverified)", with no links, no key-paste instruction and no formatting; a `ctl` message never starts an action by itself.

**`sync` items** carry a `sealed1` container with the domain `"oaiy/relay/1/sync"`, signed by the desktop host identity and sealed to the phone's X25519 key. The relay never looks inside.

**What does not belong in a mailbox:** media, push origination (the optional FCM sender is a separate add-on), bulk data, code, records and app data, anything that must survive more than 24 hours.

## 5. The HTTP API

Authenticated routes take `Authorization: Bearer <credential>`. "token" is a device capability token (section 9.1); "admission" an `aokie-adm-v2` bearer; "ticket" a JWS (section 9.2); "reply secret" the per-request secret of a reply box; "admin" the relay admin token.

| Route (POST form; alias in brackets) | Auth | Purpose | Schemas |
|---|---|---|---|
| `GET /v1/health` | none | liveness | `health` |
| `GET /v1/info` | none | static signed capability document; interactive identity proof with `X-OAIY-Nonce` | `info` |
| `GET /v1/admin/status[?diag=1]` | token (desktop) or admin | health, capacity, web-SAPI facts | `admin-status` |
| `GET /v1/admin/hold?wait=`, `GET /v1/admin/stream-probe`, `POST /v1/admin/echo`, `POST /v1/admin/capacity` | token (desktop) or admin | calibration | `admin-capacity-request`, `admin-capacity-response` |
| `POST /v1/enroll` | enrolment proof | redeem an enrolment key for a device and a token | `enroll-request`, `enroll-response` |
| `GET /v1/poll` | token | the multiplexed poll | `poll-response` |
| `POST /v1/items` | token | post up to 64 items | `post-request`, `post-response` |
| `GET /v1/items/{id}` | token | state of an item this device sent | `item-state` |
| `POST /v1/slots/{name}` (`PUT`) | token (desktop) | publish a slot with its reader list | `slot-request`, `slot-response` |
| `GET /v1/slots/{deviceId}/{name}` | token | read a slot the caller may read | `slot` |
| `GET /v1/presence` | token | who is online | `presence` |
| `POST /v1/devices/self/meta` (`PUT`) | token | name, version, capability tokens, public keys | `device-meta-request`, `ack` |
| `GET /v1/devices`; `POST /v1/devices/{id}` (`PATCH`); `POST /v1/devices/{id}/revoke` (`DELETE /v1/devices/{id}`); `POST /v1/devices/revoke` (`DELETE /v1/devices?role=phone`) | token (desktop) | list, rename, set flags or grants, revoke one, revoke every phone | `devices-list`, `device`, `device-patch-request`, `device-patch-response`, `devices-revoke-request`, `devices-revoke-response` |
| `POST /v1/roster` | token (desktop) | the desktop's authoritative roster for one app | `roster-request`, `roster-response` |
| `POST /v1/tokens/rotate` | token | new token, old one valid 10 more minutes | `token-rotate-response` |
| `POST /v1/keys` | token (desktop) | mint an enrolment key for a provider | `keys-request`, `keys-response` |
| `POST /v1/providers` | token (provider) | register signing and sealing keys and allowed origins | `providers-request`, `providers-response` |
| `POST /v1/devices/self/push` (`PUT`, `DELETE`) | token (phone) | register or remove an FCM token | `push-register-request` |
| `POST /v1/pair` | token (desktop) | open a pairing rendezvous | `pairing-create-request`, `pairing-create-response` |
| `GET /v1/pair/{pid}` | the `pid` itself | the phone fetches the offer and waits for the decision | `pairing-fetch-response` |
| `POST /v1/pair/{pid}/response` | the `pid` itself | the phone posts its signed response | `pairing-answer-request`, `pairing-answer-response` |
| `POST /v1/pair/{pid}/decision`, `.../reject`, `.../burn` | token (desktop) | approve or deny; reopen after a bad response; end the rendezvous now | `pairing-decision`, `pairing-reject-request` |
| `POST /v1/admission` (alias `POST /v1/aokie-companion/admission`) | token | Aokie admission for the plugin (desktop token) or a phone (phone token); call features | `admission-plugin-request`, `admission-plugin-response`, `admission-mobile-request`, `admission-mobile-response` |
| `GET /v1/aokie-companion/relay/challenge`, `POST` and `GET /v1/aokie-companion/relay/frames`, `GET /v1/aokie-companion/relay/stream` | admission | the Aokie compatibility routes; call features | `challenge`, `compat-frames-request`, `compat-frames-accepted`, `compat-frames-page`, `compat-error` |
| `POST /v1/replyboxes` | ticket | create a reply box | `replybox-request`, `replybox-response` |
| `POST /v1/rbx/{rid}/items`, `GET /v1/rbx/{rid}/poll`, `POST /v1/rbx/{rid}/end` (`DELETE /v1/rbx/{rid}`) | reply secret | send a request or input, read frames, finish | `rbx-item-request` |
| `OPTIONS` on the ticket and reply box routes | none | CORS preflight | |

### 5.1 `GET /v1/poll?since=&epoch=&wait=&limit=&maxBytes=&re=&peek=`

| Parameter | Default | Rule |
|---|---|---|
| `since` | 0 | cumulative, batched acknowledgement: one integer per poll acknowledges every item in this inbox with `seq <= since` and deletes its body; there is no per-item ack call. A negative, fractional or non-integer value is `400 invalid_request` |
| `epoch` | none | the epoch the client last saw; a mismatch is `reset: true` |
| `wait` | 0 | seconds the server may hold the request, at most `info.wait.max` (a larger value is clamped) |
| `limit` | 32 | 1 to 64 items; outside that range is `400 invalid_request` |
| `maxBytes` | 1,048,576 | 65,536 to 1,048,576; the first item is always returned even if larger; outside that range is `400 invalid_request` |
| `re` | none | return only items whose `hdr.re` equals this id; the request is a lookup; the response `cursor` equals `since` |
| `peek` | 0 | 1 returns items without acknowledging them or marking them delivered; the request is a lookup |

Response `200` (`poll-response.schema.json`): `{"v":1,"epoch":"<b64u>","cursor":42,"items":[...],"more":false,"time":1790000000,"hold":{"granted":true}}`. Items are in ascending `seq` and are only those with `exp > now` and not acknowledged. `more` is true when further items exist beyond `limit` or `maxBytes`; the client polls again at once. An empty result after a full wait is `200` with `items: []`. `cursor` is advisory: a client persists the highest `seq` of the items it accepted and ignores `cursor` on any non-empty response; only on `reset: true` does it adopt the server's `cursor` (which may be lower), once, and record "mailbox reset: in-flight items may be lost". A hostile `cursor` in a 200 with items therefore cannot move a client's position.

Three rules apply to consumer polls (a poll with neither `re` nor `peek`):

- **Gap.** A consumer poll whose `since` is not greater than the previous poll's `since` and which starts less than `info.wait.pollGapMs` (250) after the previous consumer poll of this device ended gets `429 rate_limited`, `Retry-After: 1`. A poll that follows a non-empty answer, or whose `since` advanced, is never refused by this rule (so a burst of three items posted together is three successful polls, not a 429 and a one second sleep).
- **Supersede.** A consumer poll from a device that already holds one ends the older within 250 ms with `items: []` and `hold.superseded: true`; it stops a client that timed out and retried from leaving a zombie held request behind.
- **Hold accounting.** The poll is counted in the hold registry (section 7.2) and may be answered as a `wait=0` poll with `hold.refused: true` and `retryAfter` when the pool is nearly full.

**Lookups.** A poll that carries `re` or `peek=1` is a lookup: a read of what is already in an inbox by a party that is not consuming it. A lookup never acknowledges (its `since` is only a lower bound), never marks an item delivered, is exempt from the gap rule, never supersedes and is never superseded, does not write presence, and counts against the `tok.req` bucket. A lookup with `wait=0` is allowed for any role on its own inbox. A lookup with `wait > 0` is allowed only for role `provider`, is clamped to `info.limits.lookupWait` (8 s), and one device may hold at most `info.limits.lookupHeld` (4) waiting lookups at once (a fifth is `429 rate_limited`, `Retry-After: 1`); a phone or a desktop asking for one is `403 forbidden`. A waiting lookup wakes on any post to that mailbox and answers only when an item with the requested `hdr.re` exists, otherwise `items: []` at its deadline.

### 5.2 `POST /v1/items`

Body `{"items":[{"to","lane","id","ttl","hdr","body"}, ...]}`, 1 to 64 items, at most 1 MiB. Response `200` (`post-response.schema.json`): `{"v":1,"results":[{"id":"...","status":"queued","seq":43},{"id":"...","status":"rejected","error":{"code":"quota_exceeded","message":"..."}}],"time":N}` with `status` one of `queued`, `duplicate` (carries the original `seq`), `rejected` (carries `error`). Errors that concern the request as a whole (`400`, `401`, `413` for a body over 1 MiB, `429`) use the normal error shape. Partial success is normal; a sender retries only the rejected items whose code is retryable (section 6).

### 5.3 `GET /v1/items/{id}?to=dev:<deviceId>&lane=<lane>`

Only for an item this device sent. `to` and `lane` are both required because an id is unique only per mailbox and lane. `200` (`item-state.schema.json`): `{"id","lane","to","state":"queued|delivered|acked|expired","seq","at","exp","deliveredAt":N,"ackedAt":N,"time"}`; `deliveredAt` and `ackedAt` are omitted until they happen. `404` otherwise.

### 5.4 Slots, presence, devices

**Slots.** `POST /v1/slots/{name}` with `{"body":"<at most 65,536 bytes>","ttl":120,"ct":"json","readers":["phone:caller_read","provider"]}`; `name` matches `[a-z0-9][a-z0-9._-]{0,63}`; TTL 1 to 3,600, default 120; only a `desktop` writes. `readers` is a list of at most 8 entries from `provider`, `desktop`, `phone` (any approved phone of this desktop) and `phone:<grant>` (an approved phone whose device row holds that grant); absent or empty means only the writer, and a name starting `private.` is writer-only whatever `readers` says. Response `200` (`slot-response.schema.json`) `{"etag":"\"<b64u>\"","exp":N,"time":N}` where the ETag is `"` + b64u(first 16 bytes of SHA-256(`body || 0x00 || ct`)) + `"`, `body` being the UTF-8 bytes of the body text exactly as posted (never decoded, whatever it holds) and `ct` the ASCII of the content type. `If-Match` gives compare-and-set (`412 precondition_failed`). `GET /v1/slots/{deviceId}/{name}` returns `slot.schema.json` with the same `ETag` header; `If-None-Match` gives `304`; an absent or expired slot is `404`; a caller who is not a listed reader is `403`; a revoked device is `401` at once. The reader list is set by the writer, which is the desktop, which is the authority for who may see what.

**Presence.** `GET /v1/presence` returns `presence.schema.json`. `online` means a consumer poll (not a lookup) is held now or started within `max(info.presenceWindow, info.wait.max + 5)` seconds. The ETag is a weak validator over the `devices` array only. A provider sees only desktops; a phone sees its desktop; a desktop sees all.

**Devices.** `POST /v1/devices/self/meta` (`device-meta-request.schema.json`) `{"name":"<=60","ver":"<=32","caps":["<=64 chars", at most 32],"ed25519":"b64u","x25519":"b64u"}` answers `ack.schema.json`; keys are recorded with a `keysChangedAt`. `GET /v1/devices` (desktop only, `devices-list.schema.json`, entries `device.schema.json`) lists the caller's own phones and providers. `POST /v1/devices/{id}` (desktop only; `device-patch-request.schema.json`, answer `device-patch-response.schema.json`) accepts `{"name"?, "flags"?, "grants"?}`; grants are patchable, so a takeover tier does not need a re-pair, and the next admission carries the new scopes. `POST /v1/devices/{id}/revoke` (desktop only; only that desktop's phones and any provider, never a desktop) answers `204` and, immediately and completely: revokes the device, revokes its tokens, purges its inbox and pending items, its party mailboxes, its slot access and its push registration, and any hold it has ends with `401 revoked` within 250 ms. `POST /v1/devices/revoke` `{"role":"phone"}` (`devices-revoke-request`, `devices-revoke-response`) does that for every phone of this desktop.

**Tokens.** `POST /v1/tokens/rotate` answers `200` `{"token":"oaiyrt1....","graceUntil":N,"time":N}` (`token-rotate-response`); the old token keeps working until `graceUntil` (now + 600). A second rotation during the grace is `409 conflict`.

**Keys and providers.** `POST /v1/keys` (desktop): `{"role":"provider","ttl":3600,"name":"FormLogic"}` gives `201 {"uri":"oaiy://enroll?...","kid":"...","exp":N,"time":N}` (`keys-request`, `keys-response`); the first desktop key comes from the installer or the admin CLI, never from the network. `POST /v1/providers` (provider token; `providers-request`): `{"name":"FormLogic","ed25519":"b64u","x25519":"b64u","origins":["https://app.example.com"]}`, at most 4 https origins with no path. `200 {"providerId":"prov-...","thumbprint":"...","time":N}`. Idempotent. A change posts a `ctl` item `{"t":"provider.updated","id":"prov-..."}` to every desktop of this relay; it changes nothing they trust until the owner re-pins.

**Roster.** `POST /v1/roster` (desktop token; `roster-request`): `{"appId":"aokie","revision":7,"thumbprints":["<sorted>", ...]}` with at most `limits.rosterMax` (16) thumbprints, possibly none. The relay validates (each 43 characters, strictly ascending bytewise, none equal to the desktop's own endpoint thumbprint), recomputes the roster hash, stores the row (desktop, appId), and revokes every phone whose `ownerDesktop` is this desktop, whose app is this app and whose thumbprint is not listed. `200 {"v":1,"hash":"...","revoked":["dev-..."],"time":N}` (`roster-response`). The revision is metadata: a value lower than the stored one from the same authenticated desktop is accepted (a reinstall restarts at 0), so there is no monotonic-revision denial of service; a revision above 2^53-1 is `400 invalid_request`. An unsorted roster, duplicates, more than 16 entries or the desktop's own thumbprint is `400 invalid_request`.

**Push registration.** `POST /v1/devices/self/push` `{"kind":"fcm","token":"<16 to 4096 printable characters>","project":"<optional>"}` gives `204`; `{"remove":true}` removes it (`push-register-request`).

## 6. Errors

Body (`error.schema.json`): `{"error":{"code":"...","message":"one sentence, safe to show a person","retryAfter":7}}`. The set of codes is closed. A client that meets an unknown code treats it by HTTP status. `retryAfter` (seconds) and the `Retry-After` header appear on 429 and 503; a 401 also sends `WWW-Authenticate: Bearer realm="oaiy-relay"`. `message` is at most 200 characters. There are no stack traces, no credentials and no internal paths, ever. A client shows a relay-supplied `message` only as data. An error body carries no `time`.

| HTTP | `code` | Meaning | Retry |
|---|---|---|---|
| 400 | `invalid_request` | malformed JSON, missing or ill-typed member, bad query value, a roster that is unsorted, has duplicates or is too large | no |
| 400 | `invalid_item` | lane, TTL, `hdr` key, `hdr.sig` or `id` failed validation | no |
| 401 | `unauthorized` | missing, wrong, expired or unknown credential (deliberately one code for all of these) | no; re-enrol or refresh |
| 401 | `revoked` | the device was revoked (only after the credential verified) | no |
| 403 | `forbidden` | authenticated but not allowed: role, lane, recipient, origin, scope, slot reader | no |
| 403 | `feature_disabled` | call features are off on this relay | no |
| 404 | `not_found` | unknown recipient, item, slot, pairing, reply box, route | no |
| 405 | `method_not_allowed` | | no |
| 409 | `conflict` | duplicate `id` with different content; second rotation; state precondition | no |
| 409 | `already_answered` | pairing rendezvous already has a response | no |
| 410 | `expired` | pairing, reply box or key past its lifetime | no |
| 412 | `precondition_failed` | slot `If-Match` mismatch | after a re-read |
| 413 | `item_too_large` | body over the lane maximum, or request over 1 MiB | no |
| 415 | `unsupported_media_type` | | no |
| 422 | `unprocessable` | well-formed but wrong: ticket lane or origin mismatch, reply box not for this desktop, a small-order key | no |
| 426 | `upgrade_required` | client level below `info.minClient` | after upgrade |
| 429 | `rate_limited` | a bucket in section 7 is empty, or the poll gap rule | yes, after `Retry-After` |
| 429 | `quota_exceeded` | mailbox full | yes, after the consumer drains |
| 500 | `internal` | | yes, backoff |
| 503 | `unavailable` | database busy or locked after retries, maintenance, host overload | yes, after `Retry-After` |

A refused hold is not an error: it is a `200` with `hold.refused` (section 7.2). The compatibility routes (section 10.6) additionally use FormLogic's codes so shipped clients behave as they do today: `invalid_token` (401), `relay_target_forbidden` (403), `relay_frame_too_large` (413), `relay_backpressure` (429), `companion_unavailable` (503). They appear in the Aokie error shape `{"error":true,"code":"...","message":"..."}` (`compat-error.schema.json`) on those routes only.

## 7. Rate limits and holds

### 7.1 Buckets

| Bucket | Key | Limit | On exceed |
|---|---|---|---|
| `ip.info` | address | `GET /v1/info`, `GET /v1/health`: 60 a minute (a nonce-bearing `info` counts) | 429 |
| `ip.pair` | address | `GET` and `POST /v1/pair/{pid}...`: 30 a minute; at most 4 pairing waits held at once | 429 |
| `ip.enroll` | address | `POST /v1/enroll`: 10 a minute | 429 |
| `ip.authfail` | address | 20 failed credential verifications a minute; a request whose credential verifies is never refused by this bucket | the failing request gets 429 with `Retry-After: 60` |
| `tok.req` | token id | bucket of 120, refill 10 a second: every authenticated request except consumer polls (lookups included) | 429 |
| `tok.items` | token id | bucket of 200, refill 20 a second: items posted | per-item `rate_limited` or 429 |
| `rbx.req` | reply box | bucket of 60, refill 5 a second: every request by a reply secret | 429 |
| `adm.req` | admission `jti` | bucket of 120, refill 10 a second: compatibility requests; the admission mint itself is 30 a minute per token | 429 |
| `pid.get` | pairing id | 60 GETs in its lifetime | 429 |
| `pid.resp` | pairing id | 3 responses accepted; a fourth is `409 already_answered` | |
| `kid.fail` | enrolment key | 5 failed proofs burn it | 401 |
| `tokid.fail` | (token id, address) | 20 wrong secrets an hour lock that id for that address for 15 minutes; the real device from another address is unaffected | 401 |

Clients honour `Retry-After` with an upper bound of 120 seconds and add up to 20 percent jitter. **Client address.** The key is `REMOTE_ADDR`, IPv6 reduced to its /64 and `::ffff:a.b.c.d` reduced to `a.b.c.d`. A forwarding header (for example `X-Forwarded-For`) is honoured only when `REMOTE_ADDR` is inside the configured trusted proxies, taking the rightmost address that is not itself a trusted proxy; otherwise it is ignored.

### 7.2 The hold registry

A hold is any request that may wait longer than zero seconds. A held request pins one worker for as long as it waits, so every hold of every kind is counted, superseded per principal and refusable.

| Hold | Principal | Class | Supersede | Wait cap |
|---|---|---|---|---|
| consumer poll, `GET /v1/poll` | device id | core if role `desktop`, otherwise edge | a newer consumer poll of the same device ends the older within 250 ms | `wait.max` |
| lookup with `wait > 0` (provider only) | device id | edge | none among lookups; at most 4 per device | 8 s |
| reply-box poll, `GET /v1/rbx/{rid}/poll` | `rid` | edge | one per box: a newer poll ends the older | `wait.max` |
| pairing wait, `GET /v1/pair/{pid}?wait=` | `pid` | edge | one per `pid` | `wait.max` |
| Aokie stream and frames wait | (desktop, appId, party) | core | one per party | 20 s |
| admin hold (calibration) | admin | exempt | none | 35 s |

With `W` the worker pool (`capacity.workers`, 5 until measured): `held_soft = max(2, floor(0.6 * W))` and `held_hard = max(3, W - 1)`. An edge hold is refused when the live count is at or above `held_soft`; a core hold when it is at or above `held_hard`. For polls, lookups, box polls, pairing waits and frame waits a refusal is a degradation, not an error: the relay runs the fetch once as if `wait=0` and answers `200` with `X-OAIY-Hold: refused` and `"hold":{"refused":true,"retryAfter":2}`; a client short-polls at `retryAfter` (2 s, then 4, 8, up to `info.wait.fallbackS`, 5) with jitter and asks for a hold again on every request. A `wait=0` request is never refused. For the stream route (a core hold) refusal at the hard limit is `503 companion_unavailable` with `Retry-After: 2`. One credential cannot pin more than: a phone token, its inbox poll and its stream (two); a reply secret, one; a `pid`, one, and at most four pairing waits per address; a provider token, four short lookups; a desktop token, its own poll and the plugin's stream.

## 8. Versioning, capability negotiation and the identity proof

`GET /v1/info` (`info.schema.json`; no authentication; `Cache-Control: public, max-age=60` without a nonce; strong ETag; `If-None-Match` gives `304`) is a static document:

```json
{"protocol":"oaiy-relay/1","minClient":1,"relayId":"rly-...","relayKey":{"algorithm":"ed25519","publicKey":"...","thumbprint":"..."},
 "software":{"name":"oaiy-relay","version":"0.1.0"},"features":["poll","items","presence","methods.post-forms"],
 "wait":{"default":20,"max":20,"pollGapMs":250,"fallbackS":5},"presenceWindow":60,
 "limits":{"batchItems":64,"batchBytes":1048576,"mailboxItems":512,"mailboxBytes":8388608,"bulkShare":0.75,"hdrBytes":512,
           "held":{"soft":3,"hard":4,"measured":false},"lanes":{"cmd":{"body":32768,"ttl":{"default":60,"min":1,"max":300}}}}}
```

The full example of the design, with every optional member, is `A6b` in `vectors.json`. The body has no `time` member, so it is static and its signature can be cached. `features` lists only what the relay implements and has enabled: `poll`, `items`, `slots`, `presence`, `replyboxes`, `tickets`, `pairing.v3`, `methods.post-forms`, and, only when the owner has enabled call features, `call`, `admission.aokie-adm-v2`, `compat.aokie-companion-relay` (with the `ring` and `sig` lanes and `call.*` slots); `compat.sse-framed-poll` only when the streaming probe passed on this host; `push.fcm` only when configured. `limits.held` shows the soft and hard hold limits and whether the pool was measured.

1. **Static signature (integrity and feature discovery, not identity).** The response carries `X-OAIY-Sig: <b64u Ed25519 signature over "oaiy/relay/1/info" || 0x00 || exact body bytes>`. It proves the bytes were once signed by the relay key; anyone can copy the body and the signature and serve them, so it proves nothing about who is answering now.
2. **Interactive proof (identity).** A request carrying `X-OAIY-Nonce: <b64u of 16 to 32 random bytes>` (22 to 43 characters; any other length is `400 invalid_request`) is answered with the same body, `Cache-Control: no-store`, and `X-OAIY-Proof: b64u(Ed25519(relay key, "oaiy/relay/1/info-proof" || 0x00 || nonce bytes || SHA-256(body) (32 bytes) || ASCII decimal of the X-OAIY-Time value))`. A client verifies that `relayKey.publicKey` hashes to the thumbprint it pinned (`f` in the enrolment or pairing key, or `offer.relay.fingerprint` for a typed pairing), that the proof verifies over its own nonce, and that `X-OAIY-Time` is the value the proof covers. Vectors `A6` and `A6b`: a replayed body and static signature with a new nonce must fail.
3. **When a client proves.** Before enrolment, before the phone's pairing, at the provider's link, and before sending a bearer on a connection it has not proved: at process start, after any failure or backoff of 60 s or more, after a network change, and at least every 300 s while polling. A client that cannot obtain a valid proof does not send its token and reports the relay as "not who it was".
4. **What the proof does and does not defeat.** It defeats a replaced server (a lapsed or hijacked name pointing at a host that never held the relay key). It does not detect a live forwarding proxy that holds a valid certificate for the relay's name and relays the challenge to the real relay: there is no channel binding to TLS.
5. Clients cache the static `info` for 10 minutes and re-read it on start and on a `426`.

## 9. Authentication

### 9.1 Device capability tokens

1. **Generation.** `id` = 8 random bytes, `secret` = 32 random bytes, both from the operating system CSPRNG. Token string: `"oaiyrt1." + b64u(id) + "." + b64u(secret)`, always 63 characters. Vector `A2`.
2. **Storage.** The relay stores `SHA-256(secret bytes)` (or `HMAC-SHA-256` with an optional server pepper), the id, the device, and times; nothing else. The secret bytes are the 32 raw bytes that the 43-character secret decodes to, not the 43 characters (vector `A2`, `secretSha256`). A copy of the database yields no credential.
3. **Verification, in this order, with the same `401 unauthorized` and body for every failure:** (a) strict parse (prefix, two dots, lengths 11 and 43, base64url alphabet, canonical last characters); (b) look up by `id`; if absent, hash a dummy secret and compare, so timing does not reveal which ids exist; (c) constant-time comparison of the stored hash with `SHA-256(secret)`; (d) `revoked_at` null and `not_after` in the future (or null); (e) the device not revoked; (f) the route's role and scope allow it. A token whose secret verified but whose device or token was revoked is `401 revoked`. Every failure counts against `ip.authfail` and `tokid.fail`.
4. **Carriage.** `Authorization: Bearer` only, with the scheme matched case-insensitively and one space. The relay reads it from `HTTP_AUTHORIZATION`, `REDIRECT_HTTP_AUTHORIZATION` and `getallheaders()`, in that order, because FastCGI-family stacks drop the header unless the vhost passes it. It never reads a URL or a cookie and never logs the header. `GET /v1/health` reports `authHeaderSeen`. A missing header is counted separately from a wrong token in admin status; on the wire both are the same `401`.
5. **Rotation and revocation:** section 5.4. Rotation every 90 days is recommended.
6. **Scope.** One token belongs to exactly one device and role; the route table in section 5 fixes what each role may call.
7. **Send discipline.** A client sends the token only after a fresh identity proof (section 8).

Replay protection is by layer: a request is protected by TLS (item posts are idempotent on `id`, polls are reads); an item at the relay by `(mailbox, lane, id)` uniqueness for 10 minutes after ack or expiry; an item at the desktop by a persisted replay cache of authenticated ids retained for `(exp - iat) + 120` seconds in provider time; a pairing response by one-use `nonce` and `jti`, one accepted response and a transcript-bound MAC; a ticket by a single-use `jti`; an admission by its 90 second life.

### 9.2 Provider tickets: JWS, EdDSA

A ticket lets a browser reach the relay after a provider has vouched for it. It is a JWS Compact Serialization (RFC 7515) with `alg` `EdDSA` over Ed25519 (RFC 8037). The signature is over the ASCII bytes of `b64u(header) + "." + b64u(payload)` exactly as transmitted; nothing is canonicalised. Vector `A1` reproduces the RFC 8037 appendix A.4 example as an anchor; `A9` is a complete ticket.

Protected header (`ticket-header.schema.json`): `{"alg":"EdDSA","typ":"oaiy-ticket+jwt","kid":"<thumbprint of the provider's Ed25519 key>"}` and nothing else (`crit`, `jku`, `jwk`, `x5u`, `x5c` are refused). Claims (`ticket-claims.schema.json`): `iss` the provider id, `aud` the relay id (a ticket for another relay is refused), `sub` an opaque member reference of at most 64 characters with no personal data, `iat` and `exp` (`exp - iat <= 300`), `jti` an id of at most 64 characters (single use), `lane` (`ai` in v1), `dev` the desktop's device id the browser may reach, `org` the browser origin, `eph` b64u of SHA-256 of the browser's ephemeral X25519 public key.

The relay verifies, and any failure is `401 unauthorized` (or `403`/`422` where noted): header `alg` exactly `EdDSA` (so `none`, `HS256` and every other algorithm confusion is refused), no `crit`, `jku`, `jwk`, `x5u`, `x5c`; `kid` matches a registered provider; the signature verifies; `aud` equals this relay; `iat <= now + 30` and `exp + 30 > now` and `exp - iat <= 300`; `lane` allowed; `dev` is an active desktop of this relay (`422` otherwise); the request's `Origin` header equals `org` and is one of the provider's registered origins (`403` otherwise); `jti` not seen. A ticket is single-use for creating one reply box. The desktop performs its own, independent verification of the same ticket.

### 9.3 The other credentials

- **Admission bearer** (`aokie-adm-v2`): section 10.6.
- **Reply secret**: 32 random bytes chosen by the browser, b64u (43 characters), sent as `Authorization: Bearer <secret>` to `/v1/rbx/{rid}/...`; the relay stores SHA-256 of it with the reply box; lifetime is the box's TTL (default 900 s), ended earlier by `DELETE`.
- **Pairing rendezvous**: the `pid` itself is the capability (128 bits, derived from the secret, section 10.1).
- **Enrolment key**: section 10.2.
- **Admin token**: `oaiyadm1.` + b64u(8 bytes) + `.` + b64u(32 bytes); the relay stores a hash; it opens the status page and the calibration routes and nothing else (it cannot mint a desktop key).

### 9.4 Clocks: which clock judges what

| Check | Window | Whose clock |
|---|---|---|
| ticket `iat`/`exp` at the relay (advisory only) | 30 s | relay |
| admission verification at the relay | 30 s | relay |
| command envelope `iat`/`exp` at the desktop | `[iat - 30, exp + 30]` | provider clock |
| ticket `iat`/`exp` at the desktop | `[iat - 30, exp + 30]` | provider clock |
| replay cache retention | `(exp - iat) + 120 s` from first sight, an absolute provider-time deadline `retain_until = exp + 60` and never shortened | provider clock |
| pairing offer and response windows | 30 s | relay-corrected time at both ends |
| enrolment key expiry | 0 | relay |
| ring `expiresAt` on a relay-profile phone | 0 | relay-corrected time |

The provider clock is the desktop's offset to the provider, taken from the provider's own TLS responses (the HTTP `Date` header or a `serverTime` member), median of the last five samples, slewed by at most 1 second a minute, applied to a monotonic local clock. An offset older than 150 s is `clock_unknown`; `|offset| > 60 s` is `clock_mismatch`; in either case signed items are refused and nothing signed is pruned.

## 10. Formats

### 10.1 Pairing v3 (a phone without FormLogic)

**Secret.** `s` is 16 bytes from the desktop's CSPRNG, used once, never sent to the relay.

**Pairing key** (QR or deep link): `oaiy://pair?v=3&u=<relay base URL, percent-encoded>&f=<relay key thumbprint, 43 chars>&s=<b64u of s, 22 chars>&x=<expiry, Unix seconds>` (`common#pairingUri`). Parse rules: scheme and host exactly as shown; `v` must be `3`; `u` an https URL with no userinfo, query or fragment; `s` exactly 22 characters decoding to 16 bytes; `f` and `x` optional; unknown parameters ignored; total length at most 512. A deep link or a scanned QR never starts pairing by itself: it opens a full-screen confirmation, and it never replaces or activates an existing relay profile without a second confirmation.

**Typed code** (`common#typedCode`): 28 characters: `s` as Crockford base32 (26 characters carrying 128 bits, the last two bits zero), then a 2 character check = the Crockford base32 encoding of the first 10 bits of `SHA-256("oaiy/pairing/3/typed" || 0x00 || s)` (`s` being its 16 raw bytes), shown in seven groups of four separated by dashes. Input is normalised (upper case, `I` and `L` read as `1`, `O` as `0`, dashes and spaces removed, `U` refused) and the check is verified locally before any network call. The typed route also asks for the relay's address (the host name), because the code carries only `s`; the phone builds `https://<host>`, fetches the offer from it and requires `offer.relay.url` to equal that URL and the relay's fresh info proof to be made by the key whose thumbprint equals `offer.relay.fingerprint`.

**Derivations** (HKDF-SHA256, RFC 5869; `salt` is the UTF-8 string `oaiy/pairing/3`; `IKM` is `s`): `pid = HKDF(info = "rendezvous", L = 16)`, which is 16 raw bytes; it is written as b64u (22 characters) in URLs and JSON, and `A3` records both spellings (`pid`, `pidHex`); `mac_key = HKDF(info = "mac", L = 32)`, which never leaves the two endpoints. Vector `A3`.

**Offer** (`pairing-offer.schema.json`): canonical JSON with exactly the members `kind` (`aokie_mobile_pairing`), `schemaVersion` (`3`), `appId`, `desktopConnectionId` (the desktop's relay `deviceId`), `desktopName` (at most 60 characters, control characters removed), `desktopEndpointKey`, `desktopX25519`, `hostIdentity` (`{"ed25519","thumbprint","x25519"}`: the phone pins them), `nonce` (b64u of 32 random bytes), `jti` (`pair-` + an id), `issuedAt`, `expiresAt` (`expiresAt - issuedAt = 600`), `relay` (`{"url","fingerprint"}`). The offer is stored on the relay as the exact text plus `mac = b64u(HMAC-SHA256(mac_key, "oaiy/pairing/3/offer-mac" || 0x00 || offer text bytes))`. The phone verifies the MAC over the received text and does not re-serialise it. The offer of `A3` is 778 bytes.

**Response** (`pairing-response.schema.json`): `{"kind":"aokie_mobile_pairing_response","schemaVersion":3,"claims":{...},"signature":"<b64u>","mac":"<b64u>"}`. `claims` (`pairing-claims.schema.json`, canonical form for signing) holds `appId`, `desktopConnectionId`, `desktopKeyThumbprint`, `deviceId`, optional `displayName` (at most 60 characters), `mobileEndpointKey`, `mobileX25519`, `pairingNonce`, `jti`, `issuedAt`, `expiresAt` (at most `issuedAt + 120`). `signature` is Ed25519 by the phone's endpoint key over `"oaiy/pairing/3/response" || 0x00 || canonical(claims)`; `mac` is HMAC-SHA256 under `mac_key` over `"oaiy/pairing/3/response-mac" || 0x00 || canonical(claims)`. Both are required.

**Short authentication string.** `sas_raw = HKDF-SHA256(IKM = desktopEd25519Pub || phoneEd25519Pub (64 bytes), salt = nonce bytes, info = "oaiy/pairing/3/sas" || 0x00 || pid, L = 8)`, where `desktopEd25519Pub` is the 32 raw bytes of the desktop endpoint key (`offer.desktopEndpointKey.publicKey` decoded), `phoneEd25519Pub` the 32 raw bytes of the phone endpoint key (`claims.mobileEndpointKey.publicKey` decoded), `nonce bytes` the 32 raw bytes of the pairing nonce (`offer.nonce` decoded), and `pid` the **raw 16 bytes** of the rendezvous id, not its 22-character b64u text and not hex: `info` is the 18 ASCII bytes of `oaiy/pairing/3/sas`, one zero byte and those 16 bytes, 35 bytes in all. Reading `pid` as the 22-character text makes a 41-byte `info` and a different, wrong SAS; `extras.sasNegative` in `vectors.json` records what that wrong reading produces for the inputs of `A3` (`24b574fd2e0d1e24` where the right value is `3563599914bdf7f9`), so that an implementation can prove it is not making that mistake (Interpretation 21). Take the top 60 bits of the 8 bytes and write them as 12 Crockford base32 characters. The phone shows 13 characters: the 12 and one check character `C = Crockford32[ SHA-256("oaiy/pairing/3/sas-check" || 0x00 || the 12 characters as ASCII)[0] >> 3 ]`, grouped `XXXX-XXXX-XXXX-C` (`common#sasCode`). The check is a public function of the twelve; it exists so that a single mistyped character is caught locally and does not cost one of the owner's three attempts.

**Approval receipt** (`approval-receipt.schema.json`). On approval the host signs `Ed25519(desktopEndpointKey, "oaiy/pairing/3/approval" || 0x00 || canonical({"appId","grants" (sorted),"issuedAt","phoneThumbprint","pid"}))` (here `pid` is a JSON string holding the 22-character b64u text) and sends it in the decision (`pairing-decision.schema.json`, member `receipt` `{"issuedAt","signature"}`). The relay stores it and returns it beside `sealedToken`; the phone verifies it with the desktop key it pinned from the MAC'd offer before storing a profile.

**The exchange.** (1) `POST /v1/pair` (desktop token; `pairing-create-request`: `pid`, `offer` text of at most 4,096 bytes, `mac`, `ttl`, `appId`, `desktopThumbprint`) opens a rendezvous; the relay parses the offer text only to check that its `appId` and `desktopEndpointKey.thumbprint` equal the request's two fields, stores the text unchanged, answers `201 {"pid","exp","time"}`, and `409 conflict` if `pid` exists. (2) `GET /v1/pair/{pid}` (no credential) answers `pairing-fetch-response`; an unknown or expired `pid` is one `404 not_found` with an identical body and timing. (3) The phone shows the owner the relay host and the desktop's name before anything is posted. (4) `POST /v1/pair/{pid}/response` `{"response":"<JSON text, at most 8192>"}` answers `202 {"state":"answered","time":N}`; a second post is `409 already_answered`; the relay posts one `pair` item (`id` = `pid`, body = the response text) to the desktop inbox; the phone waits with `GET /v1/pair/{pid}?wait=20&state=answered`. (5) The desktop verifies (shape, identifiers, lifetime against relay time with 30 s skew, nonce and `jti` not consumed, live challenge, binding to the offer, key not revoked, `mobileX25519` of 32 bytes and not of small order, the MAC in constant time, and the signature last with strict verification) and on failure calls `POST /v1/pair/{pid}/reject {"reason":"..."}`, which returns the rendezvous to `open` while fewer than 3 rejects have happened. (6) The owner types the 13 character code; three wrong 12-character entries deny the pending item and burn the rendezvous; an incomplete entry or a wrong check character is not an attempt. (7) `POST /v1/pair/{pid}/decision` (desktop token) `{"approve":true,"phone":{"ed25519","x25519","thumbprint"},"name","appId","grants":[...],"receipt":{"issuedAt","signature"}}` or `{"approve":false}`; the default grants exclude `monitor`, `consult` and `takeover`. (8) On approval the relay rejects a small-order phone X25519 key (`422`), creates the phone device, generates a token, and stores `sealed = crypto_box_seal(token, phoneX25519)` (libsodium `sodium_crypto_box_seal`); the plaintext token exists in the request's memory and nowhere else. (9) The phone's poll returns `{"v":1,"state":"approved","deviceId","sealedToken","receipt":{...},"time"}` (or `{"state":"denied"}`); it verifies the receipt, opens the box, stores the profile, and pins the desktop endpoint key and the host identity from the MAC-verified offer as the only source of its peer pin. The rendezvous is deleted 10 minutes after the phone read a terminal state, or at expiry.

Relay-side states: `open -> answered -> approved | denied`; `answered -> open` by `reject` (at most 3 times, then `expired`); `open | answered -> expired` at `exp` (at most 900 s after creation). Limits: offer 4,096 bytes, response 8,192 bytes, 3 accepted responses, 60 GETs, lifetime 900 s, 16 pending approvals at once.

### 10.2 Enrolment: the desktop and the provider

**Enrolment key** (`common#enrollUri`): `oaiy://enroll?v=1&u=<relay base URL>&f=<relay key thumbprint>&k=<kid, 11 chars>&s=<b64u of s, 22 chars>&r=<desktop|provider>&x=<expiry>`. `s` is 16 random bytes. With `salt = "oaiy/enroll/1"` and `IKM = s`: `kid = b64u(HKDF(info="id", L=8))`, and `seed = HKDF(info="sig", L=32)` is an Ed25519 seed. The relay stores the derived Ed25519 public key only, so a copy of the database yields nothing redeemable. A key is single-use, valid for one hour by default and at most 24. A relay admits at most `limits.desktops` (default 2) desktop devices.

**Redeeming.** `POST /v1/enroll` (`enroll-request.schema.json`) body `{"kid":"...","role":"desktop","name":"<=60","n":"<b64u of 16 random bytes>","keys":{"ed25519":"<b64u>","x25519":"<b64u>"}}` and the request header `X-OAIY-Proof: b64u(Ed25519(seed, "oaiy/relay/1/enroll" || 0x00 || raw request body bytes))`. The secret itself is never transmitted. The relay verifies the signature over the bytes it received, marks the key used with one conditional update (so two racing requests cannot both win), creates the device and its token and answers `201` (`enroll-response.schema.json`) `{"deviceId":"dev-...","token":"oaiyrt1....","relayId":"rly-...","time":N}`. Every failure (unknown, used, expired, wrong role, bad proof) is the same `401 unauthorized`; five failed proofs against one `kid` burn it; a small-order X25519 or Ed25519 key is `422`. The client verified a fresh identity proof before sending anything. Vector `A7`.

**Provider registration and the desktop's pin.** After redeeming, a provider calls `POST /v1/providers` with its origins. The desktop does not take the relay's word for the provider's keys: it fetches them from the provider over the existing authenticated TLS link, compares them with what the relay registered, shows the owner the fingerprint, rejects a small-order key, and stores the pin only after the owner confirms.

**Provider key rotation.** A later change of provider keys arrives as a `ctl` hint and is accepted without asking only if it carries a rotation statement signed by the currently pinned Ed25519 key. The statement (`rotation-statement.schema.json`) is shipped bytes: `{"v":1,"prev":"<thumbprint of the key being replaced>","new":{"ed25519":"...","thumbprint":"...","x25519":"..."},"serial":n,"iat":N,"exp":N}` and `signature = Ed25519(K_prev, "oaiy/relay/1/provider-rotate" || 0x00 || bytes)` (`bytes` being the statement's text bytes, the decoded content of `b` below), delivered as `ctl` `{"t":"provider.rotated","b":"<b64u bytes>","s":"<b64u signature>"}`. The desktop accepts it only when `prev` equals the thumbprint it has pinned, `serial` is greater than the last accepted serial for that provider, `exp - iat` is at most 24 hours, the time window holds in provider time, and the signature verifies with strict verification. Vector `A10`.

### 10.3 Authority envelopes: commands and results

Two standard operations are combined: an Ed25519 signature for authority and a libsodium sealed box (`crypto_box_seal`) for confidentiality to the desktop's published X25519 key. Every place that takes a peer's public key for a Diffie-Hellman or a signature refuses one of small order (vector `A12`): for X25519, after masking bit 255, a value that reduced modulo p = 2^255 - 19 is 0, 1, p-1, or one of the two order-8 values; equivalently any key whose shared secret would be all zero. For Ed25519, a pinned or registered public key of small order is refused, and every verification against a pinned key uses strict verification.

Signed bytes are shipped, not re-derived:

```
container = {"k":"<thumbprint of the signer's Ed25519 key>","b":"<b64u of the exact signed bytes>","s":"<b64u Ed25519 signature>"[,"p":"<b64u padding>"]}
signature = Ed25519(signer key, "oaiy/relay/1/cmd" || 0x00 || bytes)        (results: "oaiy/relay/1/res"; sync: "oaiy/relay/1/sync")
            where bytes = the exact signed bytes, i.e. the decoded content of "b", not the b64u text of "b"
sealed    = crypto_box_seal(container as UTF-8 JSON text, recipient X25519 public key)
item      = {to, lane:"cmd"|"res", id, ttl, hdr:{"ct":"sealed1"}, body: b64u(sealed)}
```

`container.schema.json` allows only `k`, `b`, `s`, `p`. The receiver verifies the signature over the decoded bytes and only then parses them. The optional member `p` is random padding that a signer SHOULD add so the container's length is a multiple of 256 bytes; verifiers ignore it. Vector `A8` (the sealing step is randomised and has no fixed vector).

**Command document** (`command.schema.json`; the bytes; the signer is the provider): `v` (1), `id` (the command id; equals the item `id` and any `hdr.re` on the result), `dev` (the desktop's relay device id), `to` (the desktop's provider instance id), `connector`, `command`, `payload` (an object of at most 16 KiB), `idem` (the provider's idempotency key), `iat`, `exp` (signer's clock; `exp - iat <= 300`), `src` (the provider id), `uid` (opaque member reference, at most 64 characters, audit only), and `dep`, which is reserved: a v1 signer never sets it and a v1 verifier refuses a document that carries it (`failed/unsupported`).

**Result document** (`result.schema.json`): `{"v":1,"re":"<command id>","dev":"<desktop device id>","status":"done"|"failed","result":{...}|null,"error":{...}|null,"at":N}`, signed by the host identity and sealed to the provider's X25519 key. A `done` result has a null `error`; a `failed` result has an `error` object with a `code`.

**Verification on the desktop, in this order.** Any step failing before step 4 completes drops the item; afterwards a coded, signed failure result is returned. (1) Lane `cmd`, `hdr.ct` is `sealed1`, body decodes, size within the lane cap. (2) Open the sealed box with the host's X25519 identity. (3) Parse the container; `k` must equal the thumbprint of a pinned provider signing key. (4) Verify the signature over the decoded bytes strictly. From here the item is authenticated. (5) Parse the bytes: `v` is 1; `id` equals the item id; `dev` and `to` are this desktop; `exp - iat <= 300`; no `dep`. (6) Time, in provider time: `clock_unknown` or `clock_mismatch` return `failed` with that code; outside `[iat - 30, exp + 30]` returns `failed`/`expired`. (7) Replay and ledger: only now look the `id` up in the persisted cache and, if absent, add it before executing; a repeat after completion resends the stored result. (8) Authorisation: the per-connector allow-list decides; call-control commands MUST carry `payload.callId`, a missing id is `failed`/`call_id_required`. (9) Execute with the idempotency key `relay-command-<id>`. (10) Sign the result, seal it to the provider's X25519 key, post it as a `res` item (`id` = `res-<command id>`, `hdr.re` = the command id, TTL 300 s).

### 10.4 Sealed chat through a reply box and a ticket

The chat envelope is the existing NaCl `crypto_box` tunnel. (1) The browser asks the provider for a ticket. (2) `POST /v1/replyboxes` with `Authorization: Bearer <ticket>` and `{"rid":"<b64u 16 bytes>","h":"<b64u SHA-256 of a 32-byte secret the browser keeps>","ttl":900}` (`replybox-request`) answers `201 {"rid","exp","time"}` (`replybox-response`); the relay limits live boxes to 2 per ticket `sub` and 8 per desktop. (3) `POST /v1/rbx/{rid}/items` (Bearer reply secret; `rbx-item-request.schema.json`) with lane `ai` or `ai.in`, `hdr` `{"ct":"tunnel1","eph":"<standard base64 ephPub>"}` and the standard-base64 envelope as `body`. (4) The desktop opens the envelope, requires the ticket inside it, verifies it independently in provider time, and posts sealed frames to `rbx:<rid>` as lane `ai.out`. (5) `GET /v1/rbx/{rid}/poll?since=&wait=20` returns frames with the poll semantics of section 5.1; `POST /v1/rbx/{rid}/end` ends the box. (6) Stream integrity: the browser accepts a desktop frame only when its counter is exactly the previous counter plus one, and the terminal state comes only from a sealed `final` or `error` frame.

**CORS.** Only the ticket and reply box routes answer cross-origin requests. `OPTIONS` returns `204` with `Access-Control-Allow-Origin: <the exact request origin, only if it is in the registered origins of some provider>` (never `*`), `Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS`, `Access-Control-Allow-Headers: Authorization, Content-Type, X-OAIY-Client`, `Access-Control-Max-Age: 600`, `Vary: Origin`; no `Access-Control-Allow-Credentials`. Every other route sends no CORS headers at all. If an `Origin` header is present on a reply-box request it must be in the list (`403 forbidden` otherwise); an absent `Origin` is accepted on `/v1/rbx/*` and never on `POST /v1/replyboxes`.

### 10.5 Ring hints

A `ring` item's `body` (`ring.schema.json`) is a JSON object whose members are exactly the FCM `data` map, every value a string:

| `aokieClass` | Required members (all strings) |
|---|---|
| `voice_offer` | `schemaVersion` `"1"`, `eventId`, `offerId`, `appId`, `callId`, `callEpoch`, `ownerEpoch`, `expiresAt` |
| `voice_offer_cancel` | `schemaVersion` `"1"`, `eventId`, `offerId`, optional `reason` (1 to 120 characters, no control characters) |
| `assistance_offer` | `schemaVersion` `"1"`, `eventId`, `appId`, `requestId`, `callId`, `callEpoch`, `ownerEpoch`, `expiresAt` |
| `informational` | `schemaVersion` `"1"`, `eventId`, `title` (at most 80), `body` (at most 240), `expiresAt` |

Ids match `[A-Za-z0-9_.:-]{1,200}`; `callEpoch` is 1 to 2^53-1; `ownerEpoch` 0 to 2^53-1; `expiresAt` is Unix seconds, later than now and at most now + 300 (86,400 for informational); no other members. **Signature.** A `ring` item carries `hdr.sig = b64u(Ed25519(host identity, "oaiy/relay/1/ring" || 0x00 || body text exactly as posted))`. A relay-profile phone drops a ring whose signature does not verify against the host identity it pinned at pairing. `expiresAt` is judged in relay-corrected time. Push and ring data are wake hints and never authority: no SDP, ICE, token, caption or caller number. Vector `A11`.

### 10.6 The admission issuer and the Aokie compatibility routes

Present only when call features are enabled (`403 feature_disabled` otherwise).

**Token.** `aokie-adm-v2.` + lowercase hex of the claims JSON + `.` + lowercase hex of `HMAC-SHA256(admission secret, exact claims JSON bytes)`. Claims (`admission-claims.schema.json`; in this order): `aud` = `aokie-v2-gateway`, `appId`, `subjectId`, `role` (`mobile` or `plugin`), `holderKeyThumbprint`, then for a mobile `expectedPeerKeyThumbprint`, for a plugin `approvedPeerKeyThumbprints`, `peerRosterRevision`, `peerRosterHash`, then `scopes`, `dsk` (the device id of the desktop the party belongs to; read only by the relay), `exp`, `jti` (`adm_` plus 32 hex characters). TTL 90 seconds by default and never more than 300. Vectors `A4` (888 characters for a phone) and `A4b` (a plugin token is 964 characters for one phone and grows by 92 per phone: 2,344 for sixteen). The roster hash is `b64u(SHA-256("aokie/v2/peer-roster" || 0x00 || {"approvedPeerKeyThumbprints":[...sorted],"peerRosterRevision":N}))` in canonical form; vector `A0` checks it against the Aokie README's own example.

**`POST /v1/admission`** (alias `POST /v1/aokie-companion/admission`; 30 a minute per token). Plugin admission (desktop token; `admission-plugin-request`): validation in order: `appId` and `pluginId` are safe ids; `endpointPublicKey` is `{algorithm:"ed25519", publicKey, thumbprint}` with a canonical 32-byte key of no small order and a thumbprint that recomputes, and `holderKeyThumbprint` equals it; `approvedPeerKeyThumbprints` has 1 to 16 entries, each a valid thumbprint, strictly ascending, none equal to the holder; `peerRosterRevision` is an integer from 1 to 2^53-1; `peerRosterHash` recomputes. The admission changes nothing in the roster registry. Issued with role `plugin`, scopes `["state_read","rtc_signal"]`, TTL 90. Response (`admission-plugin-response.schema.json`): exactly `accessToken`, `tokenType` `Bearer`, `expiresIn`, `expiresAt`, `gatewayUrl` (a syntactically valid `wss://<relay host>/v2/realtime` that is never dialled), `appId`, `subjectId`, `role`, `scopes`, `device`, `iceServers`, `relayOnly`, `turnCredentialExpiresAt`, `endpointPublicKey`, `holderKeyThumbprint`, `approvedPeerKeyThumbprints`, `peerRosterRevision`, `peerRosterHash` and `relay` (`{"challengeUrl","framesUrl","streamUrl"}`, plus `"mode":"poll"` for a poll-mode carrier), and no other member: the plugin's decoder is strict. It never emits `desktopConnection` or `scopeCompatibility`. The three relay URLs are byte-identical in every admission and carry no per-admission query string. Mobile admission (phone token; `admission-mobile-request`, `admission-mobile-response`): the token's device must be a phone; `deviceId` must equal it; `holderKeyThumbprint` must equal the phone's registered thumbprint; `appId` must equal the pairing's; the roster row, when it exists, must list the phone's thumbprint (otherwise `403 forbidden`); `supportedTransports` must include a transport the relay can serve, otherwise `422 unprocessable`. A phone treats the relay-supplied `expectedPeerKeyThumbprint`, `iceServers`, `scopes` and `device` as advisory: its desktop pin comes only from the MAC-verified offer, and a different expected peer is a hard error.

**ICE and TURN** (`ice-server.schema.json`). Each TURN entry is `{"urls":[...],"username":"<exp>:<subjectId>","credential":"<base64(HMAC-SHA1(secret, username))>","expiresAt":<exp>}` with `exp` 31 seconds to 24 hours ahead; STUN-only entries carry `urls` and nothing else; `turnCredentialExpiresAt` is the earliest TURN `expiresAt` or `null`; `relayOnly` needs at least one TURN entry. Vector `A5`.

**Aokie mailbox routes.** The party comes only from the verified admission: role `plugin` is `plugin`; a mobile is `mobile:<holderKeyThumbprint>`. Mailboxes are `app:<appId>@<dsk>/<party>`. Every request re-checks the device row. `GET /v1/aokie-companion/relay/challenge` returns `challenge.schema.json` (25 s lifetime; the plugin's peer set and the mobile's `expectedPeerKeyThumbprint` are mutually exclusive). `POST /v1/aokie-companion/relay/frames` `{"to":"plugin"|"mobile:<thumbprint>","frames":[...]}` (`compat-frames-request`): 1 to 64 frames; a mobile may address only `plugin`; the plugin only phones in its admission's roster whose device row is active; each frame at most `limits.lanes.sig.body` bytes (`413 relay_frame_too_large`); the party mailbox limits of section 3 apply (`429 relay_backpressure`); TTL 120 s; answer `{"accepted":N,"seq":N}` (`compat-frames-accepted`). Facade reads do not acknowledge: `since` only selects where to read from, and frames live until their TTL. `GET .../frames?since=&wait=` returns `compat-frames-page` (at most 128 frames and 1 MiB, the first always returned; `wait` up to 20, a core hold). There is no consent filter: the plugin is the consent authority.

`GET /v1/aokie-companion/relay/stream`: SSE framing over a bounded poll that flushes at once. Verify the admission and the device row; register a core hold for the party (a newer stream or frames wait for the same party ends this one within 250 ms with an `end` event); `deadline = min(now + 20 s, admission exp + 30 s)`. Before any wait send status `200`, `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-store`, `X-Accel-Buffering: no`, and write `retry: 2000\n\n: connected\n\n` and flush. Then write each frame as `id: <seq>` / `event: frame` / `data: <compact JSON {"seq","from","subjectId","grants","frame"}>`; after the first frame wait at most 50 ms more to coalesce a burst; every 2 s without output write `: keepalive\n\n`; exit at once when the client has gone. At the deadline, or when superseded, write `id: <cursor>\nevent: end\ndata: {}\n\n` and exit; `end` is never omitted. A relay advertises `compat.sse-framed-poll` only after a streaming probe passed on this host.

### 10.7 Domain strings

Every signature and MAC is over a domain string, a zero byte, and the text; a value made under one domain never verifies under another.

| Use | Domain | Made by |
|---|---|---|
| static `info` signature | `oaiy/relay/1/info` | relay key (Ed25519) |
| identity proof | `oaiy/relay/1/info-proof`, then 0x00, the nonce bytes, SHA-256(body) and the ASCII decimal time | relay key |
| enrolment proof | `oaiy/relay/1/enroll` | key derived from the enrolment secret |
| command | `oaiy/relay/1/cmd` | provider signing key |
| result | `oaiy/relay/1/res` | host identity |
| sync | `oaiy/relay/1/sync` | host identity |
| provider rotation | `oaiy/relay/1/provider-rotate` | the pinned provider key |
| ring | `oaiy/relay/1/ring` | host identity |
| pairing offer MAC | `oaiy/pairing/3/offer-mac` | HMAC-SHA256 under `mac_key` |
| pairing response | `oaiy/pairing/3/response` | phone endpoint key |
| pairing response MAC | `oaiy/pairing/3/response-mac` | HMAC-SHA256 under `mac_key` |
| approval receipt | `oaiy/pairing/3/approval` | desktop endpoint key |
| typed-code check, SAS input, SAS check | `oaiy/pairing/3/typed` (then the 16 raw bytes of `s`), `oaiy/pairing/3/sas` (then the 16 raw bytes of `pid`), `oaiy/pairing/3/sas-check` (then the 12 characters as ASCII) | SHA-256 and HKDF |
| HKDF salts | `oaiy/pairing/3`, `oaiy/enroll/1` | |
| roster hash | `aokie/v2/peer-roster` | SHA-256 (the Aokie construction) |

## 11. Response shapes

1. Every dynamic JSON success body carries `"time"` (Unix seconds). `"v": 1` appears in poll, post results, presence, device list, device patch, revoke-all, roster, `ack`, pairing fetch, admin capacity and admin status responses. Static `info` carries neither. Error bodies carry no `time`. `204` responses have no body.
2. `GET /v1/health` is exactly `{"ok":true,"time":N,"authHeaderSeen":bool}`.
3. Poll: `hold` appears only when the request asked for `wait > 0`: `{"granted":true}`, `{"refused":true,"retryAfter":N}` or `{"granted":true,"superseded":true}` (with the header `X-OAIY-Hold: granted|refused`); `reset: true` appears only on a reset (and then `items` is empty and `more` false); `hdr` in a delivered item is always an object; `rp` is optional.
4. `POST /v1/devices/self/meta`: `200 {"v":1,"time":N}`. `POST /v1/devices/{id}` (patch): `200 {"v":1,"device":{...},"time":N}`. Revoke one: `204`. `POST /v1/devices/revoke {"role":"phone"}`: `200 {"v":1,"revoked":["dev-..."],"time":N}`. Token rotate: `200 {"token","graceUntil","time"}`. Enroll: `201 {"deviceId","token","relayId","time"}`. A device list entry is `{id, role, name, ver?, createdAt, lastSeen, revokedAt, thumbprint, ownerDesktop, grants, flags:{canCmd}, push:{kind}}`; members that do not apply are `null`.
5. `POST /v1/admin/capacity` takes `{"workers":int>=1,"streamOk":bool,"maxBody":int,"maxHold":int}` and answers `200 {"v":1,"effective":{"workers","heldSoft","heldHard","waitMax","streamOk","laneBodies":{lane:int}},"time"}`. `GET /v1/admin/status` answers `{"v":1,"version","php":{"version","sapi","extensions":[...]},"db":{"driver","sizeBytes"?},"items":{"live","bytes","oldestAgeS"},"devices":[{id,role,name,online}],"holds":{"soft","hard","measured","byKind":{...},"live"},"rejected24h":{code:n},"noAuthHeader24h":n,"tokensOlderThan90d":[ids],"warnings":[strings],"time"}` plus a `diag` object with `?diag=1`. Both schemas allow additional members.
6. The first relay serves the lanes `cmd`, `res`, `ctl` and `sync` (and `ring` when call features are on); `info.limits.lanes` lists only lanes actually served, and `features` lists only implemented features (initially `poll`, `items`, `presence`, `methods.post-forms`).
7. `Retry-After` is mirrored in `error.retryAfter` on 429 and 503.

## 12. Vectors and conformance

`vectors.json` holds, for each of Appendix A0 to A12 of the design, the inputs and the expected outputs: A0 the roster hash (anchor: the Aokie README), A1 a JWS EdDSA signature (anchor: RFC 8037 A.4), A2 a token, A3 pairing v3 (secret, typed code, `pid`, offer and MAC, response claims, signature and MAC, SAS, approval receipt), A4 an admission bearer (and A4b the bearer sizes for 1 to 64 phones), A5 a TURN credential, A6 an info proof (A6b over the full info document), A7 an enrolment key and proof, A8 a command container, A9 a ticket, A10 a rotation statement, A11 a ring signature, A12 the small-order X25519 encodings with their bit-255 variants. Extras: canonical JSON cases and refusals (including `-0`), token acceptance and refusal cases, thumbprints, typed-code and SAS check-character samples, `sasNegative` (the SAS of `A3` computed from the raw bytes of `pid` and the two wrong readings that take `pid` as text), the error taxonomy, the lane table, and external anchors (RFC 2202, 4231, 5869, 7748, 8032). All keys are fixed public test seeds.

Every deterministic value is recomputed three times: by `generate_vectors.py`, by `../../tests/relay_conformance.py` (Python, from the recorded inputs, without importing the generator) and by `verify_vectors.mjs` (Node). Sealed-box outputs are randomised by the ephemeral key and have no fixed vector here; PHP-sealed and Rust-sealed fixtures, and the composition of `A8` with a sealed box, belong to the desktop and pairing packages (`DK-04`, `RL-06`).

## Schema index

| File | Validates |
|---|---|
| `common.schema.json` | shared definitions (`$defs`) |
| `error.schema.json`, `compat-error.schema.json` | error bodies |
| `health.schema.json`, `info.schema.json` | health, info |
| `item.schema.json`, `post-request.schema.json`, `post-response.schema.json`, `poll-response.schema.json`, `item-state.schema.json` | items and poll |
| `slot-request.schema.json`, `slot-response.schema.json`, `slot.schema.json`, `presence.schema.json` | slots, presence |
| `enroll-request.schema.json`, `enroll-response.schema.json` | enrolment |
| `device.schema.json`, `devices-list.schema.json`, `device-patch-request.schema.json`, `device-patch-response.schema.json`, `device-meta-request.schema.json`, `ack.schema.json`, `devices-revoke-request.schema.json`, `devices-revoke-response.schema.json` | devices |
| `roster-request.schema.json`, `roster-response.schema.json`, `token-rotate-response.schema.json`, `keys-request.schema.json`, `keys-response.schema.json`, `providers-request.schema.json`, `providers-response.schema.json`, `push-register-request.schema.json` | roster, tokens, keys, providers, push |
| `pairing-create-request.schema.json`, `pairing-create-response.schema.json`, `pairing-fetch-response.schema.json`, `pairing-answer-request.schema.json`, `pairing-answer-response.schema.json`, `pairing-offer.schema.json`, `pairing-claims.schema.json`, `pairing-response.schema.json`, `pairing-decision.schema.json`, `pairing-reject-request.schema.json`, `approval-receipt.schema.json` | pairing v3 |
| `admission-claims.schema.json`, `admission-plugin-request.schema.json`, `admission-plugin-response.schema.json`, `admission-mobile-request.schema.json`, `admission-mobile-response.schema.json`, `ice-server.schema.json`, `challenge.schema.json`, `compat-frames-request.schema.json`, `compat-frames-accepted.schema.json`, `compat-frames-page.schema.json` | admission and the Aokie compatibility routes |
| `ring.schema.json` | ring body |
| `command.schema.json`, `result.schema.json`, `container.schema.json`, `rotation-statement.schema.json`, `ctl.schema.json` | authority envelopes |
| `ticket-header.schema.json`, `ticket-claims.schema.json`, `replybox-request.schema.json`, `replybox-response.schema.json`, `rbx-item-request.schema.json` | tickets and reply boxes |
| `admin-capacity-request.schema.json`, `admin-capacity-response.schema.json`, `admin-status.schema.json` | calibration and status |

## Interpretations

Where the design is silent, ambiguous or wrong, this package makes the smallest sensible choice. An implementation follows these; the list is also what to raise with the design.

1. **`seq` starts at 1.** The design says "unsigned integer below 2^53" without a first value. Starting at 1 keeps `since = 0` (the default) meaning "nothing acknowledged" and makes `cursor = 0` mean "no item yet".
2. **Integer spellings.** JSON Schema's `integer` accepts `60.0`; the design says `ttl` must be an integer and rejects "fractional". The relay rejects any spelling with a fraction or exponent for integer members (`60.0`, `6e1`), consistent with the canonical form's "integers only". A schema cannot say this; the black-box suite must.
3. **HEAD is `405`.** The design allows exactly GET, POST, OPTIONS and the three aliases and says "anything else is 405".
4. **`hold` is present only when `wait > 0` was requested.** The design's example shows `hold` on a hold; for `wait=0` there is nothing to grant or refuse.
5. **Only `wait` is clamped.** `limit` and `maxBytes` outside their ranges, and a negative, fractional or non-numeric `since`, are `400 invalid_request`.
6. **Content type is required only when a body is present**; the charset parameter may be omitted.
7. **A provider's `deviceId` is its `prov-` id.** The design gives the enrol response as `dev-...` yet generates a `providerId` when a provider key is redeemed; the response member `deviceId` carries the `prov-` id for role `provider`.
8. **`401 revoked` needs the token row.** The design says a revoke "deletes its tokens" and also defines `401 revoked`; a deleted token could only be `unauthorized`. The relay marks the tokens revoked and keeps the rows, and answers `revoked` only after the secret verified, so nothing is disclosed to a party without the secret. A held poll of a revoked device ends with `401 revoked`.
9. **`GET /v1/items/{id}` requires both `to` and `lane`.** An id is unique only per (mailbox, lane, sender), and the sender is the caller; without `to` and `lane` the lookup is ambiguous.
10. **`hdr` may be absent on a post** (meaning `{}`); `to`, `lane`, `id` and `body` are required. The relay does not enforce a `ct` per lane.
11. **The 512 byte `hdr` cap is unreachable by the allow-list** (the largest expressible header is 405 bytes) but is still counted, in compact UTF-8 serialisation, as defence in depth.
12. **Base64url is read strictly**: a spelling whose unused low bits are not zero (for example the last character of a 43-character secret being `9` instead of `8`) is refused for tokens, nonces and every fixed-length key, so one secret has exactly one spelling. The design says only "characters outside the alphabet are rejected".
13. **`X-OAIY-Nonce`** is 16 to 32 bytes, i.e. 22 to 43 canonical base64url characters; anything else is `400 invalid_request`. The header name `X-OAIY-Proof` is used for two different proofs (a response header on `info`, a request header on `enroll`); the direction tells them apart.
14. **`limits.lanes` in `info` lists `ctl` too**, which the design's example omits, because a desktop may post it and its TTL bounds are then part of the contract. The `A6b` vector keeps the design's example verbatim.
15. **`Authorization` is `Bearer <credential>`** with the scheme matched case-insensitively and exactly one space; when the header arrives by more than one PHP source the first in the order `HTTP_AUTHORIZATION`, `REDIRECT_HTTP_AUTHORIZATION`, `getallheaders()` wins.
16. **Roster `revision` may be 0** at `POST /v1/roster` (a reinstall restarts at 0) while an admission needs 1 to 2^53-1.
17. **The design's Appendix A lists sealed-box fixtures "still to be produced by RL-01"; section 8.2 assigns them to `DK-04` and `RL-06`.** This package follows 8.2 and ships none: a randomised output cannot be a known answer, and no Rust implementation exists yet to seal the other direction.
18. **The plugin bearer sizes of Appendix C (964 for one phone, +92 each) reproduce exactly** when the plugin subject id is `aokie` (vector `A4b`); another id changes the size by twice its length difference.
19. **`ring` is `403 feature_disabled` while call features are off**, and a ring with an invalid `hdr.sig` is `400 invalid_item`, not `403`.
20. **Timing of `Retry-After` values**: 1 second for the poll gap rule and lookup limit, 5 for `quota_exceeded`, 60 for `ip.authfail`, 2 for a refused hold, as the design's tables give them; other 429/503 answers use the bucket's refill time, rounded up.
21. **The SAS input carries the raw 16 bytes of `pid`, and in general raw bytes, never their text, enter hashes, MACs, signatures and key derivations.** The design writes `info = "oaiy/pairing/3/sas" || 0x00 || pid` and elsewhere calls `pid` a 22-character b64u string, so the prose reads naturally as the text: an independent implementation built that way got `24b574fd2e0d1e24` where the vector has `3563599914bdf7f9`, and nothing in the prose could tell it it was wrong. The generator, the design's own value (`6NHN-K68M-QQVZ-5`) and the vectors use the raw bytes (a 35-byte `info`). This package says so at every place where a value with a text spelling is an input to a hash or KDF (section 1, the `pid` row of section 2, the typed code, the SAS, secret storage, the slot ETag, the signed bytes of a container and of a rotation statement, the domain table), and `extras.sasNegative` records the two wrong readings, `pid` as its b64u text (41-byte `info`, `24b574fd2e0d1e24`) and as its hex text (51-byte `info`), with the value each produces, so that an implementation can show it does not make either. The one place `pid` is text is the approval receipt, because that is a JSON string.
22. **The item ids `.` and `..` are refused** (`400 invalid_item`, and in a `GET /v1/items/{id}` path `404 not_found`) although they match `[A-Za-z0-9._-]{1,128}`. An id becomes a path segment in a URL and is a name in stores that are files or paths, and `.` and `..` are the two names that a URL parser, a proxy or a file system resolve away instead of keeping. Every other name made of these characters, `...` and `a..b` included, is an ordinary id. `common#itemId` says so with a `not`.
23. **`-0` is not an integer.** JSON's grammar admits it and JSON Schema's `integer` accepts it, but it is a second spelling of `0`, and a value that has two spellings is one that two implementations can disagree about (a signed or hashed text would differ). A request member that must be an integer (`ttl`, `since`, `wait`, `limit`, `revision`, and every other one) and every canonical-JSON input is refused if it is written `-0`; `0` and negative integers are unaffected (Interpretation 2 is the same rule for `60.0` and `6e1`). The refusal is in `extras.canonical.refused` for canonical JSON.
24. **The idempotency key of an item is (mailbox, lane, sender, id), not (mailbox, lane, id).** The design keys it without the sender, so a second provider posting an id another provider had already used got `duplicate` with the first provider's `seq` (or `conflict`), could learn that the id was in use, and could claim ids ahead of the sender that meant to use them. An id is chosen by its sender and belongs to it; `GET /v1/items/{id}` already answers only about the caller's own items, and a `res` may answer a `cmd` when the recipient of the `res` is a device that posted a `cmd` of that id to the desktop. The relay's database moves to schema version 2 for it and migrates an older database on first use.
