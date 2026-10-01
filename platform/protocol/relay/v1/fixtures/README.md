# Recorded fixtures

`vectors.json` holds known answers: values that are the same every time, so three implementations can agree on them. A sealed
box is not one of those: its ephemeral key is random, so no two are alike. The files here are **recordings** of the real relay
(and, for the Aokie decoders, of the shapes those decoders read), written so that the code that will read them, the desktop
(`DK-07`) and the phone shell, can check itself against bytes the relay actually produced.

| File | What it is | Read by |
|---|---|---|
| `sealed-token.json` | the token a pairing seals to the phone, three real ones and ten that must not open | the desktop's Rust test of `crypto_box`, the phone's opener |
| `rust-check/` | a stand-alone Rust crate that opens `sealed-token.json` with the `crypto_box` crate | a first Rust reader, to copy from |
| `pairing-ceremony.json` | one whole pairing (Appendix A3's keys and values) as the requests a desktop and a phone make and the answers the relay gives | a Rust stub relay, a phone double, a host client |
| `verify_fixtures.py`, `verify_fixtures.mjs` | two independent checks of the two files above, with no libsodium | the conformance suite (`../../../tests/relay_conformance.py`) |
| `selftest_fixtures.py` | runs those two checks on sixteen damaged copies of the two files (a flipped bit, a wrong hash, a bad receipt, a good box among the refused ones, ...) and requires each to refuse each | the conformance suite |
| `aokie/` | what the relay answers to the shipped Aokie plugin and phone (admissions, challenges, frames, streams, errors, ICE) and the rules of their decoders applied to it; its own [`README.md`](aokie/README.md) | a Rust contract test in the Aokie repository, and the conformance suite |
| `poll-client/poll-client.json` | **not a recording**: a hand-written table of what the poll loop of a native client does with each answer of the relay (rules P1 to P9 of `README.md` section 5.1.1) | the desktop's relay client (DK-03), the phone's (MOB-21a), and the two readers below |
| `poll-client/verify_poll_client.py`, `poll-client/verify_poll_client.mjs` | two readings of those rules, written from the README alone in two languages, each recomputing every case of the table; the conformance suite also runs both on twenty-six damaged tables (a clamp, a backoff cap, a jitter or a count that is wrong, a case that went missing, constants that are not the README's) and pins the number of cases and the digest of their names and requires each to refuse each | the conformance suite |

Regenerate with `php platform/relay/tests/fixtures.php --write` (it drives the relay in a temporary directory on loopback and
overwrites the files: the sealed boxes and tokens change, so commit the result). `php platform/relay/tests/fixtures.php --check`
verifies the committed files without changing them: every sealed token opens (or is refused) as its file says, and the
ceremony still is Appendix A3's.

**No token is written down.** A device token is a credential even when it belongs to a relay that never existed, and secret
scanners look for its prefix. `sealed-token.json` records each token's length and SHA-256 instead; a reader opens the box and
compares the hash. The recipient key is the fixed public test phone of Appendix A3 (`x25519Secrets.phone` in `vectors.json`).

## `sealed-token.json`

```json
{ "recipient": { "x25519Secret": "<b64u>", "x25519Public": "<b64u>" },
  "opens":   [ { "label": "...", "sealedToken": "<b64u>", "sealedBytes": 111, "plaintextLength": 63, "plaintextSha256": "<hex>" } ],
  "refused": [ { "label": "...", "sealedToken": "<b64u>" } ],
  "wrongRecipient": { "x25519Secret": "<b64u>", "x25519Public": "<b64u>" } }
```

`sealedToken` is what `GET /v1/pair/{pid}` returns once the owner has approved. It is base64url without padding of a **libsodium
sealed box** (`crypto_box_seal`): `ephemeralPublic (32) || tag (16) || ciphertext`, the ciphertext as long as the plaintext,
which is the 63 character device token as ASCII. The nonce is not sent: it is BLAKE2b with a 24 byte output over
`ephemeralPublic || recipientPublic`, and the key is HSalsa20 over the X25519 shared secret (`crypto_box_beforenm`).

A reader MUST

1. decode the base64url strictly (no padding, no other alphabet, unused low bits zero) and refuse a box under 48 bytes;
2. refuse an ephemeral key of small order (an all-zero X25519 shared secret): `refused[6]` and `refused[7]` are those, but their
   tags fail as well, so a reader that skips the rule still refuses them. **`refused[8]` and `refused[9]` are the ones that test
   the rule**: a small-order ephemeral key (`zero` and `order8-a` of vector A12) with a real `crypto_secretbox` under
   `HSalsa20(0^32, 0^16)`, which a reader that skips the check opens. libsodium refuses them itself; the Rust `crypto_box` crate
   does **not** (its `unseal` opens `refused[8]`; `rust-check` shows it), so check the shared secret before opening (`README.md` section
   10.3, vector A12);
3. open the box, and only then look at the plaintext: 63 bytes matching `oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}`, whose
   SHA-256 is `plaintextSha256`;
4. verify the approval receipt (`pairing-fetch-response.receipt`) with the desktop key it pinned from the MAC-verified offer
   **before** it stores a profile. A sealed box is anonymous: anyone can seal something to a phone's public key, so the box proves
   nothing about who made it; the receipt proves the desktop approved this phone.

In Rust (`crypto_box` with its `seal` feature, which the desktop's `Cargo.toml` does not yet enable; check the version in the
lockfile) the opening is `SecretKey::from(secret_bytes).unseal(&sealed)`. **Run**: `rust-check/` is a stand-alone crate (its own
empty `[workspace]`, no relation to any other Cargo project) that reads `sealed-token.json` with `crypto_box` 0.9.1 and opens the
three tokens (length, SHA-256 and shape as the file says), refuses the ten boxes that must not open and the wrong recipient,
and shows that `unseal` alone opens `refused[8]` (its `open` therefore checks for an all-zero shared secret with
`curve25519-dalek` first; `refused[9]`, an order-8 point, does not open with `unseal` either, because `crypto_box` multiplies by
the scalar reduced modulo the group order and so does not get the zero that RFC 7748 and libsodium get): `cargo run --release -- ../sealed-token.json` printed `21 checks, 0 mismatches` (Windows, rustc of the machine's toolchain,
crates fetched from crates.io, `Cargo.lock` committed). `DK-07` adds the desktop's own test and, in the other direction, a
Rust-sealed sample of the same shape (`opens[]` with another `recipient`) that `verify_fixtures.py` and `verify_fixtures.mjs`
open the same way, which is the reverse fixture `vectors.json` interpretation 17 asks for.

## `pairing-ceremony.json`

`steps` is the ceremony in order: the desktop opens the rendezvous, the phone fetches the offer, the phone posts its response,
the desktop's poll returns the `pair` item, the desktop approves, the phone reads the outcome. Each step has the `request`
(method, path and body; the desktop's own requests carry its device token, which is not recorded) and the relay's `response`
(status and body). Everything in it that is deterministic is Appendix A3's: the pairing secret, the 778 byte offer and its MAC,
the response claims, signature and MAC, the receipt and the short authentication string. What varies between recordings is the
phone's relay device id and the sealed token.

The verifiers re-derive the pid, both MACs, both signatures, the canonical receipt document and the SAS from the secret and the
keys, and open the sealed token.

## `poll-client/poll-client.json`

Not a recording of the relay: the relay's own side of these rules (`error.rule` on the two refusals of a poll, the pause it asks for) is
tested in `platform/relay/tests/`. This is the other side, what a client does with an answer, so that the desktop's client (DK-03) and
the phone's (MOB-21a) can be tested against the same table before there is a relay to poll. `cases` is a list; each case has an `id`, the
`rule` of README section 5.1.1 it checks (`P1` to `P9`), and either

- `replace`: `{ "msSinceLastStart": n }`, and `expect.waitMs`: how long a client that cancels a running poll waits before it starts the
  next (P1); or
- `proof`: `{ "result": "verified" | "none" | "invalid" }`, the answer to the identity proof of P9, with `state` and `info` as below; or
- `proofDue`: `{ processStart, networkChanged, longestPauseS, secondsSinceProof }`, and `expect.due`: whether a proof is due (P9); or
- `state`: the client's counters before the answer (`n429`, `nFail`, `nRefused`, `n400`), `info` (`pollGapMs` and `fallbackS` as `GET /v1/info`
  gave them), `response` (`status` or `null` with a `transport` word when nothing came back, `headers` with lower-case names, `body` as
  parsed JSON or `null`), `u` (the jitter draw, 0 up to but not including 1), and optionally `nowEpoch` (the client's own clock, for a
  `Retry-After` that is an HTTP-date with no usable `Date` header), `weReplaced: false` (the poll answered `superseded` was not replaced by
  this client) and `minClientAboveOurs` (what a `426` made the client find out by re-reading `info`); and `expect`: the `outcome` (`progress`,
  `superseded`, `idle`, `flow`, `failure`, `stop`, or `proved` for a proof), `baseS` (the pause before jitter, in seconds), `pauseS` (after:
  `baseS * (1 + 0.2 * u)`), the `state` after, the `action` (`forget_credential`, `refresh_or_reenrol`, `update_client`, `report_defect`,
  `clear_epoch`, `report_relay_changed`) and the `report` (`unreachable`, `in_flight_defect`, `duplicate_credential`, `invalid_request`): the
  names are those of README section 5.1.1 (P7).

The table also says what it holds: `constants` (the numbers of the rules, which a reader refuses unless they are exactly the README's, and
takes its numbers from), `caseCount` and `idsSha256` (the SHA-256 of the sorted case names, one per line: a case that goes missing changes
both, and the conformance suite pins both so that a table edited to agree with itself is noticed too).

A reader loads the file, runs every case through its own implementation of the rules and compares all six members; `python
poll-client/verify_poll_client.py` and `node poll-client/verify_poll_client.mjs` each print `N checks, 0 mismatches` (and take `--file`
to read another copy, which is how the conformance suite shows that they are not vacuous).
