# Recorded fixtures

`vectors.json` holds known answers: values that are the same every time, so three implementations can agree on them. A sealed
box is not one of those: its ephemeral key is random, so no two are alike. The files here are **recordings** of the real relay
(and, for the Aokie decoders, of the shapes those decoders read), written so that the code that will read them, the desktop
(`DK-07`) and the phone shell, can check itself against bytes the relay actually produced.

| File | What it is | Read by |
|---|---|---|
| `sealed-token.json` | the token a pairing seals to the phone, three real ones and eight that must not open | the desktop's Rust test of `crypto_box`, the phone's opener |
| `rust-check/` | a stand-alone Rust crate that opens `sealed-token.json` with the `crypto_box` crate | a first Rust reader, to copy from |
| `pairing-ceremony.json` | one whole pairing (Appendix A3's keys and values) as the requests a desktop and a phone make and the answers the relay gives | a Rust stub relay, a phone double, a host client |
| `verify_fixtures.py`, `verify_fixtures.mjs` | two independent checks of the two files above, with no libsodium | the conformance suite (`../../../tests/relay_conformance.py`) |
| `aokie/` | what the relay answers to the shipped Aokie plugin and phone (admissions, challenges, frames, streams, errors, ICE) and the rules of their decoders applied to it; its own [`README.md`](aokie/README.md) | a Rust contract test in the Aokie repository, and the conformance suite |

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
2. refuse an ephemeral key of small order (an all-zero X25519 shared secret): `refused[6]` and `refused[7]` are those. libsodium
   does this itself; the Rust `crypto_box` crate does **not**, so check it before opening (`README.md` section 10.3, vector A12);
3. open the box, and only then look at the plaintext: 63 bytes matching `oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}`, whose
   SHA-256 is `plaintextSha256`;
4. verify the approval receipt (`pairing-fetch-response.receipt`) with the desktop key it pinned from the MAC-verified offer
   **before** it stores a profile. A sealed box is anonymous: anyone can seal something to a phone's public key, so the box proves
   nothing about who made it; the receipt proves the desktop approved this phone.

In Rust (`crypto_box` with its `seal` feature, which the desktop's `Cargo.toml` does not yet enable; check the version in the
lockfile) the opening is `SecretKey::from(secret_bytes).unseal(&sealed)`. **Run**: `rust-check/` is a stand-alone crate (its own
empty `[workspace]`, no relation to any other Cargo project) that reads `sealed-token.json` with `crypto_box` 0.9.1 and opens the
three tokens (length, SHA-256 and shape as the file says), refuses the eight boxes that must not open and the wrong recipient:
`cargo run --release -- ../sealed-token.json` printed `18 checks, 0 mismatches` (Windows, rustc of the machine's toolchain,
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
