#!/usr/bin/env python3
"""Generate vectors.json, the known-answer vectors of oaiy-relay/1.

    python generate_vectors.py            write vectors.json next to this file
    python generate_vectors.py --check    fail unless vectors.json is exactly what this would write

Every deterministic value is produced here with Python's `cryptography` package and
is re-computed by two other programs that never import this one:
`../../tests/relay_conformance.py` (Python, from the inputs recorded in the file)
and `verify_vectors.mjs` (Node, `node:crypto` only). The values of Appendix A0 to A12
of the design are reproduced exactly; the design's numbers are the authority, and
the assertions below fail loudly if a library ever disagrees with an external
anchor (RFC 8037, RFC 8032, RFC 7748, RFC 5869, RFC 4231, RFC 2202, and the roster hash
printed in the Aokie self-host README).

Sealed-box outputs are randomised by the ephemeral key, so they have no fixed vector
here. PHP-sealed and Rust-sealed fixtures belong to DK-04 and RL-06.

Keys are the fixed test seeds below. They protect nothing.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import pathlib
import sys

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HERE = pathlib.Path(__file__).resolve().parent
RAW, RAWF = serialization.Encoding.Raw, serialization.PublicFormat.Raw
ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U
P = 2 ** 255 - 19


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Refused(ValueError):
    pass


def canon(obj) -> bytes:
    """Canonical JSON of the pairing family: keys sorted bytewise (UTF-8), no whitespace,
    integers only (a float, or an integer outside the 64-bit range, is refused), strings
    escaped as JSON with non-ASCII left as UTF-8."""
    def w(v) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            if not (-(2 ** 63) <= v <= 2 ** 64 - 1):
                raise Refused("integer out of range")
            return str(v)
        if isinstance(v, float):
            raise Refused("floating point number")
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return "[" + ",".join(w(x) for x in v) + "]"
        if isinstance(v, dict):
            keys = sorted(v, key=lambda k: k.encode("utf-8"))
            return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + w(v[k]) for k in keys) + "}"
        raise Refused("unsupported value")
    return w(obj).encode("utf-8")


def compact(obj) -> bytes:
    """Insertion-order compact JSON: the way a signer ships bytes."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hkdf(ikm: bytes, salt: bytes, info: bytes, n: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=n, salt=salt, info=info).derive(ikm)


def ed_pub(seed: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(RAW, RAWF)


def ed_sign(seed: bytes, msg: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).sign(msg)


def x_pub(secret: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(secret).public_key().public_bytes(RAW, RAWF)


def thumbprint(pub: bytes) -> str:
    jwk = '{"crv":"Ed25519","kty":"OKP","x":' + json.dumps(b64u(pub)) + "}"
    return b64u(hashlib.sha256(jwk.encode()).digest())


def crock(value: int, nbits: int) -> str:
    """Crockford base32 of an nbits-bit value, zero-padded on the right to a multiple of 5 bits."""
    pad = (-nbits) % 5
    value <<= pad
    out = ""
    for _ in range((nbits + pad) // 5):
        out = ALPHABET[value & 31] + out
        value >>= 5
    return out


def roster_hash(rev: int, ths: list[str]) -> str:
    p = {"approvedPeerKeyThumbprints": sorted(ths), "peerRosterRevision": rev}
    return b64u(hashlib.sha256(b"aokie/v2/peer-roster\x00" + canon(p)).digest())


def clamp(k: bytes) -> int:
    k = bytearray(k)
    k[0] &= 248
    k[31] &= 127
    k[31] |= 64
    return int.from_bytes(k, "little")


def x25519_ladder(k_bytes: bytes, u_bytes: bytes) -> bytes:
    """RFC 7748 Montgomery ladder in pure Python (bit 255 of u is ignored, as the RFC says)."""
    a24 = 121665
    k = clamp(k_bytes)
    u = int.from_bytes(u_bytes, "little") & ((1 << 255) - 1)
    x1, x2, z2, x3, z3, swap = u, 1, 0, u, 1, 0
    for t in range(254, -1, -1):
        kt = (k >> t) & 1
        swap ^= kt
        if swap:
            x2, x3 = x3, x2
            z2, z3 = z3, z2
        swap = kt
        a = (x2 + z2) % P
        aa = a * a % P
        b = (x2 - z2) % P
        bb = b * b % P
        e = (aa - bb) % P
        c = (x3 + z3) % P
        d = (x3 - z3) % P
        da = d * a % P
        cb = c * b % P
        x3 = (da + cb) ** 2 % P
        z3 = x1 * (da - cb) ** 2 % P
        x2 = aa * bb % P
        z2 = e * (aa + 121665 * e) % P
    if swap:
        x2, x3 = x3, x2
        z2, z3 = z3, z2
    return (x2 * pow(z2, P - 2, P) % P).to_bytes(32, "little")


def build() -> dict:
    V: dict = {}
    V["protocol"] = "oaiy-relay/1"
    V["generator"] = "generate_vectors.py"
    V["note"] = ("Known-answer vectors of Appendix A0 to A12 plus extras. Text values are exact; signed and MACed "
                 "texts are stored as strings and rebuilt from the recorded objects by the verifiers. All keys are "
                 "fixed public test seeds.")

    # ------------------------------------------------------------- external anchors
    hex_ = bytes.fromhex
    A = {}
    A["rfc5869_tc1"] = {"ikm": "0b" * 22, "salt": "000102030405060708090a0b0c", "info": "f0f1f2f3f4f5f6f7f8f9", "length": 42,
                        "okm": "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"}
    assert hkdf(hex_(A["rfc5869_tc1"]["ikm"]), hex_(A["rfc5869_tc1"]["salt"]), hex_(A["rfc5869_tc1"]["info"]), 42).hex() == A["rfc5869_tc1"]["okm"]
    A["rfc4231_tc1_hmac_sha256"] = {"key": "0b" * 20, "data": "Hi There",
                                    "mac": "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"}
    assert hmac.new(hex_("0b" * 20), b"Hi There", hashlib.sha256).hexdigest() == A["rfc4231_tc1_hmac_sha256"]["mac"]
    A["rfc2202_tc1_hmac_sha1"] = {"key": "0b" * 20, "data": "Hi There", "mac": "b617318655057264e28bc0b6fb378c8ef146be00"}
    assert hmac.new(hex_("0b" * 20), b"Hi There", hashlib.sha1).hexdigest() == A["rfc2202_tc1_hmac_sha1"]["mac"]
    A["rfc8032_test1"] = {"seed": "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
                          "public": "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "message": "",
                          "signature": "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"}
    r = A["rfc8032_test1"]
    assert ed_pub(hex_(r["seed"])).hex() == r["public"] and ed_sign(hex_(r["seed"]), b"").hex() == r["signature"]
    A["rfc7748_6_1"] = {"alicePrivate": "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a",
                        "alicePublic": "8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a",
                        "bobPrivate": "5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb",
                        "bobPublic": "de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f",
                        "shared": "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"}
    r = A["rfc7748_6_1"]
    assert x_pub(hex_(r["alicePrivate"])).hex() == r["alicePublic"] and x_pub(hex_(r["bobPrivate"])).hex() == r["bobPublic"]
    assert x25519_ladder(hex_(r["alicePrivate"]), hex_(r["bobPublic"])).hex() == r["shared"]
    V["anchors"] = A

    # ------------------------------------------------------------- keys
    seeds = {"desktopEndpoint": bytes([0x09] * 32), "phone": bytes([0x07] * 32), "relay": bytes([0x33] * 32),
             "host": bytes([0x0B] * 32), "provider": bytes([0x51] * 32), "provider2": bytes([0x52] * 32)}
    xsec = {"plugin": bytes([0xD1] * 32), "phone": bytes([0xE2] * 32), "host": bytes([0xC3] * 32),
            "provider": bytes([0xA4] * 32), "provider2": bytes([0xA5] * 32), "browserEphemeral": bytes([0x66] * 32)}
    idbytes = {"desktopDevice": bytes(range(0xA0, 0xB0)), "phoneDevice": bytes(range(0xB0, 0xC0)),
               "provider": bytes(range(0xC0, 0xD0)), "relay": bytes(range(0xD0, 0xE0))}
    ids = {"desktopDevice": "dev-" + b64u(idbytes["desktopDevice"]), "phoneDevice": "dev-" + b64u(idbytes["phoneDevice"]),
           "provider": "prov-" + b64u(idbytes["provider"]), "relay": "rly-" + b64u(idbytes["relay"])}
    pub = {k: ed_pub(v) for k, v in seeds.items()}
    xpub = {k: x_pub(v) for k, v in xsec.items()}
    V["keys"] = {
        "ed25519Seeds": {k: v.hex() for k, v in seeds.items()},
        "x25519Secrets": {k: v.hex() for k, v in xsec.items()},
        "idBytes": {k: v.hex() for k, v in idbytes.items()},
        "ids": ids,
        "ed25519Public": {k: {"publicKey": b64u(v), "thumbprint": thumbprint(v)} for k, v in pub.items()},
        "x25519Public": {k: b64u(v) for k, v in xpub.items()},
    }
    dpub, ppub, rpub, hpub, vpub = (pub[k] for k in ("desktopEndpoint", "phone", "relay", "host", "provider"))
    dth, pth, rth, hth, vth = (thumbprint(x) for x in (dpub, ppub, rpub, hpub, vpub))
    DEV, PHONE, PROV, RELAY = ids["desktopDevice"], ids["phoneDevice"], ids["provider"], ids["relay"]

    # ------------------------------------------------------------- A0 roster hash (anchor: Aokie README)
    ths0 = ["mobile_thumbprint_a", "mobile_thumbprint_b"]
    h0 = roster_hash(7, ths0)
    assert h0 == "tsKgPP1ruPCU23HfLYaUChe9jHYHCtubve77gnlfyDw"
    V["A0"] = {"inputs": {"peerRosterRevision": 7, "approvedPeerKeyThumbprints": ths0,
                          "domain": "aokie/v2/peer-roster", "hashedText": canon({"approvedPeerKeyThumbprints": ths0, "peerRosterRevision": 7}).decode()},
               "expected": {"peerRosterHash": h0, "aokieReadmeValue": "tsKgPP1ruPCU23HfLYaUChe9jHYHCtubve77gnlfyDw"}}

    # ------------------------------------------------------------- A1 JWS EdDSA (anchor: RFC 8037 A.4)
    rfc_priv = unb64u("nWGxne_9WmC6hEr0kuwsxERJxWl7MmkZcDusAxyuf2A")
    rfc_pub = unb64u("11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo")
    assert ed_pub(rfc_priv) == rfc_pub
    si = "eyJhbGciOiJFZERTQSJ9.RXhhbXBsZSBvZiBFZDI1NTE5IHNpZ25pbmc"
    sig = ed_sign(rfc_priv, si.encode())
    assert b64u(sig) == "hgyY0il_MGCjP0JzlnLWG1PPOt7-09PGcvMg3AIbQR6dWbhijcNR4ki4iylGjg5BhVsPt9g7sVvpAr_MuM0KAg"
    V["A1"] = {"inputs": {"privateKey": b64u(rfc_priv), "publicKey": b64u(rfc_pub), "signingInput": si},
               "expected": {"signature": b64u(sig)}}

    # ------------------------------------------------------------- A2 capability token
    tid, tsecret = bytes(range(1, 9)), bytes(range(0x20, 0x40))
    token = "oaiyrt1." + b64u(tid) + "." + b64u(tsecret)
    V["A2"] = {"inputs": {"idHex": tid.hex(), "secretHex": tsecret.hex(), "prefix": "oaiyrt1."},
               "expected": {"token": token, "length": len(token), "secretSha256": hashlib.sha256(tsecret).hexdigest()}}
    assert len(token) == 63

    # ------------------------------------------------------------- A3 pairing v3
    s = bytes.fromhex("a1b2c3d4e5f60718293a4b5c6d7e8f90")
    salt = b"oaiy/pairing/3"
    pid = hkdf(s, salt, b"rendezvous", 16)
    mac_key = hkdf(s, salt, b"mac", 32)
    typed26 = crock(int.from_bytes(s, "big"), 128)
    typed2 = crock(int.from_bytes(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + s).digest()[:2], "big") >> 6, 10)
    typed = typed26 + typed2
    typed_disp = "-".join(typed[i:i + 4] for i in range(0, 28, 4))
    nonce = bytes(range(0x40, 0x60))
    offer = {
        "kind": "aokie_mobile_pairing", "schemaVersion": 3, "appId": "aokie",
        "desktopConnectionId": DEV, "desktopName": "Front desk PC",
        "desktopEndpointKey": {"algorithm": "ed25519", "publicKey": b64u(dpub), "thumbprint": dth},
        "desktopX25519": b64u(xpub["plugin"]),
        "hostIdentity": {"ed25519": b64u(hpub), "thumbprint": hth, "x25519": b64u(xpub["host"])},
        "nonce": b64u(nonce), "jti": "pair-0001", "issuedAt": 1790000000, "expiresAt": 1790000600,
        "relay": {"url": "https://relay.example.com", "fingerprint": rth},
    }
    offer_text = canon(offer)
    offer_mac = hmac.new(mac_key, b"oaiy/pairing/3/offer-mac\x00" + offer_text, hashlib.sha256).digest()
    claims = {
        "appId": "aokie", "desktopConnectionId": DEV, "desktopKeyThumbprint": dth, "deviceId": PHONE, "displayName": "Test phone",
        "mobileEndpointKey": {"algorithm": "ed25519", "publicKey": b64u(ppub), "thumbprint": pth},
        "mobileX25519": b64u(xpub["phone"]), "pairingNonce": b64u(nonce), "jti": "pair-0001",
        "issuedAt": 1790000030, "expiresAt": 1790000150,
    }
    claims_c = canon(claims)
    resp_sig = ed_sign(seeds["phone"], b"oaiy/pairing/3/response\x00" + claims_c)
    resp_mac = hmac.new(mac_key, b"oaiy/pairing/3/response-mac\x00" + claims_c, hashlib.sha256).digest()
    sas_raw = hkdf(dpub + ppub, nonce, b"oaiy/pairing/3/sas\x00" + pid, 8)
    sas12 = crock(int.from_bytes(sas_raw, "big") >> 4, 60)
    sas_check = ALPHABET[hashlib.sha256(b"oaiy/pairing/3/sas-check\x00" + sas12.encode()).digest()[0] >> 3]
    sas_disp = sas12[:4] + "-" + sas12[4:8] + "-" + sas12[8:] + "-" + sas_check
    grants = ["caller_read", "captions_read", "assistance_read", "assistance_respond", "rtc_signal", "state_read"]
    receipt_doc = {"appId": "aokie", "grants": sorted(grants), "issuedAt": 1790000040, "phoneThumbprint": pth, "pid": b64u(pid)}
    receipt_text = canon(receipt_doc)
    receipt = ed_sign(seeds["desktopEndpoint"], b"oaiy/pairing/3/approval\x00" + receipt_text)
    pair_uri = ("oaiy://pair?v=3&u=https%3A%2F%2Frelay.example.com&f=" + rth + "&s=" + b64u(s) + "&x=1790000600")
    V["A3"] = {
        "inputs": {"secretHex": s.hex(), "nonceHex": nonce.hex(), "hkdfSalt": "oaiy/pairing/3",
                   "relayUrl": "https://relay.example.com", "expiresAtParam": 1790000600,
                   "offer": offer, "claims": claims, "receiptDocument": receipt_doc},
        "expected": {
            "secretB64u": b64u(s), "typedCode": typed_disp, "pid": b64u(pid), "pidHex": pid.hex(), "macKeyHex": mac_key.hex(),
            "pairingUri": pair_uri, "offerText": offer_text.decode(), "offerTextBytes": len(offer_text), "offerMac": b64u(offer_mac),
            "claimsCanonical": claims_c.decode(), "responseSignature": b64u(resp_sig), "responseMac": b64u(resp_mac),
            "sasRawHex": sas_raw.hex(), "sas12": sas12, "sasCheckChar": sas_check, "sasDisplay": sas_disp,
            "receiptText": receipt_text.decode(), "receiptSignature": b64u(receipt),
        },
    }
    assert typed_disp == "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RW" and sas_disp == "6NHN-K68M-QQVZ-5"
    assert len(offer_text) == 778, len(offer_text)

    # ------------------------------------------------------------- A4 admission token (mobile) and bearer sizes
    adm_secret = bytes([1] * 32)
    adm = {"aud": "aokie-v2-gateway", "appId": "aokie", "subjectId": PHONE, "role": "mobile", "holderKeyThumbprint": pth,
           "expectedPeerKeyThumbprint": dth, "scopes": ["state_read", "caller_read", "captions_read", "rtc_signal"],
           "dsk": DEV, "exp": 1790000090, "jti": "adm_00000000000000000000000000000001"}
    payload = compact(adm)
    tok = "aokie-adm-v2." + payload.hex() + "." + hmac.new(adm_secret, payload, hashlib.sha256).hexdigest()
    V["A4"] = {"inputs": {"secretHex": adm_secret.hex(), "claims": adm},
               "expected": {"payload": payload.decode(), "token": tok, "length": len(tok)}}
    assert len(tok) == 888

    def plugin_bearer(n: int) -> str:
        ths = sorted(b64u(hashlib.sha256(bytes([i])).digest()) for i in range(n))
        c = {"aud": "aokie-v2-gateway", "appId": "aokie", "subjectId": "aokie", "role": "plugin", "holderKeyThumbprint": dth,
             "approvedPeerKeyThumbprints": ths, "peerRosterRevision": 7, "peerRosterHash": roster_hash(7, ths),
             "scopes": ["state_read", "rtc_signal"], "dsk": DEV, "exp": 1790000090, "jti": "adm_00000000000000000000000000000001"}
        p = compact(c)
        return "aokie-adm-v2." + p.hex() + "." + hmac.new(adm_secret, p, hashlib.sha256).hexdigest()

    sizes = {"1": 964, "3": 1148, "8": 1608, "16": 2344, "32": 3816, "64": 6760}
    for n, want in sizes.items():
        assert len(plugin_bearer(int(n))) == want, (n, len(plugin_bearer(int(n))))
    V["A4b"] = {"inputs": {"secretHex": adm_secret.hex(), "pluginId": "aokie", "holderThumbprint": dth, "peerRosterRevision": 7,
                           "scopes": ["state_read", "rtc_signal"], "dsk": DEV, "exp": 1790000090,
                           "jti": "adm_00000000000000000000000000000001",
                           "peerThumbprintOf": "b64u(SHA-256(byte(i))) for i in 0..n-1, then sorted ascending"},
                "expected": {"lengthByPhones": sizes, "perPhoneCharacters": 92, "tokenForOnePhone": plugin_bearer(1)}}

    # ------------------------------------------------------------- A5 TURN REST credential
    turn_secret = b"0123456789abcdef0123456789abcdef"
    username = "1790000600:" + PHONE
    cred = base64.b64encode(hmac.new(turn_secret, username.encode(), hashlib.sha1).digest()).decode()
    V["A5"] = {"inputs": {"secret": turn_secret.decode(), "expiry": 1790000600, "subjectId": PHONE},
               "expected": {"username": username, "credential": cred}}
    assert cred == "T2oDhW95dkDEbEXjdHRbKL5eh7g="

    # ------------------------------------------------------------- A6 info proof
    info_body = ('{"protocol":"oaiy-relay/1","relayId":"' + RELAY + '"}').encode()
    inonce = bytes(range(0x10, 0x20))
    itime = 1790000000
    proof = ed_sign(seeds["relay"], b"oaiy/relay/1/info-proof\x00" + inonce + hashlib.sha256(info_body).digest() + str(itime).encode())
    static_sig = ed_sign(seeds["relay"], b"oaiy/relay/1/info\x00" + info_body)
    V["A6"] = {"inputs": {"body": info_body.decode(), "nonce": b64u(inonce), "time": itime},
               "expected": {"bodySha256": hashlib.sha256(info_body).hexdigest(), "proof": b64u(proof), "staticSignature": b64u(static_sig)}}
    # the same proof over the full 4.8 example document, compact insertion order
    info_example = {
        "protocol": "oaiy-relay/1", "minClient": 1, "relayId": RELAY,
        "relayKey": {"algorithm": "ed25519", "publicKey": b64u(rpub), "thumbprint": rth},
        "software": {"name": "oaiy-relay", "version": "0.1.0"},
        "features": ["poll", "items", "slots", "presence", "replyboxes", "tickets", "pairing.v3", "methods.post-forms", "call",
                     "admission.aokie-adm-v2", "compat.aokie-companion-relay", "compat.sse-framed-poll", "push.fcm"],
        "wait": {"default": 20, "max": 20, "pollGapMs": 250, "fallbackS": 5},
        "presenceWindow": 60,
        "limits": {"batchItems": 64, "batchBytes": 1048576, "mailboxItems": 512, "mailboxBytes": 8388608, "bulkShare": 0.75,
                   "sigItems": 1024, "sigSenderShare": 0.25, "lookupWait": 8, "lookupHeld": 4, "hdrBytes": 512, "slotBytes": 65536,
                   "rosterMax": 16, "held": {"soft": 3, "hard": 4, "measured": False},
                   "lanes": {"cmd": {"body": 32768, "ttl": {"default": 60, "min": 1, "max": 300}},
                             "res": {"body": 98304, "ttl": {"default": 300, "min": 1, "max": 3600}},
                             "ai": {"body": 393216, "ttl": {"default": 300, "min": 1, "max": 600}},
                             "ai.in": {"body": 65536, "ttl": {"default": 300, "min": 1, "max": 600}},
                             "ai.out": {"body": 393216, "ttl": {"default": 360, "min": 1, "max": 900}},
                             "ring": {"body": 4096, "ttl": {"default": 30, "min": 1, "max": 300}},
                             "sync": {"body": 65536, "ttl": {"default": 21600, "min": 1, "max": 86400}},
                             "sig": {"body": 196608, "ttl": {"default": 120, "min": 1, "max": 300}}}},
        "turn": True, "cors": ["https://app.example.com"],
    }
    ibody2 = compact(info_example)
    V["A6b"] = {"inputs": {"info": info_example, "nonce": b64u(inonce), "time": itime},
                "expected": {"bodyText": ibody2.decode(), "bodyBytes": len(ibody2), "bodySha256": hashlib.sha256(ibody2).hexdigest(),
                             "proof": b64u(ed_sign(seeds["relay"], b"oaiy/relay/1/info-proof\x00" + inonce + hashlib.sha256(ibody2).digest() + str(itime).encode())),
                             "staticSignature": b64u(ed_sign(seeds["relay"], b"oaiy/relay/1/info\x00" + ibody2))}}

    # ------------------------------------------------------------- A7 enrolment key and proof
    es = bytes.fromhex("00112233445566778899aabbccddeeff")
    esalt = b"oaiy/enroll/1"
    kid = b64u(hkdf(es, esalt, b"id", 8))
    eseed = hkdf(es, esalt, b"sig", 32)
    epub = ed_pub(eseed)
    req = {"kid": kid, "role": "desktop", "name": "Front desk PC", "n": b64u(bytes(range(0x70, 0x80))),
           "keys": {"ed25519": b64u(hpub), "x25519": b64u(xpub["host"])}}
    ebody = compact(req)
    euri = ("oaiy://enroll?v=1&u=https%3A%2F%2Frelay.example.com&f=" + rth + "&k=" + kid + "&s=" + b64u(es) + "&r=desktop&x=1790003600")
    V["A7"] = {"inputs": {"secretHex": es.hex(), "hkdfSalt": "oaiy/enroll/1", "request": req, "relayUrl": "https://relay.example.com", "expiry": 1790003600},
               "expected": {"kid": kid, "derivedPublic": b64u(epub), "requestBody": ebody.decode(),
                            "proof": b64u(ed_sign(eseed, b"oaiy/relay/1/enroll\x00" + ebody)), "uri": euri, "secretB64u": b64u(es)}}

    # ------------------------------------------------------------- A8 command container
    cmd = {"v": 1, "id": "cmd-0001", "dev": DEV, "to": "oaiy-desk-1", "connector": "aokie", "command": "call.hangup",
           "payload": {"callId": "call_0123"}, "idem": "idem-0001", "iat": 1790000000, "exp": 1790000015, "src": PROV, "uid": "u-42"}
    cbytes = compact(cmd)
    csig = ed_sign(seeds["provider"], b"oaiy/relay/1/cmd\x00" + cbytes)
    container = compact({"k": vth, "b": b64u(cbytes), "s": b64u(csig)})
    V["A8"] = {"inputs": {"command": cmd},
               "expected": {"signedBytes": cbytes.decode(), "signature": b64u(csig), "container": container.decode(), "containerBytes": len(container)}}
    assert len(container) == 493

    # ------------------------------------------------------------- A9 ticket (JWS EdDSA)
    eph_pub = xpub["browserEphemeral"]
    hdr = {"alg": "EdDSA", "typ": "oaiy-ticket+jwt", "kid": vth}
    clm = {"iss": PROV, "aud": RELAY, "sub": "member-42", "iat": 1790000000, "exp": 1790000300, "jti": "tkt-0001", "lane": "ai",
           "dev": DEV, "org": "https://app.example.com", "eph": b64u(hashlib.sha256(eph_pub).digest())}
    sin = b64u(compact(hdr)) + "." + b64u(compact(clm))
    tsig = ed_sign(seeds["provider"], sin.encode())
    V["A9"] = {"inputs": {"header": hdr, "claims": clm},
               "expected": {"ephemeralPublic": b64u(eph_pub), "signingInput": sin, "signature": b64u(tsig), "ticket": sin + "." + b64u(tsig)}}

    # ------------------------------------------------------------- A10 provider rotation statement
    new = {"ed25519": b64u(pub["provider2"]), "thumbprint": thumbprint(pub["provider2"]), "x25519": b64u(xpub["provider2"])}
    rot = compact({"v": 1, "prev": vth, "new": new, "serial": 1, "iat": 1790000000, "exp": 1790086400})
    V["A10"] = {"inputs": {"statement": json.loads(rot)},
                "expected": {"statementText": rot.decode(), "signature": b64u(ed_sign(seeds["provider"], b"oaiy/relay/1/provider-rotate\x00" + rot))}}

    # ------------------------------------------------------------- A11 ring signature
    ring = {"aokieClass": "voice_offer", "schemaVersion": "1", "eventId": "evt_ring_0001", "offerId": "toffer_0001", "appId": "aokie",
            "callId": "call_0123", "callEpoch": "7", "ownerEpoch": "4", "expiresAt": "1790000040"}
    rb = compact(ring)
    V["A11"] = {"inputs": {"body": ring}, "expected": {"bodyText": rb.decode(), "hdrSig": b64u(ed_sign(seeds["host"], b"oaiy/relay/1/ring\x00" + rb))}}

    # ------------------------------------------------------------- A12 small-order X25519 peers
    e1 = bytes.fromhex("e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800")
    e2 = bytes.fromhex("5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157")
    small = {"zero": (0).to_bytes(32, "little"), "one": (1).to_bytes(32, "little"), "order8-a": e1, "order8-b": e2,
             "p-1": (P - 1).to_bytes(32, "little"), "p": P.to_bytes(32, "little"), "p+1": (P + 1).to_bytes(32, "little")}
    key = bytes(range(1, 33))
    hi = {}
    for name, enc in small.items():
        b = bytearray(enc)
        b[31] |= 0x80
        hi[name] = bytes(b).hex()
        assert x25519_ladder(key, enc) == bytes(32) and x25519_ladder(key, bytes(b)) == bytes(32), name
    assert x25519_ladder(key, (9).to_bytes(32, "little")) != bytes(32)
    V["A12"] = {"inputs": {"staticKeyHex": key.hex(), "encodings": {k: v.hex() for k, v in small.items()}, "withBit255": hi,
                           "rule": "a peer key is refused when, after masking bit 255, its value reduced modulo p is 0, 1, p-1, or one of the two order-8 encodings; equivalently when the shared secret would be all zero"},
                "expected": {"allGiveZeroSharedSecret": True, "controlBasePointGivesNonZero": True}}

    # ------------------------------------------------------------- extras
    extras: dict = {}
    extras["canonical"] = {
        "cases": [
            {"label": "keys are sorted", "input": '{"b":1,"a":2}', "output": '{"a":2,"b":1}'},
            {"label": "whitespace is dropped, nesting is sorted", "input": '{ "z" : [ 3 , 1 , {"y":true,"x":null} ] , "a" : { } }',
             "output": '{"a":{},"z":[3,1,{"x":null,"y":true}]}'},
            {"label": "order is bytewise on the UTF-8 of the key", "input": '{"B":1,"a":2,"_":3,"1":4}', "output": '{"1":4,"B":1,"_":3,"a":2}'},
            {"label": "non-ASCII stays UTF-8, only quote, backslash and controls are escaped, slash is not",
             "input": '{"z":"é","a":"日本","k":"\\u0001\\n\\"\\\\\\/"}', "output": '{"a":"日本","k":"\\u0001\\n\\"\\\\/","z":"é"}'},
            {"label": "integers up to the 64-bit limits", "input": '{"a":9007199254740991,"b":9223372036854775807,"c":18446744073709551615,"d":-9223372036854775808,"e":0}',
             "output": '{"a":9007199254740991,"b":9223372036854775807,"c":18446744073709551615,"d":-9223372036854775808,"e":0}'},
            {"label": "empty containers, booleans and null", "input": '{"c":null,"b":true,"a":[],"d":{},"e":false}',
             "output": '{"a":[],"b":true,"c":null,"d":{},"e":false}'},
            {"label": "0 and negative integers are accepted; only the spelling -0 is not", "input": '{"a":0,"b":-1,"c":-10,"d":10}',
             "output": '{"a":0,"b":-1,"c":-10,"d":10}'},
        ],
        "refused": [
            {"label": "a fraction", "input": '{"n":0.5}'},
            {"label": "1.0 is a float even though its value is integral", "input": '{"n":1.0}'},
            {"label": "an exponent", "input": '{"n":1e2}'},
            {"label": "a float nested in an array", "input": '{"a":[1,2,3.5]}'},
            {"label": "an integer above the unsigned 64-bit range", "input": '{"n":18446744073709551616}'},
            {"label": "an integer below the signed 64-bit range", "input": '{"n":-9223372036854775809}'},
            {"label": "-0 is not a canonical spelling of the integer 0 (every integer has exactly one spelling)", "input": '{"n":-0}'},
            {"label": "-0 nested in an array", "input": '{"a":[1,-0,3]}'},
        ],
    }
    good_tok = [token, "oaiyrt1." + b64u(b"\xff" * 8) + "." + b64u(b"\x00" * 32), "oaiyrt1." + b64u(bytes(range(8))) + "." + b64u(bytes(range(32)))]
    extras["tokens"] = {
        "valid": good_tok,
        "invalid": [
            {"token": token.replace("oaiyrt1.", "oaiyrt2.", 1), "reason": "wrong prefix"},
            {"token": token.replace("oaiyrt1.", "OAIYRT1.", 1), "reason": "prefix is case sensitive"},
            {"token": token[len("oaiyrt1."):], "reason": "prefix missing"},
            {"token": "oaiyrt1." + b64u(tid)[:10] + "." + b64u(tsecret), "reason": "id one character short"},
            {"token": "oaiyrt1." + b64u(tid) + "A." + b64u(tsecret), "reason": "id one character long"},
            {"token": "oaiyrt1." + b64u(tid) + "." + b64u(tsecret)[:42], "reason": "secret one character short"},
            {"token": token + "A", "reason": "secret one character long"},
            {"token": token.replace(".", "", 2), "reason": "no dots"},
            {"token": token + ".AAAA", "reason": "a third dot and segment"},
            {"token": "oaiyrt1." + b64u(tid) + "." + b64u(tsecret).replace("_", "+", 1) if "_" in b64u(tsecret) else "oaiyrt1." + b64u(tid) + ".+" + b64u(tsecret)[1:], "reason": "standard base64 character"},
            {"token": token + "=", "reason": "padding"},
            {"token": token[:-1] + "/", "reason": "slash in the secret"},
            {"token": " " + token, "reason": "leading space"},
            {"token": token + "\n", "reason": "trailing newline"},
            {"token": "", "reason": "empty"},
            {"token": token[:-1] + "9", "reason": "last character of the secret is not canonical (unused low bits set)"},
            {"token": token.replace(b64u(tid), b64u(tid)[:-1] + "h", 1), "reason": "last character of the id is not canonical (unused low bits set)"},
        ],
        "canonicalLastCharacters": "a 43 character secret and an 11 character id carry 2 unused bits, so the last character must be one of A E I M Q U Y c g k o s w 0 4 8",
    }
    extras["thumbprints"] = [
        {"seed": "09" * 32, "publicKey": b64u(dpub), "jwk": '{"crv":"Ed25519","kty":"OKP","x":"' + b64u(dpub) + '"}', "thumbprint": dth},
        {"seed": "07" * 32, "publicKey": b64u(ppub), "jwk": '{"crv":"Ed25519","kty":"OKP","x":"' + b64u(ppub) + '"}', "thumbprint": pth},
        {"seed": "33" * 32, "publicKey": b64u(rpub), "jwk": '{"crv":"Ed25519","kty":"OKP","x":"' + b64u(rpub) + '"}', "thumbprint": rth},
    ]
    def typed_for(sb: bytes) -> str:
        t = crock(int.from_bytes(sb, "big"), 128) + crock(
            int.from_bytes(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + sb).digest()[:2], "big") >> 6, 10)
        return "-".join(t[j:j + 4] for j in range(0, 28, 4))

    extras["typedCode"] = {
        "samples": [{"secretHex": s.hex(), "typed": typed_for(s)}] + [
            {"secretHex": hashlib.sha256(bytes([i])).digest()[:16].hex(), "typed": typed_for(hashlib.sha256(bytes([i])).digest()[:16])}
            for i in range(1, 6)],
        "normalise": [
            {"input": "m6sc-7n75-yr3h-ga9t-9de6-tzmf-j0rw", "output": "M6SC7N75YR3HGA9T9DE6TZMFJ0RW"},
            {"input": "M6SC 7N75 YR3H GA9T 9DE6 TZMF J0RW", "output": "M6SC7N75YR3HGA9T9DE6TZMFJ0RW"},
            {"input": "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-JORW", "output": "M6SC7N75YR3HGA9T9DE6TZMFJ0RW"},
            {"input": "il1O0oLI", "output": "11100011"},
            {"input": "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RU", "output": None, "reason": "U is refused"},
            {"input": "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0R!", "output": None, "reason": "outside the alphabet"},
        ],
        "algorithm": "26 characters carry s (128 bits, two zero bits of padding), then 2 characters carry the first 10 bits of SHA-256(\"oaiy/pairing/3/typed\" || 0x00 || s)",
    }
    def sas_check_for(sas: str) -> str:
        return ALPHABET[hashlib.sha256(b"oaiy/pairing/3/sas-check\x00" + sas.encode()).digest()[0] >> 3]
    sas_samples = [sas12] + [crock(int.from_bytes(hashlib.sha256(b"sas" + bytes([i])).digest()[:8], "big") >> 4, 60) for i in range(1, 8)]
    extras["sasCheck"] = {"samples": [{"sas12": x, "check": sas_check_for(x)} for x in sas_samples],
                          "algorithm": "ALPHABET[SHA-256(\"oaiy/pairing/3/sas-check\" || 0x00 || the 12 characters as ASCII)[0] >> 3]"}

    # The SAS input carries the RAW 16 bytes of pid. The prose `info = "oaiy/pairing/3/sas" || 0x00 || pid` can be read as the
    # 22-character b64u text of pid (the form pid has everywhere else), which gives a different, wrong SAS. These are the wrong
    # readings, recorded so that an implementation can show that it does not make them.
    def sas_from_info(info: bytes) -> dict:
        raw = hkdf(dpub + ppub, nonce, info, 8)
        s12 = crock(int.from_bytes(raw, "big") >> 4, 60)
        return {"infoHex": info.hex(), "infoLength": len(info), "sasRawHex": raw.hex(), "sas12": s12,
                "sasDisplay": s12[:4] + "-" + s12[4:8] + "-" + s12[8:] + "-" + sas_check_for(s12)}
    sas_prefix = b"oaiy/pairing/3/sas\x00"
    sas_right = sas_from_info(sas_prefix + pid)
    sas_wrong_b64u = sas_from_info(sas_prefix + b64u(pid).encode())
    sas_wrong_hex = sas_from_info(sas_prefix + pid.hex().encode())
    assert sas_right["sasRawHex"] == sas_raw.hex() and sas_right["sasDisplay"] == sas_disp and sas_right["infoLength"] == 35
    assert sas_wrong_b64u["sasRawHex"] == "24b574fd2e0d1e24" and sas_wrong_b64u["infoLength"] == 41
    assert sas_wrong_hex["infoLength"] == 51
    assert len({sas_right["sasRawHex"], sas_wrong_b64u["sasRawHex"], sas_wrong_hex["sasRawHex"]}) == 3
    extras["sasNegative"] = {
        "note": "The inputs are those of A3. The SAS is computed from the RAW 16 bytes of pid. The two wrong readings below are what an implementation gets when it takes pid as text; neither may ever be produced or accepted.",
        "inputs": {"desktopEndpointPublicHex": dpub.hex(), "phoneEndpointPublicHex": ppub.hex(), "nonceHex": nonce.hex(),
                   "pidHex": pid.hex(), "pidB64u": b64u(pid)},
        "algorithm": "sas_raw = HKDF-SHA256(IKM = desktopEndpointPublic || phoneEndpointPublic, salt = nonce, info = \"oaiy/pairing/3/sas\" || 0x00 || PID, L = 8); sas12 = the top 60 bits as 12 Crockford base32 characters",
        "correct": dict(reading="PID = the 16 raw bytes of pid", **sas_right),
        "wrong": [
            dict(reading="PID = the 22 ASCII characters of the b64u text of pid", mustNotProduce=True, **sas_wrong_b64u),
            dict(reading="PID = the 32 ASCII characters of the lower-case hex of pid", mustNotProduce=True, **sas_wrong_hex),
        ],
    }
    extras["errors"] = [
        {"status": 400, "code": "invalid_request", "retry": "no"}, {"status": 400, "code": "invalid_item", "retry": "no"},
        {"status": 401, "code": "unauthorized", "retry": "no"}, {"status": 401, "code": "revoked", "retry": "no"},
        {"status": 403, "code": "forbidden", "retry": "no"}, {"status": 403, "code": "feature_disabled", "retry": "no"},
        {"status": 404, "code": "not_found", "retry": "no"}, {"status": 405, "code": "method_not_allowed", "retry": "no"},
        {"status": 409, "code": "conflict", "retry": "no"}, {"status": 409, "code": "already_answered", "retry": "no"},
        {"status": 410, "code": "expired", "retry": "no"}, {"status": 412, "code": "precondition_failed", "retry": "after a re-read"},
        {"status": 413, "code": "item_too_large", "retry": "no"}, {"status": 415, "code": "unsupported_media_type", "retry": "no"},
        {"status": 422, "code": "unprocessable", "retry": "no"}, {"status": 426, "code": "upgrade_required", "retry": "after upgrade"},
        {"status": 429, "code": "rate_limited", "retry": "after Retry-After"}, {"status": 429, "code": "quota_exceeded", "retry": "after the consumer drains"},
        {"status": 500, "code": "internal", "retry": "backoff"}, {"status": 503, "code": "unavailable", "retry": "after Retry-After"},
    ]
    extras["compatErrors"] = [{"status": 401, "code": "invalid_token"}, {"status": 403, "code": "relay_target_forbidden"},
                              {"status": 413, "code": "relay_frame_too_large"}, {"status": 429, "code": "relay_backpressure"},
                              {"status": 503, "code": "companion_unavailable"}]
    lanes = {
        "cmd": (32768, 60, 300), "res": (98304, 300, 3600), "ai": (393216, 300, 600), "ai.in": (65536, 300, 600),
        "ai.out": (393216, 360, 900), "pair": (16384, 900, 900), "ring": (4096, 30, 300), "ctl": (4096, 3600, 86400),
        "sync": (65536, 21600, 86400), "sig": (196608, 120, 300),
    }
    extras["lanes"] = {k: {"body": v[0], "ttl": {"default": v[1], "min": 1, "max": v[2]}} for k, v in lanes.items()}
    extras["pollGapMs"] = 250
    V["extras"] = extras
    return V


def render(V: dict) -> str:
    return json.dumps(V, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render(build())
    target = HERE / "vectors.json"
    if "--check" in argv:
        have = target.read_bytes().decode("utf-8").replace("\r\n", "\n") if target.exists() else ""   # a checkout may convert line endings
        if have != text:
            sys.stderr.write("vectors.json is not what generate_vectors.py writes; regenerate it\n")
            return 1
        print("vectors.json is current")
        return 0
    with open(target, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("wrote", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
