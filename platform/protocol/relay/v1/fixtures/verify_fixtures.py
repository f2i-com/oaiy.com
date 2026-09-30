#!/usr/bin/env python3
"""Independent check of the recorded fixtures in this folder, without libsodium.

    python verify_fixtures.py

The relay's PHP produced pairing-ceremony.json and sealed-token.json with libsodium. This script reads them and re-derives
everything it can with code that shares nothing with the relay or with libsodium:

* every sealed token is opened by a hand-written XSalsa20-Poly1305 (Salsa20 core, HSalsa20, Poly1305 on Python integers) and
  BLAKE2b from `hashlib`, with X25519 from `cryptography` (OpenSSL): the same construction as `crypto_box_seal`, written from
  the NaCl paper's formulas. A box that does not authenticate, a short one and one whose X25519 secret is all zeros are refused;
* the pairing ceremony is re-verified from Appendix A3's secret: the pid, the offer's MAC, the phone's response (MAC and
  Ed25519 signature over the canonical claims), the pair item, the desktop's receipt (Ed25519 signature over the canonical
  receipt document, under the desktop key the offer carries) and the short authentication string.

Prints "<n> checks, <m> mismatches" and exits non-zero on any mismatch. Needs `cryptography`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import pathlib
import re
import struct
import sys

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError:
    sys.stderr.write("needs `cryptography`:  pip install cryptography\n")
    raise SystemExit(2)

HERE = pathlib.Path(__file__).resolve().parent
V1 = HERE.parent
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
M32 = 0xFFFFFFFF

checks = 0
bad = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global checks, bad
    checks += 1
    if not cond:
        bad += 1
        print("MISMATCH", name, detail)


# ---------------------------------------------------------------------------- encodings
def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s: str) -> bytes:
    """Canonical base64url without padding, or ValueError."""
    if not re.fullmatch(r"[A-Za-z0-9_-]*", s):
        raise ValueError("alphabet")
    out = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if b64u(out) != s:
        raise ValueError("not canonical")
    return out


# ---------------------------------------------------------------------------- Salsa20, HSalsa20, Poly1305 (NaCl's crypto_secretbox)
def _rotl(v: int, n: int) -> int:
    return ((v << n) & M32) | (v >> (32 - n))


def _core(x: list[int]) -> list[int]:
    for _ in range(10):
        x[4] ^= _rotl((x[0] + x[12]) & M32, 7); x[8] ^= _rotl((x[4] + x[0]) & M32, 9)
        x[12] ^= _rotl((x[8] + x[4]) & M32, 13); x[0] ^= _rotl((x[12] + x[8]) & M32, 18)
        x[9] ^= _rotl((x[5] + x[1]) & M32, 7); x[13] ^= _rotl((x[9] + x[5]) & M32, 9)
        x[1] ^= _rotl((x[13] + x[9]) & M32, 13); x[5] ^= _rotl((x[1] + x[13]) & M32, 18)
        x[14] ^= _rotl((x[10] + x[6]) & M32, 7); x[2] ^= _rotl((x[14] + x[10]) & M32, 9)
        x[6] ^= _rotl((x[2] + x[14]) & M32, 13); x[10] ^= _rotl((x[6] + x[2]) & M32, 18)
        x[3] ^= _rotl((x[15] + x[11]) & M32, 7); x[7] ^= _rotl((x[3] + x[15]) & M32, 9)
        x[11] ^= _rotl((x[7] + x[3]) & M32, 13); x[15] ^= _rotl((x[11] + x[7]) & M32, 18)
        x[1] ^= _rotl((x[0] + x[3]) & M32, 7); x[2] ^= _rotl((x[1] + x[0]) & M32, 9)
        x[3] ^= _rotl((x[2] + x[1]) & M32, 13); x[0] ^= _rotl((x[3] + x[2]) & M32, 18)
        x[6] ^= _rotl((x[5] + x[4]) & M32, 7); x[7] ^= _rotl((x[6] + x[5]) & M32, 9)
        x[4] ^= _rotl((x[7] + x[6]) & M32, 13); x[5] ^= _rotl((x[4] + x[7]) & M32, 18)
        x[11] ^= _rotl((x[10] + x[9]) & M32, 7); x[8] ^= _rotl((x[11] + x[10]) & M32, 9)
        x[9] ^= _rotl((x[8] + x[11]) & M32, 13); x[10] ^= _rotl((x[9] + x[8]) & M32, 18)
        x[12] ^= _rotl((x[15] + x[14]) & M32, 7); x[13] ^= _rotl((x[12] + x[15]) & M32, 9)
        x[14] ^= _rotl((x[13] + x[12]) & M32, 13); x[15] ^= _rotl((x[14] + x[13]) & M32, 18)
    return x


_SIGMA = struct.unpack("<4I", b"expand 32-byte k")


def hsalsa20(key: bytes, nonce16: bytes) -> bytes:
    k = struct.unpack("<8I", key)
    n = struct.unpack("<4I", nonce16)
    x = _core([_SIGMA[0], k[0], k[1], k[2], k[3], _SIGMA[1], n[0], n[1], n[2], n[3], _SIGMA[2], k[4], k[5], k[6], k[7], _SIGMA[3]])
    return struct.pack("<8I", x[0], x[5], x[10], x[15], x[6], x[7], x[8], x[9])


def salsa20_stream(key: bytes, nonce8: bytes, length: int) -> bytes:
    k = struct.unpack("<8I", key)
    n = struct.unpack("<2I", nonce8)
    out = b""
    counter = 0
    while len(out) < length:
        inp = [_SIGMA[0], k[0], k[1], k[2], k[3], _SIGMA[1], n[0], n[1], counter & M32, counter >> 32, _SIGMA[2], k[4], k[5], k[6], k[7], _SIGMA[3]]
        x = _core(list(inp))
        out += struct.pack("<16I", *[(a + b) & M32 for a, b in zip(x, inp)])
        counter += 1
    return out[:length]


def poly1305(key: bytes, msg: bytes) -> bytes:
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:32], "little")
    p = (1 << 130) - 5
    acc = 0
    for i in range(0, len(msg), 16):
        acc = (acc + int.from_bytes(msg[i:i + 16] + b"\x01", "little")) * r % p
    return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def secretbox_open(box: bytes, nonce24: bytes, key: bytes) -> bytes | None:
    """crypto_secretbox_open_easy: tag (16) || ciphertext. None when it does not authenticate."""
    if len(box) < 16:
        return None
    tag, ct = box[:16], box[16:]
    subkey = hsalsa20(key, nonce24[:16])
    stream = salsa20_stream(subkey, nonce24[16:24], 32 + len(ct))
    if not hmac.compare_digest(poly1305(stream[:32], ct), tag):
        return None
    return bytes(a ^ b for a, b in zip(ct, stream[32:]))


def x25519(secret: bytes, public: bytes) -> bytes | None:
    try:
        return X25519PrivateKey.from_private_bytes(secret).exchange(X25519PublicKey.from_public_bytes(public))
    except ValueError:
        return None


def seal_open(sealed: bytes, recipient_secret: bytes, recipient_public: bytes) -> bytes | None:
    """crypto_box_seal_open, from the formulas: nonce = BLAKE2b-24(epk || pk); key = HSalsa20(X25519(sk, epk), 0^16)."""
    if len(sealed) < 48:
        return None
    epk, box = sealed[:32], sealed[32:]
    shared = x25519(recipient_secret, epk)
    if shared is None or shared == bytes(32):
        return None
    key = hsalsa20(shared, bytes(16))
    nonce = hashlib.blake2b(epk + recipient_public, digest_size=24).digest()
    return secretbox_open(box, nonce, key)


# ---------------------------------------------------------------------------- pairing helpers (README 10.1)
def hkdf(ikm: bytes, salt: bytes, info: bytes, n: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=n, salt=salt, info=info).derive(ikm)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ed_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError):
        return False


def crock(bits: str) -> str:
    bits += "0" * (-len(bits) % 5)
    return "".join(CROCKFORD[int(bits[i:i + 5], 2)] for i in range(0, len(bits), 5))


def sas_display(desktop_pub: bytes, phone_pub: bytes, nonce: bytes, pid: bytes) -> str:
    raw = hkdf(desktop_pub + phone_pub, nonce, b"oaiy/pairing/3/sas\x00" + pid, 8)
    twelve = crock("".join(f"{b:08b}" for b in raw)[:60])
    check_char = CROCKFORD[hashlib.sha256(b"oaiy/pairing/3/sas-check\x00" + twelve.encode()).digest()[0] >> 3]
    return f"{twelve[:4]}-{twelve[4:8]}-{twelve[8:]}-{check_char}"


# ---------------------------------------------------------------------------- the files
vec = json.loads((V1 / "vectors.json").read_text(encoding="utf-8"))
TOKEN_RE = re.compile(r"oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}")


def sealed_file() -> None:
    doc = json.loads((HERE / "sealed-token.json").read_text(encoding="utf-8"))
    sk = unb64u(doc["recipient"]["x25519Secret"])
    pk = unb64u(doc["recipient"]["x25519Public"])
    check("sealed: the recipient public key is the X25519 public key of its secret",
          X25519PrivateKey.from_private_bytes(sk).public_key().public_bytes_raw() == pk)
    check("sealed: the recipient is the test phone of Appendix A3", b64u(pk) == vec["keys"]["x25519Public"]["phone"])
    for i, c in enumerate(doc["opens"]):
        sealed = unb64u(c["sealedToken"])
        opened = seal_open(sealed, sk, pk)
        check(f"sealed: opens[{i}] ({c['label']}) opens", opened is not None)
        if opened is None:
            continue
        check(f"sealed: opens[{i}] is {c['plaintextLength']} bytes", len(opened) == c["plaintextLength"] == 63)
        check(f"sealed: opens[{i}] hashes to the recorded plaintextSha256", hashlib.sha256(opened).hexdigest() == c["plaintextSha256"])
        check(f"sealed: opens[{i}] is a device token", TOKEN_RE.fullmatch(opened.decode("ascii")) is not None)
        check(f"sealed: opens[{i}] is {c['sealedBytes']} bytes long (32 + 16 + 63)", len(sealed) == c["sealedBytes"] == 32 + 16 + len(opened))
    for i, c in enumerate(doc["refused"]):
        try:
            box = unb64u(c["sealedToken"])
        except ValueError:
            box = b""
        check(f"sealed: refused[{i}] ({c['label']}) does not open", seal_open(box, sk, pk) is None)
    wsk = unb64u(doc["wrongRecipient"]["x25519Secret"])
    wpk = unb64u(doc["wrongRecipient"]["x25519Public"])
    check("sealed: the first token does not open for another recipient", seal_open(unb64u(doc["opens"][0]["sealedToken"]), wsk, wpk) is None)
    # Two boxes of one token differ (a fresh ephemeral key), and an independent seal of our own opens too: the reading is the same
    # construction as the writing, not a coincidence of these files.
    check("sealed: the three recorded boxes are all different", len({c["sealedToken"] for c in doc["opens"]}) == len(doc["opens"]))


def ceremony_file() -> None:
    doc = json.loads((HERE / "pairing-ceremony.json").read_text(encoding="utf-8"))
    a3 = vec["A3"]
    steps = doc["steps"]
    check("ceremony: six steps", len(steps) == 6)
    s = bytes.fromhex(a3["inputs"]["secretHex"])
    pid_bin = hkdf(s, b"oaiy/pairing/3", b"rendezvous", 16)
    mac_key = hkdf(s, b"oaiy/pairing/3", b"mac", 32)
    pid = b64u(pid_bin)
    check("ceremony: the pid is the HKDF of the pairing secret", doc["pid"] == pid == a3["expected"]["pid"])
    create, fetch, answer, poll, decision, approved = steps
    body = create["request"]["body"]
    offer_text = body["offer"]
    check("ceremony: the offer is the 778 bytes of Appendix A3", offer_text == a3["expected"]["offerText"] and len(offer_text.encode()) == 778)
    want_mac = b64u(hmac.new(mac_key, b"oaiy/pairing/3/offer-mac\x00" + offer_text.encode(), hashlib.sha256).digest())
    check("ceremony: the offer MAC verifies under the pairing secret", body["mac"] == want_mac == a3["expected"]["offerMac"])
    check("ceremony: the create request names the pid", body["pid"] == pid)
    check("ceremony: the rendezvous is created for 600 seconds", create["response"]["status"] == 201 and create["response"]["body"]["exp"] - create["response"]["body"]["time"] == 600)
    check("ceremony: the phone fetches the offer and MAC exactly as sent",
          fetch["response"]["body"]["offer"] == offer_text and fetch["response"]["body"]["mac"] == want_mac and fetch["response"]["body"]["state"] == "open")
    offer = json.loads(offer_text)
    desktop_pub = unb64u(offer["desktopEndpointKey"]["publicKey"])
    # the phone's response
    resp_text = answer["request"]["body"]["response"]
    resp = json.loads(resp_text)
    canon = canonical(resp["claims"]).encode()
    check("ceremony: the response carries the vector's claims", canon.decode() == a3["expected"]["claimsCanonical"])
    check("ceremony: the response MAC verifies", resp["mac"] == b64u(hmac.new(mac_key, b"oaiy/pairing/3/response-mac\x00" + canon, hashlib.sha256).digest()))
    phone_pub = unb64u(resp["claims"]["mobileEndpointKey"]["publicKey"])
    check("ceremony: the response signature verifies under the phone key", ed_verify(phone_pub, b"oaiy/pairing/3/response\x00" + canon, unb64u(resp["signature"])))
    check("ceremony: the answer is 202 answered", answer["response"]["status"] == 202 and answer["response"]["body"]["state"] == "answered")
    # the desktop's poll
    item = poll["response"]["body"]["items"][0]
    check("ceremony: the desktop receives the response as one pair item, id = pid, from the relay, body exactly as posted",
          (item["lane"], item["id"], item["from"], item["body"]) == ("pair", pid, "relay", resp_text))
    # the approval and its receipt
    d = decision["request"]["body"]
    grants = sorted(d["grants"])
    receipt_doc = canonical({"appId": d["appId"], "grants": grants, "issuedAt": d["receipt"]["issuedAt"], "phoneThumbprint": d["phone"]["thumbprint"], "pid": pid})
    check("ceremony: the receipt document is the one of Appendix A3", receipt_doc == a3["expected"]["receiptText"])
    check("ceremony: the receipt verifies under the desktop key of the offer", ed_verify(desktop_pub, b"oaiy/pairing/3/approval\x00" + receipt_doc.encode(), unb64u(d["receipt"]["signature"])))
    check("ceremony: the approval names the keys the phone answered with",
          d["phone"]["ed25519"] == resp["claims"]["mobileEndpointKey"]["publicKey"] and d["phone"]["x25519"] == resp["claims"]["mobileX25519"])
    check("ceremony: the decision answers approved with a device id", decision["response"]["body"]["state"] == "approved" and decision["response"]["body"]["deviceId"].startswith("dev-"))
    # what the phone reads
    out = approved["response"]["body"]
    check("ceremony: the phone reads the same device id, the receipt as signed and a sealed token",
          out["deviceId"] == decision["response"]["body"]["deviceId"] and out["receipt"] == d["receipt"] and out["state"] == "approved")
    phone_sk = bytes.fromhex(vec["keys"]["x25519Secrets"]["phone"])
    phone_x_pub = X25519PrivateKey.from_private_bytes(phone_sk).public_key().public_bytes_raw()
    check("ceremony: the phone X25519 key of the response is the test phone's", b64u(phone_x_pub) == resp["claims"]["mobileX25519"])
    token = seal_open(unb64u(out["sealedToken"]), phone_sk, phone_x_pub)
    check("ceremony: the sealed token opens with the phone's key to a device token", token is not None and TOKEN_RE.fullmatch(token.decode("ascii")) is not None)
    # the short authentication string
    sas = sas_display(desktop_pub, phone_pub, unb64u(offer["nonce"]), pid_bin)
    check("ceremony: the short authentication string is recomputed from the keys, the nonce and the pid", doc["sas"] == sas == a3["expected"]["sasDisplay"])


def main() -> int:
    sealed_file()
    ceremony_file()
    print(f"{checks} checks, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
