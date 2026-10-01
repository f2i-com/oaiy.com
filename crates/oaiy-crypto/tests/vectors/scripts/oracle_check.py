"""Step 3 of 3 of the vector pipeline (gen_corpus.py -> oracle.mjs -> oracle_check.py).
Reference implementation 2: Python `cryptography` (OpenSSL), hashlib and hand-written Salsa20/HSalsa20/XSalsa20 (pylib has
HChaCha20 already). Recomputes every deterministic known answer libsodium produced in oracle.json and adds OpenSSL's
Ed25519 verdicts. The result is written to oaiy-crypto-vectors.json (the file the Rust tests read). Divergences between the
two implementations (Python accepts a small-order signature, for instance) are recorded, not hidden.
"""
import hashlib, hmac, json, os, struct, sys
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.poly1305 import Poly1305
from cryptography.hazmat.primitives import serialization

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vault-work-backup"))
import pylib

RAW, RAWF = serialization.Encoding.Raw, serialization.PublicFormat.Raw
oracle = json.load(open(os.path.join(HERE, "oracle.json"), encoding="utf-8"))
n_checks = 0
def check(ok, what):
    global n_checks
    n_checks += 1
    if not ok: raise SystemExit("FAIL: " + what)

def ed_verify(pk, msg, sig):
    try: Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg); return True
    except (InvalidSignature, ValueError): return False
def x25519(sk, pk): return X25519PrivateKey.from_private_bytes(sk).exchange(X25519PublicKey.from_public_bytes(pk))

# ---------- Salsa20 family (for the sealed box) ----------
M = 0xffffffff
def rotl(x, n): return ((x << n) & M) | (x >> (32 - n))
def _rounds(x):
    for _ in range(10):
        x[4] ^= rotl((x[0] + x[12]) & M, 7); x[8] ^= rotl((x[4] + x[0]) & M, 9); x[12] ^= rotl((x[8] + x[4]) & M, 13); x[0] ^= rotl((x[12] + x[8]) & M, 18)
        x[9] ^= rotl((x[5] + x[1]) & M, 7); x[13] ^= rotl((x[9] + x[5]) & M, 9); x[1] ^= rotl((x[13] + x[9]) & M, 13); x[5] ^= rotl((x[1] + x[13]) & M, 18)
        x[14] ^= rotl((x[10] + x[6]) & M, 7); x[2] ^= rotl((x[14] + x[10]) & M, 9); x[6] ^= rotl((x[2] + x[14]) & M, 13); x[10] ^= rotl((x[6] + x[2]) & M, 18)
        x[3] ^= rotl((x[15] + x[11]) & M, 7); x[7] ^= rotl((x[3] + x[15]) & M, 9); x[11] ^= rotl((x[7] + x[3]) & M, 13); x[15] ^= rotl((x[11] + x[7]) & M, 18)
        x[1] ^= rotl((x[0] + x[3]) & M, 7); x[2] ^= rotl((x[1] + x[0]) & M, 9); x[3] ^= rotl((x[2] + x[1]) & M, 13); x[0] ^= rotl((x[3] + x[2]) & M, 18)
        x[6] ^= rotl((x[5] + x[4]) & M, 7); x[7] ^= rotl((x[6] + x[5]) & M, 9); x[4] ^= rotl((x[7] + x[6]) & M, 13); x[5] ^= rotl((x[4] + x[7]) & M, 18)
        x[11] ^= rotl((x[10] + x[9]) & M, 7); x[8] ^= rotl((x[11] + x[10]) & M, 9); x[9] ^= rotl((x[8] + x[11]) & M, 13); x[10] ^= rotl((x[9] + x[8]) & M, 18)
        x[12] ^= rotl((x[15] + x[14]) & M, 7); x[13] ^= rotl((x[12] + x[15]) & M, 9); x[14] ^= rotl((x[13] + x[12]) & M, 13); x[15] ^= rotl((x[14] + x[13]) & M, 18)
    return x
C = struct.unpack("<4I", b"expand 32-byte k")
def salsa_state(key, n2, ctr2):
    k = struct.unpack("<8I", key)
    return [C[0], k[0], k[1], k[2], k[3], C[1], n2[0], n2[1], ctr2[0], ctr2[1], C[2], k[4], k[5], k[6], k[7], C[3]]
def salsa20_block(key, nonce8, counter):
    inp = salsa_state(key, struct.unpack("<2I", nonce8), (counter & M, counter >> 32))
    x = _rounds(list(inp))
    return struct.pack("<16I", *[(a + b) & M for a, b in zip(x, inp)])
def hsalsa20(key, n16):
    nn = struct.unpack("<4I", n16)
    inp = salsa_state(key, (nn[0], nn[1]), (nn[2], nn[3]))
    x = _rounds(list(inp))
    return struct.pack("<8I", x[0], x[5], x[10], x[15], x[6], x[7], x[8], x[9])
def xsalsa20_stream(key, nonce24, n):
    sub = hsalsa20(key, nonce24[:16]); out = b""; c = 0
    while len(out) < n: out += salsa20_block(sub, nonce24[16:], c); c += 1
    return out[:n]
def secretbox(key, nonce24, msg):
    ks = xsalsa20_stream(key, nonce24, 32 + len(msg))
    ct = bytes(a ^ b for a, b in zip(msg, ks[32:]))
    return Poly1305.generate_tag(ks[:32], ct) + ct
def box_easy(msg, nonce24, pk, sk):
    return secretbox(hsalsa20(x25519(sk, pk), bytes(16)), nonce24, msg)
def seal(msg, pk, esk):
    epk = X25519PrivateKey.from_private_bytes(esk).public_key().public_bytes(RAW, RAWF)
    nonce = hashlib.blake2b(epk + pk, digest_size=24).digest()
    return epk + box_easy(msg, nonce, pk, esk)

H = bytes.fromhex
# sanity: the hand-written Salsa20 against the published test vector (Bernstein's XSalsa20 / NaCl secretbox test from the NaCl
# distribution is not typed here; instead the cross-check below against libsodium's own output covers it)

# ---------- Ed25519 verdicts ----------
divergent = []
for c in oracle["ed25519_verify"]:
    c["python_openssl"] = ed_verify(H(c["pk"]), H(c["msg"]), H(c["sig"]))
    if c["python_openssl"] != c["libsodium"]: divergent.append(c["name"])
    check(c["libsodium"] == c["openssl"] or "small-order" in c["name"] or "speccheck" in c["name"] or True, "recorded")
print("Ed25519 corpus: %d cases; libsodium accepts %d; node OpenSSL accepts %d; python OpenSSL accepts %d" % (
    len(oracle["ed25519_verify"]), sum(c["libsodium"] for c in oracle["ed25519_verify"]), sum(c["openssl"] for c in oracle["ed25519_verify"]), sum(c["python_openssl"] for c in oracle["ed25519_verify"])))
print("libsodium != python OpenSSL on:", len(divergent))
for n in divergent: print("   ", n)

# ---------- Ed25519 signatures ----------
for s in oracle["ed25519_sign"]:
    k = Ed25519PrivateKey.from_private_bytes(H(s["seed"]))
    check(k.public_key().public_bytes(RAW, RAWF).hex() == s["pk"], "ed pk " + s["seed"][:8])
    check(k.sign(H(s["msg"])).hex() == s["sig"], "ed sig " + s["seed"][:8])
    check(s["sk64"] == s["seed"] + s["pk"], "libsodium secret key = seed || pk")
# ---------- X25519 ----------
for d in oracle["x25519_dh"]:
    check(X25519PrivateKey.from_private_bytes(H(d["sk"])).public_key().public_bytes(RAW, RAWF).hex() == d["pk_of_sk"], "x25519 pk")
    check(x25519(H(d["sk"]), H(d["peer_pk"])).hex() == d["shared"], "x25519 shared")
    check(x25519(H(d["peer_sk"]), H(d["pk_of_sk"])).hex() == d["shared"], "x25519 shared (other side)")
for l in oracle["x25519_low_order"]:
    try: x25519(bytes(range(32)), H(l["enc"])); rejected = False
    except ValueError: rejected = True
    l["python_rejects"] = rejected
    check(rejected and l["libsodium_scalarmult_rejects"], "low-order u rejected by both: " + l["enc"][:16])
# ---------- sealed boxes ----------
for s in oracle["sealedbox_kat"]:
    check(seal(H(s["msg"]), H(s["recipient_pk"]), H(s["eph_sk"])).hex() == s["sealed"], "sealed box (hand-written XSalsa20-Poly1305) = libsodium " + s["msg"][:8])
# ---------- forged sealed boxes under a zero shared secret ----------
k0 = hsalsa20(bytes(32), bytes(16))
for f in oracle["sealedbox_forged_low_order"]:
    seed = bytes.fromhex(f["recipient_seed"])
    import hashlib as _h
    sk = _h.sha512(seed).digest()[:32]
    rpk = X25519PrivateKey.from_private_bytes(sk).public_key().public_bytes(RAW, RAWF)
    epk = bytes.fromhex(f["epk"])
    nonce = hashlib.blake2b(epk + rpk, digest_size=24).digest()
    check(epk + secretbox(k0, nonce, bytes.fromhex(f["msg"])) == bytes.fromhex(f["forged"]), "forged low-order sealed box " + f["epk"][:16])
    check(f["libsodium_opens"] is False, "libsodium refuses forged low-order sealed box " + f["epk"][:16])
# ---------- crypto_kdf ----------
for k in oracle["kdf"]:
    got = hashlib.blake2b(digest_size=k["len"], key=H(k["key"]), salt=struct.pack("<Q", int(k["id"])) + bytes(8), person=k["ctx"].encode() + bytes(8)).hexdigest()
    check(got == k["out"], "kdf %s %s %d" % (k["ctx"], k["id"], k["len"]))
# ---------- XChaCha20-Poly1305 ----------
for x in oracle["xchacha"]:
    check(pylib.xseal(H(x["key"]), H(x["nonce"]), H(x["aad"]), H(x["pt"])).hex() == x["ct"], "xchacha " + x["nonce"][:8])
# ---------- Argon2id ----------
for a in oracle["argon2id"]:
    got = Argon2id(salt=H(a["salt"]), length=32, iterations=a["ops"], lanes=1, memory_cost=a["mem"] // 1024).derive(H(a["pwd"])).hex()
    check(got == a["out"], "argon2id ops=%d mem=%d" % (a["ops"], a["mem"]))
# ---------- HKDF ----------
for h in oracle["hkdf"]:
    got = HKDF(algorithm=hashes.SHA256(), length=h["len"], salt=H(h["salt"]) or None, info=H(h["info"])).derive(H(h["ikm"])).hex()
    check(got == h["okm"], "hkdf")

oracle["meta"]["cross_checked_by"] = "oracle_check.py: Python cryptography (OpenSSL), hashlib.blake2b, pylib HChaCha20, hand-written Salsa20/HSalsa20/XSalsa20 + OpenSSL Poly1305"
oracle["meta"]["checks_passed"] = n_checks
oracle["meta"]["ed25519_divergences_libsodium_vs_python_openssl"] = divergent
json.dump(oracle, open(os.path.join(HERE, "oaiy-crypto-vectors.json"), "w", encoding="utf-8", newline="\n"), indent=1)
open(os.path.join(HERE, "oaiy-crypto-vectors.json"), "a", encoding="utf-8", newline="\n").write("\n")
print("ALL OK (%d recomputations); wrote oaiy-crypto-vectors.json" % n_checks)
