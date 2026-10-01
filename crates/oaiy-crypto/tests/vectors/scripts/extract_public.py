"""Extracts the public known-answer vectors the oaiy-crypto tests use, from the published texts, and recomputes every
one of them with an independent implementation (Python `cryptography` + hashlib + pylib's hand-written HChaCha20).
Nothing is typed from memory: the hex comes out of the downloaded RFC / draft / trezor files (public/), the recomputation
must reproduce it, and only then is public-vectors.json written.

  python extract_public.py            (run from this folder)
"""
import hashlib, json, os, re, sys
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, "public")
sys.path.insert(0, os.path.join(HERE, "vault-work-backup"))   # pylib.py (the design's second implementation)
import pylib

RAW = serialization.Encoding.Raw
RAWF = serialization.PublicFormat.Raw

def read(name): return open(os.path.join(PUB, name), encoding="utf-8", newline="").read().replace("\r\n", "\n")
def sha(name): return hashlib.sha256(open(os.path.join(PUB, name), "rb").read()).hexdigest()
def hexblock(s): return "".join(re.findall(r"[0-9a-f]{2}", s))
def only_hex_lines(lines):
    out = ""
    for l in lines:
        m = re.fullmatch(r"\s+([0-9a-f]+)\s*", l)
        if m: out += m.group(1)
    return out

checks = []
def check(ok, what):
    checks.append((ok, what)); print(("ok   " if ok else "FAIL ") + what)
    if not ok: raise SystemExit("cross-check failed: " + what)

out = {"meta": {"note": "Extracted by extract_public.py from the published texts; every value recomputed by Python cryptography/hashlib/pylib.", "sources": {}}}
def src(key, url, fname): out["meta"]["sources"][key] = {"url": url, "sha256": sha(fname)}

# ---------------- RFC 5869 (HKDF-SHA256, cases 1-3) ----------------
t = read("rfc5869.txt")
src("rfc5869", "https://www.rfc-editor.org/rfc/rfc5869.txt", "rfc5869.txt")
cases = []
for n in (1, 2, 3):
    m = re.search(r"A\.%d\.  Test Case %d(.*?)(?=\nA\.%d\.|\Z)" % (n, n, n + 1), t, re.S)
    body = m.group(1)
    def field(name, nxt):
        mm = re.search(name + r"\s*=\s*(0x)?(.*?)(?=\n\s*(?:%s)\s*=|\Z)" % nxt, body, re.S)
        return mm.group(2)
    ikm = hexblock(re.search(r"IKM\s*=\s*0x(.*?)\(", body, re.S).group(1))
    salt = hexblock(re.search(r"salt\s*=\s*0x(.*?)\(", body, re.S).group(1)) if re.search(r"salt\s*=\s*0x", body) else ""
    info = hexblock(re.search(r"info\s*=\s*0x(.*?)\(", body, re.S).group(1)) if re.search(r"info\s*=\s*0x", body) else ""
    L = int(re.search(r"L\s*=\s*(\d+)", body).group(1))
    prk = hexblock(re.search(r"PRK\s*=\s*0x(.*?)\(", body, re.S).group(1))
    okm = hexblock(re.search(r"OKM\s*=\s*0x(.*?)\(", body, re.S).group(1))
    # independent recomputation
    import hmac
    prk2 = hmac.new(bytes.fromhex(salt) if salt else b"\0" * 32, bytes.fromhex(ikm), hashlib.sha256).digest()
    okm2 = HKDF(algorithm=hashes.SHA256(), length=L, salt=bytes.fromhex(salt) if salt else None, info=bytes.fromhex(info)).derive(bytes.fromhex(ikm))
    check(prk2.hex() == prk, "RFC 5869 case %d PRK (hmac)" % n)
    check(okm2.hex() == okm and len(okm) == 2 * L, "RFC 5869 case %d OKM (cryptography HKDF)" % n)
    cases.append({"case": n, "ikm": ikm, "salt": salt, "info": info, "len": L, "prk": prk, "okm": okm})
out["hkdf_sha256_rfc5869"] = cases

# ---------------- RFC 8032 (Ed25519 pure, section 7.1) ----------------
t = read("rfc8032.txt")
src("rfc8032", "https://www.rfc-editor.org/rfc/rfc8032.txt", "rfc8032.txt")
i0 = t.rindex("7.1.  Test Vectors for Ed25519")          # the last one: the first is in the table of contents
sec = t[i0:t.index("7.2.  Test Vectors for Ed25519ctx", i0)]
blocks = re.split(r"\n   -----TEST ", sec)[1:]
ed = []
for b in blocks:
    name = b.split("\n", 1)[0].strip()
    lines = b.split("\n")
    def section(label, end):
        i = next(k for k, l in enumerate(lines) if l.strip().startswith(label))
        j = next(k for k, l in enumerate(lines) if k > i and l.strip().startswith(end))
        return only_hex_lines(lines[i + 1:j])
    sk = section("SECRET KEY", "PUBLIC KEY")
    pk = section("PUBLIC KEY", "MESSAGE")
    mi = next(k for k, l in enumerate(lines) if l.strip().startswith("MESSAGE"))
    mlen = int(re.search(r"length (\d+) byte", lines[mi]).group(1))
    si = next(k for k, l in enumerate(lines) if l.strip().startswith("SIGNATURE"))
    msg = only_hex_lines(lines[mi + 1:si])
    sig = only_hex_lines(lines[si + 1:])
    check(len(msg) == 2 * mlen, "RFC 8032 TEST %s message length %d parsed" % (name, mlen))
    seed = bytes.fromhex(sk)
    k = Ed25519PrivateKey.from_private_bytes(seed)
    check(k.public_key().public_bytes(RAW, RAWF).hex() == pk, "RFC 8032 TEST %s public key" % name)
    check(k.sign(bytes.fromhex(msg)).hex() == sig, "RFC 8032 TEST %s signature (cryptography)" % name)
    Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk)).verify(bytes.fromhex(sig), bytes.fromhex(msg))
    ed.append({"name": name, "seed": sk, "public": pk, "message": msg, "signature": sig})
check(len(ed) == 5, "RFC 8032 has five Ed25519 vectors (1, 2, 3, 1024, SHA(abc))")
out["ed25519_rfc8032"] = ed

# ---------------- RFC 7748 (X25519) ----------------
t = read("rfc7748.txt")
src("rfc7748", "https://www.rfc-editor.org/rfc/rfc7748.txt", "rfc7748.txt")
i0 = t.rindex("5.2.  Test Vectors")
sec = t[i0:t.index("6.  Diffie-Hellman", i0)]
x = sec[sec.index("X25519:"):sec.index("X448:")]
trip = re.findall(r"Input scalar:\s*([0-9a-f]{64})\s*Input scalar as a number.*?Input u-coordinate:\s*([0-9a-f]{64})\s*Input u-coordinate as a number.*?Output u-coordinate:\s*([0-9a-f]{64})", x, re.S)
check(len(trip) == 2, "RFC 7748 5.2 has two X25519 vectors")
def x25519_py(k, u):
    return X25519PrivateKey.from_private_bytes(bytes.fromhex(k)).exchange(X25519PublicKey.from_public_bytes(bytes.fromhex(u))).hex()
scalarmult = []
for k, u, o in trip:
    check(x25519_py(k, u) == o, "RFC 7748 5.2 scalarmult recomputed (cryptography)")
    scalarmult.append({"scalar": k, "u": u, "out": o})
it = re.search(r"After one iteration:\s*([0-9a-f]{64})\s*After 1,000 iterations:\s*([0-9a-f]{64})\s*After 1,000,000 iterations:\s*([0-9a-f]{64})", sec[sec.index("X25519:"):], re.S)
k = u = "09" + "00" * 31
kk, uu = k, u
res = {}
# X25519 with python-cryptography objects, 1,000,000 iterations (about a minute)
for i in range(1, 1000001):
    kk, uu = x25519_py(kk, uu), kk
    if i == 1: res["after_1"] = kk
    if i == 1000: res["after_1000"] = kk
res["after_1000000"] = kk
check(res["after_1"] == it.group(1), "RFC 7748 iterated x1 recomputed")
check(res["after_1000"] == it.group(2), "RFC 7748 iterated x1000 recomputed")
check(res["after_1000000"] == it.group(3), "RFC 7748 iterated x1000000 recomputed")
i0 = t.rindex("6.1.  Curve25519")
sec6 = t[i0:t.index("6.2.  Curve448", i0)]
def f6(label):
    return re.search(label + r"[^\n]*\n\s*([0-9a-f]{64})", sec6).group(1)
dh = {"alice_private": re.search(r"Alice's private key, a:\s*([0-9a-f]{64})", sec6, re.S).group(1),
      "alice_public": re.search(r"Alice's public key, X25519\(a, 9\):\s*([0-9a-f]{64})", sec6, re.S).group(1),
      "bob_private": re.search(r"Bob's private key, b:\s*([0-9a-f]{64})", sec6, re.S).group(1),
      "bob_public": re.search(r"Bob's public key, X25519\(b, 9\):\s*([0-9a-f]{64})", sec6, re.S).group(1),
      "shared": re.search(r"Their shared secret, K:\s*([0-9a-f]{64})", sec6, re.S).group(1)}
base = "09" + "00" * 31
check(x25519_py(dh["alice_private"], base) == dh["alice_public"], "RFC 7748 6.1 Alice public")
check(x25519_py(dh["bob_private"], base) == dh["bob_public"], "RFC 7748 6.1 Bob public")
check(x25519_py(dh["alice_private"], dh["bob_public"]) == dh["shared"], "RFC 7748 6.1 shared (a, Bob)")
check(x25519_py(dh["bob_private"], dh["alice_public"]) == dh["shared"], "RFC 7748 6.1 shared (b, Alice)")
out["x25519_rfc7748"] = {"scalarmult": scalarmult, "iterated": {"k": k, "u": u, "after_1": res["after_1"], "after_1000": res["after_1000"], "after_1000000": res["after_1000000"]}, "dh": dh}

# ---------------- RFC 9106 (Argon2id) ----------------
t = read("rfc9106.txt")
src("rfc9106", "https://www.rfc-editor.org/rfc/rfc9106.txt", "rfc9106.txt")
i = [m.start() for m in re.finditer(r"Argon2id Test Vectors", t)][-1]
sec = t[i:t.index("6.  IANA Considerations", i)]
tag = hexblock(re.search(r"Tag:((?:\s+[0-9a-f]{2})+)", sec).group(1))
check(len(tag) == 64, "RFC 9106 Argon2id tag parsed (32 bytes)")
pwd, salt, secret, ad = bytes([1]) * 32, bytes([2]) * 16, bytes([3]) * 8, bytes([4]) * 12
tag2 = Argon2id(salt=salt, length=32, iterations=3, lanes=4, memory_cost=32, ad=ad, secret=secret).derive(pwd)
check(tag2.hex() == tag, "RFC 9106 Argon2id recomputed (cryptography, secret + ad + 4 lanes)")
out["argon2id_rfc9106"] = {"password": pwd.hex(), "salt": salt.hex(), "secret": secret.hex(), "ad": ad.hex(), "t_cost": 3, "m_cost_kib": 32, "lanes": 4, "tag_len": 32, "tag": tag}

# ---------------- draft-irtf-cfrg-xchacha-03, A.3.1 ----------------
t = read("xchacha-03.txt")
src("xchacha_draft_03", "https://www.ietf.org/archive/id/draft-irtf-cfrg-xchacha-03.txt", "xchacha-03.txt")
i = [m.start() for m in re.finditer(r"A\.3\.1\.  AEAD_XCHACHA20_POLY1305", t)][-1]
sec = t[i:t.index("A.3.2.  XChaCha20", i)]
def blk(label, nxt):
    return hexblock(re.search(label + r":\s*((?:\s*[0-9a-f]+\n)+)", sec).group(1))
pt, aad, key, iv, ct, tg = blk("Plaintext", ""), blk("AAD", ""), blk("Key", ""), blk("IV", ""), blk("Ciphertext", ""), blk("Tag", "")
check(len(key) == 64 and len(iv) == 48 and len(tg) == 32, "xchacha draft A.3.1 field sizes")
sealed = pylib.xseal(bytes.fromhex(key), bytes.fromhex(iv), bytes.fromhex(aad), bytes.fromhex(pt))
check(sealed.hex() == ct + tg, "xchacha draft A.3.1 recomputed (pylib HChaCha20 + cryptography ChaCha20Poly1305)")
out["xchacha20poly1305_draft03"] = {"key": key, "nonce": iv, "aad": aad, "plaintext": pt, "ciphertext": ct, "tag": tg}

# ---------------- BIP-39: trezor vectors (128-bit entropy) + the official list ----------------
src("bip39_vectors", "https://raw.githubusercontent.com/trezor/python-mnemonic/master/vectors.json", "bip39-vectors.json")
src("bip39_english", "https://raw.githubusercontent.com/bitcoin/bips/master/bip-0039/english.txt", "bip39-english.txt")
words = pylib.load_words(os.path.join(PUB, "bip39-english.txt"))
check(sha("bip39-english.txt") == "2f5eed53a4727b4bf8880d8f3f199efc90e58503646d9ff8eff3a2ed3b24dbda", "official english.txt SHA-256 equals the design's")
vec = json.load(open(os.path.join(PUB, "bip39-vectors.json"), encoding="utf-8"))["english"]
b39 = []
for e, m, seed, xprv in vec:
    if len(e) != 32: continue
    check(pylib.entropy_to_words(words, bytes.fromhex(e)) == m, "BIP-39 vector %s... encodes (pylib)" % e[:8])
    back, err = pylib.words_to_entropy(words, m)
    check(err is None and back.hex() == e, "BIP-39 vector %s... decodes (pylib)" % e[:8])
    b39.append({"entropy": e, "mnemonic": m})
check(len(b39) == 8, "eight 128-bit official BIP-39 vectors")
out["bip39_trezor_128"] = b39

json.dump(out, open(os.path.join(HERE, "public-vectors.json"), "w", encoding="utf-8", newline="\n"), indent=2)
open(os.path.join(HERE, "public-vectors.json"), "a", encoding="utf-8", newline="\n").write("\n")
print("\nALL OK (%d checks); wrote public-vectors.json" % len(checks))
