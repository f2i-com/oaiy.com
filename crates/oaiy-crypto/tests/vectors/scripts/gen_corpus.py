"""Step 1 of 3 of the negative-corpus pipeline (gen_corpus.py -> oracle.mjs -> oracle_check.py).

Builds the candidate cases for the oaiy-crypto negative tests with plain Python integer arithmetic (no crypto library):
the eight points of order dividing 8 on the Ed25519 curve, their canonical and non-canonical encodings, signatures that
are malleable or non-canonical, and the ed25519-speccheck cases. Writes corpus-in.json. Step 2 (oracle.mjs, libsodium)
adds libsodium's verdicts; step 3 (oracle_check.py, Python cryptography/OpenSSL) adds OpenSSL's and re-derives the parts
that are computable, and writes the final tests/vectors files.
"""
import hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
d = (-121665 * pow(121666, p - 2, p)) % p
I = pow(2, (p - 1) // 4, p)            # sqrt(-1)

def inv(x): return pow(x, p - 2, p)
def add(P, Q):                          # twisted Edwards, a = -1, affine (complete formulas)
    (x1, y1), (x2, y2) = P, Q
    x3 = (x1 * y2 + x2 * y1) * inv(1 + d * x1 * x2 * y1 * y2) % p
    y3 = (y1 * y2 + x1 * x2) * inv(1 - d * x1 * x2 * y1 * y2) % p
    return (x3, y3)
def mul(k, P):
    R = (0, 1)
    while k:
        if k & 1: R = add(R, P)
        P = add(P, P); k >>= 1
    return R
def recover_x(y, sign):
    xx = (y * y - 1) * inv(d * y * y + 1) % p
    x = pow(xx, (p + 3) // 8, p)
    if (x * x - xx) % p: x = x * I % p
    if (x * x - xx) % p: return None
    if x == 0 and sign: return None
    if x & 1 != sign: x = p - x
    return x
def enc(x, y): return (y | ((x & 1) << 255)).to_bytes(32, "little")
Bx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
By = 46316835694926478169428394003475163141307993866256225615783033603165251855960
B = (Bx, By)
assert (Bx, By) == (recover_x(By, Bx & 1), By)

# the torsion subgroup: multiply a random point by L (kills the prime-order part), then walk it
P0 = None
for y in range(2, 1000):
    x = recover_x(y, 0)
    if x is not None:
        P0 = (x, y); break
T = mul(L, P0)
tors = {(0, 1)}
cur = T
for _ in range(16):
    tors.add(cur); cur = add(cur, T)
assert len(tors) == 8, len(tors)
# order of each
def order(P):
    k, Q = 1, P
    while Q != (0, 1): Q = add(Q, P); k += 1
    return k
orders = {P: order(P) for P in tors}
assert sorted(orders.values()) == [1, 2, 4, 4, 8, 8, 8, 8]

small = []   # (name, 32-byte encoding, canonical?)
seen = set()
def note(name, b, canonical):
    if b in seen: return
    seen.add(b); small.append({"name": name, "enc": b.hex(), "canonical": canonical})
for (x, y), o in sorted(orders.items(), key=lambda t: (t[1], t[0])):
    e = enc(x, y)
    note("order %d, canonical %s" % (o, e.hex()[:8]), e, True)
    if x == 0:
        # x = 0 with the sign bit set is a non-canonical encoding of the same point
        e2 = (y | (1 << 255)).to_bytes(32, "little")
        note("order %d, x=0 with the sign bit set, non-canonical %s" % (o, e2.hex()[:8]), e2, False)
    if y + p < 2**255:
        e3 = (y + p | ((x & 1) << 255)).to_bytes(32, "little")
        note("order %d, y+p, non-canonical %s" % (o, e3.hex()[:8]), e3, False)
        if x == 0:
            e4 = (y + p | (1 << 255)).to_bytes(32, "little")
            note("order %d, y+p with the sign bit set, non-canonical %s" % (o, e4.hex()[:8]), e4, False)
# the two order-8 points of the Montgomery curve's twist do not exist on the twisted Edwards curve; nothing else to add

def h(b): return b.hex()
public = json.load(open(os.path.join(HERE, "public-vectors.json"), encoding="utf-8"))
ed = public["ed25519_rfc8032"]
speccheck = json.load(open(os.path.join(HERE, "public", "speccheck-cases.json"), encoding="utf-8"))

cases = []
def case(name, pk, msg, sig, expect_note=""):
    cases.append({"name": name, "pk": pk, "msg": msg, "sig": sig, "note": expect_note})

ident_R = (1).to_bytes(32, "little")
zero_S = bytes(32)
for v in ed[:3]:
    case("rfc8032 %s (valid: the positive control)" % v["name"], v["public"], v["message"], v["signature"])
    sig = bytes.fromhex(v["signature"]); R, S = sig[:32], int.from_bytes(sig[32:], "little")
    case("rfc8032 %s with S+L (malleated, non-canonical S)" % v["name"], v["public"], v["message"], h(R + (S + L).to_bytes(32, "little")))
    case("rfc8032 %s with S+2L" % v["name"], v["public"], v["message"], h(R + (S + 2 * L).to_bytes(32, "little")))
    case("rfc8032 %s with the three top bits of S set" % v["name"], v["public"], v["message"], h(R + (S | (7 << 253)).to_bytes(32, "little")))
    case("rfc8032 %s with S = L" % v["name"], v["public"], v["message"], h(R + L.to_bytes(32, "little")))
    case("rfc8032 %s with S = 2^256-1" % v["name"], v["public"], v["message"], h(R + b"\xff" * 32))
    case("rfc8032 %s with S = 0" % v["name"], v["public"], v["message"], h(R + zero_S))
    case("rfc8032 %s with R = the identity" % v["name"], v["public"], v["message"], h(ident_R + sig[32:]))
    case("rfc8032 %s with one bit of R flipped" % v["name"], v["public"], v["message"], h(bytes([sig[0] ^ 1]) + sig[1:]))
    case("rfc8032 %s with one bit of the message flipped" % v["name"], v["public"], (bytes([bytes.fromhex(v["message"] or "00")[0] ^ 1]) + bytes.fromhex(v["message"] or "00")[1:]).hex(), v["signature"])
    case("rfc8032 %s under another public key" % v["name"], ed[(ed.index(v) + 1) % 3]["public"], v["message"], v["signature"])
# small-order public keys: the equation holds for every message when R = identity and S = 0 (plain RFC 8032 verification accepts)
for s in small:
    for rname, R in (("R=identity", ident_R), ("R=A", bytes.fromhex(s["enc"]))):
        case("small-order A [%s], %s, S=0" % (s["name"], rname), s["enc"], b"any message at all".hex(), h(R + zero_S))
# small-order R with an honest key
case("honest A, small-order R (identity), S = 1", ed[0]["public"], ed[0]["message"], h(ident_R + (1).to_bytes(32, "little")))
# A public key of MIXED order (a·B + T, T of order 8) is a perfectly good key for every verifier; what strict verification adds is refusing a
# SMALL-ORDER R. Without that, the cofactorless equation holds for a forged (R, S) whenever h·T cancels R: with R = -k·T and h = k (mod 8),
# S = h·a gives sB = R + hA. So these two cases are accepted by plain RFC 8032 verification and refused by every strict one (libsodium, dalek's
# verify_strict), and they are the ones that tell a strict verifier from a plain one when the key itself is acceptable.
T8 = next(P for P, o in orders.items() if o == 8)
def neg(P): return ((-P[0]) % p, P[1])
def smul(k, P):
    return mul(k % 8, P) if k % 8 else (0, 1)
def sha512_int(b): return int.from_bytes(hashlib.sha512(b).digest(), "little")
a_scalar = sha512_int(b"mixed-order-secret-scalar") % L
A_mixed = add(mul(a_scalar, B), T8)
assert mul(8, A_mixed) != (0, 1)                                    # not of small order (8*A would be the identity)
assert mul(L, A_mixed) == mul(L % 8, T8)                            # mixed order: the prime-order part is gone after L, the torsion part remains
A_enc = enc(*A_mixed)
for k in (0, 3):
    R_pt = neg(smul(k, T8)) if k else (0, 1)
    R_enc = enc(*R_pt)
    assert orders[R_pt] in (1, 2, 4, 8)                             # R is of small order
    i = 0
    while True:
        m = b"mixed-order case %d attempt %d" % (k, i)
        hh = sha512_int(R_enc + A_enc + m) % L
        if hh % 8 == k: break
        i += 1
    S = (hh * a_scalar) % L
    # check the cofactorless equation by hand: S*B == R + h*A
    assert mul(S, B) == add(R_pt, mul(hh, A_mixed))
    case("mixed-order A with small-order R (k=%d): the plain equation holds, strict verification refuses" % k, A_enc.hex(), m.hex(), (R_enc + S.to_bytes(32, "little")).hex())
# random-looking garbage
for i in range(4):
    g = hashlib.sha512(b"garbage%d" % i).digest()
    case("garbage %d" % i, h(g[:32]), h(g[32:36]), h(hashlib.sha512(g).digest()))
for i, c in enumerate(speccheck):
    case("ed25519-speccheck case %d" % i, c["pub_key"], c["message"], c["signature"])

# sign-bit / y-range probes for the public key parse (no signature can exist for the non-small-order ones: only the parse is tested)
nc_parse = []
for y in range(0, 19):
    for sign in (0, 1):
        x = recover_x(y, sign)
        b = (y + p | (sign << 255)).to_bytes(32, "little")   # non-canonical y (>= p)
        nc_parse.append({"y": y, "sign": sign, "on_curve": x is not None, "enc_noncanonical": b.hex(),
                         "enc_canonical": (y | (sign << 255)).to_bytes(32, "little").hex() if x is not None else None,
                         "small_order": (x is not None and orders.get((x, y)) is not None)})

out = {"small_order_encodings": small, "cases": cases, "noncanonical_y_probe": nc_parse,
       "L": L.to_bytes(32, "little").hex(), "p_minus_1_le": (p - 1).to_bytes(32, "little").hex()}
json.dump(out, open(os.path.join(HERE, "corpus-in.json"), "w", encoding="utf-8", newline="\n"), indent=1)
print("small-order encodings:", len(small), " cases:", len(cases), " non-canonical y probes:", len(nc_parse))
for s in small: print("  ", s["canonical"], s["name"], s["enc"][:20] + "...")
