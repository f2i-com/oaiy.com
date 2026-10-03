import json, random, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from ed import *

rng = random.Random(8032)
cases = []


def add_case(group, name, pk, msg, sig, note=""):
    assert len(pk) == 32 and len(sig) == 64, (name, len(pk), len(sig))
    cases.append({"group": group, "id": f"{group}/{len(cases):03d}/{name}", "pk": pk.hex(), "msg": msg.hex(), "sig": sig.hex(), "note": note})


def rnd(n):
    return bytes(rng.getrandbits(8) for _ in range(n))


tors = torsion_points()  # canonical enc -> point
tors_list = sorted(tors.items(), key=lambda kv: (order_of(kv[1]), kv[0]))
NONID = [(b, P) for b, P in tors_list if P != ID]

# ---- honest ----------------------------------------------------------------------------------------------------------------------
honest = []
for i, mlen in enumerate([0, 1, 3, 64, 1000, 65536]):
    seed = rnd(32)
    a, prefix, A = keypair(seed)
    Ab = compress(A)
    msg = rnd(mlen)
    sig, R, k = sign_with(a, prefix, Ab, msg)
    honest.append((a, prefix, A, Ab, msg, sig))
    add_case("honest", f"random{i}-len{mlen}", Ab, msg, sig)
# the RFC 8032 vectors 1 to 3
rfc = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60", ""),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb", "72"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7", "af82"),
]
for i, (s, m) in enumerate(rfc):
    a, prefix, A = keypair(bytes.fromhex(s))
    msg = bytes.fromhex(m)
    sig, R, k = sign_with(a, prefix, compress(A), msg)
    add_case("honest", f"rfc8032-{i+1}", compress(A), msg, sig)

# ---- bit flips -------------------------------------------------------------------------------------------------------------------
a, prefix, A, Ab, msg, sig = honest[2]
for pos in [0, 1, 100, 255, 256, 300, 400, 511]:
    s2 = bytearray(sig)
    s2[pos // 8] ^= 1 << (pos % 8)
    add_case("bitflip", f"sig-bit{pos}", Ab, msg, bytes(s2))
for pos in [0, 5, 128, 254, 255]:
    k2 = bytearray(Ab)
    k2[pos // 8] ^= 1 << (pos % 8)
    add_case("bitflip", f"pk-bit{pos}", bytes(k2), msg, sig)
add_case("bitflip", "msg-bit0", Ab, bytes([msg[0] ^ 1]) + msg[1:], sig)
add_case("bitflip", "msg-trunc", Ab, msg[:-1], sig)

# ---- non-canonical S -------------------------------------------------------------------------------------------------------------
for hi, (a, prefix, A, Ab, msg, sig) in enumerate(honest[:3]):
    Rb = sig[:32]
    S = int.from_bytes(sig[32:], "little")
    for m in [1, 2, 3, 4, 7, 8, 15]:
        S2 = S + m * L
        if S2 < 2**256:
            add_case("S_noncanon", f"h{hi}-S+{m}L(top3bits={'set' if S2 >> 253 else 'clear'})", Ab, msg, Rb + S2.to_bytes(32, "little"))
    for name, S2 in [("S=L", L), ("S=L-1", L - 1), ("S=2^253", 2**253), ("S|2^255", S | 2**255), ("S=2^256-1", 2**256 - 1), ("S=S+2^252", S + 2**252), ("S=0", 0)]:
        add_case("S_noncanon", f"h{hi}-{name}", Ab, msg, Rb + (S2 % 2**256).to_bytes(32, "little"))

# ---- A of small order ------------------------------------------------------------------------------------------------------------
raw_small = []
for h in SMALL_Y_HEX:
    yb = bytes.fromhex(h)
    for sign in (0, 1):
        b = bytearray(yb)
        b[31] |= sign << 7
        raw_small.append(bytes(b))
ID_CANON = compress(ID)
ID_NONCANON = bytes.fromhex("eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f")
ID_SIGN = bytearray(ID_CANON)
ID_SIGN[31] |= 0x80
ID_SIGN = bytes(ID_SIGN)
for raw in raw_small:
    P = decompress_loose(raw)
    n = order_of(P) if P is not None else None
    for rname, Rb, Rpt in [("R=id", ID_CANON, ID), ("R=id-noncanon(ee..7f)", ID_NONCANON, ID), ("R=id-signbit", ID_SIGN, ID)]:
        # a message for which k*A = 0, so that the cofactorless equation [0]B = R + kA holds
        found = None
        for i in range(400):
            msg = f"small-A msg {i}".encode()
            k = int.from_bytes(sha512(Rb, raw, msg), "little") % L
            if P is None or mul(k % (n or 1), P) == ID:
                found = msg
                break
        if found is None:
            continue
        add_case("A_small", f"A={raw.hex()[:8]}..{raw.hex()[60:]}-order{n}-{rname}-S=0", raw, found, Rb + bytes(32), f"order {n}")
    # honest R, S = r, A small: holds when kA = 0
    r = 123456789
    Rb = compress(mul(r, B))
    for i in range(400):
        msg = f"small-A honest-R msg {i}".encode()
        k = int.from_bytes(sha512(Rb, raw, msg), "little") % L
        if P is None or mul(k % (n or 1), P) == ID:
            add_case("A_small", f"A={raw.hex()[:8]}..{raw.hex()[60:]}-order{n}-R=rB-S=r", raw, msg, Rb + r.to_bytes(32, "little"), f"order {n}")
            break
# the case that matters most: A = identity, R = identity, S = 0 verifies every message under a non-strict verifier
for i, msg in enumerate([b"", b"anything", b"message two"]):
    add_case("A_small", f"identity-all-messages-{i}", ID_CANON, msg, ID_CANON + bytes(32), "key = 0100..00, sig = 0100..00 || 00..00")

# ---- R of small order, honest A ---------------------------------------------------------------------------------------------------
a, prefix, A, Ab, _, _ = honest[1]
for Rb, name in [(ID_CANON, "R=id"), (ID_NONCANON, "R=id-noncanon"), (ID_SIGN, "R=id-signbit")]:
    for i in range(2):
        msg = f"R-small {i}".encode()
        k = int.from_bytes(sha512(Rb, Ab, msg), "little") % L
        S = (k * a) % L
        add_case("R_small", f"{name}-S=ka-{i}", Ab, msg, Rb + S.to_bytes(32, "little"), "r = 0: [S]B = kA holds as points")
for Rb, T in NONID:
    msg = b"R-small torsion"
    k = int.from_bytes(sha512(Rb, Ab, msg), "little") % L
    S = (k * a) % L
    add_case("R_small", f"R=T{Rb.hex()[:8]}-order{order_of(T)}-S=ka", Ab, msg, Rb + S.to_bytes(32, "little"), "never valid as a point equation")

# ---- R of mixed order (cofactored-valid, cofactorless-invalid) --------------------------------------------------------------------
for Tb, T in NONID:
    r = int.from_bytes(sha512(b"mixedR", Tb), "little") % L
    R = add(mul(r, B), T)
    msg = b"mixed-order R"
    sig, _, k = sign_with(a, prefix, Ab, msg, r=r, R_override=R)
    add_case("R_mixed", f"R=rB+T{Tb.hex()[:8]}-order{order_of(T)}", Ab, msg, sig, "valid only after multiplying by the cofactor")

# ---- A of mixed order ------------------------------------------------------------------------------------------------------------
for hi in range(2):
    a, prefix, A, Ab, _, _ = honest[hi]
    for Tb, T in NONID:
        Am = add(A, T)
        Amb = compress(Am)
        n = order_of(T)
        got_valid = got_invalid = 0
        for i in range(200):
            msg = f"mixed-A {i}".encode()
            sig, R, k = sign_with(a, prefix, Amb, msg)
            ok = mul(k % n, T) == ID  # kT = 0
            if ok and got_valid < 2:
                add_case("A_mixed", f"h{hi}-A+T{Tb.hex()[:8]}-order{n}-kT=0-{i}", Amb, msg, sig, "cofactorless equation holds")
                got_valid += 1
            elif not ok and got_invalid < 2:
                add_case("A_mixed", f"h{hi}-A+T{Tb.hex()[:8]}-order{n}-kT!=0-{i}", Amb, msg, sig, "cofactorless equation fails, cofactored holds")
                got_invalid += 1
            if got_valid >= 2 and got_invalid >= 2:
                break

# A mixed and R mixed with kT1 + T2 = 0 (valid cofactorless, neither point of small order)
a, prefix, A, Ab, _, _ = honest[0]
T1b, T1 = [kv for kv in NONID if order_of(kv[1]) == 8][0]
Am = add(A, T1)
Amb = compress(Am)
count = 0
for i in range(2000):
    msg = f"mixed-both {i}".encode()
    r = int.from_bytes(sha512(b"mb", bytes([i % 256, i // 256])), "little") % L
    T2b, T2 = tors_list[i % 8]
    R = add(mul(r, B), T2)
    Rb = compress(R)
    k = int.from_bytes(sha512(Rb, Amb, msg), "little") % L
    if add(mul(k % 8, T1), T2) == ID:
        S = (r + k * a) % L
        add_case("AR_mixed", f"A+T1,R=rB+T2 valid-cofactorless-{count}", Amb, msg, Rb + S.to_bytes(32, "little"), "neither point has small order; [S]B = R + kA")
        count += 1
        if count >= 3:
            break

# ---- keys not on the curve, and non-canonical y of ordinary points -------------------------------------------------------------------
cnt = 0
for y in range(2, 5000):
    if recover_x(y, 0) is None and cnt < 3:
        add_case("A_invalid", f"y={y}-not-on-curve", y.to_bytes(32, "little"), b"m", bytes(64), "")
        cnt += 1
add_case("A_invalid", "all-ff", b"\xff" * 32, b"m", b"\xff" * 64)
add_case("A_invalid", "y=p (0xed.. 7f) sign set", bytes.fromhex("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"), b"m", bytes(64))
for y in range(2, 19):
    for sign in (0, 1):
        x = recover_x(y, sign)
        if x is None:
            continue
        P = (x, y)
        # the encoding y+p is non-canonical (it fits: y + p < 2^255 for y < 19)
        nc = (y + p) | (sign << 255)
        o = None
        in_main = mul(L, P) == ID
        add_case("A_noncanon_y", f"y={y}-sign{sign}-mainsubgroup={in_main}", nc.to_bytes(32, "little"), b"m", bytes(64), "noncanonical encoding of an ordinary point")
        add_case("A_noncanon_y", f"y={y}-sign{sign}-canonical-mainsubgroup={in_main}", compress(P), b"m", bytes(64), "the canonical encoding of the same point")

json.dump(cases, open(os.path.join(os.path.dirname(__file__), "cases.json"), "w"), indent=0)
from collections import Counter

print(len(cases), "cases", Counter(c["group"] for c in cases))
