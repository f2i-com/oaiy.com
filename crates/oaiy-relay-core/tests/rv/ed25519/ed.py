"""Minimal Ed25519 (RFC 8032 reference arithmetic) for building edge cases. Not for production."""
import hashlib

p = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
d = (-121665 * pow(121666, p - 2, p)) % p
I = pow(2, (p - 1) // 4, p)  # sqrt(-1)


def inv(x):
    return pow(x, p - 2, p)


def recover_x(y, sign):
    """x from y and sign bit, None if not on the curve. y must be < p (caller reduces)."""
    x2 = ((y * y - 1) * inv(d * y * y + 1)) % p
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (p + 3) // 8, p)
    if (x * x - x2) % p != 0:
        x = (x * I) % p
    if (x * x - x2) % p != 0:
        return None
    if (x & 1) != sign:
        x = p - x
    return x


# points are affine (x, y) for simplicity (speed is irrelevant here)
def add(P, Q):
    x1, y1 = P
    x2, y2 = Q
    t = (d * x1 * x2 * y1 * y2) % p
    x3 = ((x1 * y2 + x2 * y1) * inv(1 + t)) % p
    y3 = ((y1 * y2 + x1 * x2) * inv(1 - t)) % p
    return (x3, y3)


ID = (0, 1)


def mul(s, P):
    Q = ID
    while s > 0:
        if s & 1:
            Q = add(Q, P)
        P = add(P, P)
        s >>= 1
    return Q


def neg(P):
    return ((-P[0]) % p, P[1])


By = (4 * inv(5)) % p
Bx = recover_x(By, 0)
B = (Bx, By)


def compress(P):
    x, y = P
    return int(y | ((x & 1) << 255)).to_bytes(32, "little")


def decompress_strict(b):
    """Decode a canonical encoding; None when non-canonical or not on the curve."""
    n = int.from_bytes(b, "little")
    y = n & ((1 << 255) - 1)
    sign = n >> 255
    if y >= p:
        return None
    x = recover_x(y, sign)
    if x is None:
        return None
    return (x, y)


def decompress_loose(b):
    """Decode reducing y mod p (what a non-canonical-accepting decoder does); x=0 with sign set is refused."""
    n = int.from_bytes(b, "little")
    y = (n & ((1 << 255) - 1)) % p
    sign = n >> 255
    x = recover_x(y, sign)
    if x is None:
        return None
    return (x, y)


def sha512(*parts):
    h = hashlib.sha512()
    for x in parts:
        h.update(x)
    return h.digest()


def clamp(h32):
    a = bytearray(h32)
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(a, "little")


def keypair(seed):
    h = sha512(seed)
    a = clamp(h[:32])
    return a, h[32:], mul(a, B)


def sign_with(a, prefix, A_bytes, msg, r=None, R_override=None):
    """Honest signing but with the A bytes (any encoding) put into the hash. Returns (sig, R_point, k)."""
    if r is None:
        r = int.from_bytes(sha512(prefix, msg), "little") % L
    R = mul(r, B) if R_override is None else R_override
    Rb = compress(R)
    k = int.from_bytes(sha512(Rb, A_bytes, msg), "little") % L
    S = (r + k * a) % L
    return Rb + S.to_bytes(32, "little"), R, k


def order_of(P):
    Q = P
    n = 1
    while Q != ID:
        Q = add(Q, P)
        n += 1
        if n > 16:
            return None
    return n


# the torsion subgroup: decode the seven y values of libsodium's blocklist with both signs
SMALL_Y_HEX = [
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05",
    "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a",
    "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
    "eeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
]


def torsion_points():
    pts = {}
    for h in SMALL_Y_HEX:
        yb = bytes.fromhex(h)
        for sign in (0, 1):
            b = bytearray(yb)
            b[31] |= sign << 7
            P = decompress_loose(bytes(b))
            if P is not None:
                pts[compress(P)] = P
    return pts  # canonical encodings -> point


if __name__ == "__main__":
    # self checks
    assert mul(L, B) == ID
    tp = torsion_points()
    print(len(tp), "torsion points", sorted(order_of(P) for P in tp.values()))
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    a, prefix, A = keypair(seed)
    assert compress(A).hex() == "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", compress(A).hex()
    sig, R, k = sign_with(a, prefix, compress(A), b"")
    assert sig.hex() == "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b", sig.hex()
    print("RFC 8032 test 1 reproduced")
