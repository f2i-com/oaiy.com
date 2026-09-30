#!/usr/bin/env python3
"""Conformance test for the OAIY Relay Protocol v1 (`oaiy-relay/1`).

Same three questions as the bridge protocol's `conformance.py`, in the order they rot:

1. **Every schema is a valid JSON Schema 2020-12 and every `$ref` resolves** (the
   fragment too, not only the file). A typo in a `$ref` makes a subschema silently
   unenforced, which looks exactly like a passing test.

2. **Valid documents are accepted.** Built from the known-answer vectors wherever a
   vector exists, so the schemas and the cryptography describe the same bytes.

3. **Invalid documents are REJECTED, and for the reason the label names.** Each
   negative below starts from a valid document, changes exactly one thing, and states a
   `hint` that must appear in the validator's error trail, so a document that fails for
   an accidental reason (a typo in the test) does not count as a passing negative.
   Rules that JSON Schema cannot count (bytes, ordering, arithmetic between two
   members, small-order keys, canonical spellings) are checked by the reference rule
   functions in the "Reference rules" section, which are written to be reused by the
   black-box suite (RL-04) and by other implementations' tests.

It also recomputes EVERY vector in `relay/v1/vectors.json` from its recorded inputs with
Python's `cryptography` package, without importing `generate_vectors.py`; pins the
values the design printed (`DESIGN_PINS`) so a regenerated file cannot drift from them;
runs the independent Node re-computation `verify_vectors.mjs`; and checks the README
names every schema, error code, lane and route.

Run:  python protocol/tests/relay_conformance.py
Exit: 0 only when every case behaves as declared.

Environment: RELAY_V1_DIR points the run at another copy of relay/v1 (used to
mutation-check this suite against deliberately weakened schemas).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

try:
    from jsonschema import Draft202012Validator
    from jsonschema.validators import validator_for
    from referencing import Registry, Resource
except ImportError:
    sys.stderr.write(
        "This test needs `jsonschema` (>=4.18, which bundles `referencing`) and `cryptography`:\n"
        "    pip install jsonschema cryptography\n"
    )
    raise SystemExit(2)

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.exceptions import InvalidSignature
except ImportError:
    sys.stderr.write("This test needs `cryptography`:  pip install cryptography\n")
    raise SystemExit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent / "relay"
V1 = pathlib.Path(os.environ.get("RELAY_V1_DIR") or (ROOT / "v1"))
BASE = "https://oaiy.com/schemas/relay/v1/"

# Windows consoles still default to cp1252, which cannot encode the tick/cross.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

passed = 0
failures: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    global passed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failures.append(name)
        print(f"  ✗ {name}" + (f"  -> {detail}" if detail else ""))


def section(name: str) -> None:
    print(f"\n-- {name} --")


# ---------------------------------------------------------------------------
# Schemas and a registry so cross-file $refs resolve offline.
schemas: dict[str, dict] = {}
for path in sorted(V1.glob("*.schema.json")):
    schemas[path.name] = json.loads(path.read_text(encoding="utf-8"))

registry = Registry().with_resources([(s["$id"], Resource.from_contents(s)) for s in schemas.values()])


def validator(name: str) -> Draft202012Validator:
    """`name` is a schema file stem, or `common#def` for a definition inside common.schema.json."""
    if "#" in name:
        stem, frag = name.split("#", 1)
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$ref": f"{BASE}{stem}.schema.json#/$defs/{frag}"}
    else:
        schema = schemas[name + ".schema.json"]
    return validator_for(schema)(schema, registry=registry)


def error_trail(errors) -> str:
    """Every path, message and validator keyword in the error tree (contexts included)."""
    out: list[str] = []

    def walk(e) -> None:
        out.append("/".join(str(p) for p in e.absolute_path))
        out.append(e.message)
        out.append(str(e.validator))
        for c in e.context or []:
            walk(c)

    for e in errors:
        walk(e)
    return "\n".join(out)


def problems(name: str, doc) -> list:
    return sorted(validator(name).iter_errors(doc), key=lambda e: list(e.absolute_path))


def validate(name: str, doc) -> list[str]:
    """Reusable by a black-box suite: the messages that make `doc` invalid for schema `name`."""
    return [e.message for e in problems(name, doc)]


# ---------------------------------------------------------------------------
section("schemas are well-formed")
ok("relay/v1/ contains schemas", len(schemas) > 0, f"found {len(schemas)}")
for name, schema in schemas.items():
    try:
        validator_for(schema).check_schema(schema)
        ok(f"{name} is valid JSON Schema 2020-12", True)
    except Exception as e:  # noqa: BLE001 - report, don't crash the suite
        ok(f"{name} is valid JSON Schema 2020-12", False, str(e)[:160])
    ok(f"{name} declares the expected $id", schema.get("$id") == BASE + name)
    ok(f"{name} declares a description and a title", isinstance(schema.get("description"), str) and isinstance(schema.get("title"), str))

section("every $ref resolves (file and fragment)")


def refs_of(node) -> list[str]:
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "$ref" and isinstance(v, str):
                found.append(v)
            else:
                found.extend(refs_of(v))
    elif isinstance(node, list):
        for item in node:
            found.extend(refs_of(item))
    return found


known_ids = {s["$id"]: s for s in schemas.values()}


def pointer(doc, frag: str):
    node = doc
    for part in [p for p in frag.split("/") if p]:
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


nrefs = 0
for name, schema in schemas.items():
    for ref in refs_of(schema):
        nrefs += 1
        target, _, frag = ref.partition("#")
        resolved = True
        try:
            if target:
                resolved = target in known_ids
                doc = known_ids.get(target)
            else:
                doc = schema
            if resolved and frag:
                pointer(doc, frag)
        except (KeyError, TypeError):
            resolved = False
        ok(f"{name} -> {ref.rsplit('/', 1)[-1]}", resolved, f"unresolved: {ref}")
ok("the schemas use $ref at all", nrefs > 50, f"only {nrefs}")


# ---------------------------------------------------------------------------
# Reference rules: what JSON Schema cannot say. Reusable, dependency-light.
class Refused(ValueError):
    pass


P25519 = 2 ** 255 - 19
ORDER8 = [int.from_bytes(bytes.fromhex(h), "little") & ((1 << 255) - 1) for h in (
    "e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800",
    "5f9c95bca3508c24b1d0b1559c83ef5b04445cc4581c8e86d8224eddd09f1157")]
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u_strict(s: str) -> bytes:
    """Base64url without padding, refusing any alphabet slip and any non-canonical last character."""
    if not re.fullmatch(r"[A-Za-z0-9_-]*", s):
        raise Refused("alphabet")
    out = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
    if b64u(out) != s:
        raise Refused("not canonical")
    return out


def parse_token(t: str) -> tuple[bytes, bytes]:
    parts = t.split(".")
    if len(parts) != 3 or parts[0] != "oaiyrt1" or len(parts[1]) != 11 or len(parts[2]) != 43:
        raise Refused("shape")
    return unb64u_strict(parts[1]), unb64u_strict(parts[2])


def canonical_json(text: str) -> str:
    """Canonical form of the pairing family, from JSON TEXT so a spelling like 1.0 is seen."""
    def no_float(s: str):
        raise Refused("floating point number " + s)

    def no_const(s: str):
        raise Refused("constant " + s)

    def int_of(s: str) -> int:
        if s == "-0":
            raise Refused("negative zero: not a canonical spelling of 0")
        return int(s)

    def w(v) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            if not (-(2 ** 63) <= v <= 2 ** 64 - 1):
                raise Refused("integer out of 64-bit range")
            return str(v)
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return "[" + ",".join(w(x) for x in v) + "]"
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + w(v[k]) for k in sorted(v, key=lambda k: k.encode())) + "}"

    return w(json.loads(text, parse_float=no_float, parse_constant=no_const, parse_int=int_of))


def is_small_order_x25519(enc: bytes) -> bool:
    if len(enc) != 32:
        raise Refused("length")
    v = (int.from_bytes(enc, "little") & ((1 << 255) - 1)) % P25519
    return v in (0, 1, P25519 - 1) or v in ORDER8


def strictly_ascending(items: list[str]) -> bool:
    raw = [i.encode() for i in items]
    return all(a < b for a, b in zip(raw, raw[1:]))


def hdr_bytes(hdr: dict) -> int:
    return len(json.dumps(hdr, separators=(",", ":"), ensure_ascii=False).encode())


def crock(value: int, nbits: int) -> str:
    pad = (-nbits) % 5
    value <<= pad
    out = ""
    for _ in range((nbits + pad) // 5):
        out = CROCKFORD[value & 31] + out
        value >>= 5
    return out


def normalise_typed(text: str) -> str | None:
    t = re.sub(r"[-\s]", "", text.upper()).replace("I", "1").replace("L", "1").replace("O", "0")
    return t if re.fullmatch(r"[0-9A-HJKMNP-TV-Z]*", t) else None


def typed_code_ok(text: str) -> bool:
    """The local typo check of the 28 character typed code: 26 characters of secret, then 2 of check."""
    t = normalise_typed(text)
    if t is None or len(t) != 28:
        return False
    body = 0
    for ch in t[:26]:
        body = body * 32 + CROCKFORD.index(ch)
    if body & 3:                         # the last two bits of the 130 are zero padding
        return False
    s = (body >> 2).to_bytes(16, "big")
    want = crock(int.from_bytes(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + s).digest()[:2], "big") >> 6, 10)
    return t[26:] == want


def sas_check_char(sas12: str) -> str:
    return CROCKFORD[hashlib.sha256(b"oaiy/pairing/3/sas-check\x00" + sas12.encode()).digest()[0] >> 3]


def sas_ok(text: str) -> bool:
    t = normalise_typed(text)
    return t is not None and len(t) == 13 and sas_check_char(t[:12]) == t[12]


def window_ok(doc: dict, iat: str, exp: str, longest: int, exact: bool = False) -> bool:
    """exp - iat must be positive and at most `longest` seconds (or exactly that long)."""
    span = doc[exp] - doc[iat]
    return span == longest if exact else 0 < span <= longest


def pairing_response_window_ok(claims: dict) -> bool:      # RESPONSE_TTL_SECONDS: issuedAt + 120
    return window_ok(claims, "issuedAt", "expiresAt", 120)


def pairing_offer_window_ok(offer: dict) -> bool:          # the offer lives exactly 600 seconds
    return window_ok(offer, "issuedAt", "expiresAt", 600, exact=True)


def signed_window_ok(doc: dict) -> bool:                   # command envelopes and tickets: at most 300 seconds
    return window_ok(doc, "iat", "exp", 300)


def rotation_window_ok(st: dict) -> bool:                  # a rotation statement: at most 24 hours
    return window_ok(st, "iat", "exp", 86400)


def ring_window_ok(ring: dict, now: int) -> bool:
    """A ring's expiresAt is later than now and at most now + 300 (86,400 for informational)."""
    exp = int(ring["expiresAt"])
    return now < exp <= now + (86400 if ring["aokieClass"] == "informational" else 300)


# ---------------------------------------------------------------------------
# Vectors
vec = json.loads((V1 / "vectors.json").read_text(encoding="utf-8"))


def _hkdf(ikm: bytes, salt: bytes, info: bytes, n: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=n, salt=salt, info=info).derive(ikm)


RAW, RAWF = serialization.Encoding.Raw, serialization.PublicFormat.Raw


def ed_pub(seed: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(RAW, RAWF)


def ed_sign(seed: bytes, msg: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(seed).sign(msg)


def ed_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(pub).verify(sig, msg)
        return True
    except InvalidSignature:
        return False


def x_pub(sec: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(sec).public_key().public_bytes(RAW, RAWF)


def thumb(pub: bytes) -> str:
    return b64u(hashlib.sha256(('{"crv":"Ed25519","kty":"OKP","x":"' + b64u(pub) + '"}').encode()).digest())


def hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def roster_hash(rev: int, ths: list[str]) -> str:
    text = canonical_json(json.dumps({"approvedPeerKeyThumbprints": sorted(ths), "peerRosterRevision": rev}))
    return b64u(hashlib.sha256(b"aokie/v2/peer-roster\x00" + text.encode()).digest())


def ladder(k_bytes: bytes, u_bytes: bytes) -> bytes:
    """RFC 7748 X25519 Montgomery ladder (bit 255 of u ignored)."""
    kb = bytearray(k_bytes)
    kb[0] &= 248
    kb[31] &= 127
    kb[31] |= 64
    k = int.from_bytes(kb, "little")
    u = int.from_bytes(u_bytes, "little") & ((1 << 255) - 1)
    x1, x2, z2, x3, z3, swap = u, 1, 0, u, 1, 0
    for t in range(254, -1, -1):
        kt = (k >> t) & 1
        swap ^= kt
        if swap:
            x2, x3, z2, z3 = x3, x2, z3, z2
        swap = kt
        a, b = (x2 + z2) % P25519, (x2 - z2) % P25519
        aa, bb = a * a % P25519, b * b % P25519
        e = (aa - bb) % P25519
        c, d = (x3 + z3) % P25519, (x3 - z3) % P25519
        da, cb = d * a % P25519, c * b % P25519
        x3 = (da + cb) ** 2 % P25519
        z3 = x1 * (da - cb) ** 2 % P25519
        x2 = aa * bb % P25519
        z2 = e * (aa + 121665 * e) % P25519
    if swap:
        x2, x3, z2, z3 = x3, x2, z3, z2
    return (x2 * pow(z2, P25519 - 2, P25519) % P25519).to_bytes(32, "little")


# What the design printed (Appendix A), pinned so a regenerated vectors.json cannot drift from it.
DESIGN_PINS = {
    "A0 roster hash": ("A0", "expected.peerRosterHash", "tsKgPP1ruPCU23HfLYaUChe9jHYHCtubve77gnlfyDw"),
    "A1 RFC 8037 signature": ("A1", "expected.signature", "hgyY0il_MGCjP0JzlnLWG1PPOt7-09PGcvMg3AIbQR6dWbhijcNR4ki4iylGjg5BhVsPt9g7sVvpAr_MuM0KAg"),
    "A2 token": ("A2", "expected.token", "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"),
    "A2 stored hash": ("A2", "expected.secretSha256", "72dbb7336c76780023f83da4c355f2eeea85733b13d3477697917790c1229084"),
    "A3 secret b64u": ("A3", "expected.secretB64u", "obLD1OX2BxgpOktcbX6PkA"),
    "A3 typed code": ("A3", "expected.typedCode", "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RW"),
    "A3 pid": ("A3", "expected.pid", "b5YkfMcTvJb0g1GTv3kNNQ"),
    "A3 mac key": ("A3", "expected.macKeyHex", "0a32adef7f346a81bee7534af43c154ea1f7418433de1da4b18182ab4244880d"),
    "A3 offer MAC": ("A3", "expected.offerMac", "NVXJRqSryFOpH6fU5kmvLpQzv0-rM1wzDXd8BiMoDWk"),
    "A3 response signature": ("A3", "expected.responseSignature", "uoCR5b9NLDUwK2QBH-pXtm-6I5Kj4FqM4F_BhuGDxr_qKomo7Kq1O1H14s8YyPuie14oCoaoTzHvX0KoT5UwAQ"),
    "A3 response MAC": ("A3", "expected.responseMac", "0Nw8sxSLeI6x5islDA7m0Gi_s_qP4LrHWrbPIMwxdoQ"),
    "A3 SAS raw": ("A3", "expected.sasRawHex", "3563599914bdf7f9"),
    "A3 SAS display": ("A3", "expected.sasDisplay", "6NHN-K68M-QQVZ-5"),
    "A3 receipt signature": ("A3", "expected.receiptSignature", "FU2PKdtj23VkPpxZfw-DcqwSBNWlL8OvFfPFiJ-kSufrDGbpBPKfKy0c9MfhD3IZyM_5pVgAmeiMGNep93mtDg"),
    "A3 offer length": ("A3", "expected.offerTextBytes", 778),
    "A4 token length": ("A4", "expected.length", 888),
    "A4 token tail (HMAC)": ("A4", "expected.token", "fa070168b95bc239a6ce8e23075e0dec97919e3739e9259feed7c50b640eca96"),
    "A5 credential": ("A5", "expected.credential", "T2oDhW95dkDEbEXjdHRbKL5eh7g="),
    "A6 body hash": ("A6", "expected.bodySha256", "50c0fd16de03abb48ae864b6379836b71f10025f91d0976b2198885617a09883"),
    "A6 proof": ("A6", "expected.proof", "4o_XJ-1EjzwrHgisbfhHOeJ7WymUeJWqN8WmjLf0lllRTAlNTNSmFygY9Al5HncmJRTVDv82GcI4eur-D_f_Bg"),
    "A6 static signature": ("A6", "expected.staticSignature", "Ev4PROe-bST4Z_BQLcuEVkBJQwj82o6zdeaFMOTUxiyjcUBRu6dTBCXnfUzsxNTcTdoNiUSfxE-4wCBpYMNlDA"),
    "A7 kid": ("A7", "expected.kid", "OJttnmp91Xo"),
    "A7 derived public": ("A7", "expected.derivedPublic", "UbdzJGbsg68-cAFQVB7FTivyuuaHUyJg7M_67QU97AA"),
    "A7 proof": ("A7", "expected.proof", "uUkVKr8zengn6paFCcBgs_uBFU8S7rUEDg2w3t_5279ALMEeR_WqRWBZ0lxnyiwWffD0L-qbR6k3bfbVlyK0Cg"),
    "A8 signature": ("A8", "expected.signature", "rDvULKFLe_i0a7wyxkd7teh1B4bg2fZtBbBXMempJmp26BnyUg-RhOXP0U_AeV_DHt_rICqrwO_2803WnS9dDw"),
    "A8 container length": ("A8", "expected.containerBytes", 493),
    "A9 ephemeral public": ("A9", "expected.ephemeralPublic", "IZ5NgA2paNKl_LAJx4T0dGxxOO257khEtznoMLBc9CQ"),
    "A9 signature": ("A9", "expected.signature", "8uLJTXCapVZ23JG5kVDni_qaS71qsZ1T0aomu7RrPiU0rUwrYhytLu-NFDDhJvssKzfTP5XwcuGLRr45_5TAAQ"),
    "A10 signature": ("A10", "expected.signature", "uvf5l5xuyHGkuNhz4ycg07tAbHxqDETo-4rY7ruIVNQVSUjHgpvwXS_I1006bpwM4qOOSYePS165g5zNLQRRDg"),
    "A11 hdr.sig": ("A11", "expected.hdrSig", "0lTc0PX9bX8oBmi8-jriiGEPBTD4MNP35n1GEPNNj0N3-mDwWkv0hJjOaXgiC8s_5fIz91jrtL15Eejz5H6tAQ"),
    "keys desktop thumbprint": ("keys", "ed25519Public.desktopEndpoint.thumbprint", "atZR63tNG3W2GhPpVleKnfrw1N7N6LAqOe5grE2SBR4"),
    "keys phone thumbprint": ("keys", "ed25519Public.phone.thumbprint", "--6IM5l0OosLj9yWskISYhUA3n_3CURQkmrYMSha_ck"),
    "keys relay thumbprint": ("keys", "ed25519Public.relay.thumbprint", "b7dKD2-DlMApkGljz-RJwDdNcyxwyFBpSmhz2cRBnaw"),
    "keys host thumbprint": ("keys", "ed25519Public.host.thumbprint", "8TlG3IhiRH5sWhsIGSSRnAqf9zyhEl1ir4t4cbTRCk0"),
    "keys provider thumbprint": ("keys", "ed25519Public.provider.thumbprint", "hlqNnvnuw_g7uAgV8Sy2kR0hH3orMop9alO-BBtaWo4"),
    "keys ids": ("keys", "ids.desktopDevice", "dev-oKGio6SlpqeoqaqrrK2urw"),
    "keys relay id": ("keys", "ids.relay", "rly-0NHS09TV1tfY2drb3N3e3w"),
}


def dig(obj, dotted: str):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


section("vectors: the values the design printed")
for label, (group, path, want) in DESIGN_PINS.items():
    got = dig(vec[group], path)
    if isinstance(want, str) and label == "A4 token tail (HMAC)":
        ok(f"pinned: {label}", got.endswith(want))
    else:
        ok(f"pinned: {label}", got == want, f"{got!r} != {want!r}")
for n, want in {"1": 964, "3": 1148, "8": 1608, "16": 2344, "32": 3816, "64": 6760}.items():
    ok(f"pinned: plugin bearer with {n} phone(s) is {want} characters", vec["A4b"]["expected"]["lengthByPhones"][n] == want)

section("vectors: every value recomputed in Python from its recorded inputs")
recomputed = 0


def vok(name: str, got, want) -> None:
    global recomputed
    recomputed += 1
    ok(f"recomputed: {name}", got == want, f"{got!r} != {want!r}")


K = vec["keys"]
seeds = {k: bytes.fromhex(v) for k, v in K["ed25519Seeds"].items()}
xsec = {k: bytes.fromhex(v) for k, v in K["x25519Secrets"].items()}
pub = {k: ed_pub(s) for k, s in seeds.items()}
xpub = {k: x_pub(s) for k, s in xsec.items()}
th = {k: thumb(p) for k, p in pub.items()}
for k in seeds:
    vok(f"key {k} public", b64u(pub[k]), K["ed25519Public"][k]["publicKey"])
    vok(f"key {k} thumbprint", th[k], K["ed25519Public"][k]["thumbprint"])
for k in xsec:
    vok(f"x25519 {k} public", b64u(xpub[k]), K["x25519Public"][k])
DEV = "dev-" + b64u(bytes.fromhex(K["idBytes"]["desktopDevice"]))
PHONE = "dev-" + b64u(bytes.fromhex(K["idBytes"]["phoneDevice"]))
PROV = "prov-" + b64u(bytes.fromhex(K["idBytes"]["provider"]))
RELAY = "rly-" + b64u(bytes.fromhex(K["idBytes"]["relay"]))
vok("ids", {"desktopDevice": DEV, "phoneDevice": PHONE, "provider": PROV, "relay": RELAY}, K["ids"])

# anchors
A = vec["anchors"]
vok("RFC 5869 test case 1", _hkdf(bytes.fromhex(A["rfc5869_tc1"]["ikm"]), bytes.fromhex(A["rfc5869_tc1"]["salt"]), bytes.fromhex(A["rfc5869_tc1"]["info"]), 42).hex(), A["rfc5869_tc1"]["okm"])
vok("RFC 4231 test case 1", hmac.new(bytes.fromhex(A["rfc4231_tc1_hmac_sha256"]["key"]), b"Hi There", hashlib.sha256).hexdigest(), A["rfc4231_tc1_hmac_sha256"]["mac"])
vok("RFC 2202 test case 1", hmac.new(bytes.fromhex(A["rfc2202_tc1_hmac_sha1"]["key"]), b"Hi There", hashlib.sha1).hexdigest(), A["rfc2202_tc1_hmac_sha1"]["mac"])
r = A["rfc8032_test1"]
vok("RFC 8032 test 1 public", ed_pub(bytes.fromhex(r["seed"])).hex(), r["public"])
vok("RFC 8032 test 1 signature", ed_sign(bytes.fromhex(r["seed"]), b"").hex(), r["signature"])
r = A["rfc7748_6_1"]
vok("RFC 7748 alice public", x_pub(bytes.fromhex(r["alicePrivate"])).hex(), r["alicePublic"])
vok("RFC 7748 shared secret (pure ladder)", ladder(bytes.fromhex(r["alicePrivate"]), bytes.fromhex(r["bobPublic"])).hex(), r["shared"])

# A0
a0 = vec["A0"]
vok("A0 hashed text", canonical_json(json.dumps({"approvedPeerKeyThumbprints": a0["inputs"]["approvedPeerKeyThumbprints"], "peerRosterRevision": a0["inputs"]["peerRosterRevision"]})), a0["inputs"]["hashedText"])
vok("A0 roster hash", roster_hash(a0["inputs"]["peerRosterRevision"], a0["inputs"]["approvedPeerKeyThumbprints"]), a0["expected"]["peerRosterHash"])
vok("A0 equals the Aokie README value", a0["expected"]["peerRosterHash"], a0["expected"]["aokieReadmeValue"])

# A1
a1 = vec["A1"]
priv1 = unb64u_strict(a1["inputs"]["privateKey"])
vok("A1 RFC 8037 public key", b64u(ed_pub(priv1)), a1["inputs"]["publicKey"])
vok("A1 RFC 8037 signature", b64u(ed_sign(priv1, a1["inputs"]["signingInput"].encode())), a1["expected"]["signature"])

# A2
a2 = vec["A2"]
tok2 = a2["inputs"]["prefix"] + b64u(bytes.fromhex(a2["inputs"]["idHex"])) + "." + b64u(bytes.fromhex(a2["inputs"]["secretHex"]))
vok("A2 token", tok2, a2["expected"]["token"])
vok("A2 length", len(tok2), 63)
vok("A2 stored hash", hashlib.sha256(bytes.fromhex(a2["inputs"]["secretHex"])).hexdigest(), a2["expected"]["secretSha256"])
vok("A2 strict parse", (parse_token(tok2)[0].hex(), parse_token(tok2)[1].hex()), (a2["inputs"]["idHex"], a2["inputs"]["secretHex"]))

# A3
a3 = vec["A3"]
i3, x3 = a3["inputs"], a3["expected"]
s3 = bytes.fromhex(i3["secretHex"])
nonce3 = bytes.fromhex(i3["nonceHex"])
pid3 = _hkdf(s3, i3["hkdfSalt"].encode(), b"rendezvous", 16)
mac3 = _hkdf(s3, i3["hkdfSalt"].encode(), b"mac", 32)
vok("A3 pid", b64u(pid3), x3["pid"])
vok("A3 mac key", mac3.hex(), x3["macKeyHex"])
typed3 = crock(int.from_bytes(s3, "big"), 128) + crock(int.from_bytes(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + s3).digest()[:2], "big") >> 6, 10)
vok("A3 typed code", "-".join(typed3[i:i + 4] for i in range(0, 28, 4)), x3["typedCode"])
vok("A3 typed code passes the local check", typed_code_ok(x3["typedCode"]), True)
vok("A3 pairing key", "oaiy://pair?v=3&u=" + i3["relayUrl"].replace(":", "%3A").replace("/", "%2F") + "&f=" + th["relay"] + "&s=" + b64u(s3) + "&x=" + str(i3["expiresAtParam"]), x3["pairingUri"])
offer_text = canonical_json(json.dumps(i3["offer"]))
vok("A3 offer canonical text", offer_text, x3["offerText"])
vok("A3 offer length", len(offer_text.encode()), 778)
vok("A3 offer MAC", b64u(hmac_sha256(mac3, b"oaiy/pairing/3/offer-mac\x00" + offer_text.encode())), x3["offerMac"])
vok("A3 offer key material derives from the seeds", (i3["offer"]["desktopEndpointKey"]["publicKey"], i3["offer"]["hostIdentity"]["x25519"], i3["offer"]["relay"]["fingerprint"]), (b64u(pub["desktopEndpoint"]), b64u(xpub["host"]), th["relay"]))
claims_text = canonical_json(json.dumps(i3["claims"]))
vok("A3 claims canonical text", claims_text, x3["claimsCanonical"])
sig3 = ed_sign(seeds["phone"], b"oaiy/pairing/3/response\x00" + claims_text.encode())
vok("A3 response signature", b64u(sig3), x3["responseSignature"])
vok("A3 response signature verifies", ed_verify(pub["phone"], b"oaiy/pairing/3/response\x00" + claims_text.encode(), sig3), True)
vok("A3 response MAC", b64u(hmac_sha256(mac3, b"oaiy/pairing/3/response-mac\x00" + claims_text.encode())), x3["responseMac"])
sas_raw = _hkdf(pub["desktopEndpoint"] + pub["phone"], nonce3, b"oaiy/pairing/3/sas\x00" + pid3, 8)
vok("A3 SAS raw", sas_raw.hex(), x3["sasRawHex"])
sas12 = crock(int.from_bytes(sas_raw, "big") >> 4, 60)
vok("A3 SAS 12 characters", sas12, x3["sas12"])
vok("A3 SAS check character", sas_check_char(sas12), x3["sasCheckChar"])
vok("A3 SAS display and local check", (sas12[:4] + "-" + sas12[4:8] + "-" + sas12[8:] + "-" + sas_check_char(sas12), sas_ok(x3["sasDisplay"])), (x3["sasDisplay"], True))
receipt_text = canonical_json(json.dumps(i3["receiptDocument"]))
vok("A3 receipt text", receipt_text, x3["receiptText"])
vok("A3 receipt grants are sorted", i3["receiptDocument"]["grants"], sorted(i3["receiptDocument"]["grants"]))
rsig = ed_sign(seeds["desktopEndpoint"], b"oaiy/pairing/3/approval\x00" + receipt_text.encode())
vok("A3 receipt signature", b64u(rsig), x3["receiptSignature"])
vok("A3 receipt verifies", ed_verify(pub["desktopEndpoint"], b"oaiy/pairing/3/approval\x00" + receipt_text.encode(), rsig), True)


def admission_token(secret: bytes, claims: dict) -> str:
    p = json.dumps(claims, separators=(",", ":"), ensure_ascii=False).encode()
    return "aokie-adm-v2." + p.hex() + "." + hmac_sha256(secret, p).hex()


# A4, A4b
a4 = vec["A4"]
t4 = admission_token(bytes.fromhex(a4["inputs"]["secretHex"]), a4["inputs"]["claims"])
vok("A4 admission token", t4, a4["expected"]["token"])
vok("A4 length", len(t4), 888)
a4b = vec["A4b"]
sizes = {}
for n in (1, 3, 8, 16, 32, 64):
    ths = sorted(b64u(hashlib.sha256(bytes([k])).digest()) for k in range(n))
    claims = {"aud": "aokie-v2-gateway", "appId": "aokie", "subjectId": a4b["inputs"]["pluginId"], "role": "plugin",
              "holderKeyThumbprint": a4b["inputs"]["holderThumbprint"], "approvedPeerKeyThumbprints": ths,
              "peerRosterRevision": a4b["inputs"]["peerRosterRevision"], "peerRosterHash": roster_hash(a4b["inputs"]["peerRosterRevision"], ths),
              "scopes": a4b["inputs"]["scopes"], "dsk": a4b["inputs"]["dsk"], "exp": a4b["inputs"]["exp"], "jti": a4b["inputs"]["jti"]}
    sizes[str(n)] = len(admission_token(bytes.fromhex(a4b["inputs"]["secretHex"]), claims))
    if n == 1:
        vok("A4b plugin bearer for one phone", admission_token(bytes.fromhex(a4b["inputs"]["secretHex"]), claims), a4b["expected"]["tokenForOnePhone"])
vok("A4b bearer sizes", sizes, a4b["expected"]["lengthByPhones"])

# A5
a5 = vec["A5"]
user5 = f"{a5['inputs']['expiry']}:{PHONE}"
vok("A5 username", user5, a5["expected"]["username"])
vok("A5 TURN credential", base64.b64encode(hmac.new(a5["inputs"]["secret"].encode(), user5.encode(), hashlib.sha1).digest()).decode(), a5["expected"]["credential"])


# A6, A6b
def info_proof(body: bytes, nonce_b64u: str, t: int) -> dict:
    n = unb64u_strict(nonce_b64u)
    return {"sha": hashlib.sha256(body).hexdigest(),
            "proof": b64u(ed_sign(seeds["relay"], b"oaiy/relay/1/info-proof\x00" + n + hashlib.sha256(body).digest() + str(t).encode())),
            "static": b64u(ed_sign(seeds["relay"], b"oaiy/relay/1/info\x00" + body))}


a6 = vec["A6"]
p6 = info_proof(a6["inputs"]["body"].encode(), a6["inputs"]["nonce"], a6["inputs"]["time"])
vok("A6 body hash", p6["sha"], a6["expected"]["bodySha256"])
vok("A6 proof", p6["proof"], a6["expected"]["proof"])
vok("A6 static signature", p6["static"], a6["expected"]["staticSignature"])
vok("A6 a replayed body with another nonce gives another proof", info_proof(a6["inputs"]["body"].encode(), b64u(bytes(16)), a6["inputs"]["time"])["proof"] != p6["proof"], True)
a6b = vec["A6b"]
body6b = json.dumps(a6b["inputs"]["info"], separators=(",", ":"), ensure_ascii=False)
vok("A6b info document text", body6b, a6b["expected"]["bodyText"])
p6b = info_proof(body6b.encode(), a6b["inputs"]["nonce"], a6b["inputs"]["time"])
vok("A6b proof", p6b["proof"], a6b["expected"]["proof"])
vok("A6b static signature", p6b["static"], a6b["expected"]["staticSignature"])
vok("A6b relayKey matches the relay seed", a6b["inputs"]["info"]["relayKey"], {"algorithm": "ed25519", "publicKey": b64u(pub["relay"]), "thumbprint": th["relay"]})

# A7
a7 = vec["A7"]
es = bytes.fromhex(a7["inputs"]["secretHex"])
kid7 = b64u(_hkdf(es, a7["inputs"]["hkdfSalt"].encode(), b"id", 8))
seed7 = _hkdf(es, a7["inputs"]["hkdfSalt"].encode(), b"sig", 32)
vok("A7 kid", kid7, a7["expected"]["kid"])
vok("A7 derived public", b64u(ed_pub(seed7)), a7["expected"]["derivedPublic"])
body7 = json.dumps(a7["inputs"]["request"], separators=(",", ":"), ensure_ascii=False)
vok("A7 request body", body7, a7["expected"]["requestBody"])
proof7 = ed_sign(seed7, b"oaiy/relay/1/enroll\x00" + body7.encode())
vok("A7 proof", b64u(proof7), a7["expected"]["proof"])
vok("A7 proof over one extra space fails", ed_verify(ed_pub(seed7), b"oaiy/relay/1/enroll\x00" + body7.replace('"kid":', '"kid": ').encode(), proof7), False)
vok("A7 enrolment key", "oaiy://enroll?v=1&u=" + a7["inputs"]["relayUrl"].replace(":", "%3A").replace("/", "%2F") + "&f=" + th["relay"] + "&k=" + kid7 + "&s=" + b64u(es) + "&r=desktop&x=" + str(a7["inputs"]["expiry"]), a7["expected"]["uri"])

# A8
a8 = vec["A8"]
bytes8 = json.dumps(a8["inputs"]["command"], separators=(",", ":"), ensure_ascii=False).encode()
sig8 = ed_sign(seeds["provider"], b"oaiy/relay/1/cmd\x00" + bytes8)
vok("A8 signed bytes", bytes8.decode(), a8["expected"]["signedBytes"])
vok("A8 signature", b64u(sig8), a8["expected"]["signature"])
cont8 = json.dumps({"k": th["provider"], "b": b64u(bytes8), "s": b64u(sig8)}, separators=(",", ":"))
vok("A8 container", cont8, a8["expected"]["container"])
vok("A8 container length", len(cont8), 493)

# A9
a9 = vec["A9"]
sin9 = b64u(json.dumps(a9["inputs"]["header"], separators=(",", ":")).encode()) + "." + b64u(json.dumps(a9["inputs"]["claims"], separators=(",", ":")).encode())
sig9 = ed_sign(seeds["provider"], sin9.encode())
vok("A9 ephemeral public", b64u(xpub["browserEphemeral"]), a9["expected"]["ephemeralPublic"])
vok("A9 eph claim is b64u(SHA-256(public))", a9["inputs"]["claims"]["eph"], b64u(hashlib.sha256(xpub["browserEphemeral"]).digest()))
vok("A9 signing input", sin9, a9["expected"]["signingInput"])
vok("A9 signature", b64u(sig9), a9["expected"]["signature"])
vok("A9 ticket", sin9 + "." + b64u(sig9), a9["expected"]["ticket"])

# A10, A11
a10 = vec["A10"]
text10 = json.dumps(a10["inputs"]["statement"], separators=(",", ":"), ensure_ascii=False)
vok("A10 statement text", text10, a10["expected"]["statementText"])
vok("A10 signature", b64u(ed_sign(seeds["provider"], b"oaiy/relay/1/provider-rotate\x00" + text10.encode())), a10["expected"]["signature"])
vok("A10 new key derives from the second provider seed", a10["inputs"]["statement"]["new"]["ed25519"], b64u(pub["provider2"]))
a11 = vec["A11"]
text11 = json.dumps(a11["inputs"]["body"], separators=(",", ":"), ensure_ascii=False)
vok("A11 body text", text11, a11["expected"]["bodyText"])
vok("A11 hdr.sig", b64u(ed_sign(seeds["host"], b"oaiy/relay/1/ring\x00" + text11.encode())), a11["expected"]["hdrSig"])

# A12
a12 = vec["A12"]
key12 = bytes.fromhex(a12["inputs"]["staticKeyHex"])
for name, h in a12["inputs"]["encodings"].items():
    enc = bytes.fromhex(h)
    hi = bytearray(enc)
    hi[31] |= 0x80
    vok(f"A12 {name}: all-zero shared secret", ladder(key12, enc) == bytes(32), True)
    vok(f"A12 {name}: bit 255 set is the same point", (hi.hex(), ladder(key12, bytes(hi)) == bytes(32)), (a12["inputs"]["withBit255"][name], True))
    vok(f"A12 {name}: refused by the reference rule (both spellings)", (is_small_order_x25519(enc), is_small_order_x25519(bytes(hi))), (True, True))
vok("A12 control: the X25519 base point is not refused", (is_small_order_x25519((9).to_bytes(32, "little")), ladder(key12, (9).to_bytes(32, "little")) != bytes(32)), (False, True))
vok("A12 control: a real public key is not refused", is_small_order_x25519(xpub["phone"]), False)

# extras
ex = vec["extras"]
for c in ex["canonical"]["cases"]:
    vok(f"canonical: {c['label']}", canonical_json(c["input"]), c["output"])
for c in ex["canonical"]["refused"]:
    try:
        canonical_json(c["input"])
        accepted = True
    except Refused:
        accepted = False
    vok(f"canonicaliser refuses: {c['label']}", accepted, False)
for t in ex["tokens"]["valid"]:
    try:
        parse_token(t)
        good = True
    except Refused:
        good = False
    vok(f"token accepted: {t[:24]}...", good, True)
for t in ex["tokens"]["invalid"]:
    try:
        parse_token(t["token"])
        good = True
    except Refused:
        good = False
    vok(f"token refused: {t['reason']}", good, False)
for t in ex["thumbprints"]:
    vok(f"thumbprint of seed {t['seed'][:2]}...", thumb(ed_pub(bytes.fromhex(t["seed"]))), t["thumbprint"])
for t in ex["typedCode"]["samples"]:
    s = bytes.fromhex(t["secretHex"])
    code = crock(int.from_bytes(s, "big"), 128) + crock(int.from_bytes(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + s).digest()[:2], "big") >> 6, 10)
    vok(f"typed code {t['typed']}", ("-".join(code[i:i + 4] for i in range(0, 28, 4)), typed_code_ok(t["typed"])), (t["typed"], True))
for n in ex["typedCode"]["normalise"]:
    vok(f"typed code normalisation of {n['input']!r}", normalise_typed(n["input"]), n["output"])
for c in ex["sasCheck"]["samples"]:
    vok(f"SAS check character of {c['sas12']}", (sas_check_char(c["sas12"]), sas_ok(c["sas12"] + c["check"])), (c["check"], True))

# The SAS input carries the RAW 16 bytes of pid. Reading pid as its text (b64u or hex) gives other values; those are recorded as
# negative vectors so that an implementation can show it does not make that mistake.
sasneg = ex["sasNegative"]
sni = sasneg["inputs"]
sn_dpub, sn_ppub, sn_nonce, sn_pid = (bytes.fromhex(sni[k]) for k in ("desktopEndpointPublicHex", "phoneEndpointPublicHex", "nonceHex", "pidHex"))


def sas_of_pid_reading(pid_input: bytes) -> tuple:
    info = b"oaiy/pairing/3/sas\x00" + pid_input
    raw = _hkdf(sn_dpub + sn_ppub, sn_nonce, info, 8)
    s12 = crock(int.from_bytes(raw, "big") >> 4, 60)
    return info.hex(), len(info), raw.hex(), s12, s12[:4] + "-" + s12[4:8] + "-" + s12[8:] + "-" + sas_check_char(s12)


vok("SAS negative: the recorded inputs are those of A3", (sn_dpub, sn_ppub, sn_nonce, sn_pid, sni["pidB64u"]), (pub["desktopEndpoint"], pub["phone"], nonce3, pid3, x3["pid"]))
for entry, pid_input in ((sasneg["correct"], sn_pid), (sasneg["wrong"][0], sni["pidB64u"].encode()), (sasneg["wrong"][1], sn_pid.hex().encode())):
    vok(f"SAS negative: {entry['reading']}", sas_of_pid_reading(pid_input),
        (entry["infoHex"], entry["infoLength"], entry["sasRawHex"], entry["sas12"], entry["sasDisplay"]))
vok("SAS negative: the correct reading is the SAS of A3", sasneg["correct"]["sasDisplay"], x3["sasDisplay"])
vok("SAS negative: the wrong readings are flagged and give three different values",
    ([w["mustNotProduce"] for w in sasneg["wrong"]], len({sasneg["correct"]["sasRawHex"]} | {w["sasRawHex"] for w in sasneg["wrong"]})), ([True, True], 3))
print(f"\n  {recomputed} vector values recomputed in Python")

# ---------------------------------------------------------------------------
section("vectors: the checksums catch typos as the design claims")
alphabet_others = {c: [d for d in CROCKFORD if d != c] for c in CROCKFORD}
typed = ex["typedCode"]["samples"][0]["typed"].replace("-", "")
total = caught = 0
for pos in range(28):
    for alt in alphabet_others[typed[pos]]:
        total += 1
        caught += 0 if typed_code_ok(typed[:pos] + alt + typed[pos + 1:]) else 1
ok(f"typed code: {caught} of {total} single-character substitutions rejected locally", caught / total >= 0.99, f"{caught / total:.4f}")
sas = vec["A3"]["expected"]["sas12"]
total = caught = 0
for pos in range(12):
    for alt in alphabet_others[sas[pos]]:
        total += 1
        caught += 0 if sas_ok(sas[:pos] + alt + sas[pos + 1:] + sas_check_char(sas)) else 1
ok(f"SAS: {caught} of {total} single-character substitutions caught by the check character", 0.95 <= caught / total <= 1.0, f"{caught / total:.4f}")
sas13 = sas + sas_check_char(sas)
total = caught = 0
for pos in range(13):
    for alt in alphabet_others[sas13[pos]]:
        total += 1
        caught += 0 if sas_ok(sas13[:pos] + alt + sas13[pos + 1:]) else 1
ok(f"SAS: {caught} of {total} substitutions over all 13 positions caught", caught / total >= 0.95, f"{caught / total:.4f}")

# ---------------------------------------------------------------------------
# Documents
K = vec["keys"]
TH = {k: v["thumbprint"] for k, v in K["ed25519Public"].items()}
PUB = {k: v["publicKey"] for k, v in K["ed25519Public"].items()}
XP = K["x25519Public"]
DEVID, PHONEID, PROVID, RELAYID = K["ids"]["desktopDevice"], K["ids"]["phoneDevice"], K["ids"]["provider"], K["ids"]["relay"]
NOW = 1790000000
TOKEN = vec["A2"]["expected"]["token"]
A3 = vec["A3"]
OFFER = A3["inputs"]["offer"]
CLAIMS = A3["inputs"]["claims"]
RECEIPT = A3["inputs"]["receiptDocument"]
PAIR_RESPONSE = {"kind": "aokie_mobile_pairing_response", "schemaVersion": 3, "claims": CLAIMS,
                 "signature": A3["expected"]["responseSignature"], "mac": A3["expected"]["responseMac"]}
INFO = vec["A6b"]["inputs"]["info"]
ITEM = {"seq": 42, "id": "6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a", "lane": "cmd", "from": "prov-Q1w2E3r4T5y6U7i8O9p0aB",
        "at": NOW, "exp": NOW + 60, "hdr": {"ct": "sealed1"}, "body": "AAAA", "rp": "Ry3kq0wEo2nq1c9h5c7Zab"}
CONTAINER = json.loads(vec["A8"]["expected"]["container"])
COMMAND = vec["A8"]["inputs"]["command"]
TICKET_H, TICKET_C = vec["A9"]["inputs"]["header"], vec["A9"]["inputs"]["claims"]
ROT = vec["A10"]["inputs"]["statement"]
RING = vec["A11"]["inputs"]["body"]
SIG64 = vec["A8"]["expected"]["signature"]
PLUGIN_TOKEN = vec["A4b"]["expected"]["tokenForOnePhone"]
MOBILE_TOKEN = vec["A4"]["expected"]["token"]
PHONE_TH, DESK_TH = TH["phone"], TH["desktopEndpoint"]
ROSTER_HASH = roster_hash(7, [PHONE_TH])
STUN = {"urls": ["stun:stun.example.com:3478"], "username": "", "credential": ""}   # the plugin decoder requires both members, empty on STUN
TURN = {"urls": ["turn:turn.example.com:3478?transport=udp", "turns:turn.example.com:5349?transport=tcp"],
        "username": vec["A5"]["expected"]["username"], "credential": vec["A5"]["expected"]["credential"], "expiresAt": NOW + 600}
RELAY_URLS = {"challengeUrl": "https://relay.example.com/v1/aokie-companion/relay/challenge",
              "framesUrl": "https://relay.example.com/v1/aokie-companion/relay/frames",
              "streamUrl": "https://relay.example.com/v1/aokie-companion/relay/stream"}
ENDPOINT_KEY = {"algorithm": "ed25519", "publicKey": PUB["desktopEndpoint"], "thumbprint": DESK_TH}
PLUGIN_RESPONSE = {
    "accessToken": PLUGIN_TOKEN, "tokenType": "Bearer", "expiresIn": 90, "expiresAt": NOW + 90,
    "gatewayUrl": "wss://relay.example.com/v2/realtime", "appId": "aokie", "subjectId": "aokie", "role": "plugin",
    "scopes": ["state_read", "rtc_signal"], "device": {"id": "aokie", "appId": "aokie", "subjectId": "aokie", "role": "plugin"},
    "iceServers": [STUN, TURN], "relayOnly": False,
    "turnCredentialExpiresAt": NOW + 600, "endpointPublicKey": ENDPOINT_KEY, "holderKeyThumbprint": DESK_TH,
    "approvedPeerKeyThumbprints": [PHONE_TH], "peerRosterRevision": 7, "peerRosterHash": ROSTER_HASH, "relay": RELAY_URLS}
MOBILE_RESPONSE = {
    "accessToken": MOBILE_TOKEN, "tokenType": "Bearer", "expiresIn": 90, "expiresAt": NOW + 90,
    "gatewayUrl": "wss://relay.example.com/v2/realtime", "appId": "aokie", "subjectId": PHONEID, "role": "mobile",
    "holderKeyThumbprint": PHONE_TH, "expectedPeerKeyThumbprint": DESK_TH, "scopes": ["state_read", "caller_read"],
    "iceServers": [STUN, TURN], "relayOnly": False, "turnCredentialExpiresAt": NOW + 600,
    "device": {"id": PHONEID, "appId": "aokie", "subjectId": PHONEID, "role": "mobile", "displayName": "Test phone",
               "grants": ["state_read", "caller_read"], "approvedAt": "2026-09-29T01:02:03Z", "lastSeenAt": "2026-09-29T01:05:03Z"},
    "relay": RELAY_URLS}
DEVICE = {"id": PHONEID, "role": "phone", "name": "Test phone", "ver": "0.1.0", "createdAt": NOW, "lastSeen": NOW + 5, "revokedAt": None,
          "thumbprint": PHONE_TH, "ownerDesktop": DEVID, "grants": ["state_read", "caller_read"], "flags": {"canCmd": False}, "push": {"kind": None}}
SUB_KEYS = {"ed25519": PUB["host"], "x25519": XP["host"]}
CHALLENGE_PLUGIN = {"kind": "endpoint_challenge", "schemaVersion": 2, "appId": "aokie", "subjectId": "aokie", "role": "plugin",
                    "connectionId": "relay_" + "0" * 32, "challengeNonce": "challenge_" + "1" * 32,
                    "admissionJti": "adm_" + "0" * 31 + "1", "holderKeyThumbprint": DESK_TH,
                    "approvedPeerKeyThumbprints": [PHONE_TH], "peerRosterRevision": 7, "peerRosterHash": ROSTER_HASH, "expiresAt": NOW + 25}
CHALLENGE_MOBILE = {k: v for k, v in CHALLENGE_PLUGIN.items() if k not in ("approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")}
CHALLENGE_MOBILE.update({"subjectId": PHONEID, "role": "mobile", "holderKeyThumbprint": PHONE_TH, "expectedPeerKeyThumbprint": DESK_TH})
CTL = {"provider.updated": {"t": "provider.updated", "id": PROVID},
       "provider.rotated": {"t": "provider.rotated", "b": b64u(b"statement"), "s": SIG64},
       "device.revoked": {"t": "device.revoked", "id": PHONEID},
       "token.age": {"t": "token.age", "days": 95},
       "relay.notice": {"t": "relay.notice", "level": "warn", "message": "Disk is nearly full."}}
LANES = vec["extras"]["lanes"]
MIN_INFO = {"protocol": "oaiy-relay/1", "minClient": 1, "relayId": RELAYID,
            "relayKey": {"algorithm": "ed25519", "publicKey": PUB["relay"], "thumbprint": TH["relay"]},
            "software": {"name": "oaiy-relay", "version": "0.1.0"}, "features": ["poll", "items", "presence", "methods.post-forms"],
            "wait": {"default": 20, "max": 20, "pollGapMs": 250, "fallbackS": 5}, "presenceWindow": 60,
            "limits": {"batchItems": 64, "batchBytes": 1048576, "mailboxItems": 512, "mailboxBytes": 8388608, "hdrBytes": 512,
                       "held": {"soft": 3, "hard": 4, "measured": False},
                       "lanes": {ln: LANES[ln] for ln in ("cmd", "res", "ctl", "sync")}}}
POST_CMD = {"to": "dev:" + DEVID, "lane": "cmd", "id": "cmd-0001", "ttl": 60, "hdr": {"ct": "sealed1"}, "body": b64u(b"sealed")}
POST_RING = {"to": "dev:" + PHONEID, "lane": "ring", "id": "ring-0001", "ttl": 30, "hdr": {"ct": "json", "prio": 1, "sig": vec["A11"]["expected"]["hdrSig"]}, "body": vec["A11"]["expected"]["bodyText"]}
STATUS = {"v": 1, "version": "0.1.0", "php": {"version": "8.2.0", "sapi": "fpm-fcgi", "extensions": ["sodium", "pdo_sqlite"]},
          "db": {"driver": "sqlite", "sizeBytes": 4096}, "items": {"live": 0, "bytes": 0, "oldestAgeS": None},
          "devices": [{"id": DEVID, "role": "desktop", "name": "Front desk PC", "online": True}],
          "holds": {"soft": 3, "hard": 4, "measured": False, "byKind": {"poll": 1}, "live": 1},
          "rejected24h": {"unauthorized": 2}, "noAuthHeader24h": 0, "tokensOlderThan90d": [], "warnings": ["capacity not measured"], "time": NOW}

# The largest header the allow-list can express: every value at its own limit.
MAX_HDR = {"re": "r" * 128, "ct": "sealed1", "eph": "A" * 43 + "=", "kid": "k" * 64, "prio": 1, "n": 2 ** 53 - 1, "sig": "A" * 88}

POS: dict[str, list[tuple[str, object]]] = {}


def pos(schema: str, label: str, doc) -> None:
    POS.setdefault(schema, []).append((label, doc))


# common definitions
for d, ex_ in {
        "b64u": [TOKEN.split(".")[2]], "b64u8": ["AQIDBAUGBwg"], "b64u16": ["b5YkfMcTvJb0g1GTv3kNNQ"], "key32": [DESK_TH],
        "thumbprint": [DESK_TH, PHONE_TH], "signature": [SIG64], "relayId": [RELAYID], "deviceId": [DEVID, PHONEID],
        "providerId": [PROVID], "principalId": [DEVID, PROVID], "pid": ["b5YkfMcTvJb0g1GTv3kNNQ"], "rid": ["Ry3kq0wEo2nq1c9h5c7Zab"],
        "itemId": ["cmd-0001", "6b6e21fd-6cd2-41ed-ac1a-a30a91cbad3a", "a.b_c-d", "x" * 128, "...", ".a", "a..b", "a."], "appId": ["aokie", "a" * 64, "com.acme:app_1"],
        "token": [TOKEN], "adminToken": ["oaiyadm1.AQIDBAUGBwg." + TOKEN.split(".")[2]], "epoch": ["AQIDBAUGBwg"], "kid": ["OJttnmp91Xo"],
        "unixTime": [0, NOW, 2 ** 53 - 1], "uint53": [0, 42], "seq": [1, 42, 2 ** 53 - 1], "lane": list(LANES), "role": ["desktop", "phone", "provider", "web"],
        "ct": ["text", "json", "sealed1", "tunnel1", "noise1"], "errorCode": ["invalid_request", "unavailable"], "grant": ["state_read", "rtc_signal"],
        "origin": ["https://app.example.com", "https://app.example.com:8443"], "publicUrl": ["https://relay.example.com", "https://relay.example.com:8443"],
        "inboxAddress": ["dev:" + DEVID, "rbx:Ry3kq0wEo2nq1c9h5c7Zab"],
        "mailbox": ["dev:" + DEVID, "rbx:Ry3kq0wEo2nq1c9h5c7Zab", "app:aokie@" + DEVID + "/plugin", "app:aokie@" + DEVID + "/mobile:" + PHONE_TH],
        "from": [PROVID, DEVID, "rbx:Ry3kq0wEo2nq1c9h5c7Zab", "relay"],
        "hdr": [{}, {"ct": "sealed1"}, {"re": "cmd-0001", "ct": "tunnel1", "eph": "A" * 43 + "=", "kid": "k1", "prio": 1, "n": 0, "sig": "A" * 88},
                MAX_HDR],
        "jws": [vec["A9"]["expected"]["ticket"]], "etag": ['"AQIDBAUGBwgJCgsMDQ4PEA"'], "slotName": ["call.current", "a", "private.x-1"],
        "sha256Hex": ["0" * 64], "typedCode": [A3["expected"]["typedCode"]], "sasCode": [A3["expected"]["sasDisplay"]],
        "pairingUri": [A3["expected"]["pairingUri"]], "enrollUri": [vec["A7"]["expected"]["uri"]],
        "endpointPublicKey": [ENDPOINT_KEY], "hostIdentity": [OFFER["hostIdentity"]], "name60": ["x" * 60, "Front desk PC"],
        "unixTimeOrNull": [None, NOW]}.items():
    for e in ex_:
        pos("common#" + d, f"{d}: {str(e)[:40]}", e)

pos("error", "a rate limit with retryAfter", {"error": {"code": "rate_limited", "message": "Slow down.", "retryAfter": 7}})
pos("error", "an unauthorized error", {"error": {"code": "unauthorized", "message": "Credential missing or wrong."}})
pos("compat-error", "the Aokie shape", {"error": True, "code": "relay_backpressure", "message": "Mailbox full."})
pos("health", "a healthy relay", {"ok": True, "time": NOW, "authHeaderSeen": False})
pos("info", "the design's full example (Appendix A6b)", INFO)
pos("info", "the reduced document a first relay serves", MIN_INFO)
pos("item", "the design's example item", ITEM)
pos("item", "an item with no reply box", {k: v for k, v in ITEM.items() if k != "rp"})
pos("post-request", "a command", {"items": [POST_CMD]})
pos("post-request", "a signed ring", {"items": [POST_RING]})
pos("post-request", "an item with no ttl and no hdr", {"items": [{"to": "dev:" + DEVID, "lane": "ctl", "id": "n1", "body": "{}"}]})
pos("post-request", "64 items", {"items": [dict(POST_CMD, id=f"c{i}") for i in range(64)]})
pos("post-response", "queued, duplicate and rejected", {"v": 1, "time": NOW, "results": [
    {"id": "a", "status": "queued", "seq": 43}, {"id": "b", "status": "duplicate", "seq": 41},
    {"id": "c", "status": "rejected", "error": {"code": "quota_exceeded", "message": "Mailbox full.", "retryAfter": 5}}]})
pos("poll-response", "one item", {"v": 1, "epoch": "AQIDBAUGBwg", "cursor": 42, "items": [ITEM], "more": False, "time": NOW, "hold": {"granted": True}})
pos("poll-response", "an empty poll without a hold", {"v": 1, "epoch": "AQIDBAUGBwg", "cursor": 42, "items": [], "more": False, "time": NOW})
pos("poll-response", "a refused hold", {"v": 1, "epoch": "AQIDBAUGBwg", "cursor": 42, "items": [], "more": False, "time": NOW, "hold": {"refused": True, "retryAfter": 2}})
pos("poll-response", "a superseded hold", {"v": 1, "epoch": "AQIDBAUGBwg", "cursor": 42, "items": [], "more": False, "time": NOW, "hold": {"granted": True, "superseded": True}})
pos("poll-response", "a reset", {"v": 1, "epoch": "AQIDBAUGBwg", "cursor": 7, "items": [], "more": False, "time": NOW, "reset": True})
pos("item-state", "an acked item", {"id": "cmd-0001", "lane": "cmd", "to": "dev:" + DEVID, "state": "acked", "seq": 43, "at": NOW, "exp": NOW + 60, "deliveredAt": NOW + 1, "ackedAt": NOW + 2, "time": NOW + 3})
pos("item-state", "a queued item", {"id": "cmd-0001", "lane": "cmd", "to": "dev:" + DEVID, "state": "queued", "seq": 43, "at": NOW, "exp": NOW + 60, "time": NOW})
pos("slot-request", "call.current", {"body": "{}", "ttl": 120, "ct": "json", "readers": ["phone:caller_read", "provider"]})
pos("slot-response", "a publish", {"etag": '"AQIDBAUGBwgJCgsMDQ4PEA"', "exp": NOW + 120, "time": NOW})
pos("slot", "a read", {"name": "call.current", "dev": DEVID, "body": "{}", "ct": "json", "at": NOW, "exp": NOW + 120, "time": NOW})
pos("presence", "two devices", {"v": 1, "time": NOW, "devices": [
    {"id": DEVID, "role": "desktop", "name": "Front desk PC", "online": True, "changedAt": NOW, "ver": "0.1.0", "caps": ["relay"]},
    {"id": PROVID, "role": "provider", "name": "FormLogic", "online": False, "changedAt": None}]})
pos("enroll-request", "the A7 request", vec["A7"]["inputs"]["request"])
pos("enroll-request", "a provider redemption", dict(vec["A7"]["inputs"]["request"], role="provider"))
pos("enroll-response", "a desktop", {"deviceId": DEVID, "token": TOKEN, "relayId": RELAYID, "time": NOW})
pos("enroll-response", "a provider", {"deviceId": PROVID, "token": TOKEN, "relayId": RELAYID, "time": NOW})
pos("device", "a phone", DEVICE)
pos("devices-list", "one phone", {"v": 1, "time": NOW, "devices": [DEVICE]})
pos("device-patch-request", "grants only", {"grants": ["state_read", "monitor"]})
pos("device-patch-request", "name and flags", {"name": "Kitchen phone", "flags": {"canCmd": True}})
pos("device-patch-response", "the patched device", {"v": 1, "time": NOW, "device": DEVICE})
pos("device-meta-request", "name and keys", {"name": "Front desk PC", "ver": "0.1.0", "caps": ["relay"], **SUB_KEYS})
pos("ack", "an acknowledgement", {"v": 1, "time": NOW})
pos("devices-revoke-request", "every phone", {"role": "phone"})
pos("devices-revoke-response", "two revoked", {"v": 1, "time": NOW, "revoked": [PHONEID, DEVID]})
pos("roster-request", "a sorted roster", {"appId": "aokie", "revision": 7, "thumbprints": sorted([PHONE_TH, DESK_TH])})
pos("roster-request", "an empty roster", {"appId": "aokie", "revision": 0, "thumbprints": []})
pos("roster-response", "a push", {"v": 1, "time": NOW, "hash": ROSTER_HASH, "revoked": [PHONEID]})
pos("token-rotate-response", "a rotation", {"token": TOKEN, "graceUntil": NOW + 600, "time": NOW})
pos("keys-request", "a provider key", {"role": "provider", "ttl": 3600, "name": "FormLogic"})
pos("keys-response", "a provider key", {"uri": vec["A7"]["expected"]["uri"].replace("r=desktop", "r=provider"), "kid": "OJttnmp91Xo", "exp": NOW + 3600, "time": NOW})
pos("providers-request", "FormLogic", {"name": "FormLogic", **SUB_KEYS, "origins": ["https://app.example.com"]})
pos("providers-response", "a registration", {"providerId": PROVID, "thumbprint": TH["provider"], "time": NOW})
pos("push-register-request", "register", {"kind": "fcm", "token": "x" * 16, "project": "demo"})
pos("push-register-request", "remove", {"remove": True})
pos("pairing-create-request", "the A3 offer", {"pid": A3["expected"]["pid"], "offer": A3["expected"]["offerText"], "mac": A3["expected"]["offerMac"], "ttl": 600, "appId": "aokie", "desktopThumbprint": DESK_TH})
pos("pairing-create-response", "a rendezvous", {"pid": A3["expected"]["pid"], "exp": NOW + 600, "time": NOW})
pos("pairing-fetch-response", "open", {"v": 1, "state": "open", "offer": A3["expected"]["offerText"], "mac": A3["expected"]["offerMac"], "exp": NOW + 600, "time": NOW})
pos("pairing-fetch-response", "approved", {"v": 1, "state": "approved", "deviceId": PHONEID, "sealedToken": b64u(b"x" * 100), "receipt": {"issuedAt": NOW, "signature": SIG64}, "time": NOW})
pos("pairing-fetch-response", "denied", {"v": 1, "state": "denied", "time": NOW})
pos("pairing-answer-request", "the A3 response", {"response": json.dumps(PAIR_RESPONSE)})
pos("pairing-answer-response", "answered", {"state": "answered", "time": NOW})
pos("pairing-offer", "the A3 offer (Appendix A3)", OFFER)
pos("pairing-claims", "the A3 claims", CLAIMS)
pos("pairing-claims", "claims without a display name", {k: v for k, v in CLAIMS.items() if k != "displayName"})
pos("pairing-response", "the A3 response", PAIR_RESPONSE)
pos("pairing-decision", "approve", {"approve": True, "phone": {"ed25519": PUB["phone"], "x25519": XP["phone"], "thumbprint": PHONE_TH}, "name": "Test phone", "appId": "aokie",
                                    "grants": RECEIPT["grants"], "receipt": {"issuedAt": NOW, "signature": SIG64}})
pos("pairing-decision", "deny", {"approve": False})
pos("pairing-reject-request", "a reason", {"reason": "mac mismatch"})
pos("pairing-reject-request", "no reason at all", {})
pos("pairing-decision-response", "an approval", {"v": 1, "state": "approved", "deviceId": PHONEID, "time": NOW})
pos("pairing-decision-response", "a denial (no device)", {"v": 1, "state": "denied", "time": NOW})
pos("pairing-state-response", "a reject that reopened it", {"v": 1, "state": "open", "time": NOW})
pos("pairing-state-response", "a burn, or the third reject", {"v": 1, "state": "expired", "time": NOW})
FETCH_OPEN = POS["pairing-fetch-response"][0][1]
pos("pairing-fetch-response", "open, after a granted wait", dict(FETCH_OPEN, hold={"granted": True}))
pos("pairing-fetch-response", "answered, superseded by a newer wait", dict(FETCH_OPEN, state="answered", hold={"granted": True, "superseded": True}))
pos("pairing-fetch-response", "open, the wait refused because the pool is nearly full", dict(FETCH_OPEN, hold={"refused": True, "retryAfter": 2}))
SEALED_FIXTURE = json.loads((V1 / "fixtures" / "sealed-token.json").read_text(encoding="utf-8"))
pos("sealed-token-fixture", "the recorded fixture", SEALED_FIXTURE)
pos("approval-receipt", "the A3 receipt document", RECEIPT)
pos("admission-claims", "the A4 mobile claims", vec["A4"]["inputs"]["claims"])
pos("admission-claims", "a plugin claims set", {"aud": "aokie-v2-gateway", "appId": "aokie", "subjectId": "aokie", "role": "plugin", "holderKeyThumbprint": DESK_TH,
                                                "approvedPeerKeyThumbprints": [PHONE_TH], "peerRosterRevision": 7, "peerRosterHash": ROSTER_HASH,
                                                "scopes": ["state_read", "rtc_signal"], "dsk": DEVID, "exp": NOW + 90, "jti": "adm_" + "0" * 31 + "1"})
PLUGIN_REQUEST = {"appId": "aokie", "pluginId": "aokie", "displayName": "Receptionist", "endpointPublicKey": ENDPOINT_KEY, "holderKeyThumbprint": DESK_TH,
                  "approvedPeerKeyThumbprints": [PHONE_TH], "peerRosterRevision": 7, "peerRosterHash": ROSTER_HASH, "supportedTransports": ["relay"]}
pos("admission-plugin-request", "a plugin admission request", PLUGIN_REQUEST)
pos("admission-plugin-request", "a plugin admission request without supportedTransports (the desktop's broker sends none: it means relay)",
    {k: v for k, v in PLUGIN_REQUEST.items() if k != "supportedTransports"})
pos("admission-plugin-response", "a plugin admission response", PLUGIN_RESPONSE)
pos("admission-plugin-response", "a poll-mode response", dict(PLUGIN_RESPONSE, relay=dict(RELAY_URLS, mode="poll")))
MOBILE_REQUEST = {"appId": "aokie", "deviceId": PHONEID, "displayName": "Test phone", "holderKeyThumbprint": PHONE_TH, "supportedTransports": ["relay"]}
pos("admission-mobile-request", "a phone admission request", MOBILE_REQUEST)
pos("admission-mobile-request", "a phone admission request without supportedTransports", {k: v for k, v in MOBILE_REQUEST.items() if k != "supportedTransports"})
pos("admission-mobile-response", "a phone admission response", MOBILE_RESPONSE)
pos("ice-server", "STUN", STUN)
pos("ice-server", "TURN with the A5 credential", TURN)
pos("challenge", "a plugin challenge", CHALLENGE_PLUGIN)
pos("challenge", "a phone challenge", CHALLENGE_MOBILE)
pos("compat-frames-request", "a frame to the plugin", {"to": "plugin", "frames": [{"type": "hello"}]})
pos("compat-frames-request", "a frame to a phone", {"to": "mobile:" + PHONE_TH, "frames": [{"type": "state"}]})
FRAME = {"seq": 9, "from": "plugin", "subjectId": "aokie", "grants": ["state_read"], "frame": {"type": "state"}}
pos("compat-frames-accepted", "accepted", {"accepted": 1, "seq": 9, "time": NOW})
pos("compat-frames-accepted", "sixty-four accepted", {"accepted": 64, "seq": 2 ** 53 - 1, "time": NOW})
pos("compat-frames-page", "one frame", {"frames": [FRAME], "lastSeq": 9, "time": NOW})
pos("compat-frames-page", "the tail: no frames, lastSeq is the cursor asked for", {"frames": [], "lastSeq": 9, "time": NOW})
pos("compat-frames-page", "a granted wait that found nothing", {"frames": [], "lastSeq": 9, "time": NOW, "hold": {"granted": True}})
pos("compat-frames-page", "a wait ended by a newer one", {"frames": [], "lastSeq": 9, "time": NOW, "hold": {"granted": True, "superseded": True}})
pos("compat-frames-page", "a wait the pool refused", {"frames": [FRAME], "lastSeq": 9, "time": NOW, "hold": {"refused": True, "retryAfter": 2}})
pos("compat-stream-frame", "a frame from the plugin", FRAME)
pos("compat-stream-frame", "a frame from a phone with the plugin's whole scope set", dict(FRAME, **{"from": "mobile:" + PHONE_TH, "subjectId": PHONEID, "grants": ["state_read", "caller_read", "captions_read", "assistance_read", "assistance_respond", "rtc_signal"]}))
pos("compat-stream-frame", "a frame with empty grants (a sender with no known scope)", dict(FRAME, grants=[]))
pos("ring", "the A11 voice offer", RING)
pos("ring", "a cancel", {"aokieClass": "voice_offer_cancel", "schemaVersion": "1", "eventId": "evt_2", "offerId": "toffer_0001", "reason": "answered elsewhere"})
pos("ring", "an assistance offer", {"aokieClass": "assistance_offer", "schemaVersion": "1", "eventId": "evt_3", "appId": "aokie", "requestId": "req_1", "callId": "call_0123", "callEpoch": "7", "ownerEpoch": "0", "expiresAt": "1790000040"})
pos("ring", "an informational notice", {"aokieClass": "informational", "schemaVersion": "1", "eventId": "evt_4", "title": "Front desk", "body": "The receptionist restarted.", "expiresAt": "1790086400"})
pos("command", "the A8 command", COMMAND)
RESULT_OK = {"v": 1, "re": "cmd-0001", "dev": DEVID, "status": "done", "result": {"ok": True}, "error": None, "at": NOW}
pos("result", "done", RESULT_OK)
pos("result", "failed", dict(RESULT_OK, status="failed", result=None, error={"code": "expired", "message": "The command was too old."}))
pos("container", "the A8 container", CONTAINER)
pos("container", "a padded container", dict(CONTAINER, p=b64u(b"\x00" * 64)))
pos("rotation-statement", "the A10 statement", ROT)
pos("ticket-header", "the A9 header", TICKET_H)
pos("ticket-claims", "the A9 claims", TICKET_C)
for name, doc in CTL.items():
    pos("ctl", name, doc)
pos("ctl", "an unknown t is valid and ignored by receivers", {"t": "future.thing", "x": 1})
pos("replybox-request", "a reply box", {"rid": "Ry3kq0wEo2nq1c9h5c7Zab", "h": DESK_TH, "ttl": 900})
pos("replybox-response", "a reply box", {"rid": "Ry3kq0wEo2nq1c9h5c7Zab", "exp": NOW + 900, "time": NOW})
pos("rbx-item-request", "a chat request", {"lane": "ai", "id": "req-1", "ttl": 300, "hdr": {"ct": "tunnel1", "eph": "A" * 43 + "="}, "body": "AAAA"})
pos("admin-capacity-request", "a measurement", {"workers": 8, "streamOk": True, "maxBody": 393216, "maxHold": 35})
pos("admin-capacity-response", "the effect", {"v": 1, "time": NOW, "effective": {"workers": 8, "heldSoft": 4, "heldHard": 7, "waitMax": 20, "streamOk": True, "laneBodies": {"cmd": 32768, "sig": 196608}}})
pos("admin-status", "a status", STATUS)
pos("admin-status", "a status with diagnostics", dict(STATUS, diag={"sapi": "fpm-fcgi", "authorizationSeen": True}))

section("valid documents are accepted")
n_pos = 0
for schema_name, cases in POS.items():
    for label, doc in cases:
        n_pos += 1
        stem = schema_name if "#" in schema_name else schema_name
        errs = problems(schema_name, doc)
        ok(f"{schema_name}: {label}", not errs, errs[0].message[:150] if errs else "")
print(f"\n  {n_pos} positive documents")
ok(f"hdr: the largest header the allow-list can express is {hdr_bytes(MAX_HDR)} bytes, under the 512 byte cap (the cap is defence in depth)",
   hdr_bytes(MAX_HDR) < 512)
for name in schemas:
    stem = name.replace(".schema.json", "")
    if stem == "common":
        continue
    ok(f"{stem} has at least one positive document", stem in POS)
common_defs = set(schemas["common.schema.json"]["$defs"])
ok("every common definition has a positive document", common_defs <= {k.split("#", 1)[1] for k in POS if k.startswith("common#")},
   str(common_defs - {k.split("#", 1)[1] for k in POS if k.startswith("common#")}))

# ---------------------------------------------------------------------------
# Negative documents. (schema, label, doc, hint). The hint must appear in the error trail.
DEL = object()


def mut(doc, path: str, value=DEL):
    d = copy.deepcopy(doc)
    node = d
    parts = path.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    last = parts[-1]
    if isinstance(node, list):
        last = int(last)
        if value is DEL:
            del node[last]
        else:
            node[last] = value
    elif value is DEL:
        del node[last]
    else:
        node[last] = value
    return d


NEG: list[tuple[str, str, object, str]] = []


def neg(schema: str, label: str, doc, hint: str) -> None:
    NEG.append((schema, label, doc, hint))


def p1(schema: str, i: int = 0):
    return POS[schema][i][1]


# --- items and posting (4.3, 4.4)
neg("post-request", "a reserved lane (flow) is not served in v1", mut(p1("post-request"), "items.0.lane", "flow"), "lane")
neg("post-request", "an unknown lane", mut(p1("post-request"), "items.0.lane", "chat"), "lane")
neg("post-request", "an upper-case lane", mut(p1("post-request"), "items.0.lane", "CMD"), "lane")
neg("post-request", "a party mailbox is not a valid post target", mut(p1("post-request"), "items.0.to", "app:aokie@" + DEVID + "/plugin"), "to")
neg("post-request", "a provider id cannot be a device inbox address", mut(p1("post-request"), "items.0.to", "dev:" + PROVID), "to")
neg("post-request", "a device inbox needs the dev: scheme", mut(p1("post-request"), "items.0.to", DEVID), "to")
neg("post-request", "an unknown hdr key", mut(p1("post-request"), "items.0.hdr.x", "1"), "hdr")
neg("post-request", "an unknown hdr ct", mut(p1("post-request"), "items.0.hdr.ct", "xml"), "ct")
neg("post-request", "hdr prio 2", mut(p1("post-request"), "items.0.hdr.prio", 2), "prio")
neg("post-request", "hdr n negative", mut(p1("post-request"), "items.0.hdr.n", -1), "n")
neg("post-request", "hdr eph not 44 standard base64 characters", mut(p1("post-request"), "items.0.hdr.eph", "A" * 43), "eph")
neg("post-request", "hdr sig of 89 characters", mut(p1("post-request"), "items.0.hdr.sig", "A" * 89), "sig")
neg("post-request", "hdr kid of 65 characters", mut(p1("post-request"), "items.0.hdr.kid", "k" * 65), "kid")
neg("post-request", "hdr re with a path character", mut(p1("post-request"), "items.0.hdr.re", "../x"), "re")
neg("post-request", "ttl zero", mut(p1("post-request"), "items.0.ttl", 0), "ttl")
neg("post-request", "ttl negative", mut(p1("post-request"), "items.0.ttl", -1), "ttl")
neg("post-request", "ttl fractional", mut(p1("post-request"), "items.0.ttl", 1.5), "ttl")
neg("post-request", "ttl a string", mut(p1("post-request"), "items.0.ttl", "60"), "ttl")
neg("post-request", "ttl null", mut(p1("post-request"), "items.0.ttl", None), "ttl")
neg("post-request", "cmd ttl above its maximum (301)", mut(p1("post-request"), "items.0.ttl", 301), "ttl")
neg("post-request", "ring ttl above its maximum (301)", mut(p1("post-request", 1), "items.0.ttl", 301), "ttl")
neg("post-request", "ctl ttl above its maximum (86401)", mut(p1("post-request", 2), "items.0.ttl", 86401), "ttl")
neg("post-request", "item id with a path traversal", mut(p1("post-request"), "items.0.id", "../etc/passwd"), "id")
neg("post-request", "item id of 129 characters", mut(p1("post-request"), "items.0.id", "x" * 129), "id")
neg("post-request", "empty item id", mut(p1("post-request"), "items.0.id", ""), "id")
neg("post-request", "item id with a space", mut(p1("post-request"), "items.0.id", "a b"), "id")
neg("post-request", "item id with a slash", mut(p1("post-request"), "items.0.id", "a/b"), "id")
neg("post-request", "body is an object, not an opaque string", mut(p1("post-request"), "items.0.body", {"a": 1}), "body")
neg("post-request", "cmd body one over the lane cap (32769)", mut(p1("post-request"), "items.0.body", "A" * 32769), "body")
neg("post-request", "ring body one over the lane cap (4097)", mut(p1("post-request", 1), "items.0.body", "A" * 4097), "body")
neg("post-request", "a ring item without hdr.sig", mut(p1("post-request", 1), "items.0.hdr.sig"), "sig")
neg("post-request", "a ring item without hdr", mut(p1("post-request", 1), "items.0.hdr"), "hdr")
neg("post-request", "65 items", {"items": [POST_CMD] * 65}, "items")
neg("post-request", "no items", {"items": []}, "items")
neg("post-request", "an item without a body", mut(p1("post-request"), "items.0.body"), "body")
neg("item", "seq zero (the first item of a mailbox is 1)", mut(ITEM, "seq", 0), "seq")
neg("item", "seq at 2^53", mut(ITEM, "seq", 2 ** 53), "seq")
neg("item", "an unknown lane in a delivered item", mut(ITEM, "lane", "flow.in"), "lane")
neg("item", "no hdr", mut(ITEM, "hdr"), "hdr")
neg("item", "a from that is neither a device, a provider, a reply box nor relay", mut(ITEM, "from", "someone"), "from")
neg("item", "body null (an acknowledged item is never delivered)", mut(ITEM, "body", None), "body")
neg("item", "rp is not a reply box id", mut(ITEM, "rp", "short"), "rp")
neg("post-response", "a rejected result without an error", {"v": 1, "time": NOW, "results": [{"id": "c", "status": "rejected"}]}, "error")
neg("post-response", "a queued result without a seq", {"v": 1, "time": NOW, "results": [{"id": "a", "status": "queued"}]}, "seq")
neg("post-response", "an unknown status", {"v": 1, "time": NOW, "results": [{"id": "a", "status": "pending", "seq": 1}]}, "status")
neg("post-response", "a rejected item carrying an unknown error code", {"v": 1, "time": NOW, "results": [{"id": "a", "status": "rejected", "error": {"code": "oops", "message": "x"}}]}, "code")
neg("poll-response", "reset with items", mut(p1("poll-response", 4), "items", [ITEM]), "items")
neg("poll-response", "no epoch", mut(p1("poll-response"), "epoch"), "epoch")
neg("poll-response", "epoch of the wrong length", mut(p1("poll-response"), "epoch", "AQID"), "epoch")
neg("poll-response", "a hold both granted and refused", mut(p1("poll-response"), "hold", {"granted": True, "refused": True, "retryAfter": 2}), "hold")
neg("poll-response", "a refused hold without retryAfter", mut(p1("poll-response", 2), "hold", {"refused": True}), "hold")
neg("poll-response", "reset false is not a spelling (it is absent or true)", mut(p1("poll-response"), "reset", False), "reset")
neg("poll-response", "65 items", mut(p1("poll-response"), "items", [ITEM] * 65), "items")
neg("poll-response", "no time", mut(p1("poll-response"), "time"), "time")
neg("item-state", "an unknown state", mut(p1("item-state"), "state", "lost"), "state")
neg("item-state", "no time", mut(p1("item-state"), "time"), "time")

# --- errors (4.6)
neg("error", "an unknown error code", {"error": {"code": "oops", "message": "x"}}, "code")
neg("error", "an Aokie code on a native route", {"error": {"code": "invalid_token", "message": "x"}}, "code")
neg("error", "a message of 201 characters", {"error": {"code": "internal", "message": "x" * 201}}, "message")
neg("error", "an empty message", {"error": {"code": "internal", "message": ""}}, "message")
neg("error", "no message", {"error": {"code": "internal"}}, "message")
neg("error", "an unwrapped error", {"code": "internal", "message": "x"}, "error")
neg("error", "retryAfter negative", {"error": {"code": "rate_limited", "message": "x", "retryAfter": -1}}, "retryAfter")
neg("compat-error", "error must be the boolean true", {"error": False, "code": "invalid_token", "message": "x"}, "error")
neg("compat-error", "an unknown compat code", {"error": True, "code": "nope", "message": "x"}, "code")
neg("health", "an extra member", dict(p1("health"), extra=1), "extra")
neg("health", "ok false", dict(p1("health"), ok=False), "ok")
neg("health", "no authHeaderSeen", mut(p1("health"), "authHeaderSeen"), "authHeaderSeen")
neg("info", "a foreign protocol major", mut(INFO, "protocol", "oaiy-relay/2"), "protocol")
neg("info", "a time member (the document is static)", mut(INFO, "time", NOW), "False schema")
neg("info", "relayKey algorithm rsa", mut(INFO, "relayKey.algorithm", "rsa"), "algorithm")
neg("info", "relayKey thumbprint of 42 characters", mut(INFO, "relayKey.thumbprint", INFO["relayKey"]["thumbprint"][:42]), "thumbprint")
neg("info", "a lane name that is not a lane", mut(INFO, "limits.lanes.flow", INFO["limits"]["lanes"]["cmd"]), "flow")
neg("info", "no limits.lanes", mut(INFO, "limits.lanes"), "lanes")
neg("info", "wait.max above 300", mut(INFO, "wait.max", 301), "max")
neg("info", "a feature name with capitals", mut(INFO, "features", ["Poll"]), "features")
neg("info", "hdrBytes above the 512 byte cap", mut(INFO, "limits.hdrBytes", 513), "hdrBytes")
neg("info", "rosterMax above 16", mut(INFO, "limits.rosterMax", 17), "rosterMax")

# --- slots, presence (4.5)
neg("slot-request", "more than 8 readers", mut(p1("slot-request"), "readers", ["provider", "desktop", "phone"] + [f"phone:g{i}" for i in range(6)]), "readers")
neg("slot-request", "an unknown reader (web)", mut(p1("slot-request"), "readers", ["web"]), "readers")
neg("slot-request", "ttl above 3600", mut(p1("slot-request"), "ttl", 3601), "ttl")
neg("slot-request", "body over 65536", mut(p1("slot-request"), "body", "x" * 65537), "body")
neg("slot-request", "duplicate readers", mut(p1("slot-request"), "readers", ["provider", "provider"]), "readers")
neg("slot-response", "an unquoted ETag", mut(p1("slot-response"), "etag", "AQIDBAUGBwgJCgsMDQ4PEA"), "etag")
neg("slot", "a slot name with capitals", mut(p1("slot"), "name", "Call.Current"), "name")
neg("presence", "a device with an unknown role", mut(p1("presence"), "devices.0.role", "admin"), "role")
neg("presence", "online not boolean", mut(p1("presence"), "devices.0.online", "yes"), "online")

# --- enrolment, devices, roster, tokens (4.5, 4.11)
neg("enroll-request", "role phone cannot enrol", mut(p1("enroll-request"), "role", "phone"), "role")
neg("enroll-request", "kid of the wrong length", mut(p1("enroll-request"), "kid", "OJtt"), "kid")
neg("enroll-request", "n of the wrong length", mut(p1("enroll-request"), "n", "cHFy"), "n")
neg("enroll-request", "keys without x25519", mut(p1("enroll-request"), "keys.x25519"), "x25519")
neg("enroll-request", "a name of 61 characters", mut(p1("enroll-request"), "name", "x" * 61), "name")
neg("enroll-response", "a malformed token", mut(p1("enroll-response"), "token", TOKEN[:-1]), "token")
neg("device", "canCmd as a string", mut(DEVICE, "flags.canCmd", "false"), "canCmd")
neg("device", "an unknown role", mut(DEVICE, "role", "root"), "role")
neg("device", "no push member", mut(DEVICE, "push"), "push")
neg("devices-list", "no v", mut(p1("devices-list"), "v"), "v")
neg("device-patch-request", "an empty patch", {}, "minProperties")
neg("device-patch-request", "duplicate grants", {"grants": ["state_read", "state_read"]}, "grants")
neg("device-patch-response", "no device", mut(p1("device-patch-response"), "device"), "device")
neg("device-meta-request", "33 capability tokens", {"caps": ["c"] * 33}, "caps")
neg("device-meta-request", "an empty meta", {}, "minProperties")
neg("ack", "v 2", {"v": 2, "time": NOW}, "v")
neg("devices-revoke-request", "role desktop cannot be revoked in bulk", {"role": "desktop"}, "role")
neg("devices-revoke-response", "a revoked entry that is a provider id", {"v": 1, "time": NOW, "revoked": [PROVID]}, "revoked")
neg("roster-request", "17 thumbprints", mut(p1("roster-request"), "thumbprints", sorted(b64u(hashlib.sha256(bytes([i])).digest()) for i in range(17))), "thumbprints")
neg("roster-request", "a thumbprint of 42 characters", mut(p1("roster-request"), "thumbprints", [PHONE_TH[:42]]), "thumbprints")
neg("roster-request", "duplicate thumbprints", mut(p1("roster-request"), "thumbprints", [PHONE_TH, PHONE_TH]), "thumbprints")
neg("roster-request", "revision 2^53", mut(p1("roster-request"), "revision", 2 ** 53), "revision")
neg("roster-request", "revision negative", mut(p1("roster-request"), "revision", -1), "revision")
neg("roster-request", "appId of 65 characters", mut(p1("roster-request"), "appId", "a" * 65), "appId")
neg("roster-response", "a revoked entry that is not a device id", mut(p1("roster-response"), "revoked", ["x"]), "revoked")
neg("token-rotate-response", "no graceUntil", mut(p1("token-rotate-response"), "graceUntil"), "graceUntil")
neg("keys-request", "a desktop key cannot be minted over the network", {"role": "desktop"}, "role")
neg("keys-request", "ttl above 24 hours", {"role": "provider", "ttl": 86401}, "ttl")
neg("keys-response", "an enrolment uri that is not oaiy://enroll", mut(p1("keys-response"), "uri", "https://relay.example.com"), "uri")
neg("providers-request", "5 origins", mut(p1("providers-request"), "origins", [f"https://a{i}.example.com" for i in range(5)]), "origins")
neg("providers-request", "an origin with a path", mut(p1("providers-request"), "origins", ["https://app.example.com/x"]), "origins")
neg("providers-request", "an http origin", mut(p1("providers-request"), "origins", ["http://app.example.com"]), "origins")
neg("providers-response", "no providerId", mut(p1("providers-response"), "providerId"), "providerId")
neg("push-register-request", "a token of 15 characters", {"kind": "fcm", "token": "x" * 15}, "token")
neg("push-register-request", "an unknown kind", {"kind": "apns", "token": "x" * 16}, "kind")
neg("push-register-request", "register and remove at once", {"kind": "fcm", "token": "x" * 16, "remove": True}, "oneOf")

# --- pairing (4.10)
neg("pairing-offer", "schemaVersion 2 (a v2 phone must never mistake it)", mut(OFFER, "schemaVersion", 2), "schemaVersion")
neg("pairing-offer", "an extra member (workspaceId is dropped in v3)", mut(OFFER, "workspaceId", "w1"), "workspaceId")
neg("pairing-offer", "no hostIdentity", mut(OFFER, "hostIdentity"), "hostIdentity")
neg("pairing-offer", "a relay url with a path", mut(OFFER, "relay.url", "https://relay.example.com/x"), "url")
neg("pairing-offer", "a relay url on plain http", mut(OFFER, "relay.url", "http://relay.example.com"), "url")
neg("pairing-offer", "a wrong kind", mut(OFFER, "kind", "aokie_mobile_pairing_response"), "kind")
neg("pairing-offer", "a desktopName of 61 characters", mut(OFFER, "desktopName", "x" * 61), "desktopName")
neg("pairing-offer", "a nonce of the wrong length", mut(OFFER, "nonce", OFFER["nonce"][:40]), "nonce")
neg("pairing-offer", "an extra member inside hostIdentity", mut(OFFER, "hostIdentity.extra", "x"), "extra")
neg("pairing-claims", "an extra member", mut(CLAIMS, "extra", 1), "extra")
neg("pairing-claims", "no pairingNonce", mut(CLAIMS, "pairingNonce"), "pairingNonce")
neg("pairing-claims", "mobileX25519 of the wrong length", mut(CLAIMS, "mobileX25519", CLAIMS["mobileX25519"][:42]), "mobileX25519")
neg("pairing-claims", "a float timestamp", mut(CLAIMS, "issuedAt", 1790000030.5), "issuedAt")
neg("pairing-response", "no mac (the MAC proves knowledge of the secret)", mut(PAIR_RESPONSE, "mac"), "mac")
neg("pairing-response", "no signature", mut(PAIR_RESPONSE, "signature"), "signature")
neg("pairing-response", "schemaVersion 2", mut(PAIR_RESPONSE, "schemaVersion", 2), "schemaVersion")
neg("pairing-response", "a signature of 85 characters", mut(PAIR_RESPONSE, "signature", PAIR_RESPONSE["signature"][:85]), "signature")
neg("pairing-decision", "an approval without a receipt", mut(p1("pairing-decision"), "receipt"), "receipt")
neg("pairing-decision", "an approval with duplicate grants", mut(p1("pairing-decision"), "grants", ["state_read", "state_read"]), "grants")
neg("pairing-decision", "approve as a string", {"approve": "yes"}, "approve")
neg("approval-receipt", "a pid of the wrong length", mut(RECEIPT, "pid", "short"), "pid")
neg("approval-receipt", "an extra member", mut(RECEIPT, "extra", 1), "extra")
neg("pairing-create-response", "a pid of the wrong length", mut(p1("pairing-create-response"), "pid", "short"), "pid")
neg("pairing-create-request", "no mac", mut(p1("pairing-create-request"), "mac"), "mac")
neg("pairing-create-request", "an offer over 4096 characters", mut(p1("pairing-create-request"), "offer", "x" * 4097), "offer")
neg("pairing-create-request", "ttl above 900", mut(p1("pairing-create-request"), "ttl", 901), "ttl")
neg("pairing-answer-request", "a response over 8192 characters", {"response": "x" * 8193}, "response")
neg("pairing-answer-response", "a state other than answered", {"state": "open", "time": NOW}, "state")
neg("pairing-fetch-response", "approved without a sealed token", mut(p1("pairing-fetch-response", 1), "sealedToken"), "sealedToken")
neg("pairing-fetch-response", "open without an offer", mut(p1("pairing-fetch-response"), "offer"), "offer")
neg("pairing-fetch-response", "expired is an error (410), not a state", {"v": 1, "state": "expired", "time": NOW}, "state")
neg("pairing-reject-request", "a reason of 201 characters", {"reason": "x" * 201}, "reason")
neg("pairing-reject-request", "a reason that is a number", {"reason": 5}, "reason")
neg("pairing-create-request", "ttl zero", mut(p1("pairing-create-request"), "ttl", 0), "ttl")
neg("pairing-create-request", "ttl fractional", mut(p1("pairing-create-request"), "ttl", 1.5), "ttl")
neg("pairing-create-request", "no desktopThumbprint", mut(p1("pairing-create-request"), "desktopThumbprint"), "desktopThumbprint")
neg("pairing-create-request", "a pid of 23 characters", mut(p1("pairing-create-request"), "pid", "A" * 23), "pid")
neg("pairing-create-request", "an appId of 65 characters", mut(p1("pairing-create-request"), "appId", "a" * 65), "appId")
neg("pairing-create-request", "an empty offer", mut(p1("pairing-create-request"), "offer", ""), "offer")
neg("pairing-answer-request", "no response member", {}, "response")
neg("pairing-answer-request", "an empty response", {"response": ""}, "response")
neg("pairing-answer-request", "a response that is an object, not text", {"response": {"kind": "aokie_mobile_pairing_response"}}, "response")
neg("pairing-decision", "an approval with 17 grants", mut(p1("pairing-decision"), "grants", [f"g{i}" for i in range(17)]), "grants")
neg("pairing-decision", "a grant that is not a name", mut(p1("pairing-decision"), "grants", ["State-Read"]), "grants")
neg("pairing-decision", "an approval without the phone's X25519 key", mut(p1("pairing-decision"), "phone.x25519"), "x25519")
neg("pairing-decision", "a receipt with a negative issuedAt", mut(p1("pairing-decision"), "receipt.issuedAt", -1), "issuedAt")
neg("pairing-decision", "a receipt signature of 85 characters", mut(p1("pairing-decision"), "receipt.signature", SIG64[:85]), "signature")
neg("pairing-decision", "a name of 61 characters", mut(p1("pairing-decision"), "name", "x" * 61), "name")
neg("pairing-decision-response", "state open is not an outcome", {"v": 1, "state": "open", "time": NOW}, "state")
neg("pairing-decision-response", "an approval without the device id", {"v": 1, "state": "approved", "time": NOW}, "deviceId")
neg("pairing-decision-response", "a denial that names a device", {"v": 1, "state": "denied", "deviceId": PHONEID, "time": NOW}, "deviceId")
neg("pairing-decision-response", "a device id that is a provider id", {"v": 1, "state": "approved", "deviceId": PROVID, "time": NOW}, "deviceId")
neg("pairing-decision-response", "no time", {"v": 1, "state": "denied"}, "time")
neg("pairing-state-response", "state approved is not for a reject", {"v": 1, "state": "approved", "time": NOW}, "state")
neg("pairing-state-response", "v 2", {"v": 2, "state": "open", "time": NOW}, "v")
neg("pairing-fetch-response", "a hold both granted and refused", dict(p1("pairing-fetch-response"), hold={"granted": True, "refused": True, "retryAfter": 2}), "hold")
neg("pairing-fetch-response", "a refused hold without retryAfter", dict(p1("pairing-fetch-response"), hold={"refused": True}), "hold")
neg("pairing-fetch-response", "an approval without the receipt", mut(p1("pairing-fetch-response", 1), "receipt"), "receipt")
neg("pairing-fetch-response", "an answered rendezvous without its MAC", mut(dict(p1("pairing-fetch-response"), state="answered"), "mac"), "mac")
neg("sealed-token-fixture", "a sealed token of 147 characters", mut(SEALED_FIXTURE, "opens.0.sealedToken", SEALED_FIXTURE["opens"][0]["sealedToken"][:147]), "sealedToken")
neg("sealed-token-fixture", "a plaintext length that is not a token's", mut(SEALED_FIXTURE, "opens.0.plaintextLength", 64), "plaintextLength")
neg("sealed-token-fixture", "a token written into the file", mut(SEALED_FIXTURE, "opens.0.token", "oaiyrt1.AQIDBAUGBwg.ICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8"), "token")
neg("sealed-token-fixture", "no wrongRecipient", mut(SEALED_FIXTURE, "wrongRecipient"), "wrongRecipient")
neg("sealed-token-fixture", "a refused box that is not base64url", mut(SEALED_FIXTURE, "refused.0.sealedToken", "not base64!"), "sealedToken")

# --- admission (4.14)
MOBILE_CLAIMS = vec["A4"]["inputs"]["claims"]
PLUGIN_CLAIMS = p1("admission-claims", 1)
neg("admission-claims", "a mobile claims set carrying plugin members", dict(MOBILE_CLAIMS, approvedPeerKeyThumbprints=[PHONE_TH]), "oneOf")
neg("admission-claims", "an unknown member", dict(MOBILE_CLAIMS, extra=1), "oneOf")
neg("admission-claims", "a wrong audience", dict(MOBILE_CLAIMS, aud="other"), "oneOf")
neg("admission-claims", "no dsk", mut(MOBILE_CLAIMS, "dsk"), "oneOf")
neg("admission-claims", "a jti of the wrong form", dict(MOBILE_CLAIMS, jti="adm_short"), "oneOf")
neg("admission-claims", "a plugin claims set with an empty roster", dict(PLUGIN_CLAIMS, approvedPeerKeyThumbprints=[]), "oneOf")
neg("admission-claims", "a plugin claims set with 17 thumbprints", dict(PLUGIN_CLAIMS, approvedPeerKeyThumbprints=sorted(b64u(hashlib.sha256(bytes([i])).digest()) for i in range(17))), "oneOf")
neg("admission-claims", "a plugin claims set with revision 0", dict(PLUGIN_CLAIMS, peerRosterRevision=0), "oneOf")
neg("admission-claims", "a role that is neither mobile nor plugin", dict(MOBILE_CLAIMS, role="admin"), "oneOf")
neg("admission-plugin-request", "a supportedTransports that is null", mut(PLUGIN_REQUEST, "supportedTransports", None), "supportedTransports")
neg("admission-plugin-request", "an empty supportedTransports", mut(PLUGIN_REQUEST, "supportedTransports", []), "supportedTransports")
neg("admission-plugin-request", "a websocket transport", mut(PLUGIN_REQUEST, "supportedTransports", ["websocket"]), "supportedTransports")
neg("admission-plugin-request", "an empty roster", mut(PLUGIN_REQUEST, "approvedPeerKeyThumbprints", []), "approvedPeerKeyThumbprints")
neg("admission-plugin-response", "desktopConnection (the plugin's decoder rejects it)", dict(PLUGIN_RESPONSE, desktopConnection={}), "desktopConnection")
neg("admission-plugin-response", "scopeCompatibility (the plugin's decoder rejects it)", dict(PLUGIN_RESPONSE, scopeCompatibility={}), "scopeCompatibility")
neg("admission-plugin-response", "no relay member", mut(PLUGIN_RESPONSE, "relay"), "relay")
neg("admission-plugin-response", "expiresIn 10 (the decoder needs more than its 10 second safety margin)", mut(PLUGIN_RESPONSE, "expiresIn", 10), "expiresIn")
neg("admission-plugin-response", "expiresIn 301", mut(PLUGIN_RESPONSE, "expiresIn", 301), "expiresIn")
neg("admission-plugin-response", "tokenType bearer in lower case", mut(PLUGIN_RESPONSE, "tokenType", "bearer"), "tokenType")
neg("admission-plugin-response", "a gatewayUrl on ws://", mut(PLUGIN_RESPONSE, "gatewayUrl", "ws://relay.example.com/v2/realtime"), "gatewayUrl")
neg("admission-plugin-response", "a stream url on http", mut(PLUGIN_RESPONSE, "relay.streamUrl", "http://relay.example.com/x"), "streamUrl")
neg("admission-plugin-response", "a relay mode other than poll", mut(PLUGIN_RESPONSE, "relay.mode", "sse"), "mode")
neg("admission-plugin-response", "an accessToken that is not an aokie-adm-v2 bearer", mut(PLUGIN_RESPONSE, "accessToken", "abc"), "accessToken")
neg("admission-mobile-response", "no device record (the phone decoder requires it)", mut(MOBILE_RESPONSE, "device"), "device")
neg("admission-mobile-response", "an extra member", dict(MOBILE_RESPONSE, desktopConnection={}), "desktopConnection")
neg("admission-mobile-response", "a role that is not mobile", mut(MOBILE_RESPONSE, "role", "plugin"), "role")
neg("admission-mobile-request", "a supportedTransports of an unknown value", mut(MOBILE_REQUEST, "supportedTransports", ["quic"]), "supportedTransports")
neg("admission-mobile-request", "a deviceId that is a provider id", mut(MOBILE_REQUEST, "deviceId", PROVID), "deviceId")
neg("ice-server", "a STUN entry carrying a credential", dict(STUN, username="1:x", credential="AAAAAAAAAAAAAAAAAAAAAAAAAAA="), "oneOf")
neg("ice-server", "a TURN entry without expiresAt", mut(TURN, "expiresAt"), "oneOf")
neg("ice-server", "a TURN username that is not expiry:subject", mut(TURN, "username", "nocolon"), "oneOf")
neg("ice-server", "a TURN credential that is not base64 of an HMAC-SHA1", mut(TURN, "credential", "short"), "oneOf")
neg("challenge", "both peer shapes at once", dict(CHALLENGE_PLUGIN, expectedPeerKeyThumbprint=DESK_TH), "oneOf")
neg("challenge", "neither peer shape", {k: v for k, v in CHALLENGE_PLUGIN.items() if k not in ("approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")}, "oneOf")
neg("challenge", "no challengeNonce", mut(CHALLENGE_PLUGIN, "challengeNonce"), "challengeNonce")
neg("challenge", "schemaVersion 3", mut(CHALLENGE_PLUGIN, "schemaVersion", 3), "schemaVersion")
neg("compat-frames-request", "65 frames", {"to": "plugin", "frames": [{}] * 65}, "frames")
neg("compat-frames-request", "an address that is neither plugin nor mobile:<thumbprint>", {"to": "phone", "frames": [{}]}, "to")
neg("compat-frames-request", "no frames", {"to": "plugin", "frames": []}, "frames")
neg("compat-frames-accepted", "accepted above 64", {"accepted": 65, "seq": 1, "time": NOW}, "accepted")
neg("compat-frames-page", "129 frames", {"lastSeq": 9, "time": NOW, "frames": [p1("compat-frames-page")["frames"][0]] * 129}, "frames")

# --- RL-07: the members the decoders are strict about, and the rules of the compatibility routes' answers
PAGE = p1("compat-frames-page")
neg("ice-server", "a STUN entry without username and credential (the plugin's decoder requires both members)", {"urls": STUN["urls"]}, "oneOf")
neg("ice-server", "a STUN entry without credential", mut(STUN, "credential"), "oneOf")
neg("ice-server", "a STUN entry with an expiresAt", dict(STUN, expiresAt=NOW + 600), "oneOf")
neg("ice-server", "a STUN and a TURN url in one entry (the decoders treat it as TURN)", dict(TURN, urls=STUN["urls"] + TURN["urls"]), "oneOf")
neg("ice-server", "a TURN entry without a credential", mut(TURN, "credential"), "oneOf")
neg("ice-server", "a TURN username of 513 characters", mut(TURN, "username", "1790000600:" + "a" * 502), "oneOf")
neg("ice-server", "a url that is neither stun nor turn", {"urls": ["https://stun.example.com"], "username": "", "credential": ""}, "oneOf")
neg("ice-server", "nine urls in one entry", dict(STUN, urls=["stun:s%d.example.com" % i for i in range(9)]), "oneOf")
neg("ice-server", "an unknown member", dict(TURN, realm="example"), "oneOf")
neg("compat-error", "a retryAfter member (the phone's error decoder refuses a fourth member)", {"error": True, "code": "rate_limited", "message": "x", "retryAfter": 5}, "retryAfter")
neg("compat-error", "the native nesting", {"error": {"code": "rate_limited", "message": "x"}}, "error")
neg("compat-error", "an empty message", {"error": True, "code": "rate_limited", "message": ""}, "message")
neg("compat-frames-accepted", "no time", {"accepted": 1, "seq": 9}, "time")
neg("compat-frames-accepted", "nothing accepted", {"accepted": 0, "seq": 9, "time": NOW}, "accepted")
neg("compat-frames-accepted", "an extra member", {"accepted": 1, "seq": 9, "time": NOW, "ok": True}, "ok")
neg("compat-frames-page", "no time", mut(PAGE, "time"), "time")
neg("compat-frames-page", "a negative lastSeq", mut(PAGE, "lastSeq", -1), "lastSeq")
neg("compat-frames-page", "an unknown member", dict(PAGE, more=True), "more")
neg("compat-frames-page", "a frame that has no subjectId", mut(PAGE, "frames.0.subjectId"), "subjectId")
neg("compat-frames-page", "a frame that is an array", mut(PAGE, "frames.0.frame", [1]), "frame")
neg("compat-frames-page", "a frame element with an extra member", mut(PAGE, "frames.0.extra", 1), "extra")
neg("compat-frames-page", "a hold that is both granted and refused", dict(PAGE, hold={"granted": True, "refused": True, "retryAfter": 2}), "hold")
neg("compat-stream-frame", "from is a bare role", mut(FRAME, "from", "phone"), "from")
neg("compat-stream-frame", "from names a thumbprint of 42 characters", mut(FRAME, "from", "mobile:" + "A" * 42), "from")
neg("compat-stream-frame", "seq 0", mut(FRAME, "seq", 0), "seq")
neg("compat-stream-frame", "no frame", mut(FRAME, "frame"), "frame")
neg("compat-stream-frame", "a frame that is a string", mut(FRAME, "frame", "x"), "frame")
neg("compat-stream-frame", "a subjectId with a space", mut(FRAME, "subjectId", "a b"), "subjectId")
neg("compat-stream-frame", "seventeen grants", mut(FRAME, "grants", ["state_read"] * 17), "grants")
neg("compat-stream-frame", "a grant that is not a grant name at all", mut(FRAME, "grants", ["state_read", "Delete All"]), "grants")
neg("compat-stream-frame", "a grant twice", mut(FRAME, "grants", ["state_read", "state_read"]), "grants")
neg("compat-stream-frame", "an extra member", mut(FRAME, "id", "x"), "id")
neg("admission-plugin-response", "a device record without a role", mut(PLUGIN_RESPONSE, "device", {"id": "aokie", "appId": "aokie", "subjectId": "aokie"}), "device")
neg("admission-plugin-response", "no endpointPublicKey", mut(PLUGIN_RESPONSE, "endpointPublicKey"), "endpointPublicKey")
neg("admission-plugin-response", "an empty roster", mut(PLUGIN_RESPONSE, "approvedPeerKeyThumbprints", []), "approvedPeerKeyThumbprints")
neg("admission-plugin-response", "no turnCredentialExpiresAt (a required member, null when there is no TURN)", mut(PLUGIN_RESPONSE, "turnCredentialExpiresAt"), "turnCredentialExpiresAt")
neg("admission-plugin-response", "a STUN entry without its empty credential members", mut(PLUGIN_RESPONSE, "iceServers", [{"urls": STUN["urls"]}, TURN]), "iceServers")
neg("admission-mobile-response", "a device record with an extra member (the phone's DeviceRecord is strict)", mut(MOBILE_RESPONSE, "device.email", "x"), "email")
neg("admission-mobile-response", "a device displayName of 121 characters", mut(MOBILE_RESPONSE, "device.displayName", "n" * 121), "displayName")
neg("admission-mobile-response", "an empty device displayName", mut(MOBILE_RESPONSE, "device.displayName", ""), "displayName")
neg("admission-mobile-response", "no scopes", mut(MOBILE_RESPONSE, "scopes", []), "scopes")
neg("admission-mobile-response", "a device record without lastSeenAt", mut(MOBILE_RESPONSE, "device.lastSeenAt"), "lastSeenAt")
neg("admission-mobile-response", "expiresIn 0", mut(MOBILE_RESPONSE, "expiresIn", 0), "expiresIn")
neg("admission-mobile-response", "expiresIn 301", mut(MOBILE_RESPONSE, "expiresIn", 301), "expiresIn")
neg("challenge", "an expiresAt that is a string", mut(CHALLENGE_PLUGIN, "expiresAt", str(NOW + 25)), "expiresAt")
neg("challenge", "a connectionId in capitals", mut(CHALLENGE_PLUGIN, "connectionId", "relay_" + "A" * 32), "connectionId")
neg("challenge", "an admissionJti that is short", mut(CHALLENGE_PLUGIN, "admissionJti", "adm_1"), "admissionJti")
neg("challenge", "a challengeNonce with another prefix", mut(CHALLENGE_PLUGIN, "challengeNonce", "nonce_" + "1" * 32), "challengeNonce")
neg("challenge", "the plugin's roster hash as a thumbprint of 42 characters", mut(CHALLENGE_PLUGIN, "peerRosterHash", "A" * 42), "peerRosterHash")
neg("challenge", "role admin", mut(CHALLENGE_PLUGIN, "role", "admin"), "role")

# --- ring (4.15.1)
neg("ring", "an extra member (the Android parser forbids it)", dict(RING, extra="x"), "oneOf")
neg("ring", "a non-string value (callEpoch as an integer)", dict(RING, callEpoch=7), "oneOf")
neg("ring", "an unknown class", dict(RING, aokieClass="voice_shout"), "oneOf")
neg("ring", "no offerId", mut(RING, "offerId"), "oneOf")
neg("ring", "callEpoch 0", dict(RING, callEpoch="0"), "oneOf")
neg("ring", "schemaVersion 2", dict(RING, schemaVersion="2"), "oneOf")
neg("ring", "a cancel with an empty reason", {"aokieClass": "voice_offer_cancel", "schemaVersion": "1", "eventId": "e", "offerId": "o", "reason": ""}, "oneOf")
neg("ring", "an informational title of 81 characters", {"aokieClass": "informational", "schemaVersion": "1", "eventId": "e", "title": "t" * 81, "body": "b", "expiresAt": "1"}, "oneOf")
neg("ring", "an informational body of 241 characters", {"aokieClass": "informational", "schemaVersion": "1", "eventId": "e", "title": "t", "body": "b" * 241, "expiresAt": "1"}, "oneOf")
neg("ring", "an expiresAt that is not digits", dict(RING, expiresAt="soon"), "oneOf")

# --- authority envelopes (4.12) and rotation (4.11)
neg("command", "a dep member (reserved, refused by a v1 verifier)", dict(COMMAND, dep="cmd-0"), "False schema")
neg("command", "v 2", mut(COMMAND, "v", 2), "v")
neg("command", "dev that is a provider id", mut(COMMAND, "dev", PROVID), "dev")
neg("command", "a payload that is an array", mut(COMMAND, "payload", [1]), "payload")
neg("command", "no idem", mut(COMMAND, "idem"), "idem")
neg("command", "src that is not a provider id", mut(COMMAND, "src", DEVID), "src")
neg("command", "iat as a string", mut(COMMAND, "iat", "1790000000"), "iat")
neg("result", "done carrying an error", dict(RESULT_OK, error={"code": "x"}), "error")
neg("result", "failed with no error", dict(RESULT_OK, status="failed", result=None, error=None), "error")
neg("result", "an unknown status", dict(RESULT_OK, status="ok"), "status")
neg("result", "re that is not an id", dict(RESULT_OK, re="a b"), "re")
neg("container", "an extra member", dict(CONTAINER, x="1"), "x")
neg("container", "no signature", mut(CONTAINER, "s"), "s")
neg("container", "a signature of the wrong length", mut(CONTAINER, "s", SIG64[:80]), "s")
neg("container", "a key thumbprint of 42 characters", mut(CONTAINER, "k", CONTAINER["k"][:42]), "k")
neg("container", "empty signed bytes", mut(CONTAINER, "b", ""), "b")
neg("container", "padded base64 in b", mut(CONTAINER, "b", CONTAINER["b"] + "=="), "b")
neg("rotation-statement", "serial 0", mut(ROT, "serial", 0), "serial")
neg("rotation-statement", "an extra member", dict(ROT, extra=1), "extra")
neg("rotation-statement", "no prev (the first draft's statement was replayable)", mut(ROT, "prev"), "prev")
neg("rotation-statement", "a new key without a thumbprint", mut(ROT, "new.thumbprint"), "thumbprint")

# --- tickets (4.9.2)
neg("ticket-header", "alg none", mut(TICKET_H, "alg", "none"), "alg")
neg("ticket-header", "alg HS256", mut(TICKET_H, "alg", "HS256"), "alg")
neg("ticket-header", "alg RS256", mut(TICKET_H, "alg", "RS256"), "alg")
neg("ticket-header", "a crit header", dict(TICKET_H, crit=["b64"]), "crit")
neg("ticket-header", "a jku header", dict(TICKET_H, jku="https://evil.example/keys"), "jku")
neg("ticket-header", "a jwk header", dict(TICKET_H, jwk={}), "jwk")
neg("ticket-header", "an x5u header", dict(TICKET_H, x5u="https://evil.example"), "x5u")
neg("ticket-header", "an x5c header", dict(TICKET_H, x5c=[]), "x5c")
neg("ticket-header", "typ JWT", mut(TICKET_H, "typ", "JWT"), "typ")
neg("ticket-header", "no kid", mut(TICKET_H, "kid"), "kid")
neg("ticket-claims", "lane cmd", mut(TICKET_C, "lane", "cmd"), "lane")
neg("ticket-claims", "a sub of 65 characters", mut(TICKET_C, "sub", "s" * 65), "sub")
neg("ticket-claims", "an org with a path", mut(TICKET_C, "org", "https://app.example.com/x"), "org")
neg("ticket-claims", "aud that is not a relay id", mut(TICKET_C, "aud", DEVID), "aud")
neg("ticket-claims", "iss that is a device id", mut(TICKET_C, "iss", DEVID), "iss")
neg("ticket-claims", "eph of 42 characters", mut(TICKET_C, "eph", TICKET_C["eph"][:42]), "eph")
neg("ticket-claims", "an extra member", dict(TICKET_C, extra=1), "extra")
neg("ticket-claims", "no jti", mut(TICKET_C, "jti"), "jti")
neg("ctl", "a relay.notice of 201 characters", dict(CTL["relay.notice"], message="m" * 201), "message")
neg("ctl", "a relay.notice level error", dict(CTL["relay.notice"], level="error"), "level")
neg("ctl", "a device.revoked naming a provider", dict(CTL["device.revoked"], id=PROVID), "id")
neg("ctl", "token.age days as a string", dict(CTL["token.age"], days="95"), "days")
neg("ctl", "no t", {"id": PROVID}, "t")
neg("ctl", "provider.rotated without a signature", {"t": "provider.rotated", "b": "AAAA"}, "s")

# --- reply boxes, admin (4.13, 4.18.7)
neg("replybox-request", "a rid of the wrong length", mut(p1("replybox-request"), "rid", "short"), "rid")
neg("replybox-request", "ttl above 900", mut(p1("replybox-request"), "ttl", 901), "ttl")
neg("rbx-item-request", "lane cmd through a reply box", mut(p1("rbx-item-request"), "lane", "cmd"), "lane")
neg("rbx-item-request", "hdr.ct sealed1 (a reply box carries tunnel1)", mut(p1("rbx-item-request"), "hdr.ct", "sealed1"), "ct")
neg("replybox-response", "no rid", mut(p1("replybox-response"), "rid"), "rid")
neg("admin-capacity-request", "zero workers", mut(p1("admin-capacity-request"), "workers", 0), "workers")
neg("admin-capacity-request", "streamOk as a string", mut(p1("admin-capacity-request"), "streamOk", "yes"), "streamOk")
neg("admin-capacity-response", "laneBodies naming a lane that is not a lane", mut(p1("admin-capacity-response"), "effective.laneBodies.chat", 1), "chat")
neg("admin-status", "no warnings member", mut(STATUS, "warnings"), "warnings")
neg("admin-status", "an unknown db driver", mut(STATUS, "db.driver", "oracle"), "driver")

# --- identifiers straight from common
for d, bad, why in [
        ("thumbprint", "A" * 42, "42 characters"), ("thumbprint", "A" * 44, "44 characters"), ("thumbprint", "A" * 42 + "=", "padding"),
        ("thumbprint", "A" * 42 + "+", "standard base64 character"), ("deviceId", "prov-" + "A" * 22, "provider prefix"),
        ("deviceId", "dev-" + "A" * 21, "short"), ("deviceId", "dev-" + "A" * 23, "long"), ("deviceId", "DEV-" + "A" * 22, "upper-case prefix"),
        ("providerId", "dev-" + "A" * 22, "device prefix"), ("relayId", "rly-" + "A" * 21, "short"),
        ("itemId", "a/b", "slash"), ("itemId", "..\\x", "backslash"), ("itemId", "a\u0000b", "NUL"), ("itemId", "x" * 129, "129 characters"),
        ("itemId", ".", "a single dot is a path component"), ("itemId", "..", "two dots are a path component"),
        ("appId", "a" * 65, "65 characters"), ("appId", "", "empty"), ("appId", "a b", "space"), ("appId", "a/b", "slash"),
        ("token", TOKEN[:-1], "62 characters"), ("token", TOKEN + "A", "64 characters"), ("token", "oaiyrt2." + TOKEN[8:], "prefix"),
        ("token", TOKEN.replace(".", ":"), "no dots"), ("token", " " + TOKEN, "leading space"),
        ("mailbox", "dev:" + DEVID + "/x", "suffix on a device inbox"), ("mailbox", "app:aokie@" + DEVID + "/mobile:short", "short thumbprint"),
        ("mailbox", "app:aokie@" + DEVID + "/other", "unknown party"), ("mailbox", "rbx:short", "short rid"),
        ("mailbox", "app:" + "a" * 65 + "@" + DEVID + "/plugin", "appId over 64"),
        ("unixTime", -1, "negative"), ("unixTime", 2 ** 53, "2^53"), ("unixTime", 1.5, "fractional"), ("unixTime", "1", "string"),
        ("hdr", {"x": 1}, "unknown key"), ("hdr", {"ct": "text", "prio": True}, "prio as boolean"),
        ("jws", "a.b", "two segments"), ("jws", "a.b.c.d", "four segments"), ("jws", "a.b.c=", "padding"),
        ("typedCode", "M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RI", "I is not in the alphabet"), ("typedCode", "M6SC7N75YR3HGA9T9DE6TZMFJ0RW", "no dashes"),
        ("sasCode", "6NHN-K68M-QQVZ", "no check character"),
        ("pairingUri", "oaiy://pair?v=2&u=x&s=" + "A" * 22, "v not 3"), ("pairingUri", "https://x?v=3&s=" + "A" * 22, "wrong scheme"),
        ("enrollUri", vec["A7"]["expected"]["uri"].replace("&r=desktop", "&r=phone"), "role phone"),
        ("origin", "http://app.example.com", "http"), ("origin", "https://app.example.com/", "trailing slash"),
        ("origin", "https://user@app.example.com", "userinfo"), ("publicUrl", "https://relay.example.com/", "trailing slash"),
        ("epoch", "A" * 12, "12 characters"), ("errorCode", "oops", "unknown code"), ("lane", "flow.out", "reserved lane"),
        ("seq", 0, "zero"), ("etag", "AQIDBAUGBwgJCgsMDQ4PEA", "unquoted"), ("slotName", "-a", "leading dash"),
        ("from", "phone", "a bare role"), ("inboxAddress", "app:aokie@" + DEVID + "/plugin", "party address"),
        ("name60", "x" * 61, "61 characters"), ("kid", "A" * 12, "12 characters"),
]:
    neg("common#" + d, f"{d}: {why}", bad, "")

section("invalid documents are REJECTED, for the reason the label names")
n_neg_schema = 0
covered: set[str] = set()
for schema_name, label, doc, hint in NEG:
    n_neg_schema += 1
    covered.add(schema_name)
    errs = problems(schema_name, doc)
    accepted = not errs
    if accepted:
        ok(f"{schema_name}: {label}", False, "ACCEPTED a document it must reject")
        continue
    trail = error_trail(errs)
    ok(f"{schema_name}: {label}", (hint in trail) if hint else True, f"rejected, but not for '{hint}':\n{trail[:300]}")
for name in schemas:
    stem = name.replace(".schema.json", "")
    if stem != "common":
        ok(f"{stem} has at least one negative document", stem in covered)
print(f"\n  {n_neg_schema} negative documents checked by schema")

# ---------------------------------------------------------------------------
section("rules JSON Schema cannot count: reference rules reject these")
RULES: list[tuple[str, object]] = []


def rule(label: str, rejected) -> None:
    RULES.append((label, rejected))


def raises(fn) -> bool:
    try:
        fn()
        return False
    except Refused:
        return True


rule("canonical form: a float (0.5) is refused", lambda: raises(lambda: canonical_json('{"n":0.5}')))
rule("canonical form: 1.0 is a float even though integral", lambda: raises(lambda: canonical_json('{"n":1.0}')))
rule("canonical form: an exponent is refused", lambda: raises(lambda: canonical_json('{"n":1e2}')))
rule("canonical form: a float inside an array inside a claim is refused", lambda: raises(lambda: canonical_json(json.dumps(CLAIMS).replace('"issuedAt": 1790000030', '"issuedAt": 1790000030.0'))))
rule("canonical form: NaN is refused", lambda: raises(lambda: canonical_json('{"n":NaN}')))
rule("canonical form: an integer above 2^64-1 is refused", lambda: raises(lambda: canonical_json('{"n":18446744073709551616}')))
rule("canonical form: an integer below -2^63 is refused", lambda: raises(lambda: canonical_json('{"n":-9223372036854775809}')))
rule("canonical form: -0 is refused (every integer has exactly one spelling)", lambda: raises(lambda: canonical_json('{"n":-0}')))
rule("canonical form: -0 inside an array is refused", lambda: raises(lambda: canonical_json('{"a":[1,-0,3]}')))
rule("SAS: pid read as its 22-character b64u text (a 41-byte info) does not give the SAS of A3", lambda: sas_of_pid_reading(sni["pidB64u"].encode())[4] != A3["expected"]["sasDisplay"])
rule("SAS: pid read as its 32-character hex text (a 51-byte info) does not give the SAS of A3", lambda: sas_of_pid_reading(sn_pid.hex().encode())[4] != A3["expected"]["sasDisplay"])
rule("SAS: only the raw 16 bytes of pid (a 35-byte info) give the SAS of A3", lambda: sas_of_pid_reading(sn_pid)[4] == A3["expected"]["sasDisplay"] and sas_of_pid_reading(sn_pid)[1] == 35)
rule("hdr: 513 serialised bytes is over the 512 byte cap", lambda: hdr_bytes({"pad": "x" * 503}) == 513 > 512)
rule("hdr: the serialised size counts UTF-8 bytes, not characters",
     lambda: hdr_bytes({"n": "é" * 300}) > 512 and len(json.dumps({"n": "é" * 300}, ensure_ascii=False)) < 512)
rule("roster: unsorted thumbprints are refused (not strictly ascending)", lambda: not strictly_ascending([PHONE_TH, DESK_TH] if PHONE_TH > DESK_TH else [DESK_TH, PHONE_TH]))
rule("roster: duplicates are refused (not strictly ascending)", lambda: not strictly_ascending(sorted([PHONE_TH, PHONE_TH])))
rule("roster: bytewise order puts '-' (0x2d) before 'A' (0x41)", lambda: not strictly_ascending(["A" * 43, "-" + "A" * 42]))
rule("roster: 17 thumbprints exceed rosterMax (16)", lambda: len(sorted(b64u(hashlib.sha256(bytes([i])).digest()) for i in range(17))) > 16)
rule("pairing response: expiresAt = issuedAt + 121 is refused", lambda: not pairing_response_window_ok(mut(CLAIMS, "expiresAt", CLAIMS["issuedAt"] + 121)))
rule("pairing response: expiresAt = issuedAt is refused", lambda: not pairing_response_window_ok(mut(CLAIMS, "expiresAt", CLAIMS["issuedAt"])))
rule("pairing response: expiresAt before issuedAt is refused", lambda: not pairing_response_window_ok(mut(CLAIMS, "expiresAt", CLAIMS["issuedAt"] - 5)))
rule("pairing offer: 601 seconds is refused", lambda: not pairing_offer_window_ok(mut(OFFER, "expiresAt", OFFER["issuedAt"] + 601)))
rule("pairing offer: 599 seconds is refused", lambda: not pairing_offer_window_ok(mut(OFFER, "expiresAt", OFFER["issuedAt"] + 599)))
rule("ticket: exp - iat = 301 is refused", lambda: not signed_window_ok(mut(TICKET_C, "exp", TICKET_C["iat"] + 301)))
rule("ticket: exp before iat is refused", lambda: not signed_window_ok(mut(TICKET_C, "exp", TICKET_C["iat"] - 1)))
rule("command: exp - iat = 301 is refused", lambda: not signed_window_ok(mut(COMMAND, "exp", COMMAND["iat"] + 301)))
rule("rotation statement: exp - iat above 24 hours is refused", lambda: not rotation_window_ok(mut(ROT, "exp", ROT["iat"] + 86401)))
rule("ring: a voice offer expiring at now + 301 is refused", lambda: not ring_window_ok(mut(RING, "expiresAt", str(NOW + 301)), NOW))
rule("ring: a voice offer that has already expired is refused", lambda: not ring_window_ok(mut(RING, "expiresAt", str(NOW)), NOW))
rule("ring: an informational notice expiring at now + 86401 is refused",
     lambda: not ring_window_ok({"aokieClass": "informational", "expiresAt": str(NOW + 86401)}, NOW))
rule("ctl: a body over 4096 bytes is refused", lambda: len(json.dumps(dict(CTL["relay.notice"], message="m" * 200, pad="x" * 4000)).encode()) > 4096)
rule("ring: the A11 body is signed as posted (a re-serialised body does not verify)",
     lambda: not ed_verify(pub["host"], b"oaiy/relay/1/ring\x00" + json.dumps(RING, indent=1).encode(), unb64u_strict(vec["A11"]["expected"]["hdrSig"])))
rule("ring: a ring signature does not verify under the command domain",
     lambda: not ed_verify(pub["host"], b"oaiy/relay/1/cmd\x00" + vec["A11"]["expected"]["bodyText"].encode(), unb64u_strict(vec["A11"]["expected"]["hdrSig"])))
rule("info proof: a nonce of 15 bytes is refused", lambda: not (16 <= len(unb64u_strict(b64u(bytes(15)))) <= 32))
rule("info proof: a nonce of 33 bytes is refused", lambda: not (16 <= len(unb64u_strict(b64u(bytes(33)))) <= 32))
rule("info proof: a nonce of 16 bytes is accepted", lambda: 16 <= len(unb64u_strict(b64u(bytes(16)))) <= 32)
rule("info proof: the copied static signature is not the proof", lambda: vec["A6"]["expected"]["proof"] != vec["A6"]["expected"]["staticSignature"])
rule("enrolment: a proof over the body with one extra space does not verify", lambda: not ed_verify(ed_pub(seed7), b"oaiy/relay/1/enroll\x00" + body7.replace('"kid":', '"kid": ').encode(), proof7))
rule("enrolment: a proof under another domain does not verify", lambda: not ed_verify(ed_pub(seed7), b"oaiy/relay/1/cmd\x00" + body7.encode(), proof7))
rule("JWS: the signature covers the transmitted bytes, so a re-encoded header does not verify",
     lambda: not ed_verify(pub["provider"], (b64u(json.dumps(TICKET_H, indent=1).encode()) + "." + b64u(json.dumps(TICKET_C, separators=(",", ":")).encode())).encode(), unb64u_strict(vec["A9"]["expected"]["signature"])))
rule("command: a signature made for one command does not verify for another id",
     lambda: not ed_verify(pub["provider"], b"oaiy/relay/1/cmd\x00" + json.dumps(dict(COMMAND, id="cmd-0002"), separators=(",", ":")).encode(), unb64u_strict(vec["A8"]["expected"]["signature"])))
rule("pairing: an offer text altered by one byte fails its MAC",
     lambda: b64u(hmac_sha256(mac3, b"oaiy/pairing/3/offer-mac\x00" + (offer_text + " ").encode())) != vec["A3"]["expected"]["offerMac"])
rule("pairing: the response MAC is not valid for the offer text (domain separation)",
     lambda: b64u(hmac_sha256(mac3, b"oaiy/pairing/3/response-mac\x00" + offer_text.encode())) != vec["A3"]["expected"]["offerMac"])
rule("admission: a token whose payload is altered fails its HMAC",
     lambda: admission_token(bytes.fromhex(a4["inputs"]["secretHex"]), dict(a4["inputs"]["claims"], role="plugin")) != a4["expected"]["token"])
rule("admission: the plugin bearer with one phone is 964 characters, 92 more per phone",
     lambda: vec["A4b"]["expected"]["lengthByPhones"]["3"] - vec["A4b"]["expected"]["lengthByPhones"]["1"] == 2 * 92)
rule("admission: sixteen phones fit under an 8 KB header line, sixty-four do not fit under 4 KB",
     lambda: vec["A4b"]["expected"]["lengthByPhones"]["16"] < 4096 <= vec["A4b"]["expected"]["lengthByPhones"]["64"])
for name, h in a12["inputs"]["encodings"].items():
    rule(f"X25519 peer {name} is refused", (lambda hh: lambda: is_small_order_x25519(bytes.fromhex(hh)))(h))
    rule(f"X25519 peer {name} with bit 255 set is refused", (lambda hh: lambda: is_small_order_x25519(bytes.fromhex(a12['inputs']['withBit255'][name])))(h))
rule("typed code: a wrong check character is a typo (rejected locally)", lambda: not typed_code_ok(A3["expected"]["typedCode"][:-1] + "0"))
rule("typed code: a wrong check character in the first check position is rejected", lambda: not typed_code_ok(A3["expected"]["typedCode"].replace("J0RW", "K0RW")))
rule("typed code: 27 characters is incomplete", lambda: not typed_code_ok(A3["expected"]["typedCode"][:-1]))
rule("typed code: U is refused by normalisation", lambda: normalise_typed("M6SC-7N75-YR3H-GA9T-9DE6-TZMF-J0RU") is None)
rule("typed code: non-zero padding bits in the 26th character are refused", lambda: not typed_code_ok(A3["expected"]["typedCode"].replace("-", "")[:25] + "1" + A3["expected"]["typedCode"].replace("-", "")[26:]))
rule("SAS: a wrong check character is caught", lambda: not sas_ok(A3["expected"]["sas12"] + ("0" if A3["expected"]["sasCheckChar"] != "0" else "1")))
rule("SAS: a transposed pair is caught (almost always)", lambda: not sas_ok(A3["expected"]["sasDisplay"].replace("6NHN", "N6HN")))
for t in ex["tokens"]["invalid"]:
    rule(f"token: {t['reason']}", (lambda tt: lambda: raises(lambda: parse_token(tt)))(t["token"]))
rule("item ids: a path-like id never reaches a file name (ids are matched by pattern)", lambda: validate("common#itemId", "../../data/relay.sqlite") != [])
rule("item ids: NUL byte", lambda: validate("common#itemId", "a\u0000b") != [])
rule("item ids: . and .. match the character class and are refused all the same", lambda: validate("common#itemId", ".") != [] and validate("common#itemId", "..") != [])

# --- the rendezvous state machine (README 10.1): the reference function, and what it must refuse
class Conflict(ValueError):
    pass


def pairing_next(state: str, event: str, rejects: int = 0, responses: int = 0) -> str:
    """One step of the relay-side state machine: open -> answered -> approved | denied; answered -> open by a reject (the
    third ends it); open | answered | denied -> expired by a burn or at exp. Anything else is a Conflict."""
    if state == "expired":
        raise Conflict("gone")
    if event == "response" and state == "open" and responses < 3:
        return "answered"
    if event == "reject" and state == "answered":
        return "open" if rejects + 1 < 3 else "expired"
    if event == "approve" and state == "answered":
        return "approved"
    if event == "deny" and state == "answered":
        return "denied"
    if event == "burn" and state in ("open", "answered", "denied"):
        return "expired"
    raise Conflict(f"{event} in {state}")


def refused_step(state: str, event: str, **kw) -> bool:
    try:
        pairing_next(state, event, **kw)
        return False
    except Conflict:
        return True


def pair_item_id(pid: str, n: int) -> str:
    """The id of the pair item of the n-th accepted response: the pid itself for the first, pid.n after a reject."""
    return pid if n == 1 else f"{pid}.{n}"


PID3 = A3["expected"]["pid"]
rule("pairing states: a response opens the answered state, a reject reopens it, the third reject ends it",
     lambda: (pairing_next("open", "response"), pairing_next("answered", "reject", rejects=0), pairing_next("answered", "reject", rejects=1),
              pairing_next("answered", "reject", rejects=2)) == ("answered", "open", "open", "expired"))
rule("pairing states: a second response while answered is refused (already_answered)", lambda: refused_step("answered", "response"))
rule("pairing states: a response after an approval is refused", lambda: refused_step("approved", "response"))
rule("pairing states: a response after a denial is refused", lambda: refused_step("denied", "response"))
rule("pairing states: a fourth response is refused whatever the state (three are accepted)", lambda: refused_step("open", "response", responses=3))
rule("pairing states: an approval of an open rendezvous (nobody answered) is refused", lambda: refused_step("open", "approve"))
rule("pairing states: a denial of an open rendezvous is refused", lambda: refused_step("open", "deny"))
rule("pairing states: a reject of an open rendezvous is refused", lambda: refused_step("open", "reject"))
rule("pairing states: an approval after a denial is refused", lambda: refused_step("denied", "approve"))
rule("pairing states: a reject after an approval is refused", lambda: refused_step("approved", "reject"))
rule("pairing states: an approved rendezvous is not burned (its phone has yet to read the token)", lambda: refused_step("approved", "burn"))
rule("pairing states: nothing happens to an expired rendezvous", lambda: all(refused_step("expired", e) for e in ("response", "reject", "approve", "deny", "burn")))
rule("pairing states: the state after a burn is expired, from open, answered and denied",
     lambda: all(pairing_next(s, "burn") == "expired" for s in ("open", "answered", "denied")))
rule("pairing: the item ids of three responses are all different and the first is the pid",
     lambda: [pair_item_id(PID3, n) for n in (1, 2, 3)] == [PID3, PID3 + ".2", PID3 + ".3"] and len({pair_item_id(PID3, n) for n in (1, 2, 3)}) == 3)
rule("pairing: the receipt document lists the grants sorted, so an unsorted list is another text and another signature",
     lambda: canonical_json(json.dumps(dict(A3["inputs"]["receiptDocument"], grants=list(reversed(A3["inputs"]["receiptDocument"]["grants"]))))) != A3["expected"]["receiptText"])
rule("pairing: a receipt signature does not verify for another pid",
     lambda: not ed_verify(pub["desktopEndpoint"], b"oaiy/pairing/3/approval\x00" + canonical_json(json.dumps(dict(A3["inputs"]["receiptDocument"], pid=b64u(bytes(16))))).encode(), unb64u_strict(A3["expected"]["receiptSignature"])))
rule("pairing: a receipt signature does not verify under the response domain",
     lambda: not ed_verify(pub["desktopEndpoint"], b"oaiy/pairing/3/response\x00" + A3["expected"]["receiptText"].encode(), unb64u_strict(A3["expected"]["receiptSignature"])))

n_rules = 0
for label, fn in RULES:
    n_rules += 1
    ok(f"rule: {label}", bool(fn()))
print(f"\n  {n_rules} rule-level negatives (reference rules)")

# ---------------------------------------------------------------------------
section("independent Node re-computation (verify_vectors.mjs)")
node = shutil.which("node")
node_total = 0
if node is None:
    print("  !! node is not on PATH: the independent re-computation was NOT run")
    ok("node re-computation ran", False, "install Node 18+ so the second implementation can check the vectors")
else:
    proc = subprocess.run([node, str(V1 / "verify_vectors.mjs")], capture_output=True, text=True, encoding="utf-8", timeout=120)
    tail = proc.stdout.strip().splitlines()
    m = re.search(r"(\d+) checks, (\d+) mismatches", proc.stdout)
    node_total = int(m.group(1)) if m else 0
    ok(f"node re-computation agrees ({node_total} checks)", proc.returncode == 0 and bool(m) and m.group(2) == "0", "\n".join(tail[-8:]) + proc.stderr[-300:])
    ok("node re-computation ran at least 217 checks (a checker that was silently thinned fails here)", node_total >= 217, str(node_total))

section("recorded fixtures (fixtures/): sealed tokens and the pairing ceremony, read by two independent implementations")
FIX = V1 / "fixtures"
CEREMONY = json.loads((FIX / "pairing-ceremony.json").read_text(encoding="utf-8"))
STEP_SCHEMAS = [("pairing-create-request", "pairing-create-response"), (None, "pairing-fetch-response"), ("pairing-answer-request", "pairing-answer-response"),
                (None, "poll-response"), ("pairing-decision", "pairing-decision-response"), (None, "pairing-fetch-response")]
n_fix_docs = 0
for i, (step, (req_schema, res_schema)) in enumerate(zip(CEREMONY["steps"], STEP_SCHEMAS)):
    for schema_name, body in ((req_schema, step["request"].get("body")), (res_schema, step["response"]["body"])):
        if schema_name is None or body is None:
            continue
        n_fix_docs += 1
        errs = problems(schema_name, body)
        ok(f"ceremony step {i} ({step['step']}) validates against {schema_name}", not errs, errs[0].message[:150] if errs else "")
print(f"\n  {n_fix_docs} documents of the recorded ceremony validated")
ok("the ceremony's pair item body is the response text the phone posted, byte for byte",
   CEREMONY["steps"][3]["response"]["body"]["items"][0]["body"] == CEREMONY["steps"][2]["request"]["body"]["response"])
ok("the ceremony's offer is Appendix A3's 778 byte text", CEREMONY["steps"][0]["request"]["body"]["offer"] == A3["expected"]["offerText"])
ok("no device token is written into any fixture file",
   all(re.search(r"oaiyrt1\.[A-Za-z0-9_-]{11}\.[A-Za-z0-9_-]{43}", p.read_text(encoding="utf-8")) is None for p in FIX.rglob("*.json")),
   "a token-shaped string was found")
fix_readme = (FIX / "README.md").read_text(encoding="utf-8")
for f in ("sealed-token.json", "pairing-ceremony.json", "verify_fixtures.py", "verify_fixtures.mjs"):
    ok(f"fixtures/README.md describes {f}", f"`{f}`" in fix_readme)
fix_checks = 0
for label, cmd in (("Python (no libsodium)", [sys.executable, str(FIX / "verify_fixtures.py")]), ("Node (no libsodium)", [shutil.which("node") or "node", str(FIX / "verify_fixtures.mjs")])):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=180)
        m = re.search(r"(\d+) checks, (\d+) mismatches", proc.stdout)
        n = int(m.group(1)) if m else 0
        fix_checks += n
        ok(f"independent reading of the fixtures in {label} agrees ({n} checks)", proc.returncode == 0 and bool(m) and m.group(2) == "0", (proc.stdout + proc.stderr)[-400:])
    except (OSError, subprocess.TimeoutExpired) as e:
        ok(f"independent reading of the fixtures in {label} ran", False, str(e))

try:
    proc = subprocess.run([sys.executable, str(FIX / "selftest_fixtures.py")], capture_output=True, text=True, encoding="utf-8", timeout=300)
    m = re.search(r"(\d+) damaged copies, each refused by", proc.stdout)
    n = int(m.group(1)) if m else 0
    ok(f"the two readers of the pairing fixtures are not vacuous: each of {n} damaged copies is refused by both", proc.returncode == 0 and n >= 10, (proc.stdout + proc.stderr)[-500:])
except (OSError, subprocess.TimeoutExpired) as e:
    ok("the self-test of the pairing fixture readers ran", False, str(e))
for f in ("sealed-token.json", "selftest_fixtures.py"):
    ok(f"fixtures/README.md describes {f}", f"`{f}`" in fix_readme)

section("recorded Aokie fixtures (fixtures/aokie/): admissions, challenges, frames, streams, errors and ICE, read by the decoders' rules")
AOK = FIX / "aokie"
aok = {n: json.loads((AOK / n).read_text(encoding="utf-8")) for n in ("admission.json", "challenge.json", "frames.json", "stream.json", "errors.json", "ice.json")}
n_aok_docs = 0


def aok_valid(schema_name: str, body, label: str) -> None:
    global n_aok_docs
    n_aok_docs += 1
    errs = problems(schema_name, body)
    ok(f"aokie fixture {label} validates against {schema_name}", not errs, errs[0].message[:150] if errs else "")


for c in aok["admission.json"]["cases"]:
    plug = c["role"] == "plugin"
    aok_valid("admission-plugin-request" if plug else "admission-mobile-request", c["request"]["body"], f"admission request '{c['name']}'")
    aok_valid(("admission-plugin-response" if plug else "admission-mobile-response") if c["response"]["status"] == 200 else "compat-error", c["response"]["body"], f"admission answer '{c['name']}'")
    for s_ in (c["response"]["body"].get("iceServers", []) if c["response"]["status"] == 200 else []):
        aok_valid("ice-server", s_, f"ICE entry of '{c['name']}'")
for c in aok["challenge.json"]["cases"]:
    aok_valid("challenge", c["response"]["body"], f"challenge '{c['name']}'")
for st in aok["frames.json"]["steps"]:
    post = st["request"]["method"] == "POST"
    if post:
        aok_valid("compat-frames-request", st["request"]["body"], f"frames request '{st['step']}'")
    aok_valid("compat-frames-accepted" if post else "compat-frames-page", st["response"]["body"], f"frames answer '{st['step']}'")
for c in aok["stream.json"]["cases"]:
    events = [b for b in c["body"].split("\n\n") if b]
    for b in events:
        lines = dict(l.split(": ", 1) for l in b.split("\n") if ": " in l and not l.startswith(":"))
        if lines.get("event") == "frame":
            aok_valid("compat-stream-frame", json.loads(lines["data"]), f"stream event id {lines['id']} of '{c['name']}'")
        elif lines.get("event") == "end":
            ok(f"stream '{c['name']}': the end event's data is an empty object", lines["data"] == "{}")
for c in aok["errors.json"]["cases"]:
    if c["response"]["status"] >= 400:
        aok_valid("compat-error", c["response"]["body"], f"error '{c['name']}'")
    else:
        aok_valid("compat-frames-page", c["response"]["body"], f"answer '{c['name']}'")
for c in aok["ice.json"]["cases"]:
    for s_ in c["expected"]["iceServers"]:
        aok_valid("ice-server", s_, f"ICE entry of '{c['name']}'")
print(f"\n  {n_aok_docs} documents of the Aokie fixtures validated")
aok_readme = (AOK / "README.md").read_text(encoding="utf-8")
for f in ("admission.json", "challenge.json", "frames.json", "stream.json", "errors.json", "ice.json", "aokie_decoders.py", "verify_aokie_fixtures.py"):
    ok(f"fixtures/aokie/README.md describes {f}", f"`{f}`" in aok_readme)
ok("every admission of the Aokie fixtures carries the same three relay URLs (the plugin's cursor domain)",
   len({json.dumps({k: v for k, v in c["response"]["body"]["relay"].items() if k != "mode"}, sort_keys=True) for c in aok["admission.json"]["cases"] if c["response"]["status"] == 200}) == 1)
aok_neg = 0
try:
    proc = subprocess.run([sys.executable, str(AOK / "verify_aokie_fixtures.py")], capture_output=True, text=True, encoding="utf-8", timeout=180)
    m = re.search(r"(\d+) checks, (\d+) mismatches", proc.stdout)
    n = int(m.group(1)) if m else 0
    fix_checks += n
    ok(f"the decoders' rules, transcribed in Python, accept the Aokie fixtures and refuse every damaged copy ({n} checks)", proc.returncode == 0 and bool(m) and m.group(2) == "0", (proc.stdout + proc.stderr)[-500:])
    m2 = re.search(r"(\d+) damaged documents refused", proc.stdout)
    aok_neg = int(m2.group(1)) if m2 else 0
    ok("at least a hundred damaged copies of the Aokie fixtures were refused", aok_neg >= 100)
except (OSError, subprocess.TimeoutExpired) as e:
    ok("the Aokie fixture verifier ran", False, str(e))

section("vectors.json is what generate_vectors.py writes")
proc = subprocess.run([sys.executable, str(V1 / "generate_vectors.py"), "--check"], capture_output=True, text=True, encoding="utf-8", timeout=120)
ok("vectors.json is current", proc.returncode == 0, (proc.stdout + proc.stderr)[-300:])

# ---------------------------------------------------------------------------
section("README covers the surface")
readme = (V1 / "README.md").read_text(encoding="utf-8")
for name in schemas:
    stem = name.replace(".schema.json", "")
    ok(f"README mentions {stem}", f"`{stem}`" in readme or f"{stem}.schema.json" in readme)
for e in ex["errors"]:
    ok(f"README names the error code {e['code']}", f"`{e['code']}`" in readme)
for e in ex["compatErrors"]:
    ok(f"README names the compat error code {e['code']}", f"`{e['code']}`" in readme)
for ln in LANES:
    ok(f"README names the lane {ln}", f"`{ln}`" in readme)
for route in ("/v1/health", "/v1/info", "/v1/poll", "/v1/items", "/v1/slots", "/v1/presence", "/v1/devices", "/v1/roster", "/v1/tokens/rotate",
              "/v1/keys", "/v1/providers", "/v1/pair", "/v1/admission", "/v1/replyboxes", "/v1/rbx", "/v1/enroll", "/v1/admin/status",
              "/v1/admin/hold", "/v1/admin/stream-probe", "/v1/admin/echo", "/v1/admin/capacity", "/v1/aokie-companion/relay/stream"):
    ok(f"README lists the route {route}", route in readme)
ok("README has an Interpretations section", re.search(r"^## (\d+\. )?Interpretations$", readme, re.M) is not None)
interp_section = re.split(r"^## ", re.split(r"^## (?:\d+\. )?Interpretations$", readme, flags=re.M)[-1], maxsplit=1, flags=re.M)[0]  # up to the next H2: the design defects that follow are numbered too
interp_numbers = [int(n) for n in re.findall(r"^(\d+)\. \*\*", interp_section, re.M)]
ok("the Interpretations are numbered 1 to N with no gap and no repeat, and there are at least 23", interp_numbers == list(range(1, len(interp_numbers) + 1)) and len(interp_numbers) >= 23, str(interp_numbers))
ok("README says the SAS input carries the raw 16 bytes of pid and points at extras.sasNegative", "**raw 16 bytes**" in readme and "extras.sasNegative" in readme and 'not its 22-character b64u text' in readme)
ok("README says the item ids . and .. and the spelling -0 are refused", "never `.` or `..`" in readme and "the spelling `-0`" in readme and "`-0` is not an integer" in readme)
ok("README has a Response shapes section", re.search(r"^## (\d+\. )?Response shapes$", readme, re.M) is not None)

# ---------------------------------------------------------------------------
print("\n" + "-" * 60)
n_vec_neg = len(ex["tokens"]["invalid"]) + len(ex["canonical"]["refused"]) + 2 * len(a12["inputs"]["encodings"]) + len(ex["sasNegative"]["wrong"])
print(f"negative documents: {n_neg_schema} by schema + {n_rules} by reference rule "
      f"({n_vec_neg} of the rules replay the vectors' invalid tokens, refused numbers, wrong SAS readings and small-order keys) "
      f"= {n_neg_schema + n_rules}, and {aok_neg} damaged copies of the Aokie fixtures refused by the decoders' rules")
print(f"positive documents: {n_pos}; vector values recomputed in Python: {recomputed}; node checks: {node_total}")
print(f"relay protocol conformance: {passed} passed, {len(failures)} failed")
if failures:
    print("failed:\n  - " + "\n  - ".join(failures))
raise SystemExit(1 if failures else 0)
