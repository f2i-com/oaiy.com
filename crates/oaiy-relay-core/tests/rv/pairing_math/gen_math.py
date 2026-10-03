"""Independent recomputation of the pairing arithmetic (README 10.1) from the README's words, for random inputs. Writes math_cases.json (inputs and expected outputs)."""
import base64, hashlib, hmac, json, os, random

rng = random.Random(1234)
CROCK = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def hkdf(ikm, salt, info, n):
    prk = hmac.new(salt if salt else b"\0" * 32, ikm, hashlib.sha256).digest()
    out, t, i = b"", b"", 1
    while len(out) < n:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:n]


def crock(bits_int, nbits):
    # nbits -> characters of 5 bits, most significant first, zero padded on the right
    chars = (nbits + 4) // 5
    padded = bits_int << (chars * 5 - nbits)
    return "".join(CROCK[(padded >> (5 * (chars - 1 - i))) & 31] for i in range(chars))


def typed_code(s):
    body = crock(int.from_bytes(s, "big"), 128)
    d = hashlib.sha256(b"oaiy/pairing/3/typed\0" + s).digest()
    check = crock(int.from_bytes(d, "big") >> (256 - 10), 10)
    full = body + check
    return "-".join(full[i:i + 4] for i in range(0, 28, 4))


def sas(dk, pk, nonce, pid):
    raw = hkdf(dk + pk, nonce, b"oaiy/pairing/3/sas\0" + pid, 8)
    top60 = int.from_bytes(raw, "big") >> 4
    c12 = crock(top60, 60)
    chk = CROCK[hashlib.sha256(b"oaiy/pairing/3/sas-check\0" + c12.encode()).digest()[0] >> 3]
    return raw, c12, chk


def canon(o):
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def normalise(t):
    out = ""
    for ch in t:
        u = ch.upper() if ch.isascii() else ch   # ASCII only: the README says nothing of other scripts
        if u in "- ":
            continue
        if u in "IL":
            u = "1"
        if u == "O":
            u = "0"
        if u not in CROCK:
            return None
        out += u
    return out


def parse_typed(t):
    ch = normalise(t)
    if ch is None or len(ch) != 28:
        return None
    n = 0
    for c in ch[:26]:
        n = n * 32 + CROCK.index(c)
    if n & 3:
        return None            # the last two of the 130 bits must be zero
    s = (n >> 2).to_bytes(16, "big")
    want = typed_code(s).replace("-", "")[26:]
    return s.hex() if ch[26:] == want else None


cases = []
for i in range(300):
    s = bytes(rng.getrandbits(8) for _ in range(16))
    dk = bytes(rng.getrandbits(8) for _ in range(32))
    pk = bytes(rng.getrandbits(8) for _ in range(32))
    nonce = bytes(rng.getrandbits(8) for _ in range(32))
    pidk = hkdf(s, b"oaiy/pairing/3", b"rendezvous", 16)
    mac_key = hkdf(s, b"oaiy/pairing/3", b"mac", 32)
    raw, c12, chk = sas(dk, pk, nonce, pidk)
    names = ["Zoë", "日本語", "plain", "a\"b\\c", "x\ty", " line", "emoji \U0001f600", ""]
    offer_text = canon({"kind": "t", "n": rng.choice(names), "k": i, "z": [1, 2, {"b": 1, "a": 2}]})
    claims_text = canon({"jti": "pair-" + str(i), "displayName": rng.choice(names), "issuedAt": 1790000000 + i, "appId": "aokie"})
    grants = rng.sample(["state_read", "rtc_signal", "cmd", "monitor", "consult", "takeover", "ring"], rng.randint(0, 5))
    receipt = canon({"appId": "aokie", "grants": sorted(grants), "issuedAt": 1790000000 + i, "phoneThumbprint": b64u(hashlib.sha256(pk).digest()), "pid": b64u(pidk)})
    cases.append({
        "secret": s.hex(), "dk": dk.hex(), "pk": pk.hex(), "nonce": nonce.hex(),
        "offer_text": offer_text, "claims_text": claims_text,
        "app_id": "aokie", "grants": grants, "issued_at": 1790000000 + i, "phone_thumbprint": b64u(hashlib.sha256(pk).digest()),
        "expect": {
            "pid": pidk.hex(), "mac_key": mac_key.hex(), "typed": typed_code(s), "sas_raw": raw.hex(), "sas_chars12": c12, "sas_check": chk,
            "sas_display": f"{c12[0:4]}-{c12[4:8]}-{c12[8:12]}-{chk}",
            "offer_mac": b64u(hmac.new(mac_key, b"oaiy/pairing/3/offer-mac\0" + offer_text.encode(), hashlib.sha256).digest()),
            "response_mac": b64u(hmac.new(mac_key, b"oaiy/pairing/3/response-mac\0" + claims_text.encode(), hashlib.sha256).digest()),
            "receipt_text": receipt,
        },
    })

# typed code variants: (input text, expected secret hex or null)
s = bytes(range(1, 17))
tc = typed_code(s)
variants = [tc, tc.lower(), tc.replace("-", ""), tc.replace("-", " "), " " + tc + " ", tc.replace("-", "--"),
            tc.replace("1", "I"), tc.replace("1", "l").lower(), tc.replace("0", "O"), tc.replace("0", "o"),
            tc[:-1], tc + "0", tc[:-1] + ("0" if tc[-1] != "0" else "1"), tc.replace(tc[0], "U", 1),
            tc.replace("-", "‑"), "ı" + tc[1:], tc.replace("A", "Ａ")]
# every single-character substitution of every position: exactly one character differs
alphabet = CROCK
for pos in range(0, 34):
    if tc[pos] == "-":
        continue
    for rep in "0AZ":
        if rep != tc[pos]:
            variants.append(tc[:pos] + rep + tc[pos + 1:])
# a code whose trailing 2 bits are not zero but whose check is made to fit
t2 = bytes(range(16))
body = crock(int.from_bytes(t2, "big"), 128)
bad_trailing = body[:25] + CROCK[(CROCK.index(body[25]) | 1)]
d = hashlib.sha256(b"oaiy/pairing/3/typed\0" + t2).digest()
chk2 = crock(int.from_bytes(d, "big") >> (256 - 10), 10)
variants.append(bad_trailing + chk2)
cases_typed = [{"input": v, "expect": parse_typed(v)} for v in variants]
# the first (valid) one must parse to s
assert cases_typed[0]["expect"] == s.hex()

# sas entry classification: (entry, expected class) for one fixed SAS
dk = bytes(range(32)); pk = bytes(range(32, 64)); nonce = bytes(range(64, 96)); pidk = bytes(range(100, 116))
raw, c12, chk = sas(dk, pk, nonce, pidk)
disp = f"{c12[0:4]}-{c12[4:8]}-{c12[8:12]}-{chk}"
other = "".join(CROCK[(CROCK.index(c) + 1) % 32] for c in c12)
other_chk = CROCK[hashlib.sha256(b"oaiy/pairing/3/sas-check\0" + other.encode()).digest()[0] >> 3]
bad_chk = CROCK[(CROCK.index(chk) + 1) % 32]


def cls(entry):
    ch = normalise(entry)
    if ch is None:
        return "Invalid"
    if len(ch) <= 12:
        return "Incomplete"
    if len(ch) == 13:
        t, c = ch[:12], ch[12]
        want = CROCK[hashlib.sha256(b"oaiy/pairing/3/sas-check\0" + t.encode()).digest()[0] >> 3]
        if c != want:
            return "BadCheck"
        return "Right" if t == c12 else "Wrong"
    return "Invalid"


entries = [disp, disp.lower(), disp.replace("-", ""), disp[:-1], "", disp[:5], other + other_chk, other + bad_chk, c12 + bad_chk, disp + "0", "UUUU-UUUU-UUUU-U", "!!!!", c12.replace(c12[0], "I", 1) if c12[0] == "1" else c12]
sas_entries = [{"entry": e, "expect": cls(e)} for e in entries]
json.dump({"cases": cases, "typed": cases_typed, "sas": {"dk": dk.hex(), "pk": pk.hex(), "nonce": nonce.hex(), "pid": pidk.hex(), "display": disp, "entries": sas_entries}},
          open(os.path.join(os.path.dirname(__file__), "math_cases.json"), "w"))
print(len(cases), "math cases,", len(cases_typed), "typed variants,", len(sas_entries), "sas entries;", sum(1 for c in cases_typed if c["expect"]), "typed variants that parse")
