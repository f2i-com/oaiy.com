"""F2 (c): typed code, SAS entry, identifier validators, display-name cleaning, relay URL and percent-decoding: corpora, independent Python references written from the README and
the schemas, PHP Ids.php, and comparison with the Rust runner (tests/rv_enc_corpus.rs). Usage: drive_text.py gen | php | compare"""
import hashlib, itertools, os, random, re, subprocess, sys, collections

D = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(D, "work_text")
os.makedirs(W, exist_ok=True)
PHP = os.environ.get("OAIY_PHP", r"C:\wamp64\bin\php\php8.4.15\php.exe")
rnd = random.Random(1002)
CROCK = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
B64A = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def hexl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(r + "\n")


# ---------------------------------------------------------------- typed code (README 10.1, from the text)
def crock_enc(data, nbits):
    out = ""
    pos = 0
    bits = "".join(f"{b:08b}" for b in data)
    while pos < nbits:
        chunk = bits[pos:pos + 5].ljust(5, "0") if pos + 5 <= nbits else bits[pos:nbits].ljust(5, "0")
        out += CROCK[int(chunk, 2)]
        pos += 5
    return out


def typed_make(secret):
    body = crock_enc(secret, 128)
    chk = crock_enc(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + secret).digest(), 10)
    s = body + chk
    return "-".join(s[i:i + 4] for i in range(0, 28, 4))


def normalise(t):
    out = ""
    for ch in t:
        c = ch.upper() if ch.isascii() else ch  # ASCII-only upper-casing (a non-ASCII character is never part of the alphabet)
        if c in "- ":
            continue
        if c in "IL":
            c = "1"
        elif c == "O":
            c = "0"
        if c not in CROCK:
            return None
        out += c
    return out


def typed_parse(t):
    n = normalise(t)
    if n is None:
        return ("NONE", "ERR")
    if len(n) != 28:
        return ("OK:" + n, "ERR")
    bits = "".join(f"{CROCK.index(c):05b}" for c in n[:26])
    if bits[128:] != "00":
        return ("OK:" + n, "ERR")
    secret = int(bits[:128], 2).to_bytes(16, "big")
    chk = crock_enc(hashlib.sha256(b"oaiy/pairing/3/typed\x00" + secret).digest(), 10)
    if chk != n[26:]:
        return ("OK:" + n, "ERR")
    return ("OK:" + n, "OK:" + secret.hex())


def gen_typed():
    rows = []
    cases = []
    for _ in range(3000):
        secret = bytes(rnd.getrandbits(8) for _ in range(16))
        code = typed_make(secret)
        cases.append(code)
        flat = code.replace("-", "")
        cases += [code.lower(), flat, flat.lower(), code.replace("-", " "), " " + code + " ", code.replace("-", "--"), code + "\n", "\t" + code, code + "\u00a0", code + "\u200b", "\ufeff" + code]
        # confusables
        cases.append(code.replace("1", "I", 1).replace("0", "O", 1))
        cases.append(code.replace("1", "l", 1))
        cases.append(code.replace("0", "o", 1))
        # U refused, I/L/O mapping
        cases.append(code[:3] + "U" + code[4:])
        # length
        cases += [code[:-1], code + "A", flat[:27], flat + "0", flat[:26], flat[:25]]
        # trailing bits (the 26th char's low two bits) and wrong check
        body = flat[:26]
        v = CROCK.index(body[25])
        for k in range(1, 4):
            cases.append(body[:25] + CROCK[v ^ k] + flat[26:])
        cases.append(flat[:26] + CROCK[(CROCK.index(flat[26]) + 1) % 32] + flat[27])
        cases.append(flat[:27] + CROCK[(CROCK.index(flat[27]) + 1) % 32])
        # single substitutions at random positions
        for _ in range(8):
            p = rnd.randrange(28)
            c = rnd.choice(CROCK)
            cases.append(flat[:p] + c + flat[p + 1:])
        # unicode lookalikes
        cases.append(flat[:5] + "\uff21" + flat[6:])  # fullwidth A
        cases.append(flat[:5] + "\u0131" + flat[6:])  # dotless i
        cases.append(flat[:5] + "\u212a" + flat[6:])  # Kelvin sign
        cases.append(flat[:5] + "\u017f" + flat[6:])  # long s
        cases.append(flat[:5] + "\u0660" + flat[6:])  # arabic-indic zero
    cases += ["", "-", " ", "0", "0" * 26, "0" * 28, "0" * 27, "Z" * 28, "U" * 28]
    for c in cases:
        rows.append(c.encode("utf-8").hex())
    return rows


# ---------------------------------------------------------------- SAS entry
def sas_check(chars12):
    return CROCK[hashlib.sha256(b"oaiy/pairing/3/sas-check\x00" + chars12.encode()).digest()[0] >> 3]


def sas_judge(expected12, typed):
    n = normalise(typed)
    if n is None:
        return "Invalid"
    if len(n) < 13:
        return "Incomplete"
    if len(n) > 13:
        return "Invalid"
    twelve, chk = n[:12], n[12]
    if sas_check(twelve) != chk:
        return "BadCheck"
    return "Right" if twelve == expected12 else "Wrong"


def gen_sas():
    rows = []
    meta = []
    for _ in range(3000):
        exp = "".join(rnd.choice(CROCK) for _ in range(12))
        full = exp + sas_check(exp)
        shown = f"{exp[:4]}-{exp[4:8]}-{exp[8:12]}-{full[12]}"
        cands = [shown, shown.lower(), full, shown.replace("-", " "), full[:12], full[:11], full[:5], "", full + "A", shown + "-", shown + "\n", " " + shown]
        # single substitution among the 12 (the typo the check is meant to catch) and in the check
        for p in range(13):
            for c in CROCK:
                if c != full[p] and rnd.random() < 0.08:
                    cands.append(full[:p] + c + full[p + 1:])
        # an adjacent transposition
        for p in range(12):
            if full[p] != full[p + 1]:
                cands.append(full[:p] + full[p + 1] + full[p] + full[p + 2:])
        # another SAS with a correct check
        other = "".join(rnd.choice(CROCK) for _ in range(12))
        cands.append(other + sas_check(other))
        cands.append(full[:11] + "U" + full[12:])
        cands.append(shown.replace("1", "I", 1).replace("0", "O", 1))
        for c in cands:
            rows.append(exp + "\t" + c.encode("utf-8").hex())
    return rows


# ---------------------------------------------------------------- identifiers
FNS = ["device", "provider", "relay", "principal", "pid", "epoch", "item", "app", "thumb", "grant", "token", "jti"]


def b64s(n, alpha=B64A):
    return "".join(rnd.choice(alpha) for _ in range(n))


def canon_b64(nbytes):
    import base64
    return base64.urlsafe_b64encode(bytes(rnd.getrandbits(8) for _ in range(nbytes))).decode().rstrip("=")


def mutate_str(s):
    s = list(s)
    for _ in range(rnd.randint(1, 3)):
        op = rnd.randrange(5)
        p = rnd.randrange(len(s) + 1) if s else 0
        ch = rnd.choice(["+", "/", "=", ".", " ", "\n", "\x00", "\u00e9", "\uff21", "\u0410", "-", "_", ":", "@", "~", "A", "z", "9", "\t", "\r"])
        if op == 0 and s:
            del s[min(p, len(s) - 1)]
        elif op == 1:
            s.insert(p, ch)
        elif op == 2 and s:
            s[min(p, len(s) - 1)] = ch
        elif op == 3:
            s.append(ch)
        else:
            s = s + s[: rnd.randint(0, 5)]
    return "".join(s)


def gen_ids():
    rows = []
    for fn in FNS:
        for _ in range(4000):
            if fn == "device":
                base = "dev-" + canon_b64(16)
            elif fn == "provider":
                base = "prov-" + canon_b64(16)
            elif fn == "relay":
                base = "rly-" + canon_b64(16)
            elif fn == "principal":
                base = rnd.choice(["dev-", "prov-"]) + b64s(22)
            elif fn == "pid":
                base = b64s(22)
            elif fn == "epoch":
                base = b64s(11)
            elif fn == "item":
                base = "".join(rnd.choice(B64A + "..") for _ in range(rnd.choice([1, 2, 3, 20, 128, 129])))
            elif fn == "app":
                base = "".join(rnd.choice(B64A + ".:") for _ in range(rnd.choice([1, 5, 63, 64, 65])))
            elif fn == "thumb":
                base = rnd.choice([canon_b64(32), b64s(43)])
            elif fn == "grant":
                base = rnd.choice("abcdefghijklmnopqrstuvwxyz") + "".join(rnd.choice("abcdefghijklmnopqrstuvwxyz0123456789_") for _ in range(rnd.choice([0, 5, 31, 32])))
            elif fn == "token":
                base = "oaiyrt1." + rnd.choice([canon_b64(8), b64s(11)]) + "." + rnd.choice([canon_b64(32), b64s(43)])
            else:
                base = "pair-" + b64s(rnd.choice([0, 1, 22, 43, 64, 65]))
            rows.append(fn + "\t" + base.encode().hex())
            for _ in range(2):
                rows.append(fn + "\t" + mutate_str(base).encode("utf-8").hex())
    for s in [".", "..", "...", "a/b", "a b", "", "dev-", "DEV-" + "A" * 22, "dev-" + "A" * 22 + "\n", "dev- " + "A" * 21, "\u00e9" * 22, "pair-" + "\u00e9"]:
        for fn in FNS:
            rows.append(fn + "\t" + s.encode("utf-8").hex())
    return rows


# ---------------------------------------------------------------- display names
NAMEPOOL = list("abcXYZ019 ") * 3 + ["\x00", "\x01", "\x1f", "\x7f", "\x80", "\x85", "\x9f", "\u00a0", "\u200b", "\u2028", "\u2029", "\u0301", "\u00e9", "\u65e5", "\u672c", "\U0001f600", "\U00010000", "\ufeff", "\n", "\t", "\r", "\ufffd", "\uffff"]


def gen_names():
    rows = []
    for _ in range(40000):
        n = rnd.choice([0, 1, 2, 5, 20, 59, 60, 61, 100, 130, 200])
        s = "".join(rnd.choice(NAMEPOOL) for _ in range(n))
        mx = rnd.choice([1, 2, 5, 60, 60, 60, 120, 200])
        rows.append(f"{mx}\t{s.encode('utf-8').hex()}")
    for s in ["  Front\x00desk\x7f PC\n ", "\u65e5" * 50, "\U0001f600" * 35, "\U0001f600" * 31, "a" * 80, "\x00\x01  ", "  ", " a ", "\u00a0a\u00a0"]:
        for mx in (60, 120):
            rows.append(f"{mx}\t{s.encode('utf-8').hex()}")
    return rows


# ---------------------------------------------------------------- relay URL and percent escapes
def gen_urls():
    rows = []
    hosts = ["relay.example.com", "Relay.Example.COM", "a", "-", ".", "a..b", "127.0.0.1", "localhost", "LOCALHOST", "127.0.0.1.", "localhost.", "127.0.0.2", "10.0.0.1", "[::1]", "::1", "0x7f.1", "2130706433",
             "re_lay.com", "r\u00e9lay.com", "xn--rlay-bsa.com", "a" * 63 + ".com", "a" * 253, "a" * 254, "user@relay.com", "relay.com@evil.com", "relay.com:80@evil.com", "relay.com\\@evil.com", "relay.com.", ""]
    for scheme in ["https", "HTTPS", "Https", "http", "HTTP", "ftp", "", "https:", "https:/", "https:///", "wss", "ws"]:
        for h in hosts:
            for port in ["", ":443", ":0443", ":80", ":8080", ":65535", ":65536", ":0", ":00000", ":1", ":", ":a", ":+80", ":-1", ":80:80", ":99999", ":123456", ":\u0661", ": 80"]:
                for suffix in ["", "/", "/path", "?q=1", "#f", " ", "\n", "\u0000", "/v1/info"]:
                    if rnd.random() < 0.06:
                        rows.append(f"{scheme}://{h}{port}{suffix}".encode("utf-8").hex())
    for t in ["https://relay.example.com", "https://relay.example.com:8443", "http://127.0.0.1:8080", "http://localhost", "%", "%4", "%zz", "%ff", "%41", "%2F%2f", "a+b", "a%20b", "%e2%82%ac", "%c0%af", "%ed%a0%80", "%00", "https%3A%2F%2Frelay.example.com",
              "%%", "%4%41", "%\u00e9\u00e9", "%\uff11\uff11", "%+1", "%-1", "% 1", "%1 ", "%0x", "\u00e9", "%C3%A9", "%c3%a9"]:
        rows.append(t.encode("utf-8").hex())
    for _ in range(4000):
        n = rnd.randint(0, 40)
        s = "".join(rnd.choice("abcAB09%%%-._~+ /:?#&=\u00e9") + (rnd.choice(["", "", "41", "zz", "f", "C3", "A9", "00", "ff"]) if rnd.random() < 0.3 else "") for _ in range(n))
        rows.append(s.encode("utf-8").hex())
    return rows


def url_ref(t, lax):
    m = re.fullmatch(r"(?i)(https)://([A-Za-z0-9.-]+)(:[0-9]{1,5})?", t)
    scheme = None
    if m:
        scheme = "https"
    elif lax:
        m = re.fullmatch(r"(?i)(http)://([A-Za-z0-9.-]+)(:[0-9]{1,5})?", t)
        if m:
            scheme = "http"
    if not m:
        return "ERR"
    host, port = m.group(2), m.group(3)
    if len(host) > 253:
        return "ERR"
    if port is not None:
        pn = int(port[1:])
        if pn == 0 or pn > 65535:
            return "ERR"
        port = ":" + str(pn)
    if scheme == "http" and host.lower() not in ("127.0.0.1", "localhost"):
        return "ERR"
    return f"{scheme}://{host.lower()}{port or ''}"


def pct_ref(t):
    raw = t.encode("utf-8")
    out = bytearray()
    i = 0
    while i < len(raw):
        if raw[i] == 0x25:
            h = raw[i + 1:i + 3]
            if len(h) < 2 or not re.fullmatch(rb"[0-9A-Fa-f]{2}", h):
                return "ERR"
            out.append(int(h, 16))
            i += 3
        else:
            out.append(raw[i])
            i += 1
    try:
        bytes(out).decode("utf-8")
    except UnicodeDecodeError:
        return "ERR"
    return bytes(out).hex()


# ---------------------------------------------------------------- driver
def read(path):
    return [l.rstrip("\n") for l in open(path, encoding="utf-8")]


def main():
    mode = sys.argv[1]
    if mode == "gen":
        hexl(os.path.join(W, "typed.in"), gen_typed())
        hexl(os.path.join(W, "sas.in"), gen_sas())
        hexl(os.path.join(W, "ids.in"), gen_ids())
        hexl(os.path.join(W, "name.in"), gen_names())
        hexl(os.path.join(W, "url.in"), gen_urls())
        for n in ("typed", "sas", "ids", "name", "url"):
            print(n, len(read(os.path.join(W, n + ".in"))))
        return
    if mode == "php":
        subprocess.run([PHP, os.path.join(D, "php_text.php"), W], check=True)
        return
    # compare
    typed_in = read(os.path.join(W, "typed.in"))
    typed_out = read(os.path.join(W, "typed.rust.out"))
    bad = collections.Counter()
    ex = collections.defaultdict(list)
    ok_codes = 0
    for l, r in zip(typed_in, typed_out):
        t = bytes.fromhex(l).decode("utf-8")
        norm, parsed = typed_parse(t)
        if r == "SKIP":
            continue
        rn, rp = r.split("\t")
        if rn != norm:
            bad["typed normalise"] += 1
            ex["typed normalise"].append((t, norm, rn))
        rp2 = "ERR" if rp.startswith("ERR") else rp
        if rp2 != parsed:
            bad["typed parse"] += 1
            ex["typed parse"].append((t, parsed, rp))
        ok_codes += parsed.startswith("OK:")
    print("typed:", len(typed_in), "inputs,", ok_codes, "accepted;", dict(bad) or "no disagreement with the reference")
    # round trip of the crate's own writer is checked in the Rust unit tests; here: Python writer == crate parse of it (they were accepted above)
    sas_in = read(os.path.join(W, "sas.in"))
    sas_out = read(os.path.join(W, "sas.rust.out"))
    counts = collections.Counter()
    typo_pass = collections.Counter()
    bad2 = 0
    for l, r in zip(sas_in, sas_out):
        exp, h = l.split("\t")
        t = bytes.fromhex(h).decode("utf-8")
        ref = sas_judge(exp, t)
        if r == "SKIP":
            continue
        counts[ref] += 1
        if ref != r:
            bad2 += 1
            if bad2 <= 5:
                print("SAS disagreement", exp, repr(t), ref, r)
    print("sas:", len(sas_in), "entries; reference outcomes", dict(counts), "; disagreements", bad2)
    # single-substitution miss rate: among entries that differ from the right one in exactly one position of 13 characters
    n_single = n_single_wrong = 0
    for l, r in zip(sas_in, sas_out):
        exp, h = l.split("\t")
        t = bytes.fromhex(h).decode("utf-8")
        full = exp + sas_check(exp)
        if len(t) == 13 and all(c in CROCK for c in t) and sum(a != b for a, b in zip(t, full)) == 1:
            n_single += 1
            n_single_wrong += r == "Wrong"
    print("sas single-character substitutions:", n_single, "of which counted as a wrong attempt (check still matched):", n_single_wrong, "= %.2f%%" % (100 * n_single_wrong / max(1, n_single)))
    n_tr = n_tr_wrong = 0
    for l, r in zip(sas_in, sas_out):
        exp, h = l.split("\t")
        t = bytes.fromhex(h).decode("utf-8")
        full = exp + sas_check(exp)
        diffs = [i for i in range(13) if len(t) == 13 and t[i] != full[i]]
        if len(t) == 13 and len(diffs) == 2 and diffs[1] == diffs[0] + 1 and t[diffs[0]] == full[diffs[1]] and t[diffs[1]] == full[diffs[0]]:
            n_tr += 1
            n_tr_wrong += r == "Wrong"
    print("sas adjacent transpositions:", n_tr, "of which counted as a wrong attempt:", n_tr_wrong)

    # identifiers vs the PHP relay's Ids.php and the schema patterns
    ids_in = read(os.path.join(W, "ids.in"))
    ids_out = read(os.path.join(W, "ids.rust.out"))
    ids_php = read(os.path.join(W, "ids.php.out"))
    d1 = d2 = 0
    per = collections.Counter()
    for l, r, p in zip(ids_in, ids_out, ids_php):
        fn, h = l.split("\t")
        if r == "SKIP":
            continue
        per[fn] += 1
        if r != p:
            d1 += 1
            if d1 <= 8:
                print("IDS disagreement vs PHP:", fn, bytes.fromhex(h)[:70], "rust", r, "php", p)
    print("ids:", len(ids_in), "inputs; crate vs PHP Ids.php disagreements:", d1)
    # names
    nin = read(os.path.join(W, "name.in"))
    nout = read(os.path.join(W, "name.rust.out"))
    nphp = read(os.path.join(W, "name.php.out"))
    d = 0
    for l, r, p in zip(nin, nout, nphp):
        if r == "SKIP":
            continue
        if r != p:
            d += 1
            if d <= 8:
                mx, h = l.split("\t")
                print("NAME disagreement:", mx, repr(bytes.fromhex(h).decode()[:60]), "rust", bytes.fromhex(r).decode()[:40].__repr__(), "php", bytes.fromhex(p).decode()[:40].__repr__())
    print("names:", len(nin), "inputs; crate vs PHP cleanName disagreements:", d)
    # urls
    uin = read(os.path.join(W, "url.in"))
    uout = read(os.path.join(W, "url.rust.out"))
    d = 0
    acc = 0
    for l, r in zip(uin, uout):
        if r == "SKIP":
            continue
        t = bytes.fromhex(l).decode("utf-8")
        a, b, p = r.split("\t")
        ra, rb, rp = url_ref(t, False), url_ref(t, True), pct_ref(t)
        acc += a != "ERR"
        if (a, b, p) != (ra, rb, rp):
            d += 1
            if d <= 10:
                print("URL disagreement:", repr(t), "crate", (a, b, p[:20]), "ref", (ra, rb, rp[:20]))
    print("urls:", len(uin), "inputs;", acc, "accepted as https; disagreements with the schema-pattern reference:", d)


if __name__ == "__main__":
    main()
