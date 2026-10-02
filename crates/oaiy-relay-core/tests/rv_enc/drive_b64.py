"""F2 (a): base64url differential. Writes the corpus, runs PHP B64.php, a strict Python reference and a Node round-trip reference, and (when --compare) compares them with
the Rust runner's output (tests/rv_enc_corpus.rs, corpus_b64) in the same directory."""
import base64, itertools, os, random, re, subprocess, sys

D = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(D, "work_b64")
os.makedirs(W, exist_ok=True)
PHP = os.environ.get("OAIY_PHP", r"C:\wamp64\bin\php\php8.4.15\php.exe")
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
rnd = random.Random(20261002)


def corpus():
    c = [b"", b"=", b"==", b"A", b"AA", b"AAA", b"AAAA", b"AAAAA", b"AA==", b"AAA=", b" ", b"\n", b"\x00", b"AA\n", b"AAAA\n", b"\nAAAA", b"AAAA ", b" AAAA"]
    # exhaustive 1..3 over the alphabet
    for n in (1, 2, 3):
        for t in itertools.product(ALPHA, repeat=n):
            c.append("".join(t).encode())
    # random length-4..: all tails (2 or 3 chars) after a random prefix of 4k chars
    for _ in range(40):
        pre = "".join(rnd.choice(ALPHA) for _ in range(4 * rnd.randint(0, 20)))
        for n in (2, 3):
            for t in itertools.product(ALPHA, repeat=n):
                if n == 3 and (rnd.random() < 0.75 or _ >= 6):
                    continue
                c.append((pre + "".join(t)).encode())
    # random full-length values: 16, 32, 64, 8, 22, 43, 86 bytes decoded canonical, then damaged
    for nbytes in (8, 16, 24, 32, 48, 64, 100, 1000):
        for _ in range(300):
            raw = bytes(rnd.getrandbits(8) for _ in range(nbytes))
            s = base64.urlsafe_b64encode(raw).decode().rstrip("=")
            c.append(s.encode())
            # last char flipped through the alphabet (unused bits)
            for ch in ALPHA:
                c.append((s[:-1] + ch).encode())
            # one position replaced by a hostile byte
            for _ in range(6):
                p = rnd.randrange(len(s))
                bad = rnd.choice([b"=", b"+", b"/", b" ", b"\t", b"\n", b"\r", b"\x00", b"\x7f", b"\x80", b"\xff", "\u00e9".encode(), "\uff21".encode(), "\u0410".encode(), b".", b",", b"%", b"~", b"@"])
                c.append(s[:p].encode() + bad + s[p + 1:].encode())
            c.append(s.encode() + b"=")
            c.append(s.encode() + b"==")
            c.append(s.encode() + b"\n")
            c.append(b"\n" + s.encode())
            c.append(s.encode() + b"A")
            c.append(s[:-1].encode())
            c.append(base64.b64encode(raw))  # standard alphabet with padding
    # long ones
    for n in (10000, 100000, 1000001):
        raw = bytes(rnd.getrandbits(8) for _ in range(n))
        c.append(base64.urlsafe_b64encode(raw).decode().rstrip("=").encode())
    return c


def py_ref(b):
    try:
        s = b.decode("ascii")
    except UnicodeDecodeError:
        return "ERR"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", s, re.ASCII) or len(s) % 4 == 1:
        return "ERR"
    pad = (4 - len(s) % 4) % 4
    raw = base64.b64decode(s.replace("-", "+").replace("_", "/") + "=" * pad, validate=True)
    if base64.urlsafe_b64encode(raw).decode().rstrip("=") != s:
        return "ERR"
    return "OK:" + raw.hex()


def exact(res, n):
    return "1" if res.startswith("OK:") and len(res) == 3 + 2 * n else "0"


def main():
    c = corpus()
    with open(os.path.join(W, "b64.in"), "w") as f:
        for b in c:
            f.write(b.hex() + "\n")
    print("corpus", len(c))
    py = []
    for b in c:
        r = py_ref(b)
        py.append(r + "\t" + exact(r, 16) + exact(r, 32) + exact(r, 64))
    # PHP
    subprocess.run([PHP, "-d", "memory_limit=3G", os.path.join(D, "php_b64.php"), os.path.join(W, "b64.in"), os.path.join(W, "b64.php.out")], check=True)
    # Node: lenient decode + canonical round trip check
    subprocess.run(["node", os.path.join(D, "node_b64.js"), os.path.join(W, "b64.in"), os.path.join(W, "b64.node.out")], check=True)
    php = open(os.path.join(W, "b64.php.out")).read().split("\n")[:-1]
    node = open(os.path.join(W, "b64.node.out")).read().split("\n")[:-1]
    rust_path = os.path.join(W, "b64.rust.out")
    rust = open(rust_path).read().split("\n")[:-1] if os.path.exists(rust_path) else None
    n = len(c)
    assert len(php) == n and len(node) == n and (rust is None or len(rust) == n), (len(php), len(node), rust and len(rust))
    dis = {"py_vs_php": 0, "py_vs_node": 0, "py_vs_rust": 0, "rust_roundtrip_false": 0, "rust_skip": 0}
    ex = {k: [] for k in dis}
    ok_count = sum(1 for r in py if r.startswith("OK:"))
    for i in range(n):
        if py[i] != php[i]:
            dis["py_vs_php"] += 1
            ex["py_vs_php"].append((c[i][:60], py[i][:40], php[i][:40]))
        if py[i] != node[i]:
            dis["py_vs_node"] += 1
            ex["py_vs_node"].append((c[i][:60], py[i][:40], node[i][:40]))
        if rust is not None:
            r = rust[i]
            if r == "SKIP":
                dis["rust_skip"] += 1
                if not py[i].startswith("ERR"):
                    ex["rust_skip"].append(c[i][:60])
                continue
            dec, e, rt = r.split("\t")
            rr = ("OK:" + dec[3:] if dec.startswith("OK:") else "ERR") + "\t" + e
            if rr != py[i]:
                dis["py_vs_rust"] += 1
                ex["py_vs_rust"].append((c[i][:60], py[i][:40], r[:60]))
            if rt != "1":
                dis["rust_roundtrip_false"] += 1
    print("accepted by the python reference:", ok_count, "of", n)
    print(dis)
    for k, v in ex.items():
        for e in v[:5]:
            print(k, e)


if __name__ == "__main__":
    main()
