"""F2 (b): JSON differential. Corpus -> {PHP json_decode (depth 65) and the relay's Json::decode rules, Python strict reference, Node JSON.parse, serde_json (inside the Rust runner),
this crate (general and canonical modes)}. Usage: drive_json.py gen|refs|compare [N]"""
import json, os, random, re, subprocess, sys, collections

D = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(D, "work_json")
os.makedirs(W, exist_ok=True)
PHP = os.environ.get("OAIY_PHP", r"C:\wamp64\bin\php\php8.4.15\php.exe")
rnd = random.Random(8785)

KEYS = ["a", "b", "k", "\\u0061", "\\u0062", "\u00e9", "e\u0301", "\\u00e9", "\ue000", "\uffff", "\U00010000", "\U0001f600", "\\ud83d\\ude00", "\\ue000", "\\uffff", "\\ud800\\udc00", "", " ", "A", "1", "10", "2", "_", "z"]
STRS = ["", "a", "abc", "\\\"", "\\\\", "\\/", "\\b\\f\\n\\r\\t", "\\u0000", "\\u001f", "\\u007f", "\u007f", "\\u00e9", "\u00e9", "e\u0301", "\u65e5\u672c", "\U0001f600", "\\ud83d\\ude00", "\u2028", "\u2029",
        "\uffff", "\ufffe", "\\ufffe", "\\uffff", "\\ud7ff", "\\ue000", "\\u2028", "x y", "\\u0041", "\\u00e9\\u00e9"]
BADSTRS = ["\\ud800", "\\udc00", "\\ud83d", "\\ud83dx", "\\ud83d\\u0041", "\\ude00\\ud83d", "\\u12", "\\u12G4", "\\x41", "\\", "\\ ", "\x01", "\x1f", "\n", "\t", "\r", "\x00", "\\U0041", "\\u+123"]
NUMS = ["0", "-0", "1", "-1", "12", "100", "123456789012345678", "9007199254740991", "9007199254740992", "-9007199254740991", "-9007199254740992", "9223372036854775807", "9223372036854775808",
        "-9223372036854775808", "-9223372036854775809", "18446744073709551615", "18446744073709551616", "1" * 38, "1" * 39, "-" + "1" * 38, "-" + "1" * 39, "9" * 400,
        "1.0", "1.5", "1e2", "1E+2", "1e-2", "-0.0", "0e0", "0.0e+0", "1e400", "1e-400", "1e99999999999999999999", "0.5", "-0.5", "100.0", "5.0", "6e1"]
BADNUMS = ["01", "00", "-01", "1.", ".5", "+1", "1e", "1e+", "--1", "0x1", "1_0", "Infinity", "NaN", "-Infinity", "- 1", "-", "1.e2", "1e2.5", "0.", "-.5", "1ee2", "\u0661", "1\u0662"]
WS = [" ", "\t", "\n", "\r", "", "  ", " \n "]
BADWS = ["\f", "\v", "\u00a0", "\u2003", "\ufeff", "\x00", "\x1f"]


def ws():
    r = rnd.random()
    if r < 0.6:
        return ""
    if r < 0.985:
        return rnd.choice(WS)
    return rnd.choice(BADWS)


def string(bad=0.0):
    pool = BADSTRS if rnd.random() < bad else STRS
    return '"' + "".join(rnd.choice(pool) for _ in range(rnd.randint(0, 3))) + '"'


def number(bad=0.0):
    return rnd.choice(BADNUMS if rnd.random() < bad else NUMS)


BUDGET = [0]


def value(depth, maxd, bad):
    r = rnd.random()
    BUDGET[0] -= 1
    if depth >= maxd or r < 0.35 or BUDGET[0] <= 0:
        k = rnd.random()
        if k < 0.35:
            return number(bad)
        if k < 0.7:
            return string(bad)
        return rnd.choice(["true", "false", "null"] + (["True", "nul", "nulll", "tru e", "NULL", "undefined"] if rnd.random() < bad else []))
    if r < 0.65:
        n = rnd.randint(0, 4)
        sep = "," if rnd.random() > bad * 0.5 else rnd.choice([",,", "", " ", ";"])
        body = (sep + ws()).join(ws() + value(depth + 1, maxd, bad) + ws() for _ in range(n))
        tail = "," if rnd.random() < bad * 0.5 else ""
        return "[" + body + tail + "]"
    n = rnd.randint(0, 4)
    items = []
    for _ in range(n):
        k = rnd.choice(KEYS)
        if rnd.random() < 0.08:
            items.append('"' + k + '"' + ws() + ":" + ws() + value(depth + 1, maxd, bad))
        else:
            items.append('"' + k + '"' + ws() + ":" + ws() + value(depth + 1, maxd, bad))
    tail = "," if rnd.random() < bad * 0.5 else ""
    return "{" + ("," + ws()).join(items) + tail + "}"


def nest(n, kind):
    if kind == "arr":
        return "[" * n + "1" + "]" * n
    if kind == "obj":
        return '{"a":' * n + "1" + "}" * n
    return "".join(rnd.choice(['[', '{"a":']) for _ in range(n))  # unbalanced marker, fixed below


def mixed_nest(n):
    opens = []
    s = ""
    for _ in range(n):
        if rnd.random() < 0.5:
            s += "["
            opens.append("]")
        else:
            s += '{"a":'
            opens.append("}")
    return s + "1" + "".join(reversed(opens))


def mutate(b):
    b = bytearray(b)
    for _ in range(rnd.randint(1, 3)):
        if not b:
            break
        op = rnd.randrange(6)
        p = rnd.randrange(len(b))
        if op == 0:
            del b[p]
        elif op == 1:
            b.insert(p, rnd.choice(b'{}[]",:\\ -+.eE0123456789tfnu\x00\x1f\x7f\xc0\xff\x80'))
        elif op == 2:
            b[p] = rnd.randrange(256)
        elif op == 3:
            q = rnd.randrange(len(b))
            b[p], b[q] = b[q], b[p]
        elif op == 4:
            b[p:p] = b[p:p + rnd.randint(1, 5)]
        else:
            del b[p:]
    return bytes(b)


def corpus(n):
    c = []
    hand = [b"", b" ", b"\n", b"{}", b"[]", b"null", b"true", b"false", b"0", b"-0", b"1", b"\"a\"", b"\"\"", b"{} ", b" {}", b"{}{}", b"[]]", b"[1,]", b"{\"a\":1,}", b"{\"a\"}", b"{a:1}",
            b"'a'", b"\xef\xbb\xbf{}", b"\xef\xbb\xbf[]", b"\"\xc0\xaf\"", b"\"\xed\xa0\x80\"", b"\"\xed\xb0\x80\"", b"\"\xf4\x90\x80\x80\"", b"\"\xc3\"", b"\"\xe2\x82\"", b"\"\xff\"",
            "\"\\ud83d\\ude00\"".encode(), "\"\U0001f600\"".encode(), "\"\ufffe\"".encode(), "\"\uffff\"".encode(), b"\"a\x7fb\"", b"{\"a\":1,\"a\":2}", b"{\"a\":1,\"\\u0061\":2}", b"[{\"k\":1,\"k\":1}]",
            b"{\"x\":{\"a\":1,\"b\":2,\"a\":3}}", "{\"\u00e9\":1,\"\\u00e9\":2}".encode(), "{\"\ue000\":1,\"\U00010000\":2}".encode(), "{\"\U00010000\":1,\"\uffff\":2}".encode(),
            b"[1 2]", b"[,1]", b"nul", b"True", b"01", b"1.", b".5", b"+1", b"1e", b"--1", b"\xc2\xa0[]", b"\x0b[]", b"\x0c[]", b"[]\x00", b"\x00[]", b"[\"\x00\"]", b"1e999", b"-1e999",
            b"{\"a\":0.1e1}", b"[-0]", b"[-0.0]", b"[-0e0]", b"[0e-0]", b"[1E400]", b"\"\\u0000\"", b"\"\\/\"", b"[1]\n", b"\r\n[1]\r\n", b"\t[1]\t"]
    c += hand
    # depth boundary
    for nlev in range(58, 72):
        c.append(nest(nlev, "arr").encode())
        c.append(nest(nlev, "obj").encode())
        for _ in range(3):
            c.append(mixed_nest(nlev).encode())
    c.append(("[" * 1000).encode())
    c.append(("[" * 100000).encode())
    c.append(('{"a":' * 100000).encode())
    c.append(("[" * 100000 + "]" * 100000).encode())
    # random structure
    while len(c) < n:
        bad = rnd.choice([0.0, 0.0, 0.0, 0.03, 0.1, 0.3])
        maxd = rnd.choice([2, 3, 4, 6, 10, 70])
        BUDGET[0] = rnd.choice([5, 20, 60, 200])
        t = ws() + value(0, maxd, bad) + ws()
        b = t.encode("utf-8", "surrogatepass")
        c.append(b)
        if rnd.random() < 0.4:
            c.append(mutate(b))
    return c[:n]


def hexlines(path, items):
    with open(path, "w") as f:
        for b in items:
            f.write(b.hex() + "\n")


# ---------- the Python strict reference
class Lit:
    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s


class Obj:
    __slots__ = ("pairs",)

    def __init__(self, pairs):
        self.pairs = pairs


class Dup(Exception):
    pass


class Bad(Exception):
    pass


def _hook(pairs):
    names = [k for k, _ in pairs]
    if len(set(names)) != len(names):
        raise Dup()
    return Obj(pairs)


def _const(s):
    raise Bad("constant")


def depth_of(v):
    # iterative max depth of arrays/objects
    mx = 0
    stack = [(v, 1)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, Obj):
            mx = max(mx, d)
            stack.extend((y, d + 1) for _, y in x.pairs)
        elif isinstance(x, list):
            mx = max(mx, d)
            stack.extend((y, d + 1) for y in x)
    return mx


def lone_surrogate(v):
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if any(0xD800 <= ord(ch) <= 0xDFFF for ch in x):
                return True
        elif isinstance(x, Obj):
            for k, y in x.pairs:
                stack.append(k)
                stack.append(y)
        elif isinstance(x, list):
            stack.extend(x)
    return False


def py_parse(raw):
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ("ERR", "utf8", None)
    if s.startswith("\ufeff"):
        return ("ERR", "bom", None)
    try:
        v = json.loads(s, object_pairs_hook=_hook, parse_constant=_const, parse_int=Lit, parse_float=Lit)
    except Dup:
        return ("ERR", "dup", None)
    except RecursionError:
        return ("ERR", "depth", None)
    except (json.JSONDecodeError, Bad):
        return ("ERR", "syntax", None)
    if depth_of(v) > 64:
        return ("ERR", "depth", None)
    if lone_surrogate(v):
        return ("ERR", "surrogate", None)
    return ("OK", "", v)


def ser(v, sortkeys=None):
    out = []

    def go(x):
        if isinstance(x, Obj):
            pairs = x.pairs
            if sortkeys:
                pairs = sorted(pairs, key=lambda kv: sortkeys(kv[0]))
            out.append("{")
            for i, (k, y) in enumerate(pairs):
                if i:
                    out.append(",")
                out.append(json.dumps(k, ensure_ascii=False))
                out.append(":")
                go(y)
            out.append("}")
        elif isinstance(x, list):
            out.append("[")
            for i, y in enumerate(x):
                if i:
                    out.append(",")
                go(y)
            out.append("]")
        elif isinstance(x, str):
            out.append(json.dumps(x, ensure_ascii=False))
        elif x is True:
            out.append("true")
        elif x is False:
            out.append("false")
        elif x is None:
            out.append("null")
        elif isinstance(x, Lit):
            out.append(x.s)
        else:
            raise TypeError(type(x))

    go(v)
    return "".join(out)


def canon_ok(v):
    """The protocol's canonical rules on a parsed value: integers only, no -0, -2^63 .. 2^64-1."""
    stack = [v]
    while stack:
        x = stack.pop()
        if isinstance(x, Lit):
            if re.search(r"[.eE]", x.s) or x.s == "-0":
                return "NotAnInteger"
            n = int(x.s)
            if not (-(1 << 63) <= n <= (1 << 64) - 1):
                return "IntegerRange"
        elif isinstance(x, Obj):
            stack.extend(y for _, y in x.pairs)
        elif isinstance(x, list):
            stack.extend(x)
    return None


def u16key(k):
    return k.encode("utf-16-be", "surrogatepass")


def refs(items):
    res = []
    for b in items:
        st, why, v = py_parse(b)
        g = c = None
        u16 = None
        if st == "OK":
            g = ser(v)
            if canon_ok(v) is None:
                c = ser(v, sortkeys=lambda k: k)
                u16 = ser(v, sortkeys=u16key)
        res.append((st, why, g, c, u16))
    return res


def main():
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    if mode == "gen":
        c = corpus(n)
        hexlines(os.path.join(W, "json.in"), c)
        print("corpus", len(c))
        return
    items = [bytes.fromhex(l.strip()) for l in open(os.path.join(W, "json.in"))] if mode != "gen" else None
    if mode == "refs":
        subprocess.run([PHP, "-d", "memory_limit=3G", os.path.join(D, "php_json.php"), os.path.join(W, "json.in"), os.path.join(W, "json.php.out")], check=True)
        subprocess.run(["node", "--stack-size=4000", os.path.join(D, "node_json.js"), os.path.join(W, "json.in"), os.path.join(W, "json.node.out")], check=True)
        return
    if mode == "compare":
        py = refs(items)
        php = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "json.php.out"))]
        node = [l.rstrip("\n") for l in open(os.path.join(W, "json.node.out"))]
        rust = [l.rstrip("\n").split("\t") for l in open(os.path.join(W, "json.rust.out"))]
        assert len(py) == len(php) == len(node) == len(rust) == len(items), (len(py), len(php), len(node), len(rust), len(items))
        cnt = collections.Counter()
        ex = collections.defaultdict(list)

        def note(k, i, extra=""):
            cnt[k] += 1
            if len(ex[k]) < 6:
                ex[k].append((i, items[i][:80], extra))

        for i, b in enumerate(items):
            st, why, g, c, u16 = py[i]
            rg, rc, rserde = rust[i]
            r_acc = rg.startswith("OK:")
            r_cacc = rc.startswith("OK:")
            cnt["total"] += 1
            cnt["py_accept"] += st == "OK"
            cnt["rust_general_accept"] += r_acc
            cnt["rust_canon_accept"] += r_cacc
            # general mode vs the strict python reference
            if (st == "OK") != r_acc:
                note("GEN accept differs from python reference (py=%s/%s)" % (st, why), i, rg[:30])
            elif r_acc and bytes.fromhex(rg[3:]).decode("utf-8") != g:
                note("GEN output differs from python reference", i)
            # canonical
            py_c_acc = st == "OK" and c is not None
            if py_c_acc != r_cacc:
                note("CANON accept differs from python reference", i, rc[:30])
            elif r_cacc and bytes.fromhex(rc[3:]).decode("utf-8") != c:
                note("CANON output differs from python reference (code point order)", i)
            if py_c_acc and r_cacc and c != u16:
                cnt["canon outputs where UTF-16 (RFC 8785) key order differs from the crate's bytewise order"] += 1
                if len(ex["u16"]) < 4:
                    ex["u16"].append((i, items[i][:100], ""))
            # others
            if rserde == "OK" and not r_acc:
                note("serde accepts, crate general rejects (reason py: %s)" % why, i, rg)
            if rserde == "ERR" and r_acc:
                note("serde rejects, crate general ACCEPTS", i)
            if node[i] == "OK" and not r_acc:
                note("node accepts, crate general rejects (reason py: %s)" % why, i)
            if node[i] == "ERR" and r_acc:
                note("node rejects, crate general ACCEPTS", i)
            if php[i][0] == "OK" and not r_acc:
                note("php json_decode(65) accepts, crate general rejects (reason py: %s)" % why, i)
            if php[i][0] == "ERR" and r_acc:
                note("php json_decode(65) rejects, crate general ACCEPTS", i)
            if php[i][1] == "OK" and not r_acc and why != "dup":
                note("relay Json::decode accepts, crate general rejects (reason py: %s)" % why, i)
        for k in sorted(cnt):
            print(cnt[k], k)
        for k in sorted(ex):
            if k.startswith("serde") or k.startswith("php") or k.startswith("node") or k.startswith("relay") or k.startswith("GEN") or k.startswith("CANON") or k == "u16":
                for e in ex[k]:
                    print("   ", k[:70], e)


if __name__ == "__main__":
    main()
