"""Random cases for the poll-decision differential: writes cases.jsonl (inputs) and mine.jsonl (my independent implementation's decisions).
usage: python gen_poll.py <count> <seed> <outdir>
"""
import json, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_poll as mp

count = int(sys.argv[1]); seed = int(sys.argv[2]); outdir = sys.argv[3]
for f in os.environ.get("RV_FLAGS", "").split(","):
    if f:
        mp.FLAGS[f] = True
os.makedirs(outdir, exist_ok=True)
R = random.Random(seed)

B64C = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def pick(*opts):
    """opts: (weight, value or callable)"""
    tot = sum(w for w, _ in opts)
    x = R.random() * tot
    for w, v in opts:
        x -= w
        if x <= 0:
            return v() if callable(v) else v
    return (opts[-1][1])()


def epoch():
    return pick((80, lambda: "".join(R.choice(B64C) for _ in range(11))),
                (3, lambda: "".join(R.choice(B64C) for _ in range(10))),
                (3, lambda: "".join(R.choice(B64C) for _ in range(12))),
                (2, lambda: "".join(R.choice(B64C) for _ in range(10)) + "+"),
                (2, lambda: "".join(R.choice(B64C) for _ in range(10)) + "="),
                (1, lambda: "AAAAAAAAAAÃ©"),
                (1, lambda: ""))


def jstr(s):
    return json.dumps(s, ensure_ascii=R.random() < 0.5)


def intlex(n):
    return str(n)


def cursor_lex():
    return pick((60, lambda: str(R.randint(0, 1000))),
                (6, lambda: str(R.choice([0, 2**53 - 1, 2**53 - 2]))),
                (3, "9007199254740992"), (2, "9007199254740993"), (2, "18446744073709551616"),
                (2, "123456789012345678901234567890123456789012345"),
                (3, lambda: str(-R.randint(1, 50))),
                (3, lambda: R.choice(["1.0", "5.0", "0.0", "100.0", "1e2", "1E2", "1e0", "5e-1", "1.5", "-0", "-0.0", "0e0", "12.50", "1e400"])),
                (2, '"7"'), (1, "true"), (1, "null"), (1, "[]"), (1, "{}"))


def seq_lex():
    return pick((70, lambda: str(R.randint(0, 40))),
                (6, lambda: str(R.choice([2**53 - 1, 2**53, 2**53 + 1, 2**63, 2**64]))),
                (4, lambda: R.choice(["1.0", "5.0", "7e0", "6.5", "-0", "-1", "0.0"])),
                (3, '"5"'), (2, "true"), (2, "null"), (1, "[]"), (1, "{}"))


def item_lex():
    def obj():
        if R.random() < 0.06:
            return "{}"
        parts = []
        if R.random() < 0.95:
            parts.append('"seq":' + seq_lex())
        if R.random() < 0.5:
            parts.append('"id":' + jstr("i" + str(R.randint(0, 9))))
        R.shuffle(parts)
        return "{" + ",".join(parts) + "}"
    return pick((90, obj), (3, "null"), (2, "5"), (2, '"x"'), (2, "[]"), (1, "true"))


def items_lex(since):
    kind = R.random()
    n = pick((20, 0), (35, 1), (25, lambda: R.randint(2, 4)), (10, lambda: R.randint(5, 9)), (3, 64))
    out = []
    if kind < 0.55:
        # ascending around since
        s = since + R.randint(-3, 3)
        for _ in range(n):
            s += R.choice([1, 1, 1, 2, 0, -1, 5])
            out.append('{"seq":%d}' % max(s, -2))
    else:
        out = [item_lex() for _ in range(n)]
    return "[" + ",".join(out) + "]"


def hold_lex():
    return pick((35, None),
                (12, '{"granted":true}'),
                (14, '{"superseded":true}'),
                (4, '{"superseded":false}'),
                (4, '{"superseded":"true"}'),
                (4, '{"superseded":1}'),
                (6, '{"refused":true}'),
                (12, lambda: '{"refused":true,"retryAfter":%s}' % R.choice(["0", "1", "2", "3", "7", "30", "119", "120", "121", "600", "-4", "1.0", "2.5", '"3"', "null", "true", "99999999999999999999999", "86400"])),
                (3, '{"refused":false,"retryAfter":9}'),
                (3, '{"refused":"yes"}'),
                (2, '{"superseded":true,"refused":true,"retryAfter":4}'),
                (2, "[]"), (2, "5"), (1, "null"), (1, '"x"'), (2, "{}"))


def body200(since):
    ep = epoch()
    parts = []
    cur = cursor_lex()
    if R.random() < 0.97:
        parts.append('"epoch":' + jstr(ep))
    if R.random() < 0.97:
        parts.append('"cursor":' + cur)
    if R.random() < 0.96:
        parts.append('"items":' + items_lex(since))
    elif R.random() < 0.5:
        parts.append('"items":' + pick((1, "null"), (1, "{}"), (1, '"x"'), (1, "5")))
    h = hold_lex()
    if h is not None:
        parts.append('"hold":' + h)
    r = pick((70, None), (10, "true"), (6, "false"), (2, '"true"'), (2, "1"), (1, "null"))
    if r is not None:
        parts.append('"reset":' + r)
    if R.random() < 0.3:
        parts.append('"v":1')
    if R.random() < 0.2:
        parts.append('"time":1790000000')
    if R.random() < 0.15:
        parts.append('"more":' + R.choice(["true", "false"]))
    R.shuffle(parts)
    return "{" + ",".join(parts) + "}"


def retry_after_header():
    def date(bad=False):
        d = R.choice(DAYS); day = R.randint(1, 28); m = R.choice(MONS); y = R.randint(1990, 2040)
        s = "%s, %02d %s %04d %02d:%02d:%02d GMT" % (d, day, m, y, R.randint(0, 23), R.randint(0, 59), R.randint(0, 59))
        return s
    return pick((25, lambda: str(R.randint(0, 130))),
                (6, lambda: str(R.randint(0, 999999))),
                (4, lambda: str(R.randint(1000000, 99999999))),
                (4, lambda: R.choice([" ", "\t", " \t"]) + str(R.randint(0, 200)) + R.choice([" ", "\t", ""])),
                (4, lambda: "0" * R.randint(1, 6) + str(R.randint(0, 99))),
                (4, lambda: R.choice(["+5", "-1", "1.5", "", " ", "abc", "5 6", "1e2", "0x10", "Ù£", "ï¼•", "5s", "5,6", "Ù£"])),
                (12, date),
                (3, lambda: date().replace("GMT", "UTC")),
                (2, lambda: date().replace(",", "")),
                (2, lambda: "Sunday, 06-Nov-94 08:49:37 GMT"),
                (2, lambda: "Sun Nov  6 08:49:37 1994"),
                (2, lambda: date()[:-1]),
                (2, lambda: date() + " x"),
                (2, lambda: "Sun, 31 Feb 1994 08:49:37 GMT"),
                (2, lambda: "Sun, 06 Nov 1994 25:49:37 GMT"),
                (2, lambda: "Sun, 06 Nov 1994 08:61:37 GMT"),
                (2, lambda: "Sun, 06 Nov 1994 08:49:60 GMT"),
                (2, lambda: "Sun, 06 Nov 1994 08:49:99 GMT"),
                (2, lambda: "Sun, 32 Nov 1994 08:49:37 GMT"),
                (2, lambda: "Foo, 06 Nov 1994 08:49:37 GMT"),
                (2, lambda: "Mon, 06 Nov 1994 08:49:37 GMT"),   # wrong weekday for that date
                (2, lambda: "Sun, 06 Nov 0000 08:49:37 GMT"),
                (2, lambda: "Sun, 06 nov 1994 08:49:37 GMT"),
                (2, lambda: "Sun,  06 Nov 1994 08:49:37 GMT"),
                (1, lambda: "Sun, 06 Nov 1994 08:49:37 gmt"))


def headers():
    h = {}
    if R.random() < 0.7:
        h["retry-after"] = retry_after_header()
    if R.random() < 0.25:
        h["date"] = pick((6, lambda: "Sun, 06 Nov 1994 08:49:37 GMT"), (3, lambda: "Mon, 07 Nov 2033 01:02:03 GMT"),
                         (2, "garbage"), (1, ""), (1, lambda: "Sun, 99 Nov 1994 08:49:37 GMT"))
    if R.random() < 0.05:
        h["location"] = "https://evil.example/"
    return h


def err_body(code=None, rule=None, with_ra=True):
    parts = []
    parts.append('"code":' + jstr(code or R.choice(["rate_limited", "revoked", "forbidden", "x"])))
    parts.append('"message":"m"')
    if with_ra and R.random() < 0.6:
        parts.append('"retryAfter":' + pick((50, lambda: str(R.randint(0, 200))), (10, "86400"), (6, "86401"), (4, "-1"), (4, "1.0"), (4, '"5"'),
                                           (3, "null"), (2, "true"), (2, "1e2"), (2, "99999999999999999999999")))
    if rule is not None:
        parts.append('"rule":' + (rule if rule.startswith('"') or rule in ("null", "true", "1") else jstr(rule)))
    R.shuffle(parts)
    return '{"error":{' + ",".join(parts) + "}}"


def body_any(kind):
    if kind == "429":
        r = R.random()
        if r < 0.3:
            return err_body(rule=jstr("in_flight"))
        if r < 0.5:
            return err_body(rule=jstr("gap"))
        if r < 0.62:
            return err_body(rule=pick((1, "null"), (1, "1"), (1, "true"), (1, '"IN_FLIGHT"'), (1, '"other"')))
        if r < 0.8:
            return err_body()
        if r < 0.85:
            return "nope"
        if r < 0.9:
            return ""
        if r < 0.93:
            return '{"error":"x"}'
        if r < 0.96:
            return '{"error":{"rule":"in_flight","rule":"gap"}}'   # duplicate key: strict parser refuses
        return "[]"
    if kind == "401":
        return pick((40, lambda: err_body(code="revoked")), (30, lambda: err_body(code="unauthorized")), (10, lambda: err_body(code='"revoked"')),
                    (5, "nope"), (5, ""), (5, '{"error":{"code":"Revoked"}}'), (5, '{"error":{"code":5}}'))
    return pick((50, lambda: err_body()), (20, ""), (10, "nope"), (10, "{}"), (5, "[]"), (5, "null"))


def case(i):
    since = pick((40, 0), (40, lambda: R.randint(0, 40)), (10, lambda: R.randint(0, 2**53 - 1)), (4, 2**53 - 1), (2, 2**53 - 5), (4, lambda: R.randint(0, 3)))
    st = {k: pick((50, 0), (30, lambda: R.randint(0, 6)), (10, lambda: R.randint(7, 40)), (5, lambda: R.choice([29, 30, 31, 32, 33, 34, 63, 64, 65, 100, 1000])), (5, lambda: R.randint(0, 3)))
          for k in ("n429", "nFail", "nRefused", "n400")}
    c = {"id": "r%d" % i, "state": st,
         "info": {"pollGapMs": pick((60, 250), (10, 0), (10, 5000), (20, lambda: R.randint(0, 5000))),
                  "fallbackS": pick((60, 5), (20, lambda: R.randint(1, 60)), (5, lambda: R.randint(0, 100)))},
         "u": pick((60, lambda: R.random()), (10, 0.0), (10, 0.9999999999999999), (10, 0.5), (10, lambda: R.choice([0.1, 0.25, 0.75]))),
         "since": since,
         "persisted": R.random() < 0.9,
         "weReplaced": R.random() < 0.7,
         "minClientAboveOurs": R.random() < 0.5,
         "nowEpoch": pick((60, None), (25, lambda: R.randint(0, 2_000_000_000)), (5, lambda: R.randint(-2**40, 2**40)), (3, 784111777), (3, 784111787), (2, 2**62), (2, -2**62))}
    k = R.random()
    if k < 0.04:
        c["proof"] = R.choice(["verified", "none", "invalid"])
        c["asked"] = pick((60, None), (40, lambda: R.randint(0, 100000)))
        return c
    if k < 0.12:
        c["status"] = None
        c["headers"] = {}
        c["bodyText"] = None
        return c
    s = pick((38, 200), (14, 429), (4, 400), (4, 401), (3, 403), (2, 404), (2, 408), (3, 426), (3, 500), (2, 502), (3, 503), (2, 504), (2, 599),
             (2, lambda: R.choice([100, 101, 199, 204, 206, 301, 302, 304, 307, 308])), (2, lambda: R.choice([405, 409, 410, 413, 415, 422, 451, 499])),
             (1, lambda: R.choice([0, 99, 600, 999, 65535])))
    c["status"] = s
    c["headers"] = headers()
    if s == 200:
        c["bodyText"] = pick((88, lambda: body200(since)), (3, "nope"), (2, ""), (2, "[]"), (2, "null"), (1, "{}"), (1, '{"items":[]}'),
                             (1, '{"items":[],"epoch":"AAAAAAAAAAA","cursor":1,"cursor":2}'), (1, '{"epoch":"AAAAAAAAAAA","cursor":1,"items":[],'))
    elif s == 429:
        c["bodyText"] = body_any("429")
    elif s == 401:
        c["bodyText"] = body_any("401")
    else:
        c["bodyText"] = body_any("other")
    return c


def parse_body(text):
    if text is None:
        return None

    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("dup")
            d[k] = v
        return d

    def bad(x):
        raise ValueError("const")
    try:
        return json.loads(text, object_pairs_hook=hook, parse_constant=bad, parse_int=(lambda s: -0.0 if (s == "-0" and mp.FLAGS.get("neg_zero_not_int")) else int(s)))
    except Exception:
        return None


def mine(c):
    if "proof" in c:
        d = mp.decide_proof({"state": c["state"], "u": c["u"], "proof": {"result": c["proof"]}, "asked": c.get("asked")})
        return d
    r = {"status": c["status"], "headers": c["headers"], "body": parse_body(c["bodyText"])}
    cc = dict(c)
    cc["response"] = r
    return mp.decide(cc)


with open(os.path.join(outdir, "cases.jsonl"), "w", encoding="utf-8") as fc, open(os.path.join(outdir, "mine.jsonl"), "w", encoding="utf-8") as fm:
    for i in range(count):
        c = case(i)
        fc.write(json.dumps(c, ensure_ascii=True) + "\n")
        d = mine(c)
        d = dict(d)
        d["id"] = c["id"]
        fm.write(json.dumps(d) + "\n")
print("wrote", count)



