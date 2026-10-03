#!/usr/bin/env python3
"""Writes a table in the shape of fixtures/poll-client/poll-client.json whose cases are RANDOM and whose expected answers are computed by the repository's own Python
reader of README 5.1.1 (`verify_poll_client.py`, a second implementation written from the README alone).

    python poll_diff_gen.py OUT.json [COUNT] [SEED] [no-integral-floats]

`tests/poll_differential.rs` runs this and then checks every case against this crate's `decide`; the same file is also given to `verify_poll_client.mjs --file`, so a
disagreement between the two readers of the repository on an input the table does not hold would show too. The table's 135 hand-written cases are the contract; this
widens it to the inputs nobody wrote down (a float where an integer goes, a seven-digit Retry-After, a hold that is not an object ...).
"""
import hashlib
import importlib.util
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
READER = HERE / ".." / ".." / ".." / "platform" / "protocol" / "relay" / "v1" / "fixtures" / "poll-client" / "verify_poll_client.py"

spec = importlib.util.spec_from_file_location("verify_poll_client", READER.resolve())
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
# With "no-integral-floats" a float is never one whose value is a whole number (`1.0`, `100.0`): JavaScript cannot tell that spelling from the integer, so the Node reader reads it
# as an integer where the Python reader (and this crate) read a float. See the README of this crate, "Findings".
FLOATS = [0.0, 1.0, 2.5, -1.5, 1e2]
DATES = [
    "Sun, 06 Nov 1994 08:49:37 GMT",
    "Sun, 06 Nov 1994 08:50:37 GMT",
    "Sun, 06 Nov 1994 08:48:37 GMT",
    "Thu, 01 Jan 1970 00:00:00 GMT",
    "Sun, 06 Nov 1994 08:49:37 UTC",
    "Sunday, 06-Nov-94 08:49:37 GMT",
    "Sun, 6 Nov 1994 08:49:37 GMT",
    "Fri, 31 Apr 2021 23:59:59 GMT",
    "Fri, 99 Dec 2021 99:99:99 GMT",
]


def rnd_int(r, around=None):
    k = r.randrange(10)
    if k == 0:
        return 0
    if k == 1:
        return r.randrange(1, 10)
    if k == 2:
        return r.randrange(-5, 1)
    if k == 3:
        return 2 ** 53 - 1
    if k == 4:
        return 2 ** 53
    if k == 5:
        return 10 ** 40
    if k == 6 and around is not None:
        return max(0, around + r.randrange(-3, 4))
    return r.randrange(0, 200)


def rnd_value(r, around=None):
    k = r.randrange(9)
    if k <= 4:
        return rnd_int(r, around)
    if k == 5:
        return r.choice(FLOATS)
    if k == 6:
        return r.choice([True, False, None])
    if k == 7:
        return r.choice(["7", "", "x"])
    return r.choice([[], {}, [1]])


def rnd_epoch(r):
    k = r.randrange(8)
    if k == 0:
        return "".join(r.choice(ALPHABET) for _ in range(r.choice([0, 10, 12])))
    if k == 1:
        return r.choice([5, None, ["x"]])
    return "".join(r.choice(ALPHABET) for _ in range(11))


def rnd_item(r, since):
    it = {"seq": rnd_value(r, since) if r.randrange(4) == 0 else max(0, since + r.randrange(-2, 6)), "id": "i" + str(r.randrange(100)), "lane": "ctl", "from": "relay", "at": 1, "exp": 2, "hdr": {}, "body": "x"}
    if r.randrange(12) == 0:
        del it["seq"]
    return it if r.randrange(15) else r.choice([5, "x", None, [], {"seq": rnd_int(r)}])


def rnd_hold(r):
    k = r.randrange(9)
    if k == 0:
        return {"granted": True}
    if k == 1:
        return {"granted": True, "superseded": True}
    if k == 2:
        return {"refused": True, "retryAfter": rnd_value(r)}
    if k == 3:
        return {"refused": True}
    if k == 4:
        return {"superseded": r.choice([True, False, "true", 1])}
    if k == 5:
        return {"refused": r.choice([True, False, "true", 1]), "retryAfter": rnd_int(r)}
    if k == 6:
        return r.choice([[], "x", 5, None])
    return None


def rnd_200_body(r, since):
    body = {"v": 1, "epoch": rnd_epoch(r), "cursor": rnd_value(r, since) if r.randrange(3) == 0 else rnd_int(r, since), "more": False, "time": 1790000000}
    items = [rnd_item(r, since) for _ in range(r.choice([0, 0, 1, 2, 3, 5]))]
    if r.randrange(10) == 0:
        items.sort(key=lambda i: -i["seq"] if isinstance(i, dict) and isinstance(i.get("seq"), int) else 0)
    body["items"] = items if r.randrange(14) else r.choice([{}, "x", None, 5])
    if r.randrange(5) == 0:
        body["reset"] = r.choice([True, True, False, "true", 1])
    h = rnd_hold(r)
    if h is not None:
        body["hold"] = h
    for k in list(body):
        if r.randrange(25) == 0:
            del body[k]
    return body if r.randrange(30) else r.choice([[], "x", 5, None, True])


def rnd_error_body(r, status):
    k = r.randrange(10)
    if k == 0:
        return None
    if k == 1:
        return r.choice([[], "x", 5, {"error": "x"}, {"error": []}, {"nope": 1}])
    err = {"code": r.choice(["rate_limited", "revoked", "unauthorized", "invalid_request", "forbidden", "unknown", 5]), "message": "m"}
    if r.randrange(2):
        err["retryAfter"] = rnd_value(r)
    rule = r.randrange(5)
    if rule == 0:
        err["rule"] = "gap"
    elif rule == 1:
        err["rule"] = "in_flight"
    elif rule == 2:
        err["rule"] = r.choice(["other", 5, None, ""])
    return {"error": err}


def rnd_headers(r):
    h = {}
    k = r.randrange(9)
    if k == 0:
        h["Retry-After"] = str(rnd_int(r))
    elif k == 1:
        h["retry-after"] = r.choice([" 12\t", "007", "1234567", "-1", "1.5", "", "abc", "0", "120", "121", "999999", "٣"])
    elif k == 2:
        h["Retry-After"] = r.choice(DATES)
    if r.randrange(4) == 0:
        h["Date"] = r.choice(DATES + ["garbage", ""])
    return h


def rnd_case(r, i):
    since = r.choice([0, 0, 1, 5, 100, 2 ** 53 - 1])
    state = {"n429": r.choice([0, 0, 1, 2, 3, 4, 5, 9]), "nFail": r.choice([0, 0, 1, 2, 3, 6, 20]), "nRefused": r.choice([0, 0, 1, 2, 3, 12]), "n400": r.choice([0, 0, 0, 1, 2])}
    info = {"pollGapMs": r.choice([0, 250, 250, 1000, 5000]), "fallbackS": r.choice([1, 5, 5, 30, 60])}
    status = r.choice([200] * 10 + [429] * 4 + [400] * 2 + [401] * 2 + [403, 404, 405, 408, 409, 410, 413, 415, 422, 426, 500, 502, 503, 504, 501, 301, 204, 100, None, None])
    response = {"status": status, "headers": rnd_headers(r)}
    if status is None:
        response["transport"] = r.choice(["refused", "reset", "timeout", "tls", "short body"])
    elif status == 200:
        response["body"] = rnd_200_body(r, since)
    else:
        response["body"] = rnd_error_body(r, status)
    case = {"id": f"r{i:05d}", "rule": f"P{2 + i % 7}", "what": "a random case", "state": state, "info": info, "response": response, "u": r.choice([0.0, 0.5, 0.999999, r.random()]), "since": since}
    if r.randrange(5) == 0:
        case["persisted"] = r.choice([True, False])
    if r.randrange(5) == 0:
        case["weReplaced"] = r.choice([True, False])
    if r.randrange(4) == 0:
        case["minClientAboveOurs"] = r.choice([True, False])
    if r.randrange(3) == 0:
        case["nowEpoch"] = r.choice([784111777, 784111700, 0, 1790000000])
    return case


def main():
    out = pathlib.Path(sys.argv[1])
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if len(sys.argv) > 4 and sys.argv[4] == "no-integral-floats":
        global FLOATS
        FLOATS = [2.5, -1.5, 0.5, 1e-3]
    r = random.Random(seed)
    decide, _, _ = reader.make(dict(reader.EXPECTED))
    cases = []
    for i in range(count):
        c = rnd_case(r, i)
        got = decide(c)
        c["expect"] = {"outcome": got["outcome"], "baseS": got["baseS"], "pauseS": got["pauseS"], "state": got["state"], "action": got["action"], "report": got["report"], "since": got["since"]}
        cases.append(c)
    # The proof cases and the cases the readers require (a case for every rule), kept from the real table so that the file passes the readers' own guards.
    real = json.loads((READER.parent / "poll-client.json").read_text(encoding="utf-8"))
    keep = [c for c in real["cases"] if "proof" in c or "proofDue" in c or "replace" in c]
    for c in keep:
        c = dict(c)
        c["id"] = "k-" + c["id"]
        cases.append(c)
    for rule in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"):
        if not any(c["rule"] == rule for c in cases):
            raise SystemExit("no case for " + rule)
    ids = [c["id"] for c in cases]
    doc = {
        "version": real["version"],
        "about": "random cases, expectations from verify_poll_client.py",
        "constants": real["constants"],
        "caseCount": len(cases),
        "idsSha256": reader.ids_digest(ids),
        "layoutSha256": reader.layout_digest(cases),
        "cases": cases,
    }
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} cases written to {out}")


if __name__ == "__main__":
    main()
