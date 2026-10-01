#!/usr/bin/env python3
"""Reads poll-client.json against the rules of README section 5.1.1 (the poll loop of a native client: DK-03 and MOB-21a), written here from
the README and from nothing else: `decide` below is a second implementation of those rules, and `verify_poll_client.mjs` is a third, in another
language; a client's own code is a fourth, tested against the same table.

    python verify_poll_client.py [--file poll-client.json]

Prints "N checks, M mismatches" and exits 1 on any mismatch.
"""
from __future__ import annotations

import calendar
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
IMF = re.compile(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d\d) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d\d):(\d\d):(\d\d) GMT")


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def clamp(x: float) -> float:
    return max(1, min(120, x))


def http_date(text: str):
    m = IMF.fullmatch(text.strip(" \t"))
    if not m:
        return None
    d, mon, y, hh, mm, ss = m.groups()
    return calendar.timegm((int(y), MONTHS[mon], int(d), int(hh), int(mm), int(ss)))


def retry_after(headers: dict, body, now):
    """P6: the header first (digits, or an HTTP-date measured from the answer's Date header or else the client's clock), then error.retryAfter, else None."""
    v = headers.get("retry-after")
    if v is not None:
        s = v.strip(" \t")
        if re.fullmatch(r"[0-9]{1,6}", s):
            return int(s)
        t = http_date(s)
        if t is not None:
            ref = http_date(headers["date"]) if "date" in headers else None
            if ref is None:
                ref = now
            if ref is not None:
                return max(0, t - ref)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        r = body["error"].get("retryAfter")
        if is_int(r) and 0 <= r <= 86400:
            return r
    return None


def decide(c: dict) -> dict:
    st = c["state"]
    n429, nfail, nref = st["n429"], st["nFail"], st["nRefused"]
    info, u = c["info"], c["u"]
    resp = c["response"]
    status = resp["status"]
    headers = {k.lower(): v for k, v in resp.get("headers", {}).items()}
    body = resp.get("body")
    now = c.get("nowEpoch")
    out = {"action": None, "report": []}

    def result(outcome, base, state):
        out.update(outcome=outcome, baseS=base, pauseS=base * (1 + 0.2 * u), state=state)
        return out

    items_ok = status == 200 and isinstance(body, dict) and isinstance(body.get("items"), list)
    if status == 200 and items_ok:
        hold = body.get("hold") if isinstance(body.get("hold"), dict) else {}
        if body["items"] or body.get("reset") is True:
            return result("progress", 0, {"n429": 0, "nFail": 0, "nRefused": 0})
        if hold.get("superseded") is True:
            if c.get("weReplaced", True) is False:
                out["report"] = ["duplicate_credential"]
                return result("superseded", info["pollGapMs"] / 1000, {"n429": 0, "nFail": 0, "nRefused": 0})  # another process polls: pause as after idle
            return result("superseded", 0, {"n429": 0, "nFail": 0, "nRefused": 0})
        if hold.get("refused") is True:
            r = hold.get("retryAfter")
            r = clamp(r if is_int(r) else 2)
            base = max(r, min(info["fallbackS"], r * 2 ** nref))
            return result("idle", base, {"n429": 0, "nFail": 0, "nRefused": nref + 1})
        return result("idle", info["pollGapMs"] / 1000, {"n429": 0, "nFail": 0, "nRefused": 0})
    if status == 429:
        n = n429 + 1
        d = retry_after(headers, body, now)
        d = clamp(1 if d is None else d)
        base = max(d, min(30, 2 ** (n - 1)))
        rule = body["error"].get("rule") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
        if rule == "in_flight" and n == 5:
            out["report"] = ["in_flight_defect"]
        return result("flow", base, {"n429": n, "nFail": 0, "nRefused": 0})
    stop = status is not None and 400 <= status <= 499 and status not in (408, 429)
    if stop:
        code = body["error"].get("code") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
        if status == 401:
            out["action"] = "forget_credential" if code == "revoked" else "refresh_or_reenrol"
        elif status == 426:
            out["action"] = "update_client"
        else:
            out["action"] = "report_defect"
        return result("stop", 0, dict(st))
    # everything else is a failure: no answer, 408, 5xx, 1xx, 2xx other than a valid 200, 3xx, an invalid 200
    n = nfail + 1
    base = min(60, 2 ** (n - 1))
    d = retry_after(headers, body, now) if status is not None else None
    if d is not None:
        base = max(base, clamp(d))
    if n >= 3:
        out["report"] = ["unreachable"]
    return result("failure", base, {"n429": 0, "nFail": n, "nRefused": 0})


def main() -> int:
    path = HERE / "poll-client.json"
    if "--file" in sys.argv:
        path = pathlib.Path(sys.argv[sys.argv.index("--file") + 1])
    doc = json.loads(path.read_text(encoding="utf-8"))
    checks = 0
    bad: list[str] = []

    def check(cid: str, what: str, ok: bool, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not ok:
            bad.append(f"{cid}: {what} {detail}")

    ids = set()
    for c in doc["cases"]:
        check(c["id"], "id is unique", c["id"] not in ids)
        ids.add(c["id"])
        if "replace" in c:
            wait = max(0, doc["constants"]["replaceMinMs"] - c["replace"]["msSinceLastStart"])
            check(c["id"], "waitMs", wait == c["expect"]["waitMs"], f"got {wait}, table {c['expect']['waitMs']}")
            continue
        got = decide(c)
        want = c["expect"]
        check(c["id"], "outcome", got["outcome"] == want["outcome"], f"got {got['outcome']}, table {want['outcome']}")
        check(c["id"], "baseS", abs(got["baseS"] - want["baseS"]) < 1e-9, f"got {got['baseS']}, table {want['baseS']}")
        check(c["id"], "pauseS", abs(got["pauseS"] - want["pauseS"]) < 1e-6, f"got {got['pauseS']}, table {want['pauseS']}")
        check(c["id"], "state", got["state"] == want["state"], f"got {got['state']}, table {want['state']}")
        check(c["id"], "action", got["action"] == want["action"], f"got {got['action']}, table {want['action']}")
        check(c["id"], "report", sorted(got["report"]) == sorted(want["report"]), f"got {got['report']}, table {want['report']}")
    # every rule of section 5.1.1 that a case can check has a case
    seen = {c["rule"] for c in doc["cases"]}
    for rule in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        check(rule, "has a case", rule in seen)
    for line in bad:
        print("MISMATCH", line)
    print(f"{checks} checks, {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
