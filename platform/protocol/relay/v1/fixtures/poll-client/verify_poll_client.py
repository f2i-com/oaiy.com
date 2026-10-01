#!/usr/bin/env python3
"""Reads poll-client.json against the rules of README section 5.1.1 (the poll loop of a native client: DK-03 and MOB-21a), written here from
the README and from nothing else: `decide` below is a second implementation of those rules, and `verify_poll_client.mjs` is a third, in another
language; a client's own code is a fourth, tested against the same table.

    python verify_poll_client.py [--file poll-client.json]

What it enforces beyond each case: the `constants` block of the table must be exactly the numbers the README states (EXPECTED below), and every
rule below takes its number from that block, so a table whose constants were changed is refused and so is a reader that ignores them; the
`caseCount`, `idsSha256` and `layoutSha256` of the table must match the cases it holds, so a case that went missing from it, or was relabelled or
moved, is noticed (the conformance suite pins all three, and checks EXPECTED below against the README's own text, so that a table edited to
agree with itself, or a reader and a table that agree on a number the README does not state, are noticed too).

Prints "N checks, M mismatches" and exits 1 on any mismatch.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
IMF = re.compile(r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (\d\d) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4}) (\d\d):(\d\d):(\d\d) GMT")

# The numbers README section 5.1.1 states (P1, P3, P5, P6, P7, P9). The table's own `constants` must equal this.
EXPECTED = {
    "replaceMinMs": 250, "clampMin": 1, "clampMax": 120, "retryAfterBodyMax": 86400, "retryAfterDigitsMax": 6, "jitter": 0.2,
    "backoff429Cap": 30, "backoffFailureCap": 60, "unreachableAfter": 3, "inFlightDefectAfter": 5, "refusedHoldDefaultS": 2,
    "proofEveryS": 300, "proofAfterPauseS": 60, "pollTimeoutExtraS": 10,
}
ZERO = {"n429": 0, "nFail": 0, "nRefused": 0, "n400": 0}


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


EPOCH = re.compile(r"[A-Za-z0-9_-]{11}")  # 8 bytes, base64url (common.schema.json)
MAX_SAFE = 2 ** 53 - 1  # the largest cursor and seq (uint53)


def valid_200(body) -> bool:
    """README P2: a 200 is valid when its body is an object whose items is an array, whose epoch is 11 base64url characters and whose cursor is an integer from 0 to 2^53 - 1."""
    return (isinstance(body, dict) and isinstance(body.get("items"), list) and isinstance(body.get("epoch"), str) and EPOCH.fullmatch(body["epoch"]) is not None
            and is_int(body.get("cursor")) and 0 <= body["cursor"] <= MAX_SAFE)


def accepted_seqs(items, since):
    """README P2: an item is accepted when its seq is an integer above the since the poll carried and above the seq accepted before it in the answer."""
    last, got = since, []
    for it in items:
        s = it.get("seq") if isinstance(it, dict) else None
        if is_int(s) and last < s <= MAX_SAFE:
            got.append(s)
            last = s
    return got


def http_date(text: str):
    m = IMF.fullmatch(text.strip(" \t"))
    if not m:
        return None
    d, mon, y, hh, mm, ss = m.groups()
    return calendar.timegm((int(y), MONTHS[mon], int(d), int(hh), int(mm), int(ss)))


def make(K: dict):
    """The rules, with every number taken from the constants K."""

    def clamp(x):
        return max(K["clampMin"], min(K["clampMax"], x))

    def retry_after(headers, body, now):
        v = headers.get("retry-after")
        if v is not None:
            s = v.strip(" \t")
            if re.fullmatch(r"[0-9]{1,%d}" % K["retryAfterDigitsMax"], s):
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
            if is_int(r) and 0 <= r <= K["retryAfterBodyMax"]:
                return r
        return None

    def decide(c):
        st = c["state"]
        n429, nfail, nref, n400 = st["n429"], st["nFail"], st["nRefused"], st["n400"]
        info, u = c["info"], c["u"]
        since, persisted = c.get("since", 0), c.get("persisted", True)
        out = {"action": None, "report": [], "since": since}

        def result(outcome, base, state):
            out.update(outcome=outcome, baseS=base, pauseS=base * (1 + K["jitter"] * u), state=state)
            return out

        def fail_backoff(n, d=None):
            base = min(K["backoffFailureCap"], 2 ** (n - 1))
            return max(base, clamp(d)) if d is not None else base

        if "proof" in c:  # P9: the answer to the interactive proof of section 8.3
            res = c["proof"]["result"]
            if res == "verified":
                return result("proved", 0, {**st, "nFail": 0})
            if res == "none":
                n = nfail + 1
                if n >= K["unreachableAfter"]:
                    out["report"] = ["unreachable"]
                return result("failure", fail_backoff(n), {"n429": 0, "nFail": n, "nRefused": 0, "n400": 0})
            out["action"] = "report_relay_changed"  # a replayed body, another key
            return result("stop", 0, dict(st))
        resp = c["response"]
        status = resp["status"]
        headers = {k.lower(): v for k, v in resp.get("headers", {}).items()}
        body = resp.get("body")
        now = c.get("nowEpoch")
        cleared = dict(ZERO)
        if status == 200 and valid_200(body):
            hold = body.get("hold") if isinstance(body.get("hold"), dict) else {}
            if body.get("reset") is True:
                adopted = body["cursor"]  # once, and it may be lower than the since the client had
            else:
                got = accepted_seqs(body["items"], since)
                adopted = got[-1] if got else None
            if adopted is not None:  # progress: first in the table, so a refused or superseded answer that carries an accepted item is this
                if not persisted:  # what was accepted could not be written: nothing advances, and the failure is paced like any other
                    out["report"] = ["storage_failure"]
                    n = nfail + 1
                    return result("failure", fail_backoff(n), {**cleared, "nFail": n})
                out["since"] = adopted
                return result("progress", 0, cleared)
            if hold.get("superseded") is True:
                if c.get("weReplaced", True) is False:
                    out["report"] = ["duplicate_credential"]
                    return result("superseded", info["pollGapMs"] / 1000, cleared)  # another process polls: pause as after idle
                return result("superseded", 0, cleared)
            if hold.get("refused") is True:
                r = hold.get("retryAfter")
                r = clamp(r if is_int(r) else K["refusedHoldDefaultS"])
                base = max(r, min(info["fallbackS"], r * 2 ** nref))
                return result("idle", base, {**cleared, "nRefused": nref + 1})
            return result("idle", info["pollGapMs"] / 1000, cleared)
        if status == 429:
            n = n429 + 1
            d = retry_after(headers, body, now)
            d = clamp(1 if d is None else d)
            base = max(d, min(K["backoff429Cap"], 2 ** (n - 1)))
            rule = body["error"].get("rule") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
            if rule == "in_flight":
                out["action"] = "cancel_own_polls"
                if n == K["inFlightDefectAfter"]:
                    out["report"] = ["in_flight_defect"]
            return result("flow", base, {**cleared, "n429": n})
        if status == 400:
            if n400 >= 1:  # a second in a row
                out["action"] = "report_defect"
                return result("stop", 0, dict(st))
            n = nfail + 1  # the first: retried without the stored epoch
            out["action"] = "clear_epoch"
            out["report"] = ["invalid_request"]
            return result("failure", fail_backoff(n), {**cleared, "nFail": n, "n400": 1})
        if status == 426 and c.get("minClientAboveOurs") is True:
            out["action"] = "update_client"
            return result("stop", 0, dict(st))
        if status is not None and 400 <= status <= 499 and status not in (408, 426, 429):
            code = body["error"].get("code") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
            if status == 401:
                out["action"] = "forget_credential" if code == "revoked" else "refresh_or_reenrol"
            else:
                out["action"] = "report_defect"
            return result("stop", 0, dict(st))
        # everything else is a failure: no answer, 408, 5xx, 1xx, 2xx other than a valid 200, 3xx, an invalid 200, a 426 that is not ours to obey
        n = nfail + 1
        d = retry_after(headers, body, now) if status is not None else None
        if n >= K["unreachableAfter"]:
            out["report"] = ["unreachable"]
        return result("failure", fail_backoff(n, d), {**cleared, "nFail": n})

    def replace_wait_ms(since_last_start_ms):
        return max(0, K["replaceMinMs"] - since_last_start_ms)

    def proof_due(p):
        return bool(p.get("processStart") or p.get("networkChanged") or p["longestPauseS"] >= K["proofAfterPauseS"] or p["secondsSinceProof"] >= K["proofEveryS"])

    return decide, replace_wait_ms, proof_due


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def layout_digest(cases) -> str:
    """The lines `id|rule` in the table's own order: which rule each case says it checks, and in what order."""
    return hashlib.sha256("\n".join(f"{c['id']}|{c['rule']}" for c in cases).encode("utf-8")).hexdigest()


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

    K = doc["constants"]
    check("constants", "the constants block is the numbers of README 5.1.1", K == EXPECTED, f"table {K}, README {EXPECTED}")
    ids = [c["id"] for c in doc["cases"]]
    check("caseCount", "the table holds the number of cases it says", doc.get("caseCount") == len(ids), f"says {doc.get('caseCount')}, holds {len(ids)}")
    check("idsSha256", "the digest of the case names is the one the table says (a case went missing, or was added)", doc.get("idsSha256") == ids_digest(ids))
    check("layoutSha256", "the digest of the case names with their rule labels, in the table's order, is the one the table says (a case was relabelled or moved)",
          doc.get("layoutSha256") == layout_digest(doc["cases"]))
    decide, replace_wait_ms, proof_due = make({**EXPECTED, **{k: v for k, v in K.items() if k in EXPECTED}})
    seen = set()
    for c in doc["cases"]:
        check(c["id"], "id is unique", c["id"] not in seen)
        seen.add(c["id"])
        if "replace" in c:
            wait = replace_wait_ms(c["replace"]["msSinceLastStart"])
            check(c["id"], "waitMs", wait == c["expect"]["waitMs"], f"got {wait}, table {c['expect']['waitMs']}")
            continue
        if "proofDue" in c:
            got = proof_due(c["proofDue"])
            check(c["id"], "proof due", got == c["expect"]["due"], f"got {got}, table {c['expect']['due']}")
            continue
        got = decide(c)
        want = c["expect"]
        check(c["id"], "outcome", got["outcome"] == want["outcome"], f"got {got['outcome']}, table {want['outcome']}")
        check(c["id"], "baseS", abs(got["baseS"] - want["baseS"]) < 1e-9, f"got {got['baseS']}, table {want['baseS']}")
        check(c["id"], "pauseS", abs(got["pauseS"] - want["pauseS"]) < 1e-6, f"got {got['pauseS']}, table {want['pauseS']}")
        check(c["id"], "state", got["state"] == want["state"], f"got {got['state']}, table {want['state']}")
        check(c["id"], "action", got["action"] == want["action"], f"got {got['action']}, table {want['action']}")
        check(c["id"], "report", sorted(got["report"]) == sorted(want["report"]), f"got {got['report']}, table {want['report']}")
        if "since" in want:
            check(c["id"], "since", got["since"] == want["since"], f"got {got['since']}, table {want['since']}")
    # every rule of section 5.1.1 that a case can check has a case
    rules = {c["rule"] for c in doc["cases"]}
    for rule in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"):
        check(rule, "has a case", rule in rules)
    for line in bad:
        print("MISMATCH", line)
    print(f"{checks} checks, {len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
