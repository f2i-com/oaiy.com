"""My own implementation of the poll decision function of README 5.1.1 (P1-P9), written from the README text alone.

Input case (dict):
  state {n429,nFail,nRefused,n400}, info {pollGapMs,fallbackS}, u, since (default 0), persisted (default True),
  weReplaced (default True), minClientAboveOurs (default False), nowEpoch (optional),
  response: {status:int|None, transport:str, headers:{lowercase:str}, body: parsed JSON (python) | None, bodyText: str | None}
Output: dict(outcome, baseS, pauseS, state, action, report, since)
"""
import json, re, calendar

B64 = re.compile(r'^[A-Za-z0-9_-]{11}$')
DIG = re.compile(r'^[0-9]{1,6}$')
IMF = re.compile(r'^(Mon|Tue|Wed|Thu|Fri|Sat|Sun), ([0-9]{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) ([0-9]{4}) ([0-9]{2}):([0-9]{2}):([0-9]{2}) GMT$')
MONTHS = {m: i + 1 for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
MAXI = 2**53 - 1
FLAGS = {"lenient_date": False, "seq_cap": False, "no_d_on_own": False}


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def days_from_civil(y, m, d):
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_imf(s):
    if s is None:
        return None
    s = s.strip(" \t")
    m = IMF.match(s)
    if not m:
        return None
    _, d, mon, y, hh, mm, ss = m.groups()
    d, y, hh, mm, ss = int(d), int(y), int(hh), int(mm), int(ss)
    mon = MONTHS[mon]
    if not FLAGS["lenient_date"] and (hh > 23 or mm > 59 or ss > 60):
        return None
    leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
    mdays = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mon - 1]
    if not FLAGS["lenient_date"] and not (1 <= d <= mdays):
        return None
    return days_from_civil(y, mon, d) * 86400 + hh * 3600 + mm * 60 + ss


def retry_after(resp, now_epoch):
    """P6: returns D (int seconds) or None."""
    h = resp.get("headers", {}).get("retry-after")
    if h is not None:
        h = h.strip(" \t")
        if DIG.match(h):
            return int(h)
        t = parse_imf(h)
        if t is not None:
            ref = parse_imf(resp.get("headers", {}).get("date"))
            if ref is None:
                ref = now_epoch
            if ref is not None:
                return max(0, t - ref)
            # no clock at all: unspecified
    body = resp.get("body")
    if isinstance(body, dict):
        e = body.get("error")
        if isinstance(e, dict):
            ra = e.get("retryAfter")
            if is_int(ra) and 0 <= ra <= 86400:
                return ra
    return None


def clamp(x):
    return max(1, min(120, x))


def valid200(body):
    if not isinstance(body, dict):
        return False
    if not isinstance(body.get("items"), list):
        return False
    ep = body.get("epoch")
    if not (isinstance(ep, str) and B64.match(ep)):
        return False
    c = body.get("cursor")
    if not (is_int(c) and 0 <= c <= MAXI):
        return False
    return True


def decide(case):
    st = case["state"]
    n429, nFail, nRef, n400 = st["n429"], st["nFail"], st["nRefused"], st["n400"]
    info = case["info"]
    u = case["u"]
    since = case.get("since", 0)
    persisted = case.get("persisted", True)
    weRepl = case.get("weReplaced", True)
    minAbove = case.get("minClientAboveOurs", False)
    now = case.get("nowEpoch")
    resp = case.get("response")

    def out(outcome, base, state, action=None, report=None, since_after=None):
        o = {"outcome": outcome, "baseS": 0 if base is None else base,
             "pauseS": 0.0 if base is None else base * (1 + 0.2 * u),
             "state": dict(zip(("n429", "nFail", "nRefused", "n400"), state)),
             "action": action, "report": ([] if report is None else [report]),
             "since": since if since_after is None else since_after}
        return o

    zero = (0, 0, 0, 0)
    cur = (n429, nFail, nRef, n400)

    def failure(extra_report=None, own=False, action=None, n400_up=False):
        n = nFail + 1
        base = min(60, 2 ** (min(n, 40) - 1))
        D = retry_after(resp, now) if resp else None
        if FLAGS["no_d_on_own"] and (extra_report in ("invalid_request", "storage_failure")):
            D = None
        if D is not None:
            base = max(base, clamp(D))
        rep = extra_report
        if rep is None and not own and n >= 3:
            rep = "unreachable"
        return out("failure", base, (0, n, 0, n400 + 1 if n400_up else 0), action, rep)

    if resp is None or resp.get("status") is None:
        return failure()
    status = resp["status"]
    body = resp.get("body")

    if status == 200:
        if not valid200(body):
            return failure()
        hold = body.get("hold")
        hold = hold if isinstance(hold, dict) else {}
        # reset
        if body.get("reset") is True:
            if not persisted:
                return failure("storage_failure", own=True)
            return out("progress", 0, zero, None, None, body["cursor"])
        acc = []
        last = since
        for it in body["items"]:
            if isinstance(it, dict) and is_int(it.get("seq")) and it["seq"] > since and it["seq"] > last and (not FLAGS["seq_cap"] or it["seq"] <= MAXI):
                acc.append(it["seq"])
                last = it["seq"]
        if acc:
            if not persisted:
                return failure("storage_failure", own=True)
            return out("progress", 0, zero, None, None, acc[-1])
        if hold.get("superseded") is True:
            if weRepl:
                return out("superseded", 0, zero)
            return out("superseded", info["pollGapMs"] / 1000, zero, None, "duplicate_credential")
        if hold.get("refused") is True:
            ra = hold.get("retryAfter")
            r = clamp(ra) if is_int(ra) else clamp(2)
            F = info["fallbackS"]
            k = min(nRef, 40)
            base = max(r, min(F, r * 2 ** k))
            return out("idle", base, (0, 0, nRef + 1, 0))
        return out("idle", info["pollGapMs"] / 1000, zero)

    if status == 429:
        n = n429 + 1
        D = retry_after(resp, now)
        if D is None:
            D = 1
        base = max(clamp(D), min(30, 2 ** (min(n, 40) - 1)))
        rule = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            rule = body["error"].get("rule")
        action = "cancel_own_polls" if rule == "in_flight" else None
        report = "in_flight_defect" if (rule == "in_flight" and n == 5) else None
        return out("flow", base, (n, 0, 0, 0), action, report)

    if status == 400:
        if n400 == 0:
            return failure("invalid_request", own=True, action="clear_epoch", n400_up=True)
        return out("stop", None, cur, "report_defect")

    if status == 401:
        code = None
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
        return out("stop", None, cur, "forget_credential" if code == "revoked" else "refresh_or_reenrol")

    if status == 426:
        if minAbove:
            return out("stop", None, cur, "update_client")
        return failure()

    if status == 408:
        return failure()
    if 400 <= status <= 499:
        return out("stop", None, cur, "report_defect")
    # 1xx, 2xx other, 3xx, 5xx, anything else
    return failure()


def replace_wait(ms):
    return max(0, 250 - ms)


def proof_due(pd):
    return bool(pd["processStart"] or pd["networkChanged"] or pd["longestPauseS"] >= 60 or pd["secondsSinceProof"] >= 300)


def decide_proof(case):
    st = case["state"]
    n429, nFail, nRef, n400 = st["n429"], st["nFail"], st["nRefused"], st["n400"]
    u = case["u"]
    res = case["proof"]["result"]
    if res == "verified":
        return {"outcome": "proved", "baseS": 0, "pauseS": 0.0,
                "state": {"n429": n429, "nFail": 0, "nRefused": nRef, "n400": n400}, "action": None, "report": []}
    if res == "none":
        n = nFail + 1
        base = min(60, 2 ** (min(n, 40) - 1))
        if case.get("asked") is not None:
            base = max(base, clamp(case["asked"]))
        return {"outcome": "failure", "baseS": base, "pauseS": base * (1 + 0.2 * u),
                "state": {"n429": 0, "nFail": n, "nRefused": 0, "n400": 0}, "action": None,
                "report": ["unreachable"] if n >= 3 else []}
    return {"outcome": "stop", "baseS": 0, "pauseS": 0.0,
            "state": {"n429": n429, "nFail": nFail, "nRefused": nRef, "n400": n400},
            "action": "report_relay_changed", "report": []}




