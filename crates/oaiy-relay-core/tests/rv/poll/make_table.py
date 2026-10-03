"""Builds a poll-client table (the shape of poll-client.json) from my random cases and my implementation's decisions (the flags that mimic the repository's Python reader), so that
verify_poll_client.py and verify_poll_client.mjs can be run on it.  usage: python make_table.py <cases.jsonl> <out.json> [max]"""
import json, sys, hashlib, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_poll as mp
import gen_poll_lib as g

src, out = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
for f in ("lenient_date", "seq_cap", "no_d_on_own"):
    mp.FLAGS[f] = True
K = {"replaceMinMs": 250, "clampMin": 1, "clampMax": 120, "retryAfterDigitsMax": 6, "retryAfterBodyMax": 86400, "jitter": 0.2, "backoff429Cap": 30, "backoffFailureCap": 60,
     "unreachableAfter": 3, "inFlightDefectAfter": 5, "refusedHoldDefaultS": 2, "proofEveryS": 300, "proofAfterPauseS": 60, "pollTimeoutExtraS": 10}
cases = []
for line in open(src, encoding="utf-8"):
    c = json.loads(line)
    if " 0000 " in json.dumps(c.get("headers", {})):
        continue
    if len(cases) >= mx:
        break
    if "proof" in c:
        d = mp.decide_proof({"state": c["state"], "u": c["u"], "proof": {"result": c["proof"]}})
        exp = d
        case = {"id": c["id"], "rule": "P9", "state": c["state"], "info": c["info"] if "info" in c else {"pollGapMs": 250, "fallbackS": 5}, "u": c["u"], "proof": {"result": c["proof"]}}
        case["expect"] = {k: exp[k] for k in ("outcome", "baseS", "pauseS", "state", "action", "report")}
        case["expect"]["since"] = 0
        cases.append(case)
        continue
    body = g.parse_body(c["bodyText"])
    if c["status"] is None:
        resp = {"status": None, "transport": "reset", "headers": {}, "body": None}
        pyresp = {"status": None, "headers": {}, "body": None}
    else:
        resp = {"status": c["status"], "headers": c["headers"], "body": None if body is None else json.loads(c["bodyText"], object_pairs_hook=lambda p: p)}
        pyresp = {"status": c["status"], "headers": c["headers"], "body": body}
    cc = dict(c)
    cc["response"] = pyresp
    d = mp.decide(cc)
    case = {"id": c["id"], "rule": "P2", "state": c["state"], "info": c["info"], "u": c["u"], "since": c["since"], "persisted": c["persisted"], "weReplaced": c["weReplaced"],
            "minClientAboveOurs": c["minClientAboveOurs"], "response": None, "expect": {k: d[k] for k in ("outcome", "baseS", "pauseS", "state", "action", "report", "since")}}
    if c.get("nowEpoch") is not None:
        case["nowEpoch"] = c["nowEpoch"]
    case["_resp_text"] = (c["status"], c["headers"], c["bodyText"] if body is not None else None)
    cases.append(case)

# serialise by hand so that the body keeps its own spelling (1.0, 1e2, -0)
def ser(case):
    c = dict(case)
    rt = c.pop("_resp_text", None)
    s = json.dumps(c)
    if rt is not None:
        status, headers, text = rt
        if status is None:
            r = '{"status":null,"transport":"reset","headers":{},"body":null}'
        else:
            r = '{"status":%d,"headers":%s,"body":%s}' % (status, json.dumps(headers), text if text is not None else "null")
        s = s.replace('"response": null', '"response": ' + r, 1)
    return s

ids = [c["id"] for c in cases]
doc = '{"version":1,"about":"reviewer table","constants":%s,"caseCount":%d,"idsSha256":"%s","layoutSha256":"%s","cases":[\n%s\n]}' % (
    json.dumps(K), len(cases), hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
    hashlib.sha256("\n".join(f"{c['id']}|{c['rule']}" for c in cases).encode()).hexdigest(), ",\n".join(ser(c) for c in cases))
open(out, "w", encoding="utf-8").write(doc)
print(len(cases), "cases written")
