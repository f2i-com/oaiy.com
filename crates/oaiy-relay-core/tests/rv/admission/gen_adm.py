"""Random damaged copies of the recorded Aokie admissions, with the verdict of the repository's Python port of the shipped decoders (aokie_decoders.py).
usage: python gen_adm.py <count per case> <seed> <outdir>   writes cases.jsonl and python.jsonl"""
import copy, json, random, sys, os, pathlib

AOKIE = pathlib.Path(r"E:\repos\oaiy-relay-core\platform\protocol\relay\v1\fixtures\aokie")
sys.path.insert(0, str(AOKIE))
import aokie_decoders as D

count, seed, outdir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
R = random.Random(seed)
adm = json.loads((AOKIE / "admission.json").read_text(encoding="utf-8"))
NOW = adm["relay"]["clock"]
cases = [c for c in adm["cases"] if c["response"]["status"] == 200]

POOL = [None, "", " ", "x", "Bearer", "bearer", "mobile", "plugin", 0, 1, -1, 2**53, 2**63, 2**64, 1.5, 90.0, 300, 301, 0.0, True, False, [], {}, [""], ["state_read"], ["state_read", "state_read"],
        ["state_read", "takeover"], ["nope"], "https://relay.example.com", "http://relay.example.com", "https://user@relay.example.com/x", "https://relay.example.com/x#f",
        "wss://relay.example.com/v2/realtime", "ws://relay.example.com/v2/realtime", "wss://relay.example.com/v2/other", "wss://relay.example.com:8443/v2/realtime",
        "https://other.example.com/v1/aokie-companion/relay/frames", "HTTPS://RELAY.EXAMPLE.COM/v1/x", "https://relay.example.com:443/v1/x", "https://relay.example.com:0443/v1/x",
        "turn:turn.example.com:3478", "stun:stun.example.com:3478", "a" * 130, "a" * 5000, "\u00e9", "\u0001", "poll", {"mode": "poll"}, {"challengeUrl": "https://a/b"}]


def paths(node, prefix=()):
    out = [prefix]
    if isinstance(node, dict):
        for k, v in node.items():
            out += paths(v, prefix + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += paths(v, prefix + (i,))
    return out


def get(doc, p):
    for k in p:
        doc = doc[k]
    return doc


def damage(doc):
    d = copy.deepcopy(doc)
    for _ in range(R.choice([1, 1, 1, 2, 3])):
        ps = [p for p in paths(d) if p]
        p = R.choice(ps)
        parent = get(d, p[:-1])
        op = R.choice(["del", "set", "set", "set", "add", "swap", "trunc"])
        k = p[-1]
        if op == "del":
            del parent[k]
        elif op == "set":
            parent[k] = copy.deepcopy(R.choice(POOL))
        elif op == "add":
            target = get(d, p) if isinstance(get(d, p), dict) else parent
            if isinstance(target, dict):
                target[R.choice(["extra", "mode", "relay", "device", "iceServers", "scopes"])] = copy.deepcopy(R.choice(POOL))
        elif op == "swap":
            q = R.choice(ps)
            if q[:len(p)] == p or p[:len(q)] == q:
                continue
            a, b = get(d, p), get(d, q)
            get(d, q[:-1])[q[-1]] = copy.deepcopy(a)
            parent[k] = copy.deepcopy(b)
        else:
            v = get(d, p)
            if isinstance(v, str) and len(v) > 1:
                parent[k] = v[: R.randint(0, len(v) - 1)]
            elif isinstance(v, list) and v:
                parent[k] = v[:-1]
    return d


with open(os.path.join(outdir, "cases.jsonl"), "w", encoding="utf-8") as fc, open(os.path.join(outdir, "python.jsonl"), "w", encoding="utf-8") as fp:
    n = 0
    for c in cases:
        req, body = c["request"]["body"], c["response"]["body"]
        for j in range(count):
            doc = body if j == 0 else damage(body)
            text = json.dumps(doc, ensure_ascii=R.random() < 0.5)
            cid = "%s#%d" % (c["name"], j)
            rec = {"id": cid, "kind": c["role"], "request": req, "response": text}
            try:
                if c["role"] == "plugin":
                    expect = {k: req[k] for k in ("appId", "pluginId", "endpointPublicKey", "approvedPeerKeyThumbprints", "peerRosterRevision", "peerRosterHash")}
                    out = D.plugin_admission(json.loads(text), expect, NOW)
                else:
                    session = {"gatewayUrl": body["gatewayUrl"], "appId": req["appId"], "deviceId": req["deviceId"], "discoveryRelayOnly": body["relayOnly"]}
                    out = D.mobile_admission(json.loads(text), session, req["holderKeyThumbprint"], NOW)
                verdict = {"accept": True, "transport": out["transport"]}
            except D.Refused as e:
                verdict = {"accept": False, "why": str(e)[:80]}
            except Exception as e:  # a crash of the reference
                verdict = {"accept": False, "crash": type(e).__name__, "why": str(e)[:80]}
            fc.write(json.dumps(rec) + "\n")
            fp.write(json.dumps({"id": cid, **verdict}) + "\n")
            n += 1
print("wrote", n, "cases; NOW =", NOW)
