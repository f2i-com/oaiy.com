import json, sys, collections
d = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
seen = collections.Counter()
for l in open(d + "/disagree.txt", encoding="utf-8"):
    o = json.loads(l)
    key = ",".join(o["diffs"])
    if want and key != want:
        continue
    seen[key] += 1
    if seen[key] > n:
        continue
    c = o["case"]
    print("---", o["id"], key)
    print(" status", c.get("status"), "hdr", c.get("headers"), "since", c.get("since"), "persisted", c.get("persisted"), "weRepl", c.get("weReplaced"), "now", c.get("nowEpoch"), "state", c.get("state"))
    print(" body", c.get("bodyText"))
    m, k = o["mine"], o["crate"]
    for f in ("outcome", "baseS", "state", "action", "report", "since"):
        if m.get(f) != k.get(f):
            print("   ", f, "mine", m.get(f), "crate", k.get(f))
