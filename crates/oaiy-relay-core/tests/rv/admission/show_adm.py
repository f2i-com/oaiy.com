import json, sys, collections, pathlib
d, direction, member, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
AOKIE = pathlib.Path(__file__).resolve().parents[5] / "platform" / "protocol" / "relay" / "v1" / "fixtures" / "aokie"
adm = json.loads((AOKIE / "admission.json").read_text(encoding="utf-8"))
orig = {c["name"]: c["response"]["body"] for c in adm["cases"] if c["response"]["status"] == 200}
shown = collections.Counter()
for l in open(d + "/disagree.txt", encoding="utf-8"):
    r = json.loads(l)
    if not r["why"].startswith(direction):
        continue
    name = r["id"].rsplit("#", 1)[0]
    doc = json.loads(r["response"])
    o = orig[name]
    diffs = sorted(k for k in set(o) | set(doc) if o.get(k, "<absent>") != doc.get(k, "<absent>"))
    if diffs != [member]:
        continue
    reason = r["py"].get("why", "") if not r["py"]["accept"] else "accepted"
    key = reason[:40]
    if shown[key] >= max(1, n // 4):
        continue
    shown[key] += 1
    print("---", r["id"], "| python:", reason)
    print("   orig:", json.dumps(o.get(member, "<absent>"))[:300])
    print("   mut :", json.dumps(doc.get(member, "<absent>"))[:300])
    if sum(shown.values()) >= n:
        break
