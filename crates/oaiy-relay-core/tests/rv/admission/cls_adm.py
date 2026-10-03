import json, sys, collections, pathlib
d = sys.argv[1]
AOKIE = pathlib.Path(__file__).resolve().parents[5] / "platform" / "protocol" / "relay" / "v1" / "fixtures" / "aokie"
adm = json.loads((AOKIE / "admission.json").read_text(encoding="utf-8"))
orig = {c["name"]: c["response"]["body"] for c in adm["cases"] if c["response"]["status"] == 200}
rows = [json.loads(l) for l in open(d + "/disagree.txt", encoding="utf-8")]
groups = collections.defaultdict(list)
for r in rows:
    name = r["id"].rsplit("#", 1)[0]
    try:
        doc = json.loads(r["response"])
    except Exception:
        doc = None
    o = orig[name]
    diffs = set()
    if isinstance(doc, dict):
        for k in set(o) | set(doc):
            if o.get(k, "<absent>") != doc.get(k, "<absent>"):
                diffs.add(k)
    else:
        diffs.add("<not an object>")
    direction = r["why"].split(":")[0]
    groups[(direction, tuple(sorted(diffs)))].append(r)
tot = collections.Counter()
for (direction, diffs), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    tot[direction] += len(rs)
    print("%4d  %-12s %s" % (len(rs), direction, ",".join(diffs)))
print(tot)
# show one example per top group
for (direction, diffs), rs in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:int(sys.argv[2]) if len(sys.argv) > 2 else 6]:
    r = rs[0]
    print("---", direction, diffs, r["id"], r["py"])
