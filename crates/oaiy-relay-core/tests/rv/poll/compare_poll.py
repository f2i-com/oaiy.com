import json, sys, collections
d = sys.argv[1]
mine = {}
for l in open(d + "/mine.jsonl", encoding="utf-8"):
    o = json.loads(l); mine[o["id"]] = o
crate = {}
for l in open(d + "/crate.jsonl", encoding="utf-8"):
    o = json.loads(l); crate[o["id"]] = o
cases = {}
for l in open(d + "/cases.jsonl", encoding="utf-8"):
    o = json.loads(l); cases[o["id"]] = o
assert set(mine) == set(crate), (len(mine), len(crate))
bad = []
for i, m in mine.items():
    c = crate[i]
    diffs = []
    for k in ("outcome", "state", "action", "report", "since"):
        if k in m and m[k] != c[k]:
            diffs.append(k)
    for k in ("baseS", "pauseS"):
        a, b = m[k], c[k]
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            diffs.append(k)
    if diffs:
        bad.append((i, diffs))
print(len(mine), "cases;", len(bad), "disagree")
by = collections.Counter(tuple(x[1]) for x in bad)
print(by.most_common(20))
# classify
out = open(d + "/disagree.txt", "w", encoding="utf-8")
for i, diffs in bad:
    out.write(json.dumps({"id": i, "diffs": diffs, "case": cases[i], "mine": mine[i], "crate": crate[i]}) + "\n")

