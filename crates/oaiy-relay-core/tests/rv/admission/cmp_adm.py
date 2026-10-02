import json, sys, collections
d = sys.argv[1]
py = {}
for l in open(d + "/python.jsonl", encoding="utf-8"):
    o = json.loads(l); py[o["id"]] = o
rs = {}
for l in open(d + "/rust.jsonl", encoding="utf-8"):
    o = json.loads(l); rs[o["id"]] = o
cases = {}
for l in open(d + "/cases.jsonl", encoding="utf-8"):
    o = json.loads(l); cases[o["id"]] = o
assert set(py) == set(rs)
crash = sum(1 for o in py.values() if "crash" in o)
bad = []
agree_acc = agree_ref = 0
for i, p in py.items():
    r = rs[i]
    if p["accept"] != r["accept"]:
        bad.append((i, "py accepts" if p["accept"] else "rust accepts"))
    elif p["accept"]:
        agree_acc += 1
        if (p["transport"] == "relay") != r["relay"]:
            bad.append((i, "relay advertisement usable: py %s, rust %s" % (p["transport"], r["relay"])))
    else:
        agree_ref += 1
print(len(py), "cases;", agree_acc, "both accept;", agree_ref, "both refuse;", len(bad), "disagree;", crash, "python crashes")
by = collections.Counter(b[1].split(":")[0] for b in bad)
print(by)
with open(d + "/disagree.txt", "w", encoding="utf-8") as f:
    for i, why in bad:
        f.write(json.dumps({"id": i, "why": why, "py": py[i], "response": cases[i]["response"]}) + "\n")

