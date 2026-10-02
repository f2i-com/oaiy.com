import re, sys, collections
log = sys.argv[1]
rows = []
for l in open(log, encoding="utf-8"):
    m = re.match(r"^([A-Z][0-9]{2})\s+(KILLED|SURVIVED|INVALID|TIMEOUT)\s+(\S+)\s+(.*?)\s{2,}(.*?)(\s+\[\d+ s\])?$", l.rstrip())
    if m:
        rows.append((m.group(1), m.group(2), m.group(3), m.group(4).strip(), m.group(5).strip()))
c = collections.Counter(r[1] for r in rows)
print(len(rows), dict(c))
by_prefix = collections.defaultdict(collections.Counter)
for r in rows:
    by_prefix[r[0][0]][r[1]] += 1
print({k: dict(v) for k, v in sorted(by_prefix.items())})
print("SURVIVED / TIMEOUT:")
for r in rows:
    if r[1] in ("SURVIVED", "TIMEOUT", "INVALID"):
        print(" ", r[0], r[1], "|", r[3], "|", r[4][:110])
