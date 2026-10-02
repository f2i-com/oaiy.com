import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rv_mutations import MUTATIONS
root = pathlib.Path(sys.argv[1])
ids = [m[0] for m in MUTATIONS]
assert len(ids) == len(set(ids)), "duplicate ids"
bad = 0
for (mid, area, what, file, old, new, kind) in MUTATIONS:
    t = (root / file).read_bytes().decode("utf-8").replace("\r\n", "\n")
    n = t.count(old)
    if n != 1:
        bad += 1
        print(mid, file, "occurs", n)
print(len(MUTATIONS), "mutants,", bad, "do not apply once")
