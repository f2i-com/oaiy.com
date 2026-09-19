"""Expert-cache hit rates from recorded routes (a golden file's per-layer route_ids).

Replays the tokens in order, as decode would see them, through an LFRU cache
per GPU (layers 0-19 on one, 20-39 on the other) and reports the warm hit rate
for several slot counts, plus the static upper bound: pinning the N most-used
experts of the sample. One prompt is a small sample; treat results as a first
estimate, not a law.

Usage: python cache_sim.py E:\\deepseek\\golden\\golden_long.safetensors
"""

import sys
from collections import Counter

from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="pt") as f:
    routes = [f.get_tensor(f"prefill.layer{l:02d}.route_ids").tolist() for l in range(40)]
n_tok = len(routes[0])
stream = [(l, e) for t in range(n_tok) for l in range(40) for e in routes[l][t]]  # token-major
print(f"{n_tok} tokens, {len(stream)} expert accesses, {len(set(stream))} distinct experts of 15360")


def lfru(accesses, slots):
    freq, last, cache, hits, clock = Counter(), {}, set(), 0, 0
    for key in accesses:
        clock += 1
        freq[key] += 1
        last[key] = clock
        if key in cache:
            hits += 1
            continue
        if len(cache) >= slots:
            victim = min(cache, key=lambda k: (freq[k], last[k]))
            cache.remove(victim)
        cache.add(key)
    return hits


half = len(stream) // 2
for slots in (600, 1200, 2400):
    gpu = [[a for a in stream if (a[0] < 20) == (g == 0)] for g in (0, 1)]
    warm = []
    for accesses in gpu:
        h1 = lfru(accesses[: len(accesses) // 2], slots)
        h_all = lfru(accesses, slots)
        warm.append((h_all - h1) / (len(accesses) - len(accesses) // 2))
    counts = Counter(stream)
    top = sum(c for _, c in counts.most_common(2 * slots))
    print(f"{slots:5d} slots/GPU: LFRU warm hit rate {100 * sum(warm) / 2:5.1f}%   "
          f"static best {2 * slots} experts {100 * top / len(stream):5.1f}%")
