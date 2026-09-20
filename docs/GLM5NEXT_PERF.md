# GLM-5.3-Flash: where a token goes, and the expert hierarchy

Modelled on `dsv41`/`dsv41-cuda`, which reaches **~29 tok/s warm** on this same
machine (`docs/DEEPSEEK_V41.md`). This document says what that design is, which
parts glm5next now has, what each measures at here, and what is left.

**Machine:** 2 x RTX 5090 (32,607 MiB each), Ryzen 9 9950X3D (16 cores / 32
threads, Zen 5, full-width AVX-512), 189.6 GB RAM, model on `D:`.

**Model:** GLM-5.3-Flash Q4_K_M, 5 shards, 1412 tensors. 45-layer
trunk (34 KDA + 11 MLA) plus `blk.45`, the NextN/MTP draft block. 288 experts
top-8 plus 1 shared, `leading_dense_block_count = 3`, so 42 MoE layers.

## The model, by bytes

Measured from the five shard headers, not estimated:

| Component | Bytes | Stored as | Placement |
|---|---:|---|---|
| Routed experts, 42 MoE layers x 288 | **182.44 GB** | gate/up Q4_K; down Q4_K on 24 layers, **Q6_K on 18** | VRAM + RAM + SSD |
| `blk.45`, MTP draft block | 4.81 GB | same | not loaded |
| Trunk: attention, shared experts, router, hc, embed, head | **5.71 GB** | Q4_K, Q8_0, F32 | VRAM, card 0 |
| **Total** | **192.97 GB** | | |

The mixed `ffn_down` quantisation matters: a record is gate+up+down for one
expert, so it is either **14.16 MB** (all Q4_K) or **16.32 MB** (Q6_K down), and a
fixed cache slot has to be the larger. A token routes 8 of 288 experts in each of
42 layers — 336 records, **5.07 GB**.

5.07 GB a token at 20 tok/s is **101 GB/s**. NVMe does ~7. DDR5 does ~80-100.
VRAM does ~1,800. The hierarchy is not an optimisation; it is the only way the
number works, because the bytes have to already be somewhere fast and mostly not
move at all.

---

## 1. Two walls, not one

`DeviceModel::view_with_experts` swaps in an `ExpertFfn` that writes zeros;
`forward::prof` counts wall time per phase. One token, 45 layers, **no routed
experts at all** (`measure_trunk_floor`):

| Phase | Before | Now | Share |
|---|---:|---:|---:|
| hyper-connections | 23.8 ms | 23.5 ms | 22.5% |
| KDA attention | 61.3 ms | 57.8 ms | 55.2% |
| MLA attention | 69.1 ms | 11.3 ms | 10.8% |
| FFN (router, shared) | 11.7 ms | 11.6 ms | 11.0% |
| output head | 0.6 ms | 0.6 ms | 0.6% |
| **trunk total** | **159.8 ms** | **102.7 ms** | |
| | *6.3 tok/s* | *9.7 tok/s* | |

**20 tok/s is 50 ms a token.** The trunk alone is 2x over that budget with the
experts free. No expert hierarchy, however good, gets to 20 tok/s while that
holds — which is the single most useful thing measuring this produced.

### 1.1 The fix that landed

`k_b` and `v_b` are the absorbed-MLA weights, `[64, 512, 256]` and
`[64, 256, 512]`, 8.39 M values each. `MlaW` typed them `&[f32]` while every
matrix around them was a `Mat`, because they are not flat matrices — each head
takes its own query. So 11 layers x 67.2 MB was read through host scalar code on
every token: 739 MB of DRAM traffic for 1.1 M output values.

`Backend::batched_gemv(w[b,m,k], x[b,k]) -> [b,m]` (one warp per output row) and
`forward::Bat` fixed it: **61.7 ms -> 3.9 ms**, trunk 159.8 -> 102.7 ms, for
739 MB of VRAM.

### 1.2 What is left in the trunk

| | ms | why |
|---|---:|---|
| KDA delta-rule recurrence | 22.2 | 34 layers x a 4.2 MB `[64,128,128]` state, host scalar |
| hyper-connections | 23.5 | 90 host calls a token over 4 x 4096 streams |
| KDA q/k/v projections | 12.7 | three `Mat::apply` a layer that want to be one |
| KDA decay gate | 8.0 | `f_a`/`f_b` plus a per-channel sigmoid, host |
| KDA depthwise conv + shift | 4.6 | host scalar over `3 * d_inner = 24576` channels |
| FFN router + shared expert | 11.6 | |
| MLA (all of it) | 11.3 | done |

There is also a floor of about **32 us per `Mat::apply`**
(`measure_mat_apply_latency`: 1x64 costs 32.1 us, 4096x4096 costs 118.3 us). At
~710 applies a token that is ~23 ms of Windows synchronisation latency — the cost
`dsv41-cuda/src/handoff.rs` exists to avoid. Fewer, larger launches is the only
way down.

---

## 2. The expert hierarchy, as built

```
                       +---- hot experts ---------> VRAM, both cards
                       |
SSD -> phase-aware ----+---- CPU AVX-512 execution        (not built)
       RAM cache       |
                       +---- pinned H2D staging --> GPU
```

`DeviceModel::open_tiered(path, max_len, n_gpus, ram_budget, vram_cap)` opens the
whole stack. Both budgets accept 0, meaning "size yourself from the hardware".

### 2.1 Tier 0 — VRAM, both cards

DeepSeek: `dsv41-cuda/src/expert_cache.rs`. Fixed record slots in one device
allocation, keyed `(layer, expert)`, LFRU — frequency first, recency as tiebreak.
A hit costs **zero PCIe**: the kernels read the slot in place. Frequencies halve
every `AGE_TOKENS = 128` tokens so an old topic cannot hold VRAM forever. Slots
used by the current batch are pinned (`begin_batch`). Measured there: ~1,170
slots per card, ~2,700 over both, **62.7% hits**.

glm5next has `llama-rs/src/expert_stream/device_cache.rs` — and for a long time
was not using it. Sweeping the budget over 0, 8 and 24 GB moved the warm token by
under a millisecond, because `StreamExperts` went through
`LayerStream::expert_weights`, which reconstructs over the leased *host* record
and never consults `StreamShared::device`. The VRAM tier was wired only into
`forward_with_logits`, the generic Qwen3-MoE path. `LayerStream::resolve_experts`
is the per-layer entry point that does consult it, and wiring it took the warm
token from **0.544 s to 0.139 s**.

A cache lives on one card and a kernel reads only the card it launched on, so
spanning two GPUs means partitioning **by layer**:
`StreamShared::enable_device_shards` builds one cache per backend plus a
`shard_of[layer]` map, and `StreamExperts::spread_over` deals the 42 MoE layers
out in contiguous runs. Contiguous, not round-robin, so each card sees the same
layers every token and its LFRU set converges instead of thrashing.

The crossing is free, which is the one place the current design helps:
`apply_layer` takes host `x` and returns host `out`, so the hidden state already
passes through a host copy between layers. The same round trip that costs the
trunk 32 us an apply is what makes multi-GPU need no transfer code at all.

Sizing is per card, from that card's own free memory, and deliberately unequal —
card 0 carries the trunk, card 1 carries nothing. Two bugs had to be fixed first:

- `cuMemGetInfo` reports the **current context**, so both cards claimed 26.5 GB
  free when only one held the trunk. `vram_status` now binds first. This affected
  any multi-GPU sizing, including dsv41's loader.
- The cache counts uploaded bytes, but each expert is two or three separate
  ~4.7 MB allocations and CUDA rounds each up: 23 GB charged put the card at
  24.4 GB, and a 192-token prefill then hit `CUDA_ERROR_OUT_OF_MEMORY`. Hence
  `VRAM_CHARGE_FRACTION = 0.88` and a 2 GB headroom.

The cache admits lazily rather than reserving a pool, so nvidia-smi shows both
cards near-empty at load climbing to **31-32 GB of 32.6** during a prefill.

### 2.2 Tier 1 — RAM, phase-aware

DeepSeek: `nrob::ecache`. LFRU over a random draw of `SAMPLES = 16`, leases so a
record in use is never evicted, one mutex never held across a read.

The phase-aware part is `Ecache::set_scan_layer`. A prefill runs the layers in
order and uses each layer's experts once, so the record just read is the
least-used one around and plain LFRU evicts one the same pass has not reached —
the pass eats its own future. Measured on DeepSeek: a full RAM tier served 2.6K of
the 10.2K records it held that a 5.6K-token prompt needed.

glm5next runs its MoE through `Ecache` but **does not call `set_scan_layer` yet**.
That is a one-line hook in the prefill loop and is the cheapest thing left on this
list.

The budget sizes itself from `ggml_rs_cuda::host_memory()` (the companion to
`vram_status`): everything free except `RAM_RESERVE = 24 GB`, capped at the expert
bytes. On this machine, 161 GB — 9,845 of 12,096 records, **81%**.

### 2.3 Tier 2 — CPU execution: not built, and why

From `dsv41/src/cpu_experts.rs`:

> Moving that record to a GPU costs a PCIe copy (2.6 ms on the dev machine's x2
> link, 1.3 ms on its x4); computing it here costs ~0.9-1.1 ms on 24-32 threads.

So a VRAM miss whose record is in RAM is computed, not uploaded, and it runs while
the GPU works on the layer's resident experts. `dsv41-cuda/src/model.rs` caps
uploads at `PROMOTE_PER_LAYER = 1`, admits that one only if `worth_admitting` says
it beats the LFRU victim, and launches the CPU job **before** the GPU work.

glm5next cannot do this yet. One record on the host today
(`measure_host_record_cost`, 14.16 MB, 25.2 M weights, single thread):

    dequantise      3.23 ms   (4.4 GB/s in)
    three matvecs   8.77 ms
    total          12.00 ms

The arithmetic is not the problem — the f32 detour is. Dequantising writes 101 MB
that the matvecs read back, so **202 MB of DRAM traffic sets a ~2.5 ms floor per
record whatever the thread count**. Against a 1.3 ms PCIe upload, the CPU *loses*.
A fused Q4_K dot touches 14.16 MB once: ~0.18 ms at 80 GB/s. That is the kernel
DeepSeek has (`avx512::fp4_rows`, a safe `#[target_feature]` function, bit-identical
to the portable path); `ggml-quants` has only `dequantize`. **Porting
`vec_dot_q4_K_q8_K` gates this tier**, and the 9950X3D has the AVX-512 for it.

### 2.4 Tier 3 — SSD, and pinned staging

Prefill misses cross on the copy stream into `STAGE_SLOTS = 2` staging slots so one
expert's bytes land while the previous one's kernels run; `admit_pending` then
moves the most-used misses into VRAM by frequency, from RAM only, stopping at the
first that does not beat its victim.

glm5next has the staging ring (`ggml-rs-cuda/src/transfer.rs`, 3 pinned slots) and
`GgufExpertStore::fetch_many`, which batches a layer's misses across a thread pool
— that is what took the raw read off 2.74 GB/s.

### 2.5 Two branches of the diagram that do not carry over

**`next-layer predictor -> temporary prefetch pool -> RAM only if demanded`.** Not
implemented in the reference either: `Ecache::hint` and `Ecache::prefetch` are
no-ops (`docs/ROADMAP.md` P1), and §6 there is explicit that predictive prefetch
should be enabled only once useful bytes exceed wasted bytes, because a bad
prefetch evicts demand-hot experts and eats queue depth. When it is built,
glm5next has the better signal available: `blk.45` is a real MTP draft block, so
next-token routing is a prediction rather than a heuristic.

**`Engram: 101 GB tables -> 24 readers -> deduplicated rows -> GPU`.** Does not
apply. Engram is a DeepSeek-V4.1 component (n-gram hashes over a compressed token
map, two 101 GB tables, `READERS = 24` latency-bound row readers). **glm5next has
no Engram tables.** Its analogous "read a little of something enormous" problem is
the sparse lightning indexer, which is compute over the KV cache, not storage.

---

## 3. Measured, in order

Released model, both cards, RAM auto-sized.

| | s/token | tok/s |
|---|---:|---:|
| Starting point (uncached experts) | 2.700 | 0.37 |
| `StreamExperts` + fused clamped SwiGLU | 0.612 | 1.64 |
| `k_b`/`v_b` on the GPU | 0.562 | 1.78 |
| Per-layer expert dispatch | 0.544 | 1.84 |
| **VRAM tier actually consulted** | **0.139** | **7.19** |

That last row is the short-prefix measurement, and it flatters itself: replaying
an 8-token prefix means a few hundred distinct experts serve the whole run, one
card holds all of them, and nothing is ever evicted — so the second card, the RAM
tier and the SSD look free when they are merely idle. It reports 92% VRAM hits.

`measure_tiered_on_a_document` prefills a varied prompt first. 192 tokens:

```
  2 cards, VRAM slots [1312, 1635] = 2947 (24% of 12096 records)
  RAM tier 161 GB = 9845 records (81%); experts are 182.4 GB on the SSD

  card 0: 69.9% VRAM hits, 158.4 GB uploaded, 9080 evictions
  card 1: 81.1% VRAM hits, 100.1 GB uploaded, 4846 evictions
  RAM:    69.2% hits, 124.2 GB read from the SSD

  decode 0.409 s/token (2.45 tok/s)
```

All three tiers carry real traffic there. That is the number to design against —
though the prompt is drawn from an LCG over the vocabulary, so it maximises expert
diversity and is nearer a worst case than a typical one. Real text routes with far
more locality, so the true figure is between 2.45 and 7.19, and a tokenised
document should decide it.

Correctness, unchanged throughout: host vs CUDA is 2e-5 over 154,880 logits at
scale 30, one card vs two cards is **exact**, and 128 sampled tokens come out with
72 distinct.

## 4. The budget, and what is next

    50 ms  =  trunk (<= 20 ms)  +  experts (<= 30 ms)

On the short-prefix measurement the split is already **103 ms trunk / 36 ms
experts** — the trunk is the whole problem. On the document measurement experts
are still ~300 ms, so both halves are live depending on the working set.

Ranked by measured milliseconds per unit of work:

1. **KDA delta-rule CUDA kernel** (22.2 ms). The `[64, 128, 128]` state never
   needs to leave the device; `Bat`-shaped batched ops fit the recurrence.
2. **Keep the hyper-connection streams device-resident** (23.5 ms over 90 host
   calls a token).
3. **Fuse the KDA q/k/v projections** into one launch (12.7 ms) and the decay gate
   and depthwise conv into kernels (12.6 ms together).
4. **`set_scan_layer` during prefill.** One hook, already built; it is what stops
   a prefill evicting the records it is about to need.
5. **Port `vec_dot_q4_K_q8_K` to `ggml-quants`, AVX-512.** Gates the CPU tier, and
   removes the f32 detour from every host path. Lower priority than it looked at a
   92% VRAM hit rate; back up the list at the document's 70-81%.
6. **A usage profile saved across runs** (`save_usage`/`warm`), so a fresh process
   does not start cold — DeepSeek's first answer is 2-6 tok/s and its third is 29.
