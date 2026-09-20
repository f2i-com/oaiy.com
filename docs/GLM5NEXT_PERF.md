# GLM-5.3-Flash: where a token goes, and the expert hierarchy

Modelled on `dsv41`/`dsv41-cuda`, which reaches **~29 tok/s warm** on this same
machine (`docs/DEEPSEEK_V41.md`). This document says what that design is, which
parts of it glm5next already has, what the measured numbers are here, and what
has to change to reach 20 tok/s.

**Machine:** 2 x RTX 5090 (32,607 MiB each), Ryzen 9 9950X3D (16 cores / 32
threads, Zen 5, full-width AVX-512), 189.6 GB RAM, model on `D:`.

**Model:** GLM-5.3-Flash Q4_K_M, 5 shards, 1412 tensors. 45-layer
trunk (34 KDA + 11 MLA) plus `blk.45`, the NextN/MTP draft block. 288 experts
top-8 plus 1 shared, `leading_dense_block_count = 3`, so 42 MoE layers.

---

## 1. The measurement that reframed this

Everything below follows from one number. `DeviceModel::view_with_experts` swaps
in an `ExpertFfn` that writes zeros; `forward::prof` counts wall time per phase.
One token, 45 layers, **no routed experts at all** (`measure_trunk_floor`):

| Phase | Before | After k_b/v_b | Share |
|---|---:|---:|---:|
| hyper-connections | 23.8 ms | 23.5 ms | 22.5% |
| KDA attention | 61.3 ms | 57.8 ms | 55.2% |
| MLA attention | 69.1 ms | 11.3 ms | 10.8% |
| FFN (router, shared) | 11.7 ms | 11.6 ms | 11.0% |
| output head | 0.6 ms | 0.6 ms | 0.6% |
| **trunk total** | **159.8 ms** | **102.7 ms** | |
| | *6.3 tok/s* | *9.7 tok/s* | |

**20 tok/s is 50 ms a token.** The trunk alone was 3.2x over that budget with the
experts free, and is still 2x over. No expert hierarchy, however good, reaches
20 tok/s while that is true. That is the finding: this model has **two walls**,
and the tiering addresses the second one.

### 1.1 What the first fix was

`k_b` and `v_b` are the absorbed-MLA weights, `[64, 512, 256]` and
`[64, 256, 512]`, 8.39 M values each. `MlaW` typed them `&[f32]` while every
matrix around them was a `Mat`, because they are not flat matrices -- each head
takes its own query. So 11 layers x 67.2 MB was read through host scalar code on
every token: 739 MB of DRAM traffic for 1.1 M output values.

`Backend::batched_gemv(w[b,m,k], x[b,k]) -> [b,m]` and `forward::Bat` fixed that:
**61.7 ms -> 3.9 ms**, trunk 159.8 -> 102.7 ms. 739 MB of VRAM, the cheapest 60 ms
in the model.

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

There is also a floor of about **32 us per `Mat::apply`** (`measure_mat_apply_latency`:
a 1x64 matvec costs 32.1 us, a 4096x4096 one 118.3 us). At ~710 applies a token
that is ~23 ms of pure Windows synchronisation latency -- the same cost
`dsv41-cuda/src/handoff.rs` was written to avoid. Fewer, larger launches is the
only way down.

---

## 2. The expert hierarchy

This is the part modelled directly on DeepSeek. The diagram it implements:

```
                       +---- hot experts ---------> VRAM
                       |
NVMe -> phase-aware ---+---- CPU AVX-512 execution
        RAM cache      |
                       +---- pinned H2D staging --> GPU
```

### 2.1 Why the tiers exist: the byte budget

| | glm5next Q4_K_M | DeepSeek V4.1 |
|---|---:|---:|
| One expert record | 14.16 MB (3 x 4,718,592 B) | 18.8 MB |
| Records in the model | 42 x 288 = 12,096 | 40 x 384 = 15,360 |
| All experts | 171 GB | 289 GB |
| Routed per token | 42 x 8 = 336 | 240 |
| **Bytes per token** | **4.76 GB** | 4.51 GB |

4.76 GB a token at 20 tok/s is **94 GB/s**. NVMe does ~7. DDR5 does ~80-100.
VRAM does 1,800. So the hierarchy is not an optimisation, it is the only way the
number works: the bytes have to already be somewhere fast, and mostly not move.

### 2.2 Tier 0 -- VRAM, both GPUs

DeepSeek: `dsv41-cuda/src/expert_cache.rs`. A byte-budgeted pool of fixed record
slots carved from **one device allocation**, keyed `(layer, expert)`, evicted
LFRU -- frequency first, recency as tiebreak. A hit costs **zero PCIe**: the
kernels read the slot in place. Frequencies halve every `AGE_TOKENS = 128`
decode tokens so an old topic's experts cannot hold VRAM forever. Slots used by
the current batch are pinned (`begin_batch`), so a grouped kernel can read all of
a layer's experts without a later fetch evicting an earlier one. Measured:
**~1,170 slots per card at ~22 GB, ~2,700 across both, 62.7% hits.**

glm5next has this: `llama-rs/src/expert_stream/device_cache.rs`, behind
`StreamShared::enable_device_cache`. It measured as **noise at 6 GB**, and that
is arithmetic, not a bug: 6 GB is ~370 of 12,096 records, 3%.

The fix is capacity, and capacity means the second card. Today one GPU holds the
~24 GB of non-expert matrices and the 739 MB of `k_b`/`v_b`, leaving little.
Split the trunk by layer across both cards -- which is what DeepSeek does -- and
each frees ~19 GB:

    2 x 19 GB / 14.16 MB = ~2,680 slots = 22% of the model

against DeepSeek's 62.7% hit rate at 17.6% of its model. Routing is skewed, so
the hot set is far smaller than the model; that is why 22% buys most of the
accesses.

The hidden state crosses cards once per token at the boundary layer: 4 x 4096
f32 = 64 KB. `CudaBackend::new(ordinal)` already takes a device.

### 2.3 Tier 1 -- RAM, phase-aware

DeepSeek: `nrob::ecache`. LFRU over a random draw of `SAMPLES = 16` resident
records, leases so a record in use is never evicted, one mutex never held across
a read.

The phase-aware part is `Ecache::set_scan_layer`. A prefill runs the layers in
order and uses each layer's experts once, so the record it just read is the
least-used one around and plain LFRU evicts a record the same pass has not
reached yet -- the pass eats its own future. Measured on DeepSeek: a full RAM
tier served 2.6K of the 10.2K records it held that a 5.6K-token prompt needed.
With the hint, victims come from layers the pass is already past.

glm5next has `Ecache` (`StreamExperts` runs the MoE through it). It does **not**
call `set_scan_layer` yet. That is a one-line hook in the prefill loop.

Capacity: ~150 GB of the 175 GB free is ~10,600 records, **88% of the model**.
With VRAM's ~2,680 on top -- and DeepSeek does *not* make the tiers exclusive;
`admit_pending` copies from RAM and the RAM copy stays -- coverage is high enough
that NVMe serves only the cold tail.

### 2.4 Tier 2 -- CPU execution, and why it is the decisive one

This is the part that makes DeepSeek fast, and the part glm5next cannot do yet.

From `dsv41/src/cpu_experts.rs`:

> Moving that record to a GPU costs a PCIe copy (2.6 ms on the dev machine's x2
> link, 1.3 ms on its x4); computing it here costs ~0.9-1.1 ms on 24-32 threads.
> So a VRAM miss whose record is in RAM is cheaper to compute here than to
> upload, and it runs while the GPU works on the layer's resident experts.

So a VRAM miss does not become an upload. It becomes CPU work, overlapped.
`dsv41-cuda/src/model.rs` caps uploads at `PROMOTE_PER_LAYER = 1` per layer when
a CPU pool exists, admits that one only if `worth_admitting` says it beats the
LFRU victim, and **launches the CPU job before the GPU work** so it runs
underneath. Threads are persistent and spin for 100 us before parking, because
waking a parked thread on Windows costs a fifth of a one-expert job.

**glm5next cannot do this yet, and the measurement says why.** One record on the
host today (`measure_host_record_cost`, 14.16 MB, 25.2 M weights, single thread):

    dequantise      3.23 ms   (4.4 GB/s in)
    three matvecs   8.77 ms
    total          12.00 ms

The arithmetic is not the problem -- the f32 detour is. Dequantising writes
101 MB that the matvecs then read back, so **202 MB of DRAM traffic sets a
~2.5 ms floor per record whatever the thread count**. Against a 1.3 ms PCIe
upload, computing on the CPU *loses*.

A fused Q4_K dot touches the 14.16 MB once: ~0.18 ms at 80 GB/s. That is the
kernel DeepSeek has (`avx512::fp4_rows`, a safe `#[target_feature]` function so
the crate stays free of `unsafe`, bit-identical to the portable path). `ggml-quants`
has only `dequantize` -- no `vec_dot_q4_K_q8_K`. **Porting it is the gate on
this tier**, and the 9950X3D has the AVX-512 to run it.

### 2.5 Tier 3 -- NVMe, and pinned staging

Prefill misses cross on the copy stream into `STAGE_SLOTS = 2` staging slots, so
one expert's bytes land while the previous one's kernels run; events stand in for
stream order. At the end of the prefill, `admit_pending` moves the most-used
misses into VRAM by frequency, from RAM only, stopping at the first one that does
not beat its victim. (Admitting every miss as it came evicted much of the warm
set; admitting none cost the reply speed.)

glm5next has the staging ring: `ggml-rs-cuda/src/transfer.rs`.

Reads themselves: `GgufExpertStore::fetch_many` already batches a layer's misses
across a thread pool, which is what took the raw read off 2.74 GB/s.

### 2.6 Two branches of the diagram that do not carry over

**`next-layer predictor -> temporary prefetch pool -> RAM only if demanded`.**
This is not implemented in the reference either. `Ecache::hint` and
`Ecache::prefetch` are **no-ops** -- `docs/ROADMAP.md` P1, and section 6 is
explicit that predictive prefetch should be enabled only once useful bytes
substantially exceed wasted bytes, because a bad prefetch evicts demand-hot
experts and eats queue depth. There is nothing to copy across yet; when there is,
glm5next has the better predictor available, because `blk.45` is a real MTP draft
block whose routing is a genuine next-token signal rather than a heuristic.

**`Engram: 101 GB tables -> 24 persistent readers -> deduplicated rows -> GPU`.**
Does not apply. Engram is a DeepSeek-V4.1 component (n-gram hashes over a
compressed token map, two 101 GB tables on layers 1 and 14, `READERS = 24`
threads doing latency-bound row reads). **glm5next has no Engram tables.** Its
analogous "read a little from something enormous" problem is the sparse
lightning indexer, which is pure compute over the KV cache, not storage.

---

## 3. Where glm5next is now

Measured on the released model, one RTX 5090:

| | |
|---|---|
| Load (5 shards, streaming) | 3.2 s |
| Host vs CUDA, whole model | 3e-5 over 154,880 logits at scale 30 |
| One token, host | 10.0 s |
| One token, CUDA | 1.8 s |
| Decode, cold cache | 0.636 s/token (1.57 tok/s) |
| Decode, warm cache | 0.562 s/token (1.78 tok/s) |

(0.705 / 0.612 before the `k_b`/`v_b` fix, so the ~57 ms it took off the trunk
shows up on the real decode path too.)

The warm token is **~103 ms of trunk and ~459 ms of experts**. Both numbers have
to come down by roughly 5x and 15x respectively.

## 4. The budget, and the order of work

    50 ms  =  trunk (<= 20 ms)  +  experts (<= 30 ms)

Both halves have to land. Ranked by measured milliseconds per unit of work:

1. **Port `vec_dot_q4_K_q8_K` to `ggml-quants`, AVX-512.** Gates tier 2 entirely,
   and also removes the f32 detour from every host-side path. ~2.5 ms -> ~0.2 ms
   per record.
2. **Split the trunk across both GPUs**, one expert cache per card. Takes the
   VRAM tier from 3% of records to ~22%, which is where DeepSeek's 62.7% hit rate
   comes from.
3. **CPU tier with `worth_admitting` and `PROMOTE_PER_LAYER = 1`**, launched
   before the GPU work. This is what breaks the PCIe wall.
4. **A KDA delta-rule CUDA kernel** (22.2 ms). The state is `[64, 128, 128]` per
   layer and never needs to leave the device; `Bat`-shaped batched ops fit it.
5. **Fuse the KDA q/k/v projections** into one launch (12.7 ms, three applies a
   layer) and keep the hyper-connection streams device-resident (23.5 ms).
6. **`set_scan_layer` during prefill** -- one hook, already built.
7. **A usage profile saved across runs** (`save_usage`/`warm`), so a fresh
   process does not start cold.

Items 1-3 are the hierarchy the diagram describes. Items 4-5 are the trunk, and
without them the hierarchy has nothing to win: at a 20 ms trunk and a 30 ms
expert budget the numbers close, and at a 103 ms trunk they cannot.
