# DeepSeek-V4.1-Flash on nrob — feasibility and plan

**Target:** `DeepSeek-V4.1-Flash-FP8` (510.3 GB, 55 files, commit
`d61c59ea`), weights in `E:\deepseek\model`, reference code (`inference/model.py`,
`kernel.py`, `engram.py`, `encoding/`) in `E:\deepseek\reference`.
**Machine:** Ryzen 9 9950X3D, 192 GB DDR5, 2× RTX 5090 (32 GB), 2× Crucial P3 Plus 4 TB.
**Date:** 2026-09-18. Every number below is measured on this machine or read from the
checkpoint headers unless marked *estimate*.

## Verdict

Feasible, and a good fit for nrob's design: this is a 552B sparse MoE whose active set per
token is small (6 of 384 experts × 40 layers), and nrob already has the streaming pipeline
(host cache with leases, VRAM expert cache, pinned staging, GPU routing, grouped MoE).
What does not exist yet is everything model-specific: the data formats (MXFP4 experts,
FP8 32×32-block trunk), the V4.1 forward pass (hyper-connections, compressed sparse
attention, Engram), and multi-GPU placement.

Expected batch-1 decode *(estimate, see Roofline)*: **~1.5–3 tok/s on the current
hardware**, **~5–8 tok/s with 256 GB RAM**. The limiter is RAM capacity and SSD speed,
not GPU compute.

**Measured (2026-09-19, see Status):** with the hybrid path, decode runs at **~29 tok/s
once the expert tiers are warm** (after a second round of work; 15 tok/s after the first).
- A fresh process's first answer is SSD-bound: **2–6 tok/s even when it warm-starts
  from a saved usage profile**. It measured 5.8, 2.3 and 2.0 in three runs, depending
  on how much the drive throttled.
- The estimate was too pessimistic because LFRU plus a saved usage profile hold a
  document's working set (~8,000 experts) between VRAM and RAM.

## The model, by bytes

Measured from the safetensors headers (96,085 tensors):

| Component | Bytes | Stored as | Per decode token | Placement |
|---|---:|---|---:|---|
| Routed experts, 40 layers × 384 | 288.8 GB | FP4 e2m1 + E8M0 scale per 32 (= MXFP4) | 240 records = **4.51 GB** | RAM cache + VRAM cache + SSD |
| Engram tables (layers 1, 14) | 202.8 GB | FP8 e4m3 rows of 256 + E8M0 per 32 | 48 rows ≈ **12.7 KB** | SSD, row reads |
| Dense trunk (attention, shared experts, router, hc, embed, head) | 9.9 GB | FP8 with 32×32 E8M0 blocks, BF16, F32 | all of it | VRAM, split by layer |
| MTP / DSpark draft (3 layers × 128 experts) | 7.9 GB | same as above | — | later |
| Vision tower + aligner | 1.0 GB | BF16 | — | later |

One expert record (w1, w3: `[2304, 5120]`, w2: `[5120, 2304]`, plus scales) is exactly
**18,800,640 bytes**. Repacked as ggml's `block_mxfp4` (1 B E8M0 + 16 B of nibbles per 32
values) it is the same size, losslessly — but the nibble order differs (checkpoint: low
nibble = even element, high = odd; ggml: element j low, j+16 high), so the converter must
shuffle. Verify against llama.cpp's `block_mxfp4` before relying on it.

Expanding the FP8 trunk to BF16 is exact (e4m3 × 2^k always fits bf16) and costs ~+6.8 GB
of VRAM (trunk ≈ 16.7 GB total). That is the simplest v1; a native FP8 GEMV comes later.

## What the forward pass needs (reference: `inference/model.py`)

None of this exists in nrob or llama-rs today:

- **Hyper-connections** (`Block`, `hc_split_sinkhorn`): the residual is 4 parallel copies
  of the 5120-d stream. Each sublayer computes pre/post/comb mixes from an RMS-normalized
  projection of all 20480 values; `comb` is made doubly stochastic by 20 Sinkhorn
  iterations. The mix one sublayer computes is used by the *next* one.
- **Attention** (`Attention`, `Compressor`, `Indexer`, `select_candidate_blocks`):
  - q: 5120 → 1280 (LoRA) → 64 heads × 512; a single shared KV head of 512.
  - A 128-token sliding-window ring buffer on every layer.
  - Compressed KV on layers with `compress_ratio` 1 or 2, owned by the kv-source layers
    (2, 8, 14, 20) and shared downstream.
  - An FP4-simulated indexer picks the top 512 compressed positions. There are 8
    index-source layers; layer 20 also makes a first-level pick of the top 2048 blocks.
  - Sparse attention with a per-head `attn_sink`. The output gets an inverse RoPE, then
    the 8-group low-rank projection (`wo_a` block-diagonal, then `wo_b`).
  - YaRN applies only on compressed layers, with `compress_rope_theta = 160000`.
- **MoE** (`Gate`, `Expert`): the router score is sqrt(softplus(x)). The correction bias
  picks the experts but not their weights; weights are normalized × 1.5. SwiGLU is clamped
  (`swiglu_limit = 10`: up clamped to ±10, gate clamped from above), plus one shared expert.
- **Engram** (`Engram`, `engram.py`): n-gram hashes over a *compressed* token map. This
  needs Unicode NFKC/NFD/lowercase normalization, sympy primes and numpy PCG64
  multipliers. **Precompute all of it once in Python** (`token_map` of 129,280 ints,
  primes, offsets, multipliers) and store it in a side file (`engram_meta.safetensors`);
  the runtime hash is then a
  few multiply/XOR/mod ops per token. The looked-up rows feed a gated add into the
  residual.
- **Activation quantization is part of the numerics**: every FP8/FP4 GEMM first
  quantizes activations to FP8 e4m3 with a power-of-two scale per 32 values
  (`act_quant`). KV, indexer and compressed-KV paths simulate FP8/FP4 in place.
  Validation must replicate this, or it won't match the reference.
- **Chat format**: `encoding/encoding.py` (thinking modes, DSML tool calls). The README
  warns reasoning effort defaults to *max* — thousands of reasoning tokens — so default to
  thinking off or low for interactive use at these speeds.

## This machine (measured 2026-09-18)

| Resource | Measured | Note |
|---|---:|---|
| GPU0 (PCI bus 1) H2D, pinned | **7.2 GB/s** | link trained **PCIe 5.0 x2** (card is x16) |
| GPU1 (PCI bus 3) H2D, pinned | **14.4 GB/s** | link trained **PCIe 5.0 x4** |
| GPU↔GPU peer access | none | consumer 5090s: hops go through host RAM |
| RAM read bandwidth (16 threads) | **42.8 GB/s** | 4×48 GB running at DDR5-3600 (2 DIMMs/channel) |
| SSD `E:`, unbuffered, any queue depth | **~2.0 GB/s** | 16 MiB random and 64 MiB sequential alike; link is PCIe 4.0 x4, so drive-limited |
| SSD `C:` | same model | 221 GB free |
| CPU | 16 cores, AVX-512 | relevant for the hybrid expert path |

The first 46 shards are contiguous (19–48 extents each); the two 101 GB Engram shards are
heavily fragmented (573k extents), which is harmless for row reads.
CUDA device 0 is the x2 card — until the links are fixed, single-GPU runs should prefer
device 1.

## Memory plan

- **VRAM (2 × 32 GB):**
  - GPU0: embedding + layers 0–19 trunk. GPU1: layers 20–39 trunk + head.
  - That's ~8.3 GB each with a BF16 trunk; KV is tiny (the model card says 890 B/token).
  - The remaining ~22 GB per GPU goes to the existing VRAM expert cache: ~1,170
    experts per GPU, each owning only its own layers' experts.
  - Only the 4×5120 hidden state (80 KB) crosses between GPUs once per token, via host.
- **Host RAM (192 GB):** ~160 GB for the `Ecache` host expert cache, about 8,500 experts
  (55%). Expert reads bypass the page cache (`FILE_FLAG_NO_BUFFERING`, which safe Rust
  can request via `OpenOptionsExt::custom_flags`), leaving the page cache free for hot
  Engram rows.
- **SSD:** the ~30% of experts that don't fit, plus the Engram tables, read row by row
  (2 layers × 24 rows × 2 reads per token).

## Where the experts get computed

1. **GPU streaming** (the existing nrob path). Every RAM-hit expert crosses PCIe. At
   today's links that is 4.5 GB × (1 − VRAM hit) per token at 7–14 GB/s, so ~200–300 ms
   a token before any SSD traffic.
2. **Hybrid CPU + GPU (recommended here).** The GPU runs the trunk, shared expert and
   VRAM-hit experts. The CPU computes RAM-resident experts in place, AVX-512: FP4 values
   come from a lookup table and are summed with bf16 dot products into f32. That is exact
   for FP4 × FP8-simulated activations, and only the 20 KB activation crosses PCIe.
   Routing runs on the GPU and sends only the 6 expert ids to the host. The per-token RAM
   cost is ~3.5 GB / 42.8 GB/s ≈ 80 ms, overlapped with GPU work.

Path 2 is what runs now (see Status). Its AVX-512 kernel needed no `unsafe` in the core:
since Rust 1.87 the value-taking `std::arch` intrinsics are safe inside a
`#[target_feature]` function, so `dsv41::cpu_experts::avx512` is plain safe code, and the
one `unsafe` block is the feature-checked call in `dsv41-cuda` (`cpu.rs`).

## Roofline (batch-1 decode, *estimates*)

Hit rates are guesses until measured with route traces; LFRU on real routing usually
beats the capacity fraction.

| Setup | Experts resident (RAM+VRAM) | SSD bytes/token | Bound | Decode |
|---|---:|---:|---|---:|
| Today, hybrid path | ~71% | 0.7–1.3 GB | SSD 2 GB/s | **~1.5–3 tok/s** |
| + SSD-tier experts striped over both drives | ~71% | same, 2 drives | SSD ~4 GB/s | ~2.5–4 tok/s |
| 256 GB RAM (4×64 GB), hybrid | ~93% | <0.15 GB | RAM 43 GB/s | **~5–8 tok/s** |
| + DSpark speculative decoding (MTP heads in the checkpoint) | | | | see note |

*(Measured since: ~29 tok/s warm with 192 GB, see Status. On speculation: with the
hybrid path most of a token's cost is its own CPU-side experts, which a multi-token
verify does not share, so DSpark would gain far less here than on a GPU-resident model.)*

**Prefill** is the pain point. A chunk of ≥300 tokens touches ~99% of each layer's
experts, and 50 tokens touch ~55%. So every chat turn re-reads most SSD-resident
experts once: ~20–40 s per turn today, a few seconds with 256 GB RAM. Chunks amortize
well (one read serves every token in the chunk), so prefill in large chunks.

## What nrob already had

The starting point, as of 2026-09-18. The VENDORED-LOCAL module headers are more
current than `docs/ROADMAP.md`, which predates several of these:

- `nrob::ecache::Ecache` with `HostLease` (zero-copy hits), plus the `WeightStore` seam.
- VRAM expert cache `llama-rs/src/expert_stream/device_cache.rs` (CACHE-02 / STREAM-01).
- Pinned staging ring, async H2D stream and upload tickets: `ggml-rs-cuda/src/transfer.rs` (GPU-02).
- GPU top-k routing and grouped gate/up and down MoE kernels: `ggml-rs-cuda/src/moe.rs`,
  `llama-rs/src/moe_cuda.rs` (MOE-01/02).
- BPE tokenizers and samplers in the GGUF stack.

Since 2026-09-19 nrob reads only GGUF and safetensors: the `.nrob` container, the `xdb`
store and `nrob-server` are gone, and `dsv41` shares only the cache, the store seam, the
thread pool and the JSON reader with the rest of the workspace.

## What was missing

All of this has since been built (see Status), except the prefill GEMM in item 3 and
the chat encoder and serving in item 7.

1. **Safetensors reader.** An 8-byte length, JSON header, then raw data; the std-only
   JSON parser already exists.
2. **Dtypes:**
   - MXFP4 (`GgmlType` stops at 35; upstream ggml's MXFP4 is 39).
   - The FP8 e4m3 32×32 block format (BF16 expansion for v1).
   - E8M0 scales.
3. **Kernels, CUDA** (all current kernels are F32-activation × ggml-quant GEMV):
   - MXFP4 grouped GEMV for decode, plus GEMM for prefill.
   - BF16/FP8 GEMV for the trunk.
   - Sparse gather-attention with sink.
   - hc mixes + Sinkhorn.
   - Compressor pooling, indexer scoring, Engram gather and gate.
   - FP8/FP4 act-quant simulation.
   - All compile via NVRTC for sm_120.
4. **Kernels, CPU:** the AVX-512 MXFP4 expert kernel (hybrid path).
5. **The V4.1 forward** (`model.py` → Rust, ~3–5k lines). The Engram precompute
   exists (`oracle.py` writes `engram_meta.safetensors`). No converter is needed:
   experts are read in place (see Status).
6. **Multi-GPU:** two `CudaBackend`s, layer split, host hop. `--cuda` is device 0 only today.
7. **Chat/serving:** port `encoding.py` to Rust and serve the model.

## Status

**Phase A — done (2026-09-18).**

- **The checkpoint is used in place; there is no converter.**
  - Each expert's three weight tensors are adjacent in its shard. So are its three
    scale tensors, which sit in a separate region because the writer grouped
    tensors by dtype.
  - `dsv41::expert::SafetensorsExpertStore`, nrob's `WeightStore` over the original
    shards, therefore serves any expert in **2 positioned reads**.
  - Indexing all 96,085 tensors reads only the headers: **51 ms**.
  - Page-cache bypass works from safe Rust (`OpenOptionsExt::custom_flags` plus an
    aligned scratch window): 12–15 ms per 18.8 MB expert from the SSD, 3–5 ms from
    the page cache.
- **Oracle:** `tools/dsv41/oracle.py` runs the reference `model.py` unmodified on one
  GPU.
  - `torch_kernels.py` replaces tilelang with the same arithmetic.
  - Experts and Engram rows are streamed from the shards.
  - Prompt "What is the capital of France?" → *"The capital of France is
    **Paris**."* then EOS. Prefill 81 s for 11 tokens, 2.4–6.8 s per token.
  - Writes `golden.safetensors` (every layer's output and `pre_mix`, logits, Engram
    hashes, the first decode step, expert fixtures) and `engram_meta.safetensors`
    (token map, primes, offsets, multipliers).
- **Gate:** `crates/dsv41/tests/expert_oracle.rs` fetches routed experts (3, 6) and
  (3, 13) through the store and runs `expert_forward`.
  - It matches the oracle **bit for bit in bf16: 20,480 of 20,480 values, max diff 0**,
    with and without direct I/O. Warm, the test takes 0.19 s: store open 22–33 ms,
    fetch 3–12 ms, forward ~4 ms per token (scalar, 32 threads).
  - The first cold run took 19.8 s, spent outside the timed steps. The likeliest cause
    is antivirus scanning the 48 shards on first open, but that's unconfirmed. Watch
    for it in Phase B.
  - What is proven: Rust `expert_forward` == the reference forward as re-implemented in
    plain torch (`torch_kernels.py`). The tilelang kernels themselves never ran; they
    don't build on Windows.
  - The re-implementation is grounded in three things. The unmodified `model.py`
    produces a coherent answer and a clean EOS through it. The nibble order comes from
    the reference `convert.py`. The scale rounding is transcribed from `kernel.py`'s
    `fast_round_scale`.
  - Don't expect bit-identity later on: attention and tensor-core reductions will need
    tolerances, as ROADMAP.md says.
- `nrob::json::JsDoc::members` was added (a linear walk of an object's members), with
  a test.

**Phase B — full forward on the CPU (in progress, 2026-09-19).**

- **What's built.** `crates/dsv41` now holds the whole backbone, std-only with no
  `unsafe`:
  - `config`, plus `linear` (fp8 32x32-block / bf16 / f32 weights through the reference
    `linear()`);
  - `ops` (RMSNorm, YaRN RoPE), `hc` (hyper-connections plus Sinkhorn), `attention`
    (window ring, compressor, indexer, candidate blocks, sparse attention with sink,
    grouped low-rank output);
  - `moe` (sqrt-softplus router, batched fp4 experts through `Ecache`, fp8 shared
    expert), `engram` (hashing from the exported precompute, row reads from the
    shards), `model` (the backbone).
  - The trunk loads into RAM in ~10 s (~11 GB, stored dtypes).
- **Greedy decoding reproduces the oracle token for token** (`tests/forward_oracle.rs`):
  prompt 29.6 s for 11 tokens, 2.6 s per decode token, scalar CPU.
- **Per-layer correctness uses teacher forcing** (`tests/layer_isolation.rs`), which
  feeds each layer the oracle's input:
  - Isolated error is 0.04–2% per token, and up to 5% where the router flipped at a
    near-tie: 4 of 440 prompt token-routes, 0 in decode.
  - Logits are within 0.8%.
  - Attention output is exactly 0 error whenever its input matched bit for bit
    (decode layers 5, 7, 29, 31; 7e-8 at 37).
  - The residual ~1% comes from summation order (the GPU reduces in a different order).
    It flips an occasional bf16 ulp, and fp8 activation quantization turns that into
    an fp8 step (6%) for the element concerned.
  - Free-running, those compound to ~15% relative by the last layer, while every
    greedy token still agrees. So drift is gated in isolation, not end to end.
- **Bug found by the full run:** a direct read that runs past EOF ends on an unaligned
  byte, and Windows rejects the follow-up read. Fixed in `io::read_direct`, with a
  regression test that reads the last expert of every shard both ways.
- **Deliberate deviation from the reference:** on ratio-2 decode steps that complete no
  group, the reference's owning indexers score against stale (layer-20) keys; the port
  uses the layer's own keys (see `attention.rs`). No effect until ~1k tokens of context.

**Phase B — done (2026-09-19).**

- Long prompt (247 tokens, window wrap, compressed top-k and candidate filtering
  active): free-running greedy decoding matches the oracle token for token.
- Teacher-forced isolation passes. On tokens routed like the oracle, p95 ≤ 2.1% and
  max ≤ 6.2% per layer. 1.07% of token routes flip at near-ties. Logits are within 0.6%.
- The comparison lives in `dsv41::golden` and is shared by the CPU and GPU suites.
- The oracle carries the same index-key fix as the port.

**Phase C — C1, C2 and C3 done (2026-09-19).**

- **Where it lives:** `crates/dsv41-cuda`, a member but not a default member (it needs
  CUDA). It holds the only `unsafe`: cudarc kernel launches, one typed wrapper each,
  with a SAFETY note and length asserts.
- **Kernels:** 20 NVRTC kernels (`src/kernels.cu`, `--fmad=false`). Each is tested
  against its `dsv41` CPU function (`tests/kernels.rs`):
  - The fp8/fp4 quantizers, RoPE and hc pre/post are bit-exact.
  - GEMVs, norms, attention, pooling and the Engram gate are bit-exact after bf16
    rounding, or within 0.05 bf16 steps for the f32 outputs.
  - A whole expert is within 2.4e-6 relative.
- **C1** (`GpuModel`, one GPU, trunk resident in stored dtypes, ~11 GB):
  - Every golden gate passes on both prompts: greedy tokens identical, and isolation
    closer to the oracle than the CPU path (prompt route flips 0.23% / 0.37%, logits
    within 0.5%). GPU reduction order resembles the oracle's own.
  - The host keeps routing, index top-k, Sinkhorn, hashing and Engram rows, through
    the same `dsv41` functions.
- **C2** (`expert_cache.rs`): an LFRU VRAM expert pool, ~1,085 slots (20 GB) after the
  trunk on a 32 GB card.
  - Text generation, `examples/generate.rs` (prompt ids from
    `tools/dsv41/encode.py`, text out through `dsv41::detok`):
    - Cold: 1.54 s/token. Second 64 tokens: **0.46 s/token (2.2 tok/s)**.
    - 62.7% VRAM hits overall, still warming.
  - Output is fluent and correct (the rainbow explanation).
- **C3** (layers 0–19 on cuda:1, 20–39 on cuda:0, each card caching its own layers'
  experts; ~2,700 VRAM slots in all) passes every gate on both cards. Only the hidden
  stream and the mix buffer cross cards, through the host.

**Speed work (Phase D in part) — 2026-09-19.** Target was >5 tok/s decode.

| Run (`generate.exe rainbow.ids N 1,0`) | Decode |
|---|---:|
| C3 as first built, second half of 256 tokens | 0.39 s/token (2.6 tok/s) |
| One process, 3rd answer to the same prompt (`DSV41_RUNS=3`), tiers warm | **0.065 s/token (15.0 tok/s)** |
| Same process, 2nd answer (host tier 99% warm) | 0.10 s/token (9.7 tok/s) |
| Fresh process, warm start from the usage file, healthy SSD | 5.8 tok/s over 388 tokens (second half 9.4) |
| Fresh process, warm start, SSD throttled | 2.0 tok/s over 402 tokens |
| First run ever (no usage file) | ~1.7 tok/s |

What changed, in order of effect:

- **GEMV/expert kernels were latency-bound** at ~150 GB/s: byte-at-a-time loads, and
  FP4 decoded through a `__constant__` table (lanes with different nibbles serialize).
  Found with Nsight Systems; `DSV41_PROFILE`'s synchronizing timers hid it. The rewrite
  uses 16-byte vector loads, a shared-memory FP4 table and branch-free fp8 decode.
  - The fp8, fp4 and grouped-expert kernels keep each lane's summation order, so their
    results are unchanged.
  - `gemv_bf16` (8 consecutive weights per lane) and `hc_project` (float4) reassociate.
    That is within tolerance: one more near-tie route flip on the long prompt, 38 vs 37.
  - Times: `moe_gate_up` 57 µs and `wq_b` 43 µs per decode call, measured with a decode-shape
    bench. For comparison, the old kernels' nsys averages were 421 µs and 150 µs, but
    those include the prefill's multi-token calls, so they are not a like-for-like
    comparison.
- **Hybrid decode** (`dsv41::cpu_experts`): a routed expert that misses VRAM is computed
  on the CPU from the host tier, while the GPU runs the resident ones. An 18.8 MB record
  over these PCIe links costs 1.3–2.6 ms; the CPU pass costs ~1 ms. The kernel is
  bit-identical to `expert_forward` (SIMD lanes run across rows, so no sum is
  reassociated), on a persistent pool, and results land in their expert-id rows.
  VRAM admission is LFRU with a frequency test and at most one upload per layer per
  token.
- **Grouped decode experts:** one launch per stage for all six experts, from a packed
  table (record address, output row, route weight).
- **hc_project** ran its 25 dot products serially in one block per token; now one
  block per output (62 → 8 ms per token).
- **Host overhead:** cudarc event tracking off (one stream per device), uninitialized
  allocations for fully written outputs, out-of-place activation quantizer (no copy).
- **Tiers across runs:** `GpuModel::save_usage` / `warm`. At start, each card's VRAM
  gets the hottest experts of its layers (seeded frequencies). The host tier then
  fills in the background, pausing for prefills and 50 ms after any demand read.
  Engram rows are read on a background thread from the start of each forward.
- **Bug fixed on the way:** prefill never started a new cache batch, so every slot it
  touched stayed pinned. Any prefill larger than VRAM would have failed.

Gates after all of it: the short and long golden files pass on both cards, with the
upload-every-miss path and with the hybrid path (forced with `DSV41_VRAM_EXPERT_GB=1
DSV41_CPU_THREADS=24`, ~1,300 decode experts on the CPU). Tokens match, prompt route
flips are 0.23% / 0.38%, and logits are within 0.5%.

**Second round (2026-09-19, same day).** Warm decode 15.0 → **~29 tok/s**
(3rd answer in one process; 0.034 s/token; a fresh process's first answer ~5.7). Each step was held to the
golden gates on both paths:

| Change | 3rd-answer tok/s |
|---|---:|
| start of round | 15.0 |
| AVX-512 CPU expert kernel (`cpu_experts::avx512`, safe `#[target_feature]` code; the one `unsafe` is the feature-checked call in `dsv41-cuda::cpu`) | 19.7 |
| hardware fp8 decode/rounding (`cvt.rn.f16x2.e4m3x2`, `cvt.rn.satfinite.e4m3x2`), fused quantizer + GEMV, parallel `hc_mix` / `index_scores` | 20.9 |
| launch-ahead: CPU experts hand their rows to an already-queued reduction through pinned memory (`handoff.rs`) | 21.6 |
| arithmetic e4m3 rounding on the CPU (the table search was ~0.2 ms of every CPU expert call), CPU job started before the GPU launches, 1 GiB VRAM headroom | 23.6 |
| attention reads the window and compressed caches in place, linear RoPE positions, indexer weights scaled in the kernel | 26.1 |
| router logits and x published to the host through a pinned mailbox, not two driver downloads | 27.2 |
| fused kernels: shared-expert gate/up/SwiGLU, kv norm/rope/quant/window write, inverse RoPE inside attention, hc projection + Sinkhorn (last-block ticket), fp8 `wo_a` | 28.6 |
| one-token fp8 GEMV computes two rows per warp (`gemv_fp8_token`) | 29.1 |

Everything new is bit-identical to what it replaced, except `rmsnorm`'s
1024-thread reduction and `moe_down`'s half-warp rows (both within the
tolerances; one more near-tie route flip on the short prompt, 2 vs 1 of 440).

A warm token (nsys, 36 ms at the time of the mailbox change): ~9 ms the GPU
spins on CPU experts, ~17 ms of kernels, ~10 ms idle between them.

**What limits it now:**

- **The SSD.** ~2 GB/s at best, and it falls to ~0.5 GB/s after a minute of
  sustained reads (likely thermal). A document touches ~8,000 experts (150 GB), so a
  fresh process spends minutes filling the host tier. A long-lived process (server/chat)
  pays that once; the one-shot example pays it every run.
- **CPU experts** (~23 per token): now at the RAM's bandwidth (0.44 GB a token at
  42.8 GB/s ≈ 10 ms). The memory runs at DDR5-3600 (4 DIMMs); a faster memory
  setting in the BIOS would cut this directly.
- **Kernel launches** (~25 per layer, ~6.5 µs each on WDDM): the GPU waits on the
  host in the stretches of small kernels. More fusion, or CUDA graphs with
  device-side parameters, would remove most of it.
- **Prefill** still uploads every VRAM-missing expert over the x2/x4 links (a
  24-token prompt takes ~3 s warm); routing prefill misses to the CPU as decode
  does is the next step for time-to-first-token.

**Phase E — serving (2026-09-19).**

- **Rust tokenizer** (`dsv41::tokenizer`): the checkpoint's byte-level BPE, with the
  added-token split, the three regex pre-tokenizer splits (hand-written, Unicode classes
  from a generated table), and rank-ordered merges.
  - It matches the reference `PreTrainedTokenizerFast` on all 3,638 cases of
    `tools/dsv41/tokenizer_golden.py` (667K tokens: this repo's code, CJK, digits,
    whitespace edge cases, special tokens, random Unicode).
- **Rust chat format** (`dsv41::chat`): `encoding.py` ported rule for rule, covering
  tools, DSML calls, thinking modes, effort, tasks and images.
  - It matches the reference on all 420 cases of `tools/dsv41/chat_golden.py`,
    including the 17 inputs the reference rejects.
  - A streaming parser splits a reply into reasoning, content and tool calls as it
    arrives.
- **Chunk continuation:** `forward(ids, start_pos)` now takes any number of tokens at
  any position.
  - Attention reads [window ring ++ the chunk's own keys], oldest first, as decode does.
  - Compressor groups straddle chunks; each query sees the compressed rows its own token
    completes.
  - A prompt in chunks gives the logits of one prefill *bit for bit* (rel-L2 0.0 on the
    long golden, cuts at odd positions and one chunk wider than the ring). Greedy tokens
    match the oracle.
- **Checkpoints** (`checkpoint` / `restore`, ~10 MB in host RAM): only the window rings
  and partial compressor groups need saving. The compressed caches below the checkpoint
  never change, and above it they are rewritten before being read. Restoring replays
  exactly.
- **Layer-by-layer prefill** (`prefill_layered`): every layer runs over the whole
  stretch before the next starts. Attention goes in continuing sub-chunks, the routed
  experts run over all tokens at once, and the residual stream waits in host RAM.
  - Each expert is read once per stretch, not once per chunk. Bit-exact to chunked
    prefill.
- **Exclusive tiers:** after the warm start, the VRAM-resident experts leave the RAM
  tier, whose ~2,900 freed places go to the next-hottest experts. That is about 10,400
  experts resident instead of about 8,000 once the fill completes.
- **`nrob-server`**: an OpenAI-compatible API (chat completions, streaming, tools,
  reasoning) over one model worker.
  - A prefix cache: the live state plus checkpoints every 256 prompt tokens, at
    user-turn boundaries and at the end of each layered pass.
  - Short prompt stretches run token by token through the decode path, long ones layer
    by layer.
  - The background cache fill waits for idle time.

Measured (both GPUs, 140 GB RAM tier):

- **The SSD is the wall, and it is thermal.** `bench_ssd` reads 1.7–2.0 GB/s for the
  first ~15–20 GB, then settles at **0.51 GB/s**.
  - A 1,024-token chunk touches nearly all 15,360 experts: 8,683 SSD reads (163 GB),
    286 s, **3.6 tok/s**.
- **Layered prefill:** 5,560 tokens (a harness-style system prompt with tools) in
  251 s, **22 tok/s**, about 6× the chunked rate.
- **Prefix cache:** a new chat with the same system prompt reused 5,560 of 5,580 tokens
  (10 s instead of ~4.5 min). A tool-result follow-up reused 360 of 392.
- **Decode on a new topic:** 1.4–8 tok/s while the tiers adapt (VRAM hit 22% on a
  coding prompt whose usage profile came from prose). Warm and on topic, it is still
  ~29 tok/s.
- **External Samsung T9** (USB 20 Gbps, the model copied in 11 min): `bench_ssd` holds
  **1.95 GB/s over 75 GB** where the internal drive falls to 0.51.
  - Layered prefill of 5,560 tokens: **181 s, 30.7 tok/s** (22.2 from the internal
    drive).
  - That pass reads 10,894 experts (205 GB): expert fetch 81 s, expert compute 31 s,
    the rest 69 s (attention, hyper-connections, the shared experts, and the residual
    stream's trips over PCIe).
  - A pool of 4 readers now pulls a layer's experts into RAM ahead of the compute loop.
- **VRAM frequency aging:** the VRAM cache halves its usage counts every 128 decode
  tokens, and a saved profile seeds at most 64.
  - Before, the seeded counts from an old profile (median 47, max 11,512) kept the
    wrong experts in VRAM: a new topic stayed at 22–29% VRAM hits.
  - With aging, the coding prompt's three runs went 42% → 67% → 77% VRAM hits and
    2.6 → 8.5 → **12.1 tok/s** (7.2 before).

### Vision (Phase G)

The reference (`image_processor.py`, `vision.py`) decodes with Pillow, pads to a
14-pixel grid with bicubic resampling, runs a 32-block ViT (dim 1024, 2-D RoPE) and a
3×3 aligner into LLM rows, and fills an image span `[START] (IMAGE×w NEWLINE)×h [END]`
whose positions all carry `<｜deepseek_image｜>`.

- **Decoding, our own:** `crates/nrob-image` (std-only, no `unsafe`).
  - PNG with its own inflate: every colour type, bit depth, Adam7, every filter,
    stored, fixed and dynamic deflate blocks.
  - JPEG: baseline and progressive, libjpeg-turbo's integer IDCT, fancy upsampling
    and colour tables; CMYK/YCCK through Pillow's `CMYK;I` conversion.
  - Pillow's quirks kept: alpha dropped without compositing, 16-bit samples keep the
    high byte, 16-bit grey clips at 255, no EXIF rotation.
  - **Pixel-exact to Pillow on all 111 test images**
    (`tools/dsv41/vision_golden.py`). Truncated and bit-flipped files return errors,
    never panics.
- **Preprocessing:** Pillow's two-pass fixed-point resampling and `ImageOps.pad`
  (Python's half-to-even rounding included), then the f32 normalization. The patches
  are **bit-exact** to the reference's on 9 images, and the sizing plan on 5,712 sizes.
- **The tower** (`dsv41::vision::VisionTower` on the CPU as the oracle,
  `dsv41_cuda::GpuVision` on the GPU): about 1 GB of bf16 weights on the first device,
  loaded before the expert cache sizes itself.
  - Each block alone is within ~1e-3 of the reference.
  - End to end, 1.6–2.5% (rel-L2, GPU). bf16 noise compounds through the blocks from
    12 on, which grow outlier activations (up to ~1,800). torch's own CPU bf16 run is
    3.2% from its GPU run, exact fp32 4.8%.
  - 60 ms for 1,521 patches, 160 ms for 3,672, with a SIMT GEMM and a
    one-thread-per-query attention.
- **In the model:** `GpuModel::forward_with` / `prefill_layered_with` take image
  spans (absolute positions, any chunking).
  - Image tokens take the span rows as embeddings and route with `gate.bias_vl`.
  - They enter the n-gram history as DEAD, and their Engram rows are zero, so the gate
    adds exactly nothing.
  - A split inside an image, and layered prefill, agree with one prefill bit for bit.
- **Against the oracle** (`oracle.py --image`, the mascot plus "Describe this image in
  one sentence.", 196 tokens):
  - Teacher-forced, every layer's p95 is ≤ 1.3e-2, and 38 of 7,840 token-routes differ
    (near-ties). Final logits are within 5.9e-3.
  - Free-running, the prompt is chaotic: one bf16 ulp on 1% of the image rows moves our
    own final logits by 24%, about as far as we land from the oracle (27%). The replies
    agree in substance:
    - oracle: *"The image shows a minimalist, dark-themed icon of a stylized robot with
      a smiling face, antenna, and boxy…"*
    - nrob, own pipeline end to end: *"The image displays a minimalist, line-art icon
      of a stylized robot with a smiling face, antenna, and boxy…"*
- **Serving:** `image_url` parts (base64 `data:` URLs, or local paths on a loopback
  server) are decoded and sized on the request thread. The worker encodes only the
  images the uncached part of the prompt reaches, and the prefix cache keys image
  positions by content.
  - A chart (254 image tokens) read back exactly: title, four values, bar colours.
  - A follow-up turn reused 363 cached tokens, image included (prompt 6.8 s).
  - Image tokens run token by token like other short stretches: 2.7 tok/s cold (the
    vision-biased experts are not cached yet), 8.5 tok/s warm.

## Phases

Each phase ends in something measurable. Effort figures are rough.

- **A — oracle, in-place expert store, first expert (done; see Status).**
- **B — full forward, CPU reference path (1–2 weeks).**
  - All 40 layers in f32, validated layer by layer against the golden dumps.
  - Engram, attention sharing and hyper-connections exact.
  - First correct tokens (slow).
- **C — CUDA trunk on both GPUs + VRAM expert cache (done).** First real decode
  speed and measured hit rates (route traces).
- **D — hybrid CPU experts (done, except SSD-tier striping).** AVX-512 kernel, CPU/GPU
  overlap (launch-ahead), warm tiers across runs; ~29 tok/s warm.
- **E — usable (done, 2026-09-19).** Rust tokenizer and chat format, chunk
  continuation, checkpoints, layered prefill, the OpenAI-compatible server.
- **G — vision (done, 2026-09-19; see Status, "Vision").**
- **F — next.**
  - A longer oracle golden (≥2K tokens) to verify long contexts.
  - Stripe the SSD-tier experts over both drives.
  - CUDA graphs for decode, then DSpark speculation.

## Hardware changes, by impact on this model

1. **256 GB RAM** (4×64 GB DDR5). Takes the SSD out of the decode path and most of the
   prefill path. The biggest single win.
2. **More SSD bandwidth.** The E: drive throttles to 0.51 GB/s after ~15–20 GB of
   sustained reads, a quarter of its cool speed.
   - A heatsink or airflow on that M.2 slot.
   - Striping the SSD-tier experts over both drives (C: has room for about half the
     expert shards).
   - A faster Gen5 drive for the expert store.
3. **GPU links.** Two GPUs on AM5 can get at most **x8/x8** (16 CPU graphics lanes), not
   x16/x16. x2 and x4 are below even that. To fix:
   - Reseat both cards and use a sag bracket; a riser cable is the prime suspect.
   - Check the BIOS slot bifurcation mode (Auto / x8+x8, not a RAID x4 split).
   - Check the manual's M.2 lane-sharing table.
   - As a diagnostic, force Gen4: if the width recovers to x8, it's Gen5 signal
     integrity.

   This matters most for prefill and the pure-GPU path; the hybrid decode path barely
   depends on it.
