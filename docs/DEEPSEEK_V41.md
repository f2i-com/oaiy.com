# DeepSeek-V4.1-Flash on OAIY — feasibility and plan

**Target:** `DeepSeek-V4.1-Flash-FP8` (510.3 GB, 55 files, commit
`d61c59ea`), weights in `E:\deepseek\model`, reference code (`inference/model.py`,
`kernel.py`, `engram.py`, `encoding/`) in `E:\deepseek\reference`.
**Machine:** Ryzen 9 9950X3D, 192 GB DDR5, 2× RTX 5090 (32 GB), 2× Crucial P3 Plus 4 TB.
**Date:** 2026-09-18. Every number below is measured on this machine or read from the
checkpoint headers unless marked *estimate*.

> **Where it runs now (2026-10-08).** This page is the plan and the record of the CUDA
> engine (`dsv41-cuda`), and its figures are that engine's. The engines' one GPU backend
> is WebGPU now, and the CUDA engine stands on the branch
> `backup/cuda-support-2026-10-08`. On main the server runs this model through `dsv41`'s
> CPU model with its dense trunk and busy experts on a WebGPU adapter
> ([WEBGPU.md](WEBGPU.md#deepseek-v41)): about a token a second, a long first prompt
> taking minutes, and no images yet.

## Verdict

Feasible, and a good fit for OAIY's design: this is a 552B sparse MoE whose active set per
token is small (6 of 384 experts × 40 layers), and OAIY already has the streaming pipeline
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

None of this exists in OAIY or llama-rs today:

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

1. **GPU streaming** (the existing OAIY path). Every RAM-hit expert crosses PCIe. At
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
verify does not share, so DSpark would gain far less here than on a GPU-resident model.
Since built for the EXL3 checkpoint, whose experts are all computed on the cards: 10-45%
faster where they hold what is used, see "DSpark drafting".)*

**Prefill** is the pain point. A chunk of ≥300 tokens touches ~99% of each layer's
experts, and 50 tokens touch ~55%. So every chat turn re-reads most SSD-resident
experts once: ~20–40 s per turn today, a few seconds with 256 GB RAM. Chunks amortize
well (one read serves every token in the chunk), so prefill in large chunks.

## What OAIY already had

The starting point, as of 2026-09-18. The module headers are more
current than `docs/ROADMAP.md`, which predates several of these:

- `oaiy_engine::ecache::Ecache` with `HostLease` (zero-copy hits), plus the `WeightStore` seam.
- VRAM expert cache `oaiy-llm-server/src/llm/expert_stream/device_cache.rs` (CACHE-02 / STREAM-01).
- Pinned staging ring, async H2D stream and upload tickets: `ggml-rs-cuda/src/transfer.rs` (GPU-02).
- GPU top-k routing and grouped gate/up and down MoE kernels: `ggml-rs-cuda/src/moe.rs`,
  `oaiy-llm-server/src/llm/moe_cuda.rs` (MOE-01/02).
- BPE tokenizers and samplers in the GGUF stack.

Since 2026-09-19 OAIY reads only GGUF and safetensors: the `.oaiy` container, the `xdb`
store and `oaiy-llm-server` are gone, and `dsv41` shares only the cache, the store seam, the
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
   - All compiled through NVRTC (since 2026-10-05 at build time, a cubin per GPU
     architecture: `crates/oaiy-cuda-build`).
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
  - `dsv41::expert::SafetensorsExpertStore`, OAIY's `WeightStore` over the original
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
- **Gate:** `crates/oaiy-llm-server/tests/dsv41_expert_oracle.rs` fetches routed experts (3, 6) and
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
- `oaiy_engine::json::JsDoc::members` was added (a linear walk of an object's members), with
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
- **Kernels:** 20 kernels (`src/kernels.cu`, `--fmad=false`), compiled through NVRTC at
  build time into a cubin per GPU architecture (`build.rs`). Each is tested
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
  - Text generation, `examples/dsv41_generate.rs` (prompt ids from
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
- **`oaiy-llm-server`**: an OpenAI-compatible API (chat completions, streaming, tools,
  reasoning) over one model worker.
  - A prefix cache: the live state plus checkpoints every 256 prompt tokens, at
    user-turn boundaries and at the end of each layered pass.
  - Short prompt stretches run token by token through the decode path, long ones layer
    by layer.
  - The background cache fill waits for idle time.

Measured (both GPUs, 140 GB RAM tier):

- **The SSD is the wall, and it is thermal.** `bench_ssd` reads 1.7–2.0 GB/s for the
  first ~15–20 GB, then settles at **0.51 GB/s**.
  - It recovers completely: after ~8 minutes idle the same drive reads **1.94 GB/s at
    queue depth 1 and 2.89 GB/s at depth 8** again (2026-09-20). Back-to-back runs
    measure a throttled drive, not the model.
- **Where a layered pass actually goes (nsys, 2026-09-20).** A 40 s window inside a
  16,800-token pass on both cards, which offers 80 GPU-seconds:
  - **Kernels: 14.6 GPU-s, ~18% occupancy.** `sparse_attn` 5.7 s (39%, 215 calls at
    26.6 ms), `gemm_fp4` 3.4 s (13,068 calls), `gemv_bf16` 2.1 s (192 calls, one of
    them 134 ms), `index_scores` 1.7 s (45 calls at 37 ms), `gemm_fp8` 1.1 s.
  - **Copies: 8.6 GPU-s, and more of it leaves the card than arrives** — 691
    device-to-host copies at 6.5 ms average (4.5 s) against 4.1 s host-to-device.
  - **The cards are idle ~60% of the pass** (`nvidia-smi` over the whole pass: GPU 0
    25.9%, GPU 1 18.9%, memory controller 0.5%). Expert *compute* is ~16 s of a 150 s
    pass; the experts are not the problem.
  - **The residual stream's host round-trip is.** `layered_inner` keeps `h`, `h1` and
    the mixes in host `Vec`s and, for every layer and every attention sub-chunk,
    uploads them and downloads the result (`model.rs:1068`, `:1091`, `:1152`). At
    `HC * DIM * 4` = 80 KB a token that is ~170 GB each way per pass (~21 s of DMA at
    the staged rates), but the real cost is that `download` is `clone_dtoh`, which
    blocks until the whole kernel queue drains: 17 sub-chunks x 2 loops x 60 layers is
    ~2,000 pipeline stalls a pass.
  - **Lever (1) done, 2026-09-20: 24% off a long pass.** `GpuOptions::residual_on_device`
    keeps `h`, the mixes and `h1` on the card between layers that share one, so the
    residual crosses PCIe once a pass instead of sixty times. Same 16,800 tokens from
    the T9, 6 GB headroom in both runs, nothing else changed:

    | | residual on the host | on the device |
    | --- | --- | --- |
    | pass | 151.0 s (111.2 tok/s) | **121.7 s (138.0 tok/s)** |
    | "the rest" | 107.3 s | **73.3 s** |
    | routed experts | 37.2 s | 40.8 s |
    | drive reads | 4,850 | 4,849 |

    The 34 s came out of "the rest", which is where the round trip lived. The logits
    agree to the bit (argmax 204 at 31.117918, sum 108290.638676), as do the golden
    gates and `prefill_bench`'s 4,096-token four-sub-chunk comparison.
  - **It needs headroom to engage.** At `layered_max` 20,480 the carry wants
    `2 * HC * dim * 4` a token = 3.36 GB, and `None` declines unless the card keeps
    half a gigabyte spare besides, so the server's default `headroom_gb = 2.0` leaves
    it on the host. `--vram-headroom-gb 6` turns it on and costs ~457 VRAM expert
    slots (drive reads went 4,402 to 4,850, +8.4 GB) - worth it here, and worth less
    on a drive that reads at 0.5 GB/s rather than the T9's 1.95. Sizing the carry into
    the expert cache at load, instead of asking headroom to cover it, is the follow-up.
  - **Lever (2) done: expert uploads on a copy stream.** `get_prefill` had one staging
    slot, uploaded on the compute stream, so an upload could not start until the last
    expert's kernels had finished. Two slots and a copy stream, ordered by events,
    took the pass 121.7 s to **117.3 s** (138 to 143 tok/s), uploads 33.6 s to 26.8 s.
    Logits identical over 14,628 expert uses; the gates unchanged.
  - **What the cards are doing now: the same 22%.** After both levers GPU 0 averages
    23.5% and GPU 1 20.0% across the pass, idle in 62-68% of samples. That is
    arithmetic, not a fault: kernel work is ~55 GPU-seconds and 55 / (2 x 117) is 23%.
    The levers removed stalls rather than adding parallelism, so the pass shrank
    around the same maths.
  - **Three that did not pay, measured.** Same 16,800-token pass, logits identical
    every time:
    - Staging ring 4 x 4 MB to 6 x 8 MB, so an 18.8 MB record stops wrapping it
      mid-copy: 119.1 s against 117.3 s. Reverted - once the copy has a stream of its
      own, the host is not waiting on that ring.
    - Attention sub-chunks of 2,048: 119.6 s ("the rest" 75.3 s). Of 4,096: 124.5 s
      (78.0 s). "The rest" grows with the sub-chunk, so it is attention widening and
      not host overhead; 1,024 is already the right size.
  - **`DW::forward` has no GEMM for bf16 weights** (`model.rs:409`). FP8 weights go to
    `gemm_fp8` on tensor cores whenever a call carries more than one token; bf16
    always takes `gemv_bf16`, one warp per output row looping over all `nt` tokens
    inside the kernel. Over a 16,800-token pass that is an untiled gemv replayed
    16,800 times - the nsys window put it at 10.9 ms a call (one at 134 ms) and ~8
    GPU-seconds a pass. A `gemm_bf16` closes it, at the cost of an accumulation order
    that has to stay inside the oracle bounds.
  - **Next levers, in order:** (1) a `gemm_bf16` for multi-token calls, worth ~8
    GPU-seconds a pass; (2) `sparse_attn`, 39% of kernel time at 26.6 ms a call and the
    largest single cost left; (3) the contiguous layer split, which lets one card
    compute at a time and caps average occupancy at 50%. Note (3) fights the
    layer-major order that reads each expert once a pass: pipelining two token stages
    across the two layer groups would read the experts twice (+36 s) to recover about
    as much compute, so it needs more than the obvious version to pay.
- **PCIe width is not the prefill limiter (2026-09-20).** Both cards had trained
  narrow — GPU 0 at x2, GPU 1 at x4 — from GPU sag; laying the case on its side took
  them to x4 and x8. The same 16,800-token layered pass from the T9 ran **151.8 s
  before and 150.0 s after**, with uploads 32.7 s → 32.5 s. A pass uploads ~165 GB in
  ~32 s, about 5 GB/s aggregate, where even the narrow links carried ~24 GB/s: the
  uploads are bound by the host side (RAM-tier reads, staging, per-record overhead),
  not by the bus. Measure host-side work before spending anything on lanes.
  - Two traps when re-measuring: `DSV41_MODEL` defaults to the internal drive while
    these numbers come from the T9 (`D:\deepseek\model`), and `DSV41_PROFILE` moves
    time between the lines it prints (see `prefill_bench`'s header).
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
- **Prefill ate its own future (fixed 2026-09-19).** `prefill_bench` now counts, per
  pass, the experts used, those resident when the pass began and when their layer ran,
  and the drive reads (`GpuModel::pass_stats`, no syncs). A second 5,560-token pass
  with a full RAM tier used 13,593 experts; 10,198 were resident at the start, but only
  2,640 when their layer ran, and it read 11,058 (208 GB), the same as the first pass.
  - Why: a pass uses each layer's experts once, so every record it reads is the least
    used around. RAM's LFRU then evicts, for it, a record of a layer the pass has yet
    to reach. VRAM did the same to everything but its seeded experts (unseeded, not one
    expert was in place when its layer ran, even on a second pass).
  - Fix: `Ecache::set_scan_layer`. During a prefill at layer L, room in RAM comes from
    layers below L first (done until the next pass). In VRAM, prefill does not admit
    at all: misses go through one staging slot per device
    (`DeviceExpertCache::get_transient`), and the decode hot set stays put.
  - Same pass, RAM full: **3,230 reads (61 GB) instead of 11,058 (208 GB); 121.7 s
    instead of 178.5 s (45.7 tok/s instead of 31.2)**. 10,364 of the 10,612 experts
    resident at the start were still there when used.
  - Where the time goes, profiled on a 148 s variant of the pass (no VRAM warm-up,
    5,969 reads): expert fetch ~68 s (drive reads, and uploads of RAM-resident experts
    over PCIe), expert compute ~30 s (GEMV-style FP4 kernels at ~3 TFLOPS), the rest
    ~50 s. In the 122 s pass the fetch share is ~40 s (3,230 reads, ~31 s of drive
    time, plus ~10 s of uploads); compute and the rest are the same.
- **The usage profile no longer shrinks.** The server rewrote `expert_usage.txt` after
  every request with only that process's (aged) VRAM counts. A short session replaced
  what longer ones had learned: the RAM warm-up fell from 7,596 queued experts to 850.
  The model now counts every routed expert it runs (prompt and reply) and merges the
  counts into the file, older counts halving every 200,000 tokens.
- **Tensor cores for prefill.** The prompt's routed experts (`gemm_fp4`), its fp8
  projections (`gemm_fp8`) and the grouped `wo_a` (`gemm_fp8w`) run as bf16
  `mma.sync` tiles instead of GEMVs. Every operand is exact in bf16 (e2m1 and e4m3
  values times power-of-two scales, fp8-quantized or bf16 activations), so the products
  are the reference's; only fp32 summation order changes. A token's outputs do not
  depend on how many tokens share the launch, so chunked, layered and one-shot prefill
  stay bit-identical. 5,560-token pass, RAM full: **76.0 s, 73.2 tok/s** (121.7 s
  before, 178.5 s at the start of the day).
- **VRAM learns from prompts at the end of the pass.** Frequencies count tokens (an
  expert 300 prompt tokens used counts 300); prefill misses go through the staging slot
  and, when the pass ends, the most-used ones move in from RAM where they beat the
  victim (`DeviceExpertCache::admit_pending`). Admitting mid-pass evicted experts the
  same pass needed later; admitting none cost the reply.
- **Whole requests, before and after** (`generate`, a 1,327-token code-review prompt,
  200 tokens, three runs in one process, the same starting profile each time):

  | Run | Prompt, before → after | Reply, before → after |
  |---|---|---|
  | 1 | 120.4 → 101.9 s | 3.68 → 5.37 tok/s |
  | 2 | 71.4 → 34.8 s | 7.19 → 8.42 tok/s |
  | 3 | 63.2 → 34.1 s | 6.72 → 9.36 tok/s |

  The three requests take 253 s instead of 367 s. (Staging alone, before the tensor
  cores and the end-of-pass admission: 292 s. The reply figures isolate the admission,
  since decode runs the same kernels: its third reply was 5.83 tok/s, now 9.36.)
- **The long golden's greedy tokens** flip at one step (the 6th of 8, 1.0-logit margin)
  with the tensor-core prefill. Its free-running logits are 0.175 rel-L2 from the
  oracle's, the GEMVs' 0.216 (they happened to match all 8); the two kernels' outputs
  differ by well under a quarter bf16 step, so this is the image prompt's sub-ulp
  sensitivity again, not a kernel error. In isolation the logits got closer (3.70e-3,
  was 5.05e-3) while a few more near-tie routes resolved the other way (long golden
  48 of 9,880, was 37; image prompt 56 of 7,840, was 38). The greedy gate now takes
  exact tokens, or else logits within 0.25 and at most one teacher-forced disagreement
  under 1.5 logits (calibrated bounds; the kernel tests and isolation rank above it);
  the chunked and layered tests compare with one prefill's reply, not the oracle's.
- **Token by token vs layered, re-measured with the fixed pass** (RAM full): layered
  1,000 tokens in 43.1 s, 2,000 in 65.8 s (about 20 s + 23 ms a token); token by token
  at its warm best 69 ms a token (much slower on a new topic). They cross near 450
  tokens, so `oaiy-llm-server --step-below` now defaults to 512 (was 2,048).
- **Prompt tokens fed one by one skip the output head** (`GpuModel::advance_with`):
  warm, 13.98 → 14.54 tok/s (3 ms a token); only the last prompt token needs logits.
- **VRAM frequency aging:** the VRAM cache halves its usage counts every 128 decode
  tokens, and a saved profile seeds at most 64.
  - Before, the seeded counts from an old profile (median 47, max 11,512) kept the
    wrong experts in VRAM: a new topic stayed at 22–29% VRAM hits.
  - With aging, the coding prompt's three runs went 42% → 67% → 77% VRAM hits and
    2.6 → 8.5 → **12.1 tok/s** (7.2 before).
- **Uploads go through pinned memory** (`Gpu::write`). `bench_h2d`: from pageable
  memory an upload ran at 2–4 GB/s, from pinned 14.3 GB/s on the x4 card and 7.2 on
  the x2 one (downloads from pageable memory were fine: 12.3 and 6.9). Copies of 1 MB
  and up now pass through four 4 MB pinned, write-combined buffers per device, each
  chunk's copy queued asynchronously and marked by an event that the buffer's next use
  waits for. The 5,560-token pass (RAM full) spent 28.5 s in uploads instead of 34.0,
  but it took the same time (77.3 s, 76.6–77.2 before): the drive was now the limit,
  and waiting for bytes went from 12.9 to 17.2 s.
- **The next layer's experts are read ahead** (`Speculative`, `Prefetched`). When a
  layer's experts are done and the last layer used at least 60% of its experts (a long
  prompt: 89% at 5.6K tokens), two readers start on the next layer's experts that
  neither VRAM nor RAM holds, most used first, while its attention runs. They read into
  a pool of at most 160 records (~3 GB) outside the RAM tier; a record moves into RAM
  only when its layer uses it, and what the layer leaves is dropped.
  - Admitting the guesses straight into RAM: 72.2 s, but 4,637 drive reads; each guess
    evicted a record a later pass needed.
  - Starting the next layer's reads while this layer's own were still going: 81.6–83.1 s.
    The two split the drive.
  - As built: **73.1 s, 76.1 tok/s**, 3,876 reads.
  - What is left in that pass: ~200 GB of uploads over PCIe (at least 21 s at the
    cards' rates), the drive's reads, and ~29 s of attention, hyper-connections and the
    residual stream's trips.
- **Engram rows were read one at a time.** A token needs 24 rows from each Engram
  layer's table (layers 1 and 14), each row two small reads at random places in a
  101 GB shard. `bench_engram` (new rows each run): a decode step's 48 rows took
  12.9 ms from the T9 with 16 or 24 threads alike, and a 2,000-token prompt's rows
  20.3 s. Windows serializes reads through one synchronous file handle, and the
  threads shared one per table.
  - Now each table has 24 reader threads with their own handles, living as long as the
    table (spawning threads per lookup cost 0.9 ms a layer even with every row in the
    page cache). A row several tokens share is read once, and the first Engram layer's
    rows are read before the second's, which are not needed until layer 14.
  - A decode step's rows: **2.2 ms** (0.28 ms from the page cache). The 2,000-token
    prompt's: **3.1 s**.
  - A 5,560-token prompt the model had never seen (tokio's sources, so none of its rows
    were in the page cache): layer 1 waited **23.9 s** for its rows in a 152.2 s pass.
    Another unseen prompt (aho-corasick's sources) with the fix: **1.2 s** in 130.6 s,
    with about the same drive reads.
  - Decode: see the table below.
  - Expert reads did not have the problem: large direct reads already overlap on one
    handle (`bench_ssd` with four handles per shard: the same 1.8 GB/s from the T9,
    2.6–3.0 GB/s from the internal drive). With the fix the internal drive serves Engram
    rows as fast as the T9 (3.0 ms a step); before, 23–155 ms.
- **Decode with the Engram fix** (`generate`, the rainbow prompt, 256 tokens, three
  runs in one process from the same starting profile; readers still spawned per lookup):

  | Run | Engram wait, before → after | Decode, before → after |
  |---|---|---|
  | 1 | 5.9 → 1.5 ms a token | 2.88 → 3.05 tok/s |
  | 2 | 5.7 → 1.8 ms a token | 9.24 → 10.11 tok/s |
  | 3 | 2.9 → 1.3 ms a token | 11.46 → 12.26 tok/s |

  VRAM hits were the same in each pair (45%, 76%, 85%), so the gain is the rows'. The
  persistent readers take another ~0.75 ms off a step in `bench_engram` (not yet
  measured in `generate`).

### Vision (Phase G)

The reference (`image_processor.py`, `vision.py`) decodes with Pillow, pads to a
14-pixel grid with bicubic resampling, runs a 32-block ViT (dim 1024, 2-D RoPE) and a
3×3 aligner into LLM rows, and fills an image span `[START] (IMAGE×w NEWLINE)×h [END]`
whose positions all carry `<｜deepseek_image｜>`.

- **Decoding, our own:** `crates/oaiy-image` (std-only, no `unsafe`).
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
    - OAIY, own pipeline end to end: *"The image displays a minimalist, line-art icon
      of a stylized robot with a smiling face, antenna, and boxy…"*
- **Serving:** `image_url` parts (base64 `data:` URLs, or local paths on a loopback
  server) are decoded and sized on the request thread. The worker encodes only the
  images the uncached part of the prompt reaches, and the prefix cache keys image
  positions by content.
  - A chart (254 image tokens) read back exactly: title, four values, bar colours.
  - A follow-up turn reused 363 cached tokens, image included (prompt 6.8 s).
  - Image tokens run token by token like other short stretches: 2.7 tok/s cold (the
    vision-biased experts are not cached yet), 8.5 tok/s warm.

## The CPU model, measured for a port off CUDA (2026-10-05)

`dsv41::model` (the reference the CUDA path is tested against) runs the whole checkpoint
on the CPU, its experts through the same LFRU host cache. Measured with
`crates/oaiy-llm-server/tests/dsv41_cpu_speed.rs` (`measure_where_prefill_and_warm_decode_time_goes`,
96 GB expert cache, 2,000 tokens of this repository's docs), split by the trace hook's
timestamps:

| | Time | Attention (hc mix, norm, the FP8 dense trunk's projections, attention) | MoE | Read from the drive |
|---|---:|---:|---:|---|
| Prompt of 2,000 tokens | 1,294 s | 784 s | 502 s | 226.6 GB, every expert a miss |
| Warm decode (the last 16 of 32 steps) | 2.06 s a token | 1.0 s, flat | 0.76-0.9 s | 0.45-0.66 GB a token, 85-90% of the experts from RAM |

So a prompt is compute-bound on the CPU: attention costs more than the MoE, and the
reads (226.6 GB, about a minute at this drive's rate) are a fraction of either. In
decode the attention is a second of compute that a GPU holding the 9.9 GB trunk would
all but remove, and the MoE is mostly its misses. A WebGPU port therefore puts the FP8
trunk on the GPU first, then the MXFP4 experts (a VRAM tier, and the prompt's batched
matmuls). A portable serving engine on the CPU model alone would make an Agent wait
21 minutes for its first answer.

What the WebGPU build does about it (docs/WEBGPU.md, "DeepSeek-V4.1"):
the trunk on the WebGPU adapter took the prompt to 1,081 s and showed the attention's
remaining 600 s to be its serial sparse attention and index scores, which spread over
the threads took it to 558 s; a prompt's busy experts on the GPU and the records read
eight at a time took it to 490 s. Profiled (`dsv41::profile`), that was 165 s of reads,
171 s of experts on the GPU (most of it uploading) and 36 s on the CPU, one after
another: the MoE overlapping its reads with its matmuls, the records uploaded as stored
into slots made once, a VRAM tier of the experts used most and the mixing spread over
the threads took it to 261 s, the drive's now (the checkpoint here is on a USB SSD,
1.4 GB/s). Warm decode went from 2.06 to 1.02 s a token, about 110 of a step's 240
experts computed on the GPU.

## EXL3 checkpoints (2026-10-24)

A DeepSeek-V4.1 checkpoint quantized to EXL3 (exllamav3's mul1 trellis, about 3 bits a
weight; a few layers' experts at 2) loads in the same engine. Each projection is its
`trellis`, `suh`, `svh` and `mul1` (`linear::Weight::Exl3`; `wo_a` stored a slice a
group, `Weight::Stacked`). The router is f16, which loads as bf16 where every value is one.
The checkpoint may share the original's Engram shards: a tensor two shards both hold is
the one `model.safetensors.index.json` names (`StIndex::open`). An expert record is its
three projections' parts in the order the checkpoint keeps them, about 13.3 MB and one
read from the drive.

On WebGPU (`engines::dsv41_portable::WgpuExl3Experts`), every routed expert is computed
on the cards:
- **Held experts.** Slots of each bitrate on every card the model may use (about 3,900 on
  two 32 GB cards beside the trunk) hold the most used experts. A usage profile fills them
  while idle.
- **Scratch.** Every other expert goes through scratch slots on each card. The host's
  threads write its record into the card's memory in place, through the BAR
  (`Exl3Slots::put_many`). The cards share these experts by the rate each has shown.
- **Promotion.** An expert used more than the least used one a card holds takes its slot
  by a copy of the card's own memory (`Exl3Slots::copy_slot`), and RAM lets it go.
- **Determinism.** A prompt's experts all go to the GPU. A job's sums are made in one
  order whatever shares its call, so the same prompt gives the same reply.

The CPU decodes EXL3 experts only without a GPU (`dsv41::exl3_experts`).

Measured on two RTX 5090s (one on four PCIe lanes, one on eight), 100 GB of RAM tier,
120-token replies:
- CPU experts: 0.64 tok/s.
- Cards that hold what the conversation uses: 5.5-5.8 tok/s.
- A new subject: 3.7-4.4 tok/s, the cards taking it in as it goes.

What is left is the trunk's projections: some 300 round trips a token, 85 ms of a step.
Greedy answers match the FP8 checkpoint's on factual prompts word for word.

## DSpark drafting (2026-10-24)

The checkpoint's `mtp.0..2` are DSpark's block drafter, run as DeepInfer runs it
(`DFlash`; `dsv41::dspark`):
- **What it reads.** For every row the trunk reads, the mean of the four streams entering
  layers 37-39, laid end to end, goes through `main_proj` and `main_norm`. Each drafter
  layer keeps its keys of these rows in a 128-position window. An EXL3 `main_proj` sums in
  f16, which the first rows' streams overflow (their values run to thousands), so each row
  is first scaled by a power of two; the norm after makes the scale nothing.
- **A draft.** One pass of five rows: the token the trunk chose, then four noise tokens, at
  the positions after it. They go through the drafter's three blocks. Its attention is
  window-only, and each row attends to the window and to every row of the block. Its
  experts are 128 a layer, 3 a token. Each row then gets the trunk's head, plus a rank-256
  correction by the token before it (the chosen token, then each row's draft). A
  confidence, a sigmoid of the row's state and that token's embedding, says how likely the
  trunk is to take each draft.
- **A check.** One trunk pass of the token and the drafts it checks
  (`Model::forward_rows`, every row's logits). The drafts stand up to the first one the
  trunk chose otherwise; that choice is the next token. The caches are rewound past the
  rows not kept (`Model::save`/`rewind`): their window slots are put back, and a
  compressor's partial group is made again from its last rows (`AttnState::rewind`).
- **How many drafts.** A pass checks as many as make the most tokens for the time. The
  expected tokens are the product of the confidences up to each draft. The cost is a step
  plus three fifths of one for each row past the first. This uses the drafter's
  confidences alone, so a greedy reply is the same whenever it is asked. After passes in a
  row that are unsure of every draft, drafting pauses for a few passes.
- **Experts.** An EXL3 checkpoint's 384 drafter experts are held on the cards in slots of
  their own (5 GB). A check of up to eight rows sends its experts a decode step's way: held
  experts are begun first, and read ones are kept as a step's are.
- **When it runs.** Where `OAIY_DSV41_DRAFT=1` asks for it. It was on for an EXL3
  checkpoint until a step's calls were cut (see "Fewer calls a token"); since then it no
  longer pays there.

A pass's rows are summed apart from a step's, so a drafted reply can differ from a
step-by-step one where two tokens nearly tie (the same prompt gives the same reply either
way).

Measured on the EXL3 checkpoint, same build, same usage profile, 120-token replies,
against plain decoding:

| Reply | Plain | Drafted | Tokens per pass |
|---|---|---|---|
| A familiar subject | 6.2 tok/s | 7.15 tok/s | 1.8 |
| The same again | 5.8 tok/s | 6.4 tok/s | 1.8 |
| Code | 4.9 tok/s | 7.1 tok/s | 3.5 |
| A new subject | 4.1 tok/s | 4.0 tok/s | 1.6 |

- A draft costs about 25 ms.
- A check costs 140-270 ms for one row. Each row more costs another 80-130 ms, mostly the
  uploads of its experts that no card holds.
- The fp8 checkpoint's checks send their experts to the CPU's workers. A two-row check
  costs four steps there, so drafting halves its speed, and it is off.

## Fewer calls a token (2026-10-24, night)

An EXL3 checkpoint's step made some 245 calls to its card a token, each a round trip.
It now makes about 165. Two pairs of a layer's calls are joined, each of two stages on the
card, to the bit of the calls apart:
- **The queries with the projections of x** (`exl3::chained_normed`,
  `linear::normed_together`). The queries' low-rank row is normed on the card as the host
  norms it. The squares are added in the host's order. The mean, the square root and the
  reciprocal are made correctly rounded by integer steps: the card's own `1 / sqrt`
  differed from the host's in a quarter of values.
- **The output's two projections** (`exl3::chained_after`, `linear::chained_together`).
  The grouped `wo_a` rows are rounded to bf16 on the card, then `wo_b` takes them.

A step's and a check's rows (up to eight) go these ways; a prompt's go as before. Inputs
several projections share are uploaded once. Joining the shared expert's down projection
to the router's call too was tried: it was no faster, since that call runs beside the
routed experts, and its SwiGLU on the card was not the host's to the bit.

The held experts cost about 1 ms a layer, where the same call alone took 0.3. Now:
- **No thread.** They are begun on their cards with no thread started
  (`Exl3Slots::begin_many`, a recording's `launch` and `wait`). Before, one was started for
  them and one more for each card past the first, every layer.
- **Kept vectors.** A step's jobs go through each pool's own kept vectors, with their bind
  groups kept (`StepScratch`). Bind groups kept of the recording's pooled scratch filled the
  device's store with new ones every call, the trunk's own among those it let go.
- **The lock.** A recording no longer holds the device's lock while it waits for its
  results, only while it submits.

An expert's few rows in a check were tried as jobs of a row each, a step's kernel, so each
row's sums would be a step's. A prompt's experts of two to eight rows went that way too,
which changed its replies and made them slower; with drafting off it is left out.

Now about 0.4 ms a layer.

Measured on the EXL3 checkpoint, same usage profile, 120-token greedy replies, plain
decoding; every reply the same text as before, to the word:

| Reply | Before | Now |
|---|---|---|
| A familiar subject | 6.9 tok/s | 8.8-9.1 tok/s |
| Code | 5.6 tok/s | 6.6-6.7 tok/s |
| The same again | 6.7 tok/s | 7.8-8.7 tok/s |
| A new subject | 4.4 tok/s | 4.8-5.2 tok/s |

A step of a familiar subject is now about 110 ms: the trunk's calls some 50 ms, the held
experts 16, the uploads of the experts no card holds 19 (overlapping them), drive reads 15,
and the CPU's attention, mixing and Engram some 15.

DSpark's drafting no longer pays: a check's row past the first still costs about three
quarters of a step, most of it the uploads of the experts that row adds. Drafted replies:
a familiar subject 8.4-8.8 tok/s, code 6.8, a new subject 4.4. So it is off unless asked
for, and its 5 GB on the cards holds experts instead. A check's rows are also not yet the
same bits as steps (the shared expert's and the BF16 matrices' few-row kernels), so a
drafted reply can still differ from a plain one where two tokens nearly tie.

## A layer in two calls (2026-10-25)

A decode step of an EXL3 checkpoint now makes each layer in two calls to its card:
- **The queries' call:** the projections of x and the queries, with the hyper-connection's
  mixing projection of the stream.
- **The router's call** (`exl3::run_program`, a program of steps). It holds:
  - the attention's output projection, with `wo_a`'s eight groups in one pass of the group kernels;
  - the hyper-connection's spread of the output, then its collapse into the FFN's input, then its
    norm (the host's to the bit);
  - the router and the FFN's mixing projection;
  - the shared expert whole. Its SwiGLU uses the host's own table of every bf16 gate, so its
    output is the host's to the bit; its down projection was a call of its own before;
  - behind it, the routed experts the router likely picks that the first card holds. A small kernel
    ranks the router's scores as the card makes them, takes the best eight, and looks each up in the
    layer's slot table. The held ones run, and the jobs of those it does not hold are skipped.

The held experts are a second recording, submitted right after the first, so the card goes
straight on to them. The host reads the first part, routes as before, and makes the experts no card
holds on its cores meanwhile. It then takes the second part's outputs for those it routed to. They
are the slots' own to the bit, so the replies are unchanged. Made in one recording, the held
experts' time came before the routing and the CPU's experts could no longer run beside it: 3-5%
slower than the call of their own.

A layer's experts go first to one pool of the first card that the router's call reads (pools
shared by a few layers). The rest go where there is room, as before, and are made in a call of
their own. When a slot changes hands, the slot table is written in the same submission as the
copy, so a call never sees an expert's table entry without its weights.

The mixing projections on the card sum in its own order. A reply's text therefore differs from the
build before them from the first token where the sums did, and is the same from then on.

Other findings:
- **The trunk split over the two cards** (`OAIY_DSV41_SPLIT=1`, each layer's experts beside its
  weights) was no faster. Each card then did half the work and ran in its P3 power state, its
  memory at half its clock. With the trunk on one card, that card stays in P0. The second card,
  holding experts alone, spends a reply in P5, its memory at a seventeenth of its clock. The
  driver's power setting (Prefer maximum performance) would change this; not tried.
- **The CPU's experts.** They run on three quarters of the cores (`OAIY_STEP_THREADS`). With
  every core, the step's other threads waited for them: 12.5 and 13.5 tok/s where with twelve of
  sixteen cores it was 14.2 and 14.9.
- **A round trip.** A call costs some 70 us to encode and 30 to submit. From submit to its read
  back is about 345 us, of which the card works about 150. The same call alone, back to back, is
  0.195 ms, and 0.23 to 0.30 after 2 ms idle.
- **The CPU's experts beside the cards'** (half of a layer's misses sent up to the cards' scratch
  slots while the cores make the rest) was a little slower. Later shared by measured time instead,
  see the next section.

Measured on the EXL3 checkpoint, same usage profile, 120-token greedy replies. Each run asks the four
twice in one session; these are the second round's. Every reply is the same text in each run:

| Reply | Held experts in a call of their own | Trunk split, held experts fused | Held experts behind the router's call |
|---|---|---|---|
| A familiar subject | 15.3 tok/s | 14.7 tok/s | 16.1 tok/s |
| Code | 10.8 tok/s | 10.5 tok/s | 11.0 tok/s |
| The same again | 14.7 tok/s | 14.5 tok/s | 16.1 tok/s |
| A new subject | 10.7 tok/s | 10.7 tok/s | 11.6 tok/s |

The commit before (every core's thread an expert's worker, the held experts in a call of their own)
gave 12.5, 8.9, 13.5 and 7.7 in a first round. The replies vary by
some 10% from run to run when another program compiles on the machine; the bench waits for a quiet
machine before each request.

## Attention on the card (2026-10-25)

A decode step of an EXL3 checkpoint whose trunk is on one card now makes each layer in one call:
the attention is made on that card, inside the router's program, where the host made it between the
queries' call and the router's. `OAIY_DSV41_GPU_ATTENTION=0` makes it on the host as before.

The card keeps its own copy of every layer's caches (`attention::DeviceCaches`, 67 MB at an 8,192
context): each window, each compressing source's compressed rows, partial group and index keys, the
index sources' picks, and the rotary tables. A prompt's chunks still attend on the host, so the
caches go over to whichever side is behind: up to the card before a decode step, back to the host
before a prompt continues a reply's state or a checkpoint is taken. A step reads its numbers
(position, window slot, compressed count) from a vector of words the program uploads each run, so
the recording is the same at every step and replayed.

What the step makes there (`exl3::ProgramStep`'s `Rope` to `HcMix`, kernels in `exl3/attend.rs`):
- the queries rotated; the window's key normed, rotated, quantized to fp8 and put in its slot;
- a compressing source's latent (pooled past a ratio of 1), its index key and its compressed row;
- an index source's scores of the compressed positions and its 512 picks (a radix select on the
  scores' bf16 bits, ties to the lower position, the picks in order);
- the heads' attention over the window and the picks, in three kernels: each head's positions in ten
  workgroups of 64 (scores, then each chunk's softmax and sums), then the chunks together;
- the attention's mixing coefficients, from the stream's sum of squares the host makes.

Each of them rounds as the host does, to the bit (the rotary products stored to workgroup memory
before their sums, so no fused multiply-add takes them; quotients and square roots correctly
rounded by integer steps), but for the exponentials of the softmax, a ratio-2 pool and the mixing,
which are the card's own, and the attention's sums, added a chunk of 64 positions at a time. So a
reply is no longer the host's text, but the same text every time. The candidate blocks keep every
position up to 2,048 blocks of 8 (16,384 positions), so the card has none to pick; a step past that
attends on the host.

Checked on the real checkpoint with `OAIY_DSV41_GPU_ATTENTION=check` (the host attends as before and
the card beside it, from the host's caches, each step's differences said), at a 4,152-token prompt
where the picks are real (more compressed positions than 512): over 48 steps of 40 layers, every
window row, compressed row and index key the host's to the bit, every pick set the host's, the heads'
outputs at most 0.36% off their largest value (bf16's own step is 0.39%), the mixing 2.4e-7. On the
tests' data 99.8% of the heads' outputs are the host's to the bit.

The first kernels were each a thread's long chain of reads: the mixing's sum of squares 400 us a
call, the attention 160, the index scores 250, and the replies slower than the host's attention.
Each thread's reads made eight at a time before their adds (the adds in the host's order), a thread
a key and head for the index scores, the sum of squares from the host and the heads' positions over
ten workgroups: 28, some 70 over the three kernels, and under 115.

Measured as the sections above, one session, the second round:

| Reply | Attention on the host | On the card |
|---|---|---|
| A familiar subject | 14.1 tok/s | 15.8 tok/s |
| Code | 8.8 tok/s | 10.4 tok/s |
| The same again | 13.4 tok/s | 15.2 tok/s |
| A new subject | 9.0 tok/s | 10.7 tok/s |

Another program compiled on the machine through these runs (the host's figures were 16.1, 11.0,
16.1 and 11.6 on a quiet one). At a 4,153-token prompt the host's attention was some 65 ms a token
(its sparse attention 36, the index scores and the indexer 30): a 60-token reply after it came at
8.0 tok/s with the attention on the card and 4.6 on the host. Its kernels are made while the model
loads (`exl3::make_attention_kernels`): made as the first reply needed them, they were some seconds
of it.

A layer's experts no card holds are now shared between the host's cores and the cards' scratch slots
by what each has taken of late (`WgpuExl3Experts::share_to_cards`: an expert on the cores, a record's
put at the fastest card's rate and the call after it): as many go to the cards as end no later than
the rest on the cores, one at least staying there. On idle cores an expert takes some half a
millisecond and a card's put and call two, so the cores keep a layer's few misses (the replies were
as fast either way). With the cores taken (four of them for the experts, `OAIY_STEP_THREADS=4`), code
went from 5.7 to 6.6 tok/s and a new subject from 4.5 to 5.8, every reply the same text.
`OAIY_DSV41_HYBRID=0` keeps them all on the cores.

Keeping the RAM tier resident (`--pin-ram`, `OAIY_PIN_RAM=1`: a hard minimum working set) measured
no gain. The figures its commit gives compared two pinned runs (the bench kept the first run's
setting for the second); measured again, unpinned then pinned in one session, the server's working
set fell to 115 GiB unpinned and 102 pinned (Windows did not hold the minimum), the replies 16.7,
10.9, 15.7, 11.6 tok/s unpinned and 15.7, 11.0, 15.2, 10.3 pinned. It stays off unless asked for.

## GGUF checkpoints (2026-10-25)

A GGUF of llama.cpp's `deepseek41` layout loads as the checkpoint does (`dsv41::gguf_file`): its
shards, and the Engram tables' own GGUF (`deepseek41-engram`) beside them, served under the
checkpoint's names. The tokenizer, the configuration and the Engram's n-gram hashing (its token map,
primes, offsets and multipliers) come from the GGUF's own metadata and tensors, so no
`tokenizer.json` or `engram_meta.safetensors` is needed. Checked against a checkpoint of the same
model (`a_ggufs_tokenizer_is_the_checkpoints`, given both): every token id and its bytes, the
configuration, and the hashing tables the same; its float tensors the checkpoint's to the bit.

The dense matrices go on the card through the quantized matmul (`QuantLinear`: Q8_0, Q5_K, Q6_K,
and the float ones as f16), in the same device programs as an EXL3 checkpoint's trunk, the
attention with them (its grouped output's groups each a matrix of its own there: a GGUF has no
group kernel). The routed experts are records of each layer's three tensors' blocks (a layer's types
its own: 1-bit beside 2-bit grid types), read into the RAM tier as any checkpoint's.

On the CPU alone (`dsv41::gguf_experts`: each row decoded by `ggml_quants` and multiplied in f32) the
four asks answered sensibly at 1.5 tok/s, the experts some 500 ms of each token. The trunk's matmuls
looked slow too, 125 ms a token where each takes 7 to 17 us alone (`measure_deepseek_gguf_matmuls`):
the card sat in P5 and P8 (its memory at 405 to 810 MHz) while the host made the experts.

So the experts are held on the cards as an EXL3 checkpoint's are (`WgpuExl3Experts`, its slots now
either kind): `ggml_rs_wgpu::quant::moe::QuantSlots`, a pool for each pair of types the layers have
(a 2-bit file has eight: its gate's and up's type, its down's), each slot an expert's blocks as the
quantized MoE kernels read them (IQ2_XS added to them), the misses through scratch slots, and between
the two passes the model's own SwiGLU (the sums rounded to bf16, clamped, `silu(gate) * up` rounded).
The misses go through the cards alone, never the host's cores: the host's decoding sums otherwise,
and which side made an expert would show in the reply. With a 2-bit file's experts some 7.5 MB, the
cards hold 5,715 and RAM the other 9,645: every expert resident, none read from the drive. No
drafter runs with it.

Measured as the sections above (the EXL3 checkpoint's usage profile seeding what the cards take while
idle, the same model's routing), the second round:

| Reply | 2-bit GGUF | EXL3 2.9 bpw, the same session |
|---|---|---|
| A familiar subject | 19.5 tok/s | 17.0 tok/s |
| Code | 16.8 tok/s | 11.7 tok/s |
| The same again | 20.9 tok/s | 16.2 tok/s |
| A new subject | 16.2 tok/s | 11.9 tok/s |

Every reply the same text in both rounds; the EXL3 checkpoint's the same as before.

The router's program then makes a GGUF's held experts too, as it makes an EXL3 checkpoint's
(`exl3::ProgramStep::Held` of either kind of slots, `HeldSlots`; `QuantSlots::record_fused`, the
few-row kernels skipping a candidate the pool does not hold): the same bits as their own call after
the host's routing, a round trip sooner. Two runs each, alternated, the second rounds:

| Reply | Held experts in their own call | In the router's program |
|---|---|---|
| A familiar subject | 20.2, 21.4 tok/s | 22.6, 21.7 tok/s |
| Code | 16.4, 16.6 tok/s | 17.7, 17.6 tok/s |
| The same again | 21.1, 19.9 tok/s | 22.9, 23.0 tok/s |
| A new subject | 16.0, 16.1 tok/s | 16.7, 17.0 tok/s |

Every reply the same text in all four runs (and as above).

A scratch pool's slots remember what they hold (`Scratch`: each slot's expert and when it was last
used): an expert a pass finds there still, put by a recent pass, is made from its slot as it is, the
rest put into the slots used longest ago, and a pass's misses go first to a card whose slots hold them.
A new subject's reply found a sixth to a quarter of its misses there; the uploads they skipped made
its replies some 5% faster (second rounds 16.7 and 16.5 tok/s, 15.9 and 15.8 with
`OAIY_DSV41_SCRATCH_REUSE=0`), the other asks within their noise; the same text either way. The
profile says each card's puts and their rate: the first card's, on four lanes, 7 to 8 GB/s; the
second's, on eight, 14 to 15.

A GGUF's grouped output (`wo_a`, eight Q5_K slices of 1024 by 4096) is kept on the card as one group
too (`ggml_rs_wgpu::quant::moe::QuantGroup`, Q5_K added to the experts' kernels), its eight parts one
pass of the few-row kernel in the attention's program, where each part was its heads' copy, the
copy's quantization to int8, its matmul and its rows' copy: some 30 dispatches a layer fewer. Its
sums are f32's where the parts' were from int8 activations, so the replies are other texts (the same
every time). Second rounds, one pair on a quiet machine:

| Reply | Each part its own matmul | One group |
|---|---|---|
| A familiar subject | 22.2 tok/s | 26.1 tok/s |
| Code | 17.4 tok/s | 19.0 tok/s |
| The same again | 21.7 tok/s | 26.7 tok/s |
| A new subject | 17.1 tok/s | 18.9 tok/s |

(`OAIY_DSV41_GGUF_GROUP=0`: each part its own matmul.)

A GGUF trunk's programs are replayed now, as an EXL3 checkpoint's are: the int8 rows its quantized
matmuls read were the recording pool's scratch, a new buffer each run, and a recording that takes
scratch is none a run can replay, so every program was recorded again at every step (some 165 us of
the host's work each, 140 now). What is left of it is the encoding: a replayed part of some 20
dispatches takes 90 us to encode and submit, some 4.5 us a dispatch (a hundred copies of three
bindings each, 0.6), wgpu tracking every one of a dispatch's nine bound buffers; some 118
recordings a token, 9 ms of it.

A pass's misses are shared between the cards by each card's best rate of late (the best seen, let go
of slowly), where it was by their rates of late: the second card, waking from a low power state, put
a few records in at a sixth of its rate, its rate of late fell below the first card's, every miss
went to the first after that (895 puts there to 135, 869 to 70) and none raised the second's rate
again; those two replies came at 24.0 and 24.6 tok/s where 27.5 to 29.3 by the best rates
(`OAIY_DSV41_SPLIT_BY_PEAK=0`: as before). Second rounds, two runs: 27.5 and 27.6, 20.1 and 19.8,
29.3 and 28.4, 20.4 and 19.5 tok/s.

Tried and left out: a thread taking two or four output rows of a step's expert matmul (x loaded once
for them) was no faster; a mapped write polling for its buffer rather than waiting for the device's
work whole measured the same.

A prompt's attention runs on the card too now, as a step's does: its queries' projection and rotation
(`ProgramStep::RopeRows`, each row its own position), the attention over each token's chosen rows
(`ProgramStep::AttendRows`: eight heads a workgroup, each token's list of rows, its sums in the
host's order, the exponentials the device's) and the output's grouped projection, in programs of 512
rows run once (`run_program_once`, an arena of their own). A 4,153-token prompt of the 2-bit GGUF
read in 92.6 s where 168 to 175 (some 45 tok/s where 25). Checked against the host
(a_prompts_attention_on_the_device_is_the_hosts_but_for_its_exponentials): the queries' rotation the
same bits, 99.94% of the outputs the same bits, the worst 0.4% off. The replies are other texts for
it, the same every time; the conversation test's three turns and its "again" as before.
(`OAIY_DSV41_PROMPT_ATTENTION=0`: the prompt's attention on the host.) What is left of that prompt's
time: the experts 27.6 s, the held experts 10.8, their uploads 9.8, the dense matmuls 22.4, the
host's mixing 12.7, the indexer 8.9 and its scores 6.4.

## An agent's long prompts (2026-10-26)

OAIY's Agent sends its instructions and tools before a word of the request: some 19,000 to 21,000
tokens with the media tools of a computer that has many media models (a run here: 21,310 tokens). The
2-bit GGUF read them at some 32 tok/s, 664 s before the first word, and its reply came at 2.8 tok/s.

The state a prompt's system turn leaves is kept now (`Engine`'s `system`, the first prompt of a
conversation whose system turn has 256 tokens or more: its state at the first user turn's token).
A new conversation that starts with the same turn, as each of the Agent's runs does, reads on from
there. Before, the one state kept was the last prompt's, which a new conversation does not start
with, so every run read the whole turn again. A conversation's later prompts keep none: a run that
reads a guide into its instructions would leave those in place of the ones its next run starts with.
Measured on the 2-bit GGUF with the Agent's own 33 tools and its instructions grown to 22,692 tokens:
a first request's first word after 663 s; a new conversation with the same system turn 22,671 of its
tokens reused, its first word after 3.1 s; the first's conversation one step on (the last prompt's
state) after 4.3 s. A state is every layer's caches whole, some 1 GB at a 65,536-token context, so
this one more is kept beside the other.

What that prompt's 662 s went to (the profile, parts overlapping): the dense matrices on the card
167 s, the experts on the cards 155 s, the indexer on the host 131 s and its scores 114 s (both as
the square of the prompt's length), the mixing 69 s, the experts' uploads 53 s and the held experts'
45 s. Its reply's 376 ms a token: the attention on the host past the card's reach (16,384 positions),
110 ms (the sparse attention 61, the index scores 24, the indexer 25), and the rest slower for it too,
each layer in two calls and three round trips where one call within the reach.

That reach was the candidate blocks': layer 20 (an index source of ratio 1) keeps the 2,048 best
blocks of 8 positions, by each block's best score and the newest block always, and the index sources
after it pick only within them; up to 16,384 positions every block is kept, so the card had none to
pick. It picks them now (`exl3::ProgramStep::BlockKeep`: each block's best, a `TopK` of the blocks,
their flags in a vector the step's later layers read; `ProgramStep::MaskBlocks`: a later source's
scores outside them made -infinity before its own `TopK`), and a decode step attends on the card at
any position of the context (`OAIY_DSV41_FAR_ON_CARD=0`: up to 16,384, as before). The caches there
are the context's: 464 MB at 65,536 where 149 at 16,384, the experts' slots as many as before.
Checked against the host (`the_candidate_blocks_on_the_device_are_the_hosts`: the flags, a later
source's masked scores and its picks to the bit at counts from 100 to 65,536, scores of few values
so blocks tie) and on the real checkpoint (`OAIY_DSV41_GPU_ATTENTION=check` over 235 steps at
17,244 to 17,325 positions: every index source's picks the host's, the window rows, compressed rows
and index keys to the bit, the heads' outputs at most 0.54% off). The 2-bit GGUF at 22,692 tokens:

| Reply | Past the reach on the host | On the card |
|---|---|---|
| A first request | 2.65 tok/s | 11.8 tok/s |
| A new conversation, the same system turn | 2.87 tok/s | 14.7 tok/s |
| The first's next step | 2.86 tok/s | 19.4 tok/s |

A system turn's state is kept on disk too (`--prompt-cache`, OAIY's `engines\cache\prompt-states`;
`engines::dsv41_portable::saved::SavedState` in the prompt cache every engine keeps, known by the
checkpoint's files and the context), so a later process reads on from it: OAIY started again, or the
model loaded again after another. Each vector is written up to its last value that is not a fresh
state's, in bf16 where every value written is one, and read back into a vector of its whole length:
the Agent's turn of 22,671 tokens is 75 MB. A first process read it in 661 s and kept it; a second,
started afresh, read it back and gave its first word after 12.8 s, the same reply as the first
process's own new conversation (its RAM tier not yet filled, the reply itself at 2.3 tok/s). Not
where a drafter runs (its keys are not kept) or for an incognito request.

A prompt's index scores are made on the card too (`ProgramStep::IndexScoresRows`, a row a workgroup's
`y`, each key's score the host's `index_scores`' to the bit; test
a_prompts_index_scores_on_the_device_are_the_hosts), downloaded, and the host picks from them as
before: in pieces of 512 rows, each the same program whatever the count (the last padded, the layer's
whole cache uploaded, the count a word), so the device keeps its vectors from one to the next. Made
as a new program at each count, the card's memory grew 2 GB through the 22,692-token prompt and ran
out (`OAIY_DSV41_PROMPT_INDEX=0`: the host's). That prompt read in 570.6 s where 648 to 705, its index
scores 18.9 s where 114 (the indexer 35.8 where 131), its replies the same text; the first card's
memory at most 30.8 GB.

## The host's experts, faster (2026-10-26)

An EXL3 checkpoint's cards hold some 3,760 of its 15,360 experts (a 2-bit GGUF's 5,500), so a reply
misses more of them, and the host's cores make most of a layer's misses (`step_experts_host`). One
thread makes an expert's sums at 11.6 billion weights a second, twelve at half that each: a task took
a block of eight tile columns down every tile row, so it read 768 bytes of each row's 14 KB, the
team's eighteen tasks at once. A task now takes a split's tile rows (which lie together in the
record) over a part of the columns, and each block's splits are added in order after, as
`ProjAt::step_sums` adds them; and the inputs' and outputs' roundings to f16 go sixteen at a time
(each value's own conversion was most of an input's time). The same bits (tests:
the_hosts_step_experts_together_are_each_alone, the_hosts_vector_sums_are_its_emulations, both on the
CPU alone, and a_steps_expert_on_the_host_is_the_cards_to_the_bit). An expert 0.49 ms alone where
0.68, two 0.39 each where 0.61, six 0.43 where 0.59. The EXL3 checkpoint, no usage profile, the four
asks' second rounds, two runs each in turn (old, new, new, old):

| Reply | Before | After |
|---|---|---|
| A familiar subject | 17.2, 17.2 tok/s | 19.3, 19.4 tok/s |
| Code | 11.3, 11.2 tok/s | 12.6, 12.6 tok/s |
| The same again | 17.2, 17.2 tok/s | 19.4, 19.3 tok/s |
| A new subject | 14.1, 14.1 tok/s | 16.2, 16.3 tok/s |

Every reply the same text in all four runs. Sending all of a layer's misses to the cards measured
slower (code 9.8 where 11.5).

With the host's experts that much faster, the share of a layer's misses sent to the cards
(`share_to_cards`) cost: priced by a call and each record's put at the fastest card's rate, it sent
the cards misses the host made sooner, and all of them on the host gave code 14.1 tok/s where 13.2.
The cards' share is now priced by what it has taken of late, from its handing over to its outputs
(`cards_us`, an expert's time: its thread, its puts on whichever card took them, its calls, the
cores' reads they slow among it), as the host's is. On a quiet machine that is all of them here, the
same as `OAIY_DSV41_HYBRID=0` (20.0, 14.1, 20.0, 17.0 tok/s where 19.9, 14.0, 19.9, 17.0); with the
host's experts on four threads (`OAIY_STEP_THREADS=4`, cores another program takes) the cards take
some again: 16.6, 10.8, 16.0, 13.6 where all of them here 16.4, 9.7, 16.1, 13.0.

A card past the first holds experts alone (their slots and scratch slots, made at load: no trunk, no
prompt's attention, no display), so its experts take half the default's 4 GiB of room beside them
(`ALONE`, as a tinygrad card's; `WgpuBackend::with_budget`), where no `--webgpu-gb` is given and the
trunk is not split: the EXL3 checkpoint's slots 3,931 where 3,761 (its replies within their noise:
19.7, 14.2, 20.0, 17.3 tok/s), the 2-bit GGUF's 5,856 where 5,608 (its replies not yet measured
against the default's in one session: an agent's at 22,692 tokens came at 15.3, 17.3 and 22.0 tok/s,
where an earlier run's on another build were 11.8, 14.7 and 19.4). Through that prompt the second
card's memory peaked at 30.8 GB of its 32.6, the first's at 30.6. OAIY's studio passes a budget of its
own making (`--webgpu-gb`, the card's memory less `vram_headroom_gb` and 2 GB), so in OAIY this is not
yet so.

The cards' experts were weighed by each one's uses, seeded at load from the usage profile's counts
(the most used 333, a quarter of them 17 or more): an expert another subject used most this run could
take a card's slot only past those counts, so a code reply kept missing the experts it used most and
the cards took few of them in. Seeded no more (the profile still says what fills the cards first, by
its order), as OAIY runs them (its profile, `--webgpu-gb 27`, `--ram-gb 153`), the second rounds:

| Reply | EXL3, seeded | EXL3, not | 2-bit GGUF, seeded | 2-bit GGUF, not |
|---|---|---|---|---|
| A familiar subject | 19.1 tok/s | 19.8 tok/s | 25.8 tok/s | 26.7 tok/s |
| Code | 11.4 tok/s | 13.9 tok/s | 15.1 tok/s | 17.3 tok/s |
| The same again | 19.1 tok/s | 19.8 tok/s | 25.8 tok/s | 26.9 tok/s |
| A new subject | 16.0 tok/s | 16.8 tok/s | 21.3 tok/s | 21.5 tok/s |

A code reply's misses on the first card 16,445 where 12,568 (EXL3), 8,865 where 6,569 (2-bit). Counts
divided by 8 fell between (code 12.9).

Tried and left out, on the EXL3 checkpoint as OAIY runs it: more promotions queued (64 where 16) took
the same 731 into the cards in a code reply (what may take a held expert's place decides, not the
queue); the cards' counts halved every 128 or 32 steps where 512 took 2,241 and 2,758 in and missed
fewer (11,732 and 10,146 where 12,568), and the replies were slower for the copies (code 10.2 and 8.6
tok/s where 13.9, a familiar subject 16.9 and 14.4 where 19.8).

The one-row EXL3 kernel (a step's trunk and held experts) decodes a 3-bit projection at 300 to 450
GB/s, its workgroup a tile column's: two to eight tile columns a workgroup, its tables (the codes'
values, each code's place) made once for them, the same bits, was 3-8% faster at two on the large
projections and slower on the small ones at four and eight; left out. What bounds it is the tile
loop's decode itself.

## Fewer parts, fewer instructions (2026-10-26)

That last line was wrong: with a tile's decode taken out (its table read, or its byte sum), the
one-row kernel took the same time. What it paid for was its parts. A projection's rows were summed
in parts aimed at 4,096 workgroups a matrix (at most 8), each part a workgroup's table, its eight
warps' two or three tiles each and its reduction; and a group of matrices in one call (a layer's
held experts) has as many workgroups again for each. `splits_for` now aims at some 1,024 a matrix,
2 to 8 parts, and every one-row projection (the cards' and the host's alike) is split so. At 3
bits, a call's time (`measure_one_row_splits`):

| Shape | 8 parts | Now |
|---|---|---|
| An expert's gate [2304, 5120], one matrix / four | 9.8 / 27.1 us | 9.7 / 25.0 us (4 parts) |
| An expert's down [5120, 2304], one / four / eight | 10.7 / 34.3 / 67.5 us | 8.8 / 25.1 / 47.1 us (2) |
| [10240, 2560] (Flash-Next) | 19.0 us | 15.8 us (2) |

(At one part a matrix of 1,024 tile columns went past a wave of the card's workgroups, 27.0 us
where 22.6 at two, so no fewer than two.) The sums are added in other parts, so replies' words
differ from before at places (the sky's "because of" is now "due to"), and the states kept on disk
are known by another name (v2), so the first agent run after it reads its instructions again.

The host's experts found each code by two permutes of the tile's words and three shifts; a lane's
eight codes lie at fixed steps after its first, so its stream is now made once a tile from the three
words it lies in, and each code is a shift and a mask of it (with VBMI2's double shifts where the
CPU has them; `OAIY_EXL3_STEP_NO_VBMI2`): a thread's rate 16.8 weights a ns where 12.9, the same
bits (the vector-sums test now at 1 to 4 bits a weight). Its records in the caches, six experts a
call 0.28 ms each where 0.34.

From memory the host's experts are near what this computer's memory gives them. Twelve threads read
42.5 GB/s each its part as it lies, 42.8 in whole 13.8 KB rows in turn and 34.3 in runs of 3 KB
14 KB apart (`measure_host_read_rate`; DDR5 at 3,600 MT/s, two channels: four sticks of 48 GB,
AMD's rate for two a channel), and a task of a split's rows over a part of the columns reads such
runs: six experts a call from 8 GB of records 0.44 ms each, some 30 GB/s (12 records, 160 MB,
measured 0.33: they were partly in the L3 as they came round again). Tried and left out: a task of
some warps of every column (whole rows as they lie, each warp's sums kept for a reduction after):
0.58 ms an expert, its sums through 8 to 14 MB of memory where a task's stayed in its caches; the
next tile row's tiles prefetched a row ahead, and one or two tasks a thread where three: the same.

The EXL3 checkpoint as OAIY runs it (its profile, `--webgpu-gb 27`, `--ram-gb 153`), second
rounds, two runs each in turn:

| Reply | Before | After |
|---|---|---|
| A familiar subject | 19.6, 19.6 tok/s | 19.9, 19.9 tok/s |
| Code | 13.2, 13.3 tok/s | 13.9, 13.8 tok/s |
| The same again | 19.6, 19.8 tok/s | 20.4, 20.3 tok/s |
| A new subject | 16.7, 16.8 tok/s | 16.9, 16.9 tok/s |

The code reply's experts on the host 5.2 s where 6.05 (an expert 0.43 ms where 0.48); its time on
the first card ("dense on the device") the same, 4.9 s. Qwen3.8-Flash-Next (EXL3, 3.05 bits) 69.7
to 79.6 tok/s where 66.0 to 78.4, OrcaSAQ-2-27B 51.5 to 57.4 where 49.6 to 55.0, the same asks.

What the first card's time is made of, a reply's kernels on the GPU (`OAIY_CHAIN_PROFILE`, each
dispatch timed alone, a trivial kernel some 2.6 us so): every token reads the whole trunk, 7.2 GB in
750 matrices, of which `wo_b` and the shared expert's `w2` of layers 10 to 36 are bf16 (84 and 24 MB
a layer, 2.9 GB a token, read at some 1.5 TB/s). The rest of a layer's chain is short kernels of one
thread or few: the norms as the host adds them (one thread, 13 us each, three a layer), the
hyper-connections' mixing (one thread, 28 us), the attention's scores, parts and join (30, 18 and 6
us at a short context), the candidates (15), the f32 projections of the mixing (24 rows, 11 us
each, two a layer): some 180 us a layer as timed, 7 ms a token. These are next: each is the host's
order of adds, so a parallel form needs the host's sums in that order too.

Promoting experts on one thread where four: the same (code 14.0 where 13.7 tok/s, the host's experts
6.0 s where 6.2).

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
  - Prefill: overlap expert uploads with compute on a copy stream (about 7,000 RAM
    experts cross PCIe a pass, GPU 0 on a x2 link); tensor cores for the bf16
    projections too; choose token-by-token vs layered by measured cost, not a fixed
    threshold.
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
