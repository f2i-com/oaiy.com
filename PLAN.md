# nrob — System Plan

**nrob** (**N**VMe · **R**AM · **O**n-GPU · **B**roker) is a Rust inference engine
for models far bigger than VRAM, on consumer hardware. The weights stay on disk in the
files they were published in, **GGUF** or **safetensors**, with no conversion step
and no format of our own. The engine keeps the shared trunk resident, reads each
token's routed experts in place (a few positioned reads per expert), and moves them
through three tiers:

- **NVMe:** every weight, in its original file.
- **RAM:** a bounded LFRU expert cache.
- **GPU:** the trunk plus the hottest experts.

The broker decides what is hot and who computes what. On DeepSeek-V4.1-Flash, the CPU
computes RAM-resident experts when that beats a PCIe upload.

## Products and crates

```
nrob/                       workspace root
  crates/
    nrob/                   core: WeightStore seam, Ecache, thread pool, JSON parser (std-only)
    nrob-cli/               `nrob` binary for GGUF models: run / chat / bench / info / tokenize
    dsv41/                  DeepSeek-V4.1 from safetensors: expert store, CPU reference model,
                            CPU experts (std-only)
    dsv41-cuda/             DeepSeek-V4.1 on CUDA: kernels, VRAM expert cache, hybrid decode,
                            chunk continuation, checkpoints, layer-by-layer prefill
    nrob-server/            OpenAI-compatible HTTP server for DeepSeek-V4.1 (std-only)
    gguf/                   GGUF reader; FileSource streams tensor bytes from the file
    ggml-quants/, ggml-rs/, ggml-rs-cuda/, tokenizer/
                            quant kernels, CPU/CUDA backends, tokenizers
    llama-rs/               GGUF model architectures; MoE expert streaming (expert_stream.rs)
```

The GGUF stack (`gguf` … `llama-rs`) is the repo author's own Rust code, vendored from
their `llm` workspace (see `crates/VENDORED.md`).

## Hard constraints

- **std-only core.** No external crates in `nrob` or `dsv41`. The JSON reader, the
  cache and the thread pool are hand-rolled.
- `#![forbid(unsafe_code)]` in `nrob`, `nrob-cli`, `nrob-server` and `dsv41`. `dsv41-cuda` holds the
  DeepSeek path's `unsafe` (kernel launches, pinned mappings, AVX-512 dispatch), each
  with a SAFETY note. The GGUF stack keeps its own style: `unsafe` for the CUDA backend,
  the mmap and a couple of byte views.
- Library code never prints and never exits; errors are `nrob::Error` (core, dsv41) or
  `llama_rs::LlamaError` (GGUF stack).
- **Weights are read in place.** GGUF and safetensors only; no converters, no
  container format of our own.

## Storage layer

```rust
pub trait WeightStore: Send + Sync {
    fn record_bytes(&self) -> usize;
    fn shape(&self) -> (u32, u32);           // (layers, experts)
    fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()>;
    fn direct_io(&self) -> bool { false }
}
```

Two implementations, one per file format:

- **`llama_rs::expert_stream::GgufExpertStore`:** a GGUF MoE model's experts. One
  record is `gate || up || down`, read with three positioned reads through
  `gguf::FileSource` (`GgufFile::open_streaming`). A layer's cold misses are read as a
  batch on a small thread pool.
- **`dsv41::expert::SafetensorsExpertStore`:** DeepSeek-V4.1's MXFP4 experts. Each
  expert's weights are adjacent in its shard, and so are its scales, so a record is
  two reads. These bypass the page cache.

Both sit behind `nrob::ecache::Ecache`: LFRU with a recency tiebreak (chosen from a
random draw of resident records) and zero-copy hits through `HostLease`. `dsv41-cuda`
also saves a usage profile that warms its RAM and VRAM tiers at start.

## Status

**GGUF (any architecture llama-rs knows):** resident on CPU or CUDA. The Qwen3-MoE and
Mixtral families can also stream their experts from the `.gguf` (`--budget`), with a
VRAM expert cache on CUDA (`--vram-cache`). Streamed output is token-identical to
resident.

**Safetensors (DeepSeek-V4.1-Flash, 510 GB):** hybrid CPU/GPU decode across two RTX
5090s at ~29 tok/s warm; a fresh process's first answer runs at 2–6 tok/s (SSD-bound).
It passes every golden gate.

It is served by `nrob-server`: an OpenAI-compatible API with tools and reasoning, and a
prefix cache that resumes each harness turn. Long prompts run layer by layer at
~22 tok/s. Details are in `docs/DEEPSEEK_V41.md`.

## Phase log

- **Phases 0–4 (2026-08): prototype.** An earlier engine with its own `.nrob`
  container format, a native model, a server and the `xdb` weight store; all retired
  in Phase 8.
- **Phase 5 (2026-08): any GGUF.** Added the author's own Rust GGUF stack, with MoE
  experts streamed through `Ecache` (`llama-rs/src/expert_stream.rs`). On Qwen3-30B-A3B
  it was token-identical at every cache size: 3.64 GiB peak RSS at a 256 MiB cache, and
  hit rate 11%→69% from 256 MiB to 4 GiB.
- **Phase 6 (2026-08): CUDA.** The GGUF stack's CUDA backend was wired in behind
  `--features cuda`. Speedups on an RTX 5090:
  - stories15M 55→455 tok/s.
  - qwen3-0.6b 1.67→79 tok/s.
  - Qwen3-30B-A3B 0.25→16.9 tok/s.

  Then the streaming roadmap (`docs/ROADMAP.md`) landed in llama-rs: host-cache
  leases, direct gate/up views, pinned staging with async uploads, a per-device VRAM
  expert cache, GPU top-k routing, and grouped MoE kernels.
- **Phase 7 (2026-09): DeepSeek-V4.1-Flash from safetensors.** `dsv41` and
  `dsv41-cuda`:
  - An in-place expert store and a CPU reference model validated against the Python
    oracle.
  - 20+ CUDA kernels, VRAM expert caches on two GPUs, and an AVX-512 CPU expert
    kernel.
  - Launch-ahead hybrid decode: 2.6 → 29 tok/s warm.
- **Phase 8 (2026-09-19): GGUF and safetensors only.** Removed everything format-
  specific that no longer earned its keep:
  - The `.nrob` container and native engine, `xdb` and `nrob-server`.
  - The core that remained (expert cache, JSON parser, thread pool, error type) and
    the CLI's option handling were rewritten from scratch, and the version restarted
    at 0.7.0.
  - GGUF streaming reads experts straight from the `.gguf` (`gguf::FileSource`,
    `Model::open_streaming`) instead of from an xdb copy.
  - `nrob-cli` is now a GGUF client of llama-rs.
- **Phase 9 (2026-09-19): serving DeepSeek-V4.1.**
  - Rust tokenizer and chat format, exact against the reference on 3,638 and 420
    golden cases.
  - Chunk continuation and checkpoints, bit-exact to one prefill.
  - Layer-by-layer prefill: each expert is read once per long stretch, ~6× the
    chunked rate on this SSD.
  - Exclusive VRAM/RAM tiers.
  - `nrob-server`, new code, with a prefix cache for coding harnesses.
- **Phase 10 (2026-09-19): DeepSeek-V4.1 vision.**
  - `nrob-image`: our own PNG (with inflate) and JPEG decoders and Pillow's resize,
    pixel-exact to Pillow on 111 test images.
  - The reference's preprocessing, bit-exact; the ViT and aligner on CPU (the oracle)
    and GPU (60–160 ms an image).
  - Image spans in the model (vision routing bias, Engram masking, any position,
    chunked and layered), and images in `nrob-server`.

## Testing strategy

No downloads are needed for `cargo test --workspace`:

1. Synthetic GGUF models built in Rust: tiny llama and Qwen3-MoE files written by
   `gguf::reader::write_to_vec`. They run end to end through the CLI, and streamed
   experts must give the same tokens as resident ones.
2. Known-answer tests for the cache, the pool, the JSON reader and the quant kernels.
   Every DeepSeek GPU kernel is tested against its CPU function.
3. Gated real-model tests: the Qwen3-30B-A3B streamed-vs-resident money test, and the
   DeepSeek-V4.1 golden gates (`#[ignore]`d; they need the checkpoint and the oracle's
   golden files).

## Next

- **DeepSeek-V4.1:**
  - Vision: GIF/WebP decoding; a tensor-core GEMM for the ViT (it is SIMT now,
    fine at one image a request); send large images (past ~500 tokens) through the
    layered prefill when the caches are cold (token by token runs 2.7 tok/s cold,
    8.5 warm).
  - A ≥2K-token oracle golden, to verify long contexts.
  - Striping the SSD tier over both drives (the drive throttles to 0.5 GB/s).
  - CUDA graphs for the decode step (launch overhead is ~10 ms of a ~35 ms token).
  - VRAM admission weighted by prompt tokens: prefill no longer admits (it churned),
    and decode right after a repeated prompt lost 6-23%; admit experts many prompt
    tokens used, when they beat the victim.
  - Prefill: bf16 tensor-core GEMMs (expert compute is ~30 s of a 122 s pass at
    ~3 TFLOPS), upload/compute overlap, and a cost-based choice between token-by-token
    and layered prefill (the fixed 512-token threshold is a measured average; the
    right one depends on how warm the caches are).
- **GGUF:**
  - Expert streaming for more MoE architectures (Gemma 4 MoE, Qwen3.5-MoE).
  - Hybrid CPU experts on the GGUF path.
  - Direct I/O for streamed reads.
- **Serving:** GGUF models in `nrob-server` too; the Anthropic messages API.

## Non-goals (for now)

Metal and other GPU backends, a weight format of our own, model quantization tooling,
and a crates.io release.
