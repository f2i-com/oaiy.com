# The GGUF stack

The crates `gguf`, `ggml-quants`, `ggml-rs`, `ggml-rs-cuda`, `tokenizer`, and
`llama-rs` in this directory are **vendored** from the author's own pure-Rust
llama.cpp-equivalent project:

- Source: the author's sibling `llm` workspace (copied with the owner's permission)
- Date: 2026-08-02
- Provenance: the author's own code; upstream license `MIT OR Apache-2.0`.

They are OAIY's GGUF engine: the reader, the quant kernels, the CPU and CUDA
backends, the tokenizers, and the per-architecture models. They keep their original
external dependencies and upstream style; the std-only rule for `oaiy-engine` / `dsv41`
does **not** apply to them (see `CONVENTIONS.md`).

`ggml-rs-cuda` is a workspace member but is kept out of `default-members`: building it
needs a CUDA toolchain (cudarc dynamic-linking), so the default build/test stays
CPU-only. It is pulled in by the `cuda` feature of `llama-rs` (and `oaiy-llm-cli`, which
forwards to it: `cargo build -p oaiy-llm-cli --features cuda`, then `OAIY run MODEL.gguf
"prompt" --cuda`). Its tests run under `cargo test --workspace` on a CUDA machine and
skip when no GPU is reachable.

Local modifications should be minimal and marked with a comment
(`// VENDORED-LOCAL: ...`) so future re-vendoring diffs stay readable.

## Local modifications

**Streaming experts from the `.gguf` (2026-08-03, reworked 2026-09-19).**

- `gguf` has a pluggable tensor-byte source, `TensorBytes` (`gguf/src/source.rs`).
  `FileSource` serves tensor bodies with positioned reads straight from the file.
  `GgufFile::open_streaming` parses the header into memory and attaches a
  `FileSource`, so nothing else is read until asked for.
- `llama-rs/src/expert_stream.rs` keeps MoE expert tensors on disk and fetches them
  per dispatch through `oaiy-engine`'s bounded `Ecache`. `llama-rs` has an `oaiy-engine` path
  dependency for `Ecache` / `WeightStore`; it is workspace-internal, with no cycle.
  - `GgufExpertStore` is the `WeightStore`. One record is `gate || up || down`, and
    a batch of misses is read on a small thread pool.
  - `MoeFfn` carries an optional `stream` handle.
  - `qwen3moe.rs` and `mixtral.rs` each have a `from_gguf_streaming` constructor.
  - The entry point is `Model::open_streaming(path, backend, ram_budget)`:
    `oaiy-llm run model.gguf --budget N`.
- Until 2026-09-19, streaming read from a copy of the model in OAIY's `xdb` object
  store (`llama-rs/src/xdb_container.rs`, `oaiy-llm convert`). Reading the `.gguf` in
  place replaced it, and the xdb crate was removed.

**Attention on the CPU in one pass (2026-10-05).** `CpuBackend::attention` (`ggml-rs/src/cpu.rs`) reads each query
head's KV head in place, each (query, head) row on its own thread, its dot products as 8 running sums; the default
copied the cache's prefix, repeated it for every query head of a group, and took the softmax on one thread. The
WebGPU backend, whose cache is on the host, uses it too: a 3B Llama's 2,000-token prompt on the portable build went
from 59 s to 18 s with it.

**Dense models on WebGPU, round trips and host work (2026-10-05).** `Backend::linear_q_many` (`ggml-rs`) takes one
input against several packed weights, which the WebGPU backend makes one submit and one read back; `Weight::linear_many`
(`llama-rs`) uses it for a layer's q, k and v (Llama, Qwen3, Gemma 3). A tied LM head keeps the GGUF's packed table
too (`CommonTensors::tied_packed`), and heads with it when the backend kept the dense f32 table on the host (WebGPU):
a quantized matmul on the device, not 1.6 GB of f32 on the CPU each token. `CpuBackend` spreads a prompt's
RMSNorm, RoPE (its sines and cosines made once a position, not once an element) and fused SwiGLU/GeGLU rows over
the threads. A 3B Llama's decode step on the portable build went from 87 ms to 37.

**The streaming roadmap (2026-08, `docs/ROADMAP.md`).** Each change is marked with
its roadmap id:

- Host-cache leases (CACHE-01).
- A per-device VRAM expert cache (CACHE-02 / STREAM-01,
  `llama-rs/src/expert_stream/device_cache.rs`).
- Pinned staging and async uploads (GPU-02, `ggml-rs-cuda/src/transfer.rs`).
- GPU top-k routing and grouped MoE kernels (MOE-01 / MOE-02,
  `ggml-rs-cuda/src/moe.rs`, `llama-rs/src/moe_cuda.rs`).
- Backend synchronization points for `oaiy-llm bench` (PERF-01).

DeepSeek-V4.1 (`dsv41`, `dsv41-cuda`) does not go through this stack: it has its own
safetensors reader and CUDA kernels (cudarc directly), and shares only `oaiy-engine`'s cache
and store seam.

## cudarc

`crates/cudarc` is cudarc 0.19.8 from crates.io, used in place of it through
`[patch.crates-io]` in the workspace manifest (and excluded from the workspace). Its
changes are marked `VENDORED-LOCAL`:

- `CudaSlice::drop` makes its context current before it frees, as the stream and event
  drops already do. Without it, a buffer freed on a thread where another device's context
  (or none) was current was never released: a model unloaded from its engine thread kept
  all of its VRAM, and the next model found the GPU full.
- Graph capture for whole decode steps (`CudaStream::begin_graph` and `end_graph`):
  - While a stream is being captured, its allocations come from an arena, a bump allocator
    reset each capture, so the same launches get the same addresses every step. Those
    slices are not `owned` and free nothing.
  - Frees anywhere in the context wait until the capture ends (`free_after`), because a
    free recorded into the graph would happen again at every replay.
