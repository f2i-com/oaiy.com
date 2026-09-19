# The GGUF stack

The crates `gguf`, `ggml-quants`, `ggml-rs`, `ggml-rs-cuda`, `tokenizer`, and
`llama-rs` in this directory are **vendored** from the author's own pure-Rust
llama.cpp-equivalent project:

- Source: the author's sibling `llm` workspace (copied with the owner's permission)
- Date: 2026-08-02
- Provenance: the author's own code; upstream license `MIT OR Apache-2.0`.

They are nrob's GGUF engine: the reader, the quant kernels, the CPU and CUDA
backends, the tokenizers, and the per-architecture models. They keep their original
external dependencies and upstream style; the std-only rule for `nrob` / `dsv41`
does **not** apply to them (see `CONVENTIONS.md`).

`ggml-rs-cuda` is a workspace member but is kept out of `default-members`: building it
needs a CUDA toolchain (cudarc dynamic-linking), so the default build/test stays
CPU-only. It is pulled in by the `cuda` feature of `llama-rs` (and `nrob-cli`, which
forwards to it: `cargo build -p nrob-cli --features cuda`, then `nrob run MODEL.gguf
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
  per dispatch through `nrob`'s bounded `Ecache`. `llama-rs` has a `nrob` path
  dependency for `Ecache` / `WeightStore`; it is workspace-internal, with no cycle.
  - `GgufExpertStore` is the `WeightStore`. One record is `gate || up || down`, and
    a batch of misses is read on a small thread pool.
  - `MoeFfn` carries an optional `stream` handle.
  - `qwen3moe.rs` and `mixtral.rs` each have a `from_gguf_streaming` constructor.
  - The entry point is `Model::open_streaming(path, backend, ram_budget)`:
    `nrob run model.gguf --budget N`.
- Until 2026-09-19, streaming read from a copy of the model in nrob's `xdb` object
  store (`llama-rs/src/xdb_container.rs`, `nrob convert`). Reading the `.gguf` in
  place replaced it, and the xdb crate was removed.

**The streaming roadmap (2026-08, `docs/ROADMAP.md`).** Each change is marked with
its roadmap id:

- Host-cache leases (CACHE-01).
- A per-device VRAM expert cache (CACHE-02 / STREAM-01,
  `llama-rs/src/expert_stream/device_cache.rs`).
- Pinned staging and async uploads (GPU-02, `ggml-rs-cuda/src/transfer.rs`).
- GPU top-k routing and grouped MoE kernels (MOE-01 / MOE-02,
  `ggml-rs-cuda/src/moe.rs`, `llama-rs/src/moe_cuda.rs`).
- Backend synchronization points for `nrob bench` (PERF-01).

DeepSeek-V4.1 (`dsv41`, `dsv41-cuda`) does not go through this stack: it has its own
safetensors reader and CUDA kernels (cudarc directly), and shares only `nrob`'s cache
and store seam.
