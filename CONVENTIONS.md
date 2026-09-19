# CONVENTIONS.md — engineering contract for the nrob workspace

This workspace is a Rust (edition 2021) inference engine, `nrob`, that reads GGUF and
safetensors weights in place. This file is the contract every change to the engine
crates (`nrob`, `nrob-cli`, `dsv41`, `dsv41-cuda`) is held to.

## Hard rules

- **std-only.** No external crates in `nrob` or `dsv41`: no serde, no tokio, nothing.
  The core is zero-dependency by design. Do not add a dependency "just for this one
  thing"; write the 30 lines instead. (`dsv41-cuda` depends on cudarc for the GPU; the
  GGUF stack is covered below.)
- **No `unsafe`** in `nrob`, `nrob-cli` or `dsv41`: their crate roots say
  `#![forbid(unsafe_code)]`. `dsv41-cuda` may use it for kernel launches, pinned
  memory and SIMD dispatch, with a `SAFETY:` comment on every block explaining why safe
  Rust cannot express it and what makes it sound.
- **Weights are read in place.** GGUF and safetensors, as published. No converters and
  no weight format of our own.
- **Cross-platform.** MSVC/gnu Windows and Unix, from one source. Put platform
  differences behind `#[cfg(...)]` in exactly one place per concern (see
  `gguf::source::read_exact_at` for positioned reads).

## Style

- **Naming:** free functions and methods are snake_case; types are CamelCase
  (`HostLease`). Names describe what a thing does, not where it came from.
- **Comments carry design rationale** (why the cache is LFRU, why a record is
  gate||up||down, why reads bypass the page cache). Keep them when editing code; update
  them when behavior changes.
- **Types:** prefer `&[T]` / `&mut [T]` over pointer+length pairs; out-params become
  return values; nullable values become `Option`.
- **Threads:** `std::thread`, scoped where the work borrows. Synchronization:
  `Mutex` / `Condvar`.

## Errors

- Everything fallible returns `nrob::Result<T>` with `nrob::Error`; pick the variant
  matching the failure (`Error::Format` for a malformed header, …) and put the detail
  string into the payload. The GGUF stack uses its own `LlamaError` / `GgufError`.
- **Never panic on bad external data.** A truncated shard, a corrupt header, a tensor
  of the wrong size: all `Err`. `unwrap`/`expect` are for invariants you can argue from
  code you own, never for input bytes.
- **Never print from library code.** No `println!`/`eprintln!` in `nrob` or `dsv41`.
  The CLI and the examples print; the libraries return.

## Done means

- `cargo build` for your crate compiles clean after your edit, warnings included;
  don't leave dead-code noise behind.
- `cargo test --workspace` passes. Tests you add use `std::env::temp_dir()` and clean
  up after themselves.
- A change to the DeepSeek path also passes its golden gates (README, "Testing").

## GGUF stack exception

The crates `gguf`, `ggml-quants`, `ggml-rs`, `ggml-rs-cuda`, `tokenizer` and
`llama-rs` are the author's own Rust GGUF stack, vendored from their `llm` workspace
(see `crates/VENDORED.md`). They keep their own external dependencies and upstream
style, and the std-only and no-`unsafe` rules above do not apply to them. Local
changes should be minimal and marked with a comment (`// VENDORED-LOCAL: ...`) so
re-vendoring diffs stay readable.
