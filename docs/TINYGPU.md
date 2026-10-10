# WebGPU on a Mac's eGPU, through tinygrad

macOS has no driver for an NVIDIA card in a Thunderbolt enclosure, so no WebGPU adapter sees one: wgpu's on a Mac is
Metal, and Metal knows the Mac's own GPU alone. tinygrad drives such a card itself (its TinyGPU app and driver
extension), and OAIY already sends a model's chats to tinygrad's own server there ([the Mac](MAC.md), "A card in a
Thunderbolt enclosure"). This is the other way: **the card as a WebGPU adapter**, so a program written against wgpu's
compute (OAIY's own engine among them) runs its kernels on it as they are.

Three parts, each its own:

- **`crates/wgsl-cuda`**: a WGSL compute kernel as CUDA C++. naga parses and validates the WGSL as wgpu does; its
  module is walked into one `extern "C" __global__` kernel beside a prelude of WGSL's types and built-ins
  (`src/prelude.cuh`): a vector as WGSL lays it out, its operators WGSL's (shifts by the amount modulo the width,
  comparisons a vector of bool), `dot4I8Packed` as `__dp4a`, workgroup memory `__shared__` and zeroed as WebGPU does,
  every expression named where naga emits it. `wgsl-cuda IN.wgsl OUT.cu` writes one and says its workgroup size and
  bindings.
- **`tools/tinygpu/webgpu_server.py`**: one process that holds the card through tinygrad and does what WebGPU's compute
  needs, over a Unix socket: buffers (zeroed), writes and reads, kernels (CUDA compiled by nvcc, kept by the source's
  hash), and submissions of dispatches, copies and clears, run in order on the card's compute queue.
- **`crates/wgpu-tinygpu`**: a `wgpu::Adapter` (wgpu's `custom` backend) that asks that server. A compute pipeline's
  WGSL becomes CUDA at creation; a command buffer's work goes in one message at `Queue::submit`. Compute alone: render
  pipelines, textures, samplers and queries are refused.

OAIY's engine takes it with the feature `tinygpu` (`oaiy-llm-server`, `ggml-rs-wgpu`) and
`OAIY_WEBGPU_ADAPTER=tinygpu` (or `tinygpu:<socket>`).

## Set up

1. **TinyGPU and its driver**, by tinygrad's own instructions (`docs/tinygpu.md` in tinygrad), and **NVIDIA's compiler**
   in Docker: `sh extra/setup_nvcc_osx.sh` from a tinygrad checkout (nvcc behind `~/.local/bin/nvcc`).
2. **tinygrad at the last commit with its macOS path.** On 5 September 2026 tinygrad's master took the TinyGPU client
   out ("remove hcq1 remote for now"); its commit before that, `33cd373ad`, has it, and with
   `tools/tinygpu/tinygrad-compile-server.patch` (its compile server read a cubin past 64 KB cut short) it runs
   OAIY's kernels:

   ```sh
   git clone https://github.com/tinygrad/tinygrad ~/tinygrad && cd ~/tinygrad
   git checkout -b tinygpu-macos 33cd373ad && git am /path/to/oaiy.com/tools/tinygpu/tinygrad-compile-server.patch
   uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install -e . jinja2
   ```

   Keep it out of `~/Documents` (and the Desktop): a program OAIY starts from there asks macOS for that folder first.
3. **The server**, which holds the card for as long as it runs:

   ```sh
   DEV=NV PATH=~/.local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH \
     ~/tinygrad/.venv/bin/python tools/tinygpu/webgpu_server.py
   ```

4. **A program on it**: `cargo run --release -p wgpu-tinygpu --example smoke`, or OAIY's engine built with
   `--features tinygpu` and run with `OAIY_WEBGPU_ADAPTER=tinygpu`.

## The card's link

A process that holds the card and ends without tinygrad's own exit (killed, or after the card faulted) leaves the card's
firmware up; the next process to open it finds its region (WPR2) and resets the card, and over Thunderbolt that reset
has twice not come back: "Booter failed to execute, mailbox is ffffffff", the PCI link down until the enclosure was
powered off and on (`system_profiler SPPCIDataType` says "Link down"). So the server ends by releasing the card at
SHUTDOWN, SIGTERM, SIGHUP or an interrupt, and OAIY's own launcher for tinygrad's LLM server does the same since
(`egpu_serve.py`, `egpu.rs`'s `halt`). A kernel that faults the card still costs it: the server's next start resets it.

## Checked, and not

Checked on an RTX 4090 in a Razer Core X (PCIe 3.0 x4 over Thunderbolt 3) on an M5 Pro, macOS 27, 10 October 2026:

- every one of the 309 kernels OAIY's GPU tests make translates, and nvcc compiles each for sm_89;
- a kernel translated and loaded into tinygrad runs right: 1000 of 1000 values. CUDA's `blockDim` and `gridDim` are the
  driver's words in constant bank 0, which tinygrad's launches leave 0 (its own kernels never read them): the kernels
  take the workgroup's size as constants, and the server fills the grid's words;
- the server: a buffer zeroed, written, a 100,000-value dispatch, a copy and a clear read back as they should.

A cubin of several kernels faulted every SM: tinygrad reads a program's registers from the cubin's `.nv.info` and takes
the last kernel's of several, so the server compiles one kernel a cubin and refuses any other (the card was lost to
that fault, and is off its link as this is written).

**Not checked yet:** OAIY's GPU tests on the adapter (`OAIY_WEBGPU_ADAPTER=tinygpu cargo test -p ggml-rs-wgpu
--features tinygpu`), a model on it, any speed. The tensor cores are not used (no cooperative matrices: CUDA's `wmma` is
where 16 x 16 fragments would go); a write to a buffer waits for the card first; each dispatch is a Python call.
