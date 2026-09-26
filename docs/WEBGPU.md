# WebGPU and CPU inference

nrob's GGUF models run on NVIDIA GPUs through CUDA. On a machine without CUDA
(an AMD or Intel GPU, a laptop's integrated graphics, Apple silicon) they run
through **WebGPU**, and on a machine with no usable GPU, on the **CPU**.

## How it works

`crates/ggml-rs-wgpu` is a backend for the GGUF stack beside `ggml-rs-cuda`. A
GGUF model spends nearly all of its time in one operation: multiplying
activations by quantized weight matrices (`linear_q`). The WebGPU backend
uploads those matrices to the GPU **in their GGML block layout**, so they take
the same memory as the file. It runs the multiply in WGSL. Everything else (norms,
RoPE, attention, recurrent state) runs on the CPU backend, with activations in
RAM.

- **Types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
  and IQ4_XS. Each shader's block decode is a line-by-line port of
  `ggml-quants`' `dequantize_block`, so the weights are the CPU's exactly. The
  parity test (`cargo test -p ggml-rs-wgpu`) checks every type against the CPU
  dequantization for 1, 5 and 9 input rows.
- **Budget:** WebGPU cannot report free memory, so a budget caps the weights
  placed on the GPU: 8 GiB on a discrete GPU, 2 GiB on an integrated one, or
  `--webgpu-gb N`. Weights past it (and types without a shader) stay in RAM and
  use the CPU path. A model bigger than the GPU still runs, split between the two.
- **Large tensors** are split by rows below the adapter's binding limit (a
  152k-vocabulary Q6_K output head is about 640 MB).

The CPU path got faster in the same change. `CpuBackend::linear_q` used to
inflate each whole weight matrix to F32 on every call; it now dequantizes one
row at a time per thread. Dense projections now split across threads during
decode, where one thread used to do all the work. Both produce the same bits as
before.

## Using it

```sh
# the CLI
cargo build --release -p nrob-cli --features webgpu
nrob run model.gguf "prompt" --webgpu            # or --webgpu-gb 20

# the OpenAI-compatible server, built without CUDA so it starts anywhere
cargo build --release -p nrob-server --no-default-features --features webgpu --bin nrob-server-webgpu
nrob-server-webgpu --model model.gguf --backend auto   # WebGPU, else the CPU
```

`tools/qwen-image/build.ps1` builds `nrob-server-webgpu` beside the CUDA
`nrob-server`. It is a separate executable because a CUDA build imports the
NVIDIA driver's DLLs and will not start without them. `nrob-server-webgpu`
imports only system DLLs.

In **NROB Studio**, Settings → Language model → *Runs on*:

- `auto` (the default) starts the CUDA build when an NVIDIA GPU answers. If that
  build dies while loading (a missing driver, say), the studio starts
  `nrob-server-webgpu` instead. Without an NVIDIA GPU it starts
  `nrob-server-webgpu` directly.
- `cuda`, `webgpu` and `cpu` pin one.

The Overview page shows what the model actually runs on.

`nrob-server-webgpu` serves GGUF models only. DeepSeek-V4.1 checkpoints, OrcaSAQ
(EXL3) and the observer are CUDA engines, and it says so rather than trying.
Image and video generation (`nrob-diffusion`) still need CUDA for useful speed.

## Measured (2026-09-26)

On a Ryzen 9 9950X3D with an RTX 5090 reached through Vulkan (WebGPU picked it
as the high-performance adapter), greedy decoding:

| Model | CPU | WebGPU | Tokens identical |
|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 13.6 tok/s (4.1 before the CPU change) | 21.9 tok/s | 48 of 48 |
| Qwen3.5 9B Q4_K_M | 2.7 tok/s | 9.2 tok/s | 40 of 40 |
| Qwen3.8 27B Q4_K_M | 0.79 tok/s | 4.0 tok/s (14.7 GB on the GPU); 1.16 tok/s with a 6 GB budget | 40 of 40, all three |

Through `nrob-server-webgpu` the 9B loads in 4.1 s and answers chat requests.

These runs gave the same tokens as the CPU, but that is not guaranteed in
general: the GPU sums each dot product in a different order, so a near-tie
between two tokens can eventually go the other way.

Speed is bounded by design: every projection is one upload, one dispatch and one
read-back, dozens a token, so WebGPU decode runs several times slower than
CUDA's. Keeping activations on the GPU between projections (norms and attention
in WGSL too) is the next step if it needs to be faster.
