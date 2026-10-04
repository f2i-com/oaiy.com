# WebGPU and CPU inference

OAIY's GGUF models run on NVIDIA GPUs through CUDA. On a machine without CUDA
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
  In OAIY the studio passes `--webgpu-gb` itself when `llm.webgpu_gb` is not set:
  the largest GPU's memory less `llm.vram_headroom_gb` and 2 GB for the cache and
  work buffers (27 on a 32 GB card), and nothing under 4 (an integrated GPU keeps
  the engine's default). It learns the GPUs from `nvidia-smi`, or, without an
  NVIDIA driver, from the display adapters the OS lists (on Windows the driver's
  `HardwareInformation.qwMemorySize`, on Linux amdgpu's `mem_info_vram_total`), so
  an AMD or Intel card gets a budget and the setup's "fits your GPU" labels too.
- **Which GPU:** the high-performance adapter, or the one `OAIY_WEBGPU_ADAPTER`
  names (part of its name, any case: `radeon`, `arc`, `5090`). An unknown name
  fails with the list of adapters wgpu found.
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
cargo build --release -p oaiy-llm-cli --features webgpu
oaiy-llm run model.gguf "prompt" --webgpu            # or --webgpu-gb 20

# the OpenAI-compatible server, built without CUDA so it starts anywhere
cargo build --release -p oaiy-llm-server --no-default-features --features webgpu --bin oaiy-llm-server-webgpu
oaiy-llm-server-webgpu --model model.gguf --backend auto   # WebGPU, else the CPU
```

`tools/qwen-image/build.ps1` builds `oaiy-llm-server-webgpu` beside the CUDA
`oaiy-llm-server`. It is a separate executable because a CUDA build imports the
NVIDIA driver's DLLs and will not start without them. `oaiy-llm-server-webgpu`
imports only system DLLs.

In **OAIY**, Settings → Language model → *Runs on*:

- `auto` (the default) starts the CUDA build when an NVIDIA GPU answers. If that
  build dies while loading (a missing driver, say), the studio starts
  `oaiy-llm-server-webgpu` instead. Without an NVIDIA GPU it starts
  `oaiy-llm-server-webgpu` directly.
- `cuda`, `webgpu` and `cpu` pin one.

The Overview page shows what the model actually runs on.

`oaiy-llm-server-webgpu` serves GGUF models of the types above, and OrcaSAQ (EXL3,
below). A GGUF with IQ1, IQ2 or IQ3 tensors (unsloth's smaller "UD" quants mix them
in, even UD-Q4_K_M) does not load, and OAIY's setup refuses one before it is added.
DeepSeek-V4.1 checkpoints and the observer are CUDA engines, and it says so rather
than trying. Qwen3.8-Flash-Next runs on it too (below).

## EXL3 (OrcaSAQ) without CUDA

`ggml_rs_wgpu::exl3` keeps an EXL3 projection's packed trellis words on the GPU (VRAM
use equals the checkpoint's) and decodes them inside the matmul, in WGSL: one kernel
for a decode step's single row (thread `(r, c)` decodes weight `(r, c)` of each 16x16
tile into one sum), one for a prompt's rows (each tile decoded once into workgroup
memory, 32 rows a pass). Each weight is decoded exactly as `Exl3Data::value`: the
mul1 product `(1024 + bytesum) * 1774/2^18` is exact in f32 and the one rounding to
f16 is written out by hand, so no driver's f16 rules come into it. The two Hadamard-128
transforms, the channel maps and exllamav3's f16 roundings run on the host. A weight
beyond the budget, or a computer without a GPU, decodes on the CPU (`Exl3Cpu`), slowly.
The tests check both against an independent bit-by-bit packing oracle at all eleven
supported bitrates, on the RTX 5090 and the Radeon iGPU.

OrcaSAQ-2-27B through `oaiy-llm-server-webgpu` on the RTX 5090 (2026-10-05): loaded in
19 s with 10.7 GB of EXL3 weights on the GPU; a 162-token prompt in 21.7 s and 3.9
tokens a second, against 17 s and 4.5 for Qwen3.8 27B Q4_K_M on the same path (the
rest is the host's share of every portable model); tool calls made and answered. No
PEFT adapters and no vision tower without CUDA.

## Qwen3.8-Flash-Next without CUDA

Its 512 experts a layer (and the shared one) are `ggml_rs_wgpu::exl3::Exl3MoeHost`:
each projection on the GPU while the budget holds it, the rest decoded on the CPU
(through a 65,536-entry table of mul1's values, where decoding was most of their
time), the routing on the host exactly as the CUDA kernel routes. A layer's experts
run as two batches (every gate and up, then every down), the GPU's recorded in one
encoder and read back with one submit, the CPU's an expert a thread meanwhile. The
attention, delta-net and head matrices keep their share of the budget
(`flashnext::dense_exl3_bytes`, which leaves out the n-gram table: its rows are
trellis-quantized too, but it is read from the disk); the sigmoid-gated delta-net
step runs on the host, a head a thread, checked against the CUDA kernel; the
hyper-connection matrices are f32 on the host (unpacked once, where the host op
unpacked f16 every call).

On the RTX 5090 (2026-10-05), 27 GiB budget: loaded in 22 s with 27.9 GB of its
weights in VRAM and 23 GB of RAM in use (its working set; Windows also charges the
VRAM to its commit, 52 GB in all); 1.0 tokens a second and a 162-token prompt in
35 s; tool calls made and answered. A decode step's time goes mostly to the MoE
(0.64 s), the delta-net layers (0.24 s: in place a projection waits about 2 ms,
against 0.4 ms alone) and the hyper-connections (0.12 s). With every expert on the
CPU instead it was a little faster here (1.2 tokens a second, the prompt in 28 s:
sixteen fast cores beat a layer's two GPU round trips) but took 48 GB of RAM; that
is how it ran until the reserve stopped counting the 32.6 GB n-gram table, which
left the experts none of the budget. On two GPUs with CUDA it is far faster.
No PEFT adapters and no vision tower without CUDA.
Image and video generation (`oaiy-media`) still need CUDA for useful speed.

## Measured (2026-09-26)

On a Ryzen 9 9950X3D with an RTX 5090 reached through Vulkan (WebGPU picked it
as the high-performance adapter), greedy decoding:

| Model | CPU | WebGPU | Tokens identical |
|---|---|---|---|
| Llama 3.2 1B Q4_K_M | 13.6 tok/s (4.1 before the CPU change) | 21.9 tok/s | 48 of 48 |
| Qwen3.5 9B Q4_K_M | 2.7 tok/s | 9.2 tok/s | 40 of 40 |
| Qwen3.8 27B Q4_K_M | 0.79 tok/s | 4.0 tok/s (14.7 GB on the GPU); 1.16 tok/s with a 6 GB budget | 40 of 40, all three |

Through `oaiy-llm-server-webgpu` the 9B loads in 4.1 s and answers chat requests.
Forcing Direct3D 12 (`WGPU_BACKEND=dx12`) passes the same parity test and gives
the same 1B tokens at 16.8 tok/s.

Only Vulkan, D3D12 and Metal are opened unless `WGPU_BACKEND` names others (`gl`
adds OpenGL). With OpenGL too, each instance started a thread in NVIDIA's GL
driver whose exit, as the instance dropped, deadlocked on the Windows loader lock
against another thread opening Vulkan: about one test run in forty hung.

Two adapters have been tested: this RTX 5090 and the Ryzen's integrated AMD Radeon
(RDNA 2, 2 GB), each through Vulkan and D3D12 (2026-10-04,
`OAIY_WEBGPU_ADAPTER=radeon`, with and without `WGPU_BACKEND=dx12`). Every type's
parity test passes on both, and Qwen3.5 4B answered correctly on the Radeon,
slowly (36 s for its first seven tokens, most of it the prompt on a small iGPU).
Through the studio's own budget (27 GiB on the 5090), Qwen3.5 27B and Qwen3.8 27B
Q4_K_M load in 10-20 s and make and answer tool calls. No discrete AMD or Intel
card has been tried. The shaders are plain WGSL, but driver compilers differ, so
run `cargo test -p ggml-rs-wgpu` on a new adapter before trusting it. The budgets
are not measurements: drivers often spill oversubscribed buffers to system memory
silently, so they get slower rather than failing.

These runs gave the same tokens as the CPU, but that is not guaranteed in
general: the GPU sums each dot product in a different order, so a near-tie
between two tokens can eventually go the other way.

Speed is bounded by design: every projection is one upload, one dispatch and one
read-back, dozens a token, so WebGPU decode runs several times slower than
CUDA's. Keeping activations on the GPU between projections (norms and attention
in WGSL too) is the next step if it needs to be faster.
