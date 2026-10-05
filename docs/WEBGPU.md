# WebGPU and CPU inference

OAIY's GGUF models run on NVIDIA GPUs through CUDA. On a machine without CUDA
(an AMD or Intel GPU, a laptop's integrated graphics, Apple silicon) they run
through **WebGPU**, and on a machine with no usable GPU, on the **CPU**.

## How it works

`crates/ggml-rs-wgpu` is a backend for the GGUF stack beside `ggml-rs-cuda`. A
GGUF model spends nearly all of its time in one operation: multiplying
activations by quantized weight matrices (`linear_q`). The WebGPU backend
uploads those matrices to the GPU **in their GGML block layout**, so they take
the same memory as the file. It runs the multiply in WGSL. A prompt's other work
(norms, RoPE, attention, recurrent state) runs on the CPU backend, with
activations in RAM. A Llama's, Qwen3's or Gemma 3's decode step, and each of its
prompt's chunks, runs whole on the GPU in one submit ("Dense models on the GPU",
below).

- **Types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
  and IQ4_XS. Each shader's block decode is a line-by-line port of
  `ggml-quants`' `dequantize_block`, so the weights are the CPU's exactly. The
  parity test (`cargo test -p ggml-rs-wgpu`) checks every type against the CPU
  dequantization for 1, 5, 9 and 70 input rows. A prompt (more than 8 rows) takes
  a tiled kernel: 64 tokens by 64 weight rows a workgroup, each type's own decode
  into workgroup memory and its sums in registers (a 4096 x 4096 Q4_K weight
  against 512 tokens in 7.0 ms where a row a workgroup took 24.2).
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
The observer is a CUDA engine, and it says so rather than trying. Qwen3.8-Flash-Next
and DeepSeek-V4.1 run on it too (below).

## DeepSeek-V4.1 without CUDA

The CPU model (`dsv41::model`, the reference the CUDA path is tested against) serves it,
with what measured slowest on the CPU moved (docs/DEEPSEEK_V41.md, "The CPU model,
measured"; `dsv41::profile` says where a pass's time goes):

- Its dense trunk, the 390 fp8 and bf16 matrices of 9.7 GB, on the adapter
  (`ggml_rs_wgpu::dense`; the activation still quantized and the result still rounded by
  the CPU model, so the answer matches: cosine 1.000000 and the same greedy tokens).
  Projections of one input go in one submit (a layer's eight `wo_a` groups, `wq_a` with
  `wkv`, the router with the shared expert's gate and up): a decode step's 667 round
  trips became 270.
- Its routed experts there too (`WgpuExperts`). A prompt's busy ones (eight tokens or
  more) pass through slots made once, each record uploaded as stored and its MXFP4
  matrices read in place with their e8m0 scales (`RecordSlots`); a prompt's MoE hands
  each record over as it is read, the busy ones to the GPU a group of 32 at a time and
  the rest to the CPU's workers meanwhile, so the reads and the matmuls overlap. What
  the budget has left after the trunk and the slots keeps the experts used most between
  passes, by the CUDA engine's VRAM policy (LFRU counted in tokens, aged every 128
  steps; a prompt's most used taken in from RAM once it is read): 935 of them on a
  32 GB card with a 27 GiB budget. A decode step's held experts (about 110 of its 240)
  are computed there while the CPU reads and computes the rest.
- The sparse attention, the indexer's scores and the hyper-connections' mixing spread
  over the CPU's threads (serial, the attention was most of a prompt's time and the
  mixing's dot products 29 s of it); a layer's expert records read eight at a time.

A conversation's next turn continues the state the last prompt left (a checkpoint a
token short of its end, since the next prompt writes the last reply its own way),
reading only the new tail in one chunk (`dsv41` continues a sequence by a chunk exactly
as it would token by token).

On the RTX 5090 (2026-10-05), 27 GiB budget, 96 GB of RAM for experts, the checkpoint on
a USB SSD (a Samsung T9 on a 20 Gbps port: 1.4 GB/s unbuffered, one reader or eight):

| | CPU only | + trunk on the GPU | + parallel attention | + GPU experts, parallel reads | + overlapped MoE, record slots, VRAM tier, one submit a group of projections |
|---|---:|---:|---:|---:|---:|
| 2,000-token prompt | 1,294 s | 1,081 s | 558 s | 490 s (attention 81 s, MoE 402 s) | 261 s (attention 68 s, MoE 188 s) |
| Warm decode | 2.06 s a token | 1.47 s | 1.21 s | 1.19 s | 1.02 s |

Both are now the drive's. The prompt reads 230 GB of expert records, 164 s at its rate,
and its MoE takes 188; the GPU's part of it (55 s since the prompt kernel keeps its sums
in registers, 81 s before) is hidden under the reads. A warm decode step reads about
1 GB of records the RAM and the GPU do not hold (0.4 to 1.2 GB, which its time follows).
Cosine 0.998825 to the CPU model's logits and the same greedy tokens (an expert's
outputs on the GPU are the CPU's to the bit but one in 170,000).

An Agent's tool call end to end (before the overlapped MoE): a 285-token prompt in
229 s and its call at 1.3 tokens a second; the next turn (the tool's result) reused 284
tokens and took 52 s. On an internal NVMe the reads, most of what is left, would be
several times faster.

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

## GLM-5.3-Flash without CUDA

Its GGUF (`glm5next`) streams its experts on the CPU, from RAM and the drive, as the
CUDA build streams them to the cards, and puts its dense layers on the WebGPU adapter.
The catalog's is unsloth's 4-bit dynamic GGUF of Z.ai's weights (UD-Q4_K_XL, 199.7 GB:
Q4_K, Q5_K, Q6_K and Q8_0 tensors, read from its headers). On the RTX 5090 (2026-10-05,
a GGUF of the same architecture and tensor types, 192 GB of RAM): loaded in 11 s with
6.2 GB on the GPU; the expert cache then filled to a 65 GB working set (it takes a share
of the free RAM, so a smaller computer holds fewer experts and reads more); a short
answer at 0.5 tokens a second cold and 1.3 warm.

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

Those numbers were the op-by-op path: every projection one upload, one dispatch
and one read-back, dozens a token. The next section is what replaced it for the
dense families.

## Dense models on the GPU (2026-10-05)

A Llama's, Qwen3's or Gemma 3's decode step is one submit (`ggml_rs::chain`, a
`DeviceChain` the WebGPU backend implements; `llama-rs`'s `chain_decode`): every
layer's norm, q, k and v (Qwen3's and Gemma 3's per-head norms), RoPE (its sines
and cosines made on the host as the CPU's rope makes them, a table each base and
scaling among the layers), the K and V stored into a copy of the KV cache kept on
the GPU, attention over it (split in runs of 256 positions across workgroups and
put together after; Gemma 3's local layers within their window), the output
projection, residual, FFN (SwiGLU, or Gemma's GeGLU) and residual (Gemma 3's
post-norms before each), then the head. Only the logits and the step's K and V
rows (for the host's cache) come back. The GPU's copy of the cache is brought up
to date with the rows the host wrote since (a prompt's: `KvCache::dirty_from`).
A prompt's chunk is one submit too, every layer's rows at once: RoPE from a table
of the chunk's positions, its K and V stored into the GPU's copy of the cache, and
the causal attention over it. Both run when every weight a step reads is on the
GPU; otherwise, and with `OAIY_NO_CHAIN`, the op-by-op path.

Also for the dense models on WebGPU: the CPU's attention runs in one pass (each
KV head's rows read once for its group of query heads; the default copied the
cache and took the softmax on one thread), a tied head heads with the GGUF's
packed table on the GPU (it was 1.6 GB of f32 read on the CPU every token), a
layer's q, k and v go in one submit, the CPU's RMSNorm, RoPE and SwiGLU spread a
prompt's rows over the threads, and a prompt goes in chunks of 512 tokens.

On the RTX 5090 (Vulkan), Q4_K_M, greedy:

| Model | Decode, op by op | Decode, one submit | A 2,000-token prompt |
|---|---:|---:|---:|
| Llama 3.2 3B | 35 tok/s (11.6 at 512 tokens of context before the changes above) | 73-81 tok/s; 70 at 2,000 tokens of context | 1.5 s (59.2 before the day's changes, 8.4 op by op) |
| Qwen3 0.6B | 53 tok/s | 182 tok/s | 0.8 s through the server |
| Gemma 3 4B | 28.5 tok/s | 60 tok/s | |

Each chained model gives the op-by-op path's 64 greedy tokens after 64- and
1,500-token prompts (past Gemma 3's 1,024-token window), the prompt's logits and
every step's cosine 1.000000, and the same two-turn conversation word for word
(Gemma 3's second turn reusing 130 tokens of the first's state). A steady Llama 3.2 3B step
is 0.4 ms of recording and 14 ms on the GPU; the one-row matmul reads its weights
at 170-280 GB/s however its lanes are laid out (measured each way), so a kernel
of wide loads is what would take it further.

Gemma 3 itself was wrong on every backend until this day: one RoPE base on every
layer, where its sliding-window layers take 10,000 and its global ones
`rope.freq_base` with the GGUF's linear scaling, and its SentencePiece tokens
were not llama.cpp's (a space before every line, merges out of score order,
newlines dropped from replies). Gemma 3 4B answered a markdown prompt of a few
hundred tokens with fragments of it; it answers as llama.cpp does now.
