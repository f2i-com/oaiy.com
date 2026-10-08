# WebGPU and CPU inference

OAIY's models run on a GPU through **WebGPU** (an NVIDIA, AMD or Intel GPU, a
laptop's integrated graphics, Apple silicon), and on a machine with no usable GPU,
on the **CPU**. There is no CUDA backend any more: the one that was stands on the
branch `backup/cuda-support-2026-10-08`. Where this page sets a figure beside CUDA's,
that figure is that build's; the sections below are dated, and each says what it
measured then.

## How it works

`crates/ggml-rs-wgpu` is the GGUF stack's GPU backend. A
GGUF model spends nearly all of its time in one operation: multiplying
activations by quantized weight matrices (`linear_q`). The WebGPU backend
uploads those matrices to the GPU **in their GGML block layout**, so they take
the same memory as the file. It runs the multiply in WGSL. A prompt's other work
(norms, RoPE, attention, recurrent state) runs on the CPU backend, with
activations in RAM. A Llama's, Qwen3's or Gemma 3's decode step, and each of its
prompt's chunks, runs whole on the GPU in one submit ("Dense models on the GPU",
below), and so does Qwen3.5's and Qwen3.8 27B's ("Qwen3.5 and Qwen3.8 27B on the
GPU").

- **Types:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
  and IQ4_XS. Each shader's block decode is a line-by-line port of
  `ggml-quants`' `dequantize_block`, so the weights are the CPU's exactly. The
  parity test (`cargo test -p ggml-rs-wgpu`) checks every type against the CPU
  dequantization for 1, 5, 9 and 70 input rows. A prompt (more than 8 rows) takes
  a tiled kernel: 64 tokens by 64 weight rows a workgroup, each type's own decode
  into workgroup memory and its sums in registers (a 4096 x 4096 Q4_K weight
  against 512 tokens in 7.0 ms where a row a workgroup took 24.2).
- **The K-quants (Q3_K, Q4_K, Q5_K, Q6_K)** have kernels of their own that read
  the weights wide (vec4 or word loads, where the generic kernels read a byte a
  load): for a decode step's one row of `x` (8 weight rows a workgroup, 32 lanes
  a row: 1,100-1,400 GB/s, where the generic one read 170-280), for up to 24 rows
  (the same lanes, each weight they decode applied to 8 rows of `x` a workgroup: a
  prompt's chunk of 6 tokens of Qwen3.8 27B in 73 ms where 358), and a tiled one
  for more (64 values of `k` a step, all 256 threads decoding: its chunk of 512 in
  1.3 s where the generic tiled kernel took 2.4). Q3_K's 110-byte blocks are
  padded to 112 on the GPU so they are seven vec4s, and unpadded when the weights
  come back to the host.
- **Budget:** WebGPU cannot report free memory, so a budget caps the weights
  placed on the GPU: a discrete card's memory less 4 GiB where Vulkan reports it
  (its largest device-local heap: 27.8 GiB of a 32 GB card), else 8 GiB; 2 GiB on
  an integrated GPU; or `--webgpu-gb N`. The old 8 GiB of any discrete card kept
  5 GB of Qwen3.8 27B Q3_K_M (13.4 GB) on the CPU of a 32 GB card: 2.0 tokens a
  second where the whole model on the GPU made 6.0. Weights past it (and types without a shader) stay in RAM and
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
# the CLI (WebGPU is a default feature)
cargo build --release -p oaiy-llm-cli
oaiy-llm run model.gguf "prompt" --webgpu            # or --webgpu-gb 20

# the OpenAI-compatible server: it starts anywhere
cargo build --release -p oaiy-llm-server
oaiy-llm-server --model model.gguf --backend auto   # WebGPU, else the CPU
```

`tools/qwen-image/build.ps1` builds it with the media worker and the studio. The
build also makes `oaiy-llm-server-webgpu`: the same program under the name it had
while a CUDA build stood beside it, which the desktop's installer still stages it
by. Either imports only system DLLs. `--backend` is `auto`, `webgpu` or `cpu`;
`cuda` is refused by name.

In **OAIY**, Settings → Language model → *Runs on*:

- `auto` (the default): a GPU through WebGPU when there is one, else the CPU.
- `webgpu` and `cpu` pin one. A configuration that still says `cuda` is read as
  `auto`.

The Overview page shows what the model actually runs on.

The server serves GGUF models of the types above, and OrcaSAQ (EXL3,
below). A GGUF with IQ1, IQ2 or IQ3 tensors (unsloth's smaller "UD" quants mix them
in, even UD-Q4_K_M) does not load, and OAIY's setup refuses one before it is added.
The observer has no WebGPU path yet, and says so rather than trying. Qwen3.8-Flash-Next
and DeepSeek-V4.1 run on it too (below).

## DeepSeek-V4.1

The CPU model (`dsv41::model`, the reference the CUDA engine was tested against) serves it,
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
  the budget has left after the trunk and the slots keeps experts between passes: 935
  of them on a 32 GB card with a 27 GiB budget, and a second card 1,509 more. The
  15,360 routed experts are 290 GB and fit nowhere whole, so each is in one place (a
  card or RAM, never both), and while the server idles the cards take RAM's most used
  experts in place of their least used (`WgpuExperts::rebalance`: used twice as much
  and four times more, the displaced one read back into RAM from the drive). On a
  request's own path a card takes in only what a free slot holds: a swap there is an
  upload of 18.9 MB and an expert left in no tier. A decode step's experts on the
  cards (about 190 of its 240 once a session's are there) are computed there, an
  expert's whole step one submit a card (its gate and up projections, their SwiGLU
  quantized on the device, its down projection), while the CPU reads and computes the
  rest.
- The order of the tiers is the model's own count of each expert's uses
  (`dsv41::moe::Uses`: one for each call that routed to it, halved every 512 decode
  steps, the RAM cache's own counts with it), whatever the computer has: with no GPU
  it orders RAM alone, with one its tier and RAM, with more their share after the
  first card's. A usage profile keeps the counts from one run to the next (`--usage
  FILE`, written after each request; the studio keeps it with its prompt states, and
  an incognito request adds nothing to it): a start that finds one fills the first
  card's tier with the most used experts, the other cards with the next and RAM with
  the next again, each read from the drive in that order while the server idles.
  Without one the cards and RAM are filled by number and find their order as the
  experts are used.
- RAM for experts is four fifths of the memory that is free (`--ram-gb` a ceiling on
  it, none by default), and the rule is applied again every few seconds of idle, the
  tier's own records counted free: a server started while another program still held
  memory takes the room when it comes free, and gives records up when another program
  needs a tenth of it back.
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

Since then (2026-10-08), through the server on two RTX 5090s with 152 GB of RAM for
experts and the same drive: the experts are read into RAM and onto the second card
while the server idles (130 s after it loads), and a 48-token reply then runs at 5.6
tokens a second on a first-time prompt and 6.9 to 7.7 on one read before; a 93-token
prompt takes 7.5 to 7.9 s and a 276-token one 19.6 to 19.8. The first-time figure is
after a pause of 30 s in which the cards took the experts the first requests had used
(798 of them); with requests back to back from the start it was 4.4 to 4.9. The CUDA
build on this drive, both cards too, gave 6.4 to 7.3 and 7.9 to 10.5 tokens a second,
and 5.1 to 6.1 s and 18.2 to 18.9 s. On one RTX 5090 (`--devices 0`: the trunk and
960 experts on it, no second card's share) the same replies run at 4.8 to 5.2 tokens a
second first-time and 5.4 to 6.1 read before, the prompts in 8.3 to 8.8 s and 21.5 to
22.0 s: the same code, with more of a step's experts on the CPU. With a usage profile
(here the one left by a run of the same eight requests, so as good as a profile gets:
another conversation's experts overlap these less) the next start reads its first
93-token prompt in 7.8 s where 42.7, and its replies run at 5.9 to 8.4 tokens a
second the first time through and 7.1 to 9.0 the second on two cards (3.0 to 5.6 and
6.8 to 7.8 without a profile); on one card the first prompt in 9.1 s where 50.9 and
replies at 4.9 to 6.0 and 5.3 to 7.0 (2.6 to 5.1 and 5.4 to 5.9 without). What did it, in the order of what each was worth: the experts each in one
tier and the cards rebalanced while idle (above); a decode step's dense calls through
buffers the device keeps (`dense::Arena`: one write of the inputs, one of the
parameters, kept bind groups, one read-back polled for, where each call made and freed
some ten buffers and groups); the CPU's AVX-512 expert kernel and a prompt's experts a
worker each; the sparse attention's heads on workers that stay (`dsv41::pool`); the
hyper-connections' 24 projections summed side by side; an expert's step and the shared
expert each one submit. A step on a prompt read before is 130 to 146 ms: the trunk's
228 dense calls 40, the experts 50 to 57 (the drive 22 to 29 for the one or two of its
240 that are in no tier, the CPU's 26 to 28, the cards' beside them), sparse attention
11, mixing 6. `OAIY_DSV41_PROFILE` prints each prompt's and reply's.

## EXL3 (OrcaSAQ)

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

OrcaSAQ-2-27B through the server on the RTX 5090 (2026-10-05): loaded in
19 s with 10.7 GB of EXL3 weights on the GPU; a 162-token prompt in 21.7 s and 3.9
tokens a second, against 17 s and 4.5 for Qwen3.8 27B Q4_K_M on the same path (the
rest is the host's share of every model then); tool calls made and answered.

Since then (2026-10-08) it runs chained on the device as the GGUF models do (a decode
step or a prompt's chunk recorded once and submitted whole, the transforms and the
roundings in kernels of their own): on one RTX 5090 a reply at 54 to 55 tokens a
second, a 1,960-token prompt in 1.4 s and a 7,720-token one in 5.5 to 6.9 s. The CUDA
build gave 54 to 59 tokens a second, 5.1 s and 24.4 s. The one-row and few-row
kernels are written for a tile's rate (`exl3-mm-48` and so on: a lane shifts its words
to its first code once a tile and each code is a shift by a constant, its value a read
of a table the workgroup makes), which took a decode step's 400 matrices from 21.9 ms
to 10.2. Its PEFT adapters run beside the projections they adapt
(`quant_linear::LowRank`), and its vision tower is chained too (a new picture answered
in 2.4 s; a picture prompt takes positions in three axes).

## Qwen3.8-Flash-Next

Its 512 experts a layer (and the shared one) are `ggml_rs_wgpu::exl3::Exl3MoeHost`:
each projection on the GPU while the budget holds it, the rest decoded on the CPU
(through a 65,536-entry table of mul1's values, where decoding was most of their
time), the routing on the host exactly as the CUDA kernel routed. A layer's experts
run as two batches (every gate and up, then every down), the GPU's recorded in one
encoder and read back with one submit, the CPU's an expert a thread meanwhile. The
attention, delta-net and head matrices keep their share of the budget
(`flashnext::dense_exl3_bytes`, which leaves out the n-gram table: its rows are
trellis-quantized too, but it is read from the disk); the sigmoid-gated delta-net
step runs on the host, a head a thread, checked against the CUDA kernel while there was one; the
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
left the experts none of the budget. (That was one card on that date: its layers
split over two cards since, and it reads a GGUF of the same architecture too.)

Since then (2026-10-08), its EXL3 checkpoint chained over two RTX 5090s: a reply at 68
to 69 tokens a second, 66 to 83 with its drafting head; a 1,960-token prompt in 0.54
to 0.72 s. Its PEFT adapters run on its dense projections (an adapter of its routed
experts is refused), and its vision tower is chained (a new picture answered in
1.2 s).

It also reads the GGUFs Strata runs (ISTA-DASLab's GSQ-RCO files: Q2_0 experts with
K-quant, IQ4 and f16 matrices beside them; `quant_moe`, `quant_linear`). From the Q2_0
file over two RTX 5090s (2026-10-08): a reply at 84 to 88 tokens a second, 88 to 96
drafting with the EXL3 checkpoint's prediction layer (`--mtp-from`), a 1,960-token
prompt in 0.50 to 0.56 s. Strata's published figure for the model on one RTX 5090
(its IQ2_XS file, drafting four deep with suffix drafts, the card at its full 575 W;
ours are capped at 400 W) is 179 tokens a second and 4,270 tokens a second of prompt
at 4K: this file's prompt is within a tenth of that on two cards, the reply is half. A
decode step there takes 10.9 ms (10.6 of them the GPU's kernels), a check of four
drafted rows 19.6, and a round of drafting adds some 4 ms for the drafts themselves.

The IQ2_XS file is the one Strata's figure is for. Its routed experts' gate and up
matrices are grid types (IQ2_S in 34 layers, IQ2_XXS in 11, IQ1_M in 3; their down
matrices Q2_0): a group of eight weights is one entry of ggml's grid, with its signs and
a scale. `quant_moe` decodes them on the card as it does Q2_0, the grid in a storage
buffer of the layer's, two bits a weight (as WGSL constants such tables are copied at
each call, which once reset the driver). Before these kernels every layer's experts ran
on the host's reference path: 0.9 tokens a second.

From that file over two RTX 5090s (2026-10-08): a reply at 79 to 84 tokens a second,
93 to 103 drafting. A decode step takes 11.5 ms; a check of drafts (every row's logits)
14.2, 16.2 and 18.6 ms for two, three and four rows. A check's IQ4_XS rows go through
an int8 kernel (`shaders::iq4_xs_few_q8`: each 32 of a row rounded to int8 by its own
scale, which is llama.cpp's CPU arithmetic and what the K-quants' check kernels here
do): 2.1 ms of a check of four where the f32 kernel took 7.8, more than a step's whole
2.5. A check's rows then differ from a step's by that rounding: over 36 rows the logits'
cosine is 0.99967 on average and 0.9987 at worst. So a drafted reply is not always the
plain one, token for token, even at temperature 0: where two tokens are nearly tied the
rounding can pick the other. Of two greedy replies to Strata's request one was the
plain reply and one left it at its eighth token (both read as answers). `OAIY_NO_Q8`
gives a check's rows in f32, a step's exactly, and then both were the plain replies to
the byte, at 93 to 95 tokens a second where 108 to 119. (The K-quants' check kernels
are int8 the same way, so this holds for every GGUF model that drafts; an EXL3
checkpoint's checks are a step's bit for bit.)

That was so until a Flash-Next step took its checks' kernels (`ChainRecorder::
rows_alike`, which its chain sets): a step's row of an IQ4_XS matrix from int8
activations as a check's rows are (llama.cpp's and Strata's steps take them so), and of
a K-quant's by the several-rows kernel. A check's rows are now its steps' bit for bit
from a GGUF too (22 of 22 rows from the IQ2_XS file and from the Q2_0 one, which the
model's test holds them to), so a reply is the same drafted or not; and the step is the
faster for it, its IQ4_XS matrices 1.4 ms where 2.5 (a step 10.3 ms where 10.8 at
Strata's context). The steps moved by that rounding instead: against the host's path
their logits' cosine is 0.9984 at worst where 0.9989. The 27B's chain does not ask for
it: its matrices are K-quants', and their several-rows kernel with one row is slower
than their one-row f32 one (62 tokens a second where 67.5), and the one-row int8 kernel
sums in another order than a check's; so a drafted 27B reply can still leave the plain
one at a near-tie.

Against the request Strata's benchmark makes (a synthetic Python module up to 4,096
prompt tokens, greedy, 256 tokens), the same two cards: the prompt at 3,600 tokens a
second (Strata 4,270), the reply at 78 to 81 tokens a second plain and 108 to 120
drafting (Strata 179; ours 118 but for the first reply after loading). Strata's run is
one card at 575 W; ours two at 400 W. The two logs count the same work for it: Strata's
256 tokens are 96 passes with 160 of 226 drafts taken, ours 102 with 153 of 210. What
differs is a pass's time: 21 ms here (a check of 3.4 rows 17.5 ms, its drafts 3.6 ms:
the server's decode line says these) where Strata's is 14.9 all told. A check's time is
its kernels': of a check of four rows' 18 ms the f16 matrices are 4.5 (the
hyper-connections' are f16 in the file; the IQ3_S ones are held so), the experts 4.7
with their routing, the IQ4_XS matrices 2.6, attention 2.3.

Since then two of a layer's kernels are one where they can be: the shared expert's gate
and up matrices are one matrix (a matmul and a split SwiGLU where two matmuls and a
SwiGLU), and a hyper-connection's down projection makes its gates and its up projection
its mix where each would store its sums (bit for bit the two kernels' values, which a
test holds them to; `OAIY_HC_UNFUSED` for the two kernels). A step is 1,257 dispatches
where 1,498 and takes 10.5 ms where 11.2 in the same build; a check of four rows, 1,636
where 1,877, is no faster that can be measured (17.0 ms), and Strata's request drafting
is 116 to 125 tokens a second. What a dispatch costs was measured for this
(`measure_the_dispatch_floor`): recording and encoding one is some 4 us of the CPU,
which a step does not wait for (the GPU is the slower of the two, and runs a piece
while the next is recorded: pieces of 32 to 128 dispatches give the same times); on
the GPU one that does next to nothing is 1 to 3 us. So a kernel's time is mostly its
own: its weights' bytes at some 1.4 TB a second (the widest f16 matrices reach that),
and the longest run of work any one of its threads has, which is what the small
kernels' 4 to 15 us are (the router's ranking is 129 comparisons a thread, a few rows'
sums are added up by one thread for all the rows in turn). Strata's source says the
same of itself: a window of rows there is one captured CUDA graph of more kernels than
this engine has dispatches, the dense quantised matrices against int8 activations for
a step as for a check, and only token ids read back.

A few rows' sums are each added up by a lane of its own (lane r row r's, in the order
one lane added every row's in turn: the same bits), in the f16 kernels, IQ4_XS's and
the experts'. At Strata's context (4,086 positions, past QSA's dense span, which is
what its request runs: profile with `FLASHNEXT_PAST=4096`) a check of four rows is
17.1 ms where 17.8 and of three 15.2 where 16.2, its IQ4_XS matrices 1.6 ms where 2.1;
Strata's request drafting 122 to 128 tokens a second (256 tokens in 2.00 to 2.09 s),
the replies the same to the byte. What an f16 kernel costs a dispatch is measured
apart from a model (`measure_a_few_rows_f16_matmuls`, every dispatch another copy of
the matrix): a hyper-connection's down matrix 7.8 us for one row and 10.8 for four, its
up matrix 6.8 and 7.0, a router 6.2 and 6.8; of each some 2.5 us is any dispatch's and
the rest mostly its bytes (6.6 MB a hyper-connection's matrix; a 31.5 MB one reads at
1.45 TB a second). The same matrix every time stays in the card's cache and takes 5.5,
3.9 and 4.0 us for a row: a measurement that reuses its weights flatters the kernel.

Past its dense span Flash-Next's attention (QSA: some 2,100 of a query's positions) took
a workgroup a query head, and its 24 query heads are over 2 KV heads: each key and each
value was read twelve times over, 163 us a layer for a check's four rows. A KV head's
query heads are now taken six at a time there, as the step's dense attention takes the
27B's six (the same kernel over QSA's entries; a KV head's twelve in two whole shares,
which the dense one does for Flash-Next too): 52 us a layer. A check of four rows is
15.5 ms where 17.1, of three 14.2 where 15.2, and Strata's request drafting 125 to 133
tokens a second (256 tokens in 1.93 to 2.05 s). The grouped kernel adds a run's values
up in another order than the one a head had (2e-5 of the largest value at most, which
its test holds it to), so a long greedy reply can part from the earlier one's at a
near-tie; with every block kept QSA's is the dense attention's bits, as before.

A layer's routing ranked the router's 513 logits in one workgroup, a thread some 260 to
390 turns of four comparisons: 14.6 us whatever the rows. Nine workgroups rank them now
(four threads a logit, 33 turns each) and write the down jobs by rank, and a second
kernel makes the weights a thread an expert: 6.7 us a layer for the two, the same
experts and weights. A step is 10.8 ms where 11.4 and a check of four rows 15.0 where
15.5.

A request that samples greedily (temperature 0) has its tokens picked on the GPU
(`argmax_rows`: each row's first largest logit, as the host's greedy sampling takes it;
`FlashNext::check_picks` and `step_pick`). A check read every row's logits back, a
megabyte a row, to pick a token from each: a check of two, three and four rows is 1.0
to 1.3 ms the shorter for picking them there (the same tokens, which the model's test
holds them to). Strata's request drafting: 256 tokens in 1.75 to 1.86 s, 138 to 146
tokens a second (Strata 179), where 1.93 to 2.05 s; of it the checks 1.23 to 1.32 s
(13.9 ms each of 3.4 rows), drafting 0.31 s, undoing 0.07 s. A request with a
temperature reads the logits as before.

Undoing a check's refused rows reads nothing back, so it is sent to the two cards and
not waited for (`ChainRecorder::send`; each card's next work is behind it on its
queue): 0.04 s of the request where 0.07, 256 tokens in 1.71 to 1.84 s (139 to 150
tokens a second).

A draft's head takes the int8 kernel too (a GGUF's head was half a draft's GPU time in
f32): a round of three drafts 2.4 ms where 3.0, drafting 0.25 to 0.27 s of the request,
256 tokens in 1.67 to 1.82 s (141 to 153 tokens a second; Strata 179). What the reply's
time is then, by kernel (`OAIY_CHAIN_PROFILE` in the server prints it at a request's
end; 1,613 ms of the GPU's, which is the whole reply): the IQ4_XS matrices 242 ms and
their int8 rows 53, the experts 281, the f16 matrices 2,560 wide 172, the
hyper-connections 214, QSA's attention 70 and its selection 28, the stream norms 67,
the delta nets 61, and 39 of copies (a check's backups of the delta nets' states).

On one card, which is what Strata's figure is for, this file is far from it: its
experts are 35.5 GB, so 15 of the 48 layers' run on the host (`quant_host` reads the
grid types as they lie too: 0.27 to 0.42 ms an expert a row on one AVX-512 core, where
the reference's dequantising took 18 ms a layer). One RTX 5090 at 400 W: a reply at 43
to 45 tokens a second, Strata's request at 43 (no drafting there), and its 4,086-token
prompt in 24 s, 170 tokens a second, because a prompt's every row costs the host
layers' experts a row each (146 ms a host layer for a chunk of 512). Strata keeps each
layer's most used experts on the card (71% of them fit, 98 to 99.7% of its lookups hit)
and computes the few others on the host; a whole layer on or off the card, as here,
puts a third of the model's experts on the host for every token. Experts held one by
one are what one card needs next, and what that would give was measured first, from
the model's own routing (`OAIY_HOST_ROUTE_LOG`, with `OAIY_EXPERTS_ON_HOST` to put
every layer's on the host; a prose request and Strata's):

- Within a request the routing is narrow: the most used half of the experts take 96 to
  99.8% of its lookups, and a reply never touches 42 to 48% of them. But which half
  depends on the text: a card holding the 69% another request used most hits 70 to 76%
  of this one's lookups, hardly more than any 69% would.
- So the card must take in what it misses. Doing that (the least recently used put out)
  a card of 69% hits 97.1% of the reply's lookups to Strata's request and 98.2 to 98.6%
  of the prose one's: Strata's own log says 97.8 to 99.7%, so that is how it works too.
- A miss still costs its layer a round trip to the host here, and they are spread: a
  check of 3.4 rows has 48 misses in 23 of its 48 layers, a step 15 in 12 (14 and 6 on
  prose); at 85% of the experts held, 12 layers a check. So for a reply it would be
  some 12 to 23 round trips a pass where a whole layer on the host is 15, each with a
  tenth of the host's work: a gain for one card, and no more than that. With no miss
  at all one card would run as two do now, and two are at 21 ms a pass where Strata's
  is 14.9: what separates this engine from Strata's figure is a pass's time on the
  card, not where the experts are. For a prompt it is the whole difference: 96 to 98%
  of a chunk's lookups would stay on the card where a third of them are the host's
  now.

A recording's reads are polled for 20 ms before the thread waits for them
(`OAIY_CHAIN_SPIN_MS`), as a CUDA program's are by default: a step, a check and a host
layer's round trip end within that, and each was some 0.1 ms the longer for parking
the thread (2 to 3% of a reply).

On one card (`--devices 0`) the file does not fit: its experts are 34 GB, and a 32 GB
card's budget holds 34 of the 48 layers' beside the dense matrices. The other 14
layers' routed experts run on the host between the chain's submits (`quant_host`:
each Q2_0 matrix read as it lies in the file, 0.13 ms an expert a row on one AVX-512
core where dequantising it whole took 18 ms a layer), their shared expert on the card.
So one RTX 5090: a reply at 51 to 54 tokens a second, a 160-token prompt in 0.6 s
and a 1,960-token one in 5.6 to 5.8 s (a prompt's rows cost the host's experts a row
each: 95 ms a host layer for a chunk of 512). It does not draft there (a check's rows
cost the host as much as the steps they save, measured one to three deep), and the
prediction layer's room goes to experts. Before, a model with any layer's experts off
its cards was not chained at all (160 ms a step and more). Strata on one card keeps
the experts it uses most on the card and computes the rest on the host a layer at a
time, which CUDA can wait for cheaply; a WebGPU wait costs a submit and a read-back,
so here a layer's experts are all on the card or all on the host, and the 14 host
layers are some 8 ms of a step's 19.
Image and video generation (`oaiy-media`) run on WebGPU too: each model's own page
says what of it does.

## GLM-5.3-Flash

Its GGUF (`glm5next`) streams its experts on the CPU, from RAM and the drive (the
CUDA build streamed them to the cards: that tier is not on WebGPU yet), and puts its
dense layers on the WebGPU adapter.
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

Through the server the 9B loads in 4.1 s and answers chat requests.
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
| Llama 3.2 3B | 35 tok/s (11.6 at 512 tokens of context before the changes above) | 128 tok/s with the wide K-quant kernels (73-81 before them) | 1.5 s (59.2 before the day's changes, 8.4 op by op) |
| Qwen3 0.6B | 53 tok/s | 182 tok/s | 0.8 s through the server |
| Gemma 3 4B | 28.5 tok/s | 60 tok/s | |

Each chained model gives the op-by-op path's 64 greedy tokens after 64- and
1,500-token prompts (past Gemma 3's 1,024-token window), the prompt's logits and
every step's cosine 1.000000, and the same two-turn conversation word for word
(Gemma 3's second turn reusing 130 tokens of the first's state). A steady Llama 3.2 3B step
was 0.4 ms of recording and 14 ms on the GPU, the one-row matmul reading its
weights at 170-280 GB/s however its lanes were laid out; the K-quants' wide
kernels (above) made it 7.7 ms.

## Qwen3.5 and Qwen3.8 27B on the GPU (2026-10-05)

Qwen3.5's hybrid (Qwen3.5 4B, 9B and 27B, Qwen3.8 27B: `llama-rs`'s
`chain_qwen35`) runs a decode step or a prompt's chunk in one submit too. Its
gated delta net layers' projections, causal conv and recurrence run on the GPU:
a workgroup a value head, a thread a row of its state, the row held in registers
through the run's tokens (43 us a layer for a step of Qwen3.8 27B when each
token read it from memory, 11 us held), then the head's norm and its `silu(z)`
gate. Its attention layers' q and gate halves, per-head norms, partial RoPE (64
of 256 dims), the K and V into the GPU's copy of the cache, attention, the
sigmoid gate; every layer's FFN (a gate and up of two types too) and residuals;
then the head. Qwen3.8 27B's beta-alpha projection is f32 and has a kernel of its
own. A Qwen3.5 GGUF with no output weight (the 4B's) is headed by its packed
token table, on the GPU.

The recurrent state stays on the GPU between steps: the cache's state tensors
are the chain's own buffers (`DeviceChain::alias`), so what reads them there (a
checkpoint, a conversation set aside, a disk state) reads them back (Qwen3.8
27B's 157 MB in 42 ms), and a state the host puts there (a restore, or the host
path's, which a prompt with an image still takes) is taken up at the next run.

On the RTX 5090 (Vulkan), greedy, through `oaiy-llm-server-webgpu`:

| Model | Decode, op by op | Decode, one submit | A prompt |
|---|---:|---:|---:|
| Qwen3.8 27B Q3_K_M | 6.0 tok/s (2.0 with the old 8 GiB budget) | 57 tok/s | 2,011 tokens in 5.4 s; a short turn in 0.4-0.7 s |
| Qwen3.5 9B Q4_K_M | 9.2 tok/s | 124 tok/s | a chunk of 512 tokens in 0.4 s |
| Qwen3.5 4B Q4_K_M | 9 tok/s | 169 tok/s | a chunk of 512 tokens in 0.24 s |

Each gives the host path's tokens after 64- and 1,500-token prompts, the prompt's
logits and every step's cosine 1.000000 (`a_chained_qwen35_run_answers_as_the_host_path`,
which also checks that the chain ran); the same replies through a conversation
that goes back to a checkpoint as with `OAIY_NO_CHAIN`; and Qwen3.8 27B the right
answer after up to 2,011 tokens. Its decode step is about 17 ms on the GPU, 12 of
them the matmuls (Q3_K, 8 GB of its 13.4, at 1,090 GB/s).

Against llama.cpp in one session (2026-10-09; LM Studio's CUDA build of its server and
this engine's in turn, twice over, the same card at its 400 W cap, Qwen3.8 27B Q3_K_M,
the same 15.6K-token prompt, greedy): llama.cpp wrote at 68.6, 68.6, 67.9 and 68.1
tokens a second and read the prompt at 2,416 to 2,636; this engine wrote at 68.5, 68.3,
68.0 and 66.7 after its first reply, and read the prompt in 6.44 to 6.47 s (2,410
tokens a second) after its first. So a reply is llama.cpp's speed and a prompt 92 to
100% of it. A server's first request is the slower one, here as there: the first
prompt 7.2 s (llama.cpp's first 2,416 where 2,636 after), the first reply 64.2 tokens
a second. That is not its kernels being made: all 52 pipelines take 89 ms together
(`OAIY_PIPELINE_LOG` says each), and a run of the model at load (`QwenEngine::warm_up`:
two chunks, a step, a model that drafts its drafts and checks too; half a second,
`OAIY_NO_WARMUP` to skip it) leaves the first request as slow. By fifties of tokens
(`OAIY_DECODE_LOG`) the first reply is 0.74 s a fifty where the later ones are 0.72,
but for one fifty of 0.95: one stall of a fifth of a second, once a server's life,
whose cause is not found.

Gemma 3 itself was wrong on every backend until this day: one RoPE base on every
layer, where its sliding-window layers take 10,000 and its global ones
`rope.freq_base` with the GGUF's linear scaling, and its SentencePiece tokens
were not llama.cpp's (a space before every line, merges out of score order,
newlines dropped from replies). Gemma 3 4B answered a markdown prompt of a few
hundred tokens with fragments of it; it answers as llama.cpp does now.
