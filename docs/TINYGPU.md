# WebGPU on a Mac's eGPU, through tinygrad

macOS has no driver for an NVIDIA card in a Thunderbolt enclosure, so no WebGPU adapter sees one: wgpu's on a Mac is
Metal, and Metal knows the Mac's own GPU alone. tinygrad drives such a card itself (its TinyGPU app and driver
extension), and OAIY already sends a model's chats to tinygrad's own server there ([the Mac](MAC.md), "A card in a
Thunderbolt enclosure"). This is the other way: **the card as a WebGPU adapter**, so a program written against wgpu's
compute (OAIY's own engine among them) runs its kernels on it as they are.

Four parts, each its own:

- **`crates/wgsl-cuda`**: a WGSL compute kernel as CUDA C++. naga parses and validates the WGSL as wgpu does; its
  module is walked into one `extern "C" __global__` kernel beside a prelude of WGSL's types and built-ins
  (`src/prelude.cuh`): a vector as WGSL lays it out, its operators WGSL's (shifts by the amount modulo the width,
  comparisons a vector of bool), `dot4I8Packed` as `__dp4a`, workgroup memory `__shared__` and zeroed as WebGPU does,
  every expression named where naga emits it. An index stays in its array, as WebGPU's robust access has it (naga's
  `Restrict`: clamped), a runtime-sized array's length from its binding's size, which the kernel takes after its
  pointers (so `arrayLength` is there too): an index past a buffer's end reads the card's memory past it, and a
  fault there costs the card its link, where Vulkan and Metal give the kernel a value of the buffer's. WGSL's
  cooperative matrices (16 x 16, f16's into f32 or f16 sums) are CUDA's `wmma` fragments, on the tensor cores
  (`src/coop.cuh`, in such a kernel alone): `coopLoadT` and `coopLoad` row- and column-major memory, a stride in the
  pointer's own elements (a `vec4<f16>`'s four halves each), as SPIR-V's is. Each load and store is told what its
  array has from the fragment's start: a fragment that fits is read and written in place, and one that would reach
  past the array (the last rows of a matrix whose buffer is not padded to them) goes through 256 elements of its
  warp's own in a scratch the server gives such a kernel (1 KB a warp, 64 warps on each of 160 SMs), an element past
  the array read as 0 and not written. `wgsl-cuda IN.wgsl OUT.cu` writes one and says its workgroup size and
  bindings.
- **`tools/tinygpu/webgpu_server.py`**: one process that holds the card through tinygrad and does what WebGPU's compute
  needs, over a Unix socket: buffers (zeroed), writes and reads, kernels (CUDA compiled by nvcc, kept by the source's
  hash), and submissions of dispatches, copies and clears, run in order on the card's compute queue. A submission's
  kernels go to the card as one queue of tinygrad's (each launched as the one before it ends, as tinygrad's own graphs
  chain them), and a write goes by the copy queue, which waits for what was submitted before it: neither waits for the
  card. Clients connect at once, their requests one at a time; a client's buffers are freed as it goes.
- **`crates/wgpu-tinygpu`**: a `wgpu::Adapter` (wgpu's `custom` backend) that asks that server. A compute pipeline's
  WGSL becomes CUDA at creation; a command buffer's work goes in one message at `Queue::submit`. Compute alone: render
  pipelines, textures, samplers and queries are refused.
- **`crates/webgpu-tinygpu`**: `webgpu.h`, WebGPU's standard C API, over that adapter: `libwebgpu_tinygpu.dylib`, with
  wgpu-native's ABI (its headers are in `include/`), so a program written against the header, in any language, runs
  its compute on the card. Python's `wgpu` takes it as it is:

  ```sh
  cargo build --release -p webgpu-tinygpu
  pip install wgpu numpy
  WGPU_LIB_PATH=$PWD/target/release/libwebgpu_tinygpu.dylib python tools/tinygpu/wgpu_py_smoke.py
  ```

  and C as `examples/compute.c` shows (`-lwebgpu_tinygpu`). `src/webgpu.c` reads the header's structs (the compiler
  lays them out from the header itself) and keeps references and futures; every operation is done as it is asked, so a
  callback fires at once (AllowSpontaneous), at the next `wgpuInstanceProcessEvents` or `wgpuDevicePoll`, or in
  `wgpuInstanceWaitAny`; errors go to the innermost error scope of their kind, else the uncaptured-error callback.
  Every function of the header that is not compute is there, and says so on stderr. `WEBGPU_TINYGPU_BACKEND=native`
  takes wgpu's own adapter instead (Metal on a Mac), to check a program against both.

OAIY's engine takes it with the feature `tinygpu` (`oaiy-llm-server`, `ggml-rs-wgpu`; every Mac build of the engine
has it) and `OAIY_WEBGPU_ADAPTER=tinygpu` (or `tinygpu:<socket>`).

**In OAIY**, Settings → eGPU → *Engine on the card*: **OAIY's own, over WebGPU** (`llm.egpu.engine: "webgpu"`,
[MAC.md](MAC.md)). A chat with a model ticked *On the eGPU* then starts `egpu_serve.py --webgpu`, which holds the card
in-process (this server, written beside it) and runs the app's own `oaiy-llm-server-webgpu` on it with the model; when
OAIY stops it or goes, the launcher's lifeline stops the engine and Python's exit releases the card.
The Agent's page chooses the model on the card from its header (its **eGPU** menu), and the model the engine runs
from the menu beside it.

## Without the card: the emulator

`webgpu_server.py --emulate` is the same server with no card: each kernel's CUDA is compiled for the CPU (clang++, with
`tools/tinygpu/emu/cuda_emu.h` before it) and run there, its workgroups shared out among the cores. CUDA's words are the
CPU's (`__half` is `_Float16`, atomics the compiler's); a workgroup runs on one thread, each invocation a fiber of its
own stack, so `__syncthreads()` passes to the next invocation and the workgroup goes on once every one has reached it;
workgroup memory is the thread's; a tensor core's fragment (`emu/mma.h`) is its whole matrix, every invocation doing
its warp's op. It needs python3 and clang++ alone (no tinygrad), and it is how a translation is
checked before it goes to the card, where a kernel that faults costs the card its link (below):

```sh
python3 tools/tinygpu/webgpu_server.py --emulate        # listens on ~/.cache/tinygpu-webgpu/emulator.sock
OAIY_WEBGPU_ADAPTER=tinygpu:$HOME/.cache/tinygpu-webgpu/emulator.sock \
  cargo test --release -p ggml-rs-wgpu --features tinygpu --lib -- --test-threads=1
```

`cargo test -p wgpu-tinygpu` starts one itself (`tests/emulator.rs`) and checks WGSL's meaning through the translation:
an integer divided by zero, shifts past the width, workgroup memory zeroed and shared across a barrier, atomics,
`workgroupUniformLoad`, a struct's members where WGSL lays them out, a float made an integer, a buffer kept while work
holds it.

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

The card's firmware region (WPR2) is up after any process that held the card, even one that released it at tinygrad's
own exit (read on 10 October 2026, with the card's registers read and nothing set up), so every open of the card begins
with a full reset. Over Thunderbolt that reset has come back after a clean exit and after a killed process, and has
twice not come back after a kernel faulted the card: "Booter failed to execute, mailbox is ffffffff", the PCI link down
until the enclosure was powered off and on (`system_profiler SPPCIDataType` says "Link down"). So a kernel is checked on
the emulator first (below), and the server still ends by releasing the card (SHUTDOWN, SIGTERM, SIGHUP or an
interrupt), as OAIY's launcher does (`egpu_serve.py`, which also restores Python's interrupt handler: a launcher
started from a shell in the background had SIGINT ignored, its lifeline's interrupt did nothing, and its last resort
skipped tinygrad's exit).

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

On the emulator (the same day, the card off its link):

- every one of OAIY's GPU tests passes on the adapter: 98, each kernel translated and run as CUDA (the 36 ignored are
  timings, as they are on any GPU);
- a model answers through the whole engine: Qwen3-0.6B (Q4_K_M) says "The capital of France is Paris.", as it does on
  Metal, finds "marigold" at the start of a 946-token prompt, and writes three sentences about the ocean, at about two
  tokens a second;
- all 309 kernels, translated again, compile with nvcc for sm_89, each a cubin of one kernel.

Through `webgpu.h`, on the emulator and on Metal alike: Python's `wgpu` 0.32 (its own build of wgpu-native swapped
for this library) runs a kernel with workgroup memory and a barrier over 1,048,576 values, its `compute_with_buffers`,
a write, a clear, a copy at offsets, a buffer mapped at creation and one mapped for reading, and raises
`GPUValidationError` for a kernel that is not WGSL, the device going on after; `examples/compute.c` waits for its
adapter, device and map with futures and counts 4096 numbers' Collatz steps right.

Two translations the emulator found wrong, now right: an integer's `/` and `%` by zero (WGSL's are the dividend and 0;
CUDA's are anything), and a struct whose member follows a `vec3` in its last 4 bytes (a struct's `vec3` is now its 12
bytes). And one of the adapter's: a buffer whose handle was dropped while a bind group still held it was freed before
the work ran (a buffer now lives while a bind group or an unsubmitted command buffer holds it, as wgpu's own do).

The emulator's buffers end where a page that cannot be read begins, so a kernel that reads or writes past a buffer's
end faults there as it would fault the card: none of OAIY's does, over the tests' 6,214 dispatches and a model's 8,000
more.

On the card, its link back up (the same day):

- `tests/emulator.rs` (with `TINYGPU_TEST_SOCKET` at the card's server) and every one of OAIY's GPU tests pass: 98;
- a model through OAIY's own engine (`oaiy-llm-server-webgpu`, built with `--features tinygpu`,
  `OAIY_WEBGPU_ADAPTER=tinygpu`), its weights all on the card:

  | model | load | 946-token prompt | writing | answers |
  |---|---|---|---|---|
  | Qwen3-0.6B Q4_K_M | 1.1 s | 0.5 s | 60 tokens/s | Paris, "marigold" |
  | Qwen3.8-27B Q4_K_M (15.8 GB) | 64 s | 2.2 s | 30 tokens/s | Paris, "marigold" |

  The same 27B model is 2.5 tokens a second on the Mac's own GPU (it does not all fit) and about 2 through tinygrad's
  own LLM server on this card;
- Python's `wgpu` and `examples/compute.c` through `webgpu.h`, as on the emulator;
- through OAIY 0.1.3-mac.8, the eGPU's engine set to WebGPU: a chat with Qwen3.8-27B through the gateway started the
  launcher (252 s: the card's reset, the 16 GB file read from an external drive and sent over the link), answered
  "The capital of France is Paris.", then wrote three sentences about the ocean at 19.4 tokens a second end to end;
  Stop let the engine and the card go within a second;
- what the adapter's own work costs (`cargo run --release -p wgpu-tinygpu --example costs`): a dispatch 13.7 us in a
  submission of many (45 us alone, a submission's own cost with it); writes at 1.1 to 1.4 GB/s and reads at 0.95 GB/s,
  over Thunderbolt 3.

With the tensor cores (the adapter's cooperative matrices, `TINYGPU_NO_COOP` for none):

- `tests/emulator.rs`'s cooperative matrices on the card's own tensor cores: row- and column-major loads and stores,
  a `vec4<f16>` array's stride, f32 and f16 sums, exact; and every one of OAIY's GPU tests, 98 (36 of their 380
  kernels on the tensor cores), all 380 compiled by nvcc for sm_89;
- Qwen3.8-27B reads the 946-token prompt in 0.80 s (2.23 s without: 1,180 tokens a second, not 425), and writes
  as before, about 28 tokens a second; through OAIY 0.1.3-mac.9's gateway (its eGPU's engine WebGPU), "marigold" out
  of a 4,726-token prompt in 6.3 s, and 20.8 tokens a second end to end.

Qwen-Image 2.1 (`oaiy-media`, which takes the adapter as the LLM engine does) **faulted the card** on the tensor cores
(its GGUF's blocks kept quantized, 4.5 GB): an MMU fault, a write where no memory is mapped, found as its transformer
loaded after the prompt was encoded. Its whole pipeline had run on the emulator without the tensor cores; with them
(the card's very request, 512 x 512) the emulator encoded the prompt and loaded the transformer without a fault and
with no fragment past its array (its tensor cores are too slow to go further: each invocation does its warp's whole
product), so the cause is not known (with the guards below, the emulator stages no fragment there either: none is past its array
or misaligned). Without the tensor cores (`TINYGPU_NO_COOP=1`, its transformer f16, 14 GB) it runs on the card, after
the card's reset (which came back): "A red apple on a wooden table, soft morning light, photograph" at 512 x 512 in
20 steps, 1.5 s each (the image 35 s, 82 s with the models' loading and the prompt's encoding), and at 1024 x 1024 in
30 steps, 7.1 s each (220 s; 290 s in all), the pictures as asked:

```sh
TINYGPU_NO_COOP=1 OAIY_WEBGPU_ADAPTER=tinygpu target/release/oaiy-media --request request.json
# {"base": "/Volumes/T9/Qwen-Image-2.1", "transformer": "/Volumes/T9/qwen-image-2.1-UC-Q4_K_M.gguf", "output_dir": "...",
#  "prompt": "...", "width": 1024, "height": 1024, "steps": 30}
```

**Qwen-Image-2.1-Turbo** (the distilled checkpoint: 8 steps at CFG 1, its own sigmas; AtomicChat's GGUFs, the
denoiser `Qwen-Image-2.1-Turbo-AD-Q4_K.gguf` and the abliterated text encoder
`Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf`, the base model's folder for the VAE and tokenizer) on the
card, on its tensor cores, the guards below in place (and the run that faulted run again, clean):

| 1024 x 1024 | sampling | in all |
|---|---|---|
| Qwen-Image 2.1, 30 steps at CFG 6, no tensor cores | 215 s | 290 s |
| Turbo, 8 steps, no tensor cores | 28 s | 57 s |
| Turbo, the tensor cores | 8.0 s | 38 s |
| Turbo, the tensor cores, the text encoder's blocks kept quantized | 8.0 s | 22.7 s |

```sh
OAIY_WEBGPU_ADAPTER=tinygpu target/release/oaiy-media --request request.json
# {"base": "/Volumes/T9/Qwen-Image-2.1", "transformer": ".../Qwen-Image-2.1-Turbo-AD-Q4_K.gguf",
#  "text_encoder": ".../Qwen-Image-2.1-Turbo-Abliterated-Uncensored-AD-Q4_K.gguf", "output_dir": "...",
#  "prompt": "...", "width": 1024, "height": 1024}
```

A transformer whose file is a Turbo one (or `"distilled": true`) samples Turbo's schedule (`schedule::DISTILLED`); a
GGUF text encoder in llama.cpp's names (`blk.N.attn_q`, as stable-diffusion.cpp's `--llm` takes it) is read as the
Hugging Face one is, and its K-quant blocks go to the card as they are (a third of the f16's bytes over the link, and
nothing unpacked on the CPU: the encoding 5.3 s, from 18); a GGUF's fused `img_mlp.gate_up` is split into its gate and
up halves by rows.

**Pictures in OAIY on the card.** With the eGPU's engine OAIY's own, a Qwen Image picture is made on the card
(`llm.egpu.images`, on unless switched off: Settings → eGPU → *Make pictures on it too*). The card is not opened a
second time (each open resets it): the launcher that holds it for the language model pauses that engine on a word from
OAIY (its buffers freed as its connection goes), and the image worker (`oaiy-media`, which must be in the engines'
folder or beside the app: no installer carries it) takes the card's socket as its WebGPU adapter. Chats with the model
wait meanwhile; when the queue of pictures is empty the card is given back and the engine starts again, or, where the
card was opened only for pictures, stays paused until a chat asks for its model (and is let go with the eGPU's idle
stop). The worker is kept (`oaiy-media --serve`, a job a line in and its answer a line out) with the models it loaded:
the next picture that names the same files is spared their 5 s of loading. It waits for another picture for two
minutes, the card lent meanwhile, and is let go at once when a chat waits for the card's model (and when the card is no
longer lent). A picture the card fails is made on the Mac. The engines' files (`detect.rs`) know Turbo's GGUFs: the
transformer is marked `distilled` (8 steps at CFG 1, no turbo adapter beside it) and the Qwen3-VL text encoder made for
it is attached as the model's text encoder, not added as a chat model. On the RTX 4090, through OAIY's gateway, a
1024 x 1024 Turbo picture:

| through OAIY 0.1.3-mac | the card already held | the card opened for it |
|---|---|---|
| mac.10 (the first) | 21.9 s | 27.7 s |
| mac.11 (writes streamed to the card, the prompt's embedding rows alone unpacked) | 17.5 s | 25.7 s |
| mac.12 (the socket's buffers 4 MB) | 13.7 s | 21.3 s |

Of the worker's own time, sampling is 8 s (8 steps of 1.0 s) and the VAE 1.4 s; the rest is the weights' way to the
card. That way was half the link's: a Mac gives a local socket 8 KB of buffer each way, so a write crossed it 8 KB a
call (1.36 GB/s); with 4 MB buffers (the adapter's and the server's) it is 2.42 GB/s, the link itself 2.6 (tinygrad's
staged copy from memory reaches it, as does its DMA alone). The text encoder's load and prompt then take 2.7 s (from
5.8) and the transformer's load 2.7 s (from 4.5). A large write is also taken in 16 MB pieces, each copied on to the
card as the next arrives, and the adapter sends a write's bytes without copying them into its request first.

`TINYGPU_PROFILE=1` has the server run each kernel alone and time it, and write the kernels' times, the most first, to
`~/.cache/tinygpu-webgpu/profile.txt` as each client goes (each kernel's CUDA beside it by its name). A Turbo picture's
steps are 49% the quantized matmul (2.4 ms a call, about 118 TFLOPS: 70% of the card's f16 with f32 sums) and 33% the
attention (OAIY's one pass, 10.3 ms a layer for 4,096 queries over 4,160 positions: about 27 TFLOPS). The attention
takes 32 queries a workgroup and its fragments straight from memory, so each head's keys and values are read 128 times;
it is where a step has most to gain. `OAIY_EGPU_EMULATE=1` in the eGPU's environment has OAIY's launcher hold the
emulator instead of the card (its tests run the launcher, the pause and the image worker's way in that way).

**A language model's tokens.** Qwen3.8-27B writes 28 tokens a second on the card, where its 16.5 GB of weights at the
card's 1 TB/s would allow about twice that. `TINYGPU_STATS=1` (each kind of request's count and time, the dispatches
submitted, the time between requests, every 5 s on stderr) and `TINYGPU_CPROFILE=FILE` (the server's Python profiled)
say where a token's 35 ms go: about 870 kernels in 7 submissions, each kernel 16.6 us of the server's Python (14 ms),
and 17 reads that wait for the card (14 ms); about 49 buffers made and freed besides. A kernel's 16.6 us are tinygrad's
launch: a view of memory made for each word it writes (8 million in a 200-token answer) and its QMD's fields set by
name. `TINYGPU_FAST=1` launches through `Fast` instead: a launch's arguments and its QMD written in a few copies, its
fields' places found once from tinygrad's own tables (about 6 us a kernel). Its bytes are tinygrad's:
`TINYGPU_FAST_CHECK=1` has tinygrad write each launch as well (in a queue never submitted) and compares them before the
card is given anything, a difference dropping the queue; checked so on the card, OAIY's 98 GPU tests, the adapter's
and a 200-token answer of the 27B were the same bytes, and `tools/tinygpu/fast_check.py` checks the same with no card
(a stand-in for tinygrad's NV device, thousands of random launches, chains and releases). It is not yet on by default:
its gain on the card is still to be measured. Submissions of a token are not byte for byte those of the token before
(a buffer made afresh each token is bound in some), so they cannot simply be replayed.

The adapter keeps the buffers a program lets go (up to 256 MB of them, each 64 MB or less; `TINYGPU_NO_POOL=1` for
none) for the next buffer of their size: the dozens a token makes and lets go were each two round trips to the server
and a launch of a clear there. A buffer from the pool is cleared before anything else reaches the server (WebGPU's
begin as zeros): at the head of the next submission, or in one of its own before a write or a read.

A process that holds the card and is ended by a signal it does not handle leaves the card's firmware up, and the next
open resets the card: on 2026-10-10 a server ended so (a SIGUSR1, which a server of older code had no handler for)
was followed by a reset that took the card's link down, until the enclosure was powered off and on. The server
(`ENDING`) and OAIY's launcher now take every such signal as a normal exit, the card released; the server's SIGUSR1
turns `TINYGPU_FAST_CHECK` on or off.

What could write past a buffer on the card and not on the emulator is now kept from it: a fragment not as
`wmma` takes it (32-byte aligned, its stride a multiple of 16 bytes; SPIR-V's and Metal's take any) is staged as one
past its array is, and the server refuses a copy, clear, write, read or binding past its buffer, as wgpu checks them
(a range far past a buffer's end jumps the emulator's page that cannot be read, and lands in the card's memory past
it). The server and the adapter say their protocol's version at HELLO (3), and a program of another is refused: an
older build would send a kernel's header short and the server read its first bytes as the source's. Each dispatch is
still a Python call (tinygrad's launch), batched to one doorbell a submission.
