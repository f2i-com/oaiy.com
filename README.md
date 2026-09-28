<div align="center">

# NROB

### **N**VMe · **R**AM · **O**n-GPU · **B**roker

**Your model doesn't fit on your GPU. nrob robs the NVMe for it.**

![Rust](https://img.shields.io/badge/rust-stable-orange?logo=rust)
![Core dependencies](https://img.shields.io/badge/core_dependencies-0-brightgreen)
![Unsafe](https://img.shields.io/badge/unsafe-forbidden_in_core-blue)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia)
![Weights](https://img.shields.io/badge/weights-GGUF%20%7C%20safetensors-purple)
![License](https://img.shields.io/badge/license-Apache--2.0%20%7C%20MIT-lightgrey)

</div>

nrob is a Rust inference engine for language models far bigger than your VRAM. The
weights stay on disk, in the files you downloaded: **GGUF** or **safetensors**, no
conversion step and no special format. For every token, nrob steals exactly the
experts it needs, stashes the hot ones in RAM and on the GPU, and brokers every move
between the three tiers so the next token finds its loot close at hand.

> **The big job:** DeepSeek-V4.1-Flash, a **510 GB, 552-billion-parameter** mixture of
> experts, decoding at **~29 tokens/s on one desktop** (Ryzen 9 9950X3D, 192 GB RAM,
> 2× RTX 5090), read straight from its safetensors.
> [How it was pulled off →](docs/DEEPSEEK_V41.md)

An opt-in [ternary-expert research backend](docs/TERNARY_EXPERTS.md) also supports
original MXFP4 experts during generated tool calls and local routing logs. It is
experimental; broad quality and paired speed comparisons remain unestablished.

## The heist

Every token is a small, well-planned job:

| | Tier | Role in the job |
|:-:|---|---|
| **N** | **NVMe** | *The vault.* Every weight lives here, in its original file. nrob only breaks in for what the current token needs: a few positioned reads per expert. |
| **R** | **RAM** | *The safe house.* Experts it has already lifted wait in an LFRU cache, so the next token doesn't have to go back to the vault. |
| **O** | **On-GPU** | *The getaway car.* The hottest experts ride in VRAM, and the always-needed trunk (attention, router, shared experts) never leaves it. |
| **B** | **Broker** | *The fence.* Decides what's hot, what gets promoted, and who does the work: the GPU, or the CPU straight from RAM when that beats the trip over PCIe. |

```
      NVMe (the vault)          RAM (the safe house)          GPU (the getaway car)
   ┌────────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
   │ .gguf/.safetensors │ ─▶ │ LFRU expert cache      │ ─▶ │ trunk + hot experts  │
   │ read in place      │    │ CPU computes misses    │    │ grouped MoE kernels  │
   └────────────────────┘    └────────────────────────┘    └──────────────────────┘
                 ▲                         the broker                  │
                 └──── usage counts, promotion, who computes what ─────┘
```

## Features

- **NROB Studio: a portable host for all of it.** One folder with `nrob-studio`,
  `nrob-server` and `nrob-diffusion`. Pick model files in its UI and it reads their
  headers to work out what each one is (LLM, image, video, or a part such as a
  projector, LoRA, encoder or tokenizer), then serves it on the right endpoint.
  Routes and dialects are configurable (OpenAI chat, Images and Videos, or nrob's
  job API). Image and video weights sit on the GPU, in RAM or on the SSD per your
  memory settings, and media jobs pause the LLM only when they need its GPU. See
  [the studio guide](docs/STUDIO.md).
- **OrcaSAQ2 27B from original EXL3 safetensors.** Native Rust/CUDA loading of
  mixed 3/3.5/4-bit decoder projections, a 6-bit head and int8 embeddings,
  with Qwen reasoning, XML tool calls and prompt caching. See
  [download, setup and current limits](docs/ORCASAQ.md).

- **Native Rust Qwen Image 2.1 generation.** GGUF Q4 or safetensors, four-/six-step
  Viggle turbo, queued multi-image batches, and a memory handoff from DeepSeek to
  a smaller Qwen controller. See [setup and agent usage](docs/QWEN_IMAGE.md).

- **Native Rust SDXL generation.** Single-file SDXL checkpoints, dual CLIP text
  encoders, DPM++ 2M Karras, negative prompts and queued batches. See
  [SDXL setup and checkpoint settings](docs/SDXL.md).

- **Native Rust LTX video inference.** Distilled LTX 2.3, LTX 2.5 and Sulphur-2,
  with a shared media queue, agent tool, RAM offloading and direct SSD weight
  streaming. See [video configuration and current limits](docs/LTX_VIDEO.md).

- **Native Rust speech, music and sound effects.** Qwen3-TTS or Breeze TTS 2
  speaks in a voice described in words, and a saved voice stays the same from
  line to line ([Speech](docs/SPEECH.md)). MOSS-SoundEffect v2.0 makes up to
  30 seconds of 48 kHz sound from a description ([Sound effects](docs/SOUND.md)). MiniMax Music 3 writes songs with vocals from
  lyrics and a description, up to six minutes of 44.1 kHz stereo, faster than
  real time; its 8B language model converts once to q8_0 or q4_k, so an 8 GB
  GPU can run it ([Music](docs/MUSIC.md)). All are served on OpenAI-style
  audio endpoints, and an LTX clip can follow that speech, or any audio file,
  with mouths moving to it.

- **Native Rust 3D models from a picture.** Pixal3D (TRELLIS.2's cascade with
  pixel-aligned conditioning, with DINOv3 and NAF) turns a picture of one object
  into a textured GLB in about 95 seconds on an RTX 5090. BiRefNet removes the
  picture's background, and Real-ESRGAN enlarges a small one first. Both are
  picture tools of their own too: background removal and upscaling on
  `/v1/images/*` ([Picture tools](docs/PICTURE_TOOLS.md)).

- **Get models.** NROB Studio downloads ready-to-run mainstream models from
  Hugging Face (chat, images, video, speech, music, sound effects, 3D and the
  picture tools) into folders of their own, with pause and resume, and adds
  them when they finish ([Getting models](docs/STUDIO.md#getting-models)). The mesh is closed and
  simplified, and its colours are baked into PBR textures, ready for three.js,
  game engines and 3D printing slicers ([3D models](docs/MODEL3D.md)).

- **DeepSeek-V4.1-Flash from safetensors, end to end.** The checkpoint's shards are
  indexed in place (51 ms, headers only) and each expert is served with two positioned
  reads. Hybrid CPU/GPU decode across two GPUs, per-GPU VRAM expert caches, an AVX-512
  CPU expert kernel, and caches that stay warm across runs. Token-identical to the
  reference implementation on the golden prompts. See [below](#deepseek-v41-flash).
- **An OpenAI-compatible server for coding harnesses.** `nrob-server` serves
  DeepSeek-V4.1 on `/v1/chat/completions` with streaming, tool calls, reasoning
  (`reasoning_content`), and a prefix cache that resumes each turn where the last one
  left off. See [Serving](#serving-an-openai-compatible-api).
- **Any GGUF.** Llama, Qwen2/3/3.5, Gemma 3/3n/4, Mixtral, Qwen3-MoE and more, through
  our own pure-Rust GGUF stack (`gguf`, `ggml-quants`, `ggml-rs`, `tokenizer`,
  `llama-rs`; no llama.cpp, no C++). Chat templates per architecture, and vision towers
  for several of them.
- **MoE experts streamed straight from the `.gguf`.** `--budget 12G` loads only the
  trunk and reads routed experts on demand, with parallel positioned reads, into a
  bounded RAM cache. The same quantized kernels run on the streamed bytes, so the output
  is token-identical to loading everything.
- **WebGPU and CPU for machines without CUDA.** `--webgpu` (and the
  `nrob-server-webgpu` build) runs GGUF models' quantized matmuls on any Direct3D
  12, Vulkan or Metal GPU, straight from the GGML blocks, with the rest on the
  CPU. Weights past the GPU budget run on the CPU. In testing, greedy tokens
  matched the CPU on 1B, 9B and 27B models. See [WEBGPU.md](docs/WEBGPU.md).
- **CUDA.** `--cuda` runs GGUF models on the GPU, with GPU-side routing, grouped MoE
  kernels and a VRAM expert cache (`--vram-cache`) for streamed models. Token ids are
  bit-identical to the CPU path on every tested model. The default build is CPU-only
  and needs nothing beyond Rust.
- **Std-only core.** `nrob` and `dsv41` have zero external dependencies and
  `#![forbid(unsafe_code)]`. The JSON parser, the expert cache and the thread pool are
  all hand-rolled.

## Quickstart

The easiest way in is the studio:

```sh
cargo build --release -p nrob-studio   # std-only; builds anywhere
powershell tools/qwen-image/build.ps1  # CUDA engines: nrob-diffusion and nrob-server
target/release/nrob-studio             # opens the control UI; add models from the Models page
```

To drive the engines directly instead:

Requires a recent stable Rust. The default build is CPU-only:

```sh
cargo build --release
```

Run any GGUF model:

```sh
nrob run model.gguf "What is the capital of France?" -n 64
nrob chat model.gguf --system "You are concise."
nrob info model.gguf
```

A MoE model bigger than your RAM or VRAM? Rob it one expert at a time:

```sh
nrob run Qwen3-30B-A3B-Q4_K_M.gguf "Explain rainbows." --budget 8G --stats
```

`--budget` caps resident weights plus the RAM expert cache; `--stats` prints tokens/s
and the cache's hit rate. Other subcommands: `bench` (timings plus a JSON result
schema), `tokenize`, `detokenize` (see `nrob --help`).

GPU build (needs the CUDA toolkit at build time, a driver at run time):

```sh
cargo build --release -p nrob-cli --features cuda
nrob run model.gguf "prompt" --cuda
nrob run big-moe.gguf "prompt" --cuda --budget 16G --vram-cache 20G
```

## DeepSeek-V4.1-Flash

The biggest job so far: `DeepSeek-V4.1-Flash` (FP8),
510 GB of FP8 trunk and MXFP4 experts (40 layers × 384 routed experts, 6 active per
token), plus hyper-connections, compressed sparse attention and Engram n-gram memory.
The checkpoint is used as-is: `dsv41` reads the safetensors headers and serves each
expert with two positioned reads.

```sh
echo Explain how rainbows form. > prompt.txt

# decode on both GPUs: layers 0-19 on cuda:1, 20-39 on cuda:0
cargo run -p dsv41-cuda --release --example generate -- prompt.txt 256 1,0
```

The prompt goes through nrob's own Rust tokenizer and chat format, both matched
exactly against the reference (3,638 tokenizer cases, 420 chat-format cases).
`generate` streams the text and prints per-token timing and cache hit rates. Useful
environment variables: `DSV41_MODEL` (checkpoint directory), `DSV41_RAM_GB` (RAM tier,
default 140), `DSV41_USAGE` (the saved expert-usage profile that warms the tiers at
start), `DSV41_RUNS` (answer the prompt several times in one process, as a server
would), and `DSV41_PROFILE` (per-phase timing).

| On a 9950X3D, 192 GB DDR5, 2× RTX 5090 | Decode |
|---|---:|
| Warm (RAM and VRAM tiers filled; any long-lived process) | **~29 tok/s** |
| First answer of a fresh process, warm-started from the usage profile | 2–6 tok/s (bound by the SSD) |

How it gets there, what it cost, and what limits it now (RAM bandwidth, kernel-launch
overhead, and the SSD for cold starts) is written up in
[docs/DEEPSEEK_V41.md](docs/DEEPSEEK_V41.md). Correctness is gated on golden files from
the reference implementation: greedy tokens identical, every layer checked in
isolation, every GPU kernel tested against its CPU counterpart.

### Serving: an OpenAI-compatible API

```sh
cargo build --release -p nrob-server
nrob-server --model <checkpoint dir> --usage <profile file>   # then point a harness at http://127.0.0.1:8000/v1
```

`--model` names the checkpoint directory; nothing is assumed about where it lives. The
Engram precompute (`engram_meta.safetensors`, from `tools/dsv41/oracle.py`) is read from
that directory unless `--engram-meta` names the file, and `--usage` (optional) keeps an
expert usage profile that warms the caches at start. `nrob_server::start` runs the same
server inside another program: `coder-cli` carries the model that way.

Any OpenAI-compatible client or coding harness (Aider, Cline, Continue, OpenCode, ...)
works with base URL `http://127.0.0.1:8000/v1`, model `deepseek-v4.1-flash`, and any API
key (or the one set with `--api-key`). Claude Code speaks Anthropic's API, not
OpenAI's.

- **Endpoints:** `POST /v1/chat/completions` (streamed or whole), `POST /v1/completions`,
  `GET /v1/models`.
- **Tools:** OpenAI `tools` become the model's DSML tool format and its calls come back
  as `tool_calls` (parallel calls included).
- **Reasoning:** answers directly by default. `reasoning_effort` (or DeepSeek's
  `thinking: {type: enabled}`) turns reasoning on and streams it as
  `reasoning_content`. `--thinking` makes it the default. The reasoning has a budget,
  by effort (low 2,048 tokens, medium and high 8,192, 76-99 16,384, max none) or from
  the request's `thinking.budget_tokens` (0: none): past it the server writes
  `</think>` itself, so a model going round in circles stops and answers.
- **Progress:** a streamed request also gets chunks with no `choices` (OpenAI clients
  skip them): `nrob_progress: {prompt_done, prompt_total}` while the prompt is read,
  `nrob_tool: {calls, name, parameter, chars, tail}` a few times a second while a
  tool call is being written, so a harness can show the work before the call is whole,
  and `nrob_thinking: {used, budget, done}` every 16 reasoning tokens.
- **Prefix cache:** a harness resends the whole conversation every turn. The server
  resumes from the live state or from a checkpoint (taken every 256 prompt tokens and at
  user-turn boundaries), so a turn only runs its new tokens. A new chat with the same
  system prompt skips it too.
- **Prompt states on disk** (`--prompt-cache DIR`): the state where a conversation's
  first user message begins (the system prompt and tools) and the state at the end of
  each prompt are also written to disk (tens of MB each: the window rings, the
  compressed rows and index keys so far), and a new process starts from them, so a
  restart does not read a system prompt, or a conversation it resumes, again.
  `--prompt-cache-gb` caps the disk they take (default 4).
- **Qwen hybrid prompt cache:** the native `qwen35` controller also checkpoints
  attention, delta-net and convolution state before the first user message and
  the current assistant reply. Re-rendered reasoning or tool calls reuse the
  matching prefix instead of starting over. Up to three host checkpoints are
  retained, with a 1 GiB target (one larger checkpoint is allowed). Text-only
  states use `--prompt-cache` across restarts; image-conditioned states stay in
  memory. An identical retry leaves one token to recompute the sampling logits.
  The initial uncached prompt still needs a full prefill.
- **Long prompts** run layer by layer: every expert is read once for the whole prompt
  instead of once per chunk (~6× faster here than chunked prefill), up to 20,480 tokens a
  pass (`--layered-max`); a longer prompt splits into equal passes, and a pass that runs out
  of VRAM is redone in halves. One pass over 16,800 tokens took 152 s (83 GB from the drive)
  where three passes of 5,600 took 238 s (257 GB).
- **Images:** `image_url` parts with base64 `data:` URLs (PNG or JPEG), or local file
  paths when the server listens on loopback only. See [Vision](#vision).

What to expect on this machine. The model is 510 GB; RAM and VRAM hold about two thirds
of the experts, and the rest come from the drive, so the drive matters. The internal
NVMe throttles to 0.51 GB/s under sustained reads; an external Samsung T9 (USB
20 Gbps) holds 1.95 GB/s.

| Harness request | From the T9 |
|---|---:|
| First request, a 5.6K-token prompt it has never seen | ~2.2 min prompt (42 tok/s; was ~3 min) |
| A long prompt whose experts an earlier pass left in RAM | ~73 s for 5.6K tokens (76 tok/s; was 31): only what is not resident is read, and the math runs on tensor cores |
| New chat, same system prompt | prompt cached (5,560 of 5,580 tokens) |
| Next turn of a conversation | only the new tokens |
| Decode on a new topic | ~3 tok/s at first, 8.5 then 12 tok/s as VRAM adapts to it |
| Warm, same topic | up to ~29 tok/s (measured with `generate`) |

### Vision

The checkpoint carries a 32-layer ViT and an aligner, and nrob runs them:

```sh
python - <<'EOF'
import base64, json, urllib.request
img = base64.b64encode(open("chart.png", "rb").read()).decode()
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
    {"type": "text", "text": "What does this chart show?"}]}]}
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"])
EOF
```

- **Decoding is ours.** `nrob-image` is a std-only PNG decoder (with its own inflate)
  and JPEG decoder (baseline and progressive). It decodes to the same pixels as Pillow,
  which the reference uses, on all 111 test images: every PNG colour type, bit depth
  and interlace; JPEG 4:4:4, 4:2:2, 4:2:0, grey and CMYK, restart markers, odd sizes.
  The resize and pad are Pillow's too, pixel-exact.
- **Preprocessing** (sizing, padding, normalization, patches) is bit-exact to the
  reference's `load_image` on every test image, and its sizing plan on 5,712 sizes.
- **The vision tower** runs on the first GPU: about 1 GB of weights, 60 ms for a
  typical image (1,521 patches), 160 ms for 3,672. Its output is within 1.6–2.5% of the
  reference's (the reference differs from itself by as much across torch backends).
- **In the model:** image tokens take the tower's rows, route with the checkpoint's
  vision bias (`bias_vl`) and are kept out of Engram's n-grams, as in the reference.
  Every layer matches the reference in isolation, and an image may sit anywhere in a
  conversation: the prefix cache keys it by content.
- **Measured:** a chart image (254 tokens) reads back its title and all four values.
  Image tokens run at 3–9 tok/s depending on how warm the caches are; a follow-up
  question about the same image reuses it from the prefix cache.

## Benchmarks

Measured on the development machine (RTX 5090, 32 GB VRAM; the "~24 GiB" below is the
peak VRAM *in use* during GPU-resident runs, not the card's capacity). Streamed runs
read experts from the model file under the stated RAM budget; output is
token-identical to the resident path.

| Model | CPU tok/s | GPU resident tok/s | Streaming peak RAM |
|---|---:|---:|---:|
| DeepSeek-V4.1-Flash (510 GB safetensors, hybrid CPU + 2 GPUs) | — | **~29** (warm) | 140 GiB RAM tier |
| Qwen3-30B-A3B Q4_K_M (18.5 GB) | 0.25 | 16.9 | 3.64 GiB @ 256 MiB cache |
| qwen3-0.6b Q4_K_M | 1.67 | 79 | — |
| stories15M q8_0 | 55 | 455 | — |

On Qwen3-30B-A3B the expert-cache hit rate climbs from 11% at a 256 MiB cache to 69% at
4 GiB (81% at 12 GiB); GPU-resident peaks at ~24 GiB VRAM. The GGUF CPU path is limited
by compute (dequantize per matmul), not by the disk: streamed and resident runs decode
at the same speed.

## Workspace layout

```
crates/
  nrob/          core library (std-only): expert-store seam, LFRU RAM expert cache,
                 thread pool, JSON reader
  nrob-cli/      `nrob` binary: run / chat / bench / info / tokenize for GGUF models
  dsv41/         DeepSeek-V4.1 from safetensors: in-place expert store, CPU reference
                 model, CPU experts (std-only, forbid(unsafe_code))
  dsv41-cuda/    DeepSeek-V4.1 on CUDA: kernels, VRAM expert cache, hybrid decode,
                 chunk continuation, checkpoints, layer-by-layer prefill, vision tower
  nrob-image/    image decoding (std-only): PNG with its own inflate, JPEG, Pillow's
                 resize — pixel-exact to Pillow
  nrob-server/   OpenAI-compatible HTTP server for DeepSeek-V4.1 (std-only)
  nrob-studio/   portable host (std-only): supervises nrob-server and nrob-diffusion,
                 control UI, configurable OpenAI-style gateway, model detection
  nrob-diffusion/ image, video, speech, music, sound and 3D worker (Candle): Qwen Image,
                 SDXL, LTX, Qwen3-TTS, Breeze TTS 2, MiniMax Music 3,
                 MOSS-SoundEffect, Pixal3D, BiRefNet, Real-ESRGAN, with
                 SSD/RAM/GPU block residency
  ggml-rs-wgpu/  WebGPU backend: quantized GGUF matmuls in WGSL, the rest on the CPU
  gguf, ggml-quants, ggml-rs, ggml-rs-cuda, tokenizer, llama-rs
                 our pure-Rust GGUF stack: reader, quant kernels, CPU and CUDA
                 backends, tokenizers, model architectures, expert streaming
tools/dsv41/     Python oracle (the reference model, streamed) and helpers
docs/            DEEPSEEK_V41.md (the port), ROADMAP.md (streaming roadmap)
```

`PLAN.md` is the system plan and phase log; `CONVENTIONS.md` is the engineering contract
(std-only rules, naming, error handling); `crates/VENDORED.md` documents where the GGUF
stack comes from.

## Testing

```sh
cargo test --workspace
```

240+ tests. Correctness rests on synthetic GGUF models built in Rust (no downloads
needed), golden fixtures, and known-answer tests: the CLI tests write tiny llama and
Qwen3-MoE GGUFs and check that streamed experts give the same tokens as resident ones.
CUDA tests skip when no GPU is reachable.

The DeepSeek-V4.1 golden gates need the checkpoint and the oracle's golden files, so
they are `#[ignore]`d by default:

```sh
DSV41_CUDA_DEVICES=1,0 cargo test -p dsv41-cuda --release --test gpu_model -- --ignored --nocapture --test-threads=1
```

Vision has its own golden files (`tools/dsv41/vision_golden.py` for decoding,
preprocessing and the tower; `oracle.py --image` for a whole image prompt):

```sh
cargo test -p nrob-image --release            # decoders and resize against Pillow
cargo test -p dsv41 --release --test vision_golden -- --include-ignored
cargo test -p dsv41-cuda --release --test gpu_vision -- --ignored --nocapture
cargo test -p dsv41-cuda --release --test gpu_vision_model -- --ignored --nocapture --test-threads=1
```

## Status and caveats

- The GGUF CPU path is slow on big models (Qwen3-30B-A3B: ~0.2 tok/s on CPU against
  16.9 on one GPU); use `--cuda`. The DeepSeek-V4.1 path has its own AVX-512 expert
  kernel.
- GGUF expert streaming covers the Qwen3-MoE and Mixtral families; other MoE
  architectures load resident.
- **Direct I/O** (page-cache bypass) is used by `dsv41`; GGUF streaming reads through
  the OS page cache.
- DeepSeek-V4.1 runs through `nrob-server` or the `dsv41-cuda` example; `nrob-cli`
  takes GGUF files.
- **DeepSeek-V4.1 context:** the model is trained for 1,048,576 tokens (YaRN ×16 over
  65,536). nrob's caches cost ~7 KB a token, so `--ctx` can go far (65,536 by default).
  `--ctx auto` serves every other model at the most it allows (its
  `max_position_embeddings`) and DeepSeek at the default; Studio passes it when
  the context setting is 0, its default.
  Verified against the reference oracle up to 247 tokens; longer contexts run the same
  code, but candidate filtering only engages past ~32K tokens and is unverified there.
- **DeepSeek-V4.1 vision:** PNG and JPEG only (GIF, WebP and BMP are refused with a
  clear error), and no `http(s)` image URLs (a std-only build has no TLS; send
  `data:` URLs). An image prompt is numerically touchy: one bf16 rounding step on 1%
  of an image's inputs moves the model's final logits by ~25%. So replies match the
  reference in substance and mostly in wording, not token for token (each layer
  matches as tightly as for text: 0.5% of routes differ, text prompts 0.4%).
- On a 192 GB machine the drive sets the pace for new prompts and topics: run from a
  drive that sustains its speed (the T9 does, the internal one throttles). More RAM
  would help most of all.
- The CLI is text-only. Vision towers live in `llama-rs` (see its examples).

## Provenance

nrob is written from scratch in Rust. The GGUF stack (`gguf`, `ggml-quants`, `ggml-rs`,
`ggml-rs-cuda`, `tokenizer`, `llama-rs`) is the repo author's own Rust code, vendored
from their `llm` workspace (see `crates/VENDORED.md`). The DeepSeek-V4.1 engine
follows DeepSeek's reference implementation and chat encoding, published under the MIT
License (see `NOTICE`). `nrob-image` is our own code; to decode to the same pixels as
the reference's Pillow, it follows the arithmetic of Pillow's resampling and of
libjpeg-turbo's IDCT, upsampling and colour conversion (see `NOTICE`).

## License

`nrob`, `nrob-cli`, `nrob-image`, `nrob-server`, `dsv41` and `dsv41-cuda` are licensed under **Apache-2.0**
(`LICENSE-APACHE`; see `NOTICE`).

The GGUF stack crates are dual-licensed **MIT OR Apache-2.0** (`LICENSE-MIT`,
`LICENSE-APACHE`).
