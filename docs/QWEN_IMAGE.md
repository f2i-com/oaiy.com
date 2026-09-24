# Native Qwen Image 2.1

NROB implements text-to-image and reference-image editing in Rust: Qwen3-VL-8B conditioning,
the 32-layer single-stream image transformer, FlowMatch Euler and the RGBA VAE.
The compute worker uses Candle tensor primitives; it does not launch Python,
ComfyUI or stable-diffusion.cpp. The server and existing core remain std-only.

Both published GGUF (including Q4_K_M) and BF16/F16 safetensors are read in place.
GGUF matrices remain quantized between operations. The base checkpoint directory
supplies `text_encoder`, `processor` and `vae`. A single diffusion safetensors
file or the sharded `transformer` directory can supply the image weights.

## Build and configuration

On Windows with CUDA 12.8 and Visual Studio 2022 C++ build tools:

```powershell
./tools/qwen-image/build.ps1
```

On a configured CUDA development shell, the equivalent commands are:

```sh
cargo build --release -p nrob-diffusion --features flash-attn
cargo build --release -p nrob-server
```

The first FlashAttention build compiles CUDA kernels and downloads the pinned
NVIDIA CUTLASS headers; it can take several minutes. `--features cuda` remains
available for GPUs/build environments without FlashAttention. The optimized
path targets Ampere or newer NVIDIA GPUs and preserves the causal text prefix
by evaluating it separately from bidirectional image queries. The VAE's wider
attention head uses the bounded reference implementation.

Copy `config/qwen-image.example.json` to a local configuration and set the paths
and device ordinals. The integrated controller configuration currently requires
two different CUDA devices. On this machine GPU 0 holds the Qwen 27B controller;
GPU 1 encodes prompts, drops the text encoder, then holds the diffusion model
and VAE. The standalone worker can use one GPU without a language controller.

```powershell
./target/release/nrob-server.exe --model D:/deepseek/model --image-config config/qwen-image.local.json
```

The configured controller is automatically added to NROB's model list. Generation
unloads and joins the old language worker before loading the controller, then
starts the diffusion worker. Further chat requests are routed to the controller
until explicitly released, including requests that still name DeepSeek. The
response's `model` identifies the actual controller. No previous GPU state is
retained by the unloaded worker. Failed controller loading never starts diffusion.

## Agent tool

The sibling coder-cli exposes `image_generate`. Build the updated coder-cli and
add the following in its existing `[nrob]` settings table (or set the
`NROB_IMAGE_CONFIG` environment variable):

```toml
image_config = 'E:\deepseek\nrob\config\qwen-image.local.json'
```

An example tool request is:

```json
{"action":"generate","prompt":"A friendly robot painting a landscape","n":100,"output_dir":"robot-series","weights":"gguf","steps":6}
```

One prompt is reused with seeds `seed..seed+n-1`; alternatively `prompts` must
contain exactly `n` prompts. Use `action: "status"` to inspect progress and
results, `cancel` to terminate the worker and free its VRAM, and `release` after
image work is finished to restore normal language-model selection. Release does
not immediately reload DeepSeek; the next chat request chooses the model.

Reference images are optional. Omit `images` (or pass `[]`) for text-to-image.
Pass one, two, or three ordered paths for editing or composition, and describe
the desired change in the prompt. The same references are reused for every image
in a batch. Inputs are opened for reading; results always go into a new batch folder.

```json
{"action":"generate","prompt":"Change the robot in image 1 to blue; preserve the scene.","images":["robot.png"],"output_dir":"edits"}
```

```json
{"action":"generate","prompt":"Place the character from image 1 in the setting from image 2, using the colors of image 3.","images":["character.png","setting.jpg","palette.webp"],"n":10,"output_dir":"compositions"}
```

The agent tool resolves relative references against its workspace. HTTP callers
must use absolute paths accessible on the NROB server. PNG, JPEG, and WebP are
supported, with a 32 MiB limit per file and bounded decoding. The server's existing
`--local-images` policy also governs these inputs (enabled by default on loopback,
disabled by default when listening remotely). Up to three references are supported
in both base and turbo modes. Six-step turbo remains the default for both editing
and text-to-image.

`reference_size` defaults to 1024: each reference retains its aspect ratio and is
resized to approximately that squared pixel area, rounded to multiples of 32.
Set it to 512 for faster reference encoding and less memory. Allowed values are
256–1024 in multiples of 32; reference aspect ratios must stay between 1:8 and 8:1.
Output `width` and `height` remain independent (default 1024×1024).
The VAE preserves alpha; the Qwen3-VL vision tower sees RGB composited over white.
Reference/text conditioning keys and values are cached across sampling steps and
reused across images sharing a prompt. Reference encoding and model loading add
setup time before the first output.

The tool queues work and returns immediately. A queued/running job is **not** a
successful generation. Wait for `completed` and inspect `result.data` for paths.
The server keeps the latest job in memory; PNGs and a per-image JSONL manifest
persist on disk. Completed images survive cancellation/failure. A restarted
server does not resume an interrupted batch automatically.

## HTTP and standalone interfaces

All image endpoints use the server's usual bearer authentication:

| Method | Endpoint | Effect |
|---|---|---|
| POST | `/v1/images/generations` | Queue one batch; returns HTTP 202 |
| GET | `/v1/images/status` | Latest job, progress, completion/error and paths |
| POST | `/v1/images/cancel` | Cancel the active batch |
| POST | `/v1/images/release` | Allow language-model switching after the batch |

Generation accepts `prompt`/`prompts`, `n` (1–1000), `width`/`height` (256–2048,
multiples of 32), `steps`, `seed`, `weights` (`gguf` or `safetensors`) and `turbo`
(default true). `output_dir` must be relative to the configured `output_root`;
absolute paths, parent traversal and symlinks escaping that root are rejected.
Every batch gets a unique subdirectory and never overwrites another batch.
Only one image batch can run at a time. Controller chats can continue during it.
This is an asynchronous local extension, not the OpenAI base64-image response.

For the standalone worker, use `nrob-diffusion --request request.json` or send
JSON on stdin with `--stdin`:

```json
{
  "base":"D:/Qwen-Image-2.1",
  "transformer":"E:/deepseek/nrob/models/qwen-image-2.1/qwen-image-2.1-Q4_K_M.gguf",
  "adapter":"E:/deepseek/nrob/models/qwen-image-2.1/viggle-v0.2.1-r128.safetensors",
  "output_dir":"E:/deepseek/nrob/generated-images",
  "prompt":"A friendly robot painting a landscape",
  "n":2,"width":1024,"height":1024,"steps":6,"seed":42,"device":1
}
```

Progress is JSONL on stderr; stdout contains one final JSON result. Dropping the
process releases all image allocations. CPU builds support reference tests, but
full model inference is intended for a CUDA build.

## Performance measurement

Six-step turbo remains the default. Sampling events include per-step `seconds`;
each saved image and manifest record includes `sampling_seconds`,
`decode_seconds` and `image_seconds`. The final result separates text loading,
prompt encoding, transformer loading and VAE loading from the batch total.

```powershell
./tools/qwen-image/benchmark.ps1 -Count 4 -Size 1024
./tools/qwen-image/benchmark.ps1 -Count 4 -Size 1024 -Weights safetensors
```

This standalone benchmark uses the configured image GPU and writes its request,
events, results and PNGs below `target/qwen-image-bench`. It reports the first
image separately and the median of subsequent images. It does not load the
language controller. Avoid concurrent GPU workloads or compilation when timing.

NROB shares a `.cuda-cache` directory under the configured output root between
jobs. The standalone worker defaults to `.cuda-cache` under its output directory.
An explicit `CUDA_CACHE_PATH` or `CUDA_CACHE_MAXSIZE` takes precedence; otherwise
the worker allows a 1 GiB compiled-kernel cache. The initial run still compiles
kernels and every worker invocation loads model weights. Use one batch for many
images to amortize setup; warm per-image timing is not first-request latency.

## Turbo and source references

Viggle v0.2.1 rank-128/256 adapters are applied at runtime with alpha/rank = 1,
not merged into low-precision weights. Six-step nodes are
`[1, .9375, .875, .75, .5, .25]`; four-step nodes are `[1, .75, .5, .25]`.
Both receive the resolution-dependent exponential shift; turbo disables
terminal stretching and CFG (`cfg=1`). Base mode (`turbo:false` / no standalone
adapter) defaults to 40 steps, CFG 6 and the base terminal shift of .02.
Both sampling modes support the optional ordered reference images above.

- [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- Requested GGUF weights and checksums
- [Viggle adapter and exact sampling instructions](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
- [Reference transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Reference VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)

The Rust architecture port follows the Apache-2.0 Qwen/Hugging Face reference.
Model weights retain their own Qwen Research License.

## Local validation (2026-09-24)

Optional-reference validation, 1024×1024 output, six-step turbo on RTX 5090:

| Input | Weights | Sampling + decode + save |
| --- | --- | ---: |
| No references (`images: []`) | Q4_K_M | 4.4 s |
| One reference, orange robot recolored blue | Q4_K_M | 4.7 s |
| Two references, both robots composed together | Q4_K_M | 5.1–5.3 s |
| Three references, ordered triptych composition | Q4_K_M | 5.7 s |
| Two references, both robots composed together | BF16 safetensors | 4.9 s |

These times exclude reference/text encoding, model loading, and condition-cache
preparation. Complete reference jobs took 21–61 seconds depending on loading/cache
state. The two-reference Q4 batch produced two distinct seeded outputs. Inputs'
SHA-256 hashes were unchanged. Outputs were visually inspected for the requested
edits/composition. Three-reference, text-only, and BF16 jobs used the HTTP endpoint
with the Qwen 27B controller resident on GPU 0. The controller returned a valid
`image_generate` call containing the requested reference path during generation.

Tests cover optional/empty and one-to-three reference forwarding, rejection of
four references, local-file permissions, protected agent paths, PNG/JPEG/WebP,
alpha handling, patch ordering, temporal VAE shortcuts, and block attention.
The GPU attention comparison also checks causal text interleaved between reference
blocks against the F32 CPU result. All 11 diffusion tests passed with GPU tests enabled.
The affected coder-cli library suites passed 268 tests (15 explicitly ignored).
The final NROB workspace suite passed 507 tests (77 explicitly ignored), with
the diffusion GPU test also run separately as noted above.

After the FlashAttention optimization, on RTX 5090 at 1024x1024, six steps,
CFG 1, including VAE decode and PNG writing:

| Weights | Warm image time |
| --- | ---: |
| Q4_K_M, original attention | 7.8 s |
| Q4_K_M, FlashAttention | 4.5 s |
| BF16 safetensors, FlashAttention | 4.4 s |

The optimized Q4 and BF16 results were verified with the Qwen 27B controller
resident on GPU 0 and diffusion on GPU 1. An API request omitting `steps`
generated three images at six steps; warm images took 4.52 and 4.53 seconds.
The controller answered a chat request during the job. That whole job took
33.1 seconds including setup and first-image initialization, so the warm timings
do not describe cold first-image latency. Reusing the compiled-kernel cache
reduced prompt encoding from approximately 25 seconds to 0.7 seconds.

The GPU regression compares FlashAttention with the F32 reference for causal,
mixed-prefix and bidirectional attention, including multiple batches and
non-aligned sequence lengths. All six diffusion tests passed with the GPU test
explicitly enabled. Generated Q4 and BF16 images were also visually inspected.
The full workspace suite with `--features nrob-diffusion/flash-attn` passed:
501 tests passed, zero failed, 77 explicitly ignored (the GPU attention test
was then run separately with `--include-ignored`).

Initial implementation smoke tests:

- Native GGUF six-step, 512x512: coherent apple image, 66.9 seconds including load.
- Native BF16 safetensors four-step, 512x512: coherent robot/umbrella image, 58.4 seconds.
- Two 1024x1024 images in one six-step GGUF batch with the controller resident:
  106.9 seconds; distinct seeds and output files.
- Requested Q4_K_M six-step, 512x512: 48.9 seconds after the controller handoff.
  DeepSeek was initially loaded on both cards; logs confirmed unload completed
  before Qwen3.8-27B loaded and before the image worker started.
- Qwen controller returned a valid `image_generate` tool call during sampling.
- Cancellation killed the image worker and returned GPU 1 memory to zero.
- `cargo test --workspace --release`: 498 passed, 76 explicitly ignored tests.
  Follow-up diffusion/server tests passed after final validation tests were added.
- Affected coder-cli library suites: 314 passed, 15 explicitly ignored tests;
  tool visibility/routing regression passed after final guidance changes.

These are smoke tests on two RTX 5090 cards, not numerical parity certification
against Diffusers. A full 100-image batch was not run; the batch interface accepts
100 and was exercised with two-, three- and four-image batches and a cancelled
100-image request.
The local D: checkpoint is BF16; separate F16 weights were not available to test.
