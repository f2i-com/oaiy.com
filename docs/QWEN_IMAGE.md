# Native Qwen Image 2.1

NROB implements text-to-image inference in Rust: Qwen3-VL-8B text conditioning,
the 32-layer single-stream image transformer, FlowMatch Euler and the RGBA VAE.
The compute worker uses Candle tensor primitives; it does not launch Python,
ComfyUI or stable-diffusion.cpp. The server and existing core remain std-only.

Both published GGUF (including Q4_K_M) and BF16/F16 safetensors are read in place.
GGUF matrices remain quantized between operations. The base checkpoint directory
supplies `text_encoder`, `processor` and `vae`. A single diffusion safetensors
file or the sharded `transformer` directory can supply the image weights.

## Build and configuration

On Windows with CUDA and Visual Studio C++ build tools:

```powershell
./tools/qwen-image/build.ps1
```

On a configured CUDA development shell, the equivalent commands are:

```sh
cargo build --release -p nrob-diffusion --features cuda
cargo build --release -p nrob-server
```

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

## Turbo and source references

Viggle v0.2.1 rank-128/256 adapters are applied at runtime with alpha/rank = 1,
not merged into low-precision weights. Six-step nodes are
`[1, .9375, .875, .75, .5, .25]`; four-step nodes are `[1, .75, .5, .25]`.
Both receive the resolution-dependent exponential shift; turbo disables
terminal stretching and CFG (`cfg=1`). Base mode (`turbo:false` / no standalone
adapter) defaults to 40 steps, CFG 6 and the base terminal shift of .02.
Image editing and reference-image conditioning are not implemented here.

- [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- Requested GGUF weights and checksums
- [Viggle adapter and exact sampling instructions](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)
- [Reference transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py)
- [Reference VAE](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_qwenimage21.py)

The Rust architecture port follows the Apache-2.0 Qwen/Hugging Face reference.
Model weights retain their own Qwen Research License.

## Local validation (2026-09-24)

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
100 and was exercised with two real images and a cancelled 100-image request.
The local D: checkpoint is BF16; separate F16 weights were not available to test.
