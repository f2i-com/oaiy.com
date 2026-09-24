# Native LTX video

The `nrob-diffusion` worker implements silent text-to-video for distilled
LTX 2.3, LTX 2.5 and Sulphur-2 checkpoints in Rust. Candle supplies tensor
operations and CUDA kernels. Gemma text encoding, the video transformer,
eight-step Euler sampling and the convolutional VAE run inside the worker.
FFmpeg only encodes the decoded pixels into an H.264 MP4; it does not run models.
The server and inference core remain std-only and forbid unsafe Rust.

Sulphur-2 has been validated end to end locally. Complete LTX 2.5 generation
validation is still pending the remaining checkpoint downloads; its Gemma 4
attention layers and convolutional decoder have passed reference checks.

## Models

| Model | Diffusion weights | Text encoder | Decoder |
|---|---|---|---|
| [LTX 2.3](https://huggingface.co/Lightricks/LTX-2.3) | `ltx-2.3-22b-distilled-1.1.safetensors` | Gemma 3 12B BF16, `model.*` tensor layout, with `tokenizer.json` | Included in checkpoint |
| [Sulphur-2](https://huggingface.co/SulphurAI/Sulphur-2-base) | `sulphur_distil_bf16.safetensors` | Same Gemma 3 encoder | Included in checkpoint |
| [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5) | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, including its tokenizer | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` |

Use the complete distilled checkpoints. The worker does not apply video LoRAs,
interpret FP8/NVFP4/GGUF video weights, or run the LTX 2.5 diffusion decoder.
It reads published BF16/F16/F32 safetensors directly without conversion.
LTX 2.5's Gemma 4 Unified path has distinct RMS normalization, attention scaling,
global head dimensions, shared K/V projections and proportional rotary positions.
It cannot use Gemma 3 as a substitute. Model licenses and access requirements are
available in the linked repositories.

## Configure and build

Use the same CUDA worker and server build as [Qwen Image](QWEN_IMAGE.md):

```sh
cargo build --release -p nrob-diffusion --features flash-attn
cargo build --release -p nrob-server
```

Install FFmpeg with `libx264` support. Copy `config/ltx-video.example.json` to
`config/ltx-video.local.json`, set absolute checkpoint paths, and add its absolute
path as `video_config` in the existing image/controller configuration:

```json
"video_config": "E:/deepseek/nrob/config/ltx-video.local.json"
```

Start NROB with that `--image-config`, or use coder-cli's existing
`[nrob].image_config` setting. The same lightweight controller handoff applies:
DeepSeek is unloaded before the configured Qwen controller is selected, and the
video worker runs on the other configured GPU. Image and video jobs share one
queue so their GPU allocations cannot overlap. A standalone worker can use one
GPU without a language controller.

## Agent and HTTP usage

Coder-cli exposes `video_generate`:

```json
{"action":"generate","model":"sulphur-2","prompt":"A red fox walks through a snowy forest. The camera slowly follows from the side. Snow falls softly.","width":512,"height":320,"frames":49,"fps":24,"seed":42,"memory":"auto","output_dir":"fox-videos"}
```

Use `action: "status"` for progress and completed file paths. Status includes
configured video models and whether their weight files are present. Do not
resubmit a queued/running job. `cancel` terminates the worker and releases its
CUDA context; `release` permits the original language model again once all
media work has finished. The next chat request then selects its requested model.

Equivalent authenticated endpoints are:

* `POST /v1/videos/generations` — request above without `action`; returns HTTP 202.
* `GET /v1/videos/status` — shared media job and video model readiness.
* `POST /v1/videos/cancel` — stop the active shared media job.
* `POST /v1/videos/release` — release the controller selection.

Requests generate one clip. Dimensions must be multiples of 32, from 128 to
1024. Frame counts must be `8k+1`, from 9 to 121; 49 frames at 24 fps is about
two seconds. Distilled sampling uses the trained eight-step schedule. Qwen
Image's six-step setting does not apply to LTX. This video path currently
accepts text only and does not generate audio or use reference images.

Completed jobs return an MP4 path, a middle-frame PNG preview, dimensions,
frame rate, seed, stage timings and weight-residency statistics. A JSON sidecar
records the prompt and generation details. Output folders stay under the
server's configured output root. Model and executable paths come from trusted
server configuration, never from the generation request.

## VRAM, RAM and SSD

| `memory` | Behavior |
|---|---|
| `auto` | Retains transformer blocks up to the VRAM budget; caches overflow weights in RAM up to the host budget; reads remaining weights from their checkpoint as needed. |
| `gpu` | Requires the video transformer blocks to fit the VRAM budget; otherwise returns an error. |
| `ram` | Transfers each transformer block from a bounded host cache for GPU execution; overflow beyond the RAM budget is read from the checkpoint. |
| `ssd` | Keeps no host or GPU transformer-block cache; reads each block directly from the original checkpoint on every step. |

`ram_gb` and `vram_gb` are weight-cache budgets, not total process memory limits.
The server caps requested budgets at its configured values. The worker also
caps GPU residency against current free VRAM with space reserved for activations,
global projections and a streamed block. OS file caching may service SSD reads
from RAM; no converted files or secondary weight cache are created.

All modes still need GPU space for an individual block and compute workspace.
Text encoder layers are loaded one at a time and released before denoising.
The transformer is released before VAE decoding. Ordinary clips decode with
full spatial context; larger clips use overlapping tiles sized for available
VRAM. RAM and SSD modes trade transfer time for lower VRAM requirements.

Local Sulphur validation on an RTX 5090 generated a 512×320, 49-frame clip at
24 fps through the authenticated HTTP queue in approximately 37 seconds with
GPU residency, 63 seconds with RAM offload, and 106 seconds with direct weight
reads. The three modes produced byte-identical MP4 files for the same prompt
and seed. These are individual measurements with warm file caches and concurrent
model downloads, not guaranteed latency. GPU denoising took about 14 seconds;
text encoding remained a substantial part of total time. Clips are silent.

The reported `gpu_weight_bytes`/`ram_weight_bytes` refer to transformer block
residency. `weight_bytes_read` counts checkpoint bytes requested by that store,
including conditioning projections/connectors; it is not physical disk traffic
and excludes the separate Gemma/VAE readers.

## Verification

`cargo test --workspace` covers request constraints, trusted path selection,
memory-budget limits and convolution boundary behavior. CUDA integration tests
exercise the real convolutional VAE. Opt-in reference tests compare Gemma 3
local/global attention and a Sulphur video block against the official PyTorch implementations;
the block test also requires identical RAM-cache and SSD-read results.
PyTorch is only an offline verification tool, never an inference dependency.

Generate the reference activations with `tools/ltx/gemma_reference.py --gemma
<gemma-file>` and `tools/ltx/dit_reference.py --checkpoint <distilled-file>
--source <official-LTX-2-checkout>`.
Both accept `--device cuda:0` and `--output target/ltx-golden`.

The complete 48-layer text-feature reference uses
`tools/ltx/gemma_full_reference.py --gemma <gemma-file> --config <Gemma-3-config.json>`.
The complete video-transformer reference uses
`tools/ltx/transformer_reference.py --checkpoint <distilled-file> --source
<official-LTX-2-checkout>` and needs about 26 GiB of free GPU memory.
Gemma 4 uses `tools/ltx/gemma4_reference.py --gemma <Gemma-4-file>` with both
`--mode blocks` and `--mode features`. It requires Transformers 5.17 or newer;
`--reference-deps` can point to an isolated installation for these offline checks.
The VAE reference uses `tools/ltx/vae_reference.py --vae <vae-file> --source
<official-LTX-2-checkout>`. It requires the official package's test dependencies
and also accepts an isolated dependency directory through `--reference-deps`.

On the local RTX 5090 checks, complete Gemma 3 features differed from the official
BF16 encoder by about 0.7% relative RMS; the complete Sulphur video transformer
differed by 4.2%, and LTX 2.5 convolutional VAE pixels differed by 1.44%.
These compare numerical components, not end-to-end video quality.

Set `NROB_LTX_GOLDEN`, `NROB_LTX_GEMMA`, `NROB_LTX_GEMMA4`, `NROB_LTX_CHECKPOINT`, `NROB_LTX_VAE`,
and optionally `NROB_LTX_TEST_DEVICE` before running the ignored tests:

```sh
cargo test --release -p nrob-diffusion --features flash-attn --lib ltx:: -- --include-ignored --test-threads=1
cargo test --release -p nrob-diffusion --features flash-attn --test ltx -- --include-ignored --test-threads=1
```
