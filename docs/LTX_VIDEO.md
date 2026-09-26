# Native LTX video

The `nrob-diffusion` worker implements text-to-video and image-to-video with optional start/end images for distilled
LTX 2.3, LTX 2.5 and Sulphur-2 checkpoints in Rust. Clips also get a
soundtrack generated with the picture (see [Audio](#audio)). Candle supplies tensor
operations and CUDA kernels. Gemma text encoding, the video transformer,
eight-step Euler sampling and the convolutional VAE run inside the worker.
FFmpeg only encodes the decoded pixels into an H.264 MP4; it does not run models.
The server and inference core remain std-only and forbid unsafe Rust.

Sulphur-2 and LTX 2.5 have been validated end to end locally in both text-to-video
and image-to-video modes. The original LTX 2.3 checkpoint is still downloading;
Sulphur validates the shared LTX 2.3 architecture.

## Models

| Model | Diffusion weights | Text encoder | Decoder |
|---|---|---|---|
| [LTX 2.3](https://huggingface.co/Lightricks/LTX-2.3) | `ltx-2.3-22b-distilled-1.1.safetensors` | Gemma 3 12B BF16, `model.*` tensor layout, with `tokenizer.json` | Included in checkpoint |
| [Sulphur-2](https://huggingface.co/SulphurAI/Sulphur-2-base) | `sulphur_distil_bf16.safetensors` | Same Gemma 3 encoder | Included in checkpoint |
| [LTX 2.5](https://huggingface.co/Lightricks/LTX-2.5) | `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors` | `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, including its tokenizer | `vae/ltx-2.5-video-vae-conv-bf16.safetensors` |

Use the complete distilled checkpoints. The worker reads published safetensors
directly, without conversion:

- **Weight types:** BF16, F16 and F32. It also reads fp8 (`F8_E4M3`) and int8 (`I8`)
  weights, with or without the per-tensor (or per-row) `weight_scale` that
  ComfyUI's scaled checkpoints store beside each weight. These keep their stored
  size on disk and in the RAM tier, and become BF16 on the GPU, so a 21 GB fp8
  transformer streams from SSD at half the bytes of its BF16 original.
- **Key names:** transformers saved with bare names (`patchify_proj.weight`
  rather than `model.diffusion_model.patchify_proj.weight`) load as they are.
  The `model_version` and `config` metadata must still be there.

The worker does not apply video LoRAs (merged checkpoints are fine), and does
not interpret NVFP4 or GGUF video weights. It does not run the LTX 2.5
diffusion-decoder VAE (`CausalDiffusionVAE`, with `decoder.diff_blocks.*`);
the conv VAE decodes the same latents. Latent upscalers that ship with a
release are not used: clips render in one pass at the requested size.

For example, the merged
LTX 2.5 v1.1 fine-tune
release runs as `diffusion_models/ltx25_v1.1-fp8_scaled.safetensors`
with `text_encoders/gemma4_12b_ltx25-int8.safetensors` (int8, with
its tokenizer and LTX projection). Its VAE is the diffusion-decoder one, so pair
it with the official `ltx-2.5-video-vae-conv-bf16.safetensors`. The studio
does this itself when another LTX 2.5 model is already configured.
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
accepts optional `image` (starting frame) and `end_image` (final-frame guidance),
each containing one absolute local image path. Omit both for text-to-video,
provide just `image` for ordinary image-to-video, or supply both endpoints.
An end image alone is also accepted. General reference-image arrays are not
supported, and audio is generated rather than accepted as an input.

For image-to-video, add
`"image": "E:/images/fox.png"` to the request. The worker center-crops and resizes
the image to the requested dimensions, encodes it with the native VAE, and holds
its first-frame latent fixed throughout sampling. Conditioned tokens receive a
zero timestep; generated tokens receive the current noise timestep. The first
output frame is a VAE reconstruction of the image, so pixel-perfect reproduction
is not guaranteed. PNG, JPEG and WebP are supported, with a 32 MiB input-file limit
and bounded decoded-image allocation. Existing local-file policy applies: enabled
by default on loopback, disabled on non-loopback listeners unless explicitly
enabled with `--local-images on`; `--local-images off` disables it everywhere.

For start/end guidance, add both paths:

```json
{"action":"generate","model":"sulphur-2","prompt":"The fox walks through the snow.","image":"E:/images/start.png","end_image":"E:/images/end.png","width":512,"height":320,"frames":49}
```

The end image goes through the same crop, resize and VAE encoding. Its clean
latent tokens are appended at the final frame's timestamp and guide attention
during every sampling step, then are removed before decoding. The last ordinary
video latent remains generated because it represents eight frames. This follows
the official LTX keyframe conditioner; final-frame resemblance is guidance,
not a pixel-exact constraint. Both images share a single VAE encoder load, though
an end image adds one encode and more attention tokens. LTX 2.5 also applies its
learned first-frame keyframe embedding, including in text-to-video mode.

Repeated prompts reuse the final text conditioning by default, including across
text-to-video and image-to-video jobs with different seeds or dimensions. The
cache is keyed by the exact prompt, model, weight/tokenizer paths, sizes,
modification times and cache format version. Its eight entries occupy roughly
128 MiB under `output_root/.ltx-prompt-cache`, with least-recently-used eviction.
No model weights are duplicated. Set `"prompt_cache": false` to bypass it; deleting
this cache is safe. A cache miss or corrupt entry simply recomputes conditioning.
The standalone worker enables this cache only when given `cache_dir`.

Completed jobs return an MP4 path, a middle-frame PNG preview, dimensions,
frame rate, seed, stage timings, `prompt_cache_hit`, and weight-residency statistics. A JSON sidecar
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
measures free VRAM before keeping each block on the GPU. A block stays only
while room remains to stream three more of its size, plus 2 GiB for
activations. Everything else is served from RAM or the checkpoint, so `auto`
adapts to whatever card it runs on. With a 4 GiB weight budget, a 2-second
LTX 2.5 clip with sound peaked at about 8 GiB of VRAM (CUDA context and
allocator cache included). An 8 GiB card handles short clips; 12 GiB is
comfortable. OS file caching may service SSD reads
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
text encoding remained a substantial part of total time. (These clips were
made before audio support.)

A repeated-prompt image-to-video run after these changes took 17.7 seconds for
512×320, 49 frames on the second RTX 5090: 0.51 seconds for the starting image,
0.010 seconds for cached prompt conditioning, 15.3 seconds for denoising, and
1.06 seconds for decoding. This is a warm file-cache measurement with concurrent
model downloads; new prompts still require Gemma and the text connector.
Cached and uncached conditioning produced byte-identical MP4s at the same seed.
LTX 2.5 also generated both modes locally; a cached 512×320, 49-frame text-to-video
run took 20.4 seconds. Its first Gemma 4 prompt was substantially slower (about
70 seconds for conditioning during concurrent compilation).
The worker also uses fused RMS normalization and reads only the prompt's Gemma
vocabulary rows, avoiding the complete embedding-table upload on cache misses.

A start/end Sulphur run at the same 512×320, 49-frame settings took 19.8 seconds
with cached prompt conditioning. Both Sulphur and LTX 2.5 start/end clips were
checked locally for endpoint resemblance and motion between endpoints.
A separate same-seed comparison took 16.1 seconds with only a starting image
and 17.0 seconds with both images. Final-frame PSNR against the requested end
image improved from 12.6 dB to 31.1 dB with end guidance. This is one local
example, not a quality guarantee for arbitrary image pairs.

The reported `gpu_weight_bytes`/`ram_weight_bytes` refer to transformer block
residency. `weight_bytes_read` counts checkpoint bytes requested by that store,
including conditioning projections/connectors; it is not physical disk traffic
and excludes the separate Gemma/VAE readers.

## Audio

LTX 2.3, LTX 2.5 and Sulphur are audio-video models: every transformer block
carries an audio stream (2048 wide, 32 heads of 64) next to the video stream. The two streams
attend to each other both ways. When the request names the model's `audio_vae`,
the worker generates both streams together:

- **Text:** the `audio_aggregate_embed` projection and the transformer's
  `audio_embeddings_connector` produce the audio prompt context. For LTX 2.5
  the projection is in the Gemma 4 file; for LTX 2.3 and Sulphur it is in the
  checkpoint. It is cached
  alongside the video context.
- **Latent:** a noise latent of 25 frames per second of video, 128 wide,
  denoised on the same eight-step schedule in the same transformer calls. A
  starting or ending image conditions only the picture.
- **Decoding:** the audio VAE decoder produces a stereo log-mel spectrogram. The
  vocoder turns it into 16 kHz audio, and its bandwidth extension lifts that to
  48 kHz. The result is trimmed to the clip's exact length and muxed as stereo
  AAC. All of this runs in F32, as the reference does.

Request fields:

| Field | Meaning |
|---|---|
| `audio_vae` | A file with `audio_vae.*` plus `vocoder.*` (with its bandwidth extension). LTX 2.3 and Sulphur checkpoints carry these, so for them it is the checkpoint itself; LTX 2.5 ships it separately. The audio VAEs of all three are the same layout, and in the releases tested, the same weights. |
| `audio` | `true` or `false`. It defaults to on whenever `audio_vae` is given. |

The result reports `audio`, `sample_rate` and `audio_seconds`.

Audio adds about 44% to each block's weights (5.6B parameters in all), and the
VRAM checks above count them. Compute grows by about 12%. Decoding 4 seconds of
audio takes about 0.6 s on an RTX 5090. A 4-second 768×512 clip with sound took
70 s end to end there, 26 s of it denoising, with 22 GB of weights resident and
7 GB streamed from RAM.

Prompts can describe sound as well as picture: speech in quotes, sound
effects, "unscored" for no music. Fine-tunes trained with an in-context LoRA
may document `[VISUAL]`, `[SPEECH]` and `[SOUNDS]` sections.

Limits: the official LTX 2.5 pipeline samples its first stage ancestrally and
refines at a second stage. This worker is single-stage Euler (as for video),
which may cost some audio fidelity.

## Verification

`cargo test --workspace` covers request constraints, trusted path selection,
memory-budget limits and convolution boundary behavior. CUDA integration tests
exercise the real convolutional VAE. Opt-in reference tests compare Gemma 3
local/global attention and a Sulphur video block against the official PyTorch implementations;
the block test also requires identical RAM-cache and SSD-read results.
PyTorch is only an offline verification tool, never an inference dependency.

Audio has its own references, from `tools/ltx/audio_reference.py` run against
the official checkout (strict F32, TF32 off). The opt-in tests are
`ltx::audio::tests` (audio VAE file) and the `audio_*` tests in
`ltx::transformer::tests`. Measured relative RMS errors:

| Stage | Error |
|---|---|
| Audio VAE decoder | 3e-7 |
| Vocoder | 9e-6 |
| Vocoder with bandwidth extension | 1e-5 |
| First audio-video block (video / audio) | 0.11% / 0.05% |
| Two-block velocities (video / audio) | 0.41% / 0.26% |
| Audio text connector | 0.45% |

The last three run the official BF16 transformer as the reference, cut to its
first two blocks. The same tests against the LTX 2.3 checkpoint:

- first block: 0.03% (video), 0.07% (audio);
- velocities: 0.37% (video), 0.28% (audio);
- connector: 0.50%.

The audio VAE inside the LTX 2.3 checkpoint decodes identically. The video-only path is unchanged by the audio work: the
existing video tests produce identical numbers before and after.

Quantized weights are covered separately:
- Unit tests decode fp8 and int8 with scalar and per-row scales, from the RAM
  tier and from disk.
- The fp8 table matches the reference conversion for all 256 byte values.
- An opt-in CUDA test checks that the GPU decode matches the CPU bit for bit.

On 2026-09-26 the LTX 2.5 v1.1 fine-tune (fp8_scaled transformer,
int8 Gemma 4, official conv VAE) was run on an RTX 5090 with `memory: auto`.
It produced coherent 2-second 768x512 clips at 24 fps: text-to-video in 25 s,
and image-to-video from a still in 21 s.

Generate the reference activations with `tools/ltx/gemma_reference.py --gemma
<gemma-file>` and `tools/ltx/dit_reference.py --checkpoint <distilled-file>
--source <official-LTX-2-checkout>`.
Both accept `--device cuda:0` and `--output target/ltx-golden`.

The complete 48-layer text-feature reference uses
`tools/ltx/gemma_full_reference.py --gemma <gemma-file> --config <Gemma-3-config.json>`.
The complete video-transformer reference uses
`tools/ltx/transformer_reference.py --checkpoint <distilled-file> --source
<official-LTX-2-checkout>` and needs about 26 GiB of free GPU memory.
For the starting-image reference, run it again with `--conditioned-tokens 4`.
For start/end conditioning, use `--end-frame`, which constructs the official
keyframe conditioner and compares generated-token velocities and hidden states.
Use separate output directories for different model checkpoints.
The starting-image encoder reference uses the VAE script with `--encoder` and
the Sulphur checkpoint.
Gemma 4 uses `tools/ltx/gemma4_reference.py --gemma <Gemma-4-file>` with both
`--mode blocks` and `--mode features`. It requires Transformers 5.17 or newer;
`--reference-deps` can point to an isolated installation for these offline checks.
The VAE reference uses `tools/ltx/vae_reference.py --vae <vae-file> --source
<official-LTX-2-checkout>`. It requires the official package's test dependencies
and also accepts an isolated dependency directory through `--reference-deps`.

On the local RTX 5090 checks, complete Gemma 3 features differed from the official
BF16 encoder by about 0.7% relative RMS; the complete Sulphur video transformer
differed by 3.4% after fused normalization. The synthetic mixed-timestep
image-conditioning check differed by 2.7% before the final projection and 7.8%
in generated-token velocities; the first-frame VAE encoder differed by 0.97%.
Start/end conditioning checks measured 7.9% generated-token velocity error for
Sulphur and 11.2% for LTX 2.5 on synthetic inputs. LTX 2.5's first block differed
by 0.18% and its final hidden state by 4.28%; the tests bound both early and
accumulated differences rather than only checking final velocities.
LTX 2.5 convolutional VAE pixels differed by 1.44%.
These compare numerical components, not end-to-end video quality.

Set `NROB_LTX_GOLDEN`, `NROB_LTX_GEMMA`, `NROB_LTX_GEMMA4`, `NROB_LTX_CHECKPOINT`, `NROB_LTX_VAE`,
and optionally `NROB_LTX_TEST_DEVICE` before running the ignored tests:

```sh
cargo test --release -p nrob-diffusion --features flash-attn --lib ltx:: -- --include-ignored --test-threads=1
cargo test --release -p nrob-diffusion --features flash-attn --test ltx -- --include-ignored --test-threads=1
```
