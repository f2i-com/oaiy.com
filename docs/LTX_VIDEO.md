# Native LTX video

The `oaiy-media` worker implements text-to-video and image-to-video with optional start/end images for distilled
LTX 2.3, LTX 2.5 and Sulphur-2 checkpoints in Rust. Clips also get a
soundtrack generated with the picture (see [Audio](#audio)). Gemma text encoding, the
video transformer, eight-step Euler sampling and the convolutional VAE run inside the worker.
FFmpeg only encodes the decoded pixels into an H.264 MP4; it does not run models.
The server and inference core remain std-only and forbid unsafe Rust.

> **WebGPU, since 2026-10-08.** The worker's one GPU backend is WebGPU, which a job
> gets when it names no backend: text-to-video and image-to-video, guided sampling, a
> negative prompt and LoRAs run there, and (2026-10-11) a clip's own sound, made with
> its picture: the transformer's audio stream beside its video stream, each block's two
> attending to each other, and the audio's text context, all on the GPU; the audio VAE
> and the vocoder are still Candle's on the CPU. On one RTX 5090 at its 400 W cap, LTX
> 2.3's distilled checkpoint (its one file given as `audio_vae` too), 512x320, 49
> frames: 62 s with sound where 36 s without (the eight steps 4.0 s where 3.1; the
> sound's decode on the CPU 16 s; the rest reading the checkpoint and Gemma). Against
> the reference's golden tensors (`tools/ltx/audio_reference.py`) the two streams' first
> two blocks give its velocities to 0.7% (video 0.0066, audio 0.0073 to 0.0077, with
> the audio denoised and with it frozen) and the audio connector its context to 0.7%,
> from Q8_0 weights where the reference is BF16. A soundtrack or speech to follow, a
> reference voice and two-stage refinement have no WebGPU path yet and are refused
> there; `backend: "cuda"` is refused by name, and Candle's CPU backend cannot multiply
> the model's BF16 weights. What this page says of those, of CUDA and of its timings
> describes the CUDA build, which stands on the branch `backup/cuda-support-2026-10-08`.

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
  ComfyUI Kitchen's `int8_tensorwise` weights with ConvRot (a `.comfy_quant`
  note saying `"convrot": true`) are rotated back after decoding, as Comfy does:
  each group of `convrot_groupsize` columns times the Hadamard matrix. Checkpoints
  such as the LTX 2.5 Stubelius Remix INT8 (the distilled LoRA already merged in,
  so it runs on the distilled 8-step schedule at CFG 1) load this way.
- **Key names:** transformers saved with bare names (`patchify_proj.weight`
  rather than `model.diffusion_model.patchify_proj.weight`) load as they are.
  The `model_version` and `config` metadata must still be there.

NVFP4 checkpoints load too: two E2M1 values a byte (high nibble first), an
fp8 scale per 16 in the cuBLAS tiled layout, and a tensor-wide
`weight_scale_2`; like fp8 they become BF16 on the device. A LoRA
(`lora_A`/`lora_B` pairs) is added to the weights it adapts as they are read.
The worker does not interpret GGUF video weights, and does not run the LTX 2.5
diffusion-decoder VAE (`CausalDiffusionVAE`, with `decoder.diff_blocks.*`);
the conv VAE decodes the same latents.

Clips render in one distilled pass by default. `guidance` (with a dev
transformer) and `refine` (the distilled transformer and a x2 spatial latent
upsampler) run the reference's two-stage pipeline instead: guided sampling at
half the size (30 steps; CFG 3 with the reference's negative prompt, STG on
one block, rescale 0.7, x0-space), the latent doubled, then three distilled
steps from sigma 0.909. In Studio a model's `dev_transformer` and
`spatial_upscaler` enable it and a request asks with `pipeline: "two_stage"`.
It did not make mouths follow a soundtrack measurably better and takes about
four times as long.

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

Use the same worker and server build as [Qwen Image](QWEN_IMAGE.md):

```sh
cargo build --release -p oaiy-media
cargo build --release -p oaiy-llm-server
```

Install FFmpeg with `libx264` support. Copy `config/ltx-video.example.json` to
`config/ltx-video.local.json`, set absolute checkpoint paths, and add its absolute
path as `video_config` in the existing image/controller configuration:

```json
"video_config": "E:/repos/oaiy/config/ltx-video.local.json"
```

Start OAIY with that `--image-config`, or use coder-cli's existing
`[oaiy].image_config` setting. The same lightweight controller handoff applies:
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
GPU memory; `release` permits the original language model again once all
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
| `auto` | Keeps every transformer block on the GPU in BF16 when that fits the VRAM budget, or else in INT8 when that fits (below); otherwise retains blocks up to the budget, caches overflow weights in RAM up to the host budget and reads the rest from their checkpoint as needed. |
| `gpu` | Requires the video transformer blocks to fit the VRAM budget; otherwise returns an error. |
| `ram` | Transfers each transformer block from a bounded host cache for GPU execution; overflow beyond the RAM budget is read from the checkpoint. |
| `ssd` | Keeps no host or GPU transformer-block cache; reads each block directly from the original checkpoint on every step. |

**INT8 on the GPU.** A 22B LTX transformer is about 44 GB in BF16; on a 32 GB
card `auto` used to keep half of it and stream the rest from RAM on every pass,
over a terabyte of uploads and device allocations per clip. When BF16 does not
fit but INT8 does, every block is kept on the card instead, its large 2-D
weights as symmetric per-row INT8 (one BF16 scale per output row; LoRAs merged
and fp8 or NVFP4 weights decoded first) and expanded to BF16 just before each
product. The transformer then takes about 18.6 GB, nothing streams, peak VRAM
fell from 28.1 to 22.4 GB, and 57-frame LTX 2.5 clips denoised in 29 s rather
than 35 s. The result reports `int8_weights`.

**Headroom.** The weight budget leaves 8 GiB of the card free and the video
decoder's workspace 4 GiB: on Windows, a card filled to the brim pages device
memory to system RAM, where kernels can stall into the driver's 2-second
watchdog (TDR). After a worker fails with a GPU fault (a driver error or
non-finite output), Studio waits 30 s before the next media job, so a job never
starts on a card whose driver is still recovering.

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

### Following a soundtrack

A clip can also follow a soundtrack it is given, instead of making its own: a
voice, a song, any audio. Mouths, movement and timing follow the sound, and the
clip keeps the original audio. This is the official `a2vid` pipeline's
conditioning:

1. The audio is read with FFmpeg, as stereo at its own rate (mono is doubled).
   It is resampled to 16 kHz with torchaudio's windowed-sinc filter.
2. It becomes a log-mel spectrogram: 64 slaney bins, hop 160.
3. The audio VAE's encoder turns that into the transformer's audio latent, at
   25 frames a second, cropped to the clip.
4. That latent is held through all the steps in one of two ways
   (`soundtrack_mode`):
   - `inpaint` (the default): at every step the latent is mixed with one fixed
     noise to that step's sigma, as if the audio were being generated with the
     picture, and put back again, so the model never changes it. The picture
     sees the audio as it was trained to: denoised together with it.
   - `frozen`: the reference's `a2vid` conditioning. The audio stream's sigma
     is 0 for its own timestep embeddings, its prompt modulation and its
     cross-attention scale and shift, and for the gate through which the
     video attends to the audio. The gate through which the audio attends to
     the video follows the video's sigma.

   Measured with face landmarks (inner-lip opening against the soundtrack's
   loudness, frame by frame), an LTX 2.5 fine-tune's lips matched
   a speech file at +0.31 on average with `frozen` and +0.50 with `inpaint`,
   over three seeds. The picture is denoised as usual, with start and end
   images if given.
5. The muxed audio is the input itself, cut to the clip, not a VAE round trip.

The audio VAE files of all three models include the encoder. Request fields:

| Field | Meaning |
|---|---|
| `audio_file` | An absolute path to any audio FFmpeg reads, up to 256 MiB (at most 60 s is read). |
| `speech` | A complete `kind: "speech"` worker request (see [Speech](SPEECH.md)). The worker speaks it first, frees the TTS model, and follows the result: a saved voice, an OpenAI voice name or a described one. |
| `transcript` | The words in `audio_file`, if known. They, or the speech's own text, are added to the prompt as `They say: "…"`. Without the words, the distilled models take little lip movement from the sound alone. |
| `a2v_guidance` | Audio-to-video guidance, 1 to 10 (1 turns it off). Each step also runs without the audio-video cross-attention, and the picture moves away from that result. This is the reference's modality guidance. Default 3 with guided (two-stage) sampling, as the reference; 1 with the distilled models, where it did not make mouths follow the words measurably better, deformed faces and doubled the denoising time. |
| `soundtrack_mode` | `inpaint` (default) or `frozen`, as above. |
| `frames` | Optional with a soundtrack. Without it, the clip is as long as the soundtrack: enough whole frames at `fps` to hold all of it, snapped up to 8k+1 (from 9 to 121; the reference snaps down, which cuts the last words off), the soundtrack padded with silence to match. |

A soundtrack needs `audio_vae` and keeps `audio` on. The result reports
`followed_soundtrack`, `audio_file` and `speech_seconds`.

### Lip-synced speech (ID-LoRA)

Following a soundtrack, the distilled models move lips loosely. With
[ID-LoRA](https://huggingface.co/papers/2603.10256) the clip's speech is
generated *with* the picture instead, in a given voice, and joint generation
keeps the lips in sync:

- The LoRA (`lora`: `{path, strength}`, as ID-LoRA and the LTX trainer save
  it) is added to the transformer's weights as they are read, `W + s B A`. The
  TalkVid ID-LoRA adapts only the audio stream and the audio-video
  cross-attention, so it also runs on LTX 2.5 and the fine-tunes.
- `reference_voice`, an audio file of the speaker (up to 10 s is used), is
  encoded by the audio VAE and leads the audio sequence as clean tokens
  (timestep 0) at negative times, ending one latent (40 ms) before the clip.
  The clip's own audio tokens start as noise and are denoised with the
  picture; only they are decoded.
- The prompt becomes ID-LoRA's (and the LTX 2.5 IC-LoRA's) form:
  `[VISUAL]: … [SPEECH]: "the words" [SOUNDS]: …`.
- With `identity: true`, the line (the `speech` made first, or `audio_file`)
  sets the clip's length and, without a `reference_voice`, is the voice; it
  is not followed as a soundtrack.
- `identity_guidance` (0, off, by default) runs each step again without the
  reference and pushes the audio toward the voice. ID-LoRA uses 3 with the dev
  model; on the distilled ones it crackled, and the reference alone carries
  the voice.
- The decoded speech is matched to the reference voice's loudness and kept
  under -0.3 dBFS; the vocoder's last step no longer hard-clips.

In Studio, a model with `id_lora` (and optionally `id_lora_strength`) speaks a
request's `speech` this way (`lip_sync`: auto, voice to re-speak an
`input_audio` with its `transcript`, or off). A saved voice's sample is the
reference. Discovery marks such models `lip_sync: true`. Measured with face
landmarks, the lips' best match with the loudness sat within one frame for
LTX 2.5 and Sulphur on every seed tried; a merged LTX 2.5 fine-tune's varied by
a few frames.

### Start and end images

Before encoding, a start or end image goes through one H.264 frame and back
(FFmpeg, libx264 veryfast, 4:2:0) at `image_crf`: 33 for the LTX 2.3
generation (Sulphur is one), 18 for LTX 2.5, 0 to skip, as the reference
pipelines do. The models were trained on video frames; a pristine still tended
to stay still (Sulphur's faces did not move their mouths without it).

On an RTX 5090, 5-second 768×512 talking-head clips took 166 s (LTX 2.3),
147 s (LTX 2.5) and 119 s (Sulphur), speech included. A 3-second Sulphur clip
following an MP3 took 66 s. The 121-frame limit caps a clip at about
5 seconds; a longer soundtrack is cut to the clip.

With no size asked for, Studio gives a clip with a start image the image's
shape, at 768×512's pixel count, so a portrait or square frame is not cropped
to landscape (the worker fills the size and crops what spills over).

### Negative prompts

`negative_prompt` names what a video should not show (watermark, text, logo,
extra limbs…). Studio joins the model's `negative_prompt` (a default for
all its videos) and the request's. With guided sampling (CFG above 1) it is the guidance's
negative prompt. Distilled models sample at CFG 1, where a negative prompt has
no effect, so there it steers by Normalized Attention Guidance (NAG, Chen et
al. 2025): each block's video text cross-attention also attends to the negative
prompt, and its output is pushed away from that one's (`scale`), its size held
within `tau` times the plain output's, then blended back by `alpha`. That is one
extra cross-attention per block rather than a second pass of the model. The
defaults are the reference's for video models; `nag: {scale, tau, alpha}` in the
request (or the model's config) tunes them (11, 2.5, 0.25). LTX was captioned
with tags such as `has_subtitles`; a watermark baked into a checkpoint went away
with `has_watermark, has_logo, has_text_overlay` in the negative prompt and
`nag: {tau: 3.5, alpha: 0.4}` (stronger steering also follows the prompt a little less).

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
| Resampler, 24 to 16 kHz | 1e-7 |
| Log-mel spectrogram | 6e-5 (the reference's own F32 spread on this signal is 5e-5) |
| Audio VAE encoder | 8e-7 |
| Two-block pass with frozen audio (video velocity / audio output), LTX 2.3 | 0.38% / 0.32% |

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

Set `OAIY_LTX_GOLDEN`, `OAIY_LTX_GEMMA`, `OAIY_LTX_GEMMA4`, `OAIY_LTX_CHECKPOINT`, `OAIY_LTX_VAE`,
and optionally `OAIY_LTX_TEST_DEVICE` before running the ignored tests:

```sh
cargo test --release -p oaiy-media --lib ltx:: -- --include-ignored --test-threads=1
cargo test --release -p oaiy-media --test ltx -- --include-ignored --test-threads=1
```

(Those of them that open a CUDA device were the CUDA build's and fail on main: run them
on that branch.)
