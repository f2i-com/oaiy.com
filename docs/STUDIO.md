# NROB Studio

`nrob-studio` is a portable host for running language, image and video models on
one machine with finite memory. It supervises the two engines, `nrob-server`
(language models) and `nrob-diffusion` (images and video), as subprocesses. It
serves a control UI and exposes one API whose routes you configure: paths,
methods, and the dialect each speaks.

It is std-only like the core (`#![forbid(unsafe_code)]`, no external crates), so
it builds and runs its UI on any machine. The engines need CUDA; see
[Current limits](#current-limits).

## A portable install

Put these in one folder:

```
nrob-studio.exe        the host (cargo build --release -p nrob-studio)
nrob-server.exe        language models (tools/qwen-image/build.ps1 builds both)
nrob-diffusion.exe     images and video (built with --features flash-attn)
nrob-studio.json       created with defaults on first start
```

Programs named without a folder (`"server": "nrob-server"`) are found beside
`nrob-studio`, then beside the configuration, then on `PATH`. Relative paths in
the configuration resolve against the configuration's folder. Generated media
goes to `outputs/` there, and prompt states to `cache/`. Copy the folder to
another machine and it runs there.

```sh
nrob-studio                      # opens the UI in a browser window without tabs
nrob-studio --open browser       # a normal browser tab
nrob-studio --headless           # console only; the UI still serves on its port
nrob-studio --config D:/ai/studio.json --ui-port 7000 --port 9000 --start-llm
```

The console accepts `status`, `start`, `stop`, `open`, `jobs` and `quit`.

## Adding models

On the **Models** page, choose **Browse** (a file picker over this machine's
drives) or paste a path, then choose **Check**. The studio reads the file's own
headers, never its weights, and decides what it is:

| Picked | Recognised by | Becomes |
|---|---|---|
| `.gguf` LLM | `general.architecture` (llama, qwen3, qwen35, gemma…) | a language model on the chat routes |
| `mmproj*.gguf` | `general.type: mmproj` / arch `clip` | the vision projector of a language model |
| Qwen Image `.gguf` | arch `qwen_image*` or `img_in` tensors | an image model (Q4 weights stream as Q4) |
| Qwen Image `.safetensors` | `img_in.*` + `transformer_blocks.*` | an image model |
| Diffusers folder | `model_index.json` `QwenImage*Pipeline` | an image model, or the *base* of one when it has no transformer weights |
| SDXL `.safetensors` | `modelspec.architecture` or `conditioner.embedders.*` | an SDXL image model |
| LTX 2.3 / 2.5, Sulphur | `__metadata__.model_version` and its `config` | a video model (2.3 files carry their own VAE) |
| LTX 2.5 ComfyUI folder | `diffusion_models/`, `text_encoders/`, `vae/` | a complete video model |
| LoRA `.safetensors` | `lora_A`/`lora_B` keys | the turbo adapter of a Qwen Image model |
| Text encoders | vocabulary size (Qwen3-VL ~152k, Gemma 262k) or `gemma_config` | the encoder of an image or video model |
| `tokenizer.json` | CLIP or Gemma special tokens | the tokenizer of an SDXL or LTX 2.3 model |
| EXL3 / DeepSeek folders | `config.json` | a language model |

Pickled `.bin`, `.pt` and `.ckpt` files are refused, because loading them can run code.

A model that still needs a part is added **disabled** and marked with what it
needs. The studio fills parts itself where it can:

- from another configured model of the same kind (a second LTX 2.3 model borrows
  the first one's Gemma encoder and tokenizer; a Qwen GGUF borrows a configured
  base folder and turbo adapter);
- from files near the picked one (a `clip-tokenizer/tokenizer.json` beside an SDXL
  checkpoint, a diffusers folder in a parent directory);
- later, when a part or complete model arrives that it lacks (adding
  `D:/Qwen-Image-2.1` after a Qwen GGUF completes the GGUF entry).

Adding a model also enables the gateway route that serves it, if that route had
been removed or turned off.

## Memory: SSD, RAM and GPU

The **Memory** page sets, for image jobs and video jobs separately, where
weights may live:

| Mode | Where blocks go |
|---|---|
| `gpu` | every block on the GPU for the whole job (fails if it does not fit) |
| `auto` | the GPU up to the VRAM cap, then RAM up to the RAM cap, then SSD |
| `ram` | host copies up to the RAM cap, uploaded block by block; the rest from SSD |
| `ssd` | only the block in use is off the disk |

The strip on that page shows where a model of a given size lands for the chosen
mode and caps. The worker applies the setting per block:

- **Qwen Image**: the 32 transformer blocks and the 36 Qwen3-VL text-encoder
  layers each settle on the GPU, in RAM or on the SSD
  (`nrob-diffusion/src/residency.rs`). GGUF blocks travel as their quantized
  bytes and are rebuilt on the GPU, never dequantized on the way. ConvRot/int8
  safetensors are decoded once when they settle in RAM, and again on every pass
  when streamed from the SSD.
- **SDXL**: the UNet's skip connections make block streaming awkward, so SDXL
  stages by component. The CLIP encoders, then the UNet, then the VAE each hold
  VRAM only while they run, with host copies kept up to the RAM cap.
- **LTX / Sulphur video**: the existing per-block residency
  ([LTX_VIDEO.md](LTX_VIDEO.md)).

Residency never changes the result (`tools/qwen-image/residency_parity.py`
checks it). With the same seed, `gpu`, `ram`, `ssd`
and a mixed `auto` produce byte-identical PNGs for Qwen Image (Q4 GGUF, 512²,
4 turbo steps) and for SDXL (a 2-image batch). Each job's result reports where
its blocks lived and how many bytes were streamed, and the Models page shows it
per model.

A request may ask for a different mode, or for less RAM or VRAM than the caps.
It may never ask for more.

### Sharing GPUs with the language model

`media.device` is the GPU media jobs use; `llm.devices` are the LLM's GPUs.
`media.llm_policy` decides what happens when they meet:

- `auto` (default): if the media GPU is one of the LLM's, the studio waits for
  running chat requests to finish, stops `nrob-server`, runs the job, and
  restarts the LLM once the queue is empty (`media.resume_llm`). Chat requests
  that arrive meanwhile wait, up to 15 minutes, and are then answered.
- `pause_llm`: always do that.
- `coexist`: never; both must fit.

`llm.idle_stop_minutes` stops an idle LLM so its RAM and VRAM return to the
machine. The next request loads it again.

## The API gateway

The **Endpoints** page edits `gateway.routes`. Each route has a `path`, a
`method`, a `target` and a `spec` (dialect):

| target | spec `openai` | spec `nrob` |
|---|---|---|
| `chat` | `POST` → nrob-server `/v1/chat/completions`, streamed through as it arrives | — |
| `completions` | `POST` → `/v1/completions` | — |
| `models` | the LLM's list plus image and video models (`type`: llm, image, video) | — |
| `images` | OpenAI Images (below) | nrob-server's job API: `202 {id, status_url}`, `GET …/status`, `POST …/cancel` |
| `edits` | OpenAI image edits (below) | — |
| `videos` | OpenAI Videos (below) | the same job API for video |
| `files` | generated media under the output folder | — |
| `health` | `{status, llm, media_busy}`, never behind the key | — |

Routes can be renamed, duplicated at other paths (the same service in both
dialects at once), turned off, or restored to the defaults. Saving refuses a
table that routes the same method and path twice. `gateway.api_key` requires
`Authorization: Bearer KEY` on everything but `health`. File links also accept
`?key=`, for `<img>` tags.

### OpenAI Images

`POST /v1/images/generations` with JSON:

| field | meaning |
|---|---|
| `prompt` | required |
| `model` | a configured image model; OpenAI names (`dall-e-3`, `gpt-image-1`) mean the default |
| `n` | 1–16 |
| `size` | `WxH` or `auto`; rounded to the model's grid (32 for Qwen, 64 for SDXL) |
| `response_format` | `b64_json` (default) or `url` (served from the `files` route; `gateway.public_url` sets the host) |
| extensions | `seed`, `steps`, `cfg`, `negative_prompt` (SDXL), `turbo`, `weights`, `memory`, `ram_gb`, `vram_gb` |

The reply is `{created, data: [{b64_json | url, revised_prompt}], output_format: "png", size, model}`.
Generation requests may also carry `images` (below) to edit; the nrob dialect accepts them too.

### OpenAI image edits

`POST /v1/images/edits`, as either:

- `multipart/form-data`, which the OpenAI SDKs send: one to three `image` (or
  `image[]`) files plus the text fields above (`prompt`, `model`, `n`, `size`,
  `response_format`, `seed`…);
- JSON with `images: [{image_url: "data:image/png;base64,…"}]` (or `image`, one or
  several; plain `data:` strings work too).

Qwen Image conditions on up to three references and edits from the instruction in
the prompt. `size: auto` (the default for edits) keeps the first image's aspect
ratio at about a megapixel (a 1920×1080 photo becomes 1376×768). Files are
recognised by their bytes (PNG, JPEG, WebP), not by their declared type. A `mask`
is refused rather than ignored, and SDXL models refuse edits: this worker's SDXL is
text-to-image only. Local paths are accepted only from the UI.

### OpenAI Videos

| request | reply |
|---|---|
| `POST /v1/videos` `{prompt, model?, seconds?, size?, input_reference?}` | a video object, `status: queued` |
| `GET /v1/videos/{id}` | the video object: `status` (`queued`, `in_progress`, `completed`, `failed`), `progress` 0–100 |
| `GET /v1/videos` | `{object: "list", data, first_id, last_id, has_more}` |
| `GET /v1/videos/{id}/content[?variant=thumbnail]` | the MP4, or its first-frame PNG |
| `DELETE /v1/videos/{id}` | `{id, object: "video.deleted", deleted}`; files stay on disk |

`seconds` becomes frames at `media.video.fps`, rounded to LTX's `8k+1` and capped
at 121. OpenAI sizes (`1280x720`, `720x1280`, `1792x1024`) scale down to fit LTX's
1024 limit and keep their aspect ratio (`1280x720` → `1024x576`).
`input_reference: {image_url: "data:image/png;base64,…"}` sets the starting frame.
Local paths are accepted from the UI only, and `file_id` is not supported.
Extensions: `frames`, `fps`, `seed`, `end_image`, `memory`, `ram_gb`, `vram_gb`.

## The control port

The UI port (`ui.port`, 7860) serves the page, a JSON API under `/api/`, the
playground under `/api/play/{chat,images,videos}`, and `/files/`. It can change
which programs the studio runs, so:

- it answers only requests addressed to it by name (`Host` must be this
  listener), so DNS rebinding cannot reach it;
- anything carrying an `Origin` must come from its own origin, so another web
  page in the same browser cannot drive it;
- if `ui.host` is not loopback and `gateway.api_key` is set, every API call needs
  that key (the page takes it once as `?key=`).

Keep `ui.host` on `127.0.0.1` unless you mean to control the machine remotely.

`nrob-server` runs on a private loopback port with a random key and is started
with `--watch-stdin`. The studio holds that pipe, so if the studio dies,
however it ends, the server exits and returns its GPU memory.

## Configuration reference

See [`config/studio.example.json`](../config/studio.example.json). The
sections are:

- `ui`: `host`, `port`, `open` (`app` | `browser` | `none`).
- `gateway`: `host`, `port`, `api_key`, `public_url`, `routes`.
- `llm`: `server`, `models` (`[{name, path, vision_projector?, lora?, enabled}]`),
  `default_model`, `devices`, `ctx`, `ram_gb` (expert cache, 0 = 80% of free),
  `cpu_threads`, `vram_headroom_gb`, `thinking`, `max_tokens`, `temperature`,
  `top_p`, `prompt_cache`, `prompt_cache_gb`, `vision`, `autostart`,
  `idle_stop_minutes`, `extra_args`.
- `media`: `worker`, `output_dir`, `device`, `llm_policy`, `resume_llm`,
  `keep_jobs`, and:
  - `image`: `enabled`, `default_model`, `memory`, `ram_gb`, `vram_gb` (null =
    free VRAM), and `models`. Each model has `architecture` (`qwen-image`: `base`,
    `transformer` or `safetensors_transformer`, `adapter`, `text_encoder`; `sdxl`:
    `checkpoint`, `tokenizer`, `steps`, `cfg`, `negative_prompt`, `clip_skip`),
    optional `memory`/`ram_gb`/`vram_gb`, `width`, `height`, `enabled`.
  - `video`: the same residency keys, `ffmpeg`, `fps`, and `models`, each with
    `family` (`ltx-2.3` | `ltx-2.5` | `sulphur-2`), `transformer`, `vae`,
    `text_encoder` and `tokenizer` (not for 2.5).

## Current limits

- The engines need CUDA: `nrob-server` has never built without it, and LTX video
  requires the CUDA worker. A CPU-only machine runs the studio, its UI and
  (slowly) CPU image jobs from a non-CUDA `nrob-diffusion`, but no LLM serving:
  `nrob` (the CLI) has CPU chat, not an HTTP server.
- One media job runs at a time; jobs queue. Only one language model is resident
  (nrob-server swaps on request).
- The job list lives in memory: a restart forgets jobs, not their files.
- Multipart is accepted on the edits route only; a video's `input_reference`
  goes as a `data:` URL in JSON. Edits take no mask.
- RAM-tier uploads use pageable host memory; pinned staging and prefetching the
  next block during compute are the obvious speedups not yet taken.

## Verified on this machine (2026-09-26)

On two RTX 5090s and 192 GB RAM, with the studio isolated on GPU 1 and its own ports:

- Qwen Image Q4 GGUF through `POST /v1/images/generations`: 512², RAM tier (all
  32 blocks and 36 encoder layers in RAM), 44 s including cold loads.
- Chat through the gateway: the first request started `nrob-server`
  (Qwen3.8 27B Q4 GGUF, loaded in 24 s) and streamed the reply.
- `POST /v1/videos` (Sulphur 2, 1 s at 512×320) while that LLM held GPU 1: the
  studio stopped the LLM, rendered the clip in 108 s, and restarted the LLM. A chat
  request sent mid-render waited 123 s and was then answered. `…/content` and
  `?variant=thumbnail` served the MP4 and PNG.
- A custom route `/my/sdxl` in the nrob dialect ran SDXL staged from the SSD.
- Killing the studio process left no `nrob-server` behind.
- `POST /v1/images/edits` as multipart (`curl -F image=@lighthouse.png -F prompt=…`)
  turned the sunset lighthouse into a snowy night with the beam lit, keeping the
  composition. A config written before the edits route existed gained it on load.
