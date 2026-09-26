# Live media catalog

`--image-config` accepts either the legacy image JSON file or a directory of
media manifests. Copy `config/media.example` to a local folder, edit the paths,
then pass that folder. The configured local folder is `E:/deepseek/nrob/media`.
Model weights stay at their existing paths; nothing is copied or converted.

| File | Purpose | When read |
|---|---|---|
| `controller.json` | Worker executable, output folder, controller name/path and GPU devices | Server startup |
| `image.json` | Qwen Image base components, diffusion weights, turbo adapter and default weight choice | Status and every image request |
| `video.json` | Supported distilled video checkpoints, encoders, VAE, FFmpeg, cache budgets and default model | Status and every video request |

Paths inside these files may be absolute or relative to the catalog/manifest.
The files are data, not executable plugins. Supported architectures remain
Qwen Image 2.1 and distilled LTX 2.3/2.5/Sulphur. Adding a different architecture's
weights does not implement that architecture.

An absent, invalid or `"enabled":false` image/video manifest disables that
capability and reports its error in status. It does not prevent the controller
from starting. Add a valid file or set `enabled` to true to make it available
again. Media weights load only after a generation request and are released when
the worker exits. An already queued job retains its prepared settings.

Changing `controller.json` requires restarting the server. Coder-cli notices
controller-file changes when it next starts/reuses its daemon; media-file changes
do not restart the daemon or reload the language model.

`image.json` uses `default_weights` (`gguf` or `safetensors`). The local default
is `safetensors`, pointing to `E:/stuff/REDQwen21.safetensors` through
`safetensors_transformer`; six-step turbo is unchanged. `video.json` uses
`default_model`; the local default is `sulphur-2`, pointing to
`sulphur_distil_bf16.safetensors` and using eight steps. Requests can still select
another configured option explicitly. The legacy file defaults remain GGUF and
LTX 2.3 when no default is configured.

`GET /v1/images/status` and `/v1/videos/status` include a `media` object containing
availability, defaults, image checkpoint path and video weight readiness.
Reading status loads no model weights. Coder-cli reads these same configuration
rules locally before each turn to expose only the available generation tools.

The controller name must match its entry in coder-cli's `[nrob.models]`.
Use different controller and media GPU devices. DeepSeek should be selected only
for explicit language-model requests or reasoning help after active media has
finished; media jobs keep the lightweight controller selected until release.

## Multiple image models and optional encoders

`image.json` accepts a `models` object and a `default_model`. Top-level base,
GGUF transformer, adapter and default weight choice are shared; each entry can
override them and can set `text_encoder` to an alternate Qwen3-VL safetensors file.
The processor/tokenizer and architecture config still come from `base`.

```json
{
  "enabled": true,
  "base": "D:/Qwen-Image-2.1",
  "adapter": "../models/qwen-image-2.1/viggle-v0.2.1-r128.safetensors",
  "default_weights": "safetensors",
  "default_model": "redqwen21",
  "models": {
    "redqwen21": {
      "safetensors_transformer": "E:/stuff/REDQwen21.safetensors",
      "text_encoder": "E:/stuff/qwen3vl_8b_w4a8.safetensors"
    },
    "realism": {"safetensors_transformer": "E:/stuff/RealismQwen21.safetensors"},
    "realism-w4a8": {
      "safetensors_transformer": "E:/stuff/RealismQwen21.safetensors",
      "text_encoder": "E:/stuff/qwen3vl_8b_w4a8.safetensors"
    }
  }
}
```

Use `image_generate` or `POST /v1/images/generations` with
`{"model":"realism","prompt":"A red panda on a mossy log"}`.
Omitting `model` uses REDQwen21 with the W4A8 encoder. Remove its `text_encoder`
override to use the original base encoder. `realism` uses that
original encoder; `realism-w4a8` selects the quantized encoder. Requests select trusted names; they cannot
supply checkpoint/encoder paths. Status lists named models and readiness.
Entries can be disabled with `enabled:false`. A flat legacy image manifest
still works. Reference images remain optional: zero for text-to-image, up to
three for editing/composition. The selected encoder is used for both text and
reference-image conditioning. Output manifests record the actual checkpoint,
encoder and model name.

The realism checkpoint is FP16 with fused gate/up MLP matrices. The loader reads
the two halves in place and applies turbo adapters using their original logical
layer names. REDQwen21 is BF16 with separate matrices; both use six-step turbo.

The optional encoder contains Comfy Kitchen `asym_w4a8_int8` matrices with
16-element FP8 scale groups, codebooks and ConvRot, plus rotated per-row INT8
embeddings. NROB decodes these in Rust into BF16 compute tensors in memory;
it does not yet execute fused W4A8 activation kernels. It therefore saves disk
space but does not retain 4-bit VRAM usage. No converted weights are written.
Quantized encoder provenance/censorship cannot be established from its metadata.
Format references: [Comfy Kitchen decoder](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/backends/eager/w4a8_int8.py)
and [rotation definition](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py).

Local smoke tests generated 768x768 red-panda images with both encoders at six
steps: sampling plus decoding took approximately 2.2-2.3 seconds per image.
Cold loading is additional; submit a batch to amortize model loading across
images. Timings vary with disk caching and concurrent GPU workloads.

## Weight residency

`image.json` (top level, or per model) accepts `memory` (`auto`, `gpu`, `ram`,
`ssd`), `ram_gb` and `vram_gb`, and `video.json` accepts the same keys. The
catalog values are defaults and caps: a request may name another mode, or less
RAM or VRAM, but never more. Qwen Image places each of its transformer blocks
and text-encoder layers on the GPU, in RAM or on the SSD; SDXL stages by
component. Without these keys, image jobs run `auto` with a 32 GiB RAM cap and
whatever VRAM is free. See [STUDIO.md](STUDIO.md#memory-ssd-ram-and-gpu).

## Project-owned output

Coder-cli supplies `Options.media_output_root` for the open project's
`.coder-cli/media` directory. This trusted host setting overrides the shared
controller manifest's `output_root`. Standalone NROB accepts the same override
as `--media-output-root DIR`. Relative output folders and LTX prompt caches
remain confined to that root. The worker writes images/videos and manifests
there directly, without copying them to a central output directory.
