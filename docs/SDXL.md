# SDXL image generation

Nrob's opt-in `nrob-diffusion` worker supports complete SDXL 1.0 base-compatible
single-file safetensors checkpoints, including Illustrious derivatives. It reads
the original checkpoint in place: UNet, CLIP-L, OpenCLIP-G and VAE must all be
present. The architecture code was ported from the author's `plugin-diffusion`
Rust implementation. Inference is native Rust/Candle; no Python service is used.

Build with `tools/qwen-image/build.ps1` on Windows (CUDA/Flash Attention), or
`cargo build --release -p nrob-diffusion --features flash-attn` with the CUDA
toolchain configured. The existing [media controller setup](QWEN_IMAGE.md) and
image API supervise SDXL jobs too. Restart the server/coder-cli after rebuilding.

## Catalog

Add a named entry to the `models` object in your media directory's `image.json`:

```json
"sdxl-checkpoint": {
  "architecture": "sdxl",
  "checkpoint": "E:/stuff/sdxlCheckpoint_v90.safetensors",
  "tokenizer": "E:/models/sdxl/clip-tokenizer/tokenizer.json",
  "steps": 16,
  "cfg": 2.5,
  "sampler": "dpmpp_2m",
  "scheduler": "karras",
  "clip_skip": 1
}
```

Keep the existing `default_model`, or explicitly set it to this entry to change
the image default. This does not change the language model in settings.toml.
Catalog paths may be absolute or relative to the media directory. SDXL does not
inherit the Qwen transformer, encoder or turbo adapter. Status advertises each
model's architecture, readiness and SDXL defaults. The worker and server validate
requests before starting inference; the server never accepts checkpoint paths
from tool calls.

Use the official CLIP tokenizer JSON from
[OpenAI CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14/blob/32bd64288804d66eefd0ccbe215aa642df71cc41/tokenizer.json).
`tools/sdxl/download-tokenizer.ps1` downloads that pinned file to
`E:/models/sdxl/clip-tokenizer` by default. Both encoders use its vocabulary;
CLIP-L pads with EOT, OpenCLIP-G with token zero.

## Agent usage

```json
{
  "action": "generate",
  "model": "sdxl-checkpoint",
  "prompt": "anime illustration, red fox beside a shrine, autumn leaves",
  "negative_prompt": "blurry, low quality, text, watermark",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

Call `image_generate` with this request, then poll `action: status` for progress
and saved paths. `n` reuses a prompt with consecutive seeds; `prompts` can contain
one prompt or exactly `n` prompts. Models load once per batch. Cancellation keeps
completed images. Coder-cli stores outputs and the worker's CUDA cache under the
selected project's `.coder-cli` media output directory. Direct worker tests use
the explicitly supplied `output_dir`.

The checkpoint's supplied instructions recommend **16+ steps, CFG 2.5+,
DPM++ 2M Karras** and explicitly say **no Lightning LoRA**. These are its catalog
defaults. You can override steps (2–100), CFG (1–30), negative prompt, seed and
dimensions (256–2048, multiples of 64). Eight to ten steps are an optional faster
quality tradeoff mentioned in the author's instructions.

## Current limits

Local RTX 5090 validation with this checkpoint generated and visually checked a
1024×1024 fox/shrine image at 16 steps, CFG 2.5: 6.1 seconds sampling, 3.0 seconds
decoding/PNG saving, 44.0 seconds for the full first worker run. A separate
two-prompt 1024×768 batch produced an anime wizard and lighthouse; its second
image took 4.3 seconds total (3.3 seconds sampling). Startup, weight loading and
first-use CUDA kernel preparation are paid once per batch. These are measured
examples, not latency guarantees.

- Text-to-image only; reference editing, inpainting, LoRA and SDXL Refiner are
  not implemented. Qwen image editing remains available through its own models.
- This worker exposes DPM++ 2M with Karras scheduling only.
- CLIP uses 75 content tokens plus start/end tokens. Longer prompts are truncated
  and reported as `prompt_truncated` in progress and the result manifest.
  Prompt weighting syntax and long-prompt chunking are not implemented.
- `clip_skip: 1` means the standard SDXL penultimate layer; `2` selects one layer
  earlier. UI conventions differ, so these are explicit worker semantics.
- UNet/text encoders use BF16 on CUDA; VAE decoding uses FP32 for stability.
  Peak memory increases with resolution. With `memory: "gpu"` (or `auto` when
  the checkpoint fits the VRAM cap) every component stays resident. With `ram` or
  `ssd` (or `auto` when it does not fit) the worker stages by component: the CLIP
  encoders, then the UNet, then the VAE each hold VRAM only while they run, with
  host copies kept up to `ram_gb`. The images are identical; see
  [STUDIO.md](STUDIO.md#memory-ssd-ram-and-gpu).
- Each batch saves PNG files and `manifest.jsonl` with prompts, seed, checkpoint,
  settings and load/sampling/decode timings. Different runtimes need not produce
  pixel-identical output from the same numeric seed.
