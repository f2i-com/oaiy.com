# Native FLUX.2 Klein 4B — archived initial checkpoint

**Updated:** native CUDA baseline, strength-1 LoRA image and byte-identical
strength-zero control have now succeeded. See [current validation and reproduction](KLEIN_VALIDATED.md).
The text below preserves the initial checkpoint and its inspection evidence.

This is a recoverable development checkpoint, **not a validated full-model image release**.
No baseline or LoRA image has been generated. No active desktop process was restarted,
unloaded, or terminated. No E: checkout was edited and no production weights downloaded.

## Isolation and implementation

Branch: `codex/native-flux2-klein4b`, based on OAIY
`36047c9002deaf465073f21d2857dd43eaae88d6`.
Checkout: `C:\Users\User\Documents\Codex\2026-09-30\task-2\oaiy-klein-native`.

The runtime is Rust/Candle in `crates/oaiy-media/src/klein/`:

- `transformer.rs`: original BFL safetensors layout, hidden 3072, 24 heads,
  five double streams, twenty single streams, global modulation, gated SiLU,
  adjacent-pair four-axis RoPE, joint bidirectional attention, final adaptive norm.
  Reads published files directly; shares OAIY's GPU/RAM/disk block residency.
- `text.rs`: Qwen3-4B, GQA 32/8 with head dimension 128, disabled-thinking chat
  template, 512 right-padded tokens, causal/padding masks, hidden-state tuple
  indices 9/18/27 concatenated to width 7680. Layers after the last tap are unused.
- `vae.rs` / `math.rs`: Flux2 32-channel VAE, original and diffusers VAE key
  layouts, optional quant/post-quant projections, 2x2 channel patches, 128-feature
  batch-normalization statistics with epsilon 1e-4, grid/token packing.
  Reuses OAIY's existing SDXL VAE building blocks; the VAE stays F32.
- `schedule.rs`: BFL empirical sequence-length/step-dependent exponential time
  shift, forward-velocity Euler update. Distilled recipe is four steps, CFG 1.
  The same 4B architecture accepts explicitly selected base weights, with CFG;
  base end-to-end execution is also unvalidated.
- `mod.rs`: strict request parsing, staged conditioning/denoising/decoding,
  reproducible OAIY noise, PNG output, JSON events, per-image manifest.
- Existing runtime LoRA factors are reused in `weights.rs` / `lora.rs`.
  New preflight checks exact projection dimensions and rejects foreign keys.
  No destructive weight merge or converter is required.

Worker dispatch, server catalog/request preparation, Studio validation, component
detection/attachment, discovery, and model fields recognize `flux2-klein-4b`.
Configured style LoRAs are forwarded; `use_loras:false` selects a clean baseline.
This checkpoint implements text-to-image only. Reference images and negative
prompts are rejected; 9B and FLUX.2 dev are outside the scope.

## Evidence so far

Run from the isolated checkout:

```powershell
cargo test --offline --locked -p oaiy-media -p oaiy-studio -p oaiy-llm-server --target-dir '..\klein-target' -- --test-threads=1
```

240 tests passed, no failures, 49 ignored across unit/integration tests. Logs:
`..\klein-cpu-tests.log`.

- A complete tiny native transformer matches an independent invocation of BFL's
  published forward pass within 2e-5 absolute error, across resident/RAM/disk
  modes. This tests real native arithmetic, **not full-model image quality**.
- A native Qwen3 decoder layer matches Hugging Face's independent reference,
  including GQA and right padding, within 2e-5 absolute error.
- Request tests exercise catalog selection, fixed distilled recipe, LoRA/baseline
  dispatch, and rejection of unsupported image/negative-prompt inputs.
- Actual local LoRA: all 160 tensors load on CPU as 80 finite rank-32 pairs;
  every module fits Klein 4B and is used. A 4096-wide configuration is rejected.
- Actual local Flux2 VAE: native CPU load and 16x16 encode/decode produce finite
  values and the expected `(1,1,128)` packed latent. No image artifact was saved.
  These two explicit tests passed in `..\klein-local-components.log` (4.16 s).

Reproduce the last two component tests:

```powershell
$env:OAIY_KLEIN_LORA='E:\stuff\SimpleFineVector_F2K4B_v1.safetensors'
$env:OAIY_KLEIN_VAE='E:\models\comfyui\models\vae\flux2-vae.safetensors'
cargo test --offline --locked -p oaiy-media --target-dir '..\klein-target' klein:: -- --ignored --test-threads=1
```

The fixture generator `tools/klein/reference.py` is a **CPU test tool only**.
The runtime never invokes Python. Checked-in fixtures avoid a Python requirement
for ordinary Rust tests. BFL source SHA256 used by the oracle:
`e062691d789ec80a04a5fbc421b09d3b7e264305be1dfcb6da65d68c259e18c1`.

The full workspace compiled during a test attempt, but the run was interrupted
before CUDA package tests, to preserve the pending controlled GPU window. It is
not recorded as a full workspace test pass. Two existing Studio tests share a
temporary directory; parallel testing failed with a file-not-found race, whereas
the serial run above passed. Their source was left unchanged.

## Build and weights blockers

CPU build succeeded. CUDA build command attempted:

```powershell
$env:CUDA_COMPUTE_CAP='120'
cargo build --offline --locked -p oaiy-media --features cuda --target-dir '..\klein-cuda-target'
```

It failed in `candle-kernels`: `nvcc` cannot find `cl.exe` in PATH.
CUDA 12.8 is installed. Resume in an MSVC development environment or set a
process-local `NVCC_CCBIN` to the installed compiler, with its required SDK/MSVC
environment. Log: `..\klein-cuda-build.log`. No persistent environment changes
were made. The CUDA worker has **not** compiled or executed yet.

Available local inputs:

| Input | Path / bytes |
| --- | --- |
| Qwen3-4B BF16 | `E:\models\comfyui\models\text_encoders\qwen_3_4b.safetensors`, 8,044,982,048 bytes |
| Flux2 VAE F32 | `E:\models\comfyui\models\vae\flux2-vae.safetensors`, 336,213,556 bytes |
| Style LoRA BF16 | `E:\stuff\SimpleFineVector_F2K4B_v1.safetensors`, 92,426,816 bytes |
| Candidate shared Qwen3 tokenizer | `C:\Users\User\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca\tokenizer.json`, 11,422,654 bytes |

The cached tokenizer is a candidate, not yet compared byte-for-byte with the
official Klein tokenizer. Native loading checks the three required special IDs.

**Missing:** `black-forest-labs/FLUX.2-klein-4B/flux-2-klein-4b.safetensors`.
Official file size: **7,751,105,712 bytes** (~7.75 GB / 7.22 GiB).
The official tokenizer is ~11.4 MB. The official repository is ungated and labels
this 4B variant Apache 2.0; it does not require a new account or gated acceptance.
Observed repository revision: `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
See [official model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
No large download has started; coordinate destination/authorization with the user.

Read-only GPU checkpoint: two RTX 5090s, 32,607 MiB each; GPU 0 had 30,475 MiB
free and GPU 1 had 28,142 MiB free. This is a snapshot, not a reservation.
An OAIY voice desktop process was active. Coordinate a fresh native test window
before GPU use or desktop model unloading; recheck resources at that time.

## Next steps

1. Resolve the process-local MSVC environment and compile the isolated CUDA worker.
2. Add direct server Klein catalog tests, worker request validation tests, and
   final UI controls/recipe validation; check worker help and source formatting.
   Current Studio route and classifier tests already pass.
3. Complete workspace checks when their GPU tests are authorized (or explicitly
   report a CPU-only subset). Do not relabel skipped tests as passing GPU tests.
4. Obtain the missing official transformer after the reported download details
   are accepted, and verify its published LFS hash. Verify the tokenizer.
5. During an approved isolated GPU window, generate a native baseline at 512x512,
   seed 747, four steps, CFG 1, then the identical request with the supplied LoRA
   at strength 1. Keep both PNGs/manifests, hashes, timing, logs, and GPU residency.
   Inspect output quality and demonstrate that the adapter changes the result.
6. Fix any full-model numerical/runtime issues, then package the final patch.

A standalone worker request uses:

```json
{
  "architecture":"flux2-klein-4b", "variant":"distilled",
  "transformer":"C:/PATH/flux-2-klein-4b.safetensors",
  "text_encoder":"E:/models/comfyui/models/text_encoders/qwen_3_4b.safetensors",
  "vae":"E:/models/comfyui/models/vae/flux2-vae.safetensors",
  "tokenizer":"C:/PATH/tokenizer.json",
  "prompt":"a simple clean vector illustration of a fox, flat colors, white background",
  "width":512, "height":512, "steps":4, "cfg":1, "seed":747, "device":0,
  "memory":"auto", "output_dir":"C:/PATH/isolated-klein-output",
  "loras":[{"path":"E:/stuff/SimpleFineVector_F2K4B_v1.safetensors","strength":1}]
}
```

For a baseline set `loras:[]`; then invoke the isolated CUDA worker with
`--request request.json`. Do not point a live desktop at the development binary.

## Earlier implementation inspection

The bounded read-only inspection of `E:\qwen-image2.1\plugin-diffusion` at
`582369f659b8a7b1ee4b457bc59f81bf1f2f0b60` found no Klein/FLUX2 transformer,
Qwen3-4B conditioner, Flux2 packing, model dispatch, or runtime Klein LoRA path
in source or 112 reachable commits. Shared FLUX1 math/comments are not a Klein
implementation. `native/src/inference.rs` dispatch contains Qwen Image, HiDream,
LTX and SDXL. Qwen Image20B/Edit/Plus2 Quanto, HiDreamO1 BF16/FP8, LTX2.3
Quanto/Comfy FP8/BF16 and SDXL including hires/adetailer are actual runtime paths;
Wan2.1 and SD1.5 recognition is not implementation. Low-level LTX NVFP4 loading
does not establish a wired public runtime variant.

Historical generation commits include `0ddf0df` (HiDream T2I), `629fdef` (edit),
`c9c029a` (SDXL hires/adetailer), and `3b2127e` (LTX smoke). These are historical
evidence, not runs repeated by this task. The inspector at
`native/src/inference.rs:2686` broadly treats `img_attn.qkv` keys as Qwen MMDiT;
that heuristic could mislabel this adapter. The user now believes earlier Klein
generations used ComfyUI, which is consistent with the absence of native code.

Adapter SHA256:
`135befa0ff25fa475b475d85747bb5808446eeba5a8eedf05335d07c652c8d2b`.
Metadata: `ss_base_model_version=flux2_klein_4b`, AI Toolkit 0.7.21, step747,
epoch1. All 80 A/B pairs are rank32, BF16, no alpha: five double blocks with eight
projections each and twenty single blocks with two projections each. Their
dimensions exactly match 3072-wide Klein 4B. Header identity does not prove which
base/distilled checkpoint trained the adapter. File offsets are valid and
contiguous; the header is 21,048 bytes. No code was executed from the adapter.

## References and licenses

Implementation follows the Apache-2.0
[BFL model](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/model.py),
[sampling](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py),
[text conditioning](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/text_encoder.py),
the official model configs, and Hugging Face's Qwen3 reference. Local references
are retained outside the checkout in `..\reference`. See `licenses/FLUX2-NOTICE.md`
and `licenses/Apache-2.0-FLUX2.txt`. No new runtime dependency was added.
