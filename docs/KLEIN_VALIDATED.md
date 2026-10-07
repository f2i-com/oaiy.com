# Native Klein 4B validation

> **Note (2026-10-08).** The CUDA code and the CUDA figures on this page are the CUDA build's, which stands on the branch `backup/cuda-support-2026-10-08`: the engines' one GPU backend is WebGPU now ([WEBGPU.md](WEBGPU.md)).

This is the preserved original checkpoint. See [KLEIN_INTEGRATED.md](KLEIN_INTEGRATED.md)
for the latest pinned upstream integration, scoped native HTTP runs, auth checks,
and the independent desktop/Studio service credential contract.

Native Rust/Candle inference now produces real images with the official distilled
FLUX.2 Klein 4B transformer and the supplied SimpleFineVector LoRA. Both images
were visually inspected: coherent vector foxes, clean outlines, white backgrounds.
No Python subprocess performs inference. No E: checkout or active desktop was
changed, restarted, unloaded or terminated.

The implementation and archived source inspection are documented in
[KLEIN.md](KLEIN.md). Branch: `codex/native-flux2-klein4b`; base `36047c9`.

## Evidence

All runs use the same prompt, seed 747, 512x512, four steps, CFG 1, GPU 0.

Exact prompt: `a simple clean vector illustration of a fox, flat colors, crisp outlines, white background`.
Variant: distilled. OAIY's SplitMix64/Box-Muller noise is deterministic within
this runtime; the seed does not promise identical noise to PyTorch/ComfyUI.

| Run | Total seconds | Sampling seconds | PNG SHA256 |
| --- | ---: | ---: | --- |
| Clean baseline | 48.857 | 2.292 | `7a9b3c69528a3780de568735d15caa77c38ff29a266494f8d93bd839e8214544` |
| LoRA strength 1 | 26.128 | 5.686 | `adf7bfc84732abd741febc06cb3e7e0209f44d4f9ccdf9bd8319b21792e5a37d` |
| LoRA strength 0 | 23.591 | 0.988 | Identical to baseline |

The zero-strength control is pixel-identical **and PNG-byte-identical** to the
clean baseline. Strength 1 changes 110,847 pixels (42.3%); mean absolute channel
difference is 8.657/255. These are individual observed timings with different
warm file/kernel caches, not a controlled performance benchmark.

Both inspected images depict a coherent left-facing orange fox with white chest
and tail tip, black legs and outlines, on white. The adapter changes the muzzle,
chest fur, tail boundary and contours while preserving the composition. This
single prompt already requests vector art; the comparison demonstrates adapter
application, not a general quality or style improvement.

All 25 transformer blocks stayed on GPU. Adding LoRA increased resident tensor
bytes by exactly 92,405,760, the full payload of all 80 factor pairs. Memory was
released when each worker exited. GPU 0 had about 30 GB free before the runs;
no desktop model unloading was needed. GPU 1's existing use remained unchanged.

Preserved artifacts in `C:\Users\User\Documents\Codex\2026-09-30\task-2`:

- `klein-baseline.request.json`, `klein-style.request.json`, `klein-zero.request.json`
- Each run's `.result.json` and `.events.jsonl`, plus per-image `manifest.jsonl`
- `klein-comparison.json`: paths, hashes, parameters, timing, residency and pixel checks
- Baseline: `klein-images/klein-52212-1790771883888508900/image-0001-seed-747.png`
- Style: `klein-images/klein-51716-1790771991671766600/image-0001-seed-747.png`
- Zero: `klein-images/klein-3240-1790772130176600100/image-0001-seed-747.png`
- `klein-native-tested.exe`, SHA256
  `9e765219127fc14f69729ea0b5dbb91dd4f1f926c161852f32a3819536b7b579`
- GPU preflight/sample CSVs; download, compile and regression logs

Recheck PNGs with `python tools/klein/compare.py '..'` (Pillow/NumPy analysis only).

## Inputs and compilation

The approved official download is pinned to repository revision
`e7b7dc27f91deacad38e78976d1f2b499d76a294`. Both data files match published LFS hashes:

- `klein-weights/flux-2-klein-4b.safetensors`, 7,751,105,712 bytes,
  `ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343`
- `klein-weights/tokenizer.json`, 11,422,654 bytes,
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`

The cached Qwen3-0.6B tokenizer is byte-for-byte identical to the official tokenizer.
Pinned URLs, sizes and hashes are in `klein-weights/provenance.json`; the official
Apache 2.0 weight license is retained there. No credentials were used. Existing
Qwen3-4B, Flux2 VAE and supplied LoRA files remain at their original E: paths.

Additional input SHA256 hashes, calculated read-only:

- `E:\models\comfyui\models\text_encoders\qwen_3_4b.safetensors`:
  `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a`
- `E:\models\comfyui\models\vae\flux2-vae.safetensors`:
  `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5`
- `E:\stuff\SimpleFineVector_F2K4B_v1.safetensors`:
  `135befa0ff25fa475b475d85747bb5808446eeba5a8eedf05335d07c652c8d2b`

The existing encoder/VAE files' original download revisions are not established.
The adapter header names `flux2_klein_4b`, ai-toolkit 0.7.21, step 747; it has
80 BF16 rank-32 A/B pairs. Its training parent's base-versus-distilled identity
is not established from metadata; distilled runtime compatibility is demonstrated.

CUDA compilation succeeds with installed CUDA 12.8 and MSVC 14.44. The earlier
`cl.exe` error was resolved with **process-local** `vcvars64.bat`; no install or
global environment change was needed. From the isolated checkout:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_COMPUTE_CAP=120
cargo build --offline --locked -p oaiy-media --features cuda --target-dir "..\klein-cuda-target"
```

Ready reproduction commands, with an authorized idle GPU:

```powershell
& '..\klein-native-tested.exe' --request '..\klein-baseline.request.json'
& '..\klein-native-tested.exe' --request '..\klein-style.request.json'
```

For a fresh build, substitute `..\klein-cuda-target\debug\oaiy-media.exe`.
The evidence archive includes requests; adjust their absolute input/output paths
to your files before execution. It excludes production weights and executables.
Apply its numbered patches in order with `git am --3way <patches>` onto an
isolated checkout of base `36047c9002deaf465073f21d2857dd43eaae88d6`.
The patches contain small deterministic test fixtures, not production models.

The broader workspace subset excluding GPU backend packages passed **686 tests**,
zero failures, 102 explicit ignores (`klein-regression-final.log`). All workspace
test executables compile. The GPU-inclusive full workspace **test-target** run
then passed **774 tests**, zero failures, 121 explicit ignores, across 44 test
binaries; its process exited 0 and no owned test process remained:

```powershell
cargo test --offline --locked --workspace --lib --bins --tests --target-dir '..\klein-target' -- --test-threads=1
```

Evidence: `klein-workspace-test-targets.log` and `klein-final-tests.json`. This
includes CUDA kernel, CPU-vs-CUDA, packed projections, long attention and transfer
checks. Ignored tests were not silently counted as passes. The two explicit
local adapter/VAE tests were run separately as recorded in the initial checkpoint.

The plain full-workspace command remains blocked while linking examples: Windows
LNK1104 cannot open `examples/bench.exe`. At the original base, both
`crates/llama-rs/examples/bench.rs` and `crates/oaiy-tts/examples/bench.rs` already
produce that filename; `llama-rs` and `dsv41-cuda` also share `generate.exe`.
Cargo reports these collisions in `klein-workspace-tests-final.log`. No example
target was changed by this branch. The passing command explicitly excludes
examples and doctests, so this is not a claim that the bare workspace command
passes. The broader subset's log separately includes its doctest results.

## Scope

Validated end-to-end: distilled Klein 4B **text-to-image at 512x512**, native BF16
CUDA transformer/conditioning and F32 VAE, with the supplied LoRA. Worker, server
catalog/request routing and Studio configuration/detection/discovery support this
architecture. Style LoRAs are catalog-owned; `use_loras:false` selects a baseline.

Other resolutions and full-model RAM/disk offloading are untested end-to-end;
tiny residency numerical tests pass. Explicit base 4B configuration is implemented
but base weights were not downloaded or generated. Reference images, negative
prompts, 9B and FLUX.2 dev are outside scope. No push, merge or deployment occurred;
the parent must coordinate applying the local branch to the active project.
