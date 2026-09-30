# Native Klein 4B validation

Native Rust/Candle inference now produces real images with the official distilled
FLUX.2 Klein 4B transformer and the supplied SimpleFineVector LoRA. Both images
were visually inspected: coherent vector foxes, clean outlines, white backgrounds.
No Python subprocess performs inference. No E: checkout or active desktop was
changed, restarted, unloaded or terminated.

The implementation and archived source inspection are documented in
[KLEIN.md](KLEIN.md). Branch: `codex/native-flux2-klein4b`; base `36047c9`.

## Evidence

All runs use the same prompt, seed 747, 512x512, four steps, CFG 1, GPU 0.

| Run | Total seconds | Sampling seconds | PNG SHA256 |
| --- | ---: | ---: | --- |
| Clean baseline | 48.857 | 2.292 | `7a9b3c69528a3780de568735d15caa77c38ff29a266494f8d93bd839e8214544` |
| LoRA strength 1 | 26.128 | 5.686 | `adf7bfc84732abd741febc06cb3e7e0209f44d4f9ccdf9bd8319b21792e5a37d` |
| LoRA strength 0 | 23.591 | 0.988 | Identical to baseline |

The zero-strength control is pixel-identical **and PNG-byte-identical** to the
clean baseline. Strength 1 changes 110,847 pixels (42.3%); mean absolute channel
difference is 8.657/255. These are individual observed timings with different
warm file/kernel caches, not a controlled performance benchmark.

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

The broader workspace subset excluding GPU backend packages passed **686 tests**,
zero failures, 102 explicit ignores (`klein-regression-final.log`). All workspace
test executables compile. The plain full-workspace run failed while linking
pre-existing examples that share `examples/bench.exe` (Cargo warns about filename
collisions). The full test-target run excluding examples is tracked separately;
do not claim it passed until its log confirms completion.

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
