# OrcaSAQ2 27B

Nrob reads [orcarouter/OrcaSAQ-2-27B](https://huggingface.co/orcarouter/OrcaSAQ-2-27B)
directly from its original safetensors. The published **3.21 bpw** is the
decoder's average precision, not a uniform 3-bit format. Decoder projections
use 2, 3, 3.5 and 4 bits, the output head uses 6 bits, and embeddings use int8
with one FP16 scale per vocabulary row. Four shards total 12,270,435,404 bytes.

## Download and launch

From the Nrob checkout:

```powershell
python tools/orcasaq/download.py
cargo build --release -p nrob-server
target/release/nrob-server.exe --model models/OrcaSAQ-2-27B --name orcasaq-2-27b --devices 0,1 --ctx 260000
```

The stdlib downloader pins revision `15d20d7e9ae4fd89d1a47878f69381760169445b`,
resumes partial downloads, and verifies every file against the published
SHA-256 or Git blob hash. `--verify-only` checks an existing download offline.
The manifest intentionally omits demonstration media. No model code is executed.

For the sibling coder-cli checkout, set these entries in `settings.toml`:

```toml
[nrob]
startup_model = 'orcasaq-2-27b'
context_tokens = 260000
devices = [0, 1]
prompt_cache = 'workspace' # coder-cli resolves the selected project
prompt_cache_gb = 96

[nrob.models]
'orcasaq-2-27b' = '../nrob/models/OrcaSAQ-2-27B'
```

Keep the other existing settings and model aliases. Rebuild coder-cli with
`build.bat`; `start.bat` already reads `startup_model`. An explicit model
argument still overrides the setting. With `keep_running = true`, opening the
coder-cli TUI starts Nrob in its own console while you type. A matching daemon
is reused; changing project/settings automatically restarts it after its
current request completes. `coder-cli --workspace PATH nrob start` also starts
it explicitly. No separate Python/vLLM server is needed.

## Implementation and limits

### Optional image reading

The Orca release omits vision. Nrob can pair its text weights with the original
vision encoder and trained merger from `Qwen/Qwen3.8-27B`, pinned to revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`:

```powershell
python tools/orcasaq/download.py --vision
```

This verifies the publisher's hashes and downloads one unmodified 3.97 GB
safetensors shard plus its configuration/license into `models/OrcaSAQ-2-27B/vision`.
The shard also contains some text weights; only `model.visual.*` is loaded.
No conversion, retraining, Python runtime, or separate inference server is used.

Configure coder-cli (the value names a directory, not a GGUF file):

```toml
[nrob.vision_projectors]
'orcasaq-2-27b' = '../nrob/models/OrcaSAQ-2-27B/vision'
```

The standalone equivalent is `--vision-projector orcasaq-2-27b=models/OrcaSAQ-2-27B/vision`.
`--no-vision` disables it. An invalid configured tower is an explicit load error.
Send ordinary chat `image_url` content blocks containing base64 data URLs or
permitted local image paths; multiple images and follow-up questions are supported.

This initial still-image path fits the whole image onto a white 768x768 canvas,
preserving aspect ratio, and consumes **576 visual tokens per image** plus framing.
It does not implement dynamic-resolution tiling or video. Small screenshot text
may need a closer crop. Vision tensors are expanded from the original BF16 to
FP32 in memory; on the tested two-GPU setup the last cache GPU runs the tower,
while the first retains the EXL3 text weights. The published **3.21 bpw** still
describes the text checkpoint, not the combined model.

Image prompt states use the selected project's existing disk cache. Its namespace
includes the vision shard/config metadata and preprocessing version; prefix keys
include image content, so changing an image invalidates its cached suffix. Visual
embeddings are also retained in a bounded in-memory cache. A restart can restore
the prompt state without running its cached images through the tower again.

For independent numerical validation, `tools/orcasaq/vision_reference.py` runs
the official Transformers Qwen3.5-family vision implementation on the original
weights (tested with Transformers 5.17.0). It writes a synthetic image and FP32
oracle under `target/orca-vision-research`. Then run:

```powershell
cargo test --release -p nrob-server original_vision_matches -- --ignored --nocapture
$env:NROB_TEST_VISION='1'
cargo test --release -p nrob-server real_model_distributed_cache -- --ignored --nocapture
```

The latter reserves all 260,000 KV slots with vision resident and runs image
encoding while that allocation is live. It is a capacity test, not a full 260K
image-conversation quality benchmark.

Observed validation on the two RTX 5090s:

- Original vision embeddings versus the HF FP32 oracle: RMS error `9.27e-6`,
  maximum absolute error `0.00361` over 576x5120 values (optimized CUDA path).
- Correct shape/colour descriptions, `ORCA 42` OCR, `PROJECT: VISION 73` OCR
  on a wide image, multi-image comparison and a follow-up colour question.
- After a process restart, **597/598** image-prompt tokens restored from disk
  and returned the same `ORCA 42` answer.
- Full 260K cache capacity and vision encoding passed together. The text-only
  128-token speed prompt took **2.44 seconds (52.5 tokens/s)** with vision loaded.
- This is experimental image support, not a vision-quality benchmark. One
  combined description/OCR prompt with no system message returned immediate EOS
  at temperature zero; normal sampling or ordinary chat instructions answered
  it correctly. A leading request to read text from a blank image also produced
  a spurious character once. No stop-token suppression or hidden retries were
  added to conceal these model outputs.

The optimized image encoder uses GPU axial/interleaved position rotations and
cuBLAS batched full-attention products. The score scratch is bounded to 384 MiB
(324 MiB for this tower); causal/long-context attention keeps its existing path.
Qwen prompt processing uses batches of up to 512 tokens, reducing repeated EXL3
weight reconstruction. The 260K capacity test also exercises this image batch.
Image resolution, weights and number of image tokens are unchanged.

With the process warmed up, three image encodings took **0.108 seconds each**,
versus **0.831 / 0.839 / 0.840 seconds** before optimization (about 7.8x faster).
The first isolated call still includes CUDA initialization (5.27 seconds in
this run). These encoder timings exclude language-model prompt processing.

For complete requests, three fresh-image OCR requests averaged **2.25 seconds**
(2.21 / 2.27 / 2.27), versus **3.76 seconds** (3.62 / 3.71 / 3.95) on the previous
build: about **40% less time**. Each used identical pixels, 609 prompt tokens,
11 cached text-prefix tokens and a five-token `ORCA 42` answer. PNG metadata
varied to prevent image-state/embedding cache hits without changing pixels.
Model loading and the first warm-up request are excluded. Fresh descriptions,
wide-image OCR and two-image comparisons also passed. The text-only 128-token
benchmark still generated at **51.0 tokens/s** (2.51 seconds).

The next prefill optimization distributes each 128-wide recurrent state row
over a CUDA warp, then normalizes the output in a separate kernel. It retains
FP32 arithmetic and the existing fused single-token decoder. Three warm
512-token model-only prefills took **1.114 / 1.111 / 1.114 seconds**, compared
with **1.231 / 1.231 / 1.227 seconds** before this change (about 9.5% less time).
These timings exclude vision encoding, checkpoint copies and response generation.
CPU comparisons cover nonzero initial states and 3/512-token full-width batches,
with absolute/relative tolerance `1e-4` for outputs and recurrent state; the
generic smaller-head fallback is also covered. The full 260K allocation plus
512-token image prefill and exact checkpoint restoration still passes.
On the same fresh-image OCR request used above, three complete requests now
took **2.06 / 2.06 / 2.07 seconds** (about 8% less time than 2.25 seconds).
The 128-token text response still took **2.50 seconds (51.2 tokens/s)**.

To verify image grounding independently of familiar demonstration images:

```powershell
python tools/orcasaq/vision_grounding.py --state PATH_TO_PROJECT_DAEMON_JSON --output target/vision-grounding --font C:/Windows/Fonts/arialbd.ttf
```

This manual test draws four random codes and different circle/square colours.
The questions stay identical; answers occur only in the pixels, never the
request text or filenames. It checks changed images, returning to the first
image, and blank/no-image controls. Review the saved report's shape-colour
assignments as well as its keyword checks. Initial controlled OCR returned
all four previously unseen codes correctly; revisiting the first restored
608/609 tokens and returned its original code. Blank and no-image controls
elicited invented numbers at temperature zero, and a combined OCR/colour
question sometimes omitted the code. Pixel conditioning is verified; these
experiments do not establish general vision reliability.
An additional four-card set read three codes exactly, but rendered `JBL 522`
as `JBL 52`; all four circle/square colour assignments were correct. The
grounding tool reports that missed digit as a failure instead of hiding it.
Rebuilding with the previous recurrent kernel and bypassing the image cache
produced the same `JBL 52` response, confirming this case predates the change.
A live coder-cli session called `image_read` once on a neutrally named local
card and correctly answered `LEK 427`; neither the filename nor the prompt
contained that code. Nrob's workspace suite passed 537 tests (93 ignored),
in addition to the manual full-capacity and release CUDA comparisons.

### Optional LoRA adapters

Native inference supports PEFT LoRA and rsLoRA adapters per Orca model alias (and per
Flash-Next alias; see FLASHNEXT.md): one, or several stacked (`--lora NAME=DIR` repeated, each `DIR@X` at strength X; in Studio, the model's `lora` folder plus its `loras` list). Adapters that change the same projections add up: two at full strength can overcook the model (Yes-Man and absolute-heresy, both on `down_proj`/`o_proj`/`out_proj`, garbled their grammar at 1 + 1 and read well at 1 + 0.5), so a second one usually wants a lower strength.
Use an adapter trained for `Qwen/Qwen3.8-27B`, containing
`adapter_config.json` and `adapter_model.safetensors`. Nrob reads those files
directly; the packed 3.21-bpw base weights remain unchanged. Adapter matrices
are held in FP32 on the text GPU. Dropout is disabled for inference, and scaling
uses `alpha/r` for LoRA or `alpha/sqrt(r)` for rsLoRA, times an optional
strength (`--lora-strength NAME=X`, default 1; Studio: the model's `lora_strength`).

In the selected project's `.coder-cli/config.toml`:

```toml
[nrob.lora_adapters]
'orcasaq-2-27b' = '.coder-cli/adapters/my-adapter'
```

Workspace paths resolve from the project root. In the shared `settings.toml`,
relative paths resolve from that settings file's directory. Set the entry to
an empty string to disable an inherited adapter. A new coder-cli session
detects changed adapter settings and restarts its daemon when idle. Standalone:

```powershell
nrob-server --model models/OrcaSAQ-2-27B --name orcasaq-2-27b --devices 0,1 --ctx 260000 --lora orcasaq-2-27b=PATH_TO_ADAPTER
```

Attention and FFN adapters run as a separate low-rank contribution alongside
the packed projection. Qwen's recurrent input/output channel permutations are
also applied to the adapter branch. The small, already-FP32 `in_proj_a/b`
projections combine their deltas at load time in GPU memory. Vision weights
remain unchanged. Every adapter tensor must be recognized and consumed:
unsupported targets or shapes fail loading rather than silently losing part
of an adapter. DoRA, trained biases, per-layer rank/alpha patterns, embedding
adapters, `modules_to_save`, and multiple simultaneous adapters are unsupported.
This implements inference, not adapter training or merging checkpoint files.

Prompt-cache namespaces include the complete adapter config and weight bytes.
Changing or disabling an adapter cannot reuse a prefix computed by a different
adapter. Both KV and recurrent states are still saved in the project's cache.
Quantization can affect adapter quality; an adapter trained on BF16 is not
guaranteed to reproduce the BF16 model's answers on the compressed checkpoint.

The selected `Qwen3.8-27B-LoRA` was verified at
revision `00db3147c218504b368bb8eb98d6291c95b5b2e0`. Its 933,975,024-byte weight
file has SHA-256 `2174bb16b757942f96decb2f4f2cffa0e2ebef5884478be5d2f4f18ade1cd210`.
It uses rank 64, alpha 32 and rsLoRA (scale 4), spanning 496 text projections.
CPU/CUDA tests compare adapter outputs and channel permutations with a direct
dense equation. To test full capacity and checkpoint restoration with an adapter:

```powershell
$env:NROB_TEST_ORCA='E:/models/OrcaSAQ-2-27B' # optional override for relocated weights
$env:NROB_TEST_LORA='PATH_TO_ADAPTER'
$env:NROB_TEST_VISION='1'
cargo test --release -p nrob-server real_model_distributed_cache -- --ignored --nocapture
```

With the initial adapter implementation on the two RTX 5090s, 128 generated tokens took
**4.59 seconds (27.9 tokens/s)** versus about 51 tokens/s without an adapter.
The additional rank-64 projections across almost every text layer add compute
and launch overhead. A fresh 609-token image prompt returned `PXH 448` in
2.89 seconds; repeating it reused 608 tokens and took 0.31 seconds. Its follow-up
colour question reused 615/638 tokens and correctly answered red. Arithmetic
and a structured `read_file` tool call also passed. The real-model test verifies
changed logits, exact cache restoration, and image prefill with all 260K slots
allocated. Workspace validation passed 541 tests (93 ignored); coder-cli's
config/provider/bootstrap suites passed 93 tests.

The decode adapter now uses two CUDA kernels: a split reduction for A*x, then
a fused B*(A*x) and addition into the base projection. This removes intermediate
output tensors and redundant clears. Original FP32 adapter values and rsLoRA
scaling are preserved; batched prefill retains the dense matrix implementation.
CPU/CUDA tests cover rank and dimension tails, batches and nonzero base outputs.
On the same approximately 5,600-token prompt, the identical 111-token reply improved
from 7.02 to 5.81 seconds (15.8 to 19.1 tokens/s). Short arithmetic reasoning
replies generated 51 tokens in 1.36–1.52 seconds (33.6–37.5 tokens/s). These are measured
decode times, not a claim of 50 tokens/s with the adapter or at longer contexts.

Replies and reasoning now stream during generation. In the conversation test,
the first visible text arrived in 1.48 seconds after a fresh read-ahead; the previous
implementation held it until the entire 8.8-second request completed. An identical
retry using a RAM checkpoint began streaming in 0.13 seconds. Thinking
is emitted as `reasoning_content` when enabled. Native tool XML is previewed as
an inert draft and becomes an executable API call only after full validation.

### Text runtime

- The safetensors configuration selects the native Qwen hybrid runtime.
  EXL3 weights remain packed on one CUDA device (the first configured device).
  Two 32 GB RTX 5090 GPUs are the tested configuration at 260,000 tokens.
- CUDA applies input/output Hadamard transforms and the mul1 trellis codebook.
  Half-bit rates alternate 3- and 4-bit steps. Decode reads packed weights
  directly using cooperative tile reads, split-K reductions fused with the
  output transform, and specialized bit-rate kernels. On capable GPUs DP4A
  computes the codebook byte sum exactly, and aligned bit windows reduce
  shift operations. Decode caches scratch buffers and replays each projection's
  three kernels as a CUDA graph, updating its input/output pointers per call.
  Recording uses a separate private stream; execution stays on the ordered
  compute stream. Each result owns its storage, including concurrent callers.
  This graph path requires CUDA Toolkit 12 or newer (tested with 12.8).
  Fully overwritten scratch buffers skip redundant clearing.
  Prefill reconstructs one projection into temporary FP32 scratch;
  the vocabulary head runs only for the last row. This initial implementation
  is **not verified to fit 16 GB GPUs**.
- HF grouped value heads are mapped to Nrob's tiled delta-net layout at runtime.
  RMSNorm offsets and negative-exponential A_log are applied during loading.
- The checkpoint tokenizer uses NFC normalization, combining-mark-aware words
  and individual digits. Its 140-case test corpus matches Hugging Face tokenizers.
- Text, reasoning, XML function calls, model switching and Qwen prompt caching
  use the existing server API. The published checkpoint is text-only; the
  separately configured original vision tower supplies image support above.
  MTP speculative decoding is not implemented for this loader.
- The configured **260,000-token context** is below the architecture's 262,144
  limit. FP32 KV buffers grow on demand and attention layers are distributed
  across the configured GPUs. Full capacity is about 34.08 GB total, plus
  weights, recurrent state and scratch. Split-K attention bounds scratch
  without replicating GQA keys/values. Smaller `--ctx` values are respected.
- RAM checkpoints save delta-net and convolution state while reusing the matching
  attention prefix already resident on the GPUs. A divergent or overwritten
  prefix invalidates these lightweight checkpoints before any restore. Three
  recent boundaries fit below the existing 1 GiB RAM cap; the old full snapshots
  approached that cap individually at about 6,500 tokens, evicting useful state.
  Disk restores include attention and are promoted into RAM for later requests.
  Matching prefixes remain in memory and on disk; a changed suffix may need
  replaying. Project checkpoints stream to/from disk without a second complete
  byte copy, and writer backpressure bounds pending snapshots. At 260K a
  checkpoint is about 34 GB: coder-cli's configured 96 GB budget holds recent
  prefixes. Model changes and different prompts correctly invalidate reuse.
- On this machine's two RTX 5090s, the graph decoder generated 128 tokens in
  **2.73 seconds on the first request (46.9 tokens/s)** and **2.50 seconds
  after warm-up (51.2 tokens/s)** through coder-cli's Nrob daemon. The prior
  optimized build took 2.86 seconds (44.8 tokens/s) warm, and the original
  implementation took 15.47 seconds (8.3 tokens/s). Both graph responses
  matched the original text on the same deterministic compiler-explanation
  prompt. A separate model-only benchmark, with 16 warm-up steps and 128
  measured steps, reached **51.4 tokens/s**.
  These are short-context results; long contexts and other workloads will
  differ. Generation timings exclude model loading and prompt processing.
  The warm request took 2.61 seconds overall; 38/39 prompt tokens restored
  from disk and one token was processed in 0.02 seconds. Loading took 9.8
  seconds. No claim of BF16/reference-model token identity is made.

Client sampling remains controlled by Nrob's settings and request parameters.
The model card recommends temperature 1.0, top_p 0.95 and top_k 20.

## Validation

```powershell
cargo test --workspace
cargo test --release -p ggml-rs-cuda --test exl3 --test long_attention
cargo test --release -p nrob-server benchmark_real_model_decode -- --ignored --nocapture
cargo test --release -p nrob-server benchmark_real_model_prefill -- --ignored --nocapture
cargo test --release -p nrob-server real_model_distributed_cache -- --ignored --nocapture
cargo test --release -p nrob-server tokenizer_matches_huggingface_oracle -- --ignored
python tools/orcasaq/download.py --verify-only
```

The CUDA test uses independently packed synthetic trellises and checks every
decoded weight plus GEMV, prefill and channel permutations for every supported
EXL3 rate (1 through 8 bits, including supported half-bit rates). A model-width
comparison also checks the optimized 2/3/3.5/4/6-bit decode against the scalar
CUDA layout across split counts 1/2/4/8/16/32. Tests retain earlier outputs
while reusing/resizing scratch, alternate inputs, and run concurrent callers
against the same projection to check graph parameter and output isolation. Recurrent-layer tests compare
16 consecutive decode steps with the CPU, including convolution and recurrent
state. Loader tests check format detection and grouped/tiled head mapping.
A real-model test reserves all 260,000 KV positions with weights and decode
graph scratch resident,
compares distributed versus local logits and verifies checkpoint restoration.
Attention separately checks a 260,000-token input; a full 260K text prefill and
long-context quality benchmark have not been run.
Real-model smoke tests cover short answers, arithmetic reasoning and typed
OpenAI tool calls. `tools/orcasaq/tokenizer_oracle.py` regenerates the tokenizer
fixtures with the optional Python `tokenizers` package.

The coder-cli verification reused 6,371/6,432 prompt tokens from live memory
(61 new tokens, 1.76 seconds of prefill). After restarting the daemon, the same
6,364-token request restored 6,363 tokens from the project disk cache; only one
token ran again to recover logits (0.21 seconds of prefill, 2.2 seconds for the
request including restoration and the reply, after model loading). These are
local measurements, not throughput guarantees. The existing Qwen3.8 GGUF
also reused its prior turn (43 cached tokens, 28 new tokens) and recalled the
verification word. Windows must permit writes to
the selected project's `.coder-cli/nrob-cache`; failed publications are logged.

Format references: [OrcaSAQ2 integration](https://github.com/Continuum-AI-Corp/OrcaSAQ2-kernel),
[exllamav3](https://github.com/turboderp-org/exllamav3) and its pack, frac,
reconstruct, codebook and Hadamard kernels. The EXL3 layout/math is implemented
locally, with upstream's MIT notice in
[`tools/orcasaq/LICENSE-exllamav3`](../tools/orcasaq/LICENSE-exllamav3).
