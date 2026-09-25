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

- The safetensors configuration selects the native Qwen hybrid runtime.
  EXL3 weights remain packed on one CUDA device (the first configured device).
  Two 32 GB RTX 5090 GPUs are the tested configuration at 260,000 tokens.
- CUDA applies input/output Hadamard transforms and the mul1 trellis codebook.
  Half-bit rates alternate 3- and 4-bit steps. Decode reads packed weights
  directly using cooperative tile reads, split-K reductions fused with the
  output transform, and specialized bit-rate kernels. On capable GPUs DP4A
  computes the codebook byte sum exactly, and aligned bit windows reduce
  shift operations. Fully overwritten scratch buffers skip redundant clearing.
  Prefill reconstructs one projection into temporary FP32 scratch;
  the vocabulary head runs only for the last row. This initial implementation
  is **not verified to fit 16 GB GPUs**.
- HF grouped value heads are mapped to Nrob's tiled delta-net layout at runtime.
  RMSNorm offsets and negative-exponential A_log are applied during loading.
- The checkpoint tokenizer uses NFC normalization, combining-mark-aware words
  and individual digits. Its 140-case test corpus matches Hugging Face tokenizers.
- Text, reasoning, XML function calls, model switching and Qwen prompt caching
  use the existing server API. This checkpoint is text-only. Vision and MTP
  speculative decoding are not implemented for this loader.
- The configured **260,000-token context** is below the architecture's 262,144
  limit. FP32 KV buffers grow on demand and attention layers are distributed
  across the configured GPUs. Full capacity is about 34.08 GB total, plus
  weights, recurrent state and scratch. Split-K attention bounds scratch
  without replicating GQA keys/values. Smaller `--ctx` values are respected.
- Prompt reuse restores attention, delta-net and convolution state together.
  Matching prefixes remain in memory and on disk; a changed suffix may need
  replaying. Project checkpoints stream to/from disk without a second complete
  byte copy, and writer backpressure bounds pending snapshots. At 260K a
  checkpoint is about 34 GB: coder-cli's configured 96 GB budget holds recent
  prefixes. Model changes and different prompts correctly invalidate reuse.
- On this machine's two RTX 5090s, the final native decoder generated
  128 tokens in **2.98 / 2.86 seconds (43.0 / 44.8 tokens/s)** in two runs
  through coder-cli's Nrob daemon, versus **15.47 seconds (8.3 tokens/s)**
  before optimization. Both generated the same text as the baseline on the
  deterministic short compiler-explanation prompt. A separate model-only
  32-step decode benchmark measured **46.5 tokens/s**. These are measured
  short-context results, not a claim of 50 tokens/s or 260K-context throughput.
  Generation timings exclude loading and prompt processing. The warm request
  took 2.98 seconds overall; 38/39 prompt tokens restored from disk and one
  token was processed in 0.02 seconds. Loading took 11.7 seconds in this run.
  No claim of BF16/reference-model token identity is made.

Client sampling remains controlled by Nrob's settings and request parameters.
The model card recommends temperature 1.0, top_p 0.95 and top_k 20.

## Validation

```powershell
cargo test --workspace
cargo test --release -p ggml-rs-cuda --test exl3 --test long_attention
cargo test --release -p nrob-server benchmark_real_model_decode -- --ignored --nocapture
cargo test --release -p nrob-server real_model_distributed_cache -- --ignored --nocapture
cargo test --release -p nrob-server tokenizer_matches_huggingface_oracle -- --ignored
python tools/orcasaq/download.py --verify-only
```

The CUDA test uses independently packed synthetic trellises and checks every
decoded weight plus GEMV, prefill and channel permutations for every supported
EXL3 rate (1 through 8 bits, including supported half-bit rates). A model-width
comparison also checks the optimized 2/3/3.5/4/6-bit decode against the scalar
CUDA layout across split counts 1/2/4/8/16/32. Recurrent-layer tests compare
16 consecutive decode steps with the CPU, including convolution and recurrent
state. Loader tests check format detection and grouped/tiled head mapping.
A real-model test reserves all 260,000 KV positions with weights resident,
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
