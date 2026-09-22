# Experimental ternary experts and tool-call precision

This opt-in research backend reads custom W2/G128 routed-expert records produced
by [nca_research](https://github.com/f2i-com/nca_research). It is an explicit
experimental exception to the normal read-original-weights format policy in
CONVENTIONS.md. The default safetensors/GGUF paths remain unchanged. Conversion
scripts and checkpoint manifests belong to the research repository; no model
weights are included here.

The format packs four ternary codes per byte, with an FP16 scale per128 columns
of each row:2.125 stored bits/weight. This is lossy post-training quantization,
not a model trained with Microsoft BitNet. Each record contains complete w1/w2/w3
expert matrices and scales. Dimensions, token IDs, attention, routers, shared
experts, normalization and other retained tensors are unchanged. The loader
checks format, source config, record sizes and all-layer expert coverage.

## Running the server

```powershell
cargo run --release -p nrob-server -- --model 'D:\bitnet\MODEL\model' --ternary-experts 'D:\bitnet\MODEL\experts' --name deepseek-v4.1-flash --devices 0,1 --ram-gb 140 --ctx 16384 --no-vision --port 18002 --tools-experts 'D:\deepseek\model' --expert-trace 'expert-routing-NEW.jsonl'
```

Without a ternary mapping, ordinary original-model inference remains available.
`--also-ternary NAME=DIR` and `--also-tools-experts NAME=DIR` configure named
alternatives registered with `--also NAME=MODEL_DIR`. The library Options carry
model-specific source maps; the server calls an explicit GPU loader, so original
or GGUF alternatives never inherit another model's ternary bank. The legacy
DSV41_TERNARY_DIR variable is recognized only by the standalone CLI for its
default model (and legacy single-model example loaders), not the embedded server.
Prompt-cache identity includes model/expert paths, precision policy and context
capacity. Unknown model names and tool sources without ternary banks are errors.
`--tools-experts` requires an initially ternary model and the matching original
checkpoint. It opens the original expert index without loading another trunk.
The file passed to `--expert-trace` must not exist; logs stay local and can include
generated token text. Do not commit session traces containing private code.

When a tool-enabled request generates a complete DSML calls opening tag, the
worker switches subsequent forwards to original MXFP4 experts. After a validated
complete batch, or request cleanup, it returns to ternary. It keeps the trunk
resident and two separate expert-cache banks, each served on demand by
(layer,expert). Existing RAM/VRAM expert budgets are split80% ternary/20% MXFP4.
Both banks retain hot records across switches. Generated mixed-precision states
are rolled back to a ternary prompt checkpoint; persisted prompt states have a
separate policy namespace.

This is not whole-turn original-model inference: decisions to begin a tool call,
and its inherited context, can already contain ternary errors. A switch cannot
retroactively change an emitted token. The trace records request, phase, input
position, layer, precision, expert IDs and router weights; it does not identify
causal coding/tool specialists. The input forward predicts the next token.

The runtime DSML parser accepts complete inner tags with either canonical DSML
namespaces or plain XML, with whitespace/attribute-order variations. It still
requires an explicit DSML outer envelope and valid required parameter attributes,
rejects duplicate parameters and malformed typed JSON, and finishes after the
first validated call batch. Invalid calls return a clear API error; no missing
arguments are guessed. The strict reference parser remains unchanged.

## Evidence and limits

On two RTX5090s, two fixed development prompts returned valid workspace_info
and write_file calls; a warm repeat of the first also passed. No tools/code were
executed by those HTTP probes. A later user session executed workspace_info and
list_files, but became repetitive during an Enigma coding request. Broad quality,
paired full-model speed, and original-model equivalence are not established.

The365.905GB research checkpoint saves144.393GB versus its source. Only routed
experts were ternarized; large Engram tables and other retained tensors remain.
Whole-copy hash rereading was stopped at the user's request after101.740GB;
conversion-time hashes and sampled reference checks exist. Do not infer a fully
verified archive or general quality from a successful startup.

```sh
cargo test --workspace --release --offline
cargo test -p dsv41-cuda --test precision_caches --release -- --ignored
```

The second command explicitly requires two CUDA devices. Its cache test verifies
distinct bytes for identical expert IDs across banks and reuse on return; it does
not substitute for the model-level smoke. Original golden-model gates remain
separate and ignored by default. See nca_research/docs/DEEPSEEK_TOOL_PRECISION.md
and its archived run reports for raw development outcomes and unexecuted checks.
