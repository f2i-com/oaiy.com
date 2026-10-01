# Qwen3.8-Flash-Next

Oaiy reads turboderp's EXL3 quantization of Qwen3.8-Flash-Next
([turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
branch `3.05bpw_h5_ng5`) directly from its safetensors (`crates/oaiy-llm-server/src/flashnext.rs`).
No Python runtime or conversion is involved.

The model has 125B parameters with 6B active per token, plus a 51B-parameter n-gram
table. At 3.05 bpw the weights are about 48 GB, so it runs over two GPUs; the n-gram
table (32.6 GB) stays on disk and is read as it is needed.

## Download and launch

```powershell
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 --local-dir E:\models\Qwen3.8-Flash-Next\exl3-3.05bpw
target/release/oaiy-llm-server.exe --model E:\models\Qwen3.8-Flash-Next\exl3-3.05bpw --name Qwen3.8-Flash-Next --devices 0,1
```

In Studio, add it as a model with GPUs of its own, so the others keep theirs:

```json
{"name": "Qwen3.8-Flash-Next", "path": "E:\\models\\Qwen3.8-Flash-Next\\exl3-3.05bpw",
 "vision_projector": "E:\\models\\Qwen3.8-Flash-Next\\exl3-3.05bpw", "devices": [0, 1]}
```

Studio passes a model's `devices` to oaiy-llm-server as `--devices-for NAME=0,1`. While a
model with GPUs of its own is enabled, a media job on any of those GPUs pauses the LLM
(`llm_policy` `auto`), so image and video work never shares a GPU with it.

## LoRA adapters

Flash-Next takes LoRA adapters as the 27B does. Pass `--lora NAME=PATH` (repeat it to stack
several; `PATH@X` sets a strength). In Studio, set the model's `lora` folder and `loras` list.
Two formats are read:

- **llama.cpp GGUF LoRAs.** Give the `.gguf` file, or a folder that holds just it.
  - Its tensors are renamed to Hugging Face names.
  - llama.cpp stores the Gated DeltaNet value heads tiled, where Hugging Face groups them
    by key head. That reorder is undone for `attn_qkv`, `attn_gate`, `ssm_alpha`,
    `ssm_beta` and `ssm_out`.
  - Stacked expert tensors split per expert.
  - The scale is `alpha / rank`, as in llama.cpp.
- **PEFT folders** (`adapter_config.json`) whose base is `Qwen/Qwen3.8-Flash-Next`.

What an adapter can change:

- **Projections.** The attention, Gated DeltaNet, router, indexer, n-gram and head
  projections run their base weight plus the adapter's low-rank term. Several adapters on
  one matrix are stacked into one term.
- **Experts.** Routed and shared experts take LoRA on their gate, up and down projections
  inside the grouped MoE kernels. The gate and up deltas are added before the activation;
  the down delta is added to each expert's weighted output.

An adapter tensor with no matrix to go to is an error, rather than being silently left
out.

For example, `chenrm/qwen3.8-flash-next-abliterated-lora` (rank 2, 2,656 projections: the
attention and DeltaNet outputs, the shared experts' down projections, and every expert's
down projection in five layers) loads in the same time and decodes at about 103 tokens/s.

## Two conversations on one model

The engine keeps one conversation's state on the GPUs (the cache, and the checkpoints it
returns to at message boundaries) and nothing on disk. When two conversations take turns on
it (a runner and a call's sub-agent that share only a short system prompt), each one's prompt
is read again at every switch: 31 to 36 s of prefill for 20,000 to 21,000 tokens, where the
conversation alone is cached and takes under a second.

So a state the engine is about to lose is copied to host RAM first, and comes back when a later
prompt continues it. `--park-gb` bounds the RAM (default 8, never more than half of what is
free, 0 = off); Studio's `llm.park_gb` passes it when it is set (`null` leaves the server's own
default, and an older server that does not know the flag still starts).

Measured on 1 Oct 2026 on the owner's two-GPU machine, one run, the model loaded on both GPUs:

| | |
|---|---|
| A state of 20.3k tokens | 1.7 GB, checkpoints included (5.3k tokens: 0.44 GB) |
| Parking it | 0.17 to 0.20 s (0.60 s once, right after the model had been loaded) |
| Restoring it | 0.15 s (the 5.3k-token one: 0.055 s) |
| The whole `Qwen cache:` line of a swap-in | 0.21 to 0.26 s, against 31 to 36 s of prefill before |
| Six requests, A1 B1 A2 B2 A3 B3 (a 20k and a 5k conversation) | every switch from `ram`, 96 tokens read each time |
| A2 after a swap against A2 without one | the same text, byte for byte (fresh servers) |

While a state is copied the engine holds a K slice and a V slice of one attention slot on the
device at once, 86 MB at 21,000 tokens (43 MB each); a restore uploads one slice at a time. The
tighter GPU has about 2 GB free with the model loaded, so both are small beside it; a copy that the
device refuses falls back to reading the prompt.

- It happens only when the engine displaces a state, never after each turn: a prompt that
  continues a stashed state sets the live one aside and brings that one back; a prompt that
  shares less than half of a big live state sets that one aside before it throws it away (or
  rolls it back to a checkpoint and writes over it, when the conversations share a system
  prompt that ends at a message boundary; then the stash gets copies, because the new prompt
  still needs the checkpoint). A conversation that branches (a larger share) is the
  checkpoints' business, as before.
- A state shorter than 1,024 tokens is not kept, and one is brought back only when it saves
  1,024 tokens or more than the live one does.
- The one set aside longest ago goes first when the budget is full.
- An incognito request neither takes from the stash nor adds to it, and when an incognito
  request or session ends the stash is emptied with the rest of what the engine holds.
- A copy that cannot be made or put back (a cache of another shape, a device that refuses the
  memory) falls back to reading the prompt, as before. Nothing else about a request changes.
- The log says so: `Qwen park: stashed N tokens (X MB) in Ys; M states, Z MB held`, `Qwen
  restore: N tokens (X MB) in Ys`, and `Qwen cache: S/N tokens from ram (common C) in Ys`, where
  that time includes the copies. The reply's `cache_source` is `ram`.

The decode graphs need nothing from a restore: every step captures its graph again from the
buffers at hand and updates the old one to the new addresses, so the caches may be grown or
replaced between steps, which is also what a restore does (through the same `reserve_layer`).
The Qwen3.5 hybrid and the other engines are not changed.

### Checking it

The logic runs on the CPU, with no model: `cargo test -p oaiy-llm-server --lib qwen` (set
`CUDA_VISIBLE_DEVICES=-1` if the GPUs are in use). That includes `qwen_real.rs`, which drives the real
`QwenEngine::run` with a fake model whose answers depend on every row and recurrent tensor of the
cache, and compares an engine that sets conversations aside with one that does not over randomized
scripts (`PARK_FUZZ_SEEDS` sets how many). `cuda_round_trip_is_bit_exact_on_the_real_devices` is
ignored: it opens a CUDA context on devices 0 and 1, so run it only with the model stopped.

On the real model, `crates/oaiy-llm-server/tools/verify_parking.py` (loopback only, synthetic
text; its own logic is tested against `tools/mock_server.py` with `python -m unittest
tools/test_verify_parking.py`). It needs a short model restart, so ask first:

1. Put the new `oaiy-llm-server.exe` in place: `POST http://127.0.0.1:7860/api/llm/stop`, copy
   the exe over `target\release\oaiy-llm-server.exe`, and the next request starts it (the
   Studio's Incognito setting must be off).
2. `python verify_parking.py --mode calibrate`, then `--mode sequence` with the notes it prints:
   A1 B1 A2 B2 A3 B3, which should all be `ram` after the first round, each with only the new
   turn read. A request of someone else's in the middle makes the run INCONCLUSIVE.
3. Three runs, each on a freshly started model (`--restart` stops it first): `--mode reference`
   (A1 A2, no swap), `--mode swapped` (A1 B1 A2), and a second `--mode reference`. Then
   `--mode compare` on the first two (PASS when A2 answers byte for byte the same, VACUOUS when
   A2 did not come from `ram`) and `--mode compare --baseline` on the two references (whether the
   engine reproduces itself across restarts, without which a difference means nothing).

## The architecture

It is the Qwen3.5 MoE lineage: 48 layers, three Gated DeltaNet layers to each full
attention layer, and 512 experts (10 routed plus a shared one) in every layer. Its new
parts, each following exllamav3's `qwen4_exp` implementation:

- **Gated residual.** The residual is 4 fp32 streams. Before each attention and MoE branch, a
  per-stream RMS norm and a low-rank sigmoid gate make the branch's input (the gated mean
  of the normed streams). The branch output goes back into each stream with a scalar
  `2 * sigmoid` weight. A last gate collapses the streams before the head; there is no
  final norm.
- **N-gram embedding (PLE).** Before layer 1, hashed bigrams and trigrams (8 hash heads each)
  pick rows of a trellis-quantized table (5 bits, 160 values a row). A signed-sqrt dot
  gate puts them into every stream, and a dilated causal conv (kernel 4, dilation 3) adds
  local context. Its state (the conv window and the last two token ids) is kept with the
  cache, and checkpoints take it along.
- **QSA sparse attention.** Each full attention layer has an indexer: 4 query heads and one
  raw key a token, pooled into 4-token blocks. Each query attends to its top 512 blocks
  plus its incomplete tail block. Up to 2,051 tokens that is every token, so dense
  attention runs there; past it, the blocks are scored and picked on the GPU, and a sparse
  attention kernel reads only those tokens.
- **Gated DeltaNet.** Its output gate is `sigmoid(z)` rather than Qwen3.5's `silu(z)`.
- **Vision.** The tower is Qwen3.5's, stored as EXL3 with its MLP padded to 4352.

The multi-token prediction head is not used.

## Checking it

`matches_the_reference` (in `flashnext.rs`, ignored by default) compares OAIY with
exllamav3 on the same checkpoint. Generate a reference with exllamav3 1.4.9 (the 1.5 Windows
wheels' extension is over 2 GB and does not load), then:

```powershell
$env:FLASHNEXT_MODEL = "E:\models\Qwen3.8-Flash-Next\exl3-3.05bpw"
$env:FLASHNEXT_REFERENCE = "reference.json"
cargo test --release -p oaiy-llm-server --features cuda --lib flashnext -- --ignored --nocapture
```

On a 33-token prompt the n-gram features match to 1e-5 and the top prediction agrees at
32 of 33 positions (the other is a near tie). On a 3,976-token prompt, where the sparse
attention is in use, it agrees at every position checked.

## Speed

On two RTX 5090s, decoding runs at about 113 tokens/s on a short prompt, 105 tokens/s at
4,000 tokens of context and 92 tokens/s at 64,000. Prefill runs at about 640 tokens/s. Through Studio, with
sampling and streaming, decoding runs at about 108 tokens/s. A later request that
continues an earlier one (an agent's next step) reads only what is new, from memory or from
a checkpoint at a message boundary.

What makes it that fast:

- **Graphs.** A decode step is about 1,400 small kernels. Launched one at a time, each costs
  the host 7 to 8 µs on Windows, which is longer than most of them take to run. So after the
  first step, each GPU's share of a step is captured into a CUDA graph. The graph is
  updated from the new capture each step and run as one launch. The second GPU's share is
  captured while the first GPU runs. `FLASHNEXT_GRAPHS=0` turns graphs off.
  - Capture needs a stream of the model's own (`CudaBackend::new_graphable`), not the
    device's default stream.
  - A step's temporaries come from an arena, so addresses repeat from step to step. The
    arena doubles whenever a step fills half of it, since long attention's temporaries
    grow with the context.
  - State kept between steps is updated in place: the caches grow, and the rope positions
    upload, before capture.
- **Grouped MoE.** Routing and grouping take one launch. Every routed expert and the shared
  expert then run in a few grouped EXL3 kernels, and each token's experts are summed in a
  fixed order, so a run repeats exactly.
- **f16 where the checkpoint is f16.** The gated-residual projections, the router and the
  n-gram projections stay f16 on the GPU, packed two to a word. This is exact, and they
  read half the memory.
- **Decode attention** splits the keys over many blocks, a warp per key.
- **The n-gram layer** runs on the GPU. Its 16 table rows a token are read from disk in
  parallel, one file handle per thread.

`FLASHNEXT_PROFILE=1` prints where a forward's time goes; it synchronizes after each part and
turns graphs off. The `bench::speed` test measures prefill and decode:

```powershell
$env:FLASHNEXT_DECODE = "400"
cargo test --release -p oaiy-llm-server --lib flashnext::bench::speed -- --ignored --nocapture
```
