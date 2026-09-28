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
