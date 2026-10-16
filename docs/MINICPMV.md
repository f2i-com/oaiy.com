# MiniCPM-V 4.7

OpenBMB's MiniCPM-V 4.7 (October 2026, Apache-2.0) reads pictures: a large one cut into an overview and up to nine
slices, each through a SigLIP tower that merges its tokens inside itself. Two sizes share the vision part:

| Model | Language model | GGUF (OpenBMB's) | Vision projector |
| --- | --- | --- | --- |
| [MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf) | Qwen3.5-family dense, 1.0B (`qwen35`) | `MiniCPM-V-4_7-1B-Q4_K_M.gguf`, 0.67 GB | `mmproj-model-f16.gguf`, 1.1 GB |
| [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Qwen3.5-family MoE, 35B with 3B active (`qwen35moe`: 256 experts, 8 a token) | `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`, 21.2 GB | `mmproj-model-f16.gguf`, 1.1 GB |

Each projector also comes as `mmproj-model-f16-4x.gguf`: four times the tokens a picture (no merger inside the
tower), for finer detail.

**Get models → Chat → MiniCPM-V 4.7 1B** (or **35B-A3B**) fetches the Q4_K_M file and its projector and adds them as
one model; or add the two files yourself (the projector attaches to the model). In `oaiy-llm-server`:

```
oaiy-llm-server --model MiniCPM-V-4_7-1B-Q4_K_M.gguf --name minicpm-v --vision-projector minicpm-v=mmproj-model-f16.gguf
```

Their language models are Qwen3.5 ones: they chat, think and call tools as Qwen3.5 does.

## A picture

As transformers' `MiniCPMV4_7ForConditionalGeneration` and its processor make it (`crates/oaiy-llm-server/src/llm/mmproj/
minicpmv.rs`):

- **Slices** (`plan`): the tiles the picture's area asks for at about 448 x 448 pixels each (at most 9), give or take
  one, in the grid nearest its shape; an overview of the whole at about 448 x 448; every side a multiple of 56 pixels
  (four 14-pixel patches). A picture of 448 x 448 or less is its overview alone.
- **The tower** (`MiniCpmVMmProj`, `projector_type = "minicpmv4_7"`): SigLIP-400M (27 blocks, 1152 wide, no RoPE),
  its learned positions a 70 x 70 grid each tile's patches are spread over; after its seventh block a window-attention
  merger (attention within each 2 x 2 window of patches, then the four as one), then the final merger (each 2 x 2 of
  tokens through an MLP into the language model's width): a 448 x 448 tile is 64 tokens (16x), or 256 (4x).
- **The prompt**: each picture `<image_id>N</image_id><image>` + the overview's tokens + `</image>`, then the slices row
  by row, each between `<slice>` and `</slice>`, the rows apart by a newline (`crates/oaiy-llm-server/src/engines/qwen/mod.rs`,
  `prepare_minicpmv`).
- **Canvas M-RoPE** (`canvas_positions`): a picture's tokens and markers hold the time channel at the canvas's origin
  (where `<image>` stands), rows and columns counted from it, the slices laid out over the canvas and the overview
  stretched across it; the text after a picture goes on from the origin past the canvas's longer side.

On a GPU the tower is chained (`crates/oaiy-llm-server/src/llm/chain/minicpmv.rs`): its patches taken in the mergers' order from
the start, so each merge is rows as they lie; the window attention (four tokens a window) runs on the host.

## Speed

Apple M5 Pro (Metal), the 1B Q4_K_M, a question about an 800 x 1344 picture (an overview and six slices, 487 prompt
tokens): its tiles 0.39 s each through the tower (8.5 s each op by op), the prompt read in 3.4 s, the reply at 120
tokens a second.

## Checks

Against transformers 5.21 (`tools/minicpmv/reference.py`: the 1B in float32 on the CPU, a picture and a question; its
tiles, each tile's tokens, the prompt's tokens and canvas positions, and its reply):

- `mmproj::minicpmv::tests::tiles_and_their_tokens_are_transformers` (the server's `llm`, the CPU) and
  `qwen::tests::the_minicpmv_tower_on_webgpu_is_transformers` (the server, WebGPU): the same tiles (pixels within one
  level of 255, the resize filters'), and each of transformers' tiles to the same tokens: cosine 1.000000, the largest
  difference 0.0002.
- `mmproj::minicpmv::tests::canvas_positions_are_transformers`: every token's position, and where the reply starts.
- `qwen::tests::a_minicpmv_prompt_is_transformers`: the server's prompt (its template, tokenizer and the pictures laid
  out) is transformers' token for token.

Each on a tall picture (an overview and six slices) and a wide one (an overview and a row of two), 16x; and 4x on the
tall one (1,819 tokens). Asked about a tarot card, the 1B Q4_K_M here and transformers' float32 1B each read its title
and describe it the same.

## The 35B-A3B

Its language model is Qwen3.5-MoE (`qwen35moe`, as Qwen3.6-35B-A3B's): the Qwen3.5 hybrid's 40 layers (30 gated delta
nets, 10 gated attention layers), each with 256 routed experts of which 8 a token, softmax-weighted among themselves,
and a shared expert by its gate's sigmoid. OAIY runs it as a Qwen3.5 model whose layers have routed experts
(`oaiy_llm_server::llm::Qwen35Moe`, loaded by `Qwen35Model::from_gguf_with`): the experts on the GPU in their GGUF blocks
(`ggml_rs_wgpu::quant_moe`, which now decodes Q4_K and Q6_K, a Q4_K_M file's experts), routed on the GPU, the shared
expert among them; the router f32. Chained, a layer's FFN is its router's matmul then the experts' routed sums in the
same submit as the rest of the step. Its weights are 21.3 GB on the GPU: a 24 GB card. A GPU without 16 x 16 tensor
cores (a Mac's) groups no more than 64 of a prompt's rows by expert at once, and a chained run takes its prompt in
pieces of 64 there.

On a Mac's eGPU its vision tower runs on the Mac's own GPU, which is idle while the card serves the model, and the
card's room past the weights is the attention cache's: 69,632 positions on the RTX 4090 (`docs/MAC.md`, "How much
context fits on the card"). The cache is f16 alone, 20 KB a position: a KV head's eight query heads of 256 are more
than a step's attention kernel holds at once, so it takes them in two fours. A 49,019-token prompt is read in 13.1 s.

RTX 4090 (eGPU), Q4_K_M with its projector: loaded in 2 minutes from an external SSD, all of it on the card; a reply at
about 120 tokens a second; a question about the 800 x 1344 picture (seven tiles, 487 prompt tokens) read in 5.4 s,
about the 1024 x 512 one (three tiles, 223 tokens) in 0.9 s. tinygrad's own `qwen35moe` (the eGPU plugin's server,
9 tokens a second) gives the same greedy reply to the same question, word for word.
`models::dense_webgpu_timing::a_chained_qwen35_run_answers_as_the_host_path` takes a Qwen3.5-MoE GGUF too (`QWEN35_MODEL`): a
prompt and greedy steps chained and op by op. On the 4090, 600 prompt tokens: chained 1.95 s against 5.7 s op by op, the
prompt's logits at cosine 0.99971; the two part at the first step, where both paths' two likeliest tokens are within
0.11 of a logit (a layer's top 8 of 256 experts tips on the last bits of a sum), so a MoE model is held to a cosine of
0.99 and such a near tie, a dense one to 0.9999.
