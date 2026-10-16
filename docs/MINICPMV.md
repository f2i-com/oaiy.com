# MiniCPM-V 4.7

OpenBMB's MiniCPM-V 4.7 (October 2026, Apache-2.0) reads pictures: a large one cut into an overview and up to nine
slices, each through a SigLIP tower that merges its tokens inside itself. Two sizes share the vision part:

| Model | Language model | GGUF (OpenBMB's) | Vision projector |
| --- | --- | --- | --- |
| [MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf) | Qwen3.5-family dense, 1.0B (`qwen35`) | `MiniCPM-V-4_7-1B-Q4_K_M.gguf`, 0.67 GB | `mmproj-model-f16.gguf`, 1.1 GB |
| [MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf) | Qwen3.5-family MoE, 35B with 3B active (`qwen35moe`: 256 experts, 8 a token) | `MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`, 21.2 GB | `mmproj-model-f16.gguf`, 1.1 GB |

Each projector also comes as `mmproj-model-f16-4x.gguf`: four times the tokens a picture (no merger inside the
tower), for finer detail.

**Get models → Chat → MiniCPM-V 4.7 1B** fetches the 1B's Q4_K_M file and its projector and adds them as one model; or
add the two files yourself (the projector attaches to the model). In `oaiy-llm-server`:

```
oaiy-llm-server --model MiniCPM-V-4_7-1B-Q4_K_M.gguf --name minicpm-v --vision-projector minicpm-v=mmproj-model-f16.gguf
```

Its language model is a Qwen3.5 one, and it chats, thinks and calls tools as Qwen3.5 does.

## A picture

As transformers' `MiniCPMV4_7ForConditionalGeneration` and its processor make it (`crates/llama-rs/src/mmproj/
minicpmv.rs`):

- **Slices** (`plan`): the tiles the picture's area asks for at about 448 x 448 pixels each (at most 9), give or take
  one, in the grid nearest its shape; an overview of the whole at about 448 x 448; every side a multiple of 56 pixels
  (four 14-pixel patches). A picture of 448 x 448 or less is its overview alone.
- **The tower** (`MiniCpmVMmProj`, `projector_type = "minicpmv4_7"`): SigLIP-400M (27 blocks, 1152 wide, no RoPE),
  its learned positions a 70 x 70 grid each tile's patches are spread over; after its seventh block a window-attention
  merger (attention within each 2 x 2 window of patches, then the four as one), then the final merger (each 2 x 2 of
  tokens through an MLP into the language model's width): a 448 x 448 tile is 64 tokens (16x), or 256 (4x).
- **The prompt**: each picture `<image_id>N</image_id><image>` + the overview's tokens + `</image>`, then the slices row
  by row, each between `<slice>` and `</slice>`, the rows apart by a newline (`crates/oaiy-llm-server/src/qwen.rs`,
  `prepare_minicpmv`).
- **Canvas M-RoPE** (`canvas_positions`): a picture's tokens and markers hold the time channel at the canvas's origin
  (where `<image>` stands), rows and columns counted from it, the slices laid out over the canvas and the overview
  stretched across it; the text after a picture goes on from the origin past the canvas's longer side.

On a GPU the tower is chained (`crates/llama-rs/src/chain_minicpmv.rs`): its patches taken in the mergers' order from
the start, so each merge is rows as they lie; the window attention (four tokens a window) runs on the host.

## Speed

Apple M5 Pro (Metal), the 1B Q4_K_M, a question about an 800 x 1344 picture (an overview and six slices, 487 prompt
tokens): its tiles 0.39 s each through the tower (8.5 s each op by op), the prompt read in 3.4 s, the reply at 120
tokens a second.

## Checks

Against transformers 5.21 (`tools/minicpmv/reference.py`: the 1B in float32 on the CPU, a picture and a question; its
tiles, each tile's tokens, the prompt's tokens and canvas positions, and its reply):

- `mmproj::minicpmv::tests::tiles_and_their_tokens_are_transformers` (llama-rs, the CPU) and
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

Its language model is `qwen35moe`, which OAIY's engine does not yet run: Get models does not offer it. On a Mac with
an eGPU, tinygrad's server in the eGPU plugin runs `qwen35moe` GGUFs on the card (text only: a request with a picture
goes to OAIY's engine).
