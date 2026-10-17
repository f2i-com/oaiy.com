# Qwen3-VL (dense)

Qwen's Qwen3-VL-2B/4B/8B-Instruct (Apache-2.0) read pictures and chat, and their language model is what FLUX 3 Image
Klein encodes its prompts with ([FLUX3.md](FLUX3.md)): one file serves as Klein's text encoder and as a language model
that checks pictures, the image flow's reviewer among them ([IMAGE_FLOW.md](IMAGE_FLOW.md)). Architecture `qwen3vl`.

| Part | What OAIY runs | Code |
| --- | --- | --- |
| Language model | Qwen3's dense decoder (q and k RMS-normed a head at a time, SwiGLU), its rotary pairs turning by a row's time, height and width interleaved (M-RoPE, `rope.dimension_sections` 24, 20, 20 of a 128-wide head's 64 pairs), chained on the GPU | `crates/oaiy-llm-server/src/llm/arch/qwen3.rs`, `llm/chain/decode.rs` |
| Vision tower | llama.cpp's mmproj GGUF (`qwen3vl_merger`): 24 blocks 1,024 wide (the 4B's; 27 and 1,152 for the 8B), a picture at 768 x 768 in 16-pixel patches, its learned positions and 2-D rotation, each 2 x 2 patches merged into one of 576 tokens | `llm/vision/mmproj/qwen3vl.rs`, `llm/chain/mmproj.rs` |
| DeepStack | Three more mergers read the tower's blocks 5, 11 and 17 (the 4B's); their tokens are added to the language model's states at the picture's rows after its layers 0, 1 and 2 | the same, and `engines/qwen` |
| Serving | Qwen's engine (`engines/qwen`): a picture's rows at their positions over its grid, the text after it past its side; the chat template with no thought; tools offered and called in Qwen3's own form (JSON in `<tool_call>` tags) | `engines/qwen/mod.rs` |

The files: Qwen's [Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) has the language
model (`Qwen3VL-4B-Instruct-Q4_K_M.gguf`, 2.5 GB; Q8_0 and F16 too) and the vision projector
(`mmproj-Qwen3VL-4B-Instruct-F16.gguf`, 0.84 GB). A model entry:

```json
{"name": "qwen3-vl-4b", "path": "FLUX-3-Image-Klein-Q4/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
 "vision_projector": "FLUX-3-Image-Klein-Q4/mmproj-Qwen3VL-4B-Instruct-F16.gguf"}
```

Adding the GGUF in **Models** finds it as a language model (and the projector beside it); a GGUF named for Qwen Image is
Qwen Image's text encoder instead. Only a quantized GGUF runs chained on the GPU: an F16 one runs op by op, and its
pictures are refused there.

## Checks

- The vision tower against transformers' `Qwen3VLVisionModel` (float32, the checkpoint's own weights) on the same
  picture (`tools/qwen3vl/vision_reference.py`; `qwen3vl_tower_and_deepstack_are_transformers`): its 576 tokens and
  each of its three DeepStack features at cosine 1.00000, the worst token's too.
- The whole model (Q4_K_M) against transformers' (bf16) on a question about a picture (`tools/qwen3vl/text_reference.py`):
  "The mug is dark blue and has the words "GOOD NIGHT" printed on it in white." where transformers says "... printed in
  white."; on text alone the same 25 words before Q4's rounding takes another. Without DeepStack, or with the picture's
  rows at one position each, the answer leaves transformers' sooner ("... and printed with the words ...", "... with the
  words ... in white, bold, uppercase letters"): both take effect.
- The interleave of the rotary pairs as transformers' `apply_interleaved_mrope` makes it (`each_rotary_pair_turns_by_its_axis`),
  the template (`qwen3_vl_prompts_open_no_thought`, `qwen3_vl_tools_are_offered_in_its_own_form`), and its calls read
  as the XML form's are (`json_calls_are_read_as_the_xml_ones`).

## Speed

On an M5 Pro's GPU (the 4B, Q4_K_M; the tower F16): loaded in 2.7 to 4.1 s; a picture through the tower in 1.5 s; a
606-token prompt with a picture read in 2.8 s; a reply at 33 to 34 tokens a second. In the image flow it writes the
caption and the review with no load at all between pictures (the card keeps the picture models), 41 to 42 s a run
warm, as with Qwen3.5-9B. As the Agent's model it called the flow tool from the Agent's 4,344-token prompt and
answered from the review.

As a reviewer it is the less careful of the two: in two runs of six it put a description of the picture where the words
it read belong. Qwen3.5-9B (MOSS-Image's text encoder's file) filled those fields right in its runs.
