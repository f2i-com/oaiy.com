# MOSS-Image

OpenMOSS's MOSS-Image (October 2026: MOSS-Image-Base, and MOSS-Image-SFT fine-tuned from it) runs natively in OAIY's
media worker on WebGPU, on a GPU of any make (the Mac's through Metal, NVIDIA's, AMD's, Intel's, and an NVIDIA card in a
Thunderbolt enclosure through the eGPU plugin): pictures from a description, a list of tags, or a JSON layout of boxes,
with legible lettering in English and Chinese. Architecture `moss-image`. Its weights and OpenMOSS's pipeline code are
Apache-2.0.

| Piece | What OAIY runs | Code |
| --- | --- | --- |
| Transformer (5B) | OpenMOSS's `transformer/` folder (BF16), a single safetensors file of it (ComfyUI's), or a GGUF of it (realrebelai's `MOSS-Image_GGUFs`, Q2_K to Q8_0); its blocks' matrices f16 where the tensor cores take f16 (NVIDIA's, AMD's) and there is room, else its BF16 ones as Q8_0 (a Mac's GPU); a GGUF's blocks as they are (Q3_K to Q8_0 on the tensor cores, others as f16). `OAIY_MOSS_WEIGHTS` `f16` or `q8_0` says which | `crates/oaiy-media/src/moss_image/webgpu/mod.rs` |
| Text encoder | Qwen3.5-9B's language model (its vision tower unused): the last hidden state after the final norm, `[n, 4096]`. MOSS-Image's own `text_encoder/` is Qwen's Qwen3.5-9B, its few F32 tensors as BF16 (every tensor checked), so Qwen's own release serves too, or a GGUF of it (llama.cpp's, with its own tokenizer) | `crates/oaiy-media/src/moss_image/webgpu/text.rs` |
| VAE | The FLUX.2 VAE's decoder: MOSS-Image's own `vae/` (FLUX.2's encoder with a smaller decoder, 384 wide where FLUX.2's is 512) or FLUX.2's | `crates/oaiy-media/src/sdxl/webgpu/vae.rs`, `crates/oaiy-media/src/klein/vae.rs` |
| Pipeline | The scheduler's 50 sigmas (shift 5, then each snapped to its training grid), Euler steps, guidance 3 whose negative branch skips block 22 | `crates/oaiy-media/src/moss_image/mod.rs` |

The transformer is one stream: the picture's tokens, then the text's. The text's tokens first go through a projection and
two blocks of their own (with biases, attending among themselves, once a prompt); then all of them through 42 blocks
attending together, every token modulated alike by the step's time, one modulation all the blocks share. A block is an
RMS norm (its own weight) modulated, then attention (q and k RMS-normed a head at a time, rotated by four axes of position
over each head's halves) and an MLP whose value and gate are each soft-capped (SiTU: `4 tanh(g / 4) sigmoid(g) * 25 tanh(h
/ 25)`, a kernel of its own, `situ_rows`), each added back by its gate. A picture token's positions are `(0, the prompt's
length, row, column)`, its rows and columns about the picture's centre; a text token's `(0, its index, 0, 0)`.

The text encoder runs the chat engine's Qwen3.5 ops (the delta nets' conv and recurrence, the attention's gate) a layer
at a time, every prompt through each layer: the 9B's 18 GB are never all on the GPU. A Hugging Face checkpoint is read
as llama.cpp's converter makes a GGUF of it (the delta nets' value heads from grouped to tiled order, `-exp(A_log)`, the
RMS norms' `1 + w`). The prompt is Qwen3.5's chat template of one user turn, the assistant's opened with thinking on
(`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`), at most 4,096 tokens, as
`MOSSImagePipeline.encode_prompt` makes it.

## How it was checked

- The transformer against OpenMOSS's own `MOSSImageTransformer` (`pipeline_moss_image.py`, made small with random
  weights, `tools/moss_image/reference.py`): a step's velocity, every block and block 1 skipped, cosine 1.0000000 (the
  worst error 7e-6 of an RMS of 2.9); on the card emulator too.
- The real transformer, one step on a 16 x 16 latent under a real prompt, against OpenMOSS's modules in float32
  (`tools/moss_image/velocity_reference.py`, its 42 blocks streamed): on the RTX 4090 (f16) cosine 1.000000, 0.06% of the
  velocity off; on a Mac (Q8_0) 0.99996, 0.9%; realrebelai's Q8_0 GGUF 0.99996, 0.9%; their Q4_K_M 0.997, 7.8%.
- The text encoder against Hugging Face transformers in float32 (`tools/moss_image/text_reference.py`): the same tokens,
  cosine 1.000000 for every token of a 56-token prompt and of the empty negative one, after 4 layers and after all 32.
  Unsloth's Q4_K_M GGUF of Qwen3.5-9B (its own tokenizer: the same tokens) gives states 0.995 and 0.983 of those, where
  llama.cpp's own states from the same file give 0.994 and 0.980; OAIY's and llama.cpp's are 0.999 and 0.997 alike.
- The schedule against `MOSSImageScheduler` (diffusers 0.41, torch 2.14): every sigma within f32's last place (torch makes
  its grid a SIMD vector at a time, so its last bits are the CPU's).
- OpenMOSS's published examples (`assets/images` of their repository: MOSS-Image-Base, each its prompt, negative prompt,
  size, steps, guidance, shift, skipped block and seed) made again with the Gradio app's own starting noise
  (`tools/moss_image/noise.py`): on the RTX 4090 the same pictures, seven of them (30 and 50 steps, shift 5 and 6,
  block 22, 24 or none skipped), those at cfg 3 or 4 26.8 to 31.5 dB from theirs, those at cfg 5 18.6 to 22.5: guided
  steps carry bfloat16's own rounding into the fine details, more the stronger the guidance.

## Speed

A 1024x1024 picture, 50 steps with guidance (two passes of the transformer a step), everything counted (the text
encoder and the transformer read from an external SSD, the VAE's decode):

| GPU | OpenMOSS's weights | The 4-bit set (Q4_K_M GGUFs) |
| --- | --- | --- |
| RTX 4090 (eGPU) | 132 s (1.8 s a step; 40 s reading their 28 GB) | 86 s (1.5 s a step) |
| Apple M5 Pro (Q8_0 weights) | 25 s a step (some 21 minutes) | 25 s a step |

At 512x512 the M5 Pro takes 5.3 s a step (20 steps in 2.6 minutes).

## Getting it

- **Get models → Images → MOSS-Image SFT** fetches OpenMOSS's release (27 GB: its transformer, Qwen3.5-9B, the VAE and
  the tokenizer) and adds it as one image model.
- **MOSS-Image SFT (4-bit)** fetches realrebelai's Q4_K_M GGUF of the transformer, the VAE and tokenizer from OpenMOSS's
  release, and unsloth's Q4_K_M GGUF of Qwen3.5-9B (9.7 GB, the last shared with the Qwen3.5 9B chat model's download).
- Or add the files by hand on the Models page: OpenMOSS's folder (MOSS-Image-SFT or -Base) is a whole model; a GGUF or
  safetensors file of the transformer finds its parts beside it (Qwen3.5-9B's folder, a VAE, Qwen3.5's tokenizer.json);
  a Qwen3.5-9B GGUF added to a MOSS-Image model is its text encoder.

A request takes `negative_prompt`, `steps` (50), `cfg` (3), `shift` (5) and `uncond_skip_layers` (`[22]`; `[]` turns the
skipping off), each also a model's own setting. A JSON layout goes in as the prompt, with the picture's size the layout's
(see OpenMOSS's user guide).
