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
| Distillation (optional) | A data-space DMD distillation's low-rank factors beside the blocks' matrices, its time embedding, and its 8-pass sampler without guidance ([below](#distillations-pictures-in-8-passes)) | `crates/oaiy-media/src/moss_image/distill.rs` |

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

## Prompting

MOSS-Image reads a description, a list of tags, or a JSON layout, and English best of all. OpenMOSS's guide and its
prompt enhancer (an agent skill, `skills/moss-image-prompt-enhancer` in github.com/OpenMOSS/MOSS-Image) come to this:

- Describe each visible thing concretely: subject, medium, palette, lighting, framing.
- Put the exact text in quotes, with its line breaks (`\n`) where they matter.
- Say what must not appear ("no other letters or words"). A distillation's passes take no negative prompt, so this
  goes in the prompt itself; with guidance OpenMOSS suggest `nsfw, worst quality, bad anatomy, watermark, jpeg artifact`.
- Where text placement matters (a logo, a poster), use a layout: a `summary` and flat `items`, each with a `label`
  (`object`, `character` or `text`), a `caption`, the exact `text` for text, and a box. The skill's
  `scripts/compile_prompt.py` turns a draft with boxes in pixels from the top-left (`abs`, `xyxy`) into the model's
  one-line form (`abs_centralized`, `yxyx`); the picture's width and height must be the draft's.

On the 4-bit set with its 8-step distillation, twelve logos made from short prompts mostly had their text right,
with a few faults: a monogram's background filled with garbled letters, a badge's name misspelt, a word added to an
abstract mark. Made again from prompts written this way (layouts for the monogram, the badge, the bakery's mark and a
Chinese seal), those faults were gone, and line breaks came out as written.

Its weights' precision matters more than the model's own (each pass hands its rounding to the next). Twelve prompts
at one seed, the distance of each picture from the f16 transformer's (mean absolute difference of 255): Q8_0 13 (a
portrait 1.7: the same picture), Q4_K_M 36, a different take of each with softer detail (fur, petals, the
background). Taking any one kind of the 4-bit file's tensors from the Q8_0 file instead (attention q and k, o, the
MLP's up and gate, v and down, or the first and last four blocks) brought it to 32 to 35 at best: the rounding counts
everywhere. On the RTX 4090 its 8 passes take 7.0 s at 4 bits, 7.1 s at Q8_0 and 8.2 s at f16, so the card uses the
Q8_0 file (`moss-image-sft-q8-8-step`); the 4-bit one is for a smaller card.

Checked again on one brief over six seeds (docs/IMAGE_FLOW.md#logos): the 8-step distillation writes a logo's name
twice, adds lines nobody asked for or sets the logo in a room, at full precision as at 4 bits and with its text
encoder on either GPU, where MOSS-Image's own 50 guided steps did not; a layout with its background as an item of its
own (an empty, plain white background over the whole canvas) had the lettering right in all six. Without that item, a
layout's empty space was sometimes filled with copies of the mark.

## Distillations: pictures in 8 passes

A distillation of MOSS-Image-SFT as data-space DMD makes one (arXiv 2610.12102) makes a picture in a few passes of the
transformer, without guidance or a negative prompt, in place of 50 guided steps. It is MOSS-Image's own transformer
with three things of its own:

- Rank-128 low-rank factors for each block's six projections (q, k, v, o, `ffn_up_gate`, `ffn_down`). A factor's
  change to a weight is about 0.02% of it, under a tenth of bf16's rounding step and a third of f16's. Merged into f16
  weights, a cosine of 0.8 of the change survives; merged into Q8_0, none of it does. So OAIY keeps them as f16 beside
  the matrices (f16, Q8_0 or a GGUF's) and adds `B A x` as the step runs.
- Its time embedding and the two modulations' biases, in place of the transformer's.
- Its sampler: from fixed noise `z`, each pass runs the transformer at the next of its step codes (1000, 968, 926,
  871, 791, 671, 463, 24) on `z + Q x0`, where `Q` is its 128 x 128 latent mix and `x0` the clean estimate so far
  (zero at first); the estimate becomes `z - v`, and the last is the picture's latent. Fewer passes are its first
  ones.

OAIY reads it from one file holding all of it (`layers.{i}.proj.{module}.lora_A`/`lora_B`, the replacements,
`latent_mix.Q`, `step_codes`), or from a folder of three: `lora.safetensors` (kohya's names, with its alpha),
`time_embedding.safetensors` and `dataspace.safetensors`. Added on the Models page by itself, it becomes a model of its
own, with the parts of a MOSS-Image model already in the list (named after it, `-8-step`) and the original left as it
was. Added to a MOSS-Image model, that model takes it. Its field on the model's card is **Distillation**; a request's
`steps` (1 to 8) are its passes.

How it was checked:

- On the tiny transformer (`tools/moss_image/distill_fixture.py`: OpenMOSS's transformer with random factors merged in
  float32), a step at two of its times and three passes of its sampler: cosine 1.0000000 on Metal and 0.9999998 on the
  card's emulator. The same holds for its one file, kohya's folder of three, and ComfyUI's fused q, k and v.
- On the real transformer against OpenMOSS's modules in float32 with the factors merged
  (`tools/moss_image/distill_reference.py`, a 16 x 16 latent, all 8 passes):
  - On the RTX 4090 (f16) each pass is 0.2 to 0.8% off, and the final estimate has cosine 0.986: the same picture.
  - On a Mac (Q8_0) each pass is about 5% off. The reference with its own weights rounded to Q8_0 is just as far off,
    and the error carries through the passes (the latent mix hands each pass the last's estimate), so the picture is
    another one of the same scene (cosine 0.86).
  - With a Q4_K_M GGUF of the transformer, the passes have cosine 0.93 to 0.98 and the final estimate 0.64, still the
    same scene.

At 1024x1024, everything counted:

| GPU | Its passes | The whole picture |
| --- | --- | --- |
| RTX 4090 (eGPU), OpenMOSS's weights | 8.6 s | 54 s (most of it reading 28 GB from an external SSD) |
| RTX 4090, the 4-bit set (Q4_K_M GGUFs) | 7.2 s | 21 s |
| Apple M5 Pro (Q8_0) | 116 s | 158 s (some 21 minutes for the 50 guided steps) |

Between pictures the studio keeps the worker (`oaiy-media --serve`), and with it the text encoder, the transformer and
its distillation and the VAE, for as long as the Memory page's **Keep the picture model loaded** says (2 minutes unless set; 0
lets it go after each picture), or until a chat wants the GPU. The text states of its last 16 prompts are kept too. A
GGUF text encoder's layers stay on a card (the 4-bit 9B's 5.5 GB). A checkpoint's 18 GB of f16 do not fit beside the
f16 transformer on 24 GB, so it goes on reading them a layer at a time; on a Mac, whose GPU memory is the computer's,
it does that too. On the RTX 4090 at 1024x1024, the kept worker alone:

| Picture | The 4-bit set | OpenMOSS's weights |
| --- | --- | --- |
| The first (everything loaded) | 24.4 s | 49.7 s |
| The next, a new prompt | 7.6 s | 33.3 s (its text encoder read again) |
| The next, a prompt it has had | 7.4 s | 9.8 s |

Through OAIY's API, the card lent from its language model first, the 4-bit set's first picture took 29.5 s and each
next one with a new prompt 8.6 s. A chat while the worker waits takes the card back at once; the language model then
loads again (the 35B some 100 s), as after any picture on the card.
