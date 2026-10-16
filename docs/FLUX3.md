# FLUX 3 Image Klein

Black Forest Labs' FLUX 3 Image Klein (October 2026) runs natively in OAIY's media worker on WebGPU, on a GPU of any
make (the Mac's through Metal, NVIDIA's, AMD's, Intel's, and an NVIDIA card in a Thunderbolt enclosure through the
eGPU plugin): pictures from a description, edits of one reference picture, and compositions of up to ten, through one
model. Architecture `flux3-klein`.

| Piece | What OAIY runs | Code |
| --- | --- | --- |
| Transformer (5.6B) | BFL's `model.safetensors`, the distilled checkpoint (a flow map: 16 steps, no guidance) or the base one (50 steps, guidance 3.5); its blocks' matrices f16 where the tensor cores take f16 (NVIDIA's, AMD's) and there is room, else its BF16 ones as Q8_0 (a Mac's GPU: Metal's simdgroup matrices take Q8_0, not f16); `OAIY_FLUX3_WEIGHTS` `f16` or `q8_0` says which | `crates/oaiy-media/src/flux3_wgpu.rs` |
| Text encoder | Qwen3-VL-4B-Instruct's language model (its vision tower unused), the states of every fourth layer to the 32nd side by side (20,480 wide), the prompt right-padded to the next multiple of 80 tokens | `crates/oaiy-media/src/klein_text_wgpu.rs` (`FLUX3`) |
| VAE | The FLUX.2 VAE (32 channels, 2x2 patches, batch-norm statistics): its encoder for references, its decoder for the picture | `crates/oaiy-media/src/sdxl_vae_wgpu.rs`, `crates/oaiy-media/src/klein/vae.rs` |
| Pipeline | References prepared as BFL's pipeline prepares them, the schedule `6.93 t / (1 + 5.93 t)`, a flow-map step (`x + dt v(x, t, t_next)`) or a guided Euler step | `crates/oaiy-media/src/flux3/` |

The transformer differs from FLUX.2 Klein's: the picture's tokens, its references' and the text's each go first through
five blocks of their own weights, attending only among themselves, then all three through 28 shared blocks attending
together, each kind still modulated by its own time (the picture's the step's, the rest's zero, plus the time the step
goes to in the flow map). A block is one stream (a modulated layer norm, then attention beside a SwiGLU MLP, both added
by one gate), its positions four axes of a head's quarter each (rows and columns halved; references told apart by the
last axis), and nothing has a bias.

## Speed

A 1024x1024 picture from BFL's distilled checkpoint, 16 steps, everything counted (the text encoder and the
transformer read from an external SSD, the VAE's decode):

| GPU | Text-to-image | One reference | Two references |
| --- | --- | --- | --- |
| RTX 4090 (eGPU, the 0.1.6 plugin's kernels) | 39 s | 54 s | 73 s |
| Apple M5 Pro (Q8_0 weights) | 3.7 min (11.6 s a step) | | |
| Apple M5 Pro (f16 weights) | 7.4 min (25.5 s a step) | 15.3 min | |

## Getting it

The weights are gated on Hugging Face under the **FLUX Kommunity License** (non-commercial use, with a commercial-output
exception for qualifying users: read its terms). Sign in on huggingface.co, accept the license on
[black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (approval is
automatic), and give OAIY a read token: the **Hugging Face token** field of **Get models**. Then:

- **Get models → Images → FLUX 3 Image Klein** fetches `model.safetensors`, `ae.safetensors` and `text_encoder/`
  (20.3 GB) and adds the model; or
- add BFL's release folder (or its `model.safetensors`) yourself: its `text_encoder/` folder and VAE are found beside
  it. The text encoder is byte for byte `Qwen/Qwen3-VL-4B-Instruct` (Apache 2.0, ungated) and the VAE the FLUX.2 VAE
  (`vae/diffusion_pytorch_model.safetensors` of the same repository, or Comfy-Org's `flux2-vae.safetensors`), so copies
  already on disk serve.

A model entry:

```json
"flux3": {
  "architecture": "flux3-klein",
  "transformer": "flux-3-image-klein/model.safetensors",
  "text_encoder": "flux-3-image-klein/text_encoder",
  "vae": "flux-3-image-klein/ae.safetensors"
}
```

Optional: `tokenizer` (else the text encoder folder's `tokenizer.json`), `steps` and `cfg` (else the checkpoint's own),
`reference_pixels` (a single reference's canvas at most this many pixels; OAIY's default is 1 MiB, BFL's own 4 MiB),
`loras` (below).

## LoRAs

A model's `loras` (`[{"path": "tarot.safetensors", "strength": 1}]`, or plain paths; strength 1 is as trained, from -4
to 4) stack, each merged into the transformer's weights as it loads (on the GPU, as f16; where the blocks run as Q8_0,
merged first and then quantized), so a step takes no longer with them. In the Studio, add a LoRA file and it joins the
FLUX 3 model's list, or use **LoRAs** in the model's settings; a request with `use_loras: false` makes the picture
without them. The layouts trainers save are read:

- BFL's own names (`content_mode_blocks.image.0.q_proj`, `single_blocks.3.mlp_in`, ...), as ComfyUI's and kohya's
  (`diffusion_model.` prefix, `lora_down`/`lora_up` with `alpha` tensors) and ai-toolkit's (`lora_A`/`lora_B`) keep them;
- diffusers' (`transformer.mode_blocks.image.0.attn.to_q`, `single_transformer_blocks.3.ff.linear_in`, ...), as its
  DreamBooth trainer saves them, their scale from the file's `lora_adapter_metadata` (`lora_alpha` over the rank, a
  module's own in `alpha_pattern`). diffusers keeps the final modulation's two halves the other way round from BFL
  (`norm_out.linear`: scale, then shift), so a LoRA's factors for it are turned round to fit.

Every matrix may be adapted: the blocks' attention and MLP, the inputs, the times' embeddings, the modulations and the
output. A LoRA is checked from its header before anything loads: one with a module the checkpoint does not have (one for
FLUX.2 Klein, say) or of another shape is refused, and so is a file that keeps more than LoRA factors (a distillation
extract's new layers and norm changes, a DoRA's magnitudes), which would be applied only in part.

Most FLUX 3 LoRAs are trained on the base checkpoint and used on the distilled one, as their makers advise. Those
published so far (October 2026) all fit whole: a tarot-card style (diffusers; its "more layers" variant adapts the
modulations too), an outpainting LoRA (ai-toolkit; pad the picture with flat gray `#808080`, give it as the reference,
and prompt with its instruction, at strength 1.5), and a 4-step distillation of the base (ComfyUI's names: give the base
model `steps: 4` and `cfg: 1` with it). On the RTX 4090 (the 0.1.6 plugin's kernels, none new) the tarot LoRA turns
a seed's picture into a Rider-Waite card in the same 42 s (800x1344) the picture takes without it; the outpainting LoRA
keeps a padded picture's middle where it was (27.3 dB from the original's pixels, unmoved), where without it the middle
slid 6 px (17.5 dB).

## A job

`oaiy-media` takes `architecture: "flux3-klein"`, `transformer`, `text_encoder`, `vae`, `prompt` (or `prompts` with
`n`), `output_dir`, and optionally:

- `images`: up to ten reference pictures to edit or compose from. One is fitted within its canvas (sides multiples of
  16, 64 to 4,096) and mirrored out to it; several are each fitted under 1 MiB pixels and cut to multiples of 16.
- `width` and `height`: multiples of 16, at most 2048 x 2048 in area. Without them the picture is the first
  reference's size (a single padded reference's picture cut back to the reference's own shape), or 1024 x 1024.
- `steps`, `cfg` (base only), `seed`, `device`, `reference_pixels`.
- `loras`: `[{path, strength}]` or paths.

Through the Studio (`/v1/images/generations`, `/v1/images/edits`) an edit at `size: auto` takes the reference's size
that way, its canvas about a megapixel. Discovery says `edits: true` and `max_references: 10`.

Not supported: negative prompts (the base checkpoint's guidance is against the empty prompt, as BFL's), the CPU backend, and BFL's two extra models: the **prompt upsampler** (a Qwen3.5-9B fine-tune that writes a full scene
caption from a short request) and the **safety classifier**. The model follows full scene captions far better than
short requests: write out the framing, the subject, the background and the light. Bounding boxes are part
of the prompt's text (a JSON list of elements on a 0 to 1000 grid after the caption, each tagged in it with `<id>`), so
they work as BFL's model card describes.

## Checks

- `flux3_wgpu::tests::a_tiny_transformer_on_webgpu_is_bfls`: BFL's own `dit.py` made small (32 wide, two heads, one
  block a kind, two shared; random weights rounded to f16) against the WebGPU transformer, with two references and
  without; fixture by `tools/flux3/reference.py`.
- `klein_text_wgpu::tests::flux3_text_is_hugging_faces` (ignored: the real encoder): Hugging Face's Qwen3-VL in float32
  as BFL's encoder runs it (`tools/flux3/text_reference.py`) against the WebGPU encoder, the padding included.
- `sdxl_vae_wgpu::tests::a_small_vae_encoder_on_webgpu_is_candles`, `flux3::tests::a_reference_round_trips_through_the_vae`
  (ignored: the real VAE), `flux3::references::tests`, `flux3::tests::the_schedule_is_bfls`.
- `flux3_wgpu::tests::diffusers_loras_are_diffusers`: diffusers' own `Flux3Transformer2DModel` at the tiny sizes, with two
  LoRAs applied by PEFT (`tools/flux3/lora_reference.py`: one made by PEFT and saved by diffusers, adapting every linear
  layer, the final modulation among them, one in BFL's names loaded through diffusers' conversion), against the WebGPU
  transformer with the same files merged as it loads: cosine 0.9999996, the worst error 0.008 of an RMS 2.3 (0.96 with
  the final modulation's halves left as they are stored). Without LoRAs diffusers' transformer is BFL's DiT to 9e-6.
- `flux3_wgpu::tests::loras_merged_as_it_loads_are_the_weights_merged_before`, and (ignored: real files)
  `real_loras_fit_the_checkpoint` (`OAIY_FLUX3_LORAS`, a folder of them).
- `flux3::tests::the_examples_are_bfls` (ignored: the real model): BFL's published examples made again from their records
  and their own starting noise (`tools/flux3/noise.py`), each against theirs. On the RTX 4090 (f16): text-to-image
  37.4 dB, with boxes 46.9, one reference 39.2 to 55.2 (with boxes 54.4), two references 54.1, and the busiest picture
  (a figurine's sand and pebbles) 28.2, the same picture by eye; on the Mac's GPU (Q8_0) text-to-image 34.9 dB.

BFL's own VAE file (`ae.safetensors`) keeps the 1x1 convolutions either side of the latent inside the encoder and the
decoder (`encoder.quant_conv`, `decoder.post_quant_conv`); OAIY's VAE loaders read them there as well as beside them
(before, from that file, they went unused: a picture came back at 12.9 dB where 36.1). FLUX.2 Klein's VAE is the same
file.

See also `docs/licenses/FLUX3-NOTICE.md`.
