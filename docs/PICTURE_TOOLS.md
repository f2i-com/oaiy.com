# Picture tools

NROB has two tools for pictures, natively in Rust (`nrob-diffusion`,
`kind: "picture"`):

- **Background removal** with BiRefNet: the subject kept, the rest made
  transparent. The result is an RGBA PNG at the picture's own size.
- **Upscaling** with Real-ESRGAN x4plus: a picture two or four times larger,
  with its detail restored rather than blurred. A transparent picture stays
  transparent.

Pixal3D uses both to prepare its pictures ([3D models](MODEL3D.md)).

## The models

| Tool | Download | Licence |
|---|---|---|
| BiRefNet | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) (`config.json`, `model.safetensors`) | MIT |
| Real-ESRGAN x4plus | `RealESRGAN_x4plus.pth` from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0) | BSD-3-Clause |

Get models downloads both ([Getting models](STUDIO.md#getting-models)). Or pick
the BiRefNet folder, or the `.pth`, on the **Models** page. Either one fills
`media.picture` (`background`, `upscaler`), and a 3D model borrows them.

BiRefNet's own weights are MIT. RMBG-2.0 is the same network with weights for
non-commercial use only; a RMBG-2.0 folder works too, and the Models page says
what its licence allows.

`RealESRGAN_x4plus.pth` is read by nrob's own tensor reader, which takes only
the tensors and refuses anything else in the pickle, so no code from the file
runs.

## How they work

**BiRefNet** runs its Swin-L backbone on the picture at 1024² and 512², joins
the two, and adds the finer levels' context to the coarsest. Its decoder then
works up from 32² to 1024². Each step uses modulated deformable convolutions,
an attention gate, and the picture itself laid out as patches. The logits'
sigmoid becomes the alpha, resized to the picture with Pillow's bicubic, as
BiRefNet's own code does.

**Real-ESRGAN** is basicsr's RRDBNet: 23 residual-in-residual dense blocks and
two doublings. It runs in F16 on a GPU, in 256-pixel tiles with 10 pixels of
overlap. Twice as large is four times, then halved (Real-ESRGAN's own
`outscale`).

## Speed

On an RTX 5090, removing a 1200×1000 picture's background takes about 1 s.
Making a 300-pixel picture four times larger takes about 1 s. A 600×400 one
takes under 2 s.

## API

Both answer at once, as OpenAI Images does: `{created, data: [{b64_json}],
output_format: "png", width, height, size, model, seconds_taken}`. Ask for
`response_format: "url"` to get a link to the file instead.

| Request | Fields |
|---|---|
| `POST /v1/images/background_removal` | `image`: a `data:` URL (PNG, JPEG or WebP) |
| `POST /v1/images/upscale` | `image`: a `data:` URL, up to 4 megapixels; `scale`: 2 or 4 (default 4) |

```sh
curl http://127.0.0.1:8080/v1/images/background_removal -H "Content-Type: application/json" \
  -d "{\"image\": \"data:image/png;base64,$(base64 -w0 knight.png)\"}"
```

Companion apps find both in `GET /v1/discovery` (`models.background`,
`models.upscale`, and the `background` and `upscale` endpoints).

## Verification

Each stage was compared with the models' own PyTorch code, from the same inputs:

| Model | Result |
|---|---|
| BiRefNet: every backbone level, the encoder, the squeeze block, one decoder block | within 3e-6 relative RMS |
| BiRefNet: the logits at 1024² | within 5.4e-6 relative RMS; no pixel of the matte on the other side of 0.5 |
| Real-ESRGAN x4plus | within 7e-7 relative RMS in F32, 7e-4 in F16 |
