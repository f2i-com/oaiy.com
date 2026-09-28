# Music

OAIY writes complete songs with MiniMax Music 3, natively in Rust
(`oaiy-media`, `kind: "music"`). A song has vocals and instruments, is
44.1 kHz stereo, and is up to six minutes long. You give it:

- **lyrics**, with section tags such as `[Verse]` and `[Chorus]` on lines of
  their own;
- **a description** of the style: genre, tempo, key, mood, vocals, instruments
  and production.

The studio serves it as asynchronous jobs on `/v1/audio/music`. It also serves
it on OpenAI's `/v1/audio/speech`, the way the official server takes it.

## The model

Download [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
(27 GB). On the **Models** page, pick the folder. The studio recognises it from
its `config.json` (`model_type: minimax_music3`) and adds `minimax-music3`.

It runs in two stages, and only one is in memory at a time:

1. **Composing.** The 8B language model (a Qwen3) writes 25 frames a second.
   For each frame it picks a semantic code, and the 4-layer depth decoder picks
   seven residual codes. Both use classifier-free guidance at 1.5. Each frame
   keeps the hidden states of both models, eight vectors of 4096.
2. **Rendering.** A flow-matching transformer (2.4B, 36 blocks) turns those
   hidden states into audio latents. It works in 200-frame windows that overlap
   by 100, with 30 Euler steps and guidance 1.7. Each window's first 172
   latents are pinned to the previous window's, so the song runs on without a
   seam. The Flow-VAE decoder then turns the latents into audio.

Only the vocabulary the model can emit is touched. The output head keeps its
16,384 code rows and the end-of-song row, not all 200,000, and the prompt's
embeddings are read row by row from the file.

## Smaller language models

The language model is 16.4 GB in BF16. **Make a smaller language model** on the
Models page converts it once, in under a minute. The same conversion is the
worker's `kind: "music_quantize"`. The new file is written beside the model,
and the model uses it for every song from then on. The original stays as it is.

| Format | File | GPU memory while composing (approx.) | Greedy picks that match the reference |
|---|---|---|---|
| BF16 (the original) | 16.4 GB | about 16 GB | 322 of 328 |
| `q8_0` | 8.3 GB | about 9 GB | 316 of 328 |
| `q6_k` | 6.6 GB | about 8 GB | not measured |
| `q5_k` | 5.7 GB | about 7 GB | not measured |
| `q4_k` | 4.8 GB | about 6 GB | 273 of 328 |

Only the 36 layers are quantized, in GGUF block formats. The norms, the output
rows and the codes' input rows stay at full precision, and the text embeddings
are `q8_0`.

- **`q8_0`** is nearly indistinguishable from the original. Where its greedy
  pick differs, the two candidates were within 0.31 logits.
- **`q4_k`** drifts more, but the model samples from its top 50 anyway.

## Memory

Like the other media models, music follows `memory`, `vram_gb` and `ram_gb`
(`media.music`, a model, or the request). Each stage reserves only what it
needs beside its weights: the song's KV cache while composing, and room for the
decoder while rendering. The layers or blocks that do not fit stay in RAM and
are copied up for each use.

On an RTX 5090:

| Setup | Composing | Rendering 20 s |
|---|---|---|
| BF16, all on the GPU | 30 frames/s (1.2x real time) | 23 s |
| `q4_k`, 4 GB of weights on the GPU | 27 frames/s | 20 s |
| `q4_k`, 2 GB on the GPU, the rest in RAM | 6 frames/s | 39 s |

With `q4_k`, a card with 8 GB composes faster than real time. A minute of music
takes about two minutes in all.

## API

### `POST /v1/audio/music`

This creates a song job and returns it at once. The fields:

| Field | Meaning |
|---|---|
| `prompt` | The description (also accepted as `instructions`, `description` or `style`). |
| `lyrics` | The lyrics (also accepted as `input`). Required unless `instrumental` is true. |
| `instrumental` | `true` without lyrics: a song with no words. `[Intro]\n(instrumental)` stands in for the lyrics, and "no vocals" is added to the description. |
| `duration` | Upper bound in seconds, 1 to 360 (default 60). Also accepted as `seconds`, `max_seconds`, or `max_new_tokens` in frames. The song may end sooner. |
| `seed` | Repeats a take exactly. |
| `steps`, `guidance` | The renderer's Euler steps (30) and guidance (1.7). |
| `model` | A music model; any name containing "music" means the default. |

```json
{"id": "music_...", "object": "music", "model": "minimax-music3", "status": "queued",
 "progress": 0, "prompt": "...", "lyrics": "...", "seconds": 60.0, "seed": 7,
 "sample_rate": 44100, "channels": 2, "finish_reason": null, "error": null}
```

The rest of the API works on those job objects:

| Request | Does |
|---|---|
| `GET /v1/audio/music/{id}` | Polls a job. `status` is `queued`, `in_progress`, `completed` or `failed`, and `progress` runs 0-100. When it is done, `seconds` is the song's length and `finish_reason` is `stop` (the song ended) or `length`. |
| `GET /v1/audio/music/{id}/content` | Downloads the song as WAV. Add `?format=mp3`, `opus`, `aac`, `flac` or `pcm` (16-bit, 44.1 kHz stereo, interleaved) for another format, which needs FFmpeg. |
| `GET /v1/audio/music` | Lists song jobs. |
| `POST /v1/audio/music/{id}/cancel`, `DELETE /v1/audio/music/{id}` | Stops a job, or forgets it. |

```sh
curl http://127.0.0.1:8080/v1/audio/music -H "Content-Type: application/json" -d '{
  "prompt": "Energetic indie rock at 128 BPM: jangly electric guitars, driving bass, live drums, bright male vocals.",
  "lyrics": "[Verse]\nCity lights are calling out my name\n[Chorus]\nWe run, we never look back",
  "duration": 30, "seed": 11}'
```

### Through `/v1/audio/speech`

Name a music model to get the whole song back in the reply. The official
server's form works as it is: `input` holds the lyrics, `instructions` the
description, and `max_new_tokens` counts frames.

```sh
curl http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" -d '{
  "model": "MiniMaxAI/MiniMax-Music3", "input": "[Intro]\n(instrumental)",
  "instructions": "An instrumental ambient piece, no vocals: warm analog pads, 70 BPM",
  "response_format": "wav", "seed": 3, "max_new_tokens": 250}' -o ambient.wav
```

Companion apps find the endpoints and music models in `GET /v1/discovery`.

## Writing the prompt

- Put each structure tag on its own line: `[Intro]`, `[Verse]`,
  `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Solo]`, `[Outro]`. Any text after a
  tag on the same line is dropped, as the model was trained.
- Name the instruments, the tempo and the production. Vague descriptions give
  generic arrangements.
- For an instrumental, send `instrumental: true`, or ask for one in the
  description and keep the lyrics to `[Intro]\n(instrumental)`.
- Markdown in the description (headings, bullets, bold) is removed first, and
  `<|tag value|>` becomes "tag is value". This is the model's own input
  cleaning.

## Verification

The reference activations come from the official diffusers pipeline. The
acoustic stack runs in strict F32 (TF32 off); the language model and depth
decoder in BF16, as stored. The opt-in tests are `music::acoustic::golden`,
`music::lm::tests` and `music::tests`. They need `OAIY_MUSIC_GOLDEN` and
`OAIY_MUSIC_MODEL`, plus `OAIY_MUSIC_LM` for a quantized file.

| Stage | Relative RMS error |
|---|---|
| Flow-VAE decoder, both windows (F32) | 2e-6 |
| Condition encoder (F32) | 8e-7 |
| Transformer, one pass: F32 / BF16 | 1e-6 / 1.5% |
| The whole two-window render with the reference's noise: F32 / BF16 | 4e-6 / 3.7% |
| Depth decoder (BF16) | 0.7% |
| Language model prompt pass (BF16) | 1.7% |

The prompt's tokens match the reference exactly.

With the reference's own codes fed back for 40 frames, every frame's hidden
states agree within 3%. 322 of 328 greedy choices are the same; where one
differs, the two candidates were within 0.13 logits.
