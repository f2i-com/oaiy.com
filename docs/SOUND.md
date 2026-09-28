# Sound effects

OAIY makes sound effects with MOSS-SoundEffect v2.0, natively in Rust
(`oaiy-media`, `kind: "sound"`): ambience, weather, crowds, creatures,
machines, footsteps, impacts. You describe what makes the sound, where, and how
it sounds ("heavy rain on a tin roof with distant thunder"), and get up to 30
seconds of 48 kHz mono audio.

The studio serves it as asynchronous jobs on `/v1/audio/sound_effects`, like
music.

## The model

Download [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0).
On the **Models** page, pick the folder. The studio recognises it from its
`model_index.json` (`MossSoundEffectPipeline`) and adds `moss-soundeffect`.

It has three parts:

1. **Text encoder.** Qwen3-1.7B reads the description, with the length appended
   (" duration: 4.0s"). Its last hidden states, padded to 512 rows, are what
   the transformer attends to.
2. **Transformer.** A 1.3B one-dimensional diffusion transformer (30 blocks)
   denoises 128-channel latents at 50 frames a second. It uses flow matching:
   100 Euler steps, shift 5, and guidance 4 against an empty description. It
   always works on the full 30-second window; a shorter sound is cut from it.
3. **Decoder.** A DAC decoder turns the latents into 48 kHz audio.

The decoder's `.pth` checkpoint is read without Python: a restricted reader
takes only its tensors, and anything else in the pickle is refused.

## Speed and memory

On an RTX 5090, a sound takes about 21 seconds whatever its length: 20 s for
the 100 steps and under 2 s to encode and decode. It needs about 4 GB of VRAM.

## API

### `POST /v1/audio/sound_effects`

This creates a job and returns it at once. The fields:

| Field | Meaning |
|---|---|
| `prompt` | The description (also accepted as `input`, `text` or `description`). |
| `seconds` | Length, 0.1 to 30 (default 10). Also accepted as `duration`. |
| `negative_prompt` | What to steer away from (default: nothing). |
| `seed` | Repeats a take exactly. |
| `steps`, `guidance` | The sampler's steps (100) and guidance (4; also accepted as `cfg_scale`). |
| `model` | A sound effects model; left out, the default. |

The rest works as music jobs do:

| Request | Does |
|---|---|
| `GET /v1/audio/sound_effects/{id}` | Polls a job (`queued`, `in_progress`, `completed`, `failed`; `progress` 0-100). |
| `GET /v1/audio/sound_effects/{id}/content` | Downloads the sound as WAV. Add `?format=mp3`, `opus`, `aac`, `flac` or `pcm` for another format, which needs FFmpeg. |
| `GET /v1/audio/sound_effects` | Lists sound jobs. |
| `POST /v1/audio/sound_effects/{id}/cancel`, `DELETE /v1/audio/sound_effects/{id}` | Stops a job, or forgets it. |

```sh
curl http://127.0.0.1:8080/v1/audio/sound_effects -H "Content-Type: application/json" \
  -d '{"prompt": "a wooden door creaking open slowly in an empty hall", "seconds": 4, "seed": 3}'
```

The Playground has a **Sound** tab, and companion apps find the endpoint and
the models in `GET /v1/discovery` (`models.sound`).

## Verification

From the same starting noise, the Rust pipeline's waveform correlates 0.9965
with the official diffusers pipeline's, and every second of it at least 0.995.
Their log spectrograms correlate 0.985.
