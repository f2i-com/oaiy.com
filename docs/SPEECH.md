# Speech

NROB speaks with Qwen3-TTS (12 Hz, 1.7B), natively in Rust (`nrob-diffusion`,
`kind: "speech"`). The studio serves it as OpenAI's `audio.speech`. It adds
**saved voices**: describe a voice once, keep it, and every later line uses the
same voice.

## Models

Use two folders from the Qwen3-TTS release:

| Folder | What it does |
|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Speaks in any voice described in words ("a warm, calm middle-aged woman with a slight Irish accent"). |
| `Qwen3-TTS-12Hz-1.7B-Base` | Speaks in a saved voice. Its speaker encoder and codec encoder also make the voice when you save one. |

On the **Models** page, pick either folder (or the `model.safetensors` inside
it). The studio reads `config.json` and pairs the two into one speech model,
`qwen3-tts`. VoiceDesign alone gives described voices. Saving voices also needs
Base.

## How a saved voice is made

When you save a voice:

1. VoiceDesign speaks a sample line in the described voice.
2. The Base model's speaker encoder turns that clip into a 2048-value speaker
   embedding. It is an ECAPA-TDNN over a log-mel spectrogram.
3. The codec's encoder (Mimi) turns the clip into reference codes.
4. The voice file under `voices/` keeps the embedding, the codes and the
   transcript: a few tens of kilobytes. The sample clip is kept beside it.

Later lines prompt the Base talker with this voice, in context. The codec
prefix carries the speaker embedding, and the reference transcript and the new
text run over the clip's frames. The voice therefore holds from line to line.
Two new lines in a saved voice measured 0.99 speaker similarity with it, against
0.95 for another voice.

The Playground's **Save this voice…** button uses the same description, line
and seed you just heard, so the saved voice is exactly that one.

## API

### `POST /v1/audio/speech`

OpenAI's request, plus a few extensions. It returns the audio itself.

| Field | Meaning |
|---|---|
| `input` | The text to speak (up to 20000 bytes). |
| `voice` | A saved voice's name, or an OpenAI voice name (`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`, each given a matching description). May also be `{"id": "..."}`. |
| `instructions` | Describe any voice in words; this takes precedence over an OpenAI voice name. It is ignored for saved voices. |
| `model` | A speech model; `tts-1`, `tts-1-hd` and `gpt-4o-mini-tts` mean the default. |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, or `pcm` (16-bit mono, 24 kHz). |
| `speed` | 0.25 to 4 (FFmpeg's `atempo`, so the pitch is kept). |
| `language` | Extension: `auto`, `english`, `chinese`, `japanese`, `korean`, `german`, `french`, `spanish`, `italian`, `portuguese`, `russian`. A saved voice defaults to its own. |
| `seed`, `max_seconds` | Extensions: a fixed seed repeats a take; the length cap defaults to 120 s. |

The whole clip comes back at once: `stream_format: "sse"` is not supported.
Formats other than `wav`, and any speed other than 1, need FFmpeg (the video
section's `ffmpeg`).

```sh
curl http://127.0.0.1:8080/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Good morning!", "voice": "nova"}' -o hello.mp3
```

### `/v1/audio/voices`

| Request | Does |
|---|---|
| `GET /v1/audio/voices` | Lists saved voices (name, description, language, sample text, time). |
| `POST /v1/audio/voices` | Designs and saves a voice. JSON: `name`, `description`, and optionally `sample_text`, `language`, `seed`, `replace`. Takes about 10 seconds. |
| `GET /v1/audio/voices/{name}` | One voice. |
| `GET /v1/audio/voices/{name}/sample` | Its sample clip (WAV). |
| `DELETE /v1/audio/voices/{name}` | Deletes it. |

Names are 1-64 letters, digits, spaces, `-` or `_`. In incognito, saving a voice
is refused, because incognito keeps nothing. Speech itself works as usual and
leaves nothing behind.

Companion apps find both endpoints, the speech models and the saved voices in
`GET /v1/discovery`.

### In a video

`POST /v1/videos` takes `speech` with the same fields: `text`, `voice`,
`instructions` and so on. It speaks them first and then makes a clip that
follows the speech, with mouths moving to it. A saved voice keeps a character
sounding the same from clip to clip. In the Playground's Video tab, pick
**Someone says this…** as the soundtrack. See
[following a soundtrack](LTX_VIDEO.md#following-a-soundtrack).

## Speed and memory

On an RTX 5090:

- The talker writes about 14 frames a second, against 12.5 for real time. The
  official PyTorch implementation writes 8.2.
- 8 seconds of speech takes about 9 seconds in all: 1.5 s to load, then
  speaking, then 0.3 s to decode.
- Saving a voice takes 7-10 seconds.

The talker and codec need about 3.5 GB of VRAM (BF16 talker, F32 codec). The
text embedding table stays in the file: only the prompt's rows are read.

## Verification

Reference activations come from the official `qwen-tts` package (strict F32 for
the F32 parts, TF32 off). The opt-in tests are `tts::codec`, `tts::clone` and
`tts::tests`.

| Stage | Relative RMS error |
|---|---|
| Codec decoder, every stage to the waveform (F32) | ~2e-6 |
| Speaker mel / speaker embedding (F32) | 4e-7 / 7e-5 |
| Codec encoder SEANet (F32) | 8e-6; every semantic code identical, 1428 of 1440 codes identical |
| Talker prefill embeddings / output (BF16) | 0.2% / 1.4% |
| Saved-voice prefill / Base talker output (BF16) | 0.5% / 1.5% |

The tokenizer reproduces the reference's token ids exactly. Greedy decoding
reproduces the reference's first frames; later frames differ only where BF16
near-ties flip.
