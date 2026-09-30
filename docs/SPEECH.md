# Speech

OAIY speaks with Qwen3-TTS (12 Hz, 1.7B) or [Breeze TTS 2](#breeze-tts-2),
natively in Rust (`oaiy-media`, `kind: "speech"`). The studio serves both as
OpenAI's `audio.speech`. It adds **saved voices**: describe a voice once, keep
it, and every later line uses the same voice.

For calls, [`oaiy-tts`](#realtime-speech-oaiy-tts) speaks as it goes, in real
time on the GPU, in a voice cloned from a clip of anyone speaking.

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

## Breeze TTS 2

[BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) is
an alternative to Qwen3-TTS: one folder speaks in described voices and in saved
ones. **Its weights, and what you make with them, are for research and
non-commercial use only.** The studio shows this beside the model, and
discovery reports it as the model's `license`.

On the **Models** page, pick the folder. The studio recognises it from its
`config.json` (`model_type: breeze`) and adds `breeze-tts-2`. The default
speech model stays as it was; choose Breeze per request with `model`, or make
it the default on the Models page.

It has three parts:

1. **Text encoder.** A T5Gemma 2 encoder (26 layers) reads the text and the
   voice description. Each text part is encoded on its own and projected into
   the talker.
2. **Talker.** A Qwen3 backbone (28 layers) picks each frame's first code, and
   a 12-layer depth decoder picks the other 15. Both use classifier-free
   guidance toward the description (4 by default; `cfg_scale` changes it).
3. **Codec.** Breeze uses Qwen3-TTS's 12 Hz codec (its `audio_tokenizer/`
   folder), so the audio is 24 kHz mono.

A voice is saved the same way, as reference codes and the transcript of a
sample line, which Breeze reads in context before the new text. It needs no
speaker embedding, so a voice Breeze makes has none. That has one consequence:

- A voice Qwen3-TTS made can be spoken by Breeze.
- A voice Breeze made can only be spoken by Breeze. Asking a Qwen3-TTS model
  for it is refused with a message saying so.

## API

### `POST /v1/audio/speech`

OpenAI's request, plus a few extensions. It returns the audio itself.

| Field | Meaning |
|---|---|
| `input` | The text to speak (up to 20000 bytes). |
| `voice` | A saved voice's name, or an OpenAI voice name (`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`, each given a matching description). May also be `{"id": "..."}`, or a voice itself: the object `POST /v1/audio/voices` with `keep: false` handed back (nothing needs to be saved here). |
| `instructions` | Describe any voice in words; this takes precedence over an OpenAI voice name. It is ignored for saved voices. |
| `cfg_scale` | Extension, Breeze TTS 2 only: how closely a described voice follows its description, 0 to 20 (default 4). |
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
| `POST /v1/audio/voices` | Designs and saves a voice. JSON: `name`, `description`, and optionally `sample_text`, `language`, `seed`, `replace`, and `model` (the speech model that designs it). Takes about 10 seconds. With `keep: false` (the default in incognito, where saving is refused) nothing is saved: the reply also holds `voice` (the voice itself) and `sample` (`{format: "wav", data}` in base64), for the caller to keep and send as `voice`, with `sample` inside it for a talking video's reference voice. |
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

## Realtime speech: `oaiy-tts`

Calls hear it through `crates/oaiy-voice`, OAIY's voice server (the desktop's
`oaiy-voice` service), which also runs Parakeet speech-to-text. There a voice
is a clip in the voices folder, and a clip with nothing written beside it is
transcribed by the server itself, so an MP3 alone makes a voice:

```sh
oaiy-voice --mode both --port 8783 --stt-model-dir parakeet-tdt-0.6b-v2   --tts-model-dir Qwen3-TTS-12Hz-0.6B-Base --model-dirs E:/models --voices-dir voices
curl http://127.0.0.1:8783/v1/audio/speech -H "Content-Type: application/json"   -d '{"input": "Hi, thanks for calling!", "voice": "receptionist", "response_format": "pcm"}' -o hi.pcm
```


`crates/oaiy-tts` is a library for live calls. It uses Qwen3-TTS-12Hz-0.6B-Base
(Apache-2.0; the 1.7B Base works the same) and streams 16-bit, 24 kHz PCM
while it speaks. A voice is cloned from a short clip and its transcript.
`oaiy-media` shares its layers: the talker, the code predictor, the codec, and
the encoders that make a voice.

```rust
let dev = oaiy_tts::cuda(1)?;                       // the GPU to use
let mut tts = oaiy_tts::Tts::load(model_dir, &dev)?; // loads and warms up
let voice = tts.voice_from_audio(clip, Some(transcript))?; // mp3, wav, ...
let cancel = AtomicBool::new(false);                // set it to stop (barge-in)
let report = tts.speak(text, &voice, &cancel, |pcm: &[i16]| { /* play it */ })?;
```

- **Model folder.** The 0.6B release: `config.json`, `model.safetensors`,
  `vocab.json`, `merges.txt` and `speech_tokenizer/`. The speech tokenizer is
  byte for byte the 1.7B's, so it can be a hard link.
- **Voices.** `voice_from_audio` reads the clip: a plain WAV natively,
  anything else through FFmpeg (`tts.ffmpeg`). It makes the clip 24 kHz mono,
  trims silence at both ends, and cuts it to 30 seconds at a quiet moment. The
  speaker encoder and the codec encoder then make the voice. The voice is kept
  in `tts.voice_cache`, keyed by the clip's bytes and the transcript, so a clip
  is worked through once. The transcript must say exactly what the clip says:
  words it lacks are read out before the line. Transcribe the audio
  `audio::read_clip` returns. Without a transcript, the voice is refused.
  A voice made for one model size (1024 values for the 0.6B, 2048 for the 1.7B)
  is refused by the other.
- **Streaming.** The first chunk comes after one frame, then one every
  `options.chunk_frames` (2 frames, 160 ms). The model sometimes opens with up
  to a second of silence; that is dropped, down to a tenth of a second. The
  codec carries each stage's past from chunk to chunk (the transformer's keys
  and values over its 72-frame window, each causal convolution's last inputs).
  Its audio is therefore the whole-clip decode's (117 dB SNR against it), at
  the cost of the new frames only.
- **Speed.** Each frame's code prediction (15 steps of the predictor, with the
  codes drawn on the GPU), and each chunk's decode, are captured once as CUDA
  graphs and replayed. Candle's thread-local cache of kernel parameters makes a
  graph belong to its thread: `load` warms up on its thread, and a server
  that speaks on another calls `warm_up()` there (otherwise its first line
  takes about a third of a second longer).
- **Threads.** `Tts` is `Send`. It speaks one line at a time (`&mut self`).
  For concurrent calls, use one engine per call; each takes about 2 GB.

Measured on an RTX 5090 (`cuda:1`), with a two-sentence reply (4.8 s of audio)
in a voice cloned from a 6-second clip:

| | |
|---|---|
| Frames | 10 ms each (100 a second; real time is 12.5) |
| Codec | 4 ms a 2-frame chunk |
| First audio | 50-100 ms after the call (after any opening silence) |
| Real-time factor | 0.15-0.16 |
| Device memory | 1.7 GB of weights; about 2.0-2.2 GB in all |

`cargo run --release -p oaiy-tts --features flash-attn --example speak --
--voice clip.mp3 --transcript "..." --text "..." --out out.wav --device 1`
speaks a line and prints these numbers; `--example tts_bench` times frames and codec
chunks apart. `cargo test -p oaiy-tts` runs on the CPU with no weights.

## Speed and memory

On an RTX 5090:

- The talker writes about 80 frames a second, against 12.5 for real time (the
  official PyTorch implementation writes 8.2): the code predictor's steps
  replay as a CUDA graph, as in `oaiy-tts`.
- 8 seconds of speech takes about 3 seconds in all: 1.5 s to load, 1.3 s
  speaking, then 0.3 s to decode.
- Saving a voice takes 7-10 seconds.

The talker and codec need about 3.5 GB of VRAM (BF16 talker, F32 codec). The
text embedding table stays in the file: only the prompt's rows are read.

Breeze TTS 2 writes about 12 frames a second, just under real time: the
depth decoder's 15 steps a frame dominate. A 6-second line takes about 6 s
once the model is loaded. It needs about
6.5 GB of VRAM.

## Verification

Reference activations come from the official `qwen-tts` package (strict F32 for
the F32 parts, TF32 off). The opt-in tests are `codec` and `clone` in
`oaiy-tts`, and `tts::tests` in `oaiy-media`.

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

`oaiy-tts`'s own checks run on the CPU without weights. They use tiny random
models built like the real ones. A stream decodes exactly what the whole
sequence decodes, in any chunking, past the attention window, and after a
primed reference clip. Codes drawn on the device match those drawn on the host
(greedy). A decoder step at a time matches the whole sequence through its
growing KV cache. On the GPU, streamed audio measures 117 dB SNR against the
whole-clip decode, and cloned lines transcribe back word for word.

For Breeze TTS 2, the reference is the official `breeze-tts` code with
Transformers. Greedy decoding reproduces its prompt and its first frame, with
and without guidance. The first difference is a near-tie (0.03 logits apart)
in a later codebook, as with Qwen3-TTS. Its lines transcribe back word for word
with Whisper.
