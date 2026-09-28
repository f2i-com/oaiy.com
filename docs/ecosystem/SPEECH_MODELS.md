# Speech models for realtime voice

Research for OAIY's realtime voice (September 2026): the speech-to-text and text-to-speech
models to run natively, what Aokie used, what the engines can reuse, and the OpenAI Realtime
API a local server must implement.

## Speech-to-text

### nvidia/parakeet-tdt-0.6b-v2

<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>, CC-BY-4.0, English, ~618M parameters
(FP32, a 2.47 GB checkpoint). WER 6.05; offline and full-context (up to 24 minutes per pass),
**not streaming**: run in chunks, its WER rises steeply.

- **Encoder (FastConformer):** 24 layers, d_model 1024, 8 heads, feed-forward ×4, a
  convolution module (kernel 9, BatchNorm), relative-position attention (`pos_bias_u/v`) over
  the full context, no xscaling, no biases. `dw_striding` ×8 subsampling, 256 channels (a
  3×3 stride-2 conv, then two depthwise 3×3 stride-2 + pointwise 1×1 pairs, then Linear
  4096→1024): an encoder frame is 80 ms.
- **Decoder (TDT):** a 2-layer LSTM (640 units; embedding 1025×640, blank = 1024); the joint
  is Linear 1024→640 (encoder) + Linear 640→640 (prediction), ReLU, Linear 640→**1030**
  (1024 tokens, blank, and 5 durations `[0,1,2,3,4]`); up to 10 tokens per frame.
- **Tokenizer:** SentencePiece BPE, 1024 pieces.
- **Mel front end** (NeMo `FilterbankFeatures`, matching transformers'
  `feature_extraction_parakeet.py`): 16 kHz, 128 mels, n_fft 512, window 400, hop 160,
  pre-emphasis 0.97, **non-periodic Hann**, `center=True` with **zero padding**, power
  spectrum, Slaney mels 0–8 kHz, `log(x + 2^-24)`, per-bin normalisation over the valid frames
  (**unbiased** std, + 1e-5). No dither at inference.
- **Greedy TDT** (NeMo `GreedyTDTInfer`): from t = 0 with blank into the prediction network,
  run the joint on (enc[t], pred); argmax the token logits and, separately, the 5 duration
  logits; a non-blank token is emitted and steps the LSTM; t advances by the predicted
  duration (after non-blank tokens too, unlike RNN-T); duration 0 stays on the frame (up to 10
  tokens); stop at t ≥ T. On blank with duration 0, implementations differ (force 1 at once,
  or after the 10-token cap); both give the same output, which a test should confirm.
- **`.nemo`** is an uncompressed tar: `model_config.yaml`, `model_weights.ckpt` (a zip-format
  `torch.save`; keys `encoder.pre_encode.*`, `encoder.layers.N.self_attn.linear_{q,k,v,out,
  pos}`, `decoder.prediction.dec_rnn.lstm.*`, `joint.*`), and hash-prefixed tokenizer files.
  The checkpoint also holds `preprocessor.featurizer.window` and `.fb`, to check a filterbank
  against.
- **Exports:** `istupakov/parakeet-tdt-0.6b-v2-onnx` (with `nemo128.onnx`, the mel front end:
  a reference to test against); sherpa-onnx's int8 build.

### moondream/parakeet-ultra

<https://huggingface.co/moondream/parakeet-ultra> (2026-09-22), CC-BY-4.0: a further-trained
**parakeet-tdt-0.6b-v3** (25 European languages, 8192-piece vocabulary, blank = 8192), the
same encoder and TDT head, plus a small `vad_head` (three Conv1d on the subsampler output)
that cuts long audio into pieces of 30 s or less (627.27M parameters).

- Files: `model.safetensors` (1.26 GB; matrices F16, norms/biases/BatchNorm F32),
  `tokenizer.json` (8192 BPE), `config.json`. Weight names follow transformers'
  `ParakeetForTDT`, so it loads directly. No processor config: take v3's mel settings.
- English WER 6.26 → 5.80 against v3; FLEURS 11.62 → 9.55; noisy 6.72 → 5.82.
- No streaming in the model (its SDK re-transcribes: a preview after 4 s, then every 2 s). Its
  own engine is closed, so port from transformers `models/parakeet/` (`modeling_parakeet.py`,
  `feature_extraction_parakeet.py`, `generation_parakeet.py`, `convert_nemo_to_hf.py`).
- v3 itself: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>, a transformers
  `model.safetensors` (2.51 GB) with `processor_config.json` and `tokenizer.json`.

## Text-to-speech

### Kyutai Pocket TTS

<https://huggingface.co/kyutai/pocket-tts> (gated; weights CC-BY-4.0, code MIT),
<https://huggingface.co/kyutai/pocket-tts-without-voice-cloning> (not gated),
<https://github.com/kyutai-labs/pocket-tts>, arXiv 2509.06926.

- **Architecture:** a "FlowLM" transformer (6 layers, d_model 1024, 16 heads) gives, every
  80 ms frame, a conditioning vector and an end-of-speech logit; a small MLP flow head (depth
  6, width 512) samples one continuous 32-dim latent in one step (LSD); a **Mimi-VAE** decoder
  (SEANet, ratios 6/5/4, plus a 2-layer transformer; **no RVQ**; 12.5 Hz, 1920 samples a
  frame) makes 24 kHz audio. ~100M parameters; text by a 4k SentencePiece model.
- **Voices:** 21 English presets, plus Italian, Spanish, German, Portuguese, French. A voice is
  the FlowLM KV cache after a Mimi-encoded reference prompt (`speaker_proj_weight`), saved as
  a 3.7–7.9 MB `.safetensors`. Cloning needs the gated weights. The voice samples
  (`kyutai/tts-voices`) have their own licences: settle them before shipping cloning.
- **Streaming:** audio frame by frame (`generate_audio_stream`); text in whole sentences
  (chunks up to 50 tokens). ~200 ms to the first audio and ~6× real time on 2 cores of an M4.
- **Files:** `languages/english/model.safetensors` (219 MB), `tokenizer.model`, and a
  24-layer higher-quality English model (1.3 GB).
- **Reference code:** `pocket_tts/models/{flow_lm,mimi,tts_model}.py`,
  `modules/{seanet,text_conditioner}.py`, `models/text_chunking.py`. Rust ports:
  <https://github.com/babybirdprd/pocket-tts> (Candle, MIT) and
  <https://github.com/gradium-ai/xn-ptts>.

### OpenMOSS-Team/MOSS-TTS-Realtime

<https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime>, Apache-2.0,
<https://github.com/OpenMOSS/MOSS-TTS> (`moss_tts_realtime/`).

- A Qwen3-1.7B backbone (28 layers, width 2048, 16/8 heads) plus a 4-layer depth transformer
  (~200M) writing 16 RVQ codes a frame. Codec: MOSS-Audio-Tokenizer ("Cat"), causal,
  ~1.6B parameters, 24 kHz, 12.5 Hz, 32×1024 RVQ (16 used); a separate repo, 7.1 GB.
- Streams text in as an LLM writes it (`push_text()/end_text()/drain()`, from 12 text tokens);
  audio in chunks (3 frames, 240 ms); ~180 ms to the first audio on an L20; 20 languages;
  zero-shot cloning; English WER 1.97%. `model.safetensors` bf16 4.66 GB.
- The engines' `sound/dac.rs` (MOSS-SoundEffect v2.0's DAC) is not this codec.

## What Aokie did

Everything in-process in Rust on ONNX Runtime, CPU, int8.

- Speech-to-text (`aokie.com/crates/aokie-ai/src/runtimes/parakeet_onnx/`):
  `eschmidbauer/parakeet-unified-en-0.6b-onnx`, which is **nvidia/parakeet-unified-en-0.6b**
  (RNN-T, not TDT, NVIDIA Open Model License), run once per utterance after Silero VAD and a
  200 ms pre-roll, ~0.1 RTF. Its mel code has no pre-emphasis, a periodic Hann and reflect
  padding (all unlike NeMo), and the vendored `transcribe-rs` Parakeet drops the duration
  logits: do not copy either.
- Text-to-speech (`runtimes/onnx_tts/`): `KevinAHM/pocket-tts-onnx` (five int8 graphs,
  ~150 MB), `LSD_STEPS=1`, `TEMPERATURE=0.7`, `EOS_THRESHOLD=-4.0`; a 15-frame decode chunk
  (0.7–1.0 s before the first audio); cloning through `mimi_encoder`; a pure-Rust SentencePiece
  unigram Viterbi (`tokenizer.rs`, ~200 lines).
- `aokie-voice-server`: `/health`, `/v1/models`, `/v1/audio/transcriptions`,
  `/v1/audio/speech` (pcm streamed with `X-Sample-Rate`) on 17920 (17921/17922 split).
  Neither Aokie nor the previous OAIY implements the Realtime API.

## What the engines can reuse

| Piece | Where | For |
|---|---|---|
| A reader for PyTorch checkpoints that runs no code | `crates/oaiy-media/src/sound/pth.rs` | `.nemo` weights (needs `LongStorage`, and the shared-storage LSTM tensors checked) |
| torchaudio-exact resampling, FFT, Slaney filters | `crates/oaiy-media/src/ltx/audio.rs` | 24 kHz in → 16 kHz |
| librosa mel filters, STFT as a GPU conv, a Mimi encoder | `crates/oaiy-media/src/tts/clone.rs` | the mel front end; Pocket TTS cloning |
| Qwen3 decoder with KV cache; a depth decoder | `tts/model.rs`, `tts/breeze.rs` | MOSS later |
| SentencePiece | `crates/tokenizer/src/spm.rs` | greedy longest-match: fine for Parakeet, not Pocket TTS (needs Viterbi) |
| OpenAI voices, 24 kHz pcm, base64, HF tokens | `crates/oaiy-studio/src/{speech,util,downloads}.rs` | as they are |
| A thread-per-connection HTTP server | `crates/oaiy-engine/src/http.rs`, `oaiy-studio` | a WebSocket upgrade (SHA-1 to write; std-only) |

candle-transformers 0.11 (already pinned) has streaming Mimi
(`models/mimi/{conv,seanet,transformer}.rs`, `StreamingModule`), Whisper's FFT and log-mel,
and candle-nn has an LSTM; it has no Parakeet or Conformer. Other Rust to learn from:
<https://github.com/gpu-cli/parakeet-rs> (Candle, TDT v2/v3), <https://github.com/altunenes/parakeet-rs>
(ONNX, streaming variants), <https://github.com/mudler/parakeet.cpp>, sherpa-onnx's
`DecodeOneTDT`, Kyutai's `moshi` Rust (`mimi`, `tts_streaming`, `asr`), and
<https://github.com/kyutai-labs/unmute> (STT → LLM → TTS behind an OpenAI-Realtime-style
protocol).

What stands in the way: media jobs start a fresh worker per job (~1.5 s to load); Qwen3-TTS
writes 14 frames a second against 12.5 for real time and does not stream; and a media job on
the LLM's GPU stops the LLM, while a call needs speech-to-text, the LLM and text-to-speech
loaded together.

## The OpenAI Realtime API, as a local server implements it

The beta was shut down on 12 May 2026: implement the GA names (and accept the beta aliases,
such as `response.audio.delta` and `input_audio_format: "pcm16"`, for old clients). Docs:
<https://developers.openai.com/api/docs/guides/realtime-websocket>, `/realtime-vad`,
`/realtime-conversations`; exact types in `openai-python/src/openai/types/realtime`.

- Connect: `wss://host/v1/realtime?model=…` with `Authorization: Bearer`.
- `session.update`: `{type:"realtime", output_modalities:["audio"], instructions, tools,
  audio:{input:{format:{type:"audio/pcm", rate:24000}, transcription:{model, language},
  turn_detection}, output:{format, voice, speed}}}`; pcm16 24 kHz mono little-endian, base64
  (`audio/pcmu`, `audio/pcma` too).
- Turn detection: `server_vad` (`threshold` 0.5, `prefix_padding_ms` 300,
  `silence_duration_ms` 500, `create_response`, `interrupt_response`, `idle_timeout_ms`),
  `semantic_vad` (`eagerness`), or `null` (the client commits).
- Client events: `input_audio_buffer.append|commit|clear`,
  `conversation.item.create|truncate|delete|retrieve`, `response.create|cancel`.
- Server events: `session.created|updated`; `input_audio_buffer.speech_started|
  speech_stopped|committed`; `conversation.item.added|done`;
  `conversation.item.input_audio_transcription.delta|completed`; `response.created`,
  `response.output_item.added`, `response.content_part.added`,
  `response.output_audio.delta`, `response.output_audio_transcript.delta`, their `.done`
  events, `response.output_item.done`, `response.done`; `error`, `rate_limits.updated`.
- Barge-in: `speech_started` during a response cancels it (`response.done` with
  `status:"cancelled"`); the client sends `conversation.item.truncate{item_id,
  content_index:0, audio_end_ms}` → `conversation.item.truncated`.

## Recommendation

1. A voice worker that stays loaded (speech-to-text and text-to-speech in VRAM, the LLM
   allowed beside it) and a framed streaming protocol to the host.
2. `/v1/realtime` over WebSocket in the gateway: the handshake and frames, the session
   state machine, 24 → 16 kHz in, an energy VAD to start.
3. Parakeet (mel, subsampling, encoder, joint, TDT loop), tested against `nemo128.onnx`, the
   ONNX export and transformers; also `/v1/audio/transcriptions`.
4. The LLM's stream into `response.output_audio_transcript.delta`.
5. Pocket TTS (FlowLM with KV cache, the flow head, a streaming Mimi-VAE decoder, a Viterbi
   SentencePiece) into `response.output_audio.delta`; streaming pcm on `/v1/audio/speech`.
6. Barge-in and truncation.
7. Later: Silero VAD or ultra's VAD head, streaming speech-to-text, MOSS-TTS-Realtime.

Risks: the mel front end must match NeMo exactly; relative-position attention's `rel_shift`;
TDT's edge cases; `.nemo` `LongStorage`; the streaming Mimi decoder's state across chunks;
Pocket TTS's gating and voice licences; sharing the GPU with the LLM; a hand-written
WebSocket (std-only); callers without echo cancellation setting off the VAD with the agent's
own voice (Aokie handles this in `aokie-plugin/src/aec.rs`).
