//! Qwen3-TTS spoken as it is made, on WebGPU: what [`oaiy_tts::Tts`] is on Candle (whose GPU was CUDA's), for a
//! server that streams a line's audio while the rest is still being made. The talker ([`crate::tts::webgpu::WgpuTalker`])
//! hands each frame over as it draws it, and the speech codec's decoder ([`crate::tts::webgpu::codec::WgpuCodec`]) decodes
//! the frames a chunk at a time after the frames before them ([`WgpuCodec::decode_after`]: the official chunked
//! decode's 25 frames of left context, where Candle's stream carries each stage's state). A cloned voice's speaker
//! embedding and reference codes are made on the CPU (both encoders F32), once a clip, and kept.

use crate::tts::webgpu::codec::{StreamState, WgpuCodec};
use crate::tts::webgpu::{FramesEnd, WgpuTalker};
use candle_core::{Device, Result};
use oaiy_tts::codec::SAMPLE_RATE;
use oaiy_tts::talker::FRAMES_PER_SECOND;
use oaiy_tts::voice::Voice;
use oaiy_tts::{clone, Finish, LeadIn, SpeakOptions, SpeakReport};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("speech on WebGPU: {e}"))
}

/// The voice's clip's frames a line's codec stream starts after ([`WgpuCodec::stream_after`], once a voice). A line's
/// chunks are decoded by the codec's stream ([`WgpuCodec::stream`]), which keeps each stage's state: the whole
/// decode's samples (107 dB from it, its float's rounding), each chunk at the cost of its own frames (9 ms a frame on
/// an M5 Pro's GPU). Each chunk was decoded after the 120 frames before it at first (with only the transformer
/// reading them, the stages after it the chunk and the 14 frames they reach back: 105 ms a chunk of four), 48 dB from
/// the whole decode at every chunk's edge (`a_streams_chunks_against_the_whole_decode` measures that); the clip's
/// frames before these reach the line's sound no more than they reached those chunks'.
const STREAM_CONTEXT: usize = 120;
/// The most frames a chunk has: four (0.32 s of sound). A line's first chunks are one frame and two ([`Stream`]).
const CHUNK_FRAMES: usize = 4;
/// The voices whose codec stream after their clip is kept.
const VOICE_STREAMS: usize = 8;

/// A voice's codes' hash: the codec's stream after its clip depends on them alone.
fn voice_key(voice: &Voice) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    voice.ref_codes.hash(&mut h);
    h.finish()
}

/// The engine: a Qwen3-TTS Base model resident on one WebGPU adapter.
pub struct WgpuTts {
    talker: WgpuTalker,
    codec: WgpuCodec,
    /// The codec's stream after each voice's clip, by its codes ([`voice_key`]), the latest few.
    voice_streams: Vec<(u64, StreamState)>,
    tokenizer: tokenizers::Tokenizer,
    /// Loaded (on the CPU) when the first voice is made, then kept.
    speaker_encoder: Option<clone::SpeakerEncoder>,
    speech_encoder: Option<clone::SpeechEncoder>,
    model_dir: PathBuf,
    /// FFmpeg, for voice clips that are not plain WAV (MP3 and the rest).
    pub ffmpeg: PathBuf,
    /// Where voices made from clips are kept (`None`: not kept).
    pub voice_cache: Option<PathBuf>,
    pub options: SpeakOptions,
}

impl WgpuTts {
    /// Load a Qwen3-TTS Base folder (`config.json`, `model.safetensors`, `vocab.json`, `merges.txt`,
    /// `speech_tokenizer/`) onto WebGPU device `device`, and speak a few frames so the first line does not pay for
    /// the kernels' compiling.
    pub fn load(model_dir: &Path, device: usize) -> Result<Self> {
        let config = oaiy_engine::json::Json::parse(&std::fs::read(model_dir.join("config.json")).map_err(|e| err(format!("{}: {e}", model_dir.join("config.json").display())))?)
            .map_err(candle_core::Error::wrap)?;
        if config.get("tts_model_type").and_then(|t| t.as_str()).is_some_and(|t| t != "base") {
            candle_core::bail!("{} is not a Qwen3-TTS Base model (voices are cloned with the Base model)", model_dir.display());
        }
        let tokenizer = oaiy_tts::text::tokenizer(model_dir)?;
        let talker = WgpuTalker::load(model_dir, device)?;
        // the codec on a device of its own on the same adapter, so a queue of its own: its chunks run on the GPU beside
        // the talker's steps, not after them (OAIY_TTS_ONE_QUEUE: the talker's)
        let own;
        let codec_gpu = if std::env::var_os("OAIY_TTS_ONE_QUEUE").is_some() {
            talker.gpu()
        } else {
            own = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
            &own
        };
        let codec = WgpuCodec::load(&model_dir.join("speech_tokenizer").join("model.safetensors"), codec_gpu)?;
        let mut tts = Self {
            talker,
            codec,
            voice_streams: Vec::new(),
            tokenizer,
            speaker_encoder: None,
            speech_encoder: None,
            model_dir: model_dir.to_path_buf(),
            ffmpeg: PathBuf::from("ffmpeg"),
            voice_cache: Some(std::env::temp_dir().join("oaiy-tts-voices")),
            options: SpeakOptions { chunk_frames: CHUNK_FRAMES, ..SpeakOptions::default() },
        };
        tts.warm_up()?;
        Ok(tts)
    }

    /// Speak a few frames of nothing: the kernels a line runs are compiled at their first use.
    pub fn warm_up(&mut self) -> Result<()> {
        let voice = Voice {
            name: String::new(),
            description: String::new(),
            language: "auto".into(),
            ref_text: "Hello.".into(),
            ref_codes: vec![[0; 16]; 8],
            speaker: vec![0.; self.width()],
        };
        let saved = self.options.clone();
        // (chunks of one frame and of `chunk_frames`, as a line's are)
        self.options.max_seconds = (1 + 2 * saved.chunk_frames.max(1)) as f64 / FRAMES_PER_SECOND;
        self.options.trim_leading_silence = false;
        self.options.language = "auto".into();
        let r = self.speak("Hello there, this warms the engine up.", &voice, &AtomicBool::new(false), |_| {});
        self.options = saved;
        r.map(|_| ())
    }

    /// The adapter it runs on, for a log.
    pub fn adapter(&self) -> String {
        let a = self.talker.gpu().adapter();
        format!("{} ({})", a.name, a.backend)
    }

    /// The talker's width (1024 for the 0.6B, 2048 for the 1.7B): voices are made for one size.
    pub fn width(&self) -> usize {
        self.talker.hidden()
    }

    /// A voice from a clip of someone speaking (MP3, WAV, or anything FFmpeg reads; 3 to 30 seconds is best, longer
    /// clips are cut to 30) and what it says, as [`oaiy_tts::Tts::voice_from_audio`]: kept in `voice_cache`, so a
    /// clip is only worked through once.
    pub fn voice_from_audio(&mut self, path: &Path, transcript: Option<&str>) -> Result<Voice> {
        let transcript = transcript.map(str::trim).filter(|t| !t.is_empty()).ok_or_else(|| {
            err("a cloned voice needs the transcript of its clip (what it says, word for word): transcribe the clip first (the transcript must match audio::read_clip's cut)")
        })?;
        let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        let cached = self.voice_cache.as_ref().map(|dir| oaiy_tts::voice::cache_path(dir, &bytes, transcript, self.width()));
        if let Some(v) = cached.as_ref().and_then(|p| Voice::open(p).ok()) {
            if v.ref_text == transcript && v.speaker.len() == self.width() {
                return Ok(v);
            }
        }
        let clip = oaiy_tts::audio::read_clip(path, &self.ffmpeg).map_err(|e| err(e.to_string()))?;
        let mut voice = self.voice_from_samples(&clip, transcript)?;
        voice.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(p) = cached {
            // A cache that cannot be written only costs the next call time.
            let _ = voice.save(&p);
        }
        Ok(voice)
    }

    /// A voice from 24 kHz mono samples and their transcript: the speaker embedding and the clip's codes, by the
    /// Base model's two encoders on the CPU.
    pub fn voice_from_samples(&mut self, clip: &[f32], transcript: &str) -> Result<Voice> {
        if self.speaker_encoder.is_none() {
            self.speaker_encoder = Some(clone::SpeakerEncoder::load(&self.model_dir.join("model.safetensors"), &Device::Cpu)?);
        }
        if self.speech_encoder.is_none() {
            self.speech_encoder = Some(clone::SpeechEncoder::load(&self.model_dir.join("speech_tokenizer").join("model.safetensors"), &Device::Cpu)?);
        }
        let (Some(spk), Some(enc)) = (&self.speaker_encoder, &self.speech_encoder) else { unreachable!("both loaded above") };
        let speaker = spk.embed(clip)?;
        if speaker.len() != self.width() {
            candle_core::bail!("the speaker encoder gives {} values but the talker takes {}", speaker.len(), self.width());
        }
        let ref_codes = enc.encode(clip)?;
        Ok(Voice { name: String::new(), description: String::new(), language: "auto".into(), ref_text: transcript.trim().into(), ref_codes, speaker })
    }

    /// How like `voice` the line `pcm` (16-bit at 24 kHz) sounds: the cosine of its speaker embedding and the voice's
    /// own (the speaker encoder's, on the CPU: some 30 ms a line of 3.5 s on an M5 Pro). The talker's draws now and then
    /// make a line in another speaker's voice (2 lines in 40, at the model's own sampling, in the receptionist voice):
    /// such a line scores 0.97 or less, where the voice's own lines score 0.98 or more.
    pub fn likeness(&mut self, pcm: &[i16], voice: &Voice) -> Result<f64> {
        if self.speaker_encoder.is_none() {
            self.speaker_encoder = Some(clone::SpeakerEncoder::load(&self.model_dir.join("model.safetensors"), &Device::Cpu)?);
        }
        let spk = self.speaker_encoder.as_ref().expect("loaded above");
        let audio: Vec<f32> = pcm.iter().map(|&s| s as f32 / 32768.0).collect();
        let e = spk.embed(&audio)?;
        let dot: f64 = e.iter().zip(&voice.speaker).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        Ok(dot / (norm(&e) * norm(&voice.speaker)).max(1e-30))
    }

    /// Speak `text` in `voice`, handing out 16-bit mono PCM at 24 kHz as it is made (the first chunk after
    /// `options.first_chunk_frames`, then every `options.chunk_frames`), as [`oaiy_tts::Tts::speak`]. `cancel` is
    /// checked at every frame; once it is set, nothing more is handed out.
    pub fn speak(&mut self, text: &str, voice: &Voice, cancel: &AtomicBool, mut on_chunk: impl FnMut(&[i16])) -> Result<SpeakReport> {
        let started = Instant::now();
        let o = self.options.clone();
        let mut report = SpeakReport { finish: Finish::Stop, frames: 0, audio_seconds: 0., first_chunk_seconds: None, prefill_seconds: 0., total_seconds: 0. };
        let text = text.trim();
        if text.is_empty() {
            return Ok(report);
        }
        let language = match o.language.trim() {
            "" | "auto" if !voice.language.is_empty() => voice.language.clone(),
            l => l.to_owned(),
        };
        let language = self.talker.language_id(&language)?;
        let text_ids = oaiy_tts::text::encode(&self.tokenizer, text)?;
        let ref_ids = oaiy_tts::text::encode(&self.tokenizer, &voice.ref_text)?;
        let (prefill, trailing) = self.talker.prefill_clone(&text_ids, &ref_ids, voice, language)?;
        // A line longer than its text could need is babbling: cap it (two seconds, plus 0.6 s a token, far slower
        // than anyone speaks).
        let max_frames = ((o.max_seconds.min(2. + 0.6 * text_ids.len() as f64)) * FRAMES_PER_SECOND).ceil() as usize;
        report.prefill_seconds = started.elapsed().as_secs_f64();
        // the codec's stream after the voice's clip (made once a voice, then copied for each line)
        let key = voice_key(voice);
        if !self.voice_streams.iter().any(|(k, _)| *k == key) {
            let made = self.codec.stream_after(&voice.ref_codes[voice.ref_codes.len().saturating_sub(STREAM_CONTEXT)..])?;
            if self.voice_streams.len() >= VOICE_STREAMS {
                self.voice_streams.remove(0);
            }
            self.voice_streams.push((key, made));
        }
        let Self { talker, codec, voice_streams, .. } = self;
        let state = codec.stream_copy(&voice_streams.iter().find(|(k, _)| *k == key).expect("made above").1);
        let mut emit = |samples: Vec<f32>, report: &mut SpeakReport| {
            if samples.is_empty() || cancel.load(Ordering::Relaxed) {
                return;
            }
            if report.first_chunk_seconds.is_none() {
                report.first_chunk_seconds = Some(started.elapsed().as_secs_f64());
            }
            report.audio_seconds += samples.len() as f64 / SAMPLE_RATE as f64;
            on_chunk(&oaiy_tts::audio::to_pcm16(&samples));
        };
        let mut stream = Stream::new(codec, state, &o);
        let decoding = if std::env::var_os("OAIY_TTS_SERIAL").is_some() {
            // (the talker's thread decodes each chunk between its frames, as before)
            let end = talker.frames_with(&prefill, Some(trailing), o.sampling.clone(), max_frames, |frame| {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(false);
                }
                report.frames += 1;
                let samples = stream.frame(*frame)?;
                emit(samples, &mut report);
                Ok(true)
            })?;
            report.finish = finish_of(end);
            if report.finish != Finish::Cancelled {
                let rest = stream.finish()?;
                emit(rest, &mut report);
            }
            stream.decoding
        } else {
            // The chunks decoded on a thread of their own (and the codec on a queue of its own: the GPU runs them
            // between the talker's small steps, which leave it idle most of the time), while the talker draws the next
            // frames; what is decoded is handed out at each frame, and the rest once the line's frames are all drawn.
            let finishing = AtomicBool::new(false);
            let (end, decoded) = std::thread::scope(|scope| -> Result<(FramesEnd, f64)> {
                let (frames_tx, frames_rx) = mpsc::channel::<[u32; 16]>();
                let (audio_tx, audio_rx) = mpsc::channel::<Result<Vec<f32>>>();
                let finishing = &finishing;
                let decoder = scope.spawn(move || {
                    for frame in frames_rx {
                        match stream.frame(frame) {
                            Ok(samples) if samples.is_empty() => {}
                            Ok(samples) => {
                                if audio_tx.send(Ok(samples)).is_err() {
                                    return stream.decoding;
                                }
                            }
                            Err(e) => {
                                let _ = audio_tx.send(Err(e));
                                return stream.decoding;
                            }
                        }
                    }
                    if finishing.load(Ordering::Relaxed) {
                        let _ = audio_tx.send(stream.finish());
                    }
                    stream.decoding
                });
                let drawn = talker.frames_with(&prefill, Some(trailing), o.sampling.clone(), max_frames, |frame| {
                    if cancel.load(Ordering::Relaxed) {
                        return Ok(false);
                    }
                    report.frames += 1;
                    let _ = frames_tx.send(*frame);
                    while let Ok(samples) = audio_rx.try_recv() {
                        emit(samples?, &mut report);
                    }
                    Ok(true)
                });
                finishing.store(matches!(drawn, Ok(FramesEnd::Ended | FramesEnd::Full)), Ordering::Relaxed);
                drop(frames_tx);
                if finishing.load(Ordering::Relaxed) {
                    for samples in &audio_rx {
                        emit(samples?, &mut report);
                    }
                }
                drop(audio_rx);
                let decoded = decoder.join().map_err(|_| err("the decoder's thread panicked"))?;
                Ok((drawn?, decoded))
            })?;
            report.finish = finish_of(end);
            decoded
        };
        report.total_seconds = started.elapsed().as_secs_f64();
        if std::env::var_os("OAIY_TTS_PROFILE").is_some() {
            eprintln!("tts profile: decodes {:.1} ms a frame ({decoding:.2} s), prefill {:.0} ms, all {:.2} s", decoding * 1e3 / report.frames.max(1) as f64, report.prefill_seconds * 1e3, report.total_seconds);
            if ggml_rs_wgpu::profile::chain_on() {
                for (name, ms, n) in ggml_rs_wgpu::profile::take_kernels().into_iter().take(14) {
                    eprintln!("  {name}: {ms:.1} ms in {n} dispatches ({:.1} us each)", ms * 1e3 / n.max(1) as f64);
                }
            }
            if ggml_rs_wgpu::profile::pieces_on() {
                let (busy, span, pieces) = ggml_rs_wgpu::profile::take_pieces();
                eprintln!("  the GPU busy {busy:.0} ms of {span:.0} ms in {pieces} pieces");
            }
        }
        Ok(report)
    }
}

fn finish_of(end: FramesEnd) -> Finish {
    match end {
        FramesEnd::Stopped => Finish::Cancelled,
        FramesEnd::Full => Finish::Length,
        FramesEnd::Ended => Finish::Stop,
    }
}

/// A line's frames to its sound through the codec's stream (each chunk decoded from its own frames, after the voice's
/// clip and the line's frames before it: the whole decode's samples) and the lead-in gate: what [`WgpuTts::speak`]
/// hands out (on a thread of its own, as the talker draws the next frames). Until the gate opens each frame is decoded
/// as it comes, so the first sound is out as soon as there is one; then chunks of two frames, then of `chunk_frames`,
/// each made in less time than the one before it takes to play, so a line's start has no gap.
struct Stream<'a> {
    codec: &'a WgpuCodec,
    state: StreamState,
    pending: Vec<[u32; 16]>,
    gate: LeadIn,
    /// The frames the last chunk handed out had (0: nothing out yet).
    last: usize,
    chunk_frames: usize,
    /// Seconds spent decoding.
    decoding: f64,
}

impl<'a> Stream<'a> {
    fn new(codec: &'a WgpuCodec, state: StreamState, o: &SpeakOptions) -> Self {
        Self { codec, state, pending: Vec::new(), gate: LeadIn::new(o.trim_leading_silence), last: 0, chunk_frames: o.chunk_frames.max(1), decoding: 0. }
    }

    /// The pending frames decoded, through the gate.
    fn decode(&mut self) -> Result<Vec<f32>> {
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        let at = Instant::now();
        let frames = std::mem::take(&mut self.pending);
        let samples = self.codec.stream(&mut self.state, &frames)?;
        self.decoding += at.elapsed().as_secs_f64();
        let out = self.gate.pass(samples);
        if !out.is_empty() {
            self.last = frames.len();
        }
        Ok(out)
    }

    /// The line's next frame: the sound to hand out now (none until a chunk is due).
    fn frame(&mut self, frame: [u32; 16]) -> Result<Vec<f32>> {
        self.pending.push(frame);
        // (nothing out yet: a frame at a time; then each chunk twice the last, up to `chunk_frames`)
        let want = if self.last == 0 { 1 } else { (2 * self.last).min(self.chunk_frames) };
        if self.pending.len() < want {
            return Ok(Vec::new());
        }
        self.decode()
    }

    /// The rest of the line once its frames are all drawn.
    fn finish(&mut self) -> Result<Vec<f32>> {
        let mut out = self.decode()?;
        if !self.gate.open {
            // Quiet to the end: hand out what the gate held.
            out = std::mem::take(&mut self.gate.held);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oaiy_tts::codec::LEFT_CONTEXT;

    /// A server loads the engine on one thread and speaks on another.
    #[test]
    fn the_engine_can_move_between_threads() {
        fn send<T: Send>() {}
        send::<WgpuTts>();
    }

    /// A stream's chunks against the whole decode of the same frames (`--ignored --nocapture`; `OAIY_TTS` the model's
    /// folder, `OAIY_TTS_VOICE` a saved voice, whose clip's codes are the frames): the signal-to-noise ratio of each
    /// chunk size and context, where the whole decode has every frame before.
    #[test]
    #[ignore = "needs a Qwen3-TTS Base model, a saved voice and a WebGPU adapter"]
    fn a_streams_chunks_against_the_whole_decode() -> Result<()> {
        let dir = PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let device = std::env::var("OAIY_TTS_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let voice = Voice::open(Path::new(&std::env::var("OAIY_TTS_VOICE").map_err(err)?)).map_err(err)?;
        // (the clip three times over: a line longer than any context tried, so none is the whole of it)
        let frames = &voice.ref_codes.repeat(3);
        assert!(frames.len() > 2 * STREAM_CONTEXT, "the voice's clip is too short to tell");
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let codec = WgpuCodec::load(&dir.join("speech_tokenizer").join("model.safetensors"), &gpu)?;
        let whole = codec.decode(frames)?;
        let snr = |got: &[f32], from: usize| {
            let (mut signal, mut noise) = (0f64, 0f64);
            for (a, b) in whole[from..].iter().zip(&got[from..]) {
                signal += (*a as f64).powi(2);
                noise += (*a as f64 - *b as f64).powi(2);
            }
            10. * (signal / noise.max(1e-30)).log10()
        };
        eprintln!("{} frames ({:.1} s)", frames.len(), frames.len() as f64 / FRAMES_PER_SECOND);
        let after = 80 * oaiy_tts::codec::SAMPLES_PER_FRAME;
        let mut ours = 0.;
        for (chunk, context) in [(2usize, LEFT_CONTEXT), (4, LEFT_CONTEXT), (4, 80), (4, STREAM_CONTEXT), (4, 160)] {
            let mut stream = Vec::with_capacity(whole.len());
            let started = Instant::now();
            for at in (0..frames.len()).step_by(chunk) {
                let end = (at + chunk).min(frames.len());
                stream.extend(codec.decode_after_with(&frames[..at], &frames[at..end], context)?);
            }
            let seconds = started.elapsed().as_secs_f64();
            assert_eq!(stream.len(), whole.len());
            eprintln!(
                "chunks of {chunk} after {context} frames: SNR {:.1} dB over the line, {:.1} dB past its first 80 frames; {:.1} ms a chunk ({:.2} of real time)",
                snr(&stream, 0),
                snr(&stream, after),
                seconds * 1e3 / frames.len().div_ceil(chunk) as f64,
                seconds / (frames.len() as f64 / FRAMES_PER_SECOND)
            );
            if (chunk, context) == (4, STREAM_CONTEXT) {
                ours = snr(&stream, after);
            }
        }
        assert!(ours > 40., "the stream's chunks are {ours:.1} dB from the whole decode");
        Ok(())
    }

    /// The Base model speaking two lines in a voice on WebGPU, streamed (`--ignored --nocapture`; `OAIY_TTS` the
    /// model's folder, `OAIY_TTS_VOICE` a saved voice's `.voice.json`): that it speaks, and how fast.
    /// A frame's codes drawn on the device (`Draws::Auto`: one submission a frame) are the host's (`Draws::Host`: a
    /// round trip a code) where both draw the largest: a line's first 40 frames, every one of their 16 codes. (Its
    /// sound would not tell: a greedy line in this voice is silence.)
    #[test]
    #[ignore = "needs a Qwen3-TTS Base model, a saved voice and a WebGPU adapter"]
    fn codes_drawn_on_the_device_are_the_hosts() -> Result<()> {
        let dir = PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let device = std::env::var("OAIY_TTS_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut tts = WgpuTts::load(&dir, device)?;
        let voice = Voice::open(Path::new(&std::env::var("OAIY_TTS_VOICE").map_err(err)?)).map_err(err)?;
        let text_ids = oaiy_tts::text::encode(&tts.tokenizer, "Thanks for calling, how can I help? We are open until five today.")?;
        let ref_ids = oaiy_tts::text::encode(&tts.tokenizer, &voice.ref_text)?;
        let language = tts.talker.language_id(&voice.language)?;
        let mut drawn = Vec::new();
        for draws in [oaiy_tts::talker::Draws::Host, oaiy_tts::talker::Draws::Auto] {
            let (prefill, trailing) = tts.talker.prefill_clone(&text_ids, &ref_ids, &voice, language)?;
            let sampling = oaiy_tts::talker::Sampling { greedy: true, draws, ..oaiy_tts::talker::Sampling::default() };
            let frames = tts.talker.frames(&prefill, Some(trailing), sampling, 40, |_| {})?;
            eprintln!("{draws:?}: {} frames, the first {:?}", frames.len(), frames.first());
            drawn.push(frames);
        }
        assert!(drawn[0].len() > 10, "{} frames", drawn[0].len());
        assert_eq!(drawn[0], drawn[1], "the device's codes are not the host's");
        // (OAIY_TTS_SAVE: the codes to a file, to set beside another build's)
        if let Ok(path) = std::env::var("OAIY_TTS_SAVE") {
            std::fs::write(path, format!("{:?}", drawn[0]))?;
        }
        Ok(())
    }

    /// A line spoken with seeds 1 to OAIY_TTS_SEEDS (12), its codes drawn on the host and on the device, each to a
    /// file in OAIY_TTS_OUT (`host_N.pcm`, `device_N.pcm`): for the speaker encoder to tell how like the voice each
    /// set sounds (`oaiy_tts::clone`'s `how_like_the_clip_each_line_sounds`).
    #[test]
    #[ignore = "needs a Qwen3-TTS Base model, a saved voice and a WebGPU adapter"]
    fn a_line_spoken_with_seeds_drawn_here_and_there() -> Result<()> {
        let dir = PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let out = PathBuf::from(std::env::var("OAIY_TTS_OUT").map_err(err)?);
        let seeds: u64 = std::env::var("OAIY_TTS_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        let text = std::env::var("OAIY_TTS_LINE").unwrap_or_else(|_| "Hello Jeff, thanks for calling. How can I help you today?".into());
        let mut tts = WgpuTts::load(&dir, 0)?;
        let voice = Voice::open(Path::new(&std::env::var("OAIY_TTS_VOICE").map_err(err)?)).map_err(err)?;
        // (OAIY_TTS_SUB_TEMPERATURE: the code predictor's; OAIY_TTS_DEVICE_ONLY: the device's draws alone)
        if let Some(t) = std::env::var("OAIY_TTS_SUB_TEMPERATURE").ok().and_then(|v| v.parse().ok()) {
            tts.options.sampling.sub_temperature = t;
        }
        if let Some(t) = std::env::var("OAIY_TTS_TEMPERATURE").ok().and_then(|v| v.parse().ok()) {
            tts.options.sampling.temperature = t;
        }
        let only = std::env::var_os("OAIY_TTS_DEVICE_ONLY").is_some();
        // (OAIY_TTS_FIRST_SEED: seeds from it, 1 unless said; the voice server's lines are seed 0's)
        let first: u64 = std::env::var("OAIY_TTS_FIRST_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        for seed in first..first + seeds {
            for (name, draws) in [("host", oaiy_tts::talker::Draws::Host), ("device", oaiy_tts::talker::Draws::Auto)].into_iter().filter(|(n, _)| !only || *n == "device") {
                tts.options.sampling.seed = seed;
                tts.options.sampling.draws = draws;
                let mut pcm: Vec<i16> = Vec::new();
                tts.speak(&text, &voice, &AtomicBool::new(false), |c| pcm.extend_from_slice(c))?;
                std::fs::write(out.join(format!("{name}_{seed:02}.pcm")), pcm.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<u8>>())?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs a Qwen3-TTS Base model, a saved voice and a WebGPU adapter"]
    fn a_line_is_spoken_in_chunks_as_fast_as_it_is_heard() -> Result<()> {
        let dir = PathBuf::from(std::env::var("OAIY_TTS").unwrap_or_else(|_| "E:/models/Qwen3-TTS-12Hz-0.6B-Base".into()));
        let device = std::env::var("OAIY_TTS_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let loading = Instant::now();
        let mut tts = WgpuTts::load(&dir, device)?;
        eprintln!("loaded on {} in {:.1} s", tts.adapter(), loading.elapsed().as_secs_f64());
        let voice = match std::env::var("OAIY_TTS_VOICE") {
            Ok(path) => Voice::open(Path::new(&path)).map_err(err)?,
            Err(_) => Voice { name: String::new(), description: String::new(), language: "english".into(), ref_text: "Hello.".into(), ref_codes: vec![[0; 16]; 8], speaker: vec![0.; tts.width()] },
        };
        tts.options.sampling.seed = 7;
        for line in ["Hello! This is a quick test of speech running on WebGPU.", "The second line starts sooner, because the kernels are already there, and it runs a little longer so the rate settles."] {
            let mut pcm: Vec<i16> = Vec::new();
            let mut chunks = 0;
            let r = tts.speak(line, &voice, &AtomicBool::new(false), |c| {
                chunks += 1;
                pcm.extend_from_slice(c);
            })?;
            eprintln!(
                "{} frames in {chunks} chunks: {:.2} s of audio in {:.2} s (real-time factor {:.2}), first audio after {:.0} ms, prefill {:.0} ms, {:?}",
                r.frames,
                r.audio_seconds,
                r.total_seconds,
                r.real_time_factor(),
                r.first_chunk_seconds.unwrap_or(0.) * 1e3,
                r.prefill_seconds * 1e3,
                r.finish
            );
            assert!(r.frames > 0 && !pcm.is_empty(), "nothing was spoken");
            assert!(pcm.iter().any(|&s| s.unsigned_abs() > 300), "the line is silent");
        }
        Ok(())
    }
}
