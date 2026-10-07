//! Qwen3-TTS spoken as it is made, on WebGPU: what [`oaiy_tts::Tts`] is on Candle (whose GPU was CUDA's), for a
//! server that streams a line's audio while the rest is still being made. The talker ([`crate::tts_wgpu::WgpuTalker`])
//! hands each frame over as it draws it, and the speech codec's decoder ([`crate::codec_wgpu::WgpuCodec`]) decodes
//! the frames a chunk at a time after the frames before them ([`WgpuCodec::decode_after`]: the official chunked
//! decode's 25 frames of left context, where Candle's stream carries each stage's state). A cloned voice's speaker
//! embedding and reference codes are made on the CPU (both encoders F32), once a clip, and kept.

use crate::codec_wgpu::WgpuCodec;
use crate::tts_wgpu::{FramesEnd, WgpuTalker};
use candle_core::{Device, Result};
use oaiy_tts::codec::{LEFT_CONTEXT, SAMPLE_RATE};
use oaiy_tts::talker::FRAMES_PER_SECOND;
use oaiy_tts::voice::Voice;
use oaiy_tts::{clone, Finish, LeadIn, SpeakOptions, SpeakReport};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("speech on WebGPU: {e}"))
}

/// The engine: a Qwen3-TTS Base model resident on one WebGPU adapter.
pub struct WgpuTts {
    talker: WgpuTalker,
    codec: WgpuCodec,
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
        let codec = WgpuCodec::load(&model_dir.join("speech_tokenizer").join("model.safetensors"), talker.gpu())?;
        let mut tts = Self {
            talker,
            codec,
            tokenizer,
            speaker_encoder: None,
            speech_encoder: None,
            model_dir: model_dir.to_path_buf(),
            ffmpeg: PathBuf::from("ffmpeg"),
            voice_cache: Some(std::env::temp_dir().join("oaiy-tts-voices")),
            options: SpeakOptions::default(),
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
        let Self { talker, codec, .. } = self;
        // the frames a chunk is decoded after: the voice's clip's last ones, then the line's own
        let mut before: Vec<[u32; 16]> = voice.ref_codes[voice.ref_codes.len().saturating_sub(LEFT_CONTEXT)..].to_vec();
        let mut gate = LeadIn::new(o.trim_leading_silence);
        let mut pending: Vec<[u32; 16]> = Vec::new();
        let mut emit = |samples: Vec<f32>, report: &mut SpeakReport| {
            if samples.is_empty() {
                return;
            }
            if report.first_chunk_seconds.is_none() {
                report.first_chunk_seconds = Some(started.elapsed().as_secs_f64());
            }
            report.audio_seconds += samples.len() as f64 / SAMPLE_RATE as f64;
            on_chunk(&oaiy_tts::audio::to_pcm16(&samples));
        };
        let decode = |before: &mut Vec<[u32; 16]>, pending: &mut Vec<[u32; 16]>| -> Result<Vec<f32>> {
            if pending.is_empty() {
                return Ok(Vec::new());
            }
            let samples = codec.decode_after(before, pending)?;
            before.append(pending);
            let extra = before.len().saturating_sub(LEFT_CONTEXT);
            before.drain(..extra);
            Ok(samples)
        };
        let end = talker.frames_with(&prefill, Some(trailing), o.sampling.clone(), max_frames, |frame| {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            report.frames += 1;
            pending.push(*frame);
            let want = if report.first_chunk_seconds.is_none() { o.first_chunk_frames } else { o.chunk_frames };
            if pending.len() >= want.max(1) {
                let samples = decode(&mut before, &mut pending)?;
                emit(gate.pass(samples), &mut report);
            }
            Ok(true)
        })?;
        report.finish = match end {
            FramesEnd::Stopped => Finish::Cancelled,
            FramesEnd::Full => Finish::Length,
            FramesEnd::Ended => Finish::Stop,
        };
        if report.finish != Finish::Cancelled {
            let rest = decode(&mut before, &mut pending)?;
            let mut out = gate.pass(rest);
            if !gate.open {
                // Quiet to the end: hand out what the gate held.
                out = std::mem::take(&mut gate.held);
            }
            emit(out, &mut report);
        }
        report.total_seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server loads the engine on one thread and speaks on another.
    #[test]
    fn the_engine_can_move_between_threads() {
        fn send<T: Send>() {}
        send::<WgpuTts>();
    }

    /// The Base model speaking two lines in a voice on WebGPU, streamed (`--ignored --nocapture`; `OAIY_TTS` the
    /// model's folder, `OAIY_TTS_VOICE` a saved voice's `.voice.json`): that it speaks, and how fast.
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
