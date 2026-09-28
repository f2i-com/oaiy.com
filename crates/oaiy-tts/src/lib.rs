//! OAIY's own text to speech: Qwen3-TTS (12 Hz; the 0.6B Base by default,
//! the 1.7B Base works the same) in Rust on Candle, on the GPU, streaming
//! audio as it is made, in a voice cloned from a few seconds of anyone
//! speaking.
//!
//! ```no_run
//! # fn main() -> candle_core::Result<()> {
//! use std::sync::atomic::AtomicBool;
//! let dev = oaiy_tts::cuda(1)?;
//! let mut tts = oaiy_tts::Tts::load("E:/models/Qwen3-TTS-12Hz-0.6B-Base".as_ref(), &dev)?;
//! let voice = tts.voice_from_audio("clip.mp3".as_ref(), Some("What the clip says, word for word."))?;
//! let cancel = AtomicBool::new(false);
//! tts.speak("Hello! How can I help?", &voice, &cancel, |pcm| {
//!     // 16-bit mono PCM at 24 kHz, a fraction of a second at a time.
//!     let _ = pcm;
//! })?;
//! # Ok(()) }
//! ```
//!
//! How it works: the talker, a 28-layer decoder, reads the text (after the
//! voice's reference clip: its transcript and codec codes, in context) and
//! writes one frame of 16 codec ids per 80 ms of speech; the codec's decoder
//! turns frames into audio as they come ([`codec::CodecStream`]). A voice is
//! made once from its clip (the speaker encoder and the codec's encoder) and
//! kept in a small file keyed by the clip and its transcript.
pub mod audio;
pub mod clone;
pub mod codec;
pub mod model;
pub mod sampling;
pub mod talker;
pub mod text;
pub mod voice;
pub mod weights;

pub use talker::Sampling;
pub use voice::Voice;

use candle_core::{Device, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// A CUDA device, or a clear error when this build has no CUDA.
pub fn cuda(ordinal: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(ordinal)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = ordinal;
        Err(msg("oaiy-tts was built without CUDA: build it with --features cuda (or flash-attn)"))
    }
}

/// Free and total memory on the device, when it is a GPU.
pub fn device_memory(dev: &Device) -> Option<(u64, u64)> {
    match dev {
        #[cfg(feature = "cuda")]
        Device::Cuda(d) => d.cuda_stream().context().mem_get_info().ok().map(|(free, total)| (free as u64, total as u64)),
        _ => None,
    }
}

/// How a line is spoken and streamed.
#[derive(Clone, Debug)]
pub struct SpeakOptions {
    /// `auto`, or a language the model knows (english, chinese, german, ...).
    /// `auto` defers to the voice's own language when it has one.
    pub language: String,
    pub sampling: Sampling,
    /// No line runs longer than this.
    pub max_seconds: f64,
    /// Frames (80 ms each) decoded before the first chunk.
    pub first_chunk_frames: usize,
    /// Frames per chunk after that.
    pub chunk_frames: usize,
    /// Earlier frames the codec decodes each chunk behind (the official 25).
    pub context_frames: usize,
    /// Drop the silence the model sometimes opens with (up to a second), down
    /// to a tenth of a second, so speech starts sooner.
    pub trim_leading_silence: bool,
}

impl Default for SpeakOptions {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            sampling: Sampling::default(),
            max_seconds: 120.,
            first_chunk_frames: 1,
            chunk_frames: 2,
            context_frames: codec::LEFT_CONTEXT,
            trim_leading_silence: true,
        }
    }
}

/// Why a line ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model finished the text.
    Stop,
    /// It reached `max_seconds` (or the length its text allows).
    Length,
    /// The cancel flag was set.
    Cancelled,
}

/// What speaking a line took.
#[derive(Clone, Debug)]
pub struct SpeakReport {
    pub finish: Finish,
    /// Codec frames made (80 ms each).
    pub frames: usize,
    /// Audio handed out, in seconds.
    pub audio_seconds: f64,
    /// From the call to the first chunk handed out.
    pub first_chunk_seconds: Option<f64>,
    /// From the call to the end of the prefill (the text read in).
    pub prefill_seconds: f64,
    pub total_seconds: f64,
}

impl SpeakReport {
    /// Seconds of compute per second of audio made (below 1 is faster than
    /// real time).
    pub fn real_time_factor(&self) -> f64 {
        let made = self.frames as f64 / talker::FRAMES_PER_SECOND;
        if made > 0. { self.total_seconds / made } else { 0. }
    }
}

/// The engine: a Qwen3-TTS Base model resident on one device.
pub struct Tts {
    talker: talker::Talker,
    codec: codec::CodecDecoder,
    tokenizer: tokenizers::Tokenizer,
    /// Loaded when the first voice is made, then kept.
    speaker_encoder: Option<clone::SpeakerEncoder>,
    speech_encoder: Option<clone::SpeechEncoder>,
    model_dir: PathBuf,
    dev: Device,
    /// FFmpeg, for voice clips that are not plain WAV (MP3 and the rest).
    pub ffmpeg: PathBuf,
    /// Where voices made from clips are kept (`None`: not kept).
    pub voice_cache: Option<PathBuf>,
    pub options: SpeakOptions,
}

impl Tts {
    /// Load a Qwen3-TTS Base folder (`config.json`, `model.safetensors`,
    /// `vocab.json`, `merges.txt`, `speech_tokenizer/`) onto `dev`.
    pub fn load(model_dir: &Path, dev: &Device) -> Result<Self> {
        let config = oaiy_engine::json::Json::parse(&std::fs::read(model_dir.join("config.json")).map_err(|e| msg(format!("{}: {e}", model_dir.join("config.json").display())))?)
            .map_err(candle_core::Error::wrap)?;
        if config.get("tts_model_type").and_then(|t| t.as_str()).is_some_and(|t| t != "base") {
            candle_core::bail!("{} is not a Qwen3-TTS Base model (voices are cloned with the Base model)", model_dir.display());
        }
        let tokenizer = text::tokenizer(model_dir)?;
        let talker = talker::Talker::load(model_dir, dev)?;
        let codec = codec::CodecDecoder::load(&model_dir.join("speech_tokenizer").join("model.safetensors"), dev)?;
        // Frames are made on whichever thread calls `speak`; the weights must
        // be on the device before another thread's stream reads them.
        dev.synchronize()?;
        Ok(Self {
            talker,
            codec,
            tokenizer,
            speaker_encoder: None,
            speech_encoder: None,
            model_dir: model_dir.to_path_buf(),
            dev: dev.clone(),
            ffmpeg: PathBuf::from("ffmpeg"),
            voice_cache: Some(std::env::temp_dir().join("oaiy-tts-voices")),
            options: SpeakOptions::default(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Device bytes held by the weights (talker and codec; the voice encoders
    /// too once a voice has been made).
    pub fn weight_bytes(&self) -> u64 {
        self.talker.bytes() + self.codec.bytes() + self.speaker_encoder.as_ref().map_or(0, |e| e.bytes()) + self.speech_encoder.as_ref().map_or(0, |e| e.bytes())
    }

    /// The talker's width (1024 for the 0.6B, 2048 for the 1.7B): voices are
    /// made for one size.
    pub fn width(&self) -> usize {
        self.talker.hidden()
    }

    /// A voice from a clip of someone speaking (MP3, WAV, or anything FFmpeg
    /// reads; 3 to 30 seconds is best, longer clips are cut to 30) and what it
    /// says. The transcript must match the clip word for word; without one
    /// the voice cannot be made (yet). Voices are kept in `voice_cache`, so a
    /// clip is only worked through once.
    pub fn voice_from_audio(&mut self, path: &Path, transcript: Option<&str>) -> Result<Voice> {
        let transcript = transcript.map(str::trim).filter(|t| !t.is_empty()).ok_or_else(|| {
            msg("a cloned voice needs the transcript of its clip (what it says, word for word): transcribe the clip first (the transcript must match audio::read_clip's cut)")
        })?;
        let bytes = std::fs::read(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
        let cached = self.voice_cache.as_ref().map(|dir| voice::cache_path(dir, &bytes, transcript, self.width()));
        if let Some(v) = cached.as_ref().and_then(|p| Voice::open(p).ok()) {
            if v.ref_text == transcript && v.speaker.len() == self.width() {
                return Ok(v);
            }
        }
        let clip = audio::read_clip(path, &self.ffmpeg).map_err(|e| msg(e.to_string()))?;
        let mut voice = self.voice_from_samples(&clip, transcript)?;
        voice.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(p) = cached {
            // A cache that cannot be written only costs the next call time.
            let _ = voice.save(&p);
        }
        Ok(voice)
    }

    /// A voice from 24 kHz mono samples and their transcript.
    pub fn voice_from_samples(&mut self, clip: &[f32], transcript: &str) -> Result<Voice> {
        if self.speaker_encoder.is_none() {
            self.speaker_encoder = Some(clone::SpeakerEncoder::load(&self.model_dir.join("model.safetensors"), &self.dev)?);
        }
        if self.speech_encoder.is_none() {
            self.speech_encoder = Some(clone::SpeechEncoder::load(&self.model_dir.join("speech_tokenizer").join("model.safetensors"), &self.dev)?);
        }
        let (Some(spk), Some(enc)) = (&self.speaker_encoder, &self.speech_encoder) else { unreachable!("both loaded above") };
        let speaker = spk.embed(clip)?;
        if speaker.len() != self.width() {
            candle_core::bail!("the speaker encoder gives {} values but the talker takes {}", speaker.len(), self.width());
        }
        let ref_codes = enc.encode(clip)?;
        self.dev.synchronize()?;
        Ok(Voice { name: String::new(), description: String::new(), language: "auto".into(), ref_text: transcript.trim().into(), ref_codes, speaker })
    }

    /// Speak `text` in `voice`, handing out 16-bit mono PCM at 24 kHz as it
    /// is made (the first chunk after one frame, then every
    /// `options.chunk_frames`). `cancel` is checked before every frame; once
    /// it is set, nothing more is handed out.
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
        let text_ids = text::encode(&self.tokenizer, text)?;
        let ref_ids = text::encode(&self.tokenizer, &voice.ref_text)?;
        let (prefill, trailing) = self.talker.prefill_clone(&text_ids, &ref_ids, voice, language)?;
        // A line longer than its text could need is babbling: cap it (two
        // seconds, plus 0.6 s a token, far slower than anyone speaks).
        let max_frames = ((o.max_seconds.min(2. + 0.6 * text_ids.len() as f64)) * talker::FRAMES_PER_SECOND).ceil() as usize;
        let mut g = self.talker.start(&prefill, Some(trailing), o.sampling.clone())?;
        report.prefill_seconds = started.elapsed().as_secs_f64();
        let mut stream = codec::CodecStream::primed(o.context_frames, &voice.ref_codes);
        let mut gate = LeadIn::new(o.trim_leading_silence);
        let mut pending: Vec<[u32; 16]> = Vec::new();
        let mut emit = |samples: Vec<f32>, report: &mut SpeakReport| {
            if samples.is_empty() {
                return;
            }
            if report.first_chunk_seconds.is_none() {
                report.first_chunk_seconds = Some(started.elapsed().as_secs_f64());
            }
            report.audio_seconds += samples.len() as f64 / codec::SAMPLE_RATE as f64;
            on_chunk(&audio::to_pcm16(&samples));
        };
        loop {
            if cancel.load(Ordering::Relaxed) {
                report.finish = Finish::Cancelled;
                break;
            }
            if report.frames >= max_frames {
                report.finish = Finish::Length;
                break;
            }
            let Some(frame) = self.talker.next_frame(&mut g)? else { break };
            report.frames += 1;
            pending.push(frame);
            let want = if report.first_chunk_seconds.is_none() { o.first_chunk_frames } else { o.chunk_frames };
            if pending.len() >= want.max(1) {
                let samples = stream.push(&self.codec, &pending)?;
                pending.clear();
                emit(gate.pass(samples), &mut report);
            }
        }
        if report.finish != Finish::Cancelled {
            let rest = stream.push(&self.codec, &pending)?;
            let mut out = gate.pass(rest);
            if !gate.open {
                // Quiet to the end: hand out what the gate held.
                out = std::mem::take(&mut gate.held);
            }
            emit(out, &mut report);
        }
        self.dev.synchronize()?;
        report.total_seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }
}

/// Holds back the silence a line opens with, keeping its last tenth of a
/// second as a lead-in; after a second of it, speech or not, it lets go.
struct LeadIn {
    open: bool,
    held: Vec<f32>,
    silent: usize,
}

impl LeadIn {
    const LEAD: usize = codec::SAMPLE_RATE / 10;
    const MAX_SILENCE: usize = codec::SAMPLE_RATE;
    const WINDOW: usize = codec::SAMPLE_RATE / 100;
    /// -46 dBFS: quieter than any speech, louder than the model's silence.
    const LEVEL: f32 = 0.005;

    fn new(enabled: bool) -> Self {
        Self { open: !enabled, held: Vec::new(), silent: 0 }
    }

    fn pass(&mut self, samples: Vec<f32>) -> Vec<f32> {
        if self.open {
            return samples;
        }
        let before = self.held.len();
        self.held.extend(samples);
        let loud = self.held.chunks(Self::WINDOW).position(|w| (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt() > Self::LEVEL);
        self.silent += self.held.len() - before;
        if let Some(i) = loud {
            self.open = true;
            return self.held.split_off((i * Self::WINDOW).saturating_sub(Self::LEAD));
        }
        let excess = self.held.len().saturating_sub(Self::LEAD);
        self.held.drain(..excess);
        if self.silent >= Self::MAX_SILENCE {
            self.open = true;
            return std::mem::take(&mut self.held);
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A server loads the engine on one thread and speaks on another.
    #[test]
    fn the_engine_can_move_between_threads() {
        fn send<T: Send>() {}
        send::<Tts>();
        send::<Voice>();
    }

    #[test]
    fn the_lead_in_drops_opening_silence_but_keeps_a_tenth_of_a_second() {
        let frame = codec::SAMPLES_PER_FRAME;
        let mut gate = LeadIn::new(true);
        // Three silent frames, then one that starts loud half way.
        for _ in 0..3 {
            assert!(gate.pass(vec![0.0001; frame]).is_empty());
        }
        let mut f = vec![0.0; frame / 2];
        f.extend(vec![0.3; frame / 2]);
        let out = gate.pass(f);
        assert_eq!(out.len(), LeadIn::LEAD + frame / 2);
        assert!(out[..LeadIn::LEAD].iter().all(|&s| s.abs() < 0.001));
        // Open: everything passes.
        assert_eq!(gate.pass(vec![0.; 10]).len(), 10);
        // Off: nothing is held.
        assert_eq!(LeadIn::new(false).pass(vec![0.; 7]).len(), 7);
        // A line that stays quiet is let go after a second.
        let mut quiet = LeadIn::new(true);
        let mut out = 0;
        for _ in 0..20 {
            out += quiet.pass(vec![0.001; frame]).len();
        }
        assert!(quiet.open && out > 0);
    }
}
