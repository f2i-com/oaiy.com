//! Native Rust Qwen3-TTS (12 Hz): text to speech in a voice described in
//! words (VoiceDesign), or in a saved voice (the Base model, cloning from a
//! short reference clip). The talker writes one frame of 16 codebook ids per
//! 80 ms; the speech codec turns frames into 24 kHz audio.
//!
//! A saved voice is made once: VoiceDesign speaks a sample line in the
//! described voice, then the Base model's speaker encoder and the codec's
//! encoder turn that clip into a speaker embedding and reference codes. Later
//! lines prompt the Base talker with them (in-context), so the voice holds.
//!
//! `backend` "webgpu" speaks on WebGPU, the talker ([`crate::tts_wgpu`]) and the speech codec's decoder
//! ([`crate::codec_wgpu`]) on any GPU, as a worker built with WebGPU and without CUDA does by default (Candle's talker
//! needs CUDA: its weights BF16); a voice is designed there too, its clip's speaker embedding and codes then worked out
//! on the CPU (both encoders F32). Breeze TTS 2 is not on WebGPU.
pub mod breeze;
pub mod clone;
pub mod codec;
pub mod model;

use candle_core::{Device, Result};
use oaiy_engine::json::Json;
use oaiy_tts::talker::{Sampling, Talker as Tts, FRAMES_PER_SECOND};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// The talker, the tokenizer, sampling and voices are `oaiy-tts`'s (the
// realtime engine speaks with the same code).
pub use oaiy_tts::audio::write_wav;
pub use oaiy_tts::sampling::Rng;
pub use oaiy_tts::text::tokenizer;
pub use oaiy_tts::voice::Voice;
use oaiy_tts::sampling::sample;
use oaiy_tts::text::encode;

#[derive(Clone, Debug)]
pub struct Request {
    /// The model folder (config.json, model.safetensors, vocab.json,
    /// merges.txt, speech_tokenizer/).
    pub model_dir: PathBuf,
    pub text: String,
    /// The voice, described in words (VoiceDesign).
    pub instructions: String,
    /// `auto` or a language the model knows (english, chinese, ...).
    pub language: String,
    pub output: PathBuf,
    pub seed: u64,
    pub device: usize,
    pub max_seconds: f64,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    /// Whether the request gave the repetition penalty (else each engine's default).
    pub repetition_penalty_set: bool,
    /// Breeze TTS 2: classifier-free guidance toward the instruction (1 = none).
    pub cfg_scale: Option<f64>,
    /// Argmax instead of sampling (for tests; it tends to loop on long text).
    pub greedy: bool,
    /// Speak in a saved voice (a file written by `design_voice`); `model_dir`
    /// is then the Base model.
    pub voice: Option<Voice>,
    /// On WebGPU.
    pub webgpu: bool,
}

impl Request {
    /// The talker's sampling: the request's settings, for the code predictor too.
    fn sampling(&self) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            sub_temperature: self.temperature,
            sub_top_k: self.top_k,
            greedy: self.greedy,
            seed: self.seed,
            ..Sampling::default()
        }
    }

    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_owned);
        let f = |k: &str, d: f64| j.get(k).and_then(Json::as_f64).unwrap_or(d);
        let r = Self {
            model_dir: s("model_dir").filter(|p| !p.trim().is_empty()).ok_or("speech: missing model_dir")?.into(),
            text: s("text").filter(|t| !t.trim().is_empty()).ok_or("speech: text must not be empty")?,
            instructions: s("instructions").unwrap_or_default(),
            language: s("language").unwrap_or_else(|| "auto".into()).to_lowercase(),
            output: s("output_dir").ok_or("speech: missing output_dir")?.into(),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            max_seconds: f("max_seconds", 120.),
            temperature: f("temperature", 0.9),
            top_k: j.get("top_k").and_then(Json::as_i64).unwrap_or(50).max(1) as usize,
            top_p: f("top_p", 1.0),
            repetition_penalty: f("repetition_penalty", 1.05),
            repetition_penalty_set: j.get("repetition_penalty").and_then(Json::as_f64).is_some(),
            cfg_scale: j.get("cfg_scale").and_then(Json::as_f64),
            greedy: j.get("greedy").and_then(Json::as_bool).unwrap_or(false),
            voice: match j.get("voice_file").and_then(Json::as_str) {
                None => None,
                Some(p) => {
                    let bytes = std::fs::read(p).map_err(|e| format!("voice file {p}: {e}"))?;
                    Some(Voice::from_json(&Json::parse(&bytes).map_err(|e| format!("voice file {p}: {e}"))?)?)
                }
            },
            webgpu: match s("backend").as_deref() {
                Some("webgpu") => true,
                Some("cuda" | "cpu") => false,
                Some(other) => return Err(format!("speech: backend must be webgpu, cuda or cpu, not {other}")),
                None => cfg!(feature = "webgpu"),
            },
        };
        if r.webgpu && !cfg!(feature = "webgpu") {
            return Err("speech: this build has no WebGPU (the webgpu feature)".into());
        }
        if r.text.len() > 20_000 || r.instructions.len() > 4_000 {
            return Err("speech: text is limited to 20000 bytes and instructions to 4000".into());
        }
        if !(1.0..=600.0).contains(&r.max_seconds) {
            return Err("speech: max_seconds must be 1..600".into());
        }
        if !(0.05..=2.0).contains(&r.temperature) || !(0.0..=1.0).contains(&r.top_p) || r.top_p == 0.0 {
            return Err("speech: temperature must be 0.05..2 and top_p in (0, 1]".into());
        }
        Ok(r)
    }
}

/// Make a reusable voice from a description: VoiceDesign speaks `sample`
/// in the described voice, then the Base model's encoders turn the clip into
/// a `Voice`.
#[derive(Clone, Debug)]
pub struct DesignRequest {
    pub design_dir: PathBuf,
    pub base_dir: PathBuf,
    pub name: String,
    pub description: String,
    pub sample: String,
    pub language: String,
    pub seed: u64,
    pub device: usize,
    pub output: PathBuf,
    /// The sample spoken on WebGPU (the encoders then on the CPU).
    pub webgpu: bool,
}

impl DesignRequest {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::trim).filter(|v| !v.is_empty()).map(str::to_owned);
        let r = Self {
            design_dir: s("design_model_dir").ok_or("voice: missing design_model_dir")?.into(),
            base_dir: s("base_model_dir").ok_or("voice: missing base_model_dir")?.into(),
            name: s("name").ok_or("voice: missing name")?,
            description: s("description").ok_or("voice: describe the voice")?,
            sample: s("sample_text").unwrap_or_else(|| "Hello there. This is my voice, and this is how I sound when I read a few sentences out loud.".into()),
            language: s("language").unwrap_or_else(|| "auto".into()).to_lowercase(),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            output: s("output_dir").ok_or("voice: missing output_dir")?.into(),
            webgpu: match s("backend").as_deref() {
                Some("webgpu") => true,
                Some("cuda" | "cpu") => false,
                Some(other) => return Err(format!("voice: backend must be webgpu, cuda or cpu, not {other}")),
                None => cfg!(feature = "webgpu"),
            },
        };
        if r.webgpu && !cfg!(feature = "webgpu") {
            return Err("voice: this build has no WebGPU (the webgpu feature)".into());
        }
        if r.description.len() > 4000 || r.sample.len() > 2000 || r.name.len() > 80 {
            return Err("voice: description, sample text or name too long".into());
        }
        Ok(r)
    }
}

pub fn design_voice(r: &DesignRequest, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    if breeze::is_breeze(&r.design_dir) {
        report(event("designing_voice", 0, 3));
        let (voice, clip) = breeze::design(&r.design_dir, &r.name, &r.description, &r.sample, &r.language, r.seed, r.device, &r.output)?;
        return save_voice(voice, &clip, &r.output, started, &mut report);
    }
    #[cfg(feature = "webgpu")]
    if r.webgpu {
        return design_voice_webgpu(r, started, report);
    }
    let dev = device(r.device)?;
    report(event("designing_voice", 0, 3));
    let speak = design_sample(r);
    let tok = tokenizer(&r.design_dir)?;
    let mut tts = Tts::load(&r.design_dir, &dev)?;
    let language = tts.language_id(&r.language)?;
    let instruct = encode(&tok, &r.description)?;
    let prefill = tts.prefill(&encode(&tok, &r.sample)?, Some(&instruct), language)?;
    let frames = tts.frames(&prefill, None, speak.sampling(), 375, |_| {})?;
    drop(tts);
    if frames.len() < 12 {
        candle_core::bail!("the voice sample came out too short; try a longer sample text or another seed");
    }
    report(event("designing_voice", 1, 3));
    let codec = codec::CodecDecoder::load(&r.design_dir.join("speech_tokenizer").join("model.safetensors"), &dev)?;
    let clip = codec.decode(&frames)?;
    drop(codec);
    report(event("designing_voice", 2, 3));
    let speaker = clone::SpeakerEncoder::load(&r.base_dir.join("model.safetensors"), &dev)?.embed(&clip)?;
    let ref_codes = clone::SpeechEncoder::load(&r.base_dir.join("speech_tokenizer").join("model.safetensors"), &dev)?.encode(&clip)?;
    let voice = Voice { name: r.name.clone(), description: r.description.clone(), language: r.language.clone(), ref_text: r.sample.clone(), ref_codes, speaker };
    save_voice(voice, &clip, &r.output, started, &mut report)
}

/// How a voice's sample is spoken: its text in the described voice, sampled as the reference's demo does.
fn design_sample(r: &DesignRequest) -> Request {
    Request {
        model_dir: r.design_dir.clone(),
        text: r.sample.clone(),
        instructions: r.description.clone(),
        language: r.language.clone(),
        output: r.output.clone(),
        seed: r.seed,
        device: r.device,
        max_seconds: 30.,
        temperature: 0.9,
        top_k: 50,
        top_p: 1.0,
        repetition_penalty: 1.05,
        repetition_penalty_set: true,
        cfg_scale: None,
        greedy: false,
        voice: None,
        webgpu: r.webgpu,
    }
}

/// [`design_voice`] on WebGPU: the sample spoken and decoded there ([`crate::tts_wgpu::WgpuTalker`],
/// [`crate::codec_wgpu::WgpuCodec`]), then the Base model's speaker and speech encoders (F32) on the CPU.
#[cfg(feature = "webgpu")]
fn design_voice_webgpu(r: &DesignRequest, started: Instant, mut report: impl FnMut(Json)) -> Result<Json> {
    report(event("designing_voice", 0, 3));
    let speak = design_sample(r);
    let tok = tokenizer(&r.design_dir)?;
    let mut tts = crate::tts_wgpu::WgpuTalker::load(&r.design_dir, r.device)?;
    let language = tts.language_id(&r.language)?;
    let instruct = encode(&tok, &r.description)?;
    let prefill = tts.prefill(&encode(&tok, &r.sample)?, Some(&instruct), language)?;
    let frames = tts.frames(&prefill, None, speak.sampling(), 375, |_| {})?;
    let gpu = tts.gpu().clone();
    drop(tts);
    if frames.len() < 12 {
        candle_core::bail!("the voice sample came out too short; try a longer sample text or another seed");
    }
    report(event("designing_voice", 1, 3));
    let clip = crate::codec_wgpu::WgpuCodec::load(&r.design_dir.join("speech_tokenizer").join("model.safetensors"), &gpu)?.decode(&frames)?;
    drop(gpu);
    report(event("designing_voice", 2, 3));
    let speaker = clone::SpeakerEncoder::load(&r.base_dir.join("model.safetensors"), &Device::Cpu)?.embed(&clip)?;
    let ref_codes = clone::SpeechEncoder::load(&r.base_dir.join("speech_tokenizer").join("model.safetensors"), &Device::Cpu)?.encode(&clip)?;
    let voice = Voice { name: r.name.clone(), description: r.description.clone(), language: r.language.clone(), ref_text: r.sample.clone(), ref_codes, speaker };
    save_voice(voice, &clip, &r.output, started, &mut report)
}

/// The voice file and its sample clip, written beside each other.
fn save_voice(voice: Voice, clip: &[f32], output: &Path, started: Instant, report: &mut impl FnMut(Json)) -> Result<Json> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let clip_path = output.join(format!("voice-{stamp}.wav"));
    write_wav(&clip_path, clip, codec::SAMPLE_RATE)?;
    let voice_path = clip_path.with_extension("voice.json");
    std::fs::write(&voice_path, voice.to_json().to_json())?;
    report(event("designing_voice", 3, 3));
    Ok(Json::obj([
        ("voice_file", Json::str(voice_path.to_string_lossy())),
        ("path", Json::str(clip_path.to_string_lossy())),
        ("name", Json::str(&voice.name)),
        ("duration", Json::Num(clip.len() as f64 / codec::SAMPLE_RATE as f64)),
        ("frames", Json::Int(voice.ref_codes.len() as i64)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
    ]))
}

fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

fn device(index: usize) -> Result<Device> {
    // The models run in BF16, which the CPU backend cannot multiply.
    {
        let _ = index;
        candle_core::bail!("speech need a GPU: this oaiy-media was built without CUDA (build it with --features cuda or flash-attn)")
    }
}

pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    if breeze::is_breeze(&r.model_dir) {
        return breeze::generate(r, report);
    }
    if r.voice.as_ref().is_some_and(|v| v.speaker.is_empty()) {
        candle_core::bail!("this voice was made with Breeze TTS 2; speak it with a Breeze model");
    }
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    #[cfg(feature = "webgpu")]
    if r.webgpu {
        return generate_webgpu(r, report);
    }
    let dev = device(r.device)?;
    report(event("loading_speech_model", 0, 1));
    let tok = tokenizer(&r.model_dir)?;
    let mut tts = Tts::load(&r.model_dir, &dev)?;
    let language = tts.language_id(&r.language)?;
    let text_ids = encode(&tok, &r.text)?;
    let (prefill, trailing) = match &r.voice {
        Some(voice) => {
            let ref_ids = encode(&tok, &voice.ref_text)?;
            let (p, t) = tts.prefill_clone(&text_ids, &ref_ids, voice, language)?;
            (p, Some(t))
        }
        None => {
            let instruct_ids = if r.instructions.trim().is_empty() { None } else { Some(encode(&tok, &r.instructions)?) };
            (tts.prefill(&text_ids, instruct_ids.as_deref(), language)?, None)
        }
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let max_frames = (r.max_seconds * FRAMES_PER_SECOND).ceil() as usize;
    let speak_started = Instant::now();
    let frames = tts.frames(&prefill, trailing, r.sampling(), max_frames, |n| report(event("speaking", n, max_frames)))?;
    let speak_seconds = speak_started.elapsed().as_secs_f64();
    drop(tts);
    report(event("decoding_speech", 0, 1));
    let decode_started = Instant::now();
    let codec = codec::CodecDecoder::load(&r.model_dir.join("speech_tokenizer").join("model.safetensors"), &dev)?;
    // A saved voice's clip is decoded ahead of the new frames (as context for
    // the causal decoder) and cut off again.
    let samples = match (&r.voice, frames.is_empty()) {
        (_, true) => Vec::new(),
        (Some(voice), false) => {
            let mut all = voice.ref_codes.clone();
            all.extend_from_slice(&frames);
            let wave = codec.decode(&all)?;
            wave[(voice.ref_codes.len() * codec::SAMPLES_PER_FRAME).min(wave.len())..].to_vec()
        }
        (None, false) => codec.decode(&frames)?,
    };
    // The model opens with up to most of a second of silence: a clip that
    // follows the line would spend it with the speaker's mouth shut.
    let samples = trim_leading_silence(samples, codec::SAMPLE_RATE);
    drop(codec);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("speech-{stamp}-{}.wav", r.seed));
    write_wav(&path, &samples, codec::SAMPLE_RATE)?;
    let audio_seconds = samples.len() as f64 / codec::SAMPLE_RATE as f64;
    let result = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(codec::SAMPLE_RATE as i64)),
        ("frames", Json::Int(frames.len() as i64)),
        ("duration", Json::Num(audio_seconds)),
        ("finish_reason", Json::str(if frames.len() >= max_frames { "length" } else { "stop" })),
        ("load_seconds", Json::Num(load_seconds)),
        ("speak_seconds", Json::Num(speak_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

/// [`generate`] on WebGPU: the talker ([`crate::tts_wgpu::WgpuTalker`]), then the codec's decoder
/// ([`crate::codec_wgpu::WgpuCodec`]) on its device.
#[cfg(feature = "webgpu")]
fn generate_webgpu(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    report(event("loading_speech_model", 0, 1));
    let tok = tokenizer(&r.model_dir)?;
    let mut tts = crate::tts_wgpu::WgpuTalker::load(&r.model_dir, r.device)?;
    let language = tts.language_id(&r.language)?;
    let text_ids = encode(&tok, &r.text)?;
    let (prefill, trailing) = match &r.voice {
        Some(voice) => {
            let ref_ids = encode(&tok, &voice.ref_text)?;
            let (p, t) = tts.prefill_clone(&text_ids, &ref_ids, voice, language)?;
            (p, Some(t))
        }
        None => {
            let instruct_ids = if r.instructions.trim().is_empty() { None } else { Some(encode(&tok, &r.instructions)?) };
            (tts.prefill(&text_ids, instruct_ids.as_deref(), language)?, None)
        }
    };
    let load_seconds = started.elapsed().as_secs_f64();
    let max_frames = (r.max_seconds * FRAMES_PER_SECOND).ceil() as usize;
    let speak_started = Instant::now();
    let frames = tts.frames(&prefill, trailing, r.sampling(), max_frames, |n| report(event("speaking", n, max_frames)))?;
    let speak_seconds = speak_started.elapsed().as_secs_f64();
    let gpu = tts.gpu().clone();
    drop(tts);
    report(event("decoding_speech", 0, 1));
    let decode_started = Instant::now();
    let codec = crate::codec_wgpu::WgpuCodec::load(&r.model_dir.join("speech_tokenizer").join("model.safetensors"), &gpu)?;
    let samples = match (&r.voice, frames.is_empty()) {
        (_, true) => Vec::new(),
        (Some(voice), false) => {
            let mut all = voice.ref_codes.clone();
            all.extend_from_slice(&frames);
            let wave = codec.decode(&all)?;
            wave[(voice.ref_codes.len() * codec::SAMPLES_PER_FRAME).min(wave.len())..].to_vec()
        }
        (None, false) => codec.decode(&frames)?,
    };
    let samples = trim_leading_silence(samples, codec::SAMPLE_RATE);
    drop(codec);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("speech-{stamp}-{}.wav", r.seed));
    write_wav(&path, &samples, codec::SAMPLE_RATE)?;
    let audio_seconds = samples.len() as f64 / codec::SAMPLE_RATE as f64;
    let result = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(codec::SAMPLE_RATE as i64)),
        ("frames", Json::Int(frames.len() as i64)),
        ("duration", Json::Num(audio_seconds)),
        ("finish_reason", Json::str(if frames.len() >= max_frames { "length" } else { "stop" })),
        ("load_seconds", Json::Num(load_seconds)),
        ("speak_seconds", Json::Num(speak_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
        ("backend", Json::str("webgpu")),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

/// Silence before the first sound, down to a tenth of a second of it. "Sound"
/// is a 10 ms window louder than -40 dB, so a lone click does not count.
fn trim_leading_silence(samples: Vec<f32>, rate: usize) -> Vec<f32> {
    let window = (rate / 100).max(1);
    let loud = samples.chunks(window).position(|w| (w.iter().map(|x| x * x).sum::<f32>() / w.len() as f32).sqrt() > 0.01);
    match loud {
        Some(i) => samples[(i * window).saturating_sub(rate / 10)..].to_vec(),
        None => samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use oaiy_tts::model::Cache;

    #[test]
    fn a_line_starts_a_tenth_of_a_second_before_its_first_sound() {
        let rate = 24_000;
        let mut line = vec![0.0005f32; rate / 2];
        line.extend(vec![0.3; rate]);
        let trimmed = trim_leading_silence(line, rate);
        assert_eq!(trimmed.len(), rate + rate / 10);
        assert_eq!(trim_leading_silence(vec![0.; 100], rate).len(), 100);
        assert_eq!(trim_leading_silence(vec![0.3; 100], rate).len(), 100);
    }

    fn relative(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        let actual = actual.to_dtype(DType::F32)?;
        let error = (&actual - expected)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt();
        Ok(error / expected.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt())
    }

    #[test]
    #[ignore = "requires the Qwen3-TTS Base model and clone reference dumps; OAIY_TTS_GOLDEN (clone folder), OAIY_TTS_BASE"]
    fn clone_prompt_matches_reference() -> Result<()> {
        let root = PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let base = PathBuf::from(std::env::var("OAIY_TTS_BASE").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("OAIY_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let ints = |name: &str| -> Result<Vec<u32>> {
            Ok(std::fs::read(root.join(name))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect())
        };
        let floats = |name: &str| -> Result<Vec<f32>> {
            Ok(std::fs::read(root.join(name))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
        };
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let meta = Json::parse(&std::fs::read(root.join("meta.json"))?).map_err(candle_core::Error::wrap)?;
        let ref_text = meta.get("ref_text").and_then(Json::as_str).unwrap_or_default();
        let new_text = meta.get("new_text").and_then(Json::as_str).unwrap_or_default();
        let tok = tokenizer(&base)?;
        let (ref_ids, text_ids) = (encode(&tok, ref_text)?, encode(&tok, new_text)?);
        let (want_ref, want_text) = (ints("ref_ids.i32")?, ints("input_ids.i32")?);
        assert_eq!(ref_ids, want_ref[3..want_ref.len() - 2], "reference transcript ids");
        assert_eq!(text_ids, want_text[3..want_text.len() - 5], "new text ids");
        let codes = ints("ref_code.i32")?;
        let voice = Voice {
            name: "golden".into(),
            description: String::new(),
            language: "english".into(),
            ref_text: ref_text.into(),
            ref_codes: codes.chunks_exact(16).map(|c| c.try_into().unwrap()).collect(),
            speaker: floats("spk_embedding_bf16.f32")?,
        };
        let back = Voice::from_json(&Json::parse(voice.to_json().to_json().as_bytes()).map_err(candle_core::Error::wrap)?).map_err(candle_core::Error::Msg)?;
        assert_eq!(back.ref_codes, voice.ref_codes, "voice files round-trip");
        let mut tts = Tts::load(&base, &dev)?;
        let lang = tts.language_id("english")?;
        let (prefill, trailing) = tts.prefill_clone(&text_ids, &ref_ids, &voice, lang)?;
        let n = prefill.dim(1)?;
        assert_eq!(n, 100, "prefill positions");
        let e = relative(&prefill.squeeze(0)?, &read("icl_prefill_in.f32", &[n, 2048])?)?;
        println!("clone prefill ({n} positions): relative RMS error {e}");
        assert!(e < 0.01, "clone prefill {e}");
        let e = relative(&trailing.squeeze(0)?, &read("icl_trailing_text.f32", &[1, 2048])?)?;
        println!("trailing text: relative RMS error {e}");
        assert!(e < 0.01, "trailing {e}");
        let mut cache = Cache::new(tts.decoder().layers());
        let out = tts.decoder().forward(&prefill, &mut cache)?;
        let e = relative(&out.squeeze(0)?, &read("icl_prefill_out.f32", &[n, 2048])?)?;
        println!("Base talker on the clone prefill: relative RMS error {e}");
        assert!(e < 0.03, "clone talker {e}");
        Ok(())
    }

    #[test]
    #[ignore = "requires the Qwen3-TTS VoiceDesign model and reference dumps; OAIY_TTS_GOLDEN, OAIY_TTS_MODEL"]
    fn talker_matches_reference() -> Result<()> {
        let root = PathBuf::from(std::env::var("OAIY_TTS_GOLDEN").map_err(candle_core::Error::wrap)?);
        let model = PathBuf::from(std::env::var("OAIY_TTS_MODEL").map_err(candle_core::Error::wrap)?);
        let dev = Device::new_cuda(std::env::var("OAIY_TTS_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        let ints = |name: &str| -> Result<Vec<u32>> {
            Ok(std::fs::read(root.join(name))?.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u32).collect())
        };
        let read = |name: &str, shape: &[usize]| -> Result<Tensor> {
            Tensor::from_raw_buffer(&std::fs::read(root.join(name))?, DType::F32, shape, &dev)
        };
        let meta = Json::parse(&std::fs::read(root.join("meta.json"))?).map_err(candle_core::Error::wrap)?;
        let text = meta.get("text").and_then(Json::as_str).unwrap_or_default();
        let instruct = meta.get("instruct").and_then(Json::as_str).unwrap_or_default();
        let tok = tokenizer(&model)?;
        let (input_ids, instruct_ids) = (ints("input_ids.i32")?, ints("instruct_ids.i32")?);
        let text_ids = encode(&tok, text)?;
        let inst_ids = encode(&tok, instruct)?;
        assert_eq!(text_ids, input_ids[3..input_ids.len() - 5], "text token ids");
        assert_eq!(inst_ids, instruct_ids[3..instruct_ids.len() - 2], "instruct token ids");
        let mut tts = Tts::load(&model, &dev)?;
        let lang = tts.language_id("english")?;
        let prefill = tts.prefill(&text_ids, Some(&inst_ids), lang)?;
        let n = prefill.dim(1)?;
        let e = relative(&prefill.squeeze(0)?, &read("talker_prefill_in.f32", &[n, 2048])?)?;
        println!("prefill embeddings ({n} positions): relative RMS error {e}");
        assert!(e < 0.01, "prefill {e}");
        let mut cache = Cache::new(tts.decoder().layers());
        let out = tts.decoder().forward(&prefill, &mut cache)?;
        let e = relative(&out.squeeze(0)?, &read("talker_prefill_out.f32", &[n, 2048])?)?;
        println!("talker prefill output: relative RMS error {e}");
        assert!(e < 0.03, "talker prefill {e}");
        // Greedy frames: the reference's first frames, exactly.
        let greedy = ints("greedy_codes.i32")?;
        let r = Request {
            model_dir: model.clone(),
            text: text.into(),
            instructions: instruct.into(),
            language: "english".into(),
            output: root.clone(),
            seed: 0,
            device: 0,
            max_seconds: 10.,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            repetition_penalty: 1.05,
            repetition_penalty_set: true,
            cfg_scale: None,
            greedy: true,
            voice: None,
            webgpu: false,
        };
        let frames = tts.frames(&prefill, None, r.sampling(), 4, |_| {})?;
        for (i, f) in frames.iter().enumerate() {
            let want = &greedy[i * 16..(i + 1) * 16];
            println!("frame {i}: ours {:?}\n         ref  {:?}", f, want);
        }
        assert_eq!(frames[0][0], greedy[0], "first codebook-0 id");
        let matching = frames.iter().enumerate().filter(|(i, f)| f[..] == greedy[i * 16..(i + 1) * 16]).count();
        println!("{matching} of {} greedy frames identical", frames.len());
        assert!(matching >= 2, "greedy frames diverge immediately");
        Ok(())
    }
}
