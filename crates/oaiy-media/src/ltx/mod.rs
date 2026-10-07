//! Native Rust text/image-to-video for LTX 2.3/2.5 and compatible distilled checkpoints.
//!
//! A clip can start from an image, end on one, or both. It can come with a
//! soundtrack generated with the picture, or follow one it is given: an audio
//! file, or speech made first in the same job (Qwen3-TTS). A given soundtrack
//! is encoded by the audio VAE and held fixed while the picture is denoised
//! (the official `a2vid` pipeline's frozen audio), so mouths and motion follow
//! it; the clip keeps the original audio.
mod audio;
mod cache;
pub(crate) mod store;
pub(crate) mod text;
pub(crate) mod transformer;
mod upsampler;
pub mod vae;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use store::Store;

const GIB: u64 = 1 << 30;
const SIGMAS: [f64; 9] = [
    1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.,
];

/// The reference's negative prompt for guided (non-distilled) sampling.
pub const DEFAULT_NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, \
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// Guided sampling, for a non-distilled (dev) transformer: the reference's
/// first stage. Each step runs the prompt, the negative prompt, the prompt
/// with block `stg_block`'s video self-attention passed through, and
/// (following a soundtrack) the pass without audio-video cross-attention, and
/// combines their denoised predictions.
#[derive(Clone, Debug)]
pub struct Guidance {
    pub steps: usize,
    pub cfg: f64,
    pub stg: f64,
    pub stg_block: usize,
    pub rescale: f64,
    pub negative_prompt: String,
}
impl Guidance {
    fn parse(j: &Json) -> std::result::Result<Self, String> {
        let num = |k: &str, default: f64, lo: f64, hi: f64| -> std::result::Result<f64, String> {
            match j.get(k) {
                None | Some(Json::Null) => Ok(default),
                Some(v) => v
                    .as_f64()
                    .filter(|v| (lo..=hi).contains(v))
                    .ok_or_else(|| format!("guidance.{k} must be a number from {lo} to {hi}")),
            }
        };
        let g = Self {
            steps: num("steps", 30., 2., 100.)? as usize,
            cfg: num("cfg", 3., 1., 20.)?,
            stg: num("stg", 1., 0., 10.)?,
            stg_block: num("stg_block", 28., 0., 47.)? as usize,
            rescale: num("rescale", 0.7, 0., 1.)?,
            negative_prompt: j
                .get("negative_prompt")
                .and_then(Json::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(DEFAULT_NEGATIVE_PROMPT)
                .to_owned(),
        };
        Ok(g)
    }
}

/// NAG defaults (the reference implementation's for video models): push the
/// cross-attention output 11x away from the negative's, keep its size within
/// 2.5x the plain one's, and blend a quarter of it in.
const NAG_SCALE: f64 = 11.;
const NAG_TAU: f64 = 2.5;
const NAG_ALPHA: f64 = 0.25;

/// The reference's LTX-2 schedule for guided sampling: `steps` evenly spaced
/// times, shifted by 2.05 (its default token count) and stretched so the last
/// nonzero sigma is 0.1.
fn guided_sigmas(steps: usize) -> Vec<f64> {
    let shift = 2.05f64.exp();
    let mut s: Vec<f64> = (0..=steps)
        .map(|i| 1. - i as f64 / steps as f64)
        .map(|t| if t == 0. { 0. } else { shift / (shift + (1. / t - 1.)) })
        .collect();
    let scale = (1. - s[steps - 1]) / (1. - 0.1);
    for v in s.iter_mut().take(steps) {
        *v = 1. - (1. - *v) / scale;
    }
    s
}

/// The reference's second stage (its two-stage pipelines): after guided
/// sampling at half the size, the latent is doubled by `upsampler` and
/// refined by the distilled `transformer` over the last three steps of its
/// schedule, with the soundtrack held as in stage one and no guidance.
#[derive(Clone, Debug)]
pub struct Refine {
    pub transformer: PathBuf,
    pub upsampler: PathBuf,
}

/// The longest reference voice used (ID-LoRA trained on a few seconds).
const MAX_REFERENCE_SECONDS: f64 = 10.;

/// Stage two's schedule: the distilled one from 0.909375.
const REFINE_SIGMAS: [f64; 4] = [0.909375, 0.725, 0.421875, 0.];

/// The prompt's contexts for the refining transformer: from the prompt cache
/// (keyed by that transformer), or Gemma and its connectors.
fn refine_contexts(
    r: &Request,
    transformer: &std::path::Path,
    store: &mut Store,
    dev: &Device,
    report: &mut dyn FnMut(Json),
) -> Result<(Tensor, Option<Tensor>)> {
    let refine = Request { transformer: transformer.to_path_buf(), ..r.clone() };
    let video_cache = cache::PromptCache::new(&refine);
    let audio_cache = if r.audio { cache::PromptCache::audio(&refine) } else { None };
    let cached = video_cache.as_ref().and_then(|c| c.load(dev));
    let cached_audio = audio_cache.as_ref().and_then(|c| c.load(dev));
    if let Some(context) = cached {
        if !r.audio || cached_audio.is_some() {
            return Ok((context, cached_audio));
        }
    }
    report(event("encoding_refine_prompt", 0, 48));
    let (features, audio_features) = text::encode(
        &r.text_encoder,
        r.tokenizer.as_deref(),
        store,
        &r.prompt,
        r.model == "ltx-2.5",
        r.audio,
        dev,
        |n| report(event("encoding_refine_prompt", n, 48)),
    )?;
    let context = transformer::connector(store, &features, "video_embeddings_connector", dev, |_| {})?;
    drop(features);
    let audio_context = audio_features
        .map(|f| transformer::connector(store, &f, "audio_embeddings_connector", dev, |_| {}))
        .transpose()?;
    if let Some(c) = &video_cache {
        let _ = c.save(&context);
    }
    if let (Some(c), Some(a)) = (&audio_cache, &audio_context) {
        let _ = c.save(a);
    }
    Ok((context, audio_context))
}

/// Speech loudness: the 95th percentile of 20 ms RMS, the level of its
/// voiced parts (pauses do not pull it down). None for silence.
fn speech_loudness(samples: &[f32], rate: usize) -> Option<f32> {
    let w = (rate / 50).max(1);
    let mut rms: Vec<f32> = samples.chunks(w).map(|c| (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt()).collect();
    rms.sort_by(|a, b| a.total_cmp(b));
    let at = rms.get(rms.len() * 95 / 100).copied()?;
    (at > 1e-5).then_some(at)
}

/// The generated soundtrack's level (interleaved stereo): matched to the
/// reference voice's loudness when there is one (within 4x either way), then
/// scaled so the peak stays under -0.3 dB rather than clipping.
fn set_level(interleaved: &mut [f32], rate: usize, reference: Option<f32>) {
    let left: Vec<f32> = interleaved.iter().step_by(2).copied().collect();
    let mut gain = match (reference, speech_loudness(&left, rate)) {
        (Some(want), Some(have)) => (want / have).clamp(0.25, 4.),
        _ => 1.,
    };
    let peak = interleaved.iter().fold(0f32, |m, x| m.max(x.abs())) * gain;
    const CEILING: f32 = 0.966; // -0.3 dBFS
    if peak > CEILING {
        gain *= CEILING / peak;
    }
    if gain != 1. {
        interleaved.iter_mut().for_each(|x| *x *= gain);
    }
}

/// The standard deviation of every element.
fn std_all(t: &Tensor) -> Result<f64> {
    let t = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean = t.mean_all()?.to_scalar::<f32>()?;
    Ok(t.affine(1., -mean as f64)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() as f64)
}

#[derive(Clone, Debug)]
pub struct Request {
    pub model: String,
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub vae: PathBuf,
    /// The audio VAE with its vocoder (LTX 2.3 and Sulphur checkpoints carry
    /// their own; LTX 2.5 ships it separately). With `audio`, the clip gets a
    /// soundtrack generated jointly with the picture.
    pub audio_vae: Option<PathBuf>,
    pub audio: bool,
    /// A soundtrack for the picture to follow (any format FFmpeg reads).
    pub audio_file: Option<PathBuf>,
    /// Speech to make first and follow: a `kind: "speech"` worker request.
    pub speech: Option<Json>,
    /// The words in `audio_file`, if known. They (or the speech's text) are
    /// added to the prompt: the models move lips much more readily when the
    /// prompt says what is spoken.
    pub transcript: Option<String>,
    /// No length was asked for: the clip is as long as the soundtrack (up to
    /// 121 frames).
    pub frames_from_audio: bool,
    /// Audio-to-video guidance while following a soundtrack: each step also
    /// runs without the audio-video cross-attention and pushes the picture
    /// away from that (the reference's modality guidance; 1 is off). 3 with
    /// guided sampling, as the reference; off by default for distilled models,
    /// where it did not make mouths follow the words measurably better,
    /// deforms faces, and doubles the time.
    pub a2v_guidance: f64,
    /// How a given soundtrack is held while the picture follows it: "inpaint"
    /// (the default: at every step, noised to that step's sigma as if it were
    /// being generated, and replaced again, so the picture sees audio as it
    /// was trained to, denoised together with it) or "frozen" (clean, at
    /// timestep 0, as the reference's a2v pipeline; lips followed it more
    /// loosely). The output keeps the file either way.
    pub soundtrack_mode: String,
    /// Guided sampling with a non-distilled transformer; none for distilled ones.
    pub guidance: Option<Guidance>,
    /// What the video should not show. With CFG guidance it is the guidance's
    /// negative prompt; otherwise (distilled models, CFG 1) it steers by NAG.
    pub negative_prompt: Option<String>,
    /// NAG's (scale, tau, alpha).
    pub nag: (f64, f64, f64),
    /// With `guidance`: render at half size, then upsample and refine.
    pub refine: Option<Refine>,
    /// A LoRA added to the transformer, and its strength.
    pub lora: Option<(PathBuf, f64)>,
    /// ID-LoRA: a clip of the speaker's voice. The clip's speech (the words in
    /// `transcript`) is generated with the picture, in this voice: its latent
    /// leads the audio as clean tokens at negative times.
    pub reference_voice: Option<PathBuf>,
    /// Lip-synced speech with ID-LoRA: the line (the speech made first, or the
    /// given `audio_file`) sets the clip's length and, without a
    /// `reference_voice`, is the voice too. The clip's own speech is generated
    /// with the picture, so the line is not followed as a soundtrack.
    pub identity: bool,
    /// ID-LoRA's identity guidance: each step also runs without the reference
    /// voice, and the clip's audio moves away from that result, toward the
    /// voice by `identity_guidance` times the difference (ID-LoRA uses 3 with
    /// the dev model). Off by default: on the distilled models it crackles,
    /// and the reference voice alone carries the speaker.
    pub identity_guidance: f64,
    /// The H.264 CRF the start and end images are re-compressed at (see
    /// `image_crf`); the model generation's default when not given.
    pub image_crf: Option<u32>,
    pub output: PathBuf,
    pub prompt: String,
    pub image: Option<PathBuf>,
    pub end_image: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: usize,
    pub seed: u64,
    pub device: usize,
    pub memory: String,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub ffmpeg: PathBuf,
    /// The text encoder, transformer, video decoder and image encoder on WebGPU (`backend` "webgpu": any GPU wgpu
    /// reaches, Vulkan, Metal or DX12): LTX 2.3's and 2.5's text- and image-to-video, the picture only, for now.
    pub webgpu: bool,
}
impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| {
            j.get(k)
                .and_then(Json::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("video: missing {k}"))
        };
        let n = |k: &str, default: i64| -> std::result::Result<usize, String> {
            let v = match j.get(k) {
                None => default,
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| format!("{k} must be an integer"))?,
            };
            usize::try_from(v).map_err(|_| format!("{k} must be nonnegative"))
        };
        let webgpu = match j.get("backend").and_then(Json::as_str) {
            None | Some("cuda" | "cpu") => false,
            Some("webgpu") if cfg!(feature = "webgpu") => true,
            Some("webgpu") => return Err("this build has no WebGPU (the webgpu feature)".into()),
            Some(other) => return Err(format!("backend must be cuda, cpu or webgpu, not {other}")),
        };
        let r = Self {
            model: s("model")?,
            transformer: s("transformer")?.into(),
            text_encoder: s("text_encoder")?.into(),
            tokenizer: j.get("tokenizer").and_then(Json::as_str).map(PathBuf::from),
            vae: s("vae")?.into(),
            audio_vae: j
                .get("audio_vae")
                .and_then(Json::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(PathBuf::from),
            // Audio comes with every clip whose model has an audio VAE, unless
            // the request says `"audio": false` (or runs on WebGPU, which has
            // no audio stream yet). A given soundtrack needs it.
            audio: match j.get("audio") {
                None | Some(Json::Null) => {
                    !webgpu && j.get("audio_vae").and_then(Json::as_str).is_some_and(|s| !s.trim().is_empty())
                        || j.get("audio_file").is_some_and(|v| !matches!(v, Json::Null))
                        || j.get("speech").is_some_and(|v| !matches!(v, Json::Null))
                }
                Some(v) => v.as_bool().ok_or("audio must be true or false")?,
            },
            audio_file: match j.get("audio_file") {
                None | Some(Json::Null) => None,
                Some(v) => Some(v.as_str().filter(|s| !s.trim().is_empty()).ok_or("audio_file must be an absolute local path")?.into()),
            },
            transcript: j.get("transcript").and_then(Json::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_owned),
            speech: match j.get("speech") {
                None | Some(Json::Null) => None,
                Some(v @ Json::Obj(_)) => Some(v.clone()),
                Some(_) => return Err("speech must be a speech request object".into()),
            },
            // The reference's 3 with guided (dev) sampling; off for distilled models.
            a2v_guidance: j.get("a2v_guidance").and_then(Json::as_f64).unwrap_or(if j.get("guidance").is_some_and(|g| !matches!(g, Json::Null)) { 3.0 } else { 1.0 }),
            guidance: match j.get("guidance") {
                None | Some(Json::Null) => None,
                Some(g) => Some(Guidance::parse(g)?),
            },
            negative_prompt: j.get("negative_prompt").and_then(Json::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_owned),
            nag: {
                let n = |k: &str, default: f64, lo: f64, hi: f64| -> std::result::Result<f64, String> {
                    match j.get("nag").and_then(|g| g.get(k)) {
                        None | Some(Json::Null) => Ok(default),
                        Some(v) => v.as_f64().filter(|v| (lo..=hi).contains(v)).ok_or_else(|| format!("nag.{k} must be a number from {lo} to {hi}")),
                    }
                };
                (n("scale", NAG_SCALE, 1., 20.)?, n("tau", NAG_TAU, 1., 10.)?, n("alpha", NAG_ALPHA, 0., 1.)?)
            },
            lora: match j.get("lora") {
                None | Some(Json::Null) => None,
                Some(Json::Str(p)) => Some((PathBuf::from(p), 1.0)),
                Some(l) => Some((
                    l.get("path").and_then(Json::as_str).filter(|s| !s.trim().is_empty()).map(PathBuf::from).ok_or("lora.path must be a local path")?,
                    l.get("strength").and_then(Json::as_f64).unwrap_or(1.0),
                )),
            },
            identity_guidance: j.get("identity_guidance").and_then(Json::as_f64).unwrap_or(0.0),
            image_crf: j.get("image_crf").and_then(Json::as_i64).map(|v| v.clamp(0, 51) as u32),
            identity: j.get("identity").and_then(Json::as_bool).unwrap_or(false),
            soundtrack_mode: match j.get("soundtrack_mode").and_then(Json::as_str).unwrap_or("inpaint") {
                m @ ("frozen" | "inpaint") => m.to_owned(),
                _ => return Err("soundtrack_mode must be frozen or inpaint".into()),
            },
            reference_voice: match j.get("reference_voice") {
                None | Some(Json::Null) => None,
                Some(v) => Some(v.as_str().filter(|s| !s.trim().is_empty()).ok_or("reference_voice must be a local path")?.into()),
            },
            refine: match j.get("refine") {
                None | Some(Json::Null) => None,
                Some(f) => {
                    let path = |k: &str| {
                        f.get(k)
                            .and_then(Json::as_str)
                            .filter(|s| !s.trim().is_empty())
                            .map(PathBuf::from)
                            .ok_or_else(|| format!("refine.{k} must be a local path"))
                    };
                    Some(Refine { transformer: path("transformer")?, upsampler: path("upsampler")? })
                }
            },
            frames_from_audio: j.get("frames").is_none() && (j.get("audio_file").is_some_and(|v| !matches!(v, Json::Null)) || j.get("speech").is_some_and(|v| !matches!(v, Json::Null))),
            output: s("output_dir")?.into(),
            prompt: s("prompt")?,
            cache_dir: j.get("cache_dir").and_then(Json::as_str).map(PathBuf::from),
            image: match j.get("image") {
                None | Some(Json::Null) => None,
                Some(v) => Some(
                    v.as_str()
                        .filter(|s| !s.trim().is_empty())
                        .ok_or("image must be an absolute local path")?
                        .into(),
                ),
            },
            end_image: match j.get("end_image") {
                None | Some(Json::Null) => None,
                Some(v) => Some(
                    v.as_str()
                        .filter(|s| !s.trim().is_empty())
                        .ok_or("end_image must be an absolute local path")?
                        .into(),
                ),
            },
            width: n("width", 512)?,
            height: n("height", 320)?,
            frames: n("frames", if j.get("frames").is_none() && (j.get("audio_file").is_some() || j.get("speech").is_some()) { 121 } else { 49 })?,
            fps: n("fps", 24)?,
            seed: n("seed", 0)? as u64,
            device: n("device", 0)?,
            memory: match j.get("memory") {
                None => "auto",
                Some(v) => v.as_str().ok_or("memory must be a string")?,
            }
            .into(),
            ram_bytes: (n("ram_gb", 48)? as u64)
                .checked_mul(GIB)
                .ok_or("ram_gb overflow")?,
            vram_bytes: (n("vram_gb", 26)? as u64)
                .checked_mul(GIB)
                .ok_or("vram_gb overflow")?,
            ffmpeg: j
                .get("ffmpeg")
                .and_then(Json::as_str)
                .unwrap_or("ffmpeg")
                .into(),
            webgpu,
        };
        r.validate()?;
        if let Some(v) = j.get("steps") {
            if v.as_i64() != Some(8) {
                return Err("LTX distilled models use their trained 8-step schedule".into());
            }
        }
        if j.get("n").is_some_and(|v| v.as_i64() != Some(1)) || j.get("prompts").is_some() {
            return Err("video requests generate one prompt at a time".into());
        }
        for k in ["images", "adapter"] {
            if j.get(k).is_some_and(|v| !matches!(v, Json::Null)) {
                return Err(format!(
                    "{k} is not supported by the video worker; use image for one starting frame"
                ));
            }
        }
        Ok(r)
    }
    pub fn validate(&self) -> std::result::Result<(), String> {
        if !["ltx-2.3", "ltx-2.5", "sulphur-2"].contains(&self.model.as_str()) {
            return Err("model must be ltx-2.3, ltx-2.5 or sulphur-2".into());
        }
        if !["auto", "gpu", "ram", "ssd"].contains(&self.memory.as_str()) {
            return Err("memory must be auto, gpu, ram or ssd".into());
        }
        if [self.width, self.height]
            .iter()
            .any(|n| !(128..=1024).contains(n) || n % 32 != 0)
        {
            return Err("video dimensions must be multiples of 32 in 128..1024".into());
        }
        if !(9..=121).contains(&self.frames) || (self.frames - 1) % 8 != 0 {
            return Err("frames must be 8k+1 in 9..121".into());
        }
        if !(1..=60).contains(&self.fps) || self.prompt.len() > 16384 {
            return Err("fps must be 1..60 and prompt at most 16384 bytes".into());
        }
        if self.ram_bytes > 512 * GIB || self.vram_bytes > 192 * GIB {
            return Err("weight budgets exceed supported limits".into());
        }
        if self.model != "ltx-2.5" && self.tokenizer.is_none() {
            return Err("Gemma 3 tokenizer is required".into());
        }
        if self.audio && self.audio_vae.is_none() {
            return Err("audio needs the model's audio VAE (audio_vae)".into());
        }
        if (self.audio_file.is_some() || self.speech.is_some()) && !self.audio {
            return Err("a soundtrack to follow needs the model's audio stream; leave audio on".into());
        }
        if !(1.0..=10.0).contains(&self.a2v_guidance) {
            return Err("a2v_guidance must be between 1 and 10".into());
        }
        if self.audio_file.is_some() && self.speech.is_some() {
            return Err("give audio_file or speech, not both".into());
        }
        if self.transcript.as_ref().is_some_and(|t| t.len() > 4000) {
            return Err("transcript is limited to 4000 bytes".into());
        }
        if let Some(path) = &self.audio_file {
            let meta = std::fs::metadata(path).map_err(|e| format!("video soundtrack: {e}"))?;
            if !path.is_absolute() || !meta.is_file() || meta.len() > 256 * 1024 * 1024 {
                return Err("audio_file must be an absolute local file of at most 256 MiB".into());
            }
        }
        if self.prompt.trim().is_empty() {
            return Err("prompt must not be empty".into());
        }
        if self.webgpu {
            let unsupported = [
                (self.audio || self.audio_file.is_some() || self.speech.is_some() || self.reference_voice.is_some() || self.identity, "sound (set audio to false)"),
                (self.lora.is_some(), "LoRAs"),
                (self.refine.is_some(), "two-stage refinement"),
            ];
            if let Some((_, what)) = unsupported.iter().find(|(on, _)| *on) {
                return Err(format!("WebGPU video does not support {what} yet"));
            }
        }
        for path in self.image.iter().chain(self.end_image.iter()) {
            let meta = std::fs::metadata(path).map_err(|e| format!("video endpoint image: {e}"))?;
            if !path.is_absolute() || !meta.is_file() || meta.len() > 32 * 1024 * 1024 {
                return Err("image must be an absolute local file of at most 32 MiB".into());
            }
        }
        Ok(())
    }
}
fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([
        ("stage", Json::str(stage)),
        ("current", Json::Int(current as i64)),
        ("total", Json::Int(total as i64)),
    ])
}
pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    r.validate().map_err(candle_core::Error::Msg)?;
    let started = Instant::now();
    report(event("preparing_video", 0, 1));
    // Validate the encoder before spending minutes on inference.
    let status = command(&r.ffmpeg)
        .args(["-version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if !status.success() {
        candle_core::bail!("FFmpeg is required to write the video container");
    }
    std::fs::create_dir_all(&r.output)?;
    // The soundtrack to follow: speech made first (its model is freed before
    // the video's loads), or a file; read as stereo at its own rate.
    let mut owned = r.clone();
    // The words, in the prompt as LTX prompts carry dialogue: without them the
    // picture barely takes lip movement from a soundtrack alone.
    let words = r.speech.as_ref().and_then(|s| s.get("text")).and_then(Json::as_str).map(str::trim).map(str::to_owned).or_else(|| r.transcript.clone());
    if r.reference_voice.is_some() || r.identity {
        // ID-LoRA's prompt: what is seen, what is said, what is heard.
        if !r.prompt.contains("[SPEECH]") {
            // Quoted words, and the voice speaking to the viewer with no music
            // ("unscored"), as the LTX 2.5 IC-LoRA template writes it.
            let said = words.as_deref().filter(|w| !w.is_empty()).map(|w| format!(" [SPEECH]: \"{}\"", w.trim().replace('"', "'"))).unwrap_or_default();
            owned.prompt = format!("[VISUAL]: {}{said} [SOUNDS]: Clear speech, spoken directly to the viewer. Unscored.", r.prompt.trim());
        }
    } else if let Some(w) = words.filter(|w| !w.is_empty() && !r.prompt.contains(w.as_str())) {
        owned.prompt = format!("{} They say: \"{}\"", r.prompt.trim_end(), w.replace('"', "'"));
    }
    let speech_started = Instant::now();
    if let Some(s) = &r.speech {
        let tts = crate::tts::Request::parse(s).map_err(candle_core::Error::Msg)?;
        let spoken = crate::tts::generate(&tts, &mut report)?;
        let path = spoken.get("path").and_then(Json::as_str).ok_or_else(|| candle_core::Error::Msg("speech wrote no audio".into()))?;
        owned.audio_file = Some(path.into());
        owned.speech = None;
    }
    let speech_seconds = speech_started.elapsed().as_secs_f64();
    let soundtrack_in = match &owned.audio_file {
        Some(path) => {
            report(event("reading_soundtrack", 0, 1));
            let (channels, rate) = read_audio(&r.ffmpeg, path)?;
            if owned.frames_from_audio {
                owned.frames = frames_for_audio(channels[0].len() as f64 / rate as f64, r.fps);
            }
            Some((channels, rate))
        }
        None => None,
    };
    // Lip-synced speech: the line has set the length; it is the voice unless a
    // saved voice's sample is given, and the clip's speech is generated.
    let voice_source = if owned.identity {
        owned.reference_voice.clone().or_else(|| owned.audio_file.clone())
    } else {
        owned.reference_voice.clone()
    };
    let soundtrack_in = if owned.identity { None } else { soundtrack_in };
    owned.validate().map_err(candle_core::Error::Msg)?;
    let r = &owned;
    // (a WebGPU clip's latent and guidance stay on the host: small beside its GPU's work)
    let dev = if r.webgpu { Device::Cpu } else { inference_device(r.device)? };
    let ram = if r.memory == "ssd" || r.memory == "gpu" {
        0
    } else {
        r.ram_bytes
    };
    let mut store = Store::open(&r.transformer, ram)?;
    if let Some((path, strength)) = &r.lora {
        let n = store.add_lora(path, *strength)?;
        report(event("lora_applied", n, n));
    }
    let expected_version = if r.model == "ltx-2.5" { "2.5." } else { "2.3." };
    if !store
        .index
        .metadata("model_version")
        .is_some_and(|v| v.starts_with(expected_version))
    {
        candle_core::bail!("{} requires an LTX {expected_version} checkpoint", r.model);
    }
    let config = Json::parse(
        store
            .index
            .metadata("config")
            .ok_or_else(|| {
                candle_core::Error::Msg("LTX checkpoint lacks architecture metadata".into())
            })?
            .as_bytes(),
    )
    .map_err(candle_core::Error::wrap)?;
    let architecture = config.get("transformer").ok_or_else(|| {
        candle_core::Error::Msg("LTX checkpoint lacks transformer configuration".into())
    })?;
    for (key, expected) in [
        ("num_layers", 48),
        ("num_attention_heads", 32),
        ("attention_head_dim", 128),
        ("connector_num_layers", 8),
        ("connector_num_attention_heads", 32),
        ("connector_attention_head_dim", 128),
    ] {
        if architecture.get(key).and_then(Json::as_i64) != Some(expected) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    for key in [
        "cross_attention_adaln",
        "apply_gated_attention",
        "use_middle_indices_grid",
    ] {
        if architecture.get(key).and_then(Json::as_bool) != Some(true) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    for (key, expected) in [("rope_type", "split"), ("frequencies_precision", "float64")] {
        if architecture.get(key).and_then(Json::as_str) != Some(expected) {
            candle_core::bail!("unsupported LTX configuration: {key}");
        }
    }
    if r.audio {
        for (key, expected) in [
            ("audio_num_attention_heads", 32),
            ("audio_attention_head_dim", 64),
            ("audio_cross_attention_dim", 2048),
            ("audio_connector_num_attention_heads", 32),
            ("audio_connector_attention_head_dim", 64),
        ] {
            if architecture.get(key).and_then(Json::as_i64) != Some(expected) {
                candle_core::bail!("unsupported LTX audio configuration: {key}");
            }
        }
        // Both timestep multipliers are 1000 in every LTX 2.x release; the
        // cross-attention gates depend on their ratio.
        for key in ["timestep_scale_multiplier", "av_ca_timestep_scale_multiplier"] {
            if architecture.get(key).and_then(Json::as_f64).is_some_and(|v| v != 1000.) {
                candle_core::bail!("unsupported LTX audio configuration: {key}");
            }
        }
    }
    let audio_shapes: &[(&str, Vec<usize>)] = if r.audio {
        &[
            ("audio_patchify_proj.weight", vec![2048, 128]),
            ("transformer_blocks.47.audio_scale_shift_table", vec![9, 2048]),
            ("transformer_blocks.47.scale_shift_table_a2v_ca_video", vec![5, 4096]),
        ]
    } else {
        &[]
    };
    for (key, shape) in [
        ("patchify_proj.weight", vec![4096, 128]),
        ("transformer_blocks.47.scale_shift_table", vec![9, 4096]),
    ]
    .iter()
    .chain(audio_shapes)
    {
        let name = format!("{}{key}", transformer::PREFIX);
        let info = store.index.info(&name).map_err(candle_core::Error::wrap)?;
        if &info.shape != shape {
            candle_core::bail!("unsupported LTX architecture at {name}: {:?}", info.shape);
        }
    }
    let image_started = Instant::now();
    // (a WebGPU clip's images encoded on the CPU, whose matmuls take F32, not BF16)
    let vae_dtype = if r.webgpu { DType::F32 } else { DType::BF16 };
    // Two stages: the first renders at half the size.
    let two_stage = r.guidance.is_some() && r.refine.is_some();
    let stage_size = if two_stage { (r.width / 2, r.height / 2) } else { (r.width, r.height) };
    let encode_endpoints = |size: (usize, usize), report: &mut dyn FnMut(Json)| -> Result<(Option<Tensor>, Option<Tensor>)> {
        if r.image.is_none() && r.end_image.is_none() {
            return Ok((None, None));
        }
        let encoder = vae::LtxVideoEncoder::load(&r.vae, vae::LtxVaeConfig::ltx_2_3_22b(), &dev, vae_dtype)?;
        let mut encode = |path: &Option<PathBuf>, stage: &str| -> Result<Option<Tensor>> {
            path.as_ref()
                .map(|path| {
                    report(event(stage, 0, 1));
                    encode_image(path, size, &encoder, &dev, vae_dtype, &r.ffmpeg, image_crf(r))
                })
                .transpose()
        };
        let start = encode(&r.image, "encoding_starting_image")?;
        let end = encode(&r.end_image, "encoding_ending_image")?;
        drop(encoder);
        dev.synchronize()?;
        Ok((start, end))
    };
    let (starting_latent, ending_latent) = if r.webgpu && (r.image.is_some() || r.end_image.is_some()) {
        webgpu_endpoints(r, stage_size, &dev, &mut report)?
    } else if r.image.is_some() || r.end_image.is_some() {
        let encoder = vae::LtxVideoEncoder::load(
            &r.vae,
            vae::LtxVaeConfig::ltx_2_3_22b(),
            &dev,
            vae_dtype,
        )?;
        let mut encode = |path: &Option<PathBuf>, stage: &str| -> Result<Option<Tensor>> {
            path.as_ref()
                .map(|path| {
                    report(event(stage, 0, 1));
                    encode_image(path, stage_size, &encoder, &dev, vae_dtype, &r.ffmpeg, image_crf(r))
                })
                .transpose()
        };
        let start = encode(&r.image, "encoding_starting_image")?;
        let end = encode(&r.end_image, "encoding_ending_image")?;
        drop(encoder);
        dev.synchronize()?;
        (start, end)
    } else {
        (None, None)
    };
    let image_seconds = image_started.elapsed().as_secs_f64();
    // The given soundtrack as the transformer's audio latent, held fixed.
    let audio_frames = audio_latent_frames(r.frames, r.fps);
    let mut reference_loudness = None;
    // ID-LoRA's reference voice: its whole latent, clean, before the clip.
    let reference_voice = match (&voice_source, &r.audio_vae) {
        (Some(path), Some(vae)) => {
            report(event("encoding_reference_voice", 0, 1));
            let (channels, rate) = read_audio(&r.ffmpeg, path)?;
            reference_loudness = speech_loudness(&channels[0], rate);
            let seconds = (channels[0].len() as f64 / rate as f64).min(MAX_REFERENCE_SECONDS);
            let frames = (seconds * audio::LATENT_RATE).round().max(1.) as usize;
            let encoder = audio::AudioEncoder::load(vae, &dev)?;
            let latent = encoder.latent(&channels, rate, frames, &dev)?;
            drop(encoder);
            Some(latent)
        }
        (Some(_), None) => candle_core::bail!("a reference voice needs the model's audio VAE"),
        _ => None,
    };
    let ref_tokens = reference_voice.as_ref().map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let conditioning_audio = match (&soundtrack_in, &r.audio_vae) {
        (Some((channels, rate)), Some(path)) => {
            report(event("encoding_soundtrack", 0, 1));
            let encoder = audio::AudioEncoder::load(path, &dev)?;
            let latent = encoder.latent(channels, *rate, audio_frames, &dev)?;
            drop(encoder);
            Some(latent)
        }
        _ => None,
    };
    let encode_started = Instant::now();
    let prompt_cache = cache::PromptCache::new(r);
    let audio_cache = if r.audio { cache::PromptCache::audio(r) } else { None };
    let cached = prompt_cache.as_ref().and_then(|c| c.load(&dev));
    let cached_audio = audio_cache.as_ref().and_then(|c| c.load(&dev));
    let prompt_cache_hit = cached.is_some() && (!r.audio || cached_audio.is_some());
    let mut text_encoder_seconds = 0.;
    let mut connector_seconds = 0.;
    // The negative prompt for guided sampling: video only (the audio stream
    // keeps the prompt's context in that pass, as the reference does).
    let negative_text = match (&r.guidance, &r.negative_prompt) {
        // Guided: the request's negative prompt, else the guidance's (or the reference's).
        (Some(g), n) if g.cfg != 1. => Some(n.clone().unwrap_or_else(|| g.negative_prompt.clone())),
        // Unguided: NAG, when there is one.
        (_, Some(n)) => Some(n.clone()),
        _ => None,
    };
    let (context, audio_context, negative) = if r.webgpu {
        let (context, negative) = webgpu_contexts(r, &mut store, cached, negative_text, &mut report)?;
        text_encoder_seconds = encode_started.elapsed().as_secs_f64();
        (context, None, negative)
    } else {
    let (context, audio_context) = if prompt_cache_hit {
        report(event("cached_video_prompt", 1, 1));
        (cached.unwrap_or_else(|| unreachable!()), cached_audio)
    } else {
        report(event("encoding_video_prompt", 0, 48));
        let (features, audio_features) = text::encode(
            &r.text_encoder,
            r.tokenizer.as_deref(),
            &mut store,
            &r.prompt,
            r.model == "ltx-2.5",
            r.audio,
            &dev,
            |n| report(event("encoding_video_prompt", n, 48)),
        )?;
        dev.synchronize()?;
        text_encoder_seconds = encode_started.elapsed().as_secs_f64();
        let blocks = if r.audio { 16 } else { 8 };
        report(event("video_text_connector", 0, blocks));
        let connector_started = Instant::now();
        let context = transformer::connector(&mut store, &features, "video_embeddings_connector", &dev, |n| {
            report(event("video_text_connector", n, blocks))
        })?;
        drop(features);
        let audio_context = match audio_features {
            Some(f) => Some(transformer::connector(&mut store, &f, "audio_embeddings_connector", &dev, |n| {
                report(event("video_text_connector", 8 + n, blocks))
            })?),
            None => None,
        };
        dev.synchronize()?;
        connector_seconds = connector_started.elapsed().as_secs_f64();
        if let Some(cache) = &prompt_cache {
            let _ = cache.save(&context);
        }
        if let (Some(cache), Some(a)) = (&audio_cache, &audio_context) {
            let _ = cache.save(a);
        }
        (context, audio_context)
    };
    let negative = match negative_text {
        Some(text) => {
            let neg = Request { prompt: text, audio: false, ..r.clone() };
            let neg_cache = cache::PromptCache::new(&neg);
            match neg_cache.as_ref().and_then(|c| c.load(&dev)) {
                Some(c) => Some(c),
                None => {
                    report(event("encoding_negative_prompt", 0, 48));
                    let (features, _) = text::encode(
                        &r.text_encoder,
                        r.tokenizer.as_deref(),
                        &mut store,
                        &neg.prompt,
                        r.model == "ltx-2.5",
                        false,
                        &dev,
                        |n| report(event("encoding_negative_prompt", n, 48)),
                    )?;
                    let c = transformer::connector(&mut store, &features, "video_embeddings_connector", &dev, |_| {})?;
                    if let Some(cache) = &neg_cache {
                        let _ = cache.save(&c);
                    }
                    Some(c)
                }
            }
        }
        _ => None,
    };
    (context, audio_context, negative)
    };
    let text_seconds = encode_started.elapsed().as_secs_f64();
    let gpu_budget = if r.memory == "ram" || r.memory == "ssd" {
        0
    } else {
        r.vram_bytes
    };
    #[cfg(feature = "cuda")]
    let gpu_budget = if r.webgpu {
        gpu_budget
    } else {
        let free = dev
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(candle_core::Error::wrap)?
            .0 as u64;
        // Leave room for global projections, activations and a streamed block,
        // with a margin: on Windows a card filled to the brim pages device
        // memory to system RAM, where kernels crawl into the driver's watchdog.
        gpu_budget.min(free.saturating_sub(8 * GIB))
    };
    let (f, mut h, mut w) = ((r.frames - 1) / 8 + 1, stage_size.1 / 32, stage_size.0 / 32);
    let mut model = if r.webgpu {
        webgpu_model(r, &mut store, f, h, w, &mut report)?
    } else {
        Model::Candle(transformer::Transformer::new(store, &dev, gpu_budget, r.memory == "gpu", r.audio)?)
    };
    // A negative prompt without CFG steers every step by NAG.
    if let (true, Some(context), Model::Candle(model)) = (r.guidance.as_ref().is_none_or(|g| g.cfg == 1.), &negative, &mut model) {
        model.nag = Some(transformer::Nag { context: context.clone(), scale: r.nag.0, tau: r.nag.1, alpha: r.nag.2 });
        report(event("negative_prompt_nag", 1, 1));
    }
    #[cfg(feature = "webgpu")]
    if let (true, Some(context), Model::Wgpu(model, _)) = (r.guidance.as_ref().is_none_or(|g| g.cfg == 1.), &negative, &mut model) {
        let values = context.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        model.nag = Some(crate::ltx_wgpu::Nag { rows: values.len() / 4096, context: values, scale: r.nag.0 as f32, tau: r.nag.1 as f32, alpha: r.nag.2 as f32 });
        report(event("negative_prompt_nag", 1, 1));
    }
    let int8_weights = matches!(&model, Model::Candle(m) if m.int8);
    if int8_weights {
        report(event("int8_weights", 1, 1));
    }
    let rope = transformer::Rope::video_with_end(f, h, w, r.fps, ending_latent.is_some(), &dev)?;
    let noise = crate::pipeline::noise(r.seed, f * h * w * 128);
    let mut latent = Tensor::from_vec(noise, (1, f * h * w, 128), &dev)?;
    if let Some(end) = &ending_latent {
        latent = Tensor::cat(&[&latent, end], 1)?;
    }
    latent = condition_endpoints(latent, starting_latent.as_ref(), ending_latent.as_ref())?;
    // The soundtrack: its own latent, denoised jointly on the same schedule,
    // or the given one, frozen (sigma 0) while the picture follows it.
    let audio_ropes = if r.audio {
        // A reference voice sits before the clip: its spans end one latent
        // (40 ms) before zero, as ID-LoRA was trained.
        let mut spans = transformer::audio_spans(ref_tokens);
        let shift = spans.last().map_or(0., |s| s.1) + 4. * 0.01;
        spans.iter_mut().for_each(|s| *s = (s.0 - shift, s.1 - shift));
        spans.extend(transformer::audio_spans(audio_frames));
        Some((
            transformer::Rope::audio_spans(&spans, &dev)?,
            transformer::Rope::video_cross(f, h, w, r.fps, ending_latent.is_some(), &dev)?,
        ))
    } else {
        None
    };
    let identity = if ref_tokens > 0 && r.identity_guidance > 0. {
        Some(transformer::Rope::audio(audio_frames, &dev)?)
    } else {
        None
    };
    let frozen_audio = conditioning_audio.is_some();
    // Inpainting: one fixed noise the soundtrack is mixed with at each step.
    let inpaint_noise = if frozen_audio && r.soundtrack_mode == "inpaint" {
        let noise = crate::pipeline::noise(r.seed.wrapping_add(AUDIO_SEED_OFFSET), audio_frames * 128);
        Some(Tensor::from_vec(noise, (1, audio_frames, 128), &dev)?)
    } else {
        None
    };
    let mut audio_latent = if let Some(latent) = conditioning_audio {
        Some(latent)
    } else if r.audio {
        let noise = crate::pipeline::noise(r.seed.wrapping_add(AUDIO_SEED_OFFSET), audio_frames * 128);
        let noise = Tensor::from_vec(noise, (1, audio_frames, 128), &dev)?;
        Some(match &reference_voice {
            Some(voice) => Tensor::cat(&[voice, &noise], 1)?,
            None => noise,
        })
    } else {
        None
    };
    let denoise_started = Instant::now();
    let sigmas = match &r.guidance {
        Some(g) => guided_sigmas(g.steps),
        None => SIGMAS.to_vec(),
    };
    let steps = sigmas.len() - 1;
    // Following a soundtrack, each step also runs with the audio-video
    // cross-attention skipped, and the picture moves away from that result.
    // Also with a reference voice: ID-LoRA's "bimodal" guidance (asked for).
    let modality = if frozen_audio || ref_tokens > 0 { r.a2v_guidance } else { 1. };
    let cfg = r.guidance.as_ref().map_or(1., |g| g.cfg);
    let stg = r.guidance.as_ref().map_or(0., |g| g.stg);
    let passes = 1 + usize::from(modality > 1.) + usize::from(negative.is_some() && cfg != 1.) + usize::from(stg != 0.) + usize::from(identity.is_some());
    for step in 0..steps {
        let sigma = sigmas[step];
        let audio_bf16 = match (&audio_latent, &inpaint_noise) {
            // The soundtrack noised to this step's sigma, like a generated one.
            (Some(clean), Some(noise)) => Some(((clean * (1. - sigma))? + (noise * sigma)?)?.to_dtype(DType::BF16)?),
            _ => audio_latent.as_ref().map(|a| a.to_dtype(DType::BF16)).transpose()?,
        };
        let video_bf16 = latent.to_dtype(DType::BF16)?;
        // (the WebGPU stream takes the latent's f32 values)
        #[cfg(feature = "webgpu")]
        let video_f32 = if r.webgpu { Some(latent.flatten_all()?.to_vec1::<f32>()?) } else { None };
        let mut pass = 0;
        // Without the reference voice: only the clip's own audio tokens.
        let target_bf16 = match (&audio_bf16, &identity) {
            (Some(a), Some(_)) => Some(a.narrow(1, ref_tokens, audio_frames)?),
            _ => None,
        };
        let mut run = |context: &Tensor, isolated: bool, perturb: Option<usize>, without_reference: bool| {
            let audio_input = match (&audio_bf16, &audio_context, &audio_ropes) {
                (Some(latent), Some(context), Some((rope, video_cross))) => {
                    let (latent, rope, clean_tokens) = match (without_reference, &target_bf16, &identity) {
                        (true, Some(target), Some(rope)) => (target, rope, 0),
                        _ => (latent, rope, ref_tokens),
                    };
                    Some(transformer::AudioInput {
                        latent,
                        context,
                        rope,
                        video_cross,
                        sigma: if frozen_audio && inpaint_noise.is_none() { 0. } else { sigma },
                        isolated,
                        clean_tokens,
                    })
                }
                _ => None,
            };
            let at = pass;
            pass += 1;
            match &mut model {
                Model::Candle(model) => {
                    model.skip_video_self_attn = perturb;
                    let out = model.forward(
                        &video_bf16,
                        context,
                        sigma,
                        &rope,
                        if starting_latent.is_some() { h * w } else { 0 },
                        if ending_latent.is_some() { h * w } else { 0 },
                        audio_input,
                        |n| report(event("video_denoising", (step * passes + at) * 48 + n, steps * passes * 48)),
                    );
                    model.skip_video_self_attn = None;
                    out
                }
                #[cfg(feature = "webgpu")]
                Model::Wgpu(m, table) => {
                    let latent = video_f32.as_deref().ok_or_else(|| candle_core::Error::Msg("the latent's values".into()))?;
                    let values = context.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let tokens = latent.len() / 128;
                    let clean = (if starting_latent.is_some() { h * w } else { 0 }, if ending_latent.is_some() { h * w } else { 0 });
                    let v = m.forward(latent, tokens, &values, values.len() / 4096, sigma, table, clean, h * w, perturb)?;
                    report(event("video_denoising", (step * passes + at + 1) * 48, steps * passes * 48));
                    Ok((Tensor::from_vec(v, (1, tokens, 128), &dev)?, None))
                }
            }
        };
        let (mut velocity, mut audio_velocity) = run(&context, false, None, false)?;
        // Identity guidance: the clip's audio away from what it would be
        // without the reference voice (ID-LoRA's, on the audio stream only).
        if identity.is_some() {
            let (_, without) = run(&context, false, None, true)?;
            if let (Some(with), Some(without)) = (audio_velocity.as_ref(), without) {
                let with = with.to_dtype(DType::F32)?;
                let target = with.narrow(1, ref_tokens, audio_frames)?;
                let guided = (&target + ((&target - without.to_dtype(DType::F32)?)? * r.identity_guidance)?)?;
                // Rescaled as the reference's guider does (0.7), on the denoised
                // audio: the guidance alone inflates its level until it clips.
                let a = audio_latent.as_ref().ok_or_else(|| candle_core::Error::Msg("audio latent".into()))?.narrow(1, ref_tokens, audio_frames)?;
                let x0 = |v: &Tensor| -> Result<Tensor> { &a - (v * sigma)? };
                let (plain, pushed) = (x0(&target)?, x0(&guided)?);
                let factor = 0.7 * std_all(&plain)? / std_all(&pushed)?.max(1e-12) + 0.3;
                let guided = ((&a - (pushed * factor)?)? / sigma)?;
                audio_velocity = Some(Tensor::cat(&[&with.narrow(1, 0, ref_tokens)?, &guided], 1)?);
            }
        }
        if let Some(g) = &r.guidance {
            // The reference guides the denoised prediction, x0 = x - sigma v;
            // its rescale needs that space.
            let x0 = |v: &Tensor| -> Result<Tensor> { &latent - (v.to_dtype(DType::F32)? * sigma)? };
            let cond = x0(&velocity)?;
            let mut pred = cond.clone();
            if let Some(negative) = negative.as_ref().filter(|_| g.cfg != 1.) {
                let (uncond, _) = run(negative, false, None, false)?;
                pred = (pred + ((&cond - x0(&uncond)?)? * (g.cfg - 1.))?)?;
            }
            if g.stg != 0. {
                let (perturbed, _) = run(&context, false, Some(g.stg_block), false)?;
                pred = (pred + ((&cond - x0(&perturbed)?)? * g.stg)?)?;
            }
            if modality > 1. {
                let (isolated, _) = run(&context, true, None, false)?;
                pred = (pred + ((&cond - x0(&isolated)?)? * (modality - 1.))?)?;
            }
            if g.rescale != 0. {
                let factor = std_all(&cond)? / std_all(&pred)?.max(1e-12);
                pred = (pred * (g.rescale * factor + (1. - g.rescale)))?;
            }
            velocity = ((&latent - pred)? / sigma)?;
        } else if modality > 1. {
            let (isolated, _) = run(&context, true, None, false)?;
            let cond = velocity.to_dtype(DType::F32)?;
            let delta = (&cond - isolated.to_dtype(DType::F32)?)?;
            velocity = (cond + (delta * (modality - 1.0))?)?;
        }
        let dt = sigmas[step + 1] - sigma;
        latent = (latent + (velocity.to_dtype(DType::F32)? * dt)?)?;
        latent = condition_endpoints(latent, starting_latent.as_ref(), ending_latent.as_ref())?;
        if let (Some(a), Some(v), false) = (audio_latent.as_mut(), audio_velocity, frozen_audio) {
            *a = (&*a + (v.to_dtype(DType::F32)? * dt)?)?;
            // The reference voice stays as it was given.
            if let Some(voice) = &reference_voice {
                *a = Tensor::cat(&[voice, &a.narrow(1, ref_tokens, audio_frames)?], 1)?;
            }
        }
    }
    // Only the clip's own audio is decoded.
    if ref_tokens > 0 {
        audio_latent = audio_latent.map(|a| a.narrow(1, ref_tokens, audio_frames)).transpose()?;
    }
    // Appended end-keyframe tokens guide attention but are not part of the decoded clip.
    latent = latent.narrow(1, 0, f * h * w)?.contiguous()?;
    let (mut gpu, mut host, mut disk) = match &model {
        Model::Candle(m) => m.stats(),
        #[cfg(feature = "webgpu")]
        Model::Wgpu(..) => (0, 0, 0),
    };
    dev.synchronize()?;
    drop(model);
    if let (true, Some(refine)) = (two_stage, &r.refine) {
        // Double the latent: un-normalized through the upsampler, then back.
        report(event("upsampling_latent", 0, 1));
        let stats = vae::PerChannelStatistics::load_file(&r.vae, &dev)?;
        let up = upsampler::Upsampler::load(&refine.upsampler, &dev)?;
        let x = latent.reshape((1, f, h, w, 128))?.permute((0, 4, 1, 2, 3))?.contiguous()?;
        let x = stats.normalize(&up.forward(&stats.un_normalize(&x)?)?)?;
        drop(up);
        h *= 2;
        w *= 2;
        let upscaled = x.permute((0, 2, 3, 4, 1))?.contiguous()?.reshape((1, f * h * w, 128))?;
        // The endpoints again, at the full size.
        let (start, end) = encode_endpoints((r.width, r.height), &mut report)?;
        let rope = transformer::Rope::video_with_end(f, h, w, r.fps, end.is_some(), &dev)?;
        let video_cross = if r.audio { Some(transformer::Rope::video_cross(f, h, w, r.fps, end.is_some(), &dev)?) } else { None };
        let audio_rope = if r.audio { Some(transformer::Rope::audio(audio_frames, &dev)?) } else { None };
        // Re-noised to the first refine sigma (a fresh draw from the seed).
        let s0 = REFINE_SIGMAS[0];
        let noise = crate::pipeline::noise(r.seed.wrapping_add(REFINE_SEED_OFFSET), f * h * w * 128);
        let noise = Tensor::from_vec(noise, (1, f * h * w, 128), &dev)?;
        latent = ((upscaled * (1. - s0))? + (noise * s0)?)?;
        if let Some(end) = &end {
            latent = Tensor::cat(&[&latent, end], 1)?;
        }
        latent = condition_endpoints(latent, start.as_ref(), end.as_ref())?;
        report(event("loading_refine_model", 0, 1));
        let ram = if r.memory == "ssd" || r.memory == "gpu" { 0 } else { r.ram_bytes };
        let mut store = Store::open(&refine.transformer, ram)?;
        // The prompt through the refining transformer's own text connectors:
        // a fine-tune (Sulphur, or one of LTX 2.5) may have its own.
        let (context, audio_context) = refine_contexts(r, &refine.transformer, &mut store, &dev, &mut report)?;
        let budget = if r.memory == "ram" || r.memory == "ssd" { 0 } else { r.vram_bytes };
        #[cfg(feature = "cuda")]
        let budget = {
            let free = dev.as_cuda_device()?.cuda_stream().context().mem_get_info().map_err(candle_core::Error::wrap)?.0 as u64;
            budget.min(free.saturating_sub(8 * GIB))
        };
        let mut model = transformer::Transformer::new(store, &dev, budget, r.memory == "gpu", r.audio)?;
        let audio_bf16 = audio_latent.as_ref().map(|a| a.to_dtype(DType::BF16)).transpose()?;
        let refine_steps = REFINE_SIGMAS.len() - 1;
        for step in 0..refine_steps {
            let sigma = REFINE_SIGMAS[step];
            let audio_input = match (&audio_bf16, &audio_context, &audio_rope, &video_cross) {
                (Some(latent), Some(context), Some(rope), Some(video_cross)) => Some(transformer::AudioInput {
                    latent,
                    context,
                    rope,
                    video_cross,
                    sigma: 0.,
                    isolated: false,
                    clean_tokens: 0,
                }),
                _ => None,
            };
            let (velocity, _) = model.forward(
                &latent.to_dtype(DType::BF16)?,
                &context,
                sigma,
                &rope,
                if start.is_some() { h * w } else { 0 },
                if end.is_some() { h * w } else { 0 },
                audio_input,
                |n| report(event("refining_video", step * 48 + n, refine_steps * 48)),
            )?;
            latent = (latent + (velocity.to_dtype(DType::F32)? * (REFINE_SIGMAS[step + 1] - sigma))?)?;
            latent = condition_endpoints(latent, start.as_ref(), end.as_ref())?;
        }
        latent = latent.narrow(1, 0, f * h * w)?.contiguous()?;
        let (g2, h2, d2) = model.stats();
        (gpu, host, disk) = (gpu + g2, host + h2, disk + d2);
        dev.synchronize()?;
        drop(model);
    }
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    drop(context);
    drop(audio_context);
    drop(audio_ropes);
    drop(rope);
    dev.synchronize()?;
    report(event("decoding_video", 0, 1));
    let decode_started = Instant::now();
    let pixels = if r.webgpu {
        webgpu_decode(r, &latent, f, h, w)?
    } else {
    let decoder =
        vae::LtxVideoDecoder::load(&r.vae, vae::LtxVaeConfig::ltx_2_3_22b(), &dev, DType::BF16)?;
    let latent = latent
        .reshape((1, f, h, w, 128))?
        .permute((0, 4, 1, 2, 3))?
        .contiguous()?
        .to_dtype(DType::BF16)?;
    // Decode ordinary clips together, preserving their full spatial context.
    // Large clips use overlapping tiles sized for the convolution workspace.
    let pixel_budget = 32 * 1024 * 1024usize;
    dev.synchronize()?;
    #[cfg(feature = "cuda")]
    let pixel_budget = {
        let free = dev
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(candle_core::Error::wrap)?
            .0;
        pixel_budget.min(free.saturating_sub(4usize << 30) / 512)
    };
    let pixels = if r.frames * r.height * r.width <= pixel_budget {
        decoder.decode(&latent)?.to_device(&Device::Cpu)?
    } else {
        let mut tile = 16;
        while tile > 4 && tile * tile * 32 * 32 * r.frames > pixel_budget {
            tile /= 2;
        }
        if tile * tile * 32 * 32 * r.frames > pixel_budget {
            candle_core::bail!("insufficient free VRAM for video decoder workspace; reduce frames or free GPU memory");
        }
        decoder.decode_tiled(&latent, tile, tile, tile / 4, tile / 4)?
    };
    drop(decoder);
    pixels
    };
    drop(latent);
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    // The soundtrack: audio VAE decoder and vocoder, then trimmed (or padded)
    // to exactly the clip's length.
    let audio_started = Instant::now();
    let soundtrack = match (&audio_latent, &r.audio_vae) {
        // A given soundtrack is kept as it was (the reference returns the
        // input audio, not its VAE round trip), cut to the clip.
        _ if soundtrack_in.is_some() => {
            let (channels, rate) = soundtrack_in.as_ref().ok_or_else(|| candle_core::Error::Msg("soundtrack missing".into()))?;
            let samples = (r.frames as f64 / r.fps as f64 * *rate as f64) as usize;
            let at = |c: &Vec<f32>, i: usize| c.get(i).copied().unwrap_or(0.);
            let interleaved: Vec<f32> = (0..samples).flat_map(|i| [at(&channels[0], i), at(&channels[1], i)]).collect();
            Some((interleaved, *rate))
        }
        (Some(latent), Some(path)) => {
            report(event("decoding_audio", 0, 1));
            let decoder = audio::AudioDecoder::load(path, &dev)?;
            let wave = decoder.decode(latent)?;
            let rate = decoder.sample_rate;
            drop(decoder);
            let samples = (r.frames * rate).div_ceil(r.fps);
            let have = wave.dim(2)?;
            let wave = if have >= samples { wave.narrow(2, 0, samples)? } else { wave.pad_with_zeros(2, 0, samples - have)? };
            // Interleaved stereo, as FFmpeg's f32le input wants it.
            let mut interleaved = wave.squeeze(0)?.transpose(0, 1)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            if interleaved.iter().any(|x| !x.is_finite()) {
                candle_core::bail!("nonfinite decoded audio samples");
            }
            set_level(&mut interleaved, rate, reference_loudness);
            Some((interleaved, rate))
        }
        _ => None,
    };
    drop(audio_latent);
    let audio_seconds = audio_started.elapsed().as_secs_f64();
    report(event("encoding_mp4", 0, r.frames));
    let export_started = Instant::now();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(candle_core::Error::wrap)?
        .as_nanos();
    let path = r.output.join(format!("{}-{stamp}-{}.mp4", r.model, r.seed));
    let preview = path.with_extension("png");
    let log = path.with_extension("ffmpeg.log");
    let (b, c, frames, height, width) = pixels.dims5()?;
    if (b, c, frames, height, width) != (1, 3, r.frames, r.height, r.width) {
        candle_core::bail!("unexpected decoded video shape {:?}", pixels.dims());
    }
    let log_file = std::fs::File::create(&log)?;
    // The soundtrack goes to FFmpeg as a second, raw input beside the frames.
    let audio_file = path.with_extension("f32le");
    let mut audio_args: Vec<String> = vec!["-an".into()];
    if let Some((samples, rate)) = &soundtrack {
        let bytes: Vec<u8> = samples.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(&audio_file, bytes)?;
        audio_args = ["-f", "f32le", "-ar", &rate.to_string(), "-ac", "2", "-i"]
            .iter()
            .map(|s| s.to_string())
            .chain([audio_file.to_string_lossy().into_owned()])
            .chain(["-c:a", "aac", "-b:a", "192k"].iter().map(|s| s.to_string()))
            .collect();
    }
    let spawned = command(&r.ffmpeg)
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", r.width, r.height),
            "-r",
            &r.fps.to_string(),
            "-i",
            "pipe:0",
        ])
        .args(&audio_args)
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(log_file)
        .spawn();
    let mut encoder = match spawned {
        Ok(e) => e,
        Err(e) => {
            let _ = std::fs::remove_file(&audio_file);
            return Err(e.into());
        }
    };
    let result = (|| -> Result<()> {
        let mut stdin = encoder
            .stdin
            .take()
            .ok_or_else(|| candle_core::Error::Msg("FFmpeg stdin unavailable".into()))?;
        // Transfer/cast the decoded video once. Repeated small CPU tensor
        // permutations per frame otherwise dominate short-clip export on Windows.
        let values = pixels
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if values.iter().any(|x| !x.is_finite()) {
            candle_core::bail!("nonfinite decoded video pixels");
        }
        let plane = r.height * r.width;
        let channel_stride = r.frames * plane;
        let mut bytes = vec![0u8; plane * 3];
        for i in 0..r.frames {
            for (pixel, rgb) in bytes.chunks_exact_mut(3).enumerate() {
                for (c, value) in rgb.iter_mut().enumerate() {
                    let x = values[c * channel_stride + i * plane + pixel];
                    *value = ((x + 1.) * 127.5).round().clamp(0., 255.) as u8;
                }
            }
            if i == r.frames / 2 {
                image::save_buffer(
                    &preview,
                    &bytes,
                    r.width as u32,
                    r.height as u32,
                    image::ColorType::Rgb8,
                )
                .map_err(candle_core::Error::wrap)?;
            }
            stdin.write_all(&bytes)?;
        }
        drop(stdin);
        Ok(())
    })();
    if let Err(e) = result {
        let _ = encoder.kill();
        let _ = encoder.wait();
        let _ = std::fs::remove_file(&audio_file);
        return Err(e);
    }
    let finished = encoder.wait();
    let _ = std::fs::remove_file(&audio_file);
    if !finished?.success() {
        candle_core::bail!("FFmpeg failed; see {}", log.display());
    }
    let _ = std::fs::remove_file(log);
    let result = Json::obj([
        ("model", Json::str(&r.model)),
        ("path", Json::str(path.to_string_lossy())),
        ("preview", Json::str(preview.to_string_lossy())),
        ("frames", Json::Int(r.frames as i64)),
        ("width", Json::Int(r.width as i64)),
        ("height", Json::Int(r.height as i64)),
        ("prompt", Json::str(&r.prompt)),
        ("transcript", r.transcript.as_ref().map(Json::str).unwrap_or(Json::Null)),
        (
            "image",
            r.image
                .as_ref()
                .map(|p| Json::str(p.to_string_lossy()))
                .unwrap_or(Json::Null),
        ),
        (
            "end_image",
            r.end_image
                .as_ref()
                .map(|p| Json::str(p.to_string_lossy()))
                .unwrap_or(Json::Null),
        ),
        ("image_seconds", Json::Num(image_seconds)),
        ("prompt_cache_hit", Json::Bool(prompt_cache_hit)),
        ("text_seconds", Json::Num(text_seconds)),
        ("text_encoder_seconds", Json::Num(text_encoder_seconds)),
        ("connector_seconds", Json::Num(connector_seconds)),
        ("denoise_seconds", Json::Num(denoise_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        (
            "export_seconds",
            Json::Num(export_started.elapsed().as_secs_f64()),
        ),
        ("fps", Json::Int(r.fps as i64)),
        ("steps", Json::Int(steps as i64)),
        ("guided", Json::Bool(r.guidance.is_some())),
        ("int8_weights", Json::Bool(int8_weights)),
        ("two_stage", Json::Bool(two_stage)),
        ("audio", Json::Bool(soundtrack.is_some())),
        ("followed_soundtrack", Json::Bool(soundtrack_in.is_some())),
        ("soundtrack_mode", if soundtrack_in.is_some() { Json::str(&r.soundtrack_mode) } else { Json::Null }),
        ("reference_voice_tokens", Json::Int(ref_tokens as i64)),
        ("lip_synced_speech", Json::Bool(ref_tokens > 0)),
        ("image_crf", if r.image.is_some() || r.end_image.is_some() { Json::Int(image_crf(r) as i64) } else { Json::Null }),
        ("identity_guidance", if ref_tokens > 0 { Json::Num(r.identity_guidance) } else { Json::Null }),
        ("lora", r.lora.as_ref().map_or(Json::Null, |(p, _)| Json::str(p.to_string_lossy()))),
        ("a2v_guidance", if soundtrack_in.is_some() { Json::Num(r.a2v_guidance) } else { Json::Null }),
        (
            "audio_file",
            r.audio_file.as_ref().map(|p| Json::str(p.to_string_lossy())).unwrap_or(Json::Null),
        ),
        ("speech_seconds", Json::Num(speech_seconds)),
        (
            "sample_rate",
            soundtrack.as_ref().map_or(Json::Null, |(_, rate)| Json::Int(*rate as i64)),
        ),
        ("audio_seconds", Json::Num(audio_seconds)),
        ("seed", Json::Int(r.seed as i64)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("memory", Json::str(&r.memory)),
        ("backend", Json::str(if r.webgpu { "webgpu" } else { "cuda" })),
        ("gpu_weight_bytes", Json::Int(gpu as i64)),
        ("ram_weight_bytes", Json::Int(host as i64)),
        ("weight_bytes_read", Json::Int(disk as i64)),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}
/// The transformer a clip is denoised by: Candle's, or (`backend` "webgpu") the video stream on WebGPU with its
/// tokens' rotary table.
enum Model {
    Candle(transformer::Transformer),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::ltx_wgpu::WgpuLtx, ggml_rs::DeviceVec),
}

/// The video stream of `store`'s transformer on WebGPU, for a latent of `f` frames of `h` by `w`.
#[cfg(feature = "webgpu")]
fn webgpu_model(r: &Request, store: &mut Store, f: usize, h: usize, w: usize, report: &mut dyn FnMut(Json)) -> Result<Model> {
    report(event("loading_video_model", 0, 48));
    let m = crate::ltx_wgpu::WgpuLtx::load(store, r.device, |n| report(event("loading_video_model", n, 48)))?;
    let table = m.upload(&crate::ltx_wgpu::rope_table(&crate::ltx_wgpu::video_positions(f, h, w, r.fps, r.end_image.is_some()), &[20., 2048., 2048.], 4096, 32));
    Ok(Model::Wgpu(m, table))
}
#[cfg(not(feature = "webgpu"))]
fn webgpu_model(_: &Request, _: &mut Store, _: usize, _: usize, _: usize, _: &mut dyn FnMut(Json)) -> Result<Model> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The prompt's video context and the negative prompt's (where there is one) on WebGPU: from the prompt cache where
/// it holds them (`cached` the prompt's), else Gemma, the projection and the connector on the GPU, both prompts in
/// one pass over Gemma's layers. On the host (F32 where new).
#[cfg(feature = "webgpu")]
fn webgpu_contexts(r: &Request, store: &mut Store, cached: Option<Tensor>, negative_text: Option<String>, report: &mut dyn FnMut(Json)) -> Result<(Tensor, Option<Tensor>)> {
    let negative_cache = negative_text.as_ref().and_then(|text| cache::PromptCache::new(&Request { prompt: text.clone(), audio: false, ..r.clone() }));
    let cached_negative = negative_cache.as_ref().and_then(|c| c.load(&Device::Cpu));
    let mut prompts = Vec::new();
    if cached.is_none() {
        prompts.push(r.prompt.clone());
    }
    if let (Some(text), None) = (&negative_text, &cached_negative) {
        prompts.push(text.clone());
    }
    let mut fresh = if prompts.is_empty() {
        report(event("cached_video_prompt", 1, 1));
        Vec::new()
    } else {
        report(event("encoding_video_prompt", 0, 1));
        crate::ltx_text_wgpu::contexts(&r.text_encoder, r.tokenizer.as_deref(), store, &prompts, r.device, |n, of| report(event("encoding_video_prompt", n, of)))?
    }
    .into_iter();
    let mut next = || -> Result<Tensor> {
        let values = fresh.next().ok_or_else(|| candle_core::Error::Msg("a prompt's context is missing".into()))?;
        Tensor::from_vec(values, (1, 1024, 4096), &Device::Cpu)
    };
    let context = match cached {
        Some(c) => c,
        None => {
            let c = next()?;
            if let Some(cache) = cache::PromptCache::new(r) {
                let _ = cache.save(&c);
            }
            c
        }
    };
    let negative = match (negative_text, cached_negative) {
        (Some(_), Some(c)) => Some(c),
        (Some(_), None) => {
            let c = next()?;
            if let Some(cache) = &negative_cache {
                let _ = cache.save(&c);
            }
            Some(c)
        }
        _ => None,
    };
    Ok((context, negative))
}
#[cfg(not(feature = "webgpu"))]
fn webgpu_contexts(_: &Request, _: &mut Store, _: Option<Tensor>, _: Option<String>, _: &mut dyn FnMut(Json)) -> Result<(Tensor, Option<Tensor>)> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The clip of `latent` (`[1, f h w, 128]`) by the video decoder on WebGPU: `[1, 3, frames, height, width]`.
#[cfg(feature = "webgpu")]
fn webgpu_decode(r: &Request, latent: &Tensor, f: usize, h: usize, w: usize) -> Result<Tensor> {
    let mut store = Store::open(&r.vae, 0)?;
    let decoder = crate::ltx_vae_wgpu::WgpuLtxVae::load(&mut store, r.device)?;
    drop(store);
    decoder.decode_fitted(&latent.flatten_all()?.to_vec1::<f32>()?, f, h, w)
}
#[cfg(not(feature = "webgpu"))]
fn webgpu_decode(_: &Request, _: &Tensor, _: usize, _: usize, _: usize) -> Result<Tensor> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// The clip length for a soundtrack of `seconds`: enough frames at `fps` to
/// hold all of it, snapped up to 8k+1 (the reference snaps down, which cuts
/// the last words off), within 9..=121. The soundtrack is padded with silence
/// to the clip's length.
fn frames_for_audio(seconds: f64, fps: usize) -> usize {
    let raw = ((seconds * fps as f64 - 1e-6).ceil().max(0.) as usize).clamp(9, 121);
    ((raw - 1).div_ceil(8) * 8 + 1).min(121)
}

/// Decode any audio FFmpeg reads to stereo F32 at its own sample rate (mono
/// is duplicated; more channels are mixed down). At most 60 seconds.
fn read_audio(ffmpeg: &std::path::Path, path: &std::path::Path) -> Result<([Vec<f32>; 2], usize)> {
    let out = command(ffmpeg)
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(path)
        .args(["-t", "60", "-vn", "-ac", "2", "-f", "f32le", "pipe:1"])
        .stdin(Stdio::null())
        .output()?;
    let log = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        candle_core::bail!("FFmpeg could not read the soundtrack: {}", log.lines().last().unwrap_or("").trim());
    }
    let rate = regex::Regex::new(r"Audio: [^\n]*?, (\d+) Hz")
        .map_err(candle_core::Error::wrap)?
        .captures(&log)
        .and_then(|c| c[1].parse::<usize>().ok())
        .filter(|r| (1000..=384_000).contains(r))
        .ok_or_else(|| candle_core::Error::Msg("the soundtrack has no audio stream".into()))?;
    let samples: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    if samples.len() < 2 * rate / 10 {
        candle_core::bail!("the soundtrack is shorter than a tenth of a second");
    }
    let left = samples.iter().step_by(2).copied().collect();
    let right = samples.iter().skip(1).step_by(2).copied().collect();
    Ok(([left, right], rate))
}

/// Seeds the audio noise apart from the video noise drawn with the same seed.
const AUDIO_SEED_OFFSET: u64 = 0x9e37_79b9_7f4a_7c15;
/// Stage two's noise, drawn apart from stage one's.
const REFINE_SEED_OFFSET: u64 = 0xc2b2_ae3d_27d4_eb4f;

/// Audio latent frames for a clip: 25 per second of video, rounded half to
/// even as the reference does.
fn audio_latent_frames(frames: usize, fps: usize) -> usize {
    (frames as f64 / fps as f64 * audio::LATENT_RATE).round_ties_even().max(1.) as usize
}

/// The H.264 quality an image is re-compressed at before it conditions a
/// clip, as the reference does: the models were trained on video frames, and a
/// pristine still tends to stay still. 33 for the LTX 2.3 generation (Sulphur
/// is one), 18 from 2.4 on; 0 leaves the image as it is.
fn image_crf(r: &Request) -> u32 {
    r.image_crf.unwrap_or(if r.model == "ltx-2.5" { 18 } else { 33 })
}

/// An image through one H.264 frame at `crf` and back (FFmpeg, libx264,
/// veryfast, 4:2:0), at its own size cut to even sides.
fn recompress(ffmpeg: &std::path::Path, image: image::RgbImage, crf: u32) -> Result<image::RgbImage> {
    use std::io::Write;
    let (w, h) = (image.width() / 2 * 2, image.height() / 2 * 2);
    if crf == 0 || w < 2 || h < 2 {
        return Ok(image);
    }
    let image = image::imageops::crop_imm(&image, 0, 0, w, h).to_image();
    let run = |args: &[&str], input: &[u8]| -> Result<Vec<u8>> {
        let mut child = command(ffmpeg)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().ok_or_else(|| candle_core::Error::Msg("FFmpeg stdin".into()))?;
        let input = input.to_vec();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let out = child.wait_with_output()?;
        writer.join().map_err(|_| candle_core::Error::Msg("FFmpeg writer".into()))??;
        if !out.status.success() || out.stdout.is_empty() {
            candle_core::bail!("FFmpeg could not re-compress the conditioning image");
        }
        Ok(out.stdout)
    };
    let size = format!("{w}x{h}");
    let crf = crf.to_string();
    let h264 = run(&["-hide_banner", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &size, "-i", "-", "-frames:v", "1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", &crf, "-pix_fmt", "yuv420p", "-f", "h264", "-"], image.as_raw())?;
    let rgb = run(&["-hide_banner", "-f", "h264", "-i", "-", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], &h264)?;
    image::RgbImage::from_raw(w, h, rgb.get(..(w * h * 3) as usize).map(<[u8]>::to_vec).unwrap_or_default())
        .ok_or_else(|| candle_core::Error::Msg("FFmpeg returned a short frame".into()))
}

/// The start and end images' latents (`[1, h w, 128]`) by the VAE's encoder on WebGPU.
#[cfg(feature = "webgpu")]
fn webgpu_endpoints(r: &Request, (width, height): (usize, usize), dev: &Device, report: &mut dyn FnMut(Json)) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let mut store = Store::open(&r.vae, 0)?;
    let encoder = crate::ltx_vae_wgpu::WgpuLtxImageEncoder::load(&mut store, r.device)?;
    drop(store);
    let mut encode = |path: &Option<PathBuf>, stage: &str| -> Result<Option<Tensor>> {
        path.as_ref()
            .map(|path| {
                report(event(stage, 0, 1));
                let latent = encoder.encode(&image_values(path, (width, height), &r.ffmpeg, image_crf(r))?, height, width)?;
                Tensor::from_vec(latent, (1, height / 32 * (width / 32), 128), dev)
            })
            .transpose()
    };
    Ok((encode(&r.image, "encoding_starting_image")?, encode(&r.end_image, "encoding_ending_image")?))
}
#[cfg(not(feature = "webgpu"))]
fn webgpu_endpoints(_: &Request, _: (usize, usize), _: &Device, _: &mut dyn FnMut(Json)) -> Result<(Option<Tensor>, Option<Tensor>)> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

#[allow(clippy::too_many_arguments)]
fn encode_image(
    path: &std::path::Path,
    (width, height): (usize, usize),
    encoder: &vae::LtxVideoEncoder,
    dev: &Device,
    dtype: DType,
    ffmpeg: &std::path::Path,
    crf: u32,
) -> Result<Tensor> {
    let values = image_values(path, (width, height), ffmpeg, crf)?;
    let pixels = Tensor::from_vec(values, (1, 1, height, width, 3), &dev)?
        .permute((0, 4, 1, 2, 3))?
        .contiguous()?
        .to_dtype(dtype)?;
    let latent = encoder
        .encode_means(&pixels)?
        .permute((0, 2, 3, 4, 1))?
        .contiguous()?
        .reshape((1, height / 32 * (width / 32), 128))?
        .to_dtype(DType::F32)?;
    Ok(latent)
}

/// An endpoint image's pixels as the encoder takes them: re-compressed as the clip's frames are (`crf`), resized to
/// fill `width` by `height`, each pixel's red, green and blue in -1..1 in turn.
fn image_values(path: &std::path::Path, (width, height): (usize, usize), ffmpeg: &std::path::Path, crf: u32) -> Result<Vec<f32>> {
    let mut reader = image::ImageReader::open(path)?
        .with_guessed_format()
        .map_err(candle_core::Error::wrap)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let decoded = reader.decode().map_err(candle_core::Error::wrap)?.to_rgb8();
    let pixels = image::DynamicImage::ImageRgb8(recompress(ffmpeg, decoded, crf)?)
        .resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    Ok(pixels.as_raw().iter().map(|&v| v as f32 / 127.5 - 1.).collect())
}
fn condition_endpoints(
    latent: Tensor,
    start: Option<&Tensor>,
    end: Option<&Tensor>,
) -> Result<Tensor> {
    let start_tokens = start.map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let end_tokens = end.map(|t| t.dim(1)).transpose()?.unwrap_or(0);
    let tokens = latent.dim(1)?;
    if start_tokens + end_tokens >= tokens {
        candle_core::bail!("endpoint conditioning must leave generated tokens");
    }
    let mut parts = Vec::new();
    if let Some(start) = start {
        parts.push(start.clone());
    }
    parts.push(latent.narrow(1, start_tokens, tokens - start_tokens - end_tokens)?);
    if let Some(end) = end {
        parts.push(end.clone());
    }
    Tensor::cat(&parts, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_tokens_stay_clean_without_freezing_the_last_video_latent() -> Result<()> {
        let start = Tensor::full(3f32, (1, 2, 2), &Device::Cpu)?;
        let end = Tensor::full(7f32, (1, 2, 2), &Device::Cpu)?;
        for with_start in [false, true] {
            let mut latent = Tensor::zeros((1, 8, 2), DType::F32, &Device::Cpu)?;
            for _ in 0..8 {
                latent = condition_endpoints(
                    (latent + 0.25)?,
                    with_start.then_some(&start),
                    Some(&end),
                )?;
            }
            let values = latent.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(&values[..4], &[if with_start { 3. } else { 2. }; 4]);
            assert_eq!(&values[4..12], &[2.; 8]);
            assert_eq!(&values[12..], &[7.; 4]);
        }
        Ok(())
    }
    #[test]
    fn soundtrack_level_matches_the_voice_and_never_clips() {
        let rate = 1000;
        // A loud tone (peak 1.4, RMS near 1) against a reference voice at RMS 0.4.
        let tone: Vec<f32> = (0..2 * rate).flat_map(|i| { let v = 1.4 * (i as f32 * 0.3).sin(); [v, v] }).collect();
        let mut matched = tone.clone();
        set_level(&mut matched, rate, Some(0.4));
        let left: Vec<f32> = matched.iter().step_by(2).copied().collect();
        assert!((speech_loudness(&left, rate).unwrap() - 0.4).abs() < 0.02);
        let mut limited = tone;
        set_level(&mut limited, rate, None);
        assert!(limited.iter().all(|x| x.abs() <= 0.967));
        assert_eq!(speech_loudness(&[0.; 100], rate), None);
    }
    #[test]
    fn guided_schedule_matches_the_reference() {
        let s = guided_sigmas(30);
        assert_eq!(s.len(), 31);
        assert_eq!(s[0], 1.);
        assert!((s[29] - 0.1).abs() < 1e-9);
        assert_eq!(s[30], 0.);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
        // Before stretching, t = 0.5 maps to e^2.05 / (e^2.05 + 1).
        let shifted = 2.05f64.exp() / (2.05f64.exp() + 1.);
        let last = 2.05f64.exp() / (2.05f64.exp() + 29.);
        let scale = (1. - last) / 0.9;
        assert!((s[15] - (1. - (1. - shifted) / scale)).abs() < 1e-12);
    }
    #[test]
    fn clips_follow_the_soundtrack_length() {
        assert_eq!(frames_for_audio(3.2, 24), 81);
        // "After thirty years?": 1.68 s needs 40.3 frames, so 41, not 33.
        assert_eq!(frames_for_audio(1.68, 24), 41);
        assert_eq!(frames_for_audio(1.0, 24), 25);
        assert_eq!(frames_for_audio(0.2, 24), 9);
        assert_eq!(frames_for_audio(30., 24), 121);
        let base = |extra: &str| {
            Json::parse(format!(r#"{{"model":"ltx-2.5","transformer":"t","text_encoder":"e","vae":"v","audio_vae":"a","output_dir":"o","prompt":"p"{extra}}}"#).as_bytes()).unwrap()
        };
        let r = Request::parse(&base(r#","speech":{"kind":"speech"}"#)).unwrap();
        assert!(r.frames_from_audio && r.audio && r.speech.is_some());
        let r = Request::parse(&base(r#","speech":{"kind":"speech"},"frames":49"#)).unwrap();
        assert!(!r.frames_from_audio && r.frames == 49);
        assert!(Request::parse(&base(r#","speech":{"kind":"speech"},"audio":false"#)).is_err());
        assert!(Request::parse(&base(r#","audio_file":"relative.wav""#)).is_err());
    }
    #[test]
    fn audio_length_follows_the_clip() {
        assert_eq!(audio_latent_frames(49, 24), 51);
        assert_eq!(audio_latent_frames(121, 24), 126);
        assert_eq!(audio_latent_frames(25, 25), 25);
    }
    #[test]
    fn audio_defaults_on_for_ltx_2_5_with_an_audio_vae() {
        let base = |extra: &str| {
            Json::parse(format!(r#"{{"model":"ltx-2.5","transformer":"t","text_encoder":"e","vae":"v","output_dir":"o","prompt":"p"{extra}}}"#).as_bytes()).unwrap()
        };
        assert!(Request::parse(&base(r#","audio_vae":"a""#)).unwrap().audio);
        let sulphur = Json::parse(br#"{"model":"sulphur-2","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","audio_vae":"t","output_dir":"o","prompt":"p"}"#).unwrap();
        assert!(Request::parse(&sulphur).unwrap().audio, "LTX 2.3 checkpoints carry their own audio VAE");
        assert!(!Request::parse(&base("")).unwrap().audio);
        assert!(!Request::parse(&base(r#","audio_vae":"a","audio":false"#)).unwrap().audio);
        assert!(Request::parse(&base(r#","audio":true"#)).is_err(), "audio without its VAE");
    }
    #[test]
    #[cfg(feature = "webgpu")]
    fn a_webgpu_clip_is_text_to_video_without_sound_for_now() {
        let request = |extra: &str| {
            let j = Json::parse(format!(r#"{{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","audio_vae":"t","output_dir":"o","prompt":"p","backend":"webgpu"{extra}}}"#).as_bytes()).unwrap();
            Request::parse(&j)
        };
        let r = request("").unwrap();
        assert!(r.webgpu && !r.audio, "sound off by default on WebGPU, though the checkpoint has its VAE");
        assert!(request(r#","guidance":{"steps":30}"#).unwrap().guidance.is_some(), "guided sampling, its negative prompt by CFG");
        assert!(request(r#","negative_prompt":"n""#).unwrap().negative_prompt.is_some(), "a negative prompt without CFG: by NAG");
        for (extra, what) in [
            (r#","audio":true"#, "sound"),
            (r#","lora":"l""#, "LoRAs"),
            (r#","refine":{"transformer":"t","upsampler":"u"}"#, "refinement"),
        ] {
            let e = request(extra).unwrap_err();
            assert!(e.starts_with("WebGPU video does not support") && e.contains(what), "{extra}: {e}");
        }
        let cuda = Json::parse(br#"{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","output_dir":"o","prompt":"p","backend":"cuda"}"#).unwrap();
        assert!(!Request::parse(&cuda).unwrap().webgpu);
        let other = Json::parse(br#"{"model":"ltx-2.3","transformer":"t","text_encoder":"e","tokenizer":"k","vae":"t","output_dir":"o","prompt":"p","backend":"metal"}"#).unwrap();
        assert!(Request::parse(&other).is_err());
    }
    #[test]
    fn starting_frame_is_exact_after_each_euler_update() -> Result<()> {
        let clean = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 2, 2), &Device::Cpu)?;
        let mut latent = Tensor::zeros((1, 6, 2), DType::F32, &Device::Cpu)?;
        for _ in 0..8 {
            latent = condition_endpoints((latent + 0.25)?, Some(&clean), None)?;
            assert_eq!(
                latent.narrow(1, 0, 2)?.flatten_all()?.to_vec1::<f32>()?,
                vec![1., 2., 3., 4.]
            );
        }
        assert_eq!(
            latent.narrow(1, 2, 4)?.flatten_all()?.to_vec1::<f32>()?,
            vec![2.; 8]
        );
        Ok(())
    }
}
fn inference_device(index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(index)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        candle_core::bail!("LTX video requires the CUDA worker; rebuild with --features flash-attn")
    }
}
fn command(path: &std::path::Path) -> Command {
    let mut c = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x08000000);
    }
    c
}
