//! A clip's request: what to make, from which files, on which device, and how it is guided and refined.

use super::*;

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

/// The reference's second stage (its two-stage pipelines): after guided
/// sampling at half the size, the latent is doubled by `upsampler` and
/// refined by the distilled `transformer` over the last three steps of its
/// schedule, with the soundtrack held as in stage one and no guidance.
#[derive(Clone, Debug)]
pub struct Refine {
    pub transformer: PathBuf,
    pub upsampler: PathBuf,
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
        let webgpu = crate::pipeline::backend_is_webgpu(j, "video")?;
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
            // the request says `"audio": false`. A given soundtrack needs it.
            audio: match j.get("audio") {
                None | Some(Json::Null) => {
                    j.get("audio_vae").and_then(Json::as_str).is_some_and(|s| !s.trim().is_empty())
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
            // (a clip's own sound runs there, the two streams denoised together, and a given soundtrack is followed:
            // the audio stream held as it was given while the picture's is denoised beside it)
            let unsupported = [
                (self.reference_voice.is_some() || self.identity, "a reference voice"),
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
