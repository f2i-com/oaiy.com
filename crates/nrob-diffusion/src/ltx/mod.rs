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
mod text;
mod transformer;
pub mod vae;
use candle_core::{DType, Device, Result, Tensor};
use nrob::json::Json;
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
    /// away from that (the reference's modality guidance, 3 by default; 1 is
    /// off). It is what makes mouths follow the words.
    pub a2v_guidance: f64,
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
            a2v_guidance: j.get("a2v_guidance").and_then(Json::as_f64).unwrap_or(3.0),
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
    if let Some(w) = words.filter(|w| !w.is_empty() && !r.prompt.contains(w.as_str())) {
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
    owned.validate().map_err(candle_core::Error::Msg)?;
    let r = &owned;
    let dev = inference_device(r.device)?;
    let ram = if r.memory == "ssd" || r.memory == "gpu" {
        0
    } else {
        r.ram_bytes
    };
    let mut store = Store::open(&r.transformer, ram)?;
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
    let (starting_latent, ending_latent) = if r.image.is_some() || r.end_image.is_some() {
        let encoder = vae::LtxVideoEncoder::load(
            &r.vae,
            vae::LtxVaeConfig::ltx_2_3_22b(),
            &dev,
            DType::BF16,
        )?;
        let mut encode = |path: &Option<PathBuf>, stage: &str| -> Result<Option<Tensor>> {
            path.as_ref()
                .map(|path| {
                    report(event(stage, 0, 1));
                    encode_image(path, r, &encoder, &dev)
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
    let text_seconds = encode_started.elapsed().as_secs_f64();
    let gpu_budget = if r.memory == "ram" || r.memory == "ssd" {
        0
    } else {
        r.vram_bytes
    };
    #[cfg(feature = "cuda")]
    let gpu_budget = {
        let free = dev
            .as_cuda_device()?
            .cuda_stream()
            .context()
            .mem_get_info()
            .map_err(candle_core::Error::wrap)?
            .0 as u64;
        // Leave room for one streamed block, global projections, and activations.
        gpu_budget.min(free.saturating_sub(6 * GIB))
    };
    let mut model = transformer::Transformer::new(store, &dev, gpu_budget, r.memory == "gpu", r.audio)?;
    let (f, h, w) = ((r.frames - 1) / 8 + 1, r.height / 32, r.width / 32);
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
        Some((
            transformer::Rope::audio(audio_frames, &dev)?,
            transformer::Rope::video_cross(f, h, w, r.fps, ending_latent.is_some(), &dev)?,
        ))
    } else {
        None
    };
    let frozen_audio = conditioning_audio.is_some();
    let mut audio_latent = if let Some(latent) = conditioning_audio {
        Some(latent)
    } else if r.audio {
        let noise = crate::pipeline::noise(r.seed.wrapping_add(AUDIO_SEED_OFFSET), audio_frames * 128);
        Some(Tensor::from_vec(noise, (1, audio_frames, 128), &dev)?)
    } else {
        None
    };
    let denoise_started = Instant::now();
    // Following a soundtrack, each step also runs with the audio-video
    // cross-attention skipped, and the picture moves away from that result.
    let guided = frozen_audio && r.a2v_guidance > 1.0;
    let passes = if guided { 2 } else { 1 };
    for step in 0..8 {
        let audio_bf16 = audio_latent.as_ref().map(|a| a.to_dtype(DType::BF16)).transpose()?;
        let video_bf16 = latent.to_dtype(DType::BF16)?;
        let mut run = |isolated: bool, pass: usize| {
            let audio_input = match (&audio_bf16, &audio_context, &audio_ropes) {
                (Some(latent), Some(context), Some((rope, video_cross))) => Some(transformer::AudioInput {
                    latent,
                    context,
                    rope,
                    video_cross,
                    sigma: if frozen_audio { 0. } else { SIGMAS[step] },
                    isolated,
                }),
                _ => None,
            };
            model.forward(
                &video_bf16,
                &context,
                SIGMAS[step],
                &rope,
                if starting_latent.is_some() { h * w } else { 0 },
                if ending_latent.is_some() { h * w } else { 0 },
                audio_input,
                |n| report(event("video_denoising", (step * passes + pass) * 48 + n, 8 * passes * 48)),
            )
        };
        let (mut velocity, audio_velocity) = run(false, 0)?;
        if guided {
            let (isolated, _) = run(true, 1)?;
            let cond = velocity.to_dtype(DType::F32)?;
            let delta = (&cond - isolated.to_dtype(DType::F32)?)?;
            velocity = (cond + (delta * (r.a2v_guidance - 1.0))?)?;
        }
        let dt = SIGMAS[step + 1] - SIGMAS[step];
        latent = (latent + (velocity.to_dtype(DType::F32)? * dt)?)?;
        latent = condition_endpoints(latent, starting_latent.as_ref(), ending_latent.as_ref())?;
        if let (Some(a), Some(v), false) = (audio_latent.as_mut(), audio_velocity, frozen_audio) {
            *a = (&*a + (v.to_dtype(DType::F32)? * dt)?)?;
        }
    }
    // Appended end-keyframe tokens guide attention but are not part of the decoded clip.
    latent = latent.narrow(1, 0, f * h * w)?.contiguous()?;
    let (gpu, host, disk) = model.stats();
    dev.synchronize()?;
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    drop(model);
    drop(context);
    drop(audio_context);
    drop(audio_ropes);
    drop(rope);
    dev.synchronize()?;
    report(event("decoding_video", 0, 1));
    let decode_started = Instant::now();
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
        pixel_budget.min(free.saturating_sub(2usize << 30) / 512)
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
            let interleaved = wave.squeeze(0)?.transpose(0, 1)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            if interleaved.iter().any(|x| !x.is_finite()) {
                candle_core::bail!("nonfinite decoded audio samples");
            }
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
        ("steps", Json::Int(8)),
        ("audio", Json::Bool(soundtrack.is_some())),
        ("followed_soundtrack", Json::Bool(soundtrack_in.is_some())),
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
        ("gpu_weight_bytes", Json::Int(gpu as i64)),
        ("ram_weight_bytes", Json::Int(host as i64)),
        ("weight_bytes_read", Json::Int(disk as i64)),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}
/// The clip length for a soundtrack of `seconds`: whole frames at `fps`,
/// snapped down to 8k+1 as the reference does, within 9..=121.
fn frames_for_audio(seconds: f64, fps: usize) -> usize {
    let raw = ((seconds * fps as f64) as usize).clamp(9, 121);
    (raw - 1) / 8 * 8 + 1
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

/// Audio latent frames for a clip: 25 per second of video, rounded half to
/// even as the reference does.
fn audio_latent_frames(frames: usize, fps: usize) -> usize {
    (frames as f64 / fps as f64 * audio::LATENT_RATE).round_ties_even().max(1.) as usize
}

fn encode_image(
    path: &std::path::Path,
    r: &Request,
    encoder: &vae::LtxVideoEncoder,
    dev: &Device,
) -> Result<Tensor> {
    let mut reader = image::ImageReader::open(path)?
        .with_guessed_format()
        .map_err(candle_core::Error::wrap)?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let pixels = reader
        .decode()
        .map_err(candle_core::Error::wrap)?
        .resize_to_fill(
            r.width as u32,
            r.height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let values: Vec<f32> = pixels
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 127.5 - 1.)
        .collect();
    let pixels = Tensor::from_vec(values, (1, 1, r.height, r.width, 3), &dev)?
        .permute((0, 4, 1, 2, 3))?
        .contiguous()?
        .to_dtype(DType::BF16)?;
    let latent = encoder
        .encode_means(&pixels)?
        .permute((0, 2, 3, 4, 1))?
        .contiguous()?
        .reshape((1, r.height / 32 * (r.width / 32), 128))?
        .to_dtype(DType::F32)?;
    Ok(latent)
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
    fn clips_follow_the_soundtrack_length() {
        assert_eq!(frames_for_audio(3.2, 24), 73);
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
