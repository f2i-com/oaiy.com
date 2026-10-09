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
mod inputs;
mod request;
pub(crate) mod store;
pub(crate) mod text;
pub(crate) mod transformer;
mod upsampler;
pub mod vae;
mod webgpu;
#[cfg(test)]
mod tests;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use store::Store;
// (a clip's request for the crate's users, and what the files make for each other)
pub use request::*;
use {inputs::*, webgpu::*};

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

/// The standard deviation of every element.
fn std_all(t: &Tensor) -> Result<f64> {
    let t = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean = t.mean_all()?.to_scalar::<f32>()?;
    Ok(t.affine(1., -mean as f64)?.sqr()?.mean_all()?.to_scalar::<f32>()?.sqrt() as f64)
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
        let contexts = webgpu_contexts(r, &mut store, cached, cached_audio, negative_text, &mut report)?;
        text_encoder_seconds = encode_started.elapsed().as_secs_f64();
        contexts
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
    let (f, mut h, mut w) = ((r.frames - 1) / 8 + 1, stage_size.1 / 32, stage_size.0 / 32);
    let mut model = if r.webgpu {
        webgpu_model(r, &mut store, f, h, w, audio_frames, &mut report)?
    } else {
        Model::Candle(transformer::Transformer::new(store, &dev, gpu_budget, r.memory == "gpu", r.audio)?)
    };
    // A clip with sound on a card with no room left for its steps once both streams are on it (LTX 2.3's are 20 GiB,
    // the largest clip's steps 7 more): without its sound, and said, as such a clip ran before the audio stream was
    // on WebGPU, where it would fail partway.
    #[cfg(feature = "webgpu")]
    let silenced = match &mut model {
        // (a soundtrack to follow is the audio stream's to carry: without room for that stream the clip cannot follow
        // it, and a clip that ignored its soundtrack would be the wrong clip)
        Model::Wgpu(m, _, _) if r.audio && conditioning_audio.is_some() && !m.room_for_step(f * h * w + if r.end_image.is_some() { h * w } else { 0 }) => {
            candle_core::bail!("the GPU has no room for this clip's steps beside the audio stream that following a soundtrack needs: a smaller or shorter clip, or a card with more memory")
        }
        Model::Wgpu(m, _, sound) if r.audio && !m.room_for_step(f * h * w + if r.end_image.is_some() { h * w } else { 0 }) => {
            m.drop_audio();
            *sound = None;
            report(event("sound_dropped_no_gpu_memory", 1, 1));
            Some(Request { audio: false, ..r.clone() })
        }
        _ => None,
    };
    #[cfg(feature = "webgpu")]
    let r = silenced.as_ref().unwrap_or(r);
    // A negative prompt without CFG steers every step by NAG.
    if let (true, Some(context), Model::Candle(model)) = (r.guidance.as_ref().is_none_or(|g| g.cfg == 1.), &negative, &mut model) {
        model.nag = Some(transformer::Nag { context: context.clone(), scale: r.nag.0, tau: r.nag.1, alpha: r.nag.2 });
        report(event("negative_prompt_nag", 1, 1));
    }
    #[cfg(feature = "webgpu")]
    if let (true, Some(context), Model::Wgpu(model, ..)) = (r.guidance.as_ref().is_none_or(|g| g.cfg == 1.), &negative, &mut model) {
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
                Model::Wgpu(m, table, sound) => {
                    let latent = video_f32.as_deref().ok_or_else(|| candle_core::Error::Msg("the latent's values".into()))?;
                    let values = context.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    let tokens = latent.len() / 128;
                    let clean = (if starting_latent.is_some() { h * w } else { 0 }, if ending_latent.is_some() { h * w } else { 0 });
                    let host = |t: &Tensor| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>();
                    match (&audio_bf16, &audio_context, sound.as_ref()) {
                        // a clip with sound: the two streams together, the audio at the video's sigma; a soundtrack
                        // held as it was given at sigma 0 (one mixed with noise each step at the step's). The pass
                        // without the streams' attention to each other (what guidance by a soundtrack is set against:
                        // only its picture is used) is the arm below: the picture's stream alone is that pass.
                        (Some(audio), Some(audio_context), Some((sound_table, video_cross))) if !isolated => {
                            let (a, c) = (host(audio)?, host(audio_context)?);
                            let sound_sigma = if frozen_audio && inpaint_noise.is_none() { 0. } else { sigma };
                            let pass = crate::ltx_wgpu::AudioPass { latent: &a, tokens: a.len() / 128, context: &c, rows: c.len() / 2048, sigma: sound_sigma, table: sound_table, video_cross, skip_self: None };
                            let (v, s) = m.forward_av(latent, tokens, &values, values.len() / 4096, sigma, table, clean, h * w, perturb, &pass)?;
                            report(event("video_denoising", (step * passes + at + 1) * 48, steps * passes * 48));
                            Ok((Tensor::from_vec(v, (1, tokens, 128), &dev)?, Some(Tensor::from_vec(s, (1, pass.tokens, 128), &dev)?)))
                        }
                        _ => {
                            let v = m.forward(latent, tokens, &values, values.len() / 4096, sigma, table, clean, h * w, perturb)?;
                            report(event("video_denoising", (step * passes + at + 1) * 48, steps * passes * 48));
                            Ok((Tensor::from_vec(v, (1, tokens, 128), &dev)?, None))
                        }
                    }
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
            // (a WebGPU job's: the vocoder and the bandwidth extension on its GPU, nearly all of the decoding)
            #[cfg(feature = "webgpu")]
            let wave = if r.webgpu { decoder.decode_webgpu(r.device, latent)? } else { decoder.decode(latent)? };
            #[cfg(not(feature = "webgpu"))]
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
        ("backend", Json::str(if r.webgpu { "webgpu" } else { "cpu" })),
        ("gpu_weight_bytes", Json::Int(gpu as i64)),
        ("ram_weight_bytes", Json::Int(host as i64)),
        ("weight_bytes_read", Json::Int(disk as i64)),
    ]);
    std::fs::write(path.with_extension("json"), result.to_json())?;
    Ok(result)
}

/// Seeds the audio noise apart from the video noise drawn with the same seed.
const AUDIO_SEED_OFFSET: u64 = 0x9e37_79b9_7f4a_7c15;
/// Stage two's noise, drawn apart from stage one's.
const REFINE_SEED_OFFSET: u64 = 0xc2b2_ae3d_27d4_eb4f;

fn inference_device(index: usize) -> Result<Device> {
    // (Candle's CPU backend cannot multiply the model's BF16 weights: video is WebGPU's)
    let _ = index;
    candle_core::bail!("LTX video runs on WebGPU: the cpu backend cannot multiply its BF16 weights")
}
