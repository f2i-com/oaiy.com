//! Native Rust text/image-to-video for LTX 2.3/2.5 and compatible distilled checkpoints.
mod cache;
mod store;
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
            frames: n("frames", 49)?,
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
        for k in ["images", "audio", "adapter"] {
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
    for (key, shape) in [
        ("patchify_proj.weight", vec![4096, 128]),
        ("transformer_blocks.47.scale_shift_table", vec![9, 4096]),
    ] {
        let name = format!("{}{key}", transformer::PREFIX);
        let info = store.index.info(&name).map_err(candle_core::Error::wrap)?;
        if info.shape != shape {
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
    let encode_started = Instant::now();
    let prompt_cache = cache::PromptCache::new(r);
    let cached = prompt_cache.as_ref().and_then(|c| c.load(&dev));
    let prompt_cache_hit = cached.is_some();
    let mut text_encoder_seconds = 0.;
    let mut connector_seconds = 0.;
    let context = if let Some(context) = cached {
        report(event("cached_video_prompt", 1, 1));
        context
    } else {
        report(event("encoding_video_prompt", 0, 48));
        let features = text::encode(
            &r.text_encoder,
            r.tokenizer.as_deref(),
            &mut store,
            &r.prompt,
            r.model == "ltx-2.5",
            &dev,
            |n| report(event("encoding_video_prompt", n, 48)),
        )?;
        dev.synchronize()?;
        text_encoder_seconds = encode_started.elapsed().as_secs_f64();
        report(event("video_text_connector", 0, 8));
        let connector_started = Instant::now();
        let context = transformer::connector(&mut store, &features, &dev, |n| {
            report(event("video_text_connector", n, 8))
        })?;
        drop(features);
        dev.synchronize()?;
        connector_seconds = connector_started.elapsed().as_secs_f64();
        if let Some(cache) = &prompt_cache {
            let _ = cache.save(&context);
        }
        context
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
    let mut model = transformer::Transformer::new(store, &dev, gpu_budget, r.memory == "gpu")?;
    let (f, h, w) = ((r.frames - 1) / 8 + 1, r.height / 32, r.width / 32);
    let rope = transformer::Rope::video_with_end(f, h, w, r.fps, ending_latent.is_some(), &dev)?;
    let noise = crate::pipeline::noise(r.seed, f * h * w * 128);
    let mut latent = Tensor::from_vec(noise, (1, f * h * w, 128), &dev)?;
    if let Some(end) = &ending_latent {
        latent = Tensor::cat(&[&latent, end], 1)?;
    }
    latent = condition_endpoints(latent, starting_latent.as_ref(), ending_latent.as_ref())?;
    let denoise_started = Instant::now();
    for step in 0..8 {
        let velocity = model.forward(
            &latent.to_dtype(DType::BF16)?,
            &context,
            SIGMAS[step],
            &rope,
            if starting_latent.is_some() { h * w } else { 0 },
            if ending_latent.is_some() { h * w } else { 0 },
            |n| report(event("video_denoising", step * 48 + n, 8 * 48)),
        )?;
        latent = (latent + (velocity.to_dtype(DType::F32)? * (SIGMAS[step + 1] - SIGMAS[step]))?)?;
        latent = condition_endpoints(latent, starting_latent.as_ref(), ending_latent.as_ref())?;
    }
    // Appended end-keyframe tokens guide attention but are not part of the decoded clip.
    latent = latent.narrow(1, 0, f * h * w)?.contiguous()?;
    let (gpu, host, disk) = model.stats();
    dev.synchronize()?;
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    drop(model);
    drop(context);
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
    let mut encoder = command(&r.ffmpeg)
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
            "-an",
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
        .spawn()?;
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
        return Err(e);
    }
    if !encoder.wait()?.success() {
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
        ("audio", Json::Bool(false)),
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
