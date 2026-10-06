use crate::residency::Budget;
use crate::{reference::Reference, vision::VisionEncoder};
use crate::{schedule, text::TextEncoder, transformer::Transformer, vae::Vae};
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug)]
pub struct Request {
    pub base: PathBuf,
    pub transformer: PathBuf,
    pub text_encoder: Option<PathBuf>,
    pub model: Option<String>,
    pub adapter: Option<PathBuf>,
    /// More LoRA adapters, each with its strength, applied with the turbo one.
    pub loras: Vec<(PathBuf, f64)>,
    pub output: PathBuf,
    pub prompts: Vec<String>,
    pub count: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    pub device: usize,
    pub cfg: f64,
    pub images: Vec<PathBuf>,
    pub reference_size: usize,
    /// Where the transformer and text-encoder blocks live: GPU, RAM or SSD.
    pub budget: Budget,
    /// The transformer on WebGPU (`backend` "webgpu": any GPU wgpu reaches, OAIY_WEBGPU_ADAPTER picking one), the text
    /// encoder and VAE on the build's own device.
    pub webgpu: bool,
}
impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let string = |key: &str| {
            j.get(key)
                .and_then(Json::as_str)
                .filter(|s| !s.trim().is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("{key} must be a nonempty string"))
        };
        let number = |key: &str, default: i64| -> std::result::Result<usize, String> {
            let n = match j.get(key) {
                None => default,
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| format!("{key} must be an integer"))?,
            };
            usize::try_from(n).map_err(|_| format!("{key} must be nonnegative"))
        };
        let adapter = match j.get("adapter") {
            None | Some(Json::Null) => None,
            Some(_) => Some(PathBuf::from(string("adapter")?)),
        };
        let prompts = if let Some(v) = j.get("prompts") {
            v.as_array()
                .ok_or("prompts must be an array")?
                .iter()
                .map(|p| {
                    p.as_str()
                        .filter(|s| !s.trim().is_empty())
                        .map(str::to_owned)
                        .ok_or("prompts must contain nonempty strings".to_owned())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        } else {
            vec![string("prompt")?]
        };
        let cfg = match j.get("cfg") {
            None => {
                if adapter.is_some() {
                    1.
                } else {
                    6.
                }
            }
            Some(v) => v.as_f64().ok_or("cfg must be a number")?,
        };
        let r = Self {
            images: match j.get("images") {
                None => Vec::new(),
                Some(v) => v
                    .as_array()
                    .ok_or("images must be an array of up to three local paths")?
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .filter(|s| !s.trim().is_empty())
                            .map(PathBuf::from)
                            .ok_or_else(|| "images must contain nonempty local paths".to_owned())
                    })
                    .collect::<std::result::Result<_, _>>()?,
            },
            reference_size: number("reference_size", 1024)?,
            base: string("base")?.into(),
            transformer: string("transformer")?.into(),
            text_encoder: match j.get("text_encoder") { None | Some(Json::Null) => None, _ => Some(string("text_encoder")?.into()) },
            model: j.get("model").and_then(Json::as_str).map(str::to_owned),
            adapter,
            loras: parse_loras(j)?,
            output: string("output_dir")?.into(),
            prompts,
            count: number("n", 1)?,
            width: number("width", 1024)?,
            height: number("height", 1024)?,
            steps: number(
                "steps",
                if j.get("adapter").and_then(Json::as_str).is_some() {
                    6
                } else {
                    40
                },
            )?,
            seed: number("seed", 0)? as u64,
            device: number("device", 0)?,
            cfg,
            budget: Budget::parse(j)?,
            webgpu: match j.get("backend").and_then(Json::as_str) {
                None | Some("cuda" | "cpu") => false,
                Some("webgpu") if cfg!(feature = "webgpu") => true,
                Some("webgpu") => return Err("this build has no WebGPU (the webgpu feature)".into()),
                Some(other) => return Err(format!("backend must be cuda, cpu or webgpu, not {other}")),
            },
        };
        r.validate()?;
        Ok(r)
    }
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.images.len() > 3 {
            return Err("images accepts zero, one, two, or three references".into());
        }
        if !(256..=1024).contains(&self.reference_size) || self.reference_size % 32 != 0 {
            return Err("reference_size must be a multiple of 32 between 256 and 1024".into());
        }
        for p in &self.images {
            if !p.is_file() {
                return Err(format!("reference image not found: {}", p.display()));
            }
        }
        if !(1..=1000).contains(&self.count) {
            return Err("n must be between 1 and 1000".into());
        }
        if self.prompts.is_empty() || (self.prompts.len() != 1 && self.prompts.len() != self.count)
        {
            return Err("provide one reusable prompt or exactly n prompts".into());
        }
        if self
            .prompts
            .iter()
            .any(|s| s.len() > 16_384 || s.trim().is_empty())
        {
            return Err("prompts must contain 1..16384 bytes".into());
        }
        if [self.width, self.height]
            .iter()
            .any(|n| !(256..=2048).contains(n) || n % 32 != 0)
        {
            return Err("width and height must be multiples of 32 between 256 and 2048".into());
        }
        if !self.cfg.is_finite()
            || !(1. ..=10.).contains(&self.cfg)
            || (self.adapter.is_some() && self.cfg != 1.)
        {
            return Err("CFG must be 1 for turbo, or between 1 and 10 for base".into());
        }
        schedule::sigmas(
            self.steps,
            self.width * self.height / 256,
            self.adapter.is_some(),
        )?;
        if self
            .seed
            .checked_add(self.count as u64 - 1)
            .is_none_or(|s| s > i64::MAX as u64)
        {
            return Err("seed range exceeds signed 64-bit range".into());
        }
        for (path, strength) in &self.loras {
            if !strength.is_finite() || !(-4. ..=4.).contains(strength) {
                return Err(format!("LoRA strength must be between -4 and 4 ({})", path.display()));
            }
        }
        if self.webgpu && !self.images.is_empty() {
            return Err("reference images are not supported on WebGPU yet".into());
        }
        for path in [&self.base, &self.transformer]
            .into_iter()
            .chain(self.adapter.iter()).chain(self.text_encoder.iter()).chain(self.loras.iter().map(|(p, _)| p))
        {
            if !path.exists() {
                return Err(format!("missing weights: {}", path.display()));
            }
        }
        Ok(())
    }
}

/// The transformer on its device: Candle's (CUDA or the CPU) or the WebGPU chain's.
enum Model {
    Candle(Transformer),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::qwen_wgpu::WgpuTransformer),
}

/// The VAE's decoder on its device: Candle's or the WebGPU chain's.
enum Decoder {
    Candle(Vae),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::vae_wgpu::WgpuVae),
}

impl Decoder {
    /// The picture of `latent` (`[1, h w, 64]`): `[1, 4, 16 h, 16 w]`.
    fn decode(&self, latent: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        match self {
            Decoder::Candle(v) => v.decode(latent, h, w),
            #[cfg(feature = "webgpu")]
            Decoder::Wgpu(v) => v.decode(latent, h, w),
        }
    }
}

/// A prompt's conditioning, as its model made it.
enum Prefix {
    Candle(crate::transformer::Prefix),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::qwen_wgpu::WgpuPrefix),
}

impl Model {
    fn prepare(&mut self, text: &crate::text::Conditioning, refs: &[(Tensor, usize, usize)]) -> Result<Prefix> {
        match self {
            Model::Candle(m) => m.prepare(text, refs).map(Prefix::Candle),
            #[cfg(feature = "webgpu")]
            Model::Wgpu(m) => m.prepare(text, refs).map(Prefix::Wgpu),
        }
    }

    /// The velocity at `sigma`, on `latent`'s device and in its dtype.
    fn conditioned(&mut self, latent: &Tensor, prefix: &Prefix, sigma: f64, h: usize, w: usize) -> Result<Tensor> {
        match (self, prefix) {
            (Model::Candle(m), Prefix::Candle(p)) => m.conditioned(latent, p, sigma, h, w),
            #[cfg(feature = "webgpu")]
            (Model::Wgpu(m), Prefix::Wgpu(p)) => m.conditioned(latent, p, sigma, h, w)?.to_device(latent.device())?.to_dtype(latent.dtype()),
            #[cfg(feature = "webgpu")]
            _ => candle_core::bail!("a prompt's conditioning from another model"),
        }
    }

    /// Let go of what a step keeps for the next (the VAE's decode wants the room).
    fn release_scratch(&mut self) {
        match self {
            Model::Candle(_) => {}
            #[cfg(feature = "webgpu")]
            Model::Wgpu(m) => m.release_scratch(),
        }
    }

    fn lora_notes(&self) -> &[String] {
        match self {
            Model::Candle(m) => m.lora_notes(),
            #[cfg(feature = "webgpu")]
            Model::Wgpu(m) => m.lora_notes(),
        }
    }

    fn residency(&self) -> Json {
        match self {
            Model::Candle(m) => m.residency(),
            #[cfg(feature = "webgpu")]
            Model::Wgpu(_) => Json::obj([("device", Json::str("webgpu"))]),
        }
    }
}

/// `loras`: LoRA adapters as `[{"path": …, "strength": 0.8}]` (strength 1 when left out), or plain paths.
pub(crate) fn parse_loras(j: &Json) -> std::result::Result<Vec<(PathBuf, f64)>, String> {
    let Some(list) = j.get("loras") else { return Ok(Vec::new()) };
    if matches!(list, Json::Null) {
        return Ok(Vec::new());
    }
    list.as_array()
        .ok_or("loras must be an array of {path, strength}")?
        .iter()
        .map(|l| {
            let (path, strength) = match l {
                Json::Str(_) => (l.as_str(), None),
                _ => (l.get("path").and_then(Json::as_str), l.get("strength")),
            };
            let path = path.filter(|p| !p.trim().is_empty()).ok_or("each LoRA needs a path")?;
            let strength = match strength {
                None | Some(Json::Null) => 1.,
                Some(v) => v.as_f64().ok_or("a LoRA's strength must be a number")?,
            };
            Ok((PathBuf::from(path), strength))
        })
        .collect()
}

/// Prompt encoding and diffusion are sequential to bound peak VRAM. All images
/// in a batch reuse one transformer and VAE; completed images survive failure.
pub fn generate(r: &Request, mut event: impl FnMut(Json)) -> Result<Json> {
    r.validate().map_err(candle_core::Error::Msg)?;
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(r.device)?;
    #[cfg(not(feature = "cuda"))]
    let dev = Device::Cpu;
    let dtype = if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    std::fs::create_dir_all(&r.output)?;
    let id = format!(
        "batch-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(candle_core::Error::wrap)?
            .as_nanos()
    );
    let out = r.output.join(id);
    std::fs::create_dir(&out)?;
    let mut manifest = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("manifest.jsonl"))?;
    let t = Instant::now();
    let mut references = Vec::new();
    let mut features = Vec::new();
    if !r.images.is_empty() {
        event(Json::obj([("stage", Json::str("encoding_references"))]));
        let images = r
            .images
            .iter()
            .map(|p| Reference::load(p, r.reference_size))
            .collect::<Result<Vec<_>>>()?;
        let encoder = Vae::load_encoder(&r.base, &dev, dtype)?;
        for image in &images {
            references.push((
                encoder.encode(&image.pixels(&dev, dtype)?)?,
                image.h / 16,
                image.w / 16,
            ));
        }
        drop(encoder);
        let vision = VisionEncoder::load(r.text_encoder.as_deref().unwrap_or(&r.base.join("text_encoder")), &dev, dtype)?;
        for image in &images {
            features.push(vision.encode(image)?);
        }
        drop(vision);
    }
    dev.synchronize()?;
    let reference_encoding_seconds = t.elapsed().as_secs_f64();
    event(Json::obj([("stage", Json::str("loading_text_encoder"))]));
    let text_load_start = Instant::now();
    let mut encoder = TextEncoder::load(&r.base, r.text_encoder.as_deref(), &dev, dtype, &r.budget)?;
    dev.synchronize()?;
    let text_load_seconds = text_load_start.elapsed().as_secs_f64();
    let encoding_start = Instant::now();
    let mut embeddings = Vec::new();
    for (i, prompt) in r.prompts.iter().enumerate() {
        embeddings.push(encoder.encode(prompt, &features)?);
        event(Json::obj([
            ("stage", Json::str("encoding")),
            ("completed", Json::Int((i + 1) as i64)),
        ]));
    }
    let negative = if r.cfg > 1. {
        Some(encoder.encode(" ", &features)?)
    } else {
        None
    };
    let text_residency = encoder.residency();
    drop(encoder);
    drop(features);
    dev.synchronize()?;
    let encoding_seconds = encoding_start.elapsed().as_secs_f64();
    event(Json::obj([("stage", Json::str("loading_transformer"))]));
    let load_start = Instant::now();
    let progress = |n: usize| {
        event(Json::obj([
            ("stage", Json::str("loading_transformer")),
            ("block", Json::Int(n as i64)),
            ("blocks", Json::Int(32)),
        ]))
    };
    #[cfg(feature = "webgpu")]
    let mut model = if r.webgpu {
        Model::Wgpu(crate::qwen_wgpu::WgpuTransformer::load(&r.transformer, r.adapter.as_deref(), &r.loras, progress)?)
    } else {
        Model::Candle(Transformer::load(&r.transformer, r.adapter.as_deref(), &r.loras, &dev, dtype, &r.budget, progress)?)
    };
    #[cfg(not(feature = "webgpu"))]
    let mut model = Model::Candle(Transformer::load(&r.transformer, r.adapter.as_deref(), &r.loras, &dev, dtype, &r.budget, progress)?);
    dev.synchronize()?;
    let transformer_load_seconds = load_start.elapsed().as_secs_f64();
    for note in model.lora_notes() {
        event(Json::obj([("stage", Json::str("lora_note")), ("note", Json::str(note))]));
    }
    event(Json::obj([("stage", Json::str("loading_vae"))]));
    let load_start = Instant::now();
    #[cfg(feature = "webgpu")]
    let vae = if r.webgpu { Decoder::Wgpu(crate::vae_wgpu::WgpuVae::load(&r.base)?) } else { Decoder::Candle(Vae::load(&r.base, &dev, dtype)?) };
    #[cfg(not(feature = "webgpu"))]
    let vae = Decoder::Candle(Vae::load(&r.base, &dev, dtype)?);
    dev.synchronize()?;
    let vae_load_seconds = load_start.elapsed().as_secs_f64();
    let (h, w) = (r.height / 16, r.width / 16);
    let sigmas =
        schedule::sigmas(r.steps, h * w, r.adapter.is_some()).map_err(candle_core::Error::Msg)?;
    let mut files = Vec::new();
    let mut prefix = None;
    let negative_prefix = negative
        .as_ref()
        .map(|n| model.prepare(n, &references))
        .transpose()?;
    for i in 0..r.count {
        if prefix.is_none() || embeddings.len() > 1 {
            drop(prefix.take());
            event(Json::obj([("stage", Json::str("preparing_conditioning"))]));
            prefix = Some(model.prepare(
                &embeddings[if embeddings.len() == 1 { 0 } else { i }],
                &references,
            )?);
        }
        let image_start = Instant::now();
        let pi = if embeddings.len() == 1 { 0 } else { i };
        let seed = r.seed + i as u64;
        let mut latent =
            Tensor::from_vec(noise(seed, h * w * 64), (1, h * w, 64), &dev)?.to_dtype(dtype)?;
        for (step, pair) in sigmas.windows(2).enumerate() {
            let step_start = Instant::now();
            let mut velocity =
                model.conditioned(&latent, prefix.as_ref().unwrap(), pair[0], h, w)?;
            if let Some(neg) = &negative_prefix {
                let uncond = model.conditioned(&latent, neg, pair[0], h, w)?;
                velocity = (&uncond + ((velocity - &uncond)? * r.cfg)?)?;
            }
            latent = (latent.to_dtype(DType::F32)?
                + (velocity.to_dtype(DType::F32)? * (pair[1] - pair[0]))?)?
                .to_dtype(dtype)?;
            dev.synchronize()?;
            event(Json::obj([
                ("stage", Json::str("sampling")),
                ("image", Json::Int((i + 1) as i64)),
                ("step", Json::Int((step + 1) as i64)),
                ("steps", Json::Int(r.steps as i64)),
                ("seconds", Json::Num(step_start.elapsed().as_secs_f64())),
            ]));
        }
        let sampling_seconds = image_start.elapsed().as_secs_f64();
        let decode_start = Instant::now();
        model.release_scratch();
        let rgba = vae
            .decode(&latent, h, w)?
            .to_dtype(DType::F32)?
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let decode_seconds = decode_start.elapsed().as_secs_f64();
        let save_start = Instant::now();
        if rgba.iter().any(|x| !x.is_finite()) {
            candle_core::bail!("non-finite VAE output for image {}", i + 1);
        }
        let bytes: Vec<u8> = rgba
            .into_iter()
            .map(|x| ((x.clamp(-1., 1.) + 1.) * 127.5).round() as u8)
            .collect();
        let path = out.join(format!("image-{:04}.png", i + 1));
        save_png(&path, &bytes, r.width as u32, r.height as u32)?;
        let record = Json::obj([
            ("path", Json::str(path.to_string_lossy())),
            ("seed", Json::Int(seed as i64)),
            ("prompt", Json::str(&r.prompts[pi])),
            (
                "images",
                Json::Arr(
                    r.images
                        .iter()
                        .map(|p| Json::str(p.to_string_lossy()))
                        .collect(),
                ),
            ),
            ("steps", Json::Int(r.steps as i64)),
            ("sampling_seconds", Json::Num(sampling_seconds)),
            ("decode_seconds", Json::Num(decode_seconds)),
            (
                "save_seconds",
                Json::Num(save_start.elapsed().as_secs_f64()),
            ),
            (
                "image_seconds",
                Json::Num(image_start.elapsed().as_secs_f64()),
            ),
            ("transformer", Json::str(r.transformer.to_string_lossy())),
            ("text_encoder", Json::str(r.text_encoder.clone().unwrap_or_else(||r.base.join("text_encoder")).to_string_lossy())),
            ("model", r.model.as_ref().map(Json::str).unwrap_or(Json::Null)),
            (
                "adapter",
                r.adapter
                    .as_ref()
                    .map_or(Json::Null, |p| Json::str(p.to_string_lossy())),
            ),
            (
                "loras",
                Json::Arr(r.loras.iter().map(|(p, s)| Json::obj([("path", Json::str(p.to_string_lossy())), ("strength", Json::Num(*s))])).collect()),
            ),
        ]);
        writeln!(manifest, "{}", record.to_json())?;
        manifest.flush()?;
        event(Json::obj([
            ("stage", Json::str("image_saved")),
            ("image", Json::Int((i + 1) as i64)),
            ("path", Json::str(path.to_string_lossy())),
            ("sampling_seconds", Json::Num(sampling_seconds)),
            ("decode_seconds", Json::Num(decode_seconds)),
            (
                "image_seconds",
                Json::Num(image_start.elapsed().as_secs_f64()),
            ),
        ]));
        files.push(Json::obj([
            ("path", Json::str(path.to_string_lossy())),
            ("seed", Json::Int(seed as i64)),
            ("steps", Json::Int(r.steps as i64)),
        ]));
    }
    Ok(Json::obj([
        ("data", Json::Arr(files)),
        ("output_dir", Json::str(out.to_string_lossy())),
        ("seconds", Json::Num(t.elapsed().as_secs_f64())),
        ("text_load_seconds", Json::Num(text_load_seconds)),
        ("encoding_seconds", Json::Num(encoding_seconds)),
        (
            "reference_encoding_seconds",
            Json::Num(reference_encoding_seconds),
        ),
        (
            "transformer_load_seconds",
            Json::Num(transformer_load_seconds),
        ),
        ("vae_load_seconds", Json::Num(vae_load_seconds)),
        (
            "residency",
            Json::obj([
                ("budget", r.budget.to_json()),
                ("transformer", model.residency()),
                ("text_encoder", text_residency),
            ]),
        ),
    ]))
}
fn save_png(path: &Path, bytes: &[u8], width: u32, height: u32) -> Result<()> {
    use image::ImageEncoder;
    let file = File::options().create_new(true).write(true).open(path)?;
    image::codecs::png::PngEncoder::new(file)
        .write_image(bytes, width, height, image::ExtendedColorType::Rgba8)
        .map_err(candle_core::Error::wrap)
}
pub(crate) fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let radius = (-2. * uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * uniform();
        out.push((radius * angle.cos()) as f32);
        if out.len() < n {
            out.push((radius * angle.sin()) as f32);
        }
    }
    out
}
#[test]
fn seeds_are_repeatable_and_distinct() {
    assert_eq!(noise(7, 13), noise(7, 13));
    assert_ne!(noise(7, 13), noise(8, 13));
    assert!(noise(0, 10000).iter().all(|x| x.is_finite()));
}
