//! Native single-checkpoint SDXL text-to-image. Architecture modules were ported
//! from the user's F2I plugin; loading and the worker protocol are Oaiy-owned.
pub mod config;
pub mod micro_cond;
pub mod scheduler;
pub mod text_encoder;
pub mod unet;
pub mod vae;
pub mod vae_layers;

use crate::residency::{Budget, Memory};
use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use oaiy_engine::json::Json;
use std::{collections::HashMap, io::Write, path::PathBuf, time::Instant};

#[derive(Debug)]
pub struct Request {
    pub checkpoint: PathBuf,
    pub tokenizer: PathBuf,
    pub output: PathBuf,
    pub prompts: Vec<String>,
    pub negative: String,
    pub count: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg: f64,
    pub seed: u64,
    pub device: usize,
    pub clip_skip: usize,
    /// Resident on the GPU, or staged component by component (see `Stages`).
    pub budget: Budget,
    /// The UNet and the VAE's decoder on WebGPU (`backend` "webgpu": any GPU wgpu reaches), the text encoders on
    /// the host.
    pub webgpu: bool,
}
impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let webgpu = match j.get("backend").and_then(Json::as_str) {
            None | Some("cuda" | "cpu") => false,
            Some("webgpu") if cfg!(feature = "webgpu") => true,
            Some("webgpu") => return Err("this build has no WebGPU (the webgpu feature)".into()),
            Some(other) => return Err(format!("backend must be cuda, cpu or webgpu, not {other}")),
        };
        let text = |k| {
            j.get(k)
                .and_then(Json::as_str)
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| format!("{k} must be a nonempty string"))
        };
        let number = |k, default, min, max| {
            let n = match j.get(k) {
                None => default,
                Some(v) => v
                    .as_i64()
                    .ok_or_else(|| format!("{k} must be an integer"))?,
            };
            if !(min..=max).contains(&n) {
                return Err(format!("{k} must be in {min}..{max}"));
            }
            Ok(n as usize)
        };
        let count = number("n", 1, 1, 1000)?;
        let prompts = match j.get("prompts") {
            Some(v) => v
                .as_array()
                .ok_or("prompts must be an array")?
                .iter()
                .map(|p| {
                    p.as_str()
                        .filter(|s| !s.trim().is_empty() && s.len() <= 16384)
                        .map(str::to_owned)
                        .ok_or("invalid prompt".to_owned())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            None => vec![text("prompt")?.to_owned()],
        };
        if (prompts.len() != 1 && prompts.len() != count) || prompts.iter().any(|p| p.len() > 16384)
        {
            return Err("provide one prompt or n prompts, each up to 16384 bytes".into());
        }
        let width = number("width", 1024, 256, 2048)?;
        let height = number("height", 1024, 256, 2048)?;
        if width % 64 != 0 || height % 64 != 0 {
            return Err("SDXL dimensions must be multiples of 64".into());
        }
        let cfg = match j.get("cfg") {
            None => 2.5,
            Some(v) => v.as_f64().ok_or("cfg must be numeric")?,
        };
        if !cfg.is_finite() || !(1.0..=30.0).contains(&cfg) {
            return Err("cfg must be between 1 and 30".into());
        }
        for (key, expected) in [("sampler", "dpmpp_2m"), ("scheduler", "karras")] {
            if j.get(key).is_some_and(|v| v.as_str() != Some(expected)) {
                return Err(format!("SDXL {key} must be {expected}"));
            }
        }
        for k in ["image", "images", "adapter"] {
            if j.get(k).is_some_and(|v| {
                !matches!(v, Json::Null) && !v.as_array().is_some_and(|a| a.is_empty())
            }) {
                return Err(format!(
                    "SDXL {k} is not supported; this path is text-to-image without LoRA"
                ));
            }
        }
        if j.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err("SDXL does not use the Qwen turbo adapter".into());
        }
        let negative = match j.get("negative_prompt") {
            None => String::new(),
            Some(v) => v
                .as_str()
                .filter(|s| s.len() <= 16384)
                .ok_or("negative_prompt must be a string up to 16384 bytes")?
                .into(),
        };
        let r = Self {
            checkpoint: text("checkpoint")?.into(),
            tokenizer: text("tokenizer")?.into(),
            output: text("output_dir")?.into(),
            prompts,
            negative,
            count,
            width,
            height,
            cfg,
            steps: number("steps", 16, 2, 100)?,
            seed: number("seed", 0, 0, i64::MAX - count as i64)? as u64,
            device: number("device", 0, 0, 255)?,
            clip_skip: number("clip_skip", 1, 1, 11)?,
            budget: Budget::parse(j)?,
            webgpu,
        };
        for p in [&r.checkpoint, &r.tokenizer] {
            if !p.is_file() {
                return Err(format!("missing SDXL file: {}", p.display()));
            }
        }
        Ok(r)
    }
}

const CLIP_L: &str = "conditioner.embedders.0.transformer.";
const CLIP_G: &str = "conditioner.embedders.1.model.";
const UNET: &str = "model.diffusion_model.";
const VAE: &str = "first_stage_model.";

/// One SDXL component's tensors, read on the CPU (that is where the checkpoint's
/// FP16 bytes are converted) and placed on `dev`.
fn component(
    w: &mut Weights,
    prefix: &str,
    dev: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    for name in w.names() {
        if let Some(key) = name.strip_prefix(prefix) {
            if key.ends_with("position_ids") || key == "logit_scale" {
                continue;
            }
            tensors.insert(
                key.to_owned(),
                w.tensor(&name, &Device::Cpu, dtype)?.to_device(dev)?,
            );
        }
    }
    if tensors.is_empty() {
        candle_core::bail!("checkpoint is missing SDXL component {prefix}");
    }
    Ok(tensors)
}

/// SDXL's four components, fetched per stage when they are not all resident.
///
/// The UNet's skip connections make block streaming awkward, so SDXL tiers by
/// component instead: in `ram`/`ssd` mode (or `auto` when the checkpoint does not
/// fit the VRAM budget) only the stage running holds VRAM -- the two CLIP encoders
/// while prompts are encoded, then the UNet, then the VAE. Host copies of each
/// component are kept while the RAM budget allows (`ram`/`auto`); the rest are
/// read from the SSD again for each image.
struct Stages {
    weights: Weights,
    host: HashMap<&'static str, HashMap<String, Tensor>>,
    ram_left: u64,
    host_bytes: u64,
    streamed_bytes: u64,
}
impl Stages {
    fn fetch(&mut self, prefix: &'static str, dev: &Device, dtype: DType) -> Result<HashMap<String, Tensor>> {
        if let Some(host) = self.host.get(prefix) {
            return host.iter().map(|(k, t)| Ok((k.clone(), t.to_device(dev)?))).collect();
        }
        let cpu = component(&mut self.weights, prefix, &Device::Cpu, dtype)?;
        let bytes: u64 = cpu.values().map(crate::weights::bytes_of).sum();
        self.streamed_bytes += bytes;
        let placed = cpu.iter().map(|(k, t)| Ok((k.clone(), t.to_device(dev)?))).collect::<Result<_>>()?;
        if bytes <= self.ram_left {
            self.ram_left -= bytes;
            self.host_bytes += bytes;
            self.host.insert(prefix, cpu);
        }
        Ok(placed)
    }
    fn clips(&mut self, dev: &Device, dtype: DType) -> Result<(text_encoder::ClipL, text_encoder::ClipG)> {
        Ok((
            text_encoder::ClipL::load(
                &config::CLIPConfig::clip_l_14_336(),
                VarBuilder::from_tensors(self.fetch(CLIP_L, dev, dtype)?, dtype, dev),
            )?,
            text_encoder::ClipG::load(
                &config::CLIPConfig::open_clip_g_14_laion2b(),
                VarBuilder::from_tensors(self.fetch(CLIP_G, dev, dtype)?, dtype, dev),
            )?,
        ))
    }
    fn unet(&mut self, dev: &Device, dtype: DType) -> Result<unet::UNet2DConditionModel> {
        unet::UNet2DConditionModel::load(
            &config::UNetConfig::sdxl_1_0(),
            VarBuilder::from_tensors(self.fetch(UNET, dev, dtype)?, dtype, dev),
        )
    }
    fn vae(&mut self, dev: &Device) -> Result<vae::AutoencoderKL> {
        // FP32 decoding avoids overflow in original SDXL VAEs.
        vae::AutoencoderKL::load(
            &config::VaeConfig::sdxl_default(),
            VarBuilder::from_tensors(self.fetch(VAE, dev, DType::F32)?, DType::F32, dev),
        )
    }
}

/// Every component on the GPU for the whole batch.
struct Resident {
    cl: text_encoder::ClipL,
    cg: text_encoder::ClipG,
    unet: unet::UNet2DConditionModel,
    vae: vae::AutoencoderKL,
}

fn clip_ids(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    pad: u32,
    dev: &Device,
) -> Result<(Tensor, usize, bool)> {
    let encoded = tokenizer
        .encode(prompt, false)
        .map_err(candle_core::Error::wrap)?;
    let raw = encoded.get_ids();
    let n = raw.len().min(75);
    let mut ids = vec![49406];
    ids.extend_from_slice(&raw[..n]);
    ids.push(49407);
    ids.resize(77, pad);
    Ok((Tensor::from_vec(ids, (1, 77), dev)?, n + 1, raw.len() > 75))
}

/// Prompt and negative through both CLIPs: (context, pooled label, truncated).
fn encode_prompt(
    r: &Request,
    tokenizer: &tokenizers::Tokenizer,
    cl: &text_encoder::ClipL,
    cg: &text_encoder::ClipG,
    prompt: &str,
    dev: &Device,
) -> Result<(Tensor, Tensor, bool)> {
    let mut encoded = Vec::new();
    let mut truncated = false;
    for p in [prompt, &r.negative] {
        let (l, _, tl) = clip_ids(tokenizer, p, 49407, dev)?;
        // OpenCLIP-G pads with zero; CLIP-L pads with EOT.
        let (g, eot, tg) = clip_ids(tokenizer, p, 0, dev)?;
        encoded.push(text_encoder::dual_encode(cl, cg, &l, &g, &[eot], r.clip_skip)?);
        truncated |= tl || tg;
    }
    let context = Tensor::cat(&[&encoded[0].context, &encoded[1].context], 0)?;
    // SDXL's six micro-conditioning values are height, width, top, left, height, width.
    let size = (r.height as u32, r.width as u32);
    let y = Tensor::cat(
        &encoded
            .iter()
            .map(|e| micro_cond::build_label_y(&e.pooled, size, (0, 0), size))
            .collect::<Result<Vec<_>>>()?,
        0,
    )?;
    Ok((context, y, truncated))
}

/// The official CLIP tokenizer the request names, neither padding nor truncating.
fn open_tokenizer(r: &Request) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(&r.tokenizer).map_err(candle_core::Error::wrap)?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(candle_core::Error::wrap)?;
    if tokenizer.token_to_id("<|startoftext|>") != Some(49406)
        || tokenizer.token_to_id("<|endoftext|>") != Some(49407)
    {
        candle_core::bail!("SDXL requires the official CLIP tokenizer");
    }
    Ok(tokenizer)
}

/// A new directory for this batch's images under the request's, and its manifest.
fn batch_dir(r: &Request) -> Result<(PathBuf, std::fs::File)> {
    std::fs::create_dir_all(&r.output)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(candle_core::Error::wrap)?
        .as_nanos();
    let out = r
        .output
        .join(format!("batch-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&out)?;
    let manifest = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("manifest.jsonl"))?;
    Ok((out, manifest))
}

/// Image `i`'s picture (`decoded`: `[1, 3, height, width]` in -1..1) saved in `out`, its record written to the
/// manifest and returned.
#[allow(clippy::too_many_arguments)]
fn save_image(r: &Request, out: &std::path::Path, manifest: &mut std::fs::File, i: usize, prompt: &str, seed: u64, truncated: bool, decoded: &Tensor, sampling_seconds: f64, decode_seconds: f64, image_clock: Instant, event: &mut impl FnMut(Json)) -> Result<Json> {
    let rgb = decoded
        .squeeze(0)?
        .permute((1, 2, 0))?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if rgb.iter().any(|v| !v.is_finite()) {
        candle_core::bail!("non-finite SDXL VAE output");
    }
    let bytes: Vec<u8> = rgb
        .iter()
        .map(|v| ((v.clamp(-1., 1.) + 1.) * 127.5).round() as u8)
        .collect();
    let image = image::RgbImage::from_raw(r.width as u32, r.height as u32, bytes)
        .ok_or_else(|| candle_core::Error::Msg("bad RGB dimensions".into()))?;
    let path = out.join(format!("image-{:04}.png", i + 1));
    image.save(&path).map_err(candle_core::Error::wrap)?;
    let record = Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("prompt", Json::str(prompt)),
        ("negative_prompt", Json::str(&r.negative)),
        ("seed", Json::Int(seed as i64)),
        ("steps", Json::Int(r.steps as i64)),
        ("cfg", Json::Num(r.cfg)),
        ("sampler", Json::str("dpmpp_2m")),
        ("scheduler", Json::str("karras")),
        ("width", Json::Int(r.width as i64)),
        ("height", Json::Int(r.height as i64)),
        ("clip_skip", Json::Int(r.clip_skip as i64)),
        ("prompt_truncated", Json::Bool(truncated)),
        ("checkpoint", Json::str(r.checkpoint.to_string_lossy())),
        ("sampling_seconds", Json::Num(sampling_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        (
            "image_seconds",
            Json::Num(image_clock.elapsed().as_secs_f64()),
        ),
    ]);
    writeln!(manifest, "{}", record.to_json())?;
    manifest.flush()?;
    event(Json::obj([
        ("stage", Json::str("image_saved")),
        ("image", Json::Int(i as i64 + 1)),
        ("path", Json::str(path.to_string_lossy())),
        ("prompt_truncated", Json::Bool(truncated)),
    ]));
    Ok(record)
}

/// [`generate`] with the UNet and the VAE's decoder on WebGPU ([`crate::sdxl_wgpu::WgpuUnet`],
/// [`crate::sdxl_vae_wgpu::WgpuSdxlVae`]): the two CLIP encoders on the host through Candle in f32 (a prompt's two
/// encodings, against a step's two passes of the UNet), the noise the host's from the seed (Candle seeds no generator on the CPU), a guided step's prompt and
/// negative prompt two passes of one recording, a guidance of 1 the prompt's pass alone.
#[cfg(feature = "webgpu")]
fn generate_webgpu(r: &Request, mut event: impl FnMut(Json)) -> Result<Json> {
    let clock = Instant::now();
    event(Json::obj([
        ("stage", Json::str("initializing_device")),
        ("device", Json::Int(r.device as i64)),
        ("backend", Json::str("webgpu")),
    ]));
    let dev = Device::Cpu;
    let tokenizer = open_tokenizer(r)?;
    let (out, mut manifest) = batch_dir(r)?;
    event(Json::obj([("stage", Json::str("loading_sdxl"))]));
    let mut stages = Stages {
        weights: Weights::open(&r.checkpoint)?,
        host: HashMap::new(),
        ram_left: 0,
        host_bytes: 0,
        streamed_bytes: 0,
    };
    let unet = crate::sdxl_wgpu::WgpuUnet::load(&mut stages.weights, UNET, &config::UNetConfig::sdxl_1_0(), r.device)?;
    let vae = crate::sdxl_vae_wgpu::WgpuSdxlVae::load_on(&mut stages.weights, VAE, &config::VaeConfig::sdxl_default(), unet.backend().clone())?;
    let (cl, cg) = stages.clips(&dev, DType::F32)?;
    let load_seconds = clock.elapsed().as_secs_f64();
    let sigmas = scheduler::sdxl_default_sigmas(r.steps);
    let (lh, lw) = (r.height / 8, r.width / 8);
    let row = |t: &Tensor, i: usize| -> Result<Vec<f32>> { t.narrow(0, i, 1)?.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>() };
    let mut files = Vec::new();
    for i in 0..r.count {
        let image_clock = Instant::now();
        let prompt = &r.prompts[if r.prompts.len() == 1 { 0 } else { i }];
        event(Json::obj([
            ("stage", Json::str("encoding_prompt")),
            ("image", Json::Int(i as i64 + 1)),
        ]));
        let (context, y, truncated) = encode_prompt(r, &tokenizer, &cl, &cg, prompt, &dev)?;
        // the prompt's keys and values for every cross-attention, and the negative prompt's where it guides
        let mut conds = vec![unet.prepare(&row(&context, 0)?, &row(&y, 0)?)?];
        if r.cfg != 1.0 {
            conds.push(unet.prepare(&row(&context, 1)?, &row(&y, 1)?)?);
        }
        let conds: Vec<&crate::sdxl_wgpu::Cond> = conds.iter().collect();
        let seed = r.seed + i as u64;
        let mut x = Tensor::from_vec(crate::pipeline::noise(seed, 4 * lh * lw), (1, 4, lh, lw), &dev)?.affine(sigmas[0], 0.)?;
        let mut state = scheduler::SamplerState::default();
        let sampling = Instant::now();
        for step in 0..r.steps {
            let scaled = scheduler::scale_input_for_euler(&x, sigmas[step])?;
            let t = scheduler::sigma_to_timestep(sigmas[step], 1000) as f32;
            // (the latent as its pixels' rows of channels, the UNet's tokens)
            let rows = scaled.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let eps = unet.eps(&rows, lh, lw, t, &conds)?;
            let guided: Vec<f32> = match eps.get(1) {
                Some(neg) => eps[0].iter().zip(neg).map(|(c, n)| n + (c - n) * r.cfg as f32).collect(),
                None => eps[0].clone(),
            };
            let guided = Tensor::from_vec(guided, (lh, lw, 4), &dev)?.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
            x = scheduler::dpmpp_2m_step(&x, &guided, sigmas[step], sigmas[step + 1], &mut state)?;
            event(Json::obj([
                ("stage", Json::str("sampling")),
                ("image", Json::Int(i as i64 + 1)),
                ("step", Json::Int(step as i64 + 1)),
                ("steps", Json::Int(r.steps as i64)),
            ]));
        }
        let sampling_seconds = sampling.elapsed().as_secs_f64();
        event(Json::obj([
            ("stage", Json::str("decoding")),
            ("image", Json::Int(i as i64 + 1)),
        ]));
        let decode = Instant::now();
        // (the UNet's vectors let go first: the decoder's last blocks are the largest a picture makes)
        unet.forget();
        let (rgb, oh, ow) = vae.decode(&x.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?, lh, lw)?;
        let decoded = Tensor::from_vec(rgb, (oh, ow, 3), &dev)?.permute((2, 0, 1))?.unsqueeze(0)?;
        files.push(save_image(r, &out, &mut manifest, i, prompt, seed, truncated, &decoded, sampling_seconds, decode.elapsed().as_secs_f64(), image_clock, &mut event)?);
    }
    Ok(Json::obj([
        ("data", Json::Arr(files)),
        ("output_dir", Json::str(out.to_string_lossy())),
        ("load_seconds", Json::Num(load_seconds)),
        ("seconds", Json::Num(clock.elapsed().as_secs_f64())),
        ("backend", Json::str("webgpu")),
    ]))
}

pub fn generate(r: &Request, mut event: impl FnMut(Json)) -> Result<Json> {
    #[cfg(feature = "webgpu")]
    if r.webgpu {
        return generate_webgpu(r, event);
    }
    let clock = Instant::now();
    event(Json::obj([
        ("stage", Json::str("initializing_device")),
        ("device", Json::Int(r.device as i64)),
    ]));
    let dev = Device::Cpu;
    let dtype = if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let tokenizer = open_tokenizer(r)?;
    let (out, mut manifest) = batch_dir(r)?;
    event(Json::obj([("stage", Json::str("loading_sdxl"))]));
    // The checkpoint's FP16 size is what its components take on the device.
    let checkpoint_bytes = std::fs::metadata(&r.checkpoint)?.len();
    let resident = match r.budget.memory {
        Memory::Gpu => true,
        Memory::Auto => checkpoint_bytes <= r.budget.vram_limit(&dev)?,
        Memory::Ram | Memory::Ssd => false,
    };
    let mut stages = Stages {
        weights: Weights::open(&r.checkpoint)?,
        host: HashMap::new(),
        ram_left: if matches!(r.budget.memory, Memory::Auto | Memory::Ram) { r.budget.ram_bytes } else { 0 },
        host_bytes: 0,
        streamed_bytes: 0,
    };
    let models = if resident {
        let (cl, cg) = stages.clips(&dev, dtype)?;
        let unet = stages.unet(&dev, dtype)?;
        let vae = stages.vae(&dev)?;
        Some(Resident { cl, cg, unet, vae })
    } else {
        None
    };
    let load_seconds = clock.elapsed().as_secs_f64();
    let sigmas = scheduler::sdxl_default_sigmas(r.steps);
    let mut files = Vec::new();
    for i in 0..r.count {
        let image_clock = Instant::now();
        let prompt = &r.prompts[if r.prompts.len() == 1 { 0 } else { i }];
        event(Json::obj([
            ("stage", Json::str("encoding_prompt")),
            ("image", Json::Int(i as i64 + 1)),
        ]));
        let (context, y, truncated) = match &models {
            Some(m) => encode_prompt(r, &tokenizer, &m.cl, &m.cg, prompt, &dev)?,
            None => {
                let (cl, cg) = stages.clips(&dev, dtype)?;
                encode_prompt(r, &tokenizer, &cl, &cg, prompt, &dev)?
            }
        };
        let seed = r.seed + i as u64;
        dev.set_seed(seed)?;
        let mut x = scheduler::build_initial_noise(1, r.height / 8, r.width / 8, sigmas[0], &dev)?;
        let mut state = scheduler::SamplerState::default();
        let staged_unet = if models.is_none() {
            event(Json::obj([("stage", Json::str("loading_unet")), ("image", Json::Int(i as i64 + 1))]));
            Some(stages.unet(&dev, dtype)?)
        } else {
            None
        };
        let unet = match (&models, &staged_unet) {
            (Some(m), _) => &m.unet,
            (None, Some(u)) => u,
            (None, None) => unreachable!("a staged UNet is loaded whenever none is resident"),
        };
        let sampling = Instant::now();
        for step in 0..r.steps {
            let scaled = scheduler::scale_input_for_euler(&x, sigmas[step])?.to_dtype(dtype)?;
            let input = Tensor::cat(&[&scaled, &scaled], 0)?;
            let t = scheduler::sigma_to_timestep(sigmas[step], 1000) as f32;
            let t = Tensor::from_vec(vec![t, t], (2,), &dev)?;
            let eps = unet
                .forward(&input, &t, &context, &y)?
                .to_dtype(DType::F32)?;
            let cond = eps.narrow(0, 0, 1)?;
            let neg = eps.narrow(0, 1, 1)?;
            let guided = (&neg + (&cond - &neg)?.affine(r.cfg, 0.)?)?;
            x = scheduler::dpmpp_2m_step(&x, &guided, sigmas[step], sigmas[step + 1], &mut state)?;
            event(Json::obj([
                ("stage", Json::str("sampling")),
                ("image", Json::Int(i as i64 + 1)),
                ("step", Json::Int(step as i64 + 1)),
                ("steps", Json::Int(r.steps as i64)),
            ]));
        }
        drop(staged_unet);
        drop(context);
        dev.synchronize()?;
        let sampling_seconds = sampling.elapsed().as_secs_f64();
        event(Json::obj([
            ("stage", Json::str("decoding")),
            ("image", Json::Int(i as i64 + 1)),
        ]));
        let decode = Instant::now();
        let decoded = match &models {
            Some(m) => m.vae.decode(&x)?,
            None if i == 0 => {
                event(Json::obj([("stage", Json::str("loading_vae")), ("image", Json::Int(i as i64 + 1))]));
                stages.vae(&dev)?.decode(&x)?
            }
            None => stages.vae(&dev)?.decode(&x)?,
        };
        files.push(save_image(r, &out, &mut manifest, i, prompt, seed, truncated, &decoded, sampling_seconds, decode.elapsed().as_secs_f64(), image_clock, &mut event)?);
    }
    Ok(Json::obj([
        ("data", Json::Arr(files)),
        ("output_dir", Json::str(out.to_string_lossy())),
        ("load_seconds", Json::Num(load_seconds)),
        ("seconds", Json::Num(clock.elapsed().as_secs_f64())),
        (
            "residency",
            Json::obj([
                ("budget", r.budget.to_json()),
                ("mode", Json::str(if resident { "resident" } else { "staged" })),
                ("ram_bytes", Json::Int(stages.host_bytes as i64)),
                ("disk_bytes_read", Json::Int(stages.streamed_bytes as i64)),
            ]),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_sdxl_request_without_allocating_model_memory() {
        let file = std::env::current_exe().unwrap();
        let request = Json::obj([
            ("checkpoint", Json::str(file.to_string_lossy())),
            ("tokenizer", Json::str(file.to_string_lossy())),
            (
                "output_dir",
                Json::str(std::env::temp_dir().to_string_lossy()),
            ),
            ("prompt", Json::str("a fox")),
        ]);
        let r = Request::parse(&request).unwrap();
        assert_eq!(
            (r.steps, r.cfg, r.width, r.height, r.clip_skip),
            (16, 2.5, 1024, 1024, 1)
        );
        for (key, value) in [
            ("width", Json::Int(544)),
            ("cfg", Json::Num(0.0)),
            ("cfg", Json::str("2.5")),
            ("steps", Json::Int(1)),
            ("prompts", Json::Arr(vec![])),
            ("negative_prompt", Json::Int(3)),
            ("turbo", Json::Bool(true)),
            ("images", Json::Arr(vec![Json::str("reference.png")])),
            ("clip_skip", Json::Int(0)),
            ("seed", Json::Int(i64::MAX)),
            ("sampler", Json::str("euler")),
        ] {
            let mut j = request.clone();
            if let Json::Obj(fields) = &mut j {
                fields.push((key.into(), value));
            }
            assert!(Request::parse(&j).is_err(), "{key}");
        }
    }
}
