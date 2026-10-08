//! Native FLUX.2 Klein 4B text-to-image, using published BFL-layout weights
//! and runtime LoRA factors. No Python, engine sidecar, or weight conversion.
//! On WebGPU (a job's `backend`, the default where the build has it) the text
//! encoder, the transformer and the VAE's decoder run on the GPU
//! ([`crate::klein_text_wgpu`], [`crate::klein_wgpu`], [`crate::sdxl_vae_wgpu`]).
pub mod math;
pub mod schedule;
pub mod text;
pub mod transformer;
pub mod vae;
use crate::residency::Budget;
use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(Debug)]
pub struct Request {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
    pub vae: PathBuf,
    pub loras: Vec<(PathBuf, f64)>,
    pub output: PathBuf,
    pub prompts: Vec<String>,
    pub count: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg: f64,
    pub seed: u64,
    pub device: usize,
    pub distilled: bool,
    pub budget: Budget,
    /// The text encoder and the transformer on WebGPU (`backend` "webgpu", or none named in a build that has it);
    /// else Candle on the CPU (`backend` "cpu").
    pub webgpu: bool,
}

/// The transformer a picture is sampled by: Candle's, or the chain's on WebGPU.
enum Model {
    Candle(transformer::Transformer),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::klein_wgpu::WgpuKlein),
}

impl Model {
    fn predict(&mut self, x: &Tensor, context: &Tensor, t: f64, h: usize, w: usize) -> Result<Tensor> {
        match self {
            Model::Candle(m) => m.predict(x, context, t, h, w),
            #[cfg(feature = "webgpu")]
            Model::Wgpu(m) => m.predict(x, context, t, h, w),
        }
    }

    fn residency(&self) -> Json {
        match self {
            Model::Candle(m) => m.residency(),
            #[cfg(feature = "webgpu")]
            Model::Wgpu(_) => Json::obj([("device", Json::str("webgpu"))]),
        }
    }

    /// Let go of a step's vectors where they are a GPU's (the decoder wants the room).
    fn release_scratch(&mut self) {
        match self {
            Model::Candle(_) => {}
            #[cfg(feature = "webgpu")]
            Model::Wgpu(m) => m.release_scratch(),
        }
    }
}

/// The VAE's decoder: Candle's, or SDXL's on WebGPU (the same AutoencoderKL) with the statistics its tokens are
/// unpacked by on the host.
enum Decoder {
    Candle(vae::Vae),
    #[cfg(feature = "webgpu")]
    Wgpu(crate::sdxl_vae_wgpu::WgpuSdxlVae, Vec<f32>, Vec<f32>),
}

impl Decoder {
    /// The picture of `tokens` (`[1, h w, 128]`): its pixels' rows of red, green and blue, in -1..1 or about.
    fn pixels(&self, tokens: &Tensor, h: usize, w: usize) -> Result<Vec<f32>> {
        match self {
            Decoder::Candle(vae) => vae.decode(tokens, h, w)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>(),
            #[cfg(feature = "webgpu")]
            Decoder::Wgpu(vae, mean, std) => {
                let tokens = tokens.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                Ok(vae.decode(&vae::unpack_rows(&tokens, h, w, mean, std), 2 * h, 2 * w)?.0)
            }
        }
    }
}

/// The decoder beside `model` where that is on WebGPU, else Candle's on `dev`.
fn decoder(r: &Request, model: &Model, dev: &Device) -> Result<Decoder> {
    match model {
        Model::Candle(_) => Ok(Decoder::Candle(vae::Vae::load(&r.vae, dev, false)?)),
        #[cfg(feature = "webgpu")]
        Model::Wgpu(m) => {
            let (vae, mean, std) = vae::decoder_on_webgpu(&r.vae, m.backend().clone())?;
            Ok(Decoder::Wgpu(vae, mean, std))
        }
    }
}

/// Each prompt's conditioning and, where asked, the empty prompt's, by the text encoder on WebGPU.
#[cfg(feature = "webgpu")]
fn webgpu_contexts(r: &Request, negative: bool) -> Result<(Vec<Tensor>, Option<Tensor>)> {
    let mut encoder = crate::klein_text_wgpu::WgpuKleinText::load(&r.text_encoder, &r.tokenizer, r.device)?;
    let prompts: Vec<&str> = r.prompts.iter().map(String::as_str).chain(negative.then_some("")).collect();
    let mut contexts = encoder.encode_all(&prompts)?;
    let negative = if negative { contexts.pop() } else { None };
    Ok((contexts, negative))
}

#[cfg(not(feature = "webgpu"))]
fn webgpu_contexts(_: &Request, _: bool) -> Result<(Vec<Tensor>, Option<Tensor>)> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

#[cfg(feature = "webgpu")]
fn webgpu_model(r: &Request) -> Result<Model> {
    Ok(Model::Wgpu(crate::klein_wgpu::WgpuKlein::load(&r.transformer, r.device, &r.loras, |_| {})?))
}

#[cfg(not(feature = "webgpu"))]
fn webgpu_model(_: &Request) -> Result<Model> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let webgpu = crate::pipeline::backend_is_webgpu(j, "FLUX.2 Klein")?;
        let path = |key: &str| -> std::result::Result<PathBuf, String> {
            let p = PathBuf::from(
                j.get(key)
                    .and_then(Json::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| format!("Klein requires {key}"))?,
            );
            if !p.is_file() {
                return Err(format!("{key} must be an existing file: {}", p.display()));
            }
            Ok(p)
        };
        let number =
            |key: &str, default: i64, min: i64, max: i64| -> std::result::Result<usize, String> {
                let v = match j.get(key) {
                    None => default,
                    Some(v) => v
                        .as_i64()
                        .ok_or_else(|| format!("{key} must be an integer"))?,
                };
                if !(min..=max).contains(&v) {
                    return Err(format!("{key} must be between {min} and {max}"));
                }
                Ok(v as usize)
            };
        let distilled = match j.get("variant") {
            None => true,
            Some(v) if v.as_str() == Some("distilled") => true,
            Some(v) if v.as_str() == Some("base") => false,
            _ => return Err("Klein variant must be distilled or base (4B only)".into()),
        };
        let cfg = match j.get("cfg") {
            None => {
                if distilled {
                    1.
                } else {
                    4.
                }
            }
            Some(v) => v.as_f64().ok_or("cfg must be numeric")?,
        };
        if !cfg.is_finite() || !(1. ..=10.).contains(&cfg) || (distilled && cfg != 1.) {
            return Err("Klein distilled CFG must be 1; base CFG must be between 1 and 10".into());
        }
        let steps = number("steps", if distilled { 4 } else { 50 }, 1, 100)?;
        if distilled && steps != 4 {
            return Err("Klein step-distilled 4B uses exactly 4 steps; select variant=base for the base checkpoint".into());
        }
        for key in [
            "images",
            "image",
            "input_image",
            "input_reference",
            "adapter",
        ] {
            if j.get(key).is_some_and(|v| {
                !matches!(v, Json::Null) && !v.as_array().is_some_and(|a| a.is_empty())
            }) {
                return Err("Klein currently supports text-to-image; use loras for style adapters, without Qwen turbo/reference/negative fields".into());
            }
        }
        // Generic image clients send an empty negative field for text-to-image.
        if j.get("negative_prompt").is_some_and(|v| {
            !matches!(v, Json::Null) && !v.as_str().is_some_and(|s| s.trim().is_empty())
        }) {
            return Err("Klein does not support negative prompt conditioning".into());
        }
        if j.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err("Klein does not use Qwen turbo".into());
        }
        let count = number("n", 1, 1, 1000)?;
        let prompts = match j.get("prompts") {
            Some(v) => v
                .as_array()
                .ok_or("prompts must be an array")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .filter(|s| !s.trim().is_empty() && s.len() <= 16384)
                        .map(str::to_owned)
                        .ok_or("invalid prompt".to_owned())
                })
                .collect::<std::result::Result<Vec<_>, _>>()?,
            None => vec![j
                .get("prompt")
                .and_then(Json::as_str)
                .filter(|s| !s.trim().is_empty() && s.len() <= 16384)
                .ok_or("prompt must contain 1..16384 bytes")?
                .to_owned()],
        };
        if prompts.len() != 1 && prompts.len() != count {
            return Err("provide one reusable prompt or exactly n prompts".into());
        }
        let width = number("width", 1024, 256, 2048)?;
        let height = number("height", 1024, 256, 2048)?;
        if width % 16 != 0 || height % 16 != 0 {
            return Err("Klein dimensions must be multiples of 16".into());
        }
        let loras = crate::pipeline::parse_loras(j)?;
        for (p, _) in &loras {
            if !p.is_file() {
                return Err(format!("LoRA not found: {}", p.display()));
            }
        }
        let seed = number("seed", 0, 0, i64::MAX - count as i64)? as u64;
        let output = j
            .get("output_dir")
            .and_then(Json::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or("output_dir is required")?
            .into();
        Ok(Self {
            transformer: path("transformer")?,
            text_encoder: path("text_encoder")?,
            tokenizer: path("tokenizer")?,
            vae: path("vae")?,
            loras,
            output,
            prompts,
            count,
            width,
            height,
            steps,
            cfg,
            seed,
            device: number("device", 0, 0, 31)?,
            distilled,
            budget: Budget::parse(j)?,
            webgpu,
        })
    }
}
pub fn generate(r: &Request, mut event: impl FnMut(Json)) -> Result<Json> {
    // Header preflight precedes device creation and all model allocations.
    let index = crate::weights::Weights::open(&r.transformer)?;
    transformer::Config::default().validate(&index)?;
    crate::lora::Loras::open(&r.loras)?
        .validate_modules(&transformer::Config::default().projections())?;
    drop(index);
    let dev = Device::Cpu;
    let dtype = if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let clock = Instant::now();
    event(Json::obj([
        ("stage", Json::str("encoding_text")),
        ("architecture", Json::str("flux2-klein-4b")),
    ]));
    let guided = !r.distilled && r.cfg > 1.;
    let (contexts, negative) = if r.webgpu {
        webgpu_contexts(r, guided)?
    } else {
        let mut encoder =
            text::TextEncoder::load(&r.text_encoder, &r.tokenizer, &dev, dtype, &r.budget)?;
        let mut contexts = Vec::new();
        for p in &r.prompts {
            contexts.push(encoder.encode(p)?.to_device(&Device::Cpu)?);
        }
        let negative = if guided {
            Some(encoder.encode("")?.to_device(&Device::Cpu)?)
        } else {
            None
        };
        (contexts, negative)
    };
    dev.synchronize()?;
    event(Json::obj([("stage", Json::str("loading_transformer"))]));
    let mut model = if r.webgpu {
        webgpu_model(r)?
    } else {
        Model::Candle(transformer::Transformer::load(&r.transformer, &r.loras, &dev, dtype, &r.budget)?)
    };
    event(Json::obj([("stage", Json::str("loading_vae"))]));
    let decoder = decoder(r, &model, &dev)?;
    std::fs::create_dir_all(&r.output)?;
    let batch = r.output.join(format!(
        "klein-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(candle_core::Error::wrap)?
            .as_nanos()
    ));
    std::fs::create_dir(&batch)?;
    let mut manifest = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(batch.join("manifest.jsonl"))?;
    let (h, w) = (r.height / 16, r.width / 16);
    let sigmas = schedule::sigmas(r.steps, h * w)?;
    let mut data = Vec::new();
    for i in 0..r.count {
        let seed = r.seed + i as u64;
        let prompt = &r.prompts[if r.prompts.len() == 1 { 0 } else { i }];
        let context = contexts[if contexts.len() == 1 { 0 } else { i }].to_device(&dev)?;
        let uncond = negative.as_ref().map(|t| t.to_device(&dev)).transpose()?;
        let mut x = Tensor::from_vec(
            crate::pipeline::noise(seed, h * w * 128),
            (1, 128, h, w),
            &dev,
        )?
        .to_dtype(dtype)?;
        x = math::pack(&x)?;
        let sample = Instant::now();
        for (step, t) in sigmas.windows(2).enumerate() {
            let mut velocity = model.predict(&x, &context, t[0], h, w)?;
            if let Some(u) = &uncond {
                let v = model.predict(&x, u, t[0], h, w)?;
                velocity = (&v + ((velocity - &v)? * r.cfg)?)?;
            }
            x = schedule::euler(&x, &velocity, t[0], t[1])?;
            event(Json::obj([
                ("stage", Json::str("sampling")),
                ("image", Json::Int(i as i64 + 1)),
                ("step", Json::Int(step as i64 + 1)),
                ("steps", Json::Int(r.steps as i64)),
            ]));
        }
        dev.synchronize()?;
        let sampling_seconds = sample.elapsed().as_secs_f64();
        model.release_scratch();
        let decode = Instant::now();
        let pixels = decoder.pixels(&x, h, w)?;
        let decode_seconds = decode.elapsed().as_secs_f64();
        if pixels.len() != r.width * r.height * 3 {
            candle_core::bail!("Klein's decoder made {} values for a {}x{} picture", pixels.len(), r.width, r.height);
        }
        if pixels.iter().any(|x| !x.is_finite()) {
            candle_core::bail!("Klein produced non-finite pixels; no image saved");
        }
        let bytes = pixels
            .iter()
            .map(|x| ((x.clamp(-1., 1.) + 1.) * 127.5).round() as u8)
            .collect::<Vec<_>>();
        let path = batch.join(format!("image-{:04}-seed-{seed}.png", i + 1));
        image::save_buffer(
            &path,
            &bytes,
            r.width as u32,
            r.height as u32,
            image::ColorType::Rgb8,
        )
        .map_err(candle_core::Error::wrap)?;
        let record = Json::obj([
            ("path", Json::str(path.to_string_lossy())),
            ("prompt", Json::str(prompt)),
            ("seed", Json::Int(seed as i64)),
            ("architecture", Json::str("flux2-klein-4b")),
            (
                "variant",
                Json::str(if r.distilled { "distilled" } else { "base" }),
            ),
            ("steps", Json::Int(r.steps as i64)),
            ("cfg", Json::Num(r.cfg)),
            ("width", Json::Int(r.width as i64)),
            ("height", Json::Int(r.height as i64)),
            ("transformer", Json::str(r.transformer.to_string_lossy())),
            ("backend", Json::str(if r.webgpu { "webgpu" } else { "cpu" })),
            ("noise_generator", Json::str("oaiy-splitmix64-box-muller")),
            (
                "loras",
                Json::Arr(
                    r.loras
                        .iter()
                        .map(|(p, s)| {
                            Json::obj([
                                ("path", Json::str(p.to_string_lossy())),
                                ("strength", Json::Num(*s)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("sampling_seconds", Json::Num(sampling_seconds)),
            ("decode_seconds", Json::Num(decode_seconds)),
        ]);
        writeln!(manifest, "{}", record.to_json())?;
        manifest.flush()?;
        event(Json::obj([
            ("stage", Json::str("image_saved")),
            ("image", Json::Int(i as i64 + 1)),
            ("path", Json::str(path.to_string_lossy())),
        ]));
        data.push(record);
    }
    Ok(Json::obj([
        ("data", Json::Arr(data)),
        ("output_dir", Json::str(batch.to_string_lossy())),
        ("seconds", Json::Num(clock.elapsed().as_secs_f64())),
        ("residency", model.residency()),
    ]))
}

#[cfg(test)]
mod request_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn empty_negative_from_generic_clients_is_absent_conditioning() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/klein/tiny.safetensors");
        let output = std::env::temp_dir().join(format!("oaiy-klein-request-{}", std::process::id()));
        let body = |negative| Json::obj([
            ("transformer", Json::str(fixture.to_string_lossy())),
            ("text_encoder", Json::str(fixture.to_string_lossy())),
            ("tokenizer", Json::str(fixture.to_string_lossy())),
            ("vae", Json::str(fixture.to_string_lossy())),
            ("output_dir", Json::str(output.to_string_lossy())),
            ("prompt", Json::str("fox")), ("negative_prompt", negative),
        ]);
        for value in [Json::Null, Json::str(""), Json::str(" \t")] {
            assert!(Request::parse(&body(value)).is_ok());
        }
        for value in [Json::str("blur"), Json::Bool(false), Json::Arr(Vec::new())] {
            assert!(Request::parse(&body(value)).is_err());
        }
    }
}
