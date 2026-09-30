//! Native FLUX.2 Klein 4B text-to-image, using published BFL-layout weights
//! and runtime LoRA factors. No Python, engine sidecar, or weight conversion.
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
}
impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
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
    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(r.device)?;
    #[cfg(not(feature = "cuda"))]
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
    let mut encoder =
        text::TextEncoder::load(&r.text_encoder, &r.tokenizer, &dev, dtype, &r.budget)?;
    let mut contexts = Vec::new();
    for p in &r.prompts {
        contexts.push(encoder.encode(p)?.to_device(&Device::Cpu)?);
    }
    let negative = if !r.distilled && r.cfg > 1. {
        Some(encoder.encode("")?.to_device(&Device::Cpu)?)
    } else {
        None
    };
    drop(encoder);
    dev.synchronize()?;
    event(Json::obj([("stage", Json::str("loading_transformer"))]));
    let mut model =
        transformer::Transformer::load(&r.transformer, &r.loras, &dev, dtype, &r.budget)?;
    event(Json::obj([("stage", Json::str("loading_vae"))]));
    let decoder = vae::Vae::load(&r.vae, &dev, false)?;
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
        let rgb = decoder
            .decode(&x, h, w)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?;
        let pixels = rgb
            .squeeze(0)?
            .permute((1, 2, 0))?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
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
