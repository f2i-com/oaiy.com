//! MOSS-Image (OpenMOSS's MOSS-Image-Base and MOSS-Image-SFT, and GGUFs of their transformer): text-to-image on WebGPU
//! as OpenMOSS's `MOSSImagePipeline` (`pipeline_moss_image.py`) makes a picture. The prompt (natural language, tags, or
//! a JSON layout) and the negative one through Qwen3.5-9B ([`webgpu::text`]); the transformer ([`webgpu`]) over 50
//! Euler steps of its own schedule (shifted, then snapped to its training grid) with classifier-free guidance whose
//! negative branch skips block 22 (`uncond_skip_layers`); the FLUX.2 VAE's decoder (MOSS-Image's own smaller one, or
//! FLUX.2's) last. With a distillation ([`distill`]) the transformer makes the picture in its few passes instead (8),
//! without guidance or a negative prompt.
pub mod distill;
#[cfg(feature = "webgpu")]
pub mod webgpu;

use oaiy_engine::json::Json;
use std::{io::Write, path::PathBuf, time::Instant};

/// The picture's area at most: 2048 x 2048 (the model's own range is 0.25 to 4 megapixels).
pub const MAX_PIXELS: usize = 4 * 1024 * 1024;
/// The scheduler's training steps.
const TRAINING_STEPS: usize = 1000;

#[derive(Debug)]
pub struct Request {
    pub transformer: PathBuf,
    /// Qwen3.5-9B: its folder (Hugging Face's, or MOSS-Image's `text_encoder`), a safetensors file of it, or a GGUF.
    pub text_encoder: PathBuf,
    /// Qwen3.5's tokenizer.json (None: a GGUF text encoder's own).
    pub tokenizer: Option<PathBuf>,
    pub vae: PathBuf,
    pub output: PathBuf,
    pub prompts: Vec<String>,
    pub negative_prompt: String,
    pub count: usize,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// Classifier-free guidance (1: none, no negative branch).
    pub guidance: f64,
    pub shift: f64,
    /// The blocks the negative branch leaves out (zero-based).
    pub skip_layers: Vec<usize>,
    /// A distillation ([`distill`]): its file or folder. `steps` are then its passes (8 by default), and there is no
    /// guidance, negative prompt, shift or skipped block.
    pub distill: Option<PathBuf>,
    pub seed: u64,
    pub device: usize,
}

/// `torch.linspace(start, end, steps)` in f32 as its CPU kernel makes it: the first half from the start, the rest from
/// the end.
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    (0..steps).map(|i| if i < steps / 2 { start + step * i as f32 } else { end - step * (steps - i - 1) as f32 }).collect()
}

/// `MOSSImageScheduler`'s grid of `count` sigmas: evenly from 1 to its lowest, shifted by `shift`.
fn grid(count: usize, shift: f32) -> Vec<f32> {
    let lower = (shift as f64 / (TRAINING_STEPS as f64 + shift as f64 - 1.)) as f32;
    linspace(1., lower, count).into_iter().map(|v| shift * v / (1. + (shift - 1.) * v)).collect()
}

/// The sigmas of `steps` steps and the 0 after them, as `MOSSImageScheduler.set_timesteps` makes them: the shifted
/// grid, each snapped to the nearest of the training's thousand (the lower of two as near), repeats dropped. A
/// step's time for the transformer is its sigma times 1000.
pub fn schedule(steps: usize, shift: f32) -> Vec<f32> {
    let mut training = grid(TRAINING_STEPS, 1.);
    training.reverse();
    let mut sigmas: Vec<f32> = Vec::with_capacity(steps + 1);
    for s in grid(steps, shift) {
        let upper = training.partition_point(|&v| v < s).min(TRAINING_STEPS - 1);
        let lower = upper.saturating_sub(1);
        let near = training[if (s - training[lower]).abs() <= (training[upper] - s).abs() { lower } else { upper }];
        if sigmas.last() != Some(&near) {
            sigmas.push(near);
        }
    }
    sigmas.push(0.);
    sigmas
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        if !crate::common::job::backend_is_webgpu(j, "MOSS-Image")? {
            return Err("MOSS-Image runs on WebGPU only: leave backend out".into());
        }
        let path = |key: &str, dir: bool| -> std::result::Result<PathBuf, String> {
            let p = PathBuf::from(j.get(key).and_then(Json::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("MOSS-Image requires {key}"))?);
            if !(p.is_file() || (dir && p.is_dir())) {
                return Err(format!("{key} must be an existing {}: {}", if dir { "file or folder" } else { "file" }, p.display()));
            }
            Ok(p)
        };
        let number = |key: &str, default: i64, min: i64, max: i64| -> std::result::Result<usize, String> {
            let v = match j.get(key) {
                None | Some(Json::Null) => default,
                Some(v) => v.as_i64().ok_or_else(|| format!("{key} must be an integer"))?,
            };
            if !(min..=max).contains(&v) {
                return Err(format!("{key} must be between {min} and {max}"));
            }
            Ok(v as usize)
        };
        let real = |keys: &[&str], default: f64, min: f64, max: f64| -> std::result::Result<f64, String> {
            match keys.iter().find_map(|k| j.get(k).filter(|v| !matches!(v, Json::Null))) {
                None => Ok(default),
                Some(v) => {
                    let x = v.as_f64().ok_or_else(|| format!("{} must be a number", keys[0]))?;
                    if !x.is_finite() || !(min..=max).contains(&x) {
                        return Err(format!("{} must be between {min} and {max}", keys[0]));
                    }
                    Ok(x)
                }
            }
        };
        let text_encoder = path("text_encoder", true)?;
        let tokenizer = match j.get("tokenizer") {
            None | Some(Json::Null) => {
                let beside = if text_encoder.is_dir() { text_encoder.join("tokenizer.json") } else { text_encoder.with_file_name("tokenizer.json") };
                if beside.is_file() {
                    Some(beside)
                } else if text_encoder.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
                    None
                } else {
                    return Err(format!("MOSS-Image requires Qwen3.5's tokenizer: no tokenizer given, none at {}", beside.display()));
                }
            }
            _ => Some(path("tokenizer", false)?),
        };
        for key in ["adapter", "loras", "images"] {
            if j.get(key).is_some_and(|v| !matches!(v, Json::Null) && !v.as_array().is_some_and(|a| a.is_empty())) {
                return Err(format!("MOSS-Image takes no {key}: it makes a picture from its prompt alone"));
            }
        }
        if j.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err("MOSS-Image does not use Qwen turbo".into());
        }
        let count = number("n", 1, 1, 1000)?;
        let prompt = |v: &Json| v.as_str().filter(|s| !s.trim().is_empty() && s.len() <= 65536).map(str::to_owned);
        let prompts = match j.get("prompts") {
            Some(v) => v.as_array().ok_or("prompts must be an array")?.iter().map(|v| prompt(v).ok_or("invalid prompt".to_owned())).collect::<std::result::Result<Vec<_>, _>>()?,
            None => vec![j.get("prompt").and_then(prompt).ok_or("prompt must contain 1..65536 bytes")?],
        };
        if prompts.len() != 1 && prompts.len() != count {
            return Err("provide one reusable prompt or exactly n prompts".into());
        }
        let negative_prompt = match j.get("negative_prompt") {
            None | Some(Json::Null) => String::new(),
            Some(v) => v.as_str().filter(|s| s.len() <= 65536).ok_or("negative_prompt must be a string of at most 65536 bytes")?.to_owned(),
        };
        let (width, height) = (number("width", 1024, 256, 4096)?, number("height", 1024, 256, 4096)?);
        if width % 16 != 0 || height % 16 != 0 {
            return Err("MOSS-Image's width and height must be multiples of 16".into());
        }
        if width * height > MAX_PIXELS {
            return Err(format!("MOSS-Image makes pictures of at most {MAX_PIXELS} pixels (2048 x 2048)"));
        }
        let skip_layers = match j.get("uncond_skip_layers").or_else(|| j.get("skip_layers")) {
            None | Some(Json::Null) => vec![22],
            Some(v) => v
                .as_array()
                .ok_or("uncond_skip_layers must be an array of block numbers")?
                .iter()
                .map(|v| v.as_i64().filter(|i| (0..1000).contains(i)).map(|i| i as usize).ok_or_else(|| "uncond_skip_layers must hold block numbers from 0".to_owned()))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        let seed = number("seed", 0, 0, i64::MAX - count as i64)? as u64;
        let distill = match j.get("distill") {
            None | Some(Json::Null) => None,
            Some(v) if v.as_str().is_some_and(|s| s.trim().is_empty()) => None,
            Some(_) => Some(path("distill", true)?),
        };
        let output = j.get("output_dir").and_then(Json::as_str).filter(|s| !s.trim().is_empty()).ok_or("output_dir is required")?.into();
        Ok(Self {
            transformer: path("transformer", true)?,
            text_encoder,
            tokenizer,
            vae: path("vae", false)?,
            output,
            prompts,
            negative_prompt,
            count,
            width,
            height,
            steps: number("steps", if distill.is_some() { 8 } else { 50 }, 1, 100)?,
            // (a distillation's passes take none of these: one forward each, from its own times)
            guidance: if distill.is_some() { 1. } else { real(&["cfg", "guidance"], 3., 1., 10.)? },
            shift: real(&["shift"], 5., 0.1, 20.)?,
            skip_layers: if distill.is_some() { Vec::new() } else { skip_layers },
            distill,
            seed,
            device: number("device", 0, 0, 31)?,
        })
    }
}

#[cfg(feature = "webgpu")]
pub fn generate(r: &Request, event: impl FnMut(Json)) -> candle_core::Result<Json> {
    generate_with(r, crate::common::job::noise, event)
}

#[cfg(not(feature = "webgpu"))]
pub fn generate(_: &Request, _: impl FnMut(Json)) -> candle_core::Result<Json> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// [`generate`], each picture's starting noise `noise(seed, values)` (its tokens' channels, a token a row; a test gives
/// the pipeline's own).
#[cfg(feature = "webgpu")]
pub fn generate_with(r: &Request, noise: impl Fn(u64, usize) -> Vec<f32>, mut event: impl FnMut(Json)) -> candle_core::Result<Json> {
    use crate::klein::vae;
    use crate::moss_image::webgpu::{text::WgpuMossText, Config, WgpuMoss};
    use candle_core::bail;
    // the transformer's header first: which checkpoint, before anything is loaded; a distillation's codes
    let cfg = Config::of(&crate::common::weights::Weights::open(&r.transformer)?)?;
    let mut distill = r.distill.as_deref().map(distill::Distillation::open).transpose()?;
    if let Some(d) = &distill {
        if r.steps > d.codes.len() {
            bail!("the distillation makes a picture in at most {} passes, not {}", d.codes.len(), r.steps);
        }
        if d.channels != cfg.channels {
            bail!("the distillation mixes {} channels, MOSS-Image's latent has {}", d.channels, cfg.channels);
        }
    }
    if cfg.context != 4096 {
        bail!("MOSS-Image's transformer takes text states 4,096 wide, not {}", cfg.context);
    }
    if let Some(bad) = r.skip_layers.iter().find(|&&i| i >= cfg.layers) {
        bail!("MOSS-Image's transformer has {} blocks: it cannot skip block {bad}", cfg.layers);
    }
    let guided = distill.is_none() && r.guidance != 1.;
    let clock = Instant::now();
    event(Json::obj([("stage", Json::str("encoding_text")), ("architecture", Json::str("moss-image"))]));
    let (contexts, negative) = {
        let mut encoder = WgpuMossText::load(&r.text_encoder, r.tokenizer.as_deref(), r.device)?;
        let prompts: Vec<&str> = r.prompts.iter().map(String::as_str).chain(guided.then_some(r.negative_prompt.as_str())).collect();
        let mut all = encoder.encode_all(&prompts)?;
        let negative = if guided { all.pop() } else { None };
        (all, negative)
    };
    let gpu = ggml_rs_wgpu::WgpuBackend::nth(r.device, None).map_err(crate::qwen_image::webgpu::transformer::err)?;
    event(Json::obj([("stage", Json::str("loading_vae"))]));
    let (decoder, mean, std) = vae::vae_on_webgpu(&r.vae, gpu.clone(), false)?;
    event(Json::obj([("stage", Json::str("loading_transformer"))]));
    let mut model = WgpuMoss::load_with(&r.transformer, gpu, distill.as_mut(), |_, _| {})?;
    let prompts = contexts.iter().map(|(states, n)| model.prompt(states, *n)).collect::<candle_core::Result<Vec<_>>>()?;
    let negative = negative.map(|(states, n)| model.prompt(&states, n)).transpose()?;
    std::fs::create_dir_all(&r.output)?;
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let batch = r.output.join(format!("moss-image-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&batch)?;
    let mut manifest = std::fs::OpenOptions::new().create_new(true).write(true).open(batch.join("manifest.jsonl"))?;
    let (h, w) = (r.height / 16, r.width / 16);
    let sigmas = schedule(r.steps, r.shift as f32);
    let steps = if distill.is_some() { r.steps } else { sigmas.len() - 1 };
    let mut data = Vec::new();
    for i in 0..r.count {
        let seed = r.seed + i as u64;
        let index = if r.prompts.len() == 1 { 0 } else { i };
        let (text, prompt) = (&prompts[if prompts.len() == 1 { 0 } else { i }], &r.prompts[index]);
        let mut x = noise(seed, h * w * 128);
        let sample = Instant::now();
        // a distillation's: each pass from the noise and the clean estimate so far, at its next time; the last
        // estimate is the picture's latent
        if let Some(d) = &distill {
            x = d.sample(&x, steps, |step, x, time| {
                let v = model.predict(x, h, w, time, text, &[]);
                event(Json::obj([("stage", Json::str("sampling")), ("image", Json::Int(i as i64 + 1)), ("step", Json::Int(step as i64 + 1)), ("steps", Json::Int(steps as i64))]));
                v
            })?;
        }
        for step in 0..if distill.is_some() { 0 } else { steps } {
            let (sigma, next) = (sigmas[step], sigmas[step + 1]);
            let t = sigma * TRAINING_STEPS as f32;
            let conditional = model.predict(&x, h, w, t, text, &[])?;
            let velocity: Vec<f32> = match &negative {
                Some(neg) => {
                    let unconditional = model.predict(&x, h, w, t, neg, &r.skip_layers)?;
                    let g = r.guidance as f32;
                    unconditional.iter().zip(&conditional).map(|(u, c)| u + g * (c - u)).collect()
                }
                None => conditional,
            };
            let dt = next - sigma;
            for (x, v) in x.iter_mut().zip(&velocity) {
                *x += dt * v;
            }
            event(Json::obj([("stage", Json::str("sampling")), ("image", Json::Int(i as i64 + 1)), ("step", Json::Int(step as i64 + 1)), ("steps", Json::Int(steps as i64))]));
        }
        let sampling_seconds = sample.elapsed().as_secs_f64();
        model.release_scratch();
        event(Json::obj([("stage", Json::str("decoding")), ("image", Json::Int(i as i64 + 1))]));
        let decode = Instant::now();
        let (pixels, ph, pw) = decoder.decode(&vae::unpack_rows(&x, h, w, &mean, &std), 2 * h, 2 * w)?;
        let decode_seconds = decode.elapsed().as_secs_f64();
        if (ph, pw) != (r.height, r.width) || pixels.len() != r.width * r.height * 3 {
            bail!("MOSS-Image's decoder made a {pw}x{ph} picture of {} values for {}x{}", pixels.len(), r.width, r.height);
        }
        if pixels.iter().any(|v| !v.is_finite()) {
            bail!("MOSS-Image produced non-finite pixels; no image saved");
        }
        // (the pipeline's: clamped, to 0..1, times 255 rounded)
        let bytes: Vec<u8> = pixels.iter().map(|v| ((v.clamp(-1., 1.) * 0.5 + 0.5) * 255.).round() as u8).collect();
        let picture = image::RgbImage::from_raw(r.width as u32, r.height as u32, bytes).ok_or_else(|| crate::qwen_image::webgpu::transformer::err("a picture's pixels"))?;
        let path = batch.join(format!("image-{:04}-seed-{seed}.png", i + 1));
        picture.save(&path).map_err(candle_core::Error::wrap)?;
        let record = Json::obj([
            ("path", Json::str(path.to_string_lossy())),
            ("prompt", Json::str(prompt)),
            ("negative_prompt", Json::str(&r.negative_prompt)),
            ("seed", Json::Int(seed as i64)),
            ("architecture", Json::str("moss-image")),
            ("steps", Json::Int(steps as i64)),
            ("cfg", Json::Num(r.guidance)),
            ("shift", Json::Num(r.shift)),
            ("uncond_skip_layers", Json::Arr(r.skip_layers.iter().map(|&i| Json::Int(i as i64)).collect())),
            ("distilled", Json::Bool(distill.is_some())),
            ("width", Json::Int(r.width as i64)),
            ("height", Json::Int(r.height as i64)),
            ("transformer", Json::str(r.transformer.to_string_lossy())),
            ("backend", Json::str("webgpu")),
            ("noise_generator", Json::str("oaiy-splitmix64-box-muller")),
            ("sampling_seconds", Json::Num(sampling_seconds)),
            ("decode_seconds", Json::Num(decode_seconds)),
        ]);
        writeln!(manifest, "{}", record.to_json())?;
        manifest.flush()?;
        event(Json::obj([("stage", Json::str("image_saved")), ("image", Json::Int(i as i64 + 1)), ("path", Json::str(path.to_string_lossy()))]));
        data.push(record);
    }
    Ok(Json::obj([
        ("data", Json::Arr(data)),
        ("output_dir", Json::str(batch.to_string_lossy())),
        ("seconds", Json::Num(clock.elapsed().as_secs_f64())),
        ("residency", Json::obj([("device", Json::str("webgpu"))])),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schedule is `MOSSImageScheduler`'s, as diffusers 0.41 and torch 2.14 make it in f32 on this Mac: 50 steps at
    /// shift 5 whole, and the ends of others (fewer steps snap to fewer distinct sigmas, at shift 6 too). Within a
    /// unit in f32's last place: torch's linspace makes its grid a SIMD vector at a time, its values as the CPU's
    /// vectors are wide (a Mac's 4 lanes, an x86's 8 or 16), the scalar formula here.
    #[test]
    fn the_schedule_is_the_schedulers() {
        let alike = |got: &[f32], want: &[f32]| got.len() == want.len() && got.iter().zip(want).all(|(a, b)| (a - b).abs() <= f32::EPSILON * b.abs());
        let fifty: [f32; 51] = [
            1.0, 0.9959999918937683, 0.9919999837875366, 0.9869999885559082, 0.9829999804496765, 0.9779999852180481, 0.9729999899864197, 0.9679999947547913, 0.9629999995231628,
            0.9570000171661377, 0.9520000219345093, 0.9459999799728394, 0.9390000104904175, 0.9329999685287476, 0.9259999990463257, 0.918999969959259, 0.9120000004768372,
            0.9049999713897705, 0.8970000147819519, 0.8880000114440918, 0.8799999952316284, 0.8709999918937683, 0.8610000014305115, 0.8510000109672546, 0.8399999737739563,
            0.8289999961853027, 0.8169999718666077, 0.8050000071525574, 0.7910000085830688, 0.7770000100135803, 0.7619999647140503, 0.7459999918937683, 0.7289999723434448,
            0.7109999656677246, 0.6919999718666077, 0.6710000038146973, 0.6480000019073486, 0.6229999661445618, 0.597000002861023, 0.5679999589920044, 0.5359999537467957,
            0.5009999871253967, 0.46300002932548523, 0.42100003361701965, 0.37300002574920654, 0.32100000977516174, 0.26100000739097595, 0.19300000369548798,
            0.11500000208616257, 0.024000000208616257, 0.0,
        ];
        assert!(alike(&schedule(50, 5.), &fifty), "{:?}", schedule(50, 5.));
        // (the transformer's times: f32's products, as the scheduler's `sigmas * 1000`)
        assert_eq!(schedule(50, 5.)[13] * 1000., 932.9999389648438);
        let ends = |steps: usize, shift: f32, len: usize, head: &[f32], tail: &[f32]| {
            let s = schedule(steps, shift);
            assert!(s.len() == len && alike(&s[..head.len()], head) && alike(&s[len - 3..], tail), "{steps} steps at {shift}: {s:?}");
        };
        ends(30, 5., 31, &[1.0, 0.9929999709129333, 0.9850000143051147, 0.9779999852180481, 0.968999981880188, 0.9599999785423279], &[0.17000000178813934, 0.024000000208616257, 0.0]);
        ends(8, 5., 9, &[1.0, 0.9679999947547913, 0.9259999990463257, 0.8709999918937683, 0.7910000085830688, 0.6710000038146973], &[0.46300002932548523, 0.024000000208616257, 0.0]);
        ends(4, 5., 5, &[1.0, 0.9099999666213989, 0.7170000076293945, 0.024000000208616257, 0.0], &[0.7170000076293945, 0.024000000208616257, 0.0]);
        assert_eq!(schedule(1, 5.), [1.0, 0.0]);
        ends(50, 6., 51, &[1.0, 0.996999979019165, 0.9929999709129333, 0.9890000224113464, 0.9850000143051147, 0.9819999933242798], &[0.13900001347064972, 0.03500000014901161, 0.0]);
        ends(4, 6., 5, &[1.0, 0.9240000247955322, 0.7529999613761902, 0.03500000014901161, 0.0], &[0.7529999613761902, 0.03500000014901161, 0.0]);
    }

    /// OpenMOSS's published examples (`OAIY_MOSS_EXAMPLES`: their repository's `assets/images`, each its picture and
    /// its `parameters.json`) made again from their records (the prompt, the negative one, the size, steps, cfg,
    /// shift, the skipped block, the seed) and the app's own starting noise (`tools/moss_image/noise.py`'s files in
    /// `OAIY_MOSS_NOISE`), by the transformer `OAIY_MOSS_TRANSFORMER` (MOSS-Image-Base's, as theirs) with
    /// `OAIY_MOSS_TEXT_ENCODER`, `OAIY_MOSS_TOKENIZER` and `OAIY_MOSS_VAE`: each picture as close to theirs as
    /// bfloat16's sums against these let it be (`OAIY_MOSS_ONLY`: the examples named in it, by comma).
    #[cfg(feature = "webgpu")]
    #[test]
    #[ignore = "needs MOSS-Image-Base (OAIY_MOSS_TRANSFORMER, OAIY_MOSS_TEXT_ENCODER, OAIY_MOSS_TOKENIZER, OAIY_MOSS_VAE), OpenMOSS's examples (OAIY_MOSS_EXAMPLES, OAIY_MOSS_NOISE) and a WebGPU adapter"]
    fn the_examples_are_openmosss() -> candle_core::Result<()> {
        let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let (Some(transformer), Some(text_encoder), Some(tokenizer), Some(vae), Some(examples), Some(noise)) =
            (var("OAIY_MOSS_TRANSFORMER"), var("OAIY_MOSS_TEXT_ENCODER"), var("OAIY_MOSS_TOKENIZER"), var("OAIY_MOSS_VAE"), var("OAIY_MOSS_EXAMPLES"), var("OAIY_MOSS_NOISE"))
        else {
            return Ok(());
        };
        let only = std::env::var("OAIY_MOSS_ONLY").unwrap_or_default();
        let mut names: Vec<String> = std::fs::read_dir(&examples)?.flatten().filter(|e| e.path().join("parameters.json").is_file()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        let mut worst = f64::INFINITY;
        for name in names.into_iter().filter(|n| only.is_empty() || only.split(',').any(|o| o == n)) {
            let record = Json::parse(&std::fs::read(examples.join(&name).join("parameters.json"))?).map_err(candle_core::Error::wrap)?;
            let p = record.get("parameters").ok_or_else(|| candle_core::Error::Msg(format!("{name}: no parameters")))?;
            let text = |k: &str| p.get(k).and_then(Json::as_str).unwrap_or_default().to_owned();
            let int = |k: &str| p.get(k).and_then(Json::as_i64).unwrap_or(0);
            let real = |k: &str| p.get(k).and_then(Json::as_f64).unwrap_or(0.);
            let (width, height) = (int("width") as usize, int("height") as usize);
            // (the app's noise is channel-major: a plane a channel; the sampler's a token a row)
            let planes: Vec<f32> = std::fs::read(noise.join(format!("{name}.f32")))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let tokens = width / 16 * (height / 16);
            let start: Vec<f32> = (0..tokens * 128).map(|i| planes[(i % 128) * tokens + i / 128]).collect();
            let r = Request {
                transformer: transformer.clone(),
                text_encoder: text_encoder.clone(),
                tokenizer: Some(tokenizer.clone()),
                vae: vae.clone(),
                output: std::env::temp_dir().join(format!("oaiy-moss-examples-{}", std::process::id())),
                prompts: vec![text("prompt")],
                negative_prompt: text("negative_prompt"),
                count: 1,
                width,
                height,
                steps: int("steps") as usize,
                guidance: real("cfg"),
                shift: real("shift"),
                skip_layers: text("uncond_skip_layers").split(',').filter_map(|s| s.trim().parse().ok()).collect(),
                distill: None,
                seed: int("seed") as u64,
                device: 0,
            };
            let clock = Instant::now();
            let result = generate_with(&r, |_, n| {
                assert_eq!(n, start.len(), "{name}: the noise's size is the picture's");
                start.clone()
            }, |_| {})?;
            let made = result.get("data").and_then(|d| d.at(0)).and_then(|d| d.get("path")).and_then(Json::as_str).map(PathBuf::from).unwrap();
            let (ours, theirs) = (image::open(&made).map_err(candle_core::Error::wrap)?.to_rgb8(), image::open(examples.join(&name).join("image.png")).map_err(candle_core::Error::wrap)?.to_rgb8());
            assert_eq!(ours.dimensions(), theirs.dimensions(), "{name}");
            let mse = ours.as_raw().iter().zip(theirs.as_raw()).map(|(a, b)| ((*a as f64 - *b as f64) / 255.).powi(2)).sum::<f64>() / ours.as_raw().len() as f64;
            let psnr = -10. * mse.log10();
            eprintln!("{name}: {width}x{height}, {} steps, {psnr:.1} dB against OpenMOSS's, in {:.1} s ({})", r.steps, clock.elapsed().as_secs_f64(), made.display());
            worst = worst.min(psnr);
        }
        // (guided steps carry bfloat16's own rounding into the fine details: on the RTX 4090 the same pictures, those at
        // cfg 3 or 4 26.8 to 31.5 dB from theirs, those at cfg 5 18.6 to 22.5, their figures, poses, lettering and
        // lighting the same)
        assert!(worst > 16., "the worst example {worst:.1} dB from OpenMOSS's");
        Ok(())
    }

    /// A job's request: the tokenizer beside the text encoder where none is named (a GGUF's own where there is none),
    /// the pipeline's defaults (1024 x 1024, 50 steps, cfg 3, shift 5, block 22 skipped), sizes of multiples of 16
    /// within 2048 x 2048's area.
    #[test]
    fn requests_are_checked() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/moss_image/tiny.safetensors");
        let dir = std::env::temp_dir().join(format!("oaiy-moss-request-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let gguf = dir.join("qwen.gguf");
        std::fs::write(&gguf, "GGUF").unwrap();
        let f = fixture.to_string_lossy().into_owned();
        let body = |encoder: &str, more: &str| Json::parse(format!(r#"{{"architecture":"moss-image","transformer":"{f}","vae":"{f}","text_encoder":"{encoder}","prompt":"a fox","output_dir":"{}"{more}}}"#, dir.join("out").display()).as_bytes()).unwrap();
        let folder = dir.to_string_lossy().into_owned();
        let r = Request::parse(&body(&folder, "")).unwrap();
        assert_eq!((r.tokenizer.clone(), r.width, r.height, r.steps, r.guidance, r.shift, r.skip_layers.clone(), r.negative_prompt.clone()), (Some(dir.join("tokenizer.json")), 1024, 1024, 50, 3., 5., vec![22], String::new()));
        let r = Request::parse(&body(&folder, r#","width":1184,"height":1776,"steps":30,"cfg":5,"shift":6,"uncond_skip_layers":[],"negative_prompt":"blur""#)).unwrap();
        assert_eq!((r.width, r.height, r.steps, r.guidance, r.shift, r.skip_layers.len(), r.negative_prompt.as_str()), (1184, 1776, 30, 5., 6., 0, "blur"));
        // (a GGUF's own tokenizer where none is beside it)
        std::fs::remove_file(dir.join("tokenizer.json")).unwrap();
        assert_eq!(Request::parse(&body(&gguf.to_string_lossy(), "")).unwrap().tokenizer, None);
        assert!(Request::parse(&body(&folder, "")).is_err());
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        for bad in [
            r#","width":1000,"height":1024"#,
            r#","width":4096,"height":2048"#,
            r#","steps":0"#,
            r#","cfg":0.5"#,
            r#","uncond_skip_layers":[-1]"#,
            r#","backend":"cpu""#,
            r#","loras":["/no/such/lora.safetensors"]"#,
            r#","images":["/no/such/picture.png"]"#,
        ] {
            assert!(Request::parse(&body(&folder, bad)).is_err(), "{bad}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
