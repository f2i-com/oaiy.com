//! FLUX 3 Image Klein: text-to-image, and edits of up to 10 reference pictures, through one model, as BFL's
//! reference pipeline (`flux3/pipeline.py`, `flux3/generate.py`) makes them, on WebGPU: the prompt through
//! Qwen3-VL-4B's language model ([`crate::klein::webgpu::text`] with [`crate::klein::webgpu::text::FLUX3`]), each reference
//! through the FLUX.2 VAE's encoder ([`crate::sdxl::webgpu::vae`]), the transformer ([`crate::flux3::webgpu`]) a flow map
//! over 16 steps (the distilled checkpoint: each step told the time it goes to) or 50 steps with guidance (the base
//! one), the VAE's decoder last; LoRA adapters merged into the transformer as it loads. The prompt is used as
//! written: BFL's prompt upsampler (a separate model that writes a full scene caption from a short request) is not
//! run, nor their safety classifier.
pub mod references;
#[cfg(feature = "webgpu")]
pub mod webgpu;

use oaiy_engine::json::Json;
use std::{io::Write, path::PathBuf, time::Instant};

/// The schedule's shift: `alpha t / (1 + (alpha - 1) t)` of `t` evenly from 1 to 0.
const ALPHA: f32 = 6.93;
pub const MAX_REFERENCES: usize = 10;
/// The picture's area at most: 2048 x 2048, the largest a reference's canvas takes.
pub const MAX_PIXELS: usize = 4 * 1024 * 1024;

#[derive(Debug)]
pub struct Request {
    pub transformer: PathBuf,
    /// Qwen3-VL-4B-Instruct: its folder (or a safetensors file of it).
    pub text_encoder: PathBuf,
    pub tokenizer: PathBuf,
    pub vae: PathBuf,
    /// LoRA adapters, each a path and its strength (1: as trained).
    pub loras: Vec<(PathBuf, f64)>,
    pub output: PathBuf,
    pub prompts: Vec<String>,
    pub references: Vec<PathBuf>,
    /// A single reference's canvas at most this many pixels (BFL's: 4 MiB).
    pub reference_pixels: usize,
    pub count: usize,
    /// The picture's size: given, or the first reference's (prepared), or 1024 x 1024.
    pub size: Option<(usize, usize)>,
    /// None: the checkpoint's own (16 for the distilled flow map, 50 for the base one).
    pub steps: Option<usize>,
    /// The base checkpoint's guidance (None: 3.5); the distilled one takes none.
    pub guidance: Option<f64>,
    pub seed: u64,
    pub device: usize,
}

/// `t` evenly from 1 to 0 in `steps`, shifted by `alpha` (`sampling.timesteps`), in f32 as torch makes it.
pub fn schedule(steps: usize, alpha: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            // (torch.linspace's: the first half from the start, the second from the end)
            let step = -1.0f32 / steps as f32;
            let t = if i < steps.div_ceil(2) { 1.0 + step * i as f32 } else { 0.0 - step * (steps - i) as f32 };
            alpha * t / (1.0 + (alpha - 1.0) * t)
        })
        .collect()
}

/// A picture's latent positions: (0, row, column, `l`), rows and columns times [`crate::flux3::webgpu::IMAGE_ROPE_SCALE`].
pub fn latent_ids(h: usize, w: usize, l: f64) -> Vec<[f64; 4]> {
    let s = crate::flux3::webgpu::IMAGE_ROPE_SCALE;
    (0..h * w).map(|i| [0., (i / w) as f64 * s, (i % w) as f64 * s, l]).collect()
}

/// A value as bfloat16 holds it (rounded to the nearest, ties to even): the pipeline's noise is drawn so.
fn bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        if !crate::common::job::backend_is_webgpu(j, "FLUX 3 Image Klein")? {
            return Err("FLUX 3 Image Klein runs on WebGPU only: leave backend out".into());
        }
        let path = |key: &str, dir: bool| -> std::result::Result<PathBuf, String> {
            let p = PathBuf::from(j.get(key).and_then(Json::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| format!("FLUX 3 Image Klein requires {key}"))?);
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
        let text_encoder = path("text_encoder", true)?;
        // (the folder's own tokenizer, or, for a GGUF of the encoder, the `tokenizer.json` beside it)
        let tokenizer = match j.get("tokenizer") {
            None | Some(Json::Null) if text_encoder.is_dir() => text_encoder.join("tokenizer.json"),
            None | Some(Json::Null) => text_encoder.with_file_name("tokenizer.json"),
            _ => path("tokenizer", false)?,
        };
        if !tokenizer.is_file() {
            return Err(format!("FLUX 3 Image Klein requires a tokenizer: {} is not a file", tokenizer.display()));
        }
        if j.get("adapter").is_some_and(|v| !matches!(v, Json::Null)) {
            return Err("FLUX 3 Image Klein takes no turbo adapter: give LoRAs as loras".into());
        }
        let loras = crate::common::job::parse_loras(j)?;
        for (p, strength) in &loras {
            if !p.is_file() {
                return Err(format!("LoRA not found: {}", p.display()));
            }
            if !strength.is_finite() || !(-4. ..=4.).contains(strength) {
                return Err("a LoRA's strength must be between -4 and 4".into());
            }
        }
        if j.get("negative_prompt").is_some_and(|v| !matches!(v, Json::Null) && !v.as_str().is_some_and(|s| s.trim().is_empty())) {
            return Err("FLUX 3 Image Klein has no negative prompt: describe what the picture is instead".into());
        }
        if j.get("turbo").is_some_and(|v| v.as_bool() != Some(false)) {
            return Err("FLUX 3 Image Klein does not use Qwen turbo".into());
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
        let references = match j.get("images") {
            None | Some(Json::Null) => Vec::new(),
            Some(v) => v
                .as_array()
                .ok_or(format!("images must be an array of up to {MAX_REFERENCES} local paths"))?
                .iter()
                .map(|v| v.as_str().filter(|s| !s.trim().is_empty()).map(PathBuf::from).ok_or_else(|| "images must contain nonempty local paths".to_owned()))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        if references.len() > MAX_REFERENCES {
            return Err(format!("FLUX 3 Image Klein takes at most {MAX_REFERENCES} reference pictures"));
        }
        if let Some(p) = references.iter().find(|p| !p.is_file()) {
            return Err(format!("reference picture not found: {}", p.display()));
        }
        let size = match (j.get("width").filter(|v| !matches!(v, Json::Null)), j.get("height").filter(|v| !matches!(v, Json::Null))) {
            (None, None) => None,
            _ => {
                let (width, height) = (number("width", 1024, 256, 4096)?, number("height", 1024, 256, 4096)?);
                if width % 16 != 0 || height % 16 != 0 {
                    return Err("FLUX 3 Image Klein's width and height must be multiples of 16".into());
                }
                if width * height > MAX_PIXELS {
                    return Err(format!("FLUX 3 Image Klein makes pictures of at most {MAX_PIXELS} pixels (2048 x 2048)"));
                }
                Some((width, height))
            }
        };
        let steps = match j.get("steps") {
            None | Some(Json::Null) => None,
            _ => Some(number("steps", 16, 1, 100)?),
        };
        let guidance = match j.get("cfg").or_else(|| j.get("guidance")) {
            None | Some(Json::Null) => None,
            Some(v) => {
                let g = v.as_f64().ok_or("cfg must be a number")?;
                if !g.is_finite() || !(1. ..=10.).contains(&g) {
                    return Err("cfg must be between 1 and 10".into());
                }
                Some(g)
            }
        };
        let seed = number("seed", 0, 0, i64::MAX - count as i64)? as u64;
        let output = j.get("output_dir").and_then(Json::as_str).filter(|s| !s.trim().is_empty()).ok_or("output_dir is required")?.into();
        Ok(Self {
            transformer: path("transformer", false)?,
            text_encoder,
            tokenizer,
            vae: path("vae", false)?,
            loras,
            output,
            prompts,
            references,
            reference_pixels: number("reference_pixels", MAX_PIXELS as i64, 64 * 64, MAX_PIXELS as i64)?,
            count,
            size,
            steps,
            guidance,
            seed,
            device: number("device", 0, 0, 31)?,
        })
    }
}

#[cfg(feature = "webgpu")]
pub fn generate(r: &Request, event: impl FnMut(Json)) -> candle_core::Result<Json> {
    generate_with(r, crate::common::job::noise, event)
}

/// What a worker that serves keeps of FLUX 3 Image Klein between pictures ([`generate_kept`]): its GPU, its transformer
/// (with the LoRAs merged into it) and its VAE (with its encoder where a picture had references), each for as long as
/// the next job names the same files. The text encoder is not kept: it reads its layers from its file for each prompt.
#[cfg(feature = "webgpu")]
#[derive(Default)]
pub struct Kept {
    gpu: Option<(usize, ggml_rs_wgpu::WgpuBackend)>,
    model: Option<(String, webgpu::WgpuFlux3)>,
    vae: Option<((String, bool), Vae)>,
}

#[cfg(feature = "webgpu")]
type Vae = (crate::sdxl::webgpu::vae::WgpuSdxlVae, Vec<f32>, Vec<f32>);

/// [`generate`] with what `kept` has from the jobs before it, and what this one loads kept for the next.
#[cfg(feature = "webgpu")]
pub fn generate_kept(r: &Request, kept: &mut Kept, event: impl FnMut(Json)) -> candle_core::Result<Json> {
    generate_inner(r, Some(kept), crate::common::job::noise, event)
}

#[cfg(not(feature = "webgpu"))]
pub fn generate(_: &Request, _: impl FnMut(Json)) -> candle_core::Result<Json> {
    candle_core::bail!("this build has no WebGPU (the webgpu feature)")
}

/// [`generate`], each picture's starting noise `noise(seed, values)` (a test gives the reference's own).
#[cfg(feature = "webgpu")]
pub fn generate_with(r: &Request, noise: impl Fn(u64, usize) -> Vec<f32>, event: impl FnMut(Json)) -> candle_core::Result<Json> {
    generate_inner(r, None, noise, event)
}

#[cfg(feature = "webgpu")]
fn generate_inner(r: &Request, mut kept: Option<&mut Kept>, noise: impl Fn(u64, usize) -> Vec<f32>, mut event: impl FnMut(Json)) -> candle_core::Result<Json> {
    use crate::flux3::webgpu::{Config, Tokens, WgpuFlux3};
    use crate::klein::vae;
    use candle_core::bail;
    // the transformer's header first: which checkpoint, before anything is loaded
    let cfg = Config::of(&crate::common::weights::Weights::open(&r.transformer)?)?;
    if cfg.context != 20480 {
        bail!("FLUX 3 Image Klein's transformer takes text states 20,480 wide, not {}", cfg.context);
    }
    // and its LoRAs': each fits the checkpoint (from their headers alone)
    drop(crate::flux3::webgpu::loras(&r.loras, cfg)?);
    let steps = r.steps.unwrap_or(if cfg.flow_map { 16 } else { 50 });
    let guidance = match (cfg.flow_map, r.guidance) {
        (true, None | Some(1.)) => None,
        (true, Some(_)) => bail!("the distilled FLUX 3 Image Klein samples without guidance: leave cfg out"),
        (false, g) => Some(g.unwrap_or(3.5)),
    };
    let clock = Instant::now();
    event(Json::obj([("stage", Json::str("encoding_text")), ("architecture", Json::str("flux3-klein"))]));
    let (contexts, empty) = {
        let mut encoder = crate::klein::webgpu::text::WgpuKleinText::load_spec(&r.text_encoder, &r.tokenizer, r.device, crate::klein::webgpu::text::FLUX3)?;
        let prompts: Vec<&str> = r.prompts.iter().map(String::as_str).chain(guidance.map(|_| "")).collect();
        let mut all = encoder.encode_all(&prompts)?;
        let empty = if guidance.is_some() { all.pop() } else { None };
        let host = |t: candle_core::Tensor| -> candle_core::Result<(Vec<f32>, usize)> { Ok((t.flatten_all()?.to_vec1::<f32>()?, t.dim(1)?)) };
        (all.into_iter().map(host).collect::<candle_core::Result<Vec<_>>>()?, empty.map(host).transpose()?)
    };
    // the GPU, the VAE and the transformer: the kept ones where the job names the same (a GPU of another number lets go
    // of what was kept on the last)
    let gpu = match kept.as_deref_mut().and_then(|k| k.gpu.clone()).filter(|(d, _)| *d == r.device) {
        Some((_, g)) => g,
        None => {
            let g = ggml_rs_wgpu::WgpuBackend::nth(r.device, None).map_err(crate::qwen_image::webgpu::transformer::err)?;
            if let Some(k) = kept.as_deref_mut() {
                *k = Kept { gpu: Some((r.device, g.clone())), ..Kept::default() };
            }
            g
        }
    };
    let with_encoder = !r.references.is_empty();
    let vae_key = r.vae.to_string_lossy().into_owned();
    let had = kept.as_deref_mut().and_then(|k| k.vae.take()).filter(|((path, encoder), _)| *path == vae_key && (*encoder || !with_encoder));
    let (vae_encoder, (decoder, mean, std)) = match had {
        Some(((_, encoder), v)) => (encoder, v),
        None => {
            event(Json::obj([("stage", Json::str("loading_vae"))]));
            (with_encoder, vae::vae_on_webgpu(&r.vae, gpu.clone(), with_encoder)?)
        }
    };
    // the references: each prepared as the pipeline prepares it, encoded, its tokens told apart by their last axis
    let prepared = references::prepare(&r.references, r.reference_pixels)?;
    let (mut cond, mut cond_ids) = (Vec::new(), Vec::new());
    for (index, picture) in prepared.pictures.iter().enumerate() {
        event(Json::obj([("stage", Json::str("encoding_references")), ("image", Json::Int(index as i64 + 1)), ("images", Json::Int(prepared.pictures.len() as i64))]));
        let (moments, lh, lw) = decoder.encode(&picture.rgb, picture.height, picture.width)?;
        cond.extend(vae::pack_rows(&moments, lh / 2, lw / 2, &mean, &std));
        cond_ids.extend(latent_ids(lh / 2, lw / 2, -(1. + index as f64)));
    }
    let (width, height) = r.size.or(prepared.size).unwrap_or((1024, 1024));
    // a padded single reference's picture comes out cut to the reference's own, where no size was asked for
    let crop = if r.size.is_none() { prepared.crop } else { None };
    if width % 16 != 0 || height % 16 != 0 || width * height > MAX_PIXELS {
        bail!("FLUX 3 Image Klein cannot make a {width}x{height} picture");
    }
    let model_key = format!("{}|{:?}", r.transformer.display(), r.loras);
    let had = kept.as_deref_mut().and_then(|k| k.model.take()).filter(|(key, _)| *key == model_key).map(|(_, m)| m);
    let mut model = match had {
        Some(m) => m,
        None => {
            event(Json::obj([("stage", Json::str("loading_transformer"))]));
            let m = WgpuFlux3::load_on(&r.transformer, gpu.clone(), &r.loras, |_, _| {})?;
            for note in m.lora_notes() {
                event(Json::obj([("stage", Json::str("lora_note")), ("note", Json::str(note))]));
            }
            m
        }
    };
    std::fs::create_dir_all(&r.output)?;
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let batch = r.output.join(format!("flux3-{}-{nanos}", std::process::id()));
    std::fs::create_dir(&batch)?;
    let mut manifest = std::fs::OpenOptions::new().create_new(true).write(true).open(batch.join("manifest.jsonl"))?;
    let (h, w) = (height / 16, width / 16);
    let image_ids = latent_ids(h, w, 0.);
    let times = schedule(steps, ALPHA);
    let text_ids = |n: usize| -> Vec<[f64; 4]> { (0..n).map(|p| [0., 0., 0., p as f64]).collect() };
    let mut data = Vec::new();
    for i in 0..r.count {
        let seed = r.seed + i as u64;
        let prompt = &r.prompts[if r.prompts.len() == 1 { 0 } else { i }];
        let (text, nt) = &contexts[if contexts.len() == 1 { 0 } else { i }];
        let ids = text_ids(*nt);
        // bfloat16 values, f32 as the steps go
        let mut x: Vec<f32> = noise(seed, h * w * 128).into_iter().map(bf16).collect();
        let sample = Instant::now();
        for step in 0..steps {
            let (t, next) = (times[step], times[step + 1]);
            let dt = next - t;
            let velocity = {
                let at = |x: &[f32], text: &[f32], text_ids: &[[f64; 4]], model: &mut WgpuFlux3| {
                    model.predict(&Tokens { image: x, image_ids: &image_ids, cond: &cond, cond_ids: &cond_ids, text, text_ids }, t as f64, cfg.flow_map.then_some(next as f64))
                };
                let conditional = at(&x, text, &ids, &mut model)?;
                match (&empty, guidance) {
                    (Some((uncond, nu)), Some(g)) => {
                        let unconditional = at(&x, uncond, &text_ids(*nu), &mut model)?;
                        unconditional.iter().zip(&conditional).map(|(u, c)| u + g as f32 * (c - u)).collect()
                    }
                    _ => conditional,
                }
            };
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
        if (ph, pw) != (height, width) || pixels.len() != width * height * 3 {
            bail!("FLUX 3's decoder made a {pw}x{ph} picture of {} values for {width}x{height}", pixels.len());
        }
        if pixels.iter().any(|v| !v.is_finite()) {
            bail!("FLUX 3 Image Klein produced non-finite pixels; no image saved");
        }
        // (`to_pil`: clamped, then cut to 8 bits as torch's uint8 cast does)
        let bytes: Vec<u8> = pixels.iter().map(|v| ((v.clamp(-1., 1.) + 1.) * 127.5) as u8).collect();
        let mut picture = image::RgbImage::from_raw(width as u32, height as u32, bytes).ok_or_else(|| crate::qwen_image::webgpu::transformer::err("a picture's pixels"))?;
        if let Some((left, top, right, bottom)) = crop {
            picture = image::imageops::crop_imm(&picture, left as u32, top as u32, (right - left) as u32, (bottom - top) as u32).to_image();
        }
        let path = batch.join(format!("image-{:04}-seed-{seed}.png", i + 1));
        picture.save(&path).map_err(candle_core::Error::wrap)?;
        let record = Json::obj([
            ("path", Json::str(path.to_string_lossy())),
            ("prompt", Json::str(prompt)),
            ("seed", Json::Int(seed as i64)),
            ("architecture", Json::str("flux3-klein")),
            ("variant", Json::str(if cfg.flow_map { "distilled" } else { "base" })),
            ("steps", Json::Int(steps as i64)),
            ("cfg", guidance.map_or(Json::Null, Json::Num)),
            ("width", Json::Int(picture.width() as i64)),
            ("height", Json::Int(picture.height() as i64)),
            ("references", Json::Int(r.references.len() as i64)),
            ("transformer", Json::str(r.transformer.to_string_lossy())),
            ("loras", Json::Arr(r.loras.iter().map(|(p, s)| Json::obj([("path", Json::str(p.to_string_lossy())), ("strength", Json::Num(*s))])).collect())),
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
    if let Some(k) = kept {
        k.model = Some((model_key, model));
        k.vae = Some(((vae_key, vae_encoder), (decoder, mean, std)));
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

    /// The distilled checkpoint's 16 steps are `sampling.timesteps(16, 6.93)`'s, as torch 2.11 gives them in f32.
    #[test]
    fn the_schedule_is_bfls() {
        let want: [f32; 17] = [
            1.0, 0.9904716610908508, 0.9798020720481873, 0.9677731394767761, 0.954107403755188, 0.9384464025497437, 0.920318603515625, 0.899091899394989, 0.873896598815918,
            0.8435055017471313, 0.8061264157295227, 0.7590360641479492, 0.6978851556777954, 0.615270733833313, 0.4974874258041382, 0.3160054683685303, 0.0,
        ];
        let got = schedule(16, ALPHA);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-7, "{got:?}");
        }
        assert_eq!(schedule(4, ALPHA), [1.0, 0.954107403755188, 0.873896598815918, 0.6978851556777954, 0.0]);
    }

    /// A job's request: the tokenizer the text encoder's folder's where none is named, at most ten references that
    /// exist, a size of multiples of 16 within 2048 x 2048's area, or none (the reference's, or 1024 x 1024); LoRAs
    /// that exist, at strengths from -4 to 4.
    #[test]
    fn requests_are_checked() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/flux3/tiny.safetensors");
        let dir = std::env::temp_dir().join(format!("oaiy-flux3-request-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        let body = |more: &str| {
            let f = fixture.to_string_lossy();
            Json::parse(format!(r#"{{"architecture":"flux3-klein","transformer":"{f}","vae":"{f}","text_encoder":"{}","prompt":"a fox","output_dir":"{}"{more}}}"#, dir.display(), dir.join("out").display()).as_bytes()).unwrap()
        };
        let r = Request::parse(&body("")).unwrap();
        assert_eq!((r.tokenizer.clone(), r.size, r.steps, r.guidance, r.references.len()), (dir.join("tokenizer.json"), None, None, None, 0));
        assert_eq!(Request::parse(&body(r#","width":768,"height":512,"steps":8"#)).unwrap().size, Some((768, 512)));
        // a GGUF of the text encoder: the tokenizer beside it
        let gguf = dir.join("encoder-Q4_K_M.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        let f = fixture.to_string_lossy();
        let one = Json::parse(format!(r#"{{"architecture":"flux3-klein","transformer":"{f}","vae":"{f}","text_encoder":"{}","prompt":"a fox","output_dir":"{}"}}"#, gguf.display(), dir.join("out").display()).as_bytes()).unwrap();
        assert_eq!(Request::parse(&one).unwrap().tokenizer, dir.join("tokenizer.json"));
        let image = fixture.to_string_lossy().into_owned();
        let images = |n: usize| format!(r#","images":[{}]"#, vec![format!("\"{image}\""); n].join(","));
        assert_eq!(Request::parse(&body(&images(10))).unwrap().references.len(), 10);
        let loras = |list: &str| format!(r#","loras":[{list}]"#);
        let r = Request::parse(&body(&loras(&format!(r#"{{"path":"{image}","strength":0.75}},"{image}""#)))).unwrap();
        assert_eq!(r.loras, vec![(fixture.clone(), 0.75), (fixture.clone(), 1.)]);
        for bad in [
            images(11),
            r#","images":["/no/such/picture.png"]"#.into(),
            r#","width":1000,"height":1024"#.into(),
            r#","width":4096,"height":2048"#.into(),
            r#","negative_prompt":"blur""#.into(),
            r#","backend":"cpu""#.into(),
            loras(r#""/no/such/lora.safetensors""#),
            loras(&format!(r#"{{"path":"{image}","strength":5}}"#)),
            format!(r#","adapter":"{image}""#),
        ] {
            assert!(Request::parse(&body(&bad)).is_err(), "{bad}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A real picture (`OAIY_FLUX3_PICTURE`) prepared as a single reference, through the FLUX.2 VAE (`OAIY_FLUX3_VAE`) on
    /// WebGPU and back: encoded, packed into tokens by the published statistics, unpacked and decoded, it comes back
    /// within the VAE's own loss.
    #[cfg(feature = "webgpu")]
    #[test]
    #[ignore = "needs the FLUX.2 VAE (OAIY_FLUX3_VAE), a picture (OAIY_FLUX3_PICTURE) and a WebGPU adapter"]
    fn a_reference_round_trips_through_the_vae() -> candle_core::Result<()> {
        let (Some(vae), Some(picture)) = (std::env::var_os("OAIY_FLUX3_VAE").map(PathBuf::from), std::env::var_os("OAIY_FLUX3_PICTURE").map(PathBuf::from)) else { return Ok(()) };
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(crate::qwen_image::webgpu::transformer::err)?;
        let (model, mean, std) = crate::klein::vae::vae_on_webgpu(&vae, gpu, true)?;
        let prepared = references::prepare(&[picture], 1 << 20)?;
        let p = &prepared.pictures[0];
        let t = Instant::now();
        let (moments, lh, lw) = model.encode(&p.rgb, p.height, p.width)?;
        let tokens = crate::klein::vae::pack_rows(&moments, lh / 2, lw / 2, &mean, &std);
        let encoded = t.elapsed().as_secs_f64();
        let (rgb, h, w) = model.decode(&crate::klein::vae::unpack_rows(&tokens, lh / 2, lw / 2, &mean, &std), lh, lw)?;
        assert_eq!((w, h), (p.width, p.height));
        let mse = rgb.iter().zip(&p.rgb).map(|(a, b)| ((a.clamp(-1., 1.) - b) as f64 / 2.).powi(2)).sum::<f64>() / rgb.len() as f64;
        let psnr = -10. * mse.log10();
        let rms = (tokens.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / tokens.len() as f64).sqrt();
        eprintln!("{}x{}: encoded in {encoded:.2} s, {} tokens of RMS {rms:.3}; back at {psnr:.1} dB", p.width, p.height, tokens.len() / 128);
        assert!(psnr > 28., "{psnr} dB");
        Ok(())
    }

    /// BFL's published examples (`OAIY_FLUX3_EXAMPLES`: their repository's `examples/distilled`, or `examples/base`
    /// with the base checkpoint) made again from their records (the caption the model saw, the seed, the references)
    /// and their own starting noise (`tools/flux3/noise.py`'s files in `OAIY_FLUX3_NOISE`), by the transformer
    /// `OAIY_FLUX3_TRANSFORMER` with `OAIY_FLUX3_TEXT_ENCODER` and `OAIY_FLUX3_VAE`: each picture as close to theirs as
    /// bfloat16's sums against these let it be (`OAIY_FLUX3_ONLY`: the examples whose names hold it).
    #[cfg(feature = "webgpu")]
    #[test]
    #[ignore = "needs FLUX 3 Image Klein (OAIY_FLUX3_TRANSFORMER, OAIY_FLUX3_TEXT_ENCODER, OAIY_FLUX3_VAE), BFL's examples (OAIY_FLUX3_EXAMPLES, OAIY_FLUX3_NOISE) and a WebGPU adapter"]
    fn the_examples_are_bfls() -> candle_core::Result<()> {
        let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
        let (Some(transformer), Some(text_encoder), Some(vae), Some(examples), Some(noise)) = (var("OAIY_FLUX3_TRANSFORMER"), var("OAIY_FLUX3_TEXT_ENCODER"), var("OAIY_FLUX3_VAE"), var("OAIY_FLUX3_EXAMPLES"), var("OAIY_FLUX3_NOISE")) else { return Ok(()) };
        let only = std::env::var("OAIY_FLUX3_ONLY").unwrap_or_default();
        // (a record's references are its repository's paths)
        let repo = examples.parent().and_then(|p| p.parent()).ok_or_else(|| crate::qwen_image::webgpu::transformer::err("examples/<kind> in BFL's repository"))?;
        let mut records: Vec<PathBuf> = std::fs::read_dir(&examples)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect();
        records.sort();
        let mut worst = f64::INFINITY;
        for record in records {
            let name = record.file_stem().unwrap().to_string_lossy().into_owned();
            if !name.contains(&only) {
                continue;
            }
            let j = Json::parse(&std::fs::read(&record)?).map_err(candle_core::Error::wrap)?;
            let start: Vec<f32> = std::fs::read(noise.join(format!("{name}.f32")))?.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let r = Request {
                transformer: transformer.clone(),
                text_encoder: text_encoder.clone(),
                tokenizer: text_encoder.join("tokenizer.json"),
                vae: vae.clone(),
                loras: Vec::new(),
                output: std::env::temp_dir().join(format!("oaiy-flux3-examples-{}", std::process::id())),
                prompts: vec![j.get("prompt").and_then(Json::as_str).unwrap_or_default().to_owned()],
                references: j.get("references").and_then(Json::as_array).unwrap_or(&[]).iter().filter_map(Json::as_str).map(|r| repo.join(r)).collect(),
                reference_pixels: MAX_PIXELS,
                count: 1,
                size: None,
                steps: None,
                guidance: None,
                seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0) as u64,
                device: 0,
            };
            let clock = Instant::now();
            let result = generate_with(&r, |_, n| {
                assert_eq!(n, start.len(), "{name}: the noise's size is the picture's");
                start.clone()
            }, |_| {})?;
            let made = result.get("data").and_then(|d| d.at(0)).and_then(|d| d.get("path")).and_then(Json::as_str).map(PathBuf::from).unwrap();
            let (ours, theirs) = (image::open(&made).map_err(candle_core::Error::wrap)?.to_rgb8(), image::open(examples.join(format!("{name}.png"))).map_err(candle_core::Error::wrap)?.to_rgb8());
            assert_eq!(ours.dimensions(), theirs.dimensions(), "{name}");
            let mse = ours.as_raw().iter().zip(theirs.as_raw()).map(|(a, b)| ((*a as f64 - *b as f64) / 255.).powi(2)).sum::<f64>() / ours.as_raw().len() as f64;
            let psnr = -10. * mse.log10();
            eprintln!("{name}: {}x{}, {psnr:.1} dB against BFL's, in {:.1} s ({})", ours.width(), ours.height(), clock.elapsed().as_secs_f64(), made.display());
            worst = worst.min(psnr);
        }
        assert!(worst > 20., "the worst example {worst:.1} dB from BFL's");
        Ok(())
    }

    #[test]
    fn latent_positions_are_halved_rows_and_columns() {
        assert_eq!(latent_ids(2, 3, -1.)[5], [0., 0.5, 1.0, -1.]);
    }
}
