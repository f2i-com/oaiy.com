//! Pixal3D (TencentARC, MIT): one picture of an object to a 3D model (GLB),
//! natively on Candle. TRELLIS.2's cascade with Pixal3D's pixel-aligned
//! conditioning (each voxel sees the image features where it projects):
//!
//! 1. the picture, cut out (BiRefNet) and squared, a small one enlarged
//!    (Real-ESRGAN); DINOv3 at 512 and 1024 (and NAF to upsample its
//!    features) describe it;
//! 2. a flow transformer makes the coarse structure (a 16³ latent, decoded to
//!    64³ occupancy and pooled to 32³ voxels);
//! 3. a second makes the shape latent on those voxels (at 512), whose decoder
//!    proposes the finer voxels; a third makes it again there (at 1024, or 1536);
//! 4. a fourth makes the texture latent on the same voxels;
//! 5. the decoders turn both into the surface (a flexible dual grid: one vertex
//!    per voxel, quads where edges cross) and its PBR colours; the surface is
//!    remeshed (closed and manifold), simplified, unwrapped, and its colours are
//!    baked into textures, written as a GLB.
//!
//! One model is on the GPU at a time. The reference's own steps, guidance and
//! normalization come from its pipeline.json.
pub mod bake;
pub mod decoder;
pub mod dinov3;
pub mod dit;
pub mod glb;
pub mod mesh;
pub mod naf;
pub mod prepare;
pub mod proj;
pub mod remesh;
pub mod sparse;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod verify;

use candle_core::{DType, Device, Result, Tensor};
use oaiy_engine::json::Json;
use sparse::Level;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct Request {
    pub image: PathBuf,
    /// The Pixal3D folder (pipeline.json and ckpts/).
    pub model_dir: PathBuf,
    /// DINOv3 ViT-L/16 (transformers layout).
    pub dino_dir: PathBuf,
    /// NAF's naf_release.pth.
    pub naf: PathBuf,
    pub output: PathBuf,
    pub seed: u64,
    /// 1024 or 1536: the finest grid the shape reaches.
    pub resolution: usize,
    /// The camera's horizontal field of view the picture was taken with, degrees.
    pub fov_degrees: f64,
    pub steps: Option<usize>,
    pub max_tokens: usize,
    /// The simplified mesh's triangle budget.
    pub faces: usize,
    pub device: usize,
    /// Testing: each stage's starting noise from this folder (F32 LE: ss.bin, shape_lr.bin,
    /// shape_hr.bin, tex.bin) instead of the seed's.
    pub noise_dir: Option<PathBuf>,
    /// Testing: every intermediate written here.
    pub dump_dir: Option<PathBuf>,
    /// The picture is already prepared (square, the object on black): use it as it is.
    pub prepared: bool,
    /// Testing: the voxels of the structure (coords32.i32) and of the finer shape
    /// (coords_hr.i32) from this folder, instead of the ones made here.
    pub coords_dir: Option<PathBuf>,
    /// The remeshing grid (default: the shape's own resolution).
    pub remesh: Option<usize>,
    /// The baked textures' size (a power of two, 512 to 4096), or 0 for vertex colours only.
    pub texture_size: u32,
    /// BiRefNet's folder: cuts the object out of a picture without transparency.
    pub matte: Option<PathBuf>,
    /// Real-ESRGAN x4plus's weights: enlarges a small picture before it is seen.
    pub upscaler: Option<PathBuf>,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_string);
        let model_dir = PathBuf::from(s("model_dir").ok_or("model_dir is required")?);
        let parent = model_dir.parent().map(Path::to_path_buf).unwrap_or_default();
        let r = Self {
            image: PathBuf::from(s("image").ok_or("image is required")?),
            dino_dir: s("dino_dir").map(PathBuf::from).unwrap_or_else(|| parent.join("dinov3-vitl16")),
            naf: s("naf").map(PathBuf::from).unwrap_or_else(|| parent.join("NAF").join("naf_release.pth")),
            output: PathBuf::from(s("output_dir").ok_or("output_dir is required")?),
            seed: j.get("seed").and_then(Json::as_i64).unwrap_or(0).max(0) as u64,
            resolution: j.get("resolution").and_then(Json::as_i64).unwrap_or(1024) as usize,
            fov_degrees: j.get("fov_degrees").and_then(Json::as_f64).unwrap_or(30.),
            steps: j.get("steps").and_then(Json::as_i64).map(|v| v as usize),
            max_tokens: j.get("max_tokens").and_then(Json::as_i64).unwrap_or(49152) as usize,
            faces: j.get("faces").and_then(Json::as_i64).unwrap_or(200_000) as usize,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            noise_dir: s("noise_dir").map(PathBuf::from),
            dump_dir: s("dump_dir").map(PathBuf::from),
            prepared: j.get("prepared").and_then(Json::as_bool).unwrap_or(false),
            coords_dir: s("coords_dir").map(PathBuf::from),
            remesh: j.get("remesh").and_then(Json::as_i64).map(|v| v as usize),
            texture_size: j.get("texture_size").and_then(Json::as_i64).unwrap_or(2048).max(0) as u32,
            matte: s("matte").filter(|v| !v.is_empty()).map(PathBuf::from),
            upscaler: s("upscaler").filter(|v| !v.is_empty()).map(PathBuf::from),
            model_dir,
        };
        if r.resolution != 1024 && r.resolution != 1536 {
            return Err("3d: resolution must be 1024 or 1536".into());
        }
        if !(5.0..=120.0).contains(&r.fov_degrees) {
            return Err("3d: fov_degrees must be 5..120".into());
        }
        if r.steps.is_some_and(|s| !(1..=100).contains(&s)) {
            return Err("3d: steps must be 1..100".into());
        }
        if !(1_000..=5_000_000).contains(&r.faces) {
            return Err("3d: faces must be 1000..5000000".into());
        }
        if r.remesh.is_some_and(|v| !(64..=2048).contains(&v)) {
            return Err("3d: remesh must be 64..2048".into());
        }
        if r.texture_size != 0 && !(r.texture_size.is_power_of_two() && (512..=4096).contains(&r.texture_size)) {
            return Err("3d: texture_size must be 0, 512, 1024, 2048 or 4096".into());
        }
        Ok(r)
    }
}

fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

fn device(index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(index)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = index;
        candle_core::bail!("3D models need a GPU: this oaiy-media was built without CUDA (build it with --features cuda or flash-attn)")
    }
}

/// Seeded standard normal noise (splitmix64, Box-Muller).
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (uniform(), uniform());
        let r = (-2. * u1.ln()).sqrt();
        let a = std::f64::consts::TAU * u2;
        out.push((r * a.cos()) as f32);
        out.push((r * a.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// A stage's starting noise: from the test folder, or the seed's.
fn start_noise(r: &Request, name: &str, stage: u64, n: usize) -> Result<Vec<f32>> {
    match &r.noise_dir {
        Some(dir) => {
            let p = dir.join(format!("{name}.bin"));
            let bytes = std::fs::read(&p)?;
            if bytes.len() != n * 4 {
                candle_core::bail!("3d: {} holds {} bytes, not {n} F32", p.display(), bytes.len());
            }
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        None => Ok(noise(r.seed.wrapping_mul(0x1000_0001).wrapping_add(stage), n)),
    }
}

/// Testing: voxel coordinates from the reference, when it gave them.
fn given_coords(r: &Request, name: &str) -> Result<Option<Vec<[i32; 3]>>> {
    let Some(dir) = &r.coords_dir else { return Ok(None) };
    let bytes = std::fs::read(dir.join(format!("{name}.i32")))?;
    let v: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(Some(v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()))
}

fn dump(r: &Request, name: &str, t: &Tensor) -> Result<()> {
    let Some(dir) = &r.dump_dir else { return Ok(()) };
    std::fs::create_dir_all(dir)?;
    let t = t.to_dtype(DType::F32)?.flatten_all()?;
    let v: Vec<f32> = t.to_vec1()?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.bin")), bytes)?;
    Ok(())
}

fn dump_coords(r: &Request, name: &str, coords: &[[i32; 3]]) -> Result<()> {
    let Some(dir) = &r.dump_dir else { return Ok(()) };
    std::fs::create_dir_all(dir)?;
    let bytes: Vec<u8> = coords.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(dir.join(format!("{name}.i32")), bytes)?;
    Ok(())
}

/// A sampler's settings, from pipeline.json.
struct Sampling {
    steps: usize,
    guidance: f64,
    rescale: f64,
    interval: (f64, f64),
    rescale_t: f64,
}

impl Sampling {
    fn read(pipeline: &Json, key: &str, steps: Option<usize>) -> Self {
        let p = pipeline.get("args").and_then(|a| a.get(key)).and_then(|s| s.get("params"));
        let f = |k: &str, d: f64| p.and_then(|p| p.get(k)).and_then(Json::as_f64).unwrap_or(d);
        let interval = p.and_then(|p| p.get("guidance_interval")).and_then(Json::as_array).map(|a| (a.first().and_then(Json::as_f64).unwrap_or(0.), a.get(1).and_then(Json::as_f64).unwrap_or(1.))).unwrap_or((0., 1.));
        Self { steps: steps.unwrap_or(f("steps", 12.) as usize), guidance: f("guidance_strength", 1.), rescale: f("guidance_rescale", 0.), interval, rescale_t: f("rescale_t", 1.) }
    }
}

const SIGMA_MIN: f64 = 1e-5;

/// The spread of a prediction, as the reference measures it: the dense stage with
/// `torch.std` (unbiased), the sparse ones over all values (population).
fn spread(x: &Tensor, unbiased: bool) -> Result<f64> {
    let n = x.elem_count() as f64;
    let x = x.to_dtype(DType::F32)?.flatten_all()?;
    let mean = (x.sum_all()?.to_scalar::<f32>()? as f64) / n;
    let ss = (x.affine(1., -mean)?.sqr()?.sum_all()?.to_scalar::<f32>()?) as f64;
    Ok((ss / if unbiased { n - 1. } else { n }).sqrt())
}

/// Flow-matching Euler sampling with guidance inside an interval and its rescale
/// (TRELLIS's `FlowEulerGuidanceIntervalSampler`). `model(x, t·1000, conditional)`.
fn sample(mut x: Tensor, s: &Sampling, unbiased: bool, mut model: impl FnMut(&Tensor, f64, bool) -> Result<Tensor>, mut report: impl FnMut(usize, usize)) -> Result<Tensor> {
    let ts: Vec<f64> = (0..=s.steps).map(|i| 1. - i as f64 / s.steps as f64).map(|t| s.rescale_t * t / (1. + (s.rescale_t - 1.) * t)).collect();
    for i in 0..s.steps {
        report(i, s.steps);
        let (t, t_prev) = (ts[i], ts[i + 1]);
        let guided = s.guidance != 1. && s.interval.0 <= t && t <= s.interval.1;
        let pred = if !guided {
            model(&x, 1000. * t, true)?
        } else {
            let pos = model(&x, 1000. * t, true)?;
            let neg = model(&x, 1000. * t, false)?;
            let mut pred = ((&pos * s.guidance)? + (&neg * (1. - s.guidance))?)?;
            if s.rescale > 0. {
                let k = SIGMA_MIN + (1. - SIGMA_MIN) * t;
                let x0_pos = ((&x * (1. - SIGMA_MIN))? - (&pos * k)?)?;
                let x0_cfg = ((&x * (1. - SIGMA_MIN))? - (&pred * k)?)?;
                let ratio = spread(&x0_pos, unbiased)? / spread(&x0_cfg, unbiased)?;
                let x0 = ((&x0_cfg * (s.rescale * ratio))? + (&x0_cfg * (1. - s.rescale))?)?;
                pred = (((&x * (1. - SIGMA_MIN))? - x0)? / k)?;
            }
            pred
        };
        x = (x - (pred * (t - t_prev))?)?;
    }
    report(s.steps, s.steps);
    Ok(x)
}

fn normalization(pipeline: &Json, key: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let n = pipeline.get("args").and_then(|a| a.get(key)).ok_or_else(|| candle_core::Error::Msg(format!("pipeline.json: no {key}")))?;
    let v = |k: &str| n.get(k).and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_f64).map(|x| x as f32).collect::<Vec<_>>()).unwrap_or_default();
    Ok((v("mean"), v("std")))
}

fn denormalize(x: &Tensor, (mean, std): &(Vec<f32>, Vec<f32>)) -> Result<Tensor> {
    let dev = x.device();
    let c = mean.len();
    x.broadcast_mul(&Tensor::from_vec(std.clone(), (1, c), dev)?)?.broadcast_add(&Tensor::from_vec(mean.clone(), (1, c), dev)?)
}

fn normalize(x: &Tensor, (mean, std): &(Vec<f32>, Vec<f32>)) -> Result<Tensor> {
    let dev = x.device();
    let c = mean.len();
    x.broadcast_sub(&Tensor::from_vec(mean.clone(), (1, c), dev)?)?.broadcast_div(&Tensor::from_vec(std.clone(), (1, c), dev)?)
}

/// What DINOv3 made of the picture at one size: the global tokens and the patch map.
struct Seen {
    global: Tensor,
    /// [h·w, 1024].
    patches: Tensor,
    side: usize,
    image_size: usize,
    /// The picture at that size, [3, S, S] in [0, 1] (NAF's guide).
    pixels: Tensor,
}

fn see(dino: &dinov3::Dinov3, img: &image::RgbImage, size: usize, dev: &Device) -> Result<Seen> {
    let pixels = prepare::tensor(img, size as u32, dev)?;
    let tokens = dino.forward(&prepare::imagenet(&pixels)?)?;
    let prefix = 1 + dino.registers();
    let side = size / dino.patch;
    Ok(Seen { global: tokens.narrow(0, 0, prefix)?, patches: tokens.narrow(0, prefix, side * side)?.contiguous()?, side, image_size: size, pixels })
}

/// The voxels' conditioning: each one's image features where it projects (the patch
/// map's, and with NAF the upsampled map's at `naf_size`), for voxel indices on a
/// `grid`³ grid.
fn project(seen: &Seen, naf: Option<(&naf::Naf, usize)>, coords: &[[i32; 3]], grid: usize, cam: &proj::Camera, dev: &Device) -> Result<Tensor> {
    let res = seen.image_size as f64;
    let points: Vec<(f64, f64)> = coords.iter().map(|&c| cam.project(proj::grid_point(c, grid), res)).collect();
    let lr_taps: Vec<_> = points.iter().map(|&(x, y)| proj::taps(x, y, res, seen.side, seen.side)).collect();
    let lr = proj::sample(&seen.patches, &lr_taps, dev)?;
    let Some((naf, size)) = naf else { return Ok(lr) };
    // NAF only where the bilinear taps land.
    let hr_taps: Vec<_> = points.iter().map(|&(x, y)| proj::taps(x, y, res, size, size)).collect();
    let mut pixels: Vec<u32> = hr_taps.iter().flat_map(|t| t.iter().map(|(i, _)| *i as u32)).collect();
    pixels.sort_unstable();
    pixels.dedup();
    let features = seen.patches.t()?.contiguous()?.reshape((seen.patches.dim(1)?, seen.side, seen.side))?;
    let values = naf.at_pixels(&seen.pixels, &features, size, size, &pixels)?;
    let at: std::collections::HashMap<u32, usize> = pixels.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let remapped: Vec<[(usize, f32); 4]> = hr_taps.iter().map(|t| t.map(|(i, w)| (at[&(i as u32)], w))).collect();
    let hr = proj::sample(&values, &remapped, dev)?;
    Tensor::cat(&[&lr, &hr], 1)
}

pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let dev = device(r.device)?;
    let pipeline = Json::parse(&std::fs::read(r.model_dir.join("pipeline.json"))?).map_err(candle_core::Error::wrap)?;
    let model = |key: &str| -> Result<PathBuf> {
        let rel = pipeline.get("args").and_then(|a| a.get("models")).and_then(|m| m.get(key)).and_then(Json::as_str).ok_or_else(|| candle_core::Error::Msg(format!("pipeline.json: no {key}")))?;
        Ok(r.model_dir.join(rel))
    };
    let ss_sampling = Sampling::read(&pipeline, "sparse_structure_sampler", r.steps);
    let shape_sampling = Sampling::read(&pipeline, "shape_slat_sampler", r.steps);
    let tex_sampling = Sampling::read(&pipeline, "tex_slat_sampler", r.steps);
    let shape_norm = normalization(&pipeline, "shape_slat_normalization")?;
    let tex_norm = normalization(&pipeline, "tex_slat_normalization")?;

    // 1. The picture.
    report(event("preparing_image", 0, 1));
    let prepared = if r.prepared {
        let img = image::ImageReader::open(&r.image)?.with_guessed_format()?.decode().map_err(candle_core::Error::wrap)?;
        prepare::Prepared { image: img.to_rgb8(), cutout: img.to_rgba8(), matte: "prepared", upscaled: None }
    } else {
        prepare::prepare(&r.image, &prepare::Helpers { matte: r.matte.as_deref(), upscaler: r.upscaler.as_deref(), dev: &dev })?
    };
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let cutout = r.output.join(format!("model-{stamp}-input.png"));
    prepared.cutout.save(&cutout).map_err(candle_core::Error::wrap)?;
    let fov = r.fov_degrees.to_radians();
    let cam = proj::Camera::framing(fov, 1., 512., 0.);
    report(event("encoding_image", 0, 2));
    let dino = dinov3::Dinov3::load(&r.dino_dir, &dev)?;
    let seen512 = see(&dino, &prepared.image, 512, &dev)?;
    report(event("encoding_image", 1, 2));
    let seen1024 = see(&dino, &prepared.image, 1024, &dev)?;
    drop(dino);
    dump(r, "global512", &seen512.global)?;
    dump(r, "patches512", &seen512.patches)?;
    dump(r, "patches1024", &seen1024.patches)?;
    let naf = naf::Naf::load(&r.naf, &dev)?;

    // 2. The coarse structure.
    let ss_res = 16usize;
    let dense = Level::dense(ss_res, &dev)?;
    let proj_ss = project(&seen512, None, &dense.coords, ss_res, &cam, &dev)?;
    dump(r, "proj_ss", &proj_ss)?;
    let coords32 = {
        let flow = dit::Dit::load(&model("sparse_structure_flow_model")?, &dev)?;
        let c_in = flow.config().in_channels;
        let rope = dit::rope_tables(&dense.coords, 128, &dev)?;
        let pos = flow.context(Some(&seen512.global), Some(&proj_ss), 5, &dev)?;
        let neg = flow.context(None, None, seen512.global.dim(0)?, &dev)?;
        // The noise as the reference draws it, [C, 16³], as tokens [16³, C].
        let n = ss_res * ss_res * ss_res;
        let x = Tensor::from_vec(start_noise(r, "ss", 1, c_in * n)?, (c_in, n), &dev)?.t()?.contiguous()?;
        let z = sample(x, &ss_sampling, true, |x, t, c| flow.forward(x, t, &rope, if c { &pos } else { &neg }), |i, n| report(event("making_structure", i, n)))?;
        dump(r, "ss_latent", &z)?;
        drop(flow);
        let decoder = decoder::StructureDecoder::load(&model("sparse_structure_decoder")?, &dev)?;
        let (logits, res) = decoder.forward(&z)?;
        dump(r, "ss_logits", &logits)?;
        // Occupied at 64³, max-pooled to 32³ (any occupied child).
        let occupied: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        let f = res / 32;
        let mut cells = std::collections::BTreeSet::new();
        for (i, v) in occupied.iter().enumerate() {
            if *v > 0. {
                let (x, y, z) = (i / (res * res), (i / res) % res, i % res);
                cells.insert([(x / f) as i32, (y / f) as i32, (z / f) as i32]);
            }
        }
        cells.into_iter().collect::<Vec<_>>()
    };
    let coords32 = given_coords(r, "coords32")?.unwrap_or(coords32);
    if coords32.is_empty() {
        candle_core::bail!("3d: no structure came out of the picture (is the object in view, on a plain background?)");
    }
    dump_coords(r, "coords32", &coords32)?;

    // 3. The shape at 512, the finer voxels it proposes, and the shape there.
    let lr_level = Level::new(coords32.clone(), 32, &dev)?;
    let proj_lr = project(&seen512, Some((&naf, 512)), &coords32, 32, &cam, &dev)?;
    dump(r, "proj_shape512", &proj_lr)?;
    let lr_slat = {
        let flow = dit::Dit::load(&model("shape_slat_flow_model_512")?, &dev)?;
        let rope = dit::rope_tables(&coords32, 128, &dev)?;
        let pos = flow.context(Some(&seen512.global), Some(&proj_lr), 5, &dev)?;
        let neg = flow.context(None, None, 5, &dev)?;
        let c = flow.config().in_channels;
        let x = Tensor::from_vec(start_noise(r, "shape_lr", 2, coords32.len() * c)?, (coords32.len(), c), &dev)?;
        let z = sample(x, &shape_sampling, false, |x, t, cnd| flow.forward(x, t, &rope, if cnd { &pos } else { &neg }), |i, n| report(event("making_shape", i, n * 2)))?;
        denormalize(&z, &shape_norm)?
    };
    dump(r, "shape512", &lr_slat)?;
    drop(proj_lr);
    let shape_decoder = decoder::SparseDecoder::load(&model("shape_slat_decoder")?, &dev)?;
    let fine = shape_decoder.upsample(lr_level, &lr_slat, 4)?;
    // Quantized to the high-resolution latent grid, fewer tokens if there are too many.
    let mut hr_res = r.resolution;
    let hr_coords = loop {
        let grid = hr_res / 16;
        let mut q: Vec<[i32; 3]> = fine.coords.iter().map(|c| c.map(|v| ((((v as f32) + 0.5) / 512.) * (grid as f32 - 1.)).round_ties_even() as i32)).collect();
        q.sort_unstable();
        q.dedup();
        if q.len() < r.max_tokens || hr_res == 1024 {
            break q;
        }
        hr_res -= 128;
    };
    drop(fine);
    let hr_coords = given_coords(r, "coords_hr")?.unwrap_or(hr_coords);
    let grid = hr_res / 16;
    dump_coords(r, "coords_hr", &hr_coords)?;
    let proj_hr = project(&seen1024, Some((&naf, 512)), &hr_coords, grid, &cam, &dev)?;
    dump(r, "proj_shape_hr", &proj_hr)?;
    let rope_hr = dit::rope_tables(&hr_coords, 128, &dev)?;
    let shape = {
        let flow = dit::Dit::load(&model("shape_slat_flow_model_1024")?, &dev)?;
        let pos = flow.context(Some(&seen1024.global), Some(&proj_hr), 5, &dev)?;
        let neg = flow.context(None, None, 5, &dev)?;
        let c = flow.config().in_channels;
        let x = Tensor::from_vec(start_noise(r, "shape_hr", 3, hr_coords.len() * c)?, (hr_coords.len(), c), &dev)?;
        let z = sample(x, &shape_sampling, false, |x, t, cnd| flow.forward(x, t, &rope_hr, if cnd { &pos } else { &neg }), |i, n| report(event("making_shape", n + i, n * 2)))?;
        denormalize(&z, &shape_norm)?
    };
    dump(r, "shape_hr", &shape)?;
    drop(proj_hr);

    // 4. The texture on the same voxels.
    let proj_tex = project(&seen1024, Some((&naf, 1024)), &hr_coords, grid, &cam, &dev)?;
    dump(r, "proj_tex", &proj_tex)?;
    let tex = {
        let flow = dit::Dit::load(&model("tex_slat_flow_model_1024")?, &dev)?;
        let pos = flow.context(Some(&seen1024.global), Some(&proj_tex), 5, &dev)?;
        let neg = flow.context(None, None, 5, &dev)?;
        let shape_n = normalize(&shape, &shape_norm)?;
        let c = flow.config().in_channels - shape_n.dim(1)?;
        let x = Tensor::from_vec(start_noise(r, "tex", 4, hr_coords.len() * c)?, (hr_coords.len(), c), &dev)?;
        // The shape rides along as input channels after the noise.
        let z = sample(x, &tex_sampling, false, |x, t, cnd| flow.forward(&Tensor::cat(&[x, &shape_n], 1)?, t, &rope_hr, if cnd { &pos } else { &neg }), |i, n| report(event("making_texture", i, n)))?;
        denormalize(&z, &tex_norm)?
    };
    dump(r, "tex_hr", &tex)?;
    drop(proj_tex);
    drop(naf);
    drop(seen512);
    drop(seen1024);

    // 5. Decoding, the mesh and its colours.
    report(event("decoding", 0, 2));
    let hr_level = Level::new(hr_coords.clone(), grid, &dev)?;
    let decoded = shape_decoder.forward(hr_level.clone(), &shape, None, |_, _| {})?;
    drop(shape_decoder);
    let res = decoded.level.res;
    dump(r, "shape_voxels", &decoded.feats)?;
    dump_coords(r, "voxels", &decoded.level.coords)?;
    report(event("decoding", 1, 2));
    let tex_decoder = decoder::SparseDecoder::load(&model("tex_slat_decoder")?, &dev)?;
    let attrs = tex_decoder.forward(hr_level, &tex, Some(&decoded.subdivisions), |_, _| {})?;
    drop(tex_decoder);
    let attrs = ((attrs.feats * 0.5)? + 0.5)?;
    dump(r, "tex_voxels", &attrs)?;
    report(event("meshing", 0, 3));
    let values: Vec<f32> = decoded.feats.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let colors: Vec<f32> = attrs.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let voxels = decoded.level.coords.len();
    let raw = mesh::dual_grid(&decoded.level.coords, &values, res);
    let raw_faces = raw.triangles.len();
    // A closed, manifold surface, which simplifies cleanly (as TRELLIS.2's export remeshes).
    let surface = remesh::Surface::new(&raw, r.remesh.unwrap_or(res));
    drop(raw);
    let mut mesh = remesh::remesh(&surface);
    report(event("meshing", 1, 3));
    simplify(&mut mesh, r.faces);
    report(event("meshing", 2, 3));
    let sampler = mesh::Voxels::new(&decoded.level.coords, &colors, res);
    let (mesh, textures) = if r.texture_size == 0 {
        mesh.color_from_voxels(&sampler);
        (mesh, None)
    } else {
        let baked = bake::bake(&mesh, &surface, &sampler, r.texture_size)?;
        (baked.mesh, Some(baked.textures))
    };
    drop((surface, sampler));
    let glb = glb::write(&mesh, textures.as_ref());
    let path = r.output.join(format!("model-{stamp}-{}.glb", r.seed));
    std::fs::write(&path, &glb)?;
    report(event("meshing", 3, 3));
    Ok(Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("input", Json::str(cutout.to_string_lossy())),
        ("matte", Json::str(prepared.matte)),
        ("upscaled", match prepared.upscaled {
            Some((from, to)) => Json::Arr(vec![Json::Int(from as i64), Json::Int(to as i64)]),
            None => Json::Null,
        }),
        ("resolution", Json::Int(res as i64)),
        ("voxels", Json::Int(voxels as i64)),
        ("tokens", Json::Int(hr_coords.len() as i64)),
        ("raw_faces", Json::Int(raw_faces as i64)),
        ("faces", Json::Int(mesh.triangles.len() as i64)),
        ("vertices", Json::Int(mesh.positions.len() as i64)),
        ("bytes", Json::Int(glb.len() as i64)),
        ("fov_degrees", Json::Num(r.fov_degrees)),
        ("texture_size", Json::Int(if textures.is_some() { r.texture_size as i64 } else { 0 })),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
    ]))
}

/// Simplifies the mesh to about `faces` triangles (meshoptimizer), keeping its colours:
/// edge collapses within a growing error, then (where the surface's topology still
/// blocks them) clustering.
pub fn simplify(mesh: &mut mesh::Mesh, faces: usize) {
    if mesh.triangles.len() <= faces {
        return;
    }
    let bytes: Vec<u8> = mesh.positions.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    let Ok(adapter) = meshopt::VertexDataAdapter::new(&bytes, 12, 0) else { return };
    let target = faces * 3;
    let mut indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
    for (error, options) in [(0.002, meshopt::SimplifyOptions::None), (0.01, meshopt::SimplifyOptions::Prune), (0.04, meshopt::SimplifyOptions::Prune)] {
        indices = meshopt::simplify(&indices, &adapter, target, error, options, None);
        if indices.len() <= target * 5 / 4 {
            break;
        }
    }
    if indices.len() > target * 3 / 2 {
        indices = meshopt::simplify_sloppy(&indices, &adapter, target, 0.05, None);
    }
    mesh.triangles = indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    mesh.compact();
}
