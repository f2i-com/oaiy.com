//! Picture tools (`kind: "picture"`): an object cut out of its background with
//! BiRefNet, or a picture made two or four times larger with Real-ESRGAN.
//!
//! - `remove_background`: the picture at its own size, its background
//!   transparent (the matte BiRefNet finds at 1024², resized to the picture);
//!   an RGBA PNG.
//! - `upscale`: four times larger with Real-ESRGAN (twice: four times, then
//!   halved, as Real-ESRGAN's own `outscale` does); a transparent picture keeps
//!   its alpha, resized bicubic. Up to 4 megapixels in (8192 pixels a side out).
use candle_core::{Device, Result};
use image::{imageops::FilterType, Rgba, RgbaImage};
use nrob::json::Json;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The largest picture `upscale` takes, in pixels.
pub const MAX_UPSCALE_PIXELS: u64 = 4 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    RemoveBackground,
    Upscale,
}

pub struct Request {
    pub op: Op,
    pub image: PathBuf,
    /// BiRefNet's folder, or Real-ESRGAN x4plus's weights.
    pub model: PathBuf,
    pub output: PathBuf,
    /// 2 or 4 (`upscale` only).
    pub scale: u32,
    pub device: usize,
}

impl Request {
    pub fn parse(j: &Json) -> std::result::Result<Self, String> {
        let s = |k: &str| j.get(k).and_then(Json::as_str).map(str::to_string);
        let op = match s("op").as_deref() {
            Some("remove_background") => Op::RemoveBackground,
            Some("upscale") => Op::Upscale,
            other => return Err(format!("picture: op must be remove_background or upscale, not {other:?}")),
        };
        let scale = j.get("scale").and_then(Json::as_i64).unwrap_or(4);
        if op == Op::Upscale && scale != 2 && scale != 4 {
            return Err("picture: scale must be 2 or 4".into());
        }
        Ok(Self {
            op,
            image: PathBuf::from(s("image").ok_or("picture: image is required")?),
            model: PathBuf::from(s("model").ok_or("picture: model is required")?),
            output: PathBuf::from(s("output_dir").ok_or("picture: output_dir is required")?),
            scale: scale as u32,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
        })
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
        Ok(Device::Cpu)
    }
}

pub fn run(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let dev = device(r.device)?;
    let img = image::ImageReader::open(&r.image)?.with_guessed_format()?.decode().map_err(candle_core::Error::wrap)?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let rgb: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let (out, name) = match r.op {
        Op::RemoveBackground => {
            report(event("removing_background", 0, 1));
            let net = crate::birefnet::BiRefNet::load(&r.model, &dev)?;
            let alpha = net.matte(&rgb, w as usize, h as usize)?;
            let mut out = rgba;
            for (p, a) in out.pixels_mut().zip(alpha) {
                p[3] = a;
            }
            report(event("removing_background", 1, 1));
            (out, format!("picture-{stamp}-cutout.png"))
        }
        Op::Upscale => {
            if u64::from(w) * u64::from(h) > MAX_UPSCALE_PIXELS {
                candle_core::bail!("picture: {w}×{h} is more than 4 megapixels to enlarge; make it smaller first");
            }
            let net = crate::esrgan::Esrgan::load(&r.model, &dev)?;
            let big = net.upscale_with(&rgb, w as usize, h as usize, |done, total| report(event("upscaling", done, total)))?;
            let (bw, bh) = (w as usize * crate::esrgan::SCALE, h as usize * crate::esrgan::SCALE);
            let transparent = rgba.pixels().any(|p| p[3] != 255);
            let alpha: Vec<u8> = if transparent {
                let a: Vec<u8> = rgba.pixels().map(|p| p[3]).collect();
                nrob_image::resize::resample(&a, 1, w as usize, h as usize, bw, bh, nrob_image::resize::Filter::Bicubic)
            } else {
                vec![255; bw * bh]
            };
            let mut out = RgbaImage::new(bw as u32, bh as u32);
            for (i, p) in out.pixels_mut().enumerate() {
                *p = Rgba([big[i * 3], big[i * 3 + 1], big[i * 3 + 2], alpha[i]]);
            }
            if r.scale == 2 {
                out = image::imageops::resize(&out, w * 2, h * 2, FilterType::Lanczos3);
            }
            (out, format!("picture-{stamp}-x{}.png", r.scale))
        }
    };
    let path = r.output.join(name);
    // An opaque result is written as RGB (a smaller file).
    if out.pixels().all(|p| p[3] == 255) {
        image::DynamicImage::ImageRgba8(out.clone()).to_rgb8().save(&path).map_err(candle_core::Error::wrap)?;
    } else {
        out.save(&path).map_err(candle_core::Error::wrap)?;
    }
    Ok(Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("op", Json::str(if r.op == Op::Upscale { "upscale" } else { "remove_background" })),
        ("width", Json::Int(out.width() as i64)),
        ("height", Json::Int(out.height() as i64)),
        ("source_width", Json::Int(w as i64)),
        ("source_height", Json::Int(h as i64)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
    ]))
}
