//! Picture tools (`kind: "picture"`): an object cut out of its background with
//! BiRefNet, or a picture made two or four times larger with Real-ESRGAN.
//!
//! - `remove_background`: the picture at its own size, its background
//!   transparent (the matte BiRefNet finds at 1024², resized to the picture);
//!   an RGBA PNG.
//! - `upscale`: four times larger with Real-ESRGAN (twice: four times, then
//!   halved, as Real-ESRGAN's own `outscale` does); a transparent picture keeps
//!   its alpha, resized bicubic. Up to 4 megapixels in (8192 pixels a side out).
//!
//! `backend` "webgpu" runs both on WebGPU (any GPU: [`crate::birefnet_wgpu`], [`crate::esrgan_wgpu`]), as a worker built
//! with WebGPU and without CUDA does unless told "cpu" (falling back to the CPU where no WebGPU device opens, unless
//! asked for WebGPU by name).
use candle_core::{Device, Result};
use image::{imageops::FilterType, Rgba, RgbaImage};
use oaiy_engine::json::Json;
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
    /// On WebGPU.
    pub webgpu: bool,
    /// WebGPU asked for by name (else taken by default: the CPU where it does not open).
    pub webgpu_asked: bool,
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
        // (a worker without CUDA but with WebGPU takes the GPU by default: its CPU is the other way)
        let webgpu_asked = s("backend").as_deref() == Some("webgpu");
        let webgpu = match s("backend").as_deref() {
            Some("webgpu") => true,
            Some("cuda" | "cpu") => false,
            Some(other) => return Err(format!("picture: backend must be webgpu, cuda or cpu, not {other}")),
            None => cfg!(feature = "webgpu"),
        };
        if webgpu && !cfg!(feature = "webgpu") {
            return Err("picture: this build has no WebGPU (the webgpu feature)".into());
        }
        Ok(Self {
            op,
            image: PathBuf::from(s("image").ok_or("picture: image is required")?),
            model: PathBuf::from(s("model").ok_or("picture: model is required")?),
            output: PathBuf::from(s("output_dir").ok_or("picture: output_dir is required")?),
            scale: scale as u32,
            device: j.get("device").and_then(Json::as_i64).unwrap_or(0).max(0) as usize,
            webgpu,
            webgpu_asked,
        })
    }
}

fn event(stage: &str, current: usize, total: usize) -> Json {
    Json::obj([("stage", Json::str(stage)), ("current", Json::Int(current as i64)), ("total", Json::Int(total as i64))])
}

/// `load`'s model where `r` is on WebGPU: None (the CPU's to do) where it is not, or where it fails to load and WebGPU
/// was only taken by default (no WebGPU device on this computer: said, as an event), its error where asked for.
#[cfg_attr(not(feature = "webgpu"), allow(dead_code))]
fn on_webgpu<T>(r: &Request, report: &mut impl FnMut(Json), load: impl FnOnce() -> Result<T>) -> Result<Option<T>> {
    if !r.webgpu {
        return Ok(None);
    }
    match load() {
        Ok(net) => Ok(Some(net)),
        Err(e) if r.webgpu_asked => Err(e),
        Err(e) => {
            report(Json::obj([("stage", Json::str("webgpu_unavailable")), ("error", Json::str(e.to_string()))]));
            Ok(None)
        }
    }
}

/// The object's matte in `rgb` with BiRefNet: on WebGPU where `r` is, else on `dev`.
fn matte(r: &Request, dev: &Device, rgb: &[u8], w: usize, h: usize, report: &mut impl FnMut(Json)) -> Result<Vec<u8>> {
    #[cfg(feature = "webgpu")]
    if let Some(net) = on_webgpu(r, report, || crate::birefnet_wgpu::WgpuBiRefNet::load(&r.model, r.device))? {
        return net.matte(rgb, w, h);
    }
    #[cfg(not(feature = "webgpu"))]
    let _ = report;
    crate::birefnet::BiRefNet::load(&r.model, dev)?.matte(rgb, w, h)
}

/// `rgb` four times larger with Real-ESRGAN: on WebGPU where `r` is, else on `dev`.
fn upscale(r: &Request, dev: &Device, rgb: &[u8], w: usize, h: usize, report: &mut impl FnMut(Json)) -> Result<Vec<u8>> {
    #[cfg(feature = "webgpu")]
    if let Some(net) = on_webgpu(r, report, || crate::esrgan_wgpu::WgpuEsrgan::load(&r.model, r.device))? {
        return net.upscale_with(rgb, w, h, |done, total| report(event("upscaling", done, total)));
    }
    let net = crate::esrgan::Esrgan::load(&r.model, dev)?;
    net.upscale_with(rgb, w, h, |done, total| report(event("upscaling", done, total)))
}

fn device(index: usize) -> Result<Device> {
    {
        let _ = index;
        Ok(Device::Cpu)
    }
}

pub fn run(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    // (no Candle device for a WebGPU job: a CUDA context on the card would keep its memory)
    let dev = if r.webgpu { Device::Cpu } else { device(r.device)? };
    let img = image::ImageReader::open(&r.image)?.with_guessed_format()?.decode().map_err(candle_core::Error::wrap)?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let rgb: Vec<u8> = rgba.pixels().flat_map(|p| [p[0], p[1], p[2]]).collect();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let (out, name) = match r.op {
        Op::RemoveBackground => {
            report(event("removing_background", 0, 1));
            let alpha = matte(r, &dev, &rgb, w as usize, h as usize, &mut report)?;
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
            let big = upscale(r, &dev, &rgb, w as usize, h as usize, &mut report)?;
            let (bw, bh) = (w as usize * crate::esrgan::SCALE, h as usize * crate::esrgan::SCALE);
            let transparent = rgba.pixels().any(|p| p[3] != 255);
            let alpha: Vec<u8> = if transparent {
                let a: Vec<u8> = rgba.pixels().map(|p| p[3]).collect();
                oaiy_image::resize::resample(&a, 1, w as usize, h as usize, bw, bh, oaiy_image::resize::Filter::Bicubic)
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

#[cfg(test)]
mod fallback_tests {
    use super::*;

    fn request(backend: Option<&str>) -> Request {
        let j = Json::parse(format!(r#"{{"op":"upscale","image":"a.png","model":"m.pth","output_dir":"out"{}}}"#, backend.map_or(String::new(), |b| format!(r#","backend":"{b}""#))).as_bytes()).unwrap();
        Request::parse(&j).unwrap()
    }

    /// WebGPU taken by default falls back to the CPU where it does not open (said as an event); asked for by name, its
    /// error is the job's; not taken, nothing is tried.
    #[test]
    fn webgpu_by_default_falls_back_to_the_cpu() {
        let fail = || -> Result<()> { candle_core::bail!("GPU 99: the computer has 2") };
        let mut events = Vec::new();
        let mut by_default = request(None);
        by_default.webgpu = true;
        assert!(on_webgpu(&by_default, &mut |e| events.push(e), fail).unwrap().is_none(), "the CPU's to do");
        assert!(events.iter().any(|e| e.get("stage").and_then(Json::as_str) == Some("webgpu_unavailable")), "said");
        if cfg!(feature = "webgpu") {
            let asked = request(Some("webgpu"));
            assert!(asked.webgpu && asked.webgpu_asked);
            assert!(on_webgpu(&asked, &mut |_| {}, fail).is_err(), "asked for by name: its error");
        }
        let cpu = request(Some("cpu"));
        assert!(!cpu.webgpu);
        assert!(on_webgpu(&cpu, &mut |_| {}, || -> Result<()> { panic!("not tried") }).unwrap().is_none());
    }
}
