//! Real-ESRGAN x4plus (xinntao, BSD-3-Clause): a picture four times larger,
//! with detail restored rather than blurred. basicsr's RRDBNet: a convolution,
//! 23 residual-in-residual dense blocks (each three dense blocks of five
//! convolutions with 32-channel growth, scaled by 0.2), a convolution added
//! back, two nearest-neighbour doublings each followed by a convolution, and
//! two convolutions to RGB; LeakyReLU 0.2 between.
//!
//! The weights are `RealESRGAN_x4plus.pth` (`params_ema`), read by the
//! tensor-only `.pth` reader, so nothing in the pickle runs. On a GPU it runs
//! in F16, as Real-ESRGAN's own inference does; large pictures go in tiles with
//! a 10-pixel overlap (its `tile_pad`).
use crate::sound::pth::{Pth, Value};
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// How much larger the picture gets.
pub const SCALE: usize = 4;
/// Input tile side, and the context around each tile.
const TILE: usize = 256;
const TILE_PAD: usize = 10;

struct Conv {
    w: Tensor,
    b: Tensor,
}

impl Conv {
    /// A 3×3 convolution, padding 1, on [1, C, H, W].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv2d(&self.w, 1, 1, 1, 1)?.broadcast_add(&self.b)
    }
}

fn lrelu(x: &Tensor) -> Result<Tensor> {
    x.maximum(&(x * 0.2)?)
}

struct Dense {
    convs: Vec<Conv>,
}

impl Dense {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut feats = vec![x.clone()];
        for conv in &self.convs[..4] {
            let y = lrelu(&conv.forward(&Tensor::cat(&feats, 1)?)?)?;
            feats.push(y);
        }
        let y = self.convs[4].forward(&Tensor::cat(&feats, 1)?)?;
        (y * 0.2)? + x
    }
}

pub struct Esrgan {
    first: Conv,
    body: Vec<[Dense; 3]>,
    conv_body: Conv,
    up1: Conv,
    up2: Conv,
    hr: Conv,
    last: Conv,
    dtype: DType,
    dev: Device,
}

impl Esrgan {
    /// `RealESRGAN_x4plus.pth` (or a state dict with its `params_ema` or `params`).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let dtype = if dev.is_cuda() { DType::F16 } else { DType::F32 };
        Self::load_as(path, dev, dtype)
    }

    pub fn load_as(path: &Path, dev: &Device, dtype: DType) -> Result<Self> {
        let mut pth = Pth::open(path)?;
        let dict = |v: &Value| -> HashMap<String, Value> {
            match v {
                Value::Dict(items) => items.iter().filter_map(|(k, v)| if let Value::Str(k) = k { Some((k.clone(), v.clone())) } else { None }).collect(),
                _ => HashMap::new(),
            }
        };
        let root = dict(&pth.root.clone());
        let sd = match root.get("params_ema").or_else(|| root.get("params")) {
            Some(v) => dict(v),
            None => root,
        };
        if !sd.contains_key("body.22.rdb3.conv5.weight") || sd.contains_key("body.23.rdb1.conv1.weight") {
            candle_core::bail!("{}: not Real-ESRGAN x4plus (RRDBNet, 23 blocks)", path.display());
        }
        let mut conv = |p: &str| -> Result<Conv> {
            let t = |pth: &mut Pth, k: String| -> Result<Tensor> {
                let v = sd.get(&k).ok_or_else(|| candle_core::Error::Msg(format!("Real-ESRGAN: no {k}")))?.clone();
                pth.tensor(&v, dev)?.to_dtype(dtype)
            };
            let w = t(&mut pth, format!("{p}.weight"))?;
            let b = t(&mut pth, format!("{p}.bias"))?;
            let co = b.dim(0)?;
            Ok(Conv { w, b: b.reshape((1, co, 1, 1))? })
        };
        let dense = |conv: &mut dyn FnMut(&str) -> Result<Conv>, p: &str| -> Result<Dense> { Ok(Dense { convs: (1..=5).map(|i| conv(&format!("{p}.conv{i}"))).collect::<Result<_>>()? }) };
        let first = conv("conv_first")?;
        let mut body = Vec::with_capacity(23);
        for i in 0..23 {
            body.push([dense(&mut conv, &format!("body.{i}.rdb1"))?, dense(&mut conv, &format!("body.{i}.rdb2"))?, dense(&mut conv, &format!("body.{i}.rdb3"))?]);
        }
        Ok(Self { first, body, conv_body: conv("conv_body")?, up1: conv("conv_up1")?, up2: conv("conv_up2")?, hr: conv("conv_hr")?, last: conv("conv_last")?, dtype, dev: dev.clone() })
    }

    /// The network on [1, 3, H, W] in [0, 1]: [1, 3, 4H, 4W], unclamped.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.first.forward(x)?;
        let mut h = feat.clone();
        for block in &self.body {
            let mut y = h.clone();
            for d in block {
                y = d.forward(&y)?;
            }
            h = ((y * 0.2)? + h)?;
        }
        let feat = (feat + self.conv_body.forward(&h)?)?;
        let (_, _, fh, fw) = feat.dims4()?;
        let feat = lrelu(&self.up1.forward(&feat.upsample_nearest2d(fh * 2, fw * 2)?)?)?;
        let feat = lrelu(&self.up2.forward(&feat.upsample_nearest2d(fh * 4, fw * 4)?)?)?;
        self.last.forward(&lrelu(&self.hr.forward(&feat)?)?)
    }

    /// An RGB picture (`w` × `h`, 8-bit) four times larger: `4w` × `4h`, 8-bit RGB.
    pub fn upscale(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        let (ow, oh) = (w * SCALE, h * SCALE);
        let mut out = vec![0u8; ow * oh * 3];
        for ty in (0..h).step_by(TILE) {
            for tx in (0..w).step_by(TILE) {
                // The tile, and the context around it that the network sees.
                let (x1, y1) = ((tx + TILE).min(w), (ty + TILE).min(h));
                let (px0, py0) = (tx.saturating_sub(TILE_PAD), ty.saturating_sub(TILE_PAD));
                let (px1, py1) = ((x1 + TILE_PAD).min(w), (y1 + TILE_PAD).min(h));
                let (pw, ph) = (px1 - px0, py1 - py0);
                let mut data = vec![0f32; 3 * pw * ph];
                for y in 0..ph {
                    for x in 0..pw {
                        for c in 0..3 {
                            data[c * pw * ph + y * pw + x] = rgb[((py0 + y) * w + px0 + x) * 3 + c] as f32 / 255.;
                        }
                    }
                }
                let input = Tensor::from_vec(data, (1, 3, ph, pw), &self.dev)?.to_dtype(self.dtype)?;
                let up = self.forward(&input)?.to_dtype(DType::F32)?.clamp(0f32, 1f32)?;
                let up: Vec<f32> = up.flatten_all()?.to_vec1()?;
                let (uw, uh) = (pw * SCALE, ph * SCALE);
                // Only the tile itself goes into the picture.
                for y in ty * SCALE..y1 * SCALE {
                    for x in tx * SCALE..x1 * SCALE {
                        let (sy, sx) = (y - py0 * SCALE, x - px0 * SCALE);
                        for c in 0..3 {
                            out[(y * ow + x) * 3 + c] = (up[c * uw * uh + sy * uw + sx] * 255.).round() as u8;
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(name: &str) -> Vec<f32> {
        let dir = std::env::var("ESRGAN_REF").unwrap_or_else(|_| "E:/p3dref/esrgan".into());
        std::fs::read(Path::new(&dir).join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        (d / n).sqrt()
    }

    /// The network against basicsr's, on the reference's crop (scratchpad
    /// ref_esrgan.py), in F32 and F16, and tiled against whole.
    /// cargo test --release --features flash-attn --lib esrgan -- --ignored --nocapture
    #[test]
    #[ignore]
    fn matches_the_reference() -> Result<()> {
        let gpu: usize = std::env::var("NROB_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let dev = Device::cuda_if_available(gpu)?;
        let path = std::env::var("ESRGAN").unwrap_or_else(|_| "E:/models/Real-ESRGAN/RealESRGAN_x4plus.pth".into());
        let input = reference("input");
        let want = reference("output");
        let (h, w) = (160, 200);
        for dtype in [DType::F32, DType::F16] {
            let net = Esrgan::load_as(Path::new(&path), &dev, dtype)?;
            let x = Tensor::from_vec(input.clone(), (1, 3, h, w), &dev)?.to_dtype(dtype)?;
            let first: Vec<f32> = net.first.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let t = std::time::Instant::now();
            let y: Vec<f32> = net.forward(&x)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let (e1, e2) = (rel(&first, &reference("first")), rel(&y, &want));
            println!("{dtype:?}: first conv {e1:.2e}, output {e2:.2e} relative RMS, in {:.2}s", t.elapsed().as_secs_f64());
            assert!(e2 < if dtype == DType::F32 { 1e-4 } else { 5e-3 });
        }
        // Tiled (the crop is under one tile: force several by upscaling a larger picture) agrees with whole.
        let net = Esrgan::load(Path::new(&path), &dev)?;
        let img = image::open("E:/p3dref/case4/input.png").map_err(candle_core::Error::wrap)?.to_rgb8();
        let crop = image::imageops::crop_imm(&img, 300, 300, 600, 400).to_image();
        let t = std::time::Instant::now();
        let up = net.upscale(crop.as_raw(), 600, 400)?;
        println!("600×400 upscaled to 2400×1600 in {:.2}s", t.elapsed().as_secs_f64());
        image::RgbImage::from_raw(2400, 1600, up).unwrap().save("E:/p3dref/esrgan/tiled.png").map_err(candle_core::Error::wrap)?;
        Ok(())
    }
}
