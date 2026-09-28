//! NAF (valeoai, Apache-2.0): upsamples a vision model's patch features to image
//! resolution, guided by the image. A small conv encoder makes 256-channel
//! queries at the target size (2-D RoPE on them); the keys are those queries
//! average-pooled to the patch grid; each target pixel attends, in 4 heads, to the
//! 9×9 patches around its own (NATTEN's dilated neighbourhood, windows clamped at
//! the borders), and takes their features.
//!
//! Only the pixels asked for are computed (Pixal3D samples the upsampled map at
//! the voxels' projections), so the full-resolution map is never made. F32.
use crate::sound::pth::{Pth, Value};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

struct Conv {
    w: Tensor,
    b: Tensor,
    k: usize,
}

struct Block {
    gn1: (Tensor, Tensor),
    conv1: Conv,
    gn2: (Tensor, Tensor),
    conv2: Conv,
}

struct Encoder {
    first: Conv,
    blocks: Vec<Block>,
}

pub struct Naf {
    enc: Encoder,
    sem: Encoder,
    periods: Tensor,
    heads: usize,
    kernel: usize,
}

/// Reflect-pads [C, H, W] by `p` on each side.
fn pad_reflect(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(x.clone());
    }
    let (_, h, w) = x.dims3()?;
    let mut cols = Vec::new();
    for i in (1..=p).rev() {
        cols.push(x.narrow(2, i, 1)?);
    }
    cols.push(x.clone());
    for i in 0..p {
        cols.push(x.narrow(2, w - 2 - i, 1)?);
    }
    let x = Tensor::cat(&cols, 2)?;
    let mut rows = Vec::new();
    for i in (1..=p).rev() {
        rows.push(x.narrow(1, i, 1)?);
    }
    rows.push(x.clone());
    for i in 0..p {
        rows.push(x.narrow(1, h - 2 - i, 1)?);
    }
    Tensor::cat(&rows, 1)
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = pad_reflect(x, self.k / 2)?;
        let conv = candle_nn::Conv2d::new(self.w.clone(), Some(self.b.clone()), Default::default());
        conv.forward(&x.unsqueeze(0)?)?.squeeze(0)
    }
}

fn group_norm(x: &Tensor, gn: &(Tensor, Tensor), groups: usize) -> Result<Tensor> {
    let (c, h, w) = x.dims3()?;
    let g = x.reshape((groups, c / groups * h * w))?;
    let g = g.broadcast_sub(&g.mean_keepdim(1)?)?;
    let var = g.sqr()?.mean_keepdim(1)?;
    let g = g.broadcast_div(&(var + 1e-5)?.sqrt()?)?.reshape((c, h, w))?;
    g.broadcast_mul(&gn.0.reshape((c, 1, 1))?)?.broadcast_add(&gn.1.reshape((c, 1, 1))?)
}

impl Encoder {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.first.forward(x)?;
        for b in &self.blocks {
            // EncBlock without its residual (NAF's encoder builds them with residual=False).
            let y = group_norm(&h, &b.gn1, 8)?.silu()?;
            let y = b.conv1.forward(&y)?;
            let y = group_norm(&y, &b.gn2, 8)?.silu()?;
            h = b.conv2.forward(&y)?;
        }
        Ok(h)
    }
}

/// Average-pools [C, H, W] to [C, oh, ow] (whole factors, as adaptive pooling then is).
fn avg_pool(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (c, h, w) = x.dims3()?;
    if (h, w) == (oh, ow) {
        return Ok(x.clone());
    }
    if h % oh != 0 || w % ow != 0 {
        candle_core::bail!("NAF: {h}×{w} does not pool evenly to {oh}×{ow}");
    }
    x.reshape((c, oh, h / oh, ow, w / ow))?.mean(4)?.mean(2)
}

impl Naf {
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let mut pth = Pth::open(path)?;
        let Value::Dict(items) = pth.root.clone() else { candle_core::bail!("{}: not a state dict", path.display()) };
        let sd: HashMap<String, Value> = items.into_iter().filter_map(|(k, v)| if let Value::Str(k) = k { Some((k, v)) } else { None }).collect();
        let mut t = |k: &str| -> Result<Tensor> {
            let v = sd.get(k).ok_or_else(|| candle_core::Error::Msg(format!("NAF: no {k}")))?.clone();
            pth.tensor(&v, dev)?.to_dtype(DType::F32)
        };
        let mut encoder = |name: &str, k: usize| -> Result<Encoder> {
            let p = format!("image_encoder.{name}");
            let first = Conv { w: t(&format!("{p}.0.weight"))?, b: t(&format!("{p}.0.bias"))?, k };
            let blocks = (1..=2)
                .map(|i| {
                    Ok(Block {
                        gn1: (t(&format!("{p}.{i}.norm1.weight"))?, t(&format!("{p}.{i}.norm1.bias"))?),
                        conv1: Conv { w: t(&format!("{p}.{i}.conv1.weight"))?, b: t(&format!("{p}.{i}.conv1.bias"))?, k },
                        gn2: (t(&format!("{p}.{i}.norm2.weight"))?, t(&format!("{p}.{i}.norm2.bias"))?),
                        conv2: Conv { w: t(&format!("{p}.{i}.conv2.weight"))?, b: t(&format!("{p}.{i}.conv2.bias"))?, k },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Encoder { first, blocks })
        };
        let enc = encoder("encoder", 1)?;
        let sem = encoder("sem_encoder", 3)?;
        let periods = t("image_encoder.rope.periods")?;
        Ok(Self { enc, sem, periods, heads: 4, kernel: 9 })
    }

    /// The upsampled features of `features` [C, h, w] at target pixels `pixels` (row·ow + col of an
    /// `oh`×`ow` map), guided by `image` [3, H, W] in [0, 1]: [pixels, C].
    pub fn at_pixels(&self, image: &Tensor, features: &Tensor, oh: usize, ow: usize, pixels: &[u32]) -> Result<Tensor> {
        let dev = image.device();
        let (_, ih, iw) = image.dims3()?;
        if ih > 4 * oh || iw > 4 * ow {
            candle_core::bail!("NAF: a {ih}×{iw} image for a {oh}×{ow} map needs resizing first");
        }
        let (c, h, w) = features.dims3()?;
        // Queries: both encoders, pooled to the target size, RoPE per head.
        let x = Tensor::cat(&[self.enc.forward(image)?, self.sem.forward(image)?], 0)?;
        let x = avg_pool(&x, oh, ow)?;
        let dim = x.dim(0)?;
        let hd = dim / self.heads;
        let periods: Vec<f32> = self.periods.to_vec1()?;
        let quarter = periods.len();
        let mut angles: Vec<f32> = Vec::with_capacity(oh * ow * hd);
        for i in 0..oh {
            for j in 0..ow {
                let cy = 2. * ((i as f64 + 0.5) / oh as f64) - 1.;
                let cx = 2. * ((j as f64 + 0.5) / ow as f64) - 1.;
                let row: Vec<f32> = periods.iter().map(|p| (2. * std::f64::consts::PI * cy / *p as f64) as f32).chain(periods.iter().map(|p| (2. * std::f64::consts::PI * cx / *p as f64) as f32)).collect();
                debug_assert_eq!(row.len(), 2 * quarter);
                angles.extend(row.iter().chain(row.iter()));
            }
        }
        let angles = Tensor::from_vec(angles, (oh * ow, 1, hd), dev)?;
        // [dim, oh, ow] -> [pixels, heads, hd]
        let q = x.reshape((self.heads, hd, oh * ow))?.permute((2, 0, 1))?.contiguous()?;
        let half = hd / 2;
        let rot = Tensor::cat(&[&q.narrow(2, half, half)?.neg()?, &q.narrow(2, 0, half)?], 2)?;
        let q = (q.broadcast_mul(&angles.cos()?)? + rot.broadcast_mul(&angles.sin()?)?)?;
        // Keys: the (rotated) queries pooled to the patch grid; values: the features.
        let q_map = q.permute((1, 2, 0))?.reshape((dim, oh, ow))?;
        let keys = avg_pool(&q_map, h, w)?.reshape((dim, h * w))?.t()?.contiguous()?; // [h·w, dim]
        let values = features.reshape((c, h * w))?.t()?.contiguous()?; // [h·w, C]
        let q = q.reshape((oh * ow, dim))?;
        let (dy, dx) = (oh / h, ow / w);
        let k = self.kernel;
        let start = |p: usize, len: usize| p.saturating_sub(k / 2).min(len - k);
        let scale = 1. / (hd as f64).sqrt();
        let step = 2048;
        let mut outs = Vec::new();
        let mut at = 0;
        while at < pixels.len() {
            let m = step.min(pixels.len() - at);
            let chunk = &pixels[at..at + m];
            let mut window = Vec::with_capacity(m * k * k);
            for &p in chunk {
                let (i, j) = (p as usize / ow, p as usize % ow);
                let (a0, b0) = (start(i / dy, h), start(j / dx, w));
                for a in a0..a0 + k {
                    for b in b0..b0 + k {
                        window.push((a * w + b) as u32);
                    }
                }
            }
            let window = Tensor::from_vec(window, m * k * k, dev)?;
            let qs = q.index_select(&Tensor::from_vec(chunk.to_vec(), m, dev)?, 0)?.reshape((m, self.heads, 1, hd))?;
            let ks = keys.index_select(&window, 0)?.reshape((m, k * k, self.heads, hd))?.permute((0, 2, 3, 1))?.contiguous()?; // [m, heads, hd, 81]
            let vs = values.index_select(&window, 0)?.reshape((m, k * k, self.heads, c / self.heads))?.transpose(1, 2)?.contiguous()?; // [m, heads, 81, c/heads]
            let logits = (qs.matmul(&ks)? * scale)?;
            let p = candle_nn::ops::softmax(&logits, D::Minus1)?;
            outs.push(p.matmul(&vs)?.reshape((m, c))?);
            at += m;
        }
        Tensor::cat(&outs, 0)
    }
}
