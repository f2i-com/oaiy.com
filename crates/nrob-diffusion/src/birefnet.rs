//! BiRefNet (ZhengPeng7/BiRefNet, MIT): the object in a picture, as a matte.
//! Its general-use weights, with the architecture its `birefnet.py` configures:
//!
//! - a Swin-L backbone (window 12), run on the 1024² picture and on it at 512²,
//!   whose four levels are joined channel-wise (the half-size ones upsampled);
//! - the three finer levels pooled into the coarsest (context), then a squeeze
//!   block down to 3072 channels;
//! - a decoder of four blocks from 32² up to 256², each a convolution, an ASPP
//!   of modulated deformable convolutions (kernels 1, 1, 3 and 7, and the
//!   global mean) and a convolution, gated by a learned attention map, with
//!   lateral links from the backbone and the picture itself laid out as
//!   patches at each block's size; then one channel of logits at 1024².
//!
//! BatchNorms (running statistics) are folded into the convolutions before
//! them. F32 throughout, as the reference runs it. The deformable convolution
//! is torchvision's `deform_conv2d` (bilinear taps, zero outside, times the
//! modulator), as a gather of the four corners and one matmul per chunk.
use crate::ltx::store::Store;
use crate::model3d::dinov3::layer_norm;
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

/// The side BiRefNet sees the picture at.
pub const SIZE: usize = 1024;
const WINDOW: usize = 12;
const BN_EPS: f64 = 1e-5;

struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.w.t()?)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// A convolution on [C, H, W], stride 1, padding k / 2, its BatchNorm folded in.
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    k: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let co = self.w.dim(0)?;
        let y = if self.k == 1 {
            self.w.reshape((co, c))?.matmul(&x.reshape((c, h * w))?)?.reshape((co, h, w))?
        } else {
            x.unsqueeze(0)?.conv2d(&self.w, self.k / 2, 1, 1, 1)?.squeeze(0)?
        };
        match &self.b {
            Some(b) => y.broadcast_add(&b.reshape((co, 1, 1))?),
            None => Ok(y),
        }
    }
}

/// Modulated deformable convolution (torchvision's), with a BatchNorm folded in.
struct Deform {
    offset: Conv,
    modulator: Conv,
    /// [Cout, K·Cin]: taps outer, input channels inner.
    w: Tensor,
    b: Tensor,
    k: usize,
}

struct AsppDeformable {
    branches: Vec<Deform>,
    pool: Conv,
    out: Conv,
}

struct DecBlk {
    conv_in: Conv,
    att: AsppDeformable,
    conv_out: Conv,
}

struct SwinBlock {
    norm1: (Tensor, Tensor),
    qkv: Linear,
    proj: Linear,
    /// [heads, 144, 144]: the relative position bias.
    bias: Tensor,
    norm2: (Tensor, Tensor),
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    shift: usize,
}

struct Stage {
    blocks: Vec<SwinBlock>,
    norm: (Tensor, Tensor),
    /// PatchMerging: its norm and reduction.
    merge: Option<((Tensor, Tensor), Linear)>,
}

pub struct BiRefNet {
    patch: Conv,
    patch_norm: (Tensor, Tensor),
    stages: Vec<Stage>,
    squeeze: DecBlk,
    blocks: [DecBlk; 4],
    lateral: [Conv; 3],
    /// Each block's gate: a 3×3 convolution (with BatchNorm and ReLU), then 1×1 to one channel.
    gates: [(Conv, Conv); 3],
    /// The picture as patches at each block's size, 32² to 1024²: two convolutions each.
    inputs: [(Conv, Conv); 5],
    out: Conv,
    dev: Device,
}

/// Bilinear resampling weights along one axis, PyTorch's `align_corners=True`: [out, in].
fn resize_matrix(n_in: usize, n_out: usize, dev: &Device) -> Result<Tensor> {
    let mut m = vec![0f32; n_out * n_in];
    let scale = if n_out > 1 { (n_in - 1) as f32 / (n_out - 1) as f32 } else { 0. };
    for o in 0..n_out {
        let src = scale * o as f32;
        let i0 = (src as usize).min(n_in - 1);
        let i1 = if i0 < n_in - 1 { i0 + 1 } else { i0 };
        let f = src - i0 as f32;
        m[o * n_in + i0] += 1. - f;
        m[o * n_in + i1] += f;
    }
    Tensor::from_vec(m, (n_out, n_in), dev)
}

/// `F.interpolate(x, (oh, ow), mode="bilinear", align_corners=True)` on [C, H, W].
fn interpolate(x: &Tensor, oh: usize, ow: usize) -> Result<Tensor> {
    let (_, h, w) = x.dims3()?;
    if (h, w) == (oh, ow) {
        return Ok(x.clone());
    }
    let dev = x.device();
    let rows = resize_matrix(h, oh, dev)?.broadcast_matmul(x)?;
    rows.broadcast_matmul(&resize_matrix(w, ow, dev)?.t()?)
}

/// The picture [3, S, S] laid out as patches the size of `size`²: [3·g², size, size]
/// with g = S / size, each channel one whole block (einops' `b c (hg h) (wg w) -> b (c hg wg) h w`).
fn patches(x: &Tensor, size: usize) -> Result<Tensor> {
    let (c, h, _) = x.dims3()?;
    let g = h / size;
    x.reshape((c, g, size, g, size))?.permute((0, 1, 3, 2, 4))?.reshape((c * g * g, size, size))
}

fn roll2(x: &Tensor, s: usize) -> Result<Tensor> {
    // [H, W, C] rolled by -s on both axes (torch.roll(x, (-s, -s), (0, 1))).
    let (h, w, _) = x.dims3()?;
    let x = Tensor::cat(&[&x.narrow(0, s, h - s)?, &x.narrow(0, 0, s)?], 0)?;
    Tensor::cat(&[&x.narrow(1, s, w - s)?, &x.narrow(1, 0, s)?], 1)
}

fn unroll2(x: &Tensor, s: usize) -> Result<Tensor> {
    let (h, w, _) = x.dims3()?;
    let x = Tensor::cat(&[&x.narrow(0, h - s, s)?, &x.narrow(0, 0, h - s)?], 0)?;
    Tensor::cat(&[&x.narrow(1, w - s, s)?, &x.narrow(1, 0, w - s)?], 1)
}

/// The shifted windows' mask on a padded `hp` × `wp` grid: [nW, 144, 144], -100 across regions.
fn shift_mask(hp: usize, wp: usize, shift: usize, dev: &Device) -> Result<Tensor> {
    let region = |i: usize, n: usize| if i < n - WINDOW { 0 } else if i < n - shift { 1 } else { 2 };
    let (gh, gw) = (hp / WINDOW, wp / WINDOW);
    let n = WINDOW * WINDOW;
    let mut mask = vec![0f32; gh * gw * n * n];
    for wy in 0..gh {
        for wx in 0..gw {
            let ids: Vec<usize> = (0..n).map(|t| region(wy * WINDOW + t / WINDOW, hp) * 3 + region(wx * WINDOW + t % WINDOW, wp)).collect();
            let base = (wy * gw + wx) * n * n;
            for i in 0..n {
                for j in 0..n {
                    if ids[i] != ids[j] {
                        mask[base + i * n + j] = -100.;
                    }
                }
            }
        }
    }
    Tensor::from_vec(mask, (gh * gw, n, n), dev)
}

impl SwinBlock {
    /// `x`: [H·W, C] tokens of an `h` × `w` grid.
    fn forward(&self, x: &Tensor, h: usize, w: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let c = x.dim(1)?;
        let (hp, wp) = (h.div_ceil(WINDOW) * WINDOW, w.div_ceil(WINDOW) * WINDOW);
        let mut t = layer_norm(x, Some(&self.norm1.0), Some(&self.norm1.1), 1e-5)?.reshape((h, w, c))?;
        // Padding comes after the norm: the padded tokens are zeros, and are attended to.
        if hp > h {
            t = t.pad_with_zeros(0, 0, hp - h)?;
        }
        if wp > w {
            t = t.pad_with_zeros(1, 0, wp - w)?;
        }
        if self.shift > 0 {
            t = roll2(&t, self.shift)?;
        }
        let (gh, gw) = (hp / WINDOW, wp / WINDOW);
        let n = WINDOW * WINDOW;
        let windows = t.reshape((gh, WINDOW, gw, WINDOW, c))?.permute((0, 2, 1, 3, 4))?.reshape((gh * gw, n, c))?;
        let hd = c / self.heads;
        let qkv = self.qkv.forward(&windows)?.reshape((gh * gw, n, 3, self.heads, hd))?.permute((2, 0, 3, 1, 4))?;
        let q = (qkv.get(0)?.contiguous()? * (hd as f64).powf(-0.5))?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let mut attn = q.matmul(&k.t()?)?.broadcast_add(&self.bias.unsqueeze(0)?)?;
        if let (Some(m), true) = (mask, self.shift > 0) {
            attn = attn.broadcast_add(&m.unsqueeze(1)?)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?.permute((0, 2, 1, 3))?.reshape((gh * gw, n, c))?;
        let out = self.proj.forward(&out)?;
        let mut t = out.reshape((gh, gw, WINDOW, WINDOW, c))?.permute((0, 2, 1, 3, 4))?.reshape((hp, wp, c))?;
        if self.shift > 0 {
            t = unroll2(&t, self.shift)?;
        }
        let t = t.narrow(0, 0, h)?.narrow(1, 0, w)?.reshape((h * w, c))?;
        let x = (x + t)?;
        let m = self.fc2.forward(&self.fc1.forward(&layer_norm(&x, Some(&self.norm2.0), Some(&self.norm2.1), 1e-5)?)?.gelu_erf()?)?;
        x + m
    }
}

impl Deform {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let kk = self.k * self.k;
        let pad = (self.k / 2) as f32;
        let offset: Vec<f32> = self.offset.forward(x)?.flatten_all()?.to_vec1()?;
        let modulator: Vec<f32> = (candle_nn::ops::sigmoid(&self.modulator.forward(x)?)? * 2.)?.flatten_all()?.to_vec1()?;
        // The input as rows, with a zero row for taps outside it.
        let rows = Tensor::cat(&[&x.reshape((c, h * w))?.t()?, &Tensor::zeros((1, c), DType::F32, x.device())?], 0)?.contiguous()?;
        let zero = (h * w) as u32;
        let n = h * w;
        let chunk = (1 << 21) / (kk * 4).max(1);
        let mut outs = Vec::new();
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            let mut index = Vec::with_capacity(len * kk * 4);
            let mut weight = Vec::with_capacity(len * kk * 4);
            for p in start..start + len {
                let (py, px) = ((p / w) as f32, (p % w) as f32);
                for t in 0..kk {
                    let (i, j) = ((t / self.k) as f32, (t % self.k) as f32);
                    let y = py - pad + i + offset[(2 * t) * n + p];
                    let xx = px - pad + j + offset[(2 * t + 1) * n + p];
                    let m = modulator[t * n + p];
                    if y <= -1. || y >= h as f32 || xx <= -1. || xx >= w as f32 {
                        index.extend([zero; 4]);
                        weight.extend([0f32; 4]);
                        continue;
                    }
                    let (y0, x0) = (y.floor(), xx.floor());
                    let (ly, lx) = (y - y0, xx - x0);
                    let (y0, x0) = (y0 as i64, x0 as i64);
                    for (dy, dx, wgt) in [(0, 0, (1. - ly) * (1. - lx)), (0, 1, (1. - ly) * lx), (1, 0, ly * (1. - lx)), (1, 1, ly * lx)] {
                        let (yy, xc) = (y0 + dy, x0 + dx);
                        if yy >= 0 && xc >= 0 && (yy as usize) < h && (xc as usize) < w {
                            index.push((yy as usize * w + xc as usize) as u32);
                            weight.push(wgt * m);
                        } else {
                            index.push(zero);
                            weight.push(0.);
                        }
                    }
                }
            }
            let dev = x.device();
            let index = Tensor::from_vec(index, len * kk * 4, dev)?;
            let weight = Tensor::from_vec(weight, (len * kk * 4, 1), dev)?;
            let taps = rows.index_select(&index, 0)?.broadcast_mul(&weight)?.reshape((len * kk, 4, c))?.sum(1)?.reshape((len, kk * c))?;
            outs.push(taps.matmul(&self.w.t()?)?);
            start += len;
        }
        let co = self.w.dim(0)?;
        Tensor::cat(&outs, 0)?.broadcast_add(&self.b)?.t()?.reshape((co, h, w))
    }
}

impl AsppDeformable {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, h, w) = x.dims3()?;
        let mut parts = Vec::with_capacity(self.branches.len() + 1);
        for b in &self.branches {
            parts.push(b.forward(x)?.relu()?);
        }
        let pooled = self.pool.forward(&x.mean_keepdim(2)?.mean_keepdim(1)?)?.relu()?;
        parts.push(pooled.broadcast_as((pooled.dim(0)?, h, w))?.contiguous()?);
        self.out.forward(&Tensor::cat(&parts, 0)?)?.relu()
    }
}

impl DecBlk {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.att.forward(&self.conv_in.forward(x)?.relu()?)?)
    }
}

/// Loads a model's tensors, folding BatchNorms into the convolutions before them.
struct Loader {
    store: Store,
    dev: Device,
}

impl Loader {
    fn t(&mut self, key: &str) -> Result<Tensor> {
        self.store.tensor_f32(key, &self.dev)
    }

    fn has(&self, key: &str) -> bool {
        self.store.index.get(key).is_some()
    }

    fn linear(&mut self, p: &str) -> Result<Linear> {
        let b = if self.has(&format!("{p}.bias")) { Some(self.t(&format!("{p}.bias"))?) } else { None };
        Ok(Linear { w: self.t(&format!("{p}.weight"))?, b })
    }

    fn norm(&mut self, p: &str) -> Result<(Tensor, Tensor)> {
        Ok((self.t(&format!("{p}.weight"))?, self.t(&format!("{p}.bias"))?))
    }

    /// The BatchNorm at `bn` as a scale and shift per channel.
    fn bn(&mut self, bn: &str) -> Result<(Tensor, Tensor)> {
        let g = self.t(&format!("{bn}.weight"))?;
        let beta = self.t(&format!("{bn}.bias"))?;
        let mean = self.t(&format!("{bn}.running_mean"))?;
        let var = self.t(&format!("{bn}.running_var"))?;
        let scale = g.broadcast_div(&(var + BN_EPS)?.sqrt()?)?;
        let shift = (beta - mean.broadcast_mul(&scale)?)?;
        Ok((scale, shift))
    }

    /// The convolution at `p`, and the BatchNorm at `bn` after it.
    fn conv(&mut self, p: &str, bn: Option<&str>) -> Result<Conv> {
        let w = self.t(&format!("{p}.weight"))?;
        let k = w.dim(3)?;
        let co = w.dim(0)?;
        let b = if self.has(&format!("{p}.bias")) { Some(self.t(&format!("{p}.bias"))?) } else { None };
        let Some(bn) = bn else { return Ok(Conv { w, b, k }) };
        let (scale, shift) = self.bn(bn)?;
        let w = w.broadcast_mul(&scale.reshape((co, 1, 1, 1))?)?;
        let b = match b {
            Some(b) => (b.broadcast_mul(&scale)? + shift)?,
            None => shift,
        };
        Ok(Conv { w, b: Some(b), k })
    }

    fn deform(&mut self, p: &str, bn: &str) -> Result<Deform> {
        let offset = self.conv(&format!("{p}.offset_conv"), None)?;
        let modulator = self.conv(&format!("{p}.modulator_conv"), None)?;
        let regular = self.conv(&format!("{p}.regular_conv"), Some(bn))?;
        let (co, ci, k, _) = regular.w.dims4()?;
        // [Cout, Cin, k, k] → [Cout, k·k, Cin], to match the taps' layout.
        let w = regular.w.reshape((co, ci, k * k))?.permute((0, 2, 1))?.reshape((co, k * k * ci))?.contiguous()?;
        Ok(Deform { offset, modulator, w, b: regular.b.expect("a folded BatchNorm has a bias"), k })
    }

    fn aspp(&mut self, p: &str) -> Result<AsppDeformable> {
        let mut branches = vec![self.deform(&format!("{p}.aspp1.atrous_conv"), &format!("{p}.aspp1.bn"))?];
        for i in 0..3 {
            branches.push(self.deform(&format!("{p}.aspp_deforms.{i}.atrous_conv"), &format!("{p}.aspp_deforms.{i}.bn"))?);
        }
        let pool = self.conv(&format!("{p}.global_avg_pool.1"), Some(&format!("{p}.global_avg_pool.2")))?;
        let out = self.conv(&format!("{p}.conv1"), Some(&format!("{p}.bn1")))?;
        Ok(AsppDeformable { branches, pool, out })
    }

    fn dec_blk(&mut self, p: &str) -> Result<DecBlk> {
        Ok(DecBlk {
            conv_in: self.conv(&format!("{p}.conv_in"), Some(&format!("{p}.bn_in")))?,
            att: self.aspp(&format!("{p}.dec_att"))?,
            conv_out: self.conv(&format!("{p}.conv_out"), Some(&format!("{p}.bn_out")))?,
        })
    }
}

impl BiRefNet {
    /// From a BiRefNet folder (`model.safetensors`, Swin-L, the default configuration).
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let file = if dir.is_file() { dir.to_path_buf() } else { dir.join("model.safetensors") };
        let mut l = Loader { store: Store::open(&file, 0)?, dev: dev.clone() };
        if !l.has("bb.layers.2.blocks.17.attn.qkv.weight") || l.t("bb.patch_embed.proj.weight")?.dim(0)? != 192 {
            candle_core::bail!("{}: not a BiRefNet with the Swin-L backbone", file.display());
        }
        // The relative position index, as Swin computes it: (Δy + 11)·23 + (Δx + 11).
        let n = WINDOW * WINDOW;
        let index: Vec<u32> = (0..n * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                ((i / WINDOW + WINDOW - 1 - j / WINDOW) * (2 * WINDOW - 1) + (i % WINDOW + WINDOW - 1 - j % WINDOW)) as u32
            })
            .collect();
        let index = Tensor::from_vec(index, n * n, dev)?;
        let heads = [6, 12, 24, 48];
        let depths = [2, 2, 18, 2];
        let mut stages = Vec::new();
        for s in 0..4 {
            let mut blocks = Vec::new();
            for i in 0..depths[s] {
                let p = format!("bb.layers.{s}.blocks.{i}");
                let table = l.t(&format!("{p}.attn.relative_position_bias_table"))?;
                let bias = table.index_select(&index, 0)?.reshape((n, n, heads[s]))?.permute((2, 0, 1))?.contiguous()?;
                blocks.push(SwinBlock {
                    norm1: l.norm(&format!("{p}.norm1"))?,
                    qkv: l.linear(&format!("{p}.attn.qkv"))?,
                    proj: l.linear(&format!("{p}.attn.proj"))?,
                    bias,
                    norm2: l.norm(&format!("{p}.norm2"))?,
                    fc1: l.linear(&format!("{p}.mlp.fc1"))?,
                    fc2: l.linear(&format!("{p}.mlp.fc2"))?,
                    heads: heads[s],
                    shift: if i % 2 == 0 { 0 } else { WINDOW / 2 },
                });
            }
            let merge = if s < 3 { Some((l.norm(&format!("bb.layers.{s}.downsample.norm"))?, l.linear(&format!("bb.layers.{s}.downsample.reduction"))?)) } else { None };
            stages.push(Stage { blocks, norm: l.norm(&format!("bb.norm{s}"))?, merge });
        }
        let gate = |l: &mut Loader, i: usize| -> Result<(Conv, Conv)> { Ok((l.conv(&format!("decoder.gdt_convs_{i}.0"), Some(&format!("decoder.gdt_convs_{i}.1")))?, l.conv(&format!("decoder.gdt_convs_attn_{i}.0"), None)?)) };
        let input = |l: &mut Loader, i: usize| -> Result<(Conv, Conv)> { Ok((l.conv(&format!("decoder.ipt_blk{i}.conv1"), None)?, l.conv(&format!("decoder.ipt_blk{i}.conv_out"), None)?)) };
        Ok(Self {
            patch: l.conv("bb.patch_embed.proj", None)?,
            patch_norm: l.norm("bb.patch_embed.norm")?,
            stages,
            squeeze: l.dec_blk("squeeze_module.0")?,
            blocks: [l.dec_blk("decoder.decoder_block4")?, l.dec_blk("decoder.decoder_block3")?, l.dec_blk("decoder.decoder_block2")?, l.dec_blk("decoder.decoder_block1")?],
            lateral: [l.conv("decoder.lateral_block4.conv", None)?, l.conv("decoder.lateral_block3.conv", None)?, l.conv("decoder.lateral_block2.conv", None)?],
            gates: [gate(&mut l, 4)?, gate(&mut l, 3)?, gate(&mut l, 2)?],
            inputs: [input(&mut l, 5)?, input(&mut l, 4)?, input(&mut l, 3)?, input(&mut l, 2)?, input(&mut l, 1)?],
            out: l.conv("decoder.conv_out1.0", None)?,
            dev: dev.clone(),
        })
    }

    /// The backbone's four levels of `x` [3, S, S], each [C, H, W].
    fn backbone(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        // The patch embedding: a 4×4, stride-4 convolution, then a norm over channels.
        let e = x.unsqueeze(0)?.conv2d(&self.patch.w, 0, 4, 1, 1)?.squeeze(0)?.broadcast_add(&self.patch.b.as_ref().expect("the patch embedding has a bias").reshape(((), 1, 1))?)?;
        let (c, mut h, mut w) = e.dims3()?;
        let mut t = layer_norm(&e.reshape((c, h * w))?.t()?.contiguous()?, Some(&self.patch_norm.0), Some(&self.patch_norm.1), 1e-5)?;
        let mut levels = Vec::new();
        for stage in &self.stages {
            let (hp, wp) = (h.div_ceil(WINDOW) * WINDOW, w.div_ceil(WINDOW) * WINDOW);
            let mask = shift_mask(hp, wp, WINDOW / 2, &self.dev)?;
            for b in &stage.blocks {
                t = b.forward(&t, h, w, Some(&mask))?;
            }
            let c = t.dim(1)?;
            levels.push(layer_norm(&t, Some(&stage.norm.0), Some(&stage.norm.1), 1e-5)?.t()?.reshape((c, h, w))?);
            if let Some((norm, reduction)) = &stage.merge {
                let mut g = t.reshape((h, w, c))?;
                if h % 2 == 1 || w % 2 == 1 {
                    g = g.pad_with_zeros(0, 0, h % 2)?.pad_with_zeros(1, 0, w % 2)?;
                }
                let (h2, w2) = (h.div_ceil(2), w.div_ceil(2));
                let g = g.reshape((h2, 2, w2, 2, c))?;
                let part = |dy: usize, dx: usize| -> Result<Tensor> { g.narrow(1, dy, 1)?.narrow(3, dx, 1)?.reshape((h2 * w2, c)) };
                let merged = Tensor::cat(&[&part(0, 0)?, &part(1, 0)?, &part(0, 1)?, &part(1, 1)?], 1)?;
                t = reduction.forward(&layer_norm(&merged, Some(&norm.0), Some(&norm.1), 1e-5)?)?;
                (h, w) = (h2, w2);
            }
        }
        Ok(levels)
    }

    /// The four encoder levels: the backbone at full and half size, joined, the top one with its context.
    fn encode(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let (_, s, _) = x.dims3()?;
        let full = self.backbone(x)?;
        let half = self.backbone(&interpolate(x, s / 2, s / 2)?)?;
        let mut levels = Vec::new();
        for (f, hf) in full.iter().zip(&half) {
            let (_, h, w) = f.dims3()?;
            levels.push(Tensor::cat(&[f, &interpolate(hf, h, w)?], 0)?);
        }
        let (_, h4, w4) = levels[3].dims3()?;
        let context = [interpolate(&levels[0], h4, w4)?, interpolate(&levels[1], h4, w4)?, interpolate(&levels[2], h4, w4)?, levels[3].clone()];
        levels[3] = Tensor::cat(&context, 0)?;
        Ok(levels)
    }

    /// The logits [S, S] of `x` [3, S, S] (ImageNet-normalized; S = 1024).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let levels = self.encode(x)?;
        let top = self.squeeze.forward(&levels[3])?;
        let input = |i: usize, like: &Tensor| -> Result<Tensor> {
            let (_, h, w) = like.dims3()?;
            let (a, b) = &self.inputs[i];
            b.forward(&a.forward(&interpolate(&patches(x, h)?, h, w)?)?)
        };
        let mut p = Tensor::cat(&[&top, &input(0, &top)?], 0)?;
        for i in 0..3 {
            let mut q = self.blocks[i].forward(&p)?;
            let (gate, attn) = &self.gates[i];
            q = q.broadcast_mul(&candle_nn::ops::sigmoid(&attn.forward(&gate.forward(&q)?.relu()?)?)?)?;
            let skip = &levels[2 - i];
            let (_, h, w) = skip.dims3()?;
            let up = (interpolate(&q, h, w)? + self.lateral[i].forward(skip)?)?;
            p = Tensor::cat(&[&up, &input(i + 1, &up)?], 0)?;
        }
        let (_, s, _) = x.dims3()?;
        let q = interpolate(&self.blocks[3].forward(&p)?, s, s)?;
        let q = Tensor::cat(&[&q, &input(4, &q)?], 0)?;
        self.out.forward(&q)?.squeeze(0)
    }

    /// The object's matte in an RGB picture (`w` × `h`, 8-bit), 8-bit at the
    /// picture's size, as Pixal3D's wrapper makes it: the picture resized to
    /// 1024² (bilinear), ImageNet-normalized; the logits' sigmoid, truncated to
    /// 8 bits and resized back (bicubic), both as Pillow does.
    pub fn matte(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        let small = nrob_image::resize::resample(rgb, 3, w, h, SIZE, SIZE, nrob_image::resize::Filter::Bilinear);
        let x = Tensor::from_vec(small.into_iter().map(|v| v as f32 / 255.).collect::<Vec<f32>>(), (SIZE, SIZE, 3), &self.dev)?.permute((2, 0, 1))?.contiguous()?;
        let x = crate::model3d::prepare::imagenet(&x)?;
        let prob: Vec<f32> = candle_nn::ops::sigmoid(&self.forward(&x)?)?.flatten_all()?.to_vec1()?;
        let mask: Vec<u8> = prob.into_iter().map(|p| (p * 255.) as u8).collect();
        Ok(nrob_image::resize::resample(&mask, 1, SIZE, SIZE, w, h, nrob_image::resize::Filter::Bicubic))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("BIREFNET_REF").unwrap_or_else(|_| "E:/p3dref/birefnet".into()))
    }

    fn reference(name: &str, dev: &Device) -> Tensor {
        let shapes = nrob::json::Json::parse(&std::fs::read(dir().join("shapes.json")).unwrap()).unwrap();
        let shape: Vec<usize> = match shapes.get(name).unwrap() {
            nrob::json::Json::Arr(a) => a.iter().map(|v| v.as_i64().unwrap() as usize).collect(),
            _ => panic!("no shape for {name}"),
        };
        let data: Vec<f32> = std::fs::read(dir().join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        // Batch of one: drop it.
        Tensor::from_vec(data, shape[1..].to_vec(), dev).unwrap()
    }

    /// Relative RMS error of `a` against `b`, and their largest difference.
    fn compare(name: &str, a: &Tensor, b: &Tensor) -> f32 {
        let a: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a.len(), b.len(), "{name}: sizes differ");
        let (mut d2, mut b2, mut max) = (0f64, 0f64, 0f32);
        for (x, y) in a.iter().zip(&b) {
            d2 += ((x - y) as f64).powi(2);
            b2 += (*y as f64).powi(2);
            max = max.max((x - y).abs());
        }
        let rel = (d2 / b2.max(1e-30)).sqrt() as f32;
        println!("{name}: relative RMS error {:.3e}, max difference {max:.3e}", rel);
        rel
    }

    /// Each stage against BiRefNet's own, from the reference's input
    /// (tools: scratchpad ref_birefnet.py).   cargo test --release --features flash-attn --lib birefnet -- --ignored --nocapture
    #[test]
    #[ignore]
    fn matches_the_reference() -> Result<()> {
        let gpu: usize = std::env::var("NROB_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let dev = Device::cuda_if_available(gpu)?;
        let t = std::time::Instant::now();
        let net = BiRefNet::load(Path::new(&std::env::var("BIREFNET").unwrap_or_else(|_| "E:/models/BiRefNet".into())), &dev)?;
        println!("loaded in {:.1}s", t.elapsed().as_secs_f64());
        let x = reference("input", &dev);
        let full = net.backbone(&x)?;
        for (i, f) in full.iter().enumerate() {
            assert!(compare(&format!("bb_{}", i + 1), f, &reference(&format!("bb_{}", i + 1), &dev)) < 1e-3);
        }
        let levels = net.encode(&x)?;
        for (i, f) in levels.iter().enumerate() {
            assert!(compare(&format!("enc_{}", i + 1), f, &reference(&format!("enc_{}", i + 1), &dev)) < 1e-3);
        }
        let top = net.squeeze.forward(&reference("enc_4", &dev))?;
        assert!(compare("squeeze", &top, &reference("squeeze", &dev)) < 1e-3);
        let (a, b) = &net.inputs[0];
        let ipt5 = b.forward(&a.forward(&patches(&x, 32)?)?)?;
        assert!(compare("ipt5", &ipt5, &reference("ipt5", &dev)) < 1e-3);
        let blk_in = Tensor::cat(&[&reference("squeeze", &dev), &reference("ipt5", &dev)], 0)?;
        let h = net.blocks[0].conv_in.forward(&blk_in)?.relu()?;
        assert!(compare("dec4_in", &h, &reference("dec4_in", &dev)) < 1e-3);
        assert!(compare("dec4_att", &net.blocks[0].att.forward(&reference("dec4_in", &dev))?, &reference("dec4_att", &dev)) < 1e-3);
        assert!(compare("dec4", &net.blocks[0].forward(&blk_in)?, &reference("dec4", &dev)) < 1e-3);
        let t = std::time::Instant::now();
        let logits = net.forward(&x)?;
        let logits = logits.to_device(&Device::Cpu)?;
        println!("forward in {:.2}s", t.elapsed().as_secs_f64());
        assert!(compare("logits", &logits.unsqueeze(0)?, &reference("logits", &Device::Cpu)) < 1e-3);
        // The mask agrees wherever it is not a soft edge.
        let p: Vec<f32> = candle_nn::ops::sigmoid(&logits)?.flatten_all()?.to_vec1()?;
        let q: Vec<f32> = candle_nn::ops::sigmoid(&reference("logits", &Device::Cpu))?.flatten_all()?.to_vec1()?;
        let differ = p.iter().zip(&q).filter(|(a, b)| (*a > &0.5) != (*b > &0.5)).count();
        println!("mask pixels on the other side of 0.5: {differ} of {}", p.len());
        assert!(differ * 10_000 < p.len());
        Ok(())
    }
}
