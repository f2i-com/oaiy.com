//! The FastConformer encoder (NeMo's `ConformerEncoder` as Parakeet uses it):
//! `dw_striding` subsampling by 8, then conformer layers of a half-step
//! feed-forward, relative-position self-attention, a depthwise convolution
//! module and a second half-step feed-forward.
//!
//! One utterance at a time: tensors are `(time, channels)` without a batch
//! axis, and only the valid frames are carried (NeMo masks the padding of a
//! batch before every convolution and in attention, which for a single
//! utterance is the same as not having it).
//!
//! Precision follows PyTorch autocast: weights may be f16/bf16 and matrix
//! products run in them, while the residual stream, normalisation,
//! convolutions and attention scores stay f32 (NeMo computes attention in
//! f32 even under autocast).

use candle_core::{DType, Device, Module, Tensor};

use super::weights::Weights;
use super::{bad, Result};

/// A linear layer: `x W^T + b`, the product in the weight's dtype, the result f32.
pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &mut Weights, prefix: &str, out: usize, inp: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let weight = w.tensor(&format!("{prefix}.weight"), &[out, inp], dtype, dev)?;
        let bias = w.optional(&format!("{prefix}.bias"), &[out], DType::F32, dev)?;
        Ok(Self { w: weight, b: bias })
    }

    /// A 1x1 convolution's weight `(out, in, 1)` (or `(out, in, 1, 1)`) as a linear layer.
    pub fn load_pointwise(w: &mut Weights, prefix: &str, out: usize, inp: usize, dims: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let shape: Vec<usize> = [out, inp].into_iter().chain(std::iter::repeat_n(1, dims - 2)).collect();
        let weight = w.tensor(&format!("{prefix}.weight"), &shape, dtype, dev)?.reshape((out, inp))?;
        let bias = w.optional(&format!("{prefix}.bias"), &[out], DType::F32, dev)?;
        Ok(Self { w: weight, b: bias })
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }

    pub fn out_dim(&self) -> usize {
        self.w.dims()[0]
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.to_dtype(self.w.dtype())?.matmul(&self.w.t()?)?.to_dtype(DType::F32)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

struct LayerNorm {
    w: Tensor,
    b: Tensor,
}

impl LayerNorm {
    fn load(w: &mut Weights, prefix: &str, d: usize, dev: &Device) -> Result<Self> {
        Ok(Self { w: w.tensor(&format!("{prefix}.weight"), &[d], DType::F32, dev)?, b: w.tensor(&format!("{prefix}.bias"), &[d], DType::F32, dev)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, &self.w, &self.b, 1e-5)
    }
}

/// A depthwise convolution over time, `(time, channels)` in and out, zero
/// padded to keep the length, with BatchNorm (inference statistics) folded
/// into its taps and bias.
struct DepthwiseTime {
    /// `(kernel, channels)`: row k scales the input shifted by k - pad.
    taps: Tensor,
    bias: Tensor,
    kernel: usize,
}

impl DepthwiseTime {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (t, _c) = x.dims2()?;
        let pad = (self.kernel - 1) / 2;
        let xp = x.pad_with_zeros(0, pad, pad)?;
        let mut acc = self.bias.unsqueeze(0)?.broadcast_as(x.shape())?.contiguous()?;
        for k in 0..self.kernel {
            let tap = self.taps.narrow(0, k, 1)?;
            acc = (acc + xp.narrow(0, k, t)?.broadcast_mul(&tap)?)?;
        }
        Ok(acc)
    }
}

/// A 3x3, stride-2, padding-1 convolution of `(channels, time, freq)`
/// feature maps, either depthwise (one filter per channel) or from a single
/// input channel to many (the first subsampling layer). The input is split
/// into its four even/odd phases so that each of the nine taps is a plain
/// shifted view.
struct Stride2Conv {
    /// `(channels, 3, 3)`.
    w: Vec<Tensor>,
    bias: Tensor,
}

impl Stride2Conv {
    fn load(w: &mut Weights, prefix: &str, channels: usize, dev: &Device) -> Result<Self> {
        let (v, shape) = w.f32(&format!("{prefix}.weight"))?;
        if shape != [channels, 1, 3, 3] {
            return Err(bad(format!("{prefix}.weight: shape {shape:?}, expected [{channels}, 1, 3, 3]")));
        }
        // One (channels, 1, 1) tensor per tap.
        let mut taps = Vec::with_capacity(9);
        for k in 0..9 {
            let col: Vec<f32> = (0..channels).map(|c| v[c * 9 + k]).collect();
            taps.push(Tensor::from_vec(col, (channels, 1, 1), dev)?);
        }
        let bias = w.optional(&format!("{prefix}.bias"), &[channels], DType::F32, dev)?.unwrap_or(Tensor::zeros(channels, DType::F32, dev)?);
        Ok(Self { w: taps, bias: bias.reshape((channels, 1, 1))? })
    }

    /// Output length of a stride-2, kernel-3, padding-1 convolution.
    fn out_len(n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (n - 1) / 2 + 1
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let (ho, wo) = (Self::out_len(h), Self::out_len(w));
        // Pad to (2 ho + 2, 2 wo + 2): one zero before, the rest after.
        let xp = x.pad_with_zeros(1, 1, 2 * ho + 1 - h)?.pad_with_zeros(2, 1, 2 * wo + 1 - w)?;
        let phases = xp.reshape((c, ho + 1, 2, wo + 1, 2))?;
        let out_c = self.w[0].dim(0)?;
        let mut acc = self.bias.broadcast_as((out_c, ho, wo))?.contiguous()?;
        for i in 0..3 {
            for j in 0..3 {
                let phase = phases.narrow(2, i % 2, 1)?.narrow(4, j % 2, 1)?.squeeze(4)?.squeeze(2)?;
                let tap = phase.narrow(1, i / 2, ho)?.narrow(2, j / 2, wo)?;
                acc = (acc + tap.broadcast_mul(&self.w[i * 3 + j])?)?;
            }
        }
        Ok(acc)
    }
}

/// `dw_striding` subsampling: conv 3x3/2 (1 to C channels), ReLU, then
/// twice (depthwise 3x3/2, pointwise 1x1, ReLU), then a linear layer from
/// C x (mels / 8) to d_model.
struct Subsampling {
    conv0: Stride2Conv,
    /// Each stage's depthwise convolution, then its pointwise weight
    /// `(C, C)` and bias `(C, 1)`.
    stages: Vec<(Stride2Conv, Tensor, Tensor)>,
    out: Linear,
    channels: usize,
}

impl Subsampling {
    fn load(w: &mut Weights, n_mels: usize, d_model: usize, dtype: DType, dev: &Device) -> Result<Self> {
        let channels = w.shape("encoder.pre_encode.conv.0.weight")?[0];
        let conv0 = Stride2Conv::load(w, "encoder.pre_encode.conv.0", channels, dev)?;
        let mut stages = Vec::new();
        let mut freq = Stride2Conv::out_len(n_mels);
        for (dw, pw) in [(2, 3), (5, 6)] {
            let conv = Stride2Conv::load(w, &format!("encoder.pre_encode.conv.{dw}"), channels, dev)?;
            let point = w.tensor(&format!("encoder.pre_encode.conv.{pw}.weight"), &[channels, channels, 1, 1], DType::F32, dev)?.reshape((channels, channels))?;
            let bias = w.optional(&format!("encoder.pre_encode.conv.{pw}.bias"), &[channels], DType::F32, dev)?.unwrap_or(Tensor::zeros(channels, DType::F32, dev)?);
            stages.push((conv, point, bias.reshape((channels, 1))?));
            freq = Stride2Conv::out_len(freq);
        }
        let out = Linear::load(w, "encoder.pre_encode.out", d_model, channels * freq, dtype, dev)?;
        Ok(Self { conv0, stages, out, channels })
    }

    /// `(mels, frames)` features to `(frames / 8, d_model)`.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = features.t()?.unsqueeze(0)?; // (1, time, freq)
        let mut x = self.conv0.forward(&x)?.relu()?;
        for (dw, pw, pb) in &self.stages {
            x = dw.forward(&x)?;
            let (c, h, f) = x.dims3()?;
            // Pointwise: (C, C) times (C, time * freq).
            x = pw.matmul(&x.reshape((c, h * f))?)?.broadcast_add(pb)?.reshape((c, h, f))?.relu()?;
        }
        let (c, t, f) = x.dims3()?;
        debug_assert_eq!(c, self.channels);
        // (C, time, freq) to (time, C * freq), channel-major like NeMo's flatten.
        let x = x.transpose(0, 1)?.reshape((t, c * f))?;
        self.out.forward(&x)
    }
}

struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&candle_nn::ops::silu(&self.up.forward(x)?)?)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    pos: Linear,
    /// `(heads, 1, head_dim)`.
    bias_u: Tensor,
    bias_v: Tensor,
    heads: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (t, d) = x.dims2()?;
        let (h, dk) = (self.heads, d / self.heads);
        let split = |y: Tensor, n: usize| -> Result<Tensor> { y.reshape((n, h, dk))?.transpose(0, 1)?.contiguous() };
        let q = split(self.q.forward(x)?, t)?;
        let k = split(self.k.forward(x)?, t)?;
        let v = split(self.v.forward(x)?, t)?;
        let p = split(self.pos.forward(pos_emb)?, 2 * t - 1)?;
        let ac = q.broadcast_add(&self.bias_u)?.matmul(&k.t()?)?;
        let bd = rel_shift(&q.broadcast_add(&self.bias_v)?.matmul(&p.t()?)?)?;
        let scores = ((ac + bd)? * (1.0 / (dk as f64).sqrt()))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let y = attn.matmul(&v)?.transpose(0, 1)?.reshape((t, d))?;
        self.out.forward(&y)
    }
}

/// Transformer-XL's relative shift: `(heads, t, 2t - 1)` scores against
/// relative positions t-1 .. -(t-1) become `(heads, t, t)` scores against keys,
/// `out[h, i, j] = x[h, i, t - 1 - i + j]` (relative position i - j).
pub fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (h, t, p) = x.dims3()?;
    debug_assert_eq!(p, 2 * t - 1);
    let x = x.pad_with_zeros(2, 1, 0)?; // (h, t, 2t)
    let x = x.reshape((h, 2 * t, t))?.narrow(1, 1, 2 * t - 1)?;
    x.contiguous()?.reshape((h, t, p))?.narrow(2, 0, t)
}

struct ConvModule {
    pw1: Linear,
    dw: DepthwiseTime,
    pw2: Linear,
}

impl ConvModule {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.pw1.forward(x)?;
        let d = y.dim(1)? / 2;
        // GLU over channels: first half times the sigmoid of the second.
        let y = (y.narrow(1, 0, d)? * candle_nn::ops::sigmoid(&y.narrow(1, d, d)?)?)?;
        let y = candle_nn::ops::silu(&self.dw.forward(&y)?)?;
        self.pw2.forward(&y)
    }
}

struct Layer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_att: LayerNorm,
    att: Attention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
}

impl Layer {
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let x = (x + (self.ff1.forward(&self.norm_ff1.forward(x)?)? * 0.5)?)?;
        let x = (&x + self.att.forward(&self.norm_att.forward(&x)?, pos_emb)?)?;
        let x = (&x + self.conv.forward(&self.norm_conv.forward(&x)?)?)?;
        let x = (&x + (self.ff2.forward(&self.norm_ff2.forward(&x)?)? * 0.5)?)?;
        self.norm_out.forward(&x)
    }
}

/// Sizes read from the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderShape {
    pub layers: usize,
    pub d_model: usize,
    pub heads: usize,
    pub ff: usize,
    pub kernel: usize,
    pub n_mels: usize,
}

pub struct Encoder {
    sub: Subsampling,
    layers: Vec<Layer>,
    xscale: Option<f64>,
    pub shape: EncoderShape,
    dev: Device,
}

impl Encoder {
    pub fn load(w: &mut Weights, n_mels: usize, xscaling: bool, dtype: DType, dev: &Device) -> Result<Self> {
        let d_model = w.shape("encoder.layers.0.self_attn.linear_q.weight")?[0];
        let heads = w.shape("encoder.layers.0.self_attn.pos_bias_u")?[0];
        let ff = w.shape("encoder.layers.0.feed_forward1.linear1.weight")?[0];
        let kernel = w.shape("encoder.layers.0.conv.depthwise_conv.weight")?[2];
        let mut layers_n = 0;
        while w.has(&format!("encoder.layers.{layers_n}.norm_out.weight")) {
            layers_n += 1;
        }
        if layers_n == 0 || heads == 0 || d_model % heads != 0 || kernel % 2 == 0 {
            return Err(bad(format!("not a FastConformer encoder ({layers_n} layers, d_model {d_model}, {heads} heads, kernel {kernel})")));
        }
        let shape = EncoderShape { layers: layers_n, d_model, heads, ff, kernel, n_mels };
        let sub = Subsampling::load(w, n_mels, d_model, dtype, dev)?;
        let mut layers = Vec::with_capacity(layers_n);
        for l in 0..layers_n {
            let p = format!("encoder.layers.{l}");
            let lin = |w: &mut Weights, name: &str, out: usize, inp: usize| Linear::load(w, &format!("{p}.{name}"), out, inp, dtype, dev);
            let ff1 = FeedForward { up: lin(w, "feed_forward1.linear1", ff, d_model)?, down: lin(w, "feed_forward1.linear2", d_model, ff)? };
            let ff2 = FeedForward { up: lin(w, "feed_forward2.linear1", ff, d_model)?, down: lin(w, "feed_forward2.linear2", d_model, ff)? };
            let dk = d_model / heads;
            let att = Attention {
                q: lin(w, "self_attn.linear_q", d_model, d_model)?,
                k: lin(w, "self_attn.linear_k", d_model, d_model)?,
                v: lin(w, "self_attn.linear_v", d_model, d_model)?,
                out: lin(w, "self_attn.linear_out", d_model, d_model)?,
                pos: lin(w, "self_attn.linear_pos", d_model, d_model)?,
                bias_u: w.tensor(&format!("{p}.self_attn.pos_bias_u"), &[heads, dk], DType::F32, dev)?.unsqueeze(1)?,
                bias_v: w.tensor(&format!("{p}.self_attn.pos_bias_v"), &[heads, dk], DType::F32, dev)?.unsqueeze(1)?,
                heads,
            };
            let conv = ConvModule {
                pw1: Linear::load_pointwise(w, &format!("{p}.conv.pointwise_conv1"), 2 * d_model, d_model, 3, dtype, dev)?,
                dw: load_depthwise(w, &format!("{p}.conv"), d_model, kernel, dev)?,
                pw2: Linear::load_pointwise(w, &format!("{p}.conv.pointwise_conv2"), d_model, d_model, 3, dtype, dev)?,
            };
            let norm = |w: &mut Weights, name: &str| LayerNorm::load(w, &format!("{p}.{name}"), d_model, dev);
            layers.push(Layer {
                norm_ff1: norm(w, "norm_feed_forward1")?,
                ff1,
                norm_att: norm(w, "norm_self_att")?,
                att,
                norm_conv: norm(w, "norm_conv")?,
                conv,
                norm_ff2: norm(w, "norm_feed_forward2")?,
                ff2,
                norm_out: norm(w, "norm_out")?,
            });
        }
        let xscale = xscaling.then(|| (d_model as f64).sqrt());
        Ok(Self { sub, layers, xscale, shape, dev: dev.clone() })
    }

    /// Encoder frames for `frames` input frames (the subsampling's length rule).
    pub fn out_frames(frames: usize) -> usize {
        let mut n = frames;
        for _ in 0..3 {
            n = Stride2Conv::out_len(n);
        }
        n
    }

    /// `(mels, frames)` f32 features to `(frames / 8, d_model)` f32 encodings.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let mut x = self.sub.forward(features)?;
        if let Some(s) = self.xscale {
            x = (x * s)?;
        }
        let t = x.dim(0)?;
        let pos_emb = rel_pos_emb(t, self.shape.d_model, &self.dev)?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok(x)
    }

    /// The output after the subsampling and the first `n` layers (parity tests).
    pub fn forward_layers(&self, features: &Tensor, n: usize) -> Result<Tensor> {
        let mut x = self.sub.forward(features)?;
        if let Some(s) = self.xscale {
            x = (x * s)?;
        }
        let pos_emb = rel_pos_emb(x.dim(0)?, self.shape.d_model, &self.dev)?;
        for layer in self.layers.iter().take(n) {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok(x)
    }
}

/// The depthwise convolution of a conformer layer with its BatchNorm folded in.
fn load_depthwise(w: &mut Weights, prefix: &str, d: usize, kernel: usize, dev: &Device) -> Result<DepthwiseTime> {
    let (taps, shape) = w.f32(&format!("{prefix}.depthwise_conv.weight"))?;
    if shape != [d, 1, kernel] {
        return Err(bad(format!("{prefix}.depthwise_conv.weight: shape {shape:?}")));
    }
    let bias = if w.has(&format!("{prefix}.depthwise_conv.bias")) { w.f32(&format!("{prefix}.depthwise_conv.bias"))?.0 } else { vec![0.0; d] };
    let get = |w: &mut Weights, n: &str| -> Result<Vec<f32>> {
        let (v, s) = w.f32(&format!("{prefix}.batch_norm.{n}"))?;
        if s != [d] {
            return Err(bad(format!("{prefix}.batch_norm.{n}: shape {s:?}")));
        }
        Ok(v)
    };
    let (gamma, beta, mean, var) = (get(w, "weight")?, get(w, "bias")?, get(w, "running_mean")?, get(w, "running_var")?);
    let mut folded = vec![0f32; kernel * d];
    let mut folded_bias = vec![0f32; d];
    for c in 0..d {
        let scale = gamma[c] as f64 / (var[c] as f64 + 1e-5).sqrt();
        for k in 0..kernel {
            folded[k * d + c] = (taps[c * kernel + k] as f64 * scale) as f32;
        }
        folded_bias[c] = ((bias[c] as f64 - mean[c] as f64) * scale + beta[c] as f64) as f32;
    }
    Ok(DepthwiseTime { taps: Tensor::from_vec(folded, (kernel, d), dev)?, bias: Tensor::from_vec(folded_bias, d, dev)?, kernel })
}

/// NeMo's `RelPositionalEncoding` for `t` frames: sinusoids of the relative
/// positions t-1 down to -(t-1), `(2t - 1, d)`, computed in f32 as torch does.
pub fn rel_pos_emb(t: usize, d: usize, dev: &Device) -> Result<Tensor> {
    let n = 2 * t - 1;
    let scale = -(10000f64.ln() / d as f64) as f32;
    let div: Vec<f32> = (0..d / 2).map(|i| ((2 * i) as f32 * scale).exp()).collect();
    let mut pe = vec![0f32; n * d];
    for r in 0..n {
        let pos = (t as f32 - 1.0) - r as f32;
        for (i, dv) in div.iter().enumerate() {
            let a = pos * dv;
            pe[r * d + 2 * i] = a.sin();
            pe[r * d + 2 * i + 1] = a.cos();
        }
    }
    Tensor::from_vec(pe, (n, d), dev)
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn rel_shift_picks_relative_positions() {
        // x[h, i, k] = 100 i + k: out[h, i, j] must be x[h, i, t - 1 - i + j].
        let (h, t) = (2, 4);
        let p = 2 * t - 1;
        let v: Vec<f32> = (0..h * t * p).map(|n| ((n / p) % t * 100 + n % p) as f32).collect();
        let x = Tensor::from_vec(v, (h, t, p), &Device::Cpu).unwrap();
        let out = rel_shift(&x).unwrap().to_vec3::<f32>().unwrap();
        for hh in 0..h {
            for i in 0..t {
                for j in 0..t {
                    assert_eq!(out[hh][i][j], (100 * i + (t - 1 - i + j)) as f32, "h {hh} i {i} j {j}");
                }
            }
        }
    }

    #[test]
    fn stride2_conv_matches_a_direct_convolution() {
        let dev = Device::Cpu;
        let (c, h, w) = (3usize, 7usize, 6usize);
        let x: Vec<f32> = (0..c * h * w).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.1).collect();
        let k: Vec<f32> = (0..c * 9).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.2).collect();
        let conv = Stride2Conv {
            w: (0..9).map(|t| Tensor::from_vec((0..c).map(|ch| k[ch * 9 + t]).collect::<Vec<_>>(), (c, 1, 1), &dev).unwrap()).collect(),
            bias: Tensor::from_vec(vec![0.5f32, -0.25, 0.0], (c, 1, 1), &dev).unwrap(),
        };
        let out = conv.forward(&Tensor::from_vec(x.clone(), (c, h, w), &dev).unwrap()).unwrap();
        let (ho, wo) = (Stride2Conv::out_len(h), Stride2Conv::out_len(w));
        assert_eq!(out.dims(), &[c, ho, wo]);
        let out = out.to_vec3::<f32>().unwrap();
        let bias = [0.5f32, -0.25, 0.0];
        for ch in 0..c {
            for oy in 0..ho {
                for ox in 0..wo {
                    let mut s = bias[ch];
                    for i in 0..3 {
                        for j in 0..3 {
                            let (y, xx) = (2 * oy as isize + i as isize - 1, 2 * ox as isize + j as isize - 1);
                            if y >= 0 && (y as usize) < h && xx >= 0 && (xx as usize) < w {
                                s += k[ch * 9 + i * 3 + j] * x[ch * h * w + y as usize * w + xx as usize];
                            }
                        }
                    }
                    assert!((out[ch][oy][ox] - s).abs() < 1e-5, "{ch} {oy} {ox}: {} vs {s}", out[ch][oy][ox]);
                }
            }
        }
        // One input channel to many (the first subsampling layer) broadcasts.
        let single = Tensor::from_vec(x[..h * w].to_vec(), (1, h, w), &dev).unwrap();
        assert_eq!(conv.forward(&single).unwrap().dims(), &[c, ho, wo]);
    }

    #[test]
    fn depthwise_time_matches_a_direct_convolution() {
        let dev = Device::Cpu;
        let (t, c, kernel) = (6usize, 2usize, 3usize);
        let x: Vec<f32> = (0..t * c).map(|i| (i as f32 * 0.7).sin()).collect();
        let taps: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]; // (kernel, channels)
        let dw = DepthwiseTime { taps: Tensor::from_vec(taps.clone(), (kernel, c), &dev).unwrap(), bias: Tensor::from_vec(vec![1.0f32, -1.0], c, &dev).unwrap(), kernel };
        let out = dw.forward(&Tensor::from_vec(x.clone(), (t, c), &dev).unwrap()).unwrap().to_vec2::<f32>().unwrap();
        for i in 0..t {
            for ch in 0..c {
                let mut s = [1.0f32, -1.0][ch];
                for k in 0..kernel {
                    let src = i as isize + k as isize - 1;
                    if src >= 0 && (src as usize) < t {
                        s += taps[k * c + ch] * x[src as usize * c + ch];
                    }
                }
                assert!((out[i][ch] - s).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rel_pos_emb_is_centred_on_zero() {
        let pe = rel_pos_emb(3, 4, &Device::Cpu).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(pe.len(), 5);
        // Row 2 is position 0: sin 0 = 0, cos 0 = 1.
        assert_eq!(pe[2], vec![0.0, 1.0, 0.0, 1.0]);
        // Row 0 is position +2; row 4 is -2 (odd in sin, even in cos).
        assert!((pe[0][0] - 2f32.sin()).abs() < 1e-6 && (pe[4][0] + 2f32.sin()).abs() < 1e-6);
        assert!((pe[0][3] - pe[4][3]).abs() < 1e-6);
    }

    #[test]
    fn subsampled_length_follows_nemo() {
        // NeMo's calc_length: floor((n + 2 - 3) / 2 + 1), three times.
        for n in [1usize, 2, 7, 8, 9, 100, 101, 1000, 1001] {
            let mut expect = n as f64;
            for _ in 0..3 {
                expect = ((expect - 1.0) / 2.0 + 1.0).floor();
            }
            assert_eq!(Encoder::out_frames(n), expect as usize, "{n}");
        }
    }
}
