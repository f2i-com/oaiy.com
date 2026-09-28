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
//! On a GPU every candle operation is a kernel launch costing about 10-30 us
//! whatever its size, and a 0.6B encoder over a few seconds of speech is
//! far from compute-bound; so the layers are written for few operations:
//! one product makes the four attention projections (with `pos_bias_u`,
//! `pos_bias_v` and the score scale folded in), every layer's positional
//! projection is one product per pass, the convolutions are a gather and a
//! product, and the feed-forward half-step is folded into its weights.
//!
//! [`Precision`] picks the dtypes: f32 throughout (exact), half weights with
//! f32 activations (PyTorch autocast), or half throughout, which on a GPU
//! is as fast as f32 for short inputs, faster for long ones, and half the
//! memory.

use candle_core::{DType, Device, Module, Tensor};

use super::weights::Weights;
use super::{bad, Result};

/// A linear layer: `x W^T + b`, the product in the weights' dtype. The input
/// and output are in the activations' dtype (the bias's): when that is f32
/// and the weights are half, the input is cast for the product and the result
/// cast back, as PyTorch autocast does.
pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    pub fn load(w: &mut Weights, prefix: &str, out: usize, inp: usize, p: Precision, dev: &Device) -> Result<Self> {
        let weight = w.tensor(&format!("{prefix}.weight"), &[out, inp], p.weights, dev)?;
        let bias = w.optional(&format!("{prefix}.bias"), &[out], p.act, dev)?;
        Ok(Self { w: weight, b: bias })
    }

    /// A 1x1 convolution's weight `(out, in, 1)` (or `(out, in, 1, 1)`) as a linear layer.
    pub fn load_pointwise(w: &mut Weights, prefix: &str, out: usize, inp: usize, dims: usize, p: Precision, dev: &Device) -> Result<Self> {
        let shape: Vec<usize> = [out, inp].into_iter().chain(std::iter::repeat_n(1, dims - 2)).collect();
        let weight = w.tensor(&format!("{prefix}.weight"), &shape, p.weights, dev)?.reshape((out, inp))?;
        let bias = w.optional(&format!("{prefix}.bias"), &[out], p.act, dev)?;
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
        let y = if x.dtype() == self.w.dtype() { x.matmul(&self.w.t()?)? } else { x.to_dtype(self.w.dtype())?.matmul(&self.w.t()?)?.to_dtype(x.dtype())? };
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// The dtypes a model runs in: `weights` for the matrix products, `act`
/// for everything between them. f32/f32 is exact; f16/f32 is PyTorch
/// autocast; f16/f16 keeps the whole encoder in half precision, which on a
/// GPU saves the casts around every product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    pub weights: DType,
    pub act: DType,
}

impl Precision {
    pub fn f32() -> Self {
        Self { weights: DType::F32, act: DType::F32 }
    }
}

struct LayerNorm {
    w: Tensor,
    b: Tensor,
}

impl LayerNorm {
    fn load(w: &mut Weights, prefix: &str, d: usize, p: Precision, dev: &Device) -> Result<Self> {
        Ok(Self { w: w.tensor(&format!("{prefix}.weight"), &[d], p.act, dev)?, b: w.tensor(&format!("{prefix}.bias"), &[d], p.act, dev)? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, &self.w, &self.b, 1e-5)
    }
}

/// Host values as a tensor of `dtype` on `dev`.
fn upload(v: Vec<f32>, shape: &[usize], dtype: DType, dev: &Device) -> Result<Tensor> {
    Tensor::from_vec(v, shape, &Device::Cpu)?.to_dtype(dtype)?.to_device(dev)
}

/// Per-length inputs every layer shares, made once per forward pass.
struct Frame {
    /// Every layer's `linear_pos` of the relative positions,
    /// `(layers, heads, 2t - 1, head_dim)`.
    pos: Tensor,
    /// Rows of `(input ; zero row)` that unfold a depthwise convolution's
    /// window: `t * kernel` indices, out-of-range taps pointing at the zero row.
    dw_index: Tensor,
    zero_row: Tensor,
}

/// A depthwise convolution over time, `(time, channels)` in and out, zero
/// padded to keep the length, with BatchNorm (inference statistics) folded
/// into its taps and bias. The window is unfolded with one gather, so the
/// whole convolution is a handful of kernels whatever the kernel size.
struct DepthwiseTime {
    /// `(1, kernel, channels)`: tap k scales the input shifted by k - pad.
    taps: Tensor,
    bias: Tensor,
    kernel: usize,
}

impl DepthwiseTime {
    /// The unfold indices for `t` frames.
    fn index(t: usize, kernel: usize, dev: &Device) -> Result<Tensor> {
        let pad = (kernel - 1) / 2;
        let idx: Vec<u32> = (0..t).flat_map(|i| (0..kernel).map(move |k| (i + k).checked_sub(pad).filter(|&s| s < t).unwrap_or(t) as u32)).collect();
        Tensor::from_vec(idx, t * kernel, dev)
    }

    fn forward(&self, x: &Tensor, frame: &Frame) -> Result<Tensor> {
        let (t, c) = x.dims2()?;
        let padded = Tensor::cat(&[x, &frame.zero_row], 0)?;
        let window = padded.index_select(&frame.dw_index, 0)?.reshape((t, self.kernel, c))?;
        window.broadcast_mul(&self.taps)?.sum(1)?.broadcast_add(&self.bias)
    }
}

/// A 3x3, stride-2, padding-1 convolution of `(channels, time, freq)`
/// feature maps: depthwise (one filter per channel) or from a single input
/// channel to many (the first subsampling layer). The nine taps of every
/// output position are gathered in one index_select, then weighed with one
/// matrix product.
struct Stride2Conv {
    /// `(out, 9)` from one input channel, or `(channels, 1, 9)` depthwise.
    w: Tensor,
    /// `(out, 1)`.
    bias: Tensor,
    depthwise: bool,
}

impl Stride2Conv {
    fn load(w: &mut Weights, prefix: &str, channels: usize, depthwise: bool, dev: &Device) -> Result<Self> {
        let (v, _) = w.f32_shaped(&format!("{prefix}.weight"), &[channels, 1, 3, 3])?;
        let weight = if depthwise { Tensor::from_vec(v, (channels, 1, 9), dev)? } else { Tensor::from_vec(v, (channels, 9), dev)? };
        let bias = w.optional_f32(&format!("{prefix}.bias"), &[channels])?.unwrap_or_else(|| vec![0.0; channels]);
        Ok(Self { w: weight, bias: Tensor::from_vec(bias, (channels, 1), dev)?, depthwise })
    }

    /// Output length of a stride-2, kernel-3, padding-1 convolution.
    fn out_len(n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (n - 1) / 2 + 1
        }
    }

    /// For an `(h, w)` map: `9 * ho * wo` indices into its flattened cells
    /// with one zero cell appended, tap-major.
    fn index(h: usize, w: usize, dev: &Device) -> Result<Tensor> {
        let (ho, wo) = (Self::out_len(h), Self::out_len(w));
        let mut idx = Vec::with_capacity(9 * ho * wo);
        for i in 0..3 {
            for j in 0..3 {
                for oy in 0..ho {
                    for ox in 0..wo {
                        let (y, x) = ((2 * oy + i).checked_sub(1), (2 * ox + j).checked_sub(1));
                        idx.push(match (y, x) {
                            (Some(y), Some(x)) if y < h && x < w => (y * w + x) as u32,
                            _ => (h * w) as u32,
                        });
                    }
                }
            }
        }
        Tensor::from_vec(idx, 9 * ho * wo, dev)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (c, h, w) = x.dims3()?;
        let (ho, wo) = (Self::out_len(h), Self::out_len(w));
        let flat = Tensor::cat(&[&x.reshape((c, h * w))?, &Tensor::zeros((c, 1), x.dtype(), x.device())?], 1)?;
        let window = flat.index_select(&Self::index(h, w, x.device())?, 1)?.reshape((c, 9, ho * wo))?;
        let y = if self.depthwise { self.w.matmul(&window)?.squeeze(1)? } else { self.w.matmul(&window.squeeze(0)?)? };
        let out = y.dim(0)?;
        y.broadcast_add(&self.bias)?.reshape((out, ho, wo))
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
}

impl Subsampling {
    fn load(w: &mut Weights, n_mels: usize, d_model: usize, p: Precision, dev: &Device) -> Result<Self> {
        let channels = w.shape("encoder.pre_encode.conv.0.weight")?[0];
        let conv0 = Stride2Conv::load(w, "encoder.pre_encode.conv.0", channels, false, dev)?;
        let mut stages = Vec::new();
        let mut freq = Stride2Conv::out_len(n_mels);
        for (dw, pw) in [(2, 3), (5, 6)] {
            let conv = Stride2Conv::load(w, &format!("encoder.pre_encode.conv.{dw}"), channels, true, dev)?;
            let (point, _) = w.f32_shaped(&format!("encoder.pre_encode.conv.{pw}.weight"), &[channels, channels, 1, 1])?;
            let bias = w.optional_f32(&format!("encoder.pre_encode.conv.{pw}.bias"), &[channels])?.unwrap_or_else(|| vec![0.0; channels]);
            stages.push((conv, Tensor::from_vec(point, (channels, channels), dev)?, Tensor::from_vec(bias, (channels, 1), dev)?));
            freq = Stride2Conv::out_len(freq);
        }
        let out = Linear::load(w, "encoder.pre_encode.out", d_model, channels * freq, Precision { weights: p.weights, act: DType::F32 }, dev)?;
        Ok(Self { conv0, stages, out })
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
        // (C, time, freq) to (time, C * freq), channel-major like NeMo's flatten.
        let x = x.transpose(0, 1)?.reshape((t, c * f))?;
        self.out.forward(&x)
    }
}

/// `linear2(silu(linear1(x)))`, with the conformer's half-step factor
/// folded into `linear2` (a power of two: exact in any precision).
struct FeedForward {
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn load(w: &mut Weights, prefix: &str, d: usize, ff: usize, p: Precision, dev: &Device) -> Result<Self> {
        let up = Linear::load(w, &format!("{prefix}.linear1"), ff, d, p, dev)?;
        let half = |v: Vec<f32>| v.into_iter().map(|x| x * 0.5).collect::<Vec<_>>();
        let (v, _) = w.f32_shaped(&format!("{prefix}.linear2.weight"), &[d, ff])?;
        let b = w.optional_f32(&format!("{prefix}.linear2.bias"), &[d])?;
        let down = Linear::from_parts(upload(half(v), &[d, ff], p.weights, dev)?, b.map(|b| upload(half(b), &[d], p.act, dev)).transpose()?);
        Ok(Self { up, down })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down.forward(&candle_nn::ops::silu(&self.up.forward(x)?)?)
    }
}

/// Relative-position self-attention. One product makes four projections:
/// the query with `pos_bias_u` (for content scores), the query with
/// `pos_bias_v` (for position scores), the key and the value; both queries
/// carry the 1/sqrt(head_dim) score scale.
struct Attention {
    qkv: Linear,
    out: Linear,
    heads: usize,
}

impl Attention {
    fn load(w: &mut Weights, p: &str, d: usize, heads: usize, prec: Precision, dev: &Device) -> Result<Self> {
        let dk = d / heads;
        let scale = 1.0 / (dk as f64).sqrt();
        let (wq, _) = w.f32_shaped(&format!("{p}.linear_q.weight"), &[d, d])?;
        let (wk, _) = w.f32_shaped(&format!("{p}.linear_k.weight"), &[d, d])?;
        let (wv, _) = w.f32_shaped(&format!("{p}.linear_v.weight"), &[d, d])?;
        let bq = w.optional_f32(&format!("{p}.linear_q.bias"), &[d])?.unwrap_or_else(|| vec![0.0; d]);
        let bk = w.optional_f32(&format!("{p}.linear_k.bias"), &[d])?.unwrap_or_else(|| vec![0.0; d]);
        let bv = w.optional_f32(&format!("{p}.linear_v.bias"), &[d])?.unwrap_or_else(|| vec![0.0; d]);
        let (u, _) = w.f32_shaped(&format!("{p}.pos_bias_u"), &[heads, dk])?;
        let (v, _) = w.f32_shaped(&format!("{p}.pos_bias_v"), &[heads, dk])?;
        let scaled = |x: &[f32]| x.iter().map(|&a| (a as f64 * scale) as f32).collect::<Vec<_>>();
        let mut weight = Vec::with_capacity(4 * d * d);
        weight.extend(scaled(&wq));
        weight.extend(scaled(&wq));
        weight.extend_from_slice(&wk);
        weight.extend_from_slice(&wv);
        let mut bias = Vec::with_capacity(4 * d);
        bias.extend(bq.iter().zip(&u).map(|(b, u)| ((*b as f64 + *u as f64) * scale) as f32));
        bias.extend(bq.iter().zip(&v).map(|(b, v)| ((*b as f64 + *v as f64) * scale) as f32));
        bias.extend_from_slice(&bk);
        bias.extend_from_slice(&bv);
        let qkv = Linear::from_parts(upload(weight, &[4 * d, d], prec.weights, dev)?, Some(upload(bias, &[4 * d], prec.act, dev)?));
        Ok(Self { qkv, out: Linear::load(w, &format!("{p}.linear_out"), d, d, prec, dev)?, heads })
    }

    /// `x` `(t, d)`; `pos` this layer's projected positions `(heads, 2t - 1, head_dim)`.
    fn forward(&self, x: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (t, d) = x.dims2()?;
        let (h, dk) = (self.heads, d / self.heads);
        // (t, 4d) to (4, heads, t, head_dim): q_u, q_v, k, v.
        let qkv = self.qkv.forward(x)?.reshape((t, 4, h, dk))?.permute((1, 2, 0, 3))?.contiguous()?;
        let (qu, qv, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?, qkv.get(3)?);
        let ac = qu.matmul(&k.t()?)?;
        let bd = rel_shift(&qv.matmul(&pos.t()?)?)?;
        let attn = candle_nn::ops::softmax_last_dim(&(ac + bd)?)?;
        let y = attn.matmul(&v)?.transpose(0, 1)?.reshape((t, d))?;
        self.out.forward(&y)
    }
}

/// Transformer-XL's relative shift: `(heads, t, 2t - 1)` scores against
/// relative positions t-1 .. -(t-1) become `(heads, t, t)` scores against keys,
/// `out[h, i, j] = x[h, i, t - 1 - i + j]` (relative position i - j). Within
/// a head that is the row-major data from offset t-1 read in rows of 2t-2,
/// so it is one narrow of the flattened scores and one copy.
pub fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (h, t, p) = x.dims3()?;
    debug_assert_eq!(p, 2 * t - 1);
    if t == 1 {
        return Ok(x.clone());
    }
    let flat = x.contiguous()?.reshape((h, t * p))?;
    flat.narrow(1, t - 1, t * (2 * t - 2))?.reshape((h, t, 2 * t - 2))?.narrow(2, 0, t)
}

struct ConvModule {
    pw1: Linear,
    dw: DepthwiseTime,
    pw2: Linear,
}

impl ConvModule {
    fn forward(&self, x: &Tensor, frame: &Frame) -> Result<Tensor> {
        let y = self.pw1.forward(x)?;
        let d = y.dim(1)? / 2;
        // GLU over channels: first half times the sigmoid of the second.
        let y = (y.narrow(1, 0, d)? * candle_nn::ops::sigmoid(&y.narrow(1, d, d)?)?)?;
        let y = candle_nn::ops::silu(&self.dw.forward(&y, frame)?)?;
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
    fn forward(&self, x: &Tensor, pos: &Tensor, frame: &Frame) -> Result<Tensor> {
        let x = (x + self.ff1.forward(&self.norm_ff1.forward(x)?)?)?;
        let x = (&x + self.att.forward(&self.norm_att.forward(&x)?, pos)?)?;
        let x = (&x + self.conv.forward(&self.norm_conv.forward(&x)?, frame)?)?;
        let x = (&x + self.ff2.forward(&self.norm_ff2.forward(&x)?)?)?;
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
    /// Every layer's `linear_pos`, stacked: `(layers * d_model, d_model)`.
    pos: Linear,
    xscale: Option<f64>,
    pub shape: EncoderShape,
    pub precision: Precision,
    dev: Device,
}

impl Encoder {
    pub fn load(w: &mut Weights, n_mels: usize, xscaling: bool, prec: Precision, dev: &Device) -> Result<Self> {
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
        let sub = Subsampling::load(w, n_mels, d_model, prec, dev)?;
        let mut layers = Vec::with_capacity(layers_n);
        let mut pos = Vec::with_capacity(layers_n * d_model * d_model);
        for l in 0..layers_n {
            let p = format!("encoder.layers.{l}");
            pos.extend(w.f32_shaped(&format!("{p}.self_attn.linear_pos.weight"), &[d_model, d_model])?.0);
            let conv = ConvModule {
                pw1: Linear::load_pointwise(w, &format!("{p}.conv.pointwise_conv1"), 2 * d_model, d_model, 3, prec, dev)?,
                dw: load_depthwise(w, &format!("{p}.conv"), d_model, kernel, prec.act, dev)?,
                pw2: Linear::load_pointwise(w, &format!("{p}.conv.pointwise_conv2"), d_model, d_model, 3, prec, dev)?,
            };
            let norm = |w: &mut Weights, name: &str| LayerNorm::load(w, &format!("{p}.{name}"), d_model, prec, dev);
            layers.push(Layer {
                norm_ff1: norm(w, "norm_feed_forward1")?,
                ff1: FeedForward::load(w, &format!("{p}.feed_forward1"), d_model, ff, prec, dev)?,
                norm_att: norm(w, "norm_self_att")?,
                att: Attention::load(w, &format!("{p}.self_attn"), d_model, heads, prec, dev)?,
                norm_conv: norm(w, "norm_conv")?,
                conv,
                norm_ff2: norm(w, "norm_feed_forward2")?,
                ff2: FeedForward::load(w, &format!("{p}.feed_forward2"), d_model, ff, prec, dev)?,
                norm_out: norm(w, "norm_out")?,
            });
        }
        let pos = Linear::from_parts(upload(pos, &[layers_n * d_model, d_model], prec.weights, dev)?, None);
        let xscale = xscaling.then(|| (d_model as f64).sqrt());
        Ok(Self { sub, layers, pos, xscale, shape, precision: prec, dev: dev.clone() })
    }

    /// Encoder frames for `frames` input frames (the subsampling's length rule).
    pub fn out_frames(frames: usize) -> usize {
        let mut n = frames;
        for _ in 0..3 {
            n = Stride2Conv::out_len(n);
        }
        n
    }

    fn frame(&self, t: usize) -> Result<Frame> {
        let s = &self.shape;
        let dk = s.d_model / s.heads;
        let pe = rel_pos_emb(t, s.d_model, &self.dev)?.to_dtype(self.precision.act)?;
        let pos = self.pos.forward(&pe)?.reshape((2 * t - 1, s.layers, s.heads, dk))?.permute((1, 2, 0, 3))?.contiguous()?;
        Ok(Frame { pos, dw_index: DepthwiseTime::index(t, s.kernel, &self.dev)?, zero_row: Tensor::zeros((1, s.d_model), self.precision.act, &self.dev)? })
    }

    /// `(mels, frames)` f32 features to `(frames / 8, d_model)` encodings,
    /// in the activations' dtype.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        self.forward_layers(features, self.layers.len())
    }

    /// The output after the subsampling and the first `n` layers.
    pub fn forward_layers(&self, features: &Tensor, n: usize) -> Result<Tensor> {
        let mut x = self.sub.forward(features)?.to_dtype(self.precision.act)?;
        if let Some(s) = self.xscale {
            x = (x * s)?;
        }
        let frame = self.frame(x.dim(0)?)?;
        for (l, layer) in self.layers.iter().take(n).enumerate() {
            x = layer.forward(&x, &frame.pos.get(l)?, &frame)?;
        }
        Ok(x)
    }
}

/// The depthwise convolution of a conformer layer with its BatchNorm folded in.
fn load_depthwise(w: &mut Weights, prefix: &str, d: usize, kernel: usize, act: DType, dev: &Device) -> Result<DepthwiseTime> {
    let (taps, _) = w.f32_shaped(&format!("{prefix}.depthwise_conv.weight"), &[d, 1, kernel])?;
    let bias = w.optional_f32(&format!("{prefix}.depthwise_conv.bias"), &[d])?.unwrap_or_else(|| vec![0.0; d]);
    let get = |w: &mut Weights, n: &str| -> Result<Vec<f32>> { Ok(w.f32_shaped(&format!("{prefix}.batch_norm.{n}"), &[d])?.0) };
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
    Ok(DepthwiseTime { taps: upload(folded, &[1, kernel, d], act, dev)?, bias: upload(folded_bias, &[d], act, dev)?, kernel })
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
        let one = Tensor::from_vec(vec![7f32, 8.0], (2, 1, 1), &Device::Cpu).unwrap();
        assert_eq!(rel_shift(&one).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![7.0, 8.0]);
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

    /// A direct 3x3/2 convolution with padding 1: `k` is `(out, 9)`, taking
    /// input channel `ch` for output channel `ch` when depthwise, else channel 0.
    fn direct(x: &[f32], h: usize, w: usize, k: &[f32], bias: &[f32], depthwise: bool) -> Vec<Vec<Vec<f32>>> {
        let (ho, wo) = (Stride2Conv::out_len(h), Stride2Conv::out_len(w));
        (0..bias.len())
            .map(|ch| {
                (0..ho)
                    .map(|oy| {
                        (0..wo)
                            .map(|ox| {
                                let mut s = bias[ch];
                                for i in 0..3 {
                                    for j in 0..3 {
                                        let (y, xx) = (2 * oy as isize + i as isize - 1, 2 * ox as isize + j as isize - 1);
                                        if y >= 0 && (y as usize) < h && xx >= 0 && (xx as usize) < w {
                                            let src = if depthwise { ch } else { 0 };
                                            s += k[ch * 9 + i * 3 + j] * x[src * h * w + y as usize * w + xx as usize];
                                        }
                                    }
                                }
                                s
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn stride2_conv_matches_a_direct_convolution() {
        let dev = Device::Cpu;
        let (c, h, w) = (3usize, 7usize, 6usize);
        let x: Vec<f32> = (0..c * h * w).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.1).collect();
        let k: Vec<f32> = (0..c * 9).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.2).collect();
        let bias = [0.5f32, -0.25, 0.0];
        let (ho, wo) = (Stride2Conv::out_len(h), Stride2Conv::out_len(w));
        for depthwise in [true, false] {
            let weight = if depthwise { Tensor::from_vec(k.clone(), (c, 1, 9), &dev) } else { Tensor::from_vec(k.clone(), (c, 9), &dev) };
            let conv = Stride2Conv { w: weight.unwrap(), bias: Tensor::from_vec(bias.to_vec(), (c, 1), &dev).unwrap(), depthwise };
            let input = if depthwise { Tensor::from_vec(x.clone(), (c, h, w), &dev) } else { Tensor::from_vec(x[..h * w].to_vec(), (1, h, w), &dev) };
            let out = conv.forward(&input.unwrap()).unwrap();
            assert_eq!(out.dims(), &[c, ho, wo]);
            let out = out.to_vec3::<f32>().unwrap();
            let want = direct(&x, h, w, &k, &bias, depthwise);
            for ch in 0..c {
                for oy in 0..ho {
                    for ox in 0..wo {
                        assert!((out[ch][oy][ox] - want[ch][oy][ox]).abs() < 1e-5, "{depthwise} {ch} {oy} {ox}");
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_time_matches_a_direct_convolution() {
        let dev = Device::Cpu;
        let (t, c, kernel) = (6usize, 2usize, 3usize);
        let x: Vec<f32> = (0..t * c).map(|i| (i as f32 * 0.7).sin()).collect();
        let taps: Vec<f32> = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]; // (kernel, channels)
        let dw = DepthwiseTime { taps: Tensor::from_vec(taps.clone(), (1, kernel, c), &dev).unwrap(), bias: Tensor::from_vec(vec![1.0f32, -1.0], c, &dev).unwrap(), kernel };
        let frame = Frame { pos: Tensor::zeros(1, DType::F32, &dev).unwrap(), dw_index: DepthwiseTime::index(t, kernel, &dev).unwrap(), zero_row: Tensor::zeros((1, c), DType::F32, &dev).unwrap() };
        let out = dw.forward(&Tensor::from_vec(x.clone(), (t, c), &dev).unwrap(), &frame).unwrap().to_vec2::<f32>().unwrap();
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
