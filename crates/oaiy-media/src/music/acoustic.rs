//! The acoustic stage of MiniMax Music 3: per-frame language-model hidden
//! states to a 44.1 kHz stereo waveform, after the official diffusers
//! pipeline (`MiniMaxMusic3ConditionEncoder`, `MiniMaxMusic3Transformer1DModel`,
//! `MiniMaxMusic3Vocoder`, and the chunked flow-matching loop around them).
//!
//! * The condition encoder mixes each frame's eight 4096-wide hidden states,
//!   projects them to 2048 channels and stretches 25 Hz frames onto the
//!   86.13 Hz latent timeline (nearest neighbour).
//! * A 36-block flow-matching transformer denoises 128-channel latents in
//!   200-frame windows that overlap by 100 frames; each window's first 172
//!   latents are pinned to the previous window's, so the song stays continuous.
//! * The Flow-VAE decoder (DAC-style, weight-normalised convolutions and Snake
//!   activations) turns each window into audio; the windows are cropped and
//!   joined.
//!
//! The condition encoder and vocoder run in F32. The transformer runs in BF16
//! (as the official pipeline loads it) or F32, block by block from wherever
//! the memory budget put it.
use crate::ltx::store::Store;
use crate::residency::{Budget, Resident, Tiered};
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub const SAMPLE_RATE: usize = 44_100;
/// Waveform samples per latent (8 * 8 * 4 * 2).
pub const HOP: usize = 512;
/// Language-model frames per window, and the step between windows.
pub const CHUNK_FRAMES: usize = 200;
pub const CHUNK_HOP: usize = 100;
/// Latents a window shares with the one before it.
pub(crate) const OVERLAP: usize = 172;
/// Latents each side of a window that its neighbours' audio replaces.
pub(crate) const CROP_LEFT: usize = 86;
pub(crate) const CROP_RIGHT: usize = 344 - 86;
pub const LATENT_CHANNELS: usize = 128;

fn msg(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(s.into())
}

/// Latents for `frames` language-model frames: frames * 44100/24000 * 960/512,
/// truncated, evaluated in the reference's order.
pub fn latent_len(frames: usize) -> usize {
    ((frames as f64 * 44100. / 24000. * 960. / 512.) as usize).max(1)
}

/// Where each window starts, in frames.
pub fn chunk_starts(frames: usize) -> Vec<usize> {
    if frames <= CHUNK_FRAMES {
        vec![0]
    } else {
        (0..frames - CHUNK_HOP).step_by(CHUNK_HOP).collect()
    }
}

/// The flow-matching times for `steps` Euler steps, plus the final 1.0:
/// the scheduler's inverted sigmas (`1 - linspace(1, 1/steps, steps)` in F32).
pub fn times(steps: usize) -> Vec<f32> {
    let mut t: Vec<f32> = (0..steps)
        .map(|i| {
            let sigma = if steps == 1 { 1. } else { 1. + i as f64 * (1. / steps as f64 - 1.) / (steps - 1) as f64 };
            1f32 - sigma as f32
        })
        .collect();
    t.push(1.);
    t
}

/// Conv1d with a (out, in, k) F32 weight and optional bias, zero padding.
fn conv(x: &Tensor, w: &Tensor, b: Option<&Tensor>, padding: usize, dilation: usize) -> Result<Tensor> {
    let y = x.conv1d(w, padding, 1, dilation, 1)?;
    match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
        None => Ok(y),
    }
}

/// `g * v / |v|`, the norm taken over every axis but the first.
fn weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    v.broadcast_mul(&g.broadcast_div(&norm)?)
}

/// The Flow-VAE decoder: (B, 128, L) latents to (B, 2, L * 512) audio. Each
/// stereo channel is decoded from its own 64 latent channels.
pub struct Vocoder {
    pub(crate) w: HashMap<String, Tensor>,
    pub(crate) strides: Vec<usize>,
}

impl Vocoder {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let names: Vec<String> = store.index.names().map(str::to_owned).collect();
        let mut raw = HashMap::new();
        for k in names {
            raw.insert(k.clone(), store.tensor_f32(&k, dev)?);
        }
        // Weight norm folded once, in F32.
        let mut w = HashMap::new();
        for (k, t) in &raw {
            if let Some(base) = k.strip_suffix(".weight_v") {
                let g = raw.get(&format!("{base}.weight_g")).ok_or_else(|| msg(format!("vocoder: {base} lacks weight_g")))?;
                w.insert(format!("{base}.weight"), weight_norm(g, t)?);
            } else if !k.ends_with(".weight_g") {
                w.insert(k.clone(), t.clone());
            }
        }
        let config = read_config(&dir.join("config.json"))?;
        let strides = config
            .get("upsampling_ratios")
            .and_then(oaiy_engine::json::Json::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_i64().map(|v| v as usize)).collect())
            .unwrap_or_else(|| vec![8, 8, 4, 2]);
        Ok(Self { w, strides })
    }

    fn get(&self, k: &str) -> Result<&Tensor> {
        self.w.get(k).ok_or_else(|| msg(format!("missing vocoder tensor {k}")))
    }

    fn conv(&self, x: &Tensor, p: &str, padding: usize, dilation: usize) -> Result<Tensor> {
        conv(x, self.get(&format!("{p}.weight"))?, self.w.get(&format!("{p}.bias")), padding, dilation)
    }

    /// `x + sin(alpha x)^2 / (alpha + 1e-9)`.
    fn snake(&self, x: &Tensor, p: &str) -> Result<Tensor> {
        let a = self.get(&format!("{p}.alpha"))?;
        let s = x.broadcast_mul(a)?.sin()?.sqr()?;
        x + s.broadcast_mul(&(a + 1e-9)?.recip()?)?
    }

    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let (b, c, l) = latents.dims3()?;
        let mut h = latents.reshape((b * 2, c / 2, l))?;
        h = self.conv(&h, "dec_in_proj", 0, 1)?;
        h = self.conv(&h, "conv_in", 3, 1)?;
        for (i, &s) in self.strides.iter().enumerate() {
            let p = format!("blocks.{i}");
            h = self.snake(&h, &format!("{p}.snake1"))?;
            let w = self.get(&format!("{p}.conv_t1.weight"))?;
            let bias = self.get(&format!("{p}.conv_t1.bias"))?;
            h = h.conv_transpose1d(w, s.div_ceil(2), 0, s, 1, 1)?.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?)?;
            for (u, d) in [(1, 1), (2, 3), (3, 9)] {
                let r = format!("{p}.res_unit{u}");
                let y = self.snake(&h, &format!("{r}.snake1"))?;
                let y = self.conv(&y, &format!("{r}.conv1"), 3 * d, d)?;
                let y = self.snake(&y, &format!("{r}.snake2"))?;
                let y = self.conv(&y, &format!("{r}.conv2"), 0, 1)?;
                h = (h + y)?;
            }
        }
        h = self.snake(&h, "snake_out")?;
        let wave = self.conv(&h, "conv_out", 3, 1)?.tanh()?;
        wave.reshape((b, 2, ()))
    }
}

pub(crate) fn read_config(path: &Path) -> Result<oaiy_engine::json::Json> {
    let bytes = std::fs::read(path).map_err(|e| msg(format!("{}: {e}", path.display())))?;
    oaiy_engine::json::Json::parse(&bytes).map_err(candle_core::Error::wrap)
}

/// Mixes the eight hidden states of each frame and resamples them onto the
/// latent timeline: (1, F, 8 * 4096) -> (1, latent_len(F), 2048), F32.
pub struct ConditionEncoder {
    /// softmax(layer logits) * layer scale, one weight per hidden state.
    mix: Vec<f32>,
    width: usize,
    proj_w: Tensor,
    proj_b: Tensor,
}

impl ConditionEncoder {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let logits = store.tensor_f32("layer_weight_logits", &Device::Cpu)?.to_vec1::<f32>()?;
        let scale = store.tensor_f32("layer_scale", &Device::Cpu)?.to_vec1::<f32>()?[0];
        let proj_w = store.tensor_f32("proj.weight", dev)?;
        // softmax in F32, then the scale applied to the mixed state (as two
        // separate roundings in the reference; the difference is below F32 noise).
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        Ok(Self {
            mix: e.iter().map(|v| v / sum * scale).collect(),
            width: proj_w.dim(1)?,
            proj_w,
            proj_b: store.tensor_f32("proj.bias", dev)?,
        })
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let (_, frames, _) = hidden.dims3()?;
        let hidden = hidden.to_dtype(DType::F32)?;
        let mut mixed: Option<Tensor> = None;
        for (i, &w) in self.mix.iter().enumerate() {
            let part = (hidden.narrow(2, i * self.width, self.width)? * w as f64)?;
            mixed = Some(match mixed {
                None => part,
                Some(m) => (m + part)?,
            });
        }
        let mixed = mixed.ok_or_else(|| msg("condition encoder has no layers"))?.transpose(1, 2)?.contiguous()?;
        let y = conv(&mixed, &self.proj_w, Some(&self.proj_b), 1, 1)?;
        // PyTorch "nearest": source = floor(dst * (in / out)) in F32.
        let out = latent_len(frames);
        let scale = frames as f32 / out as f32;
        let idx: Vec<u32> = (0..out).map(|i| ((i as f32 * scale).floor() as usize).min(frames - 1) as u32).collect();
        let idx = Tensor::from_vec(idx, out, y.device())?;
        y.index_select(&idx, 2)?.transpose(1, 2)?.contiguous()
    }
}

/// One transformer block's weights.
pub struct Block {
    norm1: (Tensor, Tensor),
    qkv: Tensor,
    out: Tensor,
    norm2: (Tensor, Tensor),
    ff_in: (Tensor, Tensor),
    ff_out: (Tensor, Tensor),
}

fn bytes(t: &Tensor) -> u64 {
    (t.elem_count() * t.dtype().size_in_bytes()) as u64
}

impl Resident for Block {
    fn bytes(&self) -> u64 {
        [&self.norm1.0, &self.norm1.1, &self.qkv, &self.out, &self.norm2.0, &self.norm2.1, &self.ff_in.0, &self.ff_in.1, &self.ff_out.0, &self.ff_out.1]
            .iter()
            .map(|t| bytes(t))
            .sum()
    }
    fn to_device(&self, dev: &Device) -> Result<Self> {
        let m = |t: &Tensor| t.to_device(dev);
        Ok(Self {
            norm1: (m(&self.norm1.0)?, m(&self.norm1.1)?),
            qkv: m(&self.qkv)?,
            out: m(&self.out)?,
            norm2: (m(&self.norm2.0)?, m(&self.norm2.1)?),
            ff_in: (m(&self.ff_in.0)?, m(&self.ff_in.1)?),
            ff_out: (m(&self.ff_out.0)?, m(&self.ff_out.1)?),
        })
    }
}

/// Over the last axis, as one 2-D matmul.
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    let mut dims = x.dims().to_vec();
    let n = *dims.last().ok_or_else(|| msg("linear input has no axes"))?;
    let y = x.reshape((x.elem_count() / n, n))?.matmul(&w.t()?)?;
    let y = match b {
        Some(b) => y.broadcast_add(b)?,
        None => y,
    };
    *dims.last_mut().unwrap() = y.dim(1)?;
    y.reshape(dims)
}

/// The flow-matching transformer.
pub struct Transformer {
    store: Store,
    blocks: Tiered<Block>,
    dtype: DType,
    heads: usize,
    head_dim: usize,
    rotary: usize,
    fourier: Tensor,
    time1: (Tensor, Tensor),
    time2: (Tensor, Tensor),
    preprocess: Tensor,
    proj_in: Tensor,
    proj_out: Tensor,
    postprocess: Tensor,
    dev: Device,
}

impl Transformer {
    /// `dtype`: BF16 (the official configuration) or F32.
    pub fn load(dir: &Path, dtype: DType, budget: &Budget, dev: &Device, progress: impl FnMut(usize)) -> Result<Self> {
        let config = read_config(&dir.join("config.json"))?;
        let int = |k: &str, d: usize| config.get(k).and_then(oaiy_engine::json::Json::as_i64).map_or(d, |v| v as usize);
        let (layers, heads, head_dim, rotary) = (int("num_layers", 36), int("num_attention_heads", 32), int("attention_head_dim", 64), int("rotary_dim", 32));
        let mut store = Store::open(dir, 0)?;
        let f32 = |s: &mut Store, k: &str| s.tensor_f32(k, dev);
        let conv_as_linear = |t: Tensor| -> Result<Tensor> { t.squeeze(2)?.to_dtype(dtype) };
        let fourier = f32(&mut store, "time_proj.weight")?;
        let time1 = (f32(&mut store, "time_embed.linear_1.weight")?, f32(&mut store, "time_embed.linear_1.bias")?);
        let time2 = (f32(&mut store, "time_embed.linear_2.weight")?, f32(&mut store, "time_embed.linear_2.bias")?);
        let preprocess = conv_as_linear(f32(&mut store, "preprocess_conv.weight")?)?;
        let proj_in = f32(&mut store, "proj_in.weight")?.to_dtype(dtype)?;
        let proj_out = f32(&mut store, "proj_out.weight")?.to_dtype(dtype)?;
        let postprocess = conv_as_linear(f32(&mut store, "postprocess_conv.weight")?)?;
        let blocks = Tiered::load(layers, budget, dev, |i| load_block(&mut store, i, dtype, dev), progress)?;
        Ok(Self { store, blocks, dtype, heads, head_dim, rotary, fourier, time1, time2, preprocess, proj_in, proj_out, postprocess, dev: dev.clone() })
    }

    pub fn report(&self) -> oaiy_engine::json::Json {
        self.blocks.report()
    }

    /// Velocity for `x` (B, 128, L) at flow time `t` (one per batch row),
    /// conditioned on `cond` (B, L, 2048).
    pub fn forward(&mut self, x: &Tensor, t: &[f32], cond: &Tensor) -> Result<Tensor> {
        let (b, c, l) = x.dims3()?;
        let dt = self.dtype;
        let x = x.to_dtype(dt)?;
        // [latent, zeros, condition] along channels, as (B, L, 2304).
        let h = Tensor::cat(&[&x.transpose(1, 2)?, &Tensor::zeros((b, l, c), dt, &self.dev)?, &cond.to_dtype(dt)?], 2)?.contiguous()?;
        let h = (linear(&h, &self.preprocess, None)? + &h)?;
        let h = linear(&h, &self.proj_in, None)?;
        // Random Fourier time features in F32, then the timestep MLP.
        let tt = Tensor::from_slice(t, (b, 1), &self.dev)?;
        let angles = (tt.matmul(&self.fourier.t()?)? * (2. * std::f64::consts::PI))?;
        let feats = Tensor::cat(&[angles.cos()?, angles.sin()?], 1)?;
        let temb = linear(&linear(&feats, &self.time1.0, Some(&self.time1.1))?.silu()?, &self.time2.0, Some(&self.time2.1))?;
        // The time embedding is token 0.
        let mut h = Tensor::cat(&[&temb.to_dtype(dt)?.unsqueeze(1)?, &h], 1)?;
        let s = l + 1;
        let (cos, sin) = self.rope(s)?;
        let (heads, hd, rot) = (self.heads, self.head_dim, self.rotary);
        let dev = self.dev.clone();
        let store = &mut self.store;
        for i in 0..self.blocks.counts().0 + self.blocks.counts().1 + self.blocks.counts().2 {
            h = self.blocks.with(i, |i| load_block(store, i, dt, &dev), |blk| block(blk, &h, &cos, &sin, heads, hd, rot))?;
        }
        let h = linear(&h.narrow(1, 1, l)?, &self.proj_out, None)?;
        let y = (linear(&h, &self.postprocess, None)? + &h)?;
        y.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()
    }

    /// Partial-rotary tables for `s` positions: (s, rotary/2), theta 1e4.
    fn rope(&self, s: usize) -> Result<(Tensor, Tensor)> {
        let half = self.rotary / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1. / 10000f32.powf((2 * i) as f32 / self.rotary as f32)).collect();
        let f: Vec<f32> = (0..s).flat_map(|p| inv.iter().map(move |v| p as f32 * v)).collect();
        let f = Tensor::from_vec(f, (s, half), &self.dev)?;
        Ok((f.cos()?.to_dtype(self.dtype)?, f.sin()?.to_dtype(self.dtype)?))
    }
}

fn load_block(store: &mut Store, i: usize, dtype: DType, dev: &Device) -> Result<Block> {
    let p = format!("transformer_blocks.{i}");
    let mut get = |k: &str| -> Result<Tensor> { store.tensor_f32(&format!("{p}.{k}"), dev)?.to_dtype(dtype) };
    let qkv = Tensor::cat(&[get("attn.to_q.weight")?, get("attn.to_k.weight")?, get("attn.to_v.weight")?], 0)?;
    Ok(Block {
        norm1: (get("norm1.weight")?, get("norm1.bias")?),
        qkv,
        out: get("attn.to_out.0.weight")?,
        norm2: (get("norm2.weight")?, get("norm2.bias")?),
        ff_in: (get("ff_in.weight")?, get("ff_in.bias")?),
        ff_out: (get("ff_out.weight")?, get("ff_out.bias")?),
    })
}

/// Rotate-half RoPE on the first `rot` dims of each head; the rest pass.
fn partial_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rot: usize) -> Result<Tensor> {
    let d = x.dim(3)?;
    let r = candle_nn::rotary_emb::rope_thd(&x.narrow(3, 0, rot)?.contiguous()?, cos, sin)?;
    Tensor::cat(&[&r, &x.narrow(3, rot, d - rot)?], 3)
}

fn block(b: &Block, h: &Tensor, cos: &Tensor, sin: &Tensor, heads: usize, hd: usize, rot: usize) -> Result<Tensor> {
    let (bs, s, dim) = h.dims3()?;
    let n = candle_nn::ops::layer_norm(&h.contiguous()?, &b.norm1.0, &b.norm1.1, 1e-5)?;
    let qkv = linear(&n, &b.qkv, None)?;
    let q = partial_rope(&qkv.narrow(2, 0, dim)?.reshape((bs, s, heads, hd))?, cos, sin, rot)?;
    let k = partial_rope(&qkv.narrow(2, dim, dim)?.reshape((bs, s, heads, hd))?, cos, sin, rot)?;
    let v = qkv.narrow(2, 2 * dim, dim)?.reshape((bs, s, heads, hd))?;
    let a = attention(&q, &k, &v)?.reshape((bs, s, dim))?;
    let h = (h + linear(&a, &b.out, None)?)?;
    let n = candle_nn::ops::layer_norm(&h.contiguous()?, &b.norm2.0, &b.norm2.1, 1e-5)?;
    let f = linear(&n, &b.ff_in.0, Some(&b.ff_in.1))?;
    let inner = f.dim(2)? / 2;
    // The first half is the value, the second the SiLU gate.
    let g = (f.narrow(2, 0, inner)? * f.narrow(2, inner, inner)?.silu()?)?;
    h + linear(&g, &b.ff_out.0, Some(&b.ff_out.1))?
}

/// Full (bidirectional) attention over (B, S, H, D).
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let hd = q.dim(3)?;
    let dtype = q.dtype();
    let t = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous() };
    let (q, k, v) = (t(q)?, t(k)?, t(v)?);
    let scores = (q.matmul(&k.t()?)? / (hd as f64).sqrt())?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v)?.transpose(1, 2)?.to_dtype(dtype)?.contiguous()
}

/// Everything after the language model: condition encoder, transformer and
/// vocoder, with the window loop that joins them.
pub struct Acoustic {
    pub condition: ConditionEncoder,
    pub transformer: Transformer,
    pub vocoder: Vocoder,
    pub steps: usize,
    pub guidance: f64,
}

impl Acoustic {
    /// Denoise every window of `hidden` (1, F, 32768) and decode it: returns
    /// the stereo waveform (2, samples) in [-1, 1]. `noise(k, len)` gives
    /// window `k`'s starting noise, (1, 128, len). `progress(done, total)`
    /// counts Euler steps.
    pub fn generate(&mut self, hidden: &Tensor, mut noise: impl FnMut(usize, usize) -> Result<Tensor>, mut progress: impl FnMut(usize, usize)) -> Result<Tensor> {
        let frames = hidden.dim(1)?;
        let starts = chunk_starts(frames);
        let times = times(self.steps);
        let total = starts.len() * self.steps;
        let mut prev: Option<(Tensor, Tensor)> = None;
        let mut chunks = Vec::with_capacity(starts.len());
        let dev = self.transformer.dev.clone();
        for (k, &start) in starts.iter().enumerate() {
            let end = (start + CHUNK_FRAMES).min(frames);
            let mut cond = self.condition.forward(&hidden.narrow(1, start, end - start)?.to_device(&dev)?)?;
            let len = cond.dim(1)?;
            let mut overlap = 0;
            if let Some((pl, pc)) = &prev {
                overlap = pl.dim(2)?.min(len);
                cond = Tensor::cat(&[&pc.narrow(1, 0, overlap)?, &cond.narrow(1, overlap, len - overlap)?], 1)?;
            }
            let init = noise(k, len)?.to_device(&dev)?.to_dtype(DType::F32)?;
            let prompt = if overlap > 0 { Some(init.narrow(2, 0, overlap)?) } else { None };
            let mut x = init;
            let both = Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?;
            for i in 0..self.steps {
                let t = times[i];
                if let (Some(p), Some((pl, _))) = (&prompt, &prev) {
                    // The shared latents follow the previous window's path.
                    let pinned = ((p * (1. - (1. - 1e-6) * t as f64))? + (pl.narrow(2, 0, overlap)? * t as f64)?)?;
                    x = Tensor::cat(&[&pinned, &x.narrow(2, overlap, len - overlap)?], 2)?;
                }
                let v = self.transformer.forward(&Tensor::cat(&[&x, &x], 0)?, &[t, t], &both)?;
                let (vc, vu) = (v.narrow(0, 0, 1)?, v.narrow(0, 1, 1)?);
                let guided = (&vu + ((vc - &vu)? * self.guidance)?)?;
                x = (x + (guided * (times[i + 1] - t) as f64)?)?;
                progress(k * self.steps + i + 1, total);
            }
            if let Some((pl, _)) = &prev {
                x = Tensor::cat(&[&pl.narrow(2, 0, overlap)?, &x.narrow(2, overlap, len - overlap)?], 2)?;
            }
            let os = len.saturating_sub(2 * OVERLAP);
            let oe = os.max(len.saturating_sub(OVERLAP));
            prev = Some((x.narrow(2, os, oe - os)?, cond.narrow(1, os, oe - os)?));
            chunks.push(x);
        }
        let n = chunks.len();
        let mut waves = Vec::with_capacity(n);
        for (i, latents) in chunks.iter().enumerate() {
            let w = self.vocoder.forward(latents)?.squeeze(0)?;
            let samples = w.dim(1)?;
            let left = if i == 0 { 0 } else { CROP_LEFT * HOP };
            let right = if i == n - 1 { 0 } else { CROP_RIGHT * HOP };
            waves.push(w.narrow(1, left, samples - left - right)?);
        }
        Tensor::cat(&waves, 1)?.clamp(-1f32, 1f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_and_times_follow_the_reference() {
        assert_eq!(latent_len(200), 689);
        assert_eq!(latent_len(5), 17);
        assert_eq!(chunk_starts(200), vec![0]);
        assert_eq!(chunk_starts(201), vec![0, 100]);
        assert_eq!(chunk_starts(260), vec![0, 100]);
        assert_eq!(chunk_starts(301), vec![0, 100, 200]);
        let t = times(30);
        assert_eq!(t.len(), 31);
        assert_eq!(t[0], 0.);
        assert!((t[1] - 0.033_333_36).abs() < 1e-7, "{}", t[1]);
        assert!((t[29] - 0.966_666_64).abs() < 1e-7, "{}", t[29]);
        assert_eq!(t[30], 1.);
    }
}

/// Opt-in checks against the official pipeline's activations (strict F32,
/// TF32 off): `OAIY_MUSIC_GOLDEN` (the dump folder) and `OAIY_MUSIC_MODEL`
/// (the MiniMax-Music3 folder).
#[cfg(test)]
pub(crate) mod golden {
    use super::*;
    use std::path::PathBuf;

    pub fn dirs() -> Option<(PathBuf, PathBuf)> {
        Some((PathBuf::from(std::env::var("OAIY_MUSIC_GOLDEN").ok()?), PathBuf::from(std::env::var("OAIY_MUSIC_MODEL").ok()?)))
    }
    pub fn shape(dir: &Path, name: &str) -> Vec<usize> {
        let s = oaiy_engine::json::Json::parse(&std::fs::read(dir.join("shapes.json")).unwrap()).unwrap();
        s.get(name).and_then(oaiy_engine::json::Json::as_array).unwrap().iter().map(|v| v.as_i64().unwrap() as usize).collect()
    }
    pub fn load(dir: &Path, name: &str, dev: &Device) -> Tensor {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        let v: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        Tensor::from_vec(v, shape(dir, name), dev).unwrap()
    }
    pub fn ints(dir: &Path, name: &str) -> Vec<i32> {
        std::fs::read(dir.join(name)).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
    /// RMS of the difference over RMS of the reference.
    pub fn relative(actual: &Tensor, expected: &Tensor) -> f32 {
        let d = (actual.to_dtype(DType::F32).unwrap() - expected).unwrap().sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        let e = expected.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap();
        (d / e).sqrt()
    }
    pub fn device() -> Device {
        Device::cuda_if_available(0).unwrap()
    }

    #[test]
    fn vocoder_and_condition_encoder_match_reference() -> Result<()> {
        let Some((g, m)) = dirs() else { return Ok(()) };
        let dev = device();
        let voc = Vocoder::load(&m.join("vocoder"), &dev)?;
        for i in 0..2 {
            let wave = voc.forward(&load(&g, &format!("latents{i}.f32"), &dev))?;
            let err = relative(&wave, &load(&g, &format!("wave{i}.f32"), &dev));
            println!("vocoder chunk {i}: {err:.2e}");
            assert!(err < 1e-5, "vocoder chunk {i}: {err}");
        }
        let cond = ConditionEncoder::load(&m.join("condition_encoder"), &dev)?;
        for i in 0..2 {
            let y = cond.forward(&load(&g, &format!("cond_in{i}.f32"), &dev))?;
            let err = relative(&y, &load(&g, &format!("cond_out{i}.f32"), &dev));
            println!("condition chunk {i}: {err:.2e}");
            assert!(err < 1e-5, "condition chunk {i}: {err}");
        }
        Ok(())
    }

    #[test]
    fn transformer_matches_reference() -> Result<()> {
        let Some((g, m)) = dirs() else { return Ok(()) };
        let dev = device();
        for (dtype, limit) in [(DType::F32, 1e-4), (DType::BF16, 2e-2)] {
            let mut dit = Transformer::load(&m.join("transformer"), dtype, &Budget::default(), &dev, |_| {})?;
            for i in 0..2 {
                let t = load(&g, &format!("dit_t{i}.f32"), &Device::Cpu).to_vec1::<f32>()?;
                let v = dit.forward(&load(&g, &format!("dit_x{i}.f32"), &dev), &t, &load(&g, &format!("dit_c{i}.f32"), &dev))?;
                let err = relative(&v, &load(&g, &format!("dit_v{i}.f32"), &dev));
                println!("transformer {dtype:?} pass {i}: {err:.2e}");
                assert!(err < limit, "transformer {dtype:?} pass {i}: {err}");
            }
        }
        Ok(())
    }

    #[test]
    fn windows_join_like_the_reference() -> Result<()> {
        let Some((g, m)) = dirs() else { return Ok(()) };
        let dev = device();
        // The run's frame hiddens are both windows' inputs, less the overlap.
        let (a, b) = (load(&g, "cond_in0.f32", &Device::Cpu), load(&g, "cond_in1.f32", &Device::Cpu));
        let hidden = Tensor::cat(&[&a, &b.narrow(1, CHUNK_HOP, b.dim(1)? - CHUNK_HOP)?], 1)?;
        let noise = [load(&g, "noise0.f32", &dev), load(&g, "noise1.f32", &dev)];
        let expected = load(&g, "audio.f32", &dev);
        for (dtype, limit) in [(DType::F32, 1e-3), (DType::BF16, 5e-2)] {
            let mut acoustic = Acoustic {
                condition: ConditionEncoder::load(&m.join("condition_encoder"), &dev)?,
                transformer: Transformer::load(&m.join("transformer"), dtype, &Budget::default(), &dev, |_| {})?,
                vocoder: Vocoder::load(&m.join("vocoder"), &dev)?,
                steps: 30,
                guidance: 1.7,
            };
            let started = std::time::Instant::now();
            let audio = acoustic.generate(&hidden, |k, len| { assert_eq!(noise[k].dim(2)?, len); Ok(noise[k].clone()) }, |_, _| {})?;
            assert_eq!(audio.dims(), expected.dims());
            let err = relative(&audio, &expected);
            println!("acoustic {dtype:?}: {err:.2e} in {:.1?}", started.elapsed());
            assert!(err < limit, "acoustic {dtype:?}: {err}");
        }
        Ok(())
    }
}
