//! The DAC decoder MOSS-SoundEffect v2.0 uses as its VAE (continuous, 128 latent
//! channels, 48 kHz): `post_quant_conv`, a 7-tap conv up to `decoder_dim`, then per
//! rate a Snake and a transposed conv (upsampling by the rate, halving the channels)
//! and three dilated residual units, and a last Snake, 7-tap conv to one channel and
//! tanh. Weight-normed convs are folded once. Read from the release's `.pth`
//! (interpreted, never run) or a safetensors copy of it; decoded in F32.
use super::pth::{Pth, Value};
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

fn bad(s: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(format!("DAC: {}", s.into()))
}

pub struct DacConfig {
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    /// Samples per latent frame: the product of the encoder's rates.
    pub hop: usize,
    pub sample_rate: usize,
}

/// A conv's folded weight and bias.
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
}

struct Residual {
    snake1: Tensor,
    conv1: Conv,
    dilation: usize,
    snake2: Tensor,
    conv2: Conv,
}

struct Up {
    snake: Tensor,
    conv: Conv,
    stride: usize,
    residuals: Vec<Residual>,
}

pub struct Dac {
    pub cfg: DacConfig,
    post_quant: Option<Conv>,
    first: Conv,
    ups: Vec<Up>,
    last_snake: Tensor,
    last: Conv,
}

/// `g * v / |v|`, the norm over every axis but the first.
fn weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    v.broadcast_mul(&g.broadcast_div(&norm)?)
}

/// `x + sin(alpha x)^2 / (alpha + 1e-9)`, per channel.
fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let s = x.broadcast_mul(alpha)?.sin()?.sqr()?;
    x + s.broadcast_div(&(alpha + 1e-9)?)?
}

/// The decoder's tensors, read from the checkpoint as they are asked for.
struct Loader<'a> {
    pth: Pth,
    sd: HashMap<String, Value>,
    dev: &'a Device,
}
impl Loader<'_> {
    fn has(&self, k: &str) -> bool {
        self.sd.contains_key(k)
    }
    fn get(&mut self, k: &str) -> Result<Tensor> {
        let v = self.sd.get(k).ok_or_else(|| bad(format!("no {k}")))?.clone();
        self.pth.tensor(&v, self.dev)
    }
    /// A conv's weight (weight norm folded, old or new naming) and bias.
    fn conv(&mut self, prefix: &str) -> Result<Conv> {
        let w = if self.has(&format!("{prefix}.weight_g")) {
            weight_norm(&self.get(&format!("{prefix}.weight_g"))?, &self.get(&format!("{prefix}.weight_v"))?)?
        } else if self.has(&format!("{prefix}.parametrizations.weight.original0")) {
            weight_norm(&self.get(&format!("{prefix}.parametrizations.weight.original0"))?, &self.get(&format!("{prefix}.parametrizations.weight.original1"))?)?
        } else {
            self.get(&format!("{prefix}.weight"))?
        };
        let b = if self.has(&format!("{prefix}.bias")) { Some(self.get(&format!("{prefix}.bias"))?) } else { None };
        Ok(Conv { w, b })
    }
}

fn add_bias(y: Tensor, b: &Option<Tensor>) -> Result<Tensor> {
    match b {
        Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?),
        None => Ok(y),
    }
}

impl Dac {
    /// `path`: `vae_128d_48k.pth` (or a safetensors file with the same names and a
    /// `config.json` beside it holding the constructor's arguments).
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let pth = Pth::open(path)?;
        let root = pth.root.clone();
        let kwargs = root.get("metadata").and_then(|m| m.get("kwargs")).ok_or_else(|| bad("no metadata.kwargs"))?;
        let int = |k: &str| kwargs.get(k).and_then(Value::as_i64).map(|v| v as usize).ok_or_else(|| bad(format!("no {k}")));
        let ints = |k: &str| kwargs.get(k).and_then(Value::ints).map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>()).ok_or_else(|| bad(format!("no {k}")));
        let cfg = DacConfig {
            latent_dim: int("latent_dim")?,
            decoder_dim: int("decoder_dim")?,
            decoder_rates: ints("decoder_rates")?,
            hop: ints("encoder_rates")?.iter().product(),
            sample_rate: int("sample_rate")?,
        };
        let continuous = kwargs.get("continuous").and_then(Value::as_bool).unwrap_or(false);
        let Some(Value::Dict(items)) = root.get("state_dict") else { return Err(bad("no state_dict")) };
        let mut sd: HashMap<String, Value> = HashMap::new();
        for (k, v) in items {
            if let Value::Str(k) = k {
                if k.starts_with("decoder.") || k.starts_with("post_quant_conv.") {
                    sd.insert(k.clone(), v.clone());
                }
            }
        }
        if sd.is_empty() {
            return Err(bad("the state dict has no decoder"));
        }
        let mut l = Loader { pth, sd, dev };
        let post_quant = if continuous { Some(l.conv("post_quant_conv")?) } else { None };
        let first = l.conv("decoder.model.0")?;
        let mut ups = Vec::new();
        for (i, &stride) in cfg.decoder_rates.iter().enumerate() {
            let p = format!("decoder.model.{}.block", i + 1);
            let mut residuals = Vec::new();
            for (r, dilation) in [1, 3, 9].into_iter().enumerate() {
                let q = format!("{p}.{}.block", r + 2);
                residuals.push(Residual {
                    snake1: l.get(&format!("{q}.0.alpha"))?,
                    conv1: l.conv(&format!("{q}.1"))?,
                    dilation,
                    snake2: l.get(&format!("{q}.2.alpha"))?,
                    conv2: l.conv(&format!("{q}.3"))?,
                });
            }
            ups.push(Up { snake: l.get(&format!("{p}.0.alpha"))?, conv: l.conv(&format!("{p}.1"))?, stride, residuals });
        }
        let n = cfg.decoder_rates.len();
        let last_snake = l.get(&format!("decoder.model.{}.alpha", n + 1))?;
        let last = l.conv(&format!("decoder.model.{}", n + 2))?;
        Ok(Self { cfg, post_quant, first, ups, last_snake, last })
    }

    /// Latents (1, latent_dim, L), F32, to audio (1, 1, L * hop).
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = z.clone();
        if let Some(c) = &self.post_quant {
            x = add_bias(x.conv1d(&c.w, 0, 1, 1, 1)?, &c.b)?;
        }
        x = add_bias(x.conv1d(&self.first.w, 3, 1, 1, 1)?, &self.first.b)?;
        for up in &self.ups {
            let s = up.stride;
            x = snake(&x, &up.snake)?;
            x = add_bias(x.conv_transpose1d(&up.conv.w, s.div_ceil(2), s % 2, s, 1, 1)?, &up.conv.b)?;
            for r in &up.residuals {
                let y = snake(&x, &r.snake1)?;
                let y = add_bias(y.conv1d(&r.conv1.w, 3 * r.dilation, 1, r.dilation, 1)?, &r.conv1.b)?;
                let y = snake(&y, &r.snake2)?;
                let y = add_bias(y.conv1d(&r.conv2.w, 0, 1, 1, 1)?, &r.conv2.b)?;
                x = (x + y)?;
            }
        }
        let x = snake(&x, &self.last_snake)?;
        add_bias(x.conv1d(&self.last.w, 3, 1, 1, 1)?, &self.last.b)?.tanh()
    }
}
