//! Qwen Image 2.1's VAE decoder on WebGPU, as [`crate::vae::Vae::decode`] computes it (one frame): each pixel's
//! channels a row (the tensor cores' tokens), the convolutions on the tensor cores (an implicit im2col), the weights
//! f16 (the checkpoint's f32 rounded to the nearest), each convolution's input scaled by a power of two into f16's
//! range (its residual stream reaches some 230,000). Each residual block, the attention and each upsampling its own
//! recording, its vectors let go once it has run (a 1024x1024 picture's last blocks are 1.2 GB a vector).
use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::collections::HashMap;
use std::path::Path;

/// The norms' epsilon: Wan's divides by the channels' L2 norm clamped at 1e-12 (an RMS norm with none).
const EPS: f32 = 1e-12;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A convolution's weights on the GPU as [`DeviceChain::conv_weights`] packs them (`k` by `k`, 1 or 3), and its bias.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
    k: usize,
}

pub struct WgpuVae {
    gpu: ggml_rs_wgpu::WgpuBackend,
    convs: HashMap<String, Conv>,
    gammas: HashMap<String, DeviceVec>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl WgpuVae {
    /// The decoder of the VAE under `root` (`vae/`, its config's latents' mean and deviation) on GPU `device` (as
    /// CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(root: &Path, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(&root.join("vae"))?;
        let mut convs = HashMap::new();
        let mut gammas = HashMap::new();
        for name in w.names() {
            if !(name.starts_with("decoder.") || name.starts_with("post_quant_conv.")) || name.contains("time_conv") {
                continue;
            }
            if let Some(base) = name.strip_suffix(".gamma") {
                let values = w.tensor(&name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let v = gpu.vec(values.len());
                gpu.upload(&v, &values);
                gammas.insert(base.to_owned(), v);
                continue;
            }
            let Some(base) = name.strip_suffix(".weight") else { continue };
            let t = w.tensor(&name, &Device::Cpu, DType::F32)?;
            let &[cout, cin, kh, kw] = t.dims() else { continue };
            let values = t.flatten_all()?.to_vec1::<f32>()?;
            if kh != kw || !matches!(kh, 1 | 3) {
                candle_core::bail!("{name}: a {kh}x{kw} convolution");
            }
            let wv = gpu.conv_weights(&values, cout, cin, kh).ok_or_else(|| err(format!("{name}: no tensor cores for its convolution, or a weight past f16's range")))?;
            let bias = w.tensor(&format!("{base}.bias"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let b = gpu.vec(bias.len());
            gpu.upload(&b, &bias);
            convs.insert(base.to_owned(), Conv { w: wv, b, cout, cin, k: kh });
        }
        let config = oaiy_engine::json::Json::parse(&std::fs::read(root.join("vae/config.json"))?).map_err(candle_core::Error::wrap)?;
        let list = |key: &str| -> Result<Vec<f32>> {
            config.get(key).and_then(|x| x.as_array()).ok_or_else(|| err(format!("missing {key}")))?.iter().map(|x| x.as_f64().map(|n| n as f32).ok_or_else(|| err(format!("invalid {key}")))).collect()
        };
        let (mean, std) = (list("latents_mean")?, list("latents_std")?);
        if mean.len() != 64 || std.len() != 64 {
            candle_core::bail!("the VAE's latents have {} channels, not 64", mean.len());
        }
        Ok(Self { gpu, convs, gammas, mean, std })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// Convolution `name` of an image of `h` by `w` pixels (`x` its rows of channels): its output and channels.
    fn conv(&self, rec: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, h: usize, w: usize) -> Result<(DeviceVec, usize)> {
        let c = self.convs.get(name).ok_or_else(|| err(format!("missing VAE convolution {name}")))?;
        let y = self.vec(h * w * c.cout);
        rec.conv_rows(&c.w, &c.b, c.cout, c.cin, c.k, x, h, w, &y);
        Ok((y, c.cout))
    }

    /// Norm `name` of `rows` pixels of `c` channels (each pixel's channels over their RMS, times the norm's gamma),
    /// and with `silu` through SiLU.
    fn norm(&self, rec: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, rows: usize, c: usize, silu: bool) -> Result<DeviceVec> {
        let g = self.gammas.get(name).ok_or_else(|| err(format!("missing VAE norm {name}")))?;
        let n = self.vec(rows * c);
        if silu {
            rec.rmsnorm_silu_rows(x, g, &n, rows, EPS);
        } else {
            rec.rmsnorm_rows(x, g, &n, rows, EPS);
        }
        Ok(n)
    }

    /// Residual block `p` of an image of `c` channels (its own recording): its output and channels.
    fn residual(&self, p: &str, x: &DeviceVec, h: usize, w: usize, c: usize) -> Result<(DeviceVec, usize)> {
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let shortcut = if self.convs.contains_key(&format!("{p}.conv_shortcut")) { self.conv(r, &format!("{p}.conv_shortcut"), x, h, w)?.0 } else { x.clone() };
        let t = self.norm(r, &format!("{p}.norm1"), x, h * w, c, true)?;
        let (h1, co) = self.conv(r, &format!("{p}.conv1"), &t, h, w)?;
        let t = self.norm(r, &format!("{p}.norm2"), &h1, h * w, co, true)?;
        let (h2, _) = self.conv(r, &format!("{p}.conv2"), &t, h, w)?;
        r.add(&h2, &shortcut);
        rec.finish();
        drop((shortcut, t, h1));
        self.gpu.settle();
        Ok((h2, co))
    }

    /// The middle block's attention: one head over every pixel, its channels the head.
    fn attention(&self, x: &DeviceVec, h: usize, w: usize, c: usize) -> Result<DeviceVec> {
        let p = "decoder.mid_block.attentions.0";
        let rows = h * w;
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let n = self.norm(r, &format!("{p}.norm"), x, rows, c, false)?;
        let (qkv, _) = self.conv(r, &format!("{p}.to_qkv"), &n, h, w)?;
        let (q, kv) = (self.vec(rows * c), self.vec(rows * 2 * c));
        r.copy_cols(&qkv, &q, rows, c, 3 * c, 0);
        r.copy_cols(&qkv, &kv, rows, 2 * c, 3 * c, c);
        let att = self.vec(self.gpu.attention_rows_full_out_len(rows, 1, c, rows));
        r.attention_rows_full(&q, &kv, &att, rows, 1, 1, c, rows, 1.0 / (c as f32).sqrt());
        let (proj, _) = self.conv(r, &format!("{p}.proj"), &att, h, w)?;
        r.add(&proj, x);
        rec.finish();
        drop((n, qkv, q, kv, att));
        self.gpu.settle();
        Ok(proj)
    }

    /// The picture of `latent` (`[1, h w, 64]`, the transformer's tokens): `[1, 4, 16 h, 16 w]` (RGBA in -1..1) on the
    /// CPU.
    pub fn decode(&self, latent: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        if latent.dims() != [1, h * w, 64] {
            candle_core::bail!("latent must be [1, {}, 64], not {:?}", h * w, latent.dims());
        }
        // out of the latent space: each channel times its deviation, plus its mean
        let lat = latent.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let values: Vec<f32> = lat.iter().enumerate().map(|(i, v)| v * self.std[i % 64] + self.mean[i % 64]).collect();
        let x0 = self.vec(values.len());
        self.gpu.upload(&x0, &values);
        let (mut h, mut w) = (h, w);
        let (mut x, mut c) = {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            let (x, _) = self.conv(r, "post_quant_conv", &x0, h, w)?;
            let out = self.conv(r, "decoder.conv_in", &x, h, w)?;
            rec.finish();
            out
        };
        (x, c) = self.residual("decoder.mid_block.resnets.0", &x, h, w, c)?;
        x = self.attention(&x, h, w, c)?;
        (x, c) = self.residual("decoder.mid_block.resnets.1", &x, h, w, c)?;
        for i in 0..5 {
            let (before, cin) = (x.clone(), c);
            for j in 0..3 {
                (x, c) = self.residual(&format!("decoder.up_blocks.{i}.resnets.{j}"), &x, h, w, c)?;
            }
            if i < 4 {
                let mut rec = self.gpu.begin();
                rec.keep_groups(false);
                let r = rec.as_mut();
                let up = self.vec(4 * h * w * c);
                r.upsample2x_rows(&x, &up, h, w, c);
                let (y, cout) = self.conv(r, &format!("decoder.up_blocks.{i}.upsampler.resample.1"), &up, 2 * h, 2 * w)?;
                // Wan's shortcut: the block's input repeated and shuffled into the pixels (the first frame's slot)
                r.shuffle_up_add_rows(&before, &y, h, w, cin, cout, if i < 3 { 2 } else { 1 });
                rec.finish();
                drop((up, before));
                self.gpu.settle();
                (x, c, h, w) = (y, cout, 2 * h, 2 * w);
            }
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let t = self.norm(r, "decoder.norm_out", &x, h * w, c, true)?;
        let (rgba, co) = self.conv(r, "decoder.conv_out", &t, h, w)?;
        r.read(&rgba);
        let pixels = rec.finish().pop().ok_or_else(|| err("the picture was not read"))?;
        Tensor::from_vec(pixels, (1, h, w, co), &Device::Cpu)?.permute((0, 3, 1, 2))?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WebGPU decoder gives the Candle one's picture (CPU, f32) from the published VAE (`OAIY_QWEN_IMAGE_BASE`, e.g.
    /// `D:\Qwen-Image-2.1`): a latent of 6 by 10 tokens (a 96 by 160 picture), random as a sampled one is.
    #[test]
    #[ignore = "needs Qwen Image 2.1's VAE (OAIY_QWEN_IMAGE_BASE) and a WebGPU adapter"]
    fn the_webgpu_decoder_is_the_candle_one() -> Result<()> {
        let Some(base) = std::env::var_os("OAIY_QWEN_IMAGE_BASE").map(std::path::PathBuf::from) else { return Ok(()) };
        let (h, w) = (6usize, 10usize);
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let latent: Vec<f32> = (0..h * w * 64).map(|_| (next() * 1.7) as f32).collect();
        let latent = Tensor::from_vec(latent, (1, h * w, 64), &Device::Cpu)?;
        let gpu = WgpuVae::load(&base, 0)?;
        let t = std::time::Instant::now();
        let got = gpu.decode(&latent, h, w)?;
        eprintln!("WebGPU decode {:.3} s", t.elapsed().as_secs_f64());
        let cpu = crate::vae::Vae::load(&base, &Device::Cpu, DType::F32)?;
        let want = cpu.decode(&latent, h, w)?;
        assert_eq!(got.dims(), want.dims());
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let dot: f64 = g.iter().zip(&e).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&g) * norm(&e));
        let worst = g.iter().zip(&e).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
        eprintln!("cosine {cos:.6}, the worst error {worst:.4} (pixels in -1..1, a level 1/127.5)");
        // (f16's 11 bits where a pixel is the difference of the stream's largest terms: a random latent's reach 230,000;
        // the CUDA path's BF16 has 8)
        let off = g.iter().zip(&e).filter(|(a, b)| (**a - **b).abs() > 0.05).count();
        eprintln!("{off} of {} values off by more than 0.05", g.len());
        assert!(cos > 0.9999 && off * 1000 < g.len(), "cosine {cos}, {off} values off by more than 0.05");
        Ok(())
    }
}
