//! SDXL's VAE decoder on WebGPU, as [`crate::sdxl::vae::AutoencoderKL::decode`] computes it: an image its pixels'
//! rows of channels, the convolutions on the tensor cores where the adapter has them (the weights f16), each
//! residual block, the middle's attention (one head, its channels the head) and each upsampling a recording of its
//! own, its vectors let go once it has run (a 1024x1024 picture's last blocks are 0.5 to 1 GB a vector). On the CPU
//! through Candle the decode is most of a picture's time (15 s at 512x512, where its eight steps take 1.4).
use crate::sdxl::config::VaeConfig;
use crate::weights::Weights;
use candle_core::{DType, Device, Result};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::collections::HashMap;

const GROUPS: usize = 32;
const EPS: f32 = 1e-6;
/// The floats one run of the middle attention's queries may take (its result and the kernel's scratch): 512 MB.
const ATTENTION_FLOATS: usize = 1 << 27;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
    k: usize,
}

pub struct WgpuSdxlVae {
    gpu: ggml_rs_wgpu::WgpuBackend,
    cfg: VaeConfig,
    convs: HashMap<String, Conv>,
    /// A group norm's weight and bias.
    norms: HashMap<String, (DeviceVec, DeviceVec)>,
}

impl WgpuSdxlVae {
    /// The decoder of the VAE `cfg` describes from `w`'s tensors under `prefix` (a checkpoint's
    /// `first_stage_model.`), on `gpu`.
    pub fn load_on(w: &mut Weights, prefix: &str, cfg: &VaeConfig, gpu: ggml_rs_wgpu::WgpuBackend) -> Result<Self> {
        Self::load_mapped(w, prefix, cfg, gpu, &|key| key.to_owned(), true)
    }

    /// [`Self::load_on`] of a VAE whose tensors' names `map` turns into that layout's (a diffusers-layout file's
    /// `decoder.up_blocks.0.resnets.1` its `decoder.up.3.block.1`), with a `post_quant_conv` or (`post` false: FLUX.2's
    /// published VAE) perhaps none.
    pub fn load_mapped(w: &mut Weights, prefix: &str, cfg: &VaeConfig, gpu: ggml_rs_wgpu::WgpuBackend, map: &dyn Fn(&str) -> String, post: bool) -> Result<Self> {
        let (mut convs, mut norms) = (HashMap::new(), HashMap::new());
        for name in w.names() {
            let Some(key) = name.strip_prefix(prefix) else { continue };
            let key = map(key);
            if !(key.starts_with("decoder.") || key.starts_with("post_quant_conv.")) {
                continue;
            }
            let (Some(base), Some(stored)) = (key.strip_suffix(".weight"), name.strip_suffix(".weight")) else { continue };
            let t = w.tensor(&name, &Device::Cpu, DType::F32)?;
            let dims = t.dims().to_vec();
            let values = t.flatten_all()?.to_vec1::<f32>()?;
            let bias = w.tensor(&format!("{stored}.bias"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let upload = |v: &[f32]| {
                let d = gpu.vec(v.len());
                gpu.upload(&d, v);
                d
            };
            match dims.as_slice() {
                [_] => {
                    norms.insert(base.to_owned(), (upload(&values), upload(&bias)));
                }
                // (a checkpoint keeps the attention's projections as 1x1 convolutions, or as matrices)
                &[cout, cin] | &[cout, cin, 1, 1] => {
                    let wv = gpu.conv_weights(&values, cout, cin, 1).ok_or_else(|| err(format!("{name}: a weight past f16's range")))?;
                    convs.insert(base.to_owned(), Conv { w: wv, b: upload(&bias), cout, cin, k: 1 });
                }
                &[cout, cin, 3, 3] => {
                    let wv = gpu.conv_weights(&values, cout, cin, 3).ok_or_else(|| err(format!("{name}: a weight past f16's range")))?;
                    convs.insert(base.to_owned(), Conv { w: wv, b: upload(&bias), cout, cin, k: 3 });
                }
                other => candle_core::bail!("{name}: a VAE tensor of shape {other:?}"),
            }
        }
        if !convs.contains_key("decoder.conv_in") || (post && !convs.contains_key("post_quant_conv")) {
            candle_core::bail!("the checkpoint has no VAE decoder under {prefix}");
        }
        Ok(Self { gpu, cfg: cfg.clone(), convs, norms })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn conv(&self, r: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, h: usize, w: usize) -> Result<(DeviceVec, usize)> {
        let c = self.convs.get(name).ok_or_else(|| err(format!("missing VAE convolution {name}")))?;
        let y = self.vec(h * w * c.cout);
        r.conv_rows(&c.w, &c.b, c.cout, c.cin, c.k, x, h, w, &y);
        Ok((y, c.cout))
    }

    fn norm(&self, r: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, px: usize, c: usize, silu: bool) -> Result<DeviceVec> {
        let (weight, bias) = self.norms.get(name).ok_or_else(|| err(format!("missing VAE norm {name}")))?;
        let (y, stats) = (self.vec(px * c), self.vec(GROUPS * (px.div_ceil(256) + 1) * 2));
        r.group_norm_rows(x, weight, bias, &y, &stats, px, c, GROUPS, EPS, silu);
        Ok(y)
    }

    /// Residual block `p` of an image of `c` channels (a recording of its own): its output and channels.
    fn residual(&self, p: &str, x: &DeviceVec, h: usize, w: usize, c: usize) -> Result<(DeviceVec, usize)> {
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let shortcut = if self.convs.contains_key(&format!("{p}.nin_shortcut")) { self.conv(r, &format!("{p}.nin_shortcut"), x, h, w)?.0 } else { x.clone() };
        let t = self.norm(r, &format!("{p}.norm1"), x, h * w, c, true)?;
        let (h1, co) = self.conv(r, &format!("{p}.conv1"), &t, h, w)?;
        let t2 = self.norm(r, &format!("{p}.norm2"), &h1, h * w, co, true)?;
        let (h2, _) = self.conv(r, &format!("{p}.conv2"), &t2, h, w)?;
        r.add(&h2, &shortcut);
        rec.finish();
        drop((shortcut, t, h1, t2));
        self.gpu.settle();
        Ok((h2, co))
    }

    /// The middle's attention (`p`): one head over every pixel, its channels the head.
    fn attention(&self, p: &str, x: &DeviceVec, h: usize, w: usize, c: usize) -> Result<DeviceVec> {
        let rows = h * w;
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let n = self.norm(r, &format!("{p}.norm"), x, rows, c, false)?;
        let (q, _) = self.conv(r, &format!("{p}.q"), &n, h, w)?;
        let (k, _) = self.conv(r, &format!("{p}.k"), &n, h, w)?;
        let (v, _) = self.conv(r, &format!("{p}.v"), &n, h, w)?;
        // a position's key then its value: the row the chain's attention reads
        let kv = self.vec(rows * 2 * c);
        r.store_rows(&k, &kv, rows, c, 0, 2 * c, 0);
        r.store_rows(&v, &kv, rows, c, 0, 2 * c, c);
        // the queries a run at a time, each over every key: a head 512 wide is the f32 kernel's, whose scratch grows
        // with its queries times its keys (a 1024x1024 picture's 16,384 pixels at once: 2.2 GB, past what one
        // binding takes)
        let scale = 1.0 / (c as f32).sqrt();
        let whole = self.gpu.attention_rows_full_out_len(rows, 1, c, rows);
        let run = if whole <= ATTENTION_FLOATS { rows } else { (rows * ATTENTION_FLOATS / whole / 32 * 32).max(32) };
        let att = self.vec(rows * c);
        let mut runs = Vec::new();
        for at in (0..rows).step_by(run) {
            let m = run.min(rows - at);
            let (qr, out) = (self.vec(m * c), self.vec(self.gpu.attention_rows_full_out_len(m, 1, c, rows)));
            r.copy(&q, at * c, &qr, 0, m * c);
            r.attention_rows_full(&qr, &kv, &out, m, 1, 1, c, rows, scale);
            r.copy(&out, 0, &att, at * c, m * c);
            runs.push((qr, out));
        }
        let (proj, _) = self.conv(r, &format!("{p}.proj_out"), &att, h, w)?;
        r.add(&proj, x);
        rec.finish();
        drop((n, q, k, v, kv, att, runs));
        self.gpu.settle();
        Ok(proj)
    }

    /// The picture of `latent` (`[h w, latent_channels]`, a pixel's channels a row, as the sampler leaves it): its
    /// pixels' rows of `out_channels` (`[8 h * 8 w, 3]` for SDXL's four levels) in -1..1, and its height and width.
    pub fn decode(&self, latent: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let lc = self.cfg.latent_channels;
        if latent.len() != h * w * lc {
            candle_core::bail!("a latent of {} values for {h}x{w} pixels of {lc}", latent.len());
        }
        // out of the sampler's scale
        let scale = 1.0 / self.cfg.scaling_factor as f32;
        let values: Vec<f32> = latent.iter().map(|v| v * scale).collect();
        let x0 = self.vec(values.len());
        self.gpu.upload(&x0, &values);
        let (mut h, mut w) = (h, w);
        let (mut x, mut c) = {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            let x = if self.convs.contains_key("post_quant_conv") { self.conv(r, "post_quant_conv", &x0, h, w)?.0 } else { x0.clone() };
            let out = self.conv(r, "decoder.conv_in", &x, h, w)?;
            rec.finish();
            out
        };
        (x, c) = self.residual("decoder.mid.block_1", &x, h, w, c)?;
        x = self.attention("decoder.mid.attn_1", &x, h, w, c)?;
        (x, c) = self.residual("decoder.mid.block_2", &x, h, w, c)?;
        // the levels from the deepest: each its residual blocks, then (but the last) each pixel four and a convolution
        for level in (0..self.cfg.channel_mults.len()).rev() {
            for j in 0..=self.cfg.layers_per_block {
                (x, c) = self.residual(&format!("decoder.up.{level}.block.{j}"), &x, h, w, c)?;
            }
            if level != 0 {
                let mut rec = self.gpu.begin();
                rec.keep_groups(false);
                let r = rec.as_mut();
                let up = self.vec(4 * h * w * c);
                r.upsample2x_rows(&x, &up, h, w, c);
                let (y, cout) = self.conv(r, &format!("decoder.up.{level}.upsample.conv"), &up, 2 * h, 2 * w)?;
                rec.finish();
                drop(up);
                self.gpu.settle();
                (x, c, h, w) = (y, cout, 2 * h, 2 * w);
            }
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let t = self.norm(r, "decoder.norm_out", &x, h * w, c, true)?;
        let (out, co) = self.conv(r, "decoder.conv_out", &t, h, w)?;
        rec.read_range(&out, 0, h * w * co);
        let rgb = rec.finish().pop().ok_or_else(|| err("the VAE's picture was not read"))?;
        drop((x, t, out));
        self.gpu.settle();
        Ok((rgb, h, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdxl::vae::AutoencoderKL;
    use candle_core::Tensor;
    use candle_nn::{VarBuilder, VarMap};

    /// A small VAE of SDXL's shape (its middle 512 wide, as SDXL's: the attention's one head of 512) with random
    /// weights, norms' and biases' too: the decoder's picture on WebGPU is Candle's on the CPU in f32.
    #[test]
    fn a_small_vae_decoder_on_webgpu_is_candles() -> Result<()> {
        let Ok(gpu) = ggml_rs_wgpu::WgpuBackend::new(Some(2 << 30)) else { return Ok(()) };
        let cfg = VaeConfig { base_channels: 128, channel_mults: vec![1, 4], layers_per_block: 2, latent_channels: 4, scaling_factor: 0.13025, in_channels: 3, out_channels: 3, norm_groups: 32 };
        let map = VarMap::new();
        let reference = AutoencoderKL::load(&cfg, VarBuilder::from_varmap(&map, DType::F32, &Device::Cpu).pp("first_stage_model"))?;
        let mut seed = 0x51ed_270bu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
        };
        for (name, var) in map.data().lock().unwrap().iter() {
            if var.dims().len() == 1 {
                let n = var.dims()[0];
                let weight = name.ends_with("weight");
                var.set(&Tensor::from_vec((0..n).map(|_| if weight { 1.0 + 0.3 * next() } else { 0.2 * next() }).collect::<Vec<f32>>(), n, &Device::Cpu)?)?;
            }
        }
        let dir = std::env::temp_dir().join(format!("oaiy-wgpu-sdxl-vae-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("vae.safetensors");
        map.save(&path)?;
        let vae = WgpuSdxlVae::load_on(&mut Weights::open(&path)?, "first_stage_model.", &cfg, gpu);
        std::fs::remove_dir_all(dir)?;
        let vae = vae?;
        let (h, w) = (12usize, 20usize);
        let latent: Vec<f32> = (0..4 * h * w).map(|_| 0.6 * next()).collect();
        let z = Tensor::from_vec(latent, (1, 4, h, w), &Device::Cpu)?;
        let want = reference.decode(&z)?.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let (got, oh, ow) = vae.decode(&z.squeeze(0)?.permute((1, 2, 0))?.contiguous()?.flatten_all()?.to_vec1::<f32>()?, h, w)?;
        assert_eq!((oh, ow, got.len()), (2 * h, 2 * w, want.len()));
        let (mut dot, mut gg, mut ww, mut worst) = (0f64, 0f64, 0f64, 0f64);
        for (g, t) in got.iter().zip(&want) {
            let (g, t) = (*g as f64, *t as f64);
            dot += g * t;
            gg += g * g;
            ww += t * t;
            worst = worst.max((g - t).abs());
        }
        let (cosine, rms) = (dot / (gg.sqrt() * ww.sqrt()), (ww / want.len() as f64).sqrt());
        eprintln!("a small VAE's picture: cosine {cosine:.6}, the worst error {worst:.2e} of an RMS of {rms:.3}");
        assert!(cosine > 0.9995 && worst < 0.1 * rms.max(1e-3), "cosine {cosine}, worst {worst} of {rms}");
        Ok(())
    }
}
