//! LTX 2.3's video VAE decoder on WebGPU, as [`crate::ltx::vae::LtxVideoDecoder::decode`] computes it: each voxel's
//! channels a row (the tensor cores' tokens), the 3x3x3 convolutions on the tensor cores (an implicit im2col, the
//! clip's first and last frames repeated past its ends), each input's range scaled into f16's, the weights f16 (the
//! checkpoint's BF16). Each residual block and each upsampling its own recording, settled after (a 121-frame clip's
//! last blocks are 1.5 GB a vector).
use crate::ltx::store::Store;
use candle_core::{Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::collections::HashMap;

const EPS: f32 = 1e-6;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A 3x3x3 convolution on the GPU: its packed weights, its bias, its channels out and in.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cout: usize,
    cin: usize,
}

/// The decoder's blocks from the latent up (LTX 2.3 22B's: its config's list reversed).
enum Up {
    /// Residual blocks of the channels, this many.
    Res(usize),
    /// A convolution then depth to space by (time, rows, columns).
    Shuffle(usize, usize, usize),
}

const BLOCKS: [Up; 9] = [Up::Res(2), Up::Shuffle(2, 2, 2), Up::Res(2), Up::Shuffle(2, 2, 2), Up::Res(4), Up::Shuffle(2, 1, 1), Up::Res(6), Up::Shuffle(1, 2, 2), Up::Res(4)];

pub struct WgpuLtxVae {
    gpu: ggml_rs_wgpu::WgpuBackend,
    convs: HashMap<String, Conv>,
    /// A pixel norm's weights (ones), by channels.
    ones: HashMap<usize, DeviceVec>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl WgpuLtxVae {
    /// The decoder in `store` (`vae.decoder.*`, `vae.per_channel_statistics.*`: a checkpoint's, or a VAE file's own
    /// without the `vae.`) on GPU `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(store: &mut Store, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut convs = HashMap::new();
        let prefix = if store.index.names().any(|n| n.starts_with("vae.decoder.")) { "vae." } else { "" };
        let names: Vec<String> = store.index.names().filter(|n| n.starts_with(&format!("{prefix}decoder.")) && n.ends_with(".conv.weight")).map(str::to_owned).collect();
        for name in names {
            let base = name.strip_suffix(".weight").unwrap_or(&name).to_owned();
            let t = store.tensor_f32(&name, &Device::Cpu)?;
            let &[cout, cin, 3, 3, 3] = t.dims() else { candle_core::bail!("{name}: a convolution of shape {:?}", t.dims()) };
            let w = gpu.conv3d_weights(&t.flatten_all()?.to_vec1::<f32>()?, cout, cin).ok_or_else(|| err(format!("{name}: no tensor cores, or a weight past f16's range")))?;
            let bias = store.tensor_f32(&format!("{base}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let b = gpu.vec(bias.len());
            gpu.upload(&b, &bias);
            // (kept under the checkpoint's names, as the decode asks for them)
            convs.insert(format!("vae.{}", base.strip_prefix(prefix).unwrap_or(&base)), Conv { w, b, cout, cin });
        }
        let stat = |store: &mut Store, key: &str| -> Result<Vec<f32>> { store.tensor_f32(&format!("{prefix}per_channel_statistics.{key}"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>() };
        let (std, mean) = (stat(store, "std-of-means")?, stat(store, "mean-of-means")?);
        let mut ones = HashMap::new();
        for c in [1024usize, 512, 256, 128] {
            let v = gpu.vec(c);
            gpu.upload(&v, &vec![1.0; c]);
            ones.insert(c, v);
        }
        Ok(Self { gpu, convs, ones, mean, std })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn conv(&self, r: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, f: usize, h: usize, w: usize) -> Result<(DeviceVec, usize)> {
        let c = self.convs.get(name).ok_or_else(|| err(format!("missing LTX VAE convolution {name}")))?;
        let y = self.vec(f * h * w * c.cout);
        r.conv3d_rows(&c.w, &c.b, c.cout, c.cin, x, f, h, w, &y);
        Ok((y, c.cout))
    }

    /// The pixel norm (over each voxel's channels' RMS) then SiLU.
    fn norm_silu(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, voxels: usize, c: usize) -> Result<DeviceVec> {
        let ones = self.ones.get(&c).ok_or_else(|| err(format!("no norm of {c} channels")))?;
        let n = self.vec(voxels * c);
        r.rmsnorm_silu_rows(x, ones, &n, voxels, EPS);
        Ok(n)
    }

    /// The clip of `latent` (`[f h w, 128]`, the transformer's tokens: `f` frames of `h` by `w`): `[1, 3, 8 (f - 1) +
    /// 1, 32 h, 32 w]` (RGB in -1..1) on the CPU.
    pub fn decode(&self, latent: &[f32], f: usize, h: usize, w: usize) -> Result<Tensor> {
        if latent.len() != f * h * w * 128 {
            candle_core::bail!("an LTX latent of {} values for {f}x{h}x{w} tokens", latent.len());
        }
        let values: Vec<f32> = latent.iter().enumerate().map(|(i, v)| v * self.std[i % 128] + self.mean[i % 128]).collect();
        let x0 = self.vec(values.len());
        self.gpu.upload(&x0, &values);
        let (mut f, mut h, mut w) = (f, h, w);
        let (mut x, mut c) = {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let out = self.conv(rec.as_mut(), "vae.decoder.conv_in.conv", &x0, f, h, w)?;
            rec.finish();
            out
        };
        for (i, block) in BLOCKS.iter().enumerate() {
            match *block {
                Up::Res(n) => {
                    for j in 0..n {
                        let p = format!("vae.decoder.up_blocks.{i}.res_blocks.{j}");
                        let mut rec = self.gpu.begin();
                        rec.keep_groups(false);
                        let r = rec.as_mut();
                        let v = f * h * w;
                        let t = self.norm_silu(r, &x, v, c)?;
                        let (h1, _) = self.conv(r, &format!("{p}.conv1.conv"), &t, f, h, w)?;
                        let t = self.norm_silu(r, &h1, v, c)?;
                        let (h2, _) = self.conv(r, &format!("{p}.conv2.conv"), &t, f, h, w)?;
                        r.add(&h2, &x);
                        rec.finish();
                        drop((t, h1));
                        x = h2;
                        self.gpu.settle();
                    }
                }
                Up::Shuffle(st, sh, sw) => {
                    let mut rec = self.gpu.begin();
                    rec.keep_groups(false);
                    let r = rec.as_mut();
                    let (y, packed) = self.conv(r, &format!("vae.decoder.up_blocks.{i}.conv.conv"), &x, f, h, w)?;
                    let cout = packed / (st * sh * sw);
                    // (doubling time drops the frame the repeated first one made)
                    let dropped = usize::from(st == 2);
                    let nf = f * st - dropped;
                    let out = self.vec(nf * h * sh * w * sw * cout);
                    r.depth_to_space_rows(&y, &out, f, h, w, cout, st, sh, sw, dropped);
                    rec.finish();
                    drop(y);
                    (x, c, f, h, w) = (out, cout, nf, h * sh, w * sw);
                    self.gpu.settle();
                }
            }
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let t = self.norm_silu(r, &x, f * h * w, c)?;
        let (out, co) = self.conv(r, "vae.decoder.conv_out.conv", &t, f, h, w)?;
        r.read(&out);
        let rows = rec.finish().pop().ok_or_else(|| err("the clip was not read"))?;
        // unpatchify: a voxel's 48 channels (colour, patch column, patch row) to its 4x4 pixels
        let (p, colours) = (4usize, co / 16);
        let (oh, ow) = (h * p, w * p);
        let mut pixels = vec![0f32; colours * f * oh * ow];
        for t in 0..f {
            for y in 0..h {
                for xx in 0..w {
                    let row = &rows[((t * h + y) * w + xx) * co..][..co];
                    for c in 0..colours {
                        for iw in 0..p {
                            for ih in 0..p {
                                pixels[((c * f + t) * oh + y * p + ih) * ow + xx * p + iw] = row[c * 16 + iw * p + ih];
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(pixels, (1, colours, f, oh, ow), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// How long the WebGPU decoder takes (`--ignored --nocapture`, `OAIY_LTX_NVFP4`): a 768x512 clip of 121 frames
    /// (a latent of 16 frames of 16 by 24), and the card's memory at its most.
    #[test]
    #[ignore = "a timing; needs LTX 2.3's checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn measure_a_clips_decode() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let (f, h, w) = (16usize, 16usize, 24usize);
        let latent: Vec<f32> = (0..f * h * w * 128).map(|i| (i * 7919 % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = WgpuLtxVae::load(&mut store, 0)?;
        for i in 0..2 {
            let t = std::time::Instant::now();
            let clip = gpu.decode(&latent, f, h, w)?;
            eprintln!("decode {i}: {:.2} s, {:?}; the card's memory {:?}", t.elapsed().as_secs_f64(), clip.dims(), gpu.gpu.memory_budget().map(|(b, u)| (u as f64 / 1e9, b as f64 / 1e9)));
        }
        Ok(())
    }

    /// The WebGPU decoder gives Candle's clip (on CUDA, BF16) from Lightricks' release (`OAIY_LTX_NVFP4`): a latent of
    /// 2 frames of 2 by 3 (9 frames of 64 by 96), random as a sampled one is.
    #[test]
    #[ignore = "needs LTX 2.3's checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter and CUDA (the cuda feature) for Candle's"]
    fn the_webgpu_video_decoder_is_the_candle_one() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let path = std::path::Path::new(&path);
        let (f, h, w) = (2usize, 2usize, 3usize);
        let mut seed = 0x7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let latent: Vec<f32> = (0..f * h * w * 128).map(|_| (next() * 1.5) as f32).collect();
        let mut store = Store::open(path, 0)?;
        let gpu = WgpuLtxVae::load(&mut store, 0)?;
        let t = std::time::Instant::now();
        let got = gpu.decode(&latent, f, h, w)?;
        eprintln!("WebGPU decode {:.3} s: {:?}", t.elapsed().as_secs_f64(), got.dims());
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        let decoder = crate::ltx::vae::LtxVideoDecoder::load(path, crate::ltx::vae::LtxVaeConfig::ltx_2_3_22b(), &dev, DType::BF16)?;
        let lt = Tensor::from_vec(latent, (1, f, h, w, 128), &dev)?.permute((0, 4, 1, 2, 3))?.contiguous()?.to_dtype(DType::BF16)?;
        let want = decoder.decode(&lt)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        assert_eq!(got.dims(), want.dims());
        let (g, e) = (got.flatten_all()?.to_vec1::<f32>()?, want.flatten_all()?.to_vec1::<f32>()?);
        let dot: f64 = g.iter().zip(&e).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&g) * norm(&e));
        let off = g.iter().zip(&e).filter(|(a, b)| (**a - **b).abs() > 0.05).count();
        eprintln!("cosine {cos:.6}, {off} of {} values off by more than 0.05", g.len());
        assert!(cos > 0.999, "cosine {cos}");
        Ok(())
    }
}
