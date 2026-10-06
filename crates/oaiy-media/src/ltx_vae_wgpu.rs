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

    /// [`Self::decode`] whole where the card has room for it (its activations some 260 bytes a pixel of every frame:
    /// 12 GB at 768x512 and 121 frames) and its largest vector fits a binding (the last blocks' 128 channels at a
    /// quarter of the pixels' rows and columns: 2.3 GB at 1024x576), else in overlapping tiles of the size that has, a
    /// quarter of a tile shared.
    pub fn decode_fitted(&self, latent: &[f32], f: usize, h: usize, w: usize) -> Result<Tensor> {
        // (where the API reports no budget: a card's 8 GB free)
        let free = self.gpu.memory_budget().map_or(8 << 30, |(budget, used)| budget.saturating_sub(used)) as f64 * 0.85;
        let frames = if f == 1 { 1 } else { 8 * (f - 1) + 1 };
        let binding = self.gpu.max_binding();
        let fits = |th: usize, tw: usize| (frames * th * 32 * tw * 32) as f64 * 260. <= free && (frames * th * 8 * tw * 8 * 128 * 4) as u64 <= binding;
        if fits(h, w) {
            return self.decode(latent, f, h, w);
        }
        let mut tile = 16;
        while tile > 4 && !fits(tile.min(h), tile.min(w)) {
            tile /= 2;
        }
        if !fits(tile.min(h), tile.min(w)) {
            candle_core::bail!("the card has no room for the video decoder's tiles; reduce frames or free GPU memory");
        }
        self.decode_tiled(latent, f, h, w, tile, tile / 4)
    }

    /// [`Self::decode`] in overlapping tiles of `tile` latent rows and columns (`overlap` shared, blended linearly) as
    /// [`crate::ltx::vae::LtxVideoDecoder::decode_tiled`] makes them: each tile's every frame on the GPU, the blend on
    /// the host.
    pub fn decode_tiled(&self, latent: &[f32], f: usize, h: usize, w: usize, tile: usize, overlap: usize) -> Result<Tensor> {
        use crate::ltx::vae::{build_intervals, trapezoidal_mask_1d};
        if latent.len() != f * h * w * 128 || overlap >= tile {
            candle_core::bail!("an LTX latent of {} values for {f}x{h}x{w} tokens, tiles of {tile} sharing {overlap}", latent.len());
        }
        let frames = if f == 1 { 1 } else { 8 * (f - 1) + 1 };
        let (hp, wp) = (h * 32, w * 32);
        let mut acc = vec![0f32; 3 * frames * hp * wp];
        let mut weight = vec![0f32; hp * wp];
        for &(h_lo, h_hi, top, bottom) in &build_intervals(h, tile, overlap) {
            for &(w_lo, w_hi, left, right) in &build_intervals(w, tile, overlap) {
                let (th, tw) = (h_hi - h_lo, w_hi - w_lo);
                let mut part = Vec::with_capacity(f * th * tw * 128);
                for t in 0..f {
                    for y in h_lo..h_hi {
                        let row = (t * h + y) * w;
                        part.extend_from_slice(&latent[(row + w_lo) * 128..(row + w_hi) * 128]);
                    }
                }
                let values = self.decode(&part, f, th, tw)?.flatten_all()?.to_vec1::<f32>()?;
                let (ph, pw) = (th * 32, tw * 32);
                let hm = trapezoidal_mask_1d(ph, top * 32, bottom * 32)?.to_vec1::<f32>()?;
                let wm = trapezoidal_mask_1d(pw, left * 32, right * 32)?.to_vec1::<f32>()?;
                let (y0, x0) = (h_lo * 32, w_lo * 32);
                for y in 0..ph {
                    for x in 0..pw {
                        weight[(y0 + y) * wp + x0 + x] += hm[y] * wm[x];
                    }
                }
                for plane in 0..3 * frames {
                    for y in 0..ph {
                        let (from, to) = ((plane * ph + y) * pw, (plane * hp + y0 + y) * wp + x0);
                        for x in 0..pw {
                            acc[to + x] += values[from + x] * hm[y] * wm[x];
                        }
                    }
                }
            }
        }
        for plane in acc.chunks_mut(hp * wp) {
            for (v, wt) in plane.iter_mut().zip(&weight) {
                if *wt <= 0. {
                    candle_core::bail!("a pixel no tile covered");
                }
                *v /= wt;
            }
        }
        Tensor::from_vec(acc, (1, 3, frames, hp, wp), &Device::Cpu)
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

/// LTX's video VAE's encoder on WebGPU for one image (a clip's start or end), as
/// [`crate::ltx::vae::LtxVideoEncoder::encode_means`] computes a one-frame video: its causal 3x3x3 convolutions see the
/// one frame through every time tap (their taps summed into 3x3 ones, the 2D convolutions'), its downsamplings' frames
/// the one repeated (each convolution's output and its input packed space to depth, the input's groups averaged as
/// the shortcut).
pub struct WgpuLtxImageEncoder {
    gpu: ggml_rs_wgpu::WgpuBackend,
    /// Each convolution's taps summed over time (`k` by `k`, as [`DeviceChain::conv_weights`] packs them), its bias,
    /// its channels out and in.
    convs: HashMap<String, (DeviceVec, DeviceVec, usize, usize)>,
    ones: HashMap<usize, DeviceVec>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

/// The encoder's blocks in turn: residual blocks of the channels, or a downsampling by (time, rows, columns).
const DOWN: [(usize, (usize, usize, usize)); 9] = [(4, (0, 0, 0)), (0, (1, 2, 2)), (6, (0, 0, 0)), (0, (2, 1, 1)), (4, (0, 0, 0)), (0, (2, 2, 2)), (2, (0, 0, 0)), (0, (2, 2, 2)), (2, (0, 0, 0))];

impl WgpuLtxImageEncoder {
    /// The encoder in `store` (`vae.encoder.*`, `vae.per_channel_statistics.*`: a checkpoint's, or a VAE file's own
    /// without the `vae.`) on GPU `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(store: &mut Store, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let prefix = if store.index.names().any(|n| n.starts_with("vae.encoder.")) { "vae." } else { "" };
        let names: Vec<String> = store.index.names().filter(|n| n.starts_with(&format!("{prefix}encoder.")) && n.ends_with(".conv.weight")).map(str::to_owned).collect();
        let mut convs = HashMap::new();
        for name in names {
            let base = name.strip_suffix(".weight").unwrap_or(&name).to_owned();
            let t = store.tensor_f32(&name, &Device::Cpu)?;
            let &[cout, cin, kt, 3, 3] = t.dims() else { candle_core::bail!("{name}: a convolution of shape {:?}", t.dims()) };
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            // the time taps summed: every one of them the one frame
            let mut w2 = vec![0f32; cout * cin * 9];
            for (o, taps) in w2.chunks_mut(9).enumerate() {
                for t in 0..kt {
                    for (j, x) in taps.iter_mut().enumerate() {
                        *x += v[(o * kt + t) * 9 + j];
                    }
                }
            }
            let w = gpu.conv_weights(&w2, cout, cin, 3).ok_or_else(|| err(format!("{name}: no tensor cores, or a weight past f16's range")))?;
            let bias = store.tensor_f32(&format!("{base}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let b = gpu.vec(bias.len());
            gpu.upload(&b, &bias);
            convs.insert(base.strip_prefix(prefix).unwrap_or(&base).to_owned(), (w, b, cout, cin));
        }
        let stat = |store: &mut Store, key: &str| -> Result<Vec<f32>> { store.tensor_f32(&format!("{prefix}per_channel_statistics.{key}"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>() };
        let (std, mean) = (stat(store, "std-of-means")?, stat(store, "mean-of-means")?);
        let mut ones = HashMap::new();
        for c in [128usize, 256, 512, 1024] {
            let v = gpu.vec(c);
            gpu.upload(&v, &vec![1.0; c]);
            ones.insert(c, v);
        }
        Ok(Self { gpu, convs, ones, mean, std })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn conv(&self, r: &mut dyn ChainRecorder, name: &str, x: &DeviceVec, h: usize, w: usize) -> Result<(DeviceVec, usize)> {
        let (cw, b, cout, cin) = self.convs.get(name).ok_or_else(|| err(format!("missing LTX VAE convolution {name}")))?;
        let y = self.vec(h * w * cout);
        r.conv_rows(cw, b, *cout, *cin, 3, x, h, w, &y);
        Ok((y, *cout))
    }

    fn norm_silu(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, rows: usize, c: usize) -> Result<DeviceVec> {
        let ones = self.ones.get(&c).ok_or_else(|| err(format!("no norm of {c} channels")))?;
        let n = self.vec(rows * c);
        r.rmsnorm_silu_rows(x, ones, &n, rows, EPS);
        Ok(n)
    }

    /// The latent of an image (`rgb`: its `height` rows of `width` pixels' red, green and blue in -1..1): `[h w, 128]`
    /// (`h` its height over 32), each channel's mean over its deviation as the transformer takes it.
    pub fn encode(&self, rgb: &[f32], height: usize, width: usize) -> Result<Vec<f32>> {
        if rgb.len() != height * width * 3 || height % 32 != 0 || width % 32 != 0 {
            candle_core::bail!("an image of {} values for {width}x{height}, not whole 32 by 32 pixels", rgb.len());
        }
        // 4 x 4 patches into channels: a patch's (channel, its column, its row)
        let (mut h, mut w) = (height / 4, width / 4);
        let mut patched = vec![0f32; h * w * 48];
        for py in 0..h {
            for px in 0..w {
                for c in 0..3 {
                    for r in 0..4 {
                        for q in 0..4 {
                            patched[(py * w + px) * 48 + c * 16 + r * 4 + q] = rgb[((py * 4 + q) * width + px * 4 + r) * 3 + c];
                        }
                    }
                }
            }
        }
        let x0 = self.vec(patched.len());
        self.gpu.upload(&x0, &patched);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let (mut x, mut c) = self.conv(rec.as_mut(), "encoder.conv_in.conv", &x0, h, w)?;
        rec.finish();
        for (i, &(res, (st, sh, sw))) in DOWN.iter().enumerate() {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            if res > 0 {
                for j in 0..res {
                    let p = format!("encoder.down_blocks.{i}.res_blocks.{j}");
                    let t = self.norm_silu(r, &x, h * w, c)?;
                    let (h1, _) = self.conv(r, &format!("{p}.conv1.conv"), &t, h, w)?;
                    let t = self.norm_silu(r, &h1, h * w, c)?;
                    let (h2, _) = self.conv(r, &format!("{p}.conv2.conv"), &t, h, w)?;
                    r.add(&h2, &x);
                    x = h2;
                }
            } else {
                // the convolution's output packed space to depth, plus the input's packing's groups' means
                let vol = st * sh * sw;
                let (y, cc) = self.conv(r, &format!("encoder.down_blocks.{i}.conv.conv"), &x, h, w)?;
                let (oh, ow) = (h / sh, w / sw);
                let out = self.vec(oh * ow * cc * vol);
                r.space_to_depth_rows(&y, &out, h, w, cc, st, sh, sw);
                let packed = self.vec(oh * ow * c * vol);
                r.space_to_depth_rows(&x, &packed, h, w, c, st, sh, sw);
                r.group_mean_add_rows(&packed, &out, oh * ow, c * vol, cc * vol);
                (x, c, h, w) = (out, cc * vol, oh, ow);
            }
            rec.finish();
            self.gpu.settle();
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let t = self.norm_silu(r, &x, h * w, c)?;
        let (out, co) = self.conv(r, "encoder.conv_out.conv", &t, h, w)?;
        r.read(&out);
        let out = rec.finish().pop().ok_or_else(|| err("the latent was not read"))?;
        let lc = self.mean.len();
        if co <= lc || self.std.len() != lc {
            candle_core::bail!("the encoder gives {co} channels for a latent of {lc}");
        }
        // the means (all but the last channel, the logvar), less their mean over their deviation
        Ok(out.chunks(co).flat_map(|px| px[..lc].iter().zip(self.mean.iter().zip(&self.std)).map(|(v, (m, s))| (v - m) / s)).collect())
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
        // and a 1024x1024 clip of 121 frames (past the card at once) in tiles
        let (h, w) = (32usize, 32usize);
        let latent: Vec<f32> = (0..f * h * w * 128).map(|i| (i * 7919 % 2001) as f32 / 1000.0 - 1.0).collect();
        let t = std::time::Instant::now();
        let clip = gpu.decode_fitted(&latent, f, h, w)?;
        eprintln!("1024x1024, 121 frames: {:.2} s, {:?}", t.elapsed().as_secs_f64(), clip.dims());
        Ok(())
    }

    /// The WebGPU image encoder gives Candle's latent (on CUDA, BF16) of an image (a 64 x 96 one, random as pixels
    /// are not: smooth gradients and a stripe) from LTX 2.3's checkpoint (`OAIY_LTX_NVFP4`).
    #[test]
    #[ignore = "needs LTX 2.3's checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter and CUDA (the cuda feature)"]
    fn the_webgpu_image_encoder_is_the_candle_one() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let (h, w) = (64usize, 96usize);
        let rgb: Vec<f32> = (0..h * w * 3).map(|i| { let (px, c) = (i / 3, i % 3); let (y, x) = (px / w, px % w); ((y as f32 / h as f32) * 1.6 - 0.8 + if (x / 8 + c) % 3 == 0 { 0.3 } else { -0.2 } + (x as f32 * 0.07 + c as f32).sin() * 0.2).clamp(-1., 1.) }).collect();
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = WgpuLtxImageEncoder::load(&mut store, 0)?;
        let t = std::time::Instant::now();
        let got = gpu.encode(&rgb, h, w)?;
        eprintln!("WebGPU encode {:.3} s", t.elapsed().as_secs_f64());
        drop(gpu);
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        let encoder = crate::ltx::vae::LtxVideoEncoder::load(std::path::Path::new(&path), crate::ltx::vae::LtxVaeConfig::ltx_2_3_22b(), &dev, DType::BF16)?;
        let video = Tensor::from_vec(rgb, (1, 1, h, w, 3), &dev)?.permute((0, 4, 1, 2, 3))?.contiguous()?.to_dtype(DType::BF16)?;
        let want = encoder.encode_means(&video)?.permute((0, 2, 3, 4, 1))?.contiguous()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len());
        let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&got) * norm(&want));
        eprintln!("an image's latent: cosine {cos:.6}");
        assert!(cos > 0.999, "cosine {cos}");
        Ok(())
    }

    /// The tiled decode blends its tiles into the clip a whole decode gives: a latent of 2 frames of 5 by 6 in tiles of
    /// 3 sharing 1 (the seams' blend aside, the tiles see less of their neighbours: near, not equal).
    #[test]
    #[ignore = "needs LTX 2.3's checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn tiles_blend_into_the_whole_clip() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let (f, h, w) = (2usize, 5usize, 6usize);
        let mut seed = 0x2545_f491u64;
        let latent: Vec<f32> = (0..f * h * w * 128)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.) as f32
            })
            .collect();
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = WgpuLtxVae::load(&mut store, 0)?;
        let whole = gpu.decode(&latent, f, h, w)?.flatten_all()?.to_vec1::<f32>()?;
        let tiled = gpu.decode_tiled(&latent, f, h, w, 3, 1)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(whole.len(), tiled.len());
        let dot: f64 = whole.iter().zip(&tiled).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&whole) * norm(&tiled));
        eprintln!("tiles of 3 sharing 1 against the whole clip: cosine {cos:.6}");
        assert!(cos > 0.95, "cosine {cos}");
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
