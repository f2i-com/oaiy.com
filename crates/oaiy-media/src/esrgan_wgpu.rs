//! Real-ESRGAN x4plus on WebGPU (a picture job's `backend` "webgpu": any GPU wgpu reaches, Vulkan, Metal or DX12, with
//! tensor cores or without): [`crate::esrgan`]'s network, its pixels' channels in rows (`[h * w, c]`). A dense
//! block's concatenation is one buffer of 192 channels a pixel, each convolution reading its leading channels
//! ([`ChainRecorder::conv_rows_strided`]) and its output put in its place ([`ChainRecorder::store_rows`]); the blocks'
//! 0.2 scales as a weight on the device ([`ChainRecorder::axpy_at`]). Tiles as Candle's: 256 pixels a side, 10 around.
use crate::esrgan::{SCALE, TILE, TILE_PAD};
use candle_core::{DType, Device, Result};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec};
use ggml_rs_wgpu::WgpuBackend;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("Real-ESRGAN on WebGPU: {e}"))
}

/// The leaky ReLUs' slope.
const SLOPE: f32 = 0.2;
/// The width of the network's features, a dense block's growth, and its concatenation.
const F: usize = 64;
const G: usize = 32;
const CAT: usize = F + 4 * G;

/// A 3x3 convolution's weights on the device ([`DeviceChain::conv_weights`]) and its bias.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
}

pub struct WgpuEsrgan {
    gpu: WgpuBackend,
    first: Conv,
    /// The 23 residual-in-residual blocks' three dense blocks' five convolutions.
    body: Vec<[[Conv; 5]; 3]>,
    conv_body: Conv,
    up1: Conv,
    up2: Conv,
    hr: Conv,
    last: Conv,
    /// `[0.2, 1.0]`: the blocks' residual scale, and a plain sum's.
    weights: DeviceVec,
}

impl WgpuEsrgan {
    /// `RealESRGAN_x4plus.pth` (as [`crate::esrgan::Esrgan::load`] takes it) on WebGPU device `device`.
    pub fn load(path: &Path, device: usize) -> Result<Self> {
        let gpu = WgpuBackend::nth(device, None).map_err(err)?;
        let (mut pth, sd) = crate::esrgan::state_dict(path)?;
        let mut conv = |p: &str| -> Result<Conv> {
            let mut t = |k: String| -> Result<(Vec<f32>, Vec<usize>)> {
                let v = sd.get(&k).ok_or_else(|| err(format!("no {k}")))?.clone();
                let t = pth.tensor(&v, &Device::Cpu)?.to_dtype(DType::F32)?;
                Ok((t.flatten_all()?.to_vec1::<f32>()?, t.dims().to_vec()))
            };
            let (w, shape) = t(format!("{p}.weight"))?;
            let (b, _) = t(format!("{p}.bias"))?;
            let &[cout, cin, 3, 3] = shape.as_slice() else { return Err(err(format!("{p}: a weight of {shape:?}, not a 3x3 convolution's"))) };
            let w = gpu.conv_weights(&w, cout, cin, 3).ok_or_else(|| err(format!("{p}: weights past f16's range")))?;
            let bias = gpu.vec(cout);
            gpu.upload(&bias, &b);
            Ok(Conv { w, b: bias, cin, cout })
        };
        let first = conv("conv_first")?;
        let mut body = Vec::with_capacity(23);
        for i in 0..23 {
            let mut rdb = |r: usize| -> Result<[Conv; 5]> { Ok([conv(&format!("body.{i}.rdb{r}.conv1"))?, conv(&format!("body.{i}.rdb{r}.conv2"))?, conv(&format!("body.{i}.rdb{r}.conv3"))?, conv(&format!("body.{i}.rdb{r}.conv4"))?, conv(&format!("body.{i}.rdb{r}.conv5"))?]) };
            body.push([rdb(1)?, rdb(2)?, rdb(3)?]);
        }
        let (conv_body, up1, up2, hr, last) = (conv("conv_body")?, conv("conv_up1")?, conv("conv_up2")?, conv("conv_hr")?, conv("conv_last")?);
        let weights = gpu.vec(2);
        gpu.upload(&weights, &[0.2, 1.0]);
        Ok(Self { gpu, first, body, conv_body, up1, up2, hr, last, weights })
    }

    /// The network on a tile (`w` by `h` pixels, each its red, green and blue in 0..1 in turn): `4w` by `4h` pixels the
    /// same way, unclamped.
    pub fn forward(&self, rgb: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
        let m = w * h;
        if rgb.len() != m * 3 || m == 0 {
            return Err(err(format!("a tile of {w}x{h} with {} values", rgb.len())));
        }
        let g = &self.gpu;
        let x3 = g.vec(m * 3);
        g.upload(&x3, rgb);
        let (feat, hb, x, cat, t32, t64) = (g.vec(m * F), g.vec(m * F), g.vec(m * F), g.vec(m * CAT), g.vec(m * G), g.vec(m * F));
        let (u1, v1, u2, v2, out) = (g.vec(4 * m * F), g.vec(4 * m * F), g.vec(16 * m * F), g.vec(16 * m * F), g.vec(16 * m * 3));
        let mut rec = g.begin();
        // (bind groups kept would hold each tile's own buffers to the end)
        rec.keep_groups(false);
        let r = rec.as_mut();
        let conv = |r: &mut dyn ChainRecorder, c: &Conv, x: &DeviceVec, xs: usize, (w, h): (usize, usize), y: &DeviceVec| r.conv_rows_strided(&c.w, &c.b, c.cout, c.cin, 3, x, xs, h, w, y);
        conv(r, &self.first, &x3, 3, (w, h), &feat);
        r.copy(&feat, 0, &hb, 0, m * F);
        for block in &self.body {
            r.copy(&hb, 0, &x, 0, m * F);
            for rdb in block {
                // the block's input, then each convolution's output beside it: the next one's concatenation
                r.store_rows(&x, &cat, m, F, 0, CAT, 0);
                for (k, c) in rdb[..4].iter().enumerate() {
                    conv(r, c, &cat, CAT, (w, h), &t32);
                    r.leaky_relu(&t32, &t32, m * G, SLOPE);
                    r.store_rows(&t32, &cat, m, G, 0, CAT, F + k * G);
                }
                conv(r, &rdb[4], &cat, CAT, (w, h), &t64);
                r.axpy_at(&x, &t64, &self.weights, 0, m * F);
            }
            r.axpy_at(&hb, &x, &self.weights, 0, m * F);
        }
        conv(r, &self.conv_body, &hb, F, (w, h), &t64);
        r.axpy_at(&feat, &t64, &self.weights, 1, m * F);
        r.upsample2x_rows(&feat, &u1, h, w, F);
        conv(r, &self.up1, &u1, F, (2 * w, 2 * h), &v1);
        r.leaky_relu(&v1, &v1, 4 * m * F, SLOPE);
        r.upsample2x_rows(&v1, &u2, 2 * h, 2 * w, F);
        conv(r, &self.up2, &u2, F, (4 * w, 4 * h), &v2);
        r.leaky_relu(&v2, &v2, 16 * m * F, SLOPE);
        conv(r, &self.hr, &v2, F, (4 * w, 4 * h), &u2);
        r.leaky_relu(&u2, &u2, 16 * m * F, SLOPE);
        conv(r, &self.last, &u2, F, (4 * w, 4 * h), &out);
        r.read(&out);
        rec.finish().pop().ok_or_else(|| err("the tile's output was not read"))
    }

    /// An RGB picture (`w` by `h`, 8-bit) four times larger (`4w` by `4h`, 8-bit RGB), in tiles as
    /// [`crate::esrgan::Esrgan::upscale_with`]'s, telling `progress` how many of how many are done.
    pub fn upscale_with(&self, rgb: &[u8], w: usize, h: usize, mut progress: impl FnMut(usize, usize)) -> Result<Vec<u8>> {
        let (ow, oh) = (w * SCALE, h * SCALE);
        let mut out = vec![0u8; ow * oh * 3];
        let tiles = w.div_ceil(TILE) * h.div_ceil(TILE);
        let mut done = 0;
        progress(0, tiles);
        for ty in (0..h).step_by(TILE) {
            for tx in (0..w).step_by(TILE) {
                let (x1, y1) = ((tx + TILE).min(w), (ty + TILE).min(h));
                let (px0, py0) = (tx.saturating_sub(TILE_PAD), ty.saturating_sub(TILE_PAD));
                let (px1, py1) = ((x1 + TILE_PAD).min(w), (y1 + TILE_PAD).min(h));
                let (pw, ph) = (px1 - px0, py1 - py0);
                let mut data = Vec::with_capacity(pw * ph * 3);
                for y in 0..ph {
                    for x in 0..pw {
                        data.extend((0..3).map(|c| rgb[((py0 + y) * w + px0 + x) * 3 + c] as f32 / 255.));
                    }
                }
                let up = self.forward(&data, pw, ph)?;
                let uw = pw * SCALE;
                for y in ty * SCALE..y1 * SCALE {
                    for x in tx * SCALE..x1 * SCALE {
                        let (sy, sx) = (y - py0 * SCALE, x - px0 * SCALE);
                        for c in 0..3 {
                            out[(y * ow + x) * 3 + c] = (up[(sy * uw + sx) * 3 + c].clamp(0., 1.) * 255.).round() as u8;
                        }
                    }
                }
                done += 1;
                progress(done, tiles);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The network against basicsr's (the reference's crop, as [`crate::esrgan`]'s test: E:/p3dref/esrgan), and the
    /// time a 1024x1024 picture takes. `--ignored --nocapture`; OAIY_NO_COOP for a GPU without tensor cores.
    #[test]
    #[ignore = "needs Real-ESRGAN's weights and basicsr's reference"]
    fn matches_the_reference_on_webgpu() -> Result<()> {
        let dir = std::env::var("ESRGAN_REF").unwrap_or_else(|_| "E:/p3dref/esrgan".into());
        let read = |name: &str| -> Vec<f32> { std::fs::read(Path::new(&dir).join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let path = std::env::var("ESRGAN").unwrap_or_else(|_| "E:/models/Real-ESRGAN/RealESRGAN_x4plus.pth".into());
        let device: usize = std::env::var("OAIY_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let t = std::time::Instant::now();
        let net = WgpuEsrgan::load(Path::new(&path), device)?;
        eprintln!("loaded in {:.2} s", t.elapsed().as_secs_f64());
        let (h, w) = (160usize, 200usize);
        let input = read("input");
        let want = read("output");
        // the reference's planes as rows of pixels
        let rows: Vec<f32> = (0..h * w).flat_map(|i| (0..3).map(move |c| (c, i))).map(|(c, i)| input[c * h * w + i]).collect();
        let t = std::time::Instant::now();
        let got = net.forward(&rows, w, h)?;
        let seconds = t.elapsed().as_secs_f64();
        let (uh, uw) = (4 * h, 4 * w);
        let planes: Vec<f32> = (0..3).flat_map(|c| (0..uh * uw).map(move |i| (c, i))).map(|(c, i)| got[i * 3 + c]).collect();
        let d: f64 = planes.iter().zip(&want).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let n: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (d / n).sqrt();
        eprintln!("200x160 to 800x640: relative RMS {rel:.2e} against basicsr's, in {seconds:.2} s");
        assert!(rel < 5e-3, "relative RMS {rel}");
        let img = image::open(Path::new(&dir).join("crop.png")).map_err(candle_core::Error::wrap)?.to_rgb8();
        let big = image::imageops::resize(&img, 1024, 1024, image::imageops::FilterType::Triangle);
        let t = std::time::Instant::now();
        let up = net.upscale_with(big.as_raw(), 1024, 1024, |_, _| {})?;
        eprintln!("1024x1024 to 4096x4096 in {:.2} s", t.elapsed().as_secs_f64());
        assert_eq!(up.len(), 4096 * 4096 * 3);
        Ok(())
    }
}
