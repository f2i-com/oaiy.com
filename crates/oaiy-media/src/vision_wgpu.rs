//! Qwen3-VL's vision tower on WebGPU, as [`crate::vision::VisionEncoder::encode`] computes a reference image's
//! features: its 2x16x16 patches through the patch projection, the learned positions (the 48 x 48 grid's,
//! bilinearly, on the host), 27 blocks (layer norms, full attention rotated by each patch's row and column, the GELU
//! MLP), the merger of each 2 x 2 patches' states (and the three DeepStack mergers after blocks 8, 16 and 24). The
//! weights f16 (the BF16 checkpoint's rounded), the matmuls' inputs f32.
use crate::{
    reference::{patch_positions, Reference},
    vision::Features,
    weights::Weights,
};
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use std::path::Path;

const D: usize = 1152;
const HEADS: usize = 16;
const HD: usize = 72;
const FF: usize = 4304;
const BLOCKS: usize = 27;
/// A patch's values (two frames of 3 x 16 x 16), and four patches' states side by side.
const PATCH: usize = 1536;
const MERGED: usize = 4 * D;
const OUT: usize = 4096;
const EPS: f32 = 1e-6;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A linear layer: its weight as f16 and its bias.
struct Linear {
    w: DeviceVec,
    b: DeviceVec,
    n: usize,
    k: usize,
}

impl Linear {
    fn load(w: &mut Weights, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<Self> {
        let mut none = crate::lora::Loras::open(&[])?;
        let (v, n, k) = crate::wgpu_weights::f16_matrix(w, gpu, name, &mut none)?;
        let bias = w.tensor(&format!("{name}.bias"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = gpu.vec(n);
        gpu.upload(&b, &bias);
        Ok(Self { w: v, b, n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows, the inputs as f32.
    fn forward(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows_f32(&self.w, self.n, self.k, x, y, rows);
        r.add_bias_rows(y, &self.b, rows, self.n);
    }
}

/// A layer norm's weight less one, then its bias (a modulated norm's scale and shift).
fn norm(w: &mut Weights, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<(DeviceVec, usize)> {
    let mut v = w.tensor(&format!("{name}.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let n = v.len();
    v.iter_mut().for_each(|x| *x -= 1.);
    v.extend(w.tensor(&format!("{name}.bias"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
    let d = gpu.vec(v.len());
    gpu.upload(&d, &v);
    Ok((d, n))
}

struct Block {
    norm1: DeviceVec,
    norm2: DeviceVec,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}

/// A merger of each 2 x 2 patches' states: its norm (over a patch's before the merge, or over the four's after), its
/// MLP.
struct Merger {
    norm: (DeviceVec, usize),
    fc1: Linear,
    fc2: Linear,
}

impl Merger {
    fn load(w: &mut Weights, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self { norm: norm(w, gpu, &format!("{p}.norm"))?, fc1: Linear::load(w, gpu, &format!("{p}.linear_fc1"))?, fc2: Linear::load(w, gpu, &format!("{p}.linear_fc2"))? })
    }

    /// `x`'s `n` patches (four a merged row, as the patches run) through the merger into `out` (`[n / 4, 4096]`);
    /// `normed`, `h` and `g` its scratch.
    fn record(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, n: usize, normed: &DeviceVec, h: &DeviceVec, g: &DeviceVec, out: &DeviceVec) {
        let (mods, width) = (&self.norm.0, self.norm.1);
        let rows = n * D / width;
        r.norm_mod_rows(x, normed, rows, width, mods, 0, Some(width), RowNorm::Layer, EPS);
        self.fc1.forward(r, normed, h, n / 4);
        r.gelu_erf(h, g, n / 4 * MERGED);
        self.fc2.forward(r, g, out, n / 4);
    }
}

pub struct WgpuVisionEncoder {
    gpu: ggml_rs_wgpu::WgpuBackend,
    patch: DeviceVec,
    patch_bias: DeviceVec,
    /// The learned positions' 48 x 48 grid (`[2304, 1152]`), on the host.
    pos: Vec<f32>,
    blocks: Vec<Block>,
    merger: Merger,
    deep: Vec<Merger>,
}

impl WgpuVisionEncoder {
    /// The vision tower in `root` (the text encoder's folder) on GPU `device` (as CUDA counts them;
    /// OAIY_WEBGPU_ADAPTER naming one instead).
    pub fn load(root: &Path, device: usize) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(root)?;
        let p = "model.visual";
        let patch_w = w.tensor(&format!("{p}.patch_embed.proj.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        if patch_w.len() != D * PATCH {
            candle_core::bail!("not Qwen3-VL's vision tower (a patch projection of {} values)", patch_w.len());
        }
        let words = crate::wgpu_weights::f16_words_f32(&patch_w).ok_or_else(|| err("the patch projection past f16's range"))?;
        let patch = gpu.vec(words.len());
        gpu.upload(&patch, &words);
        let bias = w.tensor(&format!("{p}.patch_embed.proj.bias"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let patch_bias = gpu.vec(D);
        gpu.upload(&patch_bias, &bias);
        let pos = w.tensor(&format!("{p}.pos_embed.weight"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
                norm1: norm(&mut w, &gpu, &format!("{b}.norm1"))?.0,
                norm2: norm(&mut w, &gpu, &format!("{b}.norm2"))?.0,
                qkv: Linear::load(&mut w, &gpu, &format!("{b}.attn.qkv"))?,
                proj: Linear::load(&mut w, &gpu, &format!("{b}.attn.proj"))?,
                fc1: Linear::load(&mut w, &gpu, &format!("{b}.mlp.linear_fc1"))?,
                fc2: Linear::load(&mut w, &gpu, &format!("{b}.mlp.linear_fc2"))?,
            });
        }
        if blocks[0].qkv.n != 3 * D || blocks[0].fc1.n != FF || pos.len() != 2304 * D {
            candle_core::bail!("not Qwen3-VL's vision tower (qkv {}, MLP {}, positions {})", blocks[0].qkv.n, blocks[0].fc1.n, pos.len());
        }
        let merger = Merger::load(&mut w, &gpu, &format!("{p}.merger"))?;
        let deep = (0..3).map(|i| Merger::load(&mut w, &gpu, &format!("{p}.deepstack_merger_list.{i}"))).collect::<Result<Vec<_>>>()?;
        Ok(Self { gpu, patch, patch_bias, pos, blocks, merger, deep })
    }

    /// `image`'s features (as [`crate::vision::VisionEncoder::encode`] gives them), on the CPU.
    pub fn encode(&self, image: &Reference) -> Result<Features> {
        let (h, w) = (image.h / 16, image.w / 16);
        let n = h * w;
        if h < 2 || w < 2 || h % 2 != 0 || w % 2 != 0 {
            candle_core::bail!("a reference of {} by {} pixels: not whole 2 x 2 patches of 16", image.w, image.h);
        }
        // each patch's learned position (the 48 x 48 grid's four nearest, bilinearly) and its rotary pairs (its row's
        // 18, then its column's), as the patches run
        let positions = patch_positions(h, w);
        let mut pos = vec![0f32; n * D];
        let mut table = Vec::with_capacity(n * HD);
        for (i, &(y, x)) in positions.iter().enumerate() {
            let (yy, xx) = (y as f32 * 47. / (h - 1) as f32, x as f32 * 47. / (w - 1) as f32);
            let (y0, x0) = (yy.floor() as usize, xx.floor() as usize);
            let (dy, dx) = (yy - y0 as f32, xx - x0 as f32);
            let row = &mut pos[i * D..(i + 1) * D];
            for (py, px, f) in [(y0, x0, (1. - dy) * (1. - dx)), (y0, (x0 + 1).min(47), (1. - dy) * dx), ((y0 + 1).min(47), x0, dy * (1. - dx)), ((y0 + 1).min(47), (x0 + 1).min(47), dy * dx)] {
                let grid = &self.pos[(py * 48 + px) * D..(py * 48 + px + 1) * D];
                row.iter_mut().zip(grid).for_each(|(a, g)| *a += g * f);
            }
            for p in [y, x] {
                for j in 0..HD / 4 {
                    let a = p as f64 / 10000f64.powf(j as f64 / (HD / 4) as f64);
                    table.extend([a.sin() as f32, a.cos() as f32]);
                }
            }
        }
        let patches = image.patches(&Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v = |len: usize| self.gpu.vec(len.max(1));
        let (pd, posd, td) = (v(n * PATCH), v(n * D), v(n * HD));
        self.gpu.upload(&pd, &patches);
        self.gpu.upload(&posd, &pos);
        self.gpu.upload(&td, &table);
        let (x, normed, qkv, q, k, vv, kv, o) = (v(n * D), v(n * D), v(n * 3 * D), v(n * D), v(n * D), v(n * D), v(n * 2 * D), v(n * D));
        let att = v(self.gpu.attention_rows_full_out_len(n, HEADS, HD, n));
        let (f, fg) = (v(n * FF), v(n * FF));
        let (mh, mg) = (v(n / 4 * MERGED), v(n / 4 * MERGED));
        let deep: Vec<DeviceVec> = (0..3).map(|_| v(n / 4 * OUT)).collect();
        let embedding = v(n / 4 * OUT);
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        r.matmul_f16_rows_f32(&self.patch, D, PATCH, &pd, &x, n);
        r.add_bias_rows(&x, &self.patch_bias, n, D);
        r.add(&x, &posd);
        rec.finish();
        let scale = 1.0 / (HD as f32).sqrt();
        // a recording a block (each's ops' own scratch let go before the next's)
        for (i, b) in self.blocks.iter().enumerate() {
            let mut rec = self.gpu.begin();
            rec.keep_groups(false);
            let r = rec.as_mut();
            r.norm_mod_rows(&x, &normed, n, D, &b.norm1, 0, Some(D), RowNorm::Layer, EPS);
            b.qkv.forward(r, &normed, &qkv, n);
            r.copy_cols(&qkv, &q, n, D, 3 * D, 0);
            r.copy_cols(&qkv, &k, n, D, 3 * D, D);
            r.copy_cols(&qkv, &vv, n, D, 3 * D, 2 * D);
            r.rope_rows(&q, n, HEADS, HD, &td, true);
            r.rope_rows(&k, n, HEADS, HD, &td, true);
            r.store_rows(&k, &kv, n, D, 0, 2 * D, 0);
            r.store_rows(&vv, &kv, n, D, 0, 2 * D, D);
            r.attention_rows_full(&q, &kv, &att, n, HEADS, HEADS, HD, n, scale);
            b.proj.forward(r, &att, &o, n);
            r.add(&x, &o);
            r.norm_mod_rows(&x, &normed, n, D, &b.norm2, 0, Some(D), RowNorm::Layer, EPS);
            b.fc1.forward(r, &normed, &f, n);
            r.gelu(&f, &fg, n * FF);
            b.fc2.forward(r, &fg, &o, n);
            r.add(&x, &o);
            if let Some(j) = [8, 16, 24].iter().position(|&l| l == i) {
                self.deep[j].record(r, &x, n, &mh, &mg, &f, &deep[j]);
            }
            rec.finish();
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        self.merger.record(r, &x, n, &normed, &mh, &mg, &embedding);
        r.read(&embedding);
        for d in &deep {
            r.read(d);
        }
        let mut reads = rec.finish().into_iter();
        let mut tensor = || -> Result<Tensor> { Tensor::from_vec(reads.next().ok_or_else(|| err("a feature was not read"))?, (1, n / 4, OUT), &Device::Cpu) };
        let embedding = tensor()?;
        let deep = (0..3).map(|_| tensor()).collect::<Result<Vec<_>>>()?;
        Ok(Features { embedding, deep, h: h / 2, w: w / 2 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WebGPU vision tower gives Candle's features (CPU, f32) of a reference image (`OAIY_QWEN_IMAGE_REFERENCE`,
    /// at 512 by 512) from Qwen Image 2.1's text encoder (`OAIY_QWEN_IMAGE_BASE`): its embedding's and its three
    /// deeper features' rows.
    #[test]
    #[ignore = "needs Qwen Image 2.1 (OAIY_QWEN_IMAGE_BASE), a reference image (OAIY_QWEN_IMAGE_REFERENCE) and a WebGPU adapter"]
    fn the_webgpu_vision_tower_is_the_candle_one() -> Result<()> {
        let (Some(base), Some(image)) = (std::env::var_os("OAIY_QWEN_IMAGE_BASE").map(std::path::PathBuf::from), std::env::var_os("OAIY_QWEN_IMAGE_REFERENCE").map(std::path::PathBuf::from)) else { return Ok(()) };
        let reference = Reference::load(&image, 512)?;
        let t = std::time::Instant::now();
        let gpu = WgpuVisionEncoder::load(&base.join("text_encoder"), 0)?;
        let loaded = t.elapsed().as_secs_f64();
        let got = gpu.encode(&reference)?;
        eprintln!("WebGPU vision tower: loaded {loaded:.1} s, encoded {:.2} s", t.elapsed().as_secs_f64() - loaded);
        drop(gpu);
        let cpu = crate::vision::VisionEncoder::load(&base.join("text_encoder"), &Device::Cpu, DType::F32)?;
        let want = cpu.encode(&reference)?;
        assert_eq!((got.h, got.w), (want.h, want.w));
        let rows = |a: &Tensor, b: &Tensor| -> Result<f64> {
            let (a, b) = (a.flatten_all()?.to_vec1::<f32>()?, b.flatten_all()?.to_vec1::<f32>()?);
            let mut worst = 1f64;
            for (x, y) in a.chunks(OUT).zip(b.chunks(OUT)) {
                let dot: f64 = x.iter().zip(y).map(|(p, q)| *p as f64 * *q as f64).sum();
                let nx = x.iter().map(|p| (*p as f64).powi(2)).sum::<f64>().sqrt();
                let ny = y.iter().map(|q| (*q as f64).powi(2)).sum::<f64>().sqrt();
                worst = worst.min(dot / (nx * ny));
            }
            Ok(worst)
        };
        let e = rows(&got.embedding, &want.embedding)?;
        eprintln!("the embedding's worst row: cosine {e:.6}");
        assert!(e > 0.999, "the embedding: cosine {e}");
        for (i, (g, w)) in got.deep.iter().zip(&want.deep).enumerate() {
            let c = rows(g, w)?;
            eprintln!("deeper features {i}: the worst row's cosine {c:.6}");
            assert!(c > 0.999, "deeper features {i}: cosine {c}");
        }
        Ok(())
    }
}
