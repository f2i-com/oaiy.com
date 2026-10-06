//! BiRefNet on WebGPU (a picture job's background removal with `backend` "webgpu": any GPU, with tensor cores or
//! without): [`crate::birefnet`]'s network, its weights as the Candle model folds them (each BatchNorm into the
//! convolution before it), its features as rows of pixels (`[h * w, c]`). The Swin-L backbone's windows gathered and
//! put back ([`ChainRecorder::window_rows`], [`ChainRecorder::unwindow_add_rows`]), their attention a kernel of its own
//! (heads 32 wide, the relative positions' bias table and the shifted windows' mask in it); a patch merge as space to
//! depth with its norm's and reduction's channels permuted to match; the decoder's modulated deformable convolutions as
//! their taps ([`ChainRecorder::deform_im2col_rows`]) times their weights, a chunk of pixels at a time.
use crate::birefnet::{BiRefNet, SIZE, WINDOW};
use candle_core::{DType, Device, Result, Tensor};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs_wgpu::WgpuBackend;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("BiRefNet on WebGPU: {e}"))
}

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// A linear layer: its f16 weight (`[n, k]`) and bias.
struct Lin {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

/// A convolution ([`DeviceChain::conv_weights`]'s weights) and its bias.
struct Conv {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
}

/// A modulated deformable convolution: its offsets' and modulators' convolutions, and its weight (f16 `[cout, k² cin]`,
/// taps outer) and bias.
struct Deform {
    offset: Conv,
    modulator: Conv,
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
}

struct Aspp {
    branches: Vec<Deform>,
    pool: Conv,
    out: Conv,
}

struct Dec {
    conv_in: Conv,
    att: Aspp,
    conv_out: Conv,
}

struct Block {
    /// The norms as modulations (`[w - 1, b]`, [`RowNorm::Layer`]).
    norm1: DeviceVec,
    qkv: Lin,
    proj: Lin,
    /// The relative positions' bias table (`[(2 WINDOW - 1)², heads]`).
    table: DeviceVec,
    norm2: DeviceVec,
    fc1: Lin,
    fc2: Lin,
    heads: usize,
    shift: usize,
}

struct Stage {
    blocks: Vec<Block>,
    norm: DeviceVec,
    c: usize,
    /// The patch merge's norm and reduction, their channels in space to depth's order.
    merge: Option<(DeviceVec, Lin)>,
}

/// A level of features: its rows, `h`, `w` and channels.
type Level = (DeviceVec, usize, usize, usize);

pub struct WgpuBiRefNet {
    gpu: WgpuBackend,
    /// The patch embedding (a 4x4 convolution of stride 4) as a matmul of space to depth's rows.
    patch: Lin,
    patch_norm: DeviceVec,
    stages: Vec<Stage>,
    squeeze: Dec,
    blocks: Vec<Dec>,
    lateral: Vec<Conv>,
    gates: Vec<(Conv, Conv)>,
    inputs: Vec<(Conv, Conv)>,
    out: Conv,
    /// `[1.0]` (a plain sum's weight), and 2.0s (the modulators' scale) enough for the largest.
    one: DeviceVec,
    twos: DeviceVec,
}

impl WgpuBiRefNet {
    /// From a BiRefNet folder (as [`BiRefNet::load`] takes it) on WebGPU device `device`.
    pub fn load(dir: &Path, device: usize) -> Result<Self> {
        let net = BiRefNet::load(dir, &Device::Cpu)?;
        let gpu = WgpuBackend::nth(device, None).map_err(err)?;
        let file = if dir.is_file() { dir.to_path_buf() } else { dir.join("model.safetensors") };
        let mut store = crate::ltx::store::Store::open(&file, 0)?;
        let g = &gpu;
        let f16 = |v: &[f32], what: &str| -> Result<DeviceVec> { g.vec_f16_rounded(v).ok_or_else(|| err(format!("{what}: past f16's range"))) };
        let f32v = |v: &[f32]| -> DeviceVec {
            let d = g.vec(v.len().max(1));
            g.upload(&d, v);
            d
        };
        let lin = |l: &crate::birefnet::Linear, what: &str| -> Result<Lin> {
            let (n, k) = l.w.dims2()?;
            Ok(Lin { w: f16(&values(&l.w)?, what)?, b: l.b.as_ref().map(|b| values(b).map(|v| f32v(&v))).transpose()?, n, k })
        };
        let mods = |norm: &(Tensor, Tensor)| -> Result<DeviceVec> {
            let (w, b) = (values(&norm.0)?, values(&norm.1)?);
            Ok(f32v(&w.iter().map(|v| v - 1.0).chain(b).collect::<Vec<_>>()))
        };
        let conv = |c: &crate::birefnet::Conv, what: &str| -> Result<Conv> {
            let (cout, cin, k, _) = c.w.dims4()?;
            let w = g.conv_weights(&values(&c.w)?, cout, cin, k).ok_or_else(|| err(format!("{what}: a {k}x{k} convolution's weights")))?;
            let b = match &c.b {
                Some(b) => values(b)?,
                None => vec![0.0; cout],
            };
            Ok(Conv { w, b: f32v(&b), cin, cout, k })
        };
        let deform = |d: &crate::birefnet::Deform, what: &str| -> Result<Deform> {
            let (cout, per) = d.w.dims2()?;
            let cin = per / (d.k * d.k);
            Ok(Deform { offset: conv(&d.offset, what)?, modulator: conv(&d.modulator, what)?, w: f16(&values(&d.w)?, what)?, b: f32v(&values(&d.b)?), cin, cout, k: d.k })
        };
        let dec = |d: &crate::birefnet::DecBlk, what: &str| -> Result<Dec> {
            Ok(Dec {
                conv_in: conv(&d.conv_in, what)?,
                att: Aspp { branches: d.att.branches.iter().map(|b| deform(b, what)).collect::<Result<_>>()?, pool: conv(&d.att.pool, what)?, out: conv(&d.att.out, what)? },
                conv_out: conv(&d.conv_out, what)?,
            })
        };
        let (pw, pb) = (values(&net.patch.w)?, net.patch.b.as_ref().map(values).transpose()?);
        let patch = Lin { w: f16(&pw, "the patch embedding")?, b: pb.map(|b| f32v(&b)), n: 192, k: 48 };
        let mut stages = Vec::new();
        for (s, st) in net.stages.iter().enumerate() {
            let c = 192 << s;
            let mut blocks = Vec::new();
            for (i, b) in st.blocks.iter().enumerate() {
                let what = format!("stage {s} block {i}");
                let table = values(&store.tensor_f32(&format!("bb.layers.{s}.blocks.{i}.attn.relative_position_bias_table"), &Device::Cpu)?)?;
                blocks.push(Block { norm1: mods(&b.norm1)?, qkv: lin(&b.qkv, &what)?, proj: lin(&b.proj, &what)?, table: f32v(&table), norm2: mods(&b.norm2)?, fc1: lin(&b.fc1, &what)?, fc2: lin(&b.fc2, &what)?, heads: b.heads, shift: b.shift });
            }
            let merge = match &st.merge {
                // Swin's merge joins (0, 0), (1, 0), (0, 1), (1, 1) a channel's whole run each; space to depth puts a
                // channel's four together (row, then column): the same channels in another order
                Some((norm, red)) => {
                    let swin = |e: usize| ((e % 2) * 2 + (e / 2) % 2) * c + e / 4;
                    let (nw, nb) = (values(&norm.0)?, values(&norm.1)?);
                    let m: Vec<f32> = (0..4 * c).map(|e| nw[swin(e)] - 1.0).chain((0..4 * c).map(|e| nb[swin(e)])).collect();
                    let (n, k) = red.w.dims2()?;
                    let w = values(&red.w)?;
                    let wp: Vec<f32> = (0..n * k).map(|i| w[(i / k) * k + swin(i % k)]).collect();
                    Some((f32v(&m), Lin { w: f16(&wp, "a patch merge")?, b: red.b.as_ref().map(|b| values(b).map(|v| f32v(&v))).transpose()?, n, k }))
                }
                None => None,
            };
            stages.push(Stage { blocks, norm: mods(&st.norm)?, c, merge });
        }
        let pair = |p: &(crate::birefnet::Conv, crate::birefnet::Conv), what: &str| -> Result<(Conv, Conv)> { Ok((conv(&p.0, what)?, conv(&p.1, what)?)) };
        Ok(Self {
            patch,
            patch_norm: mods(&net.patch_norm)?,
            stages,
            squeeze: dec(&net.squeeze, "the squeeze")?,
            blocks: net.blocks.iter().map(|b| dec(b, "a decoder block")).collect::<Result<_>>()?,
            lateral: net.lateral.iter().map(|c| conv(c, "a lateral")).collect::<Result<_>>()?,
            gates: net.gates.iter().map(|p| pair(p, "a gate")).collect::<Result<_>>()?,
            inputs: net.inputs.iter().map(|p| pair(p, "an input block")).collect::<Result<_>>()?,
            out: conv(&net.out, "the output")?,
            one: f32v(&[1.0]),
            twos: f32v(&vec![2.0; (SIZE / 4) * (SIZE / 4) * 49]),
            gpu,
        })
    }

    /// `y = W x + b` with f32 activations (the tensor cores' f16 ones put Swin-L's first level 4e-3 off BiRefNet's,
    /// where f32's 2e-6: its residual stream's large values).
    fn lin(&self, r: &mut dyn ChainRecorder, l: &Lin, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows_f32(&l.w, l.n, l.k, x, y, rows);
        if let Some(b) = &l.b {
            r.add_bias_rows(y, b, rows, l.n);
        }
    }

    fn conv(&self, r: &mut dyn ChainRecorder, c: &Conv, x: &DeviceVec, h: usize, w: usize) -> DeviceVec {
        let y = self.gpu.vec(h * w * c.cout);
        r.conv_rows(&c.w, &c.b, c.cout, c.cin, c.k, x, h, w, &y);
        y
    }

    /// A layer norm of `rows` of `n` (`mods` its weight less one, then its bias).
    fn norm(r: &mut dyn ChainRecorder, mods: &DeviceVec, x: &DeviceVec, out: &DeviceVec, rows: usize, n: usize) {
        r.norm_mod_rows(x, out, rows, n, mods, 0, Some(n), RowNorm::Layer, 1e-5);
    }

    fn relu(r: &mut dyn ChainRecorder, x: &DeviceVec, len: usize) {
        r.leaky_relu(x, x, len, 0.0);
    }

    /// The backbone's four levels of a picture `s` a side (`[s², 3]`, normalized).
    fn backbone(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, s: usize) -> Vec<Level> {
        let g = &self.gpu;
        let (mut h, mut w) = (s / 4, s / 4);
        let sd = g.vec(h * w * 48);
        r.space_to_depth_rows(x, &sd, s, s, 3, 1, 4, 4);
        let e = g.vec(h * w * 192);
        self.lin(r, &self.patch, &sd, &e, h * w);
        let mut t = g.vec(h * w * 192);
        Self::norm(r, &self.patch_norm, &e, &t, h * w, 192);
        let mut levels = Vec::new();
        for st in &self.stages {
            let c = st.c;
            let m = h * w;
            let rows = h.div_ceil(WINDOW) * WINDOW * w.div_ceil(WINDOW) * WINDOW;
            let (n1, win, qkv, att, pw, hid, mo) = (g.vec(m * c), g.vec(rows * c), g.vec(rows * 3 * c), g.vec(rows * c), g.vec(rows * c), g.vec(m * 4 * c), g.vec(m * c));
            let act = g.vec(m * 4 * c);
            for b in &st.blocks {
                Self::norm(r, &b.norm1, &t, &n1, m, c);
                r.window_rows(&n1, &win, h, w, c, WINDOW, b.shift);
                self.lin(r, &b.qkv, &win, &qkv, rows);
                r.window_attention(&qkv, &b.table, &att, h, w, b.heads, WINDOW, b.shift, 1.0 / 32f32.sqrt());
                self.lin(r, &b.proj, &att, &pw, rows);
                r.unwindow_add_rows(&pw, &t, h, w, c, WINDOW, b.shift);
                Self::norm(r, &b.norm2, &t, &n1, m, c);
                self.lin(r, &b.fc1, &n1, &hid, m);
                r.gelu_erf(&hid, &act, m * 4 * c);
                self.lin(r, &b.fc2, &act, &mo, m);
                r.axpy_at(&t, &mo, &self.one, 0, m * c);
            }
            let level = g.vec(m * c);
            Self::norm(r, &st.norm, &t, &level, m, c);
            levels.push((level, h, w, c));
            if let Some((mods, red)) = &st.merge {
                assert!(h % 2 == 0 && w % 2 == 0, "BiRefNet on WebGPU: a merge of {h}x{w}");
                let (h2, w2) = (h / 2, w / 2);
                let (m4, mn, t2) = (g.vec(h2 * w2 * 4 * c), g.vec(h2 * w2 * 4 * c), g.vec(h2 * w2 * 2 * c));
                r.space_to_depth_rows(&t, &m4, h, w, c, 1, 2, 2);
                Self::norm(r, mods, &m4, &mn, h2 * w2, 4 * c);
                self.lin(r, red, &mn, &t2, h2 * w2);
                t = t2;
                (h, w) = (h2, w2);
            }
        }
        levels
    }

    /// The four encoder levels: the backbone at full and half size, joined, the top one with its context.
    fn encode(&self, r: &mut dyn ChainRecorder, x: &DeviceVec) -> Vec<Level> {
        let g = &self.gpu;
        let full = self.backbone(r, x, SIZE);
        let xh = g.vec(SIZE * SIZE / 4 * 3);
        r.resize_bilinear_rows(x, &xh, SIZE, SIZE, 3, SIZE / 2, SIZE / 2);
        let half = self.backbone(r, &xh, SIZE / 2);
        let mut levels: Vec<Level> = Vec::new();
        for ((f, h, w, c), (hf, hh, hw, _)) in full.iter().zip(&half) {
            let (cat, up) = (g.vec(h * w * 2 * c), g.vec(h * w * c));
            r.store_rows(f, &cat, h * w, *c, 0, 2 * c, 0);
            r.resize_bilinear_rows(hf, &up, *hh, *hw, *c, *h, *w);
            r.store_rows(&up, &cat, h * w, *c, 0, 2 * c, *c);
            levels.push((cat, *h, *w, 2 * c));
        }
        let (h4, w4) = (levels[3].1, levels[3].2);
        let width: usize = levels.iter().map(|l| l.3).sum();
        let ctx = g.vec(h4 * w4 * width);
        let mut at = 0;
        for (l, h, w, c) in &levels {
            if (*h, *w) == (h4, w4) {
                r.store_rows(l, &ctx, h4 * w4, *c, 0, width, at);
            } else {
                let down = g.vec(h4 * w4 * c);
                r.resize_bilinear_rows(l, &down, *h, *w, *c, h4, w4);
                r.store_rows(&down, &ctx, h4 * w4, *c, 0, width, at);
            }
            at += c;
        }
        levels[3] = (ctx, h4, w4, width);
        levels
    }

    fn deform(&self, r: &mut dyn ChainRecorder, d: &Deform, x: &DeviceVec, h: usize, w: usize) -> DeviceVec {
        let g = &self.gpu;
        let (m, kk) = (h * w, d.k * d.k);
        let offsets = self.conv(r, &d.offset, x, h, w);
        let mconv = self.conv(r, &d.modulator, x, h, w);
        let mods = g.vec(m * kk);
        r.mul_sigmoid(&self.twos, &mconv, &mods, m * kk);
        let out = g.vec(m * d.cout);
        let per = kk * d.cin;
        // (the taps a chunk of pixels at a time: a 7x7 of 64 channels at 256x256, 822 MB at once)
        let chunk = ((1 << 24) / per).clamp(1, m);
        let (taps, part) = (g.vec(chunk * per), g.vec(chunk * d.cout));
        let mut first = 0;
        while first < m {
            let n = chunk.min(m - first);
            r.deform_im2col_rows(x, &offsets, &mods, &taps, h, w, d.cin, d.k, first, n);
            r.matmul_f16_rows_f32(&d.w, d.cout, per, &taps, &part, n);
            r.copy(&part, 0, &out, first * d.cout, n * d.cout);
            first += n;
        }
        r.add_bias_rows(&out, &d.b, m, d.cout);
        out
    }

    fn aspp(&self, r: &mut dyn ChainRecorder, a: &Aspp, x: &DeviceVec, h: usize, w: usize) -> DeviceVec {
        let g = &self.gpu;
        let m = h * w;
        let each = a.pool.cout;
        let width = (a.branches.len() + 1) * each;
        let cat = g.vec(m * width);
        for (i, d) in a.branches.iter().enumerate() {
            let y = self.deform(r, d, x, h, w);
            Self::relu(r, &y, m * d.cout);
            r.store_rows(&y, &cat, m, d.cout, 0, width, i * each);
        }
        let mean = g.vec(a.pool.cin);
        r.mean_rows(x, &mean, m, a.pool.cin);
        let pooled = self.conv(r, &a.pool, &mean, 1, 1);
        Self::relu(r, &pooled, each);
        r.broadcast_rows(&pooled, &cat, m, each, width, a.branches.len() * each);
        let y = self.conv(r, &a.out, &cat, h, w);
        Self::relu(r, &y, m * a.out.cout);
        y
    }

    fn dec(&self, r: &mut dyn ChainRecorder, d: &Dec, x: &DeviceVec, h: usize, w: usize) -> DeviceVec {
        let a = self.conv(r, &d.conv_in, x, h, w);
        Self::relu(r, &a, h * w * d.conv_in.cout);
        let b = self.aspp(r, &d.att, &a, h, w);
        self.conv(r, &d.conv_out, &b, h, w)
    }

    /// Input block `i` of the picture laid out as patches `size` a side.
    fn input(&self, r: &mut dyn ChainRecorder, i: usize, x: &DeviceVec, size: usize) -> DeviceVec {
        let g = SIZE / size;
        let p = self.gpu.vec(size * size * 3 * g * g);
        r.blocks_to_channels_rows(x, &p, SIZE, 3, size);
        let (a, b) = &self.inputs[i];
        let t = self.conv(r, a, &p, size, size);
        self.conv(r, b, &t, size, size)
    }

    /// `a` and `b` (rows of `ca` and `cb` channels) side by side.
    fn join(&self, r: &mut dyn ChainRecorder, a: &DeviceVec, ca: usize, b: &DeviceVec, cb: usize, rows: usize) -> DeviceVec {
        let out = self.gpu.vec(rows * (ca + cb));
        r.store_rows(a, &out, rows, ca, 0, ca + cb, 0);
        r.store_rows(b, &out, rows, cb, 0, ca + cb, ca);
        out
    }

    /// The logits (`[SIZE²]`) of a picture `[SIZE², 3]`, ImageNet-normalized.
    pub fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
        if x.len() != SIZE * SIZE * 3 {
            return Err(err(format!("a picture of {} values", x.len())));
        }
        let g = &self.gpu;
        let xd = g.vec(x.len());
        g.upload(&xd, x);
        let mut rec = g.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let levels = self.encode(r, &xd);
        let (l3, h3, w3, _) = &levels[3];
        let top = self.dec(r, &self.squeeze, l3, *h3, *w3);
        let ct = self.squeeze.conv_out.cout;
        let i0 = self.input(r, 0, &xd, *h3);
        let mut p = self.join(r, &top, ct, &i0, self.inputs[0].1.cout, h3 * w3);
        let (mut ph, mut pw) = (*h3, *w3);
        for i in 0..3 {
            let q = self.dec(r, &self.blocks[i], &p, ph, pw);
            let cq = self.blocks[i].conv_out.cout;
            let (gate, attn) = &self.gates[i];
            let gt = self.conv(r, gate, &q, ph, pw);
            Self::relu(r, &gt, ph * pw * gate.cout);
            let at = self.conv(r, attn, &gt, ph, pw);
            r.mul_sigmoid_rows(&q, &at, ph * pw, cq);
            let (skip, sh, sw, _) = &levels[2 - i];
            let up = g.vec(sh * sw * cq);
            r.resize_bilinear_rows(&q, &up, ph, pw, cq, *sh, *sw);
            let lat = self.conv(r, &self.lateral[i], skip, *sh, *sw);
            r.axpy_at(&up, &lat, &self.one, 0, sh * sw * cq);
            let inp = self.input(r, i + 1, &xd, *sh);
            p = self.join(r, &up, cq, &inp, self.inputs[i + 1].1.cout, sh * sw);
            (ph, pw) = (*sh, *sw);
        }
        let q = self.dec(r, &self.blocks[3], &p, ph, pw);
        let cq = self.blocks[3].conv_out.cout;
        let up = g.vec(SIZE * SIZE * cq);
        r.resize_bilinear_rows(&q, &up, ph, pw, cq, SIZE, SIZE);
        let inp = self.input(r, 4, &xd, SIZE);
        let last = self.join(r, &up, cq, &inp, self.inputs[4].1.cout, SIZE * SIZE);
        let logits = self.conv(r, &self.out, &last, SIZE, SIZE);
        r.read(&logits);
        rec.finish().pop().ok_or_else(|| err("the logits were not read"))
    }

    /// The object's matte in an RGB picture (`w` by `h`, 8-bit), as [`BiRefNet::matte`] makes it.
    pub fn matte(&self, rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
        let small = oaiy_image::resize::resample(rgb, 3, w, h, SIZE, SIZE, oaiy_image::resize::Filter::Bilinear);
        // ImageNet's normalization, as crate::model3d::prepare::imagenet
        let (mean, std) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
        let x: Vec<f32> = small.iter().enumerate().map(|(i, &v)| (v as f32 / 255. - mean[i % 3]) / std[i % 3]).collect();
        let logits = self.logits(&x)?;
        let mask: Vec<u8> = logits.into_iter().map(|l| (255. / (1. + (-l).exp())) as u8).collect();
        Ok(oaiy_image::resize::resample(&mask, 1, SIZE, SIZE, w, h, oaiy_image::resize::Filter::Bicubic))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("BIREFNET_REF").unwrap_or_else(|_| "E:/p3dref/birefnet".into()))
    }

    /// A reference's values (`[c, h, w]` planes, its batch of one dropped).
    fn reference(name: &str) -> Vec<f32> {
        std::fs::read(dir().join(format!("{name}.bin"))).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Rows of pixels (`[h w, c]`) as planes.
    fn planes(rows: &[f32], c: usize) -> Vec<f32> {
        let m = rows.len() / c;
        (0..c * m).map(|i| rows[(i % m) * c + i / m]).collect()
    }

    fn rel(name: &str, a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len(), "{name}: sizes differ");
        let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let n: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
        let r = (d / n.max(1e-30)).sqrt();
        eprintln!("{name}: relative RMS error {r:.3e}");
        r
    }

    /// The backbone's levels and the logits against BiRefNet's own (the Candle test's reference: E:/p3dref/birefnet),
    /// and the mask. `--ignored --nocapture`; OAIY_NO_COOP for a GPU without tensor cores.
    #[test]
    #[ignore = "needs BiRefNet's weights and its reference"]
    fn matches_the_reference_on_webgpu() -> Result<()> {
        let device: usize = std::env::var("OAIY_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let t = std::time::Instant::now();
        let net = WgpuBiRefNet::load(Path::new(&std::env::var("BIREFNET").unwrap_or_else(|_| "E:/models/BiRefNet".into())), device)?;
        eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
        // the reference's input as rows of pixels
        let input = reference("input");
        let x: Vec<f32> = (0..SIZE * SIZE * 3).map(|i| input[(i % 3) * SIZE * SIZE + i / 3]).collect();
        let g = &net.gpu;
        let xd = g.vec(x.len());
        g.upload(&xd, &x);
        let mut rec = g.begin();
        rec.keep_groups(false);
        let levels = net.backbone(rec.as_mut(), &xd, SIZE);
        for (l, ..) in &levels {
            rec.read(l);
        }
        let got = rec.finish();
        for (i, (rows, (.., c))) in got.iter().zip(&levels).enumerate() {
            assert!(rel(&format!("bb_{}", i + 1), &planes(rows, *c), &reference(&format!("bb_{}", i + 1))) < 2e-3);
        }
        let t = std::time::Instant::now();
        let logits = net.logits(&x)?;
        eprintln!("logits in {:.2} s", t.elapsed().as_secs_f64());
        let want = reference("logits");
        assert!(rel("logits", &logits, &want) < 5e-3);
        let differ = logits.iter().zip(&want).filter(|(a, b)| (**a > 0.0) != (**b > 0.0)).count();
        eprintln!("mask pixels on the other side of 0.5: {differ} of {}", want.len());
        assert!(differ * 10_000 < want.len());
        let t = std::time::Instant::now();
        let _ = net.logits(&x)?;
        eprintln!("again in {:.2} s", t.elapsed().as_secs_f64());
        Ok(())
    }
}
