//! Pixal3D on WebGPU (a 3D job's `backend` "webgpu": any GPU; Candle's needs CUDA): [`crate::model3d`]'s flow
//! transformers, the voxels' features as rows. A block's self-attention over the voxels (each head's q and k RMS-normed
//! by its own weights, interleaved 3-D RoPE), its cross-attention to the picture's DINOv3 tokens plus the voxel's own
//! back-projected features, and its tanh-GELU MLP, modulated by the timestep through the shared adaLN (on the host)
//! plus each block's offset; the weights f16 (the checkpoints' F32 rounded: the reference runs the blocks in BF16), the
//! input and output layers F32.
use crate::ltx::store::Store;
use crate::model3d::dit::Config;
use crate::wgpu_weights::f16_words_f32;
use candle_core::{Device, Result};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs_wgpu::WgpuBackend;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("3D on WebGPU: {e}"))
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

/// A tensor's F32 values.
fn f32s(s: &mut Store, key: &str) -> Result<Vec<f32>> {
    s.tensor_f32(key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
}

/// A linear layer: its weight `[n, k]` (f16, or F32) and bias.
struct Lin {
    w: DeviceVec,
    b: DeviceVec,
    n: usize,
    k: usize,
    f32: bool,
}

impl Lin {
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str, f32: bool) -> Result<Self> {
        let t = s.tensor_f32(&format!("{prefix}.weight"), &Device::Cpu)?;
        let (n, k) = t.dims2()?;
        let w = t.flatten_all()?.to_vec1::<f32>()?;
        let w = if f32 { upload(gpu, &w) } else { upload(gpu, &f16_words_f32(&w).ok_or_else(|| err(format!("{prefix}: past f16's range")))?) };
        Ok(Self { w, b: upload(gpu, &f32s(s, &format!("{prefix}.bias"))?), n, k, f32 })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        if self.f32 {
            r.matmul_f32_rows(&self.w, self.n, self.k, x, y, rows);
        } else {
            r.matmul_f16_rows(&self.w, self.n, self.k, x, y, rows);
        }
        r.add_bias_rows(y, &self.b, rows, self.n);
    }
}

/// A host linear layer in F32 (the timestep's path).
struct Host {
    w: Vec<f32>,
    b: Vec<f32>,
    n: usize,
    k: usize,
}

impl Host {
    fn load(s: &mut Store, prefix: &str) -> Result<Self> {
        let w = s.tensor_f32(&format!("{prefix}.weight"), &Device::Cpu)?;
        let (n, k) = w.dims2()?;
        Ok(Self { w: w.flatten_all()?.to_vec1::<f32>()?, b: f32s(s, &format!("{prefix}.bias"))?, n, k })
    }

    fn run(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n).map(|o| self.b[o] + self.w[o * self.k..(o + 1) * self.k].iter().zip(x).map(|(w, v)| w * v).sum::<f32>()).collect()
    }
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1. + (-v).exp())).collect()
}

struct Block {
    /// The block's offset to the shared modulation (`[6 channels]`, on the host).
    modulation: Vec<f32>,
    qkv: Lin,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    out: Lin,
    /// The cross-attention's input norm's weight less one, then its bias.
    norm2: DeviceVec,
    cq: Lin,
    ckv: Lin,
    cq_norm: DeviceVec,
    ck_norm: DeviceVec,
    cout: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

/// A run's conditioning: each block's cross-attention keys and values (`[tokens, 2 channels]`, the keys normed), and
/// the voxels' back-projected features (`[voxels, proj_channels]`; None for the unconditional branch: the projection's
/// bias alone).
pub struct Context {
    kv: Vec<DeviceVec>,
    tokens: usize,
    proj: Option<DeviceVec>,
}

/// A forward's work vectors for `n` voxels (one stage's every pass).
pub struct Scratch {
    n: usize,
    h: DeviceVec,
    hn: DeviceVec,
    qkv: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    o: DeviceVec,
    f1: DeviceVec,
    f1g: DeviceVec,
}

/// A flow transformer on the device.
pub struct WgpuDit3 {
    cfg: Config,
    head_dim: usize,
    input: Lin,
    t1: Host,
    t2: Host,
    ada: Host,
    blocks: Vec<Block>,
    output: Lin,
    zeros: DeviceVec,
    eps: f32,
}

/// Each head's q and k made unit length, times its weights and the square root of its width (TRELLIS's
/// `MultiHeadRMSNorm`): an RMS norm with no epsilon to speak of.
const HEAD_EPS: f32 = 1e-24;

impl WgpuDit3 {
    /// `path`: the checkpoint without its extension (its `.json` and `.safetensors`).
    pub fn load(path: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let cfg = Config::read(&path.with_extension("json"))?;
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let s = &mut store;
        let c = cfg.channels;
        let norm2 = |s: &mut Store, p: &str| -> Result<DeviceVec> {
            let m: Vec<f32> = f32s(s, &format!("{p}.weight"))?.iter().map(|v| v - 1.).chain(f32s(s, &format!("{p}.bias"))?).collect();
            Ok(upload(gpu, &m))
        };
        let mut blocks = Vec::with_capacity(cfg.blocks);
        for i in 0..cfg.blocks {
            let p = format!("blocks.{i}");
            let ca = format!("{p}.cross_attn.cross_attn_block");
            blocks.push(Block {
                modulation: f32s(s, &format!("{p}.modulation"))?,
                qkv: Lin::load(s, gpu, &format!("{p}.self_attn.to_qkv"), false)?,
                q_norm: upload(gpu, &f32s(s, &format!("{p}.self_attn.q_rms_norm.gamma"))?),
                k_norm: upload(gpu, &f32s(s, &format!("{p}.self_attn.k_rms_norm.gamma"))?),
                out: Lin::load(s, gpu, &format!("{p}.self_attn.to_out"), false)?,
                norm2: norm2(s, &format!("{p}.norm2"))?,
                cq: Lin::load(s, gpu, &format!("{ca}.to_q"), false)?,
                ckv: Lin::load(s, gpu, &format!("{ca}.to_kv"), false)?,
                cq_norm: upload(gpu, &f32s(s, &format!("{ca}.q_rms_norm.gamma"))?),
                ck_norm: upload(gpu, &f32s(s, &format!("{ca}.k_rms_norm.gamma"))?),
                cout: Lin::load(s, gpu, &format!("{ca}.to_out"), false)?,
                proj: Lin::load(s, gpu, &format!("{p}.cross_attn.proj_linear"), false)?,
                fc1: Lin::load(s, gpu, &format!("{p}.mlp.mlp.0"), false)?,
                fc2: Lin::load(s, gpu, &format!("{p}.mlp.mlp.2"), false)?,
            });
        }
        Ok(Self {
            head_dim: c / cfg.heads,
            input: Lin::load(s, gpu, "input_layer", true)?,
            t1: Host::load(s, "t_embedder.mlp.0")?,
            t2: Host::load(s, "t_embedder.mlp.2")?,
            ada: Host::load(s, "adaLN_modulation.1")?,
            output: Lin::load(s, gpu, "out_layer", true)?,
            blocks,
            zeros: upload(gpu, &vec![0f32; c]),
            eps: 1e-6,
            cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The conditioning for a run: `image` (`[tokens, cond_channels]`, DINOv3's) and `proj` (`[voxels,
    /// proj_channels]`); None for the unconditional branch (zeros).
    pub fn context(&self, gpu: &WgpuBackend, image: Option<&[f32]>, proj: Option<&[f32]>, tokens: usize) -> Context {
        let (c, heads) = (self.cfg.channels, self.cfg.heads);
        let image = match image {
            Some(t) => upload(gpu, t),
            None => upload(gpu, &vec![0f32; tokens * self.cfg.cond_channels]),
        };
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let (kv, k, kn) = (gpu.vec(tokens * 2 * c), gpu.vec(tokens * c), gpu.vec(tokens * c));
        let mut out = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let normed = gpu.vec(tokens * 2 * c);
            b.ckv.run(r, &image, &kv, tokens);
            r.copy_cols(&kv, &k, tokens, c, 2 * c, 0);
            r.rmsnorm_heads_rows(&k, &b.ck_norm, &kn, tokens, heads, HEAD_EPS);
            r.store_rows(&kn, &normed, tokens, c, 0, 2 * c, 0);
            r.copy_cols(&kv, &k, tokens, c, 2 * c, c);
            r.store_rows(&k, &normed, tokens, c, 0, 2 * c, c);
            out.push(normed);
        }
        rec.finish();
        Context { kv: out, tokens, proj: proj.map(|p| upload(gpu, p)) }
    }

    /// Every block's modulation at `timestep` (0-1000; `[blocks, 6 channels]`): the timestep's sinusoid through its MLP
    /// and the shared adaLN, plus each block's offset (F32, on the host).
    pub fn modulation(&self, timestep: f64) -> Vec<f32> {
        let half = 128;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
        let args: Vec<f32> = freqs.iter().map(|f| (timestep as f32) * f).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let t = self.t2.run(&silu(&self.t1.run(&emb)));
        let shared = self.ada.run(&silu(&t));
        self.blocks.iter().flat_map(|b| shared.iter().zip(&b.modulation).map(|(a, m)| a + m).collect::<Vec<_>>()).collect()
    }

    /// The work vectors of a pass over `n` voxels.
    pub fn scratch(&self, gpu: &WgpuBackend, n: usize) -> Scratch {
        let (c, hd, heads) = (self.cfg.channels, self.head_dim, self.cfg.heads);
        let v = |len: usize| gpu.vec(len);
        Scratch {
            n,
            h: v(n * c),
            hn: v(n * c),
            qkv: v(n * 3 * c),
            q: v(n * c),
            k: v(n * c),
            v: v(n * c),
            qn: v(n * c),
            kn: v(n * c),
            kv: v(n * 2 * c),
            att: v(gpu.attention_rows_full_out_len(n, heads, hd, n).max(gpu.attention_rows_full_out_len(n, heads, hd, 64))),
            o: v(n * c),
            f1: v(n * self.cfg.mlp),
            f1g: v(n * self.cfg.mlp),
        }
    }

    /// The velocity for voxel features `x` (`[n, in_channels]`) into `out` (`[n, out_channels]`), recorded on `r`: `mods`
    /// [`Self::modulation`]'s on the device, `table` the voxels' rotary angles (each one's (sin, cos) pairs, [`table`]),
    /// `ctx` the run's conditioning, `w` the work vectors for `n`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, w: &Scratch, r: &mut dyn ChainRecorder, x: &DeviceVec, mods: &DeviceVec, table: &DeviceVec, ctx: &Context, out: &DeviceVec) {
        let (c, heads, hd, n, eps) = (self.cfg.channels, self.cfg.heads, self.head_dim, w.n, self.eps);
        let scale = 1. / (hd as f32).sqrt();
        let Scratch { h, hn, qkv, q, k, v, qn, kn, kv, att, o, f1, f1g, .. } = w;
        self.input.run(r, x, h, n);
        for (bi, (b, ckv)) in self.blocks.iter().zip(&ctx.kv).enumerate() {
            let base = bi * 6 * c;
            // self-attention
            r.layernorm_mod_rows(h, hn, n, c, mods, base + c, Some(base), eps);
            b.qkv.run(r, hn, qkv, n);
            r.copy_cols(qkv, q, n, c, 3 * c, 0);
            r.copy_cols(qkv, k, n, c, 3 * c, c);
            r.copy_cols(qkv, v, n, c, 3 * c, 2 * c);
            r.rmsnorm_heads_rows(q, &b.q_norm, qn, n, heads, HEAD_EPS);
            r.rmsnorm_heads_rows(k, &b.k_norm, kn, n, heads, HEAD_EPS);
            r.rope_rows(qn, n, heads, hd, table, false);
            r.rope_rows(kn, n, heads, hd, table, false);
            r.store_rows(kn, kv, n, c, 0, 2 * c, 0);
            r.store_rows(v, kv, n, c, 0, 2 * c, c);
            r.attention_rows_full(qn, kv, att, n, heads, heads, hd, n, scale);
            b.out.run(r, att, o, n);
            r.add_gated_rows(h, o, n, c, mods, base + 2 * c, false);
            // cross-attention to the picture, plus the voxel's own features
            r.norm_mod_rows(h, hn, n, c, &b.norm2, 0, Some(c), RowNorm::Layer, eps);
            b.cq.run(r, hn, q, n);
            r.rmsnorm_heads_rows(q, &b.cq_norm, qn, n, heads, HEAD_EPS);
            r.attention_rows_full(qn, ckv, att, n, heads, heads, hd, ctx.tokens, scale);
            b.cout.run(r, att, o, n);
            match &ctx.proj {
                Some(p) => {
                    b.proj.run(r, p, hn, n);
                    r.add(o, hn);
                }
                None => r.add_bias_rows(o, &b.proj.b, n, c),
            }
            r.add(h, o);
            // the MLP
            r.layernorm_mod_rows(h, hn, n, c, mods, base + 4 * c, Some(base + 3 * c), eps);
            b.fc1.run(r, hn, f1, n);
            r.gelu(f1, f1g, n * self.cfg.mlp);
            b.fc2.run(r, f1g, o, n);
            r.add_gated_rows(h, o, n, c, mods, base + 5 * c, false);
        }
        r.layernorm_mod_rows(h, hn, n, c, &self.zeros, 0, None, 1e-5);
        self.output.run(r, hn, out, n);
    }
}

/// The voxels' rotary angles (`[voxels, head_dim]`, each pair's sine then cosine): 21 frequencies an axis (for a
/// head of 128), the leftover pair unrotated, as [`crate::model3d::dit::rope_tables`] makes them.
pub fn table(coords: &[[i32; 3]], head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    let per = half / 3;
    let freqs: Vec<f64> = (0..per).map(|i| 1. / 10000f64.powf(i as f64 / per as f64)).collect();
    let mut out = Vec::with_capacity(coords.len() * head_dim);
    for c in coords {
        for axis in 0..3 {
            for f in &freqs {
                let a = c[axis] as f64 * f;
                out.push(a.sin() as f32);
                out.push(a.cos() as f32);
            }
        }
        for _ in per * 3..half {
            out.push(0.);
            out.push(1.);
        }
    }
    out
}

// --- the decoders -----------------------------------------------------------------------------------------------------

/// A level's voxels: their coordinates (on the host), the grid's size, and each one's 27 neighbours' rows on the device
/// (`[voxels, 27]` u32s in a vector's words, in kernel order: x, then y, then z offset from -1 to 1; the voxel count
/// where there is none), as [`crate::model3d::sparse::Level`] has them.
pub struct WgpuLevel {
    pub coords: Vec<[i32; 3]>,
    pub res: usize,
    neighbours: DeviceVec,
}

fn key(c: [i32; 3]) -> u64 {
    ((c[0] as u64 & 0x1f_ffff) << 42) | ((c[1] as u64 & 0x1f_ffff) << 21) | (c[2] as u64 & 0x1f_ffff)
}

/// u32s as a vector's words.
fn upload_u32(gpu: &WgpuBackend, v: &[u32]) -> DeviceVec {
    upload(gpu, &v.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>())
}

impl WgpuLevel {
    pub fn new(gpu: &WgpuBackend, coords: Vec<[i32; 3]>, res: usize) -> Self {
        let n = coords.len();
        let index: std::collections::HashMap<u64, u32> = coords.iter().enumerate().map(|(i, c)| (key(*c), i as u32)).collect();
        let mut table = vec![n as u32; n * 27];
        for (i, c) in coords.iter().enumerate() {
            let mut k = 0;
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let p = [c[0] + dx, c[1] + dy, c[2] + dz];
                        if p.iter().all(|&v| v >= 0 && (v as usize) < res) {
                            if let Some(&j) = index.get(&key(p)) {
                                table[i * 27 + k] = j;
                            }
                        }
                        k += 1;
                    }
                }
            }
        }
        Self { neighbours: upload_u32(gpu, &table), coords, res }
    }

    /// Every voxel of an `res`³ grid, x slowest.
    pub fn dense(gpu: &WgpuBackend, res: usize) -> Self {
        let r = res as i32;
        Self::new(gpu, (0..r).flat_map(|x| (0..r).flat_map(move |y| (0..r).map(move |z| [x, y, z]))).collect(), res)
    }

    pub fn len(&self) -> usize {
        self.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }
}

/// Values gathered at a time for a sparse convolution (`[voxels, 27 cin]`): 64M, a quarter of a gigabyte.
const GATHERED: usize = 1 << 26;
/// Values of a vector the decoders make for a chunk of rows at a time (a binding takes at most 2 GB): 256M, a gigabyte.
const ROWS_VALUES: usize = 1 << 28;

/// A recording's sparse convolutions' gather and its products, made once (each convolution's in turn in them).
struct Gather {
    g: DeviceVec,
    t: DeviceVec,
}

impl Gather {
    fn new(gpu: &WgpuBackend) -> Self {
        Self { g: gpu.vec(GATHERED), t: gpu.vec(GATHERED / 2) }
    }
}

/// A 3x3x3 submanifold convolution: its weight `[cout, 27 cin]` (f16, the taps in the neighbour table's order), its bias.
struct SparseConv {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
}

impl SparseConv {
    /// From `weight` in the sparse (flex_gemm) layout `[cout, 3, 3, 3, cin]`, or with `dense` PyTorch's `Conv3d`
    /// `[cout, cin, 3, 3, 3]`; its output channels in the order `order` gives (`order[o]` the stored channel of output
    /// `o`, where given).
    fn load(s: &mut Store, gpu: &WgpuBackend, prefix: &str, dense: bool, order: Option<&[usize]>) -> Result<Self> {
        let t = s.tensor_f32(&format!("{prefix}.weight"), &Device::Cpu)?;
        let dims = t.dims().to_vec();
        let (cout, cin) = if dense { (dims[0], dims[1]) } else { (dims[0], dims[4]) };
        let taps = if dense { dims.get(2..5) } else { dims.get(1..4) };
        if dims.len() != 5 || taps != Some(&[3, 3, 3][..]) {
            return Err(err(format!("{prefix}: a 3x3x3 convolution of {dims:?}")));
        }
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        let b = f32s(s, &format!("{prefix}.bias"))?;
        let at = |o: usize, tap: usize, c: usize| if dense { v[(o * cin + c) * 27 + tap] } else { v[(o * 27 + tap) * cin + c] };
        let rows: Vec<usize> = order.map_or_else(|| (0..cout).collect(), |o| o.to_vec());
        let w: Vec<f32> = rows.iter().flat_map(|&o| (0..27).flat_map(move |tap| (0..cin).map(move |c| at(o, tap, c)))).collect();
        let b: Vec<f32> = rows.iter().map(|&o| b[o]).collect();
        let words = f16_words_f32(&w).ok_or_else(|| err(format!("{prefix}: past f16's range")))?;
        Ok(Self { w: upload(gpu, &words), b: upload(gpu, &b), cin, cout })
    }

    /// `x` (`[voxels, cin]`) convolved over `level` into `y` (`[voxels, cout]`).
    fn run(&self, w: &Gather, r: &mut dyn ChainRecorder, level: &WgpuLevel, x: &DeviceVec, y: &DeviceVec) {
        self.run_range(w, r, level, x, y, 0, level.len());
    }

    /// The outputs of voxels `from..from + count` into `y`'s first rows (`[count, cout]`): a chunk of voxels'
    /// neighbours at a time gathered into `w.g` (zeros where there is none), times the weight (into `w.t`, then `y`,
    /// where chunked).
    #[allow(clippy::too_many_arguments)]
    fn run_range(&self, w: &Gather, r: &mut dyn ChainRecorder, level: &WgpuLevel, x: &DeviceVec, y: &DeviceVec, from: usize, count: usize) {
        let step = (GATHERED / (27 * self.cin)).min((GATHERED / 2) / self.cout).max(1);
        let mut at = 0;
        while at < count {
            let m = step.min(count - at);
            r.gather_rows(x, &level.neighbours, &w.g, m * 27, self.cin, (from + at) * 27, level.len());
            if m == count {
                r.matmul_f16_rows(&self.w, self.cout, 27 * self.cin, &w.g, y, m);
            } else {
                r.matmul_f16_rows(&self.w, self.cout, 27 * self.cin, &w.g, &w.t, m);
                r.copy(&w.t, 0, y, at * self.cout, m * self.cout);
            }
            at += m;
        }
        r.add_bias_rows(y, &self.b, count, self.cout);
    }
}

/// A layer norm's weight less one, then its bias (or zeros: none), for [`ChainRecorder::norm_mod_rows`].
fn norm(s: &mut Store, gpu: &WgpuBackend, prefix: Option<&str>, c: usize) -> Result<DeviceVec> {
    Ok(match prefix {
        Some(p) => upload(gpu, &f32s(s, &format!("{p}.weight"))?.iter().map(|v| v - 1.).chain(f32s(s, &format!("{p}.bias"))?).collect::<Vec<_>>()),
        None => upload(gpu, &vec![0f32; 2 * c]),
    })
}

/// `x` (`[rows, c]`) layer-normed (`mods` [`norm`]'s), then SiLU'd, into `out` (`t` for the norm).
#[allow(clippy::too_many_arguments)]
fn norm_silu(r: &mut dyn ChainRecorder, x: &DeviceVec, mods: &DeviceVec, t: &DeviceVec, out: &DeviceVec, rows: usize, c: usize, eps: f32) {
    r.norm_mod_rows(x, t, rows, c, mods, 0, Some(c), RowNorm::Layer, eps);
    r.mul_sigmoid(t, t, out, rows * c);
}

struct DenseRes {
    norm1: DeviceVec,
    norm2: DeviceVec,
    conv1: SparseConv,
    conv2: SparseConv,
}

enum DenseBlock {
    Res(DenseRes),
    /// A conv to 8x the channels (its outputs reordered child-major), then a 3D pixel shuffle.
    Up(SparseConv),
}

/// The sparse structure decoder (dense 3D convolutions: its latent `[16³, 8]` to occupancy at 64³), its weights f16.
pub struct WgpuStructureDecoder {
    input: SparseConv,
    middle: Vec<DenseRes>,
    blocks: Vec<DenseBlock>,
    out_norm: DeviceVec,
    out_conv: SparseConv,
}

impl WgpuStructureDecoder {
    /// `path`: the checkpoint without its extension.
    pub fn load(path: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let config = crate::music::acoustic::read_config(&path.with_extension("json"))?;
        let args = config.get("args").ok_or_else(|| err("structure decoder config: no args"))?;
        let ints = |k: &str| args.get(k).and_then(oaiy_engine::json::Json::as_array).map(|a| a.iter().filter_map(|v| v.as_i64()).map(|v| v as usize).collect::<Vec<_>>()).unwrap_or_default();
        let channels = ints("channels");
        let res_blocks = args.get("num_res_blocks").and_then(oaiy_engine::json::Json::as_i64).unwrap_or(2) as usize;
        let middle = args.get("num_res_blocks_middle").and_then(oaiy_engine::json::Json::as_i64).unwrap_or(2) as usize;
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let s = &mut store;
        let res_block = |s: &mut Store, p: String, c: usize| -> Result<DenseRes> {
            Ok(DenseRes { norm1: norm(s, gpu, Some(&format!("{p}.norm1")), c)?, norm2: norm(s, gpu, Some(&format!("{p}.norm2")), c)?, conv1: SparseConv::load(s, gpu, &format!("{p}.conv1"), true, None)?, conv2: SparseConv::load(s, gpu, &format!("{p}.conv2"), true, None)? })
        };
        let mut blocks = Vec::new();
        let mut i = 0;
        for (level, &c) in channels.iter().enumerate() {
            for _ in 0..res_blocks {
                blocks.push(DenseBlock::Res(res_block(s, format!("blocks.{i}"), c)?));
                i += 1;
            }
            if level + 1 < channels.len() {
                // (stored channel-major, `ch 8 + sub` as pixel_shuffle_3d reads them: made child-major, `sub c + ch`)
                let c8 = channels[level + 1];
                let order: Vec<usize> = (0..8 * c8).map(|o| (o % c8) * 8 + o / c8).collect();
                blocks.push(DenseBlock::Up(SparseConv::load(s, gpu, &format!("blocks.{i}.conv"), true, Some(&order))?));
                i += 1;
            }
        }
        let first = channels.first().copied().ok_or_else(|| err("structure decoder config: no channels"))?;
        Ok(Self {
            input: SparseConv::load(s, gpu, "input_layer", true, None)?,
            middle: (0..middle).map(|m| res_block(s, format!("middle_block.{m}"), first)).collect::<Result<_>>()?,
            blocks,
            out_norm: norm(s, gpu, Some("out_layer.0"), *channels.last().unwrap())?,
            out_conv: SparseConv::load(s, gpu, "out_layer.2", true, None)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn res_block(gpu: &WgpuBackend, w: &Gather, r: &mut dyn ChainRecorder, b: &DenseRes, level: &WgpuLevel, x: &DeviceVec, c: usize) -> DeviceVec {
        let n = level.len();
        let (t, h, y) = (gpu.vec(n * c), gpu.vec(n * c), gpu.vec(n * c));
        norm_silu(r, x, &b.norm1, &t, &h, n, c, 1e-5);
        b.conv1.run(w, r, level, &h, &y);
        norm_silu(r, &y, &b.norm2, &t, &h, n, c, 1e-5);
        b.conv2.run(w, r, level, &h, &y);
        r.add(&y, x);
        y
    }

    /// The latent (`[16³, 8]`, voxels x slowest) to occupancy logits at the finest grid (on the host), and its size.
    pub fn forward(&self, gpu: &WgpuBackend, latent: &[f32]) -> Result<(Vec<f32>, usize)> {
        let mut res = 16;
        let mut level = WgpuLevel::dense(gpu, res);
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let x = upload(gpu, latent);
        let w = Gather::new(gpu);
        let mut c = self.input.cout;
        let mut h = gpu.vec(level.len() * c);
        self.input.run(&w, r, &level, &x, &h);
        for b in &self.middle {
            h = Self::res_block(gpu, &w, r, b, &level, &h, c);
        }
        for block in &self.blocks {
            match block {
                DenseBlock::Res(b) => h = Self::res_block(gpu, &w, r, b, &level, &h, c),
                DenseBlock::Up(conv) => {
                    let n = level.len();
                    let y = gpu.vec(n * conv.cout);
                    conv.run(&w, r, &level, &h, &y);
                    // each child's channels from its parent's (child-major), the finer grid's voxels x slowest
                    let next = res * 2;
                    let ids: Vec<u32> = (0..next).flat_map(|x| (0..next).flat_map(move |yy| (0..next).map(move |z| {
                        let parent = ((x / 2) * res + yy / 2) * res + z / 2;
                        (parent * 8 + (x % 2) * 4 + (yy % 2) * 2 + z % 2) as u32
                    }))).collect();
                    c = conv.cout / 8;
                    let idx = upload_u32(gpu, &ids);
                    h = gpu.vec(ids.len() * c);
                    r.gather_rows(&y, &idx, &h, ids.len(), c, 0, n * 8);
                    res = next;
                    level = WgpuLevel::dense(gpu, res);
                }
            }
        }
        let n = level.len();
        let (t, hn, out) = (gpu.vec(n * c), gpu.vec(n * c), gpu.vec(n * self.out_conv.cout));
        norm_silu(r, &h, &self.out_norm, &t, &hn, n, c, 1e-5);
        self.out_conv.run(&w, r, &level, &hn, &out);
        r.read(&out);
        Ok((rec.finish().pop().ok_or_else(|| err("the occupancy was not read"))?, res))
    }
}

/// A linear layer (f16, or F32) as the decoders have them.
fn lin(s: &mut Store, gpu: &WgpuBackend, prefix: &str, f32: bool) -> Result<Lin> {
    Lin::load(s, gpu, prefix, f32)
}

struct ConvNeXt {
    conv: SparseConv,
    norm: DeviceVec,
    fc1: Lin,
    fc2: Lin,
}

/// Up a level: channel-to-space into the children kept.
struct Up {
    norm1: DeviceVec,
    norm2: DeviceVec,
    conv1: SparseConv,
    conv2: SparseConv,
    to_subdiv: Option<Lin>,
    channels: usize,
    out: usize,
}

/// A sparse decoder (TRELLIS.2's shape or texture decoder) on the device, its weights f16 (as the reference runs them),
/// its input and output layers F32.
pub struct WgpuSparseDecoder {
    from_latent: Lin,
    levels: Vec<(Vec<ConvNeXt>, Option<Up>)>,
    output: Lin,
    channels: Vec<usize>,
    pub out_channels: usize,
}

/// What a sparse decoder made: the finest level's voxels and their values (on the host), and the children it kept at
/// each step up.
pub struct WgpuDecoded {
    pub level: WgpuLevel,
    pub feats: Vec<f32>,
    pub subdivisions: Vec<crate::model3d::sparse::Subdivision>,
}

impl WgpuSparseDecoder {
    pub fn load(path: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let config = crate::music::acoustic::read_config(&path.with_extension("json"))?;
        let args = config.get("args").ok_or_else(|| err("decoder config: no args"))?;
        let ints = |k: &str| args.get(k).and_then(oaiy_engine::json::Json::as_array).map(|a| a.iter().filter_map(|v| v.as_i64()).map(|v| v as usize).collect::<Vec<_>>()).unwrap_or_default();
        let (channels, counts) = (ints("model_channels"), ints("num_blocks"));
        let pred_subdiv = args.get("pred_subdiv").and_then(oaiy_engine::json::Json::as_bool).unwrap_or(true);
        let name = config.get("name").and_then(oaiy_engine::json::Json::as_str).unwrap_or("");
        let out_channels = if name == "FlexiDualGridVaeDecoder" { 7 } else { args.get("out_channels").and_then(oaiy_engine::json::Json::as_i64).unwrap_or(6) as usize };
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let s = &mut store;
        let mut levels = Vec::new();
        for (i, &ch) in channels.iter().enumerate() {
            let mut blocks = Vec::new();
            for j in 0..counts[i] {
                let p = format!("blocks.{i}.{j}");
                blocks.push(ConvNeXt { conv: SparseConv::load(s, gpu, &format!("{p}.conv"), false, None)?, norm: norm(s, gpu, Some(&format!("{p}.norm")), ch)?, fc1: lin(s, gpu, &format!("{p}.mlp.0"), false)?, fc2: lin(s, gpu, &format!("{p}.mlp.2"), false)? });
            }
            let up = if i + 1 < channels.len() {
                let p = format!("blocks.{i}.{}", counts[i]);
                Some(Up {
                    norm1: norm(s, gpu, Some(&format!("{p}.norm1")), ch)?,
                    norm2: norm(s, gpu, None, channels[i + 1])?,
                    conv1: SparseConv::load(s, gpu, &format!("{p}.conv1"), false, None)?,
                    conv2: SparseConv::load(s, gpu, &format!("{p}.conv2"), false, None)?,
                    to_subdiv: if pred_subdiv { Some(lin(s, gpu, &format!("{p}.to_subdiv"), false)?) } else { None },
                    channels: ch,
                    out: channels[i + 1],
                })
            } else {
                None
            };
            levels.push((blocks, up));
        }
        Ok(Self { from_latent: lin(s, gpu, "from_latent", true)?, levels, output: lin(s, gpu, "output_layer", true)?, channels, out_channels })
    }

    /// The voxels a latent would have after `times` steps up (the cascade's finer structure), without decoding them.
    pub fn upsample(&self, gpu: &WgpuBackend, level: WgpuLevel, latent: &[f32], times: usize) -> Result<WgpuLevel> {
        self.run(gpu, level, latent, None, times).map(|d| d.level)
    }

    /// From a latent (`[voxels, 32]`) on `level` to the finest level; a texture decoder follows a shape decoder's
    /// choice of children (`guide`).
    pub fn forward(&self, gpu: &WgpuBackend, level: WgpuLevel, latent: &[f32], guide: Option<&[crate::model3d::sparse::Subdivision]>) -> Result<WgpuDecoded> {
        self.run(gpu, level, latent, guide, usize::MAX)
    }

    fn run(&self, gpu: &WgpuBackend, level: WgpuLevel, latent: &[f32], guide: Option<&[crate::model3d::sparse::Subdivision]>, stop_at: usize) -> Result<WgpuDecoded> {
        use crate::model3d::sparse::Subdivision;
        let mut level = level;
        let mut n = level.len();
        let mut c = self.channels[0];
        let x = upload(gpu, latent);
        let mut h = gpu.vec(n * c);
        let mut subdivisions = Vec::new();
        // a level a recording (its children chosen on the host between them)
        let w = Gather::new(gpu);
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        self.from_latent.run(rec.as_mut(), &x, &h, n);
        for (i, (blocks, up)) in self.levels.iter().enumerate() {
            if i == stop_at {
                rec.finish();
                return Ok(WgpuDecoded { level, feats: Vec::new(), subdivisions });
            }
            let r = rec.as_mut();
            if let Some(ff) = blocks.first().map(|b| b.fc1.n) {
                // (two vectors for the blocks' stream, taking turns; the MLP a chunk of rows at a time where its inner
                // vectors would pass a binding's 2 GB)
                let chunk = (ROWS_VALUES / ff).min(n);
                let (mut other, t, f, fs) = (gpu.vec(n * c), gpu.vec(n * c), gpu.vec(chunk * ff), gpu.vec(chunk * ff));
                let (tc, oc) = if chunk < n { (Some(gpu.vec(chunk * c)), Some(gpu.vec(chunk * c))) } else { (None, None) };
                for b in blocks {
                    b.conv.run(&w, r, &level, &h, &other);
                    r.norm_mod_rows(&other, &t, n, c, &b.norm, 0, Some(c), RowNorm::Layer, 1e-6);
                    match (&tc, &oc) {
                        (Some(tc), Some(oc)) => {
                            let mut at = 0;
                            while at < n {
                                let m = chunk.min(n - at);
                                r.copy(&t, at * c, tc, 0, m * c);
                                b.fc1.run(r, tc, &f, m);
                                r.mul_sigmoid(&f, &f, &fs, m * ff);
                                b.fc2.run(r, &fs, oc, m);
                                r.copy(oc, 0, &other, at * c, m * c);
                                at += m;
                            }
                        }
                        _ => {
                            b.fc1.run(r, &t, &f, n);
                            r.mul_sigmoid(&f, &f, &fs, n * ff);
                            b.fc2.run(r, &fs, &other, n);
                        }
                    }
                    r.add(&other, &h);
                    std::mem::swap(&mut h, &mut other);
                }
            }
            let Some(up) = up else { continue };
            let sub = match (&up.to_subdiv, guide) {
                (Some(l), _) => {
                    let logits = gpu.vec(n * l.n);
                    l.run(r, &h, &logits, n);
                    r.read(&logits);
                    let flags = std::mem::replace(&mut rec, gpu.begin()).finish().pop().ok_or_else(|| err("the children were not read"))?;
                    rec.keep_groups(false);
                    let (mut parent, mut child) = (Vec::new(), Vec::new());
                    for (v, row) in flags.chunks_exact(l.n).enumerate() {
                        for (s, &f) in row.iter().enumerate() {
                            if f > 0. {
                                parent.push(v as u32);
                                child.push(s as u8);
                            }
                        }
                    }
                    Subdivision { parent, child }
                }
                (None, Some(g)) => Subdivision { parent: g[i].parent.clone(), child: g[i].child.clone() },
                (None, None) => return Err(err("a decoder that does not choose its children needs a guide")),
            };
            let r = rec.as_mut();
            let (t, hh) = (gpu.vec(n * c), gpu.vec(n * c));
            norm_silu(r, &h, &up.norm1, &t, &hh, n, c, 1e-6);
            let next = WgpuLevel::new(gpu, sub.coords_of(&level.coords), level.res * 2);
            let m = next.len();
            let (o, skip) = (up.out, up.channels / 8);
            // the first conv's outputs (8 children's channels a voxel) a range of parents at a time, under a binding's
            // limit, and their children's (channel-to-space) gathered from them; the children in their parents' order
            let parents = (ROWS_VALUES / up.conv1.cout).min(n).max(1);
            let (y, yc) = (gpu.vec(parents * up.conv1.cout), gpu.vec(m * o));
            let mut first_child = 0;
            let mut from = 0;
            while from < n {
                let count = parents.min(n - from);
                let end = sub.parent.partition_point(|&p| (p as usize) < from + count);
                up.conv1.run_range(&w, r, &level, &hh, &y, from, count);
                if end > first_child {
                    let rows: Vec<u32> = (first_child..end).map(|i| (sub.parent[i] - from as u32) * 8 + sub.child[i] as u32).collect();
                    let idx = upload_u32(gpu, &rows);
                    if first_child == 0 && end == m {
                        r.gather_rows(&y, &idx, &yc, m, o, 0, count * 8);
                    } else {
                        let part = gpu.vec(rows.len() * o);
                        r.gather_rows(&y, &idx, &part, rows.len(), o, 0, count * 8);
                        r.copy(&part, 0, &yc, first_child * o, rows.len() * o);
                    }
                }
                first_child = end;
                from += count;
            }
            // the skip's channels the same way from the block's input
            let rows: Vec<u32> = sub.parent.iter().zip(&sub.child).map(|(&p, &s)| p * 8 + s as u32).collect();
            let idx = upload_u32(gpu, &rows);
            let xs = gpu.vec(m * skip);
            r.gather_rows(&h, &idx, &xs, m, skip, 0, n * 8);
            let (t2, hn) = (gpu.vec(m * o), gpu.vec(m * o));
            norm_silu(r, &yc, &up.norm2, &t2, &hn, m, o, 1e-6);
            let h2 = gpu.vec(m * o);
            up.conv2.run(&w, r, &next, &hn, &h2);
            r.repeat_cols_add_rows(&xs, &h2, m, skip, o / skip);
            h = h2;
            c = o;
            n = m;
            level = next;
            subdivisions.push(sub);
            // (each level's work its own recording: its vectors let go before the next's)
            std::mem::replace(&mut rec, gpu.begin()).finish();
            rec.keep_groups(false);
        }
        let r = rec.as_mut();
        let (t, out) = (gpu.vec(n * c), gpu.vec(n * self.out_channels));
        r.layernorm_mod_rows(&h, &t, n, c, &upload(gpu, &vec![0f32; c]), 0, None, 1e-5);
        // (a million voxels at a time: a matmul's dispatch takes 65,535 tiles of rows)
        let chunk = (1 << 20).min(n);
        if chunk == n {
            self.output.run(r, &t, &out, n);
        } else {
            let (tc, oc) = (gpu.vec(chunk * c), gpu.vec(chunk * self.out_channels));
            let mut at = 0;
            while at < n {
                let m = chunk.min(n - at);
                r.copy(&t, at * c, &tc, 0, m * c);
                self.output.run(r, &tc, &oc, m);
                r.copy(&oc, 0, &out, at * self.out_channels, m * self.out_channels);
                at += m;
            }
        }
        r.read(&out);
        let feats = rec.finish().pop().ok_or_else(|| err("the decoded voxels were not read"))?;
        Ok(WgpuDecoded { level, feats, subdivisions })
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use std::path::PathBuf;

    fn dir() -> PathBuf {
        PathBuf::from(std::env::var("P3D_DIR").unwrap_or_else(|_| "E:/p3dref/cmp4/ref".into()))
    }

    fn models() -> PathBuf {
        PathBuf::from(std::env::var("MODELS").unwrap_or_else(|_| "E:/models".into()))
    }

    fn load(name: &str) -> Vec<f32> {
        std::fs::read(dir().join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn coords(name: &str) -> Vec<[i32; 3]> {
        let v: Vec<i32> = std::fs::read(dir().join(name)).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        v.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    fn relative(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        (a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt()
    }

    /// One velocity on the device from `x` (`[n, in]`) at `timestep`.
    fn velocity(gpu: &WgpuBackend, dit: &WgpuDit3, x: &[f32], n: usize, timestep: f64, table: &DeviceVec, ctx: &Context) -> Vec<f32> {
        let (xd, mods, out, w) = (upload(gpu, x), upload(gpu, &dit.modulation(timestep)), gpu.vec(n * dit.cfg.out_channels), dit.scratch(gpu, n));
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        dit.forward(&w, rec.as_mut(), &xd, &mods, table, ctx, &out);
        rec.as_mut().read(&out);
        rec.finish().pop().unwrap()
    }

    /// The decoders on the official pipeline's latents against its outputs (`--ignored --nocapture`, the folders as
    /// below): the structure's occupancy logits at 64³; the shape's voxels (both's, by coordinate) and their values, and
    /// the texture's, following our shape's choice of children.
    #[test]
    #[ignore = "needs Pixal3D and the reference's dumps"]
    fn the_webgpu_decoders_are_the_references() -> Result<()> {
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let ckpts = models().join("Pixal3D/ckpts");
        let started = std::time::Instant::now();
        let ss = WgpuStructureDecoder::load(&ckpts.join("ss_dec_conv3d_16l8_fp16"), &gpu)?;
        let (logits, res) = ss.forward(&gpu, &load("ss_latent.bin"))?;
        let want = load("ss_logits.bin");
        let same = logits.iter().zip(&want).filter(|(a, b)| (**a > 0.) == (**b > 0.)).count();
        eprintln!("structure ({res}^3, {:.1} s): {:.2e}; occupied alike at {} of {} voxels", started.elapsed().as_secs_f64(), relative(&logits, &want), same, want.len());
        drop(ss);
        let hr = coords("coords_hr.i32");
        let n = hr.len();
        let decoder = WgpuSparseDecoder::load(&ckpts.join("shape_dec_next_dc_f16c32_fp16"), &gpu)?;
        let started = std::time::Instant::now();
        let decoded = decoder.forward(&gpu, WgpuLevel::new(&gpu, hr.clone(), 64), &load("shape_hr_final.bin"), None)?;
        eprintln!("shape decoded in {:.1} s: {} voxels at {} from {n}", started.elapsed().as_secs_f64(), decoded.level.len(), decoded.level.res);
        drop(decoder);
        let ref_coords = coords("shape_voxels_coords.i32");
        let at: std::collections::HashMap<[i32; 3], usize> = ref_coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let common: Vec<(usize, usize)> = decoded.level.coords.iter().enumerate().filter_map(|(i, c)| at.get(c).map(|&j| (i, j))).collect();
        eprintln!("common voxels: {} ({:.3}% of ours, {:.3}% of the reference's {})", common.len(), 100. * common.len() as f64 / decoded.level.len() as f64, 100. * common.len() as f64 / ref_coords.len() as f64, ref_coords.len());
        let (mine, theirs) = (&decoded.feats, &load("shape_voxels.bin"));
        for (k, label) in [(0..3, "dual vertex logits"), (3..6, "edge flags"), (6..7, "split weight")] {
            let a: Vec<f32> = common.iter().flat_map(|&(i, _)| k.clone().map(move |c| mine[i * 7 + c])).collect();
            let b: Vec<f32> = common.iter().flat_map(|&(_, j)| k.clone().map(move |c| theirs[j * 7 + c])).collect();
            eprintln!("{label}: {:.2e}", relative(&a, &b));
        }
        let flags = common.iter().filter(|&&(i, j)| (3..6).all(|c| (mine[i * 7 + c] > 0.) == (theirs[j * 7 + c] > 0.))).count();
        eprintln!("edge flags alike on {:.3}% of the common voxels", 100. * flags as f64 / common.len() as f64);
        let tex = WgpuSparseDecoder::load(&ckpts.join("tex_dec_next_dc_f16c32_fp16"), &gpu)?;
        let attrs = tex.forward(&gpu, WgpuLevel::new(&gpu, hr, 64), &load("tex_hr_final.bin"), Some(&decoded.subdivisions))?;
        let theirs = &load("tex_voxels_raw.bin");
        let feats = &attrs.feats;
        let a: Vec<f32> = common.iter().flat_map(|&(i, _)| (0..6).map(move |c| feats[i * 6 + c])).collect();
        let b: Vec<f32> = common.iter().flat_map(|&(_, j)| (0..6).map(move |c| theirs[j * 6 + c])).collect();
        eprintln!("texture attributes: {:.2e}", relative(&a, &b));
        Ok(())
    }

    /// The structure flow model's step and the 512 shape model's on the official pipeline's dumped inputs, against its
    /// outputs (`--ignored --nocapture`; P3D_DIR, else E:/p3dref/cmp4/ref; MODELS, else E:/models; the reference ran its
    /// blocks in BF16).
    #[test]
    #[ignore = "needs Pixal3D and the reference's dumps"]
    fn the_webgpu_flow_steps_are_the_references() -> Result<()> {
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let ckpts = models().join("Pixal3D/ckpts");
        let global = load("global512.bin");
        // the structure: every voxel of 16^3, the noise as the reference draws it ([8, 4096] as tokens)
        let dit = WgpuDit3::load(&ckpts.join("ss_flow_img_dit_1_3B_64_bf16"), &gpu)?;
        let dense: Vec<[i32; 3]> = (0..16).flat_map(|x| (0..16).flat_map(move |y| (0..16).map(move |z| [x, y, z]))).collect();
        let n = dense.len();
        let planes = load("ss.bin");
        let x: Vec<f32> = (0..n * 8).map(|i| planes[(i % 8) * n + i / 8]).collect();
        let table = upload(&gpu, &super::table(&dense, 128));
        let pos = dit.context(&gpu, Some(&global), Some(&load("proj_ss.bin")), 5);
        let neg = dit.context(&gpu, None, None, 5);
        let started = std::time::Instant::now();
        let v = velocity(&gpu, &dit, &x, n, 1000., &table, &pos);
        eprintln!("structure step, conditional: {:.2e} ({:.3} s)", relative(&v, &load("step_pos.bin")), started.elapsed().as_secs_f64());
        eprintln!("structure step, unconditional: {:.2e}", relative(&velocity(&gpu, &dit, &x, n, 1000., &table, &neg), &load("step_neg.bin")));
        let half: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        eprintln!("structure step, halfway: {:.2e}", relative(&velocity(&gpu, &dit, &half, n, 400., &table, &pos), &load("step_mid.bin")));
        drop((dit, pos, neg));
        // the shape at 512 over the structure's voxels
        let dit = WgpuDit3::load(&ckpts.join("slat_flow_img2shape_dit_1_3B_512_bf16"), &gpu)?;
        let c = coords("coords32.i32");
        let n = c.len();
        let table = upload(&gpu, &super::table(&c, 128));
        let pos = dit.context(&gpu, Some(&global), Some(&load("proj_shape512.bin")), 5);
        let neg = dit.context(&gpu, None, None, 5);
        let x = load("shape_lr.bin");
        let started = std::time::Instant::now();
        let v = velocity(&gpu, &dit, &x, n, 1000., &table, &pos);
        eprintln!("shape step ({n} voxels), conditional: {:.2e} ({:.3} s)", relative(&v, &load("sparse_pos.bin")), started.elapsed().as_secs_f64());
        eprintln!("shape step, unconditional: {:.2e}", relative(&velocity(&gpu, &dit, &x, n, 1000., &table, &neg), &load("sparse_neg.bin")));
        Ok(())
    }
}
