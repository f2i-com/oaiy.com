//! MOSS-Image's transformer on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as OpenMOSS's `MOSSImageTransformer`
//! (`pipeline_moss_image.py`) computes it: one stream of the picture's tokens then the text's. The text's tokens first
//! through their own projection and two blocks of their own (biased, attending among themselves: [`WgpuMoss::prompt`],
//! once a prompt); then every token through the 42 blocks attending together, all modulated alike by the step's time
//! (one modulation the blocks share); the picture's tokens out. A block is an RMS norm (its own weight) modulated, then
//! attention (q and k RMS-normed a head at a time and rotated by four axes of position, each head's halves NeoX's)
//! and an MLP whose value and gate are each soft-capped (SiTU), each added back by its gate.
//!
//! Its matrices as [`crate::flux3::webgpu`] holds FLUX 3's: a checkpoint's floats f16 (its BF16 as Q8_0 where the
//! tensor cores take Q8_0 and not f16, or the card has no room for f16 beside a picture's work), a GGUF's K-quant
//! blocks as they are. The text encoder is [`text`]. A distillation's factors ([`crate::moss_image::distill`]) are
//! kept beside each block's matrices as f16, added as the step runs (`B A x`, in [`mul`]).
pub mod text;

use crate::common::lora::Loras;
use crate::common::weights::Weights;
use crate::moss_image::distill::{Distillation, Factors, REPLACES};
use crate::qwen_image::webgpu::transformer::{err, first, matrix_as, mul, vector, weights_q8, Low, Mat};
use candle_core::Result;
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use std::path::Path;

const EPS: f32 = 1e-6;
const ROPE_BASE: f64 = 10_000.;
/// The time's sinusoids: 128 frequencies, cosines then sines.
const FREQ: usize = 256;
/// The text's own blocks.
const TEXT_BLOCKS: usize = 2;

/// The transformer's shape, read from its weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    /// A head's width (128): four axes of position, an eighth of it each in frequencies.
    pub head_dim: usize,
    pub layers: usize,
    /// The MLP's width.
    pub mlp: usize,
    /// A latent token's channels.
    pub channels: usize,
    /// A text token's: the text encoder's states.
    pub context: usize,
}

impl Config {
    /// The shape of the checkpoint `w` holds, checked as far as its names and shapes say.
    pub fn of(w: &Weights) -> Result<Self> {
        let not = |why: String| candle_core::Error::Msg(format!("not MOSS-Image's transformer: {why}"));
        let shape = |name: &str| w.shape(name).map_err(|_| not(format!("no {name}")));
        let (video, text) = (shape("video_embedder.weight")?, shape("text_embedder.proj.weight")?);
        let (&[hidden, channels], &[h2, context]) = (video.as_slice(), text.as_slice()) else {
            return Err(not(format!("video_embedder {video:?}, text_embedder.proj {text:?}")));
        };
        let head_dim = shape("layers.0.q_norm.weight")?.iter().product::<usize>();
        let mlp = shape("layers.0.proj.ffn_up_gate.weight")?.first().copied().unwrap_or(0) / 2;
        // (ComfyUI's file keeps each block's q, k and v as one `proj.qkv`)
        let layers = (0..).take_while(|i| w.has(&format!("layers.{i}.proj.q.weight")) || w.has(&format!("layers.{i}.proj.qkv.weight"))).count();
        if h2 != hidden || head_dim == 0 || head_dim % 8 != 0 || hidden % head_dim != 0 || channels != 128 || mlp == 0 || layers == 0 {
            return Err(not(format!("hidden {hidden}, heads {head_dim} wide, latent channels {channels}, MLP {mlp}, {layers} blocks")));
        }
        if shape("shared_block_modulation.weight")? != [6 * hidden, hidden] || !w.has(&format!("text_embedder.blocks.{}.qkv.weight", TEXT_BLOCKS - 1)) {
            return Err(not("its modulation or text blocks".into()));
        }
        Ok(Self { hidden, heads: hidden / head_dim, head_dim, layers, mlp, channels, context })
    }
}

/// The bands of each axis's frequencies (an eighth of a head), as `_RoPE._default_bands` makes them in f32.
fn bands(head_dim: usize) -> Vec<f32> {
    let n = head_dim / 8;
    (0..n).map(|j| 1.0 / (ROPE_BASE as f32).powf(j as f32 / n as f32)).collect()
}

/// Each position's RoPE pairs as the chain's table has them (a pair's sine, then its cosine): NeoX's halves, pair `k` of
/// a head rotated by axis `k / bands`'s coordinate times band `k % bands`, the angle made in f32 as `_RoPE` makes it.
pub fn rope_table(coords: &[[f32; 4]], head_dim: usize) -> Vec<f32> {
    let b = bands(head_dim);
    let mut t = Vec::with_capacity(coords.len() * head_dim);
    for c in coords {
        for k in 0..head_dim / 2 {
            let phase = c[k / b.len()] * b[k % b.len()];
            t.push((phase as f64).sin() as f32);
            t.push((phase as f64).cos() as f32);
        }
    }
    t
}

/// The positions of an `h` by `w` picture's tokens (rows, then columns) after `text` text tokens, then the text's:
/// a picture's `(0, text, row, column)`, its rows and columns about its centre; the text's `(0, index, 0, 0)`.
pub fn positions(h: usize, w: usize, text: usize) -> Vec<[f32; 4]> {
    let (ch, cw) = ((h as f32 - 1.) / 2., (w as f32 - 1.) / 2.);
    let image = (0..h * w).map(|i| [0., text as f32, (i / w) as f32 - ch, (i % w) as f32 - cw]);
    image.chain((0..text).map(|i| [0., i as f32, 0., 0.])).collect()
}

/// A step's time (the scheduler's, 0 to 1000) as the transformer's 256 sinusoids: cosines, then sines, of
/// `t 10000^(-j / 128)`, made in f64 as `MOSSImageTransformer.forward` makes them.
pub fn timestep(t: f32) -> Vec<f32> {
    let half = FREQ / 2;
    let phase = |j: usize| t as f64 * 10_000f64.powf(-(j as f64) / half as f64);
    (0..half).map(|j| phase(j).cos() as f32).chain((0..half).map(|j| phase(j).sin() as f32)).collect()
}

/// `values` (`what`'s) as f16 on `gpu`.
fn f16_vec(gpu: &ggml_rs_wgpu::WgpuBackend, values: &[f32], what: &str) -> Result<DeviceVec> {
    let words = crate::common::webgpu_weights::f16_words_f32(values).ok_or_else(|| err(format!("{what}: an odd count, or past f16's range")))?;
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    Ok(v)
}

/// A linear layer's weight and (where it has one) bias.
struct Linear {
    m: Mat,
    b: Option<DeviceVec>,
}

/// A block's q, k and v: three matrices, or (ComfyUI's file) one of their rows stacked.
enum Qkv {
    Apart([Mat; 3]),
    Fused(Mat),
}

struct Block {
    qkv: Qkv,
    o: Mat,
    /// The MLP's value rows, then its gate's (`ffn_up_gate`).
    up_gate: Mat,
    down: Mat,
    qn: DeviceVec,
    kn: DeviceVec,
    /// The two norms' weights side by side, and each less 1.
    norms: DeviceVec,
    norms_less_1: DeviceVec,
}

/// A text block (the text's own: biased, no modulation).
struct TextBlock {
    qkv: Linear,
    proj: Linear,
    up: Linear,
    gate: Linear,
    down: Linear,
    norm1: DeviceVec,
    norm2: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
}

/// A prompt as the blocks take it: its text tokens through the text's projection and blocks (`[n, hidden]`).
pub struct Prompt {
    tokens: DeviceVec,
    n: usize,
}

impl Prompt {
    /// Its tokens.
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// A step's vectors for at most `rows` tokens, kept for the next step.
struct Scratch {
    rows: usize,
    latent: DeviceVec,
    x: DeviceVec,
    norm: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    o: DeviceVec,
    fused: DeviceVec,
    act: DeviceVec,
    vel: DeviceVec,
    /// The time's sinusoids, the hidden layer of its embedding, its embedding, the blocks' modulation (six vectors),
    /// the final norm's (two), and the two norms' scales and shifts side by side.
    t: DeviceVec,
    hidden: DeviceVec,
    time: DeviceVec,
    mods: DeviceVec,
    head: DeviceVec,
    scales: DeviceVec,
    shifts: DeviceVec,
    /// Each block's two norms' scales (`w (1 + scale) - 1`), then their shifts; the final norm's.
    block_mods: Vec<DeviceVec>,
    final_mods: DeviceVec,
    low: Low,
}

pub struct WgpuMoss {
    gpu: ggml_rs_wgpu::WgpuBackend,
    cfg: Config,
    video: Linear,
    text_in: Linear,
    text_blocks: Vec<TextBlock>,
    time: [Linear; 2],
    modulation: Linear,
    head_modulation: Linear,
    blocks: Vec<Block>,
    final_norm: DeviceVec,
    final_norm_less_1: DeviceVec,
    final_proj: Mat,
    /// A distillation's largest rank and widest adapted output (the running vectors of its factors; 0: none).
    rank: usize,
    widest: usize,
    /// Its blocks' matrices quantized (Q8_0 of the BF16 checkpoint's, or a GGUF's), not f16.
    quantized: bool,
    scratch: Option<Scratch>,
    /// Each picture size and prompt length's RoPE table.
    tables: Vec<((usize, usize, usize), DeviceVec)>,
}

impl WgpuMoss {
    /// The transformer at `path` on GPU `device` (as the computer counts them; OAIY_WEBGPU_ADAPTER naming one
    /// instead). `progress(done, of)` as its blocks load.
    pub fn load(path: &Path, device: usize, progress: impl FnMut(usize, usize)) -> Result<Self> {
        Self::load_on(path, ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?, progress)
    }

    /// [`Self::load`] onto `gpu` (the VAE's beside it).
    pub fn load_on(path: &Path, gpu: ggml_rs_wgpu::WgpuBackend, progress: impl FnMut(usize, usize)) -> Result<Self> {
        Self::load_with(path, gpu, None, progress)
    }

    /// [`Self::load_on`] with `distill`'s factors beside each block's projections and its time embedding and
    /// modulations' biases in place of the transformer's.
    pub fn load_with(path: &Path, gpu: ggml_rs_wgpu::WgpuBackend, mut distill: Option<&mut Distillation>, mut progress: impl FnMut(usize, usize)) -> Result<Self> {
        let mut w = Weights::open(path)?;
        let cfg = Config::of(&w)?;
        let mut none = Loras::open(&[])?;
        // its blocks' matrices f16 where the tensor cores take f16 and there is room for them beside a picture's work;
        // else its BF16 ones as Q8_0, half the size (on Metal's simdgroup matrices Q8_0's matmuls are the tensor
        // cores' and f16's are not). OAIY_MOSS_WEIGHTS `f16` or `q8_0` says which.
        let block_params = 4 * cfg.hidden * cfg.hidden + 3 * cfg.hidden * cfg.mlp;
        let f16_bytes = 2 * (block_params * cfg.layers) as u64;
        let q8 = match std::env::var("OAIY_MOSS_WEIGHTS").map(|v| v.to_ascii_lowercase()).as_deref() {
            Ok("f16") => false,
            Ok("q8_0" | "q8") => true,
            _ => !gpu.f16_tensor_cores() || weights_q8(&gpu, f16_bytes),
        };
        // (the inputs, the times, the modulations and the text's blocks f16 whatever the blocks are: run once a step or
        // a prompt, some read their inputs as f32)
        let linear = |w: &mut Weights, none: &mut Loras, name: &str, bias: bool| -> Result<Linear> {
            Ok(Linear { m: matrix_as(w, none, &gpu, name, false)?, b: if bias { Some(vector(w, &gpu, &format!("{name}.bias"), 0.)?) } else { None } })
        };
        let video = linear(&mut w, &mut none, "video_embedder", true)?;
        let text_in = linear(&mut w, &mut none, "text_embedder.proj", true)?;
        let mut text_blocks = Vec::with_capacity(TEXT_BLOCKS);
        for i in 0..TEXT_BLOCKS {
            let p = format!("text_embedder.blocks.{i}");
            text_blocks.push(TextBlock {
                qkv: linear(&mut w, &mut none, &format!("{p}.qkv"), true)?,
                proj: linear(&mut w, &mut none, &format!("{p}.proj"), true)?,
                up: linear(&mut w, &mut none, &format!("{p}.ffn_up"), true)?,
                gate: linear(&mut w, &mut none, &format!("{p}.ffn_gate"), true)?,
                down: linear(&mut w, &mut none, &format!("{p}.ffn_down"), true)?,
                norm1: vector(&mut w, &gpu, &format!("{p}.norm1.weight"), 0.)?,
                norm2: vector(&mut w, &gpu, &format!("{p}.norm2.weight"), 0.)?,
                qn: vector(&mut w, &gpu, &format!("{p}.q_norm.weight"), 0.)?,
                kn: vector(&mut w, &gpu, &format!("{p}.k_norm.weight"), 0.)?,
            });
        }
        // (a distillation's in place of these, each of the shape it replaces)
        if let Some(d) = distill.as_deref_mut() {
            for name in REPLACES {
                let Some(dw) = d.replacing(name) else { candle_core::bail!("the distillation has no {name}") };
                if dw.shape(name)? != w.shape(name)? {
                    candle_core::bail!("the distillation's {name} is {:?}, the transformer's {:?}", dw.shape(name)?, w.shape(name)?);
                }
            }
        }
        let mut time_layer = |w: &mut Weights, name: &str| -> Result<Linear> {
            match distill.as_deref_mut().and_then(|d| d.replacing(&format!("{name}.weight"))) {
                Some(dw) => linear(dw, &mut none, name, true),
                None => linear(w, &mut none, name, true),
            }
        };
        let time = [time_layer(&mut w, "time_embedding.0")?, time_layer(&mut w, "time_embedding.2")?];
        let mut modulated = |w: &mut Weights, name: &str| -> Result<Linear> {
            let m = matrix_as(w, &mut Loras::open(&[])?, &gpu, name, false)?;
            let bias = format!("{name}.bias");
            let b = match distill.as_deref_mut().and_then(|d| d.replacing(&bias)) {
                Some(dw) => vector(dw, &gpu, &bias, 0.)?,
                None => vector(w, &gpu, &bias, 0.)?,
            };
            Ok(Linear { m, b: Some(b) })
        };
        let modulation = modulated(&mut w, "shared_block_modulation")?;
        let head_modulation = modulated(&mut w, "video_head_modulation")?;
        let host = |w: &mut Weights, name: &str| -> Result<Vec<f32>> { w.tensor(name, &candle_core::Device::Cpu, candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let upload = |v: &[f32]| {
            let d = gpu.vec(v.len());
            gpu.upload(&d, v);
            d
        };
        let mut blocks = Vec::with_capacity(cfg.layers);
        let (mut rank, mut widest) = (0, 0);
        for i in 0..cfg.layers {
            let p = format!("layers.{i}");
            let fused = w.has(&format!("{p}.proj.qkv.weight"));
            let mut m = |name: &str| matrix_as(&mut w, &mut none, &gpu, &format!("{p}.proj.{name}"), q8);
            let mut qkv = if fused { Qkv::Fused(m("qkv")?) } else { Qkv::Apart([m("q")?, m("k")?, m("v")?]) };
            let (mut o, mut up_gate, mut down) = (m("o")?, m("ffn_up_gate")?, m("ffn_down")?);
            if let Some(d) = distill.as_deref_mut() {
                let mut beside = |mat: &mut Mat, f: Factors| -> Result<()> {
                    if (f.out, f.input) != (mat.n, mat.k) {
                        candle_core::bail!("{p}: the distillation's factors are for a {}x{} matrix, not {}x{}", f.out, f.input, mat.n, mat.k);
                    }
                    (rank, widest) = (rank.max(f.rank), widest.max(f.out));
                    mat.lora.push((f16_vec(&gpu, &f.a, &p)?, f16_vec(&gpu, &f.b, &p)?, f.rank));
                    Ok(())
                };
                let mut take = |module: &str| d.factors(i, module);
                match &mut qkv {
                    Qkv::Apart(mats) => {
                        for (mat, module) in mats.iter_mut().zip(["q", "k", "v"]) {
                            if let Some(f) = take(module)? {
                                beside(mat, f)?;
                            }
                        }
                    }
                    // ComfyUI's one matrix: the three factors' `A`s stacked and their `B`s along its diagonal
                    Qkv::Fused(mat) => {
                        let parts = [take("q")?, take("k")?, take("v")?];
                        if parts.iter().any(Option::is_some) {
                            let Some(r) = parts.iter().flatten().map(|f| f.rank).max() else { unreachable!() };
                            let (d3, k) = (mat.n / 3, mat.k);
                            let (mut a, mut b) = (vec![0f32; 3 * r * k], vec![0f32; mat.n * 3 * r]);
                            for (j, f) in parts.iter().enumerate() {
                                let Some(f) = f else { continue };
                                if (f.out, f.input) != (d3, k) {
                                    candle_core::bail!("{p}: the distillation's factors are for a {}x{} matrix, not {d3}x{k}", f.out, f.input);
                                }
                                a[j * r * k..(j * r + f.rank) * k].copy_from_slice(&f.a);
                                for row in 0..d3 {
                                    let at = (j * d3 + row) * 3 * r + j * r;
                                    b[at..at + f.rank].copy_from_slice(&f.b[row * f.rank..(row + 1) * f.rank]);
                                }
                            }
                            beside(mat, Factors { a, b, rank: 3 * r, out: mat.n, input: k })?;
                        }
                    }
                }
                for (mat, module) in [(&mut o, "o"), (&mut up_gate, "ffn_up_gate"), (&mut down, "ffn_down")] {
                    if let Some(f) = take(module)? {
                        beside(mat, f)?;
                    }
                }
            }
            let mut norms = host(&mut w, &format!("{p}.norm1.weight"))?;
            norms.extend(host(&mut w, &format!("{p}.norm2.weight"))?);
            let less_1: Vec<f32> = norms.iter().map(|v| v - 1.).collect();
            blocks.push(Block {
                qkv,
                o,
                up_gate,
                down,
                qn: vector(&mut w, &gpu, &format!("{p}.q_norm.weight"), 0.)?,
                kn: vector(&mut w, &gpu, &format!("{p}.k_norm.weight"), 0.)?,
                norms: upload(&norms),
                norms_less_1: upload(&less_1),
            });
            progress(i + 1, cfg.layers);
        }
        let final_values = host(&mut w, "final_norm_video.weight")?;
        let final_less_1: Vec<f32> = final_values.iter().map(|v| v - 1.).collect();
        let final_proj = matrix_as(&mut w, &mut none, &gpu, "final_linear_video", false)?;
        let quantized = blocks.iter().any(|b| matches!(b.o.w, crate::qwen_image::webgpu::transformer::W::Quant(_)));
        Ok(Self {
            final_norm: upload(&final_values),
            final_norm_less_1: upload(&final_less_1),
            gpu,
            cfg,
            video,
            text_in,
            text_blocks,
            time,
            modulation,
            head_modulation,
            blocks,
            final_proj,
            rank,
            widest,
            quantized,
            scratch: None,
            tables: Vec::new(),
        })
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    /// Whether its blocks' matrices are quantized (Q8_0 of the checkpoint's BF16 ones, or a GGUF's), not f16.
    pub fn quantized(&self) -> bool {
        self.quantized
    }

    /// The GPU it is on (the VAE beside it).
    pub fn backend(&self) -> &ggml_rs_wgpu::WgpuBackend {
        &self.gpu
    }

    /// Let go of a step's vectors and what the device keeps for the next (the VAE's decode wants the room), made
    /// again by the next step.
    pub fn release_scratch(&mut self) {
        self.scratch = None;
        self.tables.clear();
        self.gpu.release_cached();
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn upload(&self, values: &[f32]) -> DeviceVec {
        let v = self.vec(values.len());
        self.gpu.upload(&v, values);
        v
    }

    /// `y = W x (+ b)` for `rows` rows.
    fn linear(r: &mut dyn ChainRecorder, l: &Linear, x: &DeviceVec, y: &DeviceVec, rows: usize, exact: bool, low: &Low) {
        mul(r, &l.m, x, y, rows, exact, low);
        if let Some(b) = &l.b {
            r.add_bias_rows(y, b, rows, l.m.n);
        }
    }

    /// The RoPE table of an `h` by `w` picture's tokens after `text` text tokens, then the text's (made once).
    fn table(&mut self, h: usize, w: usize, text: usize) -> DeviceVec {
        if let Some((_, t)) = self.tables.iter().find(|(k, _)| *k == (h, w, text)) {
            return t.clone();
        }
        let t = self.upload(&rope_table(&positions(h, w, text), self.cfg.head_dim));
        self.tables.push(((h, w, text), t.clone()));
        t
    }

    /// A prompt's text states (`[n, context]`, the text encoder's) through the text's projection and its own two
    /// blocks: what every step takes of it.
    pub fn prompt(&mut self, states: &[f32], n: usize) -> Result<Prompt> {
        let Config { hidden: d, heads, head_dim: hd, mlp, context, .. } = self.cfg;
        if n == 0 || states.len() != n * context {
            candle_core::bail!("MOSS-Image's prompt of {} values for {n} tokens of {context}", states.len());
        }
        let input = self.upload(states);
        let table = self.upload(&rope_table(&positions(0, 0, n)[..], hd));
        let v = |len: usize| self.vec(len);
        let (x, y, qkv, q, k, vv, qq, kk, kv) = (v(n * d), v(n * d), v(n * 3 * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * d), v(n * 2 * d));
        let (att, o, u, g, fused, act) = (v(self.gpu.attention_rows_full_out_len(n, heads, hd, n)), v(n * d), v(n * mlp), v(n * mlp), v(n * 2 * mlp), v(n * mlp));
        let low = Low { t: v(1), y: v(1) };
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        // (the encoder's states read as they are)
        Self::linear(r, &self.text_in, &input, &x, n, true, &low);
        for b in &self.text_blocks {
            r.rmsnorm_rows(&x, &b.norm1, &y, n, EPS);
            Self::linear(r, &b.qkv, &y, &qkv, n, true, &low);
            r.copy_cols(&qkv, &q, n, d, 3 * d, 0);
            r.copy_cols(&qkv, &k, n, d, 3 * d, d);
            r.copy_cols(&qkv, &vv, n, d, 3 * d, 2 * d);
            r.rmsnorm_rows(&q, &b.qn, &qq, n * heads, EPS);
            r.rmsnorm_rows(&k, &b.kn, &kk, n * heads, EPS);
            r.rope_rows(&qq, n, heads, hd, &table, true);
            r.rope_rows(&kk, n, heads, hd, &table, true);
            r.store_rows(&kk, &kv, n, d, 0, 2 * d, 0);
            r.store_rows(&vv, &kv, n, d, 0, 2 * d, d);
            r.attention_rows_full(&qq, &kv, &att, n, heads, heads, hd, n, 1.0 / (hd as f32).sqrt());
            Self::linear(r, &b.proj, &first(&att, n * d), &o, n, true, &low);
            r.add(&x, &o);
            r.rmsnorm_rows(&x, &b.norm2, &y, n, EPS);
            Self::linear(r, &b.up, &y, &u, n, true, &low);
            Self::linear(r, &b.gate, &y, &g, n, true, &low);
            r.store_rows(&u, &fused, n, mlp, 0, 2 * mlp, 0);
            r.store_rows(&g, &fused, n, mlp, 0, 2 * mlp, mlp);
            r.situ_rows(&fused, &act, n, mlp);
            Self::linear(r, &b.down, &act, &o, n, true, &low);
            r.add(&x, &o);
        }
        rec.finish();
        Ok(Prompt { tokens: x, n })
    }

    fn scratch(&mut self, rows: usize) -> Scratch {
        if let Some(s) = self.scratch.take().filter(|s| s.rows >= rows) {
            return s;
        }
        drop(self.scratch.take());
        let Config { hidden: d, heads, head_dim: hd, mlp, channels, layers, .. } = self.cfg;
        let v = |len: usize| self.vec(len);
        Scratch {
            rows,
            latent: v(rows * channels),
            x: v(rows * d),
            norm: v(rows * d),
            q: v(rows * d),
            k: v(rows * d),
            v: v(rows * d),
            qq: v(rows * d),
            kk: v(rows * d),
            kv: v(rows * 2 * d),
            att: v(self.gpu.attention_rows_full_out_len(rows, heads, hd, rows)),
            o: v(rows * d),
            fused: v(rows * (2 * mlp).max(3 * d)),
            act: v(rows * mlp),
            vel: v(rows * channels),
            t: v(FREQ),
            hidden: v(d),
            time: v(d),
            mods: v(6 * d),
            head: v(2 * d),
            scales: v(2 * d),
            shifts: v(2 * d),
            block_mods: (0..layers).map(|_| v(4 * d)).collect(),
            final_mods: v(2 * d),
            low: Low { t: v(rows * self.rank), y: v(rows * self.widest) },
        }
    }

    /// The velocity of an `h` by `w` picture's latent tokens (`[h w, 128]`, a token a row, rows then columns) at the
    /// scheduler's time `t` (0 to 1000), under `prompt`, the blocks `skip` left out (the negative branch's skip-layer
    /// guidance): `[h w, 128]` on the CPU, as `MOSSImageTransformer.forward` returns it.
    pub fn predict(&mut self, latent: &[f32], h: usize, w: usize, t: f32, prompt: &Prompt, skip: &[usize]) -> Result<Vec<f32>> {
        let Config { hidden: d, heads, head_dim: hd, mlp, channels, .. } = self.cfg;
        let (ni, nt) = (h * w, prompt.n);
        if ni == 0 || latent.len() != ni * channels {
            candle_core::bail!("MOSS-Image's latent of {} values for {h}x{w} tokens of {channels}", latent.len());
        }
        let all = ni + nt;
        let table = self.table(h, w, nt);
        let s = self.scratch(all);
        self.gpu.upload(&s.latent, latent);
        self.gpu.upload(&s.t, &timestep(t));
        let mut rec = self.gpu.begin();
        // (bind groups kept would hold each step's own scratch)
        rec.keep_groups(false);
        let r = rec.as_mut();
        let low = &s.low;
        // the time's embedding: Linear, SiLU, Linear; from it (no SiLU between) the blocks' modulation and the final
        // norm's
        Self::linear(r, &self.time[0], &s.t, &s.hidden, 1, true, low);
        r.mul_sigmoid(&s.hidden, &s.hidden, &s.norm, d);
        Self::linear(r, &self.time[1], &first(&s.norm, d), &s.time, 1, true, low);
        Self::linear(r, &self.modulation, &s.time, &s.mods, 1, true, low);
        Self::linear(r, &self.head_modulation, &s.time, &s.head, 1, true, low);
        // each block's norms' modulation, its own weight in it: `w rms(x) (1 + scale) + shift` is the chain's RMS norm
        // modulated by `w (1 + scale) - 1` and `shift` (the six vectors: shift, scale and gate of the attention's,
        // then of the MLP's)
        r.copy(&s.mods, d, &s.scales, 0, d);
        r.copy(&s.mods, 4 * d, &s.scales, d, d);
        r.copy(&s.mods, 0, &s.shifts, 0, d);
        r.copy(&s.mods, 3 * d, &s.shifts, d, d);
        for (b, m) in self.blocks.iter().zip(&s.block_mods) {
            r.copy(&b.norms_less_1, 0, m, 0, 2 * d);
            r.add_gated_rows(m, &b.norms, 1, 2 * d, &s.scales, 0, false);
            r.copy(&s.shifts, 0, m, 2 * d, 2 * d);
        }
        r.copy(&self.final_norm_less_1, 0, &s.final_mods, 0, d);
        r.add_gated_rows(&s.final_mods, &self.final_norm, 1, d, &s.head, d, false);
        r.copy(&s.head, 0, &s.final_mods, d, d);
        // the picture's tokens embedded, the prompt's after them
        let x = first(&s.x, all * d);
        Self::linear(r, &self.video, &s.latent, &x, ni, false, low);
        r.copy(&prompt.tokens, 0, &x, ni * d, nt * d);
        let (norm, q, k, qq, kk) = (first(&s.norm, all * d), first(&s.q, all * d), first(&s.k, all * d), first(&s.qq, all * d), first(&s.kk, all * d));
        let scale = 1.0 / (hd as f32).sqrt();
        for (i, (b, m)) in self.blocks.iter().zip(&s.block_mods).enumerate() {
            if skip.contains(&i) {
                continue;
            }
            r.norm_mod_rows(&x, &norm, all, d, m, 0, Some(2 * d), RowNorm::Rms, EPS);
            match &b.qkv {
                Qkv::Apart([wq, wk, wv]) => {
                    mul(r, wq, &norm, &q, all, false, low);
                    mul(r, wk, &norm, &k, all, false, low);
                    mul(r, wv, &norm, &s.v, all, false, low);
                }
                Qkv::Fused(m) => {
                    // (into the MLP's scratch, three of a row's widths; then each its own)
                    mul(r, m, &norm, &s.fused, all, false, low);
                    r.copy_cols(&s.fused, &q, all, d, 3 * d, 0);
                    r.copy_cols(&s.fused, &k, all, d, 3 * d, d);
                    r.copy_cols(&s.fused, &s.v, all, d, 3 * d, 2 * d);
                }
            }
            r.rmsnorm_rows(&q, &b.qn, &qq, all * heads, EPS);
            r.rmsnorm_rows(&k, &b.kn, &kk, all * heads, EPS);
            r.rope_rows(&qq, all, heads, hd, &table, true);
            r.rope_rows(&kk, all, heads, hd, &table, true);
            r.store_rows(&kk, &s.kv, all, d, 0, 2 * d, 0);
            r.store_rows(&s.v, &s.kv, all, d, 0, 2 * d, d);
            r.attention_rows_full(&qq, &s.kv, &s.att, all, heads, heads, hd, all, scale);
            mul(r, &b.o, &first(&s.att, all * d), &s.o, all, false, low);
            r.add_gated_rows(&x, &s.o, all, d, &s.mods, 2 * d, false);
            r.norm_mod_rows(&x, &norm, all, d, m, d, Some(3 * d), RowNorm::Rms, EPS);
            mul(r, &b.up_gate, &norm, &s.fused, all, false, low);
            r.situ_rows(&s.fused, &s.act, all, mlp);
            mul(r, &b.down, &s.act, &s.o, all, false, low);
            r.add_gated_rows(&x, &s.o, all, d, &s.mods, 5 * d, false);
        }
        // the picture's tokens out
        r.norm_mod_rows(&x, &norm, ni, d, &s.final_mods, 0, Some(d), RowNorm::Rms, EPS);
        mul(r, &self.final_proj, &norm, &s.vel, ni, false, low);
        r.read_range(&s.vel, 0, ni * channels);
        let vel = rec.finish().pop().ok_or_else(|| err("the step's velocity was not read"))?;
        self.scratch = Some(s);
        Ok(vel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// The bands are `_RoPE`'s (10,000 to minus each eighth of a head's share), the positions a picture's then its
    /// text's, the times' sinusoids cosines first.
    #[test]
    fn positions_bands_and_times_are_the_pipelines() {
        assert_eq!(bands(32), [1.0, 0.1, 0.01, 0.001]);
        let p = positions(2, 3, 5);
        assert_eq!(p.len(), 6 + 5);
        assert_eq!(p[0], [0., 5., -0.5, -1.]);
        assert_eq!(p[5], [0., 5., 0.5, 1.]);
        assert_eq!(p[6 + 4], [0., 4., 0., 0.]);
        let t = timestep(500.);
        assert_eq!((t[0], t[128]), (500f64.cos() as f32, 500f64.sin() as f32));
        let table = rope_table(&[[0., 2., 3., -1.]], 32);
        // (pair 5: axis 1's band 1)
        let (sin, cos) = ((2f32 * 0.1) as f64).sin_cos();
        assert_eq!((table[10], table[11]), (sin as f32, cos as f32));
    }

    /// OpenMOSS's own transformer (`tools/moss_image/reference.py`: its sizes made small, its weights random and f16,
    /// run in f32 on the CPU) and this one on WebGPU give a step's velocity alike, with every block and with block 1
    /// skipped.
    #[test]
    fn a_tiny_transformer_on_webgpu_is_openmosss() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/moss_image/tiny.safetensors");
        let mut w = Weights::open(&path)?;
        let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(&format!("fixture.{name}"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let (latent, size, text, time) = (values("latent")?, values("size")?, values("text")?, values("time")?);
        let wants = [(Vec::new(), values("velocity")?), (vec![1], values("velocity_skipping_1")?)];
        let mut gpu = WgpuMoss::load(&path, 0, |_, _| {})?;
        assert_eq!(gpu.config(), Config { hidden: 64, heads: 2, head_dim: 32, layers: 3, mlp: 96, channels: 128, context: 48 });
        let prompt = gpu.prompt(&text, text.len() / 48)?;
        for (skip, want) in &wants {
            let got = gpu.predict(&latent, size[0] as usize, size[1] as usize, time[0], &prompt, skip)?;
            assert_eq!(got.len(), want.len());
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(&got) * norm(want));
            let rms = norm(want) / (want.len() as f64).sqrt();
            let worst = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
            eprintln!("skipping {skip:?}: cosine {cosine:.7}, the worst error {worst:.2e} of an RMS {rms:.3}");
            assert!(cosine > 0.9999 && worst < 0.01 * rms, "skipping {skip:?}: cosine {cosine}, worst {worst} of {rms}");
        }
        // (skipping a block moves it: the test tells them apart)
        let (a, b) = (&wants[0].1, &wants[1].1);
        assert!(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) > 0.05);
        Ok(())
    }

    /// ComfyUI's layout (each block's q, k and v one `proj.qkv`, their rows stacked) gives the step the separate
    /// matrices give.
    #[test]
    fn a_fused_qkv_is_the_three_matrices() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/moss_image/tiny.safetensors");
        let mut w = Weights::open(&path)?;
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for name in w.names() {
            tensors.push((name.clone(), w.shape(&name)?, w.tensor(&name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?));
        }
        let mut fused: Vec<(String, Vec<usize>, Vec<f32>)> = tensors.iter().filter(|(n, ..)| !(n.ends_with(".proj.q.weight") || n.ends_with(".proj.k.weight") || n.ends_with(".proj.v.weight"))).cloned().collect();
        for i in 0..3 {
            let part = |x: &str| tensors.iter().find(|(n, ..)| *n == format!("layers.{i}.proj.{x}.weight")).unwrap().clone();
            let (q, k, v) = (part("q"), part("k"), part("v"));
            fused.push((format!("layers.{i}.proj.qkv.weight"), vec![3 * q.1[0], q.1[1]], [q.2, k.2, v.2].concat()));
        }
        let file = std::env::temp_dir().join(format!("oaiy-moss-fused-{}.safetensors", std::process::id()));
        let list: Vec<(&str, &[usize], &[f32])> = fused.iter().map(|(n, s, v)| (n.as_str(), s.as_slice(), v.as_slice())).collect();
        crate::common::lora::write_f32(&file, &list, &[]);
        let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(&format!("fixture.{name}"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let (latent, size, text, time) = (values("latent")?, values("size")?, values("text")?, values("time")?);
        let want = values("velocity")?;
        let mut gpu = WgpuMoss::load(&file, 0, |_, _| {})?;
        let prompt = gpu.prompt(&text, text.len() / 48)?;
        let got = gpu.predict(&latent, size[0] as usize, size[1] as usize, time[0], &prompt, &[])?;
        std::fs::remove_file(&file).ok();
        let worst = got.iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
        assert!(worst < 1e-3, "{worst}");
        Ok(())
    }

    /// Every tensor of `path` (a safetensors file), each with its shape and values.
    fn all(path: &Path) -> Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
        let mut w = Weights::open(path)?;
        w.names().into_iter().map(|n| Ok((n.clone(), w.shape(&n)?, w.tensor(&n, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?))).collect()
    }

    fn write(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let list: Vec<(&str, &[usize], &[f32])> = tensors.iter().map(|(n, s, v)| (n.as_str(), s.as_slice(), v.as_slice())).collect();
        crate::common::lora::write_f32(path, &list, &[]);
    }

    /// A distillation (`tools/moss_image/distill_fixture.py`'s for the tiny transformer: OpenMOSS's transformer with
    /// its factors merged and its time embedding and biases in, in f32) gives a step's velocity at two of its times as
    /// the reference does, and its sampler's three passes from the same noise the same latent: in its one file, as
    /// kohya's factors and the two other files in a folder, and beside ComfyUI's fused q, k and v. Without it the
    /// step is another.
    #[test]
    fn a_tiny_distilled_transformer_on_webgpu_is_openmosss() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/moss_image");
        let (tiny, adapter) = (dir.join("tiny.safetensors"), dir.join("tiny_distill.safetensors"));
        let base = all(&tiny)?;
        let parts = all(&adapter)?;
        let value = |list: &[(String, Vec<usize>, Vec<f32>)], name: &str| list.iter().find(|(n, ..)| n == name).map(|t| t.2.clone()).unwrap();
        let (latent, size, text) = (value(&base, "fixture.latent"), value(&base, "fixture.size"), value(&base, "fixture.text"));
        let (h, w) = (size[0] as usize, size[1] as usize);
        let wants = [(1000., value(&parts, "fixture.velocity_1000")), (791., value(&parts, "fixture.velocity_791"))];
        let (noise, sampled) = (value(&parts, "fixture.noise"), value(&parts, "fixture.sampled"));
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let close = |got: &[f32], want: &[f32], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}");
            let cosine = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(got) * norm(want));
            let rms = norm(want) / (want.len() as f64).sqrt();
            let worst = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
            eprintln!("{what}: cosine {cosine:.7}, the worst error {worst:.2e} of an RMS {rms:.3}");
            assert!(cosine > 0.9999 && worst < 0.01 * rms, "{what}: cosine {cosine}, worst {worst} of {rms}");
        };
        let check = |model: &Path, distill: &Path, layout: &str| -> Result<()> {
            let mut d = crate::moss_image::distill::Distillation::open(distill)?;
            assert_eq!(d.codes, [1000., 791., 24.]);
            let mut gpu = WgpuMoss::load_with(model, ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?, Some(&mut d), |_, _| {})?;
            let prompt = gpu.prompt(&text, text.len() / 48)?;
            for (time, want) in &wants {
                close(&gpu.predict(&latent, h, w, *time, &prompt, &[])?, want, &format!("{layout}, time {time}"));
            }
            let got = d.sample(&noise, 3, |_, x, time| gpu.predict(x, h, w, time, &prompt, &[]))?;
            close(&got, &sampled, &format!("{layout}, three passes"));
            Ok(())
        };
        check(&tiny, &adapter, "one file")?;
        // kohya's factors (`lora_down`, `lora_up` and an alpha of twice the rank: each `B` halved to keep the step) and
        // the replacements and the sampler's in files of their own, in a folder
        let folder = std::env::temp_dir().join(format!("oaiy-moss-distill-{}", std::process::id()));
        std::fs::create_dir_all(&folder)?;
        let mut kohya = Vec::new();
        for (name, shape, values) in parts.iter().filter(|(n, ..)| n.contains(".lora_")) {
            let (module, factor) = name.rsplit_once('.').unwrap();
            let up = factor == "lora_B";
            let values = if up { values.iter().map(|v| v / 2.).collect() } else { values.clone() };
            kohya.push((format!("diffusion_model.{module}.{}.weight", if up { "lora_up" } else { "lora_down" }), shape.clone(), values));
            if !up {
                kohya.push((format!("diffusion_model.{module}.alpha"), vec![], vec![2. * shape[0] as f32]));
            }
        }
        write(&folder.join("lora.safetensors"), &kohya);
        let named = |names: &[&str]| parts.iter().filter(|(n, ..)| names.contains(&n.as_str())).cloned().collect::<Vec<_>>();
        write(&folder.join("time_embedding.safetensors"), &named(&crate::moss_image::distill::REPLACES));
        write(&folder.join("dataspace.safetensors"), &named(&["latent_mix.Q", "step_codes"]));
        check(&tiny, &folder, "kohya's, in a folder")?;
        // ComfyUI's one file: each block's q, k and v one matrix, their rows stacked
        let mut fused: Vec<_> = base.iter().filter(|(n, ..)| !(n.ends_with(".proj.q.weight") || n.ends_with(".proj.k.weight") || n.ends_with(".proj.v.weight"))).cloned().collect();
        for i in 0..3 {
            let part = |x: &str| base.iter().find(|(n, ..)| *n == format!("layers.{i}.proj.{x}.weight")).unwrap().clone();
            let (q, k, v) = (part("q"), part("k"), part("v"));
            fused.push((format!("layers.{i}.proj.qkv.weight"), vec![3 * q.1[0], q.1[1]], [q.2, k.2, v.2].concat()));
        }
        let fused_file = folder.join("fused.safetensors");
        write(&fused_file, &fused);
        check(&fused_file, &adapter, "fused q, k and v")?;
        std::fs::remove_dir_all(&folder).ok();
        // (without it the step is another: the test tells them apart)
        let mut plain = WgpuMoss::load(&tiny, 0, |_, _| {})?;
        let prompt = plain.prompt(&text, text.len() / 48)?;
        let got = plain.predict(&latent, h, w, 1000., &prompt, &[])?;
        let apart = got.iter().zip(&wants[0].1).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt() / norm(&wants[0].1);
        assert!(apart > 0.05, "{apart}");
        Ok(())
    }

    /// The sampler's input to each pass is the noise plus the latent mix of the estimate so far, a token at a time.
    #[test]
    fn a_pass_takes_the_noise_and_the_mixed_estimate() -> Result<()> {
        let file = std::env::temp_dir().join(format!("oaiy-moss-dataspace-{}.safetensors", std::process::id()));
        // two channels: Q = [[1, 2], [3, 4]]; two codes
        write(&file, &[("latent_mix.Q".into(), vec![2, 2], vec![1., 2., 3., 4.]), ("step_codes".into(), vec![2], vec![1000., 500.])]);
        let d = crate::moss_image::distill::Distillation::open(&file)?;
        std::fs::remove_file(&file).ok();
        assert_eq!(d.mix_in(&[1., 1., 0., 0.], &[1., 0., 0., 1.]), [2., 4., 2., 4.]);
        // a velocity of 1 everywhere: the first pass sees the noise; the next, the noise and the mix of `z - 1` ([0, 1])
        let mut seen = Vec::new();
        let x0 = d.sample(&[1., 2.], 2, |pass, x, time| {
            seen.push((pass, x.to_vec(), time));
            Ok(vec![1.; 2])
        })?;
        assert_eq!(seen, [(0, vec![1., 2.], 1000.), (1, vec![1. + 2., 2. + 4.], 500.)]);
        assert_eq!(x0, [0., 1.]);
        Ok(())
    }

    /// The real transformer (`OAIY_MOSS_TRANSFORMER`) with a distillation of it (`OAIY_MOSS_DISTILL`) against OpenMOSS's
    /// modules in float32 with its factors merged (`tools/moss_image/distill_reference.py`'s file,
    /// `OAIY_MOSS_DISTILL_REFERENCE`): each pass's velocity from the reference's own input, and the sampler's estimate
    /// after all its passes from the same noise. As f16 (the BF16 checkpoint's exactly) the estimate is the reference's
    /// (the reference with its weights as f16: 0.6% off); on the RTX 4090, whose tensor cores take the inputs as f16,
    /// each pass 0.2 to 0.8% off and the estimate 0.986, the same picture. Quantized, each pass is off by the weights'
    /// rounding (Q8_0's:
    /// 5% of the velocity; the reference with its weights rounded so is off as much), and the passes, each taking the
    /// last's estimate through the small latent mix, carry it on: the estimate is another picture of the same scene
    /// (the reference's own Q8_0 run: cosine 0.865 to its f32 one; this one's 0.857). A Q4_K_M GGUF's passes are 0.93
    /// to 0.98 and its estimate 0.64, still the scene.
    #[test]
    #[ignore = "needs MOSS-Image's transformer (OAIY_MOSS_TRANSFORMER), a distillation (OAIY_MOSS_DISTILL), its reference (OAIY_MOSS_DISTILL_REFERENCE) and a WebGPU adapter"]
    fn a_distilled_step_is_the_references() -> Result<()> {
        let var = |k: &str| std::env::var_os(k).map(std::path::PathBuf::from);
        let (Some(path), Some(distill), Some(reference)) = (var("OAIY_MOSS_TRANSFORMER"), var("OAIY_MOSS_DISTILL"), var("OAIY_MOSS_DISTILL_REFERENCE")) else { return Ok(()) };
        let mut w = Weights::open(&reference)?;
        let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let (noise, text, codes, sampled) = (values("noise")?, values("text")?, values("codes")?, values("sampled")?);
        let passes: Vec<(Vec<f32>, Vec<f32>)> = (0..codes.len()).map_while(|r| Some((values(&format!("pass{r}.input")).ok()?, values(&format!("pass{r}.velocity")).ok()?))).collect();
        let mut d = crate::moss_image::distill::Distillation::open(&distill)?;
        assert_eq!(d.codes, codes);
        let clock = std::time::Instant::now();
        let mut gpu = WgpuMoss::load_with(&path, ggml_rs_wgpu::WgpuBackend::nth(0, None).map_err(err)?, Some(&mut d), |_, _| {})?;
        eprintln!("loaded in {:.1} s", clock.elapsed().as_secs_f64());
        let prompt = gpu.prompt(&text, text.len() / gpu.config().context)?;
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let alike = |got: &[f32], want: &[f32]| {
            let cosine = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(got) * norm(want));
            (cosine, got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt() / norm(want))
        };
        let name = path.to_string_lossy().to_ascii_lowercase();
        let (floor, estimate_floor) = match (gpu.quantized(), name.contains("q4_") || name.contains("q3_") || name.contains("q2_")) {
            (false, _) => (0.9999, 0.98),
            (true, false) => (0.998, 0.8),
            (true, true) => (0.9, 0.5),
        };
        let mut worst = 1f64;
        for (r, (input, want)) in passes.iter().enumerate() {
            let (cosine, error) = alike(&gpu.predict(input, 16, 16, codes[r], &prompt, &[])?, want);
            eprintln!("pass {r} (time {}): cosine {cosine:.6}, error {:.3}% of the velocity", codes[r], 100. * error);
            worst = worst.min(cosine);
        }
        let got = d.sample(&noise, passes.len(), |_, x, time| gpu.predict(x, 16, 16, time, &prompt, &[]))?;
        // (OAIY_MOSS_DISTILL_DUMP: the estimate to a file, `sampled` in it, to be looked at)
        if let Some(dump) = var("OAIY_MOSS_DISTILL_DUMP") {
            crate::common::lora::write_f32(&dump, &[("sampled", &[256, 128], &got)], &[]);
        }
        let (cosine, error) = alike(&got, &sampled);
        eprintln!("{} passes from the noise: cosine {cosine:.6}, error {:.3}% of the estimate", passes.len(), 100. * error);
        assert!(worst > floor && cosine > estimate_floor, "the worst pass's cosine {worst} (floor {floor}), the estimate's {cosine} (floor {estimate_floor})");
        Ok(())
    }

    /// A step of the real transformer (`OAIY_MOSS_TRANSFORMER`: MOSS-Image-SFT's folder, or a GGUF of it) on a 16 x 16
    /// latent under a real prompt's states, with every block and with block 22 skipped, against OpenMOSS's own modules
    /// in float32 (`tools/moss_image/velocity_reference.py`'s file, `OAIY_MOSS_STEP`): as alike as the weights' own
    /// rounding lets it be (f16 or Q8_0 of the BF16 checkpoint, or a GGUF's blocks).
    #[test]
    #[ignore = "needs MOSS-Image's transformer (OAIY_MOSS_TRANSFORMER), its reference step (OAIY_MOSS_STEP) and a WebGPU adapter"]
    fn a_step_is_openmosss() -> Result<()> {
        let var = |k: &str| std::env::var_os(k).map(std::path::PathBuf::from);
        let (Some(path), Some(step)) = (var("OAIY_MOSS_TRANSFORMER"), var("OAIY_MOSS_STEP")) else { return Ok(()) };
        let mut w = Weights::open(&step)?;
        let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let (latent, text, time) = (values("latent")?, values("text")?, values("time")?);
        let wants = [(Vec::new(), values("velocity")?), (vec![22], values("velocity_skipping_22")?)];
        let clock = std::time::Instant::now();
        let mut gpu = WgpuMoss::load(&path, 0, |_, _| {})?;
        eprintln!("loaded in {:.1} s", clock.elapsed().as_secs_f64());
        let prompt = gpu.prompt(&text, text.len() / gpu.config().context)?;
        let name = path.to_string_lossy().to_ascii_lowercase();
        let floor = if name.contains("q4_") || name.contains("q3_") || name.contains("q2_") { 0.99 } else { 0.999 };
        for (skip, want) in &wants {
            let got = gpu.predict(&latent, 16, 16, time[0], &prompt, skip)?;
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(&got) * norm(want));
            let error = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt() / norm(want);
            eprintln!("skipping {skip:?}: cosine {cosine:.6}, error {:.3}% of the velocity", 100. * error);
            assert!(cosine > floor, "skipping {skip:?}: cosine {cosine}");
        }
        Ok(())
    }
}
