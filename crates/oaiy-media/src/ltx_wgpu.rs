//! LTX 2.3's transformer on WebGPU, as [`crate::ltx::transformer`] computes its video stream (text- and
//! image-to-video, its clean first and last frames' tokens at timestep 0, a negative prompt by NAG: no audio yet): its weights on the GPU as they are stored where the chain has a kernel
//! for them (NVFP4 packed, the tensor cores decoding it as they multiply: Lightricks' `-nvfp4` release's 44 blocks of
//! 48), else Q8_0 where the tensor cores take it (a BF16 checkpoint's video stream is 28 GB as f16, 15 as Q8_0), else
//! f16 (BF16 rounded); its activations f32.
use crate::ltx::store::{untile_scales, Store};
use candle_core::{Device, Result};
use dsv41::safetensors::Dtype;
use ggml_rs::{Backend, ChainRecorder, CleanRows, DeviceChain, DeviceVec, QuantizedTensor, RowNorm};
use crate::wgpu_weights::{f16_words, f16_words_f32, q8_0, q8_0_bf16};

const PREFIX: &str = "model.diffusion_model.";
/// The video stream's width, heads and head.
const D: usize = 4096;
const HEADS: usize = 32;
const HD: usize = 128;
const BLOCKS: usize = 48;
const EPS: f32 = 1e-6;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

/// A linear layer on the GPU: its weight NVFP4 (its words and its scale's vector), Q8_0 or f16, and its bias.
pub enum Weight {
    Nvfp4 { w: DeviceVec, scale: DeviceVec },
    Q8(QuantizedTensor),
    F16(DeviceVec),
}

/// A dense weight's values as stored: a BF16 checkpoint's bytes (converted on every core), else f32.
enum Values {
    Bf16(Vec<u8>),
    F32(Vec<f32>),
}

impl Values {
    fn of(store: &mut Store, key: &str) -> Result<Self> {
        Ok(match store.bf16_bytes(key)? {
            Some(bytes) => Self::Bf16(bytes),
            None => Self::F32(store.tensor_f32(key, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
        })
    }
}

/// A dense weight (`[n, k]`) on `gpu`: Q8_0 where its rows are a multiple of 256 long (the tensor cores' kernels, else
/// the f32 tiled one: without tensor cores a 512x320 clip's steps 10.3 s where f16's 11.5, 14.4 GB where 29.5), else
/// f16 (f16 throughout with OAIY_LTX_WEBGPU_WEIGHTS=f16).
fn dense(gpu: &ggml_rs_wgpu::WgpuBackend, key: &str, values: Values, n: usize, k: usize) -> Result<Weight> {
    let f16 = std::env::var("OAIY_LTX_WEBGPU_WEIGHTS").is_ok_and(|v| v.eq_ignore_ascii_case("f16"));
    if k % 256 == 0 && !f16 {
        let bytes = match values {
            Values::Bf16(b) => q8_0_bf16(&b),
            Values::F32(v) => q8_0(&v),
        };
        let size = bytes.len();
        let w = gpu.to_device_quant(QuantizedTensor::from_bytes_cpu(bytes, vec![n, k], ggml_quants::GgmlType::Q8_0));
        if !w.is_device() {
            candle_core::bail!("{key}: no room on the GPU for its {} MB (as Q8_0)", size >> 20);
        }
        return Ok(Weight::Q8(w));
    }
    let words = match values {
        Values::Bf16(b) => f16_words(&b),
        Values::F32(v) => f16_words_f32(&v),
    }
    .ok_or_else(|| err(format!("{key}: past f16's range")))?;
    let v = gpu.vec(words.len());
    gpu.upload(&v, &words);
    Ok(Weight::F16(v))
}

pub struct Linear {
    pub weight: Weight,
    pub bias: DeviceVec,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    /// `name`'s weight and bias (`{name}.weight`, `{name}.bias`; zeros where none) from `store` onto `gpu`.
    pub fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<Self> {
        let key = format!("{name}.weight");
        let info = store.index.get(&key).cloned().ok_or_else(|| err(format!("missing LTX tensor {key}")))?;
        let global_name = format!("{name}.weight_scale_2");
        let (weight, n, k) = if info.dtype == Dtype::U8 && store.index.get(&global_name).is_some() {
            let (n, half) = match info.shape.as_slice() {
                [n, h] => (*n, *h),
                _ => candle_core::bail!("{key}: an NVFP4 weight of shape {:?}", info.shape),
            };
            let k = 2 * half;
            let packed = store.index.read(&key).map_err(err)?;
            let scale_name = format!("{name}.weight_scale");
            let sinfo = store.index.get(&scale_name).cloned().ok_or_else(|| err(format!("missing {scale_name}")))?;
            let (sr, sc) = match sinfo.shape.as_slice() {
                [r, c] => (*r, *c),
                _ => candle_core::bail!("{scale_name}: scales of shape {:?}", sinfo.shape),
            };
            // (the scales' tiles cover rows to 128's: a layer of fewer rows keeps its own)
            let tiled = store.index.read(&scale_name).map_err(err)?;
            let all = untile_scales(&tiled, sr, sc)?;
            if sc != k / 16 || sr < n {
                candle_core::bail!("{scale_name}: {sr} x {sc} scales for a weight of {n} x {k}");
            }
            let scales = &all[..n * sc];
            let g = store.index.read(&global_name).map_err(err)?;
            let global = f32::from_le_bytes(g.get(..4).and_then(|b| b.try_into().ok()).ok_or_else(|| err(format!("{global_name} is not one F32")))?);
            match gpu.nvfp4_weights(&packed, scales, global, n, k) {
                Some((w, scale)) => (Weight::Nvfp4 { w, scale }, n, k),
                None => (dense(gpu, &key, Values::of(store, &key)?, n, k)?, n, k),
            }
        } else {
            let (n, k) = match info.shape.as_slice() {
                [n, k] => (*n, *k),
                _ => candle_core::bail!("{key}: a weight of shape {:?}", info.shape),
            };
            (dense(gpu, &key, Values::of(store, &key)?, n, k)?, n, k)
        };
        let bias_name = format!("{name}.bias");
        let bias: Vec<f32> = if store.index.get(&bias_name).is_some() { store.tensor_f32(&bias_name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()? } else { vec![0.0; n] };
        let b = gpu.vec(n);
        gpu.upload(&b, &bias);
        Ok(Self { weight, bias: b, n, k })
    }

    /// `y[r] = W x[r] + b` for `rows` rows.
    pub fn forward(&self, rec: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match &self.weight {
            Weight::Nvfp4 { w, scale } => rec.matmul_nvfp4_rows(w, scale, &self.bias, self.n, self.k, x, y, rows),
            Weight::Q8(w) => {
                rec.matmul_rows(w, x, y, rows);
                rec.add_bias_rows(y, &self.bias, rows, self.n);
            }
            Weight::F16(w) => {
                rec.matmul_f16_rows(w, self.n, self.k, x, y, rows);
                rec.add_bias_rows(y, &self.bias, rows, self.n);
            }
        }
    }
}

/// One attention's layers: q, k, v, its gate's logits and its output, and q's and k's RMS norms (over the width).
struct Attn {
    q: Linear,
    k: Linear,
    v: Linear,
    gate: Linear,
    out: Linear,
    qn: DeviceVec,
    kn: DeviceVec,
}

impl Attn {
    fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self {
            q: Linear::load(store, gpu, &format!("{p}.to_q"))?,
            k: Linear::load(store, gpu, &format!("{p}.to_k"))?,
            v: Linear::load(store, gpu, &format!("{p}.to_v"))?,
            gate: Linear::load(store, gpu, &format!("{p}.to_gate_logits"))?,
            out: Linear::load(store, gpu, &format!("{p}.to_out.0"))?,
            qn: vector(store, gpu, &format!("{p}.q_norm.weight"))?,
            kn: vector(store, gpu, &format!("{p}.k_norm.weight"))?,
        })
    }
}

struct Block {
    attn1: Attn,
    attn2: Attn,
    ff0: Linear,
    ff2: Linear,
    /// The block's nine modulation rows' own part (`[9, D]`), and the prompt's two (`[2, D]`).
    table: DeviceVec,
    prompt_table: DeviceVec,
}

/// A timestep's embedder (`{p}.emb.timestep_embedder.linear_1/2`) and its modulation's linear (`{p}.linear`).
struct TimeEmbed {
    t1: Linear,
    t2: Linear,
    linear: Linear,
}

impl TimeEmbed {
    fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self {
            t1: Linear::load(store, gpu, &format!("{p}.emb.timestep_embedder.linear_1"))?,
            t2: Linear::load(store, gpu, &format!("{p}.emb.timestep_embedder.linear_2"))?,
            linear: Linear::load(store, gpu, &format!("{p}.linear"))?,
        })
    }

    /// `sigma`'s modulation into `m` and its embedding into `e` (`t` its 256 sinusoids, uploaded; `s1`, `s2` scratch
    /// of the embedder's width).
    fn record(&self, r: &mut dyn ChainRecorder, t: &DeviceVec, s1: &DeviceVec, s2: &DeviceVec, e: &DeviceVec, m: &DeviceVec) {
        self.t1.forward(r, t, s1, 1);
        r.mul_sigmoid(s1, s1, s2, self.t1.n);
        self.t2.forward(r, s2, e, 1);
        r.mul_sigmoid(e, e, s1, self.t2.n);
        self.linear.forward(r, s1, m, 1);
    }
}

/// A norm's weight (or any vector) as f32 on `gpu`.
fn vector(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, name: &str) -> Result<DeviceVec> {
    let values = store.tensor_f32(name, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let v = gpu.vec(values.len());
    gpu.upload(&v, &values);
    Ok(v)
}

/// The timestep's 256 sinusoids as the reference's embedder takes them (cosines, then sines).
fn sinusoids(sigma: f64) -> Vec<f32> {
    let phase: Vec<f32> = (0..128).map(|j| (sigma as f32 * 1000.) * (-10000f32.ln() * j as f32 / 128.).exp()).collect();
    phase.iter().map(|p| p.cos()).chain(phase.iter().map(|p| p.sin())).collect()
}

/// LTX's split rotary tables for `positions` (each its coordinates, `maxima` their ranges) over `dim` channels of
/// `heads` heads, as [`crate::ltx::transformer::Rope::positions`] makes them (in f64): each (position, head) its
/// `dim / heads / 2` pairs' sine then cosine, the chain's layout.
pub fn rope_table(positions: &[Vec<f32>], maxima: &[f32], dim: usize, heads: usize) -> Vec<f32> {
    let axes = maxima.len();
    let half = dim / 2;
    let count = dim / (2 * axes);
    let pad = half - count * axes;
    let mut t = Vec::with_capacity(positions.len() * dim);
    for p in positions {
        for _ in 0..pad {
            t.extend([0f32, 1f32]);
        }
        for j in 0..count {
            let freq = 10000f64.powf(j as f64 / (count - 1) as f64) * std::f64::consts::FRAC_PI_2;
            for axis in 0..axes {
                let phase = freq * (2. * p[axis] as f64 / maxima[axis] as f64 - 1.);
                t.extend([phase.sin() as f32, phase.cos() as f32]);
            }
        }
    }
    let _ = heads;
    t
}

/// A video's tokens' coordinates (time in seconds, then its pixels' row and column), as the reference places them;
/// with `end_image`, an appended end frame's tokens at the last frame's time.
pub fn video_positions(frames: usize, height: usize, width: usize, fps: usize, end_image: bool) -> Vec<Vec<f32>> {
    let mut positions = Vec::with_capacity((frames + usize::from(end_image)) * height * width);
    for t in 0..frames {
        for y in 0..height {
            for x in 0..width {
                let start = (t * 8).saturating_sub(7);
                let end = (t + 1) * 8 - 7;
                positions.push(vec![(start + end) as f32 / (2 * fps) as f32, (y as f32 + 0.5) * 32., (x as f32 + 0.5) * 32.]);
            }
        }
    }
    if end_image {
        let last = ((frames - 1) * 8) as f32 + 0.5;
        for y in 0..height {
            for x in 0..width {
                positions.push(vec![last / fps as f32, (y as f32 + 0.5) * 32., (x as f32 + 0.5) * 32.]);
            }
        }
    }
    positions
}

pub struct WgpuLtx {
    gpu: ggml_rs_wgpu::WgpuBackend,
    patchify: Linear,
    /// LTX 2.5's learned marker of the first latent frame's tokens (added after the patchify).
    keyframe: Option<DeviceVec>,
    adaln: TimeEmbed,
    prompt: Option<TimeEmbed>,
    out_table: DeviceVec,
    proj_out: Linear,
    blocks: Vec<Block>,
    /// A negative prompt by normalised attention guidance, for every forward until cleared (as the reference's
    /// `ltx::transformer::Transformer::nag`).
    pub nag: Option<Nag>,
}

/// A negative prompt's video context (`rows` rows of `D`, the connector's, as the prompt's) and normalised attention
/// guidance's scale, tau and alpha: each block's text cross-attention is run over it too, and the two outputs mixed
/// ([`ChainRecorder::nag_mix`]) before the heads' gate and the output projection. One more cross-attention a block,
/// not a second pass of the model.
pub struct Nag {
    pub context: Vec<f32>,
    pub rows: usize,
    pub scale: f32,
    pub tau: f32,
    pub alpha: f32,
}

impl WgpuLtx {
    /// The video stream of the transformer in `store` on GPU `device` (as CUDA counts them; OAIY_WEBGPU_ADAPTER naming
    /// one instead). `progress(block)` as each loads.
    pub fn load(store: &mut Store, device: usize, mut progress: impl FnMut(usize)) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let g = |n: &str| format!("{PREFIX}{n}");
        let patchify = Linear::load(store, &gpu, &g("patchify_proj"))?;
        let adaln = TimeEmbed::load(store, &gpu, &g("adaln_single"))?;
        let prompt = if store.index.get(&g("prompt_adaln_single.linear.weight")).is_some() { Some(TimeEmbed::load(store, &gpu, &g("prompt_adaln_single"))?) } else { None };
        let out_table = vector(store, &gpu, &g("scale_shift_table"))?;
        let keyframe = if store.index.get(&g("keyframes_abs_pos_embedding")).is_some() { Some(vector(store, &gpu, &g("keyframes_abs_pos_embedding"))?) } else { None };
        let proj_out = Linear::load(store, &gpu, &g("proj_out"))?;
        if patchify.n != D || adaln.linear.n != 9 * D || out_table.len != 2 * D {
            candle_core::bail!("not LTX 2.3's video stream (patchify {}, modulation {}, output table {})", patchify.n, adaln.linear.n, out_table.len);
        }
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = g(&format!("transformer_blocks.{i}"));
            blocks.push(Block {
                attn1: Attn::load(store, &gpu, &format!("{p}.attn1"))?,
                attn2: Attn::load(store, &gpu, &format!("{p}.attn2"))?,
                ff0: Linear::load(store, &gpu, &format!("{p}.ff.net.0.proj"))?,
                ff2: Linear::load(store, &gpu, &format!("{p}.ff.net.2"))?,
                table: vector(store, &gpu, &format!("{p}.scale_shift_table"))?,
                prompt_table: vector(store, &gpu, &format!("{p}.prompt_scale_shift_table"))?,
            });
            progress(i + 1);
        }
        Ok(Self { gpu, patchify, keyframe, adaln, prompt, out_table, proj_out, blocks, nag: None })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// `values` on this transformer's GPU (a rotary table, kept for every step).
    pub fn upload(&self, values: &[f32]) -> DeviceVec {
        let v = self.vec(values.len());
        self.gpu.upload(&v, values);
        v
    }

    /// One attention: `xq`'s `tq` rows' queries over `xkv`'s `tk` rows (each rotated by its table where given), the
    /// heads gated, out through the output projection into `y`. `s` the scratch.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, xkv: &DeviceVec, tk: usize, rope: Option<&DeviceVec>, passthrough: bool, s: &Scratch, y: &DeviceVec) {
        attend(r, a, xq, tq, xkv, tk, rope, passthrough, s, y)
    }

    /// The video velocity of `latent` (`tokens` rows of 128) at `sigma` over `context` (`lc` rows of `D`, the
    /// connector's), its tokens rotated by `table` ([`rope_table`] of [`video_positions`], [`Self::upload`]ed):
    /// `[tokens, 128]`. `clean`: the leading tokens of a starting image and the trailing ones of an end image, at
    /// timestep 0; `first_frame` the first latent frame's tokens (LTX 2.5 marks them). With `skip_self` that block's
    /// self-attention passed through (spatio-temporal guidance's perturbed pass).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, table: &DeviceVec, clean: (usize, usize), first_frame: usize, skip_self: Option<usize>) -> Result<Vec<f32>> {
        self.pass(latent, tokens, context, lc, sigma, table, clean, first_frame, skip_self, false)?.pop().ok_or_else(|| err("the velocity was not read"))
    }

    /// [`Self::forward`]'s pass: its reads, the velocity last (with `trace`, every block's output before it, one after
    /// another in one vector: a read is the vector as the recording leaves it).
    #[allow(clippy::too_many_arguments)]
    fn pass(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, table: &DeviceVec, clean: (usize, usize), first_frame: usize, skip_self: Option<usize>, trace: bool) -> Result<Vec<Vec<f32>>> {
        if latent.len() != tokens * self.patchify.k || context.len() != lc * D || table.len != tokens * D {
            candle_core::bail!("an LTX step's inputs: {} latent values for {tokens} tokens, {} context for {lc}, {} rotary", latent.len(), context.len(), table.len);
        }
        let (start, end) = clean;
        if start + end >= tokens {
            candle_core::bail!("conditioning must leave generated video tokens");
        }
        // the clean tokens' rows modulated by timestep 0's set, after sigma's: the blocks' nine rows, the output's two
        let two = start + end > 0;
        let rows = if two { CleanRows { before: start, from: tokens - end, offset: 9 * D } } else { CleanRows::NONE };
        let out_rows = if two { CleanRows { offset: 2 * D, ..rows } } else { CleanRows::NONE };
        let nag = self.nag.as_ref();
        if let Some(n) = nag {
            if n.context.len() != n.rows * D || n.rows == 0 {
                candle_core::bail!("a negative context of {} values for {} rows", n.context.len(), n.rows);
            }
        }
        let ln = nag.map_or(0, |n| n.rows);
        let s = Scratch::new(&self.gpu, tokens.max(lc).max(ln));
        let (lat, ctx) = (self.vec(latent.len()), self.vec(context.len()));
        self.gpu.upload(&lat, latent);
        self.gpu.upload(&ctx, context);
        // the negative context, its modulated rows and the plain attention's output kept beside the negative's
        let (nctx, ncm, plain) = (self.vec(ln * D), self.vec(ln * D), self.vec(if nag.is_some() { tokens * D } else { 1 }));
        if let Some(n) = nag {
            self.gpu.upload(&nctx, &n.context);
        }
        let t = self.vec(256);
        self.gpu.upload(&t, &sinusoids(sigma));
        let (x, h, cm, y, f, fg) = (self.vec(tokens * D), self.vec(tokens * D), self.vec(lc * D), self.vec(tokens * D), self.vec(tokens * 4 * D), self.vec(tokens * 4 * D));
        let (emb, modulation, prompt) = (self.vec(D), self.vec(9 * D), self.vec(2 * D));
        let trail = self.vec(if trace { self.blocks.len() * tokens * D } else { 1 });
        let sets = if two { 2 } else { 1 };
        let (mm, pm, mo) = (self.vec(sets * 9 * D), self.vec(2 * D), self.vec(sets * 2 * D));
        let (s1, s2) = (self.vec(D), self.vec(D));
        // timestep 0's, for the clean tokens: its sinusoids, embedding and modulation, and a set's scratch
        let (t0, emb0, modulation0, set, out_set) = (self.vec(256), self.vec(D), self.vec(9 * D), self.vec(9 * D), self.vec(2 * D));
        if two {
            self.gpu.upload(&t0, &sinusoids(0.));
        }
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        // the timestep's modulation and embedding, the prompt's modulation
        self.adaln.record(r, &t, &s1, &s2, &emb, &modulation);
        if two {
            self.adaln.record(r, &t0, &s1, &s2, &emb0, &modulation0);
        }
        match &self.prompt {
            Some(pe) => {
                let pe_emb = self.vec(D);
                pe.record(r, &t, &s1, &s2, &pe_emb, &prompt);
            }
            None => r.copy(&self.vec(2 * D), 0, &prompt, 0, 2 * D),
        }
        self.patchify.forward(r, &lat, &x, tokens);
        if let Some(marker) = self.keyframe.as_ref().filter(|_| first_frame > 0) {
            r.add_bias_rows(&x, marker, first_frame.min(tokens), D);
        }
        for (i, b) in self.blocks.iter().enumerate() {
            // the block's modulation: the step's and its own table (shift, scale, gate: self-attention 0-2, the
            // feed-forward 3-5, text attention 6-8), the prompt's two rows the same
            if two {
                for (at, step) in [(0, &modulation), (9 * D, &modulation0)] {
                    r.copy(step, 0, &set, 0, 9 * D);
                    r.add(&set, &b.table);
                    r.copy(&set, 0, &mm, at, 9 * D);
                }
            } else {
                r.copy(&modulation, 0, &mm, 0, 9 * D);
                r.add(&mm, &b.table);
            }
            r.copy(&prompt, 0, &pm, 0, 2 * D);
            r.add(&pm, &b.prompt_table);
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, D, Some(0), RowNorm::Rms, EPS, rows);
            self.attend(r, &b.attn1, &h, tokens, &h, tokens, Some(table), skip_self == Some(i), &s, &y);
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 2 * D, false, rows);
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, 7 * D, Some(6 * D), RowNorm::Rms, EPS, rows);
            r.norm_mod_rows(&ctx, &cm, lc, D, &pm, D, Some(0), RowNorm::None, EPS);
            match nag {
                Some(n) => {
                    // the negative context modulated as the prompt's, the two attentions' outputs mixed
                    r.norm_mod_rows(&nctx, &ncm, ln, D, &pm, D, Some(0), RowNorm::None, EPS);
                    attend_guided(r, &b.attn2, &h, tokens, (&cm, lc), (&ncm, ln), n, &s, &plain, &y);
                }
                None => self.attend(r, &b.attn2, &h, tokens, &cm, lc, None, false, &s, &y),
            }
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 8 * D, false, rows);
            r.norm_mod_rows_clean(&x, &h, tokens, D, &mm, 4 * D, Some(3 * D), RowNorm::Rms, EPS, rows);
            b.ff0.forward(r, &h, &f, tokens);
            r.gelu(&f, &fg, tokens * b.ff0.n);
            b.ff2.forward(r, &fg, &y, tokens);
            r.add_gated_rows_clean(&x, &y, tokens, D, &mm, 5 * D, false, rows);
            if trace {
                r.copy(&x, 0, &trail, i * tokens * D, tokens * D);
            }
        }
        // the output: a layer norm, its table's (shift, scale) plus the timestep's embedding, the projection
        for (at, e) in [(0, &emb), (2 * D, &emb0)].into_iter().take(sets) {
            r.copy(&self.out_table, 0, &out_set, 0, 2 * D);
            r.add_bias_rows(&out_set, e, 2, D);
            r.copy(&out_set, 0, &mo, at, 2 * D);
        }
        r.norm_mod_rows_clean(&x, &h, tokens, D, &mo, D, Some(0), RowNorm::Layer, EPS, out_rows);
        let vel = self.vec(tokens * self.proj_out.n);
        self.proj_out.forward(r, &h, &vel, tokens);
        if trace {
            r.read(&trail);
        }
        r.read(&vel);
        Ok(rec.finish())
    }
}

/// One attention: `xq`'s `tq` rows' queries over `xkv`'s `tk` rows (each rotated by its table where given), the heads
/// gated, out through the output projection into `y`. With `passthrough` the attention is its value projection (the
/// reference's perturbation for spatio-temporal guidance), its gate and output still applied. `s` the scratch.
#[allow(clippy::too_many_arguments)]
fn attend(r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, xkv: &DeviceVec, tk: usize, rope: Option<&DeviceVec>, passthrough: bool, s: &Scratch, y: &DeviceVec) {
    if passthrough {
        a.v.forward(r, xkv, &s.att, tk);
        a.gate.forward(r, xq, &s.logits, tq);
        r.head_gate_rows(&s.att, &s.logits, tq, HEADS, HD);
        a.out.forward(r, &s.att, y, tq);
        return;
    }
    // (the norms take a row's width from their vectors' lengths: views of the scratch's first rows)
    let first = |v: &DeviceVec, rows: usize| DeviceVec { len: rows * D, inner: v.inner.clone() };
    a.q.forward(r, xq, &s.q, tq);
    r.rmsnorm_rows(&first(&s.q, tq), &a.qn, &first(&s.qn, tq), tq, EPS);
    a.k.forward(r, xkv, &s.k, tk);
    r.rmsnorm_rows(&first(&s.k, tk), &a.kn, &first(&s.kn, tk), tk, EPS);
    a.v.forward(r, xkv, &s.v, tk);
    if let Some(t) = rope {
        r.rope_split_rows(&s.qn, tq, HEADS, HD, t);
        r.rope_split_rows(&s.kn, tk, HEADS, HD, t);
    }
    r.store_rows(&s.kn, &s.kv, tk, D, 0, 2 * D, 0);
    r.store_rows(&s.v, &s.kv, tk, D, 0, 2 * D, D);
    r.attention_rows_full(&s.qn, &s.kv, &s.att, tq, HEADS, HEADS, HD, tk, 1.0 / (HD as f32).sqrt());
    a.gate.forward(r, xq, &s.logits, tq);
    r.head_gate_rows(&s.att, &s.logits, tq, HEADS, HD);
    a.out.forward(r, &s.att, y, tq);
}

/// A text cross-attention guided away from a negative context ([`Nag`]; the reference's `nag_attn`): `xq`'s `tq`
/// rows' queries over the prompt's context and over the negative one (`(rows, count)` each), the two outputs mixed in
/// `plain`, then the heads' gate and the output projection into `y`.
#[allow(clippy::too_many_arguments)]
fn attend_guided(r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, positive: (&DeviceVec, usize), negative: (&DeviceVec, usize), nag: &Nag, s: &Scratch, plain: &DeviceVec, y: &DeviceVec) {
    let first = |v: &DeviceVec, rows: usize| DeviceVec { len: rows * D, inner: v.inner.clone() };
    a.q.forward(r, xq, &s.q, tq);
    r.rmsnorm_rows(&first(&s.q, tq), &a.qn, &first(&s.qn, tq), tq, EPS);
    for (which, (xkv, tk)) in [positive, negative].into_iter().enumerate() {
        a.k.forward(r, xkv, &s.k, tk);
        r.rmsnorm_rows(&first(&s.k, tk), &a.kn, &first(&s.kn, tk), tk, EPS);
        a.v.forward(r, xkv, &s.v, tk);
        r.store_rows(&s.kn, &s.kv, tk, D, 0, 2 * D, 0);
        r.store_rows(&s.v, &s.kv, tk, D, 0, 2 * D, D);
        r.attention_rows_full(&s.qn, &s.kv, &s.att, tq, HEADS, HEADS, HD, tk, 1.0 / (HD as f32).sqrt());
        if which == 0 {
            r.copy(&s.att, 0, plain, 0, tq * D);
        }
    }
    r.nag_mix(plain, &s.att, tq, D, nag.scale, nag.tau, nag.alpha);
    a.gate.forward(r, xq, &s.logits, tq);
    r.head_gate_rows(plain, &s.logits, tq, HEADS, HD);
    a.out.forward(r, plain, y, tq);
}

/// An attention's vectors, for `rows` rows at the most.
struct Scratch {
    q: DeviceVec,
    qn: DeviceVec,
    k: DeviceVec,
    kn: DeviceVec,
    v: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    logits: DeviceVec,
}

impl Scratch {
    fn new(gpu: &ggml_rs_wgpu::WgpuBackend, rows: usize) -> Self {
        let v = |len: usize| gpu.vec(len.max(1));
        Self {
            q: v(rows * D),
            qn: v(rows * D),
            k: v(rows * D),
            kn: v(rows * D),
            v: v(rows * D),
            kv: v(rows * 2 * D),
            att: v(gpu.attention_rows_full_out_len(rows, HEADS, HD, rows)),
            logits: v(rows * HEADS),
        }
    }
}

/// One of a text connector's blocks: its gated self-attention and its feed-forward.
pub struct ConnectorBlock {
    attn1: Attn,
    ff0: Linear,
    ff2: Linear,
}

impl ConnectorBlock {
    pub fn load(store: &mut Store, gpu: &ggml_rs_wgpu::WgpuBackend, p: &str) -> Result<Self> {
        Ok(Self { attn1: Attn::load(store, gpu, &format!("{p}.attn1"))?, ff0: Linear::load(store, gpu, &format!("{p}.ff.net.0.proj"))?, ff2: Linear::load(store, gpu, &format!("{p}.ff.net.2"))? })
    }
}

/// The text connector over `x` (`rows` of `D`, changed in place) as [`crate::ltx::transformer::connector`] runs it:
/// each block RMS-normed (no weights) into its self-attention (rotated by `table`) and its feed-forward, each added
/// back; the result over its RMS (a new vector).
pub fn connector(gpu: &ggml_rs_wgpu::WgpuBackend, r: &mut dyn ChainRecorder, blocks: &[ConnectorBlock], x: &DeviceVec, rows: usize, table: &DeviceVec) -> DeviceVec {
    let s = Scratch::new(gpu, rows);
    let ones = gpu.vec(D);
    gpu.upload(&ones, &vec![1.0; D]);
    let v = |len: usize| gpu.vec(len.max(1));
    let (h, y, f, fg, out) = (v(rows * D), v(rows * D), v(rows * 4 * D), v(rows * 4 * D), v(rows * D));
    for b in blocks {
        r.rmsnorm_rows(x, &ones, &h, rows, EPS);
        attend(r, &b.attn1, &h, rows, &h, rows, Some(table), false, &s, &y);
        r.add(x, &y);
        r.rmsnorm_rows(x, &ones, &h, rows, EPS);
        b.ff0.forward(r, &h, &f, rows);
        r.gelu(&f, &fg, rows * b.ff0.n);
        b.ff2.forward(r, &fg, &y, rows);
        r.add(x, &y);
    }
    r.rmsnorm_rows(x, &ones, &out, rows, EPS);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// How long a step of the WebGPU video stream takes (`--ignored --nocapture`, `OAIY_LTX_NVFP4`): a 768x512 clip of
    /// 121 frames (16 latent frames of 16 by 24: 6,144 tokens) over the connector's 1,024 context rows.
    #[test]
    #[ignore = "a timing; needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn measure_a_video_step() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let (frames, h, w, fps, lc) = (16usize, 16usize, 24usize, 24usize, 1024usize);
        let tokens = frames * h * w;
        let latent: Vec<f32> = (0..tokens * 128).map(|i| ((i * 7919 % 2001) as f32 / 1000.0 - 1.0) * 1.5).collect();
        let context: Vec<f32> = (0..lc * D).map(|i| (i * 104729 % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
        let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
        for i in 0..3 {
            let t = std::time::Instant::now();
            let v = gpu.forward(&latent, tokens, &context, lc, 0.7 - 0.1 * i as f64, &table, (0, 0), h * w, None)?;
            eprintln!("step {i}: {:.2} s ({} values)", t.elapsed().as_secs_f64(), v.len());
        }
        Ok(())
    }

    /// `a`'s cosine with `b`, and its distance from `b` over `b`'s size.
    fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let diff = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (na * nb), diff / nb)
    }

    /// `t` (BF16) with every value one BF16 step off, alternately up and down: inputs as near as BF16 holds them.
    fn nudged(t: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let v: Vec<half::bf16> = t.flatten_all()?.to_vec1::<half::bf16>()?.iter().enumerate().map(|(i, x)| half::bf16::from_bits(if i % 2 == 0 { x.to_bits().wrapping_add(1) } else { x.to_bits().wrapping_sub(1) })).collect();
        candle_core::Tensor::from_vec(v, t.dims(), t.device())
    }

    /// The WebGPU video stream gives the Candle one's velocity (on CUDA, BF16) for the checkpoint `OAIY_LTX_NVFP4` names
    /// (Lightricks' NVFP4 release, or a BF16 one's Q8_0): a video of 2 latent frames by 4 by 6 and a context of 40 rows
    /// (random, as the connector's are near unit RMS) at two sigmas and spatio-temporal guidance's pass. As near as the
    /// reference is to itself, its inputs one BF16 step off, or to a tenth: the distilled release's last blocks take
    /// random inputs' BF16 noise to a fifth of the velocity ([`trace_the_video_blocks_against_candles`]: its blocks
    /// agree to 0.9999 through block 36).
    #[test]
    #[ignore = "needs an LTX 2.3 checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter, CUDA (the cuda feature) and some 60 GB of RAM"]
    fn the_webgpu_video_stream_is_the_candle_one() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let path = std::path::Path::new(&path);
        let (frames, h, w, fps, lc) = (2usize, 4usize, 6usize, 24usize, 40usize);
        let tokens = frames * h * w;
        let mut seed = 0x51ed_270bu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let latent: Vec<f32> = (0..tokens * 128).map(|_| (next() * 1.7) as f32).collect();
        let context: Vec<f32> = (0..lc * D).map(|_| (next() * 1.7) as f32).collect();
        // (a negative prompt's context, of another length)
        let negative_rows = 24usize;
        let negative: Vec<f32> = (0..negative_rows * D).map(|_| (next() * 1.7) as f32).collect();
        let t = std::time::Instant::now();
        let mut store = Store::open(path, 0)?;
        let mut gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
        eprintln!("WebGPU video stream loaded in {:.1} s", t.elapsed().as_secs_f64());
        let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
        let got: Vec<Vec<f32>> = [0.8, 0.25].iter().map(|&sigma| gpu.forward(&latent, tokens, &context, lc, sigma, &table, (0, 0), h * w, None)).collect::<Result<_>>()?;
        // and spatio-temporal guidance's pass, block 28's self-attention passed through
        let stg = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, Some(28))?;
        // and a starting image's and an end image's clean tokens (the first frame's, an appended frame's)
        let ends = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, true), &[20., 2048., 2048.], D, HEADS));
        let appended: Vec<f32> = latent.iter().chain(&latent[..h * w * 128]).map(|v| v * 0.9).collect();
        let conditioned = gpu.forward(&appended, tokens + h * w, &context, lc, 0.8, &ends, (h * w, h * w), h * w, None)?;
        // and a negative prompt without CFG: every block's text attention guided away from it (the reference's
        // defaults for video: scale 11, tau 2.5, alpha 0.25)
        gpu.nag = Some(Nag { context: negative.clone(), rows: negative_rows, scale: 11.0, tau: 2.5, alpha: 0.25 });
        let guided = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, None)?;
        drop(gpu);
        // (Candle's LTX is BF16 throughout: its CPU has no BF16 matmul, so CUDA's device OAIY_LTX_CUDA_DEVICE, 0 else)
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        // (a budget under the blocks' INT8 size: they stream as they are stored; a BF16 checkpoint's 28 GB over a
        // budget its INT8 rows fit would be those rows, not the reference)
        let mut cpu = crate::ltx::transformer::Transformer::new(Store::open(path, 0)?, &dev, 8 << 30, false, false)?;
        assert!(!cpu.int8, "the reference's blocks as INT8");
        let rope = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, false, &dev)?;
        let lt = candle_core::Tensor::from_vec(latent, (1, tokens, 128), &dev)?.to_dtype(DType::BF16)?;
        let ct = candle_core::Tensor::from_vec(context, (1, lc, D), &dev)?.to_dtype(DType::BF16)?;
        let (ln, cn) = (nudged(&lt)?, nudged(&ct)?);
        let rope_ends = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, true, &dev)?;
        let la = candle_core::Tensor::from_vec(appended, (1, tokens + h * w, 128), &dev)?.to_dtype(DType::BF16)?;
        let lan = nudged(&la)?;
        let nt = candle_core::Tensor::from_vec(negative, (1, negative_rows, D), &dev)?.to_dtype(DType::BF16)?;
        let passes_plain = got[0].clone();
        let mut want_plain: Option<Vec<f32>> = None;
        let passes = [(0.8, None, false, false, &got[0]), (0.25, None, false, false, &got[1]), (0.8, Some(28), false, false, &stg), (0.8, None, true, false, &conditioned), (0.8, None, false, true, &guided)];
        for (sigma, skip, ends, nag, got) in passes {
            let t = std::time::Instant::now();
            cpu.skip_video_self_attn = skip;
            cpu.nag = nag.then(|| crate::ltx::transformer::Nag { context: nt.clone(), scale: 11., tau: 2.5, alpha: 0.25 });
            let (rope, clean) = if ends { (&rope_ends, h * w) } else { (&rope, 0) };
            let velocity = |cpu: &mut crate::ltx::transformer::Transformer, l: &candle_core::Tensor, c: &candle_core::Tensor| -> Result<Vec<f32>> {
                cpu.forward(l, c, sigma, rope, clean, clean, None, |_| {})?.0.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
            };
            let (l, n) = if ends { (&la, &lan) } else { (&lt, &ln) };
            let want = velocity(&mut cpu, l, &ct)?;
            let seconds = t.elapsed().as_secs_f64();
            let nudged = velocity(&mut cpu, n, &cn)?;
            let (_, spread) = compare(&nudged, &want);
            let (cos, err) = compare(got, &want);
            if ends {
                // the clean tokens' rows (the first frame's, the appended one's) and the noisy ones' apart
                let row = 128;
                let (lo, hi) = (clean * row, want.len() - clean * row);
                let part = |v: &[f32], noisy: bool| -> Vec<f32> { if noisy { v[lo..hi].to_vec() } else { v[..lo].iter().chain(&v[hi..]).copied().collect() } };
                for noisy in [false, true] {
                    let (c, e) = compare(&part(got, noisy), &part(&want, noisy));
                    let (_, sp) = compare(&part(&nudged, noisy), &part(&want, noisy));
                    eprintln!("  the {} rows: cosine {c:.6}, relative error {e:.2e} (the reference's own spread {sp:.2e})", if noisy { "noisy" } else { "clean" });
                }
            }
            let what = match (skip, ends, nag) {
                (Some(b), _, _) => format!("sigma {sigma}, block {b}'s self-attention passed through"),
                (None, true, _) => format!("sigma {sigma}, a starting and an end image's tokens clean"),
                (None, false, true) => format!("sigma {sigma}, a negative prompt by NAG"),
                (None, false, false) => format!("sigma {sigma}"),
            };
            if !nag && !ends && skip.is_none() && want_plain.is_none() {
                want_plain = Some(want.clone());
            }
            if nag {
                // What the guidance changes, here and in the reference: the velocity's difference from the pass with no
                // negative prompt (a pair's passes through the same arithmetic, so its difference is the guidance's).
                let (cos, moved) = compare(got, &passes_plain);
                eprintln!("  NAG against no negative prompt: cosine {cos:.6}, relative difference {moved:.2e}");
                assert!(moved > 1e-3, "the negative prompt changes the velocity");
                let plain = want_plain.as_ref().expect("the pass with no negative prompt first");
                let ours: Vec<f32> = got.iter().zip(&passes_plain).map(|(a, b)| a - b).collect();
                let theirs: Vec<f32> = want.iter().zip(plain).map(|(a, b)| a - b).collect();
                let (dc, de) = compare(&ours, &theirs);
                // (the change is some 4% of the velocity, as the reference's own spread is: the two changes agree as
                // far as that lets them; the blocks' trace with OAIY_LTX_TRACE_NAG holds each block to four places)
                eprintln!("  the guidance's change against the reference's: cosine {dc:.6}, relative error {de:.2e}");
                assert!(dc > 0.6, "the guidance's change is the reference's: cosine {dc}");
            }
            eprintln!("{what}: cosine {cos:.6}, relative error {err:.2e} (the reference's own spread {spread:.2e}; its step {seconds:.1} s)");
            // (its BF16 rounding at every op beside one step off at the inputs: within three times that)
            assert!(err <= (3.0 * spread).max(0.1), "{what}: relative error {err} where the reference's own spread is {spread}");
        }
        Ok(())
    }

    /// Each block's output of the WebGPU video stream against Candle's (on CUDA, BF16; `OAIY_LTX_NVFP4`), the inputs
    /// as [`the_webgpu_video_stream_is_the_candle_one`]'s at sigma 0.8: where the two part.
    #[test]
    #[ignore = "a trace; needs an LTX 2.3 checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter and CUDA (the cuda feature)"]
    fn trace_the_video_blocks_against_candles() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let path = std::path::Path::new(&path);
        let (frames, h, w, fps, lc) = (2usize, 4usize, 6usize, 24usize, 40usize);
        let tokens = frames * h * w;
        let mut seed = 0x51ed_270bu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let latent: Vec<f32> = (0..tokens * 128).map(|_| (next() * 1.7) as f32).collect();
        let context: Vec<f32> = (0..lc * D).map(|_| (next() * 1.7) as f32).collect();
        // (OAIY_LTX_TRACE_NAG: a negative prompt by NAG in both, its context another 24 rows: every block's text
        // attention guided, and the first blocks, which agree to four places, held to the reference's)
        let negative: Option<Vec<f32>> = std::env::var_os("OAIY_LTX_TRACE_NAG").map(|_| (0..24 * D).map(|_| (next() * 1.7) as f32).collect());
        let mut store = Store::open(path, 0)?;
        let mut gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
        gpu.nag = negative.as_ref().map(|n| Nag { context: n.clone(), rows: 24, scale: 11.0, tau: 2.5, alpha: 0.25 });
        let table = gpu.upload(&rope_table(&video_positions(frames, h, w, fps, false), &[20., 2048., 2048.], D, HEADS));
        let got = gpu.pass(&latent, tokens, &context, lc, 0.8, &table, (0, 0), h * w, None, true)?;
        let trail = &got[0];
        drop(gpu);
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        let mut cpu = crate::ltx::transformer::Transformer::new(Store::open(path, 0)?, &dev, 8 << 30, false, false)?;
        cpu.hiddens = Some(Vec::new());
        if let Some(n) = &negative {
            let context = candle_core::Tensor::from_vec(n.clone(), (1, 24, D), &dev)?.to_dtype(DType::BF16)?;
            cpu.nag = Some(crate::ltx::transformer::Nag { context, scale: 11., tau: 2.5, alpha: 0.25 });
        }
        let rope = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, false, &dev)?;
        let lt = candle_core::Tensor::from_vec(latent, (1, tokens, 128), &dev)?.to_dtype(DType::BF16)?;
        let ct = candle_core::Tensor::from_vec(context, (1, lc, D), &dev)?.to_dtype(DType::BF16)?;
        let (v, _) = cpu.forward(&lt, &ct, 0.8, &rope, 0, 0, None, |_| {})?;
        let hiddens = cpu.hiddens.take().unwrap_or_default();
        for (i, hidden) in hiddens.iter().enumerate() {
            let want = hidden.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let (cos, rel) = compare(&trail[i * tokens * D..(i + 1) * tokens * D], &want);
            let rms = (want.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
            let peak = want.iter().fold(0f32, |m, x| m.max(x.abs()));
            eprintln!("block {i:2}: cosine {cos:.6}, relative error {rel:.2e} (the reference's RMS {rms:.3e}, its largest {peak:.3e})");
            if negative.is_some() && i < 8 {
                assert!(cos > 0.999, "block {i} with a negative prompt by NAG: cosine {cos}");
            }
        }
        let want = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (cos, rel) = compare(got.last().unwrap(), &want);
        eprintln!("velocity: cosine {cos:.6}, relative error {rel:.2e}");
        Ok(())
    }

    /// A BF16 checkpoint's layers (`OAIY_LTX_NVFP4` naming a BF16 one: LTX 2.3's distilled release) as Q8_0 on the
    /// tensor cores give their quantization's product (in f64, the kernel's own error), 300 rows; and how far that is
    /// from the BF16 weights' (the quantization's).
    #[test]
    #[ignore = "needs a BF16 LTX 2.3 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter with tensor cores"]
    fn a_q8_layer_on_the_gpu_is_its_quantizations() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = ggml_rs_wgpu::WgpuBackend::new(None).map_err(err)?;
        for name in ["model.diffusion_model.transformer_blocks.10.attn1.to_q", "model.diffusion_model.transformer_blocks.10.attn1.to_gate_logits", "model.diffusion_model.transformer_blocks.10.ff.net.0.proj", "model.diffusion_model.transformer_blocks.10.ff.net.2"] {
            let l = Linear::load(&mut store, &gpu, name)?;
            if !matches!(l.weight, Weight::Q8(_)) {
                eprintln!("{name}: not Q8_0 (an NVFP4 checkpoint's, or no tensor cores)");
                return Ok(());
            }
            let rows = 300;
            let mut seed = 0x9e37_79b9u64;
            let x: Vec<f32> = (0..rows * l.k)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed % 2001) as f32 / 1000.0 - 1.0
                })
                .collect();
            let (xd, yd) = (gpu.vec(x.len()), gpu.vec(rows * l.n));
            gpu.upload(&xd, &x);
            let mut rec = gpu.begin();
            l.forward(rec.as_mut(), &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let w = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let b = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
            let mut wq = vec![0f32; w.len()];
            ggml_quants::q8_0::dequantize(&q8_0(&w), &mut wq);
            let product = |w: &[f32]| -> Vec<f64> {
                let mut y = vec![0f64; rows * l.n];
                for r in 0..rows {
                    let xr = &x[r * l.k..(r + 1) * l.k];
                    for o in 0..l.n {
                        let wr = &w[o * l.k..(o + 1) * l.k];
                        y[r * l.n + o] = b[o] as f64 + xr.iter().zip(wr).map(|(a, c)| *a as f64 * *c as f64).sum::<f64>();
                    }
                }
                y
            };
            let (exact, bf16) = (product(&wq), product(&w));
            let rms = (bf16.iter().map(|v| v * v).sum::<f64>() / bf16.len() as f64).sqrt();
            let rel = |a: &[f64]| (got.iter().zip(a).map(|(g, e)| (*g as f64 - e).powi(2)).sum::<f64>() / a.len() as f64).sqrt() / rms;
            let quant = (exact.iter().zip(&bf16).map(|(e, f)| (e - f).powi(2)).sum::<f64>() / bf16.len() as f64).sqrt() / rms;
            eprintln!("{name} [{}, {}]: the kernel's error {:.2e} of the RMS, the GPU's from BF16's {:.2e} (Q8_0's own {quant:.2e})", l.n, l.k, rel(&exact), rel(&bf16));
            assert!(rel(&exact) < 2e-3, "{name}: the kernel's error {} of the RMS", rel(&exact));
        }
        Ok(())
    }

    /// A real NVFP4 layer of Lightricks' release (`OAIY_LTX_NVFP4`) on the tensor cores gives what its store's own
    /// decode does on the CPU (in f32), 300 rows.
    #[test]
    #[ignore = "needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4) and a WebGPU adapter"]
    fn an_nvfp4_layer_on_the_gpu_is_the_stores() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_LTX_NVFP4") else { return Ok(()) };
        let mut store = Store::open(std::path::Path::new(&path), 0)?;
        let gpu = ggml_rs_wgpu::WgpuBackend::new(None).map_err(err)?;
        // (not a gate's logits: 32 rows, their scales a tile of 128's, which the store's own decode does not take)
        for name in ["model.diffusion_model.transformer_blocks.10.attn1.to_q", "model.diffusion_model.transformer_blocks.20.ff.net.2", "model.diffusion_model.transformer_blocks.30.audio_ff.net.0.proj"] {
            let l = Linear::load(&mut store, &gpu, name)?;
            assert!(matches!(l.weight, Weight::Nvfp4 { .. }), "{name} NVFP4");
            let rows = 300;
            let mut seed = 0x9e37_79b9u64;
            let x: Vec<f32> = (0..rows * l.k)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    half::f16::from_f32((seed % 2001) as f32 / 1000.0 - 1.0).to_f32()
                })
                .collect();
            let (xd, yd) = (gpu.vec(x.len()), gpu.vec(rows * l.n));
            gpu.upload(&xd, &x);
            let mut rec = gpu.begin();
            l.forward(rec.as_mut(), &xd, &yd, rows);
            rec.read(&yd);
            let got = rec.finish().pop().unwrap();
            let w = store.tensor_f32(&format!("{name}.weight"), &Device::Cpu)?;
            let b = store.tensor_f32(&format!("{name}.bias"), &Device::Cpu)?;
            let xt = candle_core::Tensor::from_vec(x, (rows, l.k), &Device::Cpu)?;
            let want = xt.matmul(&w.t()?)?.broadcast_add(&b)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let rms = (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
            let worst = got.iter().zip(&want).map(|(a, e)| (*a as f64 - *e as f64).abs()).fold(0.0, f64::max);
            eprintln!("{name} [{}, {}]: the worst error {worst:.3e} of an RMS {rms:.3e}", l.n, l.k);
            assert!(worst <= 1e-3 * rms.max(1e-6) * 10.0, "{name}: worst {worst} of an RMS {rms}");
        }
        Ok(())
    }
}
