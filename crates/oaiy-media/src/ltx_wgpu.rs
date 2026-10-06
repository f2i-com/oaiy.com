//! LTX 2.3's transformer on WebGPU, as [`crate::ltx::transformer`] computes its video stream (text-to-video: no
//! audio, no conditioning frames, no NAG yet): its weights on the GPU as they are stored where the chain has a kernel
//! for them (NVFP4 packed, the tensor cores decoding it as they multiply: Lightricks' `-nvfp4` release's 44 blocks of
//! 48), else f16 (BF16 rounded); its activations f32.
use crate::ltx::store::{untile_scales, Store};
use candle_core::{Device, Result};
use dsv41::safetensors::Dtype;
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};

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

/// A linear layer on the GPU: its weight NVFP4 (its words and its scale's vector) or f16, and its bias.
pub enum Weight {
    Nvfp4 { w: DeviceVec, scale: DeviceVec },
    F16(DeviceVec),
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
                None => {
                    let t = store.tensor_f32(&key, &Device::Cpu)?;
                    let v = gpu.vec_f16_rounded(&t.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{key}: past f16's range")))?;
                    (Weight::F16(v), n, k)
                }
            }
        } else {
            let t = store.tensor_f32(&key, &Device::Cpu)?;
            let (n, k) = t.dims2()?;
            let v = gpu.vec_f16_rounded(&t.flatten_all()?.to_vec1::<f32>()?).ok_or_else(|| err(format!("{key}: past f16's range")))?;
            (Weight::F16(v), n, k)
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

/// A video's tokens' coordinates (time in seconds, then its pixels' row and column), as the reference places them.
pub fn video_positions(frames: usize, height: usize, width: usize, fps: usize) -> Vec<Vec<f32>> {
    let mut positions = Vec::with_capacity(frames * height * width);
    for t in 0..frames {
        for y in 0..height {
            for x in 0..width {
                let start = (t * 8).saturating_sub(7);
                let end = (t + 1) * 8 - 7;
                positions.push(vec![(start + end) as f32 / (2 * fps) as f32, (y as f32 + 0.5) * 32., (x as f32 + 0.5) * 32.]);
            }
        }
    }
    positions
}

pub struct WgpuLtx {
    gpu: ggml_rs_wgpu::WgpuBackend,
    patchify: Linear,
    adaln: TimeEmbed,
    prompt: Option<TimeEmbed>,
    out_table: DeviceVec,
    proj_out: Linear,
    blocks: Vec<Block>,
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
        Ok(Self { gpu, patchify, adaln, prompt, out_table, proj_out, blocks })
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    /// One attention: `xq`'s `tq` rows' queries over `xkv`'s `tk` rows (each rotated by its table where given), the
    /// heads gated, out through the output projection into `y`. `s` the scratch.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, r: &mut dyn ChainRecorder, a: &Attn, xq: &DeviceVec, tq: usize, xkv: &DeviceVec, tk: usize, rope: Option<&DeviceVec>, passthrough: bool, s: &Scratch, y: &DeviceVec) {
        attend(r, a, xq, tq, xkv, tk, rope, passthrough, s, y)
    }

    /// The video velocity of `latent` (`tokens` rows of 128) at `sigma` over `context` (`lc` rows of `D`, the
    /// connector's), its tokens rotated by `rope` ([`rope_table`] of [`video_positions`]): `[tokens, 128]`. With
    /// `skip_self` that block's self-attention passed through (spatio-temporal guidance's perturbed pass).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(&self, latent: &[f32], tokens: usize, context: &[f32], lc: usize, sigma: f64, rope: &[f32], skip_self: Option<usize>) -> Result<Vec<f32>> {
        if latent.len() != tokens * self.patchify.k || context.len() != lc * D || rope.len() != tokens * D {
            candle_core::bail!("an LTX step's inputs: {} latent values for {tokens} tokens, {} context for {lc}, {} rotary", latent.len(), context.len(), rope.len());
        }
        let s = Scratch::new(&self.gpu, tokens.max(lc));
        let (lat, ctx, table) = (self.vec(latent.len()), self.vec(context.len()), self.vec(rope.len()));
        self.gpu.upload(&lat, latent);
        self.gpu.upload(&ctx, context);
        self.gpu.upload(&table, rope);
        let t = self.vec(256);
        self.gpu.upload(&t, &sinusoids(sigma));
        let (x, h, cm, y, f, fg) = (self.vec(tokens * D), self.vec(tokens * D), self.vec(lc * D), self.vec(tokens * D), self.vec(tokens * 4 * D), self.vec(tokens * 4 * D));
        let (emb, modulation, prompt) = (self.vec(D), self.vec(9 * D), self.vec(2 * D));
        let (mm, pm, mo) = (self.vec(9 * D), self.vec(2 * D), self.vec(2 * D));
        let (s1, s2) = (self.vec(D), self.vec(D));
        let mut rec = self.gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        // the timestep's modulation and embedding, the prompt's modulation
        self.adaln.record(r, &t, &s1, &s2, &emb, &modulation);
        match &self.prompt {
            Some(pe) => {
                let pe_emb = self.vec(D);
                pe.record(r, &t, &s1, &s2, &pe_emb, &prompt);
            }
            None => r.copy(&self.vec(2 * D), 0, &prompt, 0, 2 * D),
        }
        self.patchify.forward(r, &lat, &x, tokens);
        for (i, b) in self.blocks.iter().enumerate() {
            // the block's modulation: the step's and its own table (shift, scale, gate: self-attention 0-2, the
            // feed-forward 3-5, text attention 6-8), the prompt's two rows the same
            r.copy(&modulation, 0, &mm, 0, 9 * D);
            r.add(&mm, &b.table);
            r.copy(&prompt, 0, &pm, 0, 2 * D);
            r.add(&pm, &b.prompt_table);
            r.norm_mod_rows(&x, &h, tokens, D, &mm, D, Some(0), RowNorm::Rms, EPS);
            self.attend(r, &b.attn1, &h, tokens, &h, tokens, Some(&table), skip_self == Some(i), &s, &y);
            r.add_gated_rows(&x, &y, tokens, D, &mm, 2 * D, false);
            r.norm_mod_rows(&x, &h, tokens, D, &mm, 7 * D, Some(6 * D), RowNorm::Rms, EPS);
            r.norm_mod_rows(&ctx, &cm, lc, D, &pm, D, Some(0), RowNorm::None, EPS);
            self.attend(r, &b.attn2, &h, tokens, &cm, lc, None, false, &s, &y);
            r.add_gated_rows(&x, &y, tokens, D, &mm, 8 * D, false);
            r.norm_mod_rows(&x, &h, tokens, D, &mm, 4 * D, Some(3 * D), RowNorm::Rms, EPS);
            b.ff0.forward(r, &h, &f, tokens);
            r.gelu(&f, &fg, tokens * b.ff0.n);
            b.ff2.forward(r, &fg, &y, tokens);
            r.add_gated_rows(&x, &y, tokens, D, &mm, 5 * D, false);
        }
        // the output: a layer norm, its table's (shift, scale) plus the timestep's embedding, the projection
        r.copy(&self.out_table, 0, &mo, 0, 2 * D);
        r.add_bias_rows(&mo, &emb, 2, D);
        r.norm_mod_rows(&x, &h, tokens, D, &mo, D, Some(0), RowNorm::Layer, EPS);
        let vel = self.vec(tokens * self.proj_out.n);
        self.proj_out.forward(r, &h, &vel, tokens);
        r.read(&vel);
        rec.finish().pop().ok_or_else(|| err("the velocity was not read"))
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
        let table = rope_table(&video_positions(frames, h, w, fps), &[20., 2048., 2048.], D, HEADS);
        for i in 0..3 {
            let t = std::time::Instant::now();
            let v = gpu.forward(&latent, tokens, &context, lc, 0.7 - 0.1 * i as f64, &table, None)?;
            eprintln!("step {i}: {:.2} s ({} values)", t.elapsed().as_secs_f64(), v.len());
        }
        Ok(())
    }

    /// The WebGPU video stream gives the Candle one's velocity (on CUDA, BF16) on Lightricks' release (`OAIY_LTX_NVFP4`): a video
    /// of 2 latent frames by 4 by 6 and a context of 40 rows (random, as the connector's are near unit RMS) at two sigmas.
    #[test]
    #[ignore = "needs LTX 2.3's NVFP4 checkpoint (OAIY_LTX_NVFP4), a WebGPU adapter and some 60 GB of RAM"]
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
        let t = std::time::Instant::now();
        let mut store = Store::open(path, 0)?;
        let gpu = WgpuLtx::load(&mut store, 0, |_| {})?;
        eprintln!("WebGPU video stream loaded in {:.1} s", t.elapsed().as_secs_f64());
        let table = rope_table(&video_positions(frames, h, w, fps), &[20., 2048., 2048.], D, HEADS);
        let got: Vec<Vec<f32>> = [0.8, 0.25].iter().map(|&sigma| gpu.forward(&latent, tokens, &context, lc, sigma, &table, None)).collect::<Result<_>>()?;
        // and spatio-temporal guidance's pass, block 28's self-attention passed through
        let stg = gpu.forward(&latent, tokens, &context, lc, 0.8, &table, Some(28))?;
        drop(gpu);
        // (Candle's LTX is BF16 throughout: its CPU has no BF16 matmul, so CUDA's device OAIY_LTX_CUDA_DEVICE, 0 else)
        #[cfg(feature = "cuda")]
        let dev = Device::new_cuda(std::env::var("OAIY_LTX_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0))?;
        #[cfg(not(feature = "cuda"))]
        let dev = Device::Cpu;
        let mut cpu = crate::ltx::transformer::Transformer::new(Store::open(path, 0)?, &dev, 20 << 30, false, false)?;
        let rope = crate::ltx::transformer::Rope::video_with_end(frames, h, w, fps, false, &dev)?;
        let lt = candle_core::Tensor::from_vec(latent, (1, tokens, 128), &dev)?.to_dtype(DType::BF16)?;
        let ct = candle_core::Tensor::from_vec(context, (1, lc, D), &dev)?.to_dtype(DType::BF16)?;
        for (i, &sigma) in [0.8, 0.25].iter().enumerate() {
            let t = std::time::Instant::now();
            let (v, _) = cpu.forward(&lt, &ct, sigma, &rope, 0, 0, None, |_| {})?;
            let want = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let dot: f64 = got[i].iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got[i]) * norm(&want));
            eprintln!("sigma {sigma}: cosine {cos:.6} (Candle's step {:.1} s)", t.elapsed().as_secs_f64());
            assert!(cos > 0.99, "sigma {sigma}: cosine {cos}");
        }
        cpu.skip_video_self_attn = Some(28);
        let (v, _) = cpu.forward(&lt, &ct, 0.8, &rope, 0, 0, None, |_| {})?;
        let want = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let dot: f64 = stg.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (norm(&stg) * norm(&want));
        eprintln!("block 28's self-attention passed through: cosine {cos:.6}");
        assert!(cos > 0.99, "the perturbed pass: cosine {cos}");
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
