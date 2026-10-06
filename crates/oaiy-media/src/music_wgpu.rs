//! MiniMax Music 3 on WebGPU (a music job's `backend` "webgpu": any GPU; Candle's needs CUDA, its language model BF16):
//! [`crate::music`]'s stages, their features as rows (`[steps, channels]`). The acoustic stage here: the condition
//! encoder (each frame's eight hidden states mixed on the host, its 3-tap convolution on the device, stretched onto the
//! latent timeline on the host), the 36-block flow-matching transformer (its weights f16: layer norms, partial rotary,
//! full attention, a gated MLP), each window's guidance and Euler steps on the device, and the Flow-VAE decoder as
//! [`crate::sound_wgpu`]'s DAC (the same layers).
use crate::ltx::store::Store;
use crate::music::acoustic::{chunk_starts, latent_len, times, Vocoder, CHUNK_FRAMES, CROP_LEFT, CROP_RIGHT, HOP, LATENT_CHANNELS, OVERLAP};
use crate::wgpu_weights::f16_words_f32;
use candle_core::{Device, Result, Tensor};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs_wgpu::WgpuBackend;
use oaiy_engine::json::Json;
use std::path::Path;

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("music on WebGPU: {e}"))
}

fn upload(gpu: &WgpuBackend, v: &[f32]) -> DeviceVec {
    let d = gpu.vec(v.len());
    gpu.upload(&d, v);
    d
}

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// Channels-first values (`[c, len]`) as rows (`[len, c]`).
fn rows_of(planes: &[f32], c: usize, len: usize) -> Vec<f32> {
    (0..len * c).map(|i| planes[(i % c) * len + i / c]).collect()
}

/// A linear layer: its f16 weight `[n, k]` and bias.
struct Lin {
    w: DeviceVec,
    b: Option<DeviceVec>,
    n: usize,
    k: usize,
}

impl Lin {
    fn new(gpu: &WgpuBackend, w: &[f32], n: usize, k: usize, b: Option<&[f32]>, name: &str) -> Result<Self> {
        if w.len() != n * k {
            return Err(err(format!("{name}: {} values, not {n} x {k}", w.len())));
        }
        let words = f16_words_f32(w).ok_or_else(|| err(format!("{name}: past f16's range")))?;
        Ok(Self { w: upload(gpu, &words), b: b.map(|b| upload(gpu, b)), n, k })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows(&self.w, self.n, self.k, x, y, rows);
        if let Some(b) = &self.b {
            r.add_bias_rows(y, b, rows, self.n);
        }
    }
}

/// A host linear layer in F32 (the timestep's MLP, run once a step's time).
struct Host {
    w: Vec<f32>,
    b: Vec<f32>,
    n: usize,
    k: usize,
}

impl Host {
    fn run(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n).map(|o| self.b[o] + self.w[o * self.k..(o + 1) * self.k].iter().zip(x).map(|(w, v)| w * v).sum::<f32>()).collect()
    }
}

struct Block {
    /// The norms' weights less one, then their biases.
    norm1: DeviceVec,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    norm2: DeviceVec,
    /// The MLP's first projection in its halves: the value, then the gate (SiLU'd).
    value: Lin,
    gate: Lin,
    ff_out: Lin,
}

/// A pass's work vectors for up to `len` latents, kept for a song's every pass.
pub struct Scratch {
    len: usize,
    h0: DeviceVec,
    h1: DeviceVec,
    hs: DeviceVec,
    hp: DeviceVec,
    n: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    kv: DeviceVec,
    o: DeviceVec,
    att: DeviceVec,
    fv: DeviceVec,
    fg: DeviceVec,
    act: DeviceVec,
    hl: DeviceVec,
    h2: DeviceVec,
}

/// The flow-matching transformer on the device.
pub struct WgpuDit {
    blocks: Vec<Block>,
    preprocess: Lin,
    proj_in: Lin,
    proj_out: Lin,
    postprocess: Lin,
    fourier: Vec<f32>,
    time1: Host,
    time2: Host,
    dim: usize,
    cond_dim: usize,
    ff: usize,
    heads: usize,
    head_dim: usize,
    rotary: usize,
    one: DeviceVec,
}

impl WgpuDit {
    /// The `transformer` folder onto `gpu` (F32 rounded to f16); `progress(blocks loaded)`.
    pub fn load(dir: &Path, gpu: &WgpuBackend, mut progress: impl FnMut(usize)) -> Result<Self> {
        let config = crate::music::acoustic::read_config(&dir.join("config.json"))?;
        let int = |k: &str, d: usize| config.get(k).and_then(Json::as_i64).map_or(d, |v| v as usize);
        let (layers, heads, head_dim, rotary, ff, cond_dim) = (int("num_layers", 36), int("num_attention_heads", 32), int("attention_head_dim", 64), int("rotary_dim", 32), int("ff_inner_dim", 8192), int("condition_dim", 2048));
        let dim = heads * head_dim;
        let mut store = Store::open(dir, 0)?;
        let s = &mut store;
        let f32s = |s: &mut Store, k: &str| -> Result<Vec<f32>> { values(&s.tensor_f32(k, &Device::Cpu)?) };
        let lin = |s: &mut Store, k: &str, bias: bool, n: usize, kk: usize| -> Result<Lin> {
            let w = f32s(s, &format!("{k}.weight"))?;
            let b = if bias { Some(f32s(s, &format!("{k}.bias"))?) } else { None };
            Lin::new(gpu, &w, n, kk, b.as_deref(), k)
        };
        let norm = |s: &mut Store, k: &str| -> Result<DeviceVec> {
            let m: Vec<f32> = f32s(s, &format!("{k}.weight"))?.iter().map(|v| v - 1.).chain(f32s(s, &format!("{k}.bias"))?).collect();
            Ok(upload(gpu, &m))
        };
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("transformer_blocks.{i}");
            let (ffw, ffb) = (f32s(s, &format!("{p}.ff_in.weight"))?, f32s(s, &format!("{p}.ff_in.bias"))?);
            if ffw.len() != 2 * ff * dim || ffb.len() != 2 * ff {
                return Err(err(format!("{p}.ff_in: not [{}, {dim}]", 2 * ff)));
            }
            let half = ff * dim;
            blocks.push(Block {
                norm1: norm(s, &format!("{p}.norm1"))?,
                q: lin(s, &format!("{p}.attn.to_q"), false, dim, dim)?,
                k: lin(s, &format!("{p}.attn.to_k"), false, dim, dim)?,
                v: lin(s, &format!("{p}.attn.to_v"), false, dim, dim)?,
                out: lin(s, &format!("{p}.attn.to_out.0"), false, dim, dim)?,
                norm2: norm(s, &format!("{p}.norm2"))?,
                value: Lin::new(gpu, &ffw[..half], ff, dim, Some(&ffb[..ff]), &p)?,
                gate: Lin::new(gpu, &ffw[half..], ff, dim, Some(&ffb[ff..]), &p)?,
                ff_out: lin(s, &format!("{p}.ff_out"), true, dim, ff)?,
            });
            progress(i + 1);
        }
        let c = LATENT_CHANNELS;
        let width = 2 * c + cond_dim;
        let host = |s: &mut Store, k: &str| -> Result<Host> {
            let w = s.tensor_f32(&format!("{k}.weight"), &Device::Cpu)?;
            let (n, kk) = w.dims2()?;
            Ok(Host { w: values(&w)?, b: f32s(s, &format!("{k}.bias"))?, n, k: kk })
        };
        Ok(Self {
            blocks,
            // (the 1x1 convolutions as matrices)
            preprocess: Lin::new(gpu, &f32s(s, "preprocess_conv.weight")?, width, width, None, "preprocess_conv")?,
            proj_in: lin(s, "proj_in", false, dim, width)?,
            proj_out: lin(s, "proj_out", false, c, dim)?,
            postprocess: Lin::new(gpu, &f32s(s, "postprocess_conv.weight")?, c, c, None, "postprocess_conv")?,
            fourier: f32s(s, "time_proj.weight")?,
            time1: host(s, "time_embed.linear_1")?,
            time2: host(s, "time_embed.linear_2")?,
            dim,
            cond_dim,
            ff,
            heads,
            head_dim,
            rotary,
            one: upload(gpu, &[1.0]),
        })
    }

    /// The time `t`'s embedding (`[dim]`): random Fourier features, then the timestep MLP (F32, on the host).
    fn temb(&self, t: f32) -> Vec<f32> {
        let angles: Vec<f32> = self.fourier.iter().map(|w| t * w * (2. * std::f64::consts::PI) as f32).collect();
        let feats: Vec<f32> = angles.iter().map(|a| a.cos()).chain(angles.iter().map(|a| a.sin())).collect();
        let h: Vec<f32> = self.time1.run(&feats).iter().map(|v| v / (1. + (-v).exp())).collect();
        self.time2.run(&h)
    }

    /// The partial rotary's table for `s` positions: each position's `rotary / 2` (sin, cos) pairs, theta 1e4 (F32 as
    /// the reference computes them).
    fn table(&self, s: usize) -> Vec<f32> {
        let half = self.rotary / 2;
        let inv: Vec<f32> = (0..half).map(|i| 1. / 10000f32.powf((2 * i) as f32 / self.rotary as f32)).collect();
        (0..s).flat_map(|p| inv.iter().flat_map(move |f| { let a = p as f32 * f; [a.sin(), a.cos()] })).collect()
    }

    /// The work vectors of a pass over up to `len` latents.
    pub fn scratch(&self, gpu: &WgpuBackend, len: usize) -> Scratch {
        let (c, dim, ff, nh, hd) = (LATENT_CHANNELS, self.dim, self.ff, self.heads, self.head_dim);
        let (width, s) = (2 * c + self.cond_dim, len + 1);
        let v = |n: usize| gpu.vec(n);
        Scratch {
            len,
            h0: v(len * width),
            h1: v(len * width),
            hs: v(s * dim),
            hp: v(len * dim),
            n: v(s * dim),
            q: v(s * dim),
            k: v(s * dim),
            vv: v(s * dim),
            kv: v(s * 2 * dim),
            o: v(s * dim),
            att: v(gpu.attention_rows_full_out_len(s, nh, hd, s)),
            fv: v(s * ff),
            fg: v(s * ff),
            act: v(s * ff),
            hl: v(len * dim),
            h2: v(len * c),
        }
    }

    /// The velocity for latents `x` (`[len, 128]`) at the time whose embedding is `temb`, conditioned on `cond` (`[len,
    /// cond_dim]`), recorded on `r` into `out` (`[len, 128]`), its work in `w` (for at least `len`; a song's passes one
    /// after another in it); `zeros` at least `len * 128` zeros, `table` [`Self::table`]'s for at least `len + 1`.
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, w: &Scratch, len: usize, r: &mut dyn ChainRecorder, x: &DeviceVec, temb: &DeviceVec, cond: &DeviceVec, zeros: &DeviceVec, table: &DeviceVec, out: &DeviceVec) {
        let (c, dim, ff, nh, hd) = (LATENT_CHANNELS, self.dim, self.ff, self.heads, self.head_dim);
        let width = 2 * c + self.cond_dim;
        assert!(len <= w.len, "music on WebGPU: a pass of {len} latents in work vectors for {}", w.len);
        let s = len + 1;
        let one = &self.one;
        let Scratch { h0, h1, hs, hp, n, q, k, vv, kv, o, att, fv, fg, act, hl, h2, .. } = w;
        // [latents, zeros, condition] along the channels, plus its 1x1 convolution
        r.store_rows(x, h0, len, c, 0, width, 0);
        r.store_rows(zeros, h0, len, c, 0, width, c);
        r.store_rows(cond, h0, len, self.cond_dim, 0, width, 2 * c);
        self.preprocess.run(r, h0, h1, len);
        r.axpy_at(h1, h0, one, 0, len * width);
        // the time's token first, then the latents'
        r.copy(temb, 0, hs, 0, dim);
        self.proj_in.run(r, h1, hp, len);
        r.copy(hp, 0, hs, dim, len * dim);
        let scale = 1. / (hd as f32).sqrt();
        for b in &self.blocks {
            r.norm_mod_rows(hs, n, s, dim, &b.norm1, 0, Some(dim), RowNorm::Layer, 1e-5);
            b.q.run(r, n, q, s);
            b.k.run(r, n, k, s);
            b.v.run(r, n, vv, s);
            r.rope_partial_rows(q, s, nh, hd, self.rotary, table);
            r.rope_partial_rows(k, s, nh, hd, self.rotary, table);
            r.store_rows(k, kv, s, dim, 0, 2 * dim, 0);
            r.store_rows(vv, kv, s, dim, 0, 2 * dim, dim);
            r.attention_rows_full(q, kv, att, s, nh, nh, hd, s, scale);
            b.out.run(r, att, o, s);
            r.axpy_at(hs, o, one, 0, s * dim);
            r.norm_mod_rows(hs, n, s, dim, &b.norm2, 0, Some(dim), RowNorm::Layer, 1e-5);
            b.value.run(r, n, fv, s);
            b.gate.run(r, n, fg, s);
            r.silu_mul(fg, fv, act, s * ff);
            b.ff_out.run(r, act, o, s);
            r.axpy_at(hs, o, one, 0, s * dim);
        }
        // the latents' tokens out, plus the last 1x1 convolution
        r.copy(hs, dim, hl, 0, len * dim);
        self.proj_out.run(r, hl, h2, len);
        self.postprocess.run(r, h2, out, len);
        r.axpy_at(out, h2, &self.one, 0, len * c);
    }
}

/// The condition encoder: the eight hidden states' mix (on the host), its 3-tap convolution (on the device).
pub struct WgpuCondition {
    mix: Vec<f32>,
    width: usize,
    out_dim: usize,
    w: DeviceVec,
    b: DeviceVec,
}

impl WgpuCondition {
    pub fn load(dir: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let logits = values(&store.tensor_f32("layer_weight_logits", &Device::Cpu)?)?;
        let scale = values(&store.tensor_f32("layer_scale", &Device::Cpu)?)?[0];
        let w = store.tensor_f32("proj.weight", &Device::Cpu)?;
        let (out_dim, width, k) = w.dims3()?;
        if k != 3 {
            return Err(err(format!("the condition encoder's convolution has {k} taps, not 3")));
        }
        // (as Candle's: softmax in F32, the scale on each weight)
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        Ok(Self {
            mix: e.iter().map(|v| v / sum * scale).collect(),
            width,
            out_dim,
            w: gpu.conv1d_weights(&values(&w)?, out_dim, width, 3).ok_or_else(|| err("the condition encoder past f16's range"))?,
            b: upload(gpu, &values(&store.tensor_f32("proj.bias", &Device::Cpu)?)?),
        })
    }

    /// `frames` frames' hidden states (`[frames, 8 width]`) to the latent timeline's condition (`[latent_len(frames),
    /// out_dim]`, on the host).
    pub fn forward(&self, gpu: &WgpuBackend, hidden: &[f32], frames: usize) -> Result<Vec<f32>> {
        let (w, layers) = (self.width, self.mix.len());
        if hidden.len() != frames * layers * w {
            return Err(err(format!("{} hidden values for {frames} frames", hidden.len())));
        }
        let mut mixed = vec![0f32; frames * w];
        for (f, m) in mixed.chunks_exact_mut(w).enumerate() {
            let row = &hidden[f * layers * w..(f + 1) * layers * w];
            for (i, &weight) in self.mix.iter().enumerate() {
                for (a, h) in m.iter_mut().zip(&row[i * w..(i + 1) * w]) {
                    *a += h * weight;
                }
            }
        }
        let x = upload(gpu, &mixed);
        let y = gpu.vec(frames * self.out_dim);
        let mut rec = gpu.begin();
        rec.as_mut().conv1d_rows(&self.w, &self.b, self.out_dim, w, 3, 1, &x, frames, &y);
        rec.as_mut().read(&y);
        let y = rec.finish().pop().ok_or_else(|| err("the condition was not read"))?;
        // PyTorch's "nearest": source floor(dst * (in / out)) in F32
        let out = latent_len(frames);
        let scale = frames as f32 / out as f32;
        let d = self.out_dim;
        Ok((0..out).flat_map(|i| {
            let src = ((i as f32 * scale).floor() as usize).min(frames - 1);
            y[src * d..(src + 1) * d].iter().copied()
        }).collect())
    }
}

/// The Flow-VAE decoder's layers as the sound effects' DAC keeps them (its weight norms folded by Candle's loader).
fn vocoder(gpu: &WgpuBackend, v: &Vocoder) -> Result<crate::sound_wgpu::Dac> {
    use crate::sound::dac::{Conv, Dac, DacConfig, Residual, Up};
    let get = |k: &str| -> Result<Tensor> { v.w.get(k).cloned().ok_or_else(|| err(format!("the vocoder lacks {k}"))) };
    let conv = |p: &str| -> Result<Conv> { Ok(Conv { w: get(&format!("{p}.weight"))?, b: v.w.get(&format!("{p}.bias")).cloned() }) };
    let mut ups = Vec::new();
    for (i, &stride) in v.strides.iter().enumerate() {
        let p = format!("blocks.{i}");
        let residuals = [(1, 1), (2, 3), (3, 9)]
            .into_iter()
            .map(|(u, dilation)| {
                let r = format!("{p}.res_unit{u}");
                Ok(Residual { snake1: get(&format!("{r}.snake1.alpha"))?, conv1: conv(&format!("{r}.conv1"))?, dilation, snake2: get(&format!("{r}.snake2.alpha"))?, conv2: conv(&format!("{r}.conv2"))? })
            })
            .collect::<Result<Vec<_>>>()?;
        ups.push(Up { snake: get(&format!("{p}.snake1.alpha"))?, conv: conv(&format!("{p}.conv_t1"))?, stride, residuals });
    }
    let first = conv("conv_in")?;
    let cfg = DacConfig { latent_dim: LATENT_CHANNELS / 2, decoder_dim: first.w.dim(0)?, decoder_rates: v.strides.clone(), hop: HOP, sample_rate: crate::music::acoustic::SAMPLE_RATE };
    let dac = Dac { cfg, post_quant: Some(conv("dec_in_proj")?), first, ups, last_snake: get("snake_out.alpha")?, last: conv("conv_out")? };
    crate::sound_wgpu::Dac::from(gpu, &dac)
}

/// The acoustic stage on the device: the condition encoder, the transformer and the vocoder, with the window loop that
/// joins them (as [`crate::music::acoustic::Acoustic`]'s).
pub struct WgpuAcoustic {
    pub gpu: WgpuBackend,
    pub condition: WgpuCondition,
    pub transformer: WgpuDit,
    vocoder: crate::sound_wgpu::Dac,
    pub steps: usize,
    pub guidance: f64,
}

impl WgpuAcoustic {
    /// The MiniMax-Music3 folder's `condition_encoder`, `transformer` and `vocoder` onto `gpu`.
    pub fn load(dir: &Path, gpu: &WgpuBackend, steps: usize, guidance: f64, progress: impl FnMut(usize)) -> Result<Self> {
        let voc = Vocoder::load(&dir.join("vocoder"), &Device::Cpu)?;
        Ok(Self {
            condition: WgpuCondition::load(&dir.join("condition_encoder"), gpu)?,
            vocoder: vocoder(gpu, &voc)?,
            transformer: WgpuDit::load(&dir.join("transformer"), gpu, progress)?,
            gpu: gpu.clone(),
            steps,
            guidance,
        })
    }

    /// One window's latents (`[len, 128]`) to its stereo audio (`len * 512` samples a side): each side decoded from its
    /// own 64 channels.
    pub fn decode(&self, latents: &[f32], len: usize) -> Result<[Vec<f32>; 2]> {
        let c = LATENT_CHANNELS;
        let side = |s: usize| -> Result<Vec<f32>> {
            let z: Vec<f32> = latents.chunks_exact(c).flat_map(|row| row[s * c / 2..(s + 1) * c / 2].iter().copied()).collect();
            self.vocoder.decode(&self.gpu, &upload(&self.gpu, &z), len)
        };
        Ok([side(0)?, side(1)?])
    }

    /// Denoise every window of `hidden` (`frames` frames of eight hidden states, `[frames, 8 * 4096]`) and decode it: the
    /// stereo waveform (each side `[samples]`) in [-1, 1]. `noise(k, len)` window `k`'s starting noise (`[128, len]`,
    /// channels first); `progress(done, total)` counts Euler steps.
    pub fn generate(&self, hidden: &[f32], frames: usize, mut noise: impl FnMut(usize, usize) -> Result<Vec<f32>>, mut progress: impl FnMut(usize, usize)) -> Result<[Vec<f32>; 2]> {
        let g = &self.gpu;
        let (c, cd) = (LATENT_CHANNELS, self.transformer.cond_dim);
        let width = hidden.len() / frames.max(1);
        let starts = chunk_starts(frames);
        let ts = times(self.steps);
        let total = starts.len() * self.steps;
        // each step's time embedding (every window's the same), and the vectors every window's passes use
        let tembs: Vec<DeviceVec> = ts[..self.steps].iter().map(|&t| upload(g, &self.transformer.temb(t))).collect();
        let longest = latent_len(CHUNK_FRAMES.min(frames));
        let zeros = upload(g, &vec![0f32; longest * cd.max(c)]);
        let table = upload(g, &self.transformer.table(longest + 1));
        let scratch = self.transformer.scratch(g, longest);
        let (x, cd_dev, vc, vu, weights) = (g.vec(longest * c), g.vec(longest * cd), g.vec(longest * c), g.vec(longest * c), g.vec(2));
        // the previous window's shared latents and condition (rows), and how many
        let mut prev: Option<(Vec<f32>, Vec<f32>, usize)> = None;
        let mut chunks = Vec::with_capacity(starts.len());
        for (k, &start) in starts.iter().enumerate() {
            let end = (start + CHUNK_FRAMES).min(frames);
            let mut cond = self.condition.forward(g, &hidden[start * width..end * width], end - start)?;
            let len = cond.len() / cd;
            let mut overlap = 0;
            if let Some((_, pc, plen)) = &prev {
                overlap = (*plen).min(len);
                cond[..overlap * cd].copy_from_slice(&pc[..overlap * cd]);
            }
            let init = rows_of(&noise(k, len)?, c, len);
            g.upload(&x, &init);
            g.upload(&cd_dev, &cond);
            for i in 0..self.steps {
                let t = ts[i];
                if let (true, Some((pl, _, _))) = (overlap > 0, &prev) {
                    // the shared latents follow the previous window's path
                    let (a, b) = ((1. - (1. - 1e-6) * t as f64) as f32, t);
                    let pinned: Vec<f32> = init[..overlap * c].iter().zip(&pl[..overlap * c]).map(|(p, l)| p * a + l * b).collect();
                    g.upload_at(&x, 0, &pinned);
                }
                let dt = ts[i + 1] - t;
                g.upload(&weights, &[dt * self.guidance as f32, dt * (1. - self.guidance as f32)]);
                let mut rec = g.begin();
                rec.keep_groups(false);
                let r = rec.as_mut();
                self.transformer.forward(&scratch, len, r, &x, &tembs[i], &cd_dev, &zeros, &table, &vc);
                self.transformer.forward(&scratch, len, r, &x, &tembs[i], &zeros, &zeros, &table, &vu);
                r.axpy_at(&x, &vc, &weights, 0, len * c);
                r.axpy_at(&x, &vu, &weights, 1, len * c);
                rec.finish();
                progress(k * self.steps + i + 1, total);
            }
            let mut rec = g.begin();
            rec.as_mut().read_range(&x, 0, len * c);
            let mut xs = rec.finish().pop().ok_or_else(|| err("the latents were not read"))?;
            if let Some((pl, _, _)) = &prev {
                xs[..overlap * c].copy_from_slice(&pl[..overlap * c]);
            }
            let os = len.saturating_sub(2 * OVERLAP);
            let oe = os.max(len.saturating_sub(OVERLAP));
            prev = Some((xs[os * c..oe * c].to_vec(), cond[os * cd..oe * cd].to_vec(), oe - os));
            chunks.push((xs, len));
        }
        let n = chunks.len();
        let mut out = [Vec::new(), Vec::new()];
        for (i, (latents, len)) in chunks.iter().enumerate() {
            let wave = self.decode(latents, *len)?;
            let samples = wave[0].len();
            let left = if i == 0 { 0 } else { CROP_LEFT * HOP };
            let right = if i == n - 1 { 0 } else { CROP_RIGHT * HOP };
            for (o, w) in out.iter_mut().zip(&wave) {
                o.extend(w[left..samples - right].iter().map(|v| v.clamp(-1., 1.)));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod golden {
    use super::*;
    use std::path::PathBuf;

    fn dirs() -> (PathBuf, PathBuf) {
        (
            PathBuf::from(std::env::var("OAIY_MUSIC_GOLDEN").unwrap_or_else(|_| "E:/deepseek/nrob/target/music3-golden".into())),
            PathBuf::from(std::env::var("OAIY_MUSIC_MODEL").unwrap_or_else(|_| "E:/models/MiniMax-Music3".into())),
        )
    }

    fn load(dir: &Path, name: &str) -> Vec<f32> {
        std::fs::read(dir.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn relative(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        (a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / b.iter().map(|y| (*y as f64).powi(2)).sum::<f64>()).sqrt()
    }

    /// The acoustic stage on WebGPU against the official pipeline's strict F32 activations (`--ignored --nocapture`;
    /// OAIY_MUSIC_GOLDEN, else E:/deepseek/nrob/target/music3-golden; OAIY_MUSIC_MODEL, else E:/models/MiniMax-Music3):
    /// the condition encoder's windows, the vocoder's, the transformer's first two passes, and the whole stage's audio
    /// from the run's frame hiddens and noise.
    #[test]
    #[ignore = "needs MiniMax-Music3 and the reference's dumps"]
    fn the_webgpu_acoustic_stage_is_the_references() -> Result<()> {
        let (g, m) = dirs();
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let started = std::time::Instant::now();
        let acoustic = WgpuAcoustic::load(&m, &gpu, 30, 1.7, |_| {})?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        let c = LATENT_CHANNELS;
        for i in 0..2 {
            let hidden = load(&g, &format!("cond_in{i}.f32"));
            let frames = hidden.len() / 32768;
            let y = acoustic.condition.forward(&gpu, &hidden, frames)?;
            eprintln!("condition window {i}: {:.2e}", relative(&y, &load(&g, &format!("cond_out{i}.f32"))));
            let latents = load(&g, &format!("latents{i}.f32"));
            let len = latents.len() / c;
            let wave = acoustic.decode(&rows_of(&latents, c, len), len)?;
            let both: Vec<f32> = wave[0].iter().chain(&wave[1]).copied().collect();
            eprintln!("vocoder window {i}: {:.2e}", relative(&both, &load(&g, &format!("wave{i}.f32"))));
        }
        let dit = &acoustic.transformer;
        for i in 0..2 {
            let x = load(&g, &format!("dit_x{i}.f32"));
            let len = x.len() / c;
            let t = load(&g, &format!("dit_t{i}.f32"))[0];
            let cond = load(&g, &format!("dit_c{i}.f32"));
            let (xd, cdd, temb, table, zeros, out) = (upload(&gpu, &rows_of(&x, c, len)), upload(&gpu, &cond), upload(&gpu, &dit.temb(t)), upload(&gpu, &dit.table(len + 1)), upload(&gpu, &vec![0f32; len * c]), gpu.vec(len * c));
            let started = std::time::Instant::now();
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            dit.forward(&dit.scratch(&gpu, len), len, rec.as_mut(), &xd, &temb, &cdd, &zeros, &table, &out);
            rec.as_mut().read(&out);
            let v = rec.finish().pop().unwrap();
            let want = rows_of(&load(&g, &format!("dit_v{i}.f32")), c, len);
            eprintln!("transformer pass {i} (t {t}): {:.2e} in {:.3} s", relative(&v, &want), started.elapsed().as_secs_f64());
        }
        // the whole stage: both windows' inputs less the overlap, and the run's noise
        let (a, b) = (load(&g, "cond_in0.f32"), load(&g, "cond_in1.f32"));
        let hidden: Vec<f32> = a.iter().chain(&b[100 * 32768..]).copied().collect();
        let frames = hidden.len() / 32768;
        let noise = [load(&g, "noise0.f32"), load(&g, "noise1.f32")];
        let started = std::time::Instant::now();
        let audio = acoustic.generate(&hidden, frames, |k, len| { assert_eq!(noise[k].len(), c * len); Ok(noise[k].clone()) }, |_, _| {})?;
        let both: Vec<f32> = audio[0].iter().chain(&audio[1]).copied().collect();
        let e = relative(&both, &load(&g, "audio.f32"));
        eprintln!("the acoustic stage ({frames} frames, 2 windows of 30 steps): {e:.2e} in {:.1} s", started.elapsed().as_secs_f64());
        Ok(())
    }

    /// One transformer pass on fixed inputs, over and over (`--ignored --nocapture`; OAIY_PASSES, else 400): each
    /// pass's time, ten to a line.
    #[test]
    #[ignore = "needs MiniMax-Music3 and the reference's dumps"]
    fn measure_repeated_transformer_passes() -> Result<()> {
        let (g, m) = dirs();
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let dit = WgpuDit::load(&m.join("transformer"), &gpu, |_| {})?;
        let c = LATENT_CHANNELS;
        // (OAIY_MUSIC_DUMP: a folder's x.f32 and cond.f32, rows as a window's pass has them, else the reference's first)
        let (x, cond) = match std::env::var("OAIY_MUSIC_DUMP") {
            Ok(d) => (load(Path::new(&d), "x.f32"), load(Path::new(&d), "cond.f32")),
            Err(_) => {
                let x = load(&g, "dit_x0.f32");
                (rows_of(&x, c, x.len() / c), load(&g, "dit_c0.f32"))
            }
        };
        let len = x.len() / c;
        let (xd, cdd, temb, table, zeros, out) = (upload(&gpu, &x), upload(&gpu, &cond), upload(&gpu, &dit.temb(0.)), upload(&gpu, &dit.table(len + 1)), upload(&gpu, &vec![0f32; len * c]), gpu.vec(len * c));
        let scratch = dit.scratch(&gpu, len);
        let passes = std::env::var("OAIY_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
        // (OAIY_STEPS: each a step's two passes and the latents' update, as a window's)
        let steps = std::env::var_os("OAIY_STEPS").is_some();
        let (vu, zc) = (gpu.vec(len * c), upload(&gpu, &vec![0f32; len * dit.cond_dim]));
        let weights = upload(&gpu, &[-0.01, 0.007]);
        let mut line = Vec::new();
        let mut first: Option<Vec<f32>> = None;
        for i in 0..passes {
            // (OAIY_SAME: every pass from the same latents, its output against the first's)
            if std::env::var_os("OAIY_SAME").is_some() {
                gpu.upload(&xd, &x);
            }
            let started = std::time::Instant::now();
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            dit.forward(&scratch, len, rec.as_mut(), &xd, &temb, &cdd, &zeros, &table, &out);
            if steps {
                dit.forward(&scratch, len, rec.as_mut(), &xd, &temb, &zc, &zeros, &table, &vu);
                rec.as_mut().axpy_at(&xd, &out, &weights, 0, len * c);
                rec.as_mut().axpy_at(&xd, &vu, &weights, 1, len * c);
            }
            if std::env::var_os("OAIY_SAME").is_some() {
                rec.as_mut().read(&out);
                let got = rec.finish().pop().unwrap();
                match &first {
                    None => first = Some(got),
                    Some(f) => {
                        let differ = f.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                        if differ > 0 {
                            eprintln!("pass {}: {differ} values differ from the first pass's", i + 1);
                        }
                    }
                }
            } else {
                rec.finish();
            }
            line.push(format!("{:.0}", started.elapsed().as_secs_f64() * 1000.));
            if let Some(ms) = std::env::var("OAIY_STEP_SLEEP_MS").ok().and_then(|v| v.parse().ok()) {
                std::thread::sleep(std::time::Duration::from_millis(ms));
            }
            if i % 10 == 9 {
                eprintln!("passes {}..{}: {} ms", i - 8, i + 1, line.join(" "));
                line.clear();
            }
        }
        Ok(())
    }

    /// A 30 s song's rendering on its own (`--ignored --nocapture`): each window's time, the run's frame hiddens
    /// repeated to 750 frames.
    #[test]
    #[ignore = "needs MiniMax-Music3 and the reference's dumps"]
    fn measure_a_songs_rendering() -> Result<()> {
        let (g, m) = dirs();
        let gpu = WgpuBackend::nth(1, None).map_err(err)?;
        let acoustic = WgpuAcoustic::load(&m, &gpu, 30, 1.7, |_| {})?;
        let run = load(&g, "frame_hiddens.f32");
        let frames = 750;
        let hidden: Vec<f32> = run.chunks_exact(32768).cycle().take(frames).flatten().copied().collect();
        let started = std::time::Instant::now();
        let mut last = 0f64;
        let mut steps = Vec::new();
        let profiled = std::env::var_os("OAIY_CHAIN_PROFILE").is_some();
        acoustic.generate(&hidden, frames, |k, len| Ok(crate::music::noise(7, k, len)?.flatten_all()?.to_vec1::<f32>()?), |done, total| {
            let t = started.elapsed().as_secs_f64();
            steps.push(format!("{:.0}", (t - last) * 1000.));
            if profiled {
                let kernels = ggml_rs_wgpu::profile::take_kernels();
                if t - last > 0.5 || done == 61 {
                    let top: Vec<String> = kernels.into_iter().take(8).map(|(n, ms, c)| format!("{n} {ms:.1} ms ({c})")).collect();
                    eprintln!("step {done}: {}", top.join(", "));
                }
            }
            last = t;
            if done % 30 == 0 {
                eprintln!("window {} of {}: {} ms", done / 30, total / 30, steps.join(" "));
                steps.clear();
            }
        }).map(|audio| {
            let bad = audio.iter().flatten().filter(|v| !v.is_finite()).count();
            let big = audio.iter().flatten().filter(|v| v.is_finite()).fold(0f32, |m, v| m.max(v.abs()));
            eprintln!("audio: {} samples a side, the largest {big:.3}, {bad} not finite", audio[0].len());
        })?;
        eprintln!("rendered in {:.1} s", started.elapsed().as_secs_f64());
        Ok(())
    }
}
