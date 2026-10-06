//! MOSS-SoundEffect v2.0 on WebGPU (a sound job's `backend` "webgpu": any GPU, with tensor cores or without; Candle's
//! needs CUDA, its models BF16): [`crate::sound`]'s pipeline, its features as rows (`[steps, channels]`). The prompt
//! through Qwen3-1.7B ([`crate::qwen3_wgpu`]), the 1-D DiT's blocks as chain ops (modulated norms, full-width RMS q/k
//! norms, interleaved RoPE, self- and cross-attention, a GELU MLP), each step's guidance and Euler update on the device;
//! the DAC decoder's convolutions ([`ChainRecorder::conv1d_rows`], [`ChainRecorder::conv_transpose1d_rows`]) and Snake
//! from the Candle loader's folded weights.
use crate::ltx::store::Store;
use crate::qwen3_wgpu::{mat, vector, Mat, WgpuQwen3};
use crate::sound::{clean, event, noise, sigmas, Request, TEXT_LEN};
use candle_core::{Device, Result, Tensor};
use ggml_rs::chain::{ChainRecorder, DeviceChain, DeviceVec, RowNorm};
use ggml_rs_wgpu::WgpuBackend;
use oaiy_engine::json::Json;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn err(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("sound on WebGPU: {e}"))
}

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// A linear layer: its f16 weight and bias.
struct Lin {
    m: Mat,
    b: Option<DeviceVec>,
}

impl Lin {
    fn load(store: &mut Store, gpu: &WgpuBackend, prefix: &str) -> Result<Self> {
        let bias = format!("{prefix}.bias");
        let b = if store.index.get(&bias).is_some() { Some(vector(store, gpu, &bias)?) } else { None };
        Ok(Self { m: mat(store, gpu, &format!("{prefix}.weight"))?, b })
    }

    fn run(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        r.matmul_f16_rows(&self.m.w, self.m.n, self.m.k, x, y, rows);
        if let Some(b) = &self.b {
            r.add_bias_rows(y, b, rows, self.m.n);
        }
    }
}

/// A host linear layer in F32 (the timestep's path, run outside autocast in the reference).
struct Host {
    w: Vec<f32>,
    b: Vec<f32>,
    n: usize,
    k: usize,
}

impl Host {
    fn load(store: &mut Store, prefix: &str) -> Result<Self> {
        let w = store.tensor_f32(&format!("{prefix}.weight"), &Device::Cpu)?;
        let (n, k) = w.dims2()?;
        Ok(Self { w: values(&w)?, b: values(&store.tensor_f32(&format!("{prefix}.bias"), &Device::Cpu)?)?, n, k })
    }

    fn run(&self, x: &[f32]) -> Vec<f32> {
        (0..self.n).map(|o| self.b[o] + self.w[o * self.k..(o + 1) * self.k].iter().zip(x).map(|(w, v)| w * v).sum::<f32>()).collect()
    }
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1. + (-v).exp())).collect()
}

struct Block {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    norm_q: DeviceVec,
    norm_k: DeviceVec,
    cq: Lin,
    ck: Lin,
    cv: Lin,
    co: Lin,
    cnorm_q: DeviceVec,
    cnorm_k: DeviceVec,
    /// The cross-attention's input norm's weight less one, then its bias.
    norm3: DeviceVec,
    ffn1: Lin,
    ffn2: Lin,
    /// `(6, dim)`: shift, scale and gate for attention, then for the MLP (on the host, the timestep's added each step).
    modulation: Vec<f32>,
}

pub(crate) struct Dit {
    dim: usize,
    heads: usize,
    ffn: usize,
    in_dim: usize,
    freq_dim: usize,
    eps: f32,
    patch: Lin,
    text1: Lin,
    text2: Lin,
    time1: Host,
    time2: Host,
    time_proj: Host,
    blocks: Vec<Block>,
    head_mod: Vec<f32>,
    head: Lin,
    one: DeviceVec,
}

impl Dit {
    fn load(dir: &Path, gpu: &WgpuBackend) -> Result<Self> {
        let cfg = crate::sound::dit::Config::read(&dir.join("config.json"))?;
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let s = &mut store;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blocks.{i}");
            let norm3 = {
                let (w, b) = (values(&s.tensor_f32(&format!("{p}.norm2.weight"), &Device::Cpu)?)?, values(&s.tensor_f32(&format!("{p}.norm2.bias"), &Device::Cpu)?)?);
                let m: Vec<f32> = w.iter().map(|v| v - 1.).chain(b).collect();
                let d = gpu.vec(m.len());
                gpu.upload(&d, &m);
                d
            };
            blocks.push(Block {
                q: Lin::load(s, gpu, &format!("{p}.attn1.to_q"))?,
                k: Lin::load(s, gpu, &format!("{p}.attn1.to_k"))?,
                v: Lin::load(s, gpu, &format!("{p}.attn1.to_v"))?,
                o: Lin::load(s, gpu, &format!("{p}.attn1.to_out.0"))?,
                norm_q: vector(s, gpu, &format!("{p}.attn1.norm_q.weight"))?,
                norm_k: vector(s, gpu, &format!("{p}.attn1.norm_k.weight"))?,
                cq: Lin::load(s, gpu, &format!("{p}.attn2.to_q"))?,
                ck: Lin::load(s, gpu, &format!("{p}.attn2.to_k"))?,
                cv: Lin::load(s, gpu, &format!("{p}.attn2.to_v"))?,
                co: Lin::load(s, gpu, &format!("{p}.attn2.to_out.0"))?,
                cnorm_q: vector(s, gpu, &format!("{p}.attn2.norm_q.weight"))?,
                cnorm_k: vector(s, gpu, &format!("{p}.attn2.norm_k.weight"))?,
                norm3,
                ffn1: Lin::load(s, gpu, &format!("{p}.ffn.net.0.proj"))?,
                ffn2: Lin::load(s, gpu, &format!("{p}.ffn.net.2"))?,
                modulation: values(&s.tensor_f32(&format!("{p}.scale_shift_table"), &Device::Cpu)?)?,
            });
        }
        // the patch embedding, a 1x1 convolution `[dim, in_dim, 1]`: a matrix
        let patch = {
            let w = values(&s.tensor_f32("patch_embedding.weight", &Device::Cpu)?)?;
            let words = crate::wgpu_weights::f16_words_f32(&w).ok_or_else(|| err("the patch embedding past f16's range"))?;
            let d = gpu.vec(words.len());
            gpu.upload(&d, &words);
            Lin { m: Mat { w: d, n: cfg.dim, k: cfg.in_dim }, b: Some(vector(s, gpu, "patch_embedding.bias")?) }
        };
        let one = gpu.vec(1);
        gpu.upload(&one, &[1.0]);
        Ok(Self {
            dim: cfg.dim,
            heads: cfg.heads,
            ffn: cfg.ffn_dim,
            in_dim: cfg.in_dim,
            freq_dim: cfg.freq_dim,
            eps: cfg.eps as f32,
            patch,
            text1: Lin::load(s, gpu, "condition_embedder.text_embedder.linear_1")?,
            text2: Lin::load(s, gpu, "condition_embedder.text_embedder.linear_2")?,
            time1: Host::load(s, "condition_embedder.time_embedder.linear_1")?,
            time2: Host::load(s, "condition_embedder.time_embedder.linear_2")?,
            time_proj: Host::load(s, "condition_embedder.time_proj")?,
            head_mod: values(&s.tensor_f32("scale_shift_table", &Device::Cpu)?)?,
            head: Lin::load(s, gpu, "proj_out")?,
            blocks,
            one,
        })
    }

    /// A prompt's text states (`[TEXT_LEN, text_dim]`) as each block's cross-attention keys and values (`[TEXT_LEN, 2
    /// dim]`, keys RMS-normed).
    fn context(&self, gpu: &WgpuBackend, text: &DeviceVec) -> Vec<DeviceVec> {
        let (s, dim) = (TEXT_LEN, self.dim);
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let (c1, c1g, c) = (gpu.vec(s * dim), gpu.vec(s * dim), gpu.vec(s * dim));
        self.text1.run(r, text, &c1, s);
        r.gelu(&c1, &c1g, s * dim);
        self.text2.run(r, &c1g, &c, s);
        let (k, kn, v) = (gpu.vec(s * dim), gpu.vec(s * dim), gpu.vec(s * dim));
        let mut out = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let kv = gpu.vec(s * 2 * dim);
            b.ck.run(r, &c, &k, s);
            r.rmsnorm_rows(&k, &b.cnorm_k, &kn, s, self.eps);
            b.cv.run(r, &c, &v, s);
            r.store_rows(&kn, &kv, s, dim, 0, 2 * dim, 0);
            r.store_rows(&v, &kv, s, dim, 0, 2 * dim, dim);
            out.push(kv);
        }
        rec.finish();
        out
    }

    /// The timestep's embedding `t` (`[dim]`) and the blocks' modulation (`[6 dim]`).
    fn timestep(&self, timestep: f64) -> (Vec<f32>, Vec<f32>) {
        let half = self.freq_dim / 2;
        let sinusoid: Vec<f32> = (0..half)
            .map(|i| (timestep * 10000f64.powf(-(i as f64) / half as f64)).cos() as f32)
            .chain((0..half).map(|i| (timestep * 10000f64.powf(-(i as f64) / half as f64)).sin() as f32))
            .collect();
        let t = self.time2.run(&silu(&self.time1.run(&sinusoid)));
        let t_mod = self.time_proj.run(&silu(&t));
        (t, t_mod)
    }

    /// The flow at `x` (`[len, in_dim]`) for one prompt's `ctx`, into `out` (`[len, out_dim]`): `mods` every block's
    /// modulation (`[layers, 6, dim]`), `head` the head's (`[2, dim]`), `table` RoPE's.
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, gpu: &WgpuBackend, r: &mut dyn ChainRecorder, x: &DeviceVec, len: usize, mods: &DeviceVec, head: &DeviceVec, ctx: &[DeviceVec], table: &DeviceVec, out: &DeviceVec) {
        let (dim, h) = (self.dim, self.heads);
        let hd = dim / h;
        let scale = 1. / (hd as f32).sqrt();
        let v = |n: usize| gpu.vec(n);
        let (xs, n, q, qn, k, kn, vv, kv, o) = (v(len * dim), v(len * dim), v(len * dim), v(len * dim), v(len * dim), v(len * dim), v(len * dim), v(len * 2 * dim), v(len * dim));
        let att = v(gpu.attention_rows_full_out_len(len, h, hd, len).max(gpu.attention_rows_full_out_len(len, h, hd, TEXT_LEN)));
        let (f1, f1g) = (v(len * self.ffn), v(len * self.ffn));
        self.patch.run(r, x, &xs, len);
        for (bi, b) in self.blocks.iter().enumerate() {
            let base = bi * 6 * dim;
            // self-attention
            r.layernorm_mod_rows(&xs, &n, len, dim, mods, base + dim, Some(base), self.eps);
            b.q.run(r, &n, &q, len);
            b.k.run(r, &n, &k, len);
            b.v.run(r, &n, &vv, len);
            r.rmsnorm_rows(&q, &b.norm_q, &qn, len, self.eps);
            r.rmsnorm_rows(&k, &b.norm_k, &kn, len, self.eps);
            r.rope_rows(&qn, len, h, hd, table, false);
            r.rope_rows(&kn, len, h, hd, table, false);
            r.store_rows(&kn, &kv, len, dim, 0, 2 * dim, 0);
            r.store_rows(&vv, &kv, len, dim, 0, 2 * dim, dim);
            r.attention_rows_full(&qn, &kv, &att, len, h, h, hd, len, scale);
            b.o.run(r, &att, &o, len);
            r.add_gated_rows(&xs, &o, len, dim, mods, base + 2 * dim, false);
            // cross-attention to the text
            r.norm_mod_rows(&xs, &n, len, dim, &b.norm3, 0, Some(dim), RowNorm::Layer, self.eps);
            b.cq.run(r, &n, &q, len);
            r.rmsnorm_rows(&q, &b.cnorm_q, &qn, len, self.eps);
            r.attention_rows_full(&qn, &ctx[bi], &att, len, h, h, hd, TEXT_LEN, scale);
            b.co.run(r, &att, &o, len);
            r.axpy_at(&xs, &o, &self.one, 0, len * dim);
            // the MLP
            r.layernorm_mod_rows(&xs, &n, len, dim, mods, base + 4 * dim, Some(base + 3 * dim), self.eps);
            b.ffn1.run(r, &n, &f1, len);
            r.gelu(&f1, &f1g, len * self.ffn);
            b.ffn2.run(r, &f1g, &o, len);
            r.add_gated_rows(&xs, &o, len, dim, mods, base + 5 * dim, false);
        }
        r.layernorm_mod_rows(&xs, &n, len, dim, head, dim, Some(0), self.eps);
        self.head.run(r, &n, out, len);
    }
}

/// A 1-D convolution's packed weights and bias.
struct Conv1 {
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
}

fn conv1(gpu: &WgpuBackend, c: &crate::sound::dac::Conv) -> Result<Conv1> {
    let (cout, cin, k) = c.w.dims3()?;
    let w = gpu.conv1d_weights(&values(&c.w)?, cout, cin, k).ok_or_else(|| err("a DAC convolution past f16's range"))?;
    let b = match &c.b {
        Some(b) => values(b)?,
        None => vec![0.; cout],
    };
    let bd = gpu.vec(cout);
    gpu.upload(&bd, &b);
    Ok(Conv1 { w, b: bd, cin, cout, k })
}

struct Residual {
    snake1: DeviceVec,
    conv1: Conv1,
    dilation: usize,
    snake2: DeviceVec,
    conv2: Conv1,
}

struct Up {
    snake: DeviceVec,
    /// The transposed convolution's weight as `[cout, k, cin]` (f32), its bias, and its shape.
    w: DeviceVec,
    b: DeviceVec,
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    residuals: Vec<Residual>,
}

/// A DAC decoder on the device (MOSS-SoundEffect's, and MiniMax Music 3's Flow-VAE decoder: the same layers).
pub(crate) struct Dac {
    post_quant: Option<Conv1>,
    first: Conv1,
    ups: Vec<Up>,
    last_snake: DeviceVec,
    last: Conv1,
    one: DeviceVec,
}

impl Dac {
    pub(crate) fn from(gpu: &WgpuBackend, d: &crate::sound::dac::Dac) -> Result<Self> {
        let f32v = |t: &Tensor| -> Result<DeviceVec> {
            let v = values(t)?;
            let dv = gpu.vec(v.len());
            gpu.upload(&dv, &v);
            Ok(dv)
        };
        let mut ups = Vec::new();
        for u in &d.ups {
            let (cin, cout, k) = u.conv.w.dims3()?;
            let w = values(&u.conv.w)?;
            let packed: Vec<f32> = (0..cout * k * cin).map(|i| {
                let (co, j, c) = (i / (k * cin), (i / cin) % k, i % cin);
                w[(c * cout + co) * k + j]
            }).collect();
            let wd = gpu.vec(packed.len());
            gpu.upload(&wd, &packed);
            let b = match &u.conv.b {
                Some(b) => values(b)?,
                None => vec![0.; cout],
            };
            let bd = gpu.vec(cout);
            gpu.upload(&bd, &b);
            let mut residuals = Vec::new();
            for r in &u.residuals {
                residuals.push(Residual { snake1: f32v(&r.snake1)?, conv1: conv1(gpu, &r.conv1)?, dilation: r.dilation, snake2: f32v(&r.snake2)?, conv2: conv1(gpu, &r.conv2)? });
            }
            ups.push(Up { snake: f32v(&u.snake)?, w: wd, b: bd, cin, cout, k, stride: u.stride, residuals });
        }
        let one = gpu.vec(1);
        gpu.upload(&one, &[1.0]);
        Ok(Self { post_quant: d.post_quant.as_ref().map(|c| conv1(gpu, c)).transpose()?, first: conv1(gpu, &d.first)?, ups, last_snake: f32v(&d.last_snake)?, last: conv1(gpu, &d.last)?, one })
    }

    /// Latents `z` (`[len, latent_dim]`) to audio (`len * hop` samples).
    pub(crate) fn decode(&self, gpu: &WgpuBackend, z: &DeviceVec, len: usize) -> Result<Vec<f32>> {
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let r = rec.as_mut();
        let conv = |r: &mut dyn ChainRecorder, c: &Conv1, x: &DeviceVec, len: usize, dilation: usize| -> DeviceVec {
            let y = gpu.vec(len * c.cout);
            r.conv1d_rows(&c.w, &c.b, c.cout, c.cin, c.k, dilation, x, len, &y);
            y
        };
        let mut x = match &self.post_quant {
            Some(c) => conv(r, c, z, len, 1),
            None => z.clone(),
        };
        x = conv(r, &self.first, &x, len, 1);
        let mut len = len;
        for u in &self.ups {
            r.snake_rows(&x, &u.snake, len, u.cin);
            let s = u.stride;
            let out = len * s;
            let y = gpu.vec(out * u.cout);
            // (PyTorch's padding ceil(s / 2) and output padding s % 2: `len s` steps)
            r.conv_transpose1d_rows(&u.w, &u.b, u.cout, u.cin, u.k, s, s.div_ceil(2), &x, len, out, &y);
            x = y;
            len = out;
            // (two vectors a stage for its residual units, not three a unit: a music window's 8 s held 5 GB so)
            let (a, b) = (gpu.vec(len * u.cout), gpu.vec(len * u.cout));
            for res in &u.residuals {
                r.copy(&x, 0, &a, 0, len * u.cout);
                r.snake_rows(&a, &res.snake1, len, u.cout);
                r.conv1d_rows(&res.conv1.w, &res.conv1.b, res.conv1.cout, res.conv1.cin, res.conv1.k, res.dilation, &a, len, &b);
                r.snake_rows(&b, &res.snake2, len, res.conv1.cout);
                r.conv1d_rows(&res.conv2.w, &res.conv2.b, res.conv2.cout, res.conv2.cin, res.conv2.k, 1, &b, len, &a);
                r.axpy_at(&x, &a, &self.one, 0, len * u.cout);
            }
        }
        r.snake_rows(&x, &self.last_snake, len, self.last.cin);
        let y = conv(r, &self.last, &x, len, 1);
        r.tanh_in_place(&y, len * self.last.cout);
        r.read(&y);
        rec.finish().pop().ok_or_else(|| err("the audio was not read"))
    }
}

/// Qwen3-1.7B's final, normed states of each prompt on WebGPU, zero-padded to `TEXT_LEN` rows (`[TEXT_LEN, hidden]`);
/// an empty prompt all zeros.
fn encode_texts(dir: &Path, prompts: &[&str], gpu: &WgpuBackend) -> Result<Vec<DeviceVec>> {
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer").join("tokenizer.json")).map_err(|e| err(format!("tokenizer: {e}")))?;
    let te = dir.join("text_encoder");
    let cfg = Json::parse(&std::fs::read(te.join("config.json"))?).map_err(candle_core::Error::wrap)?;
    let n = |k: &str| cfg.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| err(format!("text encoder config: no {k}")));
    let mut store = Store::open(&te, 0)?;
    let qwen = WgpuQwen3::load(&mut store, gpu, "model", n("num_hidden_layers")?, n("num_attention_heads")?, n("num_key_value_heads")?, n("head_dim")?, cfg.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6), cfg.get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-6) as f32)?;
    let hidden = qwen.hidden();
    let mut out = Vec::new();
    for p in prompts {
        let text = clean(p);
        let mut ids = tok.encode(text.as_str(), true).map_err(|e| err(format!("tokenizer: {e}")))?.get_ids().to_vec();
        ids.truncate(TEXT_LEN);
        let padded = gpu.vec(TEXT_LEN * hidden);
        if !ids.is_empty() {
            let t = ids.len();
            let emb = values(&store.rows("model.embed_tokens.weight", &ids, &Device::Cpu)?)?;
            let (x, states) = (gpu.vec(t * hidden), gpu.vec(t * hidden));
            gpu.upload(&x, &emb);
            let mut rec = gpu.begin();
            rec.keep_groups(false);
            qwen.prompt(gpu, rec.as_mut(), &x, t, &states);
            rec.as_mut().copy(&states, 0, &padded, 0, t * hidden);
            rec.finish();
        }
        out.push(padded);
    }
    Ok(out)
}

/// [`crate::sound::generate`] on WebGPU.
pub fn generate(r: &Request, mut report: impl FnMut(Json)) -> Result<Json> {
    let started = Instant::now();
    std::fs::create_dir_all(&r.output)?;
    let gpu = WgpuBackend::nth(r.device, None).map_err(err)?;
    let index = Json::parse(&std::fs::read(r.model_dir.join("model_index.json"))?).map_err(candle_core::Error::wrap)?;
    let full_seconds = index.get("max_inference_seconds").and_then(Json::as_i64).unwrap_or(30) as f64;
    if r.seconds > full_seconds {
        candle_core::bail!("sound: at most {full_seconds} seconds");
    }
    let seconds = (r.seconds * 10.).round() / 10.;
    let prompt = format!("{} duration: {seconds:.1}s", r.prompt.trim());
    report(event("encoding_prompt", 0, 1));
    let texts = encode_texts(&r.model_dir, &[&prompt, &r.negative_prompt], &gpu)?;
    let load_started = Instant::now();
    report(event("loading_sound_model", 0, 1));
    let dac_cpu = crate::sound::dac::Dac::load(&r.model_dir.join("vae").join("vae_128d_48k.pth"), &Device::Cpu)?;
    let dac = Dac::from(&gpu, &dac_cpu)?;
    let (hop, sample_rate, latent_dim) = (dac_cpu.cfg.hop, dac_cpu.cfg.sample_rate, dac_cpu.cfg.latent_dim);
    drop(dac_cpu);
    let dit = Dit::load(&r.model_dir.join("transformer"), &gpu)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let (positive, negative) = (dit.context(&gpu, &texts[0]), dit.context(&gpu, &texts[1]));
    drop(texts);
    let frames = (sample_rate as f64 * full_seconds) as usize / hop;
    let channels = dit.in_dim;
    if channels != latent_dim {
        candle_core::bail!("sound: the DiT's {channels} channels are not the DAC's {latent_dim}");
    }
    // the seed's noise (or a file's), `(channels, frames)`, as rows of frames
    let planes = match &r.noise_file {
        Some(p) => {
            let bytes = std::fs::read(p)?;
            if bytes.len() != channels * frames * 4 {
                candle_core::bail!("sound: {} holds {} bytes, not {channels} x {frames} F32", p.display(), bytes.len());
            }
            bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        }
        None => noise(r.seed, channels * frames),
    };
    let rows: Vec<f32> = (0..frames * channels).map(|i| planes[(i % channels) * frames + i / channels]).collect();
    let x = gpu.vec(rows.len());
    gpu.upload(&x, &rows);
    // RoPE over the frame index: the head's interleaved pairs
    let hd = dit.dim / dit.heads;
    let table: Vec<f32> = (0..frames)
        .flat_map(|p| (0..hd / 2).flat_map(move |i| {
            let a = p as f64 / 10000f64.powf(2. * i as f64 / hd as f64);
            [a.sin() as f32, a.cos() as f32]
        }))
        .collect();
    let td = gpu.vec(table.len());
    gpu.upload(&td, &table);
    let (mods, head, weights) = (gpu.vec(dit.blocks.len() * 6 * dit.dim), gpu.vec(2 * dit.dim), gpu.vec(2));
    let (vpos, vneg) = (gpu.vec(frames * channels), gpu.vec(frames * channels));
    let s = sigmas(r.steps, r.shift);
    let denoise_started = Instant::now();
    // (short steps for seconds on end, a dozen pieces each: two in flight, or a card under a power limit throttles
    // itself for most of them: 100 steps in 13 s where 36)
    gpu.pieces_in_flight_at_most(2);
    for i in 0..r.steps {
        report(event("generating_sound", i, r.steps));
        let (t, t_mod) = dit.timestep(s[i] * 1000.);
        let m: Vec<f32> = dit.blocks.iter().flat_map(|b| b.modulation.iter().zip(t_mod.iter()).map(|(a, b)| a + b)).collect();
        gpu.upload(&mods, &m);
        let hm: Vec<f32> = dit.head_mod.iter().enumerate().map(|(j, v)| v + t[j % dit.dim]).collect();
        gpu.upload(&head, &hm);
        // x += dt (neg + cfg (pos - neg)): a weight each
        let dt = (s[i + 1] - s[i]) as f32;
        let guided = r.cfg != 1.0;
        let cfg = r.cfg as f32;
        gpu.upload(&weights, &if guided { [dt * cfg, dt * (1. - cfg)] } else { [dt, 0.] });
        let mut rec = gpu.begin();
        rec.keep_groups(false);
        let rr = rec.as_mut();
        dit.forward(&gpu, rr, &x, frames, &mods, &head, &positive, &td, &vpos);
        if guided {
            dit.forward(&gpu, rr, &x, frames, &mods, &head, &negative, &td, &vneg);
        }
        rr.axpy_at(&x, &vpos, &weights, 0, frames * channels);
        if guided {
            rr.axpy_at(&x, &vneg, &weights, 1, frames * channels);
        }
        rec.finish();
    }
    gpu.pieces_in_flight_at_most(0);
    let denoise_seconds = denoise_started.elapsed().as_secs_f64();
    drop((positive, negative, dit));
    report(event("decoding_sound", 0, 1));
    let decode_started = Instant::now();
    // only the frames asked for (and a margin past them) decoded, as Candle's
    let keep = ((seconds * sample_rate as f64) as usize).div_ceil(hop);
    let margin = 64;
    let len = (keep + margin).min(frames);
    let decoded = dac.decode(&gpu, &x, len)?;
    let samples = (seconds * sample_rate as f64) as usize;
    let audio = &decoded[..samples.min(decoded.len())];
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(candle_core::Error::wrap)?.as_nanos();
    let path = r.output.join(format!("sound-{stamp}-{}.wav", r.seed));
    crate::tts::write_wav(&path, audio, sample_rate)?;
    report(event("generating_sound", r.steps, r.steps));
    Ok(Json::obj([
        ("path", Json::str(path.to_string_lossy())),
        ("sample_rate", Json::Int(sample_rate as i64)),
        ("duration", Json::Num(audio.len() as f64 / sample_rate as f64)),
        ("prompt", Json::str(&prompt)),
        ("steps", Json::Int(r.steps as i64)),
        ("cfg_scale", Json::Num(r.cfg)),
        ("load_seconds", Json::Num(load_seconds)),
        ("denoise_seconds", Json::Num(denoise_seconds)),
        ("decode_seconds", Json::Num(decode_seconds)),
        ("seconds", Json::Num(started.elapsed().as_secs_f64())),
        ("seed", Json::Int(r.seed as i64)),
        ("backend", Json::str("webgpu")),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let n = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        dot / (n(a) * n(b)).max(1e-30)
    }

    #[cfg(feature = "cuda")]
    fn read(gpu: &WgpuBackend, v: &DeviceVec) -> Vec<f32> {
        let mut rec = gpu.begin();
        rec.read(v);
        rec.finish().pop().unwrap()
    }

    /// Each stage on WebGPU against Candle's on CUDA (`--ignored --nocapture`, built with `webgpu cuda`; Candle on
    /// OAIY_SOUND_CUDA_DEVICE, 0 by default): the prompt's text states, a step's flow from the same noise and text, and
    /// the DAC's audio from the same latents.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs MOSS-SoundEffect v2.0 and CUDA"]
    fn the_webgpu_sound_is_candles() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("OAIY_SOUND").unwrap_or_else(|_| "E:/models/MOSS-SoundEffect-v2.0".into()));
        let cuda: usize = std::env::var("OAIY_SOUND_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let dev = Device::new_cuda(cuda)?;
        let gpu = WgpuBackend::new(None).map_err(err)?;
        let prompt = "a wooden door creaking open slowly duration: 3.0s";
        // the text states
        let want = crate::sound::encode_texts(&dir, &[prompt], &dev)?.remove(0).to_dtype(candle_core::DType::F32)?.squeeze(0)?;
        let got = read(&gpu, &encode_texts(&dir, &[prompt], &gpu)?[0]);
        let want_v = values(&want)?;
        let t = (0..TEXT_LEN).take_while(|&r| want_v[r * 2048..(r + 1) * 2048].iter().any(|v| *v != 0.)).count();
        let worst = (0..t).map(|r| cosine(&got[r * 2048..(r + 1) * 2048], &want_v[r * 2048..(r + 1) * 2048])).fold(1f64, f64::min);
        eprintln!("text states: {t} tokens, the worst row's cosine {worst:.6}");
        assert!(worst > 0.995);
        // a step's flow from the same noise and the same (Candle's) text
        let cdit = crate::sound::dit::Dit::load(&dir.join("transformer"), &dev)?;
        let frames = 1500;
        let noise_v = noise(7, 128 * frames);
        let x = Tensor::from_vec(noise_v.clone(), (1, 128, frames), &dev)?;
        let ctx = cdit.context(&want.unsqueeze(0)?)?;
        let flow = values(&cdit.forward(&x, 700., &[&ctx])?.squeeze(0)?)?;
        drop((cdit, ctx));
        let dit = Dit::load(&dir.join("transformer"), &gpu)?;
        let text = gpu.vec(want_v.len());
        gpu.upload(&text, &want_v);
        let gctx = dit.context(&gpu, &text);
        let rows: Vec<f32> = (0..frames * 128).map(|i| noise_v[(i % 128) * frames + i / 128]).collect();
        let (xd, out) = (gpu.vec(rows.len()), gpu.vec(rows.len()));
        gpu.upload(&xd, &rows);
        let (tt, t_mod) = dit.timestep(700.);
        let m: Vec<f32> = dit.blocks.iter().flat_map(|b| b.modulation.iter().zip(t_mod.iter()).map(|(a, b)| a + b)).collect();
        let hm: Vec<f32> = dit.head_mod.iter().enumerate().map(|(j, v)| v + tt[j % dit.dim]).collect();
        let (md, hd) = (gpu.vec(m.len()), gpu.vec(hm.len()));
        gpu.upload(&md, &m);
        gpu.upload(&hd, &hm);
        let head_dim = dit.dim / dit.heads;
        let table: Vec<f32> = (0..frames).flat_map(|p| (0..head_dim / 2).flat_map(move |i| { let a = p as f64 / 10000f64.powf(2. * i as f64 / head_dim as f64); [a.sin() as f32, a.cos() as f32] })).collect();
        let td = gpu.vec(table.len());
        gpu.upload(&td, &table);
        let mut rec = gpu.begin();
        dit.forward(&gpu, rec.as_mut(), &xd, frames, &md, &hd, &gctx, &td, &out);
        rec.read(&out);
        let got = rec.finish().pop().unwrap();
        let planes: Vec<f32> = (0..128 * frames).map(|i| got[(i % frames) * 128 + i / frames]).collect();
        let c = cosine(&planes, &flow);
        eprintln!("a step's flow: cosine {c:.6}");
        assert!(c > 0.995);
        // the DAC's audio from the same latents (the flow's, as latents)
        let cdac = crate::sound::dac::Dac::load(&dir.join("vae").join("vae_128d_48k.pth"), &dev)?;
        let len = 200;
        let z = Tensor::from_vec(flow[..128 * frames].to_vec(), (1, 128, frames), &dev)?.narrow(2, 0, len)?.contiguous()?;
        let want_audio = values(&cdac.decode(&z)?)?;
        let dac = Dac::from(&gpu, &crate::sound::dac::Dac::load(&dir.join("vae").join("vae_128d_48k.pth"), &Device::Cpu)?)?;
        let zrows: Vec<f32> = (0..len * 128).map(|i| flow[(i % 128) * frames + i / 128]).collect();
        let zd = gpu.vec(zrows.len());
        gpu.upload(&zd, &zrows);
        let got_audio = dac.decode(&gpu, &zd, len)?;
        let c = cosine(&got_audio, &want_audio);
        eprintln!("the DAC's audio ({} samples): cosine {c:.6}", want_audio.len());
        assert!(got_audio.len() == want_audio.len() && c > 0.995);
        Ok(())
    }

    /// A sound effect on WebGPU (`--ignored --nocapture`; OAIY_SOUND its folder, else E:/models/MOSS-SoundEffect-v2.0;
    /// OAIY_SOUND_STEPS the steps, 12 by default): its timings, and its audio finite and not silence.
    #[test]
    #[ignore = "needs MOSS-SoundEffect v2.0"]
    fn measure_webgpu_sound() -> Result<()> {
        let dir = std::env::var("OAIY_SOUND").unwrap_or_else(|_| "E:/models/MOSS-SoundEffect-v2.0".into());
        let steps = std::env::var("OAIY_SOUND_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        let out = std::env::temp_dir().join("oaiy-sound-webgpu");
        let j = Json::parse(format!(r#"{{"model_dir":"{dir}","prompt":"a wooden door creaking open slowly","seconds":3,"steps":{steps},"seed":5,"device":1,"backend":"webgpu","output_dir":"{}"}}"#, out.to_string_lossy().replace(char::from(92), "/")).as_bytes()).map_err(candle_core::Error::wrap)?;
        let r = Request::parse(&j).map_err(candle_core::Error::Msg)?;
        assert!(r.webgpu);
        let result = generate(&r, |e| eprintln!("{}", e.to_json()))?;
        eprintln!("{}", result.to_json());
        let path = result.get("path").and_then(Json::as_str).expect("a path");
        let bytes = std::fs::read(path)?;
        // 16-bit PCM after a 44-byte header
        let samples: Vec<f32> = bytes[44..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.).collect();
        let rms = (samples.iter().map(|v| v * v).sum::<f32>() / samples.len() as f32).sqrt();
        eprintln!("{} samples, RMS {rms:.4}", samples.len());
        assert!(samples.len() > 100_000 && rms > 1e-3 && samples.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
