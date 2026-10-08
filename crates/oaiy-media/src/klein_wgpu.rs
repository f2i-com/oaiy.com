//! FLUX.2 Klein 4B's transformer on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as [`crate::klein::transformer`]
//! computes it: a step's modulations from its time, five double blocks (the text's tokens and the picture's each
//! through their own weights, attending together), twenty single blocks over both, the picture's tokens out. Its
//! matrices as [`crate::qwen_wgpu`] holds Qwen Image's: a GGUF's K-quant blocks as they are (the Q4_K_M file's 2.6 GB;
//! each LoRA's factors beside them, added as it runs), a checkpoint's floats f16 (LoRA factors merged in).
use crate::klein::{math, transformer::Config};
use crate::qwen_wgpu::{err, first, host, matrix, mul, timestep, vector, Low, Mat};
use crate::{lora::Loras, weights::Weights};
use candle_core::{Device, Result, Tensor};
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::{Path, PathBuf};

const EPS: f32 = 1e-6;

/// One stream of a double block.
struct Stream {
    qkv: Mat,
    proj: Mat,
    mlp_in: Mat,
    mlp_out: Mat,
    qn: DeviceVec,
    kn: DeviceVec,
}

struct Double {
    img: Stream,
    txt: Stream,
}

struct Single {
    input: Mat,
    output: Mat,
    qn: DeviceVec,
    kn: DeviceVec,
}

/// A run of rows' attention vectors: the split q, k and v, and q and k normed (then rotated in place).
struct Heads {
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
}

/// A step's vectors, kept for the next step of the same sizes (`ni` picture tokens after `nt` of text).
struct Scratch {
    ni: usize,
    nt: usize,
    lat: DeviceVec,
    ctx: DeviceVec,
    /// Each position's RoPE table: the text's, the picture's, and both in turn.
    tables: [DeviceVec; 3],
    t: DeviceVec,
    time: [DeviceVec; 4],
    /// The modulations: the picture's and the text's double streams' (six each), the single blocks' (three), the
    /// output's (two).
    mods: [DeviceVec; 4],
    xt: DeviceVec,
    xi: DeviceVec,
    all: DeviceVec,
    norm: DeviceVec,
    /// A projection's wide output: a stream's q, k and v side by side, a single block's nine widths.
    wide: DeviceVec,
    text: Heads,
    image: Heads,
    both: Heads,
    kv: DeviceVec,
    att: DeviceVec,
    /// The picture's rows of the attention's output.
    ai: DeviceVec,
    o: DeviceVec,
    fused: DeviceVec,
    act: DeviceVec,
    cat: DeviceVec,
    vel: DeviceVec,
    low: Low,
}

pub struct WgpuKlein {
    gpu: ggml_rs_wgpu::WgpuBackend,
    cfg: Config,
    /// The largest rank of the LoRA factors kept beside quantized weights (0: none).
    rank: usize,
    img_in: Mat,
    txt_in: Mat,
    time_in: Mat,
    time_out: Mat,
    img_mod: Mat,
    txt_mod: Mat,
    single_mod: Mat,
    final_mod: Mat,
    final_proj: Mat,
    double: Vec<Double>,
    single: Vec<Single>,
    lora_notes: Vec<String>,
    scratch: Option<Scratch>,
}

/// Each position's RoPE pairs as the chain's table has them (a pair's sine, then its cosine), as [`math::rotary`]
/// makes the angles: four axes, each coordinate over `theta` to the pair's share of its axis.
fn rope_table(coords: &[[f32; 4]], axes: [usize; 4], theta: f64) -> Vec<f32> {
    let mut t = Vec::with_capacity(coords.len() * axes.iter().sum::<usize>());
    for coord in coords {
        for (axis, width) in axes.iter().enumerate() {
            for i in 0..width / 2 {
                let angle = coord[axis] as f64 / theta.powf((2 * i) as f64 / *width as f64);
                t.push(angle.sin() as f32);
                t.push(angle.cos() as f32);
            }
        }
    }
    t
}

impl WgpuKlein {
    /// The transformer at `path` on GPU `device` (as the computer counts them; OAIY_WEBGPU_ADAPTER naming one
    /// instead), with `loras`. `progress(block)` as each of its 25 blocks loads.
    pub fn load(path: &Path, device: usize, loras: &[(PathBuf, f64)], progress: impl FnMut(usize)) -> Result<Self> {
        Self::load_config(path, device, loras, Config::default(), progress)
    }

    fn load_config(path: &Path, device: usize, loras: &[(PathBuf, f64)], cfg: Config, mut progress: impl FnMut(usize)) -> Result<Self> {
        let gpu = ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?;
        let mut w = Weights::open(path)?;
        cfg.validate(&w)?;
        let mut lora = Loras::open(loras)?;
        lora.validate_modules(&cfg.projections())?;
        let mut m = |name: &str| matrix(&mut w, &mut lora, &gpu, name);
        let (img_in, txt_in) = (m("img_in")?, m("txt_in")?);
        let (time_in, time_out) = (m("time_in.in_layer")?, m("time_in.out_layer")?);
        let (img_mod, txt_mod, single_mod) = (m("double_stream_modulation_img.lin")?, m("double_stream_modulation_txt.lin")?, m("single_stream_modulation.lin")?);
        let (final_mod, final_proj) = (m("final_layer.adaLN_modulation.1")?, m("final_layer.linear")?);
        let stream = |w: &mut Weights, lora: &mut Loras, p: &str| -> Result<Stream> {
            let mut m = |name: &str| matrix(w, lora, &gpu, &format!("{p}_{name}"));
            let (qkv, proj, mlp_in, mlp_out) = (m("attn.qkv")?, m("attn.proj")?, m("mlp.0")?, m("mlp.2")?);
            Ok(Stream { qkv, proj, mlp_in, mlp_out, qn: vector(w, &gpu, &format!("{p}_attn.norm.query_norm.scale"), 0.)?, kn: vector(w, &gpu, &format!("{p}_attn.norm.key_norm.scale"), 0.)? })
        };
        let mut double = Vec::with_capacity(cfg.double);
        for i in 0..cfg.double {
            let p = format!("double_blocks.{i}");
            double.push(Double { img: stream(&mut w, &mut lora, &format!("{p}.img"))?, txt: stream(&mut w, &mut lora, &format!("{p}.txt"))? });
            progress(i + 1);
        }
        let mut single = Vec::with_capacity(cfg.single);
        for i in 0..cfg.single {
            let p = format!("single_blocks.{i}");
            let (input, output) = (matrix(&mut w, &mut lora, &gpu, &format!("{p}.linear1"))?, matrix(&mut w, &mut lora, &gpu, &format!("{p}.linear2"))?);
            single.push(Single { input, output, qn: vector(&mut w, &gpu, &format!("{p}.norm.query_norm.scale"), 0.)?, kn: vector(&mut w, &gpu, &format!("{p}.norm.key_norm.scale"), 0.)? });
            progress(cfg.double + i + 1);
        }
        // every block has been read once: an adapter that fit nothing is for another model
        let lora_notes = lora.check()?;
        let mats = [&img_in, &txt_in, &time_in, &time_out, &img_mod, &txt_mod, &single_mod, &final_mod, &final_proj]
            .into_iter()
            .chain(double.iter().flat_map(|b| [&b.img, &b.txt]).flat_map(|s| [&s.qkv, &s.proj, &s.mlp_in, &s.mlp_out]))
            .chain(single.iter().flat_map(|b| [&b.input, &b.output]));
        let rank = mats.flat_map(|m| m.lora.iter().map(|l| l.2)).max().unwrap_or(0);
        Ok(Self { gpu, cfg, rank, img_in, txt_in, time_in, time_out, img_mod, txt_mod, single_mod, final_mod, final_proj, double, single, lora_notes, scratch: None })
    }

    /// The GPU it is on (the VAE's decoder beside it).
    pub fn backend(&self) -> &ggml_rs_wgpu::WgpuBackend {
        &self.gpu
    }

    /// LoRA adapters that fit only in part, and what of them was left out.
    pub fn lora_notes(&self) -> &[String] {
        &self.lora_notes
    }

    /// Let go of a step's vectors and what the device keeps for the next (the VAE's decode wants the room), made
    /// again by the next step.
    pub fn release_scratch(&mut self) {
        self.scratch = None;
        self.gpu.release_cached();
    }

    fn vec(&self, len: usize) -> DeviceVec {
        self.gpu.vec(len.max(1))
    }

    fn heads(&self, rows: usize) -> Heads {
        let n = rows * self.cfg.hidden;
        Heads { q: self.vec(n), k: self.vec(n), v: self.vec(n), qq: self.vec(n), kk: self.vec(n) }
    }

    /// The vectors of a step of `ni` picture tokens after `nt` of text (kept for the next of the same sizes).
    fn scratch(&mut self, ni: usize, nt: usize) -> Scratch {
        if let Some(s) = self.scratch.take().filter(|s| s.ni == ni && s.nt == nt) {
            return s;
        }
        let (h, hd, np) = (self.cfg.hidden, self.cfg.hidden / self.cfg.heads, nt + ni);
        // (a LoRA's running vectors: nothing to speak of where no factors are kept)
        let low_rows = if self.rank == 0 { 0 } else { np };
        Scratch {
            ni,
            nt,
            lat: self.vec(ni * self.cfg.channels),
            ctx: self.vec(nt * self.cfg.context),
            tables: [self.vec(nt * hd), self.vec(ni * hd), self.vec(np * hd)],
            t: self.vec(256),
            time: [self.vec(h), self.vec(h), self.vec(h), self.vec(h)],
            mods: [self.vec(6 * h), self.vec(6 * h), self.vec(3 * h), self.vec(2 * h)],
            xt: self.vec(nt * h),
            xi: self.vec(ni * h),
            all: self.vec(np * h),
            norm: self.vec(np * h),
            wide: self.vec(np * 9 * h),
            text: self.heads(nt),
            image: self.heads(ni),
            both: self.heads(np),
            kv: self.vec(np * 2 * h),
            att: self.vec(self.gpu.attention_rows_full_out_len(np, self.cfg.heads, hd, np)),
            ai: self.vec(ni * h),
            o: self.vec(np * h),
            fused: self.vec(np * 6 * h),
            act: self.vec(np * 3 * h),
            cat: self.vec(np * 4 * h),
            vel: self.vec(ni * self.cfg.channels),
            low: Low { t: self.vec(low_rows * self.rank), y: self.vec(low_rows * 9 * h) },
        }
    }

    /// The velocity of `x` (`[1, h w, channels]`) at `t`, conditioned on `context` (`[1, nt, context]`): `[1, h w,
    /// channels]` on the CPU, as [`crate::klein::transformer::Transformer::predict`]'s.
    pub fn predict(&mut self, x: &Tensor, context: &Tensor, t: f64, h: usize, w: usize) -> Result<Tensor> {
        let cfg = self.cfg;
        let (ni, nt) = (h * w, context.dim(1)?);
        if x.dims() != [1, ni, cfg.channels] || context.dims() != [1, nt, cfg.context] || ni == 0 || nt == 0 {
            candle_core::bail!("invalid Klein input shape");
        }
        let (d, heads, hd, np) = (cfg.hidden, cfg.heads, cfg.hidden / cfg.heads, nt + ni);
        let s = self.scratch(ni, nt);
        let coords = math::ids(nt, h, w);
        self.gpu.upload(&s.lat, &host(x)?);
        self.gpu.upload(&s.ctx, &host(context)?);
        self.gpu.upload(&s.tables[0], &rope_table(&coords[..nt], cfg.axes, 2000.));
        self.gpu.upload(&s.tables[1], &rope_table(&coords[nt..], cfg.axes, 2000.));
        self.gpu.upload(&s.tables[2], &rope_table(&coords, cfg.axes, 2000.));
        self.gpu.upload(&s.t, &timestep(t));
        let mut rec = self.gpu.begin();
        // (bind groups kept would hold each step's own scratch, the tensor cores' f16 copies and partial sums)
        rec.keep_groups(false);
        let r = rec.as_mut();
        let low = &s.low;
        // the time's vector, its SiLU, and every block's modulation from that
        let [t1, t1s, vec, act] = &s.time;
        let [mi, mt, ms, mf] = &s.mods;
        mul(r, &self.time_in, &s.t, t1, 1, true, low);
        r.mul_sigmoid(t1, t1, t1s, d);
        mul(r, &self.time_out, t1s, vec, 1, true, low);
        r.mul_sigmoid(vec, vec, act, d);
        mul(r, &self.img_mod, act, mi, 1, true, low);
        mul(r, &self.txt_mod, act, mt, 1, true, low);
        mul(r, &self.single_mod, act, ms, 1, true, low);
        mul(r, &self.final_mod, act, mf, 1, true, low);
        // (the text's states may pass f16's range: read as they are)
        mul(r, &self.img_in, &s.lat, &s.xi, ni, false, low);
        mul(r, &self.txt_in, &s.ctx, &s.xt, nt, true, low);
        let scale = 1.0 / (hd as f32).sqrt();
        // a stream's q, k and v from its modulated norm: q and k normed a head at a time and rotated by its positions
        let qkv = |r: &mut dyn ChainRecorder, st: &Stream, x: &DeviceVec, m: &DeviceVec, hv: &Heads, table: &DeviceVec, rows: usize| {
            r.layernorm_mod_rows(x, &s.norm, rows, d, m, d, Some(0), EPS);
            mul(r, &st.qkv, &s.norm, &s.wide, rows, false, low);
            r.copy_cols(&s.wide, &hv.q, rows, d, 3 * d, 0);
            r.copy_cols(&s.wide, &hv.k, rows, d, 3 * d, d);
            r.copy_cols(&s.wide, &hv.v, rows, d, 3 * d, 2 * d);
            r.rmsnorm_rows(&hv.q, &st.qn, &hv.qq, rows * heads, EPS);
            r.rmsnorm_rows(&hv.k, &st.kn, &hv.kk, rows * heads, EPS);
            r.rope_rows(&hv.qq, rows, heads, hd, table, false);
            r.rope_rows(&hv.kk, rows, heads, hd, table, false);
        };
        // a stream's residuals: its attention's rows through its projection, then its gated MLP
        let residual = |r: &mut dyn ChainRecorder, st: &Stream, x: &DeviceVec, attended: &DeviceVec, m: &DeviceVec, rows: usize| {
            mul(r, &st.proj, attended, &s.o, rows, false, low);
            r.add_gated_rows(x, &s.o, rows, d, m, 2 * d, false);
            r.layernorm_mod_rows(x, &s.norm, rows, d, m, 4 * d, Some(3 * d), EPS);
            mul(r, &st.mlp_in, &s.norm, &s.fused, rows, false, low);
            r.silu_mul_split_rows(&first(&s.fused, rows * 6 * d), &first(&s.act, rows * 3 * d), rows);
            mul(r, &st.mlp_out, &s.act, &s.o, rows, false, low);
            r.add_gated_rows(x, &s.o, rows, d, m, 5 * d, false);
        };
        for b in &self.double {
            qkv(r, &b.txt, &s.xt, mt, &s.text, &s.tables[0], nt);
            qkv(r, &b.img, &s.xi, mi, &s.image, &s.tables[1], ni);
            // the text's rows then the picture's: the queries, and each position's key and value a row
            r.copy(&s.text.qq, 0, &s.both.qq, 0, nt * d);
            r.copy(&s.image.qq, 0, &s.both.qq, nt * d, ni * d);
            r.store_rows(&s.text.kk, &s.kv, nt, d, 0, 2 * d, 0);
            r.store_rows(&s.text.v, &s.kv, nt, d, 0, 2 * d, d);
            r.store_rows(&s.image.kk, &s.kv, ni, d, nt, 2 * d, 0);
            r.store_rows(&s.image.v, &s.kv, ni, d, nt, 2 * d, d);
            r.attention_rows_full(&s.both.qq, &s.kv, &s.att, np, heads, heads, hd, np, scale);
            r.copy(&s.att, nt * d, &s.ai, 0, ni * d);
            residual(r, &b.img, &s.xi, &s.ai, mi, ni);
            residual(r, &b.txt, &s.xt, &s.att, mt, nt);
        }
        r.copy(&s.xt, 0, &s.all, 0, nt * d);
        r.copy(&s.xi, 0, &s.all, nt * d, ni * d);
        for b in &self.single {
            // one projection: q, k and v, then the MLP's two halves
            r.layernorm_mod_rows(&s.all, &s.norm, np, d, ms, d, Some(0), EPS);
            mul(r, &b.input, &s.norm, &s.wide, np, false, low);
            r.copy_cols(&s.wide, &s.both.q, np, d, 9 * d, 0);
            r.copy_cols(&s.wide, &s.both.k, np, d, 9 * d, d);
            r.copy_cols(&s.wide, &s.both.v, np, d, 9 * d, 2 * d);
            r.copy_cols(&s.wide, &s.fused, np, 6 * d, 9 * d, 3 * d);
            r.rmsnorm_rows(&s.both.q, &b.qn, &s.both.qq, np * heads, EPS);
            r.rmsnorm_rows(&s.both.k, &b.kn, &s.both.kk, np * heads, EPS);
            r.rope_rows(&s.both.qq, np, heads, hd, &s.tables[2], false);
            r.rope_rows(&s.both.kk, np, heads, hd, &s.tables[2], false);
            r.store_rows(&s.both.kk, &s.kv, np, d, 0, 2 * d, 0);
            r.store_rows(&s.both.v, &s.kv, np, d, 0, 2 * d, d);
            r.attention_rows_full(&s.both.qq, &s.kv, &s.att, np, heads, heads, hd, np, scale);
            r.silu_mul_split_rows(&s.fused, &s.act, np);
            // the attention's output beside the MLP's, through the one output projection
            r.store_rows(&s.att, &s.cat, np, d, 0, 4 * d, 0);
            r.store_rows(&s.act, &s.cat, np, 3 * d, 0, 4 * d, d);
            mul(r, &b.output, &s.cat, &s.o, np, false, low);
            r.add_gated_rows(&s.all, &s.o, np, d, ms, 2 * d, false);
        }
        // the picture's tokens out
        r.copy(&s.all, nt * d, &s.xi, 0, ni * d);
        r.layernorm_mod_rows(&s.xi, &s.norm, ni, d, mf, d, Some(0), EPS);
        mul(r, &self.final_proj, &s.norm, &s.vel, ni, false, low);
        r.read(&s.vel);
        let vel = rec.finish().pop().ok_or_else(|| err("the step's velocity was not read"))?;
        self.scratch = Some(s);
        Tensor::from_vec(vel, (1, ni, cfg.channels), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// The chain's table rotates as [`math::rotate`] does: its pairs' sines and cosines are [`math::rotary`]'s.
    #[test]
    fn the_rope_table_is_rotarys_pairs() -> Result<()> {
        let coords = math::ids(3, 2, 2);
        let table = rope_table(&coords, [32; 4], 2000.);
        let (cos, sin) = math::rotary(&coords, [32; 4], 2000., &Device::Cpu)?;
        let (cos, sin) = (cos.flatten_all()?.to_vec1::<f32>()?, sin.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(table.len(), coords.len() * 128);
        for (pair, (c, s)) in table.chunks_exact(2).zip(cos.iter().zip(&sin)) {
            assert_eq!((pair[0], pair[1]), (*s, *c));
        }
        Ok(())
    }

    /// The WebGPU transformer gives the Candle one's velocity (CPU, f32, a GGUF's weights dequantized) on the
    /// published weights (`OAIY_KLEIN_TRANSFORMER`: the BFL-layout checkpoint or a GGUF of it): 16 text tokens' states
    /// (random, as an encoder's are: a few channels large) and an 8 x 8 picture's latent (KLEIN_GRID another side) at
    /// two times. The velocity is a small difference of a large stream (the last single blocks take most of it
    /// away), so what a block's sums differ by counts for more in it than in the stream: 0.9998 at 8 x 8 from the
    /// Q4_K_M file, 0.99995 at 32 x 32.
    #[test]
    #[ignore = "needs FLUX.2 Klein 4B's transformer (OAIY_KLEIN_TRANSFORMER) and a WebGPU adapter"]
    fn the_webgpu_transformer_is_the_candle_one() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_KLEIN_TRANSFORMER").map(PathBuf::from) else { return Ok(()) };
        let number = |key: &str, default: usize| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        let side = number("KLEIN_GRID", 8);
        // (KLEIN_DOUBLE, KLEIN_SINGLE: the first so many blocks of each kind only, to say where a difference is)
        let cfg = Config { double: number("KLEIN_DOUBLE", 5), single: number("KLEIN_SINGLE", 20), ..Config::default() };
        let (nt, h, w) = (16usize, side, side);
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let states: Vec<f32> = (0..nt * cfg.context).map(|i| (next() * if i % cfg.context % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let latent: Vec<f32> = (0..h * w * cfg.channels).map(|_| (next() * 1.7) as f32).collect();
        let context = Tensor::from_vec(states, (1, nt, cfg.context), &Device::Cpu)?;
        let latent = Tensor::from_vec(latent, (1, h * w, cfg.channels), &Device::Cpu)?;
        let t = std::time::Instant::now();
        let mut gpu = WgpuKlein::load_config(&path, 0, &[], cfg, |_| {})?;
        eprintln!("WebGPU transformer loaded in {:.1} s", t.elapsed().as_secs_f64());
        let mut got = Vec::new();
        for time in [0.9, 0.3] {
            let t = std::time::Instant::now();
            got.push(gpu.predict(&latent, &context, time, h, w)?.flatten_all()?.to_vec1::<f32>()?);
            eprintln!("a step at {time}: {:.3} s", t.elapsed().as_secs_f64());
        }
        drop(gpu);
        let t = std::time::Instant::now();
        // (every block resident: the default streams each from the file as it is used; a GGUF's matrices dequantized
        // whole: Candle's quantized matmul rounds the activations to 8 bits, and is 0.9987 by cosine from this itself)
        crate::weights::gguf_dense(true);
        let budget = crate::residency::Budget { memory: crate::residency::Memory::Gpu, ..Default::default() };
        let mut cpu = crate::klein::transformer::Transformer::load_config(&path, &[], &Device::Cpu, DType::F32, &budget, cfg)?;
        eprintln!("Candle transformer loaded in {:.1} s", t.elapsed().as_secs_f64());
        let mut cosines = Vec::new();
        for (i, &time) in [0.9, 0.3].iter().enumerate() {
            let want = cpu.predict(&latent, &context, time, h, w)?.flatten_all()?.to_vec1::<f32>()?;
            let dot: f64 = got[i].iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got[i]) * norm(&want));
            let rms = norm(&want) / (want.len() as f64).sqrt();
            let worst = got[i].iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0.0, f64::max);
            eprintln!("time {time}: cosine {cos:.6}, the worst error {worst:.4} of an RMS {rms:.4}");
            cosines.push(cos);
        }
        assert!(cosines.iter().all(|c| *c > 0.999), "cosines {cosines:?}");
        Ok(())
    }

    /// Each step's time over a few (`--ignored --nocapture`): the transformer of `OAIY_KLEIN_TRANSFORMER`, a
    /// 1024x1024 picture's latent (KLEIN_GRID another side) after 512 text tokens (KLEIN_TEXT another count), 4 steps
    /// (KLEIN_STEPS another count); with OAIY_CHAIN_PROFILE each step's costliest kernels.
    #[test]
    #[ignore = "a timing; needs FLUX.2 Klein 4B's transformer (OAIY_KLEIN_TRANSFORMER) and a WebGPU adapter"]
    fn measure_a_few_steps() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_KLEIN_TRANSFORMER").map(PathBuf::from) else { return Ok(()) };
        let number = |key: &str, default: usize| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        let (side, nt, steps) = (number("KLEIN_GRID", 64), number("KLEIN_TEXT", 512), number("KLEIN_STEPS", 4));
        let cfg = Config::default();
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let states: Vec<f32> = (0..nt * cfg.context).map(|i| (next() * if i % cfg.context % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let latent: Vec<f32> = (0..side * side * cfg.channels).map(|_| next() as f32).collect();
        let context = Tensor::from_vec(states, (1, nt, cfg.context), &Device::Cpu)?;
        let mut latent = Tensor::from_vec(latent, (1, side * side, cfg.channels), &Device::Cpu)?;
        let t = std::time::Instant::now();
        let mut gpu = WgpuKlein::load(&path, 0, &[], |_| {})?;
        eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
        for step in 0..steps {
            let t = std::time::Instant::now();
            let time = 1. - step as f64 / (steps + 1) as f64;
            let v = gpu.predict(&latent, &context, time, side, side)?;
            latent = (latent + (v * -0.2)?)?;
            eprintln!("step {step}: {:.3} s", t.elapsed().as_secs_f64());
            if std::env::var_os("OAIY_CHAIN_PROFILE").is_some() {
                eprintln!("  {:?}", ggml_rs_wgpu::profile::take_kernels().into_iter().take(8).collect::<Vec<_>>());
            }
        }
        Ok(())
    }
}
