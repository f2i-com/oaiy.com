//! FLUX 3 Image Klein's transformer on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as BFL's `flux3/dit.py` computes it:
//! three kinds of token, the picture's, its references' and the text's, each first through five blocks of its own
//! weights attending only among itself, then all three through 28 shared blocks attending together, each kind still
//! by its own modulation; the picture's tokens out. A block is one stream: a modulated layer norm, then attention (q
//! and k RMS-normed a head at a time and rotated by four axes of position) beside a SwiGLU MLP, both added back by
//! one gate. Every modulation comes from a token's time (the picture's the step's, the rest's zero), plus, in the
//! distilled checkpoint (a flow map), the time the step goes to. No biases anywhere.
//!
//! Its matrices as [`crate::qwen_wgpu`] holds Qwen Image's: a checkpoint's floats f16 (its BF16 as Q8_0 where the
//! card has no room for f16 beside a picture's work), a GGUF's K-quant blocks as they are.
use crate::qwen_wgpu::{err, first, matrix_as, mul, timestep, vector, weights_q8, Low, Mat};
use crate::{lora::Loras, weights::Weights};
use candle_core::Result;
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::Path;

const EPS: f32 = 1e-6;
const THETA: f64 = 10_000.;
/// What the picture's (and its references') rows and columns are scaled by as positions.
pub const IMAGE_ROPE_SCALE: f64 = 0.5;
/// The tensor that makes a checkpoint a flow map (the distilled one): the embedding of the time a step goes to.
pub const FLOW_MAP_KEY: &str = "ref_time_in.in_layer.weight";

/// The transformer's shape, read from its weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub heads: usize,
    /// A head's width (128): four axes of position, a quarter each.
    pub head_dim: usize,
    /// Each kind's own blocks.
    pub depth: usize,
    /// The shared blocks.
    pub single: usize,
    /// The MLP's width.
    pub mlp: usize,
    /// A latent token's channels.
    pub channels: usize,
    /// A text token's: the encoder's tapped states side by side.
    pub context: usize,
    pub flow_map: bool,
}

impl Config {
    /// The shape of the checkpoint `w` holds, checked as far as its names and shapes say.
    pub fn of(w: &Weights) -> Result<Self> {
        let shape = |name: &str| w.shape(name);
        let txt = shape("txt_in.weight")?;
        let img = shape("emb_in.image.weight")?;
        let (&[hidden, context], &[h2, channels]) = (txt.as_slice(), img.as_slice()) else {
            candle_core::bail!("not FLUX 3 Image Klein's transformer: txt_in {txt:?}, emb_in.image {img:?}");
        };
        let head_dim = shape("single_blocks.0.norm.query_norm.scale")?.iter().product::<usize>();
        if h2 != hidden || head_dim % 8 != 0 || hidden % head_dim != 0 || channels != 128 {
            candle_core::bail!("not FLUX 3 Image Klein's transformer: hidden {hidden}, heads {head_dim} wide, latent channels {channels}");
        }
        let count = |p: &str| (0..).take_while(|i| w.has(&format!("{p}.{i}.q_proj.weight"))).count();
        let (depth, single) = (count("content_mode_blocks.image"), count("single_blocks"));
        let mlp_in = shape("single_blocks.0.mlp_in.weight")?;
        let mlp = mlp_in.first().copied().unwrap_or(0) / 2;
        if depth == 0 || single == 0 || mlp == 0 || count("content_mode_blocks.image_cond") != depth || count("txt_mode_blocks") != depth {
            candle_core::bail!("not FLUX 3 Image Klein's transformer: {depth} blocks a kind, {single} shared");
        }
        Ok(Self { hidden, heads: hidden / head_dim, head_dim, depth, single, mlp, channels, context, flow_map: w.has(FLOW_MAP_KEY) })
    }
}

/// The kinds of token, in the order a sample's take in the shared blocks.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Image,
    Cond,
    Text,
}

const KINDS: [Kind; 3] = [Kind::Image, Kind::Cond, Kind::Text];

impl Kind {
    /// Its name in the checkpoint's modulations (and blocks: the text's are `txt_mode_blocks`).
    fn key(self) -> &'static str {
        match self {
            Kind::Image => "image",
            Kind::Cond => "image_cond",
            Kind::Text => "txt",
        }
    }
}

struct Block {
    q: Mat,
    k: Mat,
    v: Mat,
    /// The MLP's gate and up, side by side.
    mlp_in: Mat,
    attn_out: Mat,
    mlp_out: Mat,
    qn: DeviceVec,
    kn: DeviceVec,
}

/// A kind's own weights: its input projection, its five blocks, its two modulations (each block's shift, scale and
/// gate from its time's vector).
struct Mode {
    input: Mat,
    blocks: Vec<Block>,
    early: Mat,
    single: Mat,
}

/// A kind's vectors in a step: its tokens' states, each position's RoPE table, its modulations.
struct Rows {
    n: usize,
    /// Its tokens as they come in (latent or text), then their states.
    input: DeviceVec,
    x: DeviceVec,
    table: DeviceVec,
    vec: DeviceVec,
    act_vec: DeviceVec,
    early: DeviceVec,
    single: DeviceVec,
    /// Its MLP's activations, kept from the norm to the residual.
    act: DeviceVec,
}

/// A step's vectors, kept for the next step of the same sizes.
struct Scratch {
    sizes: [usize; 3],
    rows: [Rows; 3],
    /// The time's sinusoids: the step's, zero's, the target's.
    t: [DeviceVec; 3],
    /// Their embeddings, and the hidden activation in between.
    tv: [DeviceVec; 3],
    hidden: DeviceVec,
    final_mod: DeviceVec,
    norm: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qq: DeviceVec,
    kk: DeviceVec,
    /// Every attending token's normed and rotated query, and its key and value a row.
    all_q: DeviceVec,
    kv: DeviceVec,
    att: DeviceVec,
    /// A kind's rows of the shared attention's output.
    part: DeviceVec,
    o: DeviceVec,
    fused: DeviceVec,
    vel: DeviceVec,
    low: Low,
}

pub struct WgpuFlux3 {
    gpu: ggml_rs_wgpu::WgpuBackend,
    cfg: Config,
    modes: [Mode; 3],
    time_in: [Mat; 2],
    /// The flow map's embedding of the time a step goes to.
    ref_time_in: Option<[Mat; 2]>,
    single: Vec<Block>,
    final_mod: Mat,
    final_proj: Mat,
    scratch: Option<Scratch>,
}

/// A step's tokens: the picture's latent (`[n, 128]` and each token's four position coordinates), its references'
/// (empty without), and the text's encoder states (`[n, context]`).
pub struct Tokens<'a> {
    pub image: &'a [f32],
    pub image_ids: &'a [[f64; 4]],
    pub cond: &'a [f32],
    pub cond_ids: &'a [[f64; 4]],
    pub text: &'a [f32],
    pub text_ids: &'a [[f64; 4]],
}

/// Each position's RoPE pairs as the chain's table has them (a pair's sine, then its cosine): four axes a quarter of
/// a head (`head_dim`) each, each coordinate over 10,000 to the pair's share of its axis, as `dit.rope` makes the
/// angles.
pub fn rope_table(ids: &[[f64; 4]], head_dim: usize) -> Vec<f32> {
    let width = head_dim / 4;
    let mut t = Vec::with_capacity(ids.len() * head_dim);
    for id in ids {
        for coord in id {
            for i in 0..width / 2 {
                let angle = coord / THETA.powf((2 * i) as f64 / width as f64);
                t.push(angle.sin() as f32);
                t.push(angle.cos() as f32);
            }
        }
    }
    t
}

/// `t` as the reference computes with it: a time is held as bfloat16 (rounded to the nearest, ties to even) before
/// its sinusoids are made.
pub fn bf16_time(t: f64) -> f64 {
    let bits = (t as f32).to_bits();
    let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded) as f64
}

impl WgpuFlux3 {
    /// The transformer at `path` on GPU `device` (as the computer counts them; OAIY_WEBGPU_ADAPTER naming one
    /// instead). `progress(done, of)` as its blocks load.
    pub fn load(path: &Path, device: usize, progress: impl FnMut(usize, usize)) -> Result<Self> {
        Self::load_on(path, ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?, progress)
    }

    /// [`Self::load`] onto `gpu` (the VAE's beside it).
    pub fn load_on(path: &Path, gpu: ggml_rs_wgpu::WgpuBackend, mut progress: impl FnMut(usize, usize)) -> Result<Self> {
        let mut w = Weights::open(path)?;
        let cfg = Config::of(&w)?;
        let mut lora = Loras::open(&[])?;
        // its blocks' matrices f16 where the tensor cores take f16 and there is room for them beside a picture's work;
        // else its BF16 ones as Q8_0, half the size: on Metal's 8 x 8 simdgroup matrices Q8_0's matmuls are the tensor
        // cores' and f16's are not (an M5 Pro's 512x512 step 2.9 s where 6.9 as f16). OAIY_FLUX3_WEIGHTS `f16` or `q8_0`
        // says which.
        let block_params = 4 * cfg.hidden * cfg.hidden + 3 * cfg.hidden * cfg.mlp;
        let f16_bytes = 2 * (block_params * (3 * cfg.depth + cfg.single)) as u64;
        let q8 = match std::env::var("OAIY_FLUX3_WEIGHTS").map(|v| v.to_ascii_lowercase()).as_deref() {
            Ok("f16") => false,
            Ok("q8_0" | "q8") => true,
            _ => !gpu.f16_tensor_cores() || weights_q8(&gpu, f16_bytes),
        };
        let mut m = |w: &mut Weights, name: &str| matrix_as(w, &mut lora, &gpu, name, q8);
        // (the inputs, times and modulations f16 whatever the blocks are: the text's input reads its states as f32,
        // past f16's range, and a Q8_0 matmul takes its inputs as f16)
        let mut f16 = Loras::open(&[])?;
        let mut exact = |w: &mut Weights, name: &str| matrix_as(w, &mut f16, &gpu, name, false);
        let block = |w: &mut Weights, m: &mut dyn FnMut(&mut Weights, &str) -> Result<Mat>, p: &str| -> Result<Block> {
            Ok(Block {
                q: m(w, &format!("{p}.q_proj"))?,
                k: m(w, &format!("{p}.k_proj"))?,
                v: m(w, &format!("{p}.v_proj"))?,
                mlp_in: m(w, &format!("{p}.mlp_in"))?,
                attn_out: m(w, &format!("{p}.attn_out"))?,
                mlp_out: m(w, &format!("{p}.mlp_out"))?,
                qn: vector(w, &gpu, &format!("{p}.norm.query_norm.scale"), 0.)?,
                kn: vector(w, &gpu, &format!("{p}.norm.key_norm.scale"), 0.)?,
            })
        };
        let total = 3 * cfg.depth + cfg.single;
        let mut done = 0;
        let mut modes = Vec::with_capacity(3);
        for kind in KINDS {
            let (input, prefix) = match kind {
                Kind::Image => ("emb_in.image", "content_mode_blocks.image"),
                Kind::Cond => ("emb_in.image_cond", "content_mode_blocks.image_cond"),
                Kind::Text => ("txt_in", "txt_mode_blocks"),
            };
            let input = exact(&mut w, input)?;
            let mut blocks = Vec::with_capacity(cfg.depth);
            for i in 0..cfg.depth {
                blocks.push(block(&mut w, &mut m, &format!("{prefix}.{i}"))?);
                done += 1;
                progress(done, total);
            }
            let early = exact(&mut w, &format!("early_stream_modulations.{}.lin", kind.key()))?;
            let single = exact(&mut w, &format!("single_stream_modulations.{}.lin", kind.key()))?;
            modes.push(Mode { input, blocks, early, single });
        }
        let mut single = Vec::with_capacity(cfg.single);
        for i in 0..cfg.single {
            single.push(block(&mut w, &mut m, &format!("single_blocks.{i}"))?);
            done += 1;
            progress(done, total);
        }
        let time_in = [exact(&mut w, "time_in.in_layer")?, exact(&mut w, "time_in.out_layer")?];
        let ref_time_in = if cfg.flow_map { Some([exact(&mut w, "ref_time_in.in_layer")?, exact(&mut w, "ref_time_in.out_layer")?]) } else { None };
        let (final_mod, final_proj) = (exact(&mut w, "final_layer.image.adaLN_modulation.1")?, exact(&mut w, "final_layer.image.linear")?);
        let modes: [Mode; 3] = modes.try_into().map_err(|_| err("three kinds of token"))?;
        Ok(Self { gpu, cfg, modes, time_in, ref_time_in, single, final_mod, final_proj, scratch: None })
    }

    pub fn config(&self) -> Config {
        self.cfg
    }

    /// The GPU it is on (the VAE beside it).
    pub fn backend(&self) -> &ggml_rs_wgpu::WgpuBackend {
        &self.gpu
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

    fn scratch(&mut self, sizes: [usize; 3]) -> Scratch {
        if let Some(s) = self.scratch.take().filter(|s| s.sizes == sizes) {
            return s;
        }
        let Config { hidden: d, mlp, channels, context, .. } = self.cfg;
        let (most, all) = (sizes.iter().copied().max().unwrap_or(1), sizes.iter().sum::<usize>());
        let rows = |n: usize, width: usize| Rows {
            n,
            input: self.vec(n * width),
            x: self.vec(n * d),
            table: self.vec(n * self.cfg.head_dim),
            vec: self.vec(d),
            act_vec: self.vec(d),
            early: self.vec(3 * d),
            single: self.vec(3 * d),
            act: self.vec(n * mlp),
        };
        Scratch {
            sizes,
            rows: [rows(sizes[0], channels), rows(sizes[1], channels), rows(sizes[2], context)],
            t: [self.vec(256), self.vec(256), self.vec(256)],
            tv: [self.vec(d), self.vec(d), self.vec(d)],
            hidden: self.vec(d),
            final_mod: self.vec(2 * d),
            norm: self.vec(most * d),
            q: self.vec(most * d),
            k: self.vec(most * d),
            v: self.vec(most * d),
            qq: self.vec(most * d),
            kk: self.vec(most * d),
            all_q: self.vec(all * d),
            kv: self.vec(all * 2 * d),
            att: self.vec(self.gpu.attention_rows_full_out_len(all, self.cfg.heads, self.cfg.head_dim, all)),
            part: self.vec(most * d),
            o: self.vec(most * d),
            fused: self.vec(most * 2 * mlp),
            vel: self.vec(sizes[0] * channels),
            low: Low { t: self.vec(1), y: self.vec(1) },
        }
    }

    /// The velocity of the picture's tokens at time `t` (`[n, 128]` on the CPU, as `DiT.forward` returns it), the flow
    /// map's going to `target` (None for the base checkpoint, which takes none).
    pub fn predict(&mut self, tokens: &Tokens, t: f64, target: Option<f64>) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let (d, heads, hd, ch) = (cfg.hidden, cfg.heads, cfg.head_dim, cfg.channels);
        if target.is_some() != cfg.flow_map {
            candle_core::bail!("a target time is required by, and only by, the distilled (flow-map) checkpoint");
        }
        let sizes = [tokens.image_ids.len(), tokens.cond_ids.len(), tokens.text_ids.len()];
        if sizes[0] == 0 || sizes[2] == 0 || tokens.image.len() != sizes[0] * ch || tokens.cond.len() != sizes[1] * ch || tokens.text.len() != sizes[2] * cfg.context {
            candle_core::bail!("invalid FLUX 3 input shapes: {sizes:?} tokens for {}, {} and {} values", tokens.image.len(), tokens.cond.len(), tokens.text.len());
        }
        let all: usize = sizes.iter().sum();
        let s = self.scratch(sizes);
        for (rows, (values, ids)) in s.rows.iter().zip([(tokens.image, tokens.image_ids), (tokens.cond, tokens.cond_ids), (tokens.text, tokens.text_ids)]) {
            if rows.n > 0 {
                self.gpu.upload(&rows.input, values);
                self.gpu.upload(&rows.table, &rope_table(ids, hd));
            }
        }
        self.gpu.upload(&s.t[0], &timestep(bf16_time(t)));
        self.gpu.upload(&s.t[1], &timestep(0.));
        if let Some(target) = target {
            self.gpu.upload(&s.t[2], &timestep(bf16_time(target)));
        }
        let mut rec = self.gpu.begin();
        // (bind groups kept would hold each step's own scratch)
        rec.keep_groups(false);
        let r = rec.as_mut();
        let low = &s.low;
        // the times' vectors: MLPEmbedder's out(silu(in(x)))
        let embed = |r: &mut dyn ChainRecorder, mats: &[Mat; 2], t: &DeviceVec, out: &DeviceVec| {
            mul(r, &mats[0], t, &s.hidden, 1, true, low);
            r.mul_sigmoid(&s.hidden, &s.hidden, &s.norm, d);
            mul(r, &mats[1], &first(&s.norm, d), out, 1, true, low);
        };
        embed(r, &self.time_in, &s.t[0], &s.tv[0]);
        embed(r, &self.time_in, &s.t[1], &s.tv[1]);
        if let Some(mats) = &self.ref_time_in {
            embed(r, mats, &s.t[2], &s.tv[2]);
        }
        // each kind's vector (the picture's its step's time, the rest's zero; plus the target's), its SiLU, and its
        // modulations from that
        for (kind, (rows, mode)) in KINDS.iter().zip(s.rows.iter().zip(&self.modes)) {
            if rows.n == 0 {
                continue;
            }
            r.copy(&s.tv[if *kind == Kind::Image { 0 } else { 1 }], 0, &rows.vec, 0, d);
            if self.ref_time_in.is_some() {
                r.add(&rows.vec, &s.tv[2]);
            }
            r.mul_sigmoid(&rows.vec, &rows.vec, &rows.act_vec, d);
            mul(r, &mode.early, &rows.act_vec, &rows.early, 1, true, low);
            mul(r, &mode.single, &rows.act_vec, &rows.single, 1, true, low);
            // (the text's states reach some thousands, past f16's range: read as they are)
            mul(r, &mode.input, &rows.input, &rows.x, rows.n, *kind == Kind::Text, low);
        }
        mul(r, &self.final_mod, &s.rows[0].act_vec, &s.final_mod, 1, true, low);
        let scale = 1.0 / (hd as f32).sqrt();
        // a block's start for `n` rows: the modulated norm, q and k normed a head at a time and rotated, the MLP's
        // activations kept for after the attention
        let start = |r: &mut dyn ChainRecorder, b: &Block, rows: &Rows, mods: &DeviceVec| {
            let n = rows.n;
            r.layernorm_mod_rows(&rows.x, &s.norm, n, d, mods, d, Some(0), EPS);
            mul(r, &b.q, &s.norm, &s.q, n, false, low);
            mul(r, &b.k, &s.norm, &s.k, n, false, low);
            mul(r, &b.v, &s.norm, &s.v, n, false, low);
            // (the scratch is the longest kind's: a norm's rows are as wide as what it is given)
            r.rmsnorm_rows(&first(&s.q, n * d), &b.qn, &s.qq, n * heads, EPS);
            r.rmsnorm_rows(&first(&s.k, n * d), &b.kn, &s.kk, n * heads, EPS);
            r.rope_rows(&s.qq, n, heads, hd, &rows.table, false);
            r.rope_rows(&s.kk, n, heads, hd, &rows.table, false);
            mul(r, &b.mlp_in, &s.norm, &s.fused, n, false, low);
            r.silu_mul_split_rows(&first(&s.fused, n * 2 * cfg.mlp), &first(&rows.act, n * cfg.mlp), n);
        };
        // a block's end: the attention's rows (`attended`) and the MLP's through their projections, each added by the
        // one gate
        let end = |r: &mut dyn ChainRecorder, b: &Block, rows: &Rows, attended: &DeviceVec, mods: &DeviceVec| {
            let n = rows.n;
            mul(r, &b.attn_out, attended, &s.o, n, false, low);
            r.add_gated_rows(&rows.x, &s.o, n, d, mods, 2 * d, false);
            mul(r, &b.mlp_out, &rows.act, &s.o, n, false, low);
            r.add_gated_rows(&rows.x, &s.o, n, d, mods, 2 * d, false);
        };
        // each kind through its own blocks, attending among itself
        for i in 0..cfg.depth {
            for (rows, mode) in s.rows.iter().zip(&self.modes) {
                if rows.n == 0 {
                    continue;
                }
                let (b, n) = (&mode.blocks[i], rows.n);
                start(r, b, rows, &rows.early);
                r.store_rows(&s.kk, &s.kv, n, d, 0, 2 * d, 0);
                r.store_rows(&s.v, &s.kv, n, d, 0, 2 * d, d);
                r.attention_rows_full(&s.qq, &s.kv, &s.att, n, heads, heads, hd, n, scale);
                end(r, b, rows, &s.att, &rows.early);
            }
        }
        // then all of them through the shared blocks, attending together: the picture's, its references', the text's
        for b in &self.single {
            let mut at = 0;
            for rows in &s.rows {
                if rows.n == 0 {
                    continue;
                }
                start(r, b, rows, &rows.single);
                r.copy(&s.qq, 0, &s.all_q, at * d, rows.n * d);
                r.store_rows(&s.kk, &s.kv, rows.n, d, at, 2 * d, 0);
                r.store_rows(&s.v, &s.kv, rows.n, d, at, 2 * d, d);
                at += rows.n;
            }
            r.attention_rows_full(&s.all_q, &s.kv, &s.att, all, heads, heads, hd, all, scale);
            let mut at = 0;
            for rows in &s.rows {
                if rows.n == 0 {
                    continue;
                }
                r.copy(&s.att, at * d, &s.part, 0, rows.n * d);
                end(r, b, rows, &s.part, &rows.single);
                at += rows.n;
            }
        }
        // the picture's tokens out
        let image = &s.rows[0];
        r.layernorm_mod_rows(&image.x, &s.norm, image.n, d, &s.final_mod, d, Some(0), EPS);
        mul(r, &self.final_proj, &s.norm, &s.vel, image.n, false, low);
        r.read(&s.vel);
        let vel = rec.finish().pop().ok_or_else(|| err("the step's velocity was not read"))?;
        self.scratch = Some(s);
        Ok(vel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// A time is held as bfloat16, ties to even, as the reference's packed times are.
    #[test]
    fn times_round_as_bfloat16() {
        assert_eq!(bf16_time(0.5), 0.5);
        assert_eq!(bf16_time(0.7), 0.69921875);
        assert_eq!(bf16_time(1.0), 1.0);
        assert_eq!(bf16_time(0.3), 0.30078125);
        // (just past halfway between 0.00390625 and the next, 0.0039367676)
        assert_eq!(bf16_time(1. / 255.), 0.003936767578125);
    }

    /// The table's pairs are `dit.rope`'s: an axis a quarter of the head, the pair's angle its coordinate over
    /// 10,000 to twice the pair's index over the axis's width.
    #[test]
    fn the_rope_table_is_dits_angles() {
        let table = rope_table(&[[0., 1.5, 2., -3.]], 16);
        assert_eq!(table.len(), 16);
        let pair = |axis: usize, i: usize| (table[(axis * 2 + i) * 2], table[(axis * 2 + i) * 2 + 1]);
        assert_eq!(pair(0, 0), (0., 1.));
        let angle = 1.5f64 / 10_000f64.powf(0.5);
        assert_eq!(pair(1, 1), (angle.sin() as f32, angle.cos() as f32));
        assert_eq!(pair(3, 0), ((-3f64).sin() as f32, (-3f64).cos() as f32));
    }

    /// BFL's own DiT (`tools/flux3/reference.py`: its sizes made small, its weights random and f16, run in f32 on
    /// the CPU) and this one on WebGPU give a step's velocity alike: a picture with two references and without.
    #[test]
    fn a_tiny_transformer_on_webgpu_is_bfls() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/flux3/tiny.safetensors");
        let mut w = Weights::open(&path)?;
        let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(&format!("fixture.{name}"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
        let ids = |v: Vec<f32>| -> Vec<[f64; 4]> { v.chunks_exact(4).map(|c| [c[0] as f64, c[1] as f64, c[2] as f64, c[3] as f64]).collect() };
        let (image, image_ids, cond, cond_ids) = (values("image")?, ids(values("image_ids")?), values("cond")?, ids(values("cond_ids")?));
        let (text, text_ids, times) = (values("text")?, ids(values("text_ids")?), values("times")?);
        let wants = [values("velocity")?, values("velocity_without_references")?];
        let mut gpu = WgpuFlux3::load(&path, 0, |_, _| {})?;
        assert_eq!(gpu.config(), Config { hidden: 32, heads: 2, head_dim: 16, depth: 1, single: 2, mlp: 48, channels: 128, context: 48, flow_map: true });
        for (with, want) in [true, false].into_iter().zip(&wants) {
            let tokens = Tokens {
                image: &image,
                image_ids: &image_ids,
                cond: if with { &cond } else { &[] },
                cond_ids: if with { &cond_ids } else { &[] },
                text: &text,
                text_ids: &text_ids,
            };
            let got = gpu.predict(&tokens, times[0] as f64, Some(times[1] as f64))?;
            assert_eq!(got.len(), want.len());
            let dot: f64 = got.iter().zip(want).map(|(a, b)| *a as f64 * *b as f64).sum();
            let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cosine = dot / (norm(&got) * norm(want));
            let rms = norm(want) / (want.len() as f64).sqrt();
            let worst = got.iter().zip(want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
            eprintln!("references {with}: cosine {cosine:.7}, the worst error {worst:.2e} of an RMS {rms:.3}");
            // (the references move it by a quarter of its RMS)
            assert!(cosine > 0.9999 && worst < 0.01 * rms, "references {with}: cosine {cosine}, worst {worst} of {rms}");
        }
        Ok(())
    }
    /// Each step's time over a few (`--ignored --nocapture`): the transformer of `OAIY_FLUX3_TRANSFORMER`, a 1024x1024
    /// picture's latent (FLUX3_GRID another side) after 512 text tokens (FLUX3_TEXT another count) and FLUX3_REFS
    /// references of its own size, 3 steps (FLUX3_STEPS another count); with OAIY_CHAIN_PROFILE each step's costliest
    /// kernels.
    #[test]
    #[ignore = "a timing; needs FLUX 3 Image Klein's transformer (OAIY_FLUX3_TRANSFORMER) and a WebGPU adapter"]
    fn measure_a_few_steps() -> Result<()> {
        let Some(path) = std::env::var_os("OAIY_FLUX3_TRANSFORMER").map(std::path::PathBuf::from) else { return Ok(()) };
        let number = |key: &str, default: usize| std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
        let (side, nt, refs, steps) = (number("FLUX3_GRID", 64), number("FLUX3_TEXT", 512), number("FLUX3_REFS", 0), number("FLUX3_STEPS", 3));
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2. - 1.
        };
        let t = std::time::Instant::now();
        let mut gpu = WgpuFlux3::load(&path, 0, |_, _| {})?;
        eprintln!("loaded in {:.1} s", t.elapsed().as_secs_f64());
        let cfg = gpu.config();
        let text: Vec<f32> = (0..nt * cfg.context).map(|i| (next() * if i % 997 == 5 { 300. } else { 3. }) as f32).collect();
        let mut image: Vec<f32> = (0..side * side * 128).map(|_| next() as f32).collect();
        let cond: Vec<f32> = (0..refs * side * side * 128).map(|_| next() as f32).collect();
        let ids = |l: f64| -> Vec<[f64; 4]> { (0..side * side).map(|i| [0., (i / side) as f64 * 0.5, (i % side) as f64 * 0.5, l]).collect() };
        let image_ids = ids(0.);
        let cond_ids: Vec<[f64; 4]> = (0..refs).flat_map(|r| ids(-(1. + r as f64))).collect();
        let text_ids: Vec<[f64; 4]> = (0..nt).map(|p| [0., 0., 0., p as f64]).collect();
        for step in 0..steps {
            let t = std::time::Instant::now();
            let time = 1. - step as f64 / (steps + 1) as f64;
            let tokens = Tokens { image: &image, image_ids: &image_ids, cond: &cond, cond_ids: &cond_ids, text: &text, text_ids: &text_ids };
            let v = gpu.predict(&tokens, time, cfg.flow_map.then_some(time - 0.05))?;
            for (x, v) in image.iter_mut().zip(&v) {
                *x -= 0.05 * v;
            }
            eprintln!("step {step}: {:.3} s", t.elapsed().as_secs_f64());
            if std::env::var_os("OAIY_CHAIN_PROFILE").is_some() {
                eprintln!("  {:?}", ggml_rs_wgpu::profile::take_kernels().into_iter().take(10).collect::<Vec<_>>());
            }
        }
        Ok(())
    }
}
