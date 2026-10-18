//! FLUX 3 Image Klein's transformer on WebGPU ([`ggml_rs_wgpu`]'s chain of ops), as BFL's `flux3/dit.py` computes it:
//! three kinds of token, the picture's, its references' and the text's, each first through five blocks of its own
//! weights attending only among itself, then all three through 28 shared blocks attending together, each kind still
//! by its own modulation; the picture's tokens out. A block is one stream: a modulated layer norm, then attention (q
//! and k RMS-normed a head at a time and rotated by four axes of position) beside a SwiGLU MLP, both added back by
//! one gate. Every modulation comes from a token's time (the picture's the step's, the rest's zero), plus, in the
//! distilled checkpoint (a flow map), the time the step goes to. No biases anywhere.
//!
//! Its matrices as [`crate::qwen_image::webgpu::transformer`] holds Qwen Image's: a checkpoint's floats f16 (its BF16 as Q8_0 where the
//! card has no room for f16 beside a picture's work), a GGUF's K-quant blocks as they are; LoRA adapters merged into
//! them as they load (BFL's names, ComfyUI's and ai-toolkit's, or diffusers': [`diffusers_name`]).
use crate::common::lora::{Alias, Loras};
use crate::qwen_image::webgpu::transformer::{err, first, matrix_as, mul, timestep, vector, weights_q8, Low, Mat};
use crate::common::weights::Weights;
use candle_core::Result;
use ggml_rs::{ChainRecorder, DeviceChain, DeviceVec};
use std::path::{Path, PathBuf};

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

    /// Every matrix a LoRA may adapt, BFL's name for it and its rows and columns: the inputs, the times' embeddings,
    /// the modulations, each block's attention and MLP, the output.
    pub fn projections(self) -> Vec<(String, usize, usize)> {
        let (h, mlp, c) = (self.hidden, self.mlp, self.channels);
        let mut v: Vec<(String, usize, usize)> = vec![
            ("emb_in.image".into(), h, c),
            ("emb_in.image_cond".into(), h, c),
            ("txt_in".into(), h, self.context),
            ("time_in.in_layer".into(), h, 256),
            ("time_in.out_layer".into(), h, h),
            ("final_layer.image.adaLN_modulation.1".into(), 2 * h, h),
            ("final_layer.image.linear".into(), c, h),
        ];
        if self.flow_map {
            v.extend([("ref_time_in.in_layer".into(), h, 256), ("ref_time_in.out_layer".into(), h, h)]);
        }
        for kind in KINDS {
            v.push((format!("early_stream_modulations.{}.lin", kind.key()), 3 * h, h));
            v.push((format!("single_stream_modulations.{}.lin", kind.key()), 3 * h, h));
        }
        let blocks = ["content_mode_blocks.image", "content_mode_blocks.image_cond", "txt_mode_blocks"]
            .iter()
            .flat_map(|p| (0..self.depth).map(move |i| format!("{p}.{i}")))
            .chain((0..self.single).map(|i| format!("single_blocks.{i}")));
        for b in blocks {
            for (name, out, input) in [("q_proj", h, h), ("k_proj", h, h), ("v_proj", h, h), ("mlp_in", 2 * mlp, h), ("attn_out", h, h), ("mlp_out", h, mlp)] {
                v.push((format!("{b}.{name}"), out, input));
            }
        }
        v
    }
}

/// diffusers' name for a matrix BFL's checkpoint names `module` (`Flux3Transformer2DModel`'s, as its LoRAs name
/// them): the same weights, renamed, but for the final modulation, whose halves (BFL's shift then scale) diffusers
/// keeps the other way round (`norm_out.linear`: scale, then shift).
pub fn diffusers_name(module: &str) -> Option<Alias> {
    let named = |name: String| Some(Alias { name, swap_halves: false });
    let layer = |part: &str| -> Option<&'static str> {
        Some(match part {
            "q_proj" => "attn.to_q",
            "k_proj" => "attn.to_k",
            "v_proj" => "attn.to_v",
            "attn_out" => "attn.to_out.0",
            "mlp_in" => "ff.linear_in",
            "mlp_out" => "ff.linear_out",
            _ => return None,
        })
    };
    // a block's: `{prefix}.{i}.{layer}`
    let block = |prefix: &str, rest: &str| -> Option<String> {
        let (i, part) = rest.split_once('.')?;
        Some(format!("{prefix}.{i}.{}", layer(part)?))
    };
    for kind in ["image_cond", "image"] {
        if let Some(rest) = module.strip_prefix(&format!("content_mode_blocks.{kind}.")) {
            return named(block(&format!("mode_blocks.{kind}"), rest)?);
        }
    }
    if let Some(rest) = module.strip_prefix("txt_mode_blocks.") {
        return named(block("txt_mode_blocks", rest)?);
    }
    if let Some(rest) = module.strip_prefix("single_blocks.") {
        return named(block("single_transformer_blocks", rest)?);
    }
    for kind in ["early", "single"] {
        if let Some(k) = module.strip_prefix(&format!("{kind}_stream_modulations.")).and_then(|r| r.strip_suffix(".lin")) {
            return named(format!("{kind}_stream_modulations.{k}.linear"));
        }
    }
    if module == "final_layer.image.adaLN_modulation.1" {
        return Some(Alias { name: "norm_out.linear".into(), swap_halves: true });
    }
    named(
        match module {
            "txt_in" => "context_embedder",
            "emb_in.image" => "x_embedders.image",
            "emb_in.image_cond" => "x_embedders.image_cond",
            "time_in.in_layer" => "timestep_embedder.linear_1",
            "time_in.out_layer" => "timestep_embedder.linear_2",
            "ref_time_in.in_layer" => "target_timestep_embedder.linear_1",
            "ref_time_in.out_layer" => "target_timestep_embedder.linear_2",
            "final_layer.image.linear" => "proj_out",
            _ => return None,
        }
        .into(),
    )
}

/// `list`'s adapters (each a path and its strength) for the checkpoint `cfg` describes, checked from their headers
/// alone: every module one of its matrices of its shape, and nothing but LoRA factors.
pub fn loras(list: &[(PathBuf, f64)], cfg: Config) -> Result<Loras> {
    let lora = Loras::open(list)?.for_model("FLUX 3 Image Klein", Some(diffusers_name));
    lora.validate_modules(&cfg.projections())?;
    Ok(lora)
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
    lora_notes: Vec<String>,
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
    /// instead), with `loras` (each a path and its strength) merged in. `progress(done, of)` as its blocks load.
    pub fn load(path: &Path, device: usize, loras: &[(PathBuf, f64)], progress: impl FnMut(usize, usize)) -> Result<Self> {
        Self::load_on(path, ggml_rs_wgpu::WgpuBackend::nth(device, None).map_err(err)?, loras, progress)
    }

    /// [`Self::load`] onto `gpu` (the VAE's beside it).
    pub fn load_on(path: &Path, gpu: ggml_rs_wgpu::WgpuBackend, loras: &[(PathBuf, f64)], mut progress: impl FnMut(usize, usize)) -> Result<Self> {
        let mut w = Weights::open(path)?;
        let cfg = Config::of(&w)?;
        let mut lora = self::loras(loras, cfg)?;
        // its blocks' matrices f16 where the tensor cores take f16 and there is room for them beside a picture's work;
        // else its BF16 ones as Q8_0, half the size: on Metal's 8 x 8 simdgroup matrices Q8_0's matmuls are the tensor
        // cores' and f16's are not (an M5 Pro's 512x512 step 2.9 s where 6.9 as f16). OAIY_FLUX3_WEIGHTS `f16` or `q8_0`
        // says which. A LoRA's factors are merged into a matrix as f16 on the GPU, then made Q8_0.
        let block_params = 4 * cfg.hidden * cfg.hidden + 3 * cfg.hidden * cfg.mlp;
        let f16_bytes = 2 * (block_params * (3 * cfg.depth + cfg.single)) as u64;
        let q8 = match std::env::var("OAIY_FLUX3_WEIGHTS").map(|v| v.to_ascii_lowercase()).as_deref() {
            Ok("f16") => false,
            Ok("q8_0" | "q8") => true,
            _ => !gpu.f16_tensor_cores() || weights_q8(&gpu, f16_bytes),
        };
        // (the inputs, times and modulations f16 whatever the blocks are: the text's input reads its states as f32,
        // past f16's range, and a Q8_0 matmul takes its inputs as f16)
        let mut m = |w: &mut Weights, name: &str, q8: bool| matrix_as(w, &mut lora, &gpu, name, q8);
        let block = |w: &mut Weights, m: &mut dyn FnMut(&mut Weights, &str, bool) -> Result<Mat>, p: &str| -> Result<Block> {
            Ok(Block {
                q: m(w, &format!("{p}.q_proj"), q8)?,
                k: m(w, &format!("{p}.k_proj"), q8)?,
                v: m(w, &format!("{p}.v_proj"), q8)?,
                mlp_in: m(w, &format!("{p}.mlp_in"), q8)?,
                attn_out: m(w, &format!("{p}.attn_out"), q8)?,
                mlp_out: m(w, &format!("{p}.mlp_out"), q8)?,
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
            let input = m(&mut w, input, false)?;
            let mut blocks = Vec::with_capacity(cfg.depth);
            for i in 0..cfg.depth {
                blocks.push(block(&mut w, &mut m, &format!("{prefix}.{i}"))?);
                done += 1;
                progress(done, total);
            }
            let early = m(&mut w, &format!("early_stream_modulations.{}.lin", kind.key()), false)?;
            let single = m(&mut w, &format!("single_stream_modulations.{}.lin", kind.key()), false)?;
            modes.push(Mode { input, blocks, early, single });
        }
        let mut single = Vec::with_capacity(cfg.single);
        for i in 0..cfg.single {
            single.push(block(&mut w, &mut m, &format!("single_blocks.{i}"))?);
            done += 1;
            progress(done, total);
        }
        let time_in = [m(&mut w, "time_in.in_layer", false)?, m(&mut w, "time_in.out_layer", false)?];
        let ref_time_in = if cfg.flow_map { Some([m(&mut w, "ref_time_in.in_layer", false)?, m(&mut w, "ref_time_in.out_layer", false)?]) } else { None };
        let (final_mod, final_proj) = (m(&mut w, "final_layer.image.adaLN_modulation.1", false)?, m(&mut w, "final_layer.image.linear", false)?);
        let lora_notes = lora.check()?;
        let modes: [Mode; 3] = modes.try_into().map_err(|_| err("three kinds of token"))?;
        Ok(Self { gpu, cfg, modes, time_in, ref_time_in, single, final_mod, final_proj, lora_notes, scratch: None })
    }

    /// LoRA adapters that fit only in part, and what of them was left out.
    pub fn lora_notes(&self) -> &[String] {
        &self.lora_notes
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
            // (the shared blocks' attention of all the rows, and each kind's of its own in the early blocks: a kind of
            // few rows (a short prompt's text, under 16) takes the runs' kernel, whose scratch can be longer than the
            // output alone that all the rows' takes on the tensor cores)
            att: self.vec(
                sizes.iter().filter(|&&n| n > 0).map(|&n| self.gpu.attention_rows_full_out_len(n, self.cfg.heads, self.cfg.head_dim, n)).fold(self.gpu.attention_rows_full_out_len(all, self.cfg.heads, self.cfg.head_dim, all), usize::max),
            ),
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
        let mut gpu = WgpuFlux3::load(&path, 0, &[], |_, _| {})?;
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
    /// diffusers' names are `Flux3Transformer2DModel`'s, one for each of BFL's matrices, the final modulation's
    /// halves the other way round.
    #[test]
    fn diffusers_names_are_its_transformers() {
        let name = |m: &str| diffusers_name(m).map(|a| (a.name, a.swap_halves));
        assert_eq!(name("content_mode_blocks.image_cond.4.k_proj"), Some(("mode_blocks.image_cond.4.attn.to_k".into(), false)));
        assert_eq!(name("content_mode_blocks.image.0.mlp_in"), Some(("mode_blocks.image.0.ff.linear_in".into(), false)));
        assert_eq!(name("txt_mode_blocks.2.attn_out"), Some(("txt_mode_blocks.2.attn.to_out.0".into(), false)));
        assert_eq!(name("single_blocks.27.mlp_out"), Some(("single_transformer_blocks.27.ff.linear_out".into(), false)));
        assert_eq!(name("early_stream_modulations.image_cond.lin"), Some(("early_stream_modulations.image_cond.linear".into(), false)));
        assert_eq!(name("ref_time_in.out_layer"), Some(("target_timestep_embedder.linear_2".into(), false)));
        assert_eq!(name("final_layer.image.adaLN_modulation.1"), Some(("norm_out.linear".into(), true)));
        assert_eq!(name("single_blocks.0.norm.query_norm"), None);
        let cfg = Config { hidden: 3072, heads: 24, head_dim: 128, depth: 5, single: 28, mlp: 9216, channels: 128, context: 20480, flow_map: true };
        let all = cfg.projections();
        let names: std::collections::BTreeSet<String> = all.iter().map(|(m, ..)| diffusers_name(m).unwrap().name).collect();
        // (diffusers' 359 tensors less its 43 blocks' two norms)
        assert_eq!((all.len(), names.len()), (359 - 2 * 43, 359 - 2 * 43));
    }

    /// Every LoRA in `OAIY_FLUX3_LORAS` (a folder of FLUX 3 Image Klein LoRAs, in any of the layouts) fits the
    /// distilled checkpoint whole: each of its modules one of the checkpoint's matrices, of its shape.
    #[test]
    #[ignore = "needs OAIY_FLUX3_LORAS, a folder of FLUX 3 Image Klein LoRAs"]
    fn real_loras_fit_the_checkpoint() -> Result<()> {
        let Some(dir) = std::env::var_os("OAIY_FLUX3_LORAS").map(PathBuf::from) else { return Ok(()) };
        let cfg = Config { hidden: 3072, heads: 24, head_dim: 128, depth: 5, single: 28, mlp: 9216, channels: 128, context: 20480, flow_map: true };
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "safetensors")).collect();
        files.sort();
        assert!(!files.is_empty());
        for file in files {
            let mut lora = loras(&[(file.clone(), 1.)], cfg)?;
            let mut fitted = 0;
            for (module, out, input) in cfg.projections() {
                fitted += lora.factors(&module, out, input, &Device::Cpu, DType::F32)?.len();
            }
            let notes = lora.check()?;
            eprintln!("{}: {fitted} of the checkpoint's {} matrices", file.display(), cfg.projections().len());
            assert!(fitted > 0 && notes.is_empty(), "{}: {notes:?}", file.display());
        }
        Ok(())
    }

    /// The tiny fixture's step: its picture's latent and its references', the text's states, each one's positions,
    /// and the step's two times.
    struct Step {
        image: Vec<f32>,
        image_ids: Vec<[f64; 4]>,
        cond: Vec<f32>,
        cond_ids: Vec<[f64; 4]>,
        text: Vec<f32>,
        text_ids: Vec<[f64; 4]>,
        times: Vec<f32>,
    }

    impl Step {
        fn of(w: &mut Weights) -> Result<Self> {
            let mut values = |name: &str| -> Result<Vec<f32>> { w.tensor(&format!("fixture.{name}"), &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>() };
            let ids = |v: Vec<f32>| -> Vec<[f64; 4]> { v.chunks_exact(4).map(|c| [c[0] as f64, c[1] as f64, c[2] as f64, c[3] as f64]).collect() };
            Ok(Self {
                image: values("image")?,
                image_ids: ids(values("image_ids")?),
                cond: values("cond")?,
                cond_ids: ids(values("cond_ids")?),
                text: values("text")?,
                text_ids: ids(values("text_ids")?),
                times: values("times")?,
            })
        }

        /// The velocity `model` gives it, its references included.
        fn velocity(&self, model: &mut WgpuFlux3) -> Result<Vec<f32>> {
            let tokens = Tokens { image: &self.image, image_ids: &self.image_ids, cond: &self.cond, cond_ids: &self.cond_ids, text: &self.text, text_ids: &self.text_ids };
            model.predict(&tokens, self.times[0] as f64, Some(self.times[1] as f64))
        }
    }

    /// diffusers' own LoRAs (`tools/flux3/lora_reference.py`): one made with PEFT and saved by diffusers (its names, its
    /// scale in its metadata with the `to_k` projections' own alpha, the final modulation's halves the other way round
    /// from BFL's), one in BFL's names as ai-toolkit saves them. The velocity diffusers' `Flux3Transformer2DModel` gives
    /// the fixture's step with both applied (at 0.7 and 1.3) is this one's with them merged as it loads.
    #[test]
    fn diffusers_loras_are_diffusers() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/flux3");
        let path = dir.join("tiny.safetensors");
        let step = Step::of(&mut Weights::open(&path)?)?;
        let want = Weights::open(&dir.join("lora_expected.safetensors"))?.tensor("velocity", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let loras = [(dir.join("lora_diffusers.safetensors"), 0.7), (dir.join("lora_bfl.safetensors"), 1.3)];
        let mut model = WgpuFlux3::load(&path, 0, &loras, |_, _| {})?;
        assert!(model.lora_notes().is_empty(), "{:?}", model.lora_notes());
        let got = step.velocity(&mut model)?;
        let plain = step.velocity(&mut WgpuFlux3::load(&path, 0, &[], |_, _| {})?)?;
        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() / (norm(&got) * norm(&want));
        let rms = norm(&want) / (want.len() as f64).sqrt();
        let worst = |v: &[f32]| v.iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
        eprintln!("diffusers' LoRAs: cosine {cosine:.7}, the worst error {:.2e} (without them {:.2e}) of an RMS {rms:.3}", worst(&got), worst(&plain));
        assert!(cosine > 0.9999 && worst(&got) < 0.01 * rms && worst(&plain) > 0.5 * rms, "cosine {cosine}");
        Ok(())
    }

    /// LoRAs merged into the transformer as it loads (one in BFL's names as ComfyUI keeps them, with alphas; one in
    /// diffusers', its scale in its metadata and the final modulation's halves the other way round) give what the
    /// same transformer gives with them merged into its weights beforehand, on the host.
    #[test]
    fn loras_merged_as_it_loads_are_the_weights_merged_before() -> Result<()> {
        if ggml_rs_wgpu::WgpuBackend::new(Some(1 << 30)).is_err() {
            return Ok(());
        }
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/flux3/tiny.safetensors");
        let dir = std::env::temp_dir().join(format!("oaiy-flux3-lora-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let mut w = Weights::open(&fixture)?;
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        for name in w.names() {
            let shape = w.shape(&name)?;
            tensors.push((name.clone(), shape, w.tensor(&name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?));
        }
        let mut seed = 0x2545f4914f6cdd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32 * 2. - 1.
        };
        let rank = 4;
        let comfy = ["content_mode_blocks.image.0.q_proj", "single_blocks.1.mlp_in", "txt_in", "early_stream_modulations.txt.lin", "ref_time_in.out_layer"];
        let diffusers = ["content_mode_blocks.image_cond.0.v_proj", "single_blocks.0.mlp_out", "final_layer.image.adaLN_modulation.1", "emb_in.image", "time_in.in_layer"];
        // (each adapter's modules, strength, and alpha: ComfyUI's as tensors, diffusers' in its metadata)
        let adapters = [(&comfy, 0.8f32, 2f32), (&diffusers, 1.2, 8.)];
        let mut files: [Vec<(String, Vec<usize>, Vec<f32>)>; 2] = [Vec::new(), Vec::new()];
        for (which, (modules, strength, alpha)) in adapters.iter().enumerate() {
            for module in modules.iter() {
                let (_, shape, weight) = tensors.iter_mut().find(|(n, ..)| *n == format!("{module}.weight")).unwrap();
                let (out, input) = (shape[0], shape[1]);
                let a: Vec<f32> = (0..rank * input).map(|_| next()).collect();
                let mut b: Vec<f32> = (0..out * rank).map(|_| next()).collect();
                let delta = |b: &[f32]| -> Vec<f32> {
                    (0..out * input).map(|e| (0..rank).map(|r| b[e / input * rank + r] * a[r * input + e % input]).sum::<f32>() * strength * alpha / rank as f32).collect()
                };
                // (the change a quarter of the weight's RMS)
                let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
                let k = 0.25 * rms(weight) / rms(&delta(&b));
                b.iter_mut().for_each(|x| *x *= k);
                weight.iter_mut().zip(delta(&b)).for_each(|(w, d)| *w += d);
                let (name, b) = if which == 0 {
                    (format!("diffusion_model.{module}"), b)
                } else {
                    let alias = diffusers_name(module).unwrap();
                    let b = if alias.swap_halves { [&b[out / 2 * rank..], &b[..out / 2 * rank]].concat() } else { b };
                    (format!("transformer.{}", alias.name), b)
                };
                let (down, up) = if which == 0 { ("lora_down", "lora_up") } else { ("lora_A", "lora_B") };
                files[which].push((format!("{name}.{down}.weight"), vec![rank, input], a));
                files[which].push((format!("{name}.{up}.weight"), vec![out, rank], b));
                if which == 0 {
                    files[which].push((format!("{name}.alpha"), vec![], vec![*alpha]));
                }
            }
        }
        let write = |path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)], meta: &[(&str, &str)]| {
            let list: Vec<(&str, &[usize], &[f32])> = tensors.iter().map(|(n, s, v)| (n.as_str(), s.as_slice(), v.as_slice())).collect();
            crate::common::lora::write_f32(path, &list, meta);
        };
        let (merged, comfy_file, diffusers_file) = (dir.join("merged.safetensors"), dir.join("comfy.safetensors"), dir.join("diffusers.safetensors"));
        write(&merged, &tensors, &[]);
        write(&comfy_file, &files[0], &[]);
        write(&diffusers_file, &files[1], &[("lora_adapter_metadata", r#"{"transformer.r": 4, "transformer.lora_alpha": 8}"#)]);
        let step = Step::of(&mut w)?;
        let run = |path: &Path, loras: &[(PathBuf, f64)]| -> Result<Vec<f32>> { step.velocity(&mut WgpuFlux3::load(path, 0, loras, |_, _| {})?) };
        let before = run(&merged, &[])?;
        let loaded = run(&fixture, &[(comfy_file, 0.8), (diffusers_file, 1.2)])?;
        let plain = run(&fixture, &[])?;
        let rms = (before.iter().map(|a| (*a as f64).powi(2)).sum::<f64>() / before.len() as f64).sqrt();
        let worst = |v: &[f32]| v.iter().zip(&before).map(|(a, b)| (*a as f64 - *b as f64).abs()).fold(0., f64::max);
        eprintln!("merged as it loads: the worst error {:.2e}; without the LoRAs {:.2e}; of an RMS {rms:.3}", worst(&loaded), worst(&plain));
        assert!(worst(&loaded) < 0.01 * rms && worst(&plain) > 0.2 * rms);
        std::fs::remove_dir_all(&dir).ok();
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
        let mut gpu = WgpuFlux3::load(&path, 0, &[], |_, _| {})?;
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
