//! MOSS-SoundEffect v2.0's DiT: Wan 2.1's text-to-video transformer made 1-D for
//! audio latents (DAC, 128 channels). Each block: adaptive-LayerNorm self-attention
//! (RMS-normed q/k over the whole width, interleaved RoPE), cross-attention to the
//! text, and a GELU MLP, all gated by the timestep. As the reference runs it
//! (bf16 autocast): the residual stream, norms and timestep embedding in F32,
//! the matmuls and attention in BF16.
use crate::ltx::store::Store;
use crate::tts::model::Linear;
use candle_core::{DType, Device, Result, Tensor, D};
use oaiy_engine::json::Json;
use std::path::Path;

pub struct Config {
    pub dim: usize,
    pub ffn_dim: usize,
    pub heads: usize,
    pub layers: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub eps: f64,
}

impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        let j = Json::parse(&std::fs::read(path)?).map_err(candle_core::Error::wrap)?;
        let n = |k: &str| j.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("{}: no {k}", path.display())));
        let patch = j.get("patch_size").and_then(Json::as_array).map(|p| p.iter().filter_map(Json::as_i64).collect::<Vec<_>>()).unwrap_or_default();
        if patch != [1] {
            candle_core::bail!("{}: only patch_size [1] is supported", path.display());
        }
        if j.get("has_image_input").and_then(Json::as_bool) == Some(true) {
            candle_core::bail!("{}: image input is not supported", path.display());
        }
        Ok(Self {
            dim: n("dim")?,
            ffn_dim: n("ffn_dim")?,
            heads: n("num_heads")?,
            layers: n("num_layers")?,
            in_dim: n("in_dim")?,
            out_dim: n("out_dim")?,
            text_dim: n("text_dim")?,
            freq_dim: n("freq_dim")?,
            eps: j.get("eps").and_then(Json::as_f64).unwrap_or(1e-6),
        })
    }
}

/// A linear layer kept in F32 (the timestep path runs outside autocast).
struct Linear32 {
    w: Tensor,
    b: Tensor,
}
impl Linear32 {
    fn load(store: &mut Store, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self { w: store.tensor_f32(&format!("{prefix}.weight"), dev)?, b: store.tensor_f32(&format!("{prefix}.bias"), dev)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&self.w.t()?)?.broadcast_add(&self.b)
    }
}

struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Tensor,
    norm_k: Tensor,
    cq: Linear,
    ck: Linear,
    cv: Linear,
    co: Linear,
    cnorm_q: Tensor,
    cnorm_k: Tensor,
    /// The cross-attention's input LayerNorm (affine).
    norm3_w: Tensor,
    norm3_b: Tensor,
    ffn1: Linear,
    ffn2: Linear,
    /// (6, dim), F32: shift, scale and gate for attention, then for the MLP.
    modulation: Tensor,
}

/// One prompt's text, ready for every block: its keys and values, (1, S, H, D).
pub struct Context {
    kv: Vec<(Tensor, Tensor)>,
}

pub struct Dit {
    pub cfg: Config,
    patch_w: Tensor,
    patch_b: Tensor,
    text1: Linear,
    text2: Linear,
    time1: Linear32,
    time2: Linear32,
    time_proj: Linear32,
    blocks: Vec<Block>,
    /// (2, dim), F32: the head's shift and scale.
    head_mod: Tensor,
    head: Linear,
}

/// RMS norm over the last axis in F32 (the reference's `nn.RMSNorm` under autocast).
fn rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?.broadcast_mul(w)
}

/// LayerNorm over the last axis in F32, optionally affine.
fn layer_norm(x: &Tensor, eps: f64, affine: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
    let f = x.to_dtype(DType::F32)?;
    let f = f.broadcast_sub(&f.mean_keepdim(D::Minus1)?)?;
    let n = f.broadcast_div(&(f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?)?;
    match affine {
        Some((w, b)) => n.broadcast_mul(w)?.broadcast_add(b),
        None => Ok(n),
    }
}

/// Full attention over (B, S, H, D) in BF16.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let hd = q.dim(3)?;
    let t = |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous() };
    let (qt, kt, vt) = (t(q)?, t(k)?, t(v)?);
    let scores = (qt.matmul(&kt.t()?)? / (hd as f64).sqrt())?;
    candle_nn::ops::softmax_last_dim(&scores)?.matmul(&vt)?.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()
}

impl Dit {
    /// `dir`: the pipeline's `transformer` folder (config.json, diffusion_pytorch_model.safetensors).
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let cfg = Config::read(&dir.join("config.json"))?;
        let mut store = Store::open(&dir.join("diffusion_pytorch_model.safetensors"), 0)?;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("blocks.{i}");
            let lin = |store: &mut Store, name: &str| Linear::load(store, &format!("{p}.{name}"), dev);
            blocks.push(Block {
                q: lin(&mut store, "attn1.to_q")?,
                k: lin(&mut store, "attn1.to_k")?,
                v: lin(&mut store, "attn1.to_v")?,
                o: lin(&mut store, "attn1.to_out.0")?,
                norm_q: store.tensor_f32(&format!("{p}.attn1.norm_q.weight"), dev)?,
                norm_k: store.tensor_f32(&format!("{p}.attn1.norm_k.weight"), dev)?,
                cq: lin(&mut store, "attn2.to_q")?,
                ck: lin(&mut store, "attn2.to_k")?,
                cv: lin(&mut store, "attn2.to_v")?,
                co: lin(&mut store, "attn2.to_out.0")?,
                cnorm_q: store.tensor_f32(&format!("{p}.attn2.norm_q.weight"), dev)?,
                cnorm_k: store.tensor_f32(&format!("{p}.attn2.norm_k.weight"), dev)?,
                norm3_w: store.tensor_f32(&format!("{p}.norm2.weight"), dev)?,
                norm3_b: store.tensor_f32(&format!("{p}.norm2.bias"), dev)?,
                ffn1: lin(&mut store, "ffn.net.0.proj")?,
                ffn2: lin(&mut store, "ffn.net.2")?,
                modulation: store.tensor_f32(&format!("{p}.scale_shift_table"), dev)?.reshape((6, cfg.dim))?,
            });
        }
        let patch = store.tensor("patch_embedding.weight", dev, false)?;
        Ok(Self {
            patch_w: patch.reshape((cfg.dim, cfg.in_dim))?,
            patch_b: store.tensor("patch_embedding.bias", dev, false)?,
            text1: Linear::load(&mut store, "condition_embedder.text_embedder.linear_1", dev)?,
            text2: Linear::load(&mut store, "condition_embedder.text_embedder.linear_2", dev)?,
            time1: Linear32::load(&mut store, "condition_embedder.time_embedder.linear_1", dev)?,
            time2: Linear32::load(&mut store, "condition_embedder.time_embedder.linear_2", dev)?,
            time_proj: Linear32::load(&mut store, "condition_embedder.time_proj", dev)?,
            head_mod: store.tensor_f32("scale_shift_table", dev)?.reshape((2, cfg.dim))?,
            head: Linear::load(&mut store, "proj_out", dev)?,
            blocks,
            cfg,
        })
    }

    /// The text encoder's states for one prompt, (1, S, text_dim), as each block's
    /// cross-attention keys and values.
    pub fn context(&self, text: &Tensor) -> Result<Context> {
        let (h, hd) = (self.cfg.heads, self.cfg.dim / self.cfg.heads);
        let c = self.text2.forward(&self.text1.forward(&text.to_dtype(DType::BF16)?)?.gelu()?)?;
        let s = c.dim(1)?;
        let mut kv = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let k = rms(&b.ck.forward(&c)?, &b.cnorm_k, self.cfg.eps)?.to_dtype(DType::BF16)?.reshape((1, s, h, hd))?;
            let v = b.cv.forward(&c)?.reshape((1, s, h, hd))?;
            kv.push((k, v));
        }
        Ok(Context { kv })
    }

    /// The flow at `latents` (B, in_dim, L), F32, at `timestep` (0..1000), for each
    /// batch row's text (`contexts[i]`). Returns (B, out_dim, L), F32.
    pub fn forward(&self, latents: &Tensor, timestep: f64, contexts: &[&Context]) -> Result<Tensor> {
        let (batch, _, len) = latents.dims3()?;
        if contexts.len() != batch {
            candle_core::bail!("{} contexts for {batch} latents", contexts.len());
        }
        let dev = latents.device();
        let (dim, h) = (self.cfg.dim, self.cfg.heads);
        let hd = dim / h;
        let eps = self.cfg.eps;
        // The timestep: a sinusoid (cos, then sin), its embedding t, and the blocks' t_mod.
        let half = self.cfg.freq_dim / 2;
        let sinusoid: Vec<f32> = (0..half).map(|i| (timestep * 10000f64.powf(-(i as f64) / half as f64)).cos() as f32)
            .chain((0..half).map(|i| (timestep * 10000f64.powf(-(i as f64) / half as f64)).sin() as f32))
            .collect();
        let sinusoid = Tensor::from_vec(sinusoid, (1, self.cfg.freq_dim), dev)?;
        let t = self.time2.forward(&self.time1.forward(&sinusoid)?.silu()?)?;
        let t_mod = self.time_proj.forward(&t.silu()?)?.reshape((6, dim))?;
        // Patchify: a 1x1 conv over channels, i.e. a linear map of each frame.
        let x = latents.transpose(1, 2)?.to_dtype(DType::BF16)?;
        let mut x = x.broadcast_matmul(&self.patch_w.t()?)?.broadcast_add(&self.patch_b)?.to_dtype(DType::F32)?;
        // RoPE over the frame index: 64 interleaved pairs of each 128-wide head.
        let pairs = hd / 2;
        let angles: Vec<f32> = (0..len).flat_map(|p| (0..pairs).map(move |i| (p as f64 / 10000f64.powf(2. * i as f64 / hd as f64)) as f32)).collect();
        let angles = Tensor::from_vec(angles, (len, pairs), dev)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        let rope = |y: &Tensor| -> Result<Tensor> {
            // (B, L, H, D) F32 -> interleaved RoPE in F32 (the reference's real-valued form) -> BF16.
            let y = y.transpose(1, 2)?.contiguous()?;
            candle_nn::rotary_emb::rope_i(&y, &cos, &sin)?.transpose(1, 2)?.to_dtype(DType::BF16)?.contiguous()
        };
        for (block, b) in self.blocks.iter().enumerate() {
            let m = b.modulation.broadcast_add(&t_mod)?;
            let row = |i: usize| m.narrow(0, i, 1);
            let (shift_a, scale_a, gate_a) = (row(0)?, row(1)?, row(2)?);
            let (shift_f, scale_f, gate_f) = (row(3)?, row(4)?, row(5)?);
            // Self-attention.
            let n = layer_norm(&x, eps, None)?.broadcast_mul(&(scale_a + 1.)?)?.broadcast_add(&shift_a)?.to_dtype(DType::BF16)?;
            let q = rope(&rms(&b.q.forward(&n)?, &b.norm_q, eps)?.reshape((batch, len, h, hd))?)?;
            let k = rope(&rms(&b.k.forward(&n)?, &b.norm_k, eps)?.reshape((batch, len, h, hd))?)?;
            let v = b.v.forward(&n)?.reshape((batch, len, h, hd))?;
            let a = b.o.forward(&attention(&q, &k, &v)?.reshape((batch, len, dim))?)?;
            x = (x + a.to_dtype(DType::F32)?.broadcast_mul(&gate_a)?)?;
            // Cross-attention to each row's text.
            let n = layer_norm(&x, eps, Some((&b.norm3_w, &b.norm3_b)))?.to_dtype(DType::BF16)?;
            let q = rms(&b.cq.forward(&n)?, &b.cnorm_q, eps)?.to_dtype(DType::BF16)?.reshape((batch, len, h, hd))?;
            let (k, v) = if batch == 1 {
                contexts[0].kv[block].clone()
            } else {
                let ks: Vec<&Tensor> = contexts.iter().map(|c| &c.kv[block].0).collect();
                let vs: Vec<&Tensor> = contexts.iter().map(|c| &c.kv[block].1).collect();
                (Tensor::cat(&ks, 0)?, Tensor::cat(&vs, 0)?)
            };
            let c = b.co.forward(&attention(&q, &k, &v)?.reshape((batch, len, dim))?)?;
            x = (x + c.to_dtype(DType::F32)?)?;
            // MLP.
            let n = layer_norm(&x, eps, None)?.broadcast_mul(&(scale_f + 1.)?)?.broadcast_add(&shift_f)?.to_dtype(DType::BF16)?;
            let f = b.ffn2.forward(&b.ffn1.forward(&n)?.gelu()?)?;
            x = (x + f.to_dtype(DType::F32)?.broadcast_mul(&gate_f)?)?;
        }
        // Head: modulated by the timestep embedding t itself (not t_mod).
        let hm = self.head_mod.broadcast_add(&t)?;
        let n = layer_norm(&x, eps, None)?.broadcast_mul(&(hm.narrow(0, 1, 1)? + 1.)?)?.broadcast_add(&hm.narrow(0, 0, 1)?)?.to_dtype(DType::BF16)?;
        self.head.forward(&n)?.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()
    }
}
