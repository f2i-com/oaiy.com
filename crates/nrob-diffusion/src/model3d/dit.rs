//! Pixal3D's flow transformers (TRELLIS.2's `SparseStructureFlowModel` and
//! `SLatFlowModel` in "proj" mode): 30 blocks of 1536 channels, each
//! self-attention over the voxels (RMS-normed q/k, 3-D RoPE), cross-attention to
//! the image's DINOv3 tokens plus that voxel's back-projected image features
//! (`proj_linear`), and a tanh-GELU MLP, modulated by the timestep through one
//! shared adaLN plus a learned offset per block.
//!
//! The structure model runs over every voxel of a 16³ grid; the latent models
//! over the voxels a structure has. Both are the same network here, given the
//! voxels' coordinates. As the reference runs it: the blocks in BF16; norms, the
//! timestep path, the input and output layers in F32.
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor, D};
use nrob::json::Json;
use std::path::Path;

pub struct Config {
    pub channels: usize,
    pub heads: usize,
    pub blocks: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub cond_channels: usize,
    pub proj_channels: usize,
    pub mlp: usize,
    pub resolution: usize,
}

impl Config {
    pub fn read(path: &Path) -> Result<Self> {
        let j = Json::parse(&std::fs::read(path)?).map_err(candle_core::Error::wrap)?;
        let a = j.get("args").ok_or_else(|| candle_core::Error::Msg(format!("{}: no args", path.display())))?;
        let n = |k: &str| a.get(k).and_then(Json::as_i64).map(|v| v as usize).ok_or_else(|| candle_core::Error::Msg(format!("{}: no {k}", path.display())));
        if a.get("pe_mode").and_then(Json::as_str) != Some("rope") || a.get("image_attn_mode").and_then(Json::as_str) != Some("proj") || a.get("share_mod").and_then(Json::as_bool) != Some(true) {
            candle_core::bail!("{}: only RoPE, shared-modulation, projection-attention models are supported", path.display());
        }
        let channels = n("model_channels")?;
        let ratio = a.get("mlp_ratio").and_then(Json::as_f64).unwrap_or(4.);
        let cond = n("cond_channels")?;
        Ok(Self {
            channels,
            heads: n("num_heads")?,
            blocks: n("num_blocks")?,
            in_channels: n("in_channels")?,
            out_channels: n("out_channels")?,
            cond_channels: cond,
            proj_channels: a.get("proj_in_channels").and_then(Json::as_i64).map(|v| v as usize).unwrap_or(cond),
            mlp: (channels as f64 * ratio) as usize,
            resolution: n("resolution")?,
        })
    }
}

struct Lin {
    w: Tensor,
    b: Tensor,
}
impl Lin {
    fn load(store: &mut Store, prefix: &str, dtype: DType, dev: &Device) -> Result<Self> {
        Ok(Self { w: store.tensor(&format!("{prefix}.weight"), dev, false)?.to_dtype(dtype)?.t()?.contiguous()?, b: store.tensor(&format!("{prefix}.bias"), dev, false)?.to_dtype(dtype)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.to_dtype(self.w.dtype())?.broadcast_matmul(&self.w)?.broadcast_add(&self.b)
    }
}

struct Block {
    modulation: Tensor,
    qkv: Lin,
    q_norm: Tensor,
    k_norm: Tensor,
    out: Lin,
    norm2_w: Tensor,
    norm2_b: Tensor,
    cq: Lin,
    ckv: Lin,
    cq_norm: Tensor,
    ck_norm: Tensor,
    cout: Lin,
    proj: Lin,
    fc1: Lin,
    fc2: Lin,
}

pub struct Dit {
    cfg: Config,
    input: Lin,
    t1: Lin,
    t2: Lin,
    ada: Lin,
    blocks: Vec<Block>,
    output: Lin,
    head_dim: usize,
}

/// The image's conditioning for one run: the cross-attention's keys and values for
/// each block (the image tokens do not change between steps), and the voxels'
/// back-projected features. The unconditional branch has both zero.
pub struct Context {
    kv: Vec<(Tensor, Tensor)>,
    /// [N, proj_channels], or None for zeros.
    proj: Option<Tensor>,
}

fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?;
    let x = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    x.broadcast_div(&(var + eps)?.sqrt()?)
}

/// `MultiHeadRMSNorm`: each head's vector made unit length, times γ·√d. [N, H, d].
fn head_rms(x: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)? as f64;
    let xf = x.to_dtype(DType::F32)?;
    let len = xf.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
    (xf.broadcast_div(&len)?.broadcast_mul(gamma)? * d.sqrt())?.to_dtype(x.dtype())
}

/// Interleaved rotary embedding with precomputed cos/sin [N, 1, d/2]. x: [N, H, d].
fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (n, h, d) = x.dims3()?;
    let xf = x.to_dtype(DType::F32)?.reshape((n, h, d / 2, 2))?;
    let a = xf.narrow(3, 0, 1)?.squeeze(3)?;
    let b = xf.narrow(3, 1, 1)?.squeeze(3)?;
    let ra = (a.broadcast_mul(cos)? - b.broadcast_mul(sin)?)?;
    let rb = (a.broadcast_mul(sin)? + b.broadcast_mul(cos)?)?;
    Tensor::stack(&[ra, rb], 3)?.reshape((n, h, d))?.to_dtype(x.dtype())
}

/// Attention for one sequence: q [Nq, H, d], k/v [Nk, H, d].
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        let o = candle_flash_attn::flash_attn(&q.unsqueeze(0)?.contiguous()?, &k.unsqueeze(0)?.contiguous()?, &v.unsqueeze(0)?.contiguous()?, 1. / (d as f32).sqrt(), false)?;
        return o.squeeze(0);
    }
    let q = q.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
    let k = k.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(0, 1)?.contiguous()?.to_dtype(DType::F32)?;
    let s = (q.matmul(&k.t()?)? / (d as f64).sqrt())?;
    let p = candle_nn::ops::softmax_last_dim(&s)?;
    p.matmul(&v)?.transpose(0, 1)?.contiguous()?.to_dtype(DType::BF16)
}

/// The rotary angles of voxels at `coords`: 21 frequencies per axis (for d = 128),
/// the leftover pair unrotated.
pub fn rope_tables(coords: &[[i32; 3]], head_dim: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let per = half / 3;
    let freqs: Vec<f64> = (0..per).map(|i| 1. / 10000f64.powf(i as f64 / per as f64)).collect();
    let mut cos = Vec::with_capacity(coords.len() * half);
    let mut sin = Vec::with_capacity(coords.len() * half);
    for c in coords {
        for axis in 0..3 {
            for f in &freqs {
                let a = c[axis] as f64 * f;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        for _ in per * 3..half {
            cos.push(1.);
            sin.push(0.);
        }
    }
    let n = coords.len();
    Ok((Tensor::from_vec(cos, (n, 1, half), dev)?, Tensor::from_vec(sin, (n, 1, half), dev)?))
}

impl Dit {
    /// `path`: the checkpoint without its extension.
    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let cfg = Config::read(&path.with_extension("json"))?;
        let mut store = Store::open(&path.with_extension("safetensors"), 0)?;
        let bf = DType::BF16;
        let f32 = DType::F32;
        let blocks = (0..cfg.blocks)
            .map(|i| {
                let p = format!("blocks.{i}");
                Ok(Block {
                    // A plain parameter, not a linear layer: it stays F32 in the reference.
                    modulation: store.tensor_f32(&format!("{p}.modulation"), dev)?,
                    qkv: Lin::load(&mut store, &format!("{p}.self_attn.to_qkv"), bf, dev)?,
                    q_norm: store.tensor_f32(&format!("{p}.self_attn.q_rms_norm.gamma"), dev)?,
                    k_norm: store.tensor_f32(&format!("{p}.self_attn.k_rms_norm.gamma"), dev)?,
                    out: Lin::load(&mut store, &format!("{p}.self_attn.to_out"), bf, dev)?,
                    norm2_w: store.tensor_f32(&format!("{p}.norm2.weight"), dev)?,
                    norm2_b: store.tensor_f32(&format!("{p}.norm2.bias"), dev)?,
                    cq: Lin::load(&mut store, &format!("{p}.cross_attn.cross_attn_block.to_q"), bf, dev)?,
                    ckv: Lin::load(&mut store, &format!("{p}.cross_attn.cross_attn_block.to_kv"), bf, dev)?,
                    cq_norm: store.tensor_f32(&format!("{p}.cross_attn.cross_attn_block.q_rms_norm.gamma"), dev)?,
                    ck_norm: store.tensor_f32(&format!("{p}.cross_attn.cross_attn_block.k_rms_norm.gamma"), dev)?,
                    cout: Lin::load(&mut store, &format!("{p}.cross_attn.cross_attn_block.to_out"), bf, dev)?,
                    proj: Lin::load(&mut store, &format!("{p}.cross_attn.proj_linear"), bf, dev)?,
                    fc1: Lin::load(&mut store, &format!("{p}.mlp.mlp.0"), bf, dev)?,
                    fc2: Lin::load(&mut store, &format!("{p}.mlp.mlp.2"), bf, dev)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            head_dim: cfg.channels / cfg.heads,
            input: Lin::load(&mut store, "input_layer", f32, dev)?,
            t1: Lin::load(&mut store, "t_embedder.mlp.0", f32, dev)?,
            t2: Lin::load(&mut store, "t_embedder.mlp.2", f32, dev)?,
            ada: Lin::load(&mut store, "adaLN_modulation.1", f32, dev)?,
            output: Lin::load(&mut store, "out_layer", f32, dev)?,
            blocks,
            cfg,
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The conditioning for a run: `image` [T, 1024] DINOv3 tokens and `proj` [N, P]
    /// per-voxel features; None for the unconditional branch (zeros).
    pub fn context(&self, image: Option<&Tensor>, proj: Option<&Tensor>, tokens: usize, dev: &Device) -> Result<Context> {
        let heads = self.cfg.heads;
        let image = match image {
            Some(t) => t.to_dtype(DType::BF16)?,
            None => Tensor::zeros((tokens, self.cfg.cond_channels), DType::BF16, dev)?,
        };
        let kv = self
            .blocks
            .iter()
            .map(|b| {
                let kv = b.ckv.forward(&image)?;
                let t = kv.dim(0)?;
                let kv = kv.reshape((t, 2, heads, self.head_dim))?;
                let k = head_rms(&kv.narrow(1, 0, 1)?.squeeze(1)?, &b.ck_norm)?;
                let v = kv.narrow(1, 1, 1)?.squeeze(1)?.contiguous()?;
                Ok((k.contiguous()?, v))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Context { kv, proj: proj.map(|p| p.to_dtype(DType::BF16)).transpose()? })
    }

    /// The velocity at `timestep` (0-1000) for voxel features `x` [N, in] at `coords`' rotary angles.
    pub fn forward(&self, x: &Tensor, timestep: f64, rope: &(Tensor, Tensor), ctx: &Context) -> Result<Tensor> {
        let dev = x.device();
        let (n, _) = x.dims2()?;
        let c = self.cfg.channels;
        let heads = self.cfg.heads;
        let d = self.head_dim;
        // The timestep's sinusoidal embedding: [cos, sin].
        let half = 128;
        let freqs: Vec<f32> = (0..half).map(|i| (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32).collect();
        let args: Vec<f32> = freqs.iter().map(|f| (timestep as f32) * f).collect();
        let emb: Vec<f32> = args.iter().map(|a| a.cos()).chain(args.iter().map(|a| a.sin())).collect();
        let t = Tensor::from_vec(emb, (1, 256), dev)?;
        let t = self.t2.forward(&self.t1.forward(&t)?.silu()?)?;
        let mod_all = self.ada.forward(&t.silu()?)?.to_dtype(DType::BF16)?;
        let mut h = self.input.forward(&x.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?;
        for (b, (ck, cv)) in self.blocks.iter().zip(&ctx.kv) {
            // (modulation + mod) in F32, then to BF16, as the reference's type promotion does.
            let m = mod_all.to_dtype(DType::F32)?.broadcast_add(&b.modulation)?.to_dtype(DType::BF16)?;
            let chunk = |i: usize| m.narrow(1, i * c, c);
            let (shift_msa, scale_msa, gate_msa) = (chunk(0)?, chunk(1)?, chunk(2)?);
            let (shift_mlp, scale_mlp, gate_mlp) = (chunk(3)?, chunk(4)?, chunk(5)?);
            // Self-attention.
            let hn = layer_norm(&h, 1e-6)?.to_dtype(DType::BF16)?;
            let hn = hn.broadcast_mul(&(scale_msa + 1.)?)?.broadcast_add(&shift_msa)?;
            let qkv = b.qkv.forward(&hn)?.reshape((n, 3, heads, d))?;
            let q = head_rms(&qkv.narrow(1, 0, 1)?.squeeze(1)?, &b.q_norm)?;
            let k = head_rms(&qkv.narrow(1, 1, 1)?.squeeze(1)?, &b.k_norm)?;
            let v = qkv.narrow(1, 2, 1)?.squeeze(1)?.contiguous()?;
            let q = rotate(&q, &rope.0, &rope.1)?;
            let k = rotate(&k, &rope.0, &rope.1)?;
            let a = attention(&q, &k, &v)?.reshape((n, c))?;
            let a = b.out.forward(&a)?;
            h = (h + a.broadcast_mul(&gate_msa)?)?;
            // Cross-attention to the image, plus the voxel's own image features.
            let hn = layer_norm(&h, 1e-6)?.broadcast_mul(&b.norm2_w)?.broadcast_add(&b.norm2_b)?.to_dtype(DType::BF16)?;
            let q = head_rms(&b.cq.forward(&hn)?.reshape((n, heads, d))?, &b.cq_norm)?;
            let a = attention(&q, ck, cv)?.reshape((n, c))?;
            let mut a = b.cout.forward(&a)?;
            a = match &ctx.proj {
                Some(p) => (a + b.proj.forward(p)?)?,
                None => a.broadcast_add(&b.proj.b)?,
            };
            h = (h + a)?;
            // MLP.
            let hn = layer_norm(&h, 1e-6)?.to_dtype(DType::BF16)?;
            let hn = hn.broadcast_mul(&(scale_mlp + 1.)?)?.broadcast_add(&shift_mlp)?;
            let y = b.fc2.forward(&b.fc1.forward(&hn)?.gelu()?)?;
            h = (h + y.broadcast_mul(&gate_mlp)?)?;
        }
        let h = layer_norm(&h, 1e-5)?;
        self.output.forward(&h)
    }
}
