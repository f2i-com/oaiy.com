//! DINOv3 ViT (transformers' `DINOv3ViTModel`), as Pixal3D uses it: patch tokens
//! and the CLS and register tokens of an image, the last hidden state
//! normalized without affine (Pixal3D's `extract_features`, not the model's own
//! final norm). F32 throughout, as the reference runs it.
use crate::ltx::store::Store;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use oaiy_engine::json::Json;
use std::path::Path;

struct Layer {
    norm1: (Tensor, Tensor),
    q: (Tensor, Option<Tensor>),
    k: (Tensor, Option<Tensor>),
    v: (Tensor, Option<Tensor>),
    o: (Tensor, Option<Tensor>),
    scale1: Tensor,
    norm2: (Tensor, Tensor),
    up: (Tensor, Option<Tensor>),
    down: (Tensor, Option<Tensor>),
    gate: Option<(Tensor, Option<Tensor>)>,
    scale2: Tensor,
}

pub struct Dinov3 {
    patch_w: Tensor,
    patch_b: Tensor,
    cls: Tensor,
    registers: Tensor,
    layers: Vec<Layer>,
    heads: usize,
    hidden: usize,
    pub patch: usize,
    rope_theta: f64,
    eps: f64,
}

fn linear(x: &Tensor, w: &(Tensor, Option<Tensor>)) -> Result<Tensor> {
    let y = x.broadcast_matmul(&w.0.t()?)?;
    match &w.1 {
        Some(b) => y.broadcast_add(b),
        None => Ok(y),
    }
}

pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let x = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(w)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(b)?;
    }
    Ok(y)
}

/// `[x0..x_{d/2}, x_{d/2}..]` → `[-second half, first half]`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let a = x.narrow(D::Minus1, 0, d / 2)?;
    let b = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&b.neg()?, &a], D::Minus1)
}

impl Dinov3 {
    pub fn load(dir: &Path, dev: &Device) -> Result<Self> {
        let config = Json::parse(&std::fs::read(dir.join("config.json"))?).map_err(candle_core::Error::wrap)?;
        let n = |k: &str, d: i64| config.get(k).and_then(Json::as_i64).unwrap_or(d) as usize;
        let hidden = n("hidden_size", 1024);
        let heads = n("num_attention_heads", 16);
        let layers_n = n("num_hidden_layers", 24);
        let gated = config.get("use_gated_mlp").and_then(Json::as_bool).unwrap_or(false);
        let file = ["model.safetensors", "pytorch_model.safetensors"].iter().map(|f| dir.join(f)).find(|p| p.is_file()).ok_or_else(|| candle_core::Error::Msg(format!("{}: no model.safetensors", dir.display())))?;
        let mut store = Store::open(&file, 0)?;
        fn lin(store: &mut Store, p: &str, dev: &Device) -> Result<(Tensor, Option<Tensor>)> {
            let bias = format!("{p}.bias");
            let b = if store.index.get(&bias).is_some() { Some(store.tensor_f32(&bias, dev)?) } else { None };
            Ok((store.tensor_f32(&format!("{p}.weight"), dev)?, b))
        }
        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("layer.{i}");
            let s = &mut store;
            layers.push(Layer {
                norm1: (s.tensor_f32(&format!("{p}.norm1.weight"), dev)?, s.tensor_f32(&format!("{p}.norm1.bias"), dev)?),
                q: lin(s, &format!("{p}.attention.q_proj"), dev)?,
                k: lin(s, &format!("{p}.attention.k_proj"), dev)?,
                v: lin(s, &format!("{p}.attention.v_proj"), dev)?,
                o: lin(s, &format!("{p}.attention.o_proj"), dev)?,
                scale1: s.tensor_f32(&format!("{p}.layer_scale1.lambda1"), dev)?,
                norm2: (s.tensor_f32(&format!("{p}.norm2.weight"), dev)?, s.tensor_f32(&format!("{p}.norm2.bias"), dev)?),
                up: lin(s, &format!("{p}.mlp.up_proj"), dev)?,
                down: lin(s, &format!("{p}.mlp.down_proj"), dev)?,
                gate: if gated { Some(lin(s, &format!("{p}.mlp.gate_proj"), dev)?) } else { None },
                scale2: s.tensor_f32(&format!("{p}.layer_scale2.lambda1"), dev)?,
            });
        }
        let mut t = |k: &str| store.tensor_f32(k, dev);
        Ok(Self {
            patch_w: t("embeddings.patch_embeddings.weight")?,
            patch_b: t("embeddings.patch_embeddings.bias")?,
            cls: t("embeddings.cls_token")?,
            registers: t("embeddings.register_tokens")?,
            layers,
            heads,
            hidden,
            patch: n("patch_size", 16),
            rope_theta: config.get("rope_theta").and_then(Json::as_f64).unwrap_or(100.),
            eps: config.get("layer_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5),
        })
    }

    pub fn registers(&self) -> usize {
        self.registers.dim(1).unwrap_or(4)
    }

    /// `image` [3, H, W] (ImageNet-normalized) → all tokens [1 + registers + patches, hidden],
    /// the last hidden state normalized without affine.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let dev = image.device();
        let (_, h, w) = image.dims3()?;
        let (ph, pw) = (h / self.patch, w / self.patch);
        let conv = candle_nn::Conv2d::new(self.patch_w.clone(), Some(self.patch_b.clone()), candle_nn::Conv2dConfig { stride: self.patch, ..Default::default() });
        let patches = conv.forward(&image.unsqueeze(0)?)?.flatten_from(2)?.squeeze(0)?.t()?; // [P, hidden]
        let prefix = Tensor::cat(&[self.cls.squeeze(0)?, self.registers.squeeze(0)?], 0)?;
        let mut x = Tensor::cat(&[&prefix, &patches], 0)?;
        let n_prefix = prefix.dim(0)?;
        // RoPE over the patch centres in [-1, 1]: angles for y then x, the pair tiled twice.
        let head_dim = self.hidden / self.heads;
        let quarter = head_dim / 4;
        let inv: Vec<f64> = (0..quarter).map(|i| 1. / self.rope_theta.powf(i as f64 * 4. / head_dim as f64)).collect();
        let mut angles = Vec::with_capacity(ph * pw * head_dim);
        for i in 0..ph {
            for j in 0..pw {
                let cy = 2. * ((i as f64 + 0.5) / ph as f64) - 1.;
                let cx = 2. * ((j as f64 + 0.5) / pw as f64) - 1.;
                let row: Vec<f64> = inv.iter().map(|f| 2. * std::f64::consts::PI * cy * f).chain(inv.iter().map(|f| 2. * std::f64::consts::PI * cx * f)).collect();
                angles.extend(row.iter().chain(row.iter()).map(|&a| a as f32));
            }
        }
        let angles = Tensor::from_vec(angles, (ph * pw, 1, head_dim), dev)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        let tokens = x.dim(0)?;
        for l in &self.layers {
            let r = x.clone();
            let hn = layer_norm(&x, Some(&l.norm1.0), Some(&l.norm1.1), self.eps)?;
            let split = |t: Tensor| t.reshape((tokens, self.heads, head_dim));
            let q = split(linear(&hn, &l.q)?)?;
            let k = split(linear(&hn, &l.k)?)?;
            let v = split(linear(&hn, &l.v)?)?;
            let rope = |t: &Tensor| -> Result<Tensor> {
                let pre = t.narrow(0, 0, n_prefix)?;
                let p = t.narrow(0, n_prefix, tokens - n_prefix)?;
                let p = (p.broadcast_mul(&cos)? + rotate_half(&p)?.broadcast_mul(&sin)?)?;
                Tensor::cat(&[&pre, &p], 0)
            };
            let (q, k) = (rope(&q)?, rope(&k)?);
            let a = attention(&q, &k, &v)?.reshape((tokens, self.hidden))?;
            x = (linear(&a, &l.o)?.broadcast_mul(&l.scale1)? + r)?;
            let r = x.clone();
            let hn = layer_norm(&x, Some(&l.norm2.0), Some(&l.norm2.1), self.eps)?;
            let m = match &l.gate {
                Some(g) => linear(&(linear(&hn, g)?.gelu_erf()? * linear(&hn, &l.up)?)?, &l.down)?,
                None => linear(&linear(&hn, &l.up)?.gelu_erf()?, &l.down)?,
            };
            x = (m.broadcast_mul(&l.scale2)? + r)?;
        }
        layer_norm(&x, None, None, 1e-5)
    }
}

/// Exact attention in F32 for one sequence [N, H, d], in query chunks.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let q = q.transpose(0, 1)?.contiguous()?;
    let k = k.transpose(0, 1)?.contiguous()?;
    let v = v.transpose(0, 1)?.contiguous()?;
    let n = q.dim(1)?;
    let step = 1024;
    let mut outs = Vec::new();
    let mut at = 0;
    while at < n {
        let m = step.min(n - at);
        let s = (q.narrow(1, at, m)?.matmul(&k.t()?)? / (d as f64).sqrt())?;
        let p = candle_nn::ops::softmax_last_dim(&s)?;
        outs.push(p.matmul(&v)?);
        at += m;
    }
    Tensor::cat(&outs, 1)?.transpose(0, 1)?.contiguous().map(|t| t.to_dtype(DType::F32)).and_then(|t| t)
}
