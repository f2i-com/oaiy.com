//! The Qwen3-TTS talker and its code predictor: Qwen3-style decoders (GQA,
//! per-head q/k RMSNorm, SwiGLU, rotate-half RoPE) with a KV cache. For speech
//! the talker's multimodal RoPE has all three position streams equal, so it is
//! ordinary 1-D RoPE.
use crate::ltx::store::Store;
use candle_core::{DType, Device, Result, Tensor, D};

pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}
impl Linear {
    pub fn load(store: &mut Store, prefix: &str, dev: &Device) -> Result<Self> {
        let w = store.tensor(&format!("{prefix}.weight"), dev, false)?;
        let bias = format!("{prefix}.bias");
        let b = if store.index.get(&bias).is_some() { Some(store.tensor(&bias, dev, false)?) } else { None };
        Ok(Self { w, b })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.broadcast_matmul(&self.w.t()?)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
    /// Several projections of the same input as one matmul (outputs stacked).
    pub fn fused(store: &mut Store, prefixes: &[String], dev: &Device) -> Result<Self> {
        let ws = prefixes.iter().map(|p| store.tensor(&format!("{p}.weight"), dev, false)).collect::<Result<Vec<_>>>()?;
        Ok(Self { w: Tensor::cat(&ws, 0)?, b: None })
    }
}

fn rms(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, w, eps)
}

struct Layer {
    /// q, k and v projections fused (outputs: heads, kv heads, kv heads).
    qkv: Linear,
    o: Linear,
    /// Per-head q/k RMS norms (Qwen3); none for Llama-style layers.
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    input_norm: Tensor,
    post_norm: Tensor,
    /// gate and up fused.
    gate_up: Linear,
    down: Linear,
}

/// Keys and values seen so far, per layer.
pub struct Cache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    pub len: usize,
}
impl Cache {
    pub fn new(layers: usize) -> Self {
        Self { k: vec![None; layers], v: vec![None; layers], len: 0 }
    }
}

pub struct Decoder {
    layers: Vec<Layer>,
    norm: Tensor,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    cos: Tensor,
    sin: Tensor,
}

impl Decoder {
    /// `prefix`: e.g. `talker.model` (layers under `{prefix}.layers.N`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(store: &mut Store, prefix: &str, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, theta: f64, max_positions: usize, eps: f32, dev: &Device) -> Result<Self> {
        let inv: Vec<f64> = (0..head_dim / 2).map(|i| 1. / theta.powf(2. * i as f64 / head_dim as f64)).collect();
        Self::load_with(store, prefix, layers, heads, kv_heads, head_dim, &inv, max_positions, eps, true, dev)
    }

    /// As `load`, with RoPE's inverse frequencies given (scaled ones, e.g. llama3's) and the
    /// q/k norms optional (`qk_norm`: false for Llama-style layers).
    #[allow(clippy::too_many_arguments)]
    pub fn load_with(store: &mut Store, prefix: &str, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, inv: &[f64], max_positions: usize, eps: f32, qk_norm: bool, dev: &Device) -> Result<Self> {
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            let l = format!("{prefix}.layers.{i}");
            let names = |ns: &[&str]| ns.iter().map(|n| format!("{l}.{n}")).collect::<Vec<_>>();
            out.push(Layer {
                qkv: Linear::fused(store, &names(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), dev)?,
                o: Linear::load(store, &format!("{l}.self_attn.o_proj"), dev)?,
                q_norm: if qk_norm { Some(store.tensor(&format!("{l}.self_attn.q_norm.weight"), dev, false)?) } else { None },
                k_norm: if qk_norm { Some(store.tensor(&format!("{l}.self_attn.k_norm.weight"), dev, false)?) } else { None },
                input_norm: store.tensor(&format!("{l}.input_layernorm.weight"), dev, false)?,
                post_norm: store.tensor(&format!("{l}.post_attention_layernorm.weight"), dev, false)?,
                gate_up: Linear::fused(store, &names(&["mlp.gate_proj", "mlp.up_proj"]), dev)?,
                down: Linear::load(store, &format!("{l}.mlp.down_proj"), dev)?,
            });
        }
        // RoPE tables in F32 (as the reference computes them), cast to BF16.
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..max_positions).flat_map(|p| inv.iter().map(move |f| (p as f64 * f) as f32)).collect();
        let freqs = Tensor::from_vec(freqs, (max_positions, half), dev)?;
        Ok(Self {
            layers: out,
            norm: store.tensor(&format!("{prefix}.norm.weight"), dev, false)?,
            heads,
            kv_heads,
            head_dim,
            eps,
            cos: freqs.cos()?.to_dtype(DType::BF16)?,
            sin: freqs.sin()?.to_dtype(DType::BF16)?,
        })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// `x`: (1, T, hidden) BF16, continuing from `cache`. Returns the final
    /// (normed) hidden states, (1, T, hidden).
    pub fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let t = x.dim(1)?;
        let start = cache.len;
        let cos = self.cos.narrow(0, start, t)?;
        let sin = self.sin.narrow(0, start, t)?;
        let mut h = x.clone();
        let (hd, nq, nkv) = (self.head_dim, self.heads, self.kv_heads);
        for (i, l) in self.layers.iter().enumerate() {
            let n = rms(&h, &l.input_norm, self.eps)?;
            // (batch, time, heads, dim) throughout: no transposes.
            let qkv = l.qkv.forward(&n)?;
            let q = qkv.narrow(2, 0, nq * hd)?.reshape((1, t, nq, hd))?;
            let k = qkv.narrow(2, nq * hd, nkv * hd)?.reshape((1, t, nkv, hd))?;
            let q = match &l.q_norm { Some(w) => rms(&q, w, self.eps)?, None => q.contiguous()? };
            let k = match &l.k_norm { Some(w) => rms(&k, w, self.eps)?, None => k.contiguous()? };
            let v = qkv.narrow(2, (nq + nkv) * hd, nkv * hd)?.reshape((1, t, nkv, hd))?.contiguous()?;
            let q = candle_nn::rotary_emb::rope_thd(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_thd(&k, &cos, &sin)?;
            let (k, v) = match (&cache.k[i], &cache.v[i]) {
                (Some(pk), Some(pv)) => (Tensor::cat(&[pk, &k], 1)?, Tensor::cat(&[pv, &v], 1)?),
                _ => (k, v),
            };
            cache.k[i] = Some(k.clone());
            cache.v[i] = Some(v.clone());
            let a = attention(&q, &k, &v, start)?.reshape((1, t, nq * hd))?;
            h = (h + l.o.forward(&a)?)?;
            let n = rms(&h, &l.post_norm, self.eps)?;
            let gu = l.gate_up.forward(&n)?;
            let inner = gu.dim(2)? / 2;
            let m = (gu.narrow(2, 0, inner)?.silu()? * gu.narrow(2, inner, inner)?)?;
            h = (h + l.down.forward(&m)?)?;
        }
        cache.len += t;
        rms(&h, &self.norm, self.eps)
    }
}

/// Causal attention over (batch, time, heads, dim) tensors with grouped K/V
/// heads; query `i` sits at position `start + i`. Flash attention when the
/// worker has it (its causal mask is aligned to the last key, which is exactly
/// the KV-cache case); otherwise F32 softmax as the reference's SDPA computes.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, start: usize) -> Result<Tensor> {
    let hd = q.dim(3)?;
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() {
        return candle_flash_attn::flash_attn(q, k, v, 1. / (hd as f32).sqrt(), true);
    }
    let (_, nk, kv_heads, _) = k.dims4()?;
    let (_, nq, heads, _) = q.dims4()?;
    let groups = heads / kv_heads;
    let expand = |x: &Tensor| -> Result<Tensor> {
        x.transpose(1, 2)?
            .unsqueeze(2)?
            .expand((1, kv_heads, groups, nk, hd))?
            .reshape((1, heads, nk, hd))?
            .to_dtype(DType::F32)
    };
    let (kx, vx) = (expand(k)?, expand(v)?);
    let qx = q.transpose(1, 2)?.to_dtype(DType::F32)?;
    let mut scores = (qx.matmul(&kx.t()?)? / (hd as f64).sqrt())?;
    if nq > 1 {
        let mask: Vec<f32> = (0..nq).flat_map(|i| (0..nk).map(move |j| if j <= start + i { 0. } else { f32::NEG_INFINITY })).collect();
        scores = scores.broadcast_add(&Tensor::from_vec(mask, (nq, nk), q.device())?)?;
    }
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(&vx)?.to_dtype(DType::BF16)?.transpose(1, 2)?.contiguous()
}
