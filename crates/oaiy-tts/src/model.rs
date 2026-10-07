//! The Qwen3-TTS talker and its code predictor: Qwen3-style decoders (GQA,
//! per-head q/k RMSNorm, SwiGLU, rotate-half RoPE) with a KV cache. For speech
//! the talker's multimodal RoPE has all three position streams equal, so it is
//! ordinary 1-D RoPE.
use crate::weights::{tensor_bytes, TensorSource};
use candle_core::{DType, Device, Result, Tensor, D};

pub struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}
impl Linear {
    pub fn load(store: &mut (impl TensorSource + ?Sized), prefix: &str, dev: &Device) -> Result<Self> {
        let w = store.load(&format!("{prefix}.weight"), dev)?;
        let bias = format!("{prefix}.bias");
        let b = if store.has(&bias) { Some(store.load(&bias, dev)?) } else { None };
        Ok(Self { w, b })
    }
    /// `x`: (..., in) to (..., out).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = matmul_t(x, &self.w)?;
        match &self.b {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
    /// Several projections of the same input as one matmul (outputs stacked).
    pub fn fused(store: &mut (impl TensorSource + ?Sized), prefixes: &[String], dev: &Device) -> Result<Self> {
        let ws = prefixes.iter().map(|p| store.load(&format!("{p}.weight"), dev)).collect::<Result<Vec<_>>>()?;
        Ok(Self { w: Tensor::cat(&ws, 0)?, b: None })
    }
    /// Device bytes held.
    pub fn bytes(&self) -> u64 {
        tensor_bytes([&self.w].into_iter().chain(self.b.as_ref()))
    }
}

/// `x` (..., in) times `w` (out, in) transposed: (..., out), as one 2-D
/// product. A batched (broadcast) product costs cuBLAS twice the launch time
/// and runs slower, which dominates at one token.
pub fn matmul_t(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let input = *dims.last().ok_or_else(|| candle_core::Error::Msg("matmul: input has no axes".into()))?;
    let y = x.reshape((x.elem_count() / input, input))?.matmul(&w.t()?)?;
    let mut out = dims.to_vec();
    if let Some(last) = out.last_mut() {
        *last = w.dim(0)?;
    }
    y.reshape(out)
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

/// Keys and values seen so far, per layer, in buffers with room to grow:
/// a step writes its keys in place rather than copying the whole cache
/// (`Tensor::cat` onto a cache costs far more than the step's own work).
pub struct Cache {
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    pub len: usize,
}
impl Cache {
    pub fn new(layers: usize) -> Self {
        Self { k: vec![None; layers], v: vec![None; layers], len: 0 }
    }

    /// Start over, keeping the buffers.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Write `k` and `v` (1, t, kv_heads, head_dim) at `start` in layer
    /// `layer`'s buffers; the keys and values from 0 to `start + t`.
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor, start: usize) -> Result<(Tensor, Tensor)> {
        let need = start + k.dim(1)?;
        let capacity = self.k[layer].as_ref().map_or(Ok(0), |b| b.dim(1))?;
        if need > capacity {
            let grown = need.max(2 * capacity).max(32);
            let (_, _, heads, dim) = k.dims4()?;
            for (slot, new) in [(&mut self.k[layer], k), (&mut self.v[layer], v)] {
                let buffer = Tensor::zeros((1, grown, heads, dim), new.dtype(), new.device())?;
                if let (Some(old), true) = (slot.as_ref(), start > 0) {
                    buffer.slice_set(&old.narrow(1, 0, start)?, 1, 0)?;
                }
                *slot = Some(buffer);
            }
        }
        let (Some(bk), Some(bv)) = (&self.k[layer], &self.v[layer]) else { unreachable!("allocated above") };
        bk.slice_set(&k.contiguous()?, 1, start)?;
        bv.slice_set(&v.contiguous()?, 1, start)?;
        Ok((bk.narrow(1, 0, need)?, bv.narrow(1, 0, need)?))
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
    pub fn load(store: &mut (impl TensorSource + ?Sized), prefix: &str, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, theta: f64, max_positions: usize, eps: f32, dev: &Device) -> Result<Self> {
        let inv: Vec<f64> = (0..head_dim / 2).map(|i| 1. / theta.powf(2. * i as f64 / head_dim as f64)).collect();
        Self::load_with(store, prefix, layers, heads, kv_heads, head_dim, &inv, max_positions, eps, true, dev)
    }

    /// As `load`, with RoPE's inverse frequencies given (scaled ones, e.g. llama3's) and the
    /// q/k norms optional (`qk_norm`: false for Llama-style layers).
    #[allow(clippy::too_many_arguments)]
    pub fn load_with(store: &mut (impl TensorSource + ?Sized), prefix: &str, layers: usize, heads: usize, kv_heads: usize, head_dim: usize, inv: &[f64], max_positions: usize, eps: f32, qk_norm: bool, dev: &Device) -> Result<Self> {
        let mut out = Vec::with_capacity(layers);
        for i in 0..layers {
            let l = format!("{prefix}.layers.{i}");
            let names = |ns: &[&str]| ns.iter().map(|n| format!("{l}.{n}")).collect::<Vec<_>>();
            out.push(Layer {
                qkv: Linear::fused(store, &names(&["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), dev)?,
                o: Linear::load(store, &format!("{l}.self_attn.o_proj"), dev)?,
                q_norm: if qk_norm { Some(store.load(&format!("{l}.self_attn.q_norm.weight"), dev)?) } else { None },
                k_norm: if qk_norm { Some(store.load(&format!("{l}.self_attn.k_norm.weight"), dev)?) } else { None },
                input_norm: store.load(&format!("{l}.input_layernorm.weight"), dev)?,
                post_norm: store.load(&format!("{l}.post_attention_layernorm.weight"), dev)?,
                gate_up: Linear::fused(store, &names(&["mlp.gate_proj", "mlp.up_proj"]), dev)?,
                down: Linear::load(store, &format!("{l}.mlp.down_proj"), dev)?,
            });
        }
        // RoPE tables in F32 (as the reference computes them), cast to the
        // weights' type (BF16 for the checkpoints).
        let half = head_dim / 2;
        let freqs: Vec<f32> = (0..max_positions).flat_map(|p| inv.iter().map(move |f| (p as f64 * f) as f32)).collect();
        let freqs = Tensor::from_vec(freqs, (max_positions, half), dev)?;
        let norm = store.load(&format!("{prefix}.norm.weight"), dev)?;
        let dtype = norm.dtype();
        Ok(Self {
            layers: out,
            norm,
            heads,
            kv_heads,
            head_dim,
            eps,
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Positions the RoPE tables cover.
    pub fn max_positions(&self) -> usize {
        self.cos.dim(0).unwrap_or(0)
    }

    /// Device bytes held by the weights and RoPE tables.
    pub fn bytes(&self) -> u64 {
        let mut b = tensor_bytes([&self.norm, &self.cos, &self.sin]);
        for l in &self.layers {
            b += l.qkv.bytes() + l.o.bytes() + l.gate_up.bytes() + l.down.bytes();
            b += tensor_bytes([&l.input_norm, &l.post_norm].into_iter().chain(l.q_norm.as_ref()).chain(l.k_norm.as_ref()));
        }
        b
    }

    /// `x`: (1, T, hidden) BF16, continuing from `cache`. Returns the final
    /// (normed) hidden states, (1, T, hidden).
    pub fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let t = x.dim(1)?;
        let start = cache.len;
        if start + t > self.max_positions() {
            candle_core::bail!("the decoder's {} positions are used up", self.max_positions());
        }
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
            let (k, v) = cache.append(i, &k, &v, start)?;
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
/// heads; query `i` sits at position `start + i`. Flash attention when built
/// with it (its causal mask is aligned to the last key, which is exactly the
/// KV-cache case); otherwise F32 softmax as the reference's SDPA computes.
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, start: usize) -> Result<Tensor> {
    let hd = q.dim(3)?;
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
    candle_nn::ops::softmax(&scores, D::Minus1)?.matmul(&vx)?.to_dtype(q.dtype())?.transpose(1, 2)?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::InMemory;

    /// A tiny random Qwen3-style decoder: 2 layers, 4 heads (2 for K/V) of 8.
    fn tiny(dev: &Device) -> Result<Decoder> {
        let mut rng = crate::sampling::Rng::new(9);
        let mut w = InMemory { dtype: DType::F32, ..Default::default() };
        let mut put = |name: String, shape: &[usize], scale: f32, offset: f32| -> Result<()> {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n).map(|_| offset + scale * (rng.uniform() as f32 * 2. - 1.)).collect();
            w.tensors.insert(name, Tensor::from_vec(v, shape, dev)?);
            Ok(())
        };
        let (hidden, heads, kv, hd, inner) = (16usize, 4usize, 2usize, 8usize, 24usize);
        for l in 0..2 {
            let p = format!("m.layers.{l}");
            put(format!("{p}.self_attn.q_proj.weight"), &[heads * hd, hidden], 0.3, 0.)?;
            put(format!("{p}.self_attn.k_proj.weight"), &[kv * hd, hidden], 0.3, 0.)?;
            put(format!("{p}.self_attn.v_proj.weight"), &[kv * hd, hidden], 0.3, 0.)?;
            put(format!("{p}.self_attn.o_proj.weight"), &[hidden, heads * hd], 0.3, 0.)?;
            put(format!("{p}.self_attn.q_norm.weight"), &[hd], 0.1, 1.)?;
            put(format!("{p}.self_attn.k_norm.weight"), &[hd], 0.1, 1.)?;
            put(format!("{p}.input_layernorm.weight"), &[hidden], 0.1, 1.)?;
            put(format!("{p}.post_attention_layernorm.weight"), &[hidden], 0.1, 1.)?;
            put(format!("{p}.mlp.gate_proj.weight"), &[inner, hidden], 0.3, 0.)?;
            put(format!("{p}.mlp.up_proj.weight"), &[inner, hidden], 0.3, 0.)?;
            put(format!("{p}.mlp.down_proj.weight"), &[hidden, inner], 0.3, 0.)?;
        }
        put("m.norm.weight".into(), &[hidden], 0.1, 1.)?;
        Decoder::load(&mut w, "m", 2, heads, kv, hd, 10000., 128, 1e-6, dev)
    }

    #[test]
    fn a_step_at_a_time_matches_the_whole_sequence() -> Result<()> {
        let dev = Device::Cpu;
        let dec = tiny(&dev)?;
        let mut rng = crate::sampling::Rng::new(3);
        // 45 positions: past the cache's first capacity (32), so it grows.
        let x: Vec<f32> = (0..45 * 16).map(|_| rng.uniform() as f32 * 2. - 1.).collect();
        let x = Tensor::from_vec(x, (1, 45, 16), &dev)?;
        let whole = dec.forward(&x, &mut Cache::new(2))?.to_dtype(DType::F32)?;
        let mut cache = Cache::new(2);
        let mut steps = vec![dec.forward(&x.narrow(1, 0, 5)?, &mut cache)?];
        for t in 5..45 {
            steps.push(dec.forward(&x.narrow(1, t, 1)?, &mut cache)?);
        }
        assert_eq!(cache.len, 45);
        let stepped = Tensor::cat(&steps, 1)?.to_dtype(DType::F32)?;
        let err = (&stepped - &whole)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-4, "{err}");
        // Starting over reuses the buffers and gives the same again.
        cache.reset();
        let again = dec.forward(&x.narrow(1, 0, 5)?, &mut cache)?.to_dtype(DType::F32)?;
        let err = (&again - &whole.narrow(1, 0, 5)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-4, "{err}");
        Ok(())
    }
}
