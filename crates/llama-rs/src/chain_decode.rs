//! VENDORED-LOCAL: a dense model's decode step chained on the backend's device in one submit ([`ggml_rs::chain`]):
//! every layer's norm, q, k and v (and their per-head norms, Qwen3's), RoPE (its sines and cosines made here as the
//! CPU's rope makes them), the K and V stored into a copy of the KV cache kept on the device, attention over it, the
//! output projection, residual, FFN and residual, then the head; only the logits and the step's K and V rows (for the
//! host's cache) come back. The copy is brought up to date first with the rows the host wrote since (a prompt's;
//! [`KvCache::dirty_from`]). On WebGPU a 3B Llama's step, its 113 round trips one, went from 33.6 ms to 16.6.

use std::sync::{Mutex, OnceLock};

use ggml_rs::{Backend, DeviceVec, QuantizedTensor, Tensor};

use crate::config::ModelConfig;
use crate::kv_cache::KvCache;
use crate::loader::{CommonTensors, FfnPair, Weight};

/// A dense model's tensors as a chained step reads them.
pub(crate) struct Dense<'a> {
    pub cfg: &'a ModelConfig,
    pub common: &'a CommonTensors,
    /// Per-head norms of each layer's q and k before RoPE (Qwen3's), `[head_dim]` weights each.
    pub qk_norms: Option<Vec<(&'a Tensor, &'a Tensor)>>,
}

/// The chained step's state on the device, made at the first step that can use one; None when the backend has no chain
/// or a weight a step reads is not on its device.
#[derive(Default)]
pub(crate) struct ChainDecoder {
    state: OnceLock<Option<ChainState>>,
}

impl std::fmt::Debug for ChainDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChainDecoder({})", match self.state.get() { None => "not yet", Some(None) => "none", Some(Some(_)) => "on the device" })
    }
}

/// A chained step's copy of a KV cache on the device, a buffer a layer (row `t`: its K `[n_kv, head_dim]` then its V),
/// `cap` rows of them, and the attention's output and scratch.
struct ChainKv {
    layers: Vec<DeviceVec>,
    cap: usize,
    out: DeviceVec,
    /// The [`KvCache::id`] the rows are a copy of (0: none).
    owner: u64,
}

struct ChainState {
    kv: Mutex<ChainKv>,
    rope_table: DeviceVec,
    attn_norms: Vec<DeviceVec>,
    ffn_norms: Vec<DeviceVec>,
    /// Qwen3's per-head q and k norms, a layer's each.
    qk_norms: Option<Vec<(DeviceVec, DeviceVec)>>,
    output_norm: DeviceVec,
    /// the residual stream, the normed input, q, k, v (and q and k normed), a projection's output, the fused gate-up,
    /// the SwiGLU, the logits
    x: DeviceVec,
    xn: DeviceVec,
    q: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    proj: DeviceVec,
    gate_up: DeviceVec,
    act: DeviceVec,
    logits: DeviceVec,
}

fn quant(w: &Weight) -> &QuantizedTensor {
    match w {
        Weight::Quant(q) => q,
        _ => unreachable!("the chain state checked every weight"),
    }
}

impl ChainDecoder {
    fn state(&self, backend: &dyn Backend, m: &Dense<'_>) -> Option<&ChainState> {
        self.state
            .get_or_init(|| {
                let chain = backend.chain()?;
                let held = |w: &Weight| match w {
                    Weight::Quant(q) if chain.holds(q) => Some(()),
                    _ => None,
                };
                for b in &m.common.blocks {
                    for w in [&b.attn_q, &b.attn_k, &b.attn_v, &b.attn_output, &b.ffn_down] {
                        held(w)?;
                    }
                    match &b.ffn_pair {
                        FfnPair::Fused(w) => held(w)?,
                        FfnPair::Split { .. } => return None,
                    }
                }
                held(&m.common.output)?;
                let cfg = m.cfg;
                if cfg.n_kv_heads == 0 || cfg.n_heads % cfg.n_kv_heads != 0 {
                    return None;
                }
                let upload = |t: &Tensor| {
                    let t = if t.is_device() { t.to_host() } else { t.clone() };
                    let v = chain.vec(t.numel());
                    chain.upload(&v, t.data());
                    v
                };
                let ff = m.common.blocks[0].ffn_pair.ff();
                let (d, qd, kvd) = (cfg.embedding_dim, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
                Some(ChainState {
                    kv: Mutex::new(ChainKv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0 }),
                    rope_table: chain.vec(cfg.head_dim),
                    attn_norms: m.common.blocks.iter().map(|b| upload(&b.attn_norm)).collect(),
                    ffn_norms: m.common.blocks.iter().map(|b| upload(&b.ffn_norm)).collect(),
                    qk_norms: m.qk_norms.as_ref().map(|norms| norms.iter().map(|(q, k)| (upload(q), upload(k))).collect()),
                    output_norm: upload(&m.common.output_norm),
                    x: chain.vec(d),
                    xn: chain.vec(d),
                    q: chain.vec(qd),
                    k: chain.vec(kvd),
                    v: chain.vec(kvd),
                    qn: chain.vec(qd),
                    kn: chain.vec(kvd),
                    proj: chain.vec(d),
                    gate_up: chain.vec(2 * ff),
                    act: chain.vec(ff),
                    logits: chain.vec(cfg.vocab_size),
                })
            })
            .as_ref()
    }

    /// One decode step of `token`, chained, if the backend can: the logits `[1, vocab]`; None leaves the step to the
    /// model's own path.
    pub(crate) fn step(&self, backend: &dyn Backend, m: &Dense<'_>, token: u32, kv: &mut KvCache) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() {
            return None;
        }
        let st = self.state(backend, m)?;
        let chain = backend.chain()?;
        let cfg = m.cfg;
        let (n_h, n_kv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let (kvd, row) = (n_kv * hd, 2 * n_kv * hd);
        let past = kv.len;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        // room for this step's row
        if g.cap < past + 1 {
            let cap = (past + 1).next_power_of_two().max(256);
            g.layers = (0..cfg.n_layers)
                .map(|l| match g.layers.get(l) {
                    Some(old) => chain.resize(old, cap * row),
                    None => chain.vec(cap * row),
                })
                .collect();
            g.out = chain.vec(chain.attention_out_len(n_h, hd, cap));
            g.cap = cap;
        }
        // the rows the host wrote since (all of them for a cache the copy is not of)
        let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
        if from < past {
            for l in 0..cfg.n_layers {
                // the host's buffers read in place (a copy of one was its whole capacity: 0.3 s for 512 rows)
                let (kh, vh) = (kv.k_buffer(l), kv.v_buffer(l));
                let (kown, vown);
                let kd = if kh.is_device() { kown = kh.to_host(); kown.data() } else { kh.data() };
                let vd = if vh.is_device() { vown = vh.to_host(); vown.data() } else { vh.data() };
                let mut rows = Vec::with_capacity((past - from) * row);
                for t in from..past {
                    rows.extend_from_slice(&kd[t * kvd..(t + 1) * kvd]);
                    rows.extend_from_slice(&vd[t * kvd..(t + 1) * kvd]);
                }
                chain.upload_at(&g.layers[l], from * row, &rows);
            }
        }
        g.owner = kv.id;
        let theta = cfg.rope_theta;
        let table: Vec<f32> = (0..hd / 2)
            .flat_map(|j| {
                let (s, c) = (past as f32 * theta.powf(-2.0 * j as f32 / hd as f32) / 1.0).sin_cos();
                [s, c]
            })
            .collect();
        chain.upload(&st.rope_table, &table);
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        let emb = backend.embed_lookup(&m.common.tok_embd, &[token], cfg.embedding_dim);
        let emb = if emb.is_device() { emb.to_host() } else { emb };
        chain.upload(&st.x, emb.data());
        let mut rec = chain.begin();
        for l in 0..cfg.n_layers {
            let b = &m.common.blocks[l];
            rec.rmsnorm(&st.x, &st.attn_norms[l], &st.xn, cfg.rms_eps);
            rec.matmul(quant(&b.attn_q), &st.xn, &st.q);
            rec.matmul(quant(&b.attn_k), &st.xn, &st.k);
            rec.matmul(quant(&b.attn_v), &st.xn, &st.v);
            let (q, k) = match &st.qk_norms {
                Some(norms) => {
                    rec.rmsnorm_rows(&st.q, &norms[l].0, &st.qn, n_h, cfg.rms_eps);
                    rec.rmsnorm_rows(&st.k, &norms[l].1, &st.kn, n_kv, cfg.rms_eps);
                    (&st.qn, &st.kn)
                }
                None => (&st.q, &st.k),
            };
            rec.rope(q, n_h, hd, &st.rope_table, neox);
            rec.rope(k, n_kv, hd, &st.rope_table, neox);
            rec.store(k, &g.layers[l], past * row);
            rec.store(&st.v, &g.layers[l], past * row + kvd);
            rec.attention(q, &g.layers[l], &g.out, n_h, n_kv, hd, 0, past + 1, g.cap, scale);
            rec.matmul(quant(&b.attn_output), &g.out, &st.proj);
            rec.add(&st.x, &st.proj);
            rec.rmsnorm(&st.x, &st.ffn_norms[l], &st.xn, cfg.rms_eps);
            let FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("the chain state checked the pair") };
            rec.matmul(quant(gu), &st.xn, &st.gate_up);
            rec.silu_mul_split(&st.gate_up, &st.act);
            rec.matmul(quant(&b.ffn_down), &st.act, &st.proj);
            rec.add(&st.x, &st.proj);
        }
        rec.rmsnorm(&st.x, &st.output_norm, &st.xn, cfg.rms_eps);
        rec.matmul(quant(&m.common.output), &st.xn, &st.logits);
        rec.read(&st.logits);
        for l in 0..cfg.n_layers {
            rec.read_range(&g.layers[l], past * row, row);
        }
        let mut got = rec.finish().into_iter();
        let logits = got.next().expect("the logits");
        // the step's K and V rows into the host's cache too, which the copy already holds
        for (l, kvrow) in got.enumerate() {
            let k = Tensor::from_vec(kvrow[..kvd].to_vec(), vec![1, n_kv, hd]);
            let v = Tensor::from_vec(kvrow[kvd..].to_vec(), vec![1, n_kv, hd]);
            kv.append(backend, l, &k, &v);
        }
        kv.commit(1);
        kv.dirty_from = usize::MAX;
        Some(Tensor::from_vec(logits, vec![1, cfg.vocab_size]))
    }
}
