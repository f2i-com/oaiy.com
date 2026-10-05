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

/// A dense model's tensors as a chained step reads them, and what its layers do that Llama's do not.
pub(crate) struct Dense<'a> {
    pub cfg: &'a ModelConfig,
    pub common: &'a CommonTensors,
    /// Per-head norms of each layer's q and k before RoPE (Qwen3's, Gemma 3's), `[head_dim]` weights each.
    pub qk_norms: Option<Vec<(&'a Tensor, &'a Tensor)>>,
    /// Norms of each layer's attention output and FFN output before their residuals (Gemma 3's).
    pub post_norms: Option<Vec<(&'a Tensor, &'a Tensor)>>,
    /// The FFN's gate through the tanh GELU (Gemma's GeGLU), else SiLU.
    pub gelu: bool,
    /// The embedding times this (Gemma's `sqrt(d)`).
    pub embed_scale: Option<f32>,
    /// Each layer's RoPE base, its per-frequency divisors and its sliding window (Gemma 3's local and global layers);
    /// empty: the config's base for every layer, unscaled, no window.
    pub layers: Vec<(f32, Option<&'a [f32]>, Option<usize>)>,
    /// The logits through `tanh(l / c) * c`.
    pub softcap: Option<f32>,
}

impl<'a> Dense<'a> {
    /// A model whose layers are Llama's.
    pub fn plain(cfg: &'a ModelConfig, common: &'a CommonTensors) -> Dense<'a> {
        Dense { cfg, common, qk_norms: None, post_norms: None, gelu: false, embed_scale: None, layers: Vec::new(), softcap: None }
    }

    /// Layer `l`'s RoPE base, divisors and window.
    fn layer(&self, l: usize) -> (f32, Option<&'a [f32]>, Option<usize>) {
        self.layers.get(l).copied().unwrap_or((self.cfg.rope_theta, None, None))
    }

    /// The distinct RoPEs of the layers (base and divisors), and each layer's among them.
    fn ropes(&self) -> (Vec<(f32, Option<&'a [f32]>)>, Vec<usize>) {
        let mut distinct: Vec<(f32, Option<&'a [f32]>)> = Vec::new();
        let which = (0..self.cfg.n_layers)
            .map(|l| {
                let (theta, factors, _) = self.layer(l);
                let same = |(t, f): &(f32, Option<&[f32]>)| t.to_bits() == theta.to_bits() && f.map(|f| f.as_ptr()) == factors.map(|f| f.as_ptr());
                match distinct.iter().position(same) {
                    Some(i) => i,
                    None => {
                        distinct.push((theta, factors));
                        distinct.len() - 1
                    }
                }
            })
            .collect();
        (distinct, which)
    }
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
    /// A sine-and-cosine table a distinct RoPE of the layers.
    rope_tables: Vec<DeviceVec>,
    /// Gemma 3's norms of a layer's attention and FFN outputs.
    post_norms: Option<Vec<(DeviceVec, DeviceVec)>>,
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
    /// a projection's output normed (Gemma 3's post norms)
    proj_n: DeviceVec,
    gate_up: DeviceVec,
    act: DeviceVec,
    logits: DeviceVec,
}

/// Room in the device's copy of the cache for `needed` rows (grown to a power of two, its rows kept).
fn reserve(chain: &dyn ggml_rs::DeviceChain, g: &mut ChainKv, cfg: &ModelConfig, needed: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * cfg.n_kv_heads * cfg.head_dim;
    let cap = needed.next_power_of_two().max(256);
    g.layers = (0..cfg.n_layers)
        .map(|l| match g.layers.get(l) {
            Some(old) => chain.resize(old, cap * row),
            None => chain.vec(cap * row),
        })
        .collect();
    g.out = chain.vec(chain.attention_out_len(cfg.n_heads, cfg.head_dim, cap));
    g.cap = cap;
}

/// Bring the device's copy of the cache up to `past` rows: the rows the host wrote since (all of them for a cache the
/// copy is not of).
fn sync(chain: &dyn ggml_rs::DeviceChain, g: &mut ChainKv, cfg: &ModelConfig, kv: &KvCache, past: usize) {
    let (kvd, row) = (cfg.n_kv_heads * cfg.head_dim, 2 * cfg.n_kv_heads * cfg.head_dim);
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
}

/// Each distinct RoPE's sines and cosines for positions `past..past + rows`, `[rows, head_dim]`, as the CPU's rope
/// makes them.
fn rope_tables(m: &Dense<'_>, past: usize, rows: usize) -> Vec<Vec<f32>> {
    let hd = m.cfg.head_dim;
    m.ropes()
        .0
        .iter()
        .map(|(theta, factors)| {
            (past..past + rows)
                .flat_map(|pos| {
                    (0..hd / 2).flat_map(move |j| {
                        let factor = factors.map(|f| f[j]).unwrap_or(1.0);
                        let (s, c) = (pos as f32 * theta.powf(-2.0 * j as f32 / hd as f32) / factor).sin_cos();
                        [s, c]
                    })
                })
                .collect()
        })
        .collect()
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
                    rope_tables: m.ropes().0.iter().map(|_| chain.vec(cfg.head_dim)).collect(),
                    post_norms: m.post_norms.as_ref().map(|norms| norms.iter().map(|(a, f)| (upload(a), upload(f))).collect()),
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
                    proj_n: chain.vec(d),
                    gate_up: chain.vec(2 * ff),
                    act: chain.vec(ff),
                    logits: chain.vec(cfg.vocab_size),
                })
            })
            .as_ref()
    }

    /// A prompt's chunk of `tokens`, chained in one submit, if the backend can: the last token's logits `[1, vocab]`;
    /// None leaves it to the model's own path. Every layer on the device as a decode step's is, its rows at once: RoPE
    /// from a table of the chunk's positions, the K and V stored into the device's copy of the cache, the causal
    /// attention over it (Gemma 3's local layers within their window); the logits and the chunk's K and V rows (for
    /// the host's cache) come back. Op by op, every projection's input went up and its output came back: of a 3B
    /// Llama's 2,000-token prompt (8.4 s), 3 s on the host making and reading those; a layer a submit with the
    /// attention on the host, 3.8 s, 2.4 of it the attention.
    pub(crate) fn prompt(&self, backend: &dyn Backend, m: &Dense<'_>, tokens: &[u32], kv: &mut KvCache) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || tokens.len() < 2 {
            return None;
        }
        let st = self.state(backend, m)?;
        let chain = backend.chain()?;
        let cfg = m.cfg;
        let t = tokens.len();
        let (n_h, n_kv, hd, d) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.embedding_dim);
        let (qd, kvd, row) = (n_h * hd, n_kv * hd, 2 * n_kv * hd);
        let ff = m.common.blocks[0].ffn_pair.ff();
        let past = kv.len;
        let scale = 1.0 / (hd as f32).sqrt();
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        reserve(chain, &mut g, cfg, past + t);
        sync(chain, &mut g, cfg, kv, past);
        let rope_of = m.ropes().1;
        let tables: Vec<DeviceVec> = rope_tables(m, past, t)
            .iter()
            .map(|values| {
                let table = chain.vec(values.len());
                chain.upload(&table, values);
                table
            })
            .collect();
        let [x, xn, q, k, v, qn, kn, proj, proj_n, gate_up, act] =
            [t * d, t * d, t * qd, t * kvd, t * kvd, t * qd, t * kvd, t * d, t * d, t * 2 * ff, t * ff].map(|len| chain.vec(len));
        let attn = chain.vec(chain.attention_rows_out_len(t, n_h, hd, past + t));
        let (last, logits) = (chain.vec(d), chain.vec(cfg.vocab_size));
        let emb = backend.embed_lookup(&m.common.tok_embd, tokens, cfg.embedding_dim);
        let mut emb = if emb.is_device() { emb.to_host() } else { emb };
        if let Some(s) = m.embed_scale {
            emb.data_mut().iter_mut().for_each(|e| *e *= s);
        }
        chain.upload(&x, emb.data());
        let mut rec = chain.begin();
        // the chunk's vectors are its own: no bind groups kept to hold their memory after it
        rec.keep_groups(false);
        for l in 0..cfg.n_layers {
            let b = &m.common.blocks[l];
            rec.rmsnorm_rows(&x, &st.attn_norms[l], &xn, t, cfg.rms_eps);
            rec.matmul_rows(quant(&b.attn_q), &xn, &q, t);
            rec.matmul_rows(quant(&b.attn_k), &xn, &k, t);
            rec.matmul_rows(quant(&b.attn_v), &xn, &v, t);
            let (qq, kk) = match &st.qk_norms {
                Some(norms) => {
                    rec.rmsnorm_rows(&q, &norms[l].0, &qn, t * n_h, cfg.rms_eps);
                    rec.rmsnorm_rows(&k, &norms[l].1, &kn, t * n_kv, cfg.rms_eps);
                    (&qn, &kn)
                }
                None => (&q, &k),
            };
            let table = &tables[rope_of[l]];
            rec.rope_rows(qq, t, n_h, hd, table, neox);
            rec.rope_rows(kk, t, n_kv, hd, table, neox);
            rec.store_rows(kk, &g.layers[l], t, kvd, past, row, 0);
            rec.store_rows(&v, &g.layers[l], t, kvd, past, row, kvd);
            rec.attention_rows(qq, &g.layers[l], &attn, t, n_h, n_kv, hd, past, m.layer(l).2, scale);
            rec.matmul_rows(quant(&b.attn_output), &attn, &proj, t);
            match &st.post_norms {
                Some(norms) => {
                    rec.rmsnorm_rows(&proj, &norms[l].0, &proj_n, t, cfg.rms_eps);
                    rec.add(&x, &proj_n);
                }
                None => rec.add(&x, &proj),
            }
            rec.rmsnorm_rows(&x, &st.ffn_norms[l], &xn, t, cfg.rms_eps);
            let FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("the chain state checked the pair") };
            rec.matmul_rows(quant(gu), &xn, &gate_up, t);
            if m.gelu {
                rec.gelu_mul_split_rows(&gate_up, &act, t);
            } else {
                rec.silu_mul_split_rows(&gate_up, &act, t);
            }
            rec.matmul_rows(quant(&b.ffn_down), &act, &proj, t);
            match &st.post_norms {
                Some(norms) => {
                    rec.rmsnorm_rows(&proj, &norms[l].1, &proj_n, t, cfg.rms_eps);
                    rec.add(&x, &proj_n);
                }
                None => rec.add(&x, &proj),
            }
        }
        // the head of the last token's row only
        rec.rmsnorm_rows(&x, &st.output_norm, &xn, t, cfg.rms_eps);
        rec.copy(&xn, (t - 1) * d, &last, 0, d);
        rec.matmul(quant(&m.common.output), &last, &logits);
        rec.read(&logits);
        for l in 0..cfg.n_layers {
            rec.read_range(&g.layers[l], past * row, t * row);
        }
        let mut got = rec.finish().into_iter();
        let mut out = got.next().expect("the logits");
        // the chunk's K and V rows into the host's cache too, which the copy already holds
        for (l, rows) in got.enumerate() {
            kv.append_rows(backend, l, &rows, t);
        }
        kv.commit(t);
        kv.dirty_from = usize::MAX;
        if let Some(c) = m.softcap {
            out.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
        Some(Tensor::from_vec(out, vec![1, cfg.vocab_size]))
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
        reserve(chain, &mut g, cfg, past + 1);
        sync(chain, &mut g, cfg, kv, past);
        // each RoPE's sines and cosines at this position
        let rope_of = m.ropes().1;
        for (values, table) in rope_tables(m, past, 1).iter().zip(&st.rope_tables) {
            chain.upload(table, values);
        }
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        let emb = backend.embed_lookup(&m.common.tok_embd, &[token], cfg.embedding_dim);
        let mut emb = if emb.is_device() { emb.to_host() } else { emb };
        if let Some(s) = m.embed_scale {
            emb.data_mut().iter_mut().for_each(|v| *v *= s);
        }
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
            let table = &st.rope_tables[rope_of[l]];
            rec.rope(q, n_h, hd, table, neox);
            rec.rope(k, n_kv, hd, table, neox);
            rec.store(k, &g.layers[l], past * row);
            rec.store(&st.v, &g.layers[l], past * row + kvd);
            // a sliding window sees the last `w` positions, as the CPU's attention does
            let lo = m.layer(l).2.map_or(0, |w| (past + 1).saturating_sub(w));
            rec.attention(q, &g.layers[l], &g.out, n_h, n_kv, hd, lo, past + 1, g.cap, scale);
            rec.matmul(quant(&b.attn_output), &g.out, &st.proj);
            match &st.post_norms {
                Some(norms) => {
                    rec.rmsnorm(&st.proj, &norms[l].0, &st.proj_n, cfg.rms_eps);
                    rec.add(&st.x, &st.proj_n);
                }
                None => rec.add(&st.x, &st.proj),
            }
            rec.rmsnorm(&st.x, &st.ffn_norms[l], &st.xn, cfg.rms_eps);
            let FfnPair::Fused(gu) = &b.ffn_pair else { unreachable!("the chain state checked the pair") };
            rec.matmul(quant(gu), &st.xn, &st.gate_up);
            if m.gelu {
                rec.gelu_mul_split(&st.gate_up, &st.act);
            } else {
                rec.silu_mul_split(&st.gate_up, &st.act);
            }
            rec.matmul(quant(&b.ffn_down), &st.act, &st.proj);
            match &st.post_norms {
                Some(norms) => {
                    rec.rmsnorm(&st.proj, &norms[l].1, &st.proj_n, cfg.rms_eps);
                    rec.add(&st.x, &st.proj_n);
                }
                None => rec.add(&st.x, &st.proj),
            }
        }
        rec.rmsnorm(&st.x, &st.output_norm, &st.xn, cfg.rms_eps);
        rec.matmul(quant(&m.common.output), &st.xn, &st.logits);
        rec.read(&st.logits);
        for l in 0..cfg.n_layers {
            rec.read_range(&g.layers[l], past * row, row);
        }
        let mut got = rec.finish().into_iter();
        let mut logits = got.next().expect("the logits");
        if let Some(c) = m.softcap {
            logits.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
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
