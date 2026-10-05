//! VENDORED-LOCAL: Qwen3.5's hybrid chained on the backend's device, a decode step or a prompt's chunk in one submit
//! ([`ggml_rs::chain`]), as `chain_decode` chains a dense model's: the gated delta net layers' projections, conv,
//! recurrence and norm-gated output; the attention layers' q and gate halves, per-head norms, partial RoPE, the K and V
//! stored into a copy of the attention cache kept on the device, attention and its sigmoid gate; every layer's FFN and
//! residuals; then the head. Only the logits and the attention layers' new K and V rows (for the host's cache) come
//! back. Op by op, every projection's input went up and its output came back, and the recurrence ran on the host.
//!
//! The recurrent state stays on the device between steps: the cache's state tensors are the chain's own vectors
//! ([`DeviceChain::alias`]), so what reads them there (a checkpoint, a conversation set aside, a disk state) reads them
//! back, and a state put there from the host (a restore, the host path's) is taken up again at the next run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml_rs::chain::DeltaNet;
use ggml_rs::{Backend, DeviceChain, DeviceVec, QuantizedTensor, Tensor};

use crate::kv_cache::KvCache;
use crate::loader::{FfnPair, Weight};
use crate::qwen35::{Qwen35Block, Qwen35Model};

/// Rows a submit takes at most (the server's prompt chunks are 512).
const MAX_ROWS: usize = 512;

/// The most a prompt's attention scratch may take: a chunk's rows are cut to fit it (at 16K positions, 24 heads of
/// 256 take 1.6 MB a row).
const ATTENTION_SCRATCH: usize = 512 << 20;

/// Qwen3.5's chained runs: the state on the device, made at the first run that can use one; none when the backend
/// has no chain or a weight a run reads is not on its device.
#[derive(Default)]
pub struct Qwen35Chain {
    state: OnceLock<Option<State>>,
    /// Runs the chain took (a step or a prompt's chunk), for a test that has to know it ran.
    pub runs: AtomicUsize,
}

impl std::fmt::Debug for Qwen35Chain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match self.state.get() {
            None => "not yet",
            Some(None) => "none",
            Some(Some(_)) => "on the device",
        };
        write!(f, "Qwen35Chain({state}, {} runs)", self.runs.load(Ordering::Relaxed))
    }
}

/// A layer's own vectors on the device.
struct LayerVecs {
    attn_norm: DeviceVec,
    post_norm: DeviceVec,
    mixer: Mixer,
}

enum Mixer {
    /// The per-head q and k norms, and which of the attention cache's layers is this one's.
    Attention { q_norm: DeviceVec, k_norm: DeviceVec, slot: usize },
    /// The conv's weights, `ssm_a`, `dt_bias`, the output norm, the fused beta-alpha projection's weights when they
    /// are f32 (Qwen3.8 27B's; else it is a quantized weight the device holds), and which recurrent state is this one's.
    Ssm { conv_w: DeviceVec, a: DeviceVec, dt: DeviceVec, norm: DeviceVec, ba_f32: Option<DeviceVec>, slot: usize },
}

/// The model's sizes.
#[derive(Clone, Copy)]
struct Dims {
    d: usize,
    n_h: usize,
    n_kv: usize,
    hd: usize,
    rot: usize,
    ff: usize,
    nv: usize,
    nk: usize,
    dk: usize,
    dv: usize,
    /// the conv's channels: q and k of every key head, v of every value head
    ch: usize,
    kern: usize,
    vocab: usize,
}

/// The chain's copy of the attention layers' caches, a buffer a layer (row `t`: its K `[n_kv, head_dim]` then its
/// V), `cap` rows of them, and a decode step's attention output and scratch.
struct Kv {
    layers: Vec<DeviceVec>,
    cap: usize,
    out: DeviceVec,
    /// The [`KvCache::id`] the rows are a copy of (0: none).
    owner: u64,
}

/// The vectors the cache's recurrent state tensors alias, a delta net layer's each: its state and its conv's last
/// inputs.
struct Pool {
    states: Vec<DeviceVec>,
    convs: Vec<DeviceVec>,
}

/// A run's vectors for its rows.
struct Work {
    x: DeviceVec,
    xn: DeviceVec,
    qkv: DeviceVec,
    z: DeviceVec,
    ba: DeviceVec,
    conv: DeviceVec,
    core: DeviceVec,
    proj: DeviceVec,
    qfull: DeviceVec,
    q: DeviceVec,
    gate: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    gated: DeviceVec,
    /// the fused gate-up's output, or the gate's
    ffa: DeviceVec,
    /// the up projection's (a split pair's)
    ffb: DeviceVec,
    act: DeviceVec,
    table: DeviceVec,
    last: DeviceVec,
}

impl Work {
    fn new(chain: &dyn DeviceChain, s: &Dims, t: usize) -> Work {
        let v = |n: usize| chain.vec(n);
        Work {
            x: v(t * s.d),
            xn: v(t * s.d),
            qkv: v(t * s.ch),
            z: v(t * s.nv * s.dv),
            ba: v(t * 2 * s.nv),
            conv: v(t * s.ch),
            core: v(t * s.nv * s.dv),
            proj: v(t * s.d),
            qfull: v(t * s.n_h * 2 * s.hd),
            q: v(t * s.n_h * s.hd),
            gate: v(t * s.n_h * s.hd),
            k: v(t * s.n_kv * s.hd),
            v: v(t * s.n_kv * s.hd),
            qn: v(t * s.n_h * s.hd),
            kn: v(t * s.n_kv * s.hd),
            gated: v(t * s.n_h * s.hd),
            ffa: v(t * 2 * s.ff),
            ffb: v(t * s.ff),
            act: v(t * s.ff),
            table: v(t * s.rot),
            last: v(s.d),
        }
    }
}

struct State {
    dims: Dims,
    layers: Vec<LayerVecs>,
    /// The layer of each of the attention cache's buffers.
    attention_layers: Vec<usize>,
    /// The layer of each recurrent state.
    ssm_layers: Vec<usize>,
    output_norm: DeviceVec,
    kv: Mutex<Kv>,
    pool: Mutex<Pool>,
    /// A decode step's vectors, kept (and so their bind groups).
    step: Work,
    logits: DeviceVec,
}

fn quant(w: &Weight) -> &QuantizedTensor {
    match w {
        Weight::Quant(q) => q,
        _ => unreachable!("the chain state checked every weight"),
    }
}

/// The partial RoPE's sines and cosines for positions `past..past + rows`, `[rows, rot]`, as the CPU's
/// `rope_partial_neox` makes them.
fn rope_table(theta: f32, rot: usize, past: usize, rows: usize) -> Vec<f32> {
    (past..past + rows)
        .flat_map(|pos| {
            (0..rot / 2).flat_map(move |k| {
                let (s, c) = (pos as f32 * theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                [s, c]
            })
        })
        .collect()
}

/// Room in the chain's copy of the attention cache for `needed` rows (grown to a power of two, its rows kept).
fn reserve(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, slots: usize, needed: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let cap = needed.next_power_of_two().max(256);
    g.layers = (0..slots)
        .map(|i| match g.layers.get(i) {
            Some(old) => chain.resize(old, cap * row),
            None => chain.vec(cap * row),
        })
        .collect();
    g.out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
    g.cap = cap;
}

/// Bring the chain's copy of the attention cache up to `past` rows: the rows the host wrote since (all of them for a
/// cache the copy is not of).
fn sync(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, layers: &[usize], kv: &KvCache, past: usize) {
    let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
    let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
    if from < past {
        for (slot, &l) in layers.iter().enumerate() {
            let (kh, vh) = (kv.k_buffer(l), kv.v_buffer(l));
            let (kown, vown);
            let kd = if kh.is_device() { kown = kh.to_host(); kown.data() } else { kh.data() };
            let vd = if vh.is_device() { vown = vh.to_host(); vown.data() } else { vh.data() };
            let mut rows = Vec::with_capacity((past - from) * row);
            for t in from..past {
                rows.extend_from_slice(&kd[t * kvd..(t + 1) * kvd]);
                rows.extend_from_slice(&vd[t * kvd..(t + 1) * kvd]);
            }
            chain.upload_at(&g.layers[slot], from * row, &rows);
        }
    }
    g.owner = kv.id;
}

/// The vector behind a recurrent tensor of the cache, `len` long: the one it aliases, or (a tensor the host put
/// there, or none) the pool's, taken up with its values (zero for none) and aliased by the cache from now on. The
/// pool's vector is used again only when no other tensor aliases it (another cache's), else a new one replaces it.
fn adopt(chain: &dyn DeviceChain, pool: &mut DeviceVec, slot: &mut Option<Tensor>, shape: Vec<usize>) -> DeviceVec {
    let len: usize = shape.iter().product();
    if let Some(v) = slot.as_ref().and_then(|t| chain.aliased(t)).filter(|v| v.len == len) {
        return v;
    }
    let host = slot.take().map(|t| t.to_host());
    if Arc::strong_count(&pool.inner) > 1 || pool.len != len {
        *pool = chain.vec(len);
    }
    let v = pool.clone();
    match host {
        Some(h) => {
            assert_eq!(h.numel(), len, "a recurrent state of {:?} where the model's is {shape:?}", h.shape());
            chain.upload(&v, h.data());
        }
        None => chain.zero(&v),
    }
    *slot = Some(chain.alias(&v, shape));
    v
}

impl Qwen35Chain {
    fn state(&self, m: &Qwen35Model) -> Option<&State> {
        self.state
            .get_or_init(|| {
                let chain = m.backend.chain()?;
                let cfg = &m.config;
                let sc = &m.ssm_cfg;
                let held = |w: &Weight| matches!(w, Weight::Quant(q) if chain.holds(q));
                let (nv, nk, dk) = (sc.time_step_rank, sc.group_count, sc.state_size);
                let dv = if nv == 0 { 0 } else { sc.inner_size / nv };
                let (n_h, n_kv, hd, d) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.embedding_dim);
                if nv == 0 || nk == 0 || dk != dv || ![16, 32, 64, 128].contains(&dk) || !(2..=8).contains(&sc.conv_kernel) {
                    return None;
                }
                if n_kv == 0 || n_h % n_kv != 0 || cfg.rope_dim == 0 || cfg.rope_dim > hd || cfg.rope_dim % 2 != 0 || !held(&m.output) {
                    return None;
                }
                let ch = 2 * nk * dk + nv * dv;
                let ff = m.blocks.first().map(|b| match b {
                    Qwen35Block::Attention { ffn_pair, .. } | Qwen35Block::Ssm { ffn_pair, .. } => ffn_pair.ff(),
                })?;
                let pair_held = |p: &FfnPair| match p {
                    FfnPair::Fused(w) => held(w),
                    FfnPair::Split { gate, up } => held(gate) && held(up),
                };
                for b in &m.blocks {
                    let ok = match b {
                        Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, .. } => {
                            [attn_q, attn_k, attn_v, attn_output, ffn_down].into_iter().all(held)
                                && attn_q.shape()[0] == 2 * n_h * hd
                                && pair_held(ffn_pair)
                                && ffn_pair.ff() == ff
                        }
                        Qwen35Block::Ssm { attn_qkv, attn_gate, ssm_ba, ssm_out, ffn_pair, ffn_down, ssm_conv1d, .. } => {
                            [attn_qkv, attn_gate, ssm_out, ffn_down].into_iter().all(held)
                                && attn_qkv.shape()[0] == ch
                                && attn_gate.shape()[0] == nv * dv
                                && ssm_conv1d.numel() == ch * sc.conv_kernel
                                && (held(ssm_ba) || matches!(ssm_ba, Weight::Dense(t) if t.shape() == [2 * nv, d]))
                                && pair_held(ffn_pair)
                                && ffn_pair.ff() == ff
                        }
                    };
                    if !ok {
                        return None;
                    }
                }
                let upload = |t: &Tensor| {
                    let t = if t.is_device() { t.to_host() } else { t.clone() };
                    let v = chain.vec(t.numel());
                    chain.upload(&v, t.data());
                    v
                };
                let dims = Dims { d, n_h, n_kv, hd, rot: cfg.rope_dim, ff, nv, nk, dk, dv, ch, kern: sc.conv_kernel, vocab: cfg.vocab_size };
                let (mut attention_layers, mut ssm_layers) = (Vec::new(), Vec::new());
                let layers = m
                    .blocks
                    .iter()
                    .enumerate()
                    .map(|(l, b)| match b {
                        Qwen35Block::Attention { attn_norm, attn_q_norm, attn_k_norm, post_norm, .. } => {
                            attention_layers.push(l);
                            LayerVecs {
                                attn_norm: upload(attn_norm),
                                post_norm: upload(post_norm),
                                mixer: Mixer::Attention { q_norm: upload(attn_q_norm), k_norm: upload(attn_k_norm), slot: attention_layers.len() - 1 },
                            }
                        }
                        Qwen35Block::Ssm { attn_norm, ssm_conv1d, ssm_a, ssm_ba, ssm_dt_bias, ssm_norm, post_norm, .. } => {
                            ssm_layers.push(l);
                            let ba_f32 = match ssm_ba {
                                Weight::Dense(t) => Some(upload(t)),
                                _ => None,
                            };
                            LayerVecs {
                                attn_norm: upload(attn_norm),
                                post_norm: upload(post_norm),
                                mixer: Mixer::Ssm { conv_w: upload(ssm_conv1d), a: upload(ssm_a), dt: upload(ssm_dt_bias), norm: upload(ssm_norm), ba_f32, slot: ssm_layers.len() - 1 },
                            }
                        }
                    })
                    .collect();
                let pool = Pool {
                    states: ssm_layers.iter().map(|_| chain.vec(nv * dk * dv)).collect(),
                    convs: ssm_layers.iter().map(|_| chain.vec((sc.conv_kernel - 1) * ch)).collect(),
                };
                Some(State {
                    dims,
                    layers,
                    attention_layers,
                    ssm_layers,
                    output_norm: upload(&m.output_norm),
                    kv: Mutex::new(Kv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0 }),
                    pool: Mutex::new(pool),
                    step: Work::new(chain, &dims, 1),
                    logits: chain.vec(cfg.vocab_size),
                })
            })
            .as_ref()
    }

    /// Run `rows` tokens (`embeds` `[rows, d]`, on the host) after what `kv` holds, chained, if the backend can: the
    /// last token's logits `[1, vocab]`; None leaves them to the model's own path. A run of more rows than a submit
    /// takes goes in chunks.
    pub(crate) fn forward(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || rows == 0 || embeds.numel() != rows * m.config.embedding_dim {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        let emb_own;
        let emb = if embeds.is_device() {
            emb_own = embeds.to_host();
            emb_own.data()
        } else {
            embeds.data()
        };
        let s = st.dims;
        let mut logits = None;
        let mut at = 0;
        while at < rows {
            // a chunk's rows, as many as the attention's scratch has room for over the positions they reach
            let runs = (kv.len + rows).div_ceil(256).max(1);
            let per_row = (s.n_h * runs * (s.hd + 2) + s.n_h * s.hd) * 4;
            let t = (ATTENTION_SCRATCH / per_row).clamp(1, MAX_ROWS).min(rows - at);
            logits = Some(self.run(m, st, chain, &emb[at * s.d..(at + t) * s.d], t, kv));
            at += t;
        }
        logits
    }

    /// One submit of `t` rows.
    fn run(&self, m: &Qwen35Model, st: &State, chain: &dyn DeviceChain, emb: &[f32], t: usize, kv: &mut KvCache) -> Tensor {
        let s = st.dims;
        let cfg = &m.config;
        let backend: &dyn Backend = &*m.backend;
        let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
        let past = kv.len;
        let eps = cfg.rms_eps;
        let scale = 1.0 / (s.hd as f32).sqrt();
        let delta = DeltaNet { rows: t, v_heads: s.nv, k_heads: s.nk, k_dim: s.dk, v_dim: s.dv, scale_q: 1.0 / (s.dv as f32).sqrt(), eps, sigmoid_gate: false };
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        reserve(chain, &mut g, &s, st.attention_layers.len(), past + t);
        sync(chain, &mut g, &s, &st.attention_layers, kv, past);
        // the recurrent states the cache holds, as the chain's vectors
        let mut pool = st.pool.lock().unwrap_or_else(|p| p.into_inner());
        let mut states = Vec::with_capacity(st.ssm_layers.len());
        for (i, &l) in st.ssm_layers.iter().enumerate() {
            let state = adopt(chain, &mut pool.states[i], &mut kv.ssm_state[l], vec![s.nv, s.dv, s.dk]);
            let conv = adopt(chain, &mut pool.convs[i], &mut kv.ssm_conv[l], vec![s.kern - 1, s.ch]);
            states.push((state, conv));
        }
        drop(pool);
        let prompt;
        let w = if t == 1 {
            &st.step
        } else {
            prompt = Work::new(chain, &s, t);
            &prompt
        };
        let attn = if t == 1 { g.out.clone() } else { chain.vec(chain.attention_rows_out_len(t, s.n_h, s.hd, past + t)) };
        chain.upload(&w.table, &rope_table(cfg.rope_theta, s.rot, past, t));
        chain.upload(&w.x, emb);
        let mut rec = chain.begin();
        // a prompt's vectors are its own: no bind groups kept to hold them
        rec.keep_groups(t == 1);
        for (l, (b, lv)) in m.blocks.iter().zip(&st.layers).enumerate() {
            rec.rmsnorm_rows(&w.x, &lv.attn_norm, &w.xn, t, eps);
            let (ffn_pair, ffn_down) = match (b, &lv.mixer) {
                (Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, .. }, Mixer::Attention { q_norm, k_norm, slot }) => {
                    let kvl = &g.layers[*slot];
                    rec.matmul_rows(quant(attn_q), &w.xn, &w.qfull, t);
                    rec.matmul_rows(quant(attn_k), &w.xn, &w.k, t);
                    rec.matmul_rows(quant(attn_v), &w.xn, &w.v, t);
                    // each head's q is its query then its gate
                    rec.copy_cols(&w.qfull, &w.q, t * s.n_h, s.hd, 2 * s.hd, 0);
                    rec.copy_cols(&w.qfull, &w.gate, t * s.n_h, s.hd, 2 * s.hd, s.hd);
                    rec.rmsnorm_rows(&w.q, q_norm, &w.qn, t * s.n_h, eps);
                    rec.rmsnorm_rows(&w.k, k_norm, &w.kn, t * s.n_kv, eps);
                    rec.rope_partial_rows(&w.qn, t, s.n_h, s.hd, s.rot, &w.table);
                    rec.rope_partial_rows(&w.kn, t, s.n_kv, s.hd, s.rot, &w.table);
                    rec.store_rows(&w.kn, kvl, t, kvd, past, row, 0);
                    rec.store_rows(&w.v, kvl, t, kvd, past, row, kvd);
                    if t == 1 {
                        rec.attention(&w.qn, kvl, &attn, s.n_h, s.n_kv, s.hd, 0, past + 1, g.cap, scale);
                    } else {
                        rec.attention_rows(&w.qn, kvl, &attn, t, s.n_h, s.n_kv, s.hd, past, None, scale);
                    }
                    rec.mul_sigmoid(&attn, &w.gate, &w.gated, t * s.n_h * s.hd);
                    rec.matmul_rows(quant(attn_output), &w.gated, &w.proj, t);
                    (ffn_pair, ffn_down)
                }
                (Qwen35Block::Ssm { attn_qkv, attn_gate, ssm_ba, ssm_out, ffn_pair, ffn_down, .. }, Mixer::Ssm { conv_w, a, dt, norm, ba_f32, slot }) => {
                    let (state, conv) = &states[*slot];
                    rec.matmul_rows(quant(attn_qkv), &w.xn, &w.qkv, t);
                    rec.matmul_rows(quant(attn_gate), &w.xn, &w.z, t);
                    match ba_f32 {
                        Some(wf) => rec.matmul_f32_rows(wf, 2 * s.nv, s.d, &w.xn, &w.ba, t),
                        None => rec.matmul_rows(quant(ssm_ba), &w.xn, &w.ba, t),
                    }
                    rec.ssm_conv(&w.qkv, conv_w, conv, &w.conv, t, s.ch, s.kern);
                    rec.delta_net(&w.conv, &w.z, &w.ba, a, dt, norm, state, &w.core, delta);
                    rec.matmul_rows(quant(ssm_out), &w.core, &w.proj, t);
                    (ffn_pair, ffn_down)
                }
                _ => unreachable!("layer {l}'s vectors are its block's"),
            };
            rec.add(&w.x, &w.proj);
            rec.rmsnorm_rows(&w.x, &lv.post_norm, &w.xn, t, eps);
            match ffn_pair {
                FfnPair::Fused(gu) => {
                    rec.matmul_rows(quant(gu), &w.xn, &w.ffa, t);
                    rec.silu_mul_split_rows(&w.ffa, &w.act, t);
                }
                FfnPair::Split { gate, up } => {
                    rec.matmul_rows(quant(gate), &w.xn, &w.ffa, t);
                    rec.matmul_rows(quant(up), &w.xn, &w.ffb, t);
                    rec.silu_mul(&w.ffa, &w.ffb, &w.act, t * s.ff);
                }
            }
            rec.matmul_rows(quant(ffn_down), &w.act, &w.proj, t);
            rec.add(&w.x, &w.proj);
        }
        // the head of the last row only
        rec.rmsnorm_rows(&w.x, &st.output_norm, &w.xn, t, eps);
        let last = if t == 1 {
            &w.xn
        } else {
            rec.copy(&w.xn, (t - 1) * s.d, &w.last, 0, s.d);
            &w.last
        };
        rec.matmul(quant(&m.output), last, &st.logits);
        rec.read(&st.logits);
        for slot in 0..st.attention_layers.len() {
            rec.read_range(&g.layers[slot], past * row, t * row);
        }
        let mut got = rec.finish().into_iter();
        let logits = got.next().expect("the logits");
        // the run's K and V rows into the host's cache too, which the copy already holds
        for (&l, rows) in st.attention_layers.iter().zip(got) {
            let (mut kh, mut vh) = (Vec::with_capacity(t * kvd), Vec::with_capacity(t * kvd));
            for r in rows.chunks_exact(row) {
                kh.extend_from_slice(&r[..kvd]);
                vh.extend_from_slice(&r[kvd..]);
            }
            kv.append(backend, l, &Tensor::from_vec(kh, vec![t, s.n_kv, s.hd]), &Tensor::from_vec(vh, vec![t, s.n_kv, s.hd]));
        }
        kv.commit(t);
        kv.dirty_from = usize::MAX;
        self.runs.fetch_add(1, Ordering::Relaxed);
        Tensor::from_vec(logits, vec![1, s.vocab])
    }
}
