//! Gemma 4's dense layers chained on the backend's device ([`ggml_rs::chain`]), as
//! [`crate::llm::Gemma4Model::forward_embeds`] computes them op by op: the per-layer inputs (the embeddings through the PLE
//! projection and its norm, plus the tokens' rows of the PLE table, decoded on the host), then every layer's norm, q
//! (its per-head norm, RoPE by the layer's own base and divisors), k and v where the layer keeps its own (k normed, v
//! normed without a weight, both stored into a copy of the cache on the device; the last layers read an earlier
//! layer's), attention at scale 1 (the sliding layers' within their window), the output projection and its norm, the
//! GeGLU FFN and its norm, the PLE gate (`gelu(W x) * the layer's input`, projected back and normed) and the layer's
//! output scale; then the head. A prompt's chunk or a decode step is one submit; the logits and the new K and V rows
//! (for the host's cache) come back.
//!
//! Op by op, every op went to the host and most of a step's time was round trips: Gemma 4 E2B replied at 5.7 tokens a
//! second on an NVIDIA A2, and a 21-token prompt took 3.6 s.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use ggml_rs::{Backend, ChainRecorder, DeviceChain, DeviceVec, Tensor};

use crate::llm::arch::gemma4::Gemma4Model;
use crate::llm::kv_cache::KvCache;
use crate::llm::loader::{FfnPair, Weight};

/// A weight as a chained step holds it: the model's packed one (which the device holds), or a dense one uploaded
/// here, as f16 (`[n, k]`; Gemma 4 E2B's PLE gates and projections are f32 in its GGUF, 3 MB a layer read at every
/// token) or as f32 where a value is past f16's range.
enum Held {
    Packed,
    F16 { w: DeviceVec, n: usize, k: usize },
    F32 { w: DeviceVec, n: usize, k: usize },
}

impl Held {
    /// None for a weight the device cannot take.
    fn of(chain: &dyn DeviceChain, w: &Weight) -> Option<Held> {
        match w {
            Weight::Quant(q) if chain.holds(q) => Some(Held::Packed),
            Weight::Dense(t) if t.rank() == 2 => {
                let (values, n, k) = (host(t), t.dim(0), t.dim(1));
                Some(match chain.vec_f16_rounded(&values).filter(|_| k % 2 == 0 && std::env::var_os("OAIY_F16_EXACT").is_none()) {
                    Some(w) => Held::F16 { w, n, k },
                    None => Held::F32 { w: vector(chain, &values), n, k },
                })
            }
            _ => None,
        }
    }

    /// `y[r] = W x[r]` for `rows` rows (`one`: a decode step's single row, by the matrix-vector kernels).
    fn rows(&self, w: &Weight, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize, one: bool) {
        match (self, w) {
            (Held::Packed, Weight::Quant(q)) if one => r.matmul(q, x, y),
            (Held::Packed, Weight::Quant(q)) => r.matmul_rows(q, x, y, rows),
            (Held::Packed, _) => unreachable!("the chain state checked every weight"),
            _ => self.own_rows(r, x, y, rows),
        }
    }

    /// [`Self::rows`] of a weight uploaded here: a prompt's rows on the tensor cores (its inputs scaled into f16's
    /// range), a step's row exactly.
    fn own_rows(&self, r: &mut dyn ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match self {
            Held::F16 { w, n, k } if rows > 1 => r.matmul_f16_rows_ranged(w, *n, *k, x, y, rows),
            Held::F16 { w, n, k } => r.matmul_f16_rows_f32(w, *n, *k, x, y, rows),
            Held::F32 { w, n, k } => r.matmul_f32_rows(w, *n, *k, x, y, rows),
            Held::Packed => unreachable!("a packed weight is the model's"),
        }
    }
}

fn vector(chain: &dyn DeviceChain, values: &[f32]) -> DeviceVec {
    let v = chain.vec(values.len().max(1));
    chain.upload(&v, values);
    v
}

fn host(t: &Tensor) -> Vec<f32> {
    if t.is_device() { t.to_host().data().to_vec() } else { t.data().to_vec() }
}

struct Layer {
    attn_norm: DeviceVec,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    post_attn: DeviceVec,
    ffn_norm: DeviceVec,
    post_ffn: DeviceVec,
    q: Held,
    /// The layer's own k and v, None where it reads an earlier layer's cache.
    kv: Option<(Held, Held)>,
    o: Held,
    /// The fused gate and up, or the two.
    ffn: Ffn,
    down: Held,
    ff: usize,
    /// The PLE gate, its projection back and that one's norm.
    ple: Option<(Held, Held, DeviceVec)>,
    scale: Option<f32>,
    hd: usize,
    /// Its KV heads (a full-attention layer may keep fewer than the sliding ones: one, in a 12B).
    n_kv: usize,
    /// The layer whose cache its attention reads (its own, or the earlier layer it shares).
    src: usize,
    rope: usize,
    window: Option<usize>,
}

enum Ffn {
    Fused(Held),
    Split(Held, Held),
}

/// The device's copy of a cache: a buffer for each layer that keeps its own (row `t`: its K `[n_kv, hd]` then V).
struct Kv {
    layers: Vec<Option<DeviceVec>>,
    cap: usize,
    out: DeviceVec,
    /// The [`KvCache::id`] the rows are a copy of (0: none).
    owner: u64,
}

struct State {
    layers: Vec<Layer>,
    /// `per_layer_model_proj` over `sqrt(d)`, `[n_layers * pld, d]`.
    ple_proj: Option<Held>,
    /// The PLE projection's norm over `sqrt(2)` (the inputs' scale folded in).
    ple_norm: DeviceVec,
    /// Ones, a V norm's weight.
    ones: DeviceVec,
    output_norm: DeviceVec,
    /// The distinct RoPEs (base, divisors, head dim).
    ropes: Vec<(f32, Option<Vec<f32>>, usize)>,
    kv: Mutex<Kv>,
    /// A decode step's vectors, made at the first.
    step: Mutex<Option<Bufs>>,
}

/// The chained path's state, made at the first forward that can use one; None when the backend has no chain or a
/// weight is not on its device.
#[derive(Default)]
pub struct Gemma4Chain {
    state: OnceLock<Option<State>>,
}

impl std::fmt::Debug for Gemma4Chain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gemma4Chain({})", match self.state.get() { None => "not yet", Some(None) => "none", Some(Some(_)) => "on the device" })
    }
}

impl State {
    fn of(m: &Gemma4Model, chain: &dyn DeviceChain) -> Option<State> {
        let cfg = &m.config;
        let packed = |w: &Weight| match w {
            Weight::Quant(q) if chain.holds(q) => Some(Held::Packed),
            _ => None,
        };
        packed(&m.common.output)?;
        let pld = m.per_layer_dim();
        let mut ropes: Vec<(f32, Option<Vec<f32>>, usize)> = Vec::new();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let b = &m.common.blocks[l];
            let gx = &m.extras[l];
            let hd = cfg.layer_head_dim(l);
            let swa = cfg.layer_uses_sliding_window(l);
            let factors = if swa { None } else { m.rope_freq_factors.clone() };
            let theta = cfg.layer_rope_theta(l);
            let same = |r: &(f32, Option<Vec<f32>>, usize)| r.0.to_bits() == theta.to_bits() && r.1 == factors && r.2 == hd;
            let rope = match ropes.iter().position(same) {
                Some(i) => i,
                None => {
                    ropes.push((theta, factors, hd));
                    ropes.len() - 1
                }
            };
            let src = m.kv_source(l);
            let ffn = match &b.ffn_pair {
                FfnPair::Fused(w) => Ffn::Fused(Held::of(chain, w)?),
                FfnPair::Split { gate, up } => Ffn::Split(Held::of(chain, gate)?, Held::of(chain, up)?),
            };
            let ple = match (&gx.per_layer_inp_gate, &gx.per_layer_proj, &gx.per_layer_post_norm) {
                (Some(g), Some(p), Some(n)) if pld > 0 => Some((Held::of(chain, g)?, Held::of(chain, p)?, vector(chain, &host(n)))),
                _ => None,
            };
            layers.push(Layer {
                attn_norm: vector(chain, &host(&b.attn_norm)),
                q_norm: vector(chain, &host(&gx.attn_q_norm)),
                k_norm: vector(chain, &host(&gx.attn_k_norm)),
                post_attn: vector(chain, &host(&gx.attn_post_norm)),
                ffn_norm: vector(chain, &host(&b.ffn_norm)),
                post_ffn: vector(chain, &host(&gx.ffn_post_norm)),
                q: packed(&b.attn_q)?,
                kv: if src == l { Some((packed(&b.attn_k)?, packed(&b.attn_v)?)) } else { None },
                o: packed(&b.attn_output)?,
                ffn,
                down: packed(&b.ffn_down)?,
                ff: b.ffn_pair.ff(),
                ple,
                scale: gx.layer_output_scale,
                hd,
                n_kv: cfg.layer_kv_heads(l),
                src,
                rope,
                window: if swa { cfg.sliding_window } else { None },
            });
        }
        // the per-layer inputs: the projection pre-scaled by 1 / sqrt(d), the norm's weight by 1 / sqrt(2)
        let (ple_proj, ple_norm) = match &m.global {
            Some(g) if pld > 0 => {
                let w = host(&g.per_layer_model_proj);
                let s = 1.0 / (cfg.embedding_dim as f32).sqrt();
                let w: Vec<f32> = w.iter().map(|v| v * s).collect();
                let n: Vec<f32> = host(&g.per_layer_proj_norm).iter().map(|v| v * std::f32::consts::FRAC_1_SQRT_2).collect();
                let w = Weight::Dense(Tensor::from_vec(w, vec![cfg.n_layers * pld, cfg.embedding_dim]));
                (Some(Held::of(chain, &w)?), vector(chain, &n))
            }
            _ => (None, vector(chain, &[0.0])),
        };
        let max_hd = layers.iter().map(|l| l.hd).max().unwrap_or(1);
        Some(State {
            layers,
            ple_proj,
            ple_norm,
            ones: vector(chain, &vec![1.0; max_hd]),
            output_norm: vector(chain, &host(&m.common.output_norm)),
            ropes,
            kv: Mutex::new(Kv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0 }),
            step: Mutex::new(None),
        })
    }

    /// Each distinct RoPE's sines and cosines for positions `past..past + rows`, `[rows, hd]`, as the CPU's rope makes
    /// them.
    fn tables(&self, past: usize, rows: usize) -> Vec<Vec<f32>> {
        self.ropes
            .iter()
            .map(|(theta, factors, hd)| {
                (past..past + rows)
                    .flat_map(|pos| {
                        (0..hd / 2).flat_map(move |j| {
                            let factor = factors.as_ref().map(|f| f[j]).unwrap_or(1.0);
                            let (s, c) = (pos as f32 * theta.powf(-2.0 * j as f32 / *hd as f32) / factor).sin_cos();
                            [s, c]
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

/// A run's vectors for `t` rows: the residual stream and its normed copy, a projection's output and its normed copy;
/// q, k, v and their normed copies a head width each (a norm of rows reads its rows' width off its vectors); the FFN's
/// fused gate and up and its output a width each (Gemma 4's later layers' are twice as wide), and the two halves where
/// they are apart; the logits and the last row; the RoPE tables; the per-layer inputs and the PLE's scratch.
struct Bufs {
    xs: DeviceVec,
    xn: DeviceVec,
    proj: DeviceVec,
    proj_n: DeviceVec,
    heads: BTreeMap<usize, [DeviceVec; 6]>,
    ffn: BTreeMap<usize, (DeviceVec, DeviceVec, Option<(DeviceVec, DeviceVec)>)>,
    logits: DeviceVec,
    last: DeviceVec,
    tables: Vec<DeviceVec>,
    pli: Option<DeviceVec>,
    ple: Option<(DeviceVec, DeviceVec, DeviceVec, DeviceVec, DeviceVec, DeviceVec)>,
}

impl Bufs {
    fn new(chain: &dyn DeviceChain, st: &State, m: &Gemma4Model, t: usize) -> Bufs {
        let cfg = &m.config;
        let (d, n_h, pld) = (cfg.embedding_dim, cfg.n_heads, m.per_layer_dim());
        let v = |len: usize| chain.vec(len.max(1));
        let mut heads = BTreeMap::new();
        let mut ffn = BTreeMap::new();
        for l in &st.layers {
            let hd = l.hd;
            // k's and v's vectors as wide as the most KV heads a layer of this head width keeps
            let n_kv = st.layers.iter().filter(|o| o.hd == hd).map(|o| o.n_kv).max().unwrap_or(l.n_kv);
            heads.entry(hd).or_insert_with(|| [t * n_h * hd, t * n_kv * hd, t * n_kv * hd, t * n_h * hd, t * n_kv * hd, t * n_kv * hd].map(|len| v(len)));
            ffn.entry(l.ff).or_insert_with(|| {
                let split = matches!(l.ffn, Ffn::Split(..)).then(|| (v(t * l.ff), v(t * l.ff)));
                (v(t * 2 * l.ff), v(t * l.ff), split)
            });
        }
        let ple = st.ple_proj.is_some().then(|| (v(t * cfg.n_layers * pld), v(t * cfg.n_layers * pld), v(t * pld), v(t * 2 * pld), v(t * pld), v(t * pld)));
        Bufs {
            xs: v(t * d),
            xn: v(t * d),
            proj: v(t * d),
            proj_n: v(t * d),
            heads,
            ffn,
            logits: v(cfg.vocab_size),
            last: v(d),
            tables: st.ropes.iter().map(|(_, _, hd)| v(t * hd)).collect(),
            pli: ple.as_ref().map(|_| v(t * cfg.n_layers * pld)),
            ple,
        }
    }
}

impl Kv {
    /// Room for `needed` rows (grown to a power of two, the rows kept), and the attention's scratch for that many.
    fn reserve(&mut self, chain: &dyn DeviceChain, st: &State, n_h: usize, needed: usize) {
        if self.cap >= needed && self.layers.len() == st.layers.len() {
            return;
        }
        let cap = needed.next_power_of_two().max(256);
        self.layers = st
            .layers
            .iter()
            .enumerate()
            .map(|(l, layer)| {
                layer.kv.as_ref().map(|_| {
                    let row = 2 * layer.n_kv * layer.hd;
                    match self.layers.get(l).and_then(|v| v.as_ref()) {
                        Some(old) => chain.resize(old, cap * row),
                        None => chain.vec(cap * row),
                    }
                })
            })
            .collect();
        let max_hd = st.layers.iter().map(|l| l.hd).max().unwrap_or(1);
        self.out = chain.vec(chain.attention_out_len(n_h, max_hd, cap));
        self.cap = cap;
    }

    /// The device's copy brought up to `past` rows: the rows the host wrote since (all of them for another cache).
    fn sync(&mut self, chain: &dyn DeviceChain, st: &State, kv: &KvCache, past: usize) {
        let from = if self.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
        if from < past {
            for (l, layer) in st.layers.iter().enumerate() {
                let Some(buf) = self.layers[l].as_ref() else { continue };
                let kvd = layer.n_kv * layer.hd;
                let (kh, vh) = (kv.k_buffer(l), kv.v_buffer(l));
                let (kown, vown);
                let kd = if kh.is_device() { kown = kh.to_host(); kown.data() } else { kh.data() };
                let vd = if vh.is_device() { vown = vh.to_host(); vown.data() } else { vh.data() };
                let mut rows = Vec::with_capacity((past - from) * 2 * kvd);
                for t in from..past {
                    rows.extend_from_slice(&kd[t * kvd..(t + 1) * kvd]);
                    rows.extend_from_slice(&vd[t * kvd..(t + 1) * kvd]);
                }
                chain.upload_at(buf, from * 2 * kvd, &rows);
            }
        }
        self.owner = kv.id;
    }
}

impl Gemma4Chain {
    fn state(&self, m: &Gemma4Model, chain: &dyn DeviceChain) -> Option<&State> {
        self.state.get_or_init(|| State::of(m, chain)).as_ref()
    }

    /// `x` (the rows' embeddings, `[t, d]`, as the residual stream starts) through every layer and the head, chained
    /// in one submit, if the backend can: the last row's logits `[1, vocab]`; None leaves it to the model's own path.
    /// `ple_ids` are the tokens whose rows of the PLE table the layers read (a picture's soft tokens' the padding's).
    pub(crate) fn run(&self, m: &Gemma4Model, x: &[f32], ple_ids: &[u32], kv: &mut KvCache) -> Option<Tensor> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() {
            return None;
        }
        let backend: &dyn Backend = &*m.backend;
        let chain = backend.chain()?;
        let st = self.state(m, chain)?;
        let cfg = &m.config;
        let t = ple_ids.len();
        let d = cfg.embedding_dim;
        if t == 0 || x.len() != t * d {
            return None;
        }
        let one = t == 1;
        let n_h = cfg.n_heads;
        let pld = m.per_layer_dim();
        let past = kv.len;
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        g.reserve(chain, st, n_h, past + t);
        g.sync(chain, st, kv, past);
        // a decode step's vectors are made once and kept (its bind groups with them); a prompt's chunk's are its own
        let mut kept = st.step.lock().unwrap_or_else(|p| p.into_inner());
        let fresh;
        let bufs: &Bufs = if one {
            kept.get_or_insert_with(|| Bufs::new(chain, st, m, 1))
        } else {
            fresh = Bufs::new(chain, st, m, t);
            &fresh
        };
        let Bufs { xs, xn, proj, proj_n, heads, ffn: ffn_bufs, logits, last, tables, pli, ple: ple_bufs } = bufs;
        let attn = if one { None } else { Some(chain.vec(st.layers.iter().map(|l| chain.attention_rows_out_len(t, n_h, l.hd, past + t)).max().unwrap_or(1))) };
        chain.upload(xs, x);
        for (values, table) in st.tables(past, t).iter().zip(tables) {
            chain.upload(table, values);
        }
        // the per-layer inputs: the tokens' table rows (times sqrt(pld) / sqrt(2)) plus the normed projection
        if let (Some(pli), Some(gl)) = (pli, &m.global) {
            let s = (pld as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
            let rows: Vec<f32> = gl.per_layer_token_embd.rows(ple_ids).into_iter().map(|v| v * s).collect();
            chain.upload(pli, &rows);
        }
        let mut rec = chain.begin();
        if !one {
            // the chunk's vectors are its own: no bind groups kept to hold their memory after it
            rec.keep_groups(false);
        }
        let r = rec.as_mut();
        if let (Some(pli), Some((pp, pn, ..)), Some(pm)) = (pli, ple_bufs, &st.ple_proj) {
            pm.own_rows(r, xs, pp, t);
            r.rmsnorm_rows(pp, &st.ple_norm, pn, t * cfg.n_layers, cfg.rms_eps);
            r.add(pli, pn);
        }
        let neox = matches!(cfg.arch.rope_type(), ggml_rs::RopeType::NeoX);
        for (l, layer) in st.layers.iter().enumerate() {
            let (b, gx) = (&m.common.blocks[l], &m.extras[l]);
            let hd = layer.hd;
            // the KV heads of the cache this layer's attention reads (its own, or the layer's it shares, of its kind)
            let n_kv = st.layers[layer.src].n_kv;
            let (kvd, row) = (n_kv * hd, 2 * n_kv * hd);
            let table = &tables[layer.rope];
            let [q, k, vv, qn, kn, vn] = &heads[&hd];
            r.rmsnorm_rows(&xs, &layer.attn_norm, &xn, t, cfg.rms_eps);
            layer.q.rows(&b.attn_q, r, &xn, q, t, one);
            r.rmsnorm_rows(q, &layer.q_norm, qn, t * n_h, cfg.rms_eps);
            r.rope_rows(qn, t, n_h, hd, table, neox);
            if let Some((wk, wv)) = &layer.kv {
                let buf = g.layers[l].as_ref().expect("a buffer for a layer's own cache");
                wk.rows(&b.attn_k, r, &xn, k, t, one);
                wv.rows(&b.attn_v, r, &xn, vv, t, one);
                r.rmsnorm_rows(k, &layer.k_norm, kn, t * n_kv, cfg.rms_eps);
                r.rmsnorm_rows(vv, &st.ones, vn, t * n_kv, cfg.rms_eps);
                r.rope_rows(kn, t, n_kv, hd, table, neox);
                r.store_rows(kn, buf, t, kvd, past, row, 0);
                r.store_rows(vn, buf, t, kvd, past, row, kvd);
            }
            let src = g.layers[layer.src].as_ref().expect("the cache a layer reads");
            let out = match &attn {
                Some(a) => {
                    r.attention_rows(qn, src, a, t, n_h, n_kv, hd, past, layer.window, 1.0);
                    a
                }
                None => {
                    // a sliding window sees the last `w` positions, as the CPU's attention does
                    let lo = layer.window.map_or(0, |w| (past + 1).saturating_sub(w));
                    r.attention(qn, src, &g.out, n_h, n_kv, hd, lo, past + 1, g.cap, 1.0);
                    &g.out
                }
            };
            layer.o.rows(&b.attn_output, r, out, &proj, t, one);
            r.rmsnorm_rows(&proj, &layer.post_attn, &proj_n, t, cfg.rms_eps);
            r.add(&xs, &proj_n);
            r.rmsnorm_rows(&xs, &layer.ffn_norm, &xn, t, cfg.rms_eps);
            let (gate_up, act, split) = &ffn_bufs[&layer.ff];
            match (&layer.ffn, &b.ffn_pair, split) {
                (Ffn::Fused(h), FfnPair::Fused(w), _) => h.rows(w, r, &xn, gate_up, t, one),
                (Ffn::Split(hg, hu), FfnPair::Split { gate, up }, Some((gb, ub))) => {
                    hg.rows(gate, r, &xn, gb, t, one);
                    hu.rows(up, r, &xn, ub, t, one);
                    r.store_rows(gb, gate_up, t, layer.ff, 0, 2 * layer.ff, 0);
                    r.store_rows(ub, gate_up, t, layer.ff, 0, 2 * layer.ff, layer.ff);
                }
                _ => unreachable!("the chain state checked the FFN"),
            }
            r.gelu_mul_split_rows(gate_up, act, t);
            layer.down.rows(&b.ffn_down, r, act, &proj, t, one);
            r.rmsnorm_rows(&proj, &layer.post_ffn, &proj_n, t, cfg.rms_eps);
            r.add(&xs, &proj_n);
            if let (Some((hg, hp, norm)), Some(wg), Some(wp), Some(pli), Some((_, _, gate, fused, act, pl))) =
                (&layer.ple, &gx.per_layer_inp_gate, &gx.per_layer_proj, &pli, &ple_bufs)
            {
                // gelu(W x) times the layer's input, projected back, normed, added
                hg.rows(wg, r, &xs, gate, t, false);
                r.store_rows(gate, fused, t, pld, 0, 2 * pld, 0);
                r.copy_cols(pli, pl, t, pld, cfg.n_layers * pld, l * pld);
                r.store_rows(pl, fused, t, pld, 0, 2 * pld, pld);
                r.gelu_mul_split_rows(fused, act, t);
                hp.rows(wp, r, act, &proj, t, false);
                r.rmsnorm_rows(&proj, norm, &proj_n, t, cfg.rms_eps);
                r.add(&xs, &proj_n);
            }
            if let Some(s) = layer.scale {
                r.scale_in_place(&xs, t * d, s);
            }
        }
        // the head of the last row only
        r.rmsnorm_rows(&xs, &st.output_norm, &xn, t, cfg.rms_eps);
        r.copy(&xn, (t - 1) * d, &last, 0, d);
        let Weight::Quant(head) = &m.common.output else { unreachable!("the chain state checked the head") };
        r.matmul(head, &last, &logits);
        r.read(&logits);
        let own: Vec<usize> = (0..st.layers.len()).filter(|&l| st.layers[l].kv.is_some()).collect();
        for &l in &own {
            let row = 2 * st.layers[l].n_kv * st.layers[l].hd;
            r.read_range(g.layers[l].as_ref().expect("a layer's own cache"), past * row, t * row);
        }
        let mut got = rec.finish().into_iter();
        let mut out = got.next().expect("the logits");
        // the rows' K and V into the host's cache too, which the copy already holds
        for (&l, rows) in own.iter().zip(got) {
            kv.append_rows(backend, l, &rows, t);
        }
        kv.commit(t);
        kv.dirty_from = usize::MAX;
        if let Some(c) = cfg.final_logit_softcap {
            out.iter_mut().for_each(|v| *v = (*v * (1.0 / c)).tanh() * c);
        }
        Some(Tensor::from_vec(out, vec![1, cfg.vocab_size]))
    }
}
