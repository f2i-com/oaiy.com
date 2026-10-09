//! Flash-Next chained on the GPUs: a decode step, a prompt's chunk, drafts and their checks, each a recording.

use super::*;

// ---------------------------------------------------------------------------
// A decode step chained on the GPUs (WebGPU)
// ---------------------------------------------------------------------------

/// A matrix on a chain's device: f16 two to a word where its values are (the checkpoint's f16 weights; half the bytes
/// a step reads), else f32.
pub(super) struct ChainMat {
    pub(super) v: ggml_rs::DeviceVec,
    pub(super) half: bool,
}

impl ChainMat {
    pub(super) fn new(c: &dyn ggml_rs::DeviceChain, t: &Tensor) -> Self {
        let host;
        let values = if t.is_device() {
            host = t.to_host();
            host.data()
        } else {
            t.data()
        };
        if let Some(v) = c.vec_f16(values) {
            return ChainMat { v, half: true };
        }
        let v = c.vec(values.len());
        c.upload(&v, values);
        ChainMat { v, half: false }
    }

    /// `y[r] = W x[r]` for `rows` rows (`W` `[n, k]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mul(&self, rec: &mut dyn ggml_rs::ChainRecorder, n: usize, k: usize, x: &ggml_rs::DeviceVec, y: &ggml_rs::DeviceVec, rows: usize) {
        if self.half {
            rec.matmul_f16_rows(&self.v, n, k, x, y, rows)
        } else {
            rec.matmul_f32_rows(&self.v, n, k, x, y, rows)
        }
    }
}

/// A hyper-connection's matrices on its layer's device (`down` `[rank + writes, streams * hidden]`, `up` `[streams *
/// hidden, rank + writes]`), and its norm (`1 + w`).
impl HcVecs {
    /// The site's projections of `rows` rows of normed streams (`s` of `h`): the down matrix and the gates into `t`
    /// and `post`, the up matrix and the mix into `out`. f16 matrices take each projection with what follows it (one
    /// dispatch for a step's row or a check's few, where two: a decode step 10.5 ms where 11.2), `logits` then scratch
    /// that may stay unwritten.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(&self, rec: &mut dyn ggml_rs::ChainRecorder, normed: &ggml_rs::DeviceVec, t: &ggml_rs::DeviceVec, post: &ggml_rs::DeviceVec, logits: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec, rows: usize, s: usize, h: usize) {
        let n = self.rank + self.writes;
        if self.down.half && self.up.half {
            rec.hc_down_gates(&self.down.v, s * h, normed, t, post, rows, self.rank, self.writes, s);
            rec.hc_up_mix(&self.up.v, n, t, logits, normed, out, rows, s, h);
        } else {
            self.down.mul(rec, n, s * h, normed, t, rows);
            rec.hc_gates(t, post, rows, self.rank, self.writes, s);
            self.up.mul(rec, s * h, n, t, logits, rows);
            rec.hc_mix(logits, normed, out, rows, s, h);
        }
    }
}

pub(super) struct HcVecs {
    pub(super) norm: ggml_rs::DeviceVec,
    pub(super) down: ChainMat,
    pub(super) up: ChainMat,
    pub(super) rank: usize,
    pub(super) writes: usize,
}

/// A layer's own vectors on its device.
pub(super) struct ChainLayer {
    pub(super) attn_hc: HcVecs,
    pub(super) mlp_hc: HcVecs,
    pub(super) mixer: ChainMixer,
    /// The router, then the shared expert's gate, `[experts + 1, hidden]`.
    pub(super) router: ChainMat,
    /// Its experts are on the host (no GPU had room for them): the chain reads its router and its experts' input
    /// back, they run there, and their sum goes up for the layer's write-back.
    pub(super) host: bool,
}

pub(super) enum ChainMixer {
    /// The beta-alpha projection, the conv's weights, `A`, `dt_bias`, the output norm, and which of the recurrent
    /// states' pool is this layer's.
    Gdn { ba: ChainMat, conv: ggml_rs::DeviceVec, a: ggml_rs::DeviceVec, dt: ggml_rs::DeviceVec, norm: ggml_rs::DeviceVec, slot: usize },
    /// The per-head norms, the indexer's (QSA's query and pooled key norms), and which of its device's copy of the
    /// cache is this layer's.
    Attn { q_norm: ggml_rs::DeviceVec, k_norm: ggml_rs::DeviceVec, iq_norm: ggml_rs::DeviceVec, ik_norm: ggml_rs::DeviceVec, slot: usize },
}

/// A device's working vectors for a step (one row).
pub(super) struct ChainDev {
    pub(super) x: ggml_rs::DeviceVec,
    pub(super) normed: ggml_rs::DeviceVec,
    pub(super) t: ggml_rs::DeviceVec,
    pub(super) post: ggml_rs::DeviceVec,
    pub(super) post2: ggml_rs::DeviceVec,
    pub(super) logits: ggml_rs::DeviceVec,
    pub(super) y_in: ggml_rs::DeviceVec,
    pub(super) y_out: ggml_rs::DeviceVec,
    pub(super) y2_in: ggml_rs::DeviceVec,
    pub(super) moe_out: ggml_rs::DeviceVec,
    pub(super) router: ggml_rs::DeviceVec,
    pub(super) qkv: ggml_rs::DeviceVec,
    pub(super) z: ggml_rs::DeviceVec,
    pub(super) ba: ggml_rs::DeviceVec,
    pub(super) conv: ggml_rs::DeviceVec,
    pub(super) core: ggml_rs::DeviceVec,
    pub(super) qfull: ggml_rs::DeviceVec,
    pub(super) q: ggml_rs::DeviceVec,
    pub(super) gate: ggml_rs::DeviceVec,
    pub(super) k: ggml_rs::DeviceVec,
    pub(super) v: ggml_rs::DeviceVec,
    pub(super) qn: ggml_rs::DeviceVec,
    pub(super) kn: ggml_rs::DeviceVec,
    pub(super) gated: ggml_rs::DeviceVec,
    pub(super) index: ggml_rs::DeviceVec,
    pub(super) table: ggml_rs::DeviceVec,
    pub(super) mixed: ggml_rs::DeviceVec,
    pub(super) head: ggml_rs::DeviceVec,
    /// The head's rows' picks (`ChainRecorder::argmax_rows`: four values a row).
    pub(super) picks: ggml_rs::DeviceVec,
}

/// A device's copy of its attention layers' caches (row `t`: K `[kv_heads, head_dim]` then V), `cap` rows, and of their
/// raw indexer keys (`[cap, index_dim]`, QSA's pool reads them past the dense span).
pub(super) struct ChainKv {
    pub(super) layers: Vec<ggml_rs::DeviceVec>,
    pub(super) raw: Vec<ggml_rs::DeviceVec>,
    pub(super) cap: usize,
    pub(super) out: ggml_rs::DeviceVec,
    /// QSA's vectors for a step's or a check's rows, made at the first past the dense span (for this `cap`).
    pub(super) qsa: Option<QsaVecs>,
    /// The `KvCache::id` the rows are a copy of (0: none).
    pub(super) owner: u64,
}

/// QSA's vectors on a device for runs of up to `rows` rows over a cache of `cap` positions: the indexer's queries (in,
/// normed), the pooled block keys (in, normed) and the RoPE table of the blocks' starts, the block scores, each row's
/// blocks, and the attention's output and parts.
pub(super) struct QsaVecs {
    pub(super) rows: usize,
    pub(super) iq: ggml_rs::DeviceVec,
    pub(super) iqn: ggml_rs::DeviceVec,
    pub(super) pooled: ggml_rs::DeviceVec,
    pub(super) pooledn: ggml_rs::DeviceVec,
    pub(super) table: ggml_rs::DeviceVec,
    pub(super) scores: ggml_rs::DeviceVec,
    pub(super) list: ggml_rs::DeviceVec,
    pub(super) out: ggml_rs::DeviceVec,
}

/// `v`'s first `len` elements as a vector of their own (the same buffer): a chain's ops take their sizes from their
/// vectors' lengths.
pub(super) fn first(v: &ggml_rs::DeviceVec, len: usize) -> ggml_rs::DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    ggml_rs::DeviceVec { len, inner: Arc::clone(&v.inner) }
}

/// What a chained step changes: the devices' copies of the cache, and the recurrent states the cache's tensors alias.
pub(super) struct ChainMut {
    pub(super) kv: Vec<ChainKv>,
    pub(super) pool: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    /// The n-gram layer's conv window, aliased in the cache as the delta nets' states are.
    pub(super) ple_window: ggml_rs::DeviceVec,
    /// Each device's attention scratch for a run of a few rows, grown as the positions are.
    pub(super) rows_out: Vec<ggml_rs::DeviceVec>,
    /// What undoing the last check needs ([`FlashNext::rollback`]), made at the first check.
    pub(super) undo: Option<Undo>,
    /// The prediction layer's cache, made at the first draft or prompt that fills it.
    pub(super) mtp_kv: Option<MtpKv>,
    /// Where the trunk's streams kept for the prediction layer are: the position of the first row, and how many.
    pub(super) mtp_hid: (usize, usize),
}

/// The most rows a chained run takes as a check's (each row as a step's, the experts routed on the GPU, the vectors
/// kept): the token sampled and its drafts.
pub(crate) const CHECK_ROWS: usize = 8;

/// A check undone: each delta net's state and conv window as they were before it and the check's inputs to them (its
/// rows' qkv and beta-alpha, `CHECK_ROWS` rows), in the pool's order; the n-gram layer's window before it; and the
/// check's tokens and the n-gram history before them.
pub(super) struct Undo {
    pub(super) backups: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    pub(super) inputs: Vec<(ggml_rs::DeviceVec, ggml_rs::DeviceVec)>,
    pub(super) window: Option<ggml_rs::DeviceVec>,
    pub(super) tokens: Vec<u32>,
    pub(super) history: Option<Tensor>,
}

/// A run of a few rows' vectors (a check of drafts, a short chunk), kept so their bind groups are: each device's (the
/// collapse and head for every row), the n-gram layer's, and each device's attention layers' indexer keys (`[rows,
/// index_dim]` each).
pub(super) struct FewSet {
    pub(super) devs: Vec<ChainDev>,
    pub(super) ple: Option<PleVecs>,
    pub(super) keys: Vec<Vec<ggml_rs::DeviceVec>>,
    /// Each device's one row's query (a row's attention as a step's).
    pub(super) q1: Vec<ggml_rs::DeviceVec>,
}

/// The multi-token-prediction layer chained (on the last device): its norms, hyper-connections and router, the
/// embedding's broadcast weights (ones), the trunk's streams of the last run's rows (a step's, a check's, a prompt's
/// last), and a pass's vectors for 1 to [`CHECK_ROWS`] rows, made at the first of each.
pub(super) struct MtpChain {
    pub(super) enorm: ggml_rs::DeviceVec,
    pub(super) hnorm: ggml_rs::DeviceVec,
    pub(super) attn_hc: HcVecs,
    pub(super) mlp_hc: HcVecs,
    pub(super) mixer: HcVecs,
    pub(super) q_norm: ggml_rs::DeviceVec,
    pub(super) k_norm: ggml_rs::DeviceVec,
    pub(super) router: ChainMat,
    pub(super) ones: ggml_rs::DeviceVec,
    pub(super) hid: ggml_rs::DeviceVec,
    /// A draft's token, its logit and the sum of the exponentials against it (`argmax_softmax`).
    pub(super) best: ggml_rs::DeviceVec,
    pub(super) devs: [std::sync::OnceLock<MtpDev>; CHECK_ROWS],
}

/// A pass of the prediction layer's vectors for its rows: a device's working vectors, the next tokens' embeddings
/// (in, normed, projected), the hidden streams (in, normed), the attention's output, and one row's query.
pub(super) struct MtpDev {
    pub(super) dv: ChainDev,
    pub(super) e: ggml_rs::DeviceVec,
    pub(super) en: ggml_rs::DeviceVec,
    pub(super) e2: ggml_rs::DeviceVec,
    pub(super) hin: ggml_rs::DeviceVec,
    pub(super) hn: ggml_rs::DeviceVec,
    pub(super) att: ggml_rs::DeviceVec,
    pub(super) q1: ggml_rs::DeviceVec,
}

/// The prediction layer's cache (row `t`: K `[kv_heads, head_dim]` then V), `cap` rows, the decode attention's
/// scratch for it; the cache it follows ([`KvCache::id`]), and the positions its entries are true ones for
/// (`start..valid`: from the trunk's hidden states, not drafts).
pub(super) struct MtpKv {
    pub(super) layer: ggml_rs::DeviceVec,
    pub(super) cap: usize,
    pub(super) out: ggml_rs::DeviceVec,
    pub(super) owner: u64,
    pub(super) start: usize,
    pub(super) valid: usize,
}

/// The least probability the prediction layer gives a draft for it to be checked (Strata's `--spec-min-p`): a check's
/// rows cost, and an unlikely draft is seldom taken.
pub(super) const DRAFT_MIN_P: f64 = 0.5;

/// The n-gram layer on its device: its key and value projections, its norms and its conv's weights (`[streams *
/// hidden, kernel]`).
pub(super) struct ChainPle {
    pub(super) key: ChainMat,
    pub(super) value: ChainMat,
    pub(super) norm_key: ggml_rs::DeviceVec,
    pub(super) norm_query: ggml_rs::DeviceVec,
    pub(super) norm_conv: ggml_rs::DeviceVec,
    pub(super) conv: ggml_rs::DeviceVec,
}

/// The n-gram layer's working vectors for `rows` rows: the features, the key and value, the gated streams and the
/// conv's input.
pub(super) struct PleVecs {
    pub(super) emb: ggml_rs::DeviceVec,
    pub(super) key: ggml_rs::DeviceVec,
    pub(super) value: ggml_rs::DeviceVec,
    pub(super) gated: ggml_rs::DeviceVec,
    pub(super) conv_in: ggml_rs::DeviceVec,
}

pub(super) fn ple_vecs(c: &dyn ggml_rs::DeviceChain, cfg: &Config, rows: usize) -> PleVecs {
    let width = cfg.streams * cfg.hidden;
    PleVecs { emb: c.vec(rows * cfg.ple_dim), key: c.vec(rows * width), value: c.vec(rows * cfg.hidden), gated: c.vec(rows * width), conv_in: c.vec(rows * width) }
}

pub(crate) struct FnChain {
    pub(super) layers: Vec<ChainLayer>,
    pub(super) devs: Vec<ChainDev>,
    /// Each device's attention layers, in its copy's order.
    pub(super) attn_of: Vec<Vec<usize>>,
    /// The delta-net layers, in the pool's order.
    pub(super) gdn: Vec<usize>,
    pub(super) collapse: HcVecs,
    /// The widest low-rank gate's rank (the scratch's width).
    pub(super) rank: usize,
    /// Each device's attention layers' indexer keys of a step, copied out as each layer makes its own (a step's
    /// layers on a device run as one submit, and its indexer vector is every layer's).
    pub(super) keys: Vec<ggml_rs::DeviceVec>,
    /// The n-gram layer chained, where its matrices are dense (else it runs through the host), and a step's vectors.
    pub(super) ple: Option<(ChainPle, PleVecs)>,
    /// The vectors of runs of 2 to [`CHECK_ROWS`] rows, made at the first of each.
    pub(super) few: [std::sync::OnceLock<FewSet>; CHECK_ROWS - 1],
    /// A tapped run's later rows' vectors, a set a device, made at the first run that keeps a state from inside.
    pub(super) seg: std::sync::OnceLock<Vec<TapSeg>>,
    /// The multi-token-prediction layer, where it is loaded and chainable.
    pub(super) mtp: Option<MtpChain>,
    pub(super) m: std::sync::Mutex<ChainMut>,
    /// Steps the chain took, for a test that has to know it ran.
    pub(crate) runs: std::sync::atomic::AtomicUsize,
}

/// A device's working vectors for `rows` rows, the collapse and the head for `heads` of them.
pub(super) fn chain_dev(c: &dyn ggml_rs::DeviceChain, cfg: &Config, rank: usize, rows: usize, heads: usize) -> ChainDev {
    let (h, s, nh, nkv, hd) = (cfg.hidden, cfg.streams, cfg.heads, cfg.kv_heads, cfg.head_dim);
    let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
    let v = |n: usize| c.vec(rows * n);
    ChainDev {
        x: v(s * h),
        normed: v(s * h),
        t: v(rank + s),
        post: v(s),
        post2: v(s),
        logits: v(s * h),
        y_in: v(h),
        y_out: v(h),
        y2_in: v(h),
        moe_out: v(h),
        router: v(cfg.experts + 1),
        qkv: v(conv_dim),
        z: v(cfg.nv * cfg.vd),
        ba: v(2 * cfg.nv),
        conv: v(conv_dim),
        core: v(cfg.nv * cfg.vd),
        qfull: v(2 * nh * hd),
        q: v(nh * hd),
        gate: v(nh * hd),
        k: v(nkv * hd),
        v: v(nkv * hd),
        qn: v(nh * hd),
        kn: v(nkv * hd),
        gated: v(nh * hd),
        index: v((cfg.index_heads + 1) * cfg.index_dim),
        table: v(cfg.rope_dim),
        mixed: c.vec(heads * h),
        head: c.vec(heads * cfg.vocab),
        picks: c.vec(4 * heads.max(1)),
    }
}

pub(super) fn chain_packed(w: &Weight) -> Option<&dyn PackedLinear> {
    match w {
        Weight::Packed(p) => Some(p.as_ref()),
        _ => None,
    }
}

impl FlashNext {
    /// The chained step's state, made at the first step that can use one: None when a device has no chain, or a
    /// matrix is not where a chain reads it, or a layer's experts are neither there nor on the host (a layer no GPU
    /// had room for: [`ChainLayer::host`]).
    pub(super) fn chain_state(&self) -> Option<&FnChain> {
        self.chain
            .get_or_init(|| {

                let cfg = &self.config;
                let s = cfg.streams;
                let chains: Vec<&dyn ggml_rs::DeviceChain> = self.devices.iter().map(|b| b.chain()).collect::<Option<_>>()?;
                let held = |d: usize, w: &Weight| chain_packed(w).is_some_and(|p| chains[d].holds_exl3(p));
                if cfg.kd != cfg.vd || ![16, 32, 64, 128].contains(&cfg.kd) || !(2..=8).contains(&cfg.conv) || cfg.rope_dim == 0 || cfg.rope_dim > cfg.head_dim {
                    return None;
                }
                let up = |d: usize, t: &Tensor| {
                    let t = if t.is_device() { t.to_host() } else { t.clone() };
                    let v = chains[d].vec(t.numel());
                    chains[d].upload(&v, t.data());
                    v
                };
                let hc = |d: usize, m: &HyperMix| -> Option<HcVecs> {
                    if m.packed {
                        return None;
                    }
                    let writes = if m.site { s } else { 0 };
                    Some(HcVecs { norm: up(d, &m.norm), down: ChainMat::new(chains[d], &m.down), up: ChainMat::new(chains[d], &m.up), rank: m.rank, writes })
                };
                let mut attn_of = vec![Vec::new(); self.devices.len()];
                let mut gdn = Vec::new();
                let mut layers = Vec::with_capacity(cfg.layers);
                for (i, l) in self.layers.iter().enumerate() {
                    let d = l.device;
                    let dense = |w: &Weight| match w {
                        Weight::Dense(t) => Some(ChainMat::new(chains[d], t)),
                        _ => None,
                    };
                    let mixer = match &l.mixer {
                        Mixer::Gdn(g) => {
                            if ![&g.qkv, &g.z, &g.out].iter().all(|w| held(d, w)) {
                                return None;
                            }
                            gdn.push(i);
                            ChainMixer::Gdn { ba: dense(&g.ba)?, conv: up(d, &g.conv), a: up(d, &g.a), dt: up(d, &g.dt_bias), norm: up(d, &g.norm), slot: gdn.len() - 1 }
                        }
                        Mixer::Attn(a) => {
                            if ![&a.q, &a.k, &a.v, &a.o, &a.index_qk].iter().all(|w| held(d, w)) {
                                return None;
                            }
                            attn_of[d].push(i);
                            ChainMixer::Attn { q_norm: up(d, &a.q_norm), k_norm: up(d, &a.k_norm), iq_norm: up(d, &a.index_q_norm), ik_norm: up(d, &a.index_k_norm), slot: attn_of[d].len() - 1 }
                        }
                    };
                    let host = !chains[d].holds_experts(l.moe.experts.as_ref());
                    if host && (!l.moe.experts.on_host() || std::env::var_os("OAIY_NO_HOST_LAYERS").is_some()) {
                        return None;
                    }
                    layers.push(ChainLayer { attn_hc: hc(d, &l.attn_hc)?, mlp_hc: hc(d, &l.mlp_hc)?, mixer, router: dense(&l.moe.router)?, host });
                }
                // the last layer, the collapse and the head on the last device: a run's layers have written to the cache
                // before it gets there, so nothing may leave it then
                let last = self.devices.len() - 1;
                if self.layers.last().map(|l| l.device) != Some(last) || !held(last, &self.head) {
                    return None;
                }
                let collapse = hc(last, &self.collapse)?;
                let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
                // the prediction layer, where its matrices and experts are where a chain reads them
                let mtp = self.mtp.as_ref().and_then(|mp| {
                    let c = chains[last];
                    if ![&mp.fc_e, &mp.fc_h, &mp.q, &mp.k, &mp.v, &mp.o].iter().all(|w| held(last, w)) || !c.holds_experts(mp.experts.as_ref()) {
                        return None;
                    }
                    let ones = c.vec(CHECK_ROWS * s);
                    c.upload(&ones, &vec![1.0; CHECK_ROWS * s]);
                    Some(MtpChain {
                        enorm: up(last, &mp.enorm),
                        hnorm: up(last, &mp.hnorm),
                        attn_hc: hc(last, &mp.attn_hc)?,
                        mlp_hc: hc(last, &mp.mlp_hc)?,
                        mixer: hc(last, &mp.mixer)?,
                        q_norm: up(last, &mp.q_norm),
                        k_norm: up(last, &mp.k_norm),
                        router: ChainMat::new(c, &mp.router),
                        ones,
                        hid: c.vec(CHECK_ROWS * s * cfg.hidden),
                        best: c.vec(3),
                        devs: Default::default(),
                    })
                });
                let mtp_rank = mtp.as_ref().map_or(0, |m| m.attn_hc.rank.max(m.mlp_hc.rank).max(m.mixer.rank));
                let rank = layers.iter().map(|l| l.attn_hc.rank.max(l.mlp_hc.rank)).max().unwrap_or(0).max(collapse.rank).max(mtp_rank);
                let devs = chains.iter().map(|c| chain_dev(*c, cfg, rank, 1, 1)).collect();
                let keys = chains.iter().zip(&attn_of).map(|(c, a)| c.vec(a.len().max(1) * cfg.index_dim)).collect();
                let kv = chains.iter().map(|c| ChainKv { layers: Vec::new(), raw: Vec::new(), cap: 0, out: c.vec(1), qsa: None, owner: 0 }).collect();
                let pool = gdn
                    .iter()
                    .map(|&i| {
                        let c = chains[self.layers[i].device];
                        (c.vec(cfg.nv * cfg.vd * cfg.kd), c.vec((cfg.conv - 1) * conv_dim))
                    })
                    .collect();
                let pd = self.layers[cfg.ple_layer].device;
                let ple = match (&self.ple.key, &self.ple.value) {
                    (Weight::Dense(k), Weight::Dense(v)) if cfg.ple_kernel >= 1 => {
                        let c = chains[pd];
                        let p = ChainPle { key: ChainMat::new(c, k), value: ChainMat::new(c, v), norm_key: up(pd, &self.ple.norm_key), norm_query: up(pd, &self.ple.norm_query), norm_conv: up(pd, &self.ple.norm_conv), conv: up(pd, &self.ple.conv) };
                        Some((p, ple_vecs(c, cfg, 1)))
                    }
                    _ => None,
                };
                let ple_window = chains[pd].vec((cfg.ple_kernel.max(1) - 1) * cfg.ngram * s * cfg.hidden);
                let rows_out = chains.iter().map(|c| c.vec(1)).collect();
                Some(FnChain { layers, devs, attn_of, gdn, collapse, rank, keys, ple, few: Default::default(), seg: Default::default(), mtp, m: std::sync::Mutex::new(ChainMut { kv, pool, ple_window, rows_out, undo: None, mtp_kv: None, mtp_hid: (0, 0) }), runs: Default::default() })
            })
            .as_ref()
    }

    /// Steps the chain has taken (a test's check that it ran).
    #[allow(dead_code)]
    /// The chain made and its kernels compiled before a first request would wait on them (its matrices packed and
    /// uploaded, some 30 pipelines built): a short prompt and a step on a cache of their own; drafting, a draft and
    /// checks of 2 to 4 rows, each undone. False where nothing is chained.
    pub fn warm_up(&self) -> bool {
        if self.chain_state().is_none() {
            return false;
        }
        let Ok(tokens) = self.tokenizer.encode("The river town kept its market on the north bank.", false) else { return false };
        if tokens.len() < 4 {
            return false;
        }
        let mut kv = self.new_kv_cache(tokens.len() + 8);
        let step = |tokens: &[u32], kv: &mut KvCache| self.embed_text(tokens).and_then(|e| self.forward(tokens, &e, kv, None)).is_ok();
        let mut warmed = step(&tokens, &mut kv) && step(&tokens[..1], &mut kv);
        if warmed && self.drafts() {
            // every draft made, however unlikely: the layer's passes all run
            warmed = self.draft_above(&kv, &tokens[1..2], 3, 0.0).is_some();
            for rows in 2..=4 {
                warmed &= self.check(&tokens[..rows], &mut kv).is_some();
                self.rollback(&mut kv, 1);
            }
        }
        self.chain.get().and_then(|c| c.as_ref()).inspect(|c| c.runs.store(0, std::sync::atomic::Ordering::Relaxed));
        warmed
    }

    #[cfg(test)]
    pub(crate) fn chain_runs(&self) -> usize {
        self.chain.get().and_then(|c| c.as_ref()).map_or(0, |c| c.runs.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// `tokens` (a decode step's one, or a prompt's chunk; their embeddings `embeds`), chained, if the devices can: the
    /// last row's logits. See [`Self::run_chained`].
    pub(super) fn forward_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache) -> Option<Tensor> {
        let logits = self.run_chained(tokens, embeds, kv, false)?;
        Some(Tensor::from_vec(logits, vec![1, self.config.vocab]))
    }

    /// A check of `tokens` (the token sampled, then its drafts; 2 to [`CHECK_ROWS`] of them) after what `kv` holds:
    /// every row's logits (`[1, vocab]` each), each row as a step would give it. The run is kept undoable
    /// ([`Self::rollback`]). None where it cannot be chained.
    pub fn check(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        let logits = self.run_chained(tokens, &embeds, kv, true)?;
        Some(logits.chunks_exact(self.config.vocab).map(|r| Tensor::from_vec(r.to_vec(), vec![1, self.config.vocab])).collect())
    }

    /// [`Self::check`] for a request that samples greedily: each row's token alone, its largest logit's (the first of
    /// equals, as greedy sampling takes it), picked on the GPU. A check's rows of logits are a megabyte each to read
    /// back, for a token each. None as [`Self::check`] (nothing run).
    pub fn check_picks(&self, tokens: &[u32], kv: &mut KvCache) -> Option<Vec<u32>> {
        if !(2..=CHECK_ROWS).contains(&tokens.len()) || kv.len + tokens.len() > kv.max_len {
            return None;
        }
        let embeds = self.embed_text(tokens).ok()?;
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(tokens, &embeds, kv, true);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        Some(got?.chunks_exact(4).take(tokens.len()).map(|row| row[0].to_bits()).collect())
    }

    /// A decode step's token for a request that samples greedily, as [`Self::check_picks`]: `token` (its embedding
    /// `embeds`) run after what `kv` holds, the next token picked on the GPU. None where the step cannot be chained
    /// (nothing run: [`Self::forward`] is the caller's).
    pub fn step_pick(&self, token: u32, embeds: &Tensor, kv: &mut KvCache) -> Option<u32> {
        if profile::on() || self.prompt_rows() == 0 {
            return None;
        }
        self.pick.store(true, std::sync::atomic::Ordering::Relaxed);
        let got = self.run_chained(&[token], embeds, kv, false);
        self.pick.store(false, std::sync::atomic::Ordering::Relaxed);
        got.map(|row| row[0].to_bits())
    }

    /// Undo the last check's rows past its first `keep` (the token sampled and the drafts accepted): each delta net's
    /// state and conv window and the n-gram layer's window as they were before it, its first `keep` rows run through
    /// them again, the n-gram history its tokens', and the cache cut back.
    pub fn rollback(&self, kv: &mut KvCache, keep: usize) {
        use ggml_rs::DeltaNet;
        let Some(st) = self.chain_state() else { return };
        let Some(chains) = self.devices.iter().map(|b| b.chain()).collect::<Option<Vec<_>>>() else { return };
        let cfg = &self.config;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        let Some(u) = m.undo.as_mut() else { return };
        let rows = u.tokens.len();
        assert!(keep >= 1 && keep <= rows && rows <= kv.len, "a rollback of {rows} rows to {keep}");
        if keep == rows {
            return;
        }
        let few = st.few[rows - 2].get().expect("the check's vectors");
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let dn = DeltaNet { rows: keep, v_heads: cfg.nv, k_heads: cfg.nk, k_dim: cfg.kd, v_dim: cfg.vd, scale_q: 1.0 / (cfg.vd as f32).sqrt(), eps: cfg.eps, sigmoid_gate: true };
        let ple_device = self.layers[cfg.ple_layer].device;
        let slot = ple_slot(cfg);
        for (d, c) in chains.iter().enumerate() {
            let dv = &few.devs[d];
            let mut rec = c.begin();
            rec.keep_groups(true);
            for (i, &l) in st.gdn.iter().enumerate().filter(|&(_, &l)| self.layers[l].device == d) {
                let (Some(state), Some(conv)) = (kv.ssm_state[l].as_ref().and_then(|t| c.aliased(t)), kv.ssm_conv[l].as_ref().and_then(|t| c.aliased(t))) else {
                    unreachable!("a check left layer {l}'s state the chain's")
                };
                let ChainMixer::Gdn { conv: w, a, dt, norm, .. } = &st.layers[l].mixer else { unreachable!("layer {l} is a delta net") };
                let (bs, bc) = &u.backups[i];
                let (qkv, ba) = &u.inputs[i];
                rec.copy(bs, 0, &state, 0, state.len);
                rec.copy(bc, 0, &conv, 0, conv.len);
                rec.ssm_conv(qkv, w, &conv, &dv.conv, keep, conv_dim, cfg.conv);
                // the outputs are not wanted: the check's were the kept rows' already
                rec.delta_net(&dv.conv, &dv.z, ba, a, dt, norm, &state, &dv.core, dn);
            }
            if d == ple_device {
                if let (Some(backup), Some((p, _)), Some(v), Some(window)) = (&u.window, &st.ple, &few.ple, kv.ssm_conv[slot].as_ref().and_then(|t| c.aliased(t))) {
                    // the window as the kept rows leave it (the conv's sums into scratch)
                    rec.copy(backup, 0, &window, 0, window.len);
                    rec.ple_conv(&dv.logits, &dv.normed, &v.conv_in, &window, &p.conv, keep, cfg.streams * cfg.hidden, cfg.ple_kernel, cfg.ngram);
                }
            }
            // (nothing of it is read back, and each device's next run goes to its queue behind it: not waited for,
            // where the thread parked for the first card's undoing and then the second's)
            rec.send();
        }
        // the n-gram history: the one before the check, then its kept tokens
        let ctx = cfg.ngram - 1;
        let mut history: Vec<f32> = match &u.history {
            Some(t) => t.to_host().data().to_vec(),
            None => vec![cfg.ple_eos as f32; ctx],
        };
        history.extend(u.tokens[..keep].iter().map(|&t| t as f32));
        kv.ssm_state[slot] = Some(Tensor::from_vec(history[history.len() - ctx..].to_vec(), vec![ctx]));
        kv.len -= rows - keep;
        u.tokens.truncate(keep);
        if m.mtp_hid.0 + m.mtp_hid.1 == kv.len + rows - keep {
            m.mtp_hid.1 = m.mtp_hid.1.min(keep);
        }
    }

    /// `tokens` (a decode step's one, a check's few, or a prompt's chunk; their embeddings `embeds`), chained, if the
    /// devices can: one submit a layer, the layer's work all on its GPU (the previous layer's experts as the host routed
    /// them, the hyper-connections' write-back, norm, gates and mix, the delta net or the attention, the router), only
    /// the router's logits coming back (and an attention layer's K, V and indexer key for the host's cache); the n-gram
    /// features before their layer and the hand-over between devices through the host. A step's and a few rows' (up to
    /// [`CHECK_ROWS`]) experts are routed on their GPU, a device's layers one submit, their vectors kept. The last row's
    /// logits; every row's for a check (`check`: undoable, see [`Self::rollback`]). None leaves the run to `forward`'s
    /// own path: past the dense span (QSA's sparse attention), or with images.
    pub(super) fn run_chained(&self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool) -> Option<Vec<f32>> {
        let run = self.run_begin(tokens, embeds, kv, check, &mut None, None, &[])?;
        Some(run.finish(self, kv))
    }

    /// A prompt's chunks in turn, chained, each chunk's first devices' layers run as the last device runs the chunk
    /// before's (`kv` then holds them all): the last chunk's logits, or None where a chunk cannot be chained (the
    /// chunks before it run, `done` of each said; the rest the caller's). `done(i)` once chunk `i` has gone to the
    /// GPUs (its K and V in the host's cache once the next has).
    pub fn forward_chunks(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize)) -> Option<Tensor> {
        self.forward_chunks_tapped(chunks, kv, done, &[]).0
    }

    /// Whether a prompt's chunks of [`Self::prompt_rows`] (`rows` in all, after what `kv` holds) can keep their
    /// recurrent states from inside them once each of `taps` rows is in ([`Self::forward_chunks_tapped`]): chained,
    /// the n-gram layer too (its window a device's vector), and each chunk's rows after the first state it keeps few
    /// ([`TAP_ROWS`]: their recurrences go through vectors of their own). OAIY_NO_TAPS: never.
    pub fn can_tap(&self, rows: usize, kv: &KvCache, taps: &[usize]) -> bool {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || std::env::var_os("OAIY_NO_TAPS").is_some() || profile::on() || rows == 0 || taps.is_empty() {
            return false;
        }
        let most = self.prompt_rows();
        self.chain_state().is_some_and(|st| st.ple.is_some())
            && self.devices.iter().all(|b| b.chain().is_some())
            && (kv.len + rows) / self.config.index_ratio <= 4096
            && taps.windows(2).all(|w| w[0] < w[1])
            && taps[0] > 0
            && taps[taps.len() - 1] <= rows
            && (0..rows).step_by(most).all(|at| {
                let end = (at + most).min(rows);
                taps.iter().find(|&&p| p > at && p <= end).map_or(true, |&p| end - p <= TAP_ROWS)
            })
    }

    /// [`Self::forward_chunks`] with the delta nets' states and conv windows and the n-gram layer's window and history
    /// as they are once each of `taps` rows of the chunks is in (ascending, counted through the chunks): what a
    /// checkpoint there holds, the run not stopping for it. A prompt's last two, before the assistant's header and
    /// before its last token, each ended a run, and a run of few rows costs what a chunk of 64 does (its weights are
    /// decoded once whatever its rows): a follow-up turn's 21 new tokens 102 ms, then 42 and 18 for those two. The
    /// states of the chunks that ran (all of them where the logits are given).
    pub fn forward_chunks_tapped(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize]) -> (Option<Tensor>, Vec<llama_rs::Tapped>) {
        let mut tapped: Vec<llama_rs::Tapped> = Vec::new();
        // each chunk's first row among the chunks'
        let starts: Vec<usize> = chunks.iter().scan(0, |at, (t, _)| { let a = *at; *at += t.len(); Some(a) }).collect();
        let logits = self.chunks_run(chunks, kv, done, taps, &starts, &mut tapped);
        (logits, tapped)
    }

    pub(super) fn chunks_run(&self, chunks: &[(&[u32], &Tensor)], kv: &mut KvCache, done: &mut dyn FnMut(usize), taps: &[usize], starts: &[usize], tapped: &mut Vec<llama_rs::Tapped>) -> Option<Tensor> {
        // each chunk's n-gram features (random reads of a 32 GB table: some 25 ms a chunk of 512) read on a thread of
        // their own, a chunk ahead of the GPUs
        let ctx = self.config.ngram - 1;
        let mut history = self.ple_history(kv);
        let histories: Vec<Vec<i64>> = chunks
            .iter()
            .map(|(t, _)| {
                history.extend(t.iter().map(|&v| v as i64));
                let h = history.clone();
                history.drain(..history.len() - ctx);
                h
            })
            .collect();
        std::thread::scope(|sc| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Option<Vec<f32>>>(1);
            let histories = &histories;
            sc.spawn(move || {
                for h in histories {
                    if tx.send(self.ngram_embedding(h).ok()).is_err() {
                        break;
                    }
                }
            });
            let mut pending: Option<ChainedRun<'_>> = None;
            let mut last = None;
            // Over two devices each chunk in two parts, its first device's as that device still runs the chunk
            // before's (OAIY_FN_IN_TURN: a chunk whole at a time): where the n-gram layer is chained, the experts
            // routed on their GPU, the layers one device's then the other's, and each device has room for another
            // chunk's vectors.
            let ahead = std::env::var_os("OAIY_FN_IN_TURN").is_none()
                && self.devices.len() == 2
                && self.chain_state().is_some_and(|st| st.ple.is_some() && st.layers.iter().all(|l| !l.host))
                && self.config.experts <= 1024
                && self.config.top_k <= 32
                && std::env::var_os("OAIY_HOST_ROUTE").is_none()
                && self.layers.windows(2).filter(|w| w[0].device != w[1].device).count() == 1
                && self.devices.iter().all(|b| b.chain().is_some_and(|c| c.has_room(1 << 30)));
            let mut parked: Option<(usize, Box<Parked<'_>>)> = None;
            // Each device then has a chunk's work behind the one it runs, with no gap: its pieces two at a time on its
            // queue, each encoded at its turn, until the chunks are in. Without that a card under a power limit ran a
            // tenth as fast for seconds at a time (15,037 tokens in 6.6 to 13.3 s where 5.3; the chunks whole, with
            // their gaps, 5.9 to 7.7).
            struct Fed<'a>(Vec<&'a dyn ggml_rs::DeviceChain>);
            impl Drop for Fed<'_> {
                fn drop(&mut self) {
                    for c in &self.0 {
                        c.pieces_in_flight_at_most(0);
                    }
                }
            }
            let fed = Fed(if ahead { self.devices.iter().filter_map(|b| b.chain()).collect() } else { Vec::new() });
            for c in &fed.0 {
                c.pieces_in_flight_at_most(2);
            }
            let said = std::env::var_os("OAIY_FN_LOG").is_some();
            let began = std::time::Instant::now();
            for (i, (tokens, embeds)) in chunks.iter().enumerate() {
                let t0 = began.elapsed().as_secs_f64() * 1e3;
                let ple = rx.recv().ok().flatten();
                let t1 = began.elapsed().as_secs_f64() * 1e3;
                let had = pending.is_some();
                let fits = tokens.len() <= self.prompt_rows() && !profile::on() && ple.is_some();
                // (the states this chunk keeps: by its own rows)
                let local: Vec<usize> = taps.iter().filter(|&&p| p > starts[i] && p <= starts[i] + tokens.len()).map(|&p| p - starts[i]).collect();
                // (the chunk before's rest goes after this chunk's first part, or before a chunk that goes whole)
                if ahead && fits && tokens.len() > CHECK_ROWS {
                    match self.run_part(tokens, embeds, kv, false, &mut pending, ple, Stage::First, &local) {
                        Some(Went::Parked(p)) => {
                            let mut before = parked.replace((i, p));
                            self.run_rest(&mut before, &mut pending, embeds, kv, done, tapped);
                            if said {
                                eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, its first part and the chunk before's rest in {:.0}", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1);
                            }
                            continue;
                        }
                        Some(Went::Run(mut run)) => {
                            // (no second device's part after all: as a chunk whole)
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            tapped.append(&mut run.tapped);
                            if let Some(p) = pending.replace(run) {
                                p.finish(self, kv);
                            }
                            done(i);
                            continue;
                        }
                        None => {
                            self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                            if let Some(p) = pending.take() {
                                p.finish(self, kv);
                            }
                            return None;
                        }
                    }
                }
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
                let run = if fits { self.run_begin(tokens, embeds, kv, false, &mut pending, ple, &local) } else { None };
                if said {
                    eprintln!("  fn chunk {i}: at {t0:.0} ms, features waited {:.0}, begun in {:.0} (the one before {})", t1 - t0, began.elapsed().as_secs_f64() * 1e3 - t1, if had && pending.is_none() { "finished inside" } else { "left" });
                }
                let Some(mut run) = run else {
                    if let Some(p) = pending.take() {
                        p.finish(self, kv);
                    }
                    return None;
                };
                tapped.append(&mut run.tapped);
                // (one the run did not take: a chain on one device)
                if let Some(p) = pending.replace(run) {
                    p.finish(self, kv);
                }
                done(i);
            }
            if let Some(&(_, embeds)) = chunks.last() {
                self.run_rest(&mut parked, &mut pending, embeds, kv, done, tapped);
            }
            if let Some(p) = pending.take() {
                last = Some(p.finish(self, kv));
            }
            last.map(|l| Tensor::from_vec(l, vec![1, self.config.vocab]))
        })
    }

    /// A parked chunk's rest ([`Stage::Rest`]; `embeds` any chunk's, not read): its second device's layers recorded
    /// and gone behind the run before's, which is then finished, this chunk's run `pending` in its place.
    pub(super) fn run_rest<'a>(&'a self, parked: &mut Option<(usize, Box<Parked<'a>>)>, pending: &mut Option<ChainedRun<'a>>, embeds: &Tensor, kv: &mut KvCache, done: &mut dyn FnMut(usize), tapped: &mut Vec<llama_rs::Tapped>) {
        if let Some((j, q)) = parked.take() {
            let Some(Went::Run(mut run)) = self.run_part(&[], embeds, kv, false, pending, None, Stage::Rest(q), &[]) else { panic!("a chunk's second device's part") };
            tapped.append(&mut run.tapped);
            if let Some(p) = pending.replace(run) {
                p.finish(self, kv);
            }
            done(j);
        }
    }

    /// An attention layer's rows of a chained run (`t` of them from `at`, K then V each, and its indexer keys) into
    /// the host's cache.
    pub(super) fn cache_rows(&self, kv: &mut KvCache, i: usize, at: usize, t: usize, kvrows: Vec<f32>, raw: Vec<f32>) {
        let Mixer::Attn(a) = &self.layers[i].mixer else { unreachable!("layer {i} attends") };
        let b = self.devices[self.layers[i].device].as_ref();
        let id = self.config.index_dim;
        let len = kv.len;
        kv.len = at;
        kv.append(b, a.index_slot, &Tensor::from_vec(raw.clone(), vec![t, 1, id]), &Tensor::from_vec(raw, vec![t, 1, id]));
        kv.append_rows(b, i, &kvrows, t);
        kv.len = len;
    }

    /// [`Self::run_chained`] up to its last device's wait: every device's work gone (the last's running), `kv`
    /// committed; `prev` (a chunk's run before this one's, its last device still running) finished as the next
    /// device's layers are recorded, so the device holds one chunk's scratch at a time.
    pub(super) fn run_begin<'a>(&'a self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool, prev: &mut Option<ChainedRun<'a>>, ple: Option<Vec<f32>>, taps: &[usize]) -> Option<ChainedRun<'a>> {
        match self.run_part(tokens, embeds, kv, check, prev, ple, Stage::Whole, taps)? {
            Went::Run(run) => Some(run),
            Went::Parked(_) => unreachable!("a whole run stops at no device"),
        }
    }

    /// [`Self::run_begin`] whole, or a prompt's chunk over two devices in two parts ([`Self::forward_chunks`]): its
    /// first device's layers recorded and gone ([`Stage::First`]: `kv` committed, the chunk parked where its next
    /// device's layers begin), then, once the chunk after's first part is gone too, the rest from there
    /// ([`Stage::Rest`]: `tokens` and `embeds` not read, the chunk's own kept with it). So each device has its next
    /// chunk's work behind the one it runs, where a chunk whole left the first device idle while the host recorded
    /// the second's layers and stored rows, and the second while the host recorded the next chunk's first.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_part<'a>(&'a self, tokens: &[u32], embeds: &Tensor, kv: &mut KvCache, check: bool, prev: &mut Option<ChainedRun<'a>>, ple: Option<Vec<f32>>, stage: Stage<'a>, taps: &[usize]) -> Option<Went<'a>> {
        use ggml_rs::{ChainRecorder, DeltaNet};
        use std::sync::atomic::Ordering;
        if std::env::var_os("OAIY_NO_CHAIN").is_some() {
            return None;
        }
        let cfg = &self.config;
        let first_only = matches!(stage, Stage::First);
        let mut parked = match stage {
            Stage::Rest(p) => Some(p),
            _ => None,
        };
        let resumed = parked.is_some();
        let past = parked.as_ref().map_or(kv.len, |p| p.past);
        let t = parked.as_ref().map_or(tokens.len(), |p| p.t);
        let ratio = cfg.index_ratio;
        let phases = std::env::var_os("OAIY_FN_LOG").is_some() && t > 64;
        let clock = std::time::Instant::now();
        let mut marks: Vec<(&str, f64)> = Vec::new();
        let mut mark = |what: &'static str| {
            if phases {
                marks.push((what, clock.elapsed().as_secs_f64() * 1e3));
            }
        };
        // past the dense span the attention layers' queries attend to their QSA blocks (at most 4096 blocks)
        if t == 0 || (past + t) / ratio > 4096 {
            return None;
        }
        let sparse = past + t > cfg.index_budget / ratio * ratio + ratio - 1;
        // a device without QSA's kernels (a workgroup's memory too small for its selection) leaves it to the host path
        if sparse && self.devices.iter().filter_map(|b| b.chain()).any(|c| c.qsa_attention_out_len(1, cfg.heads, cfg.head_dim, cfg.index_budget / ratio, ratio) == 0) {
            return None;
        }
        let st = self.chain_state()?;
        let chains: Vec<&dyn ggml_rs::DeviceChain> = self.devices.iter().map(|b| b.chain()).collect::<Option<_>>()?;
        let (h, s) = (cfg.hidden, cfg.streams);
        let (nh, nkv, hd, rot) = (cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.rope_dim);
        let (kvd, row) = (nkv * hd, 2 * nkv * hd);
        let conv_dim = 2 * cfg.nk * cfg.kd + cfg.nv * cfg.vd;
        let eps = cfg.eps;
        let few = (2..=CHECK_ROWS).contains(&t);
        assert!(!check || few, "a check of 2 to {CHECK_ROWS} rows");
        let id_dim = cfg.index_dim;
        let mut m = st.m.lock().unwrap_or_else(|p| p.into_inner());
        // the devices' copies of the attention caches: room for this run, the rows the host wrote since (a resumed
        // chunk's when it began)
        for (d, layers) in st.attn_of.iter().enumerate().filter(|_| !resumed) {
            let g = &mut m.kv[d];
            if g.cap < past + t {
                let cap = (past + t).next_power_of_two().max(256);
                g.layers = (0..layers.len()).map(|i| match g.layers.get(i) { Some(old) => chains[d].resize(old, cap * row), None => chains[d].vec(cap * row) }).collect();
                g.raw = (0..layers.len()).map(|i| match g.raw.get(i) { Some(old) => chains[d].resize(old, cap * id_dim), None => chains[d].vec(cap * id_dim) }).collect();
                g.out = chains[d].vec(chains[d].attention_out_len(nh, hd, cap));
                g.qsa = None;
                g.cap = cap;
            }
            let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
            if from < past {
                for (slot, &l) in layers.iter().enumerate() {
                    let (kh, vh) = (kv.k_buffer(l).to_host(), kv.v_buffer(l).to_host());
                    let mut rows = Vec::with_capacity((past - from) * row);
                    for t in from..past {
                        rows.extend_from_slice(&kh.data()[t * kvd..(t + 1) * kvd]);
                        rows.extend_from_slice(&vh.data()[t * kvd..(t + 1) * kvd]);
                    }
                    chains[d].upload_at(&g.layers[slot], from * row, &rows);
                    // and its raw indexer keys
                    let Mixer::Attn(a) = &self.layers[l].mixer else { unreachable!("layer {l} attends") };
                    let raw = kv.k_buffer(a.index_slot).to_host();
                    chains[d].upload_at(&g.raw[slot], from * id_dim, &raw.data()[from * id_dim..past * id_dim]);
                }
            }
            g.owner = kv.id;
            if sparse && t <= CHECK_ROWS && !layers.is_empty() && g.qsa.is_none() {
                g.qsa = Some(self.qsa_vecs(chains[d], CHECK_ROWS, g.cap));
            }
        }
        // the delta-net layers' recurrent states as the chain's vectors (the cache aliasing them)
        let mut states = Vec::with_capacity(st.gdn.len());
        for (slot, &l) in st.gdn.iter().enumerate() {
            let c = chains[self.layers[l].device];
            let (ps, pc) = &mut m.pool[slot];
            let adopt = |pool: &mut ggml_rs::DeviceVec, t: &mut Option<Tensor>, shape: Vec<usize>| -> ggml_rs::DeviceVec {
                let len: usize = shape.iter().product();
                if let Some(v) = t.as_ref().and_then(|t| c.aliased(t)).filter(|v| v.len == len) {
                    return v;
                }
                let host = t.take().map(|t| t.to_host());
                if Arc::strong_count(&pool.inner) > 1 {
                    *pool = c.vec(len);
                }
                match host.filter(|h| h.numel() == len) {
                    Some(h) => c.upload(pool, h.data()),
                    None => c.zero(pool),
                }
                *t = Some(c.alias(pool, shape));
                pool.clone()
            };
            let sv = adopt(ps, &mut kv.ssm_state[l], vec![cfg.nv, cfg.vd, cfg.kd]);
            let cv = adopt(pc, &mut kv.ssm_conv[l], vec![cfg.conv - 1, conv_dim]);
            states.push((sv, cv));
        }
        // the n-gram layer's window as the chain's vector (the cache aliasing it), where the layer is chained
        let ple_device = self.layers[cfg.ple_layer].device;
        let ple_window = st.ple.as_ref().map(|_| {
            let c = chains[ple_device];
            let shape = vec![(cfg.ple_kernel - 1) * cfg.ngram, s * h];
            let len: usize = shape.iter().product();
            let slot = ple_slot(cfg);
            if let Some(v) = kv.ssm_conv[slot].as_ref().and_then(|t| c.aliased(t)).filter(|v| v.len == len) {
                return v;
            }
            let host = kv.ssm_conv[slot].take().map(|t| t.to_host());
            if Arc::strong_count(&m.ple_window.inner) > 1 {
                m.ple_window = c.vec(len);
            }
            match host.filter(|h| h.numel() == len) {
                Some(hv) => c.upload(&m.ple_window, hv.data()),
                None => c.zero(&m.ple_window),
            }
            kv.ssm_conv[slot] = Some(c.alias(&m.ple_window, shape));
            m.ple_window.clone()
        });
        // The states the run keeps from inside it (`taps`: after that many of its rows, ascending; a resumed chunk's
        // its own): a pair of vectors a delta net and one for the n-gram layer's window, and what the caller is given
        // of them: a checkpoint's tensors, as the cache holds its own (the n-gram layer's history its tokens').
        let (kept, mut tapped): (Vec<Tap>, Vec<llama_rs::Tapped>) = match parked.as_mut() {
            Some(p) => (std::mem::take(&mut p.kept), std::mem::take(&mut p.tapped)),
            None if taps.is_empty() => (Vec::new(), Vec::new()),
            None => {
                assert!(!check && taps.windows(2).all(|w| w[0] < w[1]) && taps[0] > 0 && taps[taps.len() - 1] <= t && t - taps[0] <= TAP_ROWS, "a run of {t} rows keeps states after {taps:?}");
                let before = self.ple_history(kv);
                let ctx = cfg.ngram - 1;
                let kept: Vec<Tap> = taps
                    .iter()
                    .map(|&row| Tap {
                        row,
                        gdn: st.gdn.iter().zip(&states).map(|(&l, (sv, cv))| (chains[self.layers[l].device].vec(sv.len), chains[self.layers[l].device].vec(cv.len))).collect(),
                        window: ple_window.as_ref().map(|w| chains[ple_device].vec(w.len)),
                    })
                    .collect();
                let tapped = kept
                    .iter()
                    .map(|tap| {
                        let none = || (0..kv.ssm_state.len()).map(|_| None).collect::<Vec<Option<Tensor>>>();
                        let (mut ss, mut cs) = (none(), none());
                        for (slot, &l) in st.gdn.iter().enumerate() {
                            let c = chains[self.layers[l].device];
                            ss[l] = Some(c.alias(&tap.gdn[slot].0, vec![cfg.nv, cfg.vd, cfg.kd]));
                            cs[l] = Some(c.alias(&tap.gdn[slot].1, vec![cfg.conv - 1, conv_dim]));
                        }
                        if let Some(w) = &tap.window {
                            cs[ple_slot(cfg)] = Some(chains[ple_device].alias(w, vec![(cfg.ple_kernel - 1) * cfg.ngram, s * h]));
                        }
                        let ids: Vec<i64> = before.iter().copied().chain(tokens[..tap.row].iter().map(|&v| v as i64)).collect();
                        ss[ple_slot(cfg)] = Some(Tensor::from_vec(ids[ids.len() - ctx..].iter().map(|&v| v as f32).collect(), vec![ctx]));
                        llama_rs::Tapped { at: past + tap.row, states: ss, convs: cs }
                    })
                    .collect();
                (kept, tapped)
            }
        };
        let segs: Option<&Vec<TapSeg>> = kept.first().filter(|tap| tap.row < t).map(|_| {
            st.seg.get_or_init(|| {
                chains
                    .iter()
                    .map(|c| {
                        let v = |n: usize| c.vec(TAP_ROWS * n);
                        TapSeg { qkv: v(conv_dim), conv: v(conv_dim), z: v(cfg.nv * cfg.vd), ba: v(2 * cfg.nv), core: v(cfg.nv * cfg.vd), x: v(s * h), gated: v(s * h), conv_in: v(s * h) }
                    })
                    .collect()
            })
        });
        // a step's vectors and a few rows' (their bind groups kept); a prompt's chunk's its own
        let keep = t == 1 || few;
        let few_set = few.then(|| {
            st.few[t - 2].get_or_init(|| {
                let pd = self.layers[cfg.ple_layer].device;
                FewSet {
                    devs: chains.iter().map(|c| chain_dev(*c, cfg, st.rank, t, t)).collect(),
                    ple: st.ple.as_ref().map(|_| ple_vecs(chains[pd], cfg, t)),
                    keys: chains.iter().zip(&st.attn_of).map(|(c, a)| a.iter().map(|_| c.vec(t * cfg.index_dim)).collect()).collect(),
                    q1: chains.iter().map(|c| c.vec(cfg.heads * cfg.head_dim)).collect(),
                }
            })
        });
        let owned: Option<Arc<Vec<ChainDev>>> = (t != 1 && few_set.is_none()).then(|| match &parked {
            Some(p) => Arc::clone(&p.devs),
            None => Arc::new(chains.iter().map(|c| chain_dev(*c, cfg, st.rank, t, 1)).collect()),
        });
        let devs: &[ChainDev] = match (&owned, few_set) {
            (Some(o), _) => o,
            (None, Some(f)) => &f.devs,
            (None, None) => &st.devs,
        };
        // a check: what undoing it needs (the delta nets' and the n-gram window's backups are taken as it runs)
        if check {
            if m.undo.is_none() {
                let ple_window = st.ple.as_ref().map(|_| chains[self.layers[cfg.ple_layer].device].vec((cfg.ple_kernel.max(1) - 1) * cfg.ngram * s * h));
                let (backups, inputs) = st
                    .gdn
                    .iter()
                    .map(|&l| {
                        let c = chains[self.layers[l].device];
                        ((c.vec(cfg.nv * cfg.vd * cfg.kd), c.vec((cfg.conv - 1) * conv_dim)), (c.vec(CHECK_ROWS * conv_dim), c.vec(CHECK_ROWS * 2 * cfg.nv)))
                    })
                    .unzip();
                m.undo = Some(Undo { backups, inputs, window: ple_window, tokens: Vec::new(), history: None });
            }
            let u = m.undo.as_mut().expect("made");
            u.tokens = tokens.to_vec();
            u.history = kv.ssm_state[ple_slot(cfg)].clone();
        }
        // the partial RoPE's sines and cosines at these positions, on every device: the next ones in order, or the
        // ones the run was given, a pair's angle then its axis's position (time, height, width, interleaved over the
        // pairs by Qwen's sections of 11, 11 and 10, as the host's `multimodal_rope::text` rotates them)
        let axis = |k: usize| if k % 3 == 1 && k < 33 { 1 } else if k % 3 == 2 && k < 30 { 2 } else { 0 };
        let given = self.placed.lock().unwrap_or_else(|p| p.into_inner()).clone().filter(|p| p.len() == t);
        let table: Vec<f32> = (0..t)
            .flat_map(|row| {
                let at = given.as_ref().map(|p| p[row]);
                (0..rot / 2).flat_map(move |k| {
                    let pos = at.map_or(past + row, |p| p[axis(k)] as usize);
                    let (sn, cs) = (pos as f32 * cfg.rope_theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                    [sn, cs]
                })
            })
            .collect();
        for (c, dv) in chains.iter().zip(devs).filter(|_| !resumed) {
            c.upload(&dv.table, &table);
        }
        // a prompt's attention scratch, on every device with attention layers (a few rows' kept, grown as positions are)
        let attn_rows: Vec<Option<ggml_rs::DeviceVec>> = match &parked {
            Some(p) => p.attn_rows.clone(),
            None => chains
                .iter()
                .enumerate()
                .map(|(d, c)| {
                    if t == 1 || st.attn_of[d].is_empty() {
                        return None;
                    }
                    let len = c.attention_rows_out_len(t, nh, hd, past + t);
                    if !few {
                        return Some(c.vec(len));
                    }
                    if m.rows_out[d].len < len {
                        m.rows_out[d] = c.vec(len.next_power_of_two());
                    }
                    Some(m.rows_out[d].clone())
                })
                .collect(),
        };
        // a prompt's chunk's QSA vectors (past the dense span) and its rows' raw indexer keys, on every device with
        // attention layers
        let prompt_qsa: Arc<Vec<Option<QsaVecs>>> = match &parked {
            Some(p) => Arc::clone(&p.qsa),
            None => Arc::new(chains.iter().enumerate().map(|(d, c)| (sparse && t > CHECK_ROWS && !st.attn_of[d].is_empty()).then(|| self.qsa_vecs(*c, t, m.kv[d].cap))).collect()),
        };
        let prompt_keys: Vec<Option<ggml_rs::DeviceVec>> = match &parked {
            Some(p) => p.keys.clone(),
            None => chains.iter().enumerate().map(|(d, c)| (t > CHECK_ROWS && !st.attn_of[d].is_empty()).then(|| c.vec(t * id_dim))).collect(),
        };
        // the n-gram features (on the n-gram layer's device, where it is chained), then the embedding in every stream
        // (a prompt's chunk's read already, as the chunk before ran)
        let ple_emb = match ple {
            _ if resumed => Vec::new(),
            Some(f) => {
                let mut history = self.ple_history(kv);
                history.extend(tokens.iter().map(|&t| t as i64));
                self.ple_carry(&history, kv);
                f
            }
            None => self.ple_embed_host(tokens, kv).ok()?,
        };

        let ple_owned: Option<Arc<PleVecs>> = (st.ple.is_some() && t != 1 && !few).then(|| match &parked {
            Some(p) => Arc::clone(p.ple.as_ref().expect("the chunk's n-gram vectors")),
            None => Arc::new(ple_vecs(chains[ple_device], cfg, t)),
        });
        let ple_vs: Option<(&ChainPle, &PleVecs)> = match &st.ple {
            Some((p, step)) if t == 1 => Some((p, step)),
            Some((p, _)) if few => few_set.and_then(|f| f.ple.as_ref()).map(|v| (p, v)),
            Some((p, _)) => ple_owned.as_deref().map(|v| (p, v)),
            None => None,
        };
        // where the layers go on from: the first, or a resumed chunk's next device's first
        let start = parked.as_ref().map_or(0, |p| p.at);
        let mut d = self.layers[start].device;
        let e = match &parked {
            Some(p) => p.e.clone(),
            None => embeds.to_host(),
        };
        if !resumed {
            if let Some((_, v)) = ple_vs {
                chains[ple_device].upload(&v.emb, &ple_emb);
            }
            let mut x0 = Vec::with_capacity(t * s * h);
            for row in e.data().chunks_exact(h).take(t) {
                for _ in 0..s {
                    x0.extend_from_slice(row);
                }
            }
            chains[d].upload(&devs[d].x, &x0);
        }
        let hc = |rec: &mut dyn ChainRecorder, dv: &ChainDev, rows: usize, hcv: &HcVecs, pending: Option<(&ggml_rs::DeviceVec, &ggml_rs::DeviceVec)>, post: &ggml_rs::DeviceVec, out: &ggml_rs::DeviceVec| {
            if let Some((y, p)) = pending {
                rec.stream_apply(&dv.x, y, p, rows, s, h);
            }
            rec.rmsnorm_streams(&dv.x, &hcv.norm, &dv.normed, rows, s, eps);
            hcv.project(&mut *rec, &dv.normed, &dv.t, post, &dv.logits, out, rows, s, h);
        };
        // The experts are routed on their GPU (each layer's router then its experts, a device's layers one submit; a
        // prompt's rows grouped by expert there too, where the tensor cores take them), else by the host between a
        // layer's submits (how many rows each expert takes is what the dispatches are sized by). OAIY_HOST_ROUTE
        // routes on the host.
        let on_gpu = cfg.experts <= 1024 && cfg.top_k <= 32 && std::env::var_os("OAIY_HOST_ROUTE").is_none();
        let undo = m.undo.as_ref().filter(|_| check);
        // a prompt's rows as the host routed them (between a layer's submits); on the GPU, each layer's are recorded
        // after its router, their sums added to the streams there
        enum Routed {
            Host(Vec<Vec<(usize, f32)>>),
            /// A layer's experts run on the host (none of its GPU's): their sums, for the device they go up to.
            Summed(usize, Vec<f32>),
        }
        let experts = |rec: &mut dyn ChainRecorder, dv: &ChainDev, layer: usize, routed: Routed| match routed {
            Routed::Host(assign) => rec.moe_rows(self.layers[layer].moe.experts.as_ref(), &dv.y2_in, &dv.moe_out, &assign),
            Routed::Summed(device, sums) => chains[device].upload(&dv.moe_out, &sums),
        };
        // an attention layer's K and V rows and its indexer keys (`[t, index_dim]`), read back, into the host's cache
        let (iq, id) = (cfg.index_heads * cfg.index_dim, cfg.index_dim);
        let to_cache = |i: usize, kvrows: Vec<f32>, raw: Vec<f32>, kv: &mut KvCache| self.cache_rows(kv, i, past, t, kvrows, raw);
        // the recording open on device `d`, and the attention layers whose rows and keys it reads (in order, before
        // whatever else it reads)
                let mut open: Option<Box<dyn ChainRecorder + '_>> = None;
        let mut attn_reads: Vec<usize> = Vec::new();
        // a device's attention layers' rows and keys read at its handoff, into the host's cache once the next device's
        // layers are recorded (as that device runs them)
        let mut to_store: Vec<(usize, Vec<f32>, Vec<f32>)> = Vec::new();
        // a device's recording (its work submitted, its reads its streams and its layers' rows) whose streams go up
        // to the next device (its work held until they do), and its attention layers read
        let mut handoffs: Vec<(Box<dyn ChainRecorder + '_>, usize, Vec<usize>)> = parked.as_mut().map(|p| std::mem::take(&mut p.handoffs)).unwrap_or_default();
        let mut pending: Option<Routed> = None;
        // (OAIY_FN_HOST_LOG: what the run's host layers cost, the waits for their read-backs and their experts)
        static HOST_LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let host_log = *HOST_LOG.get_or_init(|| std::env::var_os("OAIY_FN_HOST_LOG").is_some());
        let (mut host_layers, mut host_waited, mut host_ran) = (0usize, 0f64, 0f64);
        if resumed {
            // (as after a handoff: this device's work held till the streams are up)
            let mut r = chains[d].begin();
            r.keep_groups(keep);
            // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
            r.rows_alike(keep);
            r.hold();
            open = Some(r);
        }
        mark("ready");
        for (i, (layer, cl)) in self.layers.iter().zip(&st.layers).enumerate().skip(start) {
            let dev = layer.device;
            let ple_here = i == cfg.ple_layer;
            if dev != d && pending.is_none() && !(ple_here && ple_vs.is_none()) {
                // the next device's layers recorded as this one runs its own (its streams uploaded to the next once
                // it has, the next's work held until then)
                let mut rec = open.take().unwrap_or_else(|| {
                    let mut r = chains[d].begin();
                    r.keep_groups(keep);
                    // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                    r.rows_alike(keep);
                    r
                });
                rec.read(&devs[d].x);
                rec.flush();
                mark("first device recorded and gone");
                handoffs.push((rec, dev, std::mem::take(&mut attn_reads)));
                if first_only {
                    // the chunk parked here: its next device's layers after the next chunk's first part
                    kv.commit(t);
                    kv.dirty_from = usize::MAX;
                    return Some(Went::Parked(Box::new(Parked { past, t, e, devs: owned.clone()?, ple: ple_owned.clone(), attn_rows, qsa: prompt_qsa, keys: prompt_keys, handoffs, at: i, kept, tapped })));
                }
                // the chunk before's last device done with (its scratch back) before this chunk's work there
                                if let Some(p) = prev.take() {
                    p.finish(self, kv);
                }
                mark("the chunk before finished");
                                d = dev;
                let mut r = chains[d].begin();
                r.keep_groups(keep);
                // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                r.rows_alike(keep);
                r.hold();
                open = Some(r);
            } else if dev != d || ple_here && ple_vs.is_none() {
                if let Some(p) = prev.take() {
                    p.finish(self, kv);
                }
                // (a handoff before this one's first: its streams up to its next device before that one's held work
                // is finished below)
                for (from, to, reads) in handoffs.drain(..) {
                    let mut got = from.finish().into_iter();
                    for &a in &reads {
                        let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                        to_store.push((a, kvrows, raw));
                    }
                    chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
                }
                // what this device has pending, then the streams through the host (to the next device, or the
                // n-gram layer where it is not chained)
                let dv = &devs[d];
                let mut rec = open.take().unwrap_or_else(|| {
                    let mut r = chains[d].begin();
                    r.keep_groups(keep);
                    // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                    r.rows_alike(keep);
                    r
                });
                if let Some(routed) = pending.take() {
                    experts(&mut *rec, dv, i - 1, routed);
                    rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
                }
                rec.read(&dv.x);
                let mut got = rec.finish().into_iter();
                for &a in &attn_reads {
                    let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                    to_store.push((a, kvrows, raw));
                }
                attn_reads.clear();
                let mut x = Tensor::from_vec(got.next().expect("the streams"), vec![t, s * h]);
                if ple_here && ple_vs.is_none() {
                    let b = self.devices[dev].as_ref();
                    let emb = b.to_device(Tensor::from_vec(ple_emb.clone(), vec![t, cfg.ple_dim]));
                    x = self.ple_forward(b, &b.to_device(x), &emb, kv).to_host();
                }
                d = dev;
                chains[d].upload(&devs[d].x, x.data());
            }
            let dv = &devs[d];
            let rec: &mut dyn ChainRecorder = &mut **open.get_or_insert_with(|| {
                let mut r = chains[d].begin();
                r.keep_groups(keep);
                // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
                r.rows_alike(keep);
                // a prompt's chunk's work on its first device held till its handoff too, then let go at once: its
                // pieces submitted one by one as they were recorded ran the GPU's kernels some three times as long
                // (a chunk of 512 some 340 ms of them where held 100; a step's own few go as they are)
                if t > CHECK_ROWS && self.devices.len() > 1 {
                    r.hold();
                }
                r
            });
            if let (true, Some((p, v)), Some(window)) = (ple_here, ple_vs, &ple_window) {
                // the n-gram layer on its device: the last layer's experts written back first
                if let Some(routed) = pending.take() {
                    experts(&mut *rec, dv, i - 1, routed);
                    rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
                }
                if let Some(backup) = undo.and_then(|u| u.window.as_ref()) {
                    rec.copy(window, 0, backup, 0, window.len);
                }
                p.key.mul(&mut *rec, s * h, cfg.ple_dim, &v.emb, &v.key, t);
                p.value.mul(&mut *rec, h, cfg.ple_dim, &v.emb, &v.value, t);
                rec.ple_gate(&v.key, &dv.x, &v.value, &p.norm_key, &p.norm_query, &p.norm_conv, &v.gated, &v.conv_in, t, s, h, eps);
                if kept.is_empty() {
                    rec.ple_conv(&dv.x, &v.gated, &v.conv_in, window, &p.conv, t, s * h, cfg.ple_kernel, cfg.ngram);
                } else {
                    // the conv in parts, each kept state's rows then the window's copy: the first part where the rows
                    // are, the later ones (few) through the parts' own vectors and back
                    let w = s * h;
                    let mut done = 0;
                    for (end, to) in kept.iter().map(|tap| (tap.row, tap.window.as_ref())).chain(std::iter::once((t, None))) {
                        let n = end - done;
                        if n > 0 && done == 0 {
                            rec.ple_conv(&dv.x, &v.gated, &v.conv_in, window, &p.conv, n, w, cfg.ple_kernel, cfg.ngram);
                        } else if n > 0 {
                            let seg = &segs.expect("a tapped run's later rows' vectors")[d];
                            let (sx, sg, sc) = (first_of(&seg.x, n * w), first_of(&seg.gated, n * w), first_of(&seg.conv_in, n * w));
                            rec.copy(&dv.x, done * w, &sx, 0, n * w);
                            rec.copy(&v.gated, done * w, &sg, 0, n * w);
                            rec.copy(&v.conv_in, done * w, &sc, 0, n * w);
                            rec.ple_conv(&sx, &sg, &sc, window, &p.conv, n, w, cfg.ple_kernel, cfg.ngram);
                            rec.copy(&sx, 0, &dv.x, done * w, n * w);
                        }
                        if let Some(to) = to {
                            rec.copy(window, 0, to, 0, window.len);
                        }
                        done = end;
                    }
                }
            }
            let applied = pending.take().map(|routed| experts(&mut *rec, dv, i - 1, routed)).is_some();
            hc(&mut *rec, dv, t, &cl.attn_hc, applied.then_some((&dv.moe_out, &dv.post2)), &dv.post, &dv.y_in);
            match (&layer.mixer, &cl.mixer) {
                (Mixer::Gdn(g), ChainMixer::Gdn { ba, conv, a, dt, norm, slot }) => {
                    let (sv, cv) = &states[*slot];
                    // a check's qkv and beta-alpha rows where undoing it finds them, its state and window backed up
                    let (qkv, bav) = match undo {
                        Some(u) => (&u.inputs[*slot].0, &u.inputs[*slot].1),
                        None => (&dv.qkv, &dv.ba),
                    };
                    rec.exl3_rows(chain_packed(&g.qkv)?, &dv.y_in, qkv, t);
                    rec.exl3_rows(chain_packed(&g.z)?, &dv.y_in, &dv.z, t);
                    ba.mul(&mut *rec, 2 * cfg.nv, h, &dv.y_in, bav, t);
                    if let Some(u) = undo {
                        rec.copy(sv, 0, &u.backups[*slot].0, 0, sv.len);
                        rec.copy(cv, 0, &u.backups[*slot].1, 0, cv.len);
                    }
                    let dn = DeltaNet { rows: t, v_heads: cfg.nv, k_heads: cfg.nk, k_dim: cfg.kd, v_dim: cfg.vd, scale_q: 1.0 / (cfg.vd as f32).sqrt(), eps, sigmoid_gate: true };
                    if kept.is_empty() {
                        rec.ssm_conv(qkv, conv, cv, &dv.conv, t, conv_dim, cfg.conv);
                        rec.delta_net(&dv.conv, &dv.z, bav, a, dt, norm, sv, &dv.core, dn);
                    } else {
                        // the recurrence in parts, each kept state's rows then its copies (as the n-gram layer's)
                        let vw = cfg.nv * cfg.vd;
                        let mut done = 0;
                        for (end, to) in kept.iter().map(|tap| (tap.row, Some(&tap.gdn[*slot]))).chain(std::iter::once((t, None))) {
                            let n = end - done;
                            if n > 0 && done == 0 {
                                rec.ssm_conv(qkv, conv, cv, &dv.conv, n, conv_dim, cfg.conv);
                                rec.delta_net(&dv.conv, &dv.z, bav, a, dt, norm, sv, &dv.core, DeltaNet { rows: n, ..dn });
                            } else if n > 0 {
                                let seg = &segs.expect("a tapped run's later rows' vectors")[d];
                                let (sq, sc, sz, sb, so) = (first_of(&seg.qkv, n * conv_dim), first_of(&seg.conv, n * conv_dim), first_of(&seg.z, n * vw), first_of(&seg.ba, n * 2 * cfg.nv), first_of(&seg.core, n * vw));
                                rec.copy(qkv, done * conv_dim, &sq, 0, n * conv_dim);
                                rec.copy(&dv.z, done * vw, &sz, 0, n * vw);
                                rec.copy(bav, done * 2 * cfg.nv, &sb, 0, n * 2 * cfg.nv);
                                rec.ssm_conv(&sq, conv, cv, &sc, n, conv_dim, cfg.conv);
                                rec.delta_net(&sc, &sz, &sb, a, dt, norm, sv, &so, DeltaNet { rows: n, ..dn });
                                rec.copy(&so, 0, &dv.core, done * vw, n * vw);
                            }
                            if let Some((ts, tc)) = to {
                                rec.copy(sv, 0, ts, 0, sv.len);
                                rec.copy(cv, 0, tc, 0, cv.len);
                            }
                            done = end;
                        }
                    }
                    rec.exl3_rows(chain_packed(&g.out)?, &dv.core, &dv.y_out, t);
                }
                (Mixer::Attn(a), ChainMixer::Attn { q_norm, k_norm, iq_norm, ik_norm, slot }) => {
                    let g = &m.kv[d];
                    let kvl = &g.layers[*slot];
                    rec.exl3_rows(chain_packed(&a.q)?, &dv.y_in, &dv.qfull, t);
                    rec.exl3_rows(chain_packed(&a.k)?, &dv.y_in, &dv.k, t);
                    rec.exl3_rows(chain_packed(&a.v)?, &dv.y_in, &dv.v, t);
                    rec.exl3_rows(chain_packed(&a.index_qk)?, &dv.y_in, &dv.index, t);
                    if on_gpu {
                        // the run's indexer keys out of the vector the device's next attention layer writes (a
                        // prompt's are read from the device's copy of them)
                        match few_set {
                            Some(f) => rec.copy_cols(&dv.index, &f.keys[d][*slot], t, id, iq + id, iq),
                            None if t == 1 => rec.copy(&dv.index, iq, &st.keys[d], slot * id, id),
                            None => {}
                        }
                    }
                    // and into the device's copy of the raw keys (QSA's pool reads them past the dense span)
                    match (t, few_set, &prompt_keys[d]) {
                        (1, _, _) => rec.copy(&dv.index, iq, &g.raw[*slot], past * id, id),
                        (_, Some(f), _) => rec.store_rows(&f.keys[d][*slot], &g.raw[*slot], t, id, past, id, 0),
                        (_, None, Some(keys)) => {
                            rec.copy_cols(&dv.index, keys, t, id, iq + id, iq);
                            rec.store_rows(keys, &g.raw[*slot], t, id, past, id, 0);
                        }
                        _ => unreachable!("a run's raw keys have a place"),
                    }
                    rec.copy_cols(&dv.qfull, &dv.q, t * nh, hd, 2 * hd, 0);
                    rec.copy_cols(&dv.qfull, &dv.gate, t * nh, hd, 2 * hd, hd);
                    rec.rmsnorm_rows(&dv.q, q_norm, &dv.qn, t * nh, eps);
                    rec.rmsnorm_rows(&dv.k, k_norm, &dv.kn, t * nkv, eps);
                    rec.rope_partial_rows(&dv.qn, t, nh, hd, rot, &dv.table);
                    rec.rope_partial_rows(&dv.kn, t, nkv, hd, rot, &dv.table);
                    rec.store_rows(&dv.kn, kvl, t, kvd, past, row, 0);
                    rec.store_rows(&dv.v, kvl, t, kvd, past, row, kvd);
                    let scale = 1.0 / (hd as f32).sqrt();
                    let qsa = g.qsa.as_ref().filter(|q| q.rows >= t).or(prompt_qsa[d].as_ref());
                    let out = match (&attn_rows[d], qsa) {
                        (_, Some(q)) if sparse => {
                            // past the dense span: the indexer's queries, the cache's pooled block keys, each row's top
                            // blocks, and its attention over them and its tail block
                            let (ih, nb, keep) = (cfg.index_heads, (past + t) / ratio, cfg.index_budget / ratio);
                            let (iqv, iqn, pooled, pooledn) = (first(&q.iq, t * ih * id), first(&q.iqn, t * ih * id), first(&q.pooled, nb * id), first(&q.pooledn, nb * id));
                            rec.copy_cols(&dv.index, &iqv, t, ih * id, iq + id, 0);
                            rec.rmsnorm_rows(&iqv, iq_norm, &iqn, t * ih, eps);
                            rec.rope_partial_rows(&iqn, t, ih, id, rot, &dv.table);
                            rec.qsa_pool(&g.raw[*slot], &pooled, nb, ratio, id);
                            rec.rmsnorm_rows(&pooled, ik_norm, &pooledn, nb, eps);
                            rec.rope_partial_rows(&pooledn, nb, 1, id, rot, &q.table);
                            rec.qsa_scores(&iqn, &pooledn, &q.scores, t, ih, id, nb, past, ratio, 1.0 / (id as f32).sqrt());
                            rec.qsa_select(&q.scores, &q.list, t, nb, past, ratio, keep);
                            rec.qsa_attention(&dv.qn, kvl, &q.list, &q.out, t, nh, nkv, hd, past, ratio, keep, scale);
                            &q.out
                        }
                        (Some(scratch), _) if few => {
                            // a row at a time as a step attends (the decode kernel's sums: a check's rows a step's bit
                            // for bit)
                            let (q1, qh) = (&few_set.expect("a few rows' vectors").q1[d], nh * hd);
                            for r in 0..t {
                                rec.copy(&dv.qn, r * qh, q1, 0, qh);
                                rec.attention(q1, kvl, &g.out, nh, nkv, hd, 0, past + r + 1, g.cap, scale);
                                rec.copy(&g.out, 0, scratch, r * qh, qh);
                            }
                            scratch
                        }
                        (Some(scratch), _) => {
                            rec.attention_rows(&dv.qn, kvl, scratch, t, nh, nkv, hd, past, None, scale);
                            scratch
                        }
                        (None, _) => {
                            rec.attention(&dv.qn, kvl, &g.out, nh, nkv, hd, 0, past + 1, g.cap, scale);
                            &g.out
                        }
                    };
                    rec.mul_sigmoid(out, &dv.gate, &dv.gated, t * nh * hd);
                    rec.exl3_rows(chain_packed(&a.o)?, &dv.gated, &dv.y_out, t);
                }
                _ => unreachable!("layer {i}'s vectors are its mixer's"),
            }
            hc(&mut *rec, dv, t, &cl.mlp_hc, Some((&dv.y_out, &dv.post)), &dv.post2, &dv.y2_in);
            cl.router.mul(&mut *rec, cfg.experts + 1, h, &dv.y2_in, &dv.router, t);
            if on_gpu && !cl.host && rec.moe_routed_into(layer.moe.experts.as_ref(), &dv.y2_in, &dv.x, &dv.post2, &dv.router, cfg.top_k, t, s) {
                // the experts after it, on the device, their sums into the streams; the recording goes on
                if let ChainMixer::Attn { slot, .. } = &cl.mixer {
                    rec.read_range(&m.kv[d].layers[*slot], past * row, t * row);
                    match few_set {
                        Some(f) => rec.read(&f.keys[d][*slot]),
                        None if t == 1 => rec.read_range(&st.keys[d], slot * id, id),
                        None => rec.read_range(&m.kv[d].raw[*slot], past * id, t * id),
                    }
                    attn_reads.push(i);
                }
                continue;
            }
            // The host between this layer's submits: to route its rows, or (a layer no GPU had room for) to run its
            // experts on their input.
            let width = cfg.experts + 1;
            // (a host layer's shared expert on the device where it is there: its outputs read back with the rest)
            let shared = cl.host && rec.moe_shared(layer.moe.experts.as_ref(), &dv.y2_in, &dv.moe_out, t);
            rec.read(&dv.router);
            if cl.host {
                rec.read_range(&dv.y2_in, 0, t * h);
            }
            if shared {
                rec.read_range(&dv.moe_out, 0, t * h);
            }
            let attn_slot = match &cl.mixer {
                ChainMixer::Attn { slot, .. } => {
                    rec.read_range(&m.kv[d].layers[*slot], past * row, t * row);
                    rec.read(&dv.index);
                    true
                }
                _ => false,
            };
            // (what this device's recording waits on first: the chunk before's run, and the streams of the device
            // before, whose handoff holds this one's work)
            if let Some(p) = prev.take() {
                p.finish(self, kv);
            }
            for (from, to, reads) in handoffs.drain(..) {
                let mut got = from.finish().into_iter();
                for &a in &reads {
                    let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                    to_store.push((a, kvrows, raw));
                }
                chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
            }
            let waiting = std::time::Instant::now();
            let mut got = open.take().expect("the layer's recording").finish().into_iter();
            host_waited += waiting.elapsed().as_secs_f64() * 1e3;
            // (the device's attention layers before this one, routed on it: their rows and keys are this recording's)
            for a in attn_reads.drain(..) {
                let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                to_store.push((a, kvrows, raw));
            }
            let logits = got.next().expect("the router's logits");
            let input = cl.host.then(|| got.next().expect("the experts' input"));
            let shared = shared.then(|| got.next().expect("the shared expert's outputs"));
            if attn_slot {
                let (kvrows, index) = (got.next().expect("the run's K and V"), got.next().expect("the run's indexer keys"));
                let raw: Vec<f32> = index.chunks_exact(iq + id).take(t).flat_map(|r| r[iq..].iter().copied()).collect();
                to_cache(i, kvrows, raw, kv);
            }
            pending = Some(match input {
                Some(x) => {
                    let running = std::time::Instant::now();
                    let (x, logits) = (Tensor::from_vec(x, vec![t, h]), Tensor::from_vec(logits[..t * width].to_vec(), vec![t, width]));
                    let sums = match &shared {
                        Some(shared) => layer.moe.experts.forward_given(&x, &logits, cfg.top_k, shared),
                        None => layer.moe.experts.forward(&x, &logits, cfg.top_k),
                    };
                    host_layers += 1;
                    host_ran += running.elapsed().as_secs_f64() * 1e3;
                    Routed::Summed(d, sums.to_host().data().to_vec())
                }
                None => {
                    let assign: Vec<Vec<(usize, f32)>> = (0..t).map(|r| ggml_rs::exl3::route(&logits[r * width..(r + 1) * width], cfg.top_k)).collect();
                    route_dump(i, past, &assign);
                    Routed::Host(assign)
                }
            });
        }
        // the last layer's experts, its write-back, the streams' collapse and the head, on the last device (the
        // chain's state saw to it)
        debug_assert_eq!(d, self.devices.len() - 1);
        let dv = &devs[d];
        let mut rec = open.take().unwrap_or_else(|| {
            let mut r = chains[d].begin();
            r.keep_groups(keep);
            // (a step's row by a check's kernels: a check's rows its steps' bit for bit, and its IQ4_XS matrices the faster)
            r.rows_alike(keep);
            r
        });
        if let Some(routed) = pending.take() {
            experts(&mut *rec, dv, cfg.layers - 1, routed);
            rec.stream_apply(&dv.x, &dv.moe_out, &dv.post2, t, s, h);
        }
        // the trunk's streams for the prediction layer: a step's or a check's rows; a prompt's chunk fills the layer's
        // cache over its positions (but the last, whose next token it does not have) and keeps its last row
        let mut hid = None;
        if let (Some(mc), Some(mp)) = (&st.mtp, &self.mtp) {
            if t <= CHECK_ROWS {
                rec.copy(&dv.x, 0, &mc.hid, 0, t * s * h);
                hid = Some((past, t));
            } else {
                self.mtp_prompt(mp, mc, &mut m, chains[d], &mut *rec, &e, &dv.x, t, past, kv.id);
                rec.copy(&dv.x, (t - 1) * s * h, &mc.hid, 0, s * h);
                hid = Some((past + t - 1, 1));
            }
        }
        // (a greedy request's run: its rows' tokens picked here, those read where megabytes of logits were)
        let pick = self.pick.load(Ordering::Relaxed);
        if check {
            // every row's streams collapsed, then the head
            hc(&mut *rec, dv, t, &st.collapse, None, &dv.post, &dv.mixed);
            rec.exl3_rows(chain_packed(&self.head)?, &dv.mixed, &dv.head, t);
            if pick {
                rec.argmax_rows(&dv.head, t, cfg.vocab, &dv.picks);
                rec.read(&dv.picks);
            } else {
                rec.read(&dv.head);
            }
        } else {
            // the last row's streams collapsed (in a step's own vectors), then the head
            let one = &st.devs[d];
            if t > 1 {
                rec.copy(&dv.x, (t - 1) * s * h, &one.x, 0, s * h);
            }
            hc(&mut *rec, one, 1, &st.collapse, None, &one.post, &one.mixed);
            rec.exl3_rows(chain_packed(&self.head)?, &one.mixed, &one.head, 1);
            if pick {
                rec.argmax_rows(&one.head, 1, cfg.vocab, &one.picks);
                rec.read(&one.picks);
            } else {
                rec.read(&one.head);
            }
        }
        mark("last device recorded");
        if host_log && host_layers > 0 {
            eprintln!("    fn run of {t} at {past}: {host_layers} layers' experts on the host {host_ran:.2} ms, the waits for their inputs {host_waited:.2} ms, {:.2} ms in all so far", clock.elapsed().as_secs_f64() * 1e3);
        }
                // each handoff in turn: its device's streams (once it has run) up to the next, whose held work then goes
        for (from, to, reads) in handoffs.drain(..) {
            let mut got = from.finish().into_iter();
            mark("first device waited for and read");
            for &a in &reads {
                let (kvrows, raw) = (got.next().expect("a layer's K and V"), got.next().expect("its indexer keys"));
                to_store.push((a, kvrows, raw));
            }
            chains[to].upload(&devs[to].x, &got.next().expect("the streams"));
        }
        mark("streams up");
                // the last device's work going (held till its streams were up) as the host stores the others' rows
        rec.flush();
        mark("last device gone");
                if let Some(p) = prev.take() {
            p.finish(self, kv);
        }
        for (a, kvrows, raw) in to_store.drain(..) {
            to_cache(a, kvrows, raw, kv);
        }
        if let Some(at) = hid {
            m.mtp_hid = at;
        }
        if !resumed {
            kv.commit(t);
        }
        kv.dirty_from = usize::MAX;
        if t == 1 {
            self.decoded.store(true, Ordering::Relaxed);
        }
        st.runs.fetch_add(1, Ordering::Relaxed);
        mark("rows stored");
        if phases {
            eprintln!("    fn run of {t} at {past}: {}", marks.iter().map(|(w, ms)| format!("{w} {ms:.0}")).collect::<Vec<_>>().join(", "));
        }
        drop(kept);
        Some(Went::Run(ChainedRun { rec, attn_reads, at: past, t, tapped: std::mem::take(&mut tapped) }))
    }
}
