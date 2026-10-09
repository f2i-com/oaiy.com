//! What Flash-Next's chain holds on its devices: a layer's matrices and vectors, the caches' device copies, the
//! prediction layer's and the n-gram layer's, what a check keeps to undo; and the making of it (`chain_state`).

use super::*;

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
}
