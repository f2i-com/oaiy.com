//! DeepSeek-V4.1 attention (reference `Attention`, `Compressor`, `Indexer`,
//! `select_candidate_blocks`, `sparse_attn`).
//!
//! Every layer attends over a 128-token sliding window of its own KV. Layers
//! with `compress_ratio` > 0 also attend over up to `index_topk` compressed
//! positions reaching further back:
//!
//! - the *kv-source* layers (2, 8, 14, 20) pool `ratio` tokens into one
//!   latent (`Compressor`) and own the compressed-KV cache; the layers after
//!   a source read that source's cache;
//! - the *index-source* layers (2, 8, 14, 20, 24, 28, 32, 36) score the
//!   compressed positions with a small fp4-simulated side attention
//!   (`Indexer`) and publish their top-k; the layers after reuse it. Index
//!   keys come from the kv-source layers' latents;
//! - layer 20 also picks candidate *blocks* first (two-level top-k), and the
//!   later index sources only score inside those blocks.
//!
//! Queries are 64 heads x 512 against one shared 512-d KV; the last 64 dims
//! carry RoPE (YaRN + `compress_rope_theta` on compressed layers, plain base
//! theta on window-only layers). The output gets the query rotation
//! removed, then a block-diagonal low-rank projection over 8 groups (`wo_a`)
//! and `wo_b`.
//!
//! Prefill runs at `start_pos == 0` over the whole prompt; decode is one
//! token at a time — the same contract as the reference. A chunk of several
//! tokens can also continue a sequence (a conversation's next turn, read in one
//! pass rather than a token at a time): each of its tokens attends to exactly
//! what a decode step at its position would, in the same order, so the result
//! is the same as decoding it token by token.

use std::sync::{Arc, Mutex};

use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use crate::config::Config;
use crate::formats::{fake_quant_fp4_inplace, fake_quant_fp8_inplace, to_bf16, Fp4Scale};
use crate::linear::{load_vec, Out, Weight};
use crate::ops::{rmsnorm, Rope};
use crate::safetensors::StIndex;

/// Per-layer caches.
#[derive(Clone)]
pub struct AttnState {
    /// Sliding-window ring, `[window][head_dim]`.
    window: Vec<f32>,
    /// Compressed KV (kv-source layers), `[max_seq / ratio][head_dim]`.
    compress_kv: Vec<f32>,
    /// Partial group carried across decode steps (ratio > 1 sources).
    kv_state: Vec<f32>,
    score_state: Vec<f32>,
    /// Index keys (kv-source layers that index), `[max_seq / ratio][index_head_dim]`.
    index_k: Vec<f32>,
}

/// What attention layers hand down the stack (reference `SharedAttentionRuntime`).
#[derive(Default)]
pub struct Shared {
    compress_src: Option<usize>,
    index_k_src: Option<usize>,
    /// Per query: compressed positions to attend (offset applied, -1 = none).
    topk: Vec<Vec<i32>>,
    /// Per query: candidate mask over compressed positions (layer 20 onwards).
    candidates: Vec<Vec<bool>>,
}

struct Compressor {
    ratio: usize,
    wkv: Weight,
    wgate: Option<Weight>,
    norm: Vec<f32>,
}

struct Indexer {
    owns_k: bool,
    wq_b: Weight,
    weights_proj: Weight,
    wk: Option<Weight>,
    k_norm: Option<Vec<f32>>,
    is_candidate_source: bool,
    uses_candidates: bool,
}

pub struct Attention {
    layer: usize,
    ratio: usize,
    wq_a: Weight,
    q_norm: Vec<f32>,
    wq_b: Weight,
    wkv: Weight,
    kv_norm: Vec<f32>,
    wo_a: Weight,
    wo_b: Weight,
    attn_sink: Vec<f32>,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
    rope: Arc<Rope>,
}

impl Attention {
    /// Its dense weights by name, for [`crate::model::Model::offload`].
    pub(crate) fn weights_mut(&mut self) -> Vec<(&'static str, &mut Weight)> {
        let mut out: Vec<(&'static str, &mut Weight)> =
            vec![("wq_a", &mut self.wq_a), ("wq_b", &mut self.wq_b), ("wkv", &mut self.wkv), ("wo_a", &mut self.wo_a), ("wo_b", &mut self.wo_b)];
        if let Some(c) = &mut self.compressor {
            out.push(("compressor.wkv", &mut c.wkv));
            if let Some(g) = &mut c.wgate {
                out.push(("compressor.wgate", g));
            }
        }
        if let Some(i) = &mut self.indexer {
            out.push(("indexer.wq_b", &mut i.wq_b));
            out.push(("indexer.weights_proj", &mut i.weights_proj));
            if let Some(k) = &mut i.wk {
                out.push(("indexer.wk", k));
            }
        }
        out
    }

    pub fn load(idx: &StIndex, cfg: &Config, layer: usize, rope: Arc<Rope>) -> Result<Attention> {
        let p = format!("layers.{layer}.attn");
        let ratio = cfg.ratio(layer);
        let is_kv_source = cfg.kv_source_layers.contains(&layer);
        let is_index_source = cfg.index_source_layers.contains(&layer);
        let compressor = if is_kv_source {
            Some(Compressor {
                ratio,
                wkv: Weight::load(idx, &format!("{p}.compressor.wkv"))?,
                wgate: if ratio > 1 { Some(Weight::load(idx, &format!("{p}.compressor.wgate"))?) } else { None },
                norm: load_vec(idx, &format!("{p}.compressor.norm.weight"))?,
            })
        } else {
            None
        };
        let indexer = if is_index_source {
            let owns_k = is_kv_source;
            Some(Indexer {
                owns_k,
                wq_b: Weight::load(idx, &format!("{p}.indexer.wq_b"))?,
                weights_proj: Weight::load(idx, &format!("{p}.indexer.weights_proj"))?,
                wk: if owns_k { Some(Weight::load(idx, &format!("{p}.indexer.wk"))?) } else { None },
                k_norm: if owns_k { Some(load_vec(idx, &format!("{p}.indexer.k_norm.weight"))?) } else { None },
                is_candidate_source: cfg.candidate_source_layer == Some(layer),
                uses_candidates: cfg.candidate_source_layer.is_some_and(|c| c < layer),
            })
        } else {
            None
        };
        if ratio == 0 && (compressor.is_some() || indexer.is_some()) {
            return Err(Error::Format(format!("layer {layer}: a source layer must compress")));
        }
        Ok(Attention {
            layer,
            ratio,
            wq_a: Weight::load(idx, &format!("{p}.wq_a"))?,
            q_norm: load_vec(idx, &format!("{p}.q_norm.weight"))?,
            wq_b: Weight::load(idx, &format!("{p}.wq_b"))?,
            wkv: Weight::load(idx, &format!("{p}.wkv"))?,
            kv_norm: load_vec(idx, &format!("{p}.kv_norm.weight"))?,
            wo_a: Weight::load_fp8_as_bf16(idx, &format!("{p}.wo_a"))?,
            wo_b: Weight::load(idx, &format!("{p}.wo_b"))?,
            attn_sink: load_vec(idx, &format!("{p}.attn_sink"))?,
            compressor,
            indexer,
            rope,
        })
    }

    /// Fresh caches for this layer, sized for `max_seq` positions.
    pub fn new_state(&self, cfg: &Config, max_seq: usize) -> AttnState {
        let hd = cfg.head_dim;
        let (mut compress_kv, mut kv_state, mut score_state, mut index_k) = (vec![], vec![], vec![], vec![]);
        if let Some(c) = &self.compressor {
            compress_kv = vec![0.0; max_seq / c.ratio * hd];
            if c.ratio > 1 {
                kv_state = vec![0.0; c.ratio * hd];
                score_state = vec![f32::NEG_INFINITY; c.ratio * hd];
            }
            if self.indexer.as_ref().is_some_and(|i| i.owns_k) {
                index_k = vec![0.0; max_seq / c.ratio * cfg.index_head_dim];
            }
        }
        AttnState { window: vec![0.0; cfg.window_size * hd], compress_kv, kv_state, score_state, index_k }
    }

    /// `x`: `[t, dim]` (bf16 values) at positions `start_pos..start_pos + t`;
    /// returns `[t, dim]`. `states` holds every layer's caches (sources are
    /// read by later layers); `shared` carries the cross-layer hand-offs.
    pub fn forward(
        &self,
        cfg: &Config,
        x: &[f32],
        t: usize,
        start_pos: usize,
        states: &mut [AttnState],
        shared: &mut Shared,
    ) -> Result<Vec<f32>> {
        let (hd, nh, rd, win) = (cfg.head_dim, cfg.n_heads, cfg.rope_head_dim, cfg.window_size);
        if t == 0 {
            return Err(Error::Arg("no tokens".into()));
        }
        if start_pos + t > self.rope.max_pos() {
            return Err(Error::Arg(format!("position {} past max_seq {}", start_pos + t, self.rope.max_pos())));
        }

        // the queries' and the window's projections of x, together
        let [qa, kva]: [Vec<f32>; 2] = crate::linear::forward_together(&[
            (&self.wq_a, x, t, 0..self.wq_a.n(), Out::Bf16),
            (&self.wkv, x, t, 0..self.wkv.n(), Out::Bf16),
        ])
        .try_into()
        .expect("two projections");

        // queries
        let qr = rmsnorm(&qa, &self.q_norm, cfg.norm_eps);
        let mut q = self.wq_b.forward(&qr, t, Out::Bf16);
        for i in 0..t {
            for h in 0..nh {
                let base = (i * nh + h) * hd;
                self.rope.apply(&mut q[base + hd - rd..base + hd], start_pos + i, false);
            }
        }

        // sliding-window KV: normalize, rotate, fp8 round trip
        let mut kv = rmsnorm(&kva, &self.kv_norm, cfg.norm_eps);
        for i in 0..t {
            let row = &mut kv[i * hd..(i + 1) * hd];
            self.rope.apply(&mut row[hd - rd..], start_pos + i, false);
            fake_quant_fp8_inplace(row);
        }
        let state = &mut states[self.layer];
        let (window_kv, mut idxs): (Vec<f32>, Vec<Vec<i32>>) = if start_pos == 0 {
            if t <= win {
                state.window[..t * hd].copy_from_slice(&kv);
            } else {
                let cutoff = t % win;
                let tail = &kv[(t - win) * hd..];
                state.window[cutoff * hd..].copy_from_slice(&tail[..(win - cutoff) * hd]);
                state.window[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
            }
            let idxs = (0..t)
                .map(|i| {
                    let lo = (i + 1).saturating_sub(win);
                    (0..t.min(win)).map(|j| if lo + j > i { -1 } else { (lo + j) as i32 }).collect()
                })
                .collect();
            (kv, idxs)
        } else if t == 1 {
            state.window[(start_pos % win) * hd..(start_pos % win + 1) * hd].copy_from_slice(&kv);
            let oldest = start_pos % win + 1;
            let ring: Vec<i32> = (oldest..win)
                .chain(0..oldest)
                .map(|s| if s > start_pos { -1 } else { s as i32 })
                .collect();
            (state.window.clone(), vec![ring])
        } else {
            // A chunk continuing the sequence: the ring as the chunk found it (the `win` positions before it), then the
            // chunk's own rows. Token i (position p) lists the ring's slots in a decode step's order, oldest first, each
            // slot standing for the latest position <= p it would hold then: a position before the chunk is read from
            // the ring, one in it from the chunk's rows, one before the sequence's start is -1.
            let mut window_kv = state.window.clone();
            window_kv.extend_from_slice(&kv);
            let idxs = (0..t)
                .map(|i| {
                    let p = start_pos + i;
                    let oldest = p % win + 1;
                    (oldest..win)
                        .chain(0..oldest)
                        .map(|s| {
                            let back = (p + win - s) % win;
                            if back > p {
                                -1
                            } else if p - back >= start_pos {
                                (win + p - back - start_pos) as i32
                            } else {
                                s as i32
                            }
                        })
                        .collect()
                })
                .collect();
            // Then the ring takes the chunk's last `win` rows, each at its position's slot.
            for i in t.saturating_sub(win)..t {
                let s = (start_pos + i) % win;
                state.window[s * hd..(s + 1) * hd].copy_from_slice(&kv[i * hd..(i + 1) * hd]);
            }
            (window_kv, idxs)
        };
        let offset = window_kv.len() / hd;

        // compressed positions
        let mut kv_all = window_kv;
        if self.ratio > 0 {
            let ratio = self.ratio;
            let compress_len = (start_pos + t) / ratio;
            let latent = match &self.compressor {
                Some(c) => {
                    shared.compress_src = Some(self.layer);
                    let compressing = std::time::Instant::now();
                    let latent = c.forward(cfg, x, t, start_pos, &mut states[self.layer]);
                    crate::profile::add(crate::profile::Part::Compressor, compressing);
                    latent
                }
                None => None,
            };
            let comp_idxs = match &self.indexer {
                Some(ix) => {
                    let got = if compress_len == 0 {
                        vec![Vec::new(); t]
                    } else {
                        let indexing = std::time::Instant::now();
                        let got = self.indexer_forward(ix, cfg, x, &qr, latent.as_deref(), t, start_pos, offset, states, shared)?;
                        crate::profile::add(crate::profile::Part::Indexer, indexing);
                        got
                    };
                    shared.topk = got.clone();
                    got
                }
                None => shared.topk.clone(),
            };
            if let Some(mut lat) = latent {
                let n_c = lat.len() / hd;
                for j in 0..n_c {
                    let row = &mut lat[j * hd..(j + 1) * hd];
                    self.rope.apply(&mut row[hd - rd..], compressed_pos(start_pos, j, ratio), false);
                    fake_quant_fp4_inplace(row, 16, Fp4Scale::E4M3);
                }
                let at = start_pos / ratio * hd;
                states[self.layer].compress_kv[at..at + lat.len()].copy_from_slice(&lat);
            }
            let src = shared.compress_src.ok_or_else(|| Error::Format(format!("layer {}: no compressed-KV source", self.layer)))?;
            kv_all.extend_from_slice(&states[src].compress_kv[..compress_len * hd]);
            if comp_idxs.len() != t {
                return Err(Error::Format(format!("layer {}: index hand-off has {} queries, need {t}", self.layer, comp_idxs.len())));
            }
            for (w, c) in idxs.iter_mut().zip(comp_idxs) {
                w.extend(c);
            }
        }

        // sparse attention with sink, then undo the query rotation: every (token, head) on its own, so spread over the
        // threads (the same sums in the same order whichever thread makes them). Serial, it was most of a prompt's
        // time on the CPU and most of a decode step's attention.
        let scale = (hd as f32).powf(-0.5);
        let attending = std::time::Instant::now();
        let o = Mutex::new(vec![0.0f32; t * nh * hd]);
        // As many threads as the work pays for: `parallel_rows` starts its threads at each call (some 60 us each
        // here), and a decode step's 64 heads over a few dozen positions are a quarter of a millisecond on one. A
        // thread a pair of heads cost a step 0.076 s over its 40 layers where one thread takes 0.011 (measured, a
        // 40-token context); a prompt's work is thousands of times that and takes every thread. The time is least
        // near the square root of the work over a thread's start, so that many.
        let work: usize = idxs.iter().map(Vec::len).sum::<usize>() * nh * hd * 2;
        let threads = ((work as f64 / 6e5).sqrt() as usize).max(1);
        parallel_rows(t * nh, (t * nh).div_ceil(threads), &|b, e| {
            let mut buf = vec![0.0f32; (e - b) * hd];
            for r in b..e {
                let (i, h) = (r / nh, r % nh);
                let out = &mut buf[(r - b) * hd..(r - b + 1) * hd];
                sparse_attend(&q[r * hd..(r + 1) * hd], &kv_all, hd, &idxs[i], self.attn_sink[h], scale, out);
                self.rope.apply(&mut out[hd - rd..], start_pos + i, true);
            }
            o.lock().unwrap_or_else(|p| p.into_inner())[b * hd..e * hd].copy_from_slice(&buf);
        });
        let o = o.into_inner().unwrap_or_else(|p| p.into_inner());
        crate::profile::add(crate::profile::Part::SparseAttention, attending);

        // grouped low-rank output: wo_a is block-diagonal over o_groups
        let (g, gd, orank) = (cfg.o_groups, nh * hd / cfg.o_groups, cfg.o_lora_rank);
        let mut og = vec![0.0f32; t * g * orank];
        let xgs: Vec<Vec<f32>> = (0..g).map(|grp| (0..t).flat_map(|i| o[(i * g + grp) * gd..(i * g + grp + 1) * gd].iter().copied()).collect()).collect();
        let groups: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> =
            xgs.iter().enumerate().map(|(grp, xg)| (&self.wo_a, xg.as_slice(), t, grp * orank..(grp + 1) * orank, Out::Bf16)).collect();
        for (grp, yg) in crate::linear::forward_together(&groups).into_iter().enumerate() {
            for i in 0..t {
                og[(i * g + grp) * orank..(i * g + grp + 1) * orank].copy_from_slice(&yg[i * orank..(i + 1) * orank]);
            }
        }
        Ok(self.wo_b.forward(&og, t, Out::Bf16))
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer_forward(
        &self,
        ix: &Indexer,
        cfg: &Config,
        x: &[f32],
        qr: &[f32],
        latent: Option<&[f32]>,
        t: usize,
        start_pos: usize,
        offset: usize,
        states: &mut [AttnState],
        shared: &mut Shared,
    ) -> Result<Vec<Vec<i32>>> {
        let (ihd, inh, rd, ratio) = (cfg.index_head_dim, cfg.index_n_heads, cfg.rope_head_dim, self.ratio);
        let end_pos = start_pos + t;

        if ix.owns_k {
            if let Some(lat) = latent {
                let n_c = lat.len() / cfg.head_dim;
                let wk = ix.wk.as_ref().expect("owns_k implies wk");
                let mut k = rmsnorm(&wk.forward(lat, n_c, Out::Bf16), ix.k_norm.as_ref().expect("owns_k implies k_norm"), cfg.norm_eps);
                for j in 0..n_c {
                    let row = &mut k[j * ihd..(j + 1) * ihd];
                    self.rope.apply(&mut row[ihd - rd..], compressed_pos(start_pos, j, ratio), false);
                    fake_quant_fp4_inplace(row, 32, Fp4Scale::E8M0);
                }
                let at = start_pos / ratio * ihd;
                states[self.layer].index_k[at..at + k.len()].copy_from_slice(&k);
            }
            // DEVIATION from the reference, deliberately: `Indexer.forward` only
            // publishes `shared_attn.index_k` when a new latent arrived, so on a
            // ratio-2 decode step that completes no group, layers 2/8/14 score
            // against whatever was published last — layer 20's (ratio-1) keys
            // from the previous step, sliced to end_pos/2 rows. An owner's own
            // cache is always the right one. Unobservable while end_pos/ratio
            // <= index_topk (every position is kept), which covers the golden
            // fixtures; it matters past ~1k tokens of context.
            shared.index_k_src = Some(self.layer);
        }

        let mut q = ix.wq_b.forward(qr, t, Out::Bf16);
        for i in 0..t {
            for h in 0..inh {
                let row = &mut q[(i * inh + h) * ihd..(i * inh + h + 1) * ihd];
                self.rope.apply(&mut row[ihd - rd..], start_pos + i, false);
                fake_quant_fp4_inplace(row, 32, Fp4Scale::E8M0);
            }
        }
        let src = shared.index_k_src.ok_or_else(|| Error::Format(format!("layer {}: no index-key source", self.layer)))?;
        let n_t = end_pos / ratio;
        let keys = &states[src].index_k[..n_t * ihd];
        let wscale = (ihd as f32).powf(-0.5) * (inh as f32).powf(-0.5);
        let weights: Vec<f32> = ix.weights_proj.forward(x, t, Out::Bf16).iter().map(|w| to_bf16(w * wscale)).collect();

        let scores = index_scores(&q, keys, &weights, inh, ihd);
        let role = if ix.is_candidate_source {
            CandidateRole::Source
        } else if ix.uses_candidates {
            CandidateRole::User
        } else {
            CandidateRole::None
        };
        select_compressed(cfg, &scores, t, n_t, start_pos, ratio, offset, role, shared)
            .map_err(|e| Error::Format(format!("layer {}: {e}", self.layer)))
    }

    /// Whether this layer's attention needs its own compressed-KV / indexer
    /// work, for callers that orchestrate the pieces themselves (GPU path).
    pub fn roles(&self) -> (usize, bool, Option<CandidateRole>) {
        let role = self.indexer.as_ref().map(|ix| {
            if ix.is_candidate_source {
                CandidateRole::Source
            } else if ix.uses_candidates {
                CandidateRole::User
            } else {
                CandidateRole::None
            }
        });
        (self.ratio, self.compressor.is_some(), role)
    }
}

/// An index source's part in the two-level top-k (candidate blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandidateRole {
    None,
    /// Publishes the candidate-block mask (layer 20).
    Source,
    /// Scores only inside the published candidates (later index sources).
    User,
}

/// Indexer scores, `[t][n_t]` flattened, without the causal mask:
/// `bf16(sum_h bf16(relu(bf16(q_h . k)) * w_h))` per (query, key).
pub fn index_scores(q: &[f32], keys: &[f32], weights: &[f32], inh: usize, ihd: usize) -> Vec<f32> {
    let scoring = std::time::Instant::now();
    let (t, n_t) = (q.len() / (inh * ihd), keys.len() / ihd);
    // Every (query, key) cell on its own, spread over the threads; each summed as before.
    let out = Mutex::new(vec![0.0f32; t * n_t]);
    parallel_rows(t * n_t, 64, &|b, e| {
        let mut buf = vec![0.0f32; e - b];
        for (c, cell) in (b..e).zip(buf.iter_mut()) {
            let (i, tt) = (c / n_t, c % n_t);
            let key = &keys[tt * ihd..(tt + 1) * ihd];
            let mut acc = 0.0f32;
            for h in 0..inh {
                let qh = &q[(i * inh + h) * ihd..(i * inh + h + 1) * ihd];
                let d = to_bf16(qh.iter().zip(key).map(|(a, b)| a * b).sum::<f32>()).max(0.0);
                acc += to_bf16(d * weights[i * inh + h]);
            }
            *cell = to_bf16(acc);
        }
        out.lock().unwrap_or_else(|p| p.into_inner())[b..e].copy_from_slice(&buf);
    });
    crate::profile::add(crate::profile::Part::IndexScores, scoring);
    out.into_inner().unwrap_or_else(|p| p.into_inner())
}

/// From indexer scores to each query's compressed positions: causal mask
/// (prefill), candidate blocks (publish or apply), top-k, then position
/// order with `offset` applied and unreachable picks as -1.
#[allow(clippy::too_many_arguments)]
pub fn select_compressed(
    cfg: &Config,
    scores: &[f32],
    t: usize,
    n_t: usize,
    start_pos: usize,
    ratio: usize,
    offset: usize,
    role: CandidateRole,
    shared: &mut Shared,
) -> Result<Vec<Vec<i32>>> {
    if ratio == 0 || t.checked_mul(n_t) != Some(scores.len()) {
        return Err(Error::Format("invalid compressed score geometry".into()));
    }
    if matches!(role, CandidateRole::User)
        && (shared.candidates.len() != t || shared.candidates.iter().any(|r| r.len() != n_t))
    {
        return Err(Error::Format(
            "candidate mask missing or wrong shape".into(),
        ));
    }
    if matches!(role, CandidateRole::Source) {
        shared.candidates.clear();
        shared.candidates.reserve(t);
    }
    // A prefill score table is large. Reuse one masked row and index workspace
    // instead of cloning the complete table before selecting its sparse rows.
    // Keep masked -infinity entries in the selection: the reference tie rule
    // can select them when fewer than k valid positions exist.
    let mut row = Vec::with_capacity(n_t);
    let mut order = Vec::with_capacity(n_t);
    let mut out = Vec::with_capacity(t);
    for i in 0..t {
        let cl = (start_pos + i + 1) / ratio;
        row.clear();
        row.extend_from_slice(&scores[i * n_t..(i + 1) * n_t]);
        row.iter_mut().skip(cl).for_each(|s| *s = f32::NEG_INFINITY);
        match role {
            CandidateRole::Source => shared.candidates.push(candidate_mask(
                &row,
                cl,
                cfg.candidate_topk_blocks,
                cfg.candidate_block_size,
            )),
            CandidateRole::User => {
                for (s, &keep) in row.iter_mut().zip(&shared.candidates[i]) {
                    if !keep {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
            CandidateRole::None => {}
        }
        partition_topk(&row, cfg.index_topk, &mut order);
        // Attention consumes position order, not score order. Sort only the
        // selected k positions, avoiding both a full sort and a redundant
        // score-order sort of those k entries.
        order.sort_unstable();
        out.push(
            order
                .iter()
                .map(|&p| if p < cl { (p + offset) as i32 } else { -1 })
                .collect(),
        );
    }
    Ok(out)
}

impl Compressor {
    /// The latent(s) completed by these tokens, pre-RoPE, or `None` while a
    /// group is still filling up.
    fn forward(&self, cfg: &Config, x: &[f32], t: usize, start_pos: usize, st: &mut AttnState) -> Option<Vec<f32>> {
        let hd = cfg.head_dim;
        if self.ratio == 1 {
            return Some(rmsnorm(&self.wkv.forward(x, t, Out::Bf16), &self.norm, cfg.norm_eps));
        }
        let r = self.ratio;
        let kv = self.wkv.forward(x, t, Out::F32);
        let score = self.wgate.as_ref().expect("ratio > 1 has wgate").forward(x, t, Out::F32);
        let pooled = if start_pos == 0 {
            let rem = t % r;
            let cutoff = t - rem;
            st.kv_state[..rem * hd].copy_from_slice(&kv[cutoff * hd..]);
            st.score_state[..rem * hd].copy_from_slice(&score[cutoff * hd..]);
            if t < r {
                return None;
            }
            let mut out = Vec::with_capacity(cutoff / r * hd);
            for grp in 0..cutoff / r {
                out.extend(pool(&kv[grp * r * hd..(grp + 1) * r * hd], &score[grp * r * hd..(grp + 1) * r * hd], r, hd));
            }
            out
        } else if t == 1 {
            let slot = start_pos % r;
            st.kv_state[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
            st.score_state[slot * hd..(slot + 1) * hd].copy_from_slice(&score);
            if !(start_pos + 1).is_multiple_of(r) {
                return None;
            }
            pool(&st.kv_state, &st.score_state, r, hd)
        } else {
            // A chunk continuing the sequence (as the CUDA path does it): the partial group the state holds
            // (start_pos % r rows) first, then the chunk's rows; every complete group is pooled, the rest waits.
            let p = start_pos % r;
            let mut ckv = st.kv_state[..p * hd].to_vec();
            ckv.extend_from_slice(&kv);
            let mut csc = st.score_state[..p * hd].to_vec();
            csc.extend_from_slice(&score);
            let n = p + t;
            let rem = n % r;
            let cutoff = n - rem;
            st.kv_state[..rem * hd].copy_from_slice(&ckv[cutoff * hd..]);
            st.score_state[..rem * hd].copy_from_slice(&csc[cutoff * hd..]);
            if n < r {
                return None;
            }
            let mut out = Vec::with_capacity(cutoff / r * hd);
            for grp in 0..cutoff / r {
                out.extend(pool(&ckv[grp * r * hd..(grp + 1) * r * hd], &csc[grp * r * hd..(grp + 1) * r * hd], r, hd));
            }
            out
        };
        let bf: Vec<f32> = pooled.into_iter().map(to_bf16).collect();
        Some(rmsnorm(&bf, &self.norm, cfg.norm_eps))
    }
}

/// Softmax-gated sum over the `r` rows of one group, per feature.
pub fn pool(kv: &[f32], score: &[f32], r: usize, hd: usize) -> Vec<f32> {
    (0..hd)
        .map(|f| {
            let mx = (0..r).map(|i| score[i * hd + f]).fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = (0..r).map(|i| (score[i * hd + f] - mx).exp()).collect();
            let s: f32 = e.iter().sum();
            (0..r).map(|i| kv[i * hd + f] * (e[i] / s)).sum()
        })
        .collect()
}

/// Position a compressed latent rotates at: group j stands for its first token.
///
/// The latents a call produces are for consecutive groups, starting with
/// the group `start_pos` falls in (it completes first): `start_pos / ratio +
/// j`. That covers a prefill from 0 (group `j`), one decode token (the group
/// it completes) and a chunk continuing a sequence alike.
pub fn compressed_pos(start_pos: usize, j: usize, ratio: usize) -> usize {
    (start_pos / ratio + j) * ratio
}

/// `a . b` in sixteen running sums (one vector's lanes to the compiler), added up at the end: a head's score against
/// a position is 512 products, and summed one after another (which a compiler may not reorder) the scores were a
/// decode step's 0.08 s of sparse attention, a fifth of the step. The reference sums on a GPU in an order of its own;
/// this one is fixed, whatever the machine.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let (ca, cb) = (a.chunks_exact(LANES), b.chunks_exact(LANES));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ca.zip(cb) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            acc[i] += acc[i + width];
        }
    }
    acc[0] + tail
}

/// One query head over its gathered positions (-1 skipped), with the sink
/// logit in the denominator: `sum_j bf16(p_j) * kv_j / (sum_j p_j + e^(sink - max))`.
pub fn sparse_attend(q: &[f32], kv: &[f32], hd: usize, idxs: &[i32], sink: f32, scale: f32, out: &mut [f32]) {
    let valid: Vec<usize> = idxs.iter().filter(|&&j| j >= 0).map(|&j| j as usize).collect();
    let scores: Vec<f32> = valid.iter().map(|&j| dot(q, &kv[j * hd..(j + 1) * hd]) * scale).collect();
    let mx = scores.iter().fold(-1e30f32, |a, &b| a.max(b));
    let p: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
    let denom = p.iter().sum::<f32>() + (sink - mx).exp();
    out.fill(0.0);
    for (&j, &pj) in valid.iter().zip(&p) {
        let pb = to_bf16(pj);
        for (o, v) in out.iter_mut().zip(&kv[j * hd..(j + 1) * hd]) {
            *o += pb * v;
        }
    }
    for o in out.iter_mut() {
        *o = to_bf16(*o / denom);
    }
}

/// Indices of the `k` largest scores (ties: lower index first).
pub fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut order = Vec::new();
    partition_topk(scores, k, &mut order);
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order
}

/// Exact selection under the same total order as the reference full sort.
/// Including the index in the comparator makes ties deterministic even though
/// the partition itself is unstable. Handles NaNs and signed zero identically.
fn partition_topk(scores: &[f32], k: usize, order: &mut Vec<usize>) {
    order.clear();
    let k = k.min(scores.len());
    if k == 0 {
        return;
    }
    order.extend(0..scores.len());
    if k < order.len() {
        order.select_nth_unstable_by(k, |&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        order.truncate(k);
    }
}

/// Reference `select_candidate_blocks` for one query: keep the `topk_blocks`
/// best blocks (by their best position), always keeping the block that holds
/// the query's newest reachable position.
pub fn candidate_mask(
    scores: &[f32],
    compress_len: usize,
    topk_blocks: usize,
    bs: usize,
) -> Vec<bool> {
    let nb = scores.len().div_ceil(bs);
    let mut block: Vec<f32> = (0..nb)
        .map(|b| {
            scores[b * bs..((b + 1) * bs).min(scores.len())]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    if compress_len > 0 {
        block[(compress_len - 1) / bs] = f32::INFINITY;
    }
    let mut keep = vec![false; nb];
    let mut selected = Vec::new();
    partition_topk(&block, topk_blocks, &mut selected);
    for b in selected {
        keep[b] = block[b] > f32::NEG_INFINITY;
    }
    (0..scores.len()).map(|p| keep[p / bs]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_reference(scores: &[f32], k: usize) -> Vec<usize> {
        let mut order: Vec<_> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        order.truncate(k);
        order
    }

    #[test]
    fn partial_topk_matches_full_sort_including_float_edge_cases() {
        let mut seed = 7u64;
        for n in [0, 1, 2, 7, 511, 512, 513, 4096, 131072] {
            let scores: Vec<_> = (0..n)
                .map(|i| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    match i % 19 {
                        0 => f32::NAN,
                        1 => -f32::NAN,
                        2 => f32::INFINITY,
                        3 => f32::NEG_INFINITY,
                        4 => -0.0,
                        5 => 0.0,
                        _ => ((seed >> 32) % 257) as f32 - 128.0,
                    }
                })
                .collect();
            for k in [0, 1, 7, 512, n, n + 1] {
                assert_eq!(
                    topk_indices(&scores, k),
                    sorted_reference(&scores, k),
                    "n={n}, k={k}"
                );
            }
        }
    }

    fn selection_cfg() -> Config {
        Config {
            vocab_size: 8,
            dim: 8,
            moe_inter_dim: 8,
            n_layers: 1,
            n_heads: 1,
            head_dim: 8,
            rope_head_dim: 4,
            q_lora_rank: 4,
            o_lora_rank: 4,
            o_groups: 1,
            norm_eps: 1e-6,
            swiglu_limit: 10.0,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_seq_len: 65536,
            n_routed_experts: 2,
            n_activated_experts: 1,
            route_scale: 1.0,
            window_size: 8,
            compress_ratios: vec![2],
            kv_source_layers: vec![0],
            index_source_layers: vec![0],
            index_n_heads: 1,
            index_head_dim: 4,
            index_topk: 5,
            candidate_source_layer: Some(0),
            candidate_topk_blocks: 2,
            candidate_block_size: 4,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            engram_layer_ids: vec![],
            engram_num_embeddings: vec![],
            engram_max_ngram_size: 4,
            engram_n_heads: 1,
            engram_head_dim: 8,
            bos_token_id: 0,
            eos_token_id: 1,
            image_token_id: 2,
            vision: None,
        }
    }

    #[test]
    fn streamed_selection_matches_full_table_reference_and_masks() {
        let mut cfg = selection_cfg();
        for (t, start, ratio) in [(1, 0, 2), (7, 0, 2), (7, 50, 2), (8, 63, 1)] {
            let n = (start + t) / ratio;
            let scores: Vec<_> = (0..t * n)
                .map(|i| {
                    if i % 13 == 0 {
                        f32::NEG_INFINITY
                    } else {
                        (i % 17) as f32
                    }
                })
                .collect();
            for limit in [0, 1, 5, n, n + 1] {
                cfg.index_topk = limit;
                let mut shared = Shared::default();
                let mut masks = Vec::new();
                for role in [
                    CandidateRole::None,
                    CandidateRole::Source,
                    CandidateRole::User,
                ] {
                    let mut want = Vec::new();
                    for i in 0..t {
                        let cl = (start + i + 1) / ratio;
                        let mut row = scores[i * n..(i + 1) * n].to_vec();
                        row.iter_mut().skip(cl).for_each(|s| *s = f32::NEG_INFINITY);
                        if matches!(role, CandidateRole::Source) {
                            // Independent full-sort reference for candidate block choice.
                            let mut blocks: Vec<f32> = row
                                .chunks(4)
                                .map(|c| c.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                                .collect();
                            if cl > 0 {
                                blocks[(cl - 1) / 4] = f32::INFINITY;
                            }
                            let mut keep = vec![false; blocks.len()];
                            for b in sorted_reference(&blocks, 2) {
                                keep[b] = blocks[b] > f32::NEG_INFINITY;
                            }
                            masks.push((0..n).map(|p| keep[p / 4]).collect::<Vec<_>>());
                        } else if matches!(role, CandidateRole::User) {
                            for (s, &keep) in row.iter_mut().zip(&masks[i]) {
                                if !keep {
                                    *s = f32::NEG_INFINITY;
                                }
                            }
                        }
                        let mut pick = sorted_reference(&row, limit);
                        pick.sort_unstable();
                        want.push(
                            pick.into_iter()
                                .map(|p| if p < cl { (p + 11) as i32 } else { -1 })
                                .collect::<Vec<_>>(),
                        );
                    }
                    assert_eq!(
                        select_compressed(&cfg, &scores, t, n, start, ratio, 11, role, &mut shared)
                            .unwrap(),
                        want
                    );
                    if matches!(role, CandidateRole::Source) {
                        assert_eq!(shared.candidates, masks);
                    }
                }
            }
        }
    }

    #[test]
    fn compressed_selection_rejects_malformed_shapes() {
        let cfg = selection_cfg();
        let mut shared = Shared::default();
        assert!(select_compressed(
            &cfg,
            &[1.0],
            1,
            1,
            0,
            0,
            0,
            CandidateRole::None,
            &mut shared
        )
        .is_err());
        assert!(select_compressed(
            &cfg,
            &[1.0],
            2,
            1,
            0,
            1,
            0,
            CandidateRole::None,
            &mut shared
        )
        .is_err());
        shared.candidates = vec![vec![]];
        assert!(select_compressed(
            &cfg,
            &[1.0],
            1,
            1,
            0,
            1,
            0,
            CandidateRole::User,
            &mut shared
        )
        .is_err());
    }


    #[test]
    fn all_invalid_query_attends_to_nothing() {
        let mut out = vec![1.0f32; 4];
        sparse_attend(&[1.0; 4], &[1.0; 8], 4, &[-1, -1], 0.0, 0.5, &mut out);
        assert_eq!(out, [0.0; 4]);
    }

    #[test]
    fn candidate_mask_pins_the_newest_block() {
        let mut s = vec![0.0f32; 32];
        s[0] = 5.0; // block 0 is the best
        let m = candidate_mask(&s, 30, 1, 8); // one block allowed, newest (block 3) pinned
        assert!(m[24..].iter().all(|&k| k) && m[..24].iter().all(|&k| !k));
    }
}
