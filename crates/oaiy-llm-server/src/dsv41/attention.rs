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
//! token at a time â€” the same contract as the reference. A chunk of several
//! tokens can also continue a sequence (a conversation's next turn, read in one
//! pass rather than a token at a time): each of its tokens attends to exactly
//! what a decode step at its position would, in the same order, so the result
//! is the same as decoding it token by token.

use std::sync::{Arc, Mutex};

use oaiy_engine::backend::parallel_rows;
use oaiy_engine::{Error, Result};

use crate::dsv41::config::Config;
use crate::dsv41::formats::{fake_quant_fp4_inplace, fake_quant_fp8_inplace, to_bf16, Fp4Scale};
use crate::dsv41::linear::{load_vec, Out, Weight};
use crate::dsv41::ops::{rmsnorm, Rope};
use crate::dsv41::safetensors::StIndex;

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
    /// The compressor's last [`RECENT`] rows (ratio > 1 sources), each at its position's slot: a partial group made
    /// again from them where a draft's check kept fewer rows than it read ([`AttnState::rewind`]).
    recent_kv: Vec<f32>,
    recent_score: Vec<f32>,
}

/// Rows of a compressing source's projections [`AttnState`] keeps (more than a draft's check reads at once).
const RECENT: usize = 16;

/// Each group of the output's heads on its own row, a part of `wo_a` each (`(part, row)`): a step's jobs of the stack's
/// group ([`Attention::output_steps`]).
const DIAGONAL: [(u32, u32); 16] = {
    let mut d = [(0u32, 0u32); 16];
    let mut i = 0;
    while i < 16 {
        d[i] = (i as u32, i as u32);
        i += 1;
    }
    d
};

/// Rows of a pass whose projections go two stages a call where a device takes both (its queries with its projections of
/// x, `linear::normed_together`; its output's two, `linear::chained_together`): a step's, a draft's, a draft's check's.
const CHAINED_ROWS: usize = 8;

/// A prompt's chunk's tokens a program of its attention on the card takes ([`Attention::attend_on_card`]): its vectors
/// some 300 MB of the card's memory a program, let go of after it.
const CARD_ROWS: usize = 512;

/// Whether a prompt's chunks attend on the card where the decode steps' caches are there (OAIY_DSV41_PROMPT_ATTENTION=0:
/// on the host, as before).
fn prompt_on_card() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OAIY_DSV41_PROMPT_ATTENTION").as_deref() != Ok("0"))
}

/// What a pass of rows from `start` writes over in a layer's caches, as it was ([`AttnState::save`]).
#[derive(Clone)]
pub struct Saved {
    start: usize,
    /// The window slots of the pass's positions (each then holding a position a window before).
    window: Vec<f32>,
    kv_state: Vec<f32>,
    score_state: Vec<f32>,
}

impl AttnState {
    /// Whether `other`'s caches are these to the bit (a check of the device's copy, [`DeviceCaches`]).
    pub(crate) fn same_as(&self, other: &AttnState) -> bool {
        let bits = |a: &[f32], b: &[f32]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        bits(&self.window, &other.window) && bits(&self.compress_kv, &other.compress_kv) && bits(&self.kv_state, &other.kv_state) && bits(&self.score_state, &other.score_state) && bits(&self.index_k, &other.index_k)
    }

    /// What a pass of `t` rows from `start` will write over (`hd` the head's width): its positions' window slots and a
    /// compressing source's partial group. The compressed caches need nothing: an entry is written at its group's
    /// place and read only once its group is whole, so one made of rows not kept is made again over it.
    pub fn save(&self, hd: usize, start: usize, t: usize) -> Saved {
        let win = self.window.len() / hd;
        let window = (start..start + t.min(win)).flat_map(|p| self.window[p % win * hd..(p % win + 1) * hd].iter().copied()).collect();
        Saved { start, window, kv_state: self.kv_state.clone(), score_state: self.score_state.clone() }
    }

    /// Back to the caches as they would be had the saved pass read only its first `keep` rows: the window slots of the
    /// others as they were, and a compressing source's partial group (of `ratio` rows) made again of the rows it then
    /// holds, those before the pass from `saved`, those of it from the rows kept since.
    pub fn rewind(&mut self, hd: usize, saved: &Saved, keep: usize, ratio: usize) {
        let win = self.window.len() / hd;
        let saved_rows = saved.window.len() / hd;
        for i in keep..saved_rows {
            let s = (saved.start + i) % win;
            self.window[s * hd..(s + 1) * hd].copy_from_slice(&saved.window[i * hd..(i + 1) * hd]);
        }
        if ratio > 1 && !self.kv_state.is_empty() {
            self.kv_state.copy_from_slice(&saved.kv_state);
            self.score_state.copy_from_slice(&saved.score_state);
            let next = saved.start + keep;
            for p in (next - next % ratio).max(saved.start)..next {
                let (s, r) = (p % ratio, p % RECENT);
                self.kv_state[s * hd..(s + 1) * hd].copy_from_slice(&self.recent_kv[r * hd..(r + 1) * hd]);
                self.score_state[s * hd..(s + 1) * hd].copy_from_slice(&self.recent_score[r * hd..(r + 1) * hd]);
            }
        }
    }
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

impl Shared {
    /// The last index source's picks for the first row (positions plus the window's offset, -1 none), as a check reads
    /// them.
    pub(crate) fn first_picks(&self) -> Option<&[i32]> {
        self.topk.first().map(Vec::as_slice)
    }
}

/// Which side's attention caches hold steps the other's do not ([`DeviceCaches`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Newer {
    /// Both alike.
    Neither,
    /// The host's: a prompt's chunks were read here, or the caches were restored.
    Host,
    /// The device's: decode steps were made there.
    Device,
}

/// A step's numbers in the words of a layer's device program ([`DeviceCaches::words`]): its position, its window slot,
/// the compressed positions there are, the compressed row it completes, that row's position, its row in a partial
/// group.
const W_POS: u32 = 0;
const W_SLOT: u32 = 1;
const W_COUNT: u32 = 2;
const W_ROW: u32 = 3;
const W_CPOS: u32 = 4;
const W_GROUP: u32 = 5;
/// The word of the stream's sum of squares, its mixing's ([`crate::dsv41::hc::sumsq`]).
pub(crate) const W_SUMSQ: u32 = 6;

/// A decode step's attention caches on a device (OAIY_DSV41_GPU_ATTENTION): every layer's window, each compressing
/// source's compressed rows, partial group and index keys, the index sources' picks and the rotary tables, so the step's
/// attention is made there, in each layer's one call ([`Attention::device_steps`]). The host's [`AttnState`]s stay a
/// prompt's: its chunks are attended here, and the caches go over to whichever side is behind
/// ([`DeviceCaches::upload`], [`DeviceCaches::download`]).
pub struct DeviceCaches {
    dev: Arc<dyn ggml_rs::chain::DeviceChain>,
    layers: Vec<LayerCaches>,
    /// The latest index source's picks of compressed positions (u32 bits, `[index_topk]`), which the layers after it read.
    picks: ggml_rs::chain::DeviceVec,
    /// The candidate source's blocks of a step (u32 bits: their count, then a flag a block, 1 kept), which the index
    /// sources after it mask their scores by.
    blocks: ggml_rs::chain::DeviceVec,
    /// Each rotary flavour's tables, `(cos, sin)`.
    rope: Vec<(ggml_rs::chain::DeviceVec, ggml_rs::chain::DeviceVec)>,
    /// The most positions a step's attention reaches there (the context; with `OAIY_DSV41_FAR_ON_CARD=0` those up to
    /// which the candidate blocks keep every position, 16,384): a step past it attends on the host.
    pub reach: usize,
    /// Which side is ahead, and the positions the caches hold.
    pub newer: Newer,
    pub upto: usize,
}

/// A layer's caches on the device ([`DeviceCaches`]).
struct LayerCaches {
    ratio: usize,
    window: ggml_rs::chain::DeviceVec,
    /// A compressing source's: its compressed rows (`reach / ratio`), its partial group (past a ratio of 1) and its
    /// index keys (an indexer's that owns them).
    compress_kv: Option<ggml_rs::chain::DeviceVec>,
    kv_state: Option<(ggml_rs::chain::DeviceVec, ggml_rs::chain::DeviceVec)>,
    index_k: Option<ggml_rs::chain::DeviceVec>,
    /// Its rotary flavour (in [`DeviceCaches::rope`]), and the layers whose compressed rows and index keys it reads.
    rope: usize,
    compress_src: Option<usize>,
    index_src: Option<usize>,
}

impl DeviceCaches {
    /// Caches on `dev` for `layers` (the model's attention, in order), a context of `max_seq`.
    pub fn new(dev: Arc<dyn ggml_rs::chain::DeviceChain>, cfg: &Config, layers: &[&Attention], max_seq: usize) -> DeviceCaches {
        let (hd, ihd) = (cfg.head_dim, cfg.index_head_dim);
        // (the candidate blocks picked there too, where they no longer keep every position: OAIY_DSV41_FAR_ON_CARD=0, a
        // step past them on the host as before)
        let far = std::env::var("OAIY_DSV41_FAR_ON_CARD").as_deref() != Ok("0");
        let reach = match cfg.candidate_source_layer {
            Some(_) if !far => max_seq.min(cfg.candidate_topk_blocks * cfg.candidate_block_size),
            _ => max_seq,
        };
        let mut rope: Vec<(ggml_rs::chain::DeviceVec, ggml_rs::chain::DeviceVec)> = Vec::new();
        let mut tables: Vec<*const Rope> = Vec::new();
        let up = |v: &[f32]| {
            let d = dev.vec(v.len());
            dev.upload(&d, v);
            d
        };
        let (mut compress_src, mut index_src) = (None, None);
        let layers = layers
            .iter()
            .map(|a| {
                let flavour = match tables.iter().position(|&r| r == Arc::as_ptr(&a.rope)) {
                    Some(i) => i,
                    None => {
                        let (cos, sin) = a.rope.tables();
                        rope.push((up(cos), up(sin)));
                        tables.push(Arc::as_ptr(&a.rope));
                        tables.len() - 1
                    }
                };
                let source = a.compressor.as_ref().filter(|_| a.ratio > 0);
                if source.is_some() {
                    compress_src = Some(a.layer);
                }
                if a.indexer.as_ref().is_some_and(|ix| ix.owns_k) && source.is_some() {
                    index_src = Some(a.layer);
                }
                let rows = if a.ratio > 0 { reach / a.ratio } else { 0 };
                LayerCaches {
                    ratio: a.ratio,
                    window: dev.vec(cfg.window_size * hd),
                    compress_kv: source.map(|_| dev.vec(rows * hd)),
                    kv_state: source.filter(|c| c.ratio > 1).map(|c| (dev.vec(c.ratio * hd), dev.vec(c.ratio * hd))),
                    index_k: a.indexer.as_ref().filter(|ix| ix.owns_k && source.is_some()).map(|_| dev.vec(rows * ihd)),
                    rope: flavour,
                    compress_src: (a.ratio > 0).then_some(compress_src).flatten(),
                    index_src: (a.ratio > 0).then_some(index_src).flatten(),
                }
            })
            .collect();
        let blocks = dev.vec(1 + reach.div_ceil(cfg.candidate_block_size.max(1)));
        DeviceCaches { picks: dev.vec(cfg.index_topk.max(1)), blocks, dev, layers, rope, reach, newer: Newer::Host, upto: 0 }
    }

    /// Its bytes on the device.
    pub fn bytes(&self) -> usize {
        let v = |d: &ggml_rs::chain::DeviceVec| d.len * 4;
        let layers: usize = self
            .layers
            .iter()
            .map(|l| v(&l.window) + l.compress_kv.as_ref().map_or(0, v) + l.kv_state.as_ref().map_or(0, |(a, b)| v(a) + v(b)) + l.index_k.as_ref().map_or(0, v))
            .sum();
        layers + v(&self.picks) + v(&self.blocks) + self.rope.iter().map(|(c, s)| v(c) + v(s)).sum::<usize>()
    }

    /// Layer `l`'s step at `pos` in its program's words (u32s as f32 bits: an input as any other), `sumsq` its stream's
    /// sum of squares.
    pub fn words(&self, cfg: &Config, l: usize, pos: usize, sumsq: f32) -> [f32; 8] {
        let r = self.layers[l].ratio.max(1);
        let mut w = [pos, pos % cfg.window_size, (pos + 1) / r, pos / r, pos / r * r, pos % r, 0, 0].map(|v| f32::from_bits(v as u32));
        w[W_SUMSQ as usize] = sumsq;
        w
    }

    /// The host's caches of `positions` positions onto the device.
    pub fn upload(&self, cfg: &Config, states: &[AttnState], positions: usize) {
        let (hd, ihd) = (cfg.head_dim, cfg.index_head_dim);
        for (lc, st) in self.layers.iter().zip(states) {
            self.dev.upload(&lc.window, &st.window);
            let rows = if lc.ratio > 0 { positions / lc.ratio } else { 0 };
            if let Some(v) = lc.compress_kv.as_ref().filter(|_| rows > 0) {
                self.dev.upload_at(v, 0, &st.compress_kv[..(rows * hd).min(v.len)]);
            }
            if let Some((kv, score)) = &lc.kv_state {
                self.dev.upload(kv, &st.kv_state);
                self.dev.upload(score, &st.score_state);
            }
            if let Some(v) = lc.index_k.as_ref().filter(|_| rows > 0) {
                self.dev.upload_at(v, 0, &st.index_k[..(rows * ihd).min(v.len)]);
            }
        }
    }

    /// The device's caches of `positions` positions into the host's, in one read back.
    pub fn download(&self, cfg: &Config, states: &mut [AttnState], positions: usize) {
        let (hd, ihd) = (cfg.head_dim, cfg.index_head_dim);
        let mut rec = self.dev.begin();
        for lc in &self.layers {
            rec.read(&lc.window);
            let rows = if lc.ratio > 0 { positions / lc.ratio } else { 0 };
            if let Some(v) = lc.compress_kv.as_ref().filter(|_| rows > 0) {
                rec.read_range(v, 0, (rows * hd).min(v.len));
            }
            if let Some((kv, score)) = &lc.kv_state {
                rec.read(kv);
                rec.read(score);
            }
            if let Some(v) = lc.index_k.as_ref().filter(|_| rows > 0) {
                rec.read_range(v, 0, (rows * ihd).min(v.len));
            }
        }
        let mut got = rec.finish().into_iter();
        let mut next = || got.next().expect("a cache read back");
        for (lc, st) in self.layers.iter().zip(states.iter_mut()) {
            st.window.copy_from_slice(&next());
            let rows = if lc.ratio > 0 { positions / lc.ratio } else { 0 };
            if lc.compress_kv.is_some() && rows > 0 {
                let v = next();
                st.compress_kv[..v.len()].copy_from_slice(&v);
            }
            if lc.kv_state.is_some() {
                st.kv_state.copy_from_slice(&next());
                st.score_state.copy_from_slice(&next());
            }
            if lc.index_k.is_some() && rows > 0 {
                let v = next();
                st.index_k[..v.len()].copy_from_slice(&v);
            }
        }
    }

    /// Layer `l`'s rows a decode step at `pos` wrote on the host (its window slot, its partial group, the compressed row
    /// and index key it completed) and the picks the host made for the layers after it, `picks` (`[k]`, positions plus
    /// `offset`, -1 none), onto the device: a check's device then starts each layer's step from the host's caches.
    pub fn upload_step(&self, cfg: &Config, states: &[AttnState], l: usize, pos: usize, picks: Option<(&[i32], usize)>) {
        let (hd, ihd, win) = (cfg.head_dim, cfg.index_head_dim, cfg.window_size);
        let (lc, st) = (&self.layers[l], &states[l]);
        let slot = pos % win;
        self.dev.upload_at(&lc.window, slot * hd, &st.window[slot * hd..(slot + 1) * hd]);
        if let Some((kv, score)) = &lc.kv_state {
            self.dev.upload(kv, &st.kv_state);
            self.dev.upload(score, &st.score_state);
        }
        if lc.ratio > 0 && (pos + 1).is_multiple_of(lc.ratio) {
            let row = pos / lc.ratio;
            if let Some(v) = &lc.compress_kv {
                self.dev.upload_at(v, row * hd, &st.compress_kv[row * hd..(row + 1) * hd]);
            }
            if let Some(v) = &lc.index_k {
                self.dev.upload_at(v, row * ihd, &st.index_k[row * ihd..(row + 1) * ihd]);
            }
        }
        if let Some((picks, offset)) = picks {
            let mut words = vec![f32::from_bits(u32::MAX); self.picks.len];
            for (w, &p) in words.iter_mut().zip(picks) {
                if p >= 0 {
                    *w = f32::from_bits((p as usize - offset) as u32);
                }
            }
            self.dev.upload(&self.picks, &words);
        }
    }

    /// What a check reads back of layer `l`'s step at `pos` from the device besides its outputs: `(vector, first
    /// value, values, what)` of its window slot and the compressed row and index key it completed (where it did).
    pub(crate) fn step_reads(&self, cfg: &Config, l: usize, pos: usize) -> Vec<(&ggml_rs::chain::DeviceVec, usize, usize, &'static str)> {
        let (hd, ihd) = (cfg.head_dim, cfg.index_head_dim);
        let lc = &self.layers[l];
        let mut reads = vec![(&lc.window, pos % cfg.window_size * hd, hd, "window row")];
        if lc.ratio > 0 && (pos + 1).is_multiple_of(lc.ratio) {
            let row = pos / lc.ratio;
            let one = lc.ratio == 1;
            if let Some(v) = &lc.compress_kv {
                reads.push((v, row * hd, hd, if one { "compressed row (ratio 1)" } else { "compressed row (pooled)" }));
            }
            if let Some(v) = &lc.index_k {
                reads.push((v, row * ihd, ihd, if one { "index key (ratio 1)" } else { "index key (pooled)" }));
            }
        }
        reads
    }

    /// The host's rows of layer `l`'s step at `pos`, in [`Self::step_reads`]' order.
    pub(crate) fn host_rows<'s>(&self, cfg: &Config, states: &'s [AttnState], l: usize, pos: usize) -> Vec<&'s [f32]> {
        let (hd, ihd) = (cfg.head_dim, cfg.index_head_dim);
        let (lc, st) = (&self.layers[l], &states[l]);
        let slot = pos % cfg.window_size;
        let mut rows = vec![&st.window[slot * hd..(slot + 1) * hd]];
        if lc.ratio > 0 && (pos + 1).is_multiple_of(lc.ratio) {
            let row = pos / lc.ratio;
            if lc.compress_kv.is_some() {
                rows.push(&st.compress_kv[row * hd..(row + 1) * hd]);
            }
            if lc.index_k.is_some() {
                rows.push(&st.index_k[row * ihd..(row + 1) * ihd]);
            }
        }
        rows
    }

    /// The index sources' picks, as a check reads them.
    pub(crate) fn picks(&self) -> &ggml_rs::chain::DeviceVec {
        &self.picks
    }
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
    /// Its dense weights by name, for [`crate::dsv41::model::Model::offload`].
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
        Self::load_named(idx, cfg, layer, &format!("layers.{layer}.attn"), rope)
    }

    /// [`Attention::load`] of the weights named `prefix.*` (DSpark's drafter's `mtp.<i>.attn`), `layer` its place in
    /// `cfg`'s layers.
    pub fn load_named(idx: &StIndex, cfg: &Config, layer: usize, p: &str, rope: Arc<Rope>) -> Result<Attention> {
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
            wo_a: Weight::load_groups(idx, &format!("{p}.wo_a"), cfg.o_groups)?,
            wo_b: Weight::load(idx, &format!("{p}.wo_b"))?,
            attn_sink: load_vec(idx, &format!("{p}.attn_sink"))?,
            compressor,
            indexer,
            rope,
        })
    }

    /// The window's projection, whose rows [`Attention::store_keys`] takes.
    pub fn wkv(&self) -> &Weight {
        &self.wkv
    }

    /// The window keys of rows at `start_pos..`, from the window's projection of them, `kva` (`[t, head_dim]`, bf16
    /// values), as [`Attention::forward`] makes and keeps its own: DSpark's drafter's keys of the trunk's rows, which its
    /// drafts then attend to ([`Attention::forward_draft`]).
    pub fn store_keys(&self, cfg: &Config, kva: &[f32], start_pos: usize, state: &mut AttnState) {
        let (hd, rd, win) = (cfg.head_dim, cfg.rope_head_dim, cfg.window_size);
        let kv = rmsnorm(kva, &self.kv_norm, cfg.norm_eps);
        for (i, row) in kv.chunks_exact(hd).enumerate() {
            let p = start_pos + i;
            let mut row = row.to_vec();
            self.rope.apply(&mut row[hd - rd..], p, false);
            fake_quant_fp8_inplace(&mut row);
            state.window[p % win * hd..(p % win + 1) * hd].copy_from_slice(&row);
        }
    }

    /// DSpark's drafter's block (DeepInfer's `DFlashAttention.forward_drafts`): `t` rows `x` (`[t, dim]`, bf16 values)
    /// at `start_pos..`, each attending to the window's keys of the `win` positions before `start_pos` that there are
    /// ([`Attention::store_keys`]) and to every row of the block, its own and those after it alike (a block is drafted
    /// at once). The block's keys are not kept. A window-only layer's.
    pub fn forward_draft(&self, cfg: &Config, x: &[f32], t: usize, start_pos: usize, state: &AttnState) -> Result<Vec<f32>> {
        let (hd, nh, rd, win) = (cfg.head_dim, cfg.n_heads, cfg.rope_head_dim, cfg.window_size);
        if t == 0 || self.ratio != 0 {
            return Err(Error::Arg(format!("layer {}: a draft of {t} rows on a layer of ratio {}", self.layer, self.ratio)));
        }
        let (made_x, made_q, _) = self.queries(cfg, &[(&self.wq_a, x, t, 0..self.wq_a.n(), Out::Bf16), (&self.wkv, x, t, 0..self.wkv.n(), Out::Bf16)], None, t, true);
        let kva = made_x.into_iter().nth(1).expect("the window's projection");
        let mut q = made_q.into_iter().next().expect("the queries");
        for i in 0..t {
            for h in 0..nh {
                let base = (i * nh + h) * hd;
                self.rope.apply(&mut q[base + hd - rd..base + hd], start_pos + i, false);
            }
        }
        let mut kv = rmsnorm(&kva, &self.kv_norm, cfg.norm_eps);
        for i in 0..t {
            let row = &mut kv[i * hd..(i + 1) * hd];
            self.rope.apply(&mut row[hd - rd..], start_pos + i, false);
            fake_quant_fp8_inplace(row);
        }
        // The window's positions oldest first, then the block's rows: one list every row attends over.
        let mut keys: Vec<f32> = Vec::with_capacity((win + t) * hd);
        for p in start_pos.saturating_sub(win)..start_pos {
            keys.extend_from_slice(&state.window[p % win * hd..(p % win + 1) * hd]);
        }
        keys.extend_from_slice(&kv);
        // every (row, head) on the workers that stay ([`crate::dsv41::pool::shared`]), a share each: a decode step's
        // attention's way (threads started for it cost a millisecond a layer)
        let idxs: Arc<Vec<i32>> = Arc::new((0..(keys.len() / hd) as i32).collect());
        let scale = (hd as f32).powf(-0.5);
        let rows = t * nh;
        let pool = crate::dsv41::pool::shared().lock().unwrap_or_else(|p| p.into_inner());
        let per = rows.div_ceil(pool.threads().min(rows));
        let (q, keys) = (Arc::new(q), Arc::new(keys));
        let (sink, rope) = (Arc::new(self.attn_sink.clone()), Arc::clone(&self.rope));
        let jobs: Vec<crate::dsv41::pool::Job> = (0..rows)
            .step_by(per)
            .enumerate()
            .map(|(id, b)| {
                let e = (b + per).min(rows);
                let (q, keys, idxs, sink, rope) = (Arc::clone(&q), Arc::clone(&keys), Arc::clone(&idxs), Arc::clone(&sink), Arc::clone(&rope));
                Box::new(move || {
                    let mut buf = vec![0.0f32; (e - b) * hd];
                    for r in b..e {
                        let (i, h) = (r / nh, r % nh);
                        let out = &mut buf[(r - b) * hd..(r - b + 1) * hd];
                        sparse_attend(&q[r * hd..(r + 1) * hd], &keys, hd, &idxs, sink[h], scale, out);
                        rope.apply(&mut out[hd - rd..], start_pos + i, true);
                    }
                    (id, buf)
                }) as crate::dsv41::pool::Job
            })
            .collect();
        let o = pool.run(jobs).concat();
        drop(pool);
        Ok(self.output(cfg, &o, t))
    }

    /// Fresh caches for this layer, sized for `max_seq` positions.
    pub fn new_state(&self, cfg: &Config, max_seq: usize) -> AttnState {
        let hd = cfg.head_dim;
        let (mut compress_kv, mut kv_state, mut score_state, mut index_k) = (vec![], vec![], vec![], vec![]);
        let (mut recent_kv, mut recent_score) = (vec![], vec![]);
        if let Some(c) = &self.compressor {
            compress_kv = vec![0.0; max_seq / c.ratio * hd];
            if c.ratio > 1 {
                kv_state = vec![0.0; c.ratio * hd];
                score_state = vec![f32::NEG_INFINITY; c.ratio * hd];
                recent_kv = vec![0.0; RECENT * hd];
                recent_score = vec![f32::NEG_INFINITY; RECENT * hd];
            }
            if self.indexer.as_ref().is_some_and(|i| i.owns_k) {
                index_k = vec![0.0; max_seq / c.ratio * cfg.index_head_dim];
            }
        }
        AttnState { window: vec![0.0; cfg.window_size * hd], compress_kv, kv_state, score_state, index_k, recent_kv, recent_score }
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
        Ok(self.forward_side(cfg, x, t, start_pos, states, shared, None)?.0)
    }

    /// [`Self::forward`], and `side`'s projection of its own input (`(weight, input)`: the sublayer's mixing projection
    /// of the stream, [`crate::dsv41::hc::HcParams::on_device`]) made in the same call as the layer's projections of x,
    /// `[t, side.n()]` in f32 (a round trip of the host's work fewer).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn forward_side(
        &self,
        cfg: &Config,
        x: &[f32],
        t: usize,
        start_pos: usize,
        states: &mut [AttnState],
        shared: &mut Shared,
        side: Option<(&Weight, &[f32])>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let (o, mut sides, _) = self.forward_core(cfg, x, t, start_pos, states, shared, side.as_slice(), None)?;
        Ok((self.output(cfg, &o, t), sides.pop()))
    }

    /// [`Self::forward_side`] up to the heads' outputs (`[t, heads, head_dim]`, the rows the output projection takes:
    /// [`Self::output_steps`] puts it in a device program); with `card` (a prompt's chunk where the decode steps' caches
    /// are on a device), its attention there with its output projection ([`Self::attend_on_card`]): then the output,
    /// `[t, dim]`, and true.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn forward_core(
        &self,
        cfg: &Config,
        x: &[f32],
        t: usize,
        start_pos: usize,
        states: &mut [AttnState],
        shared: &mut Shared,
        sides: &[(&Weight, &[f32])],
        card: Option<&DeviceCaches>,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, bool)> {
        let (hd, nh, rd, win) = (cfg.head_dim, cfg.n_heads, cfg.rope_head_dim, cfg.window_size);
        if t == 0 {
            return Err(Error::Arg("no tokens".into()));
        }
        if start_pos + t > self.rope.max_pos() {
            return Err(Error::Arg(format!("position {} past max_seq {}", start_pos + t, self.rope.max_pos())));
        }

        // Every projection of x the layer makes, together: the queries' and the window's, the compressor's (its keys,
        // and their gate past a ratio of 1) and the indexer's weights. A decode step's were a round trip each, four of
        // a compressing layer's that had an indexer.
        let compress_len = if self.ratio > 0 { (start_pos + t) / self.ratio } else { 0 };
        let compressor = self.compressor.as_ref().filter(|_| self.ratio > 0);
        let indexer = self.indexer.as_ref().filter(|_| self.ratio > 0 && compress_len > 0);
        let mut of_x: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = vec![(&self.wq_a, x, t, 0..self.wq_a.n(), Out::Bf16), (&self.wkv, x, t, 0..self.wkv.n(), Out::Bf16)];
        if let Some(c) = compressor {
            of_x.push((&c.wkv, x, t, 0..c.wkv.n(), if c.ratio == 1 { Out::Bf16 } else { Out::F32 }));
            if let Some(g) = c.wgate.as_ref().filter(|_| c.ratio > 1) {
                of_x.push((g, x, t, 0..g.n(), Out::F32));
            }
        }
        if let Some(ix) = indexer {
            of_x.push((&ix.weights_proj, x, t, 0..ix.weights_proj.n(), Out::Bf16));
        }
        for &(w, input) in sides {
            of_x.push((w, input, t, 0..w.n(), Out::F32));
        }
        // and the queries (the indexer's too, of the same row); a prompt's chunk attending on the card makes its own
        // queries there
        let card = card.filter(|_| t > CHAINED_ROWS && prompt_on_card());
        let (mut made_x, made_q, qr) = self.queries(cfg, &of_x, indexer.map(|ix| &ix.wq_b), t, card.is_none());
        let side_out = made_x.split_off(made_x.len() - sides.len());
        let mut made = made_x.into_iter().skip(1);
        let kva = made.next().expect("the window's projection");
        let compressed = compressor.map(|c| (made.next().expect("the compressor's keys"), (c.ratio > 1).then(|| made.next().expect("the compressor's gate"))));
        let index_weights = indexer.map(|_| made.next().expect("the indexer's weights"));
        let mut made = made_q.into_iter();
        let rotated = |mut q: Vec<f32>| {
            for i in 0..t {
                for h in 0..nh {
                    let base = (i * nh + h) * hd;
                    self.rope.apply(&mut q[base + hd - rd..base + hd], start_pos + i, false);
                }
            }
            q
        };
        let q = card.is_none().then(|| rotated(made.next().expect("the queries")));
        let index_q = indexer.map(|_| made.next().expect("the indexer's queries"));

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
            let latent = match (compressor, compressed) {
                (Some(c), Some((keys, gate))) => {
                    shared.compress_src = Some(self.layer);
                    let compressing = std::time::Instant::now();
                    let latent = c.forward(cfg, keys, gate, t, start_pos, &mut states[self.layer]);
                    crate::dsv41::profile::add(crate::dsv41::profile::Part::Compressor, compressing);
                    latent
                }
                _ => None,
            };
            let comp_idxs = match &self.indexer {
                Some(ix) => {
                    let got = match (index_q, index_weights) {
                        (Some(q), Some(weights)) => {
                            let indexing = std::time::Instant::now();
                            let got = self.indexer_forward(ix, cfg, q, &weights, latent.as_deref(), t, start_pos, offset, states, shared)?;
                            crate::dsv41::profile::add(crate::dsv41::profile::Part::Indexer, indexing);
                            got
                        }
                        // (nothing compressed yet)
                        _ => vec![Vec::new(); t],
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

        // A prompt's chunk's attention and output projection on the card; else (or where the device took no such
        // program) here, its queries made now.
        let q = match (card, &qr, q) {
            (_, _, Some(q)) => q,
            (Some(dc), Some(qr), None) => {
                if let Some(out) = self.attend_on_card(cfg, dc, qr, &kv_all, &idxs, t, start_pos) {
                    return Ok((out, side_out, true));
                }
                rotated(self.wq_b.forward(qr, t, Out::Bf16))
            }
            _ => unreachable!("a chunk attending on the card has its queries' row"),
        };

        // sparse attention with sink, then undo the query rotation: every (token, head) on its own, so spread over the
        // threads (the same sums in the same order whichever thread makes them). Serial, it was most of a prompt's
        // time on the CPU and most of a decode step's attention.
        let scale = (hd as f32).powf(-0.5);
        let attending = std::time::Instant::now();
        let work: usize = idxs.iter().map(Vec::len).sum::<usize>() * nh * hd * 2;
        let o: Vec<f32> = if work <= POOLED_WORK {
            // A decode step's (64 heads over a few hundred positions: a millisecond or two of one thread's work), on
            // workers that are already there ([`crate::dsv41::pool`]): a share of the heads each, some 150,000 products or
            // more a share. Starting threads for it at each call (some 60 us each here) left the time least at the
            // square root of the work over a thread's start, four threads and half a millisecond a layer for a
            // 300-token context, 20 to 26 ms a token; the same sums, whichever worker makes them.
            let rows = t * nh;
            let pool = crate::dsv41::pool::shared().lock().unwrap_or_else(|p| p.into_inner());
            let per = rows.div_ceil(pool.threads().min(rows).min((work / 150_000).max(1)));
            let (q, kv_all, idxs) = (Arc::new(q), Arc::new(kv_all), Arc::new(idxs));
            let (sink, rope) = (Arc::new(self.attn_sink.clone()), Arc::clone(&self.rope));
            let jobs: Vec<crate::dsv41::pool::Job> = (0..rows)
                .step_by(per)
                .enumerate()
                .map(|(id, b)| {
                    let e = (b + per).min(rows);
                    let (q, kv_all, idxs, sink, rope) = (Arc::clone(&q), Arc::clone(&kv_all), Arc::clone(&idxs), Arc::clone(&sink), Arc::clone(&rope));
                    Box::new(move || {
                        let mut buf = vec![0.0f32; (e - b) * hd];
                        for r in b..e {
                            let (i, h) = (r / nh, r % nh);
                            let out = &mut buf[(r - b) * hd..(r - b + 1) * hd];
                            sparse_attend(&q[r * hd..(r + 1) * hd], &kv_all, hd, &idxs[i], sink[h], scale, out);
                            rope.apply(&mut out[hd - rd..], start_pos + i, true);
                        }
                        (id, buf)
                    }) as crate::dsv41::pool::Job
                })
                .collect();
            pool.run(jobs).concat()
        } else {
            // A prompt's: thousands of times a step's work, on every thread (`parallel_rows` starts its own).
            let o = Mutex::new(vec![0.0f32; t * nh * hd]);
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
            o.into_inner().unwrap_or_else(|p| p.into_inner())
        };
        crate::dsv41::profile::add(crate::dsv41::profile::Part::SparseAttention, attending);

        Ok((o, side_out, false))
    }

    /// One row's output projection of the heads' outputs `o` (`[heads, head_dim]`) as steps of a device program: each
    /// group's input, `wo_a`'s part of it, the groups' rows rounded to bf16 and laid end to end, `wo_b` of them and its
    /// sums rounded to bf16 (as [`Self::output`] makes them, to the bit). The step whose vector is the output (`[dim]`),
    /// or None where `wo_a` is not a stack of a part a group (an EXL3 checkpoint's: the program takes no other).
    pub(crate) fn output_steps<'a>(&'a self, cfg: &Config, o: &'a [f32], steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> Option<usize> {
        use crate::dsv41::linear::Step;
        let Weight::Stacked { parts, rows, group, .. } = &self.wo_a else { return None };
        let (g, gd, orank) = (cfg.o_groups, cfg.n_heads * cfg.head_dim / cfg.o_groups, cfg.o_lora_rank);
        if parts.len() != g || *rows != orank || o.len() != g * gd {
            return None;
        }
        match group {
            Some(_) if g <= DIAGONAL.len() => {
                let input = steps.len();
                steps.push(Step::Input(o));
                self.output_steps_from(cfg, input, steps)
            }
            _ => {
                let first = steps.len();
                for grp in 0..g {
                    steps.push(Step::Input(&o[grp * gd..(grp + 1) * gd]));
                }
                for (grp, part) in parts.iter().enumerate() {
                    steps.push(Step::Mul { w: part, x: first + grp, rows: 1 });
                }
                let rows_in = steps.len();
                steps.push(Step::Scratch(g * orank));
                for grp in 0..g {
                    steps.push(Step::CopyColsBf16 { src: first + g + grp, src_stride: orank, src_at: 0, dst: rows_in, dst_stride: g * orank, dst_at: grp * orank, width: orank, rows: 1 });
                }
                Some(self.output_tail(cfg, rows_in, steps))
            }
        }
    }

    /// [`Self::output_steps`] of the heads' outputs a program's step `o` makes (`[heads, head_dim]`), where `wo_a` is a
    /// stack of a part a group: one the device takes in one pass (an EXL3 checkpoint's), every group's heads a row of
    /// it, each part on its own row, their rows rounded at once; else (a GGUF's) each group's heads copied out (bf16's
    /// values already, as the attention leaves them: the copy rounds nothing) and its part of them. The step whose
    /// vector is the output (`[dim]`), or None.
    pub(crate) fn output_steps_from<'a>(&'a self, cfg: &Config, o: usize, steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> Option<usize> {
        use crate::dsv41::linear::Step;
        let Weight::Stacked { parts, rows, group, .. } = &self.wo_a else { return None };
        let (g, orank) = (cfg.o_groups, cfg.o_lora_rank);
        if parts.len() != g || *rows != orank || g > DIAGONAL.len() {
            return None;
        }
        let Some(group) = group else {
            let gd = cfg.n_heads * cfg.head_dim / g;
            let mut made = Vec::with_capacity(g);
            for (grp, part) in parts.iter().enumerate() {
                let input = steps.len();
                steps.push(Step::Scratch(gd));
                steps.push(Step::CopyColsBf16 { src: o, src_stride: g * gd, src_at: grp * gd, dst: input, dst_stride: gd, dst_at: 0, width: gd, rows: 1 });
                made.push(steps.len());
                steps.push(Step::Mul { w: part, x: input, rows: 1 });
            }
            let rows_in = steps.len();
            steps.push(Step::Scratch(g * orank));
            for (grp, &m) in made.iter().enumerate() {
                steps.push(Step::CopyColsBf16 { src: m, src_stride: orank, src_at: 0, dst: rows_in, dst_stride: g * orank, dst_at: grp * orank, width: orank, rows: 1 });
            }
            return Some(self.output_tail(cfg, rows_in, steps));
        };
        steps.push(Step::MulGroup { group, x: o, jobs: &DIAGONAL[..g] });
        let rows_in = steps.len();
        steps.push(Step::Scratch(g * orank));
        steps.push(Step::CopyColsBf16 { src: rows_in - 1, src_stride: g * orank, src_at: 0, dst: rows_in, dst_stride: g * orank, dst_at: 0, width: g * orank, rows: 1 });
        Some(self.output_tail(cfg, rows_in, steps))
    }

    /// `wo_b` of the groups' rows `rows_in` and its sums rounded to bf16: the output's step.
    fn output_tail<'a>(&'a self, cfg: &Config, rows_in: usize, steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> usize {
        self.output_tail_rows(cfg, rows_in, 1, steps)
    }

    /// [`Self::output_tail`] of `rows` rows.
    fn output_tail_rows<'a>(&'a self, cfg: &Config, rows_in: usize, rows: usize, steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> usize {
        use crate::dsv41::linear::Step;
        let d = cfg.dim;
        let sums = steps.len();
        steps.push(Step::Mul { w: &self.wo_b, x: rows_in, rows });
        let out = steps.len();
        steps.push(Step::Scratch(rows * d));
        steps.push(Step::CopyColsBf16 { src: sums, src_stride: d, src_at: 0, dst: out, dst_stride: d, dst_at: 0, width: d, rows });
        out
    }

    /// The output projection of `rows` tokens' heads' outputs a program's step `o` makes (`[rows, heads, head_dim]`), as
    /// [`Self::output`] makes it: each group's heads copied out (bf16's values already) and its part of them, the groups'
    /// rows rounded to bf16 and laid end to end, then `wo_b` and its sums rounded. The step whose vector is the output
    /// (`[rows, dim]`), or None where `wo_a` is not a stack of a part a group.
    fn output_rows_steps<'a>(&'a self, cfg: &Config, o: usize, rows: usize, steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> Option<usize> {
        use crate::dsv41::linear::Step;
        let Weight::Stacked { parts, rows: orank, .. } = &self.wo_a else { return None };
        let (g, gd) = (cfg.o_groups, cfg.n_heads * cfg.head_dim / cfg.o_groups);
        if parts.len() != g || *orank != cfg.o_lora_rank {
            return None;
        }
        let orank = *orank;
        let mut made = Vec::with_capacity(g);
        for (grp, part) in parts.iter().enumerate() {
            let input = steps.len();
            steps.push(Step::Scratch(rows * gd));
            steps.push(Step::CopyColsBf16 { src: o, src_stride: g * gd, src_at: grp * gd, dst: input, dst_stride: gd, dst_at: 0, width: gd, rows });
            made.push(steps.len());
            steps.push(Step::Mul { w: part, x: input, rows });
        }
        let rows_in = steps.len();
        steps.push(Step::Scratch(rows * g * orank));
        for (grp, &m) in made.iter().enumerate() {
            steps.push(Step::CopyColsBf16 { src: m, src_stride: orank, src_at: 0, dst: rows_in, dst_stride: g * orank, dst_at: grp * orank, width: orank, rows });
        }
        Some(self.output_tail_rows(cfg, rows_in, rows, steps))
    }

    /// A prompt's chunk's attention on the card (`dc`'s rotary tables), from the queries' normed low-rank rows `qr`
    /// (`[t, q_lora_rank]`), the rows `kv` it attends over (the window's, then the compressed ones, as the host gathers
    /// them) and each token's positions in them `idxs` (the host's order, -1 none): its queries (`wq_b`, rounded to bf16
    /// and rotated), each (token, head)'s sparse attention, rotated back, and the output projection, [`CARD_ROWS`]
    /// tokens a program: `[t, dim]`. Only the output comes back, where the queries and the heads' outputs (half a GB each
    /// at 4,096 rows) went to the host and back. None where the device takes no such program.
    #[allow(clippy::too_many_arguments)]
    fn attend_on_card(&self, cfg: &Config, dc: &DeviceCaches, qr: &[f32], kv: &[f32], idxs: &[Vec<i32>], t: usize, start_pos: usize) -> Option<Vec<f32>> {
        use crate::dsv41::linear::Step;
        let (hd, nh, rd) = (cfg.head_dim, cfg.n_heads, cfg.rope_head_dim);
        let qd = qr.len() / t;
        let len = idxs.iter().map(Vec::len).max()?.max(1);
        let (cos, sin) = &dc.rope[dc.layers[self.layer].rope];
        let scale = (hd as f32).powf(-0.5);
        let mut out = Vec::with_capacity(t * cfg.dim);
        for r0 in (0..t).step_by(CARD_ROWS) {
            let n = CARD_ROWS.min(t - r0);
            let pos: Vec<f32> = (r0..r0 + n).map(|i| f32::from_bits((start_pos + i) as u32)).collect();
            let picks: Vec<f32> = idxs[r0..r0 + n].iter().flat_map(|row| (0..len).map(move |e| f32::from_bits(row.get(e).filter(|&&j| j >= 0).map_or(u32::MAX, |&j| j as u32)))).collect();
            let rotate = |x: usize, inverse: bool| Step::RopeRows { x, rows: n * nh, stride: hd, at: hd - rd, half: rd / 2, cos: 4, sin: 5, pos: 6, group: nh, inverse };
            let mut steps: Vec<Step<'_>> = vec![
                Step::Input(&qr[r0 * qd..(r0 + n) * qd]),
                Step::Mul { w: &self.wq_b, x: 0, rows: n },
                Step::Scratch(n * nh * hd),
                Step::CopyColsBf16 { src: 1, src_stride: nh * hd, src_at: 0, dst: 2, dst_stride: nh * hd, dst_at: 0, width: nh * hd, rows: n },
                Step::State(cos),
                Step::State(sin),
                Step::Input(&pos),
                rotate(2, false),
                Step::Input(kv),
                Step::Input(&picks),
                Step::AttendRows { q: 2, kv: 8, idxs: 9, tokens: n, heads: nh, dim: hd, len, sink: &self.attn_sink, scale },
                rotate(10, true),
            ];
            let a = self.output_rows_steps(cfg, 10, n, &mut steps)?;
            let got = crate::dsv41::linear::run_program_once(&steps, &[(a, 0, n * cfg.dim)])?;
            out.extend_from_slice(&got[0]);
        }
        Some(out)
    }

    /// The grouped low-rank output of `t` rows of the attention's output `o` (`wo_a` block-diagonal over the groups,
    /// then `wo_b`): a step's or a check's two projections in one call where a device holds both, the groups' rows
    /// rounded there (the second was a round trip of its own, a layer); else a call each.
    pub(crate) fn output(&self, cfg: &Config, o: &[f32], t: usize) -> Vec<f32> {
        let (g, gd, orank) = (cfg.o_groups, cfg.n_heads * cfg.head_dim / cfg.o_groups, cfg.o_lora_rank);
        let xgs: Vec<Vec<f32>> = (0..g).map(|grp| (0..t).flat_map(|i| o[(i * g + grp) * gd..(i * g + grp + 1) * gd].iter().copied()).collect()).collect();
        if t <= CHAINED_ROWS {
            let first: Vec<(&Weight, &[f32], std::ops::Range<usize>)> = xgs.iter().enumerate().map(|(grp, xg)| (&self.wo_a, xg.as_slice(), grp * orank..(grp + 1) * orank)).collect();
            if let Some(y) = crate::dsv41::linear::chained_together(&first, &self.wo_b, t, Out::Bf16) {
                return y;
            }
        }
        let groups: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> =
            xgs.iter().enumerate().map(|(grp, xg)| (&self.wo_a, xg.as_slice(), t, grp * orank..(grp + 1) * orank, Out::Bf16)).collect();
        let mut og = vec![0.0f32; t * g * orank];
        for (grp, yg) in crate::dsv41::linear::forward_together(&groups).into_iter().enumerate() {
            for i in 0..t {
                og[(i * g + grp) * orank..(i * g + grp + 1) * orank].copy_from_slice(&yg[i * orank..(i + 1) * orank]);
            }
        }
        self.wo_b.forward(&og, t, Out::Bf16)
    }

    /// The queries (`wq_b`, and `also`'s: the indexer's, of the same row) of `t` rows' low-rank row (`wq_a` of x, the
    /// first of `of_x`, normed), and every projection of x in `of_x`: a step's or a check's in one call where a device
    /// takes both stages, the norm made there (a round trip fewer a layer); else a call each, the norm here.
    /// `(of_x's outputs, the queries and also's, the normed row where made here)`; without `with_q` (a prompt's chunk
    /// whose queries its attention's program makes on the device, [`Self::attend_on_card`]) no queries, `also`'s alone.
    #[allow(clippy::type_complexity)]
    fn queries(&self, cfg: &Config, of_x: &[(&Weight, &[f32], usize, std::ops::Range<usize>, Out)], also: Option<&Weight>, t: usize, with_q: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Option<Vec<f32>>) {
        let of_qr: Vec<(&Weight, std::ops::Range<usize>, Out)> = std::iter::once(&self.wq_b).filter(|_| with_q).chain(also).map(|w| (w, 0..w.n(), Out::Bf16)).collect();
        if let Some((x, q)) = (t <= CHAINED_ROWS && with_q).then(|| crate::dsv41::linear::normed_together(of_x, 0, &self.q_norm, cfg.norm_eps, &of_qr)).flatten() {
            return (x, q, None);
        }
        let made_x = crate::dsv41::linear::forward_together(of_x);
        let qr = rmsnorm(&made_x[0], &self.q_norm, cfg.norm_eps);
        let with: Vec<(&Weight, &[f32], usize, std::ops::Range<usize>, Out)> = of_qr.iter().map(|(w, rows, out)| (*w, &qr[..], t, rows.clone(), *out)).collect();
        let made_q = if with.is_empty() { Vec::new() } else { crate::dsv41::linear::forward_together(&with) };
        (made_x, made_q, Some(qr))
    }

    #[allow(clippy::too_many_arguments)]
    /// The indexer's choice of compressed positions for each token, from its queries (`wq_b` of the queries' row) and
    /// its weights' projection of x (`weights_proj`, rounded to bf16), made with the layer's other projections.
    #[allow(clippy::too_many_arguments)]
    fn indexer_forward(
        &self,
        ix: &Indexer,
        cfg: &Config,
        mut q: Vec<f32>,
        weights: &[f32],
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
            // against whatever was published last â€” layer 20's (ratio-1) keys
            // from the previous step, sliced to end_pos/2 rows. An owner's own
            // cache is always the right one. Unobservable while end_pos/ratio
            // <= index_topk (every position is kept), which covers the golden
            // fixtures; it matters past ~1k tokens of context.
            shared.index_k_src = Some(self.layer);
        }

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
        let weights: Vec<f32> = weights.iter().map(|w| to_bf16(w * wscale)).collect();

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

    /// A decode step's attention of one row at `pos` as steps of a device program, its caches `dc`'s (this layer's at
    /// `layer`): from the row `x` (a step's vector, the attention's normed input) the projections [`Self::forward_core`]
    /// makes, the queries rotated, the window's key normed, rotated, quantized and put in its slot, a compressing
    /// source's partial group and latent (pooled past a ratio of 1), its index key and compressed row where it completes
    /// one, an index source's scores and picks, and the heads' attention over the window and the picks, rotated back:
    /// each as the host makes it, to the bit, but for the exponentials of the attention, a ratio-2 pool and the
    /// mixing. `words` is the step's numbers ([`DeviceCaches::words`]). The step of the heads' outputs (`[heads,
    /// head_dim]`, as `forward_core` gives them), or None where the caches cannot take the step (past their reach).
    pub(crate) fn device_steps<'a>(&'a self, cfg: &Config, dc: &'a DeviceCaches, x: usize, words: usize, pos: usize, steps: &mut Vec<crate::dsv41::linear::Step<'a>>) -> Option<usize> {
        use crate::dsv41::linear::Step;
        let (hd, nh, rd, win, ihd, inh) = (cfg.head_dim, cfg.n_heads, cfg.rope_head_dim, cfg.window_size, cfg.index_head_dim, cfg.index_n_heads);
        let lc = &dc.layers[self.layer];
        if pos >= dc.reach || lc.ratio != self.ratio {
            return None;
        }
        let half = rd / 2;
        let push = |steps: &mut Vec<Step<'a>>, s: Step<'a>| {
            steps.push(s);
            steps.len() - 1
        };
        let (cos, sin) = (&dc.rope[lc.rope].0, &dc.rope[lc.rope].1);
        let cos = push(steps, Step::State(cos));
        let sin = push(steps, Step::State(sin));
        let rotate = |steps: &mut Vec<Step<'a>>, v: usize, rows: usize, width: usize, word: u32, inverse: bool| {
            steps.push(Step::Rope { x: v, rows, stride: width, at: width - rd, half, cos, sin, pos: words, word, inverse });
        };
        // the projections of x, and the queries' of the normed low-rank row
        let qa = push(steps, Step::Mul { w: &self.wq_a, x, rows: 1 });
        let kva = push(steps, Step::Mul { w: &self.wkv, x, rows: 1 });
        let source = self.compressor.as_ref().filter(|_| self.ratio > 0);
        let ckv = source.map(|c| push(steps, Step::Mul { w: &c.wkv, x, rows: 1 }));
        let cgate = source.and_then(|c| c.wgate.as_ref().filter(|_| c.ratio > 1)).map(|g| push(steps, Step::Mul { w: g, x, rows: 1 }));
        let indexer = self.indexer.as_ref().filter(|_| self.ratio > 0);
        let iw = indexer.map(|ix| push(steps, Step::Mul { w: &ix.weights_proj, x, rows: 1 }));
        let qr = push(steps, Step::Norm { x: qa, weight: &self.q_norm, eps: cfg.norm_eps, rows: 1, width: self.wq_a.n() });
        let qs = push(steps, Step::Mul { w: &self.wq_b, x: qr, rows: 1 });
        let iqs = indexer.map(|ix| push(steps, Step::Mul { w: &ix.wq_b, x: qr, rows: 1 }));
        let bf16 = |steps: &mut Vec<Step<'a>>, src: usize, len: usize| {
            let dst = push(steps, Step::Scratch(len));
            steps.push(Step::CopyColsBf16 { src, src_stride: len, src_at: 0, dst, dst_stride: len, dst_at: 0, width: len, rows: 1 });
            dst
        };
        let q = bf16(steps, qs, nh * hd);
        rotate(steps, q, nh, hd, W_POS, false);
        // the window's key into its slot
        let kv = push(steps, Step::Norm { x: kva, weight: &self.kv_norm, eps: cfg.norm_eps, rows: 1, width: hd });
        rotate(steps, kv, 1, hd, W_POS, false);
        steps.push(Step::Fp8 { x: kv, len: hd });
        let window = push(steps, Step::State(&lc.window));
        steps.push(Step::PutRow { src: kv, dst: window, width: hd, pos: words, word: W_SLOT });
        // a compressing source's latent, where the step completes a group
        if let (Some(c), Some(ckv)) = (source, ckv) {
            let latent = match (&lc.kv_state, cgate) {
                (None, _) => Some(push(steps, Step::Norm { x: ckv, weight: &c.norm, eps: cfg.norm_eps, rows: 1, width: hd })),
                (Some((kvs, scores)), Some(gate)) => {
                    let (kvs, scores) = (push(steps, Step::State(kvs)), push(steps, Step::State(scores)));
                    steps.push(Step::PutRow { src: ckv, dst: kvs, width: hd, pos: words, word: W_GROUP });
                    steps.push(Step::PutRow { src: gate, dst: scores, width: hd, pos: words, word: W_GROUP });
                    (pos + 1).is_multiple_of(c.ratio).then(|| {
                        let pooled = push(steps, Step::Pool { kv: kvs, score: scores, ratio: c.ratio, width: hd });
                        push(steps, Step::Norm { x: pooled, weight: &c.norm, eps: cfg.norm_eps, rows: 1, width: hd })
                    })
                }
                (Some(_), None) => return None,
            };
            if let Some(latent) = latent {
                if let (Some(ix), Some(keys)) = (indexer.filter(|ix| ix.owns_k), &lc.index_k) {
                    let k = push(steps, Step::Mul { w: ix.wk.as_ref()?, x: latent, rows: 1 });
                    let k = push(steps, Step::Norm { x: k, weight: ix.k_norm.as_ref()?, eps: cfg.norm_eps, rows: 1, width: ihd });
                    rotate(steps, k, 1, ihd, W_CPOS, false);
                    steps.push(Step::Fp4 { x: k, len: ihd, block: 32, e4m3: false });
                    let keys = push(steps, Step::State(keys));
                    steps.push(Step::PutRow { src: k, dst: keys, width: ihd, pos: words, word: W_ROW });
                }
                rotate(steps, latent, 1, hd, W_CPOS, false);
                steps.push(Step::Fp4 { x: latent, len: hd, block: 16, e4m3: true });
                let rows = push(steps, Step::State(lc.compress_kv.as_ref()?));
                steps.push(Step::PutRow { src: latent, dst: rows, width: hd, pos: words, word: W_ROW });
            }
        }
        // an index source's scores of the compressed positions and its picks of them: the candidate source's best blocks
        // flagged (its own picks from all of them), a later source's scores outside those made -infinity first
        let picks = (self.ratio > 0).then(|| push(steps, Step::State(&dc.picks)));
        if let (Some(ix), Some(iw), Some(iqs), Some(picks)) = (indexer, iw, iqs, picks) {
            let iq = bf16(steps, iqs, inh * ihd);
            rotate(steps, iq, inh, ihd, W_POS, false);
            steps.push(Step::Fp4 { x: iq, len: inh * ihd, block: 32, e4m3: false });
            let weights = bf16(steps, iw, inh);
            let keys = push(steps, Step::State(dc.layers[lc.index_src?].index_k.as_ref()?));
            let scale = (ihd as f32).powf(-0.5) * (inh as f32).powf(-0.5);
            let scores = push(steps, Step::IndexScores { q: iq, keys, weights, heads: inh, dim: ihd, scale, pos: words, word: W_COUNT });
            let bs = cfg.candidate_block_size;
            if ix.is_candidate_source {
                let blocks = push(steps, Step::State(&dc.blocks));
                push(steps, Step::BlockKeep { scores, blocks, k: cfg.candidate_topk_blocks, bs, pos: words, word: W_COUNT });
            } else if ix.uses_candidates {
                let blocks = push(steps, Step::State(&dc.blocks));
                steps.push(Step::MaskBlocks { scores, blocks, bs, pos: words, word: W_COUNT });
            }
            steps.push(Step::TopK { scores, dst: picks, k: cfg.index_topk, pos: words, word: W_COUNT });
        }
        // the heads over the window and the picks, rotated back
        let compressed = match picks {
            Some(picks) => Some((push(steps, Step::State(dc.layers[lc.compress_src?].compress_kv.as_ref()?)), picks, cfg.index_topk)),
            None => None,
        };
        let o = push(steps, Step::Attend { q, heads: nh, dim: hd, window, win, compressed, sink: &self.attn_sink, scale: (hd as f32).powf(-0.5), pos: words, word: W_POS });
        rotate(steps, o, nh, hd, W_POS, true);
        Some(o)
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
    crate::dsv41::profile::add(crate::dsv41::profile::Part::IndexScores, scoring);
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
    /// group is still filling up, from its projections of x (made with the
    /// layer's others): `kv` (`wkv`'s, bf16 at a ratio of 1, else f32) and,
    /// past a ratio of 1, `score` (`wgate`'s, f32).
    fn forward(&self, cfg: &Config, kv: Vec<f32>, score: Option<Vec<f32>>, t: usize, start_pos: usize, st: &mut AttnState) -> Option<Vec<f32>> {
        let hd = cfg.head_dim;
        if self.ratio == 1 {
            return Some(rmsnorm(&kv, &self.norm, cfg.norm_eps));
        }
        let r = self.ratio;
        let score = score.expect("ratio > 1 has wgate");
        for i in t.saturating_sub(RECENT)..t {
            let s = (start_pos + i) % RECENT;
            st.recent_kv[s * hd..(s + 1) * hd].copy_from_slice(&kv[i * hd..(i + 1) * hd]);
            st.recent_score[s * hd..(s + 1) * hd].copy_from_slice(&score[i * hd..(i + 1) * hd]);
        }
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

/// The most products of a call's sparse attention that go to the workers that stay ([`crate::dsv41::pool::shared`]): a
/// decode step's at the longest context it attends over (the window and every selected position, 64 heads) is some 42
/// million; a prompt's chunk is past it and starts its own threads.
const POOLED_WORK: usize = 64_000_000;

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
            dspark: None,
        }
    }

    /// A draft's check rewound ([`AttnState::save`], [`AttnState::rewind`]) leaves the window and a compressing
    /// source's partial group as reading only the rows it kept would have: what follows is what follows those rows read
    /// alone, wherever the kept rows end in a group and however many the check read past them.
    #[test]
    fn a_check_rewound_is_the_rows_it_kept_read_alone() {
        let cfg = selection_cfg();
        let (hd, win) = (cfg.head_dim, cfg.window_size);
        let c = Compressor { ratio: 2, wkv: Weight::F32 { w: vec![0.0; hd], n: 1, k: hd }, wgate: None, norm: vec![1.0; hd] };
        // a row at a position, the token's (salt 0) or a draft's the check does not keep (salt 1)
        let row = |p: usize, salt: f32| -> Vec<f32> { (0..hd).map(|f| ((p * 7 + f) as f32 * 0.37 + salt).sin()).collect() };
        let fresh = || AttnState {
            window: vec![0.0; win * hd],
            compress_kv: Vec::new(),
            kv_state: vec![0.0; 2 * hd],
            score_state: vec![f32::NEG_INFINITY; 2 * hd],
            index_k: Vec::new(),
            recent_kv: vec![0.0; RECENT * hd],
            recent_score: vec![f32::NEG_INFINITY; RECENT * hd],
        };
        // rows at `start..` into the window (each at its position's slot, as a pass keeps them) and the compressor:
        // the latents it completes
        let pass = |st: &mut AttnState, start: usize, salts: &[f32]| -> Option<Vec<f32>> {
            let kv: Vec<f32> = salts.iter().enumerate().flat_map(|(i, &s)| row(start + i, s)).collect();
            let score: Vec<f32> = salts.iter().enumerate().flat_map(|(i, &s)| row(start + i, s + 2.0)).collect();
            for i in 0..salts.len() {
                let s = (start + i) % win;
                st.window[s * hd..(s + 1) * hd].copy_from_slice(&kv[i * hd..(i + 1) * hd]);
            }
            c.forward(&cfg, kv, Some(score), salts.len(), start, st)
        };
        for (start, read, keep) in [(5, 4, 2), (5, 4, 1), (6, 5, 3), (6, 3, 3), (5, 6, 5), (4, 2, 1)] {
            let (mut a, mut b) = (fresh(), fresh());
            pass(&mut a, 0, &[0.0; 4]);
            pass(&mut b, 0, &[0.0; 4]);
            if start > 4 {
                pass(&mut a, 4, &vec![0.0; start - 4]);
                pass(&mut b, 4, &vec![0.0; start - 4]);
            }
            // a: the check's rows, those past `keep` not the token's, then rewound; b: the kept rows alone
            let saved = a.save(hd, start, read);
            let salts: Vec<f32> = (0..read).map(|i| if i < keep { 0.0 } else { 1.0 }).collect();
            pass(&mut a, start, &salts);
            a.rewind(hd, &saved, keep, 2);
            pass(&mut b, start, &vec![0.0; keep]);
            // then the same rows after, one and several at a time
            let mut at = start + keep;
            for n in [1, 1, 3, 2] {
                let salts = vec![0.0; n];
                assert_eq!(pass(&mut a, at, &salts), pass(&mut b, at, &salts), "a check from {start} of {read} rows keeping {keep}: the latents of {n} rows at {at}");
                at += n;
            }
            assert_eq!((&a.window, &a.kv_state, &a.score_state), (&b.window, &b.kv_state, &b.score_state), "a check from {start} of {read} rows keeping {keep}: the caches");
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
