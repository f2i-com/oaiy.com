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
//!
//! With the model's multi-token-prediction layer the chain also drafts tokens ([`Qwen35Chain::draft`]) and checks them
//! ([`Qwen35Chain::check`]): a run of the sampled token and its drafts, every row's logits back; a draft the sampler
//! does not pick is undone ([`Qwen35Chain::rollback`]): each delta net's state and conv window as they were before the
//! check (kept then), its accepted rows run through them again.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ggml_rs::chain::DeltaNet;
use ggml_rs::{Backend, DeviceChain, DeviceVec, QuantizedTensor, Tensor};

use crate::kv_cache::KvCache;
use crate::loader::{FfnPair, Weight};
use crate::qwen35::{is_no_ffn, Qwen35Block, Qwen35Model, Qwen35Moe};

// (parts of `impl Qwen35Chain`, each with what it alone uses)
mod drafts;
mod two_devices;

/// Rows a submit takes at most over two cards (the server's prompt chunks are 512), twice that on one.
const MAX_ROWS: usize = 512;

/// A prompt's pieces on a device's queue at once ([`DeviceChain::pieces_in_flight_at_most`]), for a run of
/// [`FEW_ROWS`] or more: a card under a power limit (an RTX 5090 at 402 W of its 575) otherwise runs a tenth as fast
/// for seconds at a time through a long prompt's chunks (the 27B's 15,360 tokens on one card: some 7 s in, a chunk
/// 2.7 s where 0.26, for 3 to 15 s at a time; over two cards the second's half of a chunk 1.05 s where 0.05: 15,646
/// tokens in 8.7 to 14.9 s, where now 4.3 to 5.7). So fed it does so once, for a second or two, and settles.
const PIECES_IN_FLIGHT: usize = 2;
const FEW_ROWS: usize = 64;

/// Devices' queues holding a few pieces at once ([`PIECES_IN_FLIGHT`]) until this is dropped.
struct FewInFlight<'a>(Vec<&'a dyn DeviceChain>);

impl<'a> FewInFlight<'a> {
    fn on(chains: impl IntoIterator<Item = &'a dyn DeviceChain>) -> Self {
        let chains: Vec<_> = chains.into_iter().collect();
        for c in &chains {
            c.pieces_in_flight_at_most(PIECES_IN_FLIGHT);
        }
        Self(chains)
    }
}

impl Drop for FewInFlight<'_> {
    fn drop(&mut self) {
        for c in &self.0 {
            c.pieces_in_flight_at_most(0);
        }
    }
}

/// The most a prompt's attention scratch may take: a chunk's rows are cut to fit it (at 16K positions, 24 heads of
/// 256 take 1.6 MB a row where the backend's attention leaves each run of positions' part there; a kernel that
/// leaves none, 25 KB).
const ATTENTION_SCRATCH: usize = 512 << 20;

/// Qwen3.5's chained runs: the state on the device, made at the first run that can use one; none when the backend
/// has no chain or a weight a run reads is not on its device.
#[derive(Default)]
pub struct Qwen35Chain {
    state: OnceLock<Option<State>>,
    /// Runs the chain took (a step or a prompt's chunk), for a test that has to know it ran.
    pub runs: AtomicUsize,
    /// A second device for prompts ([`Self::split_onto`]), and its share of their layers, made at the first prompt
    /// that can use it.
    second: OnceLock<Arc<dyn Backend>>,
    split: OnceLock<Option<Split>>,
    /// Prompts on the first device alone all the same (a test's comparison).
    pub split_off: AtomicBool,
    /// The positions of a run whose rows are not the next ones in order ([`Self::forward_positions`]: a picture's
    /// tokens, and the text after one): the cache's length the run began at, and each of its rows' three axes.
    positions: Mutex<Option<(usize, Vec<[u32; 3]>)>>,
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
    /// A MoE layer's router (`[experts + 1, d]`, f32: the shared expert's gate its last row).
    router: Option<DeviceVec>,
}

enum Mixer {
    /// The per-head q and k norms, and which of the attention cache's layers is this one's.
    Attention { q_norm: DeviceVec, k_norm: DeviceVec, slot: usize },
    /// An LFM2 short conv's weights (`[d, kernel]`), and which recurrent state is this one's (its conv window; the
    /// delta-net state there a placeholder).
    Conv { conv_w: DeviceVec, slot: usize },
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
    /// A MoE model's router's rows (its experts and the shared one's gate); 0: dense.
    router: usize,
    /// The rows a piece takes at most where a MoE model's experts route no more on its device at once (0: as many).
    moe_rows: usize,
    /// Whether an attention layer's q has a gate half (Qwen3.5's; LFM2's has none).
    gated: bool,
    /// An LFM2 short conv's in-projection's width (`3 d`: each row's B, C and x); 0: none.
    bcx: usize,
}

/// The chain's copy of the attention layers' caches, a buffer a layer (row `t`: its K `[n_kv, head_dim]` then its
/// V), `cap` rows of them, and a decode step's attention output and scratch.
struct Kv {
    layers: Vec<DeviceVec>,
    cap: usize,
    out: DeviceVec,
    /// The [`KvCache::id`] the rows are a copy of (0: none).
    owner: u64,
    /// The layers' rows as f16, two values a word ([`ggml_rs::ChainRecorder::halve`]), for a step's attention deep in
    /// a long cache ([`HALVES_FROM`]); none until such a step, or where the device has no such attention. Their
    /// first `halved` rows are the layers': whatever writes a layer's rows from a position on lowers it to there.
    half: Vec<DeviceVec>,
    halved: usize,
    /// The `cap` at which the device had no room for the halves (not asked again until the cache grows).
    refused: usize,
    /// The rows it holds of `owner`'s (as its last run left them), the first `owner_host` of which the host's cache
    /// holds as well (the rest the device's alone: [`ChainRows`]); `gone`, the owner's cache dropped.
    rows: usize,
    owner_host: usize,
    gone: bool,
    /// Rows of caches whose copy this was, set aside as another's run took it ([`stash_owner`]): each cache's, from
    /// its first the host did not hold (a slot's rows each K then V, f32), until that cache runs again or is gone.
    stash: Vec<(u64, Stash)>,
}

struct Stash {
    from: usize,
    to: usize,
    slots: Vec<Vec<f32>>,
}

impl Kv {
    fn new(out: DeviceVec) -> Kv {
        Kv { layers: Vec::new(), cap: 0, out, owner: 0, half: Vec::new(), halved: 0, refused: 0, rows: 0, owner_host: 0, gone: false, stash: Vec::new() }
    }

    /// Rows `from..to` of attention slot `slot`, each its K then its V, read from the device (from the halves where the
    /// cache is held as halves alone).
    fn read_rows(&self, chain: &dyn DeviceChain, slot: usize, row: usize, from: usize, to: usize, halves_only: bool) -> Vec<f32> {
        let mut rec = chain.begin();
        if halves_only {
            rec.read_range(&self.half[slot], from * row / 2, (to - from) * row / 2);
        } else {
            rec.read_range(&self.layers[slot], from * row, (to - from) * row);
        }
        let got = rec.finish().pop().expect("the rows read");
        if halves_only { from_halves(&got) } else { got }
    }
}

/// VENDORED-LOCAL: the rows of a cache this chain's copy of the attention cache holds alone, for the cache to read
/// ([`crate::kv_cache::DeviceRows`]): a chained run writes its K and V rows there and nowhere else (they came back to
/// the host's cache as well, every run, which then held all of them a second time in the computer's memory: the 27B's
/// 128 KB a token, beside the card's copy).
struct ChainRows {
    kv: Arc<Mutex<Kv>>,
    backend: Arc<dyn Backend>,
    /// each of the model's layers' attention slot (none: a recurrent layer)
    slots: Vec<Option<usize>>,
    n_kv: usize,
    hd: usize,
    halves_only: bool,
}

impl std::fmt::Debug for ChainRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChainRows({} attention layers)", self.slots.iter().flatten().count())
    }
}

impl crate::kv_cache::DeviceRows for ChainRows {
    fn holds(&self, layer: usize) -> bool {
        self.slots.get(layer).copied().flatten().is_some()
    }

    fn rows(&self, id: u64, layer: usize, from: usize, to: usize) -> Option<(Tensor, Tensor)> {
        let slot = self.slots.get(layer).copied().flatten()?;
        let (kvd, row) = (self.n_kv * self.hd, 2 * self.n_kv * self.hd);
        let g = self.kv.lock().unwrap_or_else(|p| p.into_inner());
        let flat = if from >= to {
            Vec::new()
        } else if let Some((_, st)) = g.stash.iter().find(|(owner, _)| *owner == id) {
            if from < st.from || to > st.to {
                return None;
            }
            st.slots[slot][(from - st.from) * row..(to - st.from) * row].to_vec()
        } else if g.owner == id && to <= g.rows && g.cap >= to {
            g.read_rows(self.backend.chain()?, slot, row, from, to, self.halves_only)
        } else {
            return None;
        };
        let n = to.saturating_sub(from);
        let (mut k, mut v) = (Vec::with_capacity(n * kvd), Vec::with_capacity(n * kvd));
        for r in flat.chunks_exact(row) {
            k.extend_from_slice(&r[..kvd]);
            v.extend_from_slice(&r[kvd..]);
        }
        Some((Tensor::from_vec(k, vec![n, self.n_kv, self.hd]), Tensor::from_vec(v, vec![n, self.n_kv, self.hd])))
    }

    fn forget(&self, id: u64) {
        let mut g = self.kv.lock().unwrap_or_else(|p| p.into_inner());
        g.stash.retain(|(owner, _)| *owner != id);
        if g.owner == id {
            g.gone = true;
        }
    }
}

/// Before the copy becomes another cache's: the rows of its owner's that the device held alone set aside (read to
/// the host), for that cache to read and to be brought back from. Nothing where the owner is gone, or the host held
/// them all.
fn stash_owner(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, slots: usize, halves_only: bool) {
    let (from, to) = (g.owner_host, g.rows.min(g.cap));
    if g.owner == 0 || g.gone || from >= to {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let rows: Vec<Vec<f32>> = (0..slots).map(|slot| g.read_rows(chain, slot, row, from, to, halves_only)).collect();
    let owner = g.owner;
    g.stash.retain(|(o, _)| *o != owner);
    g.stash.push((owner, Stash { from, to, slots: rows }));
}

/// A step's attention reads the cache's f16 halves from this many positions on: its parts are then bound by the
/// bytes they read (the 27B at 15,888 positions: 130 MB a layer at half the card's bandwidth, 2.3 ms of a step's
/// 15.7), where under some 4,000 a workgroup's own time is what they take.
const HALVES_FROM: usize = 4096;

/// [`HALVES_FROM`], or OAIY_HALVES_FROM's (a test's: 1 has every step read the halves).
fn halves_from() -> usize {
    static FROM: OnceLock<usize> = OnceLock::new();
    *FROM.get_or_init(|| std::env::var("OAIY_HALVES_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(HALVES_FROM))
}

/// The vectors the cache's recurrent state tensors alias, a delta net layer's each: its state and its conv's last
/// inputs.
struct Pool {
    states: Vec<DeviceVec>,
    convs: Vec<DeviceVec>,
}

/// A projection a chain multiplies by: a GGUF's quantized matrix, or a packed one its device holds (EXL3: OrcaSAQ's
/// checkpoint, whose every projection is one, its channel maps the matrix's own).
#[derive(Clone, Copy)]
enum W<'a> {
    Quant(&'a QuantizedTensor),
    Packed(&'a dyn ggml_rs::exl3::PackedLinear),
    /// A MoE layer's FFN of no shared expert (LFM2's: `qwen35::no_ffn`), never multiplied by.
    Absent,
}

impl<'a> W<'a> {
    /// `w`, which the chain's state checked the device holds.
    fn of(w: &'a Weight) -> Self {
        match w {
            Weight::Quant(q) => W::Quant(q),
            Weight::Packed(p) => W::Packed(p.as_ref()),
            w if is_no_ffn(w) => W::Absent,
            _ => unreachable!("the chain state checked every weight"),
        }
    }

    /// Whether `chain`'s device holds `w`, so a chain can multiply by it.
    fn held(chain: &dyn DeviceChain, w: &Weight) -> bool {
        match w {
            Weight::Quant(q) => chain.holds(q),
            Weight::Packed(p) => chain.holds_exl3(p.as_ref()),
            _ => false,
        }
    }

    /// `y`'s `rows` rows = `x`'s times this.
    fn rows(self, rec: &mut dyn ggml_rs::ChainRecorder, x: &DeviceVec, y: &DeviceVec, rows: usize) {
        match self {
            W::Quant(q) => rec.matmul_rows(q, x, y, rows),
            W::Packed(p) => rec.exl3_rows(p, x, y, rows),
            W::Absent => unreachable!("a MoE layer's experts are its FFN"),
        }
    }

    /// `y` = one row `x` times this.
    fn one(self, rec: &mut dyn ggml_rs::ChainRecorder, x: &DeviceVec, y: &DeviceVec) {
        match self {
            W::Quant(q) => rec.matmul(q, x, y),
            W::Packed(p) => rec.exl3_rows(p, x, y, 1),
            W::Absent => unreachable!("a MoE layer's experts are its FFN"),
        }
    }
}

/// A layer's projections a run reads: the model's own (on the first device), or their copies on the second.
#[derive(Clone, Copy)]
enum LayerW<'a> {
    Attention { q: W<'a>, k: W<'a>, v: W<'a>, o: W<'a>, ffn: FfnW<'a>, down: W<'a>, moe: Option<&'a Qwen35Moe> },
    /// `ba` none where the fused beta-alpha projection's weights are f32 (the layer's vectors hold them); `moe` a MoE
    /// layer's experts (its `ffn` and `down` then its shared expert's, unread here: the experts hold it)
    Ssm { qkv: W<'a>, gate: W<'a>, ba: Option<W<'a>>, out: W<'a>, ffn: FfnW<'a>, down: W<'a>, moe: Option<&'a Qwen35Moe> },
    /// An LFM2 short conv's projections
    Conv { in_proj: W<'a>, out: W<'a>, ffn: FfnW<'a>, down: W<'a>, moe: Option<&'a Qwen35Moe> },
}

/// An FFN's gate and up projections: fused, or a pair.
#[derive(Clone, Copy)]
enum FfnW<'a> {
    Fused(W<'a>),
    Split(W<'a>, W<'a>),
}

impl<'a> LayerW<'a> {
    /// The model's own block's.
    fn of(b: &'a Qwen35Block) -> Self {
        let ffn = |p: &'a FfnPair| match p {
            FfnPair::Fused(w) => FfnW::Fused(W::of(w)),
            FfnPair::Split { gate, up } => FfnW::Split(W::of(gate), W::of(up)),
        };
        match b {
            Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, moe, .. } => {
                LayerW::Attention { q: W::of(attn_q), k: W::of(attn_k), v: W::of(attn_v), o: W::of(attn_output), ffn: ffn(ffn_pair), down: W::of(ffn_down), moe: moe.as_deref() }
            }
            Qwen35Block::Ssm { attn_qkv, attn_gate, ssm_ba, ssm_out, ffn_pair, ffn_down, moe, .. } => LayerW::Ssm {
                qkv: W::of(attn_qkv),
                gate: W::of(attn_gate),
                ba: match ssm_ba {
                    Weight::Quant(q) => Some(W::Quant(q)),
                    _ => None,
                },
                out: W::of(ssm_out),
                ffn: ffn(ffn_pair),
                down: W::of(ffn_down),
                moe: moe.as_deref(),
            },
            Qwen35Block::Conv { in_proj, out_proj, ffn_pair, ffn_down, moe, .. } => {
                LayerW::Conv { in_proj: W::of(in_proj), out: W::of(out_proj), ffn: ffn(ffn_pair), down: W::of(ffn_down), moe: moe.as_deref() }
            }
        }
    }
}

/// A layer's quantized weights copied to the second device ([`LayerW`]'s, owned).
enum SplitW {
    Attention { q: QuantizedTensor, k: QuantizedTensor, v: QuantizedTensor, o: QuantizedTensor, ffn: SplitFfn, down: QuantizedTensor },
    Ssm { qkv: QuantizedTensor, gate: QuantizedTensor, ba: Option<QuantizedTensor>, out: QuantizedTensor, ffn: SplitFfn, down: QuantizedTensor },
}

enum SplitFfn {
    Fused(QuantizedTensor),
    Split(QuantizedTensor, QuantizedTensor),
}

impl SplitW {
    /// `w`'s copies on `chain`'s device: None where one cannot be made there.
    fn copy(chain: &dyn DeviceChain, w: LayerW<'_>) -> Option<SplitW> {
        let c = |w: W<'_>| match w {
            W::Quant(q) => chain.copy_weight(q),
            W::Packed(_) | W::Absent => None,
        };
        let ffn = |f: FfnW<'_>| -> Option<SplitFfn> {
            Some(match f {
                FfnW::Fused(gu) => SplitFfn::Fused(c(gu)?),
                FfnW::Split(g, u) => SplitFfn::Split(c(g)?, c(u)?),
            })
        };
        Some(match w {
            // (a MoE layer's experts are its device's: not copied; nor is a short conv)
            LayerW::Attention { moe: Some(_), .. } | LayerW::Ssm { moe: Some(_), .. } | LayerW::Conv { .. } => return None,
            LayerW::Attention { q, k, v, o, ffn: f, down, moe: None } => SplitW::Attention { q: c(q)?, k: c(k)?, v: c(v)?, o: c(o)?, ffn: ffn(f)?, down: c(down)? },
            LayerW::Ssm { qkv, gate, ba, out, ffn: f, down, moe: None } => SplitW::Ssm {
                qkv: c(qkv)?,
                gate: c(gate)?,
                ba: match ba {
                    Some(b) => Some(c(b)?),
                    None => None,
                },
                out: c(out)?,
                ffn: ffn(f)?,
                down: c(down)?,
            },
        })
    }

    fn view(&self) -> LayerW<'_> {
        fn ffn(f: &SplitFfn) -> FfnW<'_> {
            match f {
                SplitFfn::Fused(gu) => FfnW::Fused(W::Quant(gu)),
                SplitFfn::Split(g, u) => FfnW::Split(W::Quant(g), W::Quant(u)),
            }
        }
        match self {
            SplitW::Attention { q, k, v, o, ffn: f, down } => {
                LayerW::Attention { q: W::Quant(q), k: W::Quant(k), v: W::Quant(v), o: W::Quant(o), ffn: ffn(f), down: W::Quant(down), moe: None }
            }
            SplitW::Ssm { qkv, gate, ba, out, ffn: f, down } => {
                LayerW::Ssm { qkv: W::Quant(qkv), gate: W::Quant(gate), ba: ba.as_ref().map(W::Quant), out: W::Quant(out), ffn: ffn(f), down: W::Quant(down), moe: None }
            }
        }
    }
}

/// The layers from `from` on and the head, copied to a second device for prompts: each chunk's first layers run on
/// the first device as the second runs the chunk before's last ones (steps and checks run on the first, which holds
/// every layer).
struct Split {
    from: usize,
    weights: Vec<SplitW>,
    /// the layers' vectors there, their slots the second's own
    layers: Vec<LayerVecs>,
    /// for each of the second's attention slots and recurrent states, the first's
    attention_slots: Vec<usize>,
    ssm_slots: Vec<usize>,
    output_norm: DeviceVec,
    output: QuantizedTensor,
    logits: DeviceVec,
    kv: Mutex<SplitKv>,
    pool: Pool,
    /// A tapped run's later rows' vectors on the second device, made at the first run that keeps a state there.
    seg: std::sync::OnceLock<Seg>,
}

/// The second device's copy of its layers' attention cache: its rows up to `upto` are the cache's (`kv.owner`'s).
struct SplitKv {
    kv: Kv,
    upto: usize,
}

/// Rows a check of drafted tokens takes at most (the sampled token and its drafts).
pub const SPEC_ROWS: usize = 8;

/// The least probability (under the prediction layer) a draft is checked at: Strata's `--spec-min-p` default.
const DRAFT_MIN_P: f64 = 0.5;

/// The multi-token-prediction layer's own copy of its attention cache (row `p`: its K then V), `cap` rows, and a
/// row's attention output and scratch; its entries for positions `start..valid` are the layer's at those positions
/// (from the trunk's hidden state and the token after), of the cache `owner`.
struct MtpKv {
    layer: DeviceVec,
    cap: usize,
    out: DeviceVec,
    start: usize,
    valid: usize,
    owner: u64,
}

/// The layer's vectors for up to [`SPEC_ROWS`] rows.
struct MtpWork {
    e: DeviceVec,
    h: DeviceVec,
    en: DeviceVec,
    hn: DeviceVec,
    cat: DeviceVec,
    x: DeviceVec,
    xn: DeviceVec,
    qfull: DeviceVec,
    q: DeviceVec,
    gate: DeviceVec,
    k: DeviceVec,
    v: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    q1: DeviceVec,
    att: DeviceVec,
    gated: DeviceVec,
    proj: DeviceVec,
    ffa: DeviceVec,
    ffb: DeviceVec,
    act: DeviceVec,
    table: DeviceVec,
    last: DeviceVec,
    logits: DeviceVec,
    /// A draft's token, its logit and the sum of the exponentials against it (`argmax_softmax`).
    best: DeviceVec,
}

/// What drafting and checking drafts need on the device: the multi-token-prediction layer's norms, its cache and
/// vectors; every row's hidden state after the output norm, of the last run (a check's rows, or a run's last); a
/// check's logits; and each delta net's state and conv window as they were before the last check, with the check's
/// inputs to them (its rows' qkv and beta-alpha).
struct Spec {
    enorm: DeviceVec,
    hnorm: DeviceVec,
    head_norm: DeviceVec,
    attn_norm: DeviceVec,
    post_norm: DeviceVec,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    kv: Mutex<MtpKv>,
    work: MtpWork,
    hid: DeviceVec,
    /// Where `hid`'s rows are: the position of its first, and how many.
    hid_at: Mutex<(usize, usize)>,
    logits: DeviceVec,
    backups: Vec<(DeviceVec, DeviceVec)>,
    inputs: Vec<(DeviceVec, DeviceVec)>,
    scratch_conv: DeviceVec,
    scratch_core: DeviceVec,
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
    /// A MoE layer's router's logits
    router: DeviceVec,
    /// An LFM2 short conv's in-projection's output (each row's B, C and x)
    bcx: DeviceVec,
}

impl Work {
    /// Its first `t` rows' vectors (the same buffers): a pooled set's for a chunk of fewer rows than it has room for.
    fn view(&self, s: &Dims, t: usize) -> Work {
        Work {
            x: first(&self.x, t * s.d),
            xn: first(&self.xn, t * s.d),
            qkv: first(&self.qkv, t * s.ch),
            z: first(&self.z, t * s.nv * s.dv),
            ba: first(&self.ba, t * 2 * s.nv),
            conv: first(&self.conv, t * s.ch),
            core: first(&self.core, t * s.nv * s.dv),
            proj: first(&self.proj, t * s.d),
            qfull: first(&self.qfull, t * s.n_h * 2 * s.hd),
            q: first(&self.q, t * s.n_h * s.hd),
            gate: first(&self.gate, t * s.n_h * s.hd),
            k: first(&self.k, t * s.n_kv * s.hd),
            v: first(&self.v, t * s.n_kv * s.hd),
            qn: first(&self.qn, t * s.n_h * s.hd),
            kn: first(&self.kn, t * s.n_kv * s.hd),
            gated: first(&self.gated, t * s.n_h * s.hd),
            ffa: first(&self.ffa, t * 2 * s.ff),
            ffb: first(&self.ffb, t * s.ff),
            act: first(&self.act, t * s.ff),
            table: first(&self.table, t * s.rot),
            last: self.last.clone(),
            router: first(&self.router, t * s.router.max(1)),
            bcx: first(&self.bcx, t * s.bcx.max(1)),
        }
    }

    fn new(chain: &dyn DeviceChain, s: &Dims, t: usize) -> Work {
        Self::made(s, t, |n| chain.vec(n))
    }

    /// The words [`Self::new`]'s vectors hold between them, none of them made.
    fn words(s: &Dims, t: usize) -> usize {
        let mut words = 0;
        Self::made(s, t, |n| {
            words += n;
            DeviceVec { len: n, inner: Arc::new(()) }
        });
        words
    }

    /// A set for `t` rows, each vector `v`'s of its length.
    fn made(s: &Dims, t: usize, mut v: impl FnMut(usize) -> DeviceVec) -> Work {
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
            router: v(t * s.router.max(1)),
            bcx: v(t * s.bcx.max(1)),
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
    kv: Arc<Mutex<Kv>>,
    /// Its copy's rows, for the caches it runs ([`ChainRows`]).
    rows: Arc<ChainRows>,
    pool: Mutex<Pool>,
    /// A decode step's vectors, kept (and so their bind groups).
    step: Work,
    logits: DeviceVec,
    /// Drafting and checking, where the model has a multi-token-prediction layer the device holds.
    spec: Option<Spec>,
    /// A prompt's chunks' vectors, each set room for the most rows a chunk has, and their attention's scratch: taken
    /// by a chunk and given back once it has run (each chunk's new ones, some 25 buffers and 0.5 GB, were the
    /// allocator's every 200 ms).
    prompt_sets: Mutex<Vec<(Work, DeviceVec)>>,
    /// A tapped run's later rows' vectors, made at the first run that keeps a state from inside.
    seg: std::sync::OnceLock<Seg>,
    /// The attention cache held as f16 halves alone, with no f32 copy (`DeviceChain::halves_only`: a card with tensor
    /// cores, and no drafting layer, whose cache is f32): [`reserve_halves`]. The 27B's cache at 32,768 positions was
    /// 4.3 GB of f32 and 2.1 of halves beside it on a 24 GB card with 15.8 GB of weights; as halves alone the card holds
    /// 77,824 positions beside a prompt's work ([`Qwen35Chain::context_room`]), and a step's attention read the halves
    /// already.
    halves_only: bool,
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

/// [`rope_table`] for rows whose positions are given, each in three axes (time, height, width), as the CPU's
/// `multimodal_rope::text` rotates them: a pair's angle is its axis's position times the pair's frequency, the axes
/// interleaved over the pairs by Qwen's sections of 11, 11 and 10 (`rot` 64, the one layout there is); a row whose
/// three are equal (text) is [`rope_table`]'s at that position.
fn rope_table_at(theta: f32, rot: usize, positions: &[[u32; 3]]) -> Vec<f32> {
    let axis = |k: usize| {
        if k % 3 == 1 && k < 33 {
            1
        } else if k % 3 == 2 && k < 30 {
            2
        } else {
            0
        }
    };
    positions
        .iter()
        .flat_map(|pos| {
            (0..rot / 2).flat_map(move |k| {
                let (s, c) = (pos[axis(k)] as f32 * theta.powf(-2.0 * k as f32 / rot as f32)).sin_cos();
                [s, c]
            })
        })
        .collect()
}

#[cfg(test)]
mod position_tests {
    use super::{rope_table, rope_table_at};
    use ggml_rs::Tensor;

    /// The table for rows at given positions rotates as the CPU's `multimodal_rope::text` does (a picture's tokens,
    /// placed by row and column), and for text (three equal axes) is the plain table at that position.
    #[test]
    fn a_table_at_given_positions_is_the_multimodal_rotation() {
        let (theta, rot, width, heads) = (10_000_000.0f32, 64usize, 80usize, 2usize);
        let text = [[7u32; 3], [8; 3], [1234; 3]];
        assert_eq!(rope_table_at(theta, rot, &text[..2]), rope_table(theta, rot, 7, 2));
        assert_eq!(rope_table_at(theta, rot, &text[2..]), rope_table(theta, rot, 1234, 1));
        // a picture's tokens: one time, rows and columns their own
        let positions = [[5u32, 5, 5], [5, 5, 6], [5, 6, 5], [5, 9, 14], [6, 6, 6]];
        let x: Vec<f32> = (0..positions.len() * heads * width).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut want = Tensor::from_vec(x.clone(), vec![positions.len(), heads, width]);
        crate::multimodal_rope::text(&*ggml_rs::default_backend(), &mut want, &positions, rot, theta);
        let table = rope_table_at(theta, rot, &positions);
        let mut got = x;
        for row in 0..positions.len() {
            for k in 0..rot / 2 {
                let (sin, cos) = (table[(row * rot / 2 + k) * 2], table[(row * rot / 2 + k) * 2 + 1]);
                for head in 0..heads {
                    let a = (row * heads + head) * width + k;
                    let c = a + rot / 2;
                    (got[a], got[c]) = (got[a] * cos - got[c] * sin, got[a] * sin + got[c] * cos);
                }
            }
        }
        let worst = got.iter().zip(want.to_host().data()).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
        // (and the axes matter: the same rows at their time alone rotate otherwise)
        assert_ne!(rope_table_at(theta, rot, &positions[3..4]), rope_table(theta, rot, 5, 1));
    }

    /// A device that only counts: the bytes of the vectors it has made that are still held, and the most at once.
    struct Counting {
        live: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        most: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        /// The vectors it still makes before it refuses one, as a device with no room does (a panic): none refused
        /// where this is `usize::MAX`.
        left: std::sync::atomic::AtomicUsize,
    }

    struct Counted(usize, std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.fetch_sub(self.0, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl ggml_rs::DeviceChain for Counting {
        fn vec(&self, len: usize) -> ggml_rs::DeviceVec {
            use std::sync::atomic::Ordering::SeqCst;
            if self.left.load(SeqCst) != usize::MAX && self.left.fetch_sub(1, SeqCst) == 0 {
                self.left.store(usize::MAX, SeqCst);
                panic!("the device has no room");
            }
            let now = self.live.fetch_add(len * 4, SeqCst) + len * 4;
            self.most.fetch_max(now, SeqCst);
            ggml_rs::DeviceVec { len, inner: std::sync::Arc::new(Counted(len * 4, std::sync::Arc::clone(&self.live))) }
        }
        fn zero(&self, _: &ggml_rs::DeviceVec) {}
        fn alias(&self, _: &ggml_rs::DeviceVec, _: Vec<usize>) -> Tensor {
            unimplemented!()
        }
        fn aliased(&self, _: &Tensor) -> Option<ggml_rs::DeviceVec> {
            None
        }
        fn upload_at(&self, _: &ggml_rs::DeviceVec, _: usize, _: &[f32]) {}
        fn resize(&self, _: &ggml_rs::DeviceVec, len: usize) -> ggml_rs::DeviceVec {
            self.vec(len)
        }
        fn holds(&self, _: &ggml_rs::QuantizedTensor) -> bool {
            false
        }
        fn holds_exl3(&self, _: &dyn ggml_rs::exl3::PackedLinear) -> bool {
            false
        }
        fn holds_experts(&self, _: &dyn ggml_rs::exl3::Experts) -> bool {
            false
        }
        fn attention_out_len(&self, n_h: usize, head_dim: usize, _: usize) -> usize {
            n_h * head_dim
        }
        fn attention_rows_out_len(&self, _: usize, _: usize, _: usize, _: usize) -> usize {
            0
        }
        fn begin(&self) -> Box<dyn ggml_rs::ChainRecorder + '_> {
            unimplemented!()
        }
    }

    /// A device that refuses a vector part way through a cache's growth (no room: a panic, which the model's thread
    /// is brought back from) leaves a cache that is nobody's and of no rows, whichever way it was growing: the next
    /// prompt makes every layer and the output anew. Left at its old size with some layers grown, some gone and no
    /// output, the next prompt that fitted the old size read past what was there, and so did every one after.
    #[test]
    fn a_growth_the_device_refuses_leaves_a_cache_the_next_prompt_makes_anew() {
        use std::sync::atomic::Ordering::SeqCst;
        let s = super::Dims { d: 5120, n_h: 24, n_kv: 4, hd: 256, rot: 0, ff: 0, nv: 0, nk: 0, dk: 0, dv: 0, ch: 0, kern: 0, vocab: 0, router: 0, moe_rows: 0, gated: true, bcx: 0 };
        let (slots, row) = (16, 2 * s.n_kv * s.hd);
        for keep in [1024, 0] {
            let chain = Counting { live: Default::default(), most: Default::default(), left: usize::MAX.into() };
            let mut g = super::Kv::new(ggml_rs::DeviceChain::vec(&chain, 1));
            super::reserve(&chain, &mut g, &s, slots, 1024, 0);
            g.owner = 7;
            // (the output's stand-in, then five layers, and the sixth refused)
            chain.left.store(6, SeqCst);
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| super::reserve(&chain, &mut g, &s, slots, 2048, keep)));
            assert!(refused.is_err());
            assert_eq!((g.cap, g.owner), (0, 0), "keep {keep}: nobody's, of no rows");
            // the next prompt, one the old size would have held: whoever asks finds the copy is not theirs, keeps
            // nothing of it, and it is whole again
            let cache = crate::kv_cache::KvCache::new_lazy_per_layer_kv(Vec::new(), 4096, &[], &[]);
            assert_eq!(super::kept(&g, &cache, 512), 0);
            super::reserve(&chain, &mut g, &s, slots, 900, 0);
            assert_eq!((g.cap, g.layers.len()), (1024, slots));
            assert!(g.layers.iter().all(|l| l.len == 1024 * row), "keep {keep}: {:?}", g.layers.iter().map(|l| l.len).collect::<Vec<_>>());
            assert_eq!(g.out.len, ggml_rs::DeviceChain::attention_out_len(&chain, s.n_h, s.hd, 1024));
        }
    }

    /// The 27B's attention cache on a 24 GB card: 16 layers of 4 KV heads of 256, grown from 16,384 rows to 32,768 by
    /// a prompt past the first's length. All the new layers made beside all the old ones is 6 GB at once, which the
    /// card did not have beside the model's weights. A layer at a time it is the new cache and one old layer; and
    /// with no row of the old one kept (a prompt begun again), the new cache alone.
    #[test]
    fn a_cache_grows_without_holding_the_old_one_beside_the_new() {
        use std::sync::atomic::Ordering::SeqCst;
        let s = super::Dims { d: 5120, n_h: 24, n_kv: 4, hd: 256, rot: 0, ff: 0, nv: 0, nk: 0, dk: 0, dv: 0, ch: 0, kern: 0, vocab: 0, router: 0, moe_rows: 0, gated: true, bcx: 0 };
        let (slots, row) = (16, 2 * s.n_kv * s.hd);
        let layer = |cap: usize| cap * row * 4;
        for keep in [16_384, 0] {
            let chain = Counting { live: Default::default(), most: Default::default(), left: usize::MAX.into() };
            let mut g = super::Kv::new(ggml_rs::DeviceChain::vec(&chain, 1));
            super::reserve(&chain, &mut g, &s, slots, 15_088, 0);
            assert_eq!((g.cap, g.layers.len()), (16_384, slots));
            // (its halves, as a step deep in it makes them)
            g.half = (0..slots).map(|_| ggml_rs::DeviceChain::vec(&chain, 16_384 * row / 2)).collect();
            let old = chain.live.load(SeqCst);
            assert!(old >= slots * layer(16_384) * 3 / 2);
            chain.most.store(old, SeqCst);
            super::reserve(&chain, &mut g, &s, slots, 17_621, keep);
            assert_eq!((g.cap, g.layers.len(), g.half.len()), (32_768, slots, 0));
            let (new, most) = (chain.live.load(SeqCst), chain.most.load(SeqCst));
            assert!(new >= slots * layer(32_768) && new < slots * layer(32_768) + (1 << 20), "{new}");
            // the most at once, past what the grown cache holds: one old layer where rows are kept, nothing where none
            // (but the one value that stands for the attention's output meanwhile)
            let over = most.max(new) - new;
            let allowed = if keep == 0 { 4 } else { layer(16_384) + 4 };
            assert!(most <= old.max(new + allowed), "keep {keep}: {most} at once, of {old} before and {new} after ({over} over)");
        }
    }
}

/// Room in the chain's copy of the attention cache for `needed` rows (grown to a power of two). Its first `keep` rows
/// are the ones still wanted of it (what [`sync`] will not write again: [`kept`]); with none, nothing of it is.
///
/// The old vectors go as the new ones come: what is made from the layers' rows first (the halves, the attention's
/// output), then a layer at a time, or every layer before any new one where no row is kept. Made all at once beside
/// the old ones, a cache grown from 16,384 rows to 32,768 took both caches' memory together, 6 GB of the 27B's on a
/// 24 GB card with 15.8 GB of weights on it, and the halves' gigabyte: the card had no room ("Allocation of
/// 256.00 MB failed"), and a 17,621-token prompt that came after a 15,088-token one failed, every time.
fn reserve(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, slots: usize, needed: usize, keep: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let cap = needed.next_power_of_two().max(256);
    // Nobody's copy, of no rows, until it is whole again: a device that has no room for one of the vectors below
    // ends this where it stands (its refusal is a panic the model's thread is brought back from), with some layers
    // grown and some not and no output. Left as a cache of the old size, the next prompt that fitted the old size
    // would have read past the layers that were there, and every request after failed until the engine was started
    // again. As it is, the next one makes all of it anew and writes every row.
    let owner = std::mem::take(&mut g.owner);
    g.cap = 0;
    // (and the scratch the device keeps for its next run: the room it takes is the cache's first)
    chain.let_go_of_spare();
    // (the halves' vectors are the old layers': made again, and filled, at the next step that reads them)
    g.half.clear();
    g.halved = 0;
    g.out = chain.vec(1);
    if keep == 0 {
        g.layers.clear();
    }
    g.layers.truncate(slots);
    for i in 0..slots {
        if i < g.layers.len() {
            // (the old layer's vector is let go as this one takes its place)
            g.layers[i] = chain.resize(&g.layers[i], cap * row);
        } else {
            g.layers.push(chain.vec(cap * row));
        }
    }
    g.out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
    g.cap = cap;
    g.owner = owner;
}

/// [`reserve`] for a cache held as f16 halves alone ([`State::halves_only`]): `g.half` a vector a slot of `cap * row / 2`
/// words, and no `g.layers`. Grown in steps of 4,096 rows, with as many to spare for the reply: a power of two past what
/// was needed was a 24 GB card's whole room for the 27B (65,536 rows for a 40,000-token prompt). The old vectors go as
/// the new ones come, and the copy is nobody's until it is whole, as [`reserve`]'s.
fn reserve_halves(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, slots: usize, needed: usize, keep: usize) {
    if g.cap >= needed && g.half.len() == slots && g.layers.is_empty() {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let cap = halves_cap(needed);
    let owner = std::mem::take(&mut g.owner);
    g.cap = 0;
    chain.let_go_of_spare();
    g.layers.clear();
    g.halved = 0;
    g.out = chain.vec(1);
    if keep == 0 {
        g.half.clear();
    }
    g.half.truncate(slots);
    for i in 0..slots {
        if i < g.half.len() {
            g.half[i] = chain.resize(&g.half[i], cap * row / 2);
        } else {
            g.half.push(chain.vec(cap * row / 2));
        }
    }
    g.out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
    g.cap = cap;
    g.owner = owner;
}

/// The attention cache's copy on the device let go: nobody's, of no rows (the next run makes it and writes every row).
fn drop_cache(chain: &dyn DeviceChain, g: &mut Kv) {
    g.owner = 0;
    g.cap = 0;
    g.layers.clear();
    g.half.clear();
    g.halved = 0;
    g.out = chain.vec(1);
}

/// The rows [`reserve_halves`] holds for `needed`: as many again as 4,096 past it, in steps of 4,096.
fn halves_cap(needed: usize) -> usize {
    (needed + 4096).div_ceil(4096) * 4096
}

/// 768 MB of the recordings' own scratch and reads, in vectors an adapter does not keep to hand out again once let go.
const SPARE: [usize; 2] = [(384 << 20) / 4; 2];

/// A prompt chunk's attention scratch over a cache of `cap` rows, in words.
fn scratch_words(chain: &dyn DeviceChain, s: &Dims, cap: usize) -> usize {
    let need = chain.attention_rows_out_len(2 * MAX_ROWS, s.n_h, s.hd, cap).min(ATTENTION_SCRATCH / 4);
    need + need / 2
}

/// What a prompt's run makes on the device beside an attention cache of `cap` rows, in words: the attention's
/// output, a second chunk's vectors and their attention's scratch (a chunk records while the one before runs), three
/// states kept from inside a run and their rows' vectors, and the recordings' own ([`SPARE`]).
fn beside_words(chain: &dyn DeviceChain, st: &State, cap: usize) -> usize {
    let s = st.dims;
    let (state, conv) = (s.nv * s.dv * s.dk, (s.kern - 1) * s.ch);
    chain.attention_out_len(s.n_h, s.hd, cap) + Work::words(&s, 2 * MAX_ROWS) + 2 * scratch_words(chain, &s, cap) + 3 * st.ssm_layers.len() * (state + conv) + TAP_ROWS * 2 * (s.ch + s.nv * s.dv + s.nv) + SPARE.iter().sum::<usize>()
}

/// `make`, or None where the device refused one of its vectors for want of room (its panic [`ggml_rs::chain::no_room`]
/// knows, which is not said as it is: a refusal is the answer looked for); any other panic goes on as it came.
fn if_room<T>(make: impl FnOnce() -> T) -> Option<T> {
    type Hook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync>;
    let before: Arc<Hook> = Arc::new(std::panic::take_hook());
    let said = Arc::clone(&before);
    std::panic::set_hook(Box::new(move |info| {
        if !ggml_rs::chain::no_room(info.payload()) {
            said(info);
        }
    }));
    let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(make));
    // (this hook let go, and with it its hold on the one before, which is put back)
    drop(std::panic::take_hook());
    if let Ok(hook) = Arc::try_unwrap(before) {
        std::panic::set_hook(hook);
    }
    match made {
        Ok(made) => Some(made),
        Err(panic) if ggml_rs::chain::no_room(panic.as_ref()) => None,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// [`sync`] for a cache held as halves: the host's rows from those the copy has, as f16 halves (two a word, rounded to
/// nearest, as the device's `pack2x16float` rounds them).
fn sync_halves(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, layers: &[usize], kv: &KvCache, past: usize) {
    let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
    let kvd = s.n_kv * s.hd;
    if from < past {
        for (slot, &l) in layers.iter().enumerate() {
            let (kh, vh) = (kv.k_buffer(l), kv.v_buffer(l));
            let (kown, vown);
            let kd = if kh.is_device() { kown = kh.to_host(); kown.data() } else { kh.data() };
            let vd = if vh.is_device() { vown = vh.to_host(); vown.data() } else { vh.data() };
            let mut words: Vec<f32> = Vec::with_capacity((past - from) * kvd);
            for t in from..past {
                for pair in kd[t * kvd..(t + 1) * kvd].as_chunks::<2>().0.iter().chain(vd[t * kvd..(t + 1) * kvd].as_chunks::<2>().0) {
                    let w = u32::from(half::f16::from_f32(pair[0]).to_bits()) | (u32::from(half::f16::from_f32(pair[1]).to_bits()) << 16);
                    words.push(f32::from_bits(w));
                }
            }
            chain.upload_at(&g.half[slot], from * kvd, &words);
        }
    }
    g.owner = kv.id;
}

/// A cache's rows read back as halves (two a word) as f32 values, for the host's cache.
fn from_halves(words: &[f32]) -> Vec<f32> {
    words
        .iter()
        .flat_map(|w| {
            let b = w.to_bits();
            [half::f16::from_bits(b as u16).to_f32(), half::f16::from_bits((b >> 16) as u16).to_f32()]
        })
        .collect()
}

/// The rows of the chain's copy that are still the cache's, of its first `past`: those [`sync`] leaves as they are
/// (none for a cache the copy is not of, or one begun again).
fn kept(g: &Kv, kv: &KvCache, past: usize) -> usize {
    if g.owner == kv.id {
        kv.dirty_from.min(past)
    } else {
        0
    }
}

/// Bring the chain's copy of the attention cache up to `past` rows: the rows the host wrote since (all of them for a
/// cache the copy is not of).
fn sync(chain: &dyn DeviceChain, g: &mut Kv, s: &Dims, layers: &[usize], kv: &KvCache, past: usize) {
    let from = if g.owner == kv.id { kv.dirty_from.min(past) } else { 0 };
    upload_rows(chain, g, s, layers, kv, from, past);
    g.halved = g.halved.min(from);
    g.owner = kv.id;
}

/// Rows `from..past` of the host's cache, of `layers` (a buffer of `g`'s each, in turn), into the chain's copy.
fn upload_rows(chain: &dyn DeviceChain, g: &Kv, s: &Dims, layers: &[usize], kv: &KvCache, from: usize, past: usize) {
    let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
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

/// `t`'s values in a vector of `chain`'s.
fn upload_tensor(chain: &dyn DeviceChain, t: &Tensor) -> DeviceVec {
    let t = if t.is_device() { t.to_host() } else { t.clone() };
    let v = chain.vec(t.numel());
    chain.upload(&v, t.data());
    v
}

/// Block `b`'s vectors on `chain`'s device, `slot` its attention cache's buffer or its recurrent state (whichever it
/// has).
fn layer_vecs(chain: &dyn DeviceChain, b: &Qwen35Block, slot: usize) -> LayerVecs {
    let up = |t: &Tensor| upload_tensor(chain, t);
    let router = |moe: &Option<Box<Qwen35Moe>>| moe.as_ref().map(|m| up(&m.router.to_host()));
    match b {
        Qwen35Block::Attention { attn_norm, attn_q_norm, attn_k_norm, post_norm, moe, .. } => {
            LayerVecs { attn_norm: up(attn_norm), post_norm: up(post_norm), mixer: Mixer::Attention { q_norm: up(attn_q_norm), k_norm: up(attn_k_norm), slot }, router: router(moe) }
        }
        Qwen35Block::Conv { attn_norm, conv, post_norm, moe, .. } => {
            LayerVecs { attn_norm: up(attn_norm), post_norm: up(post_norm), mixer: Mixer::Conv { conv_w: up(conv), slot }, router: router(moe) }
        }
        Qwen35Block::Ssm { attn_norm, ssm_conv1d, ssm_a, ssm_ba, ssm_dt_bias, ssm_norm, post_norm, moe, .. } => {
            let ba_f32 = match ssm_ba {
                Weight::Dense(t) => Some(up(t)),
                _ => None,
            };
            LayerVecs {
                attn_norm: up(attn_norm),
                post_norm: up(post_norm),
                mixer: Mixer::Ssm { conv_w: up(ssm_conv1d), a: up(ssm_a), dt: up(ssm_dt_bias), norm: up(ssm_norm), ba_f32, slot },
                router: router(moe),
            }
        }
    }
}

/// What a run's layers use besides their weights and vectors: the attention cache's copy (a buffer a slot, room for
/// `cap` rows), each delta net's state and conv window (by slot), the run's vectors, and its attention's output and
/// scratch.
struct Bound<'a> {
    kvl: &'a [DeviceVec],
    /// The layers' f16 halves and how many of their rows are the layers' so far, where a step reads them (or a run
    /// of few rows, a row at a time: `one`, a step's query vector and its attention's output).
    half: Option<(&'a [DeviceVec], usize)>,
    one: Option<(&'a DeviceVec, &'a DeviceVec)>,
    /// The cache held as `half` alone ([`State::halves_only`]): the rows stored there, and every attention reads it.
    halves_only: bool,
    cap: usize,
    states: &'a [(DeviceVec, DeviceVec)],
    w: &'a Work,
    attn: &'a DeviceVec,
    /// The states kept from inside the run (by row, each once), and the vectors its rows after the first of them go
    /// through.
    taps: &'a [Tap],
    seg: Option<&'a Seg>,
}

/// The most rows of a run after the first state it keeps from inside ([`Tap`]): those rows' recurrence goes through
/// vectors of its own this long (a prompt's tail: the assistant's header and its last token).
const TAP_ROWS: usize = 64;

/// A state kept from inside a run: each delta net's state and conv window as they are once the run's first `row` rows
/// are through it (a pair a slot), copied there as the run goes.
struct Tap {
    row: usize,
    vecs: Vec<(DeviceVec, DeviceVec)>,
}

/// The vectors a tapped run's later rows' recurrence goes through ([`TAP_ROWS`] of them, a set a chain: each layer's
/// in turn): the conv's input and output, the gate, the beta-alpha and the delta net's output.
struct Seg {
    qkv: DeviceVec,
    z: DeviceVec,
    ba: DeviceVec,
    conv: DeviceVec,
    core: DeviceVec,
}

impl Seg {
    fn new(chain: &dyn DeviceChain, s: &Dims) -> Self {
        Seg { qkv: chain.vec(TAP_ROWS * s.ch.max(s.bcx)), z: chain.vec(TAP_ROWS * s.nv * s.dv), ba: chain.vec(TAP_ROWS * 2 * s.nv), conv: chain.vec(TAP_ROWS * s.ch), core: chain.vec(TAP_ROWS * s.nv * s.dv) }
    }
}

/// A run's recurrent states as they were with `at` rows in the cache ([`Qwen35Chain::forward_tapped`]): each layer's
/// delta-net state and conv window (none for an attention layer), on the device, as [`KvCache`] holds its own.
pub struct Tapped {
    pub at: usize,
    pub states: Vec<Option<Tensor>>,
    pub convs: Vec<Option<Tensor>>,
}

/// `layers` of a run of `t` rows at `past`, each its weights and vectors; `added` whether a residual's add is pending
/// (fused with the norm after it), as it is after them. With `check` (a check of drafts) each delta net's state and
/// window before it and its inputs are kept, what a rollback starts over from.
#[allow(clippy::too_many_arguments)]
fn record_layers<'w>(rec: &mut dyn ggml_rs::ChainRecorder, s: &Dims, eps: f32, layers: impl Iterator<Item = (LayerW<'w>, &'w LayerVecs)>, b: &Bound<'_>, t: usize, past: usize, check: Option<&Spec>, added: &mut bool) {
    let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
    let scale = 1.0 / (s.hd as f32).sqrt();
    let delta = DeltaNet { rows: t, v_heads: s.nv, k_heads: s.nk, k_dim: s.dk, v_dim: s.dv, scale_q: 1.0 / (s.dv as f32).sqrt(), eps, sigmoid_gate: false };
    let w = b.w;
    for (lw, lv) in layers {
        if *added {
            rec.add_rmsnorm_rows(&w.x, &w.proj, &lv.attn_norm, &w.xn, t, eps);
        } else {
            rec.rmsnorm_rows(&w.x, &lv.attn_norm, &w.xn, t, eps);
        }
        let (ffn, down, moe) = match (lw, &lv.mixer) {
            (LayerW::Attention { q, k, v, o, ffn, down, moe }, Mixer::Attention { q_norm, k_norm, slot }) => {
                // each head's q is its query then its gate (LFM2's its query alone)
                q.rows(rec, &w.xn, if s.gated { &w.qfull } else { &w.q }, t);
                k.rows(rec, &w.xn, &w.k, t);
                v.rows(rec, &w.xn, &w.v, t);
                if s.gated {
                    rec.copy_cols(&w.qfull, &w.q, t * s.n_h, s.hd, 2 * s.hd, 0);
                    rec.copy_cols(&w.qfull, &w.gate, t * s.n_h, s.hd, 2 * s.hd, s.hd);
                }
                rec.rmsnorm_rows(&w.q, q_norm, &w.qn, t * s.n_h, eps);
                rec.rmsnorm_rows(&w.k, k_norm, &w.kn, t * s.n_kv, eps);
                rec.rope_partial_rows(&w.qn, t, s.n_h, s.hd, s.rot, &w.table);
                rec.rope_partial_rows(&w.kn, t, s.n_kv, s.hd, s.rot, &w.table);
                if b.halves_only {
                    // the cache's halves alone: the rows stored as halves; a step's attention over them, a prompt's on
                    // the tensor cores, and a few rows' (fewer than they take) each a step's
                    let half = &b.half.expect("a cache held as halves").0[*slot];
                    rec.store_rows_halved(&w.kn, half, t, kvd, past, row, 0);
                    rec.store_rows_halved(&w.v, half, t, kvd, past, row, kvd);
                    if t == 1 {
                        rec.attention_halved(&w.qn, half, b.attn, s.n_h, s.n_kv, s.hd, 0, past + 1, b.cap, scale);
                    } else if !(t > SPEC_ROWS && rec.attention_rows_halved(&w.qn, half, b.attn, t, s.n_h, s.n_kv, s.hd, past, scale)) {
                        let (q1, o1) = b.one.expect("a step's query and output vectors");
                        let qd = s.n_h * s.hd;
                        for r in 0..t {
                            rec.copy(&w.qn, r * qd, q1, 0, qd);
                            rec.attention_halved(q1, half, o1, s.n_h, s.n_kv, s.hd, 0, past + r + 1, b.cap, scale);
                            rec.copy(o1, 0, b.attn, r * qd, qd);
                        }
                    }
                    gate_out(rec, s, b.attn, w, o, t);
                    (ffn, down, moe)
                } else {
                let kvl = &b.kvl[*slot];
                rec.store_rows(&w.kn, kvl, t, kvd, past, row, 0);
                rec.store_rows(&w.v, kvl, t, kvd, past, row, kvd);
                if let (1, Some((half, done))) = (t, b.half) {
                    // the rows since the halves were last brought up (this step's, and a prompt's before it)
                    rec.halve(kvl, &half[*slot], done * row, (past + 1 - done) * row);
                    rec.attention_halved(&w.qn, &half[*slot], b.attn, s.n_h, s.n_kv, s.hd, 0, past + 1, b.cap, scale);
                } else if let (Some((half, done)), Some((q1, o1))) = (b.half, b.one) {
                    // a few rows deep in a long cache (a check of drafted tokens): each row's attention a step's, the
                    // halves brought up to the run's last row and each row's keys those up to its own position (the
                    // rows' own kernel took 2.6 ms a row of the sixteen layers at 12,800 positions where a step's 1.3)
                    rec.halve(kvl, &half[*slot], done * row, (past + t - done) * row);
                    let qd = s.n_h * s.hd;
                    for r in 0..t {
                        rec.copy(&w.qn, r * qd, q1, 0, qd);
                        rec.attention_halved(q1, &half[*slot], o1, s.n_h, s.n_kv, s.hd, 0, past + r + 1, b.cap, scale);
                        rec.copy(o1, 0, b.attn, r * qd, qd);
                    }
                } else if t == 1 {
                    rec.attention(&w.qn, kvl, b.attn, s.n_h, s.n_kv, s.hd, 0, past + 1, b.cap, scale);
                } else {
                    rec.attention_rows(&w.qn, kvl, b.attn, t, s.n_h, s.n_kv, s.hd, past, None, scale);
                }
                gate_out(rec, s, b.attn, w, o, t);
                (ffn, down, moe)
                }
            }
            (LayerW::Conv { in_proj, out, ffn, down, moe }, Mixer::Conv { conv_w, slot }) => {
                let (state, conv) = &b.states[*slot];
                debug_assert!(check.is_none(), "a model of short convs drafts nothing");
                in_proj.rows(rec, &w.xn, &w.bcx, t);
                if b.taps.is_empty() {
                    rec.short_conv(&w.bcx, conv_w, conv, &w.conv, t, s.d, s.kern);
                } else {
                    // in parts, as a delta net's: the first where the rows are, the later ones through the parts' own
                    // vectors and back, each kept state copied (the delta-net state a placeholder, copied as it is)
                    let mut done = 0;
                    for (end, keep) in b.taps.iter().map(|tap| (tap.row, Some(&tap.vecs[*slot]))).chain(std::iter::once((t, None))) {
                        let n = end - done;
                        if n > 0 && done == 0 {
                            rec.short_conv(&w.bcx, conv_w, conv, &w.conv, n, s.d, s.kern);
                        } else if n > 0 {
                            let seg = b.seg.expect("a tapped run's later rows' vectors");
                            rec.copy(&w.bcx, done * s.bcx, &seg.qkv, 0, n * s.bcx);
                            rec.short_conv(&seg.qkv, conv_w, conv, &seg.conv, n, s.d, s.kern);
                            rec.copy(&seg.conv, 0, &w.conv, done * s.d, n * s.d);
                        }
                        if let Some((sv, cv)) = keep {
                            rec.copy(state, 0, sv, 0, state.len);
                            rec.copy(conv, 0, cv, 0, conv.len);
                        }
                        done = end;
                    }
                }
                out.rows(rec, &w.conv, &w.proj, t);
                (ffn, down, moe)
            }
            (LayerW::Ssm { qkv, gate, ba, out, ffn, down, moe }, Mixer::Ssm { conv_w, a, dt, norm, ba_f32, slot }) => {
                let (state, conv) = &b.states[*slot];
                qkv.rows(rec, &w.xn, &w.qkv, t);
                gate.rows(rec, &w.xn, &w.z, t);
                match (ba_f32, ba) {
                    (Some(wf), _) => rec.matmul_f32_rows(wf, 2 * s.nv, s.d, &w.xn, &w.ba, t),
                    (None, Some(q)) => q.rows(rec, &w.xn, &w.ba, t),
                    (None, None) => unreachable!("a delta net's beta-alpha projection is f32 or quantized"),
                }
                if let Some(sp) = check {
                    // what a rollback starts over from: the state and window before, the rows' inputs
                    let (bs, bc) = &sp.backups[*slot];
                    let (qkv, ba) = &sp.inputs[*slot];
                    rec.copy(state, 0, bs, 0, state.len);
                    rec.copy(conv, 0, bc, 0, conv.len);
                    rec.copy(&w.qkv, 0, qkv, 0, t * s.ch);
                    rec.copy(&w.ba, 0, ba, 0, t * 2 * s.nv);
                }
                if b.taps.is_empty() {
                    rec.ssm_conv(&w.qkv, conv_w, conv, &w.conv, t, s.ch, s.kern);
                    rec.delta_net(&w.conv, &w.z, &w.ba, a, dt, norm, state, &w.core, delta);
                } else {
                    // the recurrence in parts, each kept state's rows and then its copies: the first part where the
                    // rows are, the later ones (a prompt's tail: few) through the parts' own vectors and back
                    let vw = s.nv * s.dv;
                    let mut done = 0;
                    for (end, keep) in b.taps.iter().map(|tap| (tap.row, Some(&tap.vecs[*slot]))).chain(std::iter::once((t, None))) {
                        let n = end - done;
                        if n > 0 && done == 0 {
                            rec.ssm_conv(&w.qkv, conv_w, conv, &w.conv, n, s.ch, s.kern);
                            rec.delta_net(&w.conv, &w.z, &w.ba, a, dt, norm, state, &w.core, DeltaNet { rows: n, ..delta });
                        } else if n > 0 {
                            let seg = b.seg.expect("a tapped run's later rows' vectors");
                            rec.copy(&w.qkv, done * s.ch, &seg.qkv, 0, n * s.ch);
                            rec.copy(&w.z, done * vw, &seg.z, 0, n * vw);
                            rec.copy(&w.ba, done * 2 * s.nv, &seg.ba, 0, n * 2 * s.nv);
                            rec.ssm_conv(&seg.qkv, conv_w, conv, &seg.conv, n, s.ch, s.kern);
                            rec.delta_net(&seg.conv, &seg.z, &seg.ba, a, dt, norm, state, &seg.core, DeltaNet { rows: n, ..delta });
                            rec.copy(&seg.core, 0, &w.core, done * vw, n * vw);
                        }
                        if let Some((sv, cv)) = keep {
                            rec.copy(state, 0, sv, 0, state.len);
                            rec.copy(conv, 0, cv, 0, conv.len);
                        }
                        done = end;
                    }
                }
                out.rows(rec, &w.core, &w.proj, t);
                (ffn, down, moe)
            }
            _ => unreachable!("a layer's vectors are its block's"),
        };
        rec.add_rmsnorm_rows(&w.x, &w.proj, &lv.post_norm, &w.xn, t, eps);
        match (moe, &lv.router) {
            // a MoE layer: its router's logits (f32), then its experts routed and summed on the device, the shared one
            // among them, into the FFN's output
            (Some(m), Some(router)) => {
                rec.matmul_f32_rows(router, s.router, s.d, &w.xn, &w.router, t);
                let routed = rec.moe_routed(&*m.experts, &w.xn, &w.proj, &w.router, m.top_k, t);
                assert!(routed, "a chained MoE layer's experts route its {t} rows on the device");
            }
            _ => {
                match ffn {
                    FfnW::Fused(gu) => {
                        gu.rows(rec, &w.xn, &w.ffa, t);
                        rec.silu_mul_split_rows(&w.ffa, &w.act, t);
                    }
                    FfnW::Split(gate, up) => {
                        gate.rows(rec, &w.xn, &w.ffa, t);
                        up.rows(rec, &w.xn, &w.ffb, t);
                        rec.silu_mul(&w.ffa, &w.ffb, &w.act, t * s.ff);
                    }
                }
                down.rows(rec, &w.act, &w.proj, t);
            }
        }
        *added = true;
    }
}

/// An attention layer's output (`attn`'s first rows) through its gate's sigmoid where its q has one, then its output
/// projection into `w.proj`.
fn gate_out(rec: &mut dyn ggml_rs::ChainRecorder, s: &Dims, attn: &DeviceVec, w: &Work, o: W<'_>, t: usize) {
    if s.gated {
        rec.mul_sigmoid(attn, &w.gate, &w.gated, t * s.n_h * s.hd);
        o.rows(rec, &w.gated, &w.proj, t);
    } else {
        o.rows(rec, &first(attn, t * s.n_h * s.hd), &w.proj, t);
    }
}

/// A run's K and V rows of layer `l` (`t` of them from `at`, as the chain's copy holds them) into the host's cache.
fn cache_rows(backend: &dyn Backend, kv: &mut KvCache, l: usize, at: usize, t: usize, rows: &[f32]) {
    let len = kv.len;
    kv.len = at;
    kv.append_rows(backend, l, rows, t);
    kv.len = len;
}

/// A chained run gone to the GPU: its recording (its reads the logits, then each attention layer's rows), and where
/// those rows go in the host's cache.
pub(crate) struct Qwen35Run<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    layers: Vec<usize>,
    past: usize,
    t: usize,
    vocab: usize,
    /// A prompt chunk's pooled vectors, back to the pool once the run is done
    set: Option<(&'a Mutex<Vec<(Work, DeviceVec)>>, (Work, DeviceVec))>,
    /// The states it keeps from inside (theirs once the run has run: read after it, as anything of the device's)
    taps: Vec<Tapped>,
    /// Its rows read back as halves (the cache held as halves alone).
    halves: bool,
}

impl Qwen35Run<'_> {
    /// Waited for: each row's logits (a check's every row's, else the last's), its K and V rows into the host's
    /// cache, which the device's copy already holds.
    fn finish(self, backend: &dyn Backend, kv: &mut KvCache) -> Vec<Tensor> {
        let mut got = self.rec.finish().into_iter();
        if let Some((pool, set)) = self.set {
            pool.lock().unwrap_or_else(|p| p.into_inner()).push(set);
        }
        let logits = got.next().expect("the logits");
        let logits: Vec<Tensor> = logits.chunks_exact(self.vocab).map(|l| Tensor::from_vec(l.to_vec(), vec![1, self.vocab])).collect();
        let len = kv.len;
        kv.len = self.past;
        for (&l, rows) in self.layers.iter().zip(got) {
            let rows = if self.halves { from_halves(&rows) } else { rows };
            kv.append_rows(backend, l, &rows, self.t);
        }
        kv.len = len;
        kv.dirty_from = usize::MAX;
        logits
    }
}

impl Qwen35Chain {
    /// VENDORED-LOCAL: whether its device holds the attention cache as f16 halves alone (`State::halves_only`), where
    /// it runs a chain.
    pub fn cache_in_halves(&self, m: &Qwen35Model) -> bool {
        self.state(m).is_some_and(|st| st.halves_only)
    }

    /// The most positions, `want` at most, whose attention cache the model's device holds beside what a prompt's run
    /// makes with it, found by making them where that is how a device says (one that refuses a vector as it is asked
    /// for: the card's adapter). Beside the cache a prompt's run has a second chunk's vectors and their attention's
    /// scratch (a chunk records while the one before runs), three states kept from inside a run and their rows'
    /// vectors, and 768 MB of the recordings' own scratch and reads: their room is taken first, then the cache 4,096
    /// positions at a time until the device refuses (once, where trying whole caches of this size and that was a
    /// dozen refusals, each a traceback in the card's log) or `want` has its room. What that says fits is then made
    /// as a run would make it, each layer's vector whole, and a device that refuses that is asked for 4,096 positions
    /// fewer. All of it is let go after, and the cache the warm-up made before (the next run makes its own
    /// and writes every row). None where the device cannot say, a second device takes a prompt's later layers, or the
    /// model runs no chain. The 27B on a 24 GB card: a 79,273-token prompt left the card 651 MB, and one of 91,093
    /// failed halfway for want of room; this finds 77,824 positions, and a prompt of 77,630 leaves 427 MB (whole
    /// vectors of 128 MB do not take the small gaps a run's own vectors do, so it stops a step short of the most).
    pub fn context_room(&self, m: &Qwen35Model, want: usize) -> Option<usize> {
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        if !chain.refuses_at_once() || self.second.get().is_some() || want == 0 {
            return None;
        }
        let s = st.dims;
        let row = 2 * s.n_kv * s.hd;
        let slots = st.attention_layers.len();
        drop_cache(chain, &mut st.kv.lock().unwrap_or_else(|p| p.into_inner()));
        chain.let_go_of_spare();
        // (a layer's words a position, and the rows a cache of `positions` is made with: `reserve_halves`', `reserve`'s)
        let words = if st.halves_only { row / 2 } else { row };
        let cap_of = |positions: usize| if st.halves_only { halves_cap(positions) } else { positions.next_power_of_two().max(256) };
        let (state, conv) = (s.nv * s.dv * s.dk, (s.kern - 1) * s.ch);
        let scratch_of = |cap: usize| scratch_words(chain, &s, cap);
        // what a prompt's run makes beside a cache of `cap` rows, and the words of it
        let beside = |cap: usize| {
            let out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
            let set = Work::new(chain, &s, 2 * MAX_ROWS);
            let scratch: Vec<DeviceVec> = (0..2).map(|_| chain.vec(scratch_of(cap))).collect();
            let kept: Vec<DeviceVec> = (0..3).flat_map(|_| st.ssm_layers.iter().flat_map(|_| [chain.vec(state), chain.vec(conv)])).collect();
            let seg = Seg::new(chain, &s);
            let spare: Vec<DeviceVec> = SPARE.iter().map(|&n| chain.vec(n)).collect();
            (out, set, scratch, kept, seg, spare)
        };
        let beside_words = |cap: usize| beside_words(chain, st, cap);
        // The cache's room, measured: as many words as those in vectors of 128 MB or more (which no adapter keeps to
        // hand out again: small ones kept so, where a run's own lay, left the whole vectors of the cache no place
        // between them), then the cache 4,096 rows at a time (or as many of those as make 128 MB), every layer's rows
        // in one vector a step, until the device refuses or the cache `want` is made with has its room.
        let large = (128usize << 20) / 4;
        let step = 4096 * large.div_ceil(slots.max(1) * 4096 * words).max(1);
        let most = cap_of(want).div_ceil(step);
        let mut steps = 0usize;
        let mut held: Vec<DeviceVec> = Vec::new();
        let all = if_room(|| {
            let mut left = beside_words(cap_of(want));
            while left > 0 {
                let n = if left < 2 * large { left } else { large };
                held.push(chain.vec_once(n));
                left -= n;
            }
            for _ in 0..most {
                held.push(chain.vec_once(slots * step * words));
                steps += 1;
            }
        });
        let fitted = all.is_some();
        drop(held);
        chain.let_go_of_spare();
        if fitted {
            return Some(want);
        }
        // the positions a cache of that many rows serves (what the reservation adds past them taken off), and that
        // cache as a run makes it; fewer where the device has not that in whole vectors
        let rows = steps * step;
        let mut room = if st.halves_only { rows.saturating_sub(4096) } else if rows >= 256 { 1usize << rows.ilog2() } else { 0 };
        while room > 0 {
            let cap = cap_of(room);
            // (asked for as a run asks, a second time where refused: the card's server gives back what was just let
            // go a request late, and a vector of a new size straight after is refused once for that alone)
            let made = if_room(|| ((0..slots).map(|_| chain.vec(cap * words)).collect::<Vec<_>>(), beside(cap)));
            let fits = made.is_some();
            drop(made);
            chain.let_go_of_spare();
            if fits {
                break;
            }
            room = if st.halves_only { room.saturating_sub(4096) } else { room / 2 };
        }
        Some(room.min(want))
    }

    /// Whether the model's device has room for its attention cache at `want` positions beside what a prompt's run
    /// makes with it (what [`Self::context_room`] makes vectors of), by the device's own figure for its memory, before
    /// any of it is made: a device of wgpu's own does not refuse a vector it has no room for, it pages it, and every
    /// run after slows to a crawl. The cache's rows as a run makes them: the halves alone, or the f32 rows and the
    /// halves a step deep in the cache reads beside them, at the rows the cache is grown to for `want`. What the
    /// device holds of a cache now (the warm-up's) is room the larger one takes; `spare` bytes are left beside it (what
    /// a run makes that this does not count). None where the device says nothing of its memory, refuses a vector as it
    /// is asked for (measured so by [`Self::context_room`]), a second device takes a prompt's later layers, or the
    /// model runs no chain.
    pub fn context_fits(&self, m: &Qwen35Model, want: usize, spare: u64) -> Option<bool> {
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        if chain.refuses_at_once() || !chain.room_known() || self.second.get().is_some() || want == 0 {
            return None;
        }
        let s = st.dims;
        let row = 2 * s.n_kv * s.hd;
        let (cap, words) = if st.halves_only { (halves_cap(want), row / 2) } else { (want.next_power_of_two().max(256), row + row / 2) };
        let held: usize = {
            let g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
            g.layers.iter().chain(&g.half).map(|v| v.len).sum::<usize>() + g.out.len
        };
        let need = st.attention_layers.len() * cap * words + beside_words(chain, st, cap);
        Some(chain.has_room((need.saturating_sub(held) as u64) * 4 + spare))
    }

    /// What the chain keeps on its device between runs let go: its prompt chunks' pooled vectors and their scratch,
    /// and with `cache` its attention cache's copy too (the next run makes it again and writes every row), then the
    /// device's own spare. `cache`: after a run the device refused room for, which a run of the same prompt may then
    /// have; without it, after a picture's prompt, whose run keeps vectors of its own beside a text prompt's (1.4 GB
    /// of them on the 27B at 61,000 positions, and a text prompt after it had then no room for its own).
    pub fn let_go_of_spare(&self, m: &Qwen35Model, cache: bool) {
        let (Some(Some(st)), Some(chain)) = (self.state.get(), m.backend.chain()) else { return };
        st.prompt_sets.lock().unwrap_or_else(|p| p.into_inner()).clear();
        if cache {
            drop_cache(chain, &mut st.kv.lock().unwrap_or_else(|p| p.into_inner()));
        }
        chain.let_go_of_spare();
    }

    fn state(&self, m: &Qwen35Model) -> Option<&State> {
        self.state
            .get_or_init(|| {
                let chain = m.backend.chain()?;
                let cfg = &m.config;
                let sc = &m.ssm_cfg;
                let held = |w: &Weight| W::held(chain, w);
                // (an LFM2 model's recurrent layers are short convs: no delta net, whose sizes are then placeholders of
                // one, its states' too; its attention layers' q has no gate)
                let lfm2 = m.blocks.iter().any(|b| matches!(b, Qwen35Block::Conv { .. }));
                let (nv, nk, dk) = if lfm2 { (1, 1, 1) } else { (sc.time_step_rank, sc.group_count, sc.state_size) };
                let dv = if lfm2 { 1 } else if nv == 0 { 0 } else { sc.inner_size / nv };
                let (n_h, n_kv, hd, d) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.embedding_dim);
                if (!lfm2 && (nv == 0 || nk == 0 || dk != dv || ![16, 32, 64, 128].contains(&dk))) || !(2..=8).contains(&sc.conv_kernel) {
                    return None;
                }
                if lfm2 && m.blocks.iter().any(|b| matches!(b, Qwen35Block::Ssm { .. })) {
                    return None;
                }
                let gated = !lfm2;
                if n_kv == 0 || n_h % n_kv != 0 || cfg.rope_dim == 0 || cfg.rope_dim > hd || cfg.rope_dim % 2 != 0 || !held(&m.output) {
                    return None;
                }
                let ch = if lfm2 { d } else { 2 * nk * dk + nv * dv };
                // (the dense FFNs' width: a MoE model's first layer's where every layer has experts, its shared
                // expert's)
                let has_moe = |b: &Qwen35Block| match b {
                    Qwen35Block::Attention { moe, .. } | Qwen35Block::Ssm { moe, .. } | Qwen35Block::Conv { moe, .. } => moe.is_some(),
                };
                let ff = m.blocks.iter().find(|b| !has_moe(b)).or(m.blocks.first()).map(|b| match b {
                    Qwen35Block::Attention { ffn_pair, .. } | Qwen35Block::Ssm { ffn_pair, .. } | Qwen35Block::Conv { ffn_pair, .. } => ffn_pair.ff(),
                })?;
                let pair_held = |p: &FfnPair| match p {
                    FfnPair::Fused(w) => held(w),
                    FfnPair::Split { gate, up } => held(gate) && held(up),
                };
                // (a MoE layer's experts on this device, routed on it: none on the host, none read from the host's
                // memory as they run)
                let moe_held = |moe: &Option<Box<Qwen35Moe>>| {
                    moe.as_ref().is_none_or(|m| chain.holds_experts(&*m.experts) && !m.experts.on_host() && !m.experts.part_held() && !m.experts.host_rest() && m.router.shape() == [m.router.dim(0), d])
                };
                let routers: Vec<usize> = m.blocks.iter().filter_map(|b| match b {
                    Qwen35Block::Attention { moe, .. } | Qwen35Block::Ssm { moe, .. } | Qwen35Block::Conv { moe, .. } => moe.as_ref().map(|m| m.router.dim(0)),
                }).collect();
                let router = routers.first().copied().unwrap_or(0);
                // (LFM2's first layers dense, the rest with experts; Qwen's every layer one or the other)
                if routers.iter().any(|&r| r != router) || (router > 0 && !lfm2 && routers.len() != m.blocks.len()) {
                    return None;
                }
                // a layer's FFN: dense and held, or (a MoE layer's of no shared expert) none
                let ffn_ok = |p: &FfnPair, down: &Weight, moe: &Option<Box<Qwen35Moe>>| (moe.is_some() && is_no_ffn(down)) || (held(down) && pair_held(p) && p.ff() == ff);
                for b in &m.blocks {
                    let ok = match b {
                        Qwen35Block::Attention { moe, .. } | Qwen35Block::Ssm { moe, .. } | Qwen35Block::Conv { moe, .. } if !moe_held(moe) => false,
                        Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, moe, .. } => {
                            [attn_q, attn_k, attn_v, attn_output].into_iter().all(held)
                                && attn_q.shape()[0] == if gated { 2 } else { 1 } * n_h * hd
                                && ffn_ok(ffn_pair, ffn_down, moe)
                        }
                        Qwen35Block::Conv { in_proj, out_proj, conv, ffn_pair, ffn_down, moe, .. } => {
                            [in_proj, out_proj].into_iter().all(held) && in_proj.shape()[0] == 3 * d && conv.numel() == d * sc.conv_kernel && ffn_ok(ffn_pair, ffn_down, moe)
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
                let upload = |t: &Tensor| upload_tensor(chain, t);
                let dims = Dims { d, n_h, n_kv, hd, rot: cfg.rope_dim, ff, nv, nk, dk, dv, ch, kern: sc.conv_kernel, vocab: cfg.vocab_size, router, moe_rows: if router > 0 { m.moe_rows } else { 0 }, gated, bcx: if lfm2 { 3 * d } else { 0 } };
                let (mut attention_layers, mut ssm_layers) = (Vec::new(), Vec::new());
                let layers = m
                    .blocks
                    .iter()
                    .enumerate()
                    .map(|(l, b)| {
                        let slot = match b {
                            Qwen35Block::Attention { .. } => {
                                attention_layers.push(l);
                                attention_layers.len() - 1
                            }
                            Qwen35Block::Ssm { .. } | Qwen35Block::Conv { .. } => {
                                ssm_layers.push(l);
                                ssm_layers.len() - 1
                            }
                        };
                        layer_vecs(chain, b, slot)
                    })
                    .collect();
                let pool = Pool {
                    states: ssm_layers.iter().map(|_| chain.vec(nv * dk * dv)).collect(),
                    convs: ssm_layers.iter().map(|_| chain.vec((sc.conv_kernel - 1) * ch)).collect(),
                };
                let spec = m.mtp.as_ref().and_then(|mtp| {
                    let Qwen35Block::Attention { attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v, attn_output, post_norm, ffn_pair, ffn_down, .. } = &mtp.block else { return None };
                    let ok = [&mtp.eh_proj, attn_q, attn_k, attn_v, attn_output, ffn_down].into_iter().all(held) && pair_held(ffn_pair) && ffn_pair.ff() == ff && attn_q.shape()[0] == 2 * n_h * hd;
                    if !ok {
                        return None;
                    }
                    let r = SPEC_ROWS;
                    let v = |n: usize| chain.vec(n);
                    let work = MtpWork {
                        e: v(r * d),
                        h: v(r * d),
                        en: v(r * d),
                        hn: v(r * d),
                        cat: v(r * 2 * d),
                        x: v(r * d),
                        xn: v(r * d),
                        qfull: v(r * n_h * 2 * hd),
                        q: v(r * n_h * hd),
                        gate: v(r * n_h * hd),
                        k: v(r * n_kv * hd),
                        v: v(r * n_kv * hd),
                        qn: v(r * n_h * hd),
                        kn: v(r * n_kv * hd),
                        q1: v(n_h * hd),
                        att: v(r * n_h * hd),
                        gated: v(r * n_h * hd),
                        proj: v(r * d),
                        ffa: v(r * 2 * ff),
                        ffb: v(r * ff),
                        act: v(r * ff),
                        table: v(r * cfg.rope_dim),
                        last: v(d),
                        logits: v(cfg.vocab_size),
                        best: v(3),
                    };
                    Some(Spec {
                        enorm: upload(&mtp.enorm),
                        hnorm: upload(&mtp.hnorm),
                        head_norm: upload(&mtp.head_norm),
                        attn_norm: upload(attn_norm),
                        post_norm: upload(post_norm),
                        q_norm: upload(attn_q_norm),
                        k_norm: upload(attn_k_norm),
                        kv: Mutex::new(MtpKv { layer: chain.vec(1), cap: 0, out: chain.vec(1), start: 0, valid: 0, owner: 0 }),
                        work,
                        hid: v(r * d),
                        hid_at: Mutex::new((0, 0)),
                        logits: v(r * cfg.vocab_size),
                        backups: ssm_layers.iter().map(|_| (v(nv * dk * dv), v((sc.conv_kernel - 1) * ch))).collect(),
                        inputs: ssm_layers.iter().map(|_| (v(r * ch), v(r * 2 * nv))).collect(),
                        scratch_conv: v(r * ch),
                        scratch_core: v(r * nv * dv),
                    })
                });
                let halves_only = spec.is_none() && chain.halves_only(dims.n_h, dims.n_kv, dims.hd);
                let kv = Arc::new(Mutex::new(Kv::new(chain.vec(1))));
                let slots = (0..m.blocks.len()).map(|l| attention_layers.iter().position(|&a| a == l)).collect();
                let rows = Arc::new(ChainRows { kv: Arc::clone(&kv), backend: Arc::clone(&m.backend), slots, n_kv: dims.n_kv, hd: dims.hd, halves_only });
                Some(State {
                    dims,
                    layers,
                    attention_layers,
                    ssm_layers,
                    output_norm: upload(&m.output_norm),
                    kv,
                    rows,
                    pool: Mutex::new(pool),
                    step: Work::new(chain, &dims, 1),
                    logits: chain.vec(cfg.vocab_size),
                    halves_only,
                    spec,
                    prompt_sets: Mutex::new(Vec::new()),
                    seg: Default::default(),
                })
            })
            .as_ref()
    }

    /// Run `rows` tokens (`embeds` `[rows, d]`, on the host) after what `kv` holds, chained, if the backend can: the
    /// last token's logits `[1, vocab]`; None leaves them to the model's own path. A run of more rows than a submit
    /// takes goes in chunks.
    pub(crate) fn forward(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache) -> Option<Tensor> {
        self.forward_with(m, embeds, rows, kv, &[]).map(|(logits, _)| logits)
    }

    /// [`Self::forward`] for rows at `positions` (one a row, in three axes: a picture's tokens are placed by their row
    /// and column, and the text after a picture counts on from fewer positions than the picture has tokens), on the
    /// model's own device: op by op, a prompt with a picture was read at 7 tokens/s and answered at 6. None where
    /// the model's rotation is not Qwen's 64 (the axes' layout is its own).
    pub(crate) fn forward_positions(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache, positions: &[[u32; 3]]) -> Option<Tensor> {
        if m.config.rope_dim != 64 || positions.len() != rows {
            return None;
        }
        *self.positions.lock().unwrap_or_else(|p| p.into_inner()) = Some((kv.len, positions.to_vec()));
        let out = self.forward_with(m, embeds, rows, kv, &[]).map(|(logits, _)| logits);
        *self.positions.lock().unwrap_or_else(|p| p.into_inner()) = None;
        out
    }

    /// The RoPE table of `rows` rows after `past`: at the positions a run was given ([`Self::forward_positions`]),
    /// else at `past..past + rows`.
    fn table(&self, theta: f32, rot: usize, past: usize, rows: usize) -> Vec<f32> {
        match &*self.positions.lock().unwrap_or_else(|p| p.into_inner()) {
            Some((base, at)) if past >= *base && past - base + rows <= at.len() => rope_table_at(theta, rot, &at[past - base..past - base + rows]),
            _ => rope_table(theta, rot, past, rows),
        }
    }

    /// [`Self::forward`] with the delta nets' states and conv windows as they are once each of `taps` rows of the run
    /// is in (ascending, 1 to `rows`), on the device: what a checkpoint at those positions holds, without the run
    /// stopping there (a prompt's last two, before the assistant's header and before its last token, were a run of 6
    /// rows and a run of 1 after the prompt's: 44 ms of a 3,158-token prompt's 1.32 s, 48 of a follow-up turn's 105).
    /// None, and nothing run, where [`Self::can_tap`] is false.
    pub(crate) fn forward_tapped(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache, taps: &[usize]) -> Option<(Tensor, Vec<Tapped>)> {
        if embeds.numel() != rows * m.config.embedding_dim || !self.can_tap(m, rows, kv, taps) {
            return None;
        }
        self.forward_with(m, embeds, rows, kv, taps)
    }

    /// [`Self::forward_tapped`] (with no `taps`: [`Self::forward`]) for rows at `positions`, as
    /// [`Self::forward_positions`] places them: the text after a picture, in a conversation whose picture the cache
    /// holds already, keeps its states from inside the run as any text does (it went op by op through the run's ends
    /// before: a turn's 772 new tokens in 3.1 s, where 447 took 0.7). None, and nothing run, where it cannot.
    pub(crate) fn forward_tapped_at(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache, taps: &[usize], positions: &[[u32; 3]]) -> Option<(Tensor, Vec<Tapped>)> {
        if m.config.rope_dim != 64 || positions.len() != rows || embeds.numel() != rows * m.config.embedding_dim || (!taps.is_empty() && !self.can_tap(m, rows, kv, taps)) {
            return None;
        }
        *self.positions.lock().unwrap_or_else(|p| p.into_inner()) = Some((kv.len, positions.to_vec()));
        let out = self.forward_with(m, embeds, rows, kv, taps);
        *self.positions.lock().unwrap_or_else(|p| p.into_inner()) = None;
        out
    }

    /// Whether a run of `rows` after what `kv` holds can keep its recurrent states from inside it once each of `taps`
    /// rows is in ([`Self::forward_tapped`]): chained (on one device or a prompt's chunks over two, each keeping its
    /// own layers'), and each chunk's rows after the first state it keeps few ([`TAP_ROWS`]: their recurrence goes
    /// through vectors of its own; a state early in a long chunk is a run's end still). OAIY_NO_TAPS: never.
    pub fn can_tap(&self, m: &Qwen35Model, rows: usize, kv: &KvCache, taps: &[usize]) -> bool {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || std::env::var_os("OAIY_NO_TAPS").is_some() || rows == 0 || taps.is_empty() {
            return false;
        }
        let (Some(st), Some(chain)) = (self.state(m), m.backend.chain()) else { return false };
        let chunks = self.chunks(chain, &st.dims, kv.len, rows);
        taps.windows(2).all(|w| w[0] < w[1])
            && taps[0] > 0
            && taps[taps.len() - 1] <= rows
            && chunks.iter().all(|&(at, t)| taps.iter().find(|&&p| p > at && p <= at + t).map_or(true, |&p| at + t - p <= TAP_ROWS))
    }

    /// The chunks of a run of `rows` after `past` positions (each its first row and its rows): as many rows as the
    /// attention's scratch has room for over the positions they reach (the backend's own length of it for the rows
    /// asked; past the room, as many as fit a part a run of positions), up to a chunk of 1,024 on one card (some 1%
    /// the faster), 512 where a second one takes their later layers (the more chunks the more of them run together).
    /// (A chunk cut in two past 10,580 positions whatever the kernel was a fifth of its time: the matmuls' rows by
    /// halves.)
    fn chunks(&self, chain: &dyn DeviceChain, s: &Dims, past: usize, rows: usize) -> Vec<(usize, usize)> {
        let most = if self.second.get().is_some() { MAX_ROWS } else { 2 * MAX_ROWS };
        // (a MoE model's experts, where its device routes no more at once)
        let most = if s.moe_rows > 0 { most.min(s.moe_rows) } else { most };
        let mut chunks = Vec::new();
        let mut at = 0;
        while at < rows {
            let want = most.min(rows - at);
            let t = if chain.attention_rows_out_len(want, s.n_h, s.hd, past + at + want) * 4 <= ATTENTION_SCRATCH {
                want
            } else {
                let runs = (past + at + rows).div_ceil(256).max(1);
                let per_row = (s.n_h * runs * (s.hd + 2) + s.n_h * s.hd) * 4;
                (ATTENTION_SCRATCH / per_row).clamp(1, want)
            };
            chunks.push((at, t));
            at += t;
        }
        chunks
    }

    /// [`Self::forward`], keeping the states `taps` name (none: as it is; some: [`Self::can_tap`] said so).
    fn forward_with(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache, taps: &[usize]) -> Option<(Tensor, Vec<Tapped>)> {
        if std::env::var_os("OAIY_NO_CHAIN").is_some() || rows == 0 || embeds.numel() != rows * m.config.embedding_dim {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        // a prompt's pieces a few at once on each device's queue; a step's, and a check's few rows', as recorded
        let _few = (rows >= FEW_ROWS).then(|| FewInFlight::on(std::iter::once(chain).chain(self.second.get().and_then(|b| b.chain()))));
        let emb_own;
        let emb = if embeds.is_device() {
            emb_own = embeds.to_host();
            emb_own.data()
        } else {
            embeds.data()
        };
        let s = st.dims;
        let chunks = self.chunks(chain, &s, kv.len, rows);
        // (a run at given positions stays on the model's own device: the second's chunks build their own tables)
        let placed = self.positions.lock().unwrap_or_else(|p| p.into_inner()).is_some();
        if chunks.len() > 1 && !placed {
            if let (Some(sp), Some(chain1)) = (self.split(m, st), self.second.get().and_then(|b| b.chain())) {
                return Some(self.forward_split(m, st, sp, chain, chain1, emb, &chunks, kv, taps));
            }
        }
        let backend: &dyn Backend = &*m.backend;
        // each chunk recorded and gone to the GPU as the chunk before runs, the chunk before's rows read back and
        // into the host's cache as this one runs
        // (OAIY_CHUNKS_LOG: when each chunk's recording began and ended and when the chunk before it was done, ms)
        let mut log = std::env::var_os("OAIY_CHUNKS_LOG").map(|_| (std::time::Instant::now(), Vec::new()));
        let said = |what: String, log: &mut Option<(std::time::Instant, Vec<String>)>| {
            if let Some((t0, lines)) = log {
                lines.push(format!("{:.1} {what}", t0.elapsed().as_secs_f64() * 1e3));
            }
        };
        let mut pending: Option<Qwen35Run<'_>> = None;
        let mut tapped = Vec::new();
        for &(at, t) in &chunks {
            said(format!("rows {at}..{} at {}: recording", at + t, kv.len), &mut log);
            // (the states this chunk keeps: by its own rows)
            let local: Vec<usize> = taps.iter().filter(|&&p| p > at && p <= at + t).map(|&p| p - at).collect();
            let mut run = self.run_begin(m, st, chain, &emb[at * s.d..(at + t) * s.d], t, kv, false, &local);
            tapped.append(&mut run.taps);
            said("recorded".to_string(), &mut log);
            if let Some(p) = pending.replace(run) {
                p.finish(backend, kv);
                said("the chunk before done".to_string(), &mut log);
            }
        }
        let out = pending.map(|p| p.finish(backend, kv).pop().expect("the logits"));
        said("the last done".to_string(), &mut log);
        if let Some((_, lines)) = &log {
            eprintln!("a prompt's {} chunks: {}", chunks.len(), lines.join("; "));
        }
        out.map(|logits| (logits, tapped))
    }

    /// One submit of `t` rows: the last row's logits, or (a check) every row's.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, m: &Qwen35Model, st: &State, chain: &dyn DeviceChain, emb: &[f32], t: usize, kv: &mut KvCache, checking: bool) -> Vec<Tensor> {
        self.run_begin(m, st, chain, emb, t, kv, checking, &[]).finish(&*m.backend, kv)
    }

    /// [`Self::run`] up to its wait: its work gone to the GPU, `kv` committed (its rows the device's copy's; into
    /// the host's cache at the finish).
    #[allow(clippy::too_many_arguments)]
    fn run_begin<'a>(&self, m: &Qwen35Model, st: &'a State, chain: &'a dyn DeviceChain, emb: &[f32], t: usize, kv: &mut KvCache, checking: bool, taps: &[usize]) -> Qwen35Run<'a> {
        let s = st.dims;
        let cfg = &m.config;
        let row = 2 * s.n_kv * s.hd;
        let past = kv.len;
        let eps = cfg.rms_eps;
        self.split_written(kv, past);
        // The rows a run writes are the device's copy's alone, read from it where they are wanted (the cache's
        // `DeviceRows`); with a second device for prompts they come back to the host's cache as ever, which that
        // device's copy is made from.
        let alone = self.second.get().is_none();
        if alone {
            kv.device_rows = Some(Arc::clone(&st.rows) as Arc<dyn crate::kv_cache::DeviceRows>);
            // (rows of its own the device held alone, set aside as another cache's run took the copy: to the host,
            // which the copy is made again from below)
            let theirs = st.kv.lock().unwrap_or_else(|p| p.into_inner()).owner != kv.id;
            if theirs && kv.host_valid() < past {
                kv.materialize(&*m.backend, past);
            }
        }
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        if g.owner != kv.id {
            stash_owner(chain, &mut g, &s, st.attention_layers.len(), st.halves_only);
            g.stash.retain(|(owner, _)| *owner != kv.id);
        }
        // (the rows from `past` on are written on the device: whatever the host held there is stale)
        kv.host_rows = kv.host_rows.min(past);
        let keep = kept(&g, kv, past);
        let needed = (past + t).max(kv.expect.min(kv.max_len));
        if st.halves_only {
            reserve_halves(chain, &mut g, &s, st.attention_layers.len(), needed, keep);
            sync_halves(chain, &mut g, &s, &st.attention_layers, kv, past);
        } else {
            reserve(chain, &mut g, &s, st.attention_layers.len(), needed, keep);
            sync(chain, &mut g, &s, &st.attention_layers, kv, past);
        }
        // every row to here is the device's now: what the host held of them (a restore's, a run op by op) let go
        if alone && kv.host_rows > 0 {
            kv.release_host();
        }
        (g.rows, g.owner_host, g.gone) = (past + t, kv.host_valid(), false);
        // the layers' rows from `past` on are written here: their halves' rows end there; a step deep in a long cache
        // reads the halves, brought up to its own row as it goes
        g.halved = g.halved.min(past);
        // (a step's, or a few rows': a check of drafted tokens, each row's attention then a step's; OAIY_FEW_ROWS_F32:
        // a few rows' by their own kernel over the cache as it is)
        let few = t > 1 && t <= SPEC_ROWS && std::env::var_os("OAIY_FEW_ROWS_F32").is_none();
        let mut halves = !st.halves_only && (t == 1 || few) && past + t >= halves_from() && chain.attention_halves(s.n_h, s.n_kv, s.hd);
        if halves && g.half.len() != g.layers.len() {
            // half the cache's size again (1.07 GB at 16,384 positions, 2.1 at 32,768): only where the device says it
            // has that and more to spare, else the step reads the cache as it is
            let words = g.cap * row / 2;
            if g.refused != g.cap && chain.has_room((g.layers.len() * words * 4) as u64) {
                g.half = (0..g.layers.len()).map(|_| chain.vec(words)).collect();
                g.halved = 0;
            } else {
                g.refused = g.cap;
                halves = false;
            }
        }
        let halved = g.halved;
        if halves {
            g.halved = past + t;
        }
        // the recurrent states the cache holds, as the chain's vectors
        let mut pool = st.pool.lock().unwrap_or_else(|p| p.into_inner());
        let mut states = Vec::with_capacity(st.ssm_layers.len());
        for (i, &l) in st.ssm_layers.iter().enumerate() {
            let state = adopt(chain, &mut pool.states[i], &mut kv.ssm_state[l], vec![s.nv, s.dv, s.dk]);
            let conv = adopt(chain, &mut pool.convs[i], &mut kv.ssm_conv[l], vec![s.kern - 1, s.ch]);
            states.push((state, conv));
        }
        drop(pool);
        // the states kept from inside the run (`taps`: after that many of its rows, ascending): a pair of vectors a
        // slot for each, and the vectors the rows after the first of them go through
        assert!(
            !(checking && !taps.is_empty()) && taps.windows(2).all(|w| w[0] < w[1]) && taps.first().map_or(true, |&r| r > 0 && t - r.min(t) <= TAP_ROWS) && taps.last().map_or(true, |&r| r <= t),
            "chain: a run of {t} rows keeps states after {taps:?}"
        );
        let kept: Vec<Tap> = taps.iter().map(|&row| Tap { row, vecs: states.iter().map(|(state, conv)| (chain.vec(state.len), chain.vec(conv.len))).collect() }).collect();
        let seg = kept.first().filter(|tap| tap.row < t).map(|_| st.seg.get_or_init(|| Seg::new(chain, &s)));
        // a prompt's chunk's vectors a pooled set's (room for a chunk of the most rows), its attention's scratch grown
        // as the positions it reaches do
        let mut set = None;
        let prompt;
        let w = if t == 1 {
            &st.step
        } else {
            let taken = st.prompt_sets.lock().unwrap_or_else(|p| p.into_inner()).pop();
            let (full, scratch) = taken.filter(|(f, _)| f.x.len >= t * s.d).unwrap_or_else(|| (Work::new(chain, &s, t.max(2 * MAX_ROWS)), chain.vec(1)));
            let need = chain.attention_rows_out_len(t, s.n_h, s.hd, past + t);
            let scratch = if scratch.len >= need { scratch } else { chain.vec(need + need / 2) };
            prompt = full.view(&s, t);
            set = Some((full, scratch));
            &prompt
        };
        let attn = match &set {
            None => g.out.clone(),
            Some((_, scratch)) => first(scratch, chain.attention_rows_out_len(t, s.n_h, s.hd, past + t)),
        };
        chain.upload(&w.table, &self.table(cfg.rope_theta, s.rot, past, t));
        chain.upload(&w.x, emb);
        let mut rec = chain.begin();
        // a prompt's vectors are its own: no bind groups kept to hold them
        rec.keep_groups(t == 1);
        // each residual's add waits for the norm after it (the next layer's, or the output's): one dispatch for both
        let mut added = false;
        let bound = if st.halves_only {
            Bound { kvl: &[], half: Some((&g.half[..], past + t)), one: (t > 1).then(|| (&st.step.qn, &g.out)), halves_only: true, cap: g.cap, states: &states, w, attn: &attn, taps: &kept, seg }
        } else {
            Bound { kvl: &g.layers, half: halves.then(|| (&g.half[..], halved)), one: (halves && few).then(|| (&st.step.qn, &g.out)), halves_only: false, cap: g.cap, states: &states, w, attn: &attn, taps: &kept, seg }
        };
        let check = if checking { st.spec.as_ref() } else { None };
        record_layers(&mut *rec, &s, eps, m.blocks.iter().map(LayerW::of).zip(&st.layers), &bound, t, past, check, &mut added);
        // the head of the last row only, or of a check's every row; with a prediction layer, the hidden states after the
        // output norm kept (a check's rows, or the last), and a prompt's chunk through the layer too (its cache)
        if added {
            rec.add_rmsnorm_rows(&w.x, &w.proj, &st.output_norm, &w.xn, t, eps);
        } else {
            rec.rmsnorm_rows(&w.x, &st.output_norm, &w.xn, t, eps);
        }
        if let Some(sp) = &st.spec {
            let rows = if checking { t } else { 1 };
            // (the layer's cache first: a chunk's carries on from the hidden state the run before kept)
            if !checking && t > 1 {
                self.mtp_prompt(m, st, sp, chain, &mut *rec, emb, &w.xn, t, past, kv.id);
            }
            rec.copy(&w.xn, (t - rows) * s.d, &sp.hid, 0, rows * s.d);
            *sp.hid_at.lock().unwrap_or_else(|p| p.into_inner()) = (past + t - rows, rows);
        }
        if checking {
            let sp = st.spec.as_ref().expect("a check is a drafting chain's");
            W::of(&m.output).rows(&mut *rec, &w.xn, &sp.logits, t);
            rec.read_range(&sp.logits, 0, t * s.vocab);
        } else {
            let last = if t == 1 {
                &w.xn
            } else {
                rec.copy(&w.xn, (t - 1) * s.d, &w.last, 0, s.d);
                &w.last
            };
            W::of(&m.output).one(&mut *rec, last, &st.logits);
            rec.read(&st.logits);
        }
        if !alone {
            for slot in 0..st.attention_layers.len() {
                if st.halves_only {
                    rec.read_range(&g.half[slot], past * row / 2, t * row / 2);
                } else {
                    rec.read_range(&g.layers[slot], past * row, t * row);
                }
            }
        }
        rec.flush();
        kv.commit(t);
        kv.dirty_from = usize::MAX;
        self.runs.fetch_add(1, Ordering::Relaxed);
        // the kept states as the cache holds a layer's (theirs once the run has run)
        let taps = kept
            .iter()
            .map(|tap| {
                let none = || (0..kv.ssm_state.len()).map(|_| None).collect::<Vec<Option<Tensor>>>();
                let (mut states, mut convs) = (none(), none());
                for (i, &l) in st.ssm_layers.iter().enumerate() {
                    states[l] = Some(chain.alias(&tap.vecs[i].0, vec![s.nv, s.dv, s.dk]));
                    convs[l] = Some(chain.alias(&tap.vecs[i].1, vec![s.kern - 1, s.ch]));
                }
                Tapped { at: past + tap.row, states, convs }
            })
            .collect();
        Qwen35Run { rec, layers: if alone { Vec::new() } else { st.attention_layers.clone() }, past, t, vocab: s.vocab, set: set.map(|v| (&st.prompt_sets, v)), taps, halves: st.halves_only }
    }
}

/// `v`'s first `len` elements, as a vector of their own (the same buffer): a chain's ops take their sizes from their
/// vectors' lengths, and the layer's are kept for [`SPEC_ROWS`] rows.
fn first(v: &DeviceVec, len: usize) -> DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    DeviceVec { len, inner: Arc::clone(&v.inner) }
}
