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
use crate::qwen35::{Qwen35Block, Qwen35Model};

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
    /// The layers' rows as f16, two values a word ([`ggml_rs::ChainRecorder::halve`]), for a step's attention deep in
    /// a long cache ([`HALVES_FROM`]); none until such a step, or where the device has no such attention. Their
    /// first `halved` rows are the layers': whatever writes a layer's rows from a position on lowers it to there.
    half: Vec<DeviceVec>,
    halved: usize,
    /// The `cap` at which the device had no room for the halves (not asked again until the cache grows).
    refused: usize,
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
}

impl<'a> W<'a> {
    /// `w`, which the chain's state checked the device holds.
    fn of(w: &'a Weight) -> Self {
        match w {
            Weight::Quant(q) => W::Quant(q),
            Weight::Packed(p) => W::Packed(p.as_ref()),
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
        }
    }

    /// `y` = one row `x` times this.
    fn one(self, rec: &mut dyn ggml_rs::ChainRecorder, x: &DeviceVec, y: &DeviceVec) {
        match self {
            W::Quant(q) => rec.matmul(q, x, y),
            W::Packed(p) => rec.exl3_rows(p, x, y, 1),
        }
    }
}

/// A layer's projections a run reads: the model's own (on the first device), or their copies on the second.
#[derive(Clone, Copy)]
enum LayerW<'a> {
    Attention { q: W<'a>, k: W<'a>, v: W<'a>, o: W<'a>, ffn: FfnW<'a>, down: W<'a> },
    /// `ba` none where the fused beta-alpha projection's weights are f32 (the layer's vectors hold them)
    Ssm { qkv: W<'a>, gate: W<'a>, ba: Option<W<'a>>, out: W<'a>, ffn: FfnW<'a>, down: W<'a> },
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
            Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, .. } => {
                LayerW::Attention { q: W::of(attn_q), k: W::of(attn_k), v: W::of(attn_v), o: W::of(attn_output), ffn: ffn(ffn_pair), down: W::of(ffn_down) }
            }
            Qwen35Block::Ssm { attn_qkv, attn_gate, ssm_ba, ssm_out, ffn_pair, ffn_down, .. } => LayerW::Ssm {
                qkv: W::of(attn_qkv),
                gate: W::of(attn_gate),
                ba: match ssm_ba {
                    Weight::Quant(q) => Some(W::Quant(q)),
                    _ => None,
                },
                out: W::of(ssm_out),
                ffn: ffn(ffn_pair),
                down: W::of(ffn_down),
            },
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
            W::Packed(_) => None,
        };
        let ffn = |f: FfnW<'_>| -> Option<SplitFfn> {
            Some(match f {
                FfnW::Fused(gu) => SplitFfn::Fused(c(gu)?),
                FfnW::Split(g, u) => SplitFfn::Split(c(g)?, c(u)?),
            })
        };
        Some(match w {
            LayerW::Attention { q, k, v, o, ffn: f, down } => SplitW::Attention { q: c(q)?, k: c(k)?, v: c(v)?, o: c(o)?, ffn: ffn(f)?, down: c(down)? },
            LayerW::Ssm { qkv, gate, ba, out, ffn: f, down } => SplitW::Ssm {
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
                LayerW::Attention { q: W::Quant(q), k: W::Quant(k), v: W::Quant(v), o: W::Quant(o), ffn: ffn(f), down: W::Quant(down) }
            }
            SplitW::Ssm { qkv, gate, ba, out, ffn: f, down } => {
                LayerW::Ssm { qkv: W::Quant(qkv), gate: W::Quant(gate), ba: ba.as_ref().map(W::Quant), out: W::Quant(out), ffn: ffn(f), down: W::Quant(down) }
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
        }
    }

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
    /// Drafting and checking, where the model has a multi-token-prediction layer the device holds.
    spec: Option<Spec>,
    /// A prompt's chunks' vectors, each set room for the most rows a chunk has, and their attention's scratch: taken
    /// by a chunk and given back once it has run (each chunk's new ones, some 25 buffers and 0.5 GB, were the
    /// allocator's every 200 ms).
    prompt_sets: Mutex<Vec<(Work, DeviceVec)>>,
    /// A tapped run's later rows' vectors, made at the first run that keeps a state from inside.
    seg: std::sync::OnceLock<Seg>,
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
    // (the halves' vectors are the old layers': made again, and filled, at the next step that reads them)
    g.half.clear();
    g.halved = 0;
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
    match b {
        Qwen35Block::Attention { attn_norm, attn_q_norm, attn_k_norm, post_norm, .. } => {
            LayerVecs { attn_norm: up(attn_norm), post_norm: up(post_norm), mixer: Mixer::Attention { q_norm: up(attn_q_norm), k_norm: up(attn_k_norm), slot } }
        }
        Qwen35Block::Ssm { attn_norm, ssm_conv1d, ssm_a, ssm_ba, ssm_dt_bias, ssm_norm, post_norm, .. } => {
            let ba_f32 = match ssm_ba {
                Weight::Dense(t) => Some(up(t)),
                _ => None,
            };
            LayerVecs {
                attn_norm: up(attn_norm),
                post_norm: up(post_norm),
                mixer: Mixer::Ssm { conv_w: up(ssm_conv1d), a: up(ssm_a), dt: up(ssm_dt_bias), norm: up(ssm_norm), ba_f32, slot },
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
        Seg { qkv: chain.vec(TAP_ROWS * s.ch), z: chain.vec(TAP_ROWS * s.nv * s.dv), ba: chain.vec(TAP_ROWS * 2 * s.nv), conv: chain.vec(TAP_ROWS * s.ch), core: chain.vec(TAP_ROWS * s.nv * s.dv) }
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
        let (ffn, down) = match (lw, &lv.mixer) {
            (LayerW::Attention { q, k, v, o, ffn, down }, Mixer::Attention { q_norm, k_norm, slot }) => {
                let kvl = &b.kvl[*slot];
                q.rows(rec, &w.xn, &w.qfull, t);
                k.rows(rec, &w.xn, &w.k, t);
                v.rows(rec, &w.xn, &w.v, t);
                // each head's q is its query then its gate
                rec.copy_cols(&w.qfull, &w.q, t * s.n_h, s.hd, 2 * s.hd, 0);
                rec.copy_cols(&w.qfull, &w.gate, t * s.n_h, s.hd, 2 * s.hd, s.hd);
                rec.rmsnorm_rows(&w.q, q_norm, &w.qn, t * s.n_h, eps);
                rec.rmsnorm_rows(&w.k, k_norm, &w.kn, t * s.n_kv, eps);
                rec.rope_partial_rows(&w.qn, t, s.n_h, s.hd, s.rot, &w.table);
                rec.rope_partial_rows(&w.kn, t, s.n_kv, s.hd, s.rot, &w.table);
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
                rec.mul_sigmoid(b.attn, &w.gate, &w.gated, t * s.n_h * s.hd);
                o.rows(rec, &w.gated, &w.proj, t);
                (ffn, down)
            }
            (LayerW::Ssm { qkv, gate, ba, out, ffn, down }, Mixer::Ssm { conv_w, a, dt, norm, ba_f32, slot }) => {
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
                (ffn, down)
            }
            _ => unreachable!("a layer's vectors are its block's"),
        };
        rec.add_rmsnorm_rows(&w.x, &w.proj, &lv.post_norm, &w.xn, t, eps);
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
        *added = true;
    }
}

/// A run's K and V rows of layer `l` (`t` of them from `at`, as the chain's copy holds them) into the host's cache.
fn cache_rows(backend: &dyn Backend, kv: &mut KvCache, l: usize, at: usize, t: usize, rows: &[f32]) {
    let len = kv.len;
    kv.len = at;
    kv.append_rows(backend, l, rows, t);
    kv.len = len;
}

/// A split prompt's run ([`Qwen35Chain::forward_split`]): what its chunks share.
struct SplitRun<'a> {
    chained: &'a Qwen35Chain,
    m: &'a Qwen35Model,
    st: &'a State,
    sp: &'a Split,
    chain: &'a dyn DeviceChain,
    chain1: &'a dyn DeviceChain,
    emb: &'a [f32],
    /// each chunk's first row (of the prompt's) and rows; the cache's rows before the prompt
    chunks: &'a [(usize, usize)],
    past0: usize,
    /// the first's copy of the attention cache (its buffers, and rows of room), and the second's of its layers
    kv0: (&'a [DeviceVec], usize),
    kv1: (&'a [DeviceVec], usize),
    states: &'a [(DeviceVec, DeviceVec)],
    states1: &'a [(DeviceVec, DeviceVec)],
    /// the first's attention slots of the layers it runs, and the layer of each of the second's slots
    slots0: Vec<usize>,
    layers1: &'a [usize],
    owner: u64,
    /// The rows of the run after which the recurrent states are kept (ascending), those states as its chunks record
    /// them (a device's layers' at its part of a chunk), and the cache's layers.
    taps: &'a [usize],
    kept: std::cell::RefCell<Vec<Tapped>>,
    layers_n: usize,
    /// Whether the second's recurrent states go up from the first's (else they are zero: a new conversation's)
    states_up: bool,
    /// OAIY_SPLIT_LOG: when each step began and ended (ms from the run's start), for the log
    log: Option<(std::time::Instant, std::cell::RefCell<Vec<String>>)>,
}

/// A split run's chunk gone to the first device: its recording (its reads the residual stream and the last layer's
/// output, at the first chunk the second's recurrent states, then the first's layers' K and V rows).
struct FirstRun<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    i: usize,
}

/// A split run's chunk gone to the second device: its recording (its reads the last chunk's logits, the hidden states
/// where a prediction layer's cache takes them, the second's layers' K and V rows, at the last chunk its recurrent
/// states).
struct SecondRun<'a> {
    rec: Box<dyn ggml_rs::ChainRecorder + 'a>,
    i: usize,
}

impl<'a> SplitRun<'a> {
    /// The time since the run began, ms (for the log).
    fn now(&self) -> f64 {
        self.log.as_ref().map_or(0.0, |(t, _)| t.elapsed().as_secs_f64() * 1e3)
    }

    fn note(&self, what: String) {
        if let Some((_, l)) = &self.log {
            l.borrow_mut().push(what);
        }
    }

    /// The states chunk `i` keeps on one device (`c`, whose recurrent slots are `states`): for each of the run's kept
    /// rows in the chunk a pair of vectors for each slot whose layer the device runs (`layer_of`: that layer, in the
    /// cache's order), the run's kept states given them (theirs once that part of the chunk has run).
    fn tap(&self, i: usize, c: &dyn DeviceChain, states: &[(DeviceVec, DeviceVec)], layer_of: &dyn Fn(usize) -> Option<usize>) -> Vec<Tap> {
        let s = self.st.dims;
        let (at, t) = self.chunks[i];
        let mut kept = self.kept.borrow_mut();
        let mut spare: Option<DeviceVec> = None;
        self.taps
            .iter()
            .filter(|&&p| p > at && p <= at + t)
            .map(|&p| {
                let abs = self.past0 + p;
                if !kept.iter().any(|k| k.at == abs) {
                    let none = || (0..self.layers_n).map(|_| None).collect::<Vec<Option<Tensor>>>();
                    kept.push(Tapped { at: abs, states: none(), convs: none() });
                }
                let entry = kept.iter_mut().find(|k| k.at == abs).expect("the kept state");
                let vecs = states
                    .iter()
                    .enumerate()
                    .map(|(slot, (state, conv))| match layer_of(slot) {
                        Some(l) => {
                            let (sv, cv) = (c.vec(state.len), c.vec(conv.len));
                            entry.states[l] = Some(c.alias(&sv, vec![s.nv, s.dv, s.dk]));
                            entry.convs[l] = Some(c.alias(&cv, vec![s.kern - 1, s.ch]));
                            (sv, cv)
                        }
                        // (a slot whose layer the other device runs: never copied to)
                        None => {
                            let v = spare.get_or_insert_with(|| c.vec(1)).clone();
                            (v.clone(), v)
                        }
                    })
                    .collect();
                Tap { row: p - at, vecs }
            })
            .collect()
    }

    /// Chunk `i`'s layers before the split's, gone to the first device after the prediction layer's work for the
    /// chunks whose hidden states are back (`due`).
    fn first(&self, i: usize, due: &mut Vec<(usize, Vec<f32>)>) -> FirstRun<'a> {
        let t0 = self.now();
        let (s, cfg) = (self.st.dims, &self.m.config);
        let (at, t) = self.chunks[i];
        let pos = self.past0 + at;
        let row = 2 * s.n_kv * s.hd;
        let c = self.chain;
        let w = Work::new(c, &s, t);
        let attn = c.vec(c.attention_rows_out_len(t, s.n_h, s.hd, pos + t));
        c.upload(&w.table, &rope_table(cfg.rope_theta, s.rot, pos, t));
        c.upload(&w.x, &self.emb[at * s.d..(at + t) * s.d]);
        let mut rec = c.begin();
        rec.keep_groups(false);
        for (j, hidden) in due.drain(..) {
            self.mtp(&mut *rec, j, &hidden);
        }
        let mut added = false;
        let kept = self.tap(i, c, self.states, &|j| Some(self.st.ssm_layers[j]).filter(|&l| l < self.sp.from));
        let seg = kept.first().filter(|tap| tap.row < t).map(|_| self.st.seg.get_or_init(|| Seg::new(c, &s)));
        let bound = Bound { kvl: self.kv0.0, half: None, one: None, cap: self.kv0.1, states: self.states, w: &w, attn: &attn, taps: &kept, seg };
        let from = self.sp.from;
        record_layers(&mut *rec, &s, cfg.rms_eps, self.m.blocks[..from].iter().map(LayerW::of).zip(&self.st.layers[..from]), &bound, t, pos, None, &mut added);
        // to the second: the residual stream and the last layer's output (their add fused with its first layer's norm)
        rec.read(&w.x);
        rec.read(&w.proj);
        if i == 0 && self.states_up {
            for &j in &self.sp.ssm_slots {
                rec.read(&self.states[j].0);
                rec.read(&self.states[j].1);
            }
        }
        for &a in &self.slots0 {
            rec.read_range(&self.kv0.0[a], pos * row, t * row);
        }
        rec.flush();
        self.note(format!("first {i}: recorded {t0:.1}..{:.1}", self.now()));
        FirstRun { rec, i }
    }

    /// Chunk `f`'s layers from the split's on and the head, recorded on the second device as the first runs its first
    /// ones, gone once those have run and their output is up (their K and V rows into the host's cache).
    fn hand_off(&self, f: FirstRun<'a>, kv: &mut KvCache) -> SecondRun<'a> {
        let t0 = self.now();
        let (s, cfg) = (self.st.dims, &self.m.config);
        let i = f.i;
        let (at, t) = self.chunks[i];
        let pos = self.past0 + at;
        let last = i + 1 == self.chunks.len();
        let row = 2 * s.n_kv * s.hd;
        let c = self.chain1;
        let w = Work::new(c, &s, t);
        let attn = c.vec(c.attention_rows_out_len(t, s.n_h, s.hd, pos + t));
        c.upload(&w.table, &rope_table(cfg.rope_theta, s.rot, pos, t));
        let mut rec = c.begin();
        rec.keep_groups(false);
        rec.hold();
        let mut added = true;
        let kept = self.tap(i, c, self.states1, &|k| Some(self.st.ssm_layers[self.sp.ssm_slots[k]]));
        let seg = kept.first().filter(|tap| tap.row < t).map(|_| self.sp.seg.get_or_init(|| Seg::new(c, &s)));
        let bound = Bound { kvl: self.kv1.0, half: None, one: None, cap: self.kv1.1, states: self.states1, w: &w, attn: &attn, taps: &kept, seg };
        record_layers(&mut *rec, &s, cfg.rms_eps, self.sp.weights.iter().map(SplitW::view).zip(&self.sp.layers), &bound, t, pos, None, &mut added);
        rec.add_rmsnorm_rows(&w.x, &w.proj, &self.sp.output_norm, &w.xn, t, cfg.rms_eps);
        if last {
            rec.copy(&w.xn, (t - 1) * s.d, &w.last, 0, s.d);
            rec.matmul(&self.sp.output, &w.last, &self.sp.logits);
            rec.read(&self.sp.logits);
        }
        if self.st.spec.is_some() {
            rec.read(&w.xn);
        }
        for a in 0..self.layers1.len() {
            rec.read_range(&self.kv1.0[a], pos * row, t * row);
        }
        if last {
            for (state, conv) in self.states1 {
                rec.read(state);
                rec.read(conv);
            }
        }
        // the first's part done: its output up, the second's going, then the first's layers' rows into the host's cache
        let t1 = self.now();
        let mut got = f.rec.finish().into_iter();
        let t2 = self.now();
        c.upload(&w.x, &got.next().expect("the residual stream"));
        c.upload(&w.proj, &got.next().expect("the last layer's output"));
        if i == 0 && self.states_up {
            for (state, conv) in self.states1 {
                c.upload(state, &got.next().expect("a recurrent state"));
                c.upload(conv, &got.next().expect("its conv window"));
            }
        }
        rec.flush();
        let t3 = self.now();
        for &a in &self.slots0 {
            cache_rows(&*self.m.backend, kv, self.st.attention_layers[a], pos, t, &got.next().expect("a layer's K and V"));
        }
        self.note(format!("hand off {i}: recorded {t0:.1}..{t1:.1}, the first's done {t2:.1}, the second's gone {t3:.1}, stored {:.1}", self.now()));
        SecondRun { rec, i }
    }

    /// Chunk `r` done on the second device: its layers' K and V rows into the host's cache and the first's copy, its
    /// hidden states due for the prediction layer's cache, at the last chunk its logits and the recurrent states back
    /// on the first.
    fn finish(&self, r: SecondRun<'a>, kv: &mut KvCache, due: &mut Vec<(usize, Vec<f32>)>, logits: &mut Option<Vec<f32>>) {
        let s = self.st.dims;
        let (at, t) = self.chunks[r.i];
        let pos = self.past0 + at;
        let last = r.i + 1 == self.chunks.len();
        let row = 2 * s.n_kv * s.hd;
        let t0 = self.now();
        let mut got = r.rec.finish().into_iter();
        let t1 = self.now();
        if last {
            *logits = got.next();
        }
        if self.st.spec.is_some() {
            due.push((r.i, got.next().expect("the hidden states")));
        }
        for (k, &l) in self.layers1.iter().enumerate() {
            let rows = got.next().expect("a layer's K and V");
            self.chain.upload_at(&self.kv0.0[self.sp.attention_slots[k]], pos * row, &rows);
            cache_rows(&*self.m.backend, kv, l, pos, t, &rows);
        }
        if last {
            for &j in &self.sp.ssm_slots {
                self.chain.upload(&self.states[j].0, &got.next().expect("a recurrent state"));
                self.chain.upload(&self.states[j].1, &got.next().expect("its conv window"));
            }
        }
        self.note(format!("finish {}: waited {t0:.1}..{t1:.1}, stored {:.1}", r.i, self.now()));
    }

    /// The prediction layer's cache at chunk `j` (its hidden states after the output norm back from the second), and
    /// the chunk's last hidden state kept: as a run on the first device alone does.
    fn mtp(&self, rec: &mut dyn ggml_rs::ChainRecorder, j: usize, hidden: &[f32]) {
        let Some(spec) = &self.st.spec else { return };
        let s = self.st.dims;
        let (at, t) = self.chunks[j];
        let pos = self.past0 + at;
        let hv = self.chain.vec(t * s.d);
        self.chain.upload(&hv, hidden);
        if t > 1 {
            self.chained.mtp_prompt(self.m, self.st, spec, self.chain, rec, &self.emb[at * s.d..(at + t) * s.d], &hv, t, pos, self.owner);
        }
        rec.copy(&hv, (t - 1) * s.d, &spec.hid, 0, s.d);
        *spec.hid_at.lock().unwrap_or_else(|p| p.into_inner()) = (pos + t - 1, 1);
    }

    /// The prediction layer's work for the chunks `due`, on the first device (which has run its last chunk), gone.
    fn mtp_only(&self, due: &mut Vec<(usize, Vec<f32>)>) -> Box<dyn ggml_rs::ChainRecorder + 'a> {
        let mut rec = self.chain.begin();
        rec.keep_groups(false);
        for (j, hidden) in due.drain(..) {
            self.mtp(&mut *rec, j, &hidden);
        }
        rec.flush();
        rec
    }
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
            kv.append_rows(backend, l, &rows, self.t);
        }
        kv.len = len;
        kv.dirty_from = usize::MAX;
        logits
    }
}

impl Qwen35Chain {
    fn state(&self, m: &Qwen35Model) -> Option<&State> {
        self.state
            .get_or_init(|| {
                let chain = m.backend.chain()?;
                let cfg = &m.config;
                let sc = &m.ssm_cfg;
                let held = |w: &Weight| W::held(chain, w);
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
                let upload = |t: &Tensor| upload_tensor(chain, t);
                let dims = Dims { d, n_h, n_kv, hd, rot: cfg.rope_dim, ff, nv, nk, dk, dv, ch, kern: sc.conv_kernel, vocab: cfg.vocab_size };
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
                            Qwen35Block::Ssm { .. } => {
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
                    let Qwen35Block::Attention { attn_norm, attn_q, attn_q_norm, attn_k, attn_k_norm, attn_v, attn_output, post_norm, ffn_pair, ffn_down } = &mtp.block else { return None };
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
                Some(State {
                    dims,
                    layers,
                    attention_layers,
                    ssm_layers,
                    output_norm: upload(&m.output_norm),
                    kv: Mutex::new(Kv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0, half: Vec::new(), halved: 0, refused: 0 }),
                    pool: Mutex::new(pool),
                    step: Work::new(chain, &dims, 1),
                    logits: chain.vec(cfg.vocab_size),
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

    /// A second device for prompts: a prompt's chunks run over both ([`Self::forward`]), the layers from the middle
    /// on and the head copied there at the first prompt that can use them (steps and checks stay on the model's own).
    pub fn split_onto(&self, backend: Arc<dyn Backend>) {
        let _ = self.second.set(backend);
    }

    /// The second device's share of a prompt's layers, made at the first prompt that can use it: None where there
    /// is no second device, no room on it for them, or prompts are not split (`split_off`, OAIY_NO_SPLIT).
    fn split(&self, m: &Qwen35Model, st: &State) -> Option<&Split> {
        if self.split_off.load(Ordering::Relaxed) || std::env::var_os("OAIY_NO_SPLIT").is_some() {
            return None;
        }
        self.split
            .get_or_init(|| {
                let chain = self.second.get()?.chain()?;
                let s = st.dims;
                let n = m.blocks.len();
                // half the layers on each, unless asked otherwise (OAIY_SPLIT_AT: the second's first layer)
                let from = std::env::var("OAIY_SPLIT_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(n / 2);
                if from == 0 || from >= n {
                    return None;
                }
                // (a GGUF's alone: a packed model's prompts stay on the device its projections were made on)
                let Weight::Quant(head) = &m.output else { return None };
                let output = chain.copy_weight(head)?;
                let weights = m.blocks[from..].iter().map(|b| SplitW::copy(chain, LayerW::of(b))).collect::<Option<Vec<_>>>()?;
                let (mut attention_slots, mut ssm_slots) = (Vec::new(), Vec::new());
                let layers = (from..n)
                    .map(|l| {
                        let slot = match &st.layers[l].mixer {
                            Mixer::Attention { slot, .. } => {
                                attention_slots.push(*slot);
                                attention_slots.len() - 1
                            }
                            Mixer::Ssm { slot, .. } => {
                                ssm_slots.push(*slot);
                                ssm_slots.len() - 1
                            }
                        };
                        layer_vecs(chain, &m.blocks[l], slot)
                    })
                    .collect();
                let pool = Pool { states: ssm_slots.iter().map(|_| chain.vec(s.nv * s.dk * s.dv)).collect(), convs: ssm_slots.iter().map(|_| chain.vec((s.kern - 1) * s.ch)).collect() };
                Some(Split {
                    from,
                    weights,
                    layers,
                    attention_slots,
                    ssm_slots,
                    output_norm: upload_tensor(chain, &m.output_norm),
                    output,
                    logits: chain.vec(s.vocab),
                    kv: Mutex::new(SplitKv { kv: Kv { layers: Vec::new(), cap: 0, out: chain.vec(1), owner: 0, half: Vec::new(), halved: 0, refused: 0 }, upto: 0 }),
                    pool,
                    seg: Default::default(),
                })
            })
            .as_ref()
    }

    /// The second device's share of a prompt's layers made (their weights copied there) and its kernels compiled: a
    /// prompt of two chunks over both devices, on a cache of its own (`tokens` repeated). False where prompts do not
    /// split.
    pub(crate) fn warm_split(&self, m: &Qwen35Model, tokens: &[u32]) -> bool {
        let (Some(st), Some(chain)) = (self.state(m), m.backend.chain()) else { return false };
        let (Some(sp), Some(chain1)) = (self.split(m, st), self.second.get().and_then(|b| b.chain())) else { return false };
        let rows = 2 * 64;
        let tokens: Vec<u32> = tokens.iter().copied().cycle().take(rows).collect();
        if tokens.len() < rows {
            return false;
        }
        let e = m.embed_text(&tokens).to_host();
        let mut kv = KvCache::new(&*m.backend, m.config.n_layers, rows + 16, m.config.n_kv_heads, m.config.head_dim);
        self.forward_split(m, st, sp, chain, chain1, e.data(), &[(0, 64), (64, 64)], &mut kv, &[]);
        true
    }

    /// Rows of `kv` from `past` on written on the first device alone (a step, a check, a prompt there): the second's
    /// copy of its layers' cache holds the cache's up to there at most (and none the host wrote since).
    fn split_written(&self, kv: &KvCache, past: usize) {
        if let Some(Some(sp)) = self.split.get() {
            let mut g1 = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
            if g1.kv.owner == kv.id {
                g1.upto = g1.upto.min(kv.dirty_from).min(past);
            }
        }
    }

    /// [`Self::forward`]'s `chunks` (each its first row of the prompt's and its rows) over two devices: each chunk's
    /// layers before the split's on the first as the second runs the chunk before's from there on, and the head. The
    /// second's copy of its layers' attention cache is brought up to the prompt first, and its recurrent states are
    /// the first's (read back at the first chunk's handoff), the first's again after (steps run there); its layers'
    /// K and V rows go into the host's cache and the first's copy. With a prediction layer each chunk's hidden states
    /// come back to the first for the layer's cache. The last chunk's logits.
    #[allow(clippy::too_many_arguments)]
    fn forward_split(&self, m: &Qwen35Model, st: &State, sp: &Split, chain: &dyn DeviceChain, chain1: &dyn DeviceChain, emb: &[f32], chunks: &[(usize, usize)], kv: &mut KvCache, taps: &[usize]) -> (Tensor, Vec<Tapped>) {
        let s = st.dims;
        let past0 = kv.len;
        let end = past0 + chunks.iter().map(|c| c.1).sum::<usize>();
        // the first's copy of the cache (every layer's) up to the prompt, and room for all of it
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        // (the prompt's rows are written from here on: the halves' end there)
        g.halved = g.halved.min(past0);
        reserve(chain, &mut g, &s, st.attention_layers.len(), end);
        sync(chain, &mut g, &s, &st.attention_layers, kv, past0);
        // the second's, of its layers: the rows it lacks (the host's since, and those the first wrote)
        let mut g1 = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        let layers1: Vec<usize> = sp.attention_slots.iter().map(|&a| st.attention_layers[a]).collect();
        reserve(chain1, &mut g1.kv, &s, layers1.len(), end);
        let from1 = if g1.kv.owner == kv.id { g1.upto.min(kv.dirty_from).min(past0) } else { 0 };
        upload_rows(chain1, &g1.kv, &s, &layers1, kv, from1, past0);
        g1.kv.owner = kv.id;
        kv.dirty_from = usize::MAX;
        // the recurrent states the cache holds, as the first's vectors; the second's sent up from them, or zero for a
        // new conversation's
        let fresh = sp.ssm_slots.iter().all(|&j| kv.ssm_state[st.ssm_layers[j]].is_none() && kv.ssm_conv[st.ssm_layers[j]].is_none());
        let mut pool = st.pool.lock().unwrap_or_else(|p| p.into_inner());
        let states: Vec<(DeviceVec, DeviceVec)> = st
            .ssm_layers
            .iter()
            .enumerate()
            .map(|(i, &l)| (adopt(chain, &mut pool.states[i], &mut kv.ssm_state[l], vec![s.nv, s.dv, s.dk]), adopt(chain, &mut pool.convs[i], &mut kv.ssm_conv[l], vec![s.kern - 1, s.ch])))
            .collect();
        drop(pool);
        let states1: Vec<(DeviceVec, DeviceVec)> = sp.pool.states.iter().cloned().zip(sp.pool.convs.iter().cloned()).collect();
        if fresh {
            for (state, conv) in &states1 {
                chain1.zero(state);
                chain1.zero(conv);
            }
        }
        let n = chunks.len();
        let mut logits = None;
        let tapped;
        {
            let run = SplitRun {
                chained: self,
                m,
                st,
                sp,
                chain,
                chain1,
                emb,
                chunks,
                past0,
                kv0: (&g.layers, g.cap),
                kv1: (&g1.kv.layers, g1.kv.cap),
                states: &states,
                states1: &states1,
                slots0: (0..st.attention_layers.len()).filter(|&a| st.attention_layers[a] < sp.from).collect(),
                layers1: &layers1,
                owner: kv.id,
                taps,
                kept: Default::default(),
                layers_n: kv.ssm_state.len(),
                states_up: !fresh,
                log: std::env::var_os("OAIY_SPLIT_LOG").map(|_| (std::time::Instant::now(), Default::default())),
            };
            // each chunk's first layers gone as the first runs the chunk before's, whose last ones then go to the
            // second as it runs the one before that
            let (mut firsts, mut seconds) = (VecDeque::new(), VecDeque::new());
            // the chunks whose hidden states are back for the prediction layer's cache, and its work on the first
            // after that one's last chunk
            let mut due = Vec::new();
            let mut tail = Vec::new();
            for i in 0..n {
                firsts.push_back(run.first(i, &mut due));
                if i >= 1 {
                    seconds.push_back(run.hand_off(firsts.pop_front().expect("a chunk"), kv));
                }
                if i >= 2 {
                    run.finish(seconds.pop_front().expect("a chunk"), kv, &mut due, &mut logits);
                }
            }
            seconds.push_back(run.hand_off(firsts.pop_front().expect("a chunk"), kv));
            while let Some(r) = seconds.pop_front() {
                run.finish(r, kv, &mut due, &mut logits);
                if !due.is_empty() {
                    tail.push(run.mtp_only(&mut due));
                }
            }
            for r in tail {
                r.finish();
            }
            if let Some((t, l)) = &run.log {
                eprintln!("split run of {n} chunks, {:.1} ms:\n  {}", t.elapsed().as_secs_f64() * 1e3, l.borrow().join("\n  "));
            }
            tapped = run.kept.take();
        }
        g1.upto = end;
        kv.len = end;
        kv.dirty_from = usize::MAX;
        self.runs.fetch_add(n, Ordering::Relaxed);
        (Tensor::from_vec(logits.expect("the last chunk's logits"), vec![1, s.vocab]), tapped)
    }

    /// Whether the chain drafts and checks tokens (the model's multi-token-prediction layer on the device).
    pub(crate) fn drafts(&self, m: &Qwen35Model) -> bool {
        std::env::var_os("OAIY_NO_CHAIN").is_none() && self.state(m).is_some_and(|st| st.spec.is_some())
    }

    /// A check of `rows` tokens (`embeds` `[rows, d]`: the token sampled, then its drafts; at most [`SPEC_ROWS`]) after
    /// what `kv` holds: every row's logits (`[1, vocab]` each). The run is kept undoable ([`Self::rollback`]).
    pub(crate) fn check(&self, m: &Qwen35Model, embeds: &Tensor, rows: usize, kv: &mut KvCache) -> Option<Vec<Tensor>> {
        if !self.drafts(m) || rows == 0 || rows > SPEC_ROWS || embeds.numel() != rows * m.config.embedding_dim || kv.len + rows > kv.max_len {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        let h = embeds.to_host();
        Some(self.run(m, st, chain, h.data(), rows, kv, true))
    }

    /// Undo a check's rows past its first `keep` (the sampled token and the drafts accepted): each delta net's state
    /// and conv window as they were before it, its first `keep` rows run through them again, and the caches cut back.
    pub(crate) fn rollback(&self, m: &Qwen35Model, kv: &mut KvCache, rows: usize, keep: usize) {
        let (Some(st), Some(chain)) = (self.state(m), m.backend.chain()) else { return };
        let Some(sp) = &st.spec else { return };
        assert!(keep >= 1 && keep <= rows && rows <= kv.len, "a rollback of {rows} rows to {keep}");
        if keep == rows {
            return;
        }
        let s = st.dims;
        let delta = DeltaNet { rows: keep, v_heads: s.nv, k_heads: s.nk, k_dim: s.dk, v_dim: s.dv, scale_q: 1.0 / (s.dv as f32).sqrt(), eps: m.config.rms_eps, sigmoid_gate: false };
        let mut rec = chain.begin();
        for (slot, &l) in st.ssm_layers.iter().enumerate() {
            let (Some(state), Some(conv)) = (kv.ssm_state[l].as_ref().and_then(|t| chain.aliased(t)), kv.ssm_conv[l].as_ref().and_then(|t| chain.aliased(t))) else {
                unreachable!("a check left layer {l}'s state the chain's")
            };
            let Mixer::Ssm { conv_w, a, dt, norm, .. } = &st.layers[l].mixer else { unreachable!("layer {l} is a delta net") };
            let (bs, bc) = &sp.backups[slot];
            let (qkv, ba) = &sp.inputs[slot];
            rec.copy(bs, 0, &state, 0, state.len);
            rec.copy(bc, 0, &conv, 0, conv.len);
            rec.ssm_conv(qkv, conv_w, &conv, &sp.scratch_conv, keep, s.ch, s.kern);
            // the outputs are not wanted: the check's were the accepted rows' already
            rec.delta_net(&sp.scratch_conv, &sp.scratch_conv, ba, a, dt, norm, &state, &sp.scratch_core, delta);
        }
        rec.finish();
        kv.len -= rows - keep;
        let mut at = sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        at.1 = at.1.min(keep);
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
        let mut g = st.kv.lock().unwrap_or_else(|p| p.into_inner());
        reserve(chain, &mut g, &s, st.attention_layers.len(), past + t);
        sync(chain, &mut g, &s, &st.attention_layers, kv, past);
        // the layers' rows from `past` on are written here: their halves' rows end there; a step deep in a long cache
        // reads the halves, brought up to its own row as it goes
        g.halved = g.halved.min(past);
        // (a step's, or a few rows': a check of drafted tokens, each row's attention then a step's; OAIY_FEW_ROWS_F32:
        // a few rows' by their own kernel over the cache as it is)
        let few = t > 1 && t <= SPEC_ROWS && std::env::var_os("OAIY_FEW_ROWS_F32").is_none();
        let mut halves = (t == 1 || few) && past + t >= halves_from() && chain.attention_halves(s.n_h, s.n_kv, s.hd);
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
        let bound = Bound { kvl: &g.layers, half: halves.then(|| (&g.half[..], halved)), one: (halves && few).then(|| (&st.step.qn, &g.out)), cap: g.cap, states: &states, w, attn: &attn, taps: &kept, seg };
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
        for slot in 0..st.attention_layers.len() {
            rec.read_range(&g.layers[slot], past * row, t * row);
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
        Qwen35Run { rec, layers: st.attention_layers.clone(), past, t, vocab: s.vocab, set: set.map(|v| (&st.prompt_sets, v)), taps }
    }

    /// The prediction layer's cache at a prompt's chunk (`t` rows at `past`, its embeddings `emb`, its hidden states
    /// after the output norm `hidden`): its entries for the chunk's positions but the last (whose next token the chunk
    /// does not have), each from the row's hidden state and the next row's token, where its entries run unbroken from
    /// the cache's start up to the chunk (else its entries start over at the chunk). Where they run up to the position
    /// before it, and the run before kept its hidden state there (the chunk before's last row), that position's entry
    /// too: its next token is the chunk's first.
    #[allow(clippy::too_many_arguments)]
    fn mtp_prompt(&self, m: &Qwen35Model, st: &State, sp: &Spec, chain: &dyn DeviceChain, rec: &mut dyn ggml_rs::ChainRecorder, emb: &[f32], hidden: &DeviceVec, t: usize, past: usize, owner: u64) {
        let s = st.dims;
        let mtp = m.mtp.as_ref().expect("a drafting chain's model has its layer");
        let mut g = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        mtp_reserve(chain, &mut g, &s, past + t);
        let (hid_at, hid_rows) = *sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        let carry = past > 0 && g.owner == owner && g.start == 0 && g.valid + 1 == past && hid_rows > 0 && hid_at + hid_rows == past;
        if !carry && (g.owner != owner || g.valid != past || g.start != 0) {
            // a chunk's attention reaches back to the cache's start: entries from 0, or none
            if past != 0 {
                g.owner = 0;
                g.valid = 0;
                return;
            }
            g.start = 0;
        }
        g.owner = owner;
        let lead = usize::from(carry);
        let rows = t - 1 + lead;
        if rows == 0 {
            return;
        }
        let at = past - lead;
        // the next tokens' embeddings and the hidden states (the kept one carried, then the chunk's rows but its last),
        // in a prompt's own vectors
        let v = |n: usize| chain.vec(n);
        let (e, en, hn, cat, x, xn) = (v(rows * s.d), v(rows * s.d), v(rows * s.d), v(rows * 2 * s.d), v(rows * s.d), v(rows * s.d));
        chain.upload(&e, &emb[(1 - lead) * s.d..t * s.d]);
        let eps = m.config.rms_eps;
        let hid = if carry {
            let h = v(rows * s.d);
            rec.copy(&sp.hid, (hid_rows - 1) * s.d, &h, 0, s.d);
            rec.copy(hidden, 0, &h, s.d, (t - 1) * s.d);
            h
        } else {
            first(hidden, rows * s.d)
        };
        rec.rmsnorm_rows(&e, &sp.enorm, &en, rows, eps);
        rec.rmsnorm_rows(&hid, &sp.hnorm, &hn, rows, eps);
        // (each row's two halves side by side: two dispatches, where a copy a half a row was 2,046 a chunk of 1,024,
        // each with its parameters to make)
        rec.store_rows(&en, &cat, rows, s.d, 0, 2 * s.d, 0);
        rec.store_rows(&hn, &cat, rows, s.d, 0, 2 * s.d, s.d);
        rec.matmul_rows(quant(&mtp.eh_proj), &cat, &x, rows);
        // its K and V into its cache (as its block makes them: the rest of the block, whose output nothing reads
        // here, not run)
        let Qwen35Block::Attention { attn_k, attn_v, .. } = &mtp.block else { unreachable!("the prediction layer attends") };
        let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
        let (k, vv, kn, table) = (v(rows * kvd), v(rows * kvd), v(rows * kvd), v(rows * s.rot));
        chain.upload(&table, &rope_table(m.config.rope_theta, s.rot, at, rows));
        rec.rmsnorm_rows(&x, &sp.attn_norm, &xn, rows, eps);
        rec.matmul_rows(quant(attn_k), &xn, &k, rows);
        rec.matmul_rows(quant(attn_v), &xn, &vv, rows);
        rec.rmsnorm_rows(&k, &sp.k_norm, &kn, rows * s.n_kv, eps);
        rec.rope_partial_rows(&kn, rows, s.n_kv, s.hd, s.rot, &table);
        rec.store_rows(&kn, &g.layer, rows, kvd, at, row, 0);
        rec.store_rows(&vv, &g.layer, rows, kvd, at, row, kvd);
        g.valid = at + rows;
    }

    /// Draft up to `k` tokens after `next` (the token sampled for position `kv.len`): the prediction layer's entries
    /// caught up to it first (each position from the trunk's hidden state there, kept from the last run, and the token
    /// after, `tokens` the tokens at positions `hidden's first + 1..=kv.len`, ending with `next`), its last giving the
    /// first draft; each further draft from the layer's own output and the draft before. A draft the layer gives less
    /// than [`DRAFT_MIN_P`] ends them (none where the first is such): a check's rows cost, and an unlikely draft is
    /// seldom taken. None where the chain does not draft or the hidden states it needs are not kept.
    pub(crate) fn draft(&self, m: &Qwen35Model, kv: &KvCache, tokens: &[u32], k: usize) -> Option<Vec<u32>> {
        if !self.drafts(m) || k == 0 {
            return None;
        }
        let st = self.state(m)?;
        let chain = m.backend.chain()?;
        let sp = st.spec.as_ref()?;
        let s = st.dims;
        let mtp = m.mtp.as_ref()?;
        let n = kv.len;
        let (hid_at, hid_rows) = *sp.hid_at.lock().unwrap_or_else(|p| p.into_inner());
        // the rows the hidden states cover, up to the trunk's last position
        if hid_rows == 0 || hid_at + hid_rows != n || tokens.len() < hid_rows {
            return None;
        }
        let tokens = &tokens[tokens.len() - hid_rows..];
        let mut g = sp.kv.lock().unwrap_or_else(|p| p.into_inner());
        mtp_reserve(chain, &mut g, &s, n + k + 1);
        // entries from the hidden states' first row on; before it, the run of true ones if it reaches it, else none
        if g.owner != kv.id || g.valid < hid_at || g.valid > n {
            g.start = hid_at;
        }
        g.owner = kv.id;
        let eps = m.config.rms_eps;
        let wk = &sp.work;
        let mut drafts = Vec::with_capacity(k);
        // pass 0: the caught-up rows (the hidden states kept); then a row a draft (the layer's own output)
        for pass in 0..k {
            let (rows, at) = if pass == 0 { (hid_rows, hid_at) } else { (1, n + pass - 1) };
            let next_tokens: Vec<u32> = if pass == 0 { tokens.to_vec() } else { vec![drafts[pass - 1]] };
            let e = m.embed_text(&next_tokens).to_host();
            chain.upload(&wk.e, e.data());
            chain.upload(&wk.table, &rope_table(m.config.rope_theta, s.rot, at, rows));
            let (ev, hv, env, hnv, cat) = (first(&wk.e, rows * s.d), first(&wk.h, rows * s.d), first(&wk.en, rows * s.d), first(&wk.hn, rows * s.d), first(&wk.cat, rows * 2 * s.d));
            let mut rec = chain.begin();
            if pass == 0 {
                rec.copy(&sp.hid, 0, &hv, 0, rows * s.d);
            } else {
                rec.copy(&wk.last, 0, &hv, 0, s.d);
            }
            rec.rmsnorm_rows(&ev, &sp.enorm, &env, rows, eps);
            rec.rmsnorm_rows(&hv, &sp.hnorm, &hnv, rows, eps);
            rec.store_rows(&env, &cat, rows, s.d, 0, 2 * s.d, 0);
            rec.store_rows(&hnv, &cat, rows, s.d, 0, 2 * s.d, s.d);
            let w = MtpRows::of(wk, &s, rows);
            rec.matmul_rows(quant(&mtp.eh_proj), &cat, &w.x, rows);
            mtp_block(m, mtp, sp, &s, &mut *rec, &g, &w, rows, at, Some((&wk.q1, g.start)));
            // the head of the last row: the next draft; the layer's own output of it, the next pass's hidden state
            rec.copy(&w.x, (rows - 1) * s.d, &wk.last, 0, s.d);
            let xn1 = first(&wk.xn, s.d);
            rec.rmsnorm_rows(&wk.last, &sp.head_norm, &xn1, 1, eps);
            rec.matmul(quant(&m.output), &xn1, &wk.logits);
            // the draft and its probability under the layer (the largest logit's share of their exponentials)
            rec.argmax_softmax(&wk.logits, &wk.best);
            rec.read(&wk.best);
            let got = rec.finish().pop().expect("the draft");
            let (best, total) = (got[0].to_bits(), got[2] as f64);
            if pass == 0 {
                // the layer's entries are its own up to the trunk's last position, the draft taken or not
                g.valid = n;
            }
            if 1.0 / total < DRAFT_MIN_P {
                break;
            }
            drafts.push(best);
        }
        Some(drafts)
    }
}

/// `v`'s first `len` elements, as a vector of their own (the same buffer): a chain's ops take their sizes from their
/// vectors' lengths, and the layer's are kept for [`SPEC_ROWS`] rows.
fn first(v: &DeviceVec, len: usize) -> DeviceVec {
    assert!(len <= v.len, "{len} of a vector of {}", v.len);
    DeviceVec { len, inner: Arc::clone(&v.inner) }
}

/// The prediction layer's vectors for a pass's rows.
struct MtpRows {
    x: DeviceVec,
    xn: DeviceVec,
    qfull: DeviceVec,
    q: DeviceVec,
    gate: DeviceVec,
    k: DeviceVec,
    vv: DeviceVec,
    qn: DeviceVec,
    kn: DeviceVec,
    att: DeviceVec,
    gated: DeviceVec,
    proj: DeviceVec,
    ffa: DeviceVec,
    ffb: DeviceVec,
    act: DeviceVec,
    table: DeviceVec,
}

impl MtpRows {
    /// A pass's `rows` of the kept vectors.
    fn of(wk: &MtpWork, s: &Dims, rows: usize) -> MtpRows {
        let qh = s.n_h * s.hd;
        MtpRows {
            x: first(&wk.x, rows * s.d),
            xn: first(&wk.xn, rows * s.d),
            qfull: first(&wk.qfull, rows * 2 * qh),
            q: first(&wk.q, rows * qh),
            gate: first(&wk.gate, rows * qh),
            k: first(&wk.k, rows * s.n_kv * s.hd),
            vv: first(&wk.v, rows * s.n_kv * s.hd),
            qn: first(&wk.qn, rows * qh),
            kn: first(&wk.kn, rows * s.n_kv * s.hd),
            att: first(&wk.att, rows * qh),
            gated: first(&wk.gated, rows * qh),
            proj: first(&wk.proj, rows * s.d),
            ffa: first(&wk.ffa, rows * 2 * s.ff),
            ffb: first(&wk.ffb, rows * s.ff),
            act: first(&wk.act, rows * s.ff),
            table: first(&wk.table, rows * s.rot),
        }
    }
}

/// Room in the prediction layer's cache for `needed` rows (its rows kept).
fn mtp_reserve(chain: &dyn DeviceChain, g: &mut MtpKv, s: &Dims, needed: usize) {
    if g.cap >= needed {
        return;
    }
    let row = 2 * s.n_kv * s.hd;
    let cap = needed.next_power_of_two().max(256);
    g.layer = if g.cap == 0 { chain.vec(cap * row) } else { chain.resize(&g.layer, cap * row) };
    g.out = chain.vec(chain.attention_out_len(s.n_h, s.hd, cap));
    g.cap = cap;
}

/// The prediction layer's block on `rows` rows of `w.x` at positions `at..at + rows` (its RoPE table in `w.table`):
/// its attention over its cache (each row over the positions before it from `lo`, a row at a time, where `one` gives
/// a query's vector and `lo`; else a prompt's from the cache's start), then its FFN, `w.x` the block's output.
#[allow(clippy::too_many_arguments)]
fn mtp_block(m: &Qwen35Model, mtp: &crate::qwen35::Qwen35Mtp, sp: &Spec, s: &Dims, rec: &mut dyn ggml_rs::ChainRecorder, g: &MtpKv, w: &MtpRows, rows: usize, at: usize, one: Option<(&DeviceVec, usize)>) {
    let Qwen35Block::Attention { attn_q, attn_k, attn_v, attn_output, ffn_pair, ffn_down, .. } = &mtp.block else { unreachable!("the prediction layer attends") };
    let eps = m.config.rms_eps;
    let (kvd, row) = (s.n_kv * s.hd, 2 * s.n_kv * s.hd);
    let scale = 1.0 / (s.hd as f32).sqrt();
    let t = rows;
    rec.rmsnorm_rows(&w.x, &sp.attn_norm, &w.xn, t, eps);
    rec.matmul_rows(quant(attn_q), &w.xn, &w.qfull, t);
    rec.matmul_rows(quant(attn_k), &w.xn, &w.k, t);
    rec.matmul_rows(quant(attn_v), &w.xn, &w.vv, t);
    rec.copy_cols(&w.qfull, &w.q, t * s.n_h, s.hd, 2 * s.hd, 0);
    rec.copy_cols(&w.qfull, &w.gate, t * s.n_h, s.hd, 2 * s.hd, s.hd);
    rec.rmsnorm_rows(&w.q, &sp.q_norm, &w.qn, t * s.n_h, eps);
    rec.rmsnorm_rows(&w.k, &sp.k_norm, &w.kn, t * s.n_kv, eps);
    rec.rope_partial_rows(&w.qn, t, s.n_h, s.hd, s.rot, &w.table);
    rec.rope_partial_rows(&w.kn, t, s.n_kv, s.hd, s.rot, &w.table);
    rec.store_rows(&w.kn, &g.layer, t, kvd, at, row, 0);
    rec.store_rows(&w.vv, &g.layer, t, kvd, at, row, kvd);
    let qh = s.n_h * s.hd;
    match one {
        Some((q1, lo)) => {
            for r in 0..t {
                rec.copy(&w.qn, r * qh, q1, 0, qh);
                rec.attention(q1, &g.layer, &g.out, s.n_h, s.n_kv, s.hd, lo.min(at + r), at + r + 1, g.cap, scale);
                rec.copy(&g.out, 0, &w.att, r * qh, qh);
            }
        }
        None => rec.attention_rows(&w.qn, &g.layer, &w.att, t, s.n_h, s.n_kv, s.hd, at, None, scale),
    }
    rec.mul_sigmoid(&w.att, &w.gate, &w.gated, t * qh);
    rec.matmul_rows(quant(attn_output), &w.gated, &w.proj, t);
    rec.add_rmsnorm_rows(&w.x, &w.proj, &sp.post_norm, &w.xn, t, eps);
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
