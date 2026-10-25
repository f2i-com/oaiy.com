//! A model's few steps between its projections, recorded with them as one call ([`run_program`]): a decode step of
//! DeepSeek-V4.1 made a round trip of each sublayer's projections, its hyper-connections' mixing (four copies of the
//! residual stream collapsed into a sublayer's input and spread back after it) and its norms on the host between them.
//! A program is the list of its steps, each making a vector (or writing into one) that later steps take by its step's
//! number, and the vectors read back at the end.

use super::*;

/// A step of a [`run_program`] program. A step's vector is known by the step's number; a step that writes into
/// another's vector ([`ProgramStep::CopyColsBf16`]) makes none of its own.
pub enum ProgramStep<'a> {
    /// The host's values.
    Input(&'a [f32]),
    /// `rows` rows of a vector of `len` values, written by the steps after.
    Scratch(usize),
    /// `m` of `rows` rows of vector `x` (`[rows, k]`): its sums, `[rows, n]`.
    Mul { m: ChainMat<'a>, x: usize, rows: usize },
    /// `jobs` of a group (`(part, row of x)` each, x's rows `k` wide) in one pass: each job's sums, `[jobs, n]`.
    MulGroup { g: GroupMat<'a>, x: usize, jobs: &'a [(u32, u32)] },
    /// `width` columns from `src_at` of `rows` rows of `src` (rows `src_stride` apart), each rounded to bf16, into
    /// `dst` from `dst_at` (rows `dst_stride` apart).
    CopyColsBf16 { src: usize, src_stride: usize, src_at: usize, dst: usize, dst_stride: usize, dst_at: usize, width: usize, rows: usize },
    /// The host's rmsnorm of `rows` rows of `x` (`width` each) by `weight`, its output rounded to bf16, to the bit
    /// (`chained_normed`'s norm).
    Norm { x: usize, weight: &'a [f32], eps: f32, rows: usize, width: usize },
    /// A hyper-connection's spread of a sublayer's output: `out` (`[d]`) and the residual's copies `res` (`[HC, d]`)
    /// by `mix` (`[pre 4, post 4, comb 4 x 4]`): copy `j`'s `bf16(post[j] out + sum_i comb[i][j] res[i])`, `[HC, d]`.
    HcPost { out: usize, res: usize, mix: usize, d: usize },
    /// A hyper-connection's collapse of the copies `h` (`[HC, d]`) into a sublayer's input by `mix`'s pre:
    /// `bf16(sum_i pre[i] h[i])`, `[d]`.
    HcPre { h: usize, mix: usize, d: usize },
    /// A gated unit's activation of `gate` and `up` (`len` each, rounded to bf16 first, as the host rounds a gated
    /// unit's projections): `bf16(silu[bits(g) >> 16] * clamp(u, ±limit))` (`limit` > 0; no clamp else), `silu` the
    /// host's value of every bf16 gate by its bits (`min(g, limit)` taken in them, 65,536 values), so each element is
    /// the host's to the bit, `[len]`.
    SwigluBf16 { gate: usize, up: usize, len: usize, silu: &'a [f32], limit: f32 },
    /// A layer's routed experts its card is to make in the router's own call: the `k` best of `logits` (`bias.len()` of
    /// them) by the router's score, `sqrt(softplus(l)) + bias`, as the card makes it (its own exponential: a guess of the
    /// host's ranking, near it and not always its order), each looked up in `table` (each expert's slot in the pool the
    /// [`ProgramStep::Held`] after it takes, `u32::MAX` where that pool does not hold it). Its vector, as u32 bits: the
    /// held pass's down jobs (`(slot, job)` each), then the `k` experts, then their slots, `[4k]`.
    Candidates { logits: usize, bias: &'a [f32], table: &'a DeviceVec, k: usize },
    /// The held experts of `cands` (a [`ProgramStep::Candidates`] step) in `slots`, on row 0 of `x` (`[hidden]`): each
    /// one's output, unweighted, as the slots' `forward` makes it (to the bit), `[k, hidden]`; a candidate the pool does
    /// not hold skipped (its row whatever it was).
    Held { slots: HeldSlots<'a>, x: usize, cands: usize, limit: f32 },
    /// A vector the caller keeps on the device from run to run (a layer's caches, the rotary tables): the step's vector
    /// is it, read and written in place by the steps that name it.
    State(&'a DeviceVec),
    /// The host's rotary rotation in place (DeepSeek-V4.1's `Rope::apply`, to the bit): `rows` rows of `x` (`stride`
    /// apart), the `2 * half` values from `at` of each as pairs, to the position the word `word` of `pos` (an input of
    /// u32 words, each step's numbers) says, or back from it (`inverse`), by the tables `cos` and `sin` (`[positions,
    /// half]`), each value rounded to bf16.
    Rope { x: usize, rows: usize, stride: usize, at: usize, half: usize, cos: usize, sin: usize, pos: usize, word: u32, inverse: bool },
    /// The host's fp8 round trip of `len` values of `x` in place (`fake_quant_fp8_inplace`, blocks of 32), to the bit.
    Fp8 { x: usize, len: usize },
    /// The host's fp4 round trip of `len` values of `x` in place (`fake_quant_fp4_inplace`), blocks of `block` (16 or
    /// 32), each's scale on the e4m3 grid (`e4m3`) or a power of two, to the bit.
    Fp4 { x: usize, len: usize, block: usize, e4m3: bool },
    /// `width` values of `src` into `dst`'s row the word `word` of `pos` says (rows `width` apart).
    PutRow { src: usize, dst: usize, width: usize, pos: usize, word: u32 },
    /// The host's softmax-gated pool of `ratio` rows of `kv` by `score` (`[ratio, width]` each), rounded to bf16, its
    /// exponentials the device's own: `[width]`.
    Pool { kv: usize, score: usize, ratio: usize, width: usize },
    /// The host's index scores (`index_scores`, to the bit) of `heads` queries of `dim` (`q`) against the first keys of
    /// `keys` (`[.., dim]`) the word `word` of `pos` counts, the heads weighed by `weights` (`[heads]`, each rounded to
    /// bf16 after `scale`): `[keys' rows]`, those past the count as they were.
    IndexScores { q: usize, keys: usize, weights: usize, heads: usize, dim: usize, scale: f32, pos: usize, word: u32 },
    /// The `k` best of the first `scores` the word `word` of `pos` counts (the host's order: total, ties to the lower
    /// position), their positions in order into `dst` (u32 bits), `u32::MAX` past them; every one where there are no
    /// more than `k`.
    TopK { scores: usize, dst: usize, k: usize, pos: usize, word: u32 },
    /// A candidate source's blocks (DeepSeek-V4.1's `candidate_mask` of a decode step): of the first `scores` the word
    /// `word` of `pos` counts, each block of `bs`'s best, the newest block's taken as the best of all, and the `k` best
    /// blocks by the host's order (a [`ProgramStep::TopK`] of them) flagged in `blocks` (a [`ProgramStep::State`] of u32
    /// words: the blocks' count, then a block's flag each, 1 kept): `[blocks' most]`, each block's best.
    BlockKeep { scores: usize, blocks: usize, k: usize, bs: usize, pos: usize, word: u32 },
    /// The first `scores` the word `word` of `pos` counts that lie in a block `blocks` does not keep (a
    /// [`ProgramStep::BlockKeep`]'s, blocks of `bs`) made -infinity in place, as the host masks a later index source's.
    MaskBlocks { scores: usize, blocks: usize, bs: usize, pos: usize, word: u32 },
    /// A decode step's attention of `heads` queries of `dim` (`q`, rotated) over the window's `win` slots (`window`,
    /// `[win, dim]`: the ring, the position the word `word` of `pos` says) and, where `compressed` is `(rows, picks,
    /// count)`, the `count` rows of `rows` that `picks` names (u32, `u32::MAX` none): the host's `sparse_attend`, with
    /// `sink` (`[heads]`) in the denominators and `scale` on the scores (to the bit), its exponentials the device's own
    /// and its sums a chunk of 64 positions at a time: `[heads, dim]`, rounded to bf16 and not rotated back.
    Attend { q: usize, heads: usize, dim: usize, window: usize, win: usize, compressed: Option<(usize, usize, usize)>, sink: &'a [f32], scale: f32, pos: usize, word: u32 },
    /// [`ProgramStep::Rope`] of a prompt's rows: row `r` of `x` at the position `pos`'s word `r / group` says (a token's
    /// `group` rows: its heads'), the positions an input of u32 words, a token's each.
    RopeRows { x: usize, rows: usize, stride: usize, at: usize, half: usize, cos: usize, sin: usize, pos: usize, group: usize, inverse: bool },
    /// A prompt's attention: `tokens` tokens' `heads` queries of `dim` (`q`, rotated), each token over its own positions
    /// `idxs` (`[tokens, len]`, u32 rows of `kv` (`[.., dim]`) in the host's order, `u32::MAX` none), `sink` (`[heads]`)
    /// in the denominators and `scale` on the scores: the host's `sparse_attend` of each (token, head), its scores and
    /// sums in the host's order, its exponentials the device's own: `[tokens, heads, dim]`, rounded to bf16 and not
    /// rotated back.
    AttendRows { q: usize, kv: usize, idxs: usize, tokens: usize, heads: usize, dim: usize, len: usize, sink: &'a [f32], scale: f32 },
    /// The host's mixing coefficients of a hyper-connected stream (`n` wide) from its 24 projections `proj` and its sum
    /// of squares, the word `word` of `pos` (the host's: `mixes_of`, by `base` and `scale`: its scale exact, the
    /// coefficients' exponentials and Sinkhorn's divisions the device's own), laid out as a [`ProgramStep::HcPost`]
    /// takes them: `[24]`.
    HcMix { proj: usize, pos: usize, word: u32, base: &'a [f32], scale: &'a [f32], n: usize, norm_eps: f32, iters: usize, hc_eps: f32 },
}

/// The parts a [`ProgramStep::MulGroup`] multiplies: an EXL3 checkpoint's, or a GGUF's.
#[derive(Clone, Copy)]
pub enum GroupMat<'a> {
    Exl3(&'a Exl3Group),
    Quant(&'a crate::quant::moe::QuantGroup),
}

/// A [`ProgramStep::MulGroup`]'s vectors beside its input and output, as its parts' kind takes them.
enum GroupMatScratch {
    Exl3(GroupScratch),
    Quant(crate::quant::moe::QuantGroupScratch),
}

impl GroupMat<'_> {
    fn kn(&self) -> (usize, usize) {
        match self {
            GroupMat::Exl3(g) => g.kn(),
            GroupMat::Quant(g) => g.kn(),
        }
    }

    fn count(&self) -> usize {
        match self {
            GroupMat::Exl3(g) => g.count(),
            GroupMat::Quant(g) => g.count(),
        }
    }

    fn words(&self) -> &wgpu::Buffer {
        match self {
            GroupMat::Exl3(g) => g.words(),
            GroupMat::Quant(g) => g.words(),
        }
    }

    fn scratch_kept(&self, b: &WgpuBackend, jobs: &[(u32, u32)]) -> GroupMatScratch {
        match self {
            GroupMat::Exl3(g) => GroupMatScratch::Exl3(g.scratch_kept(b, jobs)),
            GroupMat::Quant(g) => GroupMatScratch::Quant(g.scratch_kept(b, jobs)),
        }
    }

    fn record(&self, rec: &mut crate::chain::Recorder<'_>, s: &GroupMatScratch, x: &DeviceVec, y: &DeviceVec) {
        match (self, s) {
            (GroupMat::Exl3(g), GroupMatScratch::Exl3(s)) => g.record(rec, s, x, y),
            (GroupMat::Quant(g), GroupMatScratch::Quant(s)) => g.record(rec, s, x, y),
            _ => unreachable!("a group's scratch is of its parts' kind"),
        }
    }
}

/// The slots a [`ProgramStep::Held`] makes its experts from: an EXL3 checkpoint's, or a GGUF's.
#[derive(Clone, Copy)]
pub enum HeldSlots<'a> {
    Exl3(&'a Exl3Slots),
    Quant(&'a crate::quant::moe::QuantSlots),
}

/// A [`ProgramStep::Held`]'s vectors beside its inputs and output, as its slots' kind takes them.
enum HeldScratch {
    Exl3(FusedScratch),
    Quant(crate::quant::moe::QuantFused),
}

impl HeldSlots<'_> {
    fn on(&self, b: &WgpuBackend) -> bool {
        match self {
            HeldSlots::Exl3(s) => s.on(b),
            HeldSlots::Quant(s) => s.on(b),
        }
    }

    fn hidden(&self) -> usize {
        match self {
            HeldSlots::Exl3(s) => s.hidden(),
            HeldSlots::Quant(s) => s.hidden(),
        }
    }

    fn words(&self) -> &wgpu::Buffer {
        match self {
            HeldSlots::Exl3(s) => s.words(),
            HeldSlots::Quant(s) => s.words(),
        }
    }

    fn fused_scratch(&self, k: usize) -> HeldScratch {
        match self {
            HeldSlots::Exl3(s) => HeldScratch::Exl3(s.fused_scratch(k)),
            HeldSlots::Quant(s) => HeldScratch::Quant(s.fused_scratch(k)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_fused(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, jgu: &DeviceVec, jd: &DeviceVec, k: usize, s: &HeldScratch, out: &DeviceVec, limit: f32) {
        match (self, s) {
            (HeldSlots::Exl3(slots), HeldScratch::Exl3(s)) => slots.record_fused(rec, x, jgu, jd, k, s, out, limit),
            (HeldSlots::Quant(slots), HeldScratch::Quant(s)) => slots.record_fused(rec, x, jgu, jd, k, s, out, limit),
            _ => unreachable!("a held step's scratch is of its slots' kind"),
        }
    }
}

/// [`ProgramStep::Candidates`]' kernel: one workgroup, a thread an expert (up to 512): each one's score, then its rank
/// among them (ties to the lower index), the `k` best's jobs written.
const CANDIDATES: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(6) var<storage, read_write> jd: array<u32>;
@group(0) @binding(7) var<storage, read_write> jgu: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> score: array<f32, 512>;

@compute @workgroup_size(512)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    if (t < n) {
        let v = logits[t];
        var sp = v;
        if (v < 20.0) { sp = log(1.0 + exp(v)); }
        score[t] = sqrt(sp) + bias[t];
    }
    workgroupBarrier();
    if (t >= n) { return; }
    let s = score[t];
    var rank = 0u;
    for (var i = 0u; i < n; i++) {
        let o = score[i];
        if (o > s || (o == s && i < t)) { rank++; }
    }
    if (rank >= k) { return; }
    let slot = table[t];
    let none = 0xffffffffu;
    jd[2u * rank] = slot;
    jd[2u * rank + 1u] = rank;
    jd[2u * k + rank] = t;
    jd[3u * k + rank] = slot;
    jgu[4u * rank] = select(2u * slot, none, slot == none);
    jgu[4u * rank + 1u] = 0u;
    jgu[4u * rank + 2u] = select(2u * slot + 1u, none, slot == none);
    jgu[4u * rank + 3u] = 0u;
}
"#;

/// `x` as an input of `steps`: the step of an input of the same host rows already there (a layer's projections of x
/// take one upload), else a new one. Its step's number.
pub(super) fn input_once<'a>(steps: &mut Vec<ProgramStep<'a>>, x: &'a [f32]) -> usize {
    let same = steps.iter().position(|s| matches!(s, ProgramStep::Input(y) if y.as_ptr() == x.as_ptr() && y.len() == x.len()));
    same.unwrap_or_else(|| {
        steps.push(ProgramStep::Input(x));
        steps.len() - 1
    })
}

/// The copies of a hyper-connected residual stream (DeepSeek-V4.1's `hc_mult`).
const HC: usize = 4;

/// [`ProgramStep::HcPost`]'s kernel: a thread an element of `out`, each copy's sum in the host's order.
const HC_POST: &str = r#"
@group(0) @binding(0) var<storage, read> out: array<f32>;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read> mix: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let k = id.x;
    if (k >= d) { return; }
    let o = out[k];
    let r0 = res[k];
    let r1 = res[d + k];
    let r2 = res[2u * d + k];
    let r3 = res[3u * d + k];
    for (var j = 0u; j < 4u; j++) {
        let r = mix[8u + j] * r0 + mix[12u + j] * r1 + mix[16u + j] * r2 + mix[20u + j] * r3;
        y[j * d + k] = bf16(mix[4u + j] * o + r);
    }
}
"#;

/// [`ProgramStep::SwigluBf16`]'s kernel: a thread an element, `p[0].y` the limit's bits.
const SWIGLU_BF16: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> silu: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p[0].x) { return; }
    let lim = bitcast<f32>(p[0].y);
    var u = bf16(up[i]);
    if (lim > 0.0) { u = clamp(u, -lim, lim); }
    y[i] = bf16(silu[bitcast<u32>(bf16(gate[i])) >> 16u] * u);
}
"#;

/// [`ProgramStep::HcPre`]'s kernel: a thread an element.
const HC_PRE: &str = r#"
@group(0) @binding(0) var<storage, read> h: array<f32>;
@group(0) @binding(2) var<storage, read> mix: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = p[0].x;
    let k = id.x;
    if (k >= d) { return; }
    y[k] = bf16(mix[0] * h[k] + mix[1] * h[d + k] + mix[2] * h[2u * d + k] + mix[3] * h[3u * d + k]);
}
"#;

/// Every kernel of a decode step's attention on the device ([`ProgramStep::Rope`] to [`ProgramStep::HcMix`]) made now,
/// on `b`: made as a model's first reply needed them, they were some seconds of it.
pub fn make_attention_kernels(b: &WgpuBackend) {
    let kernels: [(&'static str, &'static str); 16] = [
        ("ds-rope", attend::DS_ROPE),
        ("ds-fp8", attend::DS_FP8),
        ("ds-fp4", attend::DS_FP4),
        ("ds-put-row", attend::DS_PUT),
        ("ds-pool", attend::DS_POOL),
        ("ds-index-scores", attend::DS_INDEX_SCORES),
        ("ds-topk", attend::DS_TOPK),
        ("ds-block-best", attend::DS_BLOCK_BEST),
        ("ds-block-flags", attend::DS_BLOCK_FLAGS),
        ("ds-mask-blocks", attend::DS_MASK_BLOCKS),
        ("ds-attend-scores", attend::DS_ATTEND_SCORES),
        ("ds-attend-parts", attend::DS_ATTEND_PARTS),
        ("ds-attend-join", attend::DS_ATTEND_JOIN),
        ("ds-attend-rows", attend::DS_ATTEND_ROWS),
        ("ds-hc-mix", attend::DS_HC_MIX),
        ("exl3-rmsnorm-host", NORM_AS_HOST),
    ];
    for (name, source) in kernels {
        b.gpu.named_pipeline_wide(name, || source.to_string());
    }
}

/// Run `steps` on `b` as one recording and one read back: the vectors `reads` names (`(step, first value, values)`),
/// in order. None where a matrix is not one this device's chained recordings take, or a step names a vector it cannot
/// (a later step's, or one of another shape).
pub fn run_program(b: &WgpuBackend, steps: &[ProgramStep<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
    let (now, _) = run_program_parts(b, steps, reads, steps.len(), &[])?;
    Some(now)
}

/// The rest of a program run in two parts ([`run_program_parts`]): its reads, once its recording is done.
pub struct ProgramLater<'a> {
    launched: Option<crate::chain::Launched<'a>>,
    /// (its vectors, which the recording uses, kept until it is done)
    _arena: Arc<Mutex<ProgramArena>>,
}

impl ProgramLater<'_> {
    /// The later part's reads, in order, once it is done.
    pub fn wait(self) -> Vec<Vec<f32>> {
        self.launched.map(|l| l.wait()).unwrap_or_default()
    }
}

/// [`run_program`] with vectors of its own, made for this run and let go of after it, and recorded afresh: a prompt's
/// chunk's program (its vectors some hundreds of MB, of a shape a run seldom has again), which an arena kept for each
/// of a model's layers would hold to the end.
pub fn run_program_once(b: &WgpuBackend, steps: &[ProgramStep<'_>], reads: &[(usize, usize, usize)]) -> Option<Vec<Vec<f32>>> {
    let (now, _) = run_parts(b, steps, reads, steps.len(), &[], false)?;
    Some(now)
}

/// [`run_program`] in two recordings submitted one after the other, `steps[..split]` then the rest: the first's reads
/// (`reads`) returned once it is done, the second's (`later`) left to read when the caller wants them, the card making
/// them meanwhile (a layer's held experts after its router's call: the host routes and makes the rest of the layer's
/// experts while the card makes those). With nothing after `split`, one recording.
pub fn run_program_parts<'a>(b: &'a WgpuBackend, steps: &[ProgramStep<'_>], reads: &[(usize, usize, usize)], split: usize, later: &[(usize, usize, usize)]) -> Option<(Vec<Vec<f32>>, ProgramLater<'a>)> {
    run_parts(b, steps, reads, split, later, true)
}

/// [`run_program_parts`]; `keep`: its vectors an arena the device keeps for the program's next run (else its own, let
/// go of after it, [`run_program_once`]).
fn run_parts<'a>(b: &'a WgpuBackend, steps: &[ProgramStep<'_>], reads: &[(usize, usize, usize)], split: usize, later: &[(usize, usize, usize)], keep: bool) -> Option<(Vec<Vec<f32>>, ProgramLater<'a>)> {
    use ggml_rs::ChainRecorder;
    let entered = std::time::Instant::now();
    // each step's vector's length (0: it makes none), checked against what the steps after it ask of it
    let mut lens: Vec<usize> = Vec::with_capacity(steps.len());
    let mut rows_one = true;
    for (i, step) in steps.iter().enumerate() {
        let before = |v: usize| (v < i && lens[v] > 0).then(|| lens[v]);
        let len = match step {
            ProgramStep::Input(x) => x.len(),
            ProgramStep::Scratch(len) => *len,
            ProgramStep::Mul { m, x, rows } => {
                let (k, n) = chain_shape(b, m)?;
                rows_one &= *rows == 1;
                (before(*x)? >= rows * k).then_some(rows * n)?
            }
            ProgramStep::MulGroup { g, x, jobs } => {
                let (k, n) = g.kn();
                let rows = jobs.iter().map(|&(_, r)| r as usize + 1).max()?;
                (jobs.iter().all(|&(m, _)| (m as usize) < g.count()) && before(*x)? >= rows * k).then_some(jobs.len() * n)?
            }
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => {
                let ok = before(*src)? >= (rows - 1) * src_stride + src_at + width && before(*dst)? >= (rows - 1) * dst_stride + dst_at + width;
                ok.then_some(0)?
            }
            ProgramStep::Norm { x, weight, rows, width, .. } => (before(*x)? >= rows * width && weight.len() == *width && *width <= 1 << 20).then_some(rows * width)?,
            ProgramStep::HcPost { out, res, mix, d } => (before(*out)? >= *d && before(*res)? >= HC * d && before(*mix)? >= 24).then_some(HC * d)?,
            ProgramStep::HcPre { h, mix, d } => (before(*h)? >= HC * d && before(*mix)? >= 24).then_some(*d)?,
            ProgramStep::SwigluBf16 { gate, up, len, silu, .. } => (before(*gate)? >= *len && before(*up)? >= *len && silu.len() == 1 << 16).then_some(*len)?,
            ProgramStep::Candidates { logits, bias, table, k } => {
                let n = bias.len();
                (before(*logits)? >= n && (1..=512).contains(&n) && (1..=n.min(64)).contains(k) && table.len >= n).then_some(4 * k)?
            }
            ProgramStep::Held { slots, x, cands, .. } => {
                let k = before(*cands)? / 4;
                (matches!(steps[*cands], ProgramStep::Candidates { .. }) && slots.on(b) && before(*x)? >= slots.hidden()).then_some(k * slots.hidden())?
            }
            ProgramStep::State(v) => v.len,
            ProgramStep::Rope { x, rows, stride, at, half, cos, sin, pos, word, .. } => {
                let ok = *rows > 0 && *half > 0 && at + 2 * half <= *stride && before(*x)? >= (rows - 1) * stride + at + 2 * half && before(*cos)? >= *half && before(*sin)? == before(*cos)? && before(*pos)? > *word as usize;
                ok.then_some(0)?
            }
            ProgramStep::Fp8 { x, len } => (before(*x)? >= *len).then_some(0)?,
            ProgramStep::Fp4 { x, len, block, .. } => (matches!(block, 16 | 32) && before(*x)? >= *len).then_some(0)?,
            ProgramStep::PutRow { src, dst, width, pos, word } => (*width > 0 && before(*src)? >= *width && before(*dst)? >= *width && before(*pos)? > *word as usize).then_some(0)?,
            ProgramStep::Pool { kv, score, ratio, width } => (*ratio > 0 && before(*kv)? >= ratio * width && before(*score)? >= ratio * width).then_some(*width)?,
            ProgramStep::IndexScores { q, keys, weights, heads, dim, pos, word, .. } => {
                let ok = (1..=64).contains(heads) && 256 % heads == 0 && *dim > 0 && dim % 8 == 0 && heads * dim <= 4096 && before(*q)? >= heads * dim && before(*weights)? >= *heads && before(*pos)? > *word as usize;
                (ok && before(*keys)? >= *dim).then_some(before(*keys)? / dim)?
            }
            ProgramStep::TopK { scores, dst, k, pos, word } => (*k > 0 && before(*scores)? > 0 && before(*dst)? >= *k && before(*pos)? > *word as usize).then_some(0)?,
            ProgramStep::BlockKeep { scores, blocks, k, bs, pos, word } => {
                let most = before(*scores)?.div_ceil((*bs).max(1));
                (*k > 0 && *bs > 0 && before(*blocks)? > most && before(*pos)? > *word as usize).then_some(most)?
            }
            ProgramStep::MaskBlocks { scores, blocks, bs, pos, word } => (*bs > 0 && before(*blocks)? > before(*scores)?.div_ceil(*bs) && before(*pos)? > *word as usize).then_some(0)?,
            ProgramStep::Attend { q, heads, dim, window, win, compressed, sink, pos, word, .. } => {
                let count = compressed.map_or(0, |(_, _, c)| c);
                let picks = match compressed {
                    Some((rows, picks, count)) => before(*rows)? >= *dim && before(*picks)? >= *count,
                    None => true,
                };
                let ok = *heads > 0 && *dim > 0 && dim % 128 == 0 && *dim <= 512 && *win > 0 && win + count <= attend::ATTEND_MOST && sink.len() >= *heads;
                (ok && picks && before(*q)? >= heads * dim && before(*window)? >= win * dim && before(*pos)? > *word as usize).then_some(heads * dim)?
            }
            ProgramStep::HcMix { proj, pos, word, base, scale, n, .. } => (*n > 0 && before(*proj)? >= 24 && before(*pos)? > *word as usize && base.len() >= 24 && scale.len() >= 3).then_some(24)?,
            ProgramStep::RopeRows { x, rows, stride, at, half, cos, sin, pos, group, .. } => {
                // (a thread a pair, the workgroups along one axis)
                let ok = *rows > 0 && *half > 0 && *group > 0 && rows * half <= 65535 * 256 && at + 2 * half <= *stride && before(*x)? >= (rows - 1) * stride + at + 2 * half && before(*cos)? >= *half && before(*sin)? == before(*cos)?;
                (ok && before(*pos)? >= rows.div_ceil(*group)).then_some(0)?
            }
            ProgramStep::AttendRows { q, kv, idxs, tokens, heads, dim, len, sink, .. } => {
                let ok = *tokens > 0 && *heads > 0 && heads % attend::ATTEND_ROWS_HEADS == 0 && *dim > 0 && dim % 128 == 0 && *dim <= 512 && *len > 0 && *len <= attend::ATTEND_MOST && sink.len() >= *heads;
                (ok && before(*q)? >= tokens * heads * dim && before(*kv)? >= *dim && before(*idxs)? >= tokens * len).then_some(tokens * heads * dim)?
            }
        };
        lens.push(len);
    }
    if split > steps.len() || reads.iter().chain(later).any(|&(s, at, len)| s >= steps.len() || at + len > lens[s]) {
        return None;
    }
    // The program's vectors kept from its first run (an arena a program: a layer's), so its dispatches' bind groups are
    // the same every step and kept with them; the inputs written each run, the norms' weights and the groups' job lists
    // once. Taken from the recording's pooled scratch, every dispatch made its bind group (and a parameters' buffer)
    // again at every step, some 95 us of the host's work a call, and the kept groups' store started over every few
    // tokens.
    let kept = if keep { arena(b, steps, &lens) } else { Arc::new(Mutex::new(make_arena(b, steps, &lens))) };
    let mut arena = kept.lock().unwrap_or_else(|p| p.into_inner());
    // (its dispatches as its first run recorded them, where that was all it made: not recorded again)
    let replay = (keep && !crate::profile::chain_on() && !crate::profile::pieces_on()).then(|| arena.recipe.as_ref().filter(|(s, ..)| *s == split).cloned()).flatten();
    for (step, v) in steps.iter().zip(&arena.vecs) {
        if let (ProgramStep::Input(x), Some(v)) = (step, v) {
            DeviceChain::upload(b, v, x);
        }
    }
    let (vecs, norm_weights, groups, tables, fused) = (&arena.vecs, &arena.norms, &arena.groups, &arena.tables, &arena.fused);
    let v = |i: usize| match &steps[i] {
        ProgramStep::State(kept) => *kept,
        _ => vecs[i].as_ref().expect("a step's vector"),
    };
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU vector").clone();
    let (dummy, drw) = (b.gpu.dummy().clone(), b.gpu.dummy_rw().clone());
    let record = |rec: &mut crate::chain::Recorder<'_>, i: usize| {
        match &steps[i] {
            ProgramStep::Input(_) | ProgramStep::Scratch(_) => {}
            ProgramStep::Mul { m, x, rows } => {
                let kn = chain_shape(b, m).expect("checked above");
                chain_mul(rec, *m, kn, v(*x), v(i), *rows);
            }
            ProgramStep::MulGroup { g, x, .. } => g.record(rec, groups[i].as_ref().expect("its scratch"), v(*x), v(i)),
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => {
                rec.copy_cols_bf16(v(*src), *src_stride, *src_at, v(*dst), *dst_stride, *dst_at, *width, *rows);
            }
            ProgramStep::Norm { x, eps, rows, width, .. } => {
                let w = norm_weights[i].as_ref().expect("its weight");
                rec.dispatch_wide("exl3-rmsnorm-host", NORM_AS_HOST, [&buf(v(*x)), &buf(w), &dummy, &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*width as u32, 0, eps.to_bits()], (*rows as u32, 1, 1));
            }
            ProgramStep::HcPost { out, res, mix, d } => {
                rec.dispatch_wide("ds-hc-post", HC_POST, [&buf(v(*out)), &buf(v(*res)), &buf(v(*mix)), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*d as u32], ((*d as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::HcPre { h, mix, d } => {
                rec.dispatch_wide("ds-hc-pre", HC_PRE, [&buf(v(*h)), &dummy, &buf(v(*mix)), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*d as u32], ((*d as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::SwigluBf16 { gate, up, len, limit, .. } => {
                let table = tables[i].as_ref().expect("its table");
                rec.dispatch_wide("ds-swiglu-bf16", SWIGLU_BF16, [&buf(v(*gate)), &buf(v(*up)), &buf(table), &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*len as u32, limit.to_bits()], ((*len as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::Candidates { logits, bias, table, k } => {
                let (bias_v, jgu) = (norm_weights[i].as_ref().expect("its bias"), tables[i].as_ref().expect("its gate and up jobs"));
                rec.dispatch_wide("ds-candidates", CANDIDATES, [&buf(v(*logits)), &buf(bias_v), &buf(table), &dummy, &dummy, &dummy, &buf(v(i)), &buf(jgu)], &[bias.len() as u32, *k as u32], (1, 1, 1));
            }
            ProgramStep::Held { slots, x, cands, limit } => {
                let jgu = tables[*cands].as_ref().expect("its candidates' gate and up jobs");
                slots.record_fused(rec, v(*x), jgu, v(*cands), lens[*cands] / 4, fused[i].as_ref().expect("its scratch"), v(i), *limit);
            }
            ProgramStep::State(_) => {}
            ProgramStep::Rope { x, rows, stride, at, half, cos, sin, pos, word, inverse } => {
                let words = [*rows as u32, *stride as u32, *at as u32, *half as u32, *word, u32::from(*inverse)];
                rec.dispatch_wide("ds-rope", attend::DS_ROPE, [&buf(v(*cos)), &buf(v(*sin)), &buf(v(*pos)), &dummy, &dummy, &dummy, &buf(v(*x)), &drw], &words, (((rows * half) as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::Fp8 { x, len } => {
                let words = [*len as u32, (1.0f32 / 448.0).to_bits(), 1e-4f32.to_bits()];
                rec.dispatch_wide("ds-fp8", attend::DS_FP8, [&dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &buf(v(*x)), &drw], &words, ((*len as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::Fp4 { x, len, block, e4m3 } => {
                let words = [*len as u32, *block as u32, u32::from(*e4m3), 0, (6.0 * 2f32.powi(-126)).to_bits(), (6.0 * 2f32.powi(-9)).to_bits(), (1.0f32 / 6.0).to_bits()];
                rec.dispatch_wide("ds-fp4", attend::DS_FP4, [&dummy, &dummy, &dummy, &dummy, &dummy, &dummy, &buf(v(*x)), &drw], &words, ((*len as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::PutRow { src, dst, width, pos, word } => {
                rec.dispatch_wide("ds-put-row", attend::DS_PUT, [&buf(v(*src)), &dummy, &buf(v(*pos)), &dummy, &dummy, &dummy, &buf(v(*dst)), &drw], &[*width as u32, *word], ((*width as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::Pool { kv, score, ratio, width } => {
                rec.dispatch_wide("ds-pool", attend::DS_POOL, [&buf(v(*kv)), &buf(v(*score)), &dummy, &dummy, &dummy, &dummy, &buf(v(i)), &drw], &[*width as u32, *ratio as u32], ((*width as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::IndexScores { q, keys, weights, heads, dim, scale, pos, word } => {
                // (the workgroups over the keys a count says, `256 / heads` a pass each: as many as the keys there are
                // room for take, up to 256)
                let groups = (lens[i] as u32).div_ceil((256 / heads) as u32).clamp(1, 256);
                let words = [*heads as u32, *dim as u32, *word, scale.to_bits(), groups];
                rec.dispatch_wide("ds-index-scores", attend::DS_INDEX_SCORES, [&buf(v(*q)), &buf(v(*keys)), &buf(v(*weights)), &buf(v(*pos)), &dummy, &dummy, &buf(v(i)), &drw], &words, (groups, 1, 1));
            }
            ProgramStep::TopK { scores, dst, k, pos, word } => {
                rec.dispatch_wide("ds-topk", attend::DS_TOPK, [&buf(v(*scores)), &dummy, &dummy, &buf(v(*pos)), &dummy, &dummy, &buf(v(*dst)), &drw], &[*k as u32, *word, 0], (1, 1, 1));
            }
            ProgramStep::BlockKeep { scores, blocks, k, bs, pos, word } => {
                // (each block's best and the count, the best blocks picked by their count, word 0 of `blocks`, then flagged)
                let picks = buf(tables[i].as_ref().expect("its picks"));
                let most = lens[i] as u32;
                rec.dispatch_wide("ds-block-best", attend::DS_BLOCK_BEST, [&buf(v(*scores)), &dummy, &dummy, &buf(v(*pos)), &dummy, &dummy, &buf(v(i)), &buf(v(*blocks))], &[*bs as u32, *word], (most.div_ceil(256), 1, 1));
                rec.dispatch_wide("ds-topk", attend::DS_TOPK, [&buf(v(i)), &dummy, &dummy, &buf(v(*blocks)), &dummy, &dummy, &picks, &drw], &[*k as u32, 0, 0], (1, 1, 1));
                rec.dispatch_wide("ds-block-flags", attend::DS_BLOCK_FLAGS, [&picks, &dummy, &dummy, &dummy, &dummy, &dummy, &buf(v(*blocks)), &drw], &[*k as u32], ((*k as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::MaskBlocks { scores, blocks, bs, pos, word } => {
                let n = lens[*scores] as u32;
                rec.dispatch_wide("ds-mask-blocks", attend::DS_MASK_BLOCKS, [&dummy, &buf(v(*blocks)), &dummy, &buf(v(*pos)), &dummy, &dummy, &buf(v(*scores)), &drw], &[*bs as u32, *word], (n.div_ceil(256), 1, 1));
            }
            ProgramStep::Attend { q, heads, dim, window, win, compressed, scale, pos, word, .. } => {
                let sink = norm_weights[i].as_ref().expect("its sink");
                let (rows, picks, count) = match compressed {
                    Some((rows, picks, count)) => (buf(v(*rows)), buf(v(*picks)), *count),
                    None => (dummy.clone(), dummy.clone(), 0),
                };
                // (a head's positions over ten workgroups, their scores, then each chunk's softmax and sums, then the
                // chunks' together: in the scratch between)
                let scratch = buf(tables[i].as_ref().expect("its scratch"));
                let words = [*dim as u32, *win as u32, count as u32, *word, scale.to_bits(), *heads as u32];
                let at = [&buf(v(*q)), &buf(v(*window)), &rows, &picks, &buf(v(*pos)), &buf(sink), &scratch, &drw];
                let grid = (*heads as u32, attend::ATTEND_CHUNKS as u32, 1);
                rec.dispatch_wide("ds-attend-scores", attend::DS_ATTEND_SCORES, at, &words, grid);
                rec.dispatch_wide("ds-attend-parts", attend::DS_ATTEND_PARTS, at, &words, grid);
                rec.dispatch_wide("ds-attend-join", attend::DS_ATTEND_JOIN, [&dummy, &dummy, &dummy, &dummy, &dummy, &buf(sink), &scratch, &buf(v(i))], &words, (*heads as u32, 1, 1));
            }
            ProgramStep::HcMix { proj, pos, word, n, norm_eps, iters, hc_eps, .. } => {
                let (base, scale) = (norm_weights[i].as_ref().expect("its base"), tables[i].as_ref().expect("its scale"));
                let words = [*n as u32, *iters as u32, norm_eps.to_bits(), hc_eps.to_bits(), *word];
                rec.dispatch_wide("ds-hc-mix", attend::DS_HC_MIX, [&buf(v(*proj)), &dummy, &buf(base), &buf(scale), &buf(v(*pos)), &dummy, &buf(v(i)), &drw], &words, (1, 1, 1));
            }
            ProgramStep::RopeRows { x, rows, stride, at, half, cos, sin, pos, group, inverse } => {
                let words = [*rows as u32, *stride as u32, *at as u32, *half as u32, 0, u32::from(*inverse), *group as u32];
                rec.dispatch_wide("ds-rope", attend::DS_ROPE, [&buf(v(*cos)), &buf(v(*sin)), &buf(v(*pos)), &dummy, &dummy, &dummy, &buf(v(*x)), &drw], &words, (((rows * half) as u32).div_ceil(256), 1, 1));
            }
            ProgramStep::AttendRows { q, kv, idxs, tokens, heads, dim, len, scale, .. } => {
                let sink = norm_weights[i].as_ref().expect("its sink");
                let words = [*dim as u32, *heads as u32, *len as u32, 0, scale.to_bits()];
                let grid = ((heads / attend::ATTEND_ROWS_HEADS) as u32, (*tokens as u32).min(65535), (*tokens as u32).div_ceil(65535));
                rec.dispatch_wide("ds-attend-rows", attend::DS_ATTEND_ROWS, [&buf(v(*q)), &buf(v(*kv)), &buf(v(*idxs)), &dummy, &dummy, &buf(sink), &buf(v(i)), &drw], &words, grid);
            }
        }
    };
    // (each part a recording of its own, submitted as it is made: the second follows the first on the queue)
    let part = |range: std::ops::Range<usize>, reads: &[(usize, usize, usize)], recipe: Option<Vec<crate::Dispatch>>| {
        let mut rec = crate::chain::Recorder::new(b);
        rec.keep_groups(rows_one);
        match recipe {
            Some(dispatches) => rec.replay(dispatches),
            None => {
                for i in range {
                    record(&mut rec, i);
                }
            }
        }
        let made = rec.recipe();
        for &(s, at, len) in reads {
            rec.read_range(v(s), at, len);
        }
        (Box::new(rec).launch(), made)
    };
    let (first, made_first) = part(0..split, reads, replay.as_ref().map(|r| r.1.clone()));
    crate::profile::add(&crate::profile::PROGRAM_PREP, entered);
    let (then, made_then) = match (split < steps.len()).then(|| part(split..steps.len(), later, replay.as_ref().map(|r| r.2.clone()))) {
        Some((launched, made)) => (Some(launched), made),
        None => (None, Some(Vec::new())),
    };
    if let (None, Some(a), Some(c)) = (&replay, made_first, made_then) {
        arena.recipe = Some((split, a, c));
    }
    drop(arena);
    let now = first.wait();
    Some((now, ProgramLater { launched: then, _arena: kept }))
}

/// A program's vectors on its device ([`run_program`]'s arena): each step's, each norm's weight (each candidates' bias),
/// each group's scratch, each activation's table (each candidates' gate and up jobs), each held pass's scratch.
pub(crate) struct ProgramArena {
    vecs: Vec<Option<DeviceVec>>,
    norms: Vec<Option<DeviceVec>>,
    groups: Vec<Option<GroupMatScratch>>,
    tables: Vec<Option<DeviceVec>>,
    fused: Vec<Option<HeldScratch>>,
    /// Its two parts' dispatches as its first run recorded them, and where it was split: each run's after it the same
    /// work (its vectors, bind groups and kernels these), not recorded again.
    recipe: Option<(usize, Vec<crate::Dispatch>, Vec<crate::Dispatch>)>,
}

/// Arenas a device keeps at most before they start over (a program a layer and kind: some two hundred a model).
const ARENAS: usize = 4096;

/// `steps`' arena on `b` (vectors `lens` long), made at the program's first run and kept by the device (let go with it):
/// a program is known by its steps' kinds and shapes, its matrices, its norms' weights and its groups' jobs.
fn arena(b: &WgpuBackend, steps: &[ProgramStep<'_>], lens: &[usize]) -> Arc<Mutex<ProgramArena>> {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // (a matrix, a group or a set of slots known by its buffer's id, which is never another's while it lives or after,
    // a host's values by where they are and some of them: an arena keeps its recording, bound to those, and an
    // address freed and taken again by another matrix left a run reading the old one's weights)
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().cloned();
    let host = |v: &[f32]| (v.as_ptr() as usize, v.len(), [0, v.len() / 2, v.len().saturating_sub(1)].map(|i| v.get(i).map_or(0, |x| x.to_bits())));
    let mat = |m: &ChainMat<'_>, h: &mut std::collections::hash_map::DefaultHasher| match m {
        // (another packed matrix, a GGUF's, by where it lives: a model's matrices live as long as it)
        ChainMat::Exl3(l) => (0u8, l.as_any().and_then(|a| a.downcast_ref::<Exl3Gpu>()).and_then(|g| g.single_chunk()).map(|(w, _)| w.clone()), *l as *const dyn ggml_rs::exl3::PackedLinear as *const () as usize).hash(h),
        ChainMat::F16 { w, n, k } | ChainMat::Bf16 { w, n, k } | ChainMat::F32 { w, n, k } => (1u8, buf(w), *n, *k).hash(h),
    };
    for (step, &len) in steps.iter().zip(lens) {
        len.hash(&mut h);
        match step {
            ProgramStep::Input(_) => 0u8.hash(&mut h),
            ProgramStep::Scratch(_) => 1u8.hash(&mut h),
            ProgramStep::Mul { m, x, rows } => {
                (2u8, *x, *rows).hash(&mut h);
                mat(m, &mut h);
            }
            ProgramStep::MulGroup { g, x, jobs } => (3u8, g.words().clone(), *x, *jobs).hash(&mut h),
            ProgramStep::CopyColsBf16 { src, src_stride, src_at, dst, dst_stride, dst_at, width, rows } => (4u8, [*src, *src_stride, *src_at, *dst, *dst_stride, *dst_at, *width, *rows]).hash(&mut h),
            ProgramStep::Norm { x, weight, eps, rows, width } => (5u8, *x, host(weight), eps.to_bits(), *rows, *width).hash(&mut h),
            ProgramStep::HcPost { out, res, mix, d } => (6u8, [*out, *res, *mix, *d]).hash(&mut h),
            ProgramStep::HcPre { h: x, mix, d } => (7u8, [*x, *mix, *d]).hash(&mut h),
            ProgramStep::SwigluBf16 { gate, up, len, silu, limit } => (8u8, [*gate, *up, *len], host(silu), limit.to_bits()).hash(&mut h),
            ProgramStep::Candidates { logits, bias, table, k } => (9u8, [*logits, *k], host(bias), buf(table)).hash(&mut h),
            ProgramStep::Held { slots, x, cands, limit } => (10u8, [*x, *cands], slots.words().clone(), limit.to_bits()).hash(&mut h),
            ProgramStep::State(v) => (11u8, buf(v)).hash(&mut h),
            ProgramStep::Rope { x, rows, stride, at, half, cos, sin, pos, word, inverse } => (12u8, [*x, *rows, *stride, *at, *half, *cos, *sin, *pos], *word, *inverse).hash(&mut h),
            ProgramStep::Fp8 { x, len } => (13u8, [*x, *len]).hash(&mut h),
            ProgramStep::Fp4 { x, len, block, e4m3 } => (14u8, [*x, *len, *block], *e4m3).hash(&mut h),
            ProgramStep::PutRow { src, dst, width, pos, word } => (15u8, [*src, *dst, *width, *pos], *word).hash(&mut h),
            ProgramStep::Pool { kv, score, ratio, width } => (16u8, [*kv, *score, *ratio, *width]).hash(&mut h),
            ProgramStep::IndexScores { q, keys, weights, heads, dim, scale, pos, word } => (17u8, [*q, *keys, *weights, *heads, *dim, *pos], scale.to_bits(), *word).hash(&mut h),
            ProgramStep::TopK { scores, dst, k, pos, word } => (18u8, [*scores, *dst, *k, *pos], *word).hash(&mut h),
            ProgramStep::BlockKeep { scores, blocks, k, bs, pos, word } => (23u8, [*scores, *blocks, *k, *bs, *pos], *word).hash(&mut h),
            ProgramStep::MaskBlocks { scores, blocks, bs, pos, word } => (24u8, [*scores, *blocks, *bs, *pos], *word).hash(&mut h),
            ProgramStep::Attend { q, heads, dim, window, win, compressed, sink, scale, pos, word } => (19u8, [*q, *heads, *dim, *window, *win, *pos], *compressed, host(sink), scale.to_bits(), *word).hash(&mut h),
            ProgramStep::HcMix { proj, pos, word, base, scale, n, norm_eps, iters, hc_eps } => (20u8, [*proj, *pos, *n, *iters], *word, host(base), host(scale), norm_eps.to_bits(), hc_eps.to_bits()).hash(&mut h),
            ProgramStep::RopeRows { x, rows, stride, at, half, cos, sin, pos, group, inverse } => (21u8, [*x, *rows, *stride, *at, *half, *cos, *sin, *pos, *group], *inverse).hash(&mut h),
            ProgramStep::AttendRows { q, kv, idxs, tokens, heads, dim, len, sink, scale } => (22u8, [*q, *kv, *idxs, *tokens, *heads, *dim, *len], host(sink), scale.to_bits()).hash(&mut h),
        }
    }
    let key = h.finish();
    let mut kept = b.gpu.programs.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(a) = kept.get(&key) {
        return Arc::clone(a);
    }
    if kept.len() >= ARENAS {
        kept.clear();
    }
    let made = Arc::new(Mutex::new(make_arena(b, steps, lens)));
    kept.insert(key, Arc::clone(&made));
    made
}

/// `steps`' vectors on `b` (`lens` long), their norms' weights and groups' and passes' scratch, as an arena holds them.
fn make_arena(b: &WgpuBackend, steps: &[ProgramStep<'_>], lens: &[usize]) -> ProgramArena {
    let up = |values: &[f32]| {
        let v = b.vec(values.len());
        DeviceChain::upload(b, &v, values);
        v
    };
    ProgramArena {
        // (a kept vector is the caller's own)
        vecs: steps.iter().zip(lens).map(|(s, &len)| (len > 0 && !matches!(s, ProgramStep::State(_))).then(|| b.vec(len))).collect(),
        norms: steps
            .iter()
            .map(|s| match s {
                ProgramStep::Norm { weight, .. } => Some(up(weight)),
                ProgramStep::Candidates { bias, .. } => Some(up(bias)),
                ProgramStep::Attend { sink, .. } | ProgramStep::AttendRows { sink, .. } => Some(up(sink)),
                ProgramStep::HcMix { base, .. } => Some(up(base)),
                _ => None,
            })
            .collect(),
        groups: steps.iter().map(|s| if let ProgramStep::MulGroup { g, jobs, .. } = s { Some(g.scratch_kept(b, jobs)) } else { None }).collect(),
        tables: steps
            .iter()
            .map(|s| match s {
                ProgramStep::SwigluBf16 { silu, .. } => Some(up(silu)),
                ProgramStep::Candidates { k, .. } => Some(b.vec(4 * k)),
                ProgramStep::HcMix { scale, .. } => Some(up(scale)),
                // (an attention's scratch: its heads' scores, chunks' sums and chunks' parts)
                ProgramStep::Attend { heads, dim, .. } => Some(b.vec(heads * (attend::ATTEND_MOST + attend::ATTEND_CHUNKS + attend::ATTEND_CHUNKS * dim))),
                // (the best blocks' picks)
                ProgramStep::BlockKeep { k, .. } => Some(b.vec(*k)),
                _ => None,
            })
            .collect(),
        fused: steps.iter().zip(lens).map(|(s, &len)| if let ProgramStep::Held { slots, .. } = s { Some(slots.fused_scratch(len / slots.hidden())) } else { None }).collect(),
        recipe: None,
    }
}