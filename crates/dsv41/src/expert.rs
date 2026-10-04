//! Routed experts served straight from the checkpoint shards, and the expert
//! forward pass.
//!
//! # Record
//!
//! OAIY's cache layer ([`oaiy_engine::ecache::Ecache`]) deals in uniform byte
//! records addressed by `(layer, expert)`. One DeepSeek-V4.1 expert record
//! is the six checkpoint tensors in a fixed order:
//!
//! ```text
//!   w1.weight | w2.weight | w3.weight | w1.scale | w2.scale | w3.scale
//!   5,898,240   5,898,240   5,898,240   368,640    368,640    368,640   = 18,800,640 bytes
//! ```
//!
//! w1/w3 are `[2304, 5120]` and w2 is `[5120, 2304]` in packed e2m1 (two per
//! byte, low nibble first), each with one e8m0 scale per 32 along K. That
//! order is the order the checkpoint stores them in, so the six tensors
//! coalesce into two contiguous runs (weights, then scales, which the writer
//! grouped by dtype into a separate region) and a fetch is two positioned
//! reads. Runs are discovered at open rather than assumed: a checkpoint
//! written in another order still works, one read per tensor.

use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use oaiy_engine::store::WeightStore;
use oaiy_engine::{Error, Result};

use crate::formats::{e8m0_to_f32, fake_quant_fp8, to_bf16, FP4_VALUES};
use crate::io::{open_read, read_direct, read_exact_at, AlignedScratch};
use crate::safetensors::{Dtype, StIndex};

pub const DIM: usize = 5120;
pub const INTER: usize = 2304;
/// Elements per e8m0 scale (weights) and per fp8 activation scale.
pub const BLOCK: usize = 32;

/// Byte ranges of the six parts inside a record.
pub const W1: std::ops::Range<usize> = 0..W;
pub const W2: std::ops::Range<usize> = W..2 * W;
pub const W3: std::ops::Range<usize> = 2 * W..3 * W;
pub const S1: std::ops::Range<usize> = 3 * W..3 * W + S;
pub const S2: std::ops::Range<usize> = 3 * W + S..3 * W + 2 * S;
pub const S3: std::ops::Range<usize> = 3 * W + 2 * S..RECORD_BYTES;
const W: usize = INTER * DIM / 2;
const S: usize = INTER * DIM / BLOCK;
pub const RECORD_BYTES: usize = 3 * W + 3 * S;

/// Suffixes in record order, with the dtype and byte size each must have.
const PARTS: [(&str, Dtype, usize); 6] = [
    ("w1.weight", Dtype::I8, W),
    ("w2.weight", Dtype::I8, W),
    ("w3.weight", Dtype::I8, W),
    ("w1.scale", Dtype::F8E8M0, S),
    ("w2.scale", Dtype::F8E8M0, S),
    ("w3.scale", Dtype::F8E8M0, S),
];

/// One contiguous file range landing at `dst_off` inside the record.
#[derive(Clone, Copy, Debug)]
struct Run {
    shard: u32,
    file_off: u64,
    dst_off: usize,
    len: usize,
}

/// `WeightStore` over the routed experts of a safetensors checkpoint.
pub struct SafetensorsExpertStore {
    files: Vec<File>,
    direct: bool,
    layers: u32,
    experts: u32,
    /// Runs of expert `(l, e)` are `runs[starts[i]..starts[i + 1]]`, i = l * experts + e.
    runs: Vec<Run>,
    starts: Vec<u32>,
    /// Scratch for direct reads, one per concurrent fetch (checked out, never shared).
    scratch: Mutex<Vec<AlignedScratch>>,
}

impl SafetensorsExpertStore {
    /// Index experts `layers.{0..layers}.ffn.experts.{0..experts}` of the
    /// checkpoint. `direct` asks for page-cache-bypassing reads (see `io`).
    pub fn open(idx: &StIndex, layers: u32, experts: u32, direct: bool) -> Result<Self> {
        let mut runs: Vec<Run> = Vec::new();
        let mut starts = Vec::with_capacity((layers * experts) as usize + 1);
        for l in 0..layers {
            for e in 0..experts {
                let first_run = runs.len();
                starts.push(first_run as u32);
                let mut dst_off = 0;
                for (suffix, dtype, bytes) in PARTS {
                    let name = format!("layers.{l}.ffn.experts.{e}.{suffix}");
                    let t = idx.info(&name)?;
                    if t.dtype != dtype || t.nbytes as usize != bytes {
                        return Err(Error::Format(format!(
                            "{name}: {:?} x {} bytes, expected {dtype:?} x {bytes}",
                            t.dtype, t.nbytes
                        )));
                    }
                    let run = Run { shard: t.shard as u32, file_off: t.start, dst_off, len: bytes };
                    // extend this expert's previous run when the part follows it in the same file
                    let own_prev = if runs.len() > first_run { runs.last_mut() } else { None };
                    match own_prev {
                        Some(prev)
                            if prev.shard == run.shard
                                && prev.file_off + prev.len as u64 == run.file_off
                                && prev.dst_off + prev.len == run.dst_off =>
                        {
                            prev.len += run.len
                        }
                        _ => runs.push(run),
                    }
                    dst_off += bytes;
                }
            }
        }
        starts.push(runs.len() as u32);

        let mut files = Vec::with_capacity(idx.shard_count());
        let mut all_direct = direct;
        for i in 0..idx.shard_count() {
            let (f, d) = open_read(idx.shard_path(i), direct)?;
            all_direct &= d;
            files.push(f);
        }
        Ok(SafetensorsExpertStore {
            files,
            direct: all_direct,
            layers,
            experts,
            runs,
            starts,
            scratch: Mutex::new(Vec::new()),
        })
    }

    /// Positioned reads per expert (2 for the DeepSeek-V4.1 checkpoints).
    pub fn reads_per_expert(&self, layer: u32, expert: u32) -> usize {
        let i = (layer * self.experts + expert) as usize;
        (self.starts[i + 1] - self.starts[i]) as usize
    }

    /// Open the checkpoint directory and index all 40 x 384 routed experts.
    pub fn open_dir(dir: &Path, direct: bool) -> Result<Self> {
        Self::open(&StIndex::open(dir)?, 40, 384, direct)
    }
}

impl WeightStore for SafetensorsExpertStore {
    fn record_bytes(&self) -> usize {
        RECORD_BYTES
    }

    fn shape(&self) -> (u32, u32) {
        (self.layers, self.experts)
    }

    fn fetch(&self, layer: u32, expert: u32, dst: &mut [u8]) -> Result<()> {
        if layer >= self.layers || expert >= self.experts {
            return Err(Error::Arg(format!("expert ({layer}, {expert}) out of range")));
        }
        if dst.len() != RECORD_BYTES {
            return Err(Error::Arg(format!("record buffer is {} bytes, need {RECORD_BYTES}", dst.len())));
        }
        let i = (layer * self.experts + expert) as usize;
        let runs = &self.runs[self.starts[i] as usize..self.starts[i + 1] as usize];
        let ctx = |e: std::io::Error| Error::Io(std::io::Error::new(e.kind(), format!("expert ({layer}, {expert}): {e}")));
        if self.direct {
            let mut scratch = self.scratch.lock().unwrap_or_else(|p| p.into_inner()).pop().unwrap_or_default();
            let res = runs.iter().try_for_each(|r| {
                read_direct(&self.files[r.shard as usize], &mut dst[r.dst_off..r.dst_off + r.len], r.file_off, &mut scratch)
            });
            self.scratch.lock().unwrap_or_else(|p| p.into_inner()).push(scratch);
            res.map_err(ctx)
        } else {
            runs.iter()
                .try_for_each(|r| read_exact_at(&self.files[r.shard as usize], &mut dst[r.dst_off..r.dst_off + r.len], r.file_off))
                .map_err(ctx)
        }
    }

    fn direct_io(&self) -> bool {
        self.direct
    }
}

/// `y[t][r] = sum_k x[t][k] * dequant(w)[r, k]` for one packed-fp4 matrix
/// `[rows, k]` with per-32 e8m0 scales and `x` holding `nt` rows (already
/// fake-quantized). Each weight row is decoded once for all `nt` tokens;
/// every dot product runs in the same order whatever `nt` is, so batching
/// never changes a result. Output rows are split across threads.
fn fp4_matmul(x: &[f32], nt: usize, w: &[u8], s: &[u8], rows: usize, k: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * k / 2);
    debug_assert_eq!(s.len(), rows * k / BLOCK);
    debug_assert_eq!(x.len(), nt * k);
    let mut yt = vec![0.0f32; rows * nt]; // [rows][nt] so each thread owns a contiguous run
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows);
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, out) in yt.chunks_mut(per * nt).enumerate() {
            scope.spawn(move || {
                let mut row = vec![0.0f32; k];
                for (j, yr) in out.chunks_mut(nt).enumerate() {
                    let r = t * per + j;
                    let wrow = &w[r * k / 2..(r + 1) * k / 2];
                    for (i, &byte) in wrow.iter().enumerate() {
                        row[2 * i] = FP4_VALUES[(byte & 0x0f) as usize];
                        row[2 * i + 1] = FP4_VALUES[(byte >> 4) as usize];
                    }
                    let srow = &s[r * k / BLOCK..(r + 1) * k / BLOCK];
                    for (tok, yv) in yr.iter_mut().enumerate() {
                        let xr = &x[tok * k..(tok + 1) * k];
                        let mut acc = 0.0f32;
                        for (b, &sb) in srow.iter().enumerate() {
                            let (xb, wb) = (&xr[b * BLOCK..(b + 1) * BLOCK], &row[b * BLOCK..(b + 1) * BLOCK]);
                            let mut part = 0.0f32;
                            for i in 0..BLOCK / 2 {
                                part += xb[2 * i] * wb[2 * i] + xb[2 * i + 1] * wb[2 * i + 1];
                            }
                            acc += part * e8m0_to_f32(sb);
                        }
                        *yv = acc;
                    }
                }
            });
        }
    });
    let mut y = vec![0.0f32; nt * rows];
    for r in 0..rows {
        for tok in 0..nt {
            y[tok * rows + r] = yt[r * nt + tok];
        }
    }
    y
}

/// One routed expert on one token, as the reference `Expert.forward`:
/// fp8-quantized input, fp4 w1/w3 with outputs rounded to bf16, SwiGLU with
/// the training clamps (up to +-limit, gate from above), optional route
/// weight, bf16 round, fp8-quantized again for w2, output rounded to bf16.
/// `x` is the bf16 activation (as f32 values); `record` is a fetched record.
pub fn expert_forward(record: &[u8], x: &[f32], route_weight: Option<f32>, swiglu_limit: f32) -> Vec<f32> {
    expert_forward_batch(record, x, route_weight.as_ref().map(std::slice::from_ref), swiglu_limit)
}

/// [`expert_forward`] for every row of `x` (`[nt, DIM]`) at once, reading the
/// record once; `route_weights` has one weight per row. Identical results.
pub fn expert_forward_batch(record: &[u8], x: &[f32], route_weights: Option<&[f32]>, swiglu_limit: f32) -> Vec<f32> {
    assert_eq!(record.len(), RECORD_BYTES);
    assert_eq!(x.len() % DIM, 0);
    let nt = x.len() / DIM;
    if let Some(w) = route_weights {
        assert_eq!(w.len(), nt);
    }
    let xq = fake_quant_fp8(x, BLOCK);
    let gate = fp4_matmul(&xq, nt, &record[W1], &record[S1], INTER, DIM);
    let up = fp4_matmul(&xq, nt, &record[W3], &record[S3], INTER, DIM);
    let hq = fake_quant_fp8(&swiglu(&gate, &up, route_weights, swiglu_limit), BLOCK);
    fp4_matmul(&hq, nt, &record[W2], &record[S2], DIM, INTER).into_iter().map(to_bf16).collect()
}

/// The middle of [`expert_forward_batch`]: from the gate and up sums (`[rows, INTER]`, f32) to the activation the down
/// projection takes (before its fp8 quantization): each rounded to bf16, clamped, `silu(gate) * up`, times the row's
/// routing weight, rounded again.
pub fn swiglu(gate: &[f32], up: &[f32], route_weights: Option<&[f32]>, swiglu_limit: f32) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .enumerate()
        .map(|(i, (&g, &u))| {
            let (mut g, mut u) = (to_bf16(g), to_bf16(u));
            if swiglu_limit > 0.0 {
                u = u.clamp(-swiglu_limit, swiglu_limit);
                g = g.min(swiglu_limit);
            }
            let mut v = g / (1.0 + (-g).exp()) * u;
            if let Some(w) = route_weights {
                v *= w[i / INTER];
            }
            to_bf16(v)
        })
        .collect()
}

/// One expert's share of a prompt for an [`ExpertsKernel`]: its record, its tokens' rows (`[rows, DIM]`) and their
/// routing weights.
pub struct ExpertJob<'a> {
    pub record: &'a [u8],
    pub x: &'a [f32],
    pub weights: &'a [f32],
}

/// [`expert_forward_batch`] for several experts at once, their matmuls made elsewhere (a GPU's): each job's output,
/// `[rows, DIM]`, in order. It quantizes the activations and takes the SwiGLU as [`expert_forward_batch`] does (with
/// `fake_quant_fp8` and [`swiglu`]), so only the sums are made there.
pub trait ExpertsKernel: Send + Sync {
    fn forward(&self, jobs: &[ExpertJob<'_>], swiglu_limit: f32) -> Vec<Vec<f32>>;

    /// Which of a layer's `experts`, each used by `tokens[i]` of a pass's tokens, the device keeps (their records
    /// resident there, so nobody need read them): [`Self::forward_held`] computes those. Every use counts toward what
    /// it keeps, held or not. By default it keeps none.
    fn holds(&self, layer: u32, experts: &[u32], tokens: &[usize]) -> Vec<bool> {
        let _ = (layer, tokens);
        vec![false; experts.len()]
    }

    /// [`Self::forward`] for experts it holds, `(expert, x, weights)` each, from their resident records.
    fn forward_held(&self, layer: u32, jobs: &[(u32, &[f32], &[f32])], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let _ = (layer, jobs, swiglu_limit);
        panic!("this experts kernel holds no experts")
    }

    /// Records a decode step has just read for experts it did not hold: it keeps any used more often than what it
    /// would replace.
    fn offer(&self, layer: u32, records: &[(u32, &[u8])]) {
        let _ = (layer, records);
    }

    /// A forward pass of `tokens` tokens has ended: after a prompt it may take in the experts the prompt used most
    /// (from `cache`, the RAM tier, never from the drive), after a decode step age what it counts.
    fn pass_done(&self, tokens: usize, cache: &oaiy_engine::ecache::Ecache, store: &dyn WeightStore) {
        let _ = (tokens, cache, store);
    }
}
