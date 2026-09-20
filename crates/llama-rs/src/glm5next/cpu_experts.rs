//! VENDORED-LOCAL: GLM-5.3-Flash. Routed experts on the CPU, from the RAM tier.
//!
//! The third tier of the expert hierarchy in `docs/GLM5NEXT_PERF.md`, and the one
//! that finally pays. From `dsv41/src/cpu_experts.rs`, whose argument is the whole
//! reason this exists:
//!
//! > Moving that record to a GPU costs a PCIe copy (2.6 ms on the dev machine's x2
//! > link, 1.3 ms on its x4); computing it here costs ~0.9-1.1 ms on 24-32
//! > threads. So a VRAM miss whose record is in RAM is cheaper to compute here than
//! > to upload, and it runs while the GPU works on the layer's resident experts.
//!
//! On this machine the same comparison, measured: a record is 14.16 MB, an upload
//! of it costs **1.12 ms** on card 0's four PCIe lanes, and the AVX-512 dot
//! kernels in `ggml-quants` do the whole record in **~1.4 ms on one thread** and
//! are DRAM-bound long before the thread count runs out. So a miss goes to the CPU
//! and the GPU never waits for it.
//!
//! **What it computes.** Exactly the FFN the device path computes, from the leased
//! record bytes and never expanding them to f32:
//!
//! ```text
//!   gate = W_gate . x          [n_ff]        fused Q4_K/Q6_K dot
//!   up   = W_up   . x          [n_ff]
//!   h    = clamp(silu(gate)) * clamp(up)     the text FFN clamps AFTER the silu
//!   out  = W_down . h          [n_embd]
//! ```
//!
//! **Numerics.** The dot kernels are bit-identical to `dequantize` then a
//! sequential dot when they run scalar, and within f32 unit roundoff (5e-8 of
//! `sum |w x|`) when they run AVX-512. The GPU computes the same FFN in a different
//! order again, so a routed expert's contribution differs in its last bits
//! depending on which tier served it -- and which tier that is depends on cache
//! state rather than on the prompt. The model already shows 2e-5 between its host
//! and CUDA paths, so this sits well under the noise, but it is a real property of
//! running a hybrid and worth stating rather than discovering.

use std::sync::Arc;

use ggml_quants::GgmlType;
use nrob::ecache::HostLease;
use rayon::prelude::*;

use crate::expert_stream::ExpertLayout;
use crate::{LlamaError, Result};

/// What the CPU tier has done since the process started, so a slow token can be
/// attributed rather than guessed at. Three relaxed adds per layer.
pub static RECORDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(records, calls, seconds)` since the last [`reset_stats`].
pub fn stats() -> (u64, u64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        RECORDS.load(Relaxed),
        CALLS.load(Relaxed),
        NANOS.load(Relaxed) as f64 / 1e9,
    )
}

pub fn reset_stats() {
    use std::sync::atomic::Ordering::Relaxed;
    RECORDS.store(0, Relaxed);
    CALLS.store(0, Relaxed);
    NANOS.store(0, Relaxed);
}

/// One expert to compute: its record, and the routing weight its output gets.
pub struct CpuJob {
    pub lease: HostLease,
    pub weight: f32,
    /// Where this expert's output belongs in the caller's ordering.
    pub slot: usize,
}

/// The CPU tier. Holds nothing but the dispatch decision; the parallelism comes
/// from rayon's global pool, which is already persistent.
///
/// `dsv41` hand-rolled a thread pool because its decode needed two barriers per
/// layer and it wanted a 100 us spin before parking -- waking a parked thread on
/// Windows costs a fifth of a one-expert job. Here a layer is one parallel map
/// over independent experts with a single join, which is what rayon is for, and the
/// jobs are ~1.4 ms each rather than ~0.1 ms, so wake-up latency is a rounding
/// error rather than a fifth of the work.
pub struct CpuExperts {
    avx512: bool,
}

impl Default for CpuExperts {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuExperts {
    pub fn new() -> Self {
        Self {
            avx512: ggml_quants::q4_k::has_avx512(),
        }
    }

    /// Whether the AVX-512 kernels are in use. Without them the scalar fused path
    /// still runs, at about an eighth of the throughput -- which no longer beats a
    /// PCIe upload on one thread, but still does across a few.
    pub fn avx512(&self) -> bool {
        self.avx512
    }

    /// Compute every job's FFN, in parallel, and hand back `(slot, weight, output)`.
    ///
    /// `x` is the layer input, `limit` the SwiGLU clamp. Each job is independent,
    /// so this is a plain parallel map -- the cost of an expert is entirely its own
    /// 14.16 MB.
    pub fn run(
        &self,
        jobs: &[CpuJob],
        layout: &ExpertLayout,
        layer: usize,
        x: &[f32],
        limit: f32,
        n_embd: usize,
    ) -> Result<Vec<(usize, f32, Vec<f32>)>> {
        use std::sync::atomic::Ordering::Relaxed;
        let avx = self.avx512;
        let t0 = std::time::Instant::now();
        // In sequence, not `par_iter`: each record already spreads its rows across
        // the whole pool, so running records concurrently would only fragment it.
        let r: Result<Vec<(usize, f32, Vec<f32>)>> = jobs
            .iter()
            .map(|job| {
                let mut out = vec![0.0f32; n_embd];
                ffn(&job.lease, layout, layer, x, limit, avx, &mut out)?;
                Ok((job.slot, job.weight, out))
            })
            .collect();
        NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
        RECORDS.fetch_add(jobs.len() as u64, Relaxed);
        CALLS.fetch_add(1, Relaxed);
        r
    }
}

/// One expert's FFN from its record bytes.
///
/// The record is the three regions end to end, each padded to the model-wide
/// maximum for that part, which is how `GgufExpertStore` lays it out.
pub fn ffn(
    record: &[u8],
    layout: &ExpertLayout,
    layer: usize,
    x: &[f32],
    limit: f32,
    avx512: bool,
    out: &mut [f32],
) -> Result<()> {
    let [pg, pu, pd] = layout.part_views(layer);
    let (n_ff, n_embd) = (pg.shape[0], pg.shape[1]);

    if x.len() != n_embd || out.len() != n_embd {
        return Err(LlamaError::Config(format!(
            "cpu_experts: x {} / out {} against n_embd {n_embd}",
            x.len(),
            out.len()
        )));
    }

    let mut gate = vec![0.0f32; n_ff];
    let mut up = vec![0.0f32; n_ff];
    dot_par(
        &record[pg.region..pg.region + pg.len],
        x,
        n_ff,
        n_embd,
        pg.dtype,
        avx512,
        &mut gate,
    )?;
    dot_par(
        &record[pu.region..pu.region + pu.len],
        x,
        n_ff,
        n_embd,
        pu.dtype,
        avx512,
        &mut up,
    )?;

    // The text FFN clamps the activation, not the pre-activation -- llama.cpp's
    // `ffn_silu_clamped`. The vision tower is the other way round; this is not it.
    let mut h = vec![0.0f32; n_ff];
    for (i, t) in h.iter_mut().enumerate() {
        let mut a = gate[i];
        a /= 1.0 + (-a).exp();
        if limit > 0.0 {
            if a > limit {
                a = limit;
            }
            *t = a * up[i].clamp(-limit, limit);
        } else {
            *t = a * up[i];
        }
    }

    dot_par(
        &record[pd.region..pd.region + pd.len],
        &h,
        n_embd,
        n_ff,
        pd.dtype,
        avx512,
        out,
    )
}

/// Output rows per rayon task.
///
/// A layer routes 8 experts and the VRAM tier usually holds most of them, so only
/// two to four records reach the CPU per layer. Parallelising over *records* would
/// therefore use two to four of the 32 threads and leave a 1.4 ms record on the
/// critical path -- which is exactly what the first version did, and why it halved
/// the PCIe traffic without moving the token time.
///
/// Splitting the rows instead gives every record the whole pool: 2048 rows at 64 a
/// task is 32 tasks, and rows are independent so each one still sums in its own
/// order. 64 is small enough to fill the pool and large enough that a task is tens
/// of microseconds rather than a scheduling cost.
const ROWS_PER_TASK: usize = 64;

/// [`dot`] with the row range split across the thread pool.
///
/// Every row is summed exactly as the single-threaded version sums it -- the split
/// is over output rows, which are independent -- so this changes the schedule and
/// not the arithmetic.
fn dot_par(
    w: &[u8],
    v: &[f32],
    n_rows: usize,
    k: usize,
    dtype: GgmlType,
    avx512: bool,
    out: &mut [f32],
) -> Result<()> {
    let row_bytes = w.len() / n_rows;
    out.par_chunks_mut(ROWS_PER_TASK)
        .enumerate()
        .try_for_each(|(ci, chunk)| {
            let r0 = ci * ROWS_PER_TASK;
            let bytes = &w[r0 * row_bytes..(r0 + chunk.len()) * row_bytes];
            dot(bytes, v, chunk.len(), k, dtype, avx512, chunk)
        })
}

/// `out[r] = dot(W[r], v)` for a packed `[n_rows, k]` weight.
fn dot(
    w: &[u8],
    v: &[f32],
    n_rows: usize,
    k: usize,
    dtype: GgmlType,
    avx512: bool,
    out: &mut [f32],
) -> Result<()> {
    match dtype {
        GgmlType::Q4_K => {
            #[cfg(target_arch = "x86_64")]
            if avx512 {
                // SAFETY: `avx512` was set from `has_avx512()`, which checks both
                // avx512f and avx512bw.
                unsafe { ggml_quants::q4_k::avx512::dot_rows(w, v, n_rows, k, out) };
                return Ok(());
            }
            ggml_quants::q4_k::dot_rows(w, v, n_rows, k, out);
            Ok(())
        }
        GgmlType::Q6_K => {
            #[cfg(target_arch = "x86_64")]
            if avx512 {
                // SAFETY: as above.
                unsafe { ggml_quants::q6_k::avx512::dot_rows(w, v, n_rows, k, out) };
                return Ok(());
            }
            ggml_quants::q6_k::dot_rows(w, v, n_rows, k, out);
            Ok(())
        }
        // Only the two the released model uses have fused kernels. Anything else
        // belongs on the GPU rather than silently dequantising 14 MB here.
        other => Err(LlamaError::Config(format!(
            "cpu_experts: no fused dot for {other:?}; this expert must go to the GPU"
        ))),
    }
}

/// A shared handle, so one instance is made per model rather than per layer.
pub type SharedCpuExperts = Arc<CpuExperts>;
