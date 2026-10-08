//! Routed experts on the CPU for single-token decode, straight from the RAM
//! tier: the other half of the hybrid split in `docs/DEEPSEEK_V41.md`.
//!
//! A decode step touches each routed expert once, for one token, so the
//! work is one pass over its 18.8 MB record. Moving that record to a GPU
//! costs a PCIe copy (2.6 ms on the dev machine's x2 link, 1.3 ms on its x4);
//! computing it here costs ~0.9-1.1 ms on 24-32 threads (compute-bound at
//! ~1.4 cycles per weight, mostly the scalar FP4 decode; DRAM alone would
//! allow ~0.45 ms). So a VRAM miss whose record is in RAM is cheaper to
//! compute here than to upload, and it runs while the GPU works on the
//! layer's resident experts.
//!
//! Numerics are [`expert_forward`](crate::expert::expert_forward)'s exactly:
//! each output row is summed in the same order, and the SIMD lanes run
//! across rows (16 rows at a time) rather than along one, so nothing is
//! reassociated. The tests hold this path bit-identical to the oracle.
//!
//! The row kernel is pluggable ([`RowKernel`]): [`fp4_rows`] is portable;
//! [`avx512::fp4_rows`] does the same arithmetic with AVX-512 (decode by
//! `vpermps` from a 16-entry table), still bit-identical. It is a safe
//! `#[target_feature]` function, so this crate stays free of `unsafe`; the
//! caller that verified the CPU features (the crate `dsv41-simd`) makes the call.
//!
//! Threads are persistent ([`crate::pool`]): decode needs two parallel phases
//! per layer (gate/up, then down), and spawning 16+ threads 80 times per token
//! would cost more than the work. Jobs own their inputs through `Arc`s, so the
//! pool needs no lifetime tricks.

use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::expert::{BLOCK, DIM, INTER, RECORD_BYTES, S1, S2, S3, W1, W2, W3};
use crate::formats::{e8m0_to_f32, fake_quant_fp8, to_bf16, FP4_VALUES};
use crate::pool::{Job, Pool};

/// Rows computed together, one per SIMD lane.
const LANES: usize = 16;

/// Both e2m1 values of a byte, low nibble first: one load decodes two weights.
const PAIRS: [[f32; 2]; 256] = {
    let mut t = [[0.0f32; 2]; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = [FP4_VALUES[i & 0x0f], FP4_VALUES[i >> 4]];
        i += 1;
    }
    t
};

/// Blocks decoded per step. Decoding runs one step ahead of the dot
/// products: lanes are written as scalars and read back as vectors, and a
/// vector load that overlaps stores still in flight stalls until they
/// commit, so each step reads what the previous step wrote.
const STEP: usize = 4;

type Decoded = [[[f32; LANES]; BLOCK]; STEP]; // [block][element][row]

/// Decode blocks `b0..b0 + STEP` of up to 16 rows into `buf`.
fn decode(rows: &[&[u8]], b0: usize, buf: &mut Decoded) {
    const HALF: usize = BLOCK / 2;
    for (lane, row) in rows.iter().enumerate() {
        for (j, blk) in buf.iter_mut().enumerate() {
            let bytes: &[u8; HALF] = row[(b0 + j) * HALF..(b0 + j + 1) * HALF].try_into().expect("block");
            for (i, &byte) in bytes.iter().enumerate() {
                let [lo, hi] = PAIRS[byte as usize];
                blk[2 * i][lane] = lo;
                blk[2 * i + 1][lane] = hi;
            }
        }
    }
}

/// `out[j] = sum_k x[k] * dequant(w)[r0 + j, k]` for rows `r0..r0 + out.len()`
/// of a packed-fp4 `[rows, k]` matrix with per-32 e8m0 scales, each row
/// summed exactly as `expert::fp4_matmul` does (pairs within a 32-block in
/// order, then the scaled block into the row total), 16 rows per pass.
pub fn fp4_rows(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
    const HALF: usize = BLOCK / 2;
    let (row_bytes, row_scales) = (k / 2, k / BLOCK);
    assert_eq!(row_scales % STEP, 0);
    let mut bufs: [Box<Decoded>; 2] = [Box::new([[[0.0; LANES]; BLOCK]; STEP]), Box::new([[[0.0; LANES]; BLOCK]; STEP])];
    for (g, dst) in out.chunks_mut(LANES).enumerate() {
        let r = r0 + g * LANES;
        let n = dst.len();
        let rows: Vec<&[u8]> = (0..n).map(|lane| &w[(r + lane) * row_bytes..(r + lane + 1) * row_bytes]).collect();
        let scales: Vec<&[u8]> = (0..n).map(|lane| &s[(r + lane) * row_scales..(r + lane + 1) * row_scales]).collect();
        let mut acc = [0.0f32; LANES];
        decode(&rows, 0, &mut bufs[0]);
        for (step, b0) in (0..row_scales).step_by(STEP).enumerate() {
            let [a, b] = &mut bufs;
            let (cur, next) = if step % 2 == 0 { (&*a, b) } else { (&*b, a) };
            if b0 + STEP < row_scales {
                decode(&rows, b0 + STEP, next);
            }
            for (j, blk) in cur.iter().enumerate() {
                let b = b0 + j;
                let xb: &[f32; BLOCK] = x[b * BLOCK..(b + 1) * BLOCK].try_into().expect("block");
                let mut part = [0.0f32; LANES];
                for i in 0..HALF {
                    let (x0, x1, w0, w1) = (xb[2 * i], xb[2 * i + 1], &blk[2 * i], &blk[2 * i + 1]);
                    for lane in 0..LANES {
                        part[lane] += x0 * w0[lane] + x1 * w1[lane];
                    }
                }
                let mut sc = [0.0f32; LANES];
                for (v, sr) in sc.iter_mut().zip(&scales) {
                    *v = e8m0_to_f32(sr[b]);
                }
                for lane in 0..LANES {
                    acc[lane] += part[lane] * sc[lane];
                }
            }
        }
        dst.copy_from_slice(&acc[..n]);
    }
}

/// A row kernel: `out[j] = sum_k x[k] * dequant(w)[r0 + j, k]` for a packed-fp4
/// matrix of row length `k` with per-32 e8m0 scales, each row summed in
/// [`fp4_rows`]'s order (so every kernel gives the same bits).
pub type RowKernel = fn(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]);

#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    //! [`fp4_rows`](super::fp4_rows) with AVX-512F/BW: the same 16-rows-per-
    //! lane structure and the same operations in the same order, but each
    //! step on a whole register. Only value-taking intrinsics are used, so
    //! the functions here are safe to *define*; calling [`fp4_rows`] needs the
    //! features checked first (`is_x86_feature_detected!`).
    use std::arch::x86_64::*;

    use crate::expert::BLOCK;
    use crate::formats::{e8m0_to_f32, FP4_VALUES};

    #[target_feature(enable = "avx512f,avx512bw")]
    fn load16(b: &[u8]) -> __m128i {
        let lo = i64::from_le_bytes(b[..8].try_into().expect("8 bytes"));
        let hi = i64::from_le_bytes(b[8..16].try_into().expect("8 bytes"));
        _mm_set_epi64x(hi, lo)
    }

    /// 16 rows of 16 bytes to 16 columns: `out[c]` holds byte `c` of every row.
    #[target_feature(enable = "avx512f,avx512bw")]
    fn transpose(r: &[__m128i; 16]) -> [__m128i; 16] {
        let mut b = [_mm_setzero_si128(); 16];
        for j in 0..8 {
            b[2 * j] = _mm_unpacklo_epi8(r[2 * j], r[2 * j + 1]);
            b[2 * j + 1] = _mm_unpackhi_epi8(r[2 * j], r[2 * j + 1]);
        }
        // b[4q + 0/1] rows 4q..4q+1 cols 0-7 / 8-15, b[4q + 2/3] rows 4q+2..4q+3
        let mut c = [_mm_setzero_si128(); 16];
        for q in 0..4 {
            let (b0, b1, b2, b3) = (b[4 * q], b[4 * q + 1], b[4 * q + 2], b[4 * q + 3]);
            c[4 * q] = _mm_unpacklo_epi16(b0, b2); // cols 0-3
            c[4 * q + 1] = _mm_unpackhi_epi16(b0, b2); // cols 4-7
            c[4 * q + 2] = _mm_unpacklo_epi16(b1, b3); // cols 8-11
            c[4 * q + 3] = _mm_unpackhi_epi16(b1, b3); // cols 12-15
        }
        // c[4q + m]: rows 4q..4q+3, cols 4m..4m+3
        let mut d = [_mm_setzero_si128(); 16];
        for h in 0..2 {
            for m in 0..4 {
                let (lo, hi) = (c[8 * h + m], c[8 * h + 4 + m]);
                d[8 * h + 2 * m] = _mm_unpacklo_epi32(lo, hi); // cols 4m, 4m+1
                d[8 * h + 2 * m + 1] = _mm_unpackhi_epi32(lo, hi); // cols 4m+2, 4m+3
            }
        }
        // d[8h + p]: rows 8h..8h+7, cols 2p, 2p+1
        let mut out = [_mm_setzero_si128(); 16];
        for p in 0..8 {
            out[2 * p] = _mm_unpacklo_epi64(d[p], d[8 + p]);
            out[2 * p + 1] = _mm_unpackhi_epi64(d[p], d[8 + p]);
        }
        out
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    fn lanes(v: __m512) -> [f32; 16] {
        let q = [_mm512_extractf32x4_ps::<0>(v), _mm512_extractf32x4_ps::<1>(v), _mm512_extractf32x4_ps::<2>(v), _mm512_extractf32x4_ps::<3>(v)];
        let mut o = [0.0f32; 16];
        for (j, x) in q.iter().enumerate() {
            o[4 * j] = f32::from_bits(_mm_extract_ps::<0>(*x) as u32);
            o[4 * j + 1] = f32::from_bits(_mm_extract_ps::<1>(*x) as u32);
            o[4 * j + 2] = f32::from_bits(_mm_extract_ps::<2>(*x) as u32);
            o[4 * j + 3] = f32::from_bits(_mm_extract_ps::<3>(*x) as u32);
        }
        o
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    fn from_lanes(a: &[f32; 16]) -> __m512 {
        _mm512_setr_ps(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
    }


    #[target_feature(enable = "avx512f,avx512bw")]
    pub fn ternary_rows(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
        assert!(k.is_multiple_of(128) && x.len()>=k && w.len()>=(r0+out.len())*k/4 && s.len()>=(r0+out.len())*k/64);
        let mask = _mm512_set1_epi32(3);
        for (g, dst) in out.chunks_mut(16).enumerate() {
            let r=r0+g*16; let n=dst.len();
            let mut acc=_mm512_setzero_ps();
            for b in 0..k/64 {
                let mut rr=[_mm_setzero_si128();16];
                for lane in 0..n { rr[lane]=load16(&w[(r+lane)*k/4+b*16..][..16]); }
                let cols=transpose(&rr);
                let mut sc=[0.0f32;16];
                for lane in 0..n {
                    let off=((r+lane)*k/128+b/2)*2;
                    sc[lane]=crate::formats::f16_to_f32(u16::from_le_bytes([s[off],s[off+1]]));
                }
                for half in 0..2 {
                    let mut part=_mm512_setzero_ps();
                    for i in 0..8 {
                        let q=_mm512_cvtepu8_epi32(cols[half*8+i]);
                        let codes=[q,_mm512_srli_epi32::<2>(q),_mm512_srli_epi32::<4>(q),_mm512_srli_epi32::<6>(q)];
                        for pair in 0..2 {
                            let c=b*64+half*32+i*4+pair*2;
                            let a=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_si512(codes[pair*2],mask),_mm512_set1_epi32(1)));
                            let z=_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_si512(codes[pair*2+1],mask),_mm512_set1_epi32(1)));
                            let v=_mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(x[c]),a),_mm512_mul_ps(_mm512_set1_ps(x[c+1]),z));
                            part=_mm512_add_ps(part,v);
                        }
                    }
                    acc=_mm512_add_ps(acc,_mm512_mul_ps(part,from_lanes(&sc)));
                }
            }
            dst.copy_from_slice(&lanes(acc)[..n]);
        }
    }

    /// [`super::fp4_rows`], 16 rows per register.
    ///
    /// # Safety
    ///
    /// The body is safe code; calling it from code compiled without these
    /// features is `unsafe` only because the CPU must support AVX-512F and
    /// AVX-512BW. Check with `is_x86_feature_detected!` first (as
    /// `dsv41-cuda`'s `cpu::row_kernel` does).
    #[target_feature(enable = "avx512f,avx512bw")]
    pub fn fp4_rows(x: &[f32], w: &[u8], s: &[u8], k: usize, r0: usize, out: &mut [f32]) {
        const HALF: usize = BLOCK / 2;
        let (row_bytes, row_scales) = (k / 2, k / BLOCK);
        let table = from_lanes(&FP4_VALUES);
        for (g, dst) in out.chunks_mut(16).enumerate() {
            let r = r0 + g * 16;
            let n = dst.len();
            let rows: Vec<&[u8]> = (0..n).map(|lane| &w[(r + lane) * row_bytes..(r + lane + 1) * row_bytes]).collect();
            let scales: Vec<&[u8]> = (0..n).map(|lane| &s[(r + lane) * row_scales..(r + lane + 1) * row_scales]).collect();
            let mut acc = _mm512_setzero_ps();
            for (b, xb) in x[..k].chunks_exact(BLOCK).enumerate() {
                let mut v = [_mm_setzero_si128(); 16];
                for (lane, row) in rows.iter().enumerate() {
                    v[lane] = load16(&row[b * HALF..(b + 1) * HALF]);
                }
                let cols = transpose(&v);
                let mut part = _mm512_setzero_ps();
                for (i, col) in cols.iter().enumerate() {
                    let idx = _mm512_cvtepu8_epi32(*col);
                    let lo = _mm512_permutexvar_ps(idx, table); // low nibble: permutexvar reads 4 bits
                    let hi = _mm512_permutexvar_ps(_mm512_srli_epi32::<4>(idx), table);
                    let t = _mm512_add_ps(_mm512_mul_ps(_mm512_set1_ps(xb[2 * i]), lo), _mm512_mul_ps(_mm512_set1_ps(xb[2 * i + 1]), hi));
                    part = _mm512_add_ps(part, t);
                }
                let mut sc = [0.0f32; 16];
                for (v, sr) in sc.iter_mut().zip(&scales) {
                    *v = e8m0_to_f32(sr[b]);
                }
                acc = _mm512_add_ps(acc, _mm512_mul_ps(part, from_lanes(&sc)));
            }
            dst.copy_from_slice(&lanes(acc)[..n]);
        }
    }
}

/// Split `rows` into `parts` runs aligned to [`LANES`], as `(start, len)`.
fn runs(rows: usize, parts: usize) -> Vec<(usize, usize)> {
    let groups = rows.div_ceil(LANES);
    let per = groups.div_ceil(parts.max(1));
    (0..groups)
        .step_by(per.max(1))
        .map(|g| (g * LANES, ((g + per) * LANES).min(rows) - g * LANES))
        .collect()
}

/// What a finished job hands back: each expert's output, or why there is none.
pub type Outputs = Result<Vec<Vec<f32>>, String>;

/// Called on the coordinator thread when a job finishes (or fails).
pub type Done = Box<dyn FnOnce(Outputs) + Send>;

struct Request {
    records: Vec<Arc<Vec<u8>>>,
    weights: Vec<f32>,
    x: Vec<f32>,
    swiglu_limit: f32,
    done: Done,
}

/// Routed experts for one token on a persistent thread pool.
///
/// A coordinator thread owns the pool and runs one job at a time, so a job
/// can be started without waiting for it ([`spawn`](Self::spawn)): the GPU
/// side of a decode step keeps going and learns of the result through the
/// completion callback.
pub struct CpuExperts {
    tx: Option<Sender<Request>>,
    coordinator: Option<JoinHandle<()>>,
    threads: usize,
}

impl CpuExperts {
    /// `threads` workers (0: one per available hardware thread).
    pub fn new(threads: usize) -> CpuExperts {
        Self::with_kernel(threads, fp4_rows)
    }

    /// [`new`](Self::new) with a faster row kernel (same results).
    pub fn with_kernel(threads: usize, kernel: RowKernel) -> CpuExperts {
        Self::with_format(threads, kernel, RECORD_BYTES)
    }

    pub fn with_format(threads: usize, kernel: RowKernel, record_bytes: usize) -> CpuExperts {
        assert!(record_bytes == RECORD_BYTES || record_bytes == crate::ternary::RECORD_BYTES);
        let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
        let (tx, rx) = channel::<Request>();
        let coordinator = std::thread::Builder::new()
            .name("dsv41-experts".into())
            .spawn(move || {
                let engine = Engine { pool: Pool::new(threads, "dsv41-expert"), threads, kernel, record_bytes };
                while let Ok(req) = rx.recv() {
                    let Request { records, weights, x, swiglu_limit, done } = req;
                    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| engine.forward(&records, &weights, &x, swiglu_limit)))
                        .map_err(|e| e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_else(|| "CPU expert job panicked".into()));
                    done(out);
                }
            })
            .expect("spawn expert coordinator");
        CpuExperts { tx: Some(tx), coordinator: Some(coordinator), threads }
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Start [`forward`](Self::forward) and return at once; `done` gets the
    /// outputs (or the failure) on the coordinator thread. Jobs run in order.
    pub fn spawn(&self, records: Vec<Arc<Vec<u8>>>, weights: Vec<f32>, x: Vec<f32>, swiglu_limit: f32, done: Done) {
        let req = Request { records, weights, x, swiglu_limit, done };
        if let Err(e) = self.tx.as_ref().expect("coordinator running").send(req) {
            (e.0.done)(Err("CPU expert coordinator is gone".into()));
        }
    }

    /// Each expert's output for the one-token activation `x` (`[DIM]`, bf16
    /// values), bf16-rounded: `out[e]` equals
    /// `expert_forward(&records[e], x, Some(weights[e]), swiglu_limit)`.
    pub fn forward(&self, records: &[Arc<Vec<u8>>], weights: &[f32], x: &[f32], swiglu_limit: f32) -> Vec<Vec<f32>> {
        let (tx, rx) = channel();
        self.spawn(records.to_vec(), weights.to_vec(), x.to_vec(), swiglu_limit, Box::new(move |out| drop(tx.send(out))));
        rx.recv().expect("CPU expert job answered").expect("CPU expert job")
    }
}

impl Drop for CpuExperts {
    fn drop(&mut self) {
        self.tx.take(); // the coordinator finishes queued jobs, then leaves
        if let Some(h) = self.coordinator.take() {
            let _ = h.join();
        }
    }
}

/// The pool and kernel a coordinator runs jobs with.
struct Engine {
    pool: Pool,
    threads: usize,
    kernel: RowKernel,
    record_bytes: usize,
}

impl Engine {
    fn forward(&self, records: &[Arc<Vec<u8>>], weights: &[f32], x: &[f32], swiglu_limit: f32) -> Vec<Vec<f32>> {
        assert_eq!(records.len(), weights.len());
        assert_eq!(x.len(), DIM);
        assert!(records.iter().all(|r| r.len() == self.record_bytes));
        let m = records.len();
        if m == 0 {
            return Vec::new();
        }
        let parts = self.threads.div_ceil(m);
        let xq: Arc<Vec<f32>> = Arc::new(fake_quant_fp8(x, BLOCK));

        // gate/up and SwiGLU: h[e] = bf16(silu(g) * u * weight)
        let mut jobs: Vec<Job> = Vec::new();
        let mut spans = Vec::new();
        for (e, rec) in records.iter().enumerate() {
            for (r0, len) in runs(INTER, parts) {
                let (rec, xq, w, fp4_rows) = (Arc::clone(rec), Arc::clone(&xq), weights[e], self.kernel);
                let id = jobs.len();
                spans.push((e, r0));
                jobs.push(Box::new(move || {
                    let (mut g, mut u) = (vec![0.0f32; len], vec![0.0f32; len]);
                    fp4_rows(&xq, &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::W1 } else { W1 }], &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::S1 } else { S1 }], DIM, r0, &mut g);
                    fp4_rows(&xq, &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::W3 } else { W3 }], &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::S3 } else { S3 }], DIM, r0, &mut u);
                    let h = g
                        .iter()
                        .zip(&u)
                        .map(|(&g, &u)| {
                            let (mut g, mut u) = (to_bf16(g), to_bf16(u));
                            if swiglu_limit > 0.0 {
                                u = u.clamp(-swiglu_limit, swiglu_limit);
                                g = g.min(swiglu_limit);
                            }
                            to_bf16(g / (1.0 + (-g).exp()) * u * w)
                        })
                        .collect();
                    (id, h)
                }));
            }
        }
        let mut h = vec![0.0f32; m * INTER];
        for ((e, r0), part) in spans.iter().zip(self.pool.run(jobs)) {
            h[e * INTER + r0..][..part.len()].copy_from_slice(&part);
        }
        let hq: Arc<Vec<f32>> = Arc::new(fake_quant_fp8(&h, BLOCK));

        // down: out[e] = bf16(w2 . hq[e])
        let mut jobs: Vec<Job> = Vec::new();
        let mut spans = Vec::new();
        for (e, rec) in records.iter().enumerate() {
            for (r0, len) in runs(DIM, parts) {
                let (rec, hq, fp4_rows) = (Arc::clone(rec), Arc::clone(&hq), self.kernel);
                let id = jobs.len();
                spans.push((e, r0));
                jobs.push(Box::new(move || {
                    let mut y = vec![0.0f32; len];
                    fp4_rows(&hq[e * INTER..(e + 1) * INTER], &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::W2 } else { W2 }], &rec[if rec.len() == crate::ternary::RECORD_BYTES { crate::ternary::S2 } else { S2 }], INTER, r0, &mut y);
                    (id, y.into_iter().map(to_bf16).collect())
                }));
            }
        }
        let mut out = vec![vec![0.0f32; DIM]; m];
        for ((e, r0), part) in spans.iter().zip(self.pool.run(jobs)) {
            out[*e][*r0..][..part.len()].copy_from_slice(&part);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert::expert_forward;

    fn record(seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        let mut rec: Vec<u8> = (0..RECORD_BYTES)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect();
        for r in [S1, S2, S3] {
            for (i, b) in rec[r].iter_mut().enumerate() {
                *b = 120 + ((i as u64 + seed) % 6) as u8;
            }
        }
        rec
    }

    #[test]
    fn runs_cover_rows() {
        for (rows, parts) in [(2304, 32), (5120, 5), (2304, 1), (40, 7), (16, 3)] {
            let r = runs(rows, parts);
            assert!(r.len() <= parts, "{rows}/{parts}: {} runs", r.len());
            let mut next = 0;
            for &(s, len) in &r {
                assert_eq!(s, next);
                assert!(len > 0 && s % LANES == 0);
                next = s + len;
            }
            assert_eq!(next, rows);
        }
    }

    #[test]
    fn matches_expert_forward_bit_for_bit() {
        let recs: Vec<Arc<Vec<u8>>> = (0..3).map(|e| Arc::new(record(40 + e))).collect();
        let mut s = 7u64;
        let x: Vec<f32> = (0..DIM)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                to_bf16(((s >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 8.0)
            })
            .collect();
        let weights = [0.4f32, 1.1, 0.25];
        for threads in [1, 5, 16] {
            let cpu = CpuExperts::new(threads);
            let got = cpu.forward(&recs, &weights, &x, 10.0);
            for (e, rec) in recs.iter().enumerate() {
                let want = expert_forward(rec, &x, Some(weights[e]), 10.0);
                assert!(got[e].iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "expert {e}, {threads} threads");
            }
        }
    }

    /// A prompt's expert of a few rows, a row at a time through the row kernel: the bits of `expert_forward_batch`,
    /// which decodes each weight row once for all of them (this CPU's SIMD kernel the same: `dsv41-simd`'s test).
    #[test]
    fn a_few_rows_through_a_row_kernel_are_the_batchs_bit_for_bit() {
        use crate::expert::{expert_forward_batch, expert_forward_rows};
        let rec = record(91);
        let mut s = 11u64;
        let x: Vec<f32> = (0..3 * DIM)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                to_bf16(((s >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 8.0)
            })
            .collect();
        let weights = [0.4f32, 1.1, 0.25];
        let want = expert_forward_batch(&rec, &x, Some(&weights), 10.0);
        let got = expert_forward_rows(fp4_rows, &rec, &x, Some(&weights), 10.0);
        assert_eq!(got.len(), want.len());
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        // (and with no routing weight, as the shared path gives none)
        let got = expert_forward_rows(fp4_rows, &rec, &x[..DIM], None, 0.0);
        assert!(got.iter().zip(&expert_forward_batch(&rec, &x[..DIM], None, 0.0)).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}
