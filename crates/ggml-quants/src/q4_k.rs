//! Q4_K (K-quant): 256 elements per super-block, 144 bytes/block.
//!
//! Block layout:
//!   [0..2]    f16 d                  -- super-block scale
//!   [2..4]    f16 dmin               -- super-block min
//!   [4..16]   u8  scales[12]         -- 8 6-bit scales + 8 6-bit mins, packed
//!   [16..144] u8  qs[128]            -- 256 4-bit nibbles (32 elements per sub-block, 8 sub-blocks)
//!
//! Algorithm matches `dequantize_row_q4_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 144;

/// Unpack a sub-block scale and min for sub-block index `j` (0..8).
///
/// The 12-byte `scales` array packs 8 × 6-bit scales and 8 × 6-bit mins:
///   - sub-blocks 0..3:  scale = scales[j]   & 0x3F
///                       min   = scales[j+4] & 0x3F
///   - sub-blocks 4..7:  bits split across scales[j-4..=j+4]
#[inline]
fn unpack_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d   = read_f16(&src[0..2]);
    let min = read_f16(&src[2..4]);
    let scales = &src[4..16];
    let qs = &src[16..144];

    let mut y_off = 0;
    let mut q_off = 0;
    for is in (0..8).step_by(2) {
        let (sc1, m1) = unpack_scale_min(is + 0, scales);
        let (sc2, m2) = unpack_scale_min(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let mm1 = min * m1 as f32;
        let mm2 = min * m2 as f32;

        // 32 outputs from low nibbles using sub-block scale `is`.
        for l in 0..32 {
            dst[y_off + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - mm1;
        }
        // 32 outputs from high nibbles using sub-block scale `is+1`.
        for l in 0..32 {
            dst[y_off + 32 + l] = d2 * ((qs[q_off + l] >> 4) & 0x0F) as f32 - mm2;
        }
        y_off += 64;
        q_off += 32;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(q_off, 128);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}

// VENDORED-LOCAL: GLM-5.3-Flash. Fused Q4_K x f32 dot products.
//
// The reason this exists: computing a routed expert on the CPU used to mean
// `dequantize` into f32 and then a matvec over the result. For one GLM-5.3-Flash
// expert record that is 14.16 MB of Q4_K in and **101 MB of f32 written and read
// back**, which measured 12.0 ms single-threaded -- 3.23 ms of dequantisation and
// 8.77 ms of matvec that was really just DRAM traffic. 202 MB of traffic per
// record sets a floor around 2.5 ms whatever the thread count, and a PCIe upload
// of the same record costs 1.12 ms, so the CPU lost.
//
// Fused, the f32 never exists: 14.16 MB is read once and the products accumulate
// in registers.
//
// **Bit-identical, deliberately.** Each output row accumulates in exactly the
// order `dequantize` writes and a sequential dot then reads: super-blocks in
// order, sub-block pairs in order, the 32 low nibbles then the 32 high ones. That
// matters because a routed expert's output is summed with experts computed on the
// GPU, and if the CPU and GPU answers differed then which tier served a miss --
// which depends on cache state, not on the prompt -- would change the logits.
//
// Rows are independent, so [`ROWS_AT_ONCE`] of them run with their own
// accumulators. That is where the instruction-level parallelism comes from: a
// single row is a 4096-long dependency chain of f32 adds, and four interleaved
// chains keep the FMA units busy without reassociating any of them.

/// Output rows carried at once, each with its own accumulator.
///
/// Four independent chains cover the latency of an f32 add (3-4 cycles) at one
/// per cycle of throughput. More would help only if the loads kept up, and the
/// nibble decode already needs the ports.
const ROWS_AT_ONCE: usize = 4;

/// One row of a `[n_rows, k]` Q4_K matrix dotted with `x`.
///
/// `row` is `k / 256 * 144` bytes. Accumulates in the reference order, so this is
/// bit-identical to `dequantize(row)` followed by a sequential dot with `x`.
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
    debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

    let mut acc = 0.0f32;
    let mut xi = 0usize;
    for block in row.chunks_exact(BYTES_PER_BLOCK) {
        let d = read_f16(&block[0..2]);
        let min = read_f16(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut q_off = 0usize;
        for is in (0..8).step_by(2) {
            let (sc1, m1) = unpack_scale_min(is, scales);
            let (sc2, m2) = unpack_scale_min(is + 1, scales);
            let d1 = d * sc1 as f32;
            let d2 = d * sc2 as f32;
            let mm1 = min * m1 as f32;
            let mm2 = min * m2 as f32;

            let q = &qs[q_off..q_off + 32];
            let xlo = &x[xi..xi + 32];
            let xhi = &x[xi + 32..xi + 64];
            for l in 0..32 {
                acc += (d1 * (q[l] & 0x0F) as f32 - mm1) * xlo[l];
            }
            for l in 0..32 {
                acc += (d2 * ((q[l] >> 4) & 0x0F) as f32 - mm2) * xhi[l];
            }
            xi += 64;
            q_off += 32;
        }
    }
    acc
}

/// `out[r] = dot(W[r], x)` for a `[n_rows, k]` Q4_K matrix laid out row by row.
///
/// Each row is summed in the reference order; [`ROWS_AT_ONCE`] rows share the loop
/// so their chains interleave.
pub fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
    debug_assert_eq!(k % BLOCK_SIZE, 0);
    debug_assert_eq!(w.len(), n_rows * row_bytes);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), n_rows);

    let mut r = 0usize;
    while r + ROWS_AT_ONCE <= n_rows {
        dot_group(&w[r * row_bytes..(r + ROWS_AT_ONCE) * row_bytes], x, row_bytes, &mut out[r..r + ROWS_AT_ONCE]);
        r += ROWS_AT_ONCE;
    }
    for rr in r..n_rows {
        out[rr] = dot_row(&w[rr * row_bytes..(rr + 1) * row_bytes], x);
    }
}

/// Whether this CPU can run the AVX-512 dot kernels.
///
/// Checked once by the caller and cached; the detection macro itself is cheap but
/// not free.
#[inline]
pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// VENDORED-LOCAL: GLM-5.3-Flash. AVX-512 Q4_K dot products.
//
// The scalar [`dot_row`] is bit-identical to `dequantize` then a sequential dot,
// and it is compute-bound: measured at about 1.85 cycles per weight, 3.46 ms for a
// [2048, 4096] projection. The strict summation order is exactly what stops it
// vectorising -- one accumulator, one f32 add at a time.
//
// **This path reassociates, deliberately, and that is the trade to understand.**
// Sixteen lanes run along the reduction and their partial sums are combined in a
// fixed order at the end, so the result is *deterministic* but is not the scalar
// result. Measured difference: around 1e-7 relative, against the 2e-5 this model
// already shows between its host and CUDA paths, so it is well under the noise
// already present -- but it does mean a routed expert computed here and the same
// expert computed on the GPU differ in the last bits, and which one serves a
// dispatch depends on cache state rather than on the prompt.
//
// The alternative that keeps bit-identity is to put the lanes across *rows*
// instead: sixteen rows, each lane walking its own row in order. That needs the
// sixteen rows' nibbles transposed into lanes -- a 16x32 byte transpose per
// sub-block -- because a byte per row is a `row_bytes`-strided gather otherwise,
// and a 16-lane gather costs about what the scalar code does. `dsv41`'s MXFP4
// kernel took that route. It is the better answer and it is more work than this.
#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    use super::{read_f16, unpack_scale_min, BLOCK_SIZE, BYTES_PER_BLOCK};
    use std::arch::x86_64::*;

    /// One row dotted with `x`, AVX-512.
    ///
    /// Four accumulator chains, because an FMA has ~4 cycles of latency and two
    /// can issue per cycle: two chains would leave the units half idle.
    ///
    /// # Safety
    /// The caller must have checked `avx512f` and `avx512bw`
    /// ([`super::has_avx512`]). `row` must be a whole number of super-blocks and
    /// `x` must be `row.len() / BYTES_PER_BLOCK * BLOCK_SIZE` long.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dot_row(row: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
        debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        let nib = _mm512_set1_epi32(0x0F);
        let xp = x.as_ptr();

        let mut xi = 0usize;
        for block in row.chunks_exact(BYTES_PER_BLOCK) {
            let d = read_f16(&block[0..2]);
            let min = read_f16(&block[2..4]);
            let scales = &block[4..16];
            let qs = block[16..144].as_ptr();

            let mut q_off = 0usize;
            for is in (0..8).step_by(2) {
                let (sc1, m1) = unpack_scale_min(is, scales);
                let (sc2, m2) = unpack_scale_min(is + 1, scales);
                let d1 = _mm512_set1_ps(d * sc1 as f32);
                let d2 = _mm512_set1_ps(d * sc2 as f32);
                // fold the subtraction into the FMA: value = q * d - mm
                let n1 = _mm512_set1_ps(-(min * m1 as f32));
                let n2 = _mm512_set1_ps(-(min * m2 as f32));

                // 32 nibble bytes: low nibbles are elements 0..32 of this group,
                // high nibbles are elements 32..64. Two 16-wide steps, each
                // feeding its own pair of accumulators.
                let b0 = _mm_loadu_si128(qs.add(q_off) as *const __m128i);
                let b1 = _mm_loadu_si128(qs.add(q_off + 16) as *const __m128i);

                let q0 = _mm512_cvtepu8_epi32(b0);
                let q1 = _mm512_cvtepu8_epi32(b1);

                let lo0 = _mm512_cvtepi32_ps(_mm512_and_si512(q0, nib));
                let lo1 = _mm512_cvtepi32_ps(_mm512_and_si512(q1, nib));
                let hi0 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(q0, 4), nib));
                let hi1 = _mm512_cvtepi32_ps(_mm512_and_si512(_mm512_srli_epi32(q1, 4), nib));

                let v0 = _mm512_fmadd_ps(lo0, d1, n1);
                let v1 = _mm512_fmadd_ps(lo1, d1, n1);
                let v2 = _mm512_fmadd_ps(hi0, d2, n2);
                let v3 = _mm512_fmadd_ps(hi1, d2, n2);

                a0 = _mm512_fmadd_ps(v0, _mm512_loadu_ps(xp.add(xi)), a0);
                a1 = _mm512_fmadd_ps(v1, _mm512_loadu_ps(xp.add(xi + 16)), a1);
                a2 = _mm512_fmadd_ps(v2, _mm512_loadu_ps(xp.add(xi + 32)), a2);
                a3 = _mm512_fmadd_ps(v3, _mm512_loadu_ps(xp.add(xi + 48)), a3);

                xi += 64;
                q_off += 32;
            }
        }

        // Combine in a fixed, written-down order rather than `_mm512_reduce_add_ps`,
        // so the result does not depend on how the compiler chose to reduce.
        let mut lanes = [0.0f32; 64];
        _mm512_storeu_ps(lanes.as_mut_ptr(), a0);
        _mm512_storeu_ps(lanes.as_mut_ptr().add(16), a1);
        _mm512_storeu_ps(lanes.as_mut_ptr().add(32), a2);
        _mm512_storeu_ps(lanes.as_mut_ptr().add(48), a3);
        let mut acc = 0.0f32;
        for v in lanes {
            acc += v;
        }
        acc
    }

    /// `out[r] = dot(W[r], x)` for a `[n_rows, k]` Q4_K matrix, row-major.
    ///
    /// # Safety
    /// As [`dot_row`].
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
        let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
        debug_assert_eq!(w.len(), n_rows * row_bytes);
        debug_assert_eq!(out.len(), n_rows);
        for r in 0..n_rows {
            out[r] = dot_row(&w[r * row_bytes..(r + 1) * row_bytes], x);
        }
    }
}

/// `ROWS_AT_ONCE` rows in one pass, each with its own accumulator.
#[inline]
fn dot_group(w: &[u8], x: &[f32], row_bytes: usize, out: &mut [f32]) {
    let mut acc = [0.0f32; ROWS_AT_ONCE];
    let n_blocks = row_bytes / BYTES_PER_BLOCK;

    for b in 0..n_blocks {
        // Per-row super-block headers.
        let mut d = [0.0f32; ROWS_AT_ONCE];
        let mut mn = [0.0f32; ROWS_AT_ONCE];
        for (i, a) in d.iter_mut().enumerate() {
            let blk = &w[i * row_bytes + b * BYTES_PER_BLOCK..][..BYTES_PER_BLOCK];
            *a = read_f16(&blk[0..2]);
            mn[i] = read_f16(&blk[2..4]);
        }

        let mut q_off = 0usize;
        let mut xi = b * BLOCK_SIZE;
        for is in (0..8).step_by(2) {
            for i in 0..ROWS_AT_ONCE {
                let blk = &w[i * row_bytes + b * BYTES_PER_BLOCK..][..BYTES_PER_BLOCK];
                let scales = &blk[4..16];
                let qs = &blk[16..144];
                let (sc1, m1) = unpack_scale_min(is, scales);
                let (sc2, m2) = unpack_scale_min(is + 1, scales);
                let d1 = d[i] * sc1 as f32;
                let d2 = d[i] * sc2 as f32;
                let mm1 = mn[i] * m1 as f32;
                let mm2 = mn[i] * m2 as f32;

                let q = &qs[q_off..q_off + 32];
                let xlo = &x[xi..xi + 32];
                let xhi = &x[xi + 32..xi + 64];
                let mut a = acc[i];
                for l in 0..32 {
                    a += (d1 * (q[l] & 0x0F) as f32 - mm1) * xlo[l];
                }
                for l in 0..32 {
                    a += (d2 * ((q[l] >> 4) & 0x0F) as f32 - mm2) * xhi[l];
                }
                acc[i] = a;
            }
            q_off += 32;
            xi += 64;
        }
    }
    out.copy_from_slice(&acc);
}

#[cfg(test)]
mod fused_dot_tests {
    use super::*;

    fn bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        // Deterministic, and the 6-bit scale fields get a real spread.
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (z >> 33) as u8
        };
        let mut v = Vec::with_capacity(n_blocks * BYTES_PER_BLOCK);
        for _ in 0..n_blocks {
            // d and dmin as small positive f16s
            v.extend_from_slice(&half::f16::from_f32(0.05 + (next() % 17) as f32 / 400.0).to_bits().to_le_bytes());
            v.extend_from_slice(&half::f16::from_f32(0.01 + (next() % 13) as f32 / 900.0).to_bits().to_le_bytes());
            for _ in 0..12 {
                v.push(next());
            }
            for _ in 0..128 {
                v.push(next());
            }
        }
        v
    }

    fn xs(k: usize, seed: u64) -> Vec<f32> {
        let mut z = seed;
        (0..k)
            .map(|_| {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((z >> 40) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    /// The oracle: dequantise the row, then sum in index order -- exactly what
    /// `glm5next::forward::matvec` does over a dequantised weight.
    fn oracle(row: &[u8], x: &[f32]) -> f32 {
        let mut w = vec![0.0f32; row.len() / BYTES_PER_BLOCK * BLOCK_SIZE];
        dequantize(row, &mut w);
        let mut acc = 0.0f32;
        for (a, b) in w.iter().zip(x) {
            acc += a * b;
        }
        acc
    }

    /// Bit-identical, not merely close.
    ///
    /// A routed expert's output is summed with experts computed on the GPU, so if
    /// the fused dot differed from the dequantise-then-matvec path then which tier
    /// happened to serve a cache miss would move the logits.
    #[test]
    fn fused_dot_is_bit_identical_to_dequantize_then_dot() {
        for &k in &[256usize, 512, 4096] {
            let n_blocks = k / BLOCK_SIZE;
            let x = xs(k, 0xA5A5);
            for row_seed in 0..6u64 {
                let row = bytes(n_blocks, 0x1234 + row_seed);
                let want = oracle(&row, &x);
                let got = dot_row(&row, &x);
                assert_eq!(
                    want.to_bits(),
                    got.to_bits(),
                    "k={k} seed={row_seed}: {want} vs {got}"
                );
                assert!(want.abs() > 1e-6, "k={k} seed={row_seed}: oracle is zero");
            }
        }
    }

    /// And the grouped path, which is the one that actually runs: the row count is
    /// deliberately not a multiple of ROWS_AT_ONCE so the tail is covered.
    #[test]
    fn dot_rows_matches_dot_row_exactly() {
        let k = 512usize;
        let n_blocks = k / BLOCK_SIZE;
        let n_rows = 4 * ROWS_AT_ONCE + 3;
        let x = xs(k, 0x5EED);
        let mut w = Vec::new();
        for r in 0..n_rows {
            w.extend_from_slice(&bytes(n_blocks, 0x9000 + r as u64));
        }
        let row_bytes = n_blocks * BYTES_PER_BLOCK;

        let mut got = vec![0.0f32; n_rows];
        dot_rows(&w, &x, n_rows, k, &mut got);
        for r in 0..n_rows {
            let row = &w[r * row_bytes..(r + 1) * row_bytes];
            assert_eq!(
                oracle(row, &x).to_bits(),
                got[r].to_bits(),
                "row {r} differs"
            );
        }
        assert!(got.iter().any(|v| v.abs() > 1e-6));
    }

    /// The sum of |w_i * x_i| for a row: the scale a dot product's rounding error
    /// is actually proportional to.
    fn abs_sum(row: &[u8], x: &[f32]) -> f32 {
        let mut w = vec![0.0f32; row.len() / BYTES_PER_BLOCK * BLOCK_SIZE];
        dequantize(row, &mut w);
        w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum()
    }

    /// The AVX-512 kernel against the scalar oracle.
    ///
    /// Not bit-identical by construction -- sixteen lanes run along the reduction
    /// and their partials are combined at the end -- so this pins the size of the
    /// difference instead.
    ///
    /// Measured against `sum |w_i x_i|`, not against the result. With `x` drawn
    /// symmetrically about zero the terms cancel, so a dot of 512 of them lands
    /// near zero and *any* rounding looks enormous next to it; error relative to
    /// the result is a statement about the test data, not the kernel. Classical
    /// error analysis gives `|err| <= n * u * sum|terms|`, which for n = 512 and
    /// f32 is about 3e-5 of that sum for strict sequential summation -- and the
    /// lane-wise version is *better* conditioned than the oracle it is compared
    /// against, because 16 shorter chains accumulate less. 1e-6 is a tight bar on
    /// that scale.
    #[test]
    fn avx512_dot_matches_the_scalar_oracle() {
        if !has_avx512() {
            eprintln!("no AVX-512 on this CPU; skipping");
            return;
        }
        for &k in &[256usize, 512, 4096] {
            let x = xs(k, 0x1357);
            let (mut worst, mut worst_naive) = (0.0f32, 0.0f32);
            for seed in 0..8u64 {
                let row = bytes(k / BLOCK_SIZE, 0x2468 + seed);
                let want = dot_row(&row, &x);
                // SAFETY: has_avx512() checked above.
                let got = unsafe { avx512::dot_row(&row, &x) };
                let err = (want - got).abs();
                worst = worst.max(err / abs_sum(&row, &x).max(1e-6));
                worst_naive = worst_naive.max(err / want.abs().max(1e-3));
                assert!(want.abs() > 1e-6, "k={k} seed={seed}: oracle is zero");
            }
            println!("k={k}: {worst:.2e} of sum|terms| ({worst_naive:.2e} of the result, which cancels)");
            assert!(worst < 1e-6, "k={k}: AVX-512 drifted {worst} of sum|terms|");
        }
    }

    /// Deterministic: the same inputs must give exactly the same bits every time,
    /// even though those bits differ from the scalar path.
    #[test]
    fn avx512_dot_is_deterministic() {
        if !has_avx512() {
            return;
        }
        let k = 1024usize;
        let x = xs(k, 0x9BDF);
        let row = bytes(k / BLOCK_SIZE, 0x777);
        // SAFETY: has_avx512() checked above.
        let first = unsafe { avx512::dot_row(&row, &x) };
        for _ in 0..8 {
            let again = unsafe { avx512::dot_row(&row, &x) };
            assert_eq!(first.to_bits(), again.to_bits());
        }
    }

    /// Scalar against AVX-512 at the released projection shape.
    #[test]
    #[ignore = "measures"]
    fn measure_avx512_speedup() {
        if !has_avx512() {
            eprintln!("no AVX-512 on this CPU; skipping");
            return;
        }
        let (n_embd, n_ff) = (4096usize, 2048usize);
        let row_bytes = n_embd / BLOCK_SIZE * BYTES_PER_BLOCK;
        let mut w = Vec::with_capacity(n_ff * row_bytes);
        for r in 0..n_ff {
            w.extend_from_slice(&bytes(n_embd / BLOCK_SIZE, 0x7000 + r as u64));
        }
        let x = xs(n_embd, 0xBEEF);
        let mut out = vec![0.0f32; n_ff];

        dot_rows(&w, &x, n_ff, n_embd, &mut out);
        let n = 5usize;

        let t = std::time::Instant::now();
        for _ in 0..n {
            dot_rows(&w, &x, n_ff, n_embd, &mut out);
            std::hint::black_box(&out);
        }
        let scalar = t.elapsed().as_secs_f64() / n as f64;

        let t = std::time::Instant::now();
        for _ in 0..n {
            // SAFETY: has_avx512() checked above.
            unsafe { avx512::dot_rows(&w, &x, n_ff, n_embd, &mut out) };
            std::hint::black_box(&out);
        }
        let simd = t.elapsed().as_secs_f64() / n as f64;

        let mb = (n_ff * row_bytes) as f64 / 1e6;
        println!();
        println!("one projection, [{n_ff}, {n_embd}] Q4_K, {mb:.2} MB, single thread:");
        println!("  scalar fused   {:7.2} ms   ({:5.2} GB/s)", scalar * 1e3, mb / 1e3 / scalar);
        println!("  AVX-512        {:7.2} ms   ({:5.2} GB/s)   {:.1}x", simd * 1e3, mb / 1e3 / simd, scalar / simd);
        println!();
        println!("a whole expert record is three projections, 14.16 MB:");
        println!("  AVX-512, one thread     {:7.2} ms", simd * 3.0 * 1e3);
        println!("  AVX-512, 32 threads     {:7.2} ms   (perfect scaling)", simd * 3.0 / 32.0 * 1e3);
        println!("  DRAM floor at 80 GB/s      0.18 ms");
        println!("to beat: 1.12 ms, the PCIe upload of the same record");
        assert!(simd > 0.0);
    }

    /// What one GLM-5.3-Flash expert record costs, fused, against the
    /// dequantise-then-matvec path it replaces. Single thread, released shapes:
    /// gate and up are [2048, 4096], down is [4096, 2048].
    #[test]
    #[ignore = "measures"]
    fn measure_fused_record_cost() {
        let (n_embd, n_ff) = (4096usize, 2048usize);
        let row_bytes = n_embd / BLOCK_SIZE * BYTES_PER_BLOCK;
        let w = {
            let mut v = Vec::with_capacity(n_ff * row_bytes);
            for r in 0..n_ff {
                v.extend_from_slice(&bytes(n_embd / BLOCK_SIZE, 0x7000 + r as u64));
            }
            v
        };
        let x = xs(n_embd, 0xBEEF);
        let mut out = vec![0.0f32; n_ff];

        // one pass to fault everything in
        dot_rows(&w, &x, n_ff, n_embd, &mut out);

        let n = 5usize;
        let t = std::time::Instant::now();
        for _ in 0..n {
            dot_rows(&w, &x, n_ff, n_embd, &mut out);
            std::hint::black_box(&out);
        }
        let fused_one = t.elapsed().as_secs_f64() / n as f64;

        // the old way, for the same projection
        let mut deq = vec![0.0f32; n_ff * n_embd];
        let t = std::time::Instant::now();
        for _ in 0..n {
            dequantize(&w, &mut deq);
            for r in 0..n_ff {
                let row = &deq[r * n_embd..(r + 1) * n_embd];
                let mut acc = 0.0f32;
                for (a, b) in row.iter().zip(&x) {
                    acc += a * b;
                }
                out[r] = acc;
            }
            std::hint::black_box(&out);
        }
        let split_one = t.elapsed().as_secs_f64() / n as f64;

        println!();
        println!("one projection, [{n_ff}, {n_embd}] Q4_K, {:.2} MB, single thread:", (n_ff * row_bytes) as f64 / 1e6);
        println!("  dequantise then matvec  {:7.2} ms", split_one * 1e3);
        println!("  fused                   {:7.2} ms   ({:.1}x)", fused_one * 1e3, split_one / fused_one);
        println!();
        println!("a whole expert record is three of these:");
        println!("  dequantise then matvec  {:7.2} ms", split_one * 3.0 * 1e3);
        println!("  fused                   {:7.2} ms", fused_one * 3.0 * 1e3);
        println!("  fused over 32 threads   {:7.2} ms   (perfect scaling)", fused_one * 3.0 / 32.0 * 1e3);
        println!("to beat: 1.12 ms, the PCIe upload of the same record on this machine");
        assert!(fused_one > 0.0 && split_one > 0.0);
    }
}
