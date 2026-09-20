//! Q6_K (K-quant): 256 elements per super-block, 210 bytes/block.
//!
//! Block layout:
//!   [0..128]   u8 ql[128]         -- low 4 bits of each value
//!   [128..192] u8 qh[64]          -- high 2 bits, packed
//!   [192..208] i8 scales[16]      -- one signed scale per 16-element group
//!   [208..210] f16 d              -- super-block scale
//!
//! Each value is reconstructed as `(low4 | (high2 << 4)) - 32` × `d` × `scales[group]`.
//! Algorithm matches `dequantize_row_q6_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 210;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let ql = &src[0..128];
    let qh = &src[128..192];
    // scales are signed
    let scales: &[i8] = unsafe {
        std::slice::from_raw_parts(src[192..208].as_ptr() as *const i8, 16)
    };
    let d = read_f16(&src[208..210]);

    let mut y_off = 0;
    let mut ql_off = 0;
    let mut qh_off = 0;
    let mut sc_off = 0;

    for _ in 0..2 {
        // Process 128 outputs per outer iteration.
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l]      & 0x0F) | (((qh[qh_off + l] >> 0) & 0x3) << 4)) as i8 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 0x3) << 4)) as i8 - 32;
            let q3 = ((ql[ql_off + l]      >> 4)   | (((qh[qh_off + l] >> 4) & 0x3) << 4)) as i8 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4)   | (((qh[qh_off + l] >> 6) & 0x3) << 4)) as i8 - 32;
            dst[y_off + l]       = d * scales[sc_off + is]     as f32 * q1 as f32;
            dst[y_off + l + 32]  = d * scales[sc_off + is + 2] as f32 * q2 as f32;
            dst[y_off + l + 64]  = d * scales[sc_off + is + 4] as f32 * q3 as f32;
            dst[y_off + l + 96]  = d * scales[sc_off + is + 6] as f32 * q4 as f32;
        }
        y_off  += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(ql_off, 128);
    debug_assert_eq!(qh_off, 64);
    debug_assert_eq!(sc_off, 16);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}

// VENDORED-LOCAL: GLM-5.3-Flash. Fused Q6_K x f32 dot products.
//
// The companion to `q4_k::dot_rows`, and needed for the same reason: GLM-5.3-Flash
// quantises `ffn_down_exps` at Q6_K on 18 of its 42 MoE layers, so a third of the
// expert bytes on the CPU path come through here.
//
// **Bit-identical to `dequantize` then a sequential dot**, which takes a little
// care because `dequantize_block` writes a block out of order: within each
// 128-output half it emits index `l`, `l+32`, `l+64` and `l+96` together, from one
// `ql`/`qh` byte. Summing in index order therefore means four passes over the same
// 32 bytes -- q1 for `l` in 0..32, then q2, then q3, then q4 -- rather than one
// interleaved pass. Four passes over 32 bytes is nothing; the bytes are in L1
// after the first.
const ROWS_AT_ONCE: usize = 4;

/// One row of a `[n_rows, k]` Q6_K matrix dotted with `x`, in index order.
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
    debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

    let mut acc = 0.0f32;
    let mut xi = 0usize;
    for block in row.chunks_exact(BYTES_PER_BLOCK) {
        let d = read_f16(&block[208..210]);
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for _ in 0..2 {
            let ql = &block[ql_off..ql_off + 64];
            let qh = &block[128 + qh_off..128 + qh_off + 32];
            let sc = &block[192 + sc_off..192 + sc_off + 8];

            // The four quarters, in the order their outputs are indexed.
            for quarter in 0..4usize {
                // shift into qh, and which 32 ql bytes / which scale pair
                let (shift, ql_base, sc_base) = match quarter {
                    0 => (0u32, 0usize, 0usize),
                    1 => (2, 32, 2),
                    2 => (4, 0, 4),
                    _ => (6, 32, 6),
                };
                let high_nibble = quarter >= 2;
                for l in 0..32usize {
                    let low = if high_nibble {
                        ql[ql_base + l] >> 4
                    } else {
                        ql[ql_base + l] & 0x0F
                    };
                    let q = (low | (((qh[l] >> shift) & 0x3) << 4)) as i8 - 32;
                    let s = sc[sc_base + l / 16] as f32;
                    acc += d * s * q as f32 * x[xi + l];
                }
                xi += 32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    acc
}

/// `out[r] = dot(W[r], x)` for a `[n_rows, k]` Q6_K matrix, row-major.
pub fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
    debug_assert_eq!(k % BLOCK_SIZE, 0);
    debug_assert_eq!(w.len(), n_rows * row_bytes);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), n_rows);

    // Groups of ROWS_AT_ONCE so their accumulator chains interleave; each row is
    // still summed strictly in index order.
    let mut r = 0usize;
    while r + ROWS_AT_ONCE <= n_rows {
        for i in 0..ROWS_AT_ONCE {
            out[r + i] = dot_row(&w[(r + i) * row_bytes..(r + i + 1) * row_bytes], x);
        }
        r += ROWS_AT_ONCE;
    }
    for rr in r..n_rows {
        out[rr] = dot_row(&w[rr * row_bytes..(rr + 1) * row_bytes], x);
    }
}

/// Whether this CPU can run the AVX-512 dot kernels.
#[inline]
pub fn has_avx512() -> bool {
    crate::q4_k::has_avx512()
}

// VENDORED-LOCAL: GLM-5.3-Flash. AVX-512 Q6_K dot products.
//
// GLM-5.3-Flash quantises `ffn_down_exps` at Q6_K on 18 of its 42 MoE layers, so a
// third of the bytes the CPU expert tier touches come through here and the scalar
// fused version -- 1.85 cycles a weight -- would otherwise dominate the record.
//
// Same trade as `q4_k::avx512`: sixteen lanes along the reduction, four
// accumulator chains, partials combined in a written-down fixed order. Not
// bit-identical to the scalar path, deterministic, and within f32 unit roundoff of
// it.
//
// The layout is fiddlier than Q4_K. A 256-value super-block is two halves of 128,
// each half four quarters of 32, and a quarter takes its low 4 bits from one of
// two 32-byte `ql` runs and its high 2 bits from a 2-bit field of `qh` -- shift 0,
// 2, 4, 6 for quarters 0..3 -- with the nibble half (low or high of `ql`) changing
// at quarter 2. Scales are **signed** i8, one per 16 values, so a 16-lane step
// lands on exactly one scale and it can be broadcast.
#[cfg(target_arch = "x86_64")]
pub mod avx512 {
    use super::{read_f16, BLOCK_SIZE, BYTES_PER_BLOCK};
    use std::arch::x86_64::*;

    /// One row dotted with `x`, AVX-512.
    ///
    /// # Safety
    /// The caller must have checked `avx512f` and `avx512bw`
    /// ([`super::has_avx512`]). `row` must be a whole number of super-blocks and
    /// `x` must be `row.len() / BYTES_PER_BLOCK * BLOCK_SIZE` long.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn dot_row(row: &[u8], x: &[f32]) -> f32 {
        debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
        debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

        let mut acc = [
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
            _mm512_setzero_ps(),
        ];
        let lo4 = _mm512_set1_epi32(0x0F);
        let two = _mm512_set1_epi32(0x03);
        let bias = _mm512_set1_epi32(32);
        let xp = x.as_ptr();

        let mut xi = 0usize;
        for block in row.chunks_exact(BYTES_PER_BLOCK) {
            let d = read_f16(&block[208..210]);
            let qlp = block.as_ptr();
            let qhp = block.as_ptr().add(128);
            // Signed scales.
            let scp = block.as_ptr().add(192) as *const i8;

            for half in 0..2usize {
                let ql = qlp.add(half * 64);
                let qh = qhp.add(half * 32);
                let sc = scp.add(half * 8);

                for quarter in 0..4usize {
                    // (shift into qh, which 32-byte ql run, which scale pair)
                    let (shift, ql_base, sc_base) = match quarter {
                        0 => (0i32, 0usize, 0usize),
                        1 => (2, 32, 2),
                        2 => (4, 0, 4),
                        _ => (6, 32, 6),
                    };
                    let high_nibble = quarter >= 2;

                    // Two 16-lane steps; each lands on one scale (one per 16).
                    for step in 0..2usize {
                        let o = step * 16;
                        let qlv = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                            ql.add(ql_base + o) as *const __m128i,
                        ));
                        let qhv = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                            qh.add(o) as *const __m128i,
                        ));
                        let low = if high_nibble {
                            _mm512_srli_epi32(qlv, 4)
                        } else {
                            _mm512_and_si512(qlv, lo4)
                        };
                        let high = _mm512_slli_epi32(
                            _mm512_and_si512(_mm512_srlv_epi32(qhv, _mm512_set1_epi32(shift)), two),
                            4,
                        );
                        // 0..63, then the -32 offset: no i8 wrap to worry about.
                        let q = _mm512_sub_epi32(_mm512_or_si512(low, high), bias);
                        let qf = _mm512_cvtepi32_ps(q);

                        let dsc = _mm512_set1_ps(d * (*sc.add(sc_base + step)) as f32);
                        let xv = _mm512_loadu_ps(xp.add(xi + o));
                        acc[quarter] =
                            _mm512_fmadd_ps(_mm512_mul_ps(qf, xv), dsc, acc[quarter]);
                    }
                    xi += 32;
                }
            }
        }

        let mut lanes = [0.0f32; 64];
        for (i, a) in acc.iter().enumerate() {
            _mm512_storeu_ps(lanes.as_mut_ptr().add(i * 16), *a);
        }
        let mut total = 0.0f32;
        for v in lanes {
            total += v;
        }
        total
    }

    /// `out[r] = dot(W[r], x)` for a `[n_rows, k]` Q6_K matrix, row-major.
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

#[cfg(test)]
mod fused_dot_tests {
    use super::*;

    fn bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (z >> 33) as u8
        };
        let mut v = Vec::with_capacity(n_blocks * BYTES_PER_BLOCK);
        for _ in 0..n_blocks {
            for _ in 0..192 {
                v.push(next());
            }
            // signed scales, kept modest so products stay finite
            for _ in 0..16 {
                v.push((next() % 63) as u8);
            }
            v.extend_from_slice(&half::f16::from_f32(0.02 + (next() % 11) as f32 / 500.0).to_bits().to_le_bytes());
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

    fn oracle(row: &[u8], x: &[f32]) -> f32 {
        let mut w = vec![0.0f32; row.len() / BYTES_PER_BLOCK * BLOCK_SIZE];
        dequantize(row, &mut w);
        let mut acc = 0.0f32;
        for (a, b) in w.iter().zip(x) {
            acc += a * b;
        }
        acc
    }

    fn abs_sum(row: &[u8], x: &[f32]) -> f32 {
        let mut w = vec![0.0f32; row.len() / BYTES_PER_BLOCK * BLOCK_SIZE];
        dequantize(row, &mut w);
        w.iter().zip(x).map(|(a, b)| (a * b).abs()).sum()
    }

    /// The AVX-512 kernel against the scalar oracle, relative to `sum |w_i x_i|`
    /// -- see the Q4_K test for why that is the right scale and the result is not.
    #[test]
    fn avx512_dot_matches_the_scalar_oracle() {
        if !has_avx512() {
            eprintln!("no AVX-512 on this CPU; skipping");
            return;
        }
        for &k in &[256usize, 512, 2048] {
            let x = xs(k, 0x2244);
            let mut worst = 0.0f32;
            for seed in 0..8u64 {
                let row = bytes(k / BLOCK_SIZE, 0x8800 + seed);
                let want = dot_row(&row, &x);
                // SAFETY: has_avx512() checked above.
                let got = unsafe { avx512::dot_row(&row, &x) };
                worst = worst.max((want - got).abs() / abs_sum(&row, &x).max(1e-6));
                assert!(want.abs() > 1e-6, "k={k} seed={seed}: oracle is zero");
            }
            println!("k={k}: {worst:.2e} of sum|terms|");
            assert!(worst < 1e-6, "k={k}: AVX-512 drifted {worst} of sum|terms|");
        }
    }

    #[test]
    fn avx512_dot_is_deterministic() {
        if !has_avx512() {
            return;
        }
        let k = 1024usize;
        let x = xs(k, 0x3355);
        let row = bytes(k / BLOCK_SIZE, 0x999);
        // SAFETY: has_avx512() checked above.
        let first = unsafe { avx512::dot_row(&row, &x) };
        for _ in 0..8 {
            assert_eq!(first.to_bits(), unsafe { avx512::dot_row(&row, &x) }.to_bits());
        }
    }

    /// Scalar against AVX-512 at the released `ffn_down_exps` shape.
    #[test]
    #[ignore = "measures"]
    fn measure_avx512_speedup() {
        if !has_avx512() {
            return;
        }
        let (n_embd, n_ff) = (4096usize, 2048usize);
        let row_bytes = n_ff / BLOCK_SIZE * BYTES_PER_BLOCK;
        let mut w = Vec::with_capacity(n_embd * row_bytes);
        for r in 0..n_embd {
            w.extend_from_slice(&bytes(n_ff / BLOCK_SIZE, 0x5000 + r as u64));
        }
        let x = xs(n_ff, 0xCAFE);
        let mut out = vec![0.0f32; n_embd];

        dot_rows(&w, &x, n_embd, n_ff, &mut out);
        let n = 5usize;
        let t = std::time::Instant::now();
        for _ in 0..n {
            dot_rows(&w, &x, n_embd, n_ff, &mut out);
            std::hint::black_box(&out);
        }
        let scalar = t.elapsed().as_secs_f64() / n as f64;
        let t = std::time::Instant::now();
        for _ in 0..n {
            // SAFETY: has_avx512() checked above.
            unsafe { avx512::dot_rows(&w, &x, n_embd, n_ff, &mut out) };
            std::hint::black_box(&out);
        }
        let simd = t.elapsed().as_secs_f64() / n as f64;
        let mb = (n_embd * row_bytes) as f64 / 1e6;
        println!();
        println!("ffn_down [{n_embd}, {n_ff}] Q6_K, {mb:.2} MB, single thread:");
        println!("  scalar fused   {:7.2} ms   ({:5.2} GB/s)", scalar * 1e3, mb / 1e3 / scalar);
        println!("  AVX-512        {:7.2} ms   ({:5.2} GB/s)   {:.1}x", simd * 1e3, mb / 1e3 / simd, scalar / simd);
        assert!(simd > 0.0);
    }

    /// Bit-identical, including the out-of-order write pattern this has to undo.
    #[test]
    fn fused_dot_is_bit_identical_to_dequantize_then_dot() {
        for &k in &[256usize, 512, 2048] {
            let x = xs(k, 0xC0DE);
            for seed in 0..6u64 {
                let row = bytes(k / BLOCK_SIZE, 0x4321 + seed);
                let want = oracle(&row, &x);
                let got = dot_row(&row, &x);
                assert_eq!(want.to_bits(), got.to_bits(), "k={k} seed={seed}: {want} vs {got}");
                assert!(want.abs() > 1e-6, "k={k} seed={seed}: oracle is zero");
            }
        }
    }

    #[test]
    fn dot_rows_matches_dot_row_exactly() {
        let k = 512usize;
        let n_rows = 3 * ROWS_AT_ONCE + 2;
        let x = xs(k, 0xF00D);
        let mut w = Vec::new();
        for r in 0..n_rows {
            w.extend_from_slice(&bytes(k / BLOCK_SIZE, 0xB000 + r as u64));
        }
        let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
        let mut got = vec![0.0f32; n_rows];
        dot_rows(&w, &x, n_rows, k, &mut got);
        for r in 0..n_rows {
            assert_eq!(
                oracle(&w[r * row_bytes..(r + 1) * row_bytes], &x).to_bits(),
                got[r].to_bits(),
                "row {r} differs"
            );
        }
        assert!(got.iter().any(|v| v.abs() > 1e-6));
    }
}

