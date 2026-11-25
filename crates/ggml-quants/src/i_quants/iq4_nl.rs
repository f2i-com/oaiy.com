//! IQ4_NL (importance-weighted 4-bit, non-linear): 32 elements per block, 18 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..18]  u8  qs[16]      (32 4-bit indices into the lookup table)
//!
//! Dequant: y[i] = d * KVALUES[qs[i/2] >> (4 * (i & 1))]
//!
//! The non-linear part is the 16-entry lookup table — values are spaced to better
//! match Gaussian-distributed weights than uniform 4-bit quantization.
//!
//! Algorithm matches `dequantize_row_iq4_nl` in `ggml-quants.c`.

use std::sync::OnceLock;

use crate::{read_f16, read_f16_fast};

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

/// Non-linear lookup table from upstream ggml-quants. Spaced to approximate
/// a Gaussian distribution of weights better than uniform [-8, 7] would.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
];

const KVALUES_F32: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0,
];

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qs = &src[2..18];

    for i in 0..16 {
        let lo = (qs[i] & 0x0F) as usize;
        let hi = ((qs[i] >> 4) & 0x0F) as usize;
        dst[i]      = d * KVALUES_IQ4NL[lo] as f32;
        dst[i + 16] = d * KVALUES_IQ4NL[hi] as f32;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}

/// Fused scalar dot product of one IQ4_NL weight row with `x`.
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
    debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

    let mut acc = 0.0f32;
    for (block, xb) in row.chunks_exact(BYTES_PER_BLOCK).zip(x.chunks_exact(BLOCK_SIZE)) {
        let d = read_f16_fast(&block[0..2]);
        let qs = &block[2..18];
        let mut s = 0.0f32;
        for i in 0..16 {
            let b = qs[i];
            s += KVALUES_F32[(b & 0x0F) as usize] * xb[i];
            s += KVALUES_F32[(b >> 4) as usize] * xb[i + 16];
        }
        acc += d * s;
    }
    acc
}

#[derive(Clone, Copy)]
enum SimdLevel {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Portable,
}

#[inline]
fn simd_level() -> SimdLevel {
    static LEVEL: OnceLock<SimdLevel> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                return SimdLevel::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Portable
    })
}

/// Fused SIMD-accelerated dot product of one IQ4_NL weight row with `x`.
///
/// Reassociates floating-point additions in a fixed deterministic SIMD lane order.
#[inline]
pub fn dot_row_fast(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len() % BYTES_PER_BLOCK, 0, "iq4_nl::dot_row_fast: row not block-aligned");
    assert_eq!(
        (row.len() / BYTES_PER_BLOCK).checked_mul(BLOCK_SIZE),
        Some(x.len()),
        "iq4_nl::dot_row_fast: x length mismatch"
    );
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // SAFETY: CPU features avx512f + avx512bw + f16c verified by simd_level(),
            // and row/x lengths asserted above.
            unsafe { avx512::dot_row(row, x) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // SAFETY: CPU features avx2 + fma + f16c verified by simd_level(),
            // and row/x lengths asserted above.
            unsafe { avx2::dot_row(row, x) }
        }
        SimdLevel::Portable => dot_row_portable(row, x),
    }
}

/// Fused SIMD-accelerated matrix-vector product `out[r] = dot(W[r], x)` for `n_rows` IQ4_NL rows.
///
/// Batches multiple output rows per `x` load to maximize FMA throughput and minimize L1 traffic.
pub fn dot_rows_fast(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    assert_eq!(k % BLOCK_SIZE, 0, "iq4_nl::dot_rows_fast: k must be multiple of 32");
    let expected_w = (k / BLOCK_SIZE)
        .checked_mul(BYTES_PER_BLOCK)
        .and_then(|row_bytes| n_rows.checked_mul(row_bytes));
    assert_eq!(expected_w, Some(w.len()), "iq4_nl::dot_rows_fast: w length mismatch");
    assert_eq!(x.len(), k, "iq4_nl::dot_rows_fast: x length mismatch");
    assert_eq!(out.len(), n_rows, "iq4_nl::dot_rows_fast: out length mismatch");

    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // SAFETY: CPU features avx512f + avx512bw + f16c verified by simd_level(),
            // and all slice dimensions asserted above.
            unsafe { avx512::dot_rows(w, x, n_rows, k, out) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // SAFETY: CPU features avx2 + fma + f16c verified by simd_level(),
            // and all slice dimensions asserted above.
            unsafe { avx2::dot_rows(w, x, n_rows, k, out) }
        }
        SimdLevel::Portable => dot_rows_portable(w, x, n_rows, k, out),
    }
}

fn dot_row_portable(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (block, xb) in row.chunks_exact(BYTES_PER_BLOCK).zip(x.chunks_exact(BLOCK_SIZE)) {
        let d = read_f16_fast(&block[0..2]);
        let qs = &block[2..18];
        let mut s = [0.0f32; 8];
        for g in 0..2 {
            let base = g * 8;
            for j in 0..8 {
                let b = qs[base + j];
                s[j] += KVALUES_F32[(b & 0x0F) as usize] * xb[base + j]
                    + KVALUES_F32[(b >> 4) as usize] * xb[base + 16 + j];
            }
        }
        for j in 0..8 {
            acc[j] += d * s[j];
        }
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

fn dot_rows_portable(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
    for r in 0..n_rows {
        out[r] = dot_row_portable(&w[r * row_bytes..(r + 1) * row_bytes], x);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{BLOCK_SIZE, BYTES_PER_BLOCK, KVALUES_IQ4NL};
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn f16_from_ptr(p: *const u8) -> f32 {
        let bits = u16::from_le_bytes([*p, *p.add(1)]);
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    #[inline(always)]
    unsafe fn hsum512(v: __m512) -> f32 {
        let lo = _mm512_castps512_ps256(v);
        let hi = _mm512_extractf32x8_ps(v, 1);
        let sum256 = _mm256_add_ps(lo, hi);
        let lo128 = _mm256_castps256_ps128(sum256);
        let hi128 = _mm256_extractf128_ps(sum256, 1);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    /// # Safety
    /// Caller must guarantee `avx512f`, `avx512bw`, `f16c`, and valid block-aligned lengths.
    #[target_feature(enable = "avx512f,avx512bw,f16c")]
    pub unsafe fn dot_row(row: &[u8], x: &[f32]) -> f32 {
        let n_blocks = row.len() / BYTES_PER_BLOCK;
        let lut = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask0f = _mm_set1_epi8(0x0F);

        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut wp = row.as_ptr();
        let mut xp = x.as_ptr();

        let pairs = n_blocks / 2;
        for _ in 0..pairs {
            // Block 0
            let d0 = _mm512_set1_ps(f16_from_ptr(wp));
            let q0 = _mm_loadu_si128(wp.add(2) as *const __m128i);
            let lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q0, mask0f))));
            let hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q0, 4), mask0f))));
            let x00 = _mm512_loadu_ps(xp);
            let x01 = _mm512_loadu_ps(xp.add(16));
            let p0 = _mm512_fmadd_ps(hi0, x01, _mm512_mul_ps(lo0, x00));
            acc0 = _mm512_fmadd_ps(p0, d0, acc0);

            // Block 1
            let wp1 = wp.add(BYTES_PER_BLOCK);
            let d1 = _mm512_set1_ps(f16_from_ptr(wp1));
            let q1 = _mm_loadu_si128(wp1.add(2) as *const __m128i);
            let lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q1, mask0f))));
            let hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q1, 4), mask0f))));
            let x10 = _mm512_loadu_ps(xp.add(32));
            let x11 = _mm512_loadu_ps(xp.add(48));
            let p1 = _mm512_fmadd_ps(hi1, x11, _mm512_mul_ps(lo1, x10));
            acc1 = _mm512_fmadd_ps(p1, d1, acc1);

            wp = wp.add(2 * BYTES_PER_BLOCK);
            xp = xp.add(2 * BLOCK_SIZE);
        }

        if n_blocks & 1 != 0 {
            let d = _mm512_set1_ps(f16_from_ptr(wp));
            let q = _mm_loadu_si128(wp.add(2) as *const __m128i);
            let lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q, mask0f))));
            let hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), mask0f))));
            let x0 = _mm512_loadu_ps(xp);
            let x1 = _mm512_loadu_ps(xp.add(16));
            let p = _mm512_fmadd_ps(hi, x1, _mm512_mul_ps(lo, x0));
            acc0 = _mm512_fmadd_ps(p, d, acc0);
        }

        hsum512(_mm512_add_ps(acc0, acc1))
    }

    /// # Safety
    /// Caller must guarantee `avx512f`, `avx512bw`, `f16c`, and valid dimensions.
    #[target_feature(enable = "avx512f,avx512bw,f16c")]
    pub unsafe fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
        let n_blocks = k / BLOCK_SIZE;
        let row_bytes = n_blocks * BYTES_PER_BLOCK;
        let lut = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask0f = _mm_set1_epi8(0x0F);

        let mut r = 0usize;
        while r + 4 <= n_rows {
            let mut p0 = w.as_ptr().add(r * row_bytes);
            let mut p1 = p0.add(row_bytes);
            let mut p2 = p1.add(row_bytes);
            let mut p3 = p2.add(row_bytes);
            let mut xp = x.as_ptr();

            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();

            for _ in 0..n_blocks {
                let x0 = _mm512_loadu_ps(xp);
                let x1 = _mm512_loadu_ps(xp.add(16));

                // Row 0
                let d0 = _mm512_set1_ps(f16_from_ptr(p0));
                let q0 = _mm_loadu_si128(p0.add(2) as *const __m128i);
                let lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q0, mask0f))));
                let hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q0, 4), mask0f))));
                a0 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi0, x1, _mm512_mul_ps(lo0, x0)), d0, a0);

                // Row 1
                let d1 = _mm512_set1_ps(f16_from_ptr(p1));
                let q1 = _mm_loadu_si128(p1.add(2) as *const __m128i);
                let lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q1, mask0f))));
                let hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q1, 4), mask0f))));
                a1 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi1, x1, _mm512_mul_ps(lo1, x0)), d1, a1);

                // Row 2
                let d2 = _mm512_set1_ps(f16_from_ptr(p2));
                let q2 = _mm_loadu_si128(p2.add(2) as *const __m128i);
                let lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q2, mask0f))));
                let hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q2, 4), mask0f))));
                a2 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi2, x1, _mm512_mul_ps(lo2, x0)), d2, a2);

                // Row 3
                let d3 = _mm512_set1_ps(f16_from_ptr(p3));
                let q3 = _mm_loadu_si128(p3.add(2) as *const __m128i);
                let lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(q3, mask0f))));
                let hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q3, 4), mask0f))));
                a3 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi3, x1, _mm512_mul_ps(lo3, x0)), d3, a3);

                p0 = p0.add(BYTES_PER_BLOCK);
                p1 = p1.add(BYTES_PER_BLOCK);
                p2 = p2.add(BYTES_PER_BLOCK);
                p3 = p3.add(BYTES_PER_BLOCK);
                xp = xp.add(BLOCK_SIZE);
            }

            out[r] = hsum512(a0);
            out[r + 1] = hsum512(a1);
            out[r + 2] = hsum512(a2);
            out[r + 3] = hsum512(a3);
            r += 4;
        }

        while r < n_rows {
            out[r] = dot_row(&w[r * row_bytes..(r + 1) * row_bytes], x);
            r += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{BLOCK_SIZE, BYTES_PER_BLOCK, KVALUES_IQ4NL};
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn f16_from_ptr(p: *const u8) -> f32 {
        let bits = u16::from_le_bytes([*p, *p.add(1)]);
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32)))
    }

    #[inline(always)]
    unsafe fn hsum256(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let sums = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
    }

    /// # Safety
    /// Caller must guarantee `avx2`, `fma`, `f16c`, and valid block-aligned lengths.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_row(row: &[u8], x: &[f32]) -> f32 {
        let n_blocks = row.len() / BYTES_PER_BLOCK;
        let lut = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask0f = _mm_set1_epi8(0x0F);

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut wp = row.as_ptr();
        let mut xp = x.as_ptr();

        for _ in 0..n_blocks {
            let d = _mm256_set1_ps(f16_from_ptr(wp));
            let q = _mm_loadu_si128(wp.add(2) as *const __m128i);
            let lo_i8 = _mm_shuffle_epi8(lut, _mm_and_si128(q, mask0f));
            let hi_i8 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), mask0f));

            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo_i8));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo_i8)));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_i8));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi_i8)));

            let p0 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xp.add(16)), _mm256_mul_ps(f0, _mm256_loadu_ps(xp)));
            let p1 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp.add(24)), _mm256_mul_ps(f1, _mm256_loadu_ps(xp.add(8))));
            acc0 = _mm256_fmadd_ps(p0, d, acc0);
            acc1 = _mm256_fmadd_ps(p1, d, acc1);

            wp = wp.add(BYTES_PER_BLOCK);
            xp = xp.add(BLOCK_SIZE);
        }

        hsum256(_mm256_add_ps(acc0, acc1))
    }

    /// # Safety
    /// Caller must guarantee `avx2`, `fma`, `f16c`, and valid dimensions.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
        let n_blocks = k / BLOCK_SIZE;
        let row_bytes = n_blocks * BYTES_PER_BLOCK;
        let lut = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
        let mask0f = _mm_set1_epi8(0x0F);

        let mut r = 0usize;
        while r + 2 <= n_rows {
            let mut p0 = w.as_ptr().add(r * row_bytes);
            let mut p1 = p0.add(row_bytes);
            let mut xp = x.as_ptr();
            let mut a00 = _mm256_setzero_ps();
            let mut a01 = _mm256_setzero_ps();
            let mut a10 = _mm256_setzero_ps();
            let mut a11 = _mm256_setzero_ps();

            for _ in 0..n_blocks {
                let x0 = _mm256_loadu_ps(xp);
                let x1 = _mm256_loadu_ps(xp.add(8));
                let x2 = _mm256_loadu_ps(xp.add(16));
                let x3 = _mm256_loadu_ps(xp.add(24));

                let d0 = _mm256_set1_ps(f16_from_ptr(p0));
                let q0 = _mm_loadu_si128(p0.add(2) as *const __m128i);
                let lo0 = _mm_shuffle_epi8(lut, _mm_and_si128(q0, mask0f));
                let hi0 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q0, 4), mask0f));
                let s00 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi0)),
                    x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo0)), x0),
                );
                let s01 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi0))),
                    x3,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo0))), x1),
                );
                a00 = _mm256_fmadd_ps(s00, d0, a00);
                a01 = _mm256_fmadd_ps(s01, d0, a01);

                let d1 = _mm256_set1_ps(f16_from_ptr(p1));
                let q1 = _mm_loadu_si128(p1.add(2) as *const __m128i);
                let lo1 = _mm_shuffle_epi8(lut, _mm_and_si128(q1, mask0f));
                let hi1 = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q1, 4), mask0f));
                let s10 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi1)),
                    x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo1)), x0),
                );
                let s11 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(hi1))),
                    x3,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(lo1))), x1),
                );
                a10 = _mm256_fmadd_ps(s10, d1, a10);
                a11 = _mm256_fmadd_ps(s11, d1, a11);

                p0 = p0.add(BYTES_PER_BLOCK);
                p1 = p1.add(BYTES_PER_BLOCK);
                xp = xp.add(BLOCK_SIZE);
            }

            out[r] = hsum256(_mm256_add_ps(a00, a01));
            out[r + 1] = hsum256(_mm256_add_ps(a10, a11));
            r += 2;
        }

        if r < n_rows {
            out[r] = dot_row(&w[r * row_bytes..(r + 1) * row_bytes], x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blocks(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (z >> 33) as u8
        };
        let mut v = Vec::with_capacity(n_blocks * BYTES_PER_BLOCK);
        for _ in 0..n_blocks {
            let d = (next() as f32 / 255.0 - 0.5) * 0.08;
            v.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for _ in 0..16 {
                v.push(next());
            }
        }
        v
    }

    fn make_x(k: usize, seed: u64) -> Vec<f32> {
        let mut z = seed;
        (0..k)
            .map(|_| {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((z >> 40) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn fast_dot_matches_dequantize_reference() {
        for &(n_rows, k) in &[(1, 32), (3, 96), (4, 128), (7, 256), (17, 1152)] {
            let n_blocks = k / BLOCK_SIZE;
            let row_bytes = n_blocks * BYTES_PER_BLOCK;
            let w = make_blocks(n_rows * n_blocks, 0x1234_5678 ^ (n_rows as u64) ^ ((k as u64) << 16));
            let x = make_x(k, 0x9E37_79B9 ^ (k as u64));

            let mut deq = vec![0.0f32; n_rows * k];
            dequantize(&w, &mut deq);
            let mut expected = vec![0.0f32; n_rows];
            for r in 0..n_rows {
                expected[r] = deq[r * k..(r + 1) * k]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>() as f32;
            }

            let mut got = vec![0.0f32; n_rows];
            dot_rows_fast(&w, &x, n_rows, k, &mut got);

            for r in 0..n_rows {
                let row = &w[r * row_bytes..(r + 1) * row_bytes];
                let scalar = dot_row(row, &x);
                let single = dot_row_fast(row, &x);
                let port = dot_row_portable(row, &x);
                let tol = expected[r].abs() * 1e-4 + 1e-4;
                assert!((scalar - expected[r]).abs() <= tol, "scalar r={r}: {scalar} vs {}", expected[r]);
                assert!((single - expected[r]).abs() <= tol, "single r={r}: {single} vs {}", expected[r]);
                assert!((got[r] - expected[r]).abs() <= tol, "rows r={r}: {} vs {}", got[r], expected[r]);
                assert!((port - expected[r]).abs() <= tol, "portable r={r}: {port} vs {}", expected[r]);
            }
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                let mut got_avx2 = vec![0.0f32; n_rows];
                // SAFETY: features verified above.
                unsafe { avx2::dot_rows(&w, &x, n_rows, k, &mut got_avx2) };
                for r in 0..n_rows {
                    let tol = expected[r].abs() * 1e-4 + 1e-4;
                    assert!((got_avx2[r] - expected[r]).abs() <= tol, "avx2 r={r}: {} vs {}", got_avx2[r], expected[r]);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "w length mismatch")]
    fn dot_rows_fast_rejects_overflowing_dims() {
        let x = [0.0f32; BLOCK_SIZE];
        dot_rows_fast(&[], &x, 1usize << (usize::BITS - 1), BLOCK_SIZE, &mut []);
    }

    #[test]
    #[ignore = "microbenchmark"]
    fn measure_iq4_nl_bandwidth() {
        let k = 2304usize;
        let n_rows = 2304usize;
        let n_blocks = k / BLOCK_SIZE;
        let row_bytes = n_blocks * BYTES_PER_BLOCK;
        // 64 matrices (~191 MB total) so we exceed the 128 MB L3 cache and measure true DRAM bandwidth.
        let n_mats = 64usize;
        let w = make_blocks(n_mats * n_rows * n_blocks, 0xCAFE);
        let x = make_x(k, 0xBEEF);
        let mut out = vec![0.0f32; n_rows];

        // Warmup + L2-hot single-matrix 1-thread throughput (2.98 MB fits in L3/L2)
        dot_rows_fast(&w[..n_rows * row_bytes], &x, n_rows, k, &mut out);
        let t_hot = std::time::Instant::now();
        for _ in 0..64 {
            dot_rows_fast(&w[..n_rows * row_bytes], &x, n_rows, k, &mut out);
            std::hint::black_box(&out);
        }
        let dt_hot = t_hot.elapsed().as_secs_f64();
        let bytes = (n_mats * n_rows * row_bytes) as f64;
        eprintln!("iq4_nl 1-thread L3-hot: {:.2} GB/s", bytes / dt_hot / 1e9);

        // Raw 16-thread DRAM read bandwidth ceiling (u64 sum)
        let t_raw = std::time::Instant::now();
        std::thread::scope(|s| {
            let chunk = w.len() / 16;
            for t in 0..16 {
                let sl = &w[t * chunk..(t + 1) * chunk];
                s.spawn(move || {
                    let mut sum = 0u64;
                    for c in sl.chunks_exact(64) {
                        sum = sum.wrapping_add(u64::from_le_bytes(c[0..8].try_into().unwrap()));
                    }
                    std::hint::black_box(sum);
                });
            }
        });
        let dt_raw = t_raw.elapsed().as_secs_f64();
        eprintln!("raw DRAM 16-thread: {:.2} GB/s", (w.len() as f64) / dt_raw / 1e9);

        // 1-thread DRAM
        let t0 = std::time::Instant::now();
        for m in 0..n_mats {
            dot_rows_fast(&w[m * n_rows * row_bytes..(m + 1) * n_rows * row_bytes], &x, n_rows, k, &mut out);
        }
        let dt1 = t0.elapsed().as_secs_f64();
        eprintln!("iq4_nl 1-thread DRAM: {:.2} GB/s ({:.2} ms for {:.1} MB)", bytes / dt1 / 1e9, dt1 * 1e3, bytes / 1e6);

        // 8-thread and 16-thread DRAM
        for n_thr in [8usize, 16usize] {
            let per = n_mats / n_thr;
            let t1 = std::time::Instant::now();
            std::thread::scope(|s| {
                for t in 0..n_thr {
                    let w_ref = &w;
                    let x_ref = &x;
                    s.spawn(move || {
                        let mut local_out = vec![0.0f32; n_rows];
                        for m in t * per..(t + 1) * per {
                            dot_rows_fast(
                                &w_ref[m * n_rows * row_bytes..(m + 1) * n_rows * row_bytes],
                                x_ref,
                                n_rows,
                                k,
                                &mut local_out,
                            );
                        }
                        std::hint::black_box(&local_out);
                    });
                }
            });
            let dtn = t1.elapsed().as_secs_f64();
            eprintln!("iq4_nl {n_thr}-thread DRAM: {:.2} GB/s ({:.2} ms for {:.1} MB)", bytes / dtn / 1e9, dtn * 1e3, bytes / 1e6);
        }
    }
}
