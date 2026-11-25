//! Q8_0: 32 elements per block, 34 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..34]  i8  qs[32]
//!
//! Dequant: y = qs * d.

use std::sync::OnceLock;

use crate::{read_f16, read_f16_fast};

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 34;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qs = &src[2..34];

    for i in 0..32 {
        let q = qs[i] as i8;
        dst[i] = q as f32 * d;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}

/// Fused scalar dot product of one Q8_0 weight row with `x`.
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
    debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

    let mut acc = 0.0f32;
    for (block, xb) in row.chunks_exact(BYTES_PER_BLOCK).zip(x.chunks_exact(BLOCK_SIZE)) {
        let d = read_f16_fast(&block[0..2]);
        let qs = &block[2..34];
        let mut s = 0.0f32;
        for i in 0..32 {
            s += (qs[i] as i8) as f32 * xb[i];
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

/// Fused SIMD-accelerated dot product of one Q8_0 weight row with `x`.
#[inline]
pub fn dot_row_fast(row: &[u8], x: &[f32]) -> f32 {
    assert_eq!(row.len() % BYTES_PER_BLOCK, 0, "q8_0::dot_row_fast: row not block-aligned");
    assert_eq!(
        (row.len() / BYTES_PER_BLOCK).checked_mul(BLOCK_SIZE),
        Some(x.len()),
        "q8_0::dot_row_fast: x length mismatch"
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

/// Fused SIMD-accelerated matrix-vector product `out[r] = dot(W[r], x)` for `n_rows` Q8_0 rows.
pub fn dot_rows_fast(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    assert_eq!(k % BLOCK_SIZE, 0, "q8_0::dot_rows_fast: k must be multiple of 32");
    let expected_w = (k / BLOCK_SIZE)
        .checked_mul(BYTES_PER_BLOCK)
        .and_then(|row_bytes| n_rows.checked_mul(row_bytes));
    assert_eq!(expected_w, Some(w.len()), "q8_0::dot_rows_fast: w length mismatch");
    assert_eq!(x.len(), k, "q8_0::dot_rows_fast: x length mismatch");
    assert_eq!(out.len(), n_rows, "q8_0::dot_rows_fast: out length mismatch");

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
        let qs = &block[2..34];
        let mut s = [0.0f32; 8];
        for g in 0..4 {
            let base = g * 8;
            for j in 0..8 {
                s[j] += (qs[base + j] as i8) as f32 * xb[base + j];
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
    use super::{BLOCK_SIZE, BYTES_PER_BLOCK};
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
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut wp = row.as_ptr();
        let mut xp = x.as_ptr();

        let pairs = n_blocks / 2;
        for _ in 0..pairs {
            let d0 = _mm512_set1_ps(f16_from_ptr(wp));
            let f00 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(2) as *const __m128i)));
            let f01 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(18) as *const __m128i)));
            let p0 = _mm512_fmadd_ps(f01, _mm512_loadu_ps(xp.add(16)), _mm512_mul_ps(f00, _mm512_loadu_ps(xp)));
            acc0 = _mm512_fmadd_ps(p0, d0, acc0);

            let wp1 = wp.add(BYTES_PER_BLOCK);
            let d1 = _mm512_set1_ps(f16_from_ptr(wp1));
            let f10 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp1.add(2) as *const __m128i)));
            let f11 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp1.add(18) as *const __m128i)));
            let p1 = _mm512_fmadd_ps(f11, _mm512_loadu_ps(xp.add(48)), _mm512_mul_ps(f10, _mm512_loadu_ps(xp.add(32))));
            acc1 = _mm512_fmadd_ps(p1, d1, acc1);

            wp = wp.add(2 * BYTES_PER_BLOCK);
            xp = xp.add(2 * BLOCK_SIZE);
        }

        if n_blocks & 1 != 0 {
            let d = _mm512_set1_ps(f16_from_ptr(wp));
            let f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(2) as *const __m128i)));
            let f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp.add(18) as *const __m128i)));
            let p = _mm512_fmadd_ps(f1, _mm512_loadu_ps(xp.add(16)), _mm512_mul_ps(f0, _mm512_loadu_ps(xp)));
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

                let d0 = _mm512_set1_ps(f16_from_ptr(p0));
                let lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p0.add(2) as *const __m128i)));
                let hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p0.add(18) as *const __m128i)));
                a0 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi0, x1, _mm512_mul_ps(lo0, x0)), d0, a0);

                let d1 = _mm512_set1_ps(f16_from_ptr(p1));
                let lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p1.add(2) as *const __m128i)));
                let hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p1.add(18) as *const __m128i)));
                a1 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi1, x1, _mm512_mul_ps(lo1, x0)), d1, a1);

                let d2 = _mm512_set1_ps(f16_from_ptr(p2));
                let lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p2.add(2) as *const __m128i)));
                let hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p2.add(18) as *const __m128i)));
                a2 = _mm512_fmadd_ps(_mm512_fmadd_ps(hi2, x1, _mm512_mul_ps(lo2, x0)), d2, a2);

                let d3 = _mm512_set1_ps(f16_from_ptr(p3));
                let lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p3.add(2) as *const __m128i)));
                let hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(p3.add(18) as *const __m128i)));
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
    use super::{BLOCK_SIZE, BYTES_PER_BLOCK};
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
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut wp = row.as_ptr();
        let mut xp = x.as_ptr();

        for _ in 0..n_blocks {
            let d = _mm256_set1_ps(f16_from_ptr(wp));
            let q0 = _mm_loadu_si128(wp.add(2) as *const __m128i);
            let q1 = _mm_loadu_si128(wp.add(18) as *const __m128i);

            let f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
            let f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q0)));
            let f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
            let f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q1)));

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
                let q0a = _mm_loadu_si128(p0.add(2) as *const __m128i);
                let q0b = _mm_loadu_si128(p0.add(18) as *const __m128i);
                let s00 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0b)),
                    x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0a)), x0),
                );
                let s01 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q0b))),
                    x3,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q0a))), x1),
                );
                a00 = _mm256_fmadd_ps(s00, d0, a00);
                a01 = _mm256_fmadd_ps(s01, d0, a01);

                let d1 = _mm256_set1_ps(f16_from_ptr(p1));
                let q1a = _mm_loadu_si128(p1.add(2) as *const __m128i);
                let q1b = _mm_loadu_si128(p1.add(18) as *const __m128i);
                let s10 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1b)),
                    x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1a)), x0),
                );
                let s11 = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q1b))),
                    x3,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(q1a))), x1),
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

/// `src` (a multiple of 32 values) into Q8_0 blocks in `dst` as ggml's reference quantizer makes them: each block's
/// scale its largest magnitude over 127 (as f16), its values rounded to the nearest step.
pub fn quantize(src: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(src.len() % BLOCK_SIZE, 0);
    debug_assert_eq!(dst.len(), src.len() / BLOCK_SIZE * BYTES_PER_BLOCK);
    for (x, out) in src.chunks_exact(BLOCK_SIZE).zip(dst.chunks_exact_mut(BYTES_PER_BLOCK)) {
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for (q, v) in out[2..].iter_mut().zip(x) {
            *q = (v * id).round().clamp(-127.0, 127.0) as i8 as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block comes back within half a step of each value (its step its largest magnitude over 127), with the scale's
    /// f16 rounding times the quant on top (a subnormal scale's too), and zeros as zeros.
    #[test]
    fn quantized_blocks_come_back_within_half_a_step() {
        let src: Vec<f32> = (0..96).map(|i| if i < 32 { ((i * 37 % 61) as f32 - 30.0) * 0.173 } else if i < 64 { 0.0 } else { (i as f32 * 0.7).sin() * 1e-3 }).collect();
        let mut bytes = vec![0u8; 3 * BYTES_PER_BLOCK];
        quantize(&src, &mut bytes);
        let mut back = vec![0f32; 96];
        dequantize(&bytes, &mut back);
        for (block, (x, y)) in src.chunks(32).zip(back.chunks(32)).enumerate() {
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = amax / 127.0;
            let stored = crate::read_f16(&bytes[block * BYTES_PER_BLOCK..block * BYTES_PER_BLOCK + 2]);
            let bound = 0.5 * stored + 127.0 * (stored - step).abs() + 1e-12;
            for (a, b) in x.iter().zip(y) {
                assert!((a - b).abs() <= bound, "block {block}: {a} came back {b} (step {step}, stored {stored})");
            }
        }
        assert!(back[32..64].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn fast_dot_matches_dequantize_reference() {
        for &(n_rows, k) in &[(1, 32), (3, 96), (4, 128), (7, 256), (17, 1152)] {
            let n_blocks = k / BLOCK_SIZE;
            let row_bytes = n_blocks * BYTES_PER_BLOCK;
            let mut z = 0x517C_C1B7u64 ^ (n_rows as u64) ^ ((k as u64) << 16);
            let mut next = || {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (z >> 33) as u8
            };
            let mut w = Vec::with_capacity(n_rows * row_bytes);
            for _ in 0..n_rows * n_blocks {
                let d = (next() as f32 / 255.0 - 0.5) * 0.08;
                w.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
                for _ in 0..32 {
                    w.push(next());
                }
            }
            let x: Vec<f32> = (0..k)
                .map(|_| {
                    z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((z >> 40) as f32 / (1u32 << 23) as f32) - 1.0
                })
                .collect();

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
}
