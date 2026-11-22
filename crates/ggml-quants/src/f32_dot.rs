//! SIMD-accelerated F32 dot product and 4-row batched matrix-vector product.
//!
//! Uses runtime CPU feature detection (`AVX-512F + FMA` -> `AVX2 + FMA` -> portable 8-lane)
//! so binaries built without `-C target-cpu=native` still get full 16-wide / 8-wide FMA throughput
//! on dense F32 layers (routers, controllers, relative bias tables, attention Q·K dots).

use std::sync::OnceLock;

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
            if std::arch::is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return SimdLevel::Avx2;
            }
        }
        SimdLevel::Portable
    })
}

/// Fused SIMD dot product of two equal-length `f32` slices.
#[inline]
pub fn dot_fast(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "f32_dot::dot_fast: length mismatch");
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // SAFETY: avx512f verified by simd_level(), lengths asserted equal above.
            unsafe { avx512::dot(a, b) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // SAFETY: avx2 + fma verified by simd_level(), lengths asserted equal above.
            unsafe { avx2::dot(a, b) }
        }
        SimdLevel::Portable => dot_portable(a, b),
    }
}

/// Fused SIMD matrix-vector product `out[r] = dot(w[r], x)` for `[n_rows, k]` row-major `w`.
///
/// Batches 4 output rows per `x` load on AVX-512 (2 rows on AVX2) to maximize FMA utilization.
#[inline]
pub fn dot_rows_fast(w: &[f32], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    assert_eq!(w.len(), n_rows * k, "f32_dot::dot_rows_fast: w length mismatch");
    assert_eq!(x.len(), k, "f32_dot::dot_rows_fast: x length mismatch");
    assert_eq!(out.len(), n_rows, "f32_dot::dot_rows_fast: out length mismatch");
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // SAFETY: avx512f verified by simd_level(), all dimensions asserted above.
            unsafe { avx512::dot_rows(w, x, n_rows, k, out) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // SAFETY: avx2 + fma verified by simd_level(), all dimensions asserted above.
            unsafe { avx2::dot_rows(w, x, n_rows, k, out) }
        }
        SimdLevel::Portable => dot_rows_portable(w, x, n_rows, k, out),
    }
}

/// Weighted accumulate `dst[i] += weight * src[i]` using SIMD.
#[inline]
pub fn axpy_fast(dst: &mut [f32], src: &[f32], weight: f32) {
    assert_eq!(dst.len(), src.len(), "f32_dot::axpy_fast: length mismatch");
    match simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => {
            // SAFETY: avx512f verified by simd_level(), lengths asserted equal above.
            unsafe { avx512::axpy(dst, src, weight) }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => {
            // SAFETY: avx2 + fma verified by simd_level(), lengths asserted equal above.
            unsafe { avx2::axpy(dst, src, weight) }
        }
        SimdLevel::Portable => {
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d += weight * s;
            }
        }
    }
}

fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let chunks = n / 8;
    let mut acc = [0.0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        let aa = &a[base..base + 8];
        let bb = &b[base..base + 8];
        for j in 0..8 {
            acc[j] += aa[j] * bb[j];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for i in chunks * 8..n {
        sum += a[i] * b[i];
    }
    sum
}

fn dot_rows_portable(w: &[f32], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    for r in 0..n_rows {
        out[r] = dot_portable(&w[r * k..(r + 1) * k], x);
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use std::arch::x86_64::*;

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
    /// Caller must guarantee `avx512f` and `a.len() == b.len()`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        let mut i = 0usize;
        while i + 64 <= n {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i)), a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(i + 16)), _mm512_loadu_ps(bp.add(i + 16)), a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(i + 32)), _mm512_loadu_ps(bp.add(i + 32)), a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(i + 48)), _mm512_loadu_ps(bp.add(i + 48)), a3);
            i += 64;
        }
        while i + 16 <= n {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(ap.add(i)), _mm512_loadu_ps(bp.add(i)), a0);
            i += 16;
        }
        let mut sum = hsum512(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        sum
    }

    /// # Safety
    /// Caller must guarantee `avx512f` and valid slice dimensions.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_rows(w: &[f32], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let mut r = 0usize;
        while r + 4 <= n_rows {
            let p0 = wp.add(r * k);
            let p1 = p0.add(k);
            let p2 = p1.add(k);
            let p3 = p2.add(k);
            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();
            let mut i = 0usize;
            while i + 16 <= k {
                let xv = _mm512_loadu_ps(xp.add(i));
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(i)), xv, a0);
                a1 = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(i)), xv, a1);
                a2 = _mm512_fmadd_ps(_mm512_loadu_ps(p2.add(i)), xv, a2);
                a3 = _mm512_fmadd_ps(_mm512_loadu_ps(p3.add(i)), xv, a3);
                i += 16;
            }
            let mut s0 = hsum512(a0);
            let mut s1 = hsum512(a1);
            let mut s2 = hsum512(a2);
            let mut s3 = hsum512(a3);
            while i < k {
                let xv = *xp.add(i);
                s0 += *p0.add(i) * xv;
                s1 += *p1.add(i) * xv;
                s2 += *p2.add(i) * xv;
                s3 += *p3.add(i) * xv;
                i += 1;
            }
            out[r] = s0;
            out[r + 1] = s1;
            out[r + 2] = s2;
            out[r + 3] = s3;
            r += 4;
        }
        while r < n_rows {
            out[r] = dot(&w[r * k..(r + 1) * k], x);
            r += 1;
        }
    }

    /// # Safety
    /// Caller must guarantee `avx512f` and `dst.len() == src.len()`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn axpy(dst: &mut [f32], src: &[f32], weight: f32) {
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let vw = _mm512_set1_ps(weight);
        let mut i = 0usize;
        while i + 16 <= n {
            let d = _mm512_loadu_ps(dp.add(i));
            let s = _mm512_loadu_ps(sp.add(i));
            _mm512_storeu_ps(dp.add(i), _mm512_fmadd_ps(s, vw, d));
            i += 16;
        }
        while i < n {
            *dp.add(i) += *sp.add(i) * weight;
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

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
    /// Caller must guarantee `avx2`, `fma`, and `a.len() == b.len()`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 8)), _mm256_loadu_ps(bp.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 16)), _mm256_loadu_ps(bp.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i + 24)), _mm256_loadu_ps(bp.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), a0);
            i += 8;
        }
        let mut sum = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        while i < n {
            sum += *ap.add(i) * *bp.add(i);
            i += 1;
        }
        sum
    }

    /// # Safety
    /// Caller must guarantee `avx2`, `fma`, and valid slice dimensions.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_rows(w: &[f32], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
        let wp = w.as_ptr();
        let xp = x.as_ptr();
        let mut r = 0usize;
        while r + 2 <= n_rows {
            let p0 = wp.add(r * k);
            let p1 = p0.add(k);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 8 <= k {
                let xv = _mm256_loadu_ps(xp.add(i));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(i)), xv, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(i)), xv, a1);
                i += 8;
            }
            let mut s0 = hsum256(a0);
            let mut s1 = hsum256(a1);
            while i < k {
                let xv = *xp.add(i);
                s0 += *p0.add(i) * xv;
                s1 += *p1.add(i) * xv;
                i += 1;
            }
            out[r] = s0;
            out[r + 1] = s1;
            r += 2;
        }
        if r < n_rows {
            out[r] = dot(&w[r * k..(r + 1) * k], x);
        }
    }

    /// # Safety
    /// Caller must guarantee `avx2`, `fma`, and `dst.len() == src.len()`.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(dst: &mut [f32], src: &[f32], weight: f32) {
        let n = dst.len();
        let dp = dst.as_mut_ptr();
        let sp = src.as_ptr();
        let vw = _mm256_set1_ps(weight);
        let mut i = 0usize;
        while i + 8 <= n {
            let d = _mm256_loadu_ps(dp.add(i));
            let s = _mm256_loadu_ps(sp.add(i));
            _mm256_storeu_ps(dp.add(i), _mm256_fmadd_ps(s, vw, d));
            i += 8;
        }
        while i < n {
            *dp.add(i) += *sp.add(i) * weight;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_dot_and_axpy_match_reference() {
        for &(n_rows, k) in &[(1, 16), (3, 19), (5, 64), (7, 131), (17, 2304)] {
            let w: Vec<f32> = (0..n_rows * k)
                .map(|i: usize| ((i.wrapping_mul(37) % 101) as f32 - 50.0) * 0.03)
                .collect();
            let x: Vec<f32> = (0..k)
                .map(|i: usize| ((i.wrapping_mul(53) % 97) as f32 - 48.0) * 0.02)
                .collect();
            let mut got = vec![0.0f32; n_rows];
            dot_rows_fast(&w, &x, n_rows, k, &mut got);
            for r in 0..n_rows {
                let want = w[r * k..(r + 1) * k]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum::<f64>() as f32;
                let single = dot_fast(&w[r * k..(r + 1) * k], &x);
                let tol = want.abs() * 1e-4 + 1e-4;
                assert!((got[r] - want).abs() <= tol, "rows r={r}: {} vs {want}", got[r]);
                assert!((single - want).abs() <= tol, "single r={r}: {single} vs {want}");
            }
            let mut dst = x.clone();
            axpy_fast(&mut dst, &w[..k], 0.75);
            for i in 0..k {
                let want = x[i] + 0.75 * w[i];
                assert!((dst[i] - want).abs() < 1e-5);
            }
        }
    }
}
