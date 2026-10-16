//! Scalar codecs for DeepSeek-V4.1's number formats, and the activation
//! quantization the reference applies before every fp8/fp4 GEMM.
//!
//! - bf16 / f16: storage for norms, router, embeddings; bf16 is also the
//!   activation dtype, so outputs are rounded through it at the same points
//!   the reference rounds (`to_bf16` below).
//! - fp8 e4m3fn: dense trunk weights (with one e8m0 scale per 32x32 tile)
//!   and every quantized activation.
//! - fp4 e2m1: routed experts, two per byte, low nibble = even element, one
//!   e8m0 scale per 32 along K — the OCP MXFP4 element format.
//! - e8m0: a bare exponent, value 2^(bits-127).
//!
//! Everything here is exact where the reference is exact: products of an
//! fp4 value, an fp8 value and power-of-two scales need at most 6 significant
//! bits, so they are exact in f32 and only the accumulation order can differ
//! from the reference's tensor-core sums.

/// bf16 bits -> f32 (exact).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 -> bf16 bits, round to nearest even (NaN stays NaN) — torch's `.to(bfloat16)`.
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// Round an f32 through bf16 and back — the reference's `.to(bfloat16)` on an activation.
#[inline]
pub fn to_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// IEEE half bits -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: renormalize
            let shift = m.leading_zeros() - 21; // bring the top set bit to position 10
            sign | ((113 - shift) << 23) | (((m << shift) & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// e8m0 scale byte -> f32 power of two (255 is NaN).
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 0xff {
        f32::NAN
    } else if b == 0 {
        f32::from_bits(0x0040_0000) // 2^-127, subnormal in f32
    } else {
        f32::from_bits((b as u32) << 23)
    }
}

/// fp8 e4m3fn byte -> f32 (exact; 0x7f / 0xff are NaN, no infinities).
#[inline]
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    FP8_E4M3_TABLE[b as usize]
}

const fn fp8_e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xf) as i32;
    let mant = (b & 0x7) as u32;
    if exp == 0xf && mant == 0x7 {
        return f32::NAN;
    }
    let v = if exp == 0 {
        // subnormal: mant/8 * 2^-6
        mant as f32 / 8.0 / 64.0
    } else {
        let m = 1.0 + mant as f32 / 8.0;
        let mut scale = 1.0f32;
        let mut e = exp - 7;
        while e > 0 {
            scale *= 2.0;
            e -= 1;
        }
        while e < 0 {
            scale /= 2.0;
            e += 1;
        }
        m * scale
    };
    sign * v
}

static FP8_E4M3_TABLE: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = fp8_e4m3_decode(i as u8);
        i += 1;
    }
    t
};

/// f32 -> fp8 e4m3fn, round to nearest even. The caller clamps to +-448 first
/// (as the reference kernel does), so saturation never happens here.
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7f;
    }
    // Largest code whose value is <= a, then round between it and the next.
    // Positive codes 0x00..=0x7e are monotonic in value.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    if a >= FP8_E4M3_TABLE[0x7e] {
        return sign | 0x7e;
    }
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if FP8_E4M3_TABLE[mid as usize] <= a {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let below = FP8_E4M3_TABLE[lo as usize];
    if below == a {
        return sign | lo;
    }
    let above = FP8_E4M3_TABLE[lo as usize + 1];
    let (d_lo, d_hi) = (a - below, above - a);
    let code = if d_lo < d_hi || (d_lo == d_hi && lo % 2 == 0) { lo } else { lo + 1 };
    sign | code
}

/// e2m1 nibble -> value (bit 3 is the sign); scale separately.
pub const FP4_VALUES: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2^ceil(log2(x)) for positive finite x, exactly as the reference kernel's
/// `fast_round_scale`: read the exponent, add one unless the mantissa is zero.
#[inline]
pub fn pow2_ceil(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    let e = exp - 127 + i32::from(man != 0);
    f32::from_bits(((e + 127) as u32) << 23)
}

pub const FP8_MAX: f32 = 448.0;

/// `fp8_e4m3_to_f32(f32_to_fp8_e4m3(v))` for `|v| <= 448`, by arithmetic:
/// round to nearest (ties to even) on the e4m3 grid, whose step is 2^-9
/// below 2^-6 (subnormals) and 2^(e-3) in the binade [2^e, 2^(e+1)). The
/// table search of [`f32_to_fp8_e4m3`] costs a few dozen ns a value, which
/// is most of a one-token CPU expert once its matmuls are vectorized.
#[inline]
pub fn round_e4m3(v: f32) -> f32 {
    let a = v.abs();
    if a.is_nan() {
        return f32::NAN;
    }
    if a >= FP8_MAX {
        return FP8_MAX.copysign(v);
    }
    let q = if a < 0.015625 {
        (a * 512.0).round_ties_even() * (1.0 / 512.0)
    } else {
        // a is a normal f32 here, so its exponent field is its binade
        let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
        let step = f32::from_bits(((e - 3 + 127) as u32) << 23);
        (a / step).round_ties_even() * step
    };
    q.copysign(v)
}

/// The reference `act_quant(x, 32, "ue8m0")` followed by dequantization: per
/// block of 32, scale = pow2_ceil(amax / 448) (amax floored at 1e-4), values
/// clamped to +-448 and rounded to fp8 e4m3. Returns what the GEMM actually
/// multiplies: q * scale, exact in f32.
pub fn fake_quant_fp8(x: &[f32], block: usize) -> Vec<f32> {
    debug_assert_eq!(x.len() % block, 0);
    let inv = 1.0f32 / FP8_MAX;
    let mut out = Vec::with_capacity(x.len());
    for chunk in x.chunks_exact(block) {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_ceil(amax * inv);
        for &v in chunk {
            out.push(round_e4m3((v / s).clamp(-FP8_MAX, FP8_MAX)) * s);
        }
    }
    out
}

/// Reference `act_quant(x, 32, "ue8m0", inplace=True)`: the fp8 round trip of
/// [`fake_quant_fp8`] written back into `x` (values are bf16-exact already;
/// the final bf16 round mirrors the reference's copy into a bf16 tensor).
pub fn fake_quant_fp8_inplace(x: &mut [f32]) {
    for chunk in x.chunks_mut(32) {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-4);
        let s = pow2_ceil(amax * (1.0 / FP8_MAX));
        for v in chunk.iter_mut() {
            *v = to_bf16(round_e4m3((*v / s).clamp(-FP8_MAX, FP8_MAX)) * s);
        }
    }
}

/// Scale format of an fp4 round trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp4Scale {
    /// Power-of-two scale (indexer): pow2_ceil(amax / 6), amax >= 6 * 2^-126.
    E8M0,
    /// fp8 e4m3 scale (compressed KV): e4m3(amax / 6), amax >= 6 * 2^-9.
    E4M3,
}

/// Reference `fp4_act_quant(x, block, inplace=True)`: e2m1 round trip per
/// `block` values, rounded to bf16.
pub fn fake_quant_fp4_inplace(x: &mut [f32], block: usize, kind: Fp4Scale) {
    for chunk in x.chunks_mut(block) {
        let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s = match kind {
            Fp4Scale::E8M0 => pow2_ceil(amax.max(6.0 * 2f32.powi(-126)) * (1.0 / 6.0)),
            Fp4Scale::E4M3 => fp8_e4m3_to_f32(f32_to_fp8_e4m3(amax.max(6.0 * 2f32.powi(-9)) / 6.0)),
        };
        for v in chunk.iter_mut() {
            *v = to_bf16(round_fp4((*v / s).clamp(-6.0, 6.0)) * s);
        }
    }
}

/// Nearest e2m1 value, ties to the even code.
pub fn round_fp4(v: f32) -> f32 {
    let m = v.abs();
    let grid = &FP4_VALUES[..8];
    let hi = grid.iter().position(|&g| g >= m).unwrap_or(7);
    let lo = hi.saturating_sub(1);
    let (dl, dh) = (m - grid[lo], grid[hi] - m);
    let code = if dh < dl || (dh == dl && hi % 2 == 0) { hi } else { lo };
    grid[code].copysign(v)
}

#[cfg(test)]
mod tests {

    #[test]
    fn round_e4m3_matches_the_table() {
        let table = |v: f32| fp8_e4m3_to_f32(f32_to_fp8_e4m3(v));
        let mut vals: Vec<f32> = Vec::new();
        // every code, the midpoints between neighbours (ties), and a hair either side
        for c in 0u8..0x7e {
            let (a, b) = (fp8_e4m3_to_f32(c), fp8_e4m3_to_f32(c + 1));
            let m = (a + b) / 2.0;
            vals.extend([a, b, m, f32::from_bits(m.to_bits() + 1), f32::from_bits(m.to_bits().saturating_sub(1))]);
        }
        // a log-uniform sweep over the whole range, both signs
        let mut s = 0x1234_5678u64;
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f32 / (1u64 << 53) as f32;
            vals.push((u * 40.0 - 31.0).exp2() * if s & 1 == 0 { 1.0 } else { -1.0 });
        }
        vals.extend([0.0, -0.0, 448.0, -448.0, 1e-30, -1e-30]);
        for &v in vals.iter().filter(|v| v.abs() <= FP8_MAX) {
            for v in [v, -v] {
                assert_eq!(round_e4m3(v).to_bits(), table(v).to_bits(), "{v:e}");
            }
        }
    }

    use super::*;

    #[test]
    fn fp4_rounding_ties_to_even() {
        let got: Vec<f32> = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.3, -0.75, 6.0].iter().map(|&v| round_fp4(v)).collect();
        assert_eq!(got, [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.5, -1.0, 6.0]);
    }

    #[test]
    fn fp8_known_values() {
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0x7e), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9)); // smallest subnormal
        assert_eq!(fp8_e4m3_to_f32(0x08), 2f32.powi(-6)); // smallest normal
        assert_eq!(fp8_e4m3_to_f32(0xb8), -1.0);
        assert!(fp8_e4m3_to_f32(0x7f).is_nan());
    }

    #[test]
    fn fp8_roundtrip_and_ties() {
        for b in 0u8..=0x7e {
            let v = fp8_e4m3_to_f32(b);
            assert_eq!(f32_to_fp8_e4m3(v), b, "code {b:#x}");
            assert_eq!(f32_to_fp8_e4m3(-v) & 0x7f, b);
        }
        // 1.0 (0x38) and 1.125 (0x39): the midpoint rounds to the even code
        assert_eq!(f32_to_fp8_e4m3(1.0625), 0x38);
        // 1.125 (0x39) and 1.25 (0x3a): midpoint goes up to the even code
        assert_eq!(f32_to_fp8_e4m3(1.1875), 0x3a);
        assert_eq!(f32_to_fp8_e4m3(500.0), 0x7e);
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        // 1 + 2^-8 is the midpoint between 1.0 and the next bf16: ties to even
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3f82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_decoding() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn scales_and_pow2() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(124), 0.125);
        assert_eq!(pow2_ceil(1.0), 1.0);
        assert_eq!(pow2_ceil(1.5), 2.0);
        assert_eq!(pow2_ceil(448.0), 512.0);
        assert_eq!(pow2_ceil(3e-5), 2f32.powi(-15));
    }

    #[test]
    fn fake_quant_is_idempotent_on_its_output() {
        let x: Vec<f32> = (0..64).map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.173).collect();
        let q = fake_quant_fp8(&x, 32);
        assert_eq!(fake_quant_fp8(&q, 32), q);
        for (a, b) in x.iter().zip(&q) {
            assert!((a - b).abs() <= a.abs() * 0.0625 + 1e-6, "{a} -> {b}");
        }
    }
}
