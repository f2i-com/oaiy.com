//! Block-quantization formats compatible with ggml / llama.cpp.
//!
//! Every supported [`GgmlType`] has a fixed *block layout* (number of elements
//! per block, byte size of each block) and a dequantization routine that
//! produces `f32` output. The block-byte layouts here MUST stay byte-for-byte
//! compatible with upstream `ggml-quants.c`.
//!
//! Naming matches upstream — `Q4_0`, `Q4_K`, `IQ2_XXS`, etc.

#![deny(rust_2018_idioms)]
#![allow(non_camel_case_types)]

pub mod dtype;
pub mod error;

pub mod iq4_nl;
pub mod iq4_xs;
pub mod q2_0;
pub mod q2_k;
pub mod q3_k;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q5_k;
pub mod q6_k;
pub mod q8_0;

pub use dtype::GgmlType;
pub use error::{QuantError, Result};

use half::f16;

/// Dequantize a tensor body into a flat `f32` buffer.
///
/// `src` is the raw tensor bytes (multiple-of-block sized).
/// `dst` must already be sized to the element count.
///
/// Returns `Err` for unsupported dtypes — see [`is_supported`] to gate.
pub fn dequantize(dtype: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    if n % dtype.block_size() != 0 {
        return Err(QuantError::NotBlockAligned {
            n,
            block: dtype.block_size(),
        });
    }
    let expected_bytes = (n / dtype.block_size()) * dtype.type_size();
    if src.len() != expected_bytes {
        return Err(QuantError::WrongLength {
            got: src.len(),
            expected: expected_bytes,
        });
    }

    match dtype {
        GgmlType::F32 => copy_f32(src, dst),
        GgmlType::F16 => dequant_f16(src, dst),
        GgmlType::BF16 => dequant_bf16(src, dst),
        GgmlType::Q2_0 => q2_0::dequantize(src, dst),
        GgmlType::Q4_0 => q4_0::dequantize(src, dst),
        GgmlType::Q4_1 => q4_1::dequantize(src, dst),
        GgmlType::Q5_0 => q5_0::dequantize(src, dst),
        GgmlType::Q5_1 => q5_1::dequantize(src, dst),
        GgmlType::Q8_0 => q8_0::dequantize(src, dst),
        GgmlType::IQ4_NL => iq4_nl::dequantize(src, dst),
        GgmlType::IQ4_XS => iq4_xs::dequantize(src, dst),
        GgmlType::Q2_K => q2_k::dequantize(src, dst),
        GgmlType::Q3_K => q3_k::dequantize(src, dst),
        GgmlType::Q4_K => q4_k::dequantize(src, dst),
        GgmlType::Q5_K => q5_k::dequantize(src, dst),
        GgmlType::Q6_K => q6_k::dequantize(src, dst),
        other => return Err(QuantError::Unsupported(other)),
    }
    Ok(())
}

pub fn is_supported(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
            | GgmlType::Q2_0
            | GgmlType::Q4_0 | GgmlType::Q4_1
            | GgmlType::Q5_0 | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::IQ4_NL | GgmlType::IQ4_XS
            | GgmlType::Q2_K | GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

fn copy_f32(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len() * 4);
    for (chunk, out) in src.chunks_exact(4).zip(dst.iter_mut()) {
        *out = f32::from_le_bytes(chunk.try_into().unwrap());
    }
}

fn dequant_f16(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len() * 2);
    for (chunk, out) in src.chunks_exact(2).zip(dst.iter_mut()) {
        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
        *out = f16::from_bits(bits).to_f32();
    }
}

fn dequant_bf16(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len() * 2);
    for (chunk, out) in src.chunks_exact(2).zip(dst.iter_mut()) {
        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
        // bfloat16 is a truncated f32 — pad to the high half.
        *out = f32::from_bits((bits as u32) << 16);
    }
}

/// Read an f16 from a 2-byte little-endian slice.
#[inline]
pub(crate) fn read_f16(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn write_f16_le(out: &mut Vec<u8>, x: f32) {
        out.extend_from_slice(&f16::from_f32(x).to_bits().to_le_bytes());
    }

    #[test]
    fn f32_passthrough() {
        let xs: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
        let bytes: Vec<u8> = xs.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut out = vec![0.0; 16];
        dequantize(GgmlType::F32, &bytes, &mut out).unwrap();
        assert_eq!(out, xs);
    }

    #[test]
    fn f16_dequant_roundtrip() {
        let xs: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 4.0).collect();
        let bytes: Vec<u8> = xs.iter().flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes()).collect();
        let mut out = vec![0.0; 16];
        dequantize(GgmlType::F16, &bytes, &mut out).unwrap();
        for (a, b) in out.iter().zip(xs.iter()) {
            assert!((a - b).abs() < 1e-3, "f16 dequant: {a} vs {b}");
        }
    }

    #[test]
    fn bf16_dequant_keeps_sign() {
        // bfloat16 truncates the mantissa; sign + exponent survive.
        let xs = [-2.0_f32, 0.0, 1.5, 32.0];
        let mut bytes = Vec::new();
        for x in &xs {
            let trunc = (x.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&trunc.to_le_bytes());
        }
        let mut out = vec![0.0; 4];
        dequantize(GgmlType::BF16, &bytes, &mut out).unwrap();
        assert_eq!(out, xs);
    }

    /// Round-trip a Q8_0 block manually: encode known values, decode, compare.
    #[test]
    fn q8_0_known_block() {
        // d = 0.25, qs = -8, -4, 0, 4, ... 32 values in steps of 1
        let d: f32 = 0.25;
        let qs: Vec<i8> = (-16..16).collect();
        let mut bytes = Vec::new();
        write_f16_le(&mut bytes, d);
        for q in &qs { bytes.push(*q as u8); }
        assert_eq!(bytes.len(), 34);

        let mut out = vec![0.0; 32];
        dequantize(GgmlType::Q8_0, &bytes, &mut out).unwrap();

        for (i, q) in qs.iter().enumerate() {
            let expected = (*q as f32) * f16::from_f32(d).to_f32();
            assert!((out[i] - expected).abs() < 1e-6,
                    "q8_0 i={i}: {} vs {}", out[i], expected);
        }
    }

    /// Q4_0 known block. d=1.0, all nibbles = 8 (zero after -8 offset) -> all zeros.
    #[test]
    fn q4_0_zero_block() {
        let mut bytes = Vec::new();
        write_f16_le(&mut bytes, 1.0);
        // Each byte = 0x88 (low nibble 8, high nibble 8 -> both decode to 0)
        bytes.extend_from_slice(&[0x88; 16]);
        let mut out = vec![0.0; 32];
        dequantize(GgmlType::Q4_0, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }

    /// Q4_0 with d=2.0 and varying nibbles. nibble n -> (n - 8) * 2.0.
    #[test]
    fn q4_0_known_block() {
        let mut bytes = Vec::new();
        write_f16_le(&mut bytes, 2.0);
        // Pattern: low nibble = i, high nibble = i  -> elements 0..16 = i-8, 16..32 = i-8
        for i in 0..16u8 {
            bytes.push((i << 4) | i);
        }
        let mut out = vec![0.0; 32];
        dequantize(GgmlType::Q4_0, &bytes, &mut out).unwrap();
        let d = f16::from_f32(2.0).to_f32();
        for i in 0..16 {
            let expected = (i as f32 - 8.0) * d;
            assert!((out[i] - expected).abs() < 1e-5);
            assert!((out[i + 16] - expected).abs() < 1e-5);
        }
    }

    /// Q5_0: like Q4_0 but with an extra high bit. With qh=0, high bit is always 0,
    /// so nibble n decodes to (n - 16) * d.
    #[test]
    fn q5_0_no_high_bits() {
        let mut bytes = Vec::new();
        write_f16_le(&mut bytes, 1.0);            // d
        bytes.extend_from_slice(&[0u8; 4]);       // qh = 0
        for i in 0..16u8 { bytes.push((i << 4) | i); }
        let mut out = vec![0.0; 32];
        dequantize(GgmlType::Q5_0, &bytes, &mut out).unwrap();
        let d = f16::from_f32(1.0).to_f32();
        for i in 0..16 {
            let expected = (i as f32 - 16.0) * d;
            assert!((out[i] - expected).abs() < 1e-5);
        }
    }

    /// Q5_0: all qh bits set -> the 5th bit is always 1. nibble n decodes to (n + 16 - 16) = n.
    #[test]
    fn q5_0_all_high_bits() {
        let mut bytes = Vec::new();
        write_f16_le(&mut bytes, 1.0);
        bytes.extend_from_slice(&0xFFFFFFFFu32.to_le_bytes());
        for i in 0..16u8 { bytes.push((i << 4) | i); }
        let mut out = vec![0.0; 32];
        dequantize(GgmlType::Q5_0, &bytes, &mut out).unwrap();
        for i in 0..16 {
            let expected = i as f32;
            assert!((out[i] - expected).abs() < 1e-5);
            assert!((out[i + 16] - expected).abs() < 1e-5);
        }
    }

    /// Q4_K with all scales / mins zero and uniform nibbles -> all output is zero.
    #[test]
    fn q4_k_all_zero_output() {
        let mut bytes = vec![0u8; q4_k::BYTES_PER_BLOCK];
        // d, dmin = 1.0 just to make sure we don't NaN — but scales/mins all zero => 0 output.
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        bytes[0..2].copy_from_slice(&one);
        bytes[2..4].copy_from_slice(&one);
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q4_K, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }

    /// IQ4_NL: with qs=0 (low nibble 0 -> KVALUES[0] = -127), check decode against
    /// the canonical table.
    #[test]
    fn iq4_nl_known_block() {
        let mut bytes = Vec::with_capacity(iq4_nl::BYTES_PER_BLOCK);
        let d = 0.5_f32;
        bytes.extend_from_slice(&f16::from_f32(d).to_bits().to_le_bytes());
        // Pattern: low nibble = i, high nibble = 15 - i
        for i in 0..16u8 {
            bytes.push((i & 0x0F) | (((15 - i) & 0x0F) << 4));
        }
        let mut out = vec![0.0; 32];
        dequantize(GgmlType::IQ4_NL, &bytes, &mut out).unwrap();
        let d16 = f16::from_f32(d).to_f32();
        for i in 0..16 {
            let lo_expected = d16 * iq4_nl::KVALUES_IQ4NL[i] as f32;
            let hi_expected = d16 * iq4_nl::KVALUES_IQ4NL[15 - i] as f32;
            assert!((out[i] - lo_expected).abs() < 1e-5,
                    "iq4_nl lo i={i}: {} vs {}", out[i], lo_expected);
            assert!((out[i + 16] - hi_expected).abs() < 1e-5,
                    "iq4_nl hi i={i}: {} vs {}", out[i + 16], hi_expected);
        }
    }

    /// IQ4_XS: scales packed such that every sub-block decodes to scale=32 (i.e. signed=0)
    /// -> all zero output. Encoding: scales_l=0x00, scales_h with 2 bits per sub-block = 0b10
    /// (so combined ls = 0x20 = 32).
    #[test]
    fn iq4_xs_zero_signed_scales() {
        let mut bytes = vec![0xFFu8; iq4_xs::BYTES_PER_BLOCK];
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        bytes[0..2].copy_from_slice(&one);
        // scales_h: 2 bits per sub-block × 8 sub-blocks. Each pair = 0b10.
        // 0b10 repeated 8 times = 0b1010_1010_1010_1010 = 0xAAAA
        bytes[2] = 0xAA; bytes[3] = 0xAA;
        // scales_l[4]: low 4 bits = 0 for every sub-block
        for i in 0..4 { bytes[4 + i] = 0x00; }
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::IQ4_XS, &bytes, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.0, "iq4_xs i={i}: {v}");
        }
    }

    /// Q2_K with d=dmin=0 -> all zero output regardless of qs/scales.
    #[test]
    fn q2_k_zero_d_dmin_zeros_output() {
        let mut bytes = vec![0xFFu8; q2_k::BYTES_PER_BLOCK];
        bytes[80] = 0; bytes[81] = 0;   // d (f16) = 0
        bytes[82] = 0; bytes[83] = 0;   // dmin (f16) = 0
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q2_K, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }

    /// Q3_K with d=0 -> all zero output regardless of remaining bytes.
    #[test]
    fn q3_k_zero_d_zeros_output() {
        let mut bytes = vec![0xFFu8; q3_k::BYTES_PER_BLOCK];
        // d (f16 little-endian) = 0
        bytes[108] = 0; bytes[109] = 0;
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q3_K, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }

    /// Q3_K canonical block: d=1.0, all scales decode to 32 (so scale_signed = scale - 32 = 0)
    /// -> output = 0 regardless of qs/hmask.
    /// Encoding: scale value 32 = 0b100000. Low 4 bits = 0, high 2 bits = 0b10.
    /// In the 12-byte packed scales:
    ///   scales_packed[0..8]   hold low 4 bits of scales[0..8] in low nibble,
    ///                         and low 4 bits of scales[8..16] in high nibble.
    ///   scales_packed[8..12]  hold high 2 bits of all 16 scales packed two bits per scale.
    /// For all-32 scales: bytes 0..8 = 0x00, bytes 8..12 = 0xAA (0b10101010).
    #[test]
    fn q3_k_zero_scales_after_offset() {
        let mut bytes = vec![0xFFu8; q3_k::BYTES_PER_BLOCK];   // qs/hmask non-zero
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        bytes[108..110].copy_from_slice(&one);
        for j in 0..8 { bytes[96 + j] = 0x00; }
        for j in 0..4 { bytes[96 + 8 + j] = 0xAA; }
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q3_K, &bytes, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, 0.0, "i={i}: {v}");
        }
    }

    /// Q5_K with d=dmin=1.0, scales[..]=1, mins=0, qs=0, qh=0 ->
    /// all values dequant to 0 (low nibble 0 + high bit 0 = 0; * 1 - 0 = 0).
    #[test]
    fn q5_k_zero_block() {
        let mut bytes = vec![0u8; q5_k::BYTES_PER_BLOCK];
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        bytes[0..2].copy_from_slice(&one);
        bytes[2..4].copy_from_slice(&one);
        // 8 6-bit scales = 1, 8 6-bit mins = 0  ->  scales[0..4]=0x01, scales[4..8]=0x00
        for i in 0..4 { bytes[4 + i] = 0x01; }
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q5_K, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }

    /// Q5_K with high bits set and a known low nibble — verifies high-bit promotion.
    /// d=1.0, dmin=1.0, sub-block 0 scale=1, mins=0, qs[0]=0x01 -> low nibble 1.
    /// qh[0] = 0x01 -> high bit set in sub-block 0 only.
    /// Expected dst[0] = 1.0 * (1 + 16) - 0 = 17.0.
    #[test]
    fn q5_k_high_bit_promotion() {
        let mut bytes = vec![0u8; q5_k::BYTES_PER_BLOCK];
        let one = f16::from_f32(1.0).to_bits().to_le_bytes();
        bytes[0..2].copy_from_slice(&one);
        bytes[2..4].copy_from_slice(&one);
        for i in 0..4 { bytes[4 + i] = 0x01; }   // 8 6-bit scales = 1
        bytes[16] = 0x01;                         // qh[0]: bit 0 set => sub-block 0 high bit on for value 0
        bytes[48] = 0x01;                         // qs[0] low nibble = 1
        let mut out = vec![0.0; 256];
        dequantize(GgmlType::Q5_K, &bytes, &mut out).unwrap();
        let d = f16::from_f32(1.0).to_f32();
        let expected = d * 1.0 * (1.0 + 16.0);
        assert!((out[0] - expected).abs() < 1e-5, "out[0]={} expected={}", out[0], expected);
    }

    /// Q6_K with d=0, scales=0 -> all zero output regardless of ql/qh.
    #[test]
    fn q6_k_zero_scales_zeros_output() {
        let mut bytes = vec![0xFFu8; q6_k::BYTES_PER_BLOCK];
        // scales[16] = 0 (i8)
        for i in 0..16 { bytes[192 + i] = 0; }
        // d = 0 (f16)
        bytes[208] = 0; bytes[209] = 0;
        let mut out = vec![1.0; 256];
        dequantize(GgmlType::Q6_K, &bytes, &mut out).unwrap();
        for v in out { assert_eq!(v, 0.0); }
    }
}
