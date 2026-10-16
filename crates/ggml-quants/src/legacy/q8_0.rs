//! Q8_0: 32 elements per block, 34 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..34]  i8  qs[32]
//!
//! Dequant: y = qs * d.

use crate::read_f16;

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
}
