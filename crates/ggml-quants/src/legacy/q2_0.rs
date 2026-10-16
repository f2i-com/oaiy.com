//! Q2_0: 64 elements per block, 18 bytes/block (2.25 bits a weight).
//!
//! Block layout (little-endian):
//!   [0..2]  f16 d            -- per-block scale
//!   [2..18] u8 qs[16]        -- 64 2-bit codes, four a byte
//!
//! Dequant: y = (code - 1) * d, so a code of {0, 1, 2, 3} is {-1, 0, +1, +2} times the scale.
//! Element i's code is bits `(i % 4) * 2` of `qs[i / 4]`.
//!
//! Algorithm matches `dequantize_row_q2_0` in `ggml-quants.c` (ggml type 42): the type the GSQ-RCO GGUFs keep
//! Qwen3.8-Flash-Next's routed experts in.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 64;
pub const BYTES_PER_BLOCK: usize = 18;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qs = &src[2..18];

    for (i, out) in dst.iter_mut().enumerate() {
        let code = (qs[i / 4] >> ((i % 4) * 2)) & 3;
        *out = (code as i32 - 1) as f32 * d;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's codes are -1, 0, +1 and +2 times its scale, element i's at bits `(i % 4) * 2` of byte `i / 4`.
    #[test]
    fn a_block_is_its_codes_times_its_scale() {
        let mut block = [0u8; BYTES_PER_BLOCK];
        block[..2].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        // elements 0..4 the codes 0, 1, 2, 3; element 63 the code 3; element 5 the code 2; the rest 0
        block[2] = 0b11_10_01_00;
        block[3] = 0b00_00_10_00;
        block[17] = 0b11_00_00_00;
        let mut out = [9.0f32; BLOCK_SIZE];
        dequantize(&block, &mut out);
        assert_eq!(&out[..6], &[-0.25, 0.0, 0.25, 0.5, -0.25, 0.25]);
        assert_eq!(out[63], 0.5);
        assert!(out[6..63].iter().all(|v| *v == -0.25));
    }
}
