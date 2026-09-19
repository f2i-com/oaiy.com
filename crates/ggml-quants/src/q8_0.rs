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
