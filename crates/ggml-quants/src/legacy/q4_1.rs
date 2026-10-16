//! Q4_1: 32 elements per block, 20 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..4]   f16 m            -- per-block min
//!   [4..20]  u8  qs[16]
//!
//! Dequant: y = nibble * d + m.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 20;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let m = read_f16(&src[2..4]);
    let qs = &src[4..20];

    for i in 0..16 {
        let lo = (qs[i] & 0x0F) as f32;
        let hi = ((qs[i] >> 4) & 0x0F) as f32;
        dst[i]      = lo * d + m;
        dst[i + 16] = hi * d + m;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
