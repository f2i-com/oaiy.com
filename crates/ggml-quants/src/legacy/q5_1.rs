//! Q5_1: 32 elements per block, 24 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..4]   f16 m
//!   [4..8]   u32 qh
//!   [8..24]  u8  qs[16]
//!
//! Dequant: y = (nibble | (qh_bit << 4)) * d + m.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 24;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let m = read_f16(&src[2..4]);
    let qh = u32::from_le_bytes([src[4], src[5], src[6], src[7]]);
    let qs = &src[8..24];

    for i in 0..16 {
        let xh0 = ((qh >> i) & 0x1) as u8;
        let xh1 = ((qh >> (i + 16)) & 0x1) as u8;
        let lo = ((qs[i] & 0x0F) | (xh0 << 4)) as f32;
        let hi = (((qs[i] >> 4) & 0x0F) | (xh1 << 4)) as f32;
        dst[i]      = lo * d + m;
        dst[i + 16] = hi * d + m;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
