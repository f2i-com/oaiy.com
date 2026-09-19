//! Q5_0: 32 elements per block, 22 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..6]   u32 qh           -- high 5th bit of each element
//!   [6..22]  u8  qs[16]
//!
//! Dequant: y = ((nibble | (qh_bit << 4)) - 16) * d.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 22;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
    let qs = &src[6..22];

    for i in 0..16 {
        let xh0 = ((qh >> i) & 0x1) as u8;
        let xh1 = ((qh >> (i + 16)) & 0x1) as u8;
        let lo = ((qs[i] & 0x0F) | (xh0 << 4)) as i32 - 16;
        let hi = (((qs[i] >> 4) & 0x0F) | (xh1 << 4)) as i32 - 16;
        dst[i]      = lo as f32 * d;
        dst[i + 16] = hi as f32 * d;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
