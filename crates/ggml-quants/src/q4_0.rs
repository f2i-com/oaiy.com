//! Q4_0: 32 elements per block, 18 bytes/block.
//!
//! Block layout (little-endian):
//!   [0..2]  f16 d            -- per-block scale
//!   [2..18] u8 qs[16]        -- 32 4-bit nibbles, packed two-per-byte
//!
//! Dequant: y = (nibble - 8) * d.
//! The low nibble of qs[i] holds element i; the high nibble holds element i+16.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qs = &src[2..18];

    for i in 0..16 {
        let lo = (qs[i] & 0x0F) as i32 - 8;
        let hi = ((qs[i] >> 4) & 0x0F) as i32 - 8;
        dst[i]      = lo as f32 * d;
        dst[i + 16] = hi as f32 * d;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
