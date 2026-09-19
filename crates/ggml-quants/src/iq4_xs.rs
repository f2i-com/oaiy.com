//! IQ4_XS (importance-weighted 4-bit, super-block): 256 elements per super-block, 136 bytes.
//!
//! Block layout:
//!   [0..2]    f16 d
//!   [2..4]    u16 scales_h     -- high 2 bits of each of 8 sub-block 6-bit signed scales
//!   [4..8]    u8  scales_l[4]  -- low 4 bits, two sub-blocks per byte
//!   [8..136]  u8  qs[128]      -- 256 4-bit indices into KVALUES_IQ4NL
//!
//! Per sub-block ib in 0..8:
//!   ls = (scales_l[ib/2] >> (4*(ib%2)) & 0xF) | (((scales_h >> (2*ib)) & 0x3) << 4)
//!   dl = d * (ls - 32)
//!   for j in 0..16:
//!       y[j]    = dl * KVALUES_IQ4NL[qs[ib*16 + j] & 0xF]
//!       y[j+16] = dl * KVALUES_IQ4NL[qs[ib*16 + j] >> 4]
//!
//! Algorithm matches `dequantize_row_iq4_xs` in `ggml-quants.c`.

use crate::iq4_nl::KVALUES_IQ4NL;
use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 136;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let scales_h = u16::from_le_bytes([src[2], src[3]]);
    let scales_l = &src[4..8];
    let qs = &src[8..136];

    let mut y_off = 0;
    for ib in 0..8 {
        let lo4 = ((scales_l[ib / 2] >> (4 * (ib & 1))) & 0x0F) as u32;
        let hi2 = ((scales_h as u32 >> (2 * ib)) & 0x03) << 4;
        let ls = (lo4 | hi2) as i32;
        let dl = d * (ls - 32) as f32;
        let q_off = ib * 16;
        for j in 0..16 {
            let lo = (qs[q_off + j] & 0x0F) as usize;
            let hi = ((qs[q_off + j] >> 4) & 0x0F) as usize;
            dst[y_off + j]      = dl * KVALUES_IQ4NL[lo] as f32;
            dst[y_off + 16 + j] = dl * KVALUES_IQ4NL[hi] as f32;
        }
        y_off += 32;
    }
    debug_assert_eq!(y_off, 256);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
