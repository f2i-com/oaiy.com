//! Q5_K (K-quant): 256 elements per super-block, 176 bytes/block.
//!
//! Block layout:
//!   [0..2]    f16 d                  -- super-block scale
//!   [2..4]    f16 dmin               -- super-block min
//!   [4..16]   u8  scales[12]         -- 8 6-bit scales + 8 6-bit mins (same packing as Q4_K)
//!   [16..48]  u8  qh[32]             -- 5th bit of each value (one bit per value, 8 values per byte)
//!   [48..176] u8  qs[128]            -- 256 4-bit nibbles
//!
//! Each value reconstructs to:
//!   nibble + (qh_bit << 4) -> in [0..32]
//!   y = d_sub_block * v - min_sub_block
//!
//! Algorithm matches `dequantize_row_q5_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 176;

/// Same packing as Q4_K — 12 bytes hold 8 6-bit scales + 8 6-bit mins.
#[inline]
fn unpack_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 0x3F, scales[j + 4] & 0x3F)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4)   | ((scales[j]     >> 6) << 4);
        (d, m)
    }
}

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d   = read_f16(&src[0..2]);
    let min = read_f16(&src[2..4]);
    let scales = &src[4..16];
    let qh = &src[16..48];
    let qs = &src[48..176];

    let mut y_off = 0;
    let mut q_off = 0;
    // Bit masks for the high bit. Each iteration of the outer loop bumps both by 2.
    let mut u1: u8 = 0x01;
    let mut u2: u8 = 0x02;
    for is in (0..8).step_by(2) {
        let (sc1, m1) = unpack_scale_min(is + 0, scales);
        let (sc2, m2) = unpack_scale_min(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let mm1 = min * m1 as f32;
        let mm2 = min * m2 as f32;

        for l in 0..32 {
            let high = if (qh[l] & u1) != 0 { 16 } else { 0 };
            let v = (qs[q_off + l] & 0x0F) as i32 + high;
            dst[y_off + l] = d1 * v as f32 - mm1;
        }
        for l in 0..32 {
            let high = if (qh[l] & u2) != 0 { 16 } else { 0 };
            let v = ((qs[q_off + l] >> 4) & 0x0F) as i32 + high;
            dst[y_off + 32 + l] = d2 * v as f32 - mm2;
        }
        y_off += 64;
        q_off += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(q_off, 128);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
