//! Q2_K (K-quant): 256 elements per super-block, 84 bytes/block.
//!
//! Block layout:
//!   [0..16]   u8  scales[16]   -- 16 4-bit scales (low nibble) + 16 4-bit mins (high nibble)
//!   [16..80]  u8  qs[64]       -- 256 2-bit quants, 4 values per byte
//!   [80..82]  f16 d
//!   [82..84]  f16 dmin
//!
//! Each value reconstructs to:
//!   raw2 = (qs[i] >> shift) & 3
//!   y = d * scale_nibble * raw2 - dmin * min_nibble
//!
//! Algorithm matches `dequantize_row_q2_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 84;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let scales = &src[0..16];
    let qs = &src[16..80];
    let d   = read_f16(&src[80..82]);
    let min = read_f16(&src[82..84]);

    let mut y_off = 0;
    let mut q_base = 0usize;
    let mut is = 0usize;
    for _outer in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let sc1 = scales[is]; is += 1;
            let dl1 = d * (sc1 & 0x0F) as f32;
            let ml1 = min * ((sc1 >> 4) & 0x0F) as f32;
            for l in 0..16 {
                let raw = ((qs[q_base + l] >> shift) & 0x3) as i32;
                dst[y_off + l] = dl1 * raw as f32 - ml1;
            }
            let sc2 = scales[is]; is += 1;
            let dl2 = d * (sc2 & 0x0F) as f32;
            let ml2 = min * ((sc2 >> 4) & 0x0F) as f32;
            for l in 0..16 {
                let raw = ((qs[q_base + l + 16] >> shift) & 0x3) as i32;
                dst[y_off + 16 + l] = dl2 * raw as f32 - ml2;
            }
            y_off += 32;
            shift += 2;
        }
        q_base += 32;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(q_base, 64);
    debug_assert_eq!(is, 16);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
