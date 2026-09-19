//! Q4_K (K-quant): 256 elements per super-block, 144 bytes/block.
//!
//! Block layout:
//!   [0..2]    f16 d                  -- super-block scale
//!   [2..4]    f16 dmin               -- super-block min
//!   [4..16]   u8  scales[12]         -- 8 6-bit scales + 8 6-bit mins, packed
//!   [16..144] u8  qs[128]            -- 256 4-bit nibbles (32 elements per sub-block, 8 sub-blocks)
//!
//! Algorithm matches `dequantize_row_q4_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 144;

/// Unpack a sub-block scale and min for sub-block index `j` (0..8).
///
/// The 12-byte `scales` array packs 8 × 6-bit scales and 8 × 6-bit mins:
///   - sub-blocks 0..3:  scale = scales[j]   & 0x3F
///                       min   = scales[j+4] & 0x3F
///   - sub-blocks 4..7:  bits split across scales[j-4..=j+4]
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
    let qs = &src[16..144];

    let mut y_off = 0;
    let mut q_off = 0;
    for is in (0..8).step_by(2) {
        let (sc1, m1) = unpack_scale_min(is + 0, scales);
        let (sc2, m2) = unpack_scale_min(is + 1, scales);
        let d1 = d * sc1 as f32;
        let d2 = d * sc2 as f32;
        let mm1 = min * m1 as f32;
        let mm2 = min * m2 as f32;

        // 32 outputs from low nibbles using sub-block scale `is`.
        for l in 0..32 {
            dst[y_off + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - mm1;
        }
        // 32 outputs from high nibbles using sub-block scale `is+1`.
        for l in 0..32 {
            dst[y_off + 32 + l] = d2 * ((qs[q_off + l] >> 4) & 0x0F) as f32 - mm2;
        }
        y_off += 64;
        q_off += 32;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(q_off, 128);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
