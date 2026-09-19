//! Q6_K (K-quant): 256 elements per super-block, 210 bytes/block.
//!
//! Block layout:
//!   [0..128]   u8 ql[128]         -- low 4 bits of each value
//!   [128..192] u8 qh[64]          -- high 2 bits, packed
//!   [192..208] i8 scales[16]      -- one signed scale per 16-element group
//!   [208..210] f16 d              -- super-block scale
//!
//! Each value is reconstructed as `(low4 | (high2 << 4)) - 32` × `d` × `scales[group]`.
//! Algorithm matches `dequantize_row_q6_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 210;

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let ql = &src[0..128];
    let qh = &src[128..192];
    // scales are signed
    let scales: &[i8] = unsafe {
        std::slice::from_raw_parts(src[192..208].as_ptr() as *const i8, 16)
    };
    let d = read_f16(&src[208..210]);

    let mut y_off = 0;
    let mut ql_off = 0;
    let mut qh_off = 0;
    let mut sc_off = 0;

    for _ in 0..2 {
        // Process 128 outputs per outer iteration.
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[ql_off + l]      & 0x0F) | (((qh[qh_off + l] >> 0) & 0x3) << 4)) as i8 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 0x3) << 4)) as i8 - 32;
            let q3 = ((ql[ql_off + l]      >> 4)   | (((qh[qh_off + l] >> 4) & 0x3) << 4)) as i8 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4)   | (((qh[qh_off + l] >> 6) & 0x3) << 4)) as i8 - 32;
            dst[y_off + l]       = d * scales[sc_off + is]     as f32 * q1 as f32;
            dst[y_off + l + 32]  = d * scales[sc_off + is + 2] as f32 * q2 as f32;
            dst[y_off + l + 64]  = d * scales[sc_off + is + 4] as f32 * q3 as f32;
            dst[y_off + l + 96]  = d * scales[sc_off + is + 6] as f32 * q4 as f32;
        }
        y_off  += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    debug_assert_eq!(y_off, 256);
    debug_assert_eq!(ql_off, 128);
    debug_assert_eq!(qh_off, 64);
    debug_assert_eq!(sc_off, 16);
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
