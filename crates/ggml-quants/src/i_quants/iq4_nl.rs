//! IQ4_NL (importance-weighted 4-bit, non-linear): 32 elements per block, 18 bytes/block.
//!
//! Block layout:
//!   [0..2]   f16 d
//!   [2..18]  u8  qs[16]      (32 4-bit indices into the lookup table)
//!
//! Dequant: y[i] = d * KVALUES[qs[i/2] >> (4 * (i & 1))]
//!
//! The non-linear part is the 16-entry lookup table — values are spaced to better
//! match Gaussian-distributed weights than uniform 4-bit quantization.
//!
//! Algorithm matches `dequantize_row_iq4_nl` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 32;
pub const BYTES_PER_BLOCK: usize = 18;

/// Non-linear lookup table from upstream ggml-quants. Spaced to approximate
/// a Gaussian distribution of weights better than uniform [-8, 7] would.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
];

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let d = read_f16(&src[0..2]);
    let qs = &src[2..18];

    for i in 0..16 {
        let lo = (qs[i] & 0x0F) as usize;
        let hi = ((qs[i] >> 4) & 0x0F) as usize;
        dst[i]      = d * KVALUES_IQ4NL[lo] as f32;
        dst[i + 16] = d * KVALUES_IQ4NL[hi] as f32;
    }
}

pub fn dequantize(src: &[u8], dst: &mut [f32]) {
    for (block, out) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
        dequantize_block(block, out);
    }
}
