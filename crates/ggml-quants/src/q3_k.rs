//! Q3_K (K-quant): 256 elements per super-block, 110 bytes/block.
//!
//! Block layout:
//!   [0..32]    u8 hmask[32]      -- high (3rd) bit of each value, 1 bit per value
//!   [32..96]   u8 qs[64]         -- low 2 bits of each value, 4 values per byte
//!   [96..108]  u8 scales[12]     -- 16 6-bit signed scales, packed
//!   [108..110] f16 d             -- super-block scale
//!
//! Each value reconstructs to:
//!   raw3 = (qs[i] >> shift) & 3   (low 2 bits)
//!   if hmask bit set: q3 = raw3
//!   else:             q3 = raw3 - 4
//!   y = d * (scale[is] - 32) * q3
//!
//! Scale unpacking uses kmask1=0x03030303, kmask2=0x0f0f0f0f to reassemble
//! 16 6-bit values from the 12-byte packed `scales` array.
//!
//! Algorithm matches `dequantize_row_q3_K` in `ggml-quants.c`.

use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
pub const BYTES_PER_BLOCK: usize = 110;

/// Unpack the 12-byte scales array into 16 signed scale bytes (each in roughly [0..63]).
#[inline]
fn unpack_scales(scales: &[u8]) -> [i8; 16] {
    debug_assert_eq!(scales.len(), 12);
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes([scales[0], scales[1], scales[2], scales[3]]);
    aux[1] = u32::from_le_bytes([scales[4], scales[5], scales[6], scales[7]]);
    aux[2] = u32::from_le_bytes([scales[8], scales[9], scales[10], scales[11]]);

    let tmp = aux[2];
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2)        | (((tmp >> 2) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2)        | (((tmp >> 0) & KMASK1) << 4);

    let mut out = [0i8; 16];
    for i in 0..4 {
        let bytes = aux[i].to_le_bytes();
        out[i * 4 + 0] = bytes[0] as i8;
        out[i * 4 + 1] = bytes[1] as i8;
        out[i * 4 + 2] = bytes[2] as i8;
        out[i * 4 + 3] = bytes[3] as i8;
    }
    out
}

#[inline]
pub fn dequantize_block(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), BYTES_PER_BLOCK);
    debug_assert_eq!(dst.len(), BLOCK_SIZE);

    let hmask = &src[0..32];
    let qs = &src[32..96];
    let scales = unpack_scales(&src[96..108]);
    let d_all = read_f16(&src[108..110]);

    let mut y_off = 0;
    let mut q_base = 0usize;
    let mut m: u8 = 1;
    let mut is = 0usize;
    for _outer in 0..2 {
        let mut shift = 0;
        for _j in 0..4 {
            let dl1 = d_all * (scales[is] as f32 - 32.0); is += 1;
            for l in 0..16 {
                let raw = ((qs[q_base + l] >> shift) & 0x3) as i32;
                let q3 = raw - if (hmask[l] & m) != 0 { 0 } else { 4 };
                dst[y_off + l] = dl1 * q3 as f32;
            }
            let dl2 = d_all * (scales[is] as f32 - 32.0); is += 1;
            for l in 0..16 {
                let raw = ((qs[q_base + l + 16] >> shift) & 0x3) as i32;
                let q3 = raw - if (hmask[l + 16] & m) != 0 { 0 } else { 4 };
                dst[y_off + 16 + l] = dl2 * q3 as f32;
            }
            y_off += 32;
            shift += 2;
            m <<= 1;
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
