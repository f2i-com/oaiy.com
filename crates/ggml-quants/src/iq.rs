//! The IQ types whose groups of weights are looked up in a grid: IQ2_XXS, IQ2_XS, IQ2_S (2 to 2.5 bits a weight, 8
//! weights a grid entry), IQ3_XXS, IQ3_S (3 to 3.4, 4 weights an entry) and IQ1_S, IQ1_M (1.56 and 1.75, 8 weights
//! of -1, 0 or +1 an entry). 256 weights a block, 32 a sub-block with its own scale; a group's signs come from a
//! 7-bit pattern (`KSIGNS_IQ2XS`, the eighth its parity) or a byte.
//!
//! Each `dequantize` matches its `dequantize_row_*` in `ggml-quants.c`; the tables are ggml's ([`crate::iq_tables`]).
//! The types the GSQ-RCO GGUFs of Qwen3.8-Flash-Next keep their routed experts and some dense matrices in.

use crate::iq_tables::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use crate::read_f16;

pub const BLOCK_SIZE: usize = 256;
/// IQ1's offset of a group's values (`IQ1S_DELTA`, `IQ1M_DELTA`).
const IQ1_DELTA: f32 = 0.125;

/// `-1` where bit `j` of `signs` is set.
#[inline]
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 }
}

/// A group of 8 from a u64 grid entry (a magnitude a byte) with its signs, times `scale`.
#[inline]
fn group8(grid: u64, signs: u8, scale: f32, out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        *o = scale * ((grid >> (8 * j)) & 0xff) as f32 * sign(signs, j);
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

/// IQ2_XXS, 66 bytes a block: `d`, then per 32 weights four grid indices (a byte each) and a u32 of four 7-bit sign
/// patterns with the sub-block's 4-bit scale on top.
pub mod iq2_xxs {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 64;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib32 in 0..8 {
                let at = 2 + 8 * ib32;
                let aux = u32_at(b, at + 4);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    group8(IQ2XXS_GRID[b[at + l] as usize], signs, db, &mut y[32 * ib32 + 8 * l..]);
                }
            }
        }
    }
}

/// IQ2_XS, 74 bytes a block: `d`, 32 u16 (a 9-bit grid index under a 7-bit sign pattern, a group of 8 each), then a
/// byte a sub-block of two 4-bit scales (its first 16 weights', its last 16's).
pub mod iq2_xs {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 64 + 8;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib32 in 0..8 {
                let scales = b[66 + ib32];
                let db = [d * (0.5 + (scales & 15) as f32) * 0.25, d * (0.5 + (scales >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let q = u16_at(b, 2 + 2 * (4 * ib32 + l));
                    group8(IQ2XS_GRID[(q & 511) as usize], KSIGNS_IQ2XS[(q >> 9) as usize], db[l / 2], &mut y[32 * ib32 + 8 * l..]);
                }
            }
        }
    }
}

/// IQ2_S, 82 bytes a block: `d`, 32 bytes of grid indices' low 8 bits, 32 bytes of signs (a byte a group of 8), a
/// byte a sub-block of its four groups' indices' two high bits, a byte a sub-block of two 4-bit scales.
pub mod iq2_s {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 64 + 8 + 8;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib32 in 0..8 {
                let (qh, scales) = (b[66 + ib32] as usize, b[74 + ib32]);
                let db = [d * (0.5 + (scales & 15) as f32) * 0.25, d * (0.5 + (scales >> 4) as f32) * 0.25];
                for l in 0..4 {
                    let index = b[2 + 4 * ib32 + l] as usize | ((qh << (8 - 2 * l)) & 0x300);
                    group8(IQ2S_GRID[index], b[34 + 4 * ib32 + l], db[l / 2], &mut y[32 * ib32 + 8 * l..]);
                }
            }
        }
    }
}

/// A group of 4 from a u32 grid entry, bits `at..at + 4` of `signs` its signs.
#[inline]
fn group4(grid: u32, signs: u8, at: usize, scale: f32, out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate().take(4) {
        *o = scale * ((grid >> (8 * j)) & 0xff) as f32 * sign(signs, at + j);
    }
}

/// IQ3_XXS, 98 bytes a block: `d`, 64 bytes of grid indices (a group of 4 each), then a u32 a sub-block of four
/// 7-bit sign patterns (two groups each) with its 4-bit scale on top.
pub mod iq3_xxs {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 64 + 32;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib32 in 0..8 {
                let aux = u32_at(b, 66 + 4 * ib32);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    let q = 2 + 8 * ib32 + 2 * l;
                    let out = &mut y[32 * ib32 + 8 * l..];
                    group4(IQ3XXS_GRID[b[q] as usize], signs, 0, db, out);
                    group4(IQ3XXS_GRID[b[q + 1] as usize], signs, 4, db, &mut out[4..]);
                }
            }
        }
    }
}

/// IQ3_S, 110 bytes a block: `d`, 64 bytes of grid indices' low 8 bits, a byte a sub-block of its eight groups'
/// ninth bits, 32 bytes of signs (a byte two groups), then a byte two sub-blocks of their 4-bit scales.
pub mod iq3_s {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 64 + 8 + 32 + 4;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib32 in 0..8 {
                let scales = b[106 + ib32 / 2];
                let db = d * (1 + 2 * ((scales >> (4 * (ib32 % 2))) & 15) as u32) as f32;
                let qh = b[66 + ib32] as usize;
                for l in 0..4 {
                    let q = 2 + 8 * ib32 + 2 * l;
                    let signs = b[74 + 4 * ib32 + l];
                    let out = &mut y[32 * ib32 + 8 * l..];
                    group4(IQ3S_GRID[b[q] as usize | ((qh << (8 - 2 * l)) & 256)], signs, 0, db, out);
                    group4(IQ3S_GRID[b[q + 1] as usize | ((qh << (7 - 2 * l)) & 256)], signs, 4, db, &mut out[4..]);
                }
            }
        }
    }
}

/// A group of 8 from an IQ1 grid entry (a signed byte each: -1, 0 or +1), each plus `delta`, times `scale`.
#[inline]
fn group1(grid: u64, delta: f32, scale: f32, out: &mut [f32]) {
    for (j, o) in out.iter_mut().enumerate().take(8) {
        *o = scale * (((grid >> (8 * j)) & 0xff) as u8 as i8 as f32 + delta);
    }
}

/// IQ1_S, 50 bytes a block: `d`, 32 bytes of grid indices' low 8 bits, then a u16 a sub-block: its four groups'
/// indices' three high bits, its 3-bit scale and the sign of its groups' offset.
pub mod iq1_s {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 2 + 32 + 16;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let d = read_f16(&b[0..2]);
            for ib in 0..8 {
                let qh = u16_at(b, 34 + 2 * ib) as usize;
                let dl = d * (2 * ((qh >> 12) & 7) + 1) as f32;
                let delta = if qh & 0x8000 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
                for l in 0..4 {
                    let index = b[2 + 4 * ib + l] as usize | (((qh >> (3 * l)) & 7) << 8);
                    group1(IQ1S_GRID[index], delta, dl, &mut y[32 * ib + 8 * l..]);
                }
            }
        }
    }
}

/// IQ1_M, 56 bytes a block and no `d` field: 32 bytes of grid indices' low 8 bits, 16 bytes each two groups' high
/// bits and offsets' signs, then four u16 whose top nibbles together are the block's f16 scale, the rest each
/// sub-block's two 3-bit scales (its first 16 weights', its last 16's).
pub mod iq1_m {
    use super::*;
    pub const BYTES_PER_BLOCK: usize = 32 + 16 + 8;
    pub fn dequantize(src: &[u8], dst: &mut [f32]) {
        for (b, y) in src.chunks_exact(BYTES_PER_BLOCK).zip(dst.chunks_exact_mut(BLOCK_SIZE)) {
            let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
            let scale = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            let d = half::f16::from_bits(scale).to_f32();
            for ib in 0..8 {
                let s = sc[ib / 2] >> (6 * (ib % 2));
                let dl = [d * (2 * (s & 7) + 1) as f32, d * (2 * ((s >> 3) & 7) + 1) as f32];
                let qh = [b[32 + 2 * ib] as usize, b[33 + 2 * ib] as usize];
                for l in 0..4 {
                    let h = qh[l / 2] >> (4 * (l % 2));
                    let index = b[4 * ib + l] as usize | ((h & 7) << 8);
                    let delta = if h & 8 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
                    group1(IQ1S_GRID[index], delta, dl[l / 2], &mut y[32 * ib + 8 * l..]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_bits().to_le_bytes()
    }

    /// The type table's block sizes are the decoders' (ggml's `sizeof(block_iq*)`: 66, 74, 82, 98, 110, 50 and 56).
    #[test]
    fn the_type_tables_blocks_are_the_decoders() {
        use crate::GgmlType as T;
        let sizes = [
            (T::IQ2_XXS, iq2_xxs::BYTES_PER_BLOCK, 66),
            (T::IQ2_XS, iq2_xs::BYTES_PER_BLOCK, 74),
            (T::IQ2_S, iq2_s::BYTES_PER_BLOCK, 82),
            (T::IQ3_XXS, iq3_xxs::BYTES_PER_BLOCK, 98),
            (T::IQ3_S, iq3_s::BYTES_PER_BLOCK, 110),
            (T::IQ1_S, iq1_s::BYTES_PER_BLOCK, 50),
            (T::IQ1_M, iq1_m::BYTES_PER_BLOCK, 56),
        ];
        for (t, bytes, ggml) in sizes {
            assert_eq!((t.type_size(), t.block_size(), bytes), (ggml, BLOCK_SIZE, ggml), "{t:?}");
        }
    }

    /// Each type's block by hand: its first groups' grid entries, signs and scales where its layout puts them.
    #[test]
    fn a_blocks_groups_are_their_grid_entries_signs_and_scales() {
        // IQ2_XXS: group 0 of sub-block 0 the grid's entry 5 with sign pattern 3, the sub-block's scale 7
        let mut b = vec![0u8; iq2_xxs::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(0.5));
        b[2] = 5;
        b[6..10].copy_from_slice(&(3u32 | (7 << 28)).to_le_bytes());
        let mut y = vec![0f32; 256];
        iq2_xxs::dequantize(&b, &mut y);
        let db = 0.5 * 7.5 * 0.25;
        for j in 0..8 {
            let s = if KSIGNS_IQ2XS[3] & (1 << j) != 0 { -1.0 } else { 1.0 };
            assert_eq!(y[j], db * ((IQ2XXS_GRID[5] >> (8 * j)) & 255) as f32 * s, "IQ2_XXS weight {j}");
        }
        assert_eq!(y[8], db * (IQ2XXS_GRID[0] & 255) as f32, "the next group: entry 0, no sign");
        // IQ2_XS: group 2 of sub-block 1 the entry 300 with pattern 100, the sub-block's second scale 9
        let mut b = vec![0u8; iq2_xs::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(0.25));
        b[2 + 2 * 6..2 + 2 * 6 + 2].copy_from_slice(&(300u16 | (100 << 9)).to_le_bytes());
        b[66 + 1] = 9 << 4;
        iq2_xs::dequantize(&b, &mut y);
        for j in 0..8 {
            let s = if KSIGNS_IQ2XS[100] & (1 << j) != 0 { -1.0 } else { 1.0 };
            assert_eq!(y[32 + 16 + j], 0.25 * 9.5 * 0.25 * ((IQ2XS_GRID[300] >> (8 * j)) & 255) as f32 * s, "IQ2_XS weight {j}");
        }
        // IQ2_S: group 3 of sub-block 2 the entry 0x2a7 (its two high bits from qh), a sign byte, the second scale 4
        let mut b = vec![0u8; iq2_s::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(1.0));
        b[2 + 4 * 2 + 3] = 0xa7;
        b[66 + 2] = 2 << 6;
        b[34 + 4 * 2 + 3] = 0b1000_0101;
        b[74 + 2] = 4 << 4;
        iq2_s::dequantize(&b, &mut y);
        for j in 0..8 {
            let s = if 0b1000_0101u8 & (1 << j) != 0 { -1.0 } else { 1.0 };
            assert_eq!(y[64 + 24 + j], 4.5 * 0.25 * ((IQ2S_GRID[0x2a7] >> (8 * j)) & 255) as f32 * s, "IQ2_S weight {j}");
        }
        // IQ3_XXS: sub-block 0's first pair of groups the entries 17 and 200, pattern 9, scale 3
        let mut b = vec![0u8; iq3_xxs::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(2.0));
        (b[2], b[3]) = (17, 200);
        b[66..70].copy_from_slice(&(9u32 | (3 << 28)).to_le_bytes());
        iq3_xxs::dequantize(&b, &mut y);
        for j in 0..8 {
            let s = if KSIGNS_IQ2XS[9] & (1 << j) != 0 { -1.0 } else { 1.0 };
            let g = if j < 4 { IQ3XXS_GRID[17] >> (8 * j) } else { IQ3XXS_GRID[200] >> (8 * (j - 4)) };
            assert_eq!(y[j], 2.0 * 3.5 * 0.5 * (g & 255) as f32 * s, "IQ3_XXS weight {j}");
        }
        // IQ3_S: sub-block 1's first pair the entries 0x105 and 0x0f0 (the first's ninth bit from qh), a sign byte,
        // its scale 6 (the high nibble of the pair's byte)
        let mut b = vec![0u8; iq3_s::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(0.5));
        (b[2 + 8], b[2 + 9]) = (0x05, 0xf0);
        b[66 + 1] = 1;
        b[74 + 4] = 0b0011_0010;
        b[106] = 6 << 4;
        iq3_s::dequantize(&b, &mut y);
        for j in 0..8 {
            let s = if 0b0011_0010u8 & (1 << j) != 0 { -1.0 } else { 1.0 };
            let g = if j < 4 { IQ3S_GRID[0x105] >> (8 * j) } else { IQ3S_GRID[0x0f0] >> (8 * (j - 4)) };
            assert_eq!(y[32 + j], 0.5 * 13.0 * (g & 255) as f32 * s, "IQ3_S weight {j}");
        }
        // IQ1_S: sub-block 3's group 1 the entry 0x5c3 (three high bits from qh), scale 5, the offset negative
        let mut b = vec![0u8; iq1_s::BYTES_PER_BLOCK];
        b[..2].copy_from_slice(&f16(0.5));
        b[2 + 4 * 3 + 1] = 0xc3;
        b[34 + 6..34 + 8].copy_from_slice(&((5u16 << 3) | (5 << 12) | 0x8000).to_le_bytes());
        iq1_s::dequantize(&b, &mut y);
        for j in 0..8 {
            let g = ((IQ1S_GRID[0x5c3] >> (8 * j)) & 255) as u8 as i8 as f32;
            assert_eq!(y[96 + 8 + j], 0.5 * 11.0 * (g - 0.125), "IQ1_S weight {j}");
        }
        // IQ1_M: the block's scale 0.75 spread over the four u16s' top nibbles; sub-block 2's group 3 the entry
        // 0x2e1 with a negative offset, its second scale 4
        let mut b = vec![0u8; iq1_m::BYTES_PER_BLOCK];
        let bits = half::f16::from_f32(0.75).to_bits();
        let mut sc = [(bits & 15) << 12, ((bits >> 4) & 15) << 12, ((bits >> 8) & 15) << 12, (bits >> 12) << 12];
        sc[1] |= 4 << 3;
        for (i, v) in sc.iter().enumerate() {
            b[48 + 2 * i..50 + 2 * i].copy_from_slice(&v.to_le_bytes());
        }
        b[4 * 2 + 3] = 0xe1;
        b[32 + 2 * 2 + 1] = (2 | 8) << 4;
        iq1_m::dequantize(&b, &mut y);
        for j in 0..8 {
            let g = ((IQ1S_GRID[0x2e1] >> (8 * j)) & 255) as u8 as i8 as f32;
            assert_eq!(y[64 + 24 + j], 0.75 * 9.0 * (g - 0.125), "IQ1_M weight {j}");
        }
    }
}
