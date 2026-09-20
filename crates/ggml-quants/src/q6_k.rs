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

// VENDORED-LOCAL: GLM-5.3-Flash. Fused Q6_K x f32 dot products.
//
// The companion to `q4_k::dot_rows`, and needed for the same reason: GLM-5.3-Flash
// quantises `ffn_down_exps` at Q6_K on 18 of its 42 MoE layers, so a third of the
// expert bytes on the CPU path come through here.
//
// **Bit-identical to `dequantize` then a sequential dot**, which takes a little
// care because `dequantize_block` writes a block out of order: within each
// 128-output half it emits index `l`, `l+32`, `l+64` and `l+96` together, from one
// `ql`/`qh` byte. Summing in index order therefore means four passes over the same
// 32 bytes -- q1 for `l` in 0..32, then q2, then q3, then q4 -- rather than one
// interleaved pass. Four passes over 32 bytes is nothing; the bytes are in L1
// after the first.
const ROWS_AT_ONCE: usize = 4;

/// One row of a `[n_rows, k]` Q6_K matrix dotted with `x`, in index order.
pub fn dot_row(row: &[u8], x: &[f32]) -> f32 {
    debug_assert_eq!(row.len() % BYTES_PER_BLOCK, 0);
    debug_assert_eq!(x.len(), row.len() / BYTES_PER_BLOCK * BLOCK_SIZE);

    let mut acc = 0.0f32;
    let mut xi = 0usize;
    for block in row.chunks_exact(BYTES_PER_BLOCK) {
        let d = read_f16(&block[208..210]);
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;

        for _ in 0..2 {
            let ql = &block[ql_off..ql_off + 64];
            let qh = &block[128 + qh_off..128 + qh_off + 32];
            let sc = &block[192 + sc_off..192 + sc_off + 8];

            // The four quarters, in the order their outputs are indexed.
            for quarter in 0..4usize {
                // shift into qh, and which 32 ql bytes / which scale pair
                let (shift, ql_base, sc_base) = match quarter {
                    0 => (0u32, 0usize, 0usize),
                    1 => (2, 32, 2),
                    2 => (4, 0, 4),
                    _ => (6, 32, 6),
                };
                let high_nibble = quarter >= 2;
                for l in 0..32usize {
                    let low = if high_nibble {
                        ql[ql_base + l] >> 4
                    } else {
                        ql[ql_base + l] & 0x0F
                    };
                    let q = (low | (((qh[l] >> shift) & 0x3) << 4)) as i8 - 32;
                    let s = sc[sc_base + l / 16] as f32;
                    acc += d * s * q as f32 * x[xi + l];
                }
                xi += 32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
    acc
}

/// `out[r] = dot(W[r], x)` for a `[n_rows, k]` Q6_K matrix, row-major.
pub fn dot_rows(w: &[u8], x: &[f32], n_rows: usize, k: usize, out: &mut [f32]) {
    let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
    debug_assert_eq!(k % BLOCK_SIZE, 0);
    debug_assert_eq!(w.len(), n_rows * row_bytes);
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(out.len(), n_rows);

    // Groups of ROWS_AT_ONCE so their accumulator chains interleave; each row is
    // still summed strictly in index order.
    let mut r = 0usize;
    while r + ROWS_AT_ONCE <= n_rows {
        for i in 0..ROWS_AT_ONCE {
            out[r + i] = dot_row(&w[(r + i) * row_bytes..(r + i + 1) * row_bytes], x);
        }
        r += ROWS_AT_ONCE;
    }
    for rr in r..n_rows {
        out[rr] = dot_row(&w[rr * row_bytes..(rr + 1) * row_bytes], x);
    }
}

#[cfg(test)]
mod fused_dot_tests {
    use super::*;

    fn bytes(n_blocks: usize, seed: u64) -> Vec<u8> {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (z >> 33) as u8
        };
        let mut v = Vec::with_capacity(n_blocks * BYTES_PER_BLOCK);
        for _ in 0..n_blocks {
            for _ in 0..192 {
                v.push(next());
            }
            // signed scales, kept modest so products stay finite
            for _ in 0..16 {
                v.push((next() % 63) as u8);
            }
            v.extend_from_slice(&half::f16::from_f32(0.02 + (next() % 11) as f32 / 500.0).to_bits().to_le_bytes());
        }
        v
    }

    fn xs(k: usize, seed: u64) -> Vec<f32> {
        let mut z = seed;
        (0..k)
            .map(|_| {
                z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((z >> 40) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect()
    }

    fn oracle(row: &[u8], x: &[f32]) -> f32 {
        let mut w = vec![0.0f32; row.len() / BYTES_PER_BLOCK * BLOCK_SIZE];
        dequantize(row, &mut w);
        let mut acc = 0.0f32;
        for (a, b) in w.iter().zip(x) {
            acc += a * b;
        }
        acc
    }

    /// Bit-identical, including the out-of-order write pattern this has to undo.
    #[test]
    fn fused_dot_is_bit_identical_to_dequantize_then_dot() {
        for &k in &[256usize, 512, 2048] {
            let x = xs(k, 0xC0DE);
            for seed in 0..6u64 {
                let row = bytes(k / BLOCK_SIZE, 0x4321 + seed);
                let want = oracle(&row, &x);
                let got = dot_row(&row, &x);
                assert_eq!(want.to_bits(), got.to_bits(), "k={k} seed={seed}: {want} vs {got}");
                assert!(want.abs() > 1e-6, "k={k} seed={seed}: oracle is zero");
            }
        }
    }

    #[test]
    fn dot_rows_matches_dot_row_exactly() {
        let k = 512usize;
        let n_rows = 3 * ROWS_AT_ONCE + 2;
        let x = xs(k, 0xF00D);
        let mut w = Vec::new();
        for r in 0..n_rows {
            w.extend_from_slice(&bytes(k / BLOCK_SIZE, 0xB000 + r as u64));
        }
        let row_bytes = k / BLOCK_SIZE * BYTES_PER_BLOCK;
        let mut got = vec![0.0f32; n_rows];
        dot_rows(&w, &x, n_rows, k, &mut got);
        for r in 0..n_rows {
            assert_eq!(
                oracle(&w[r * row_bytes..(r + 1) * row_bytes], &x).to_bits(),
                got[r].to_bits(),
                "row {r} differs"
            );
        }
        assert!(got.iter().any(|v| v.abs() > 1e-6));
    }
}

