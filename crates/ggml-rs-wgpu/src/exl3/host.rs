//! The format on the host: a projection's transforms and maps, the codebook and a tile's positions, and the
//! CPU's own projection (`Exl3Cpu`), which the kernels are checked against.

use super::*;

/// 1/√128: the Hadamard transforms' normalisation.
pub(super) const ISQRT128: f32 = 0.088_388_35;
/// Input rows one GPU pass multiplies: a prompt's rows decode each weight once a pass. 32 is what [`MANY`]'s two
/// rows a thread give a workgroup of 256.
pub(super) const ROWS: usize = 32;

pub(super) fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// The unnormalised Walsh-Hadamard transform of each 128-wide block, as exllamav3's `had` kernel computes it.
pub(super) fn had(x: &mut [f32]) {
    for block in x.chunks_exact_mut(128) {
        let mut s = 1;
        while s < 128 {
            for i in 0..128 {
                if i & s == 0 {
                    let (a, b) = (block[i], block[i + s]);
                    block[i] = a + b;
                    block[i + s] = a - b;
                }
            }
            s *= 2;
        }
    }
}

/// What both halves of a projection need on the host: the maps and the scales.
#[derive(Debug)]
pub(super) struct Transform {
    pub(super) k: usize,
    pub(super) n: usize,
    pub(super) suh: Vec<f32>,
    pub(super) svh: Vec<f32>,
    pub(super) input_map: Vec<u32>,
    pub(super) output_map: Vec<u32>,
}

impl Transform {
    /// The input row as the matmul takes it: mapped, scaled, transformed and rounded.
    pub(super) fn pre(&self, x: &[f32], out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = half(x[self.input_map[i] as usize]) * self.suh[i];
        }
        had(out);
        for v in out.iter_mut() {
            *v = half(*v * ISQRT128);
        }
    }

    /// The matmul's sums for one row as the caller takes them: rounded, transformed, scaled and mapped back.
    pub(super) fn post(&self, y: &mut [f32], out: &mut [f32]) {
        for v in y.iter_mut() {
            *v = half(*v);
        }
        had(y);
        for (c, v) in y.iter_mut().enumerate() {
            *v = half(*v * ISQRT128 * self.svh[c]);
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = y[self.output_map[i] as usize];
        }
    }

    /// `x` as host rows, `[m, k]`.
    pub(super) fn rows<'a>(&self, x: &'a Tensor, host: &'a mut Option<Tensor>) -> (&'a [f32], usize, Vec<usize>) {
        if x.is_device() {
            *host = Some(x.to_host());
        }
        let host: &'a Option<Tensor> = host;
        let x = host.as_ref().unwrap_or(x);
        let m = x.numel() / self.k;
        let mut shape = x.shape().to_vec();
        *shape.last_mut().expect("x has a last axis") = self.n;
        (x.data(), m, shape)
    }
}

/// Where each of a 16×16 tile's 256 weights starts in its tile's bit stream, for a bitrate: `(first word, second word,
/// right shift of the pair)`, indexed by `row * 16 + column`.
pub(super) fn positions(tile_words: usize) -> Vec<(usize, usize, u32)> {
    let nw = tile_words / 2;
    (0..256)
        .map(|t| {
            let (r, c) = (t / 16, t % 16);
            let lane = (r % 8 / 2) + 4 * (c % 8);
            let j = (r % 2) + 2 * (r / 8) + 4 * (c / 8);
            let i = lane * 8 + j;
            let end = (i + 1) * (tile_words / 16) + if tile_words % 16 == 8 { i.div_ceil(2) } else { 0 };
            let start = (end + nw * 32 - 16) % (nw * 32);
            (start / 32, (start / 32 + 1) % nw, (48 - start % 32) as u32)
        })
        .collect()
}

/// mul1's decode of a 16-bit code, as `ggml_rs::exl3::mul1` (in the closed form the CUDA kernel uses).
#[inline]
pub(super) fn decode(code: u32) -> f32 {
    let x = code.wrapping_mul(0x83dc_d12d);
    let sum = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    half((1024 + sum) as f32 * (1774.0 / 262_144.0) - 10.382_812_5)
}

/// Every code's weight, decoded once: the CPU's matmul looks each one up (256 KB, in cache), where decoding (its f16
/// rounding in software) was most of a CPU-resident expert's time.
fn codebook() -> &'static [f32] {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..65536).map(decode).collect())
}

// ---------------------------------------------------------------------------
// On the CPU
// ---------------------------------------------------------------------------

/// An EXL3 projection decoded on the CPU, tile by tile, every call: for a weight beyond the GPU budget, or a computer
/// without a GPU. Slow (each weight is decoded for each row), but the model still runs.
#[derive(Debug)]
pub struct Exl3Cpu {
    pub(super) t: Transform,
    pub(super) words: Vec<u32>,
    tile_words: usize,
    shape: [usize; 2],
}

impl Exl3Cpu {
    pub fn new(data: Exl3Data) -> Result<Self, String> {
        data.validate()?;
        let (k, n) = (data.suh.len(), data.svh.len());
        Ok(Self {
            shape: [n, k],
            tile_words: data.tile_words,
            words: data.words,
            t: Transform { k, n, suh: data.suh, svh: data.svh, input_map: data.input_map, output_map: data.output_map },
        })
    }
}

impl PackedLinear for Exl3Cpu {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.words.len() * 4 + (self.t.k + self.t.n) * 8
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        let (k, n) = (self.t.k, self.t.n);
        let mut host = None;
        let (data, m, shape) = self.t.rows(x, &mut host);
        let mut xh = vec![0f32; m * k];
        for (row, out) in data.chunks_exact(k).zip(xh.chunks_exact_mut(k)) {
            self.t.pre(row, out);
        }
        let mut y = self.matmul(&xh, m, std::thread::available_parallelism().map_or(4, |n| n.get()));
        let mut out = vec![0f32; m * n];
        for (yr, or) in y.chunks_exact_mut(n).zip(out.chunks_exact_mut(n)) {
            self.t.post(yr, or);
        }
        Tensor::from_vec(out, shape)
    }
}

impl Exl3Cpu {
    /// The matmul's sums `[m, n]` for `m` prepared rows, on up to `threads` threads.
    pub(super) fn matmul(&self, xh: &[f32], m: usize, threads: usize) -> Vec<f32> {
        tiles_matmul(&self.words[..], self.t.k, self.t.n, self.tile_words, xh, m, threads)
    }
}

/// A projection's packed words where they lie: decoded (`[u32]`), or as a record's little-endian bytes ([`LeWords`]).
pub(super) trait Words: Sync {
    fn word(&self, i: usize) -> u32;
}

impl Words for [u32] {
    #[inline(always)]
    fn word(&self, i: usize) -> u32 {
        self[i]
    }
}

/// Words as a file stores them: four little-endian bytes each.
pub(super) struct LeWords<'a>(pub(super) &'a [u8]);

impl Words for LeWords<'_> {
    #[inline(always)]
    fn word(&self, i: usize) -> u32 {
        u32::from_le_bytes([self.0[4 * i], self.0[4 * i + 1], self.0[4 * i + 2], self.0[4 * i + 3]])
    }
}

/// The matmul's sums `[m, n]` of a `k -> n` projection's packed `words` for `m` prepared rows, on up to `threads`
/// threads, each a run of tile columns.
pub(super) fn tiles_matmul<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], m: usize, threads: usize) -> Vec<f32> {
    let ntiles = n / 16;
    // Each thread takes a run of tile columns, all rows: its sums are its own.
    let threads = threads.min(ntiles).max(1);
    let per = ntiles.div_ceil(threads);
    let mut y = vec![0f32; m * n];
    let parts: Vec<Vec<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads).map(|th| scope.spawn(move || tile_columns(words, k, n, tile_words, xh, m, th * per..((th + 1) * per).min(ntiles)))).collect();
        handles.into_iter().map(|h| h.join().expect("an EXL3 CPU worker panicked")).collect()
    });
    for (th, acc) in parts.into_iter().enumerate() {
        let a = th * per * 16;
        let width = acc.len() / m.max(1);
        // A thread past the last tile column (40 columns over 32 threads leaves the last ones none) has no sums,
        // and its offset is past the row's end.
        if width == 0 {
            continue;
        }
        for row in 0..m {
            y[row * n + a..row * n + a + width].copy_from_slice(&acc[row * width..(row + 1) * width]);
        }
    }
    y
}

/// mul1's value of a code by the byte sum of its product (`decode`'s, which depends on nothing else: 1,021 values,
/// 4 KB, where every code's table is 256 KB).
fn by_bytesum() -> &'static [f32; 1024] {
    static TABLE: std::sync::OnceLock<[f32; 1024]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|sum| half((1024 + sum as u32) as f32 * (1774.0 / 262_144.0) - 10.382_812_5)))
}

/// The byte sum of a code's product with mul1's multiplier (two bytes at a time, then the halves).
#[inline(always)]
fn bytesum(code: u32) -> usize {
    let x = code.wrapping_mul(0x83dc_d12d);
    let pairs = (x & 0x00ff_00ff) + ((x >> 8) & 0x00ff_00ff);
    ((pairs & 0xffff) + (pairs >> 16)) as usize
}

/// The sums of tile columns `cols` of a `k -> n` projection's packed `words` for `m` prepared rows: `[m, 16 cols]`,
/// each tile decoded once for every row.
pub(super) fn tile_columns<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], m: usize, cols: std::ops::Range<usize>) -> Vec<f32> {
    // (a whole number of bits a weight, up to 6: each of a tile's 32 lanes' eight codes from one 64-bit window; on Arm
    // four codes a vector. OAIY_EXL3_CPU: `table` for every code's table, `lanes` for the lanes without vectors.)
    static ASKED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let asked = ASKED.get_or_init(|| std::env::var("OAIY_EXL3_CPU").ok()).as_deref();
    if tile_words % 16 == 0 && (16..=96).contains(&tile_words) && asked != Some("table") {
        #[cfg(target_arch = "aarch64")]
        if asked != Some("lanes") {
            // SAFETY: NEON is part of every AArch64 target this is built for (the function asks for nothing else).
            return unsafe { neon::tile_columns(words, k, n, tile_words, xh, m, cols) };
        }
        return tile_columns_lanes(words, k, n, tile_words, xh, m, cols);
    }
    let (ktiles, ntiles, nw) = (k / 16, n / 16, tile_words / 2);
    let pos = positions_of(tile_words);
    let book = codebook();
    let width = cols.len() * 16;
    let mut acc = vec![0f32; m * width];
    let mut w = [0f32; 256];
    // a tile's words, read once (a weight's code spans two of them)
    let mut tile = [0u32; 64];
    for kt in 0..ktiles {
        for nt in cols.clone() {
            let base = (kt * ntiles + nt) * nw;
            for (i, t) in tile[..nw].iter_mut().enumerate() {
                *t = words.word(base + i);
            }
            for (slot, &(w0, w1, sh)) in pos.iter().enumerate() {
                let pair = ((tile[w0] as u64) << 32) | tile[w1] as u64;
                w[slot] = book[((pair >> sh) & 0xffff) as usize];
            }
            for row in 0..m {
                let xs = &xh[row * k + kt * 16..row * k + kt * 16 + 16];
                let at = row * width + (nt - cols.start) * 16;
                let out = &mut acc[at..at + 16];
                for r in 0..16 {
                    let xv = xs[r];
                    for c in 0..16 {
                        out[c] += xv * w[r * 16 + c];
                    }
                }
            }
        }
    }
    acc
}

/// [`tile_columns`] of a whole number of bits a weight (`b`, 1 to 6): a tile's weights in its bit stream go 32 lanes of
/// eight, lane `l`'s codes `j` at bits `b (8 l + j + 1) - 16` on (16 bits, the stream read round), weight `(r, c)` of
/// the tile with `r` = `2 (l % 4) + (j % 2) + 8 ((j / 2) % 2)` and `c` = `l / 4 + 8 (j / 4)` (as [`positions`] lays
/// them). A lane's eight are one 64-bit window's, each a shift by a constant, its value by its product's byte sum
/// ([`by_bytesum`]): the same values as [`decode`]'s, in the same sums' order.
fn tile_columns_lanes<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], m: usize, cols: std::ops::Range<usize>) -> Vec<f32> {
    let (ktiles, ntiles, nw, b) = (k / 16, n / 16, tile_words / 2, tile_words / 16);
    let bits = nw * 32;
    let book = by_bytesum();
    let width = cols.len() * 16;
    let mut acc = vec![0f32; m * width];
    let mut w = [0f32; 256];
    let mut tile = [0u32; 64];
    // each lane's window: its first word and the shift into it (the stream read round)
    let starts: Vec<(usize, u32)> = (0..32).map(|l| {
        let first = (b * (8 * l + 1) + bits - 16) % bits;
        (first / 32, (first % 32) as u32)
    }).collect();
    for kt in 0..ktiles {
        for nt in cols.clone() {
            let base = (kt * ntiles + nt) * nw;
            for (i, t) in tile[..nw].iter_mut().enumerate() {
                *t = words.word(base + i);
            }
            for (l, &(q, sh)) in starts.iter().enumerate() {
                let (a, bb, c) = (tile[q] as u64, tile[(q + 1) % nw] as u64, tile[(q + 2) % nw] as u64);
                // 64 bits of the stream from the lane's first code's first bit
                let win = ((a << 32 | bb) << sh) | ((c << sh) >> 32);
                let (r0, c0) = (2 * (l % 4), l / 4);
                for j in 0..8 {
                    let code = ((win >> (48 - b * j)) & 0xffff) as u32;
                    let (r, c) = (r0 + (j % 2) + 8 * ((j / 2) % 2), c0 + 8 * (j / 4));
                    w[r * 16 + c] = book[bytesum(code)];
                }
            }
            for row in 0..m {
                let xs = &xh[row * k + kt * 16..row * k + kt * 16 + 16];
                let at = row * width + (nt - cols.start) * 16;
                let out = &mut acc[at..at + 16];
                for r in 0..16 {
                    let xv = xs[r];
                    for c in 0..16 {
                        out[c] += xv * w[r * 16 + c];
                    }
                }
            }
        }
    }
    acc
}

/// [`tile_columns_lanes`] on Arm's vectors: a lane's eight codes are four shifts of its window (two codes each), their
/// values four at a time in the closed form ([`decode`]'s: the product's bytes summed in pairs, the one rounding to f16
/// the conversion's, to nearest even as `half` rounds), and a lane's two quads of values (rows `2a, 2a + 1, 2a + 8, 2a
/// + 9` of columns `l / 4` and `l / 4 + 8`) each multiplied by the row's same four inputs. An output's sum is kept as
/// four partial sums (one a lane of the quads), added up at the end: another order than the lanes without vectors
/// add in, the same for a row whatever rows are beside it.
#[cfg(target_arch = "aarch64")]
pub(super) mod neon {
    use super::Words;
    use std::arch::aarch64::*;

    #[inline(always)]
    fn load(s: &[f32]) -> float32x4_t {
        let s: &[f32; 4] = s[..4].try_into().expect("four floats");
        // SAFETY: `s` is four floats.
        unsafe { vld1q_f32(s.as_ptr()) }
    }

    #[inline(always)]
    fn store(v: float32x4_t, s: &mut [f32]) {
        let s: &mut [f32; 4] = (&mut s[..4]).try_into().expect("four floats");
        // SAFETY: `s` is four floats.
        unsafe { vst1q_f32(s.as_mut_ptr(), v) }
    }

    /// mul1's values of four codes.
    #[inline]
    #[target_feature(enable = "neon")]
    fn values(codes: uint32x4_t) -> float32x4_t {
        let x = vmulq_u32(codes, vdupq_n_u32(0x83dc_d12d));
        let sum = vpaddlq_u16(vpaddlq_u8(vreinterpretq_u8_u32(x)));
        let f = vcvtq_f32_u32(vaddq_u32(sum, vdupq_n_u32(1024)));
        // (the product and the difference are both exact in f32, so fused or not they are the same)
        let v = vfmaq_f32(vdupq_n_f32(-10.382_812_5), f, vdupq_n_f32(1774.0 / 262_144.0));
        vcvt_f32_f16(vcvt_f16_f32(v))
    }

    /// [`values`] of four codes, for a test.
    #[cfg(test)]
    pub(in super::super) fn values_of(codes: [u32; 4]) -> [f32; 4] {
        let mut out = [0f32; 4];
        // SAFETY: NEON is part of every AArch64 target; the load and the store are of four values each.
        unsafe { vst1q_f32(out.as_mut_ptr(), values(vld1q_u32(codes.as_ptr()))) };
        out
    }

    #[target_feature(enable = "neon")]
    pub(super) fn tile_columns<W: Words + ?Sized>(words: &W, k: usize, n: usize, tile_words: usize, xh: &[f32], m: usize, cols: std::ops::Range<usize>) -> Vec<f32> {
        let (ktiles, ntiles, nw, b) = (k / 16, n / 16, tile_words / 2, (tile_words / 16) as i64);
        let bits = nw * 32;
        let width = cols.len() * 16;
        // each row's inputs as the lanes take them: a tile's sixteen as four quads, quad `a` rows 2a, 2a + 1, 2a + 8, 2a + 9
        let mut xq = vec![0f32; m * k];
        for row in 0..m {
            for kt in 0..ktiles {
                let xs = &xh[row * k + kt * 16..row * k + kt * 16 + 16];
                let q = &mut xq[(row * ktiles + kt) * 16..(row * ktiles + kt) * 16 + 16];
                for a in 0..4 {
                    q[4 * a..4 * a + 4].copy_from_slice(&[xs[2 * a], xs[2 * a + 1], xs[2 * a + 8], xs[2 * a + 9]]);
                }
            }
        }
        // each lane's window: its three words and the shift into the first (the stream read round)
        let starts: [(usize, usize, usize, u32); 32] = std::array::from_fn(|l| {
            let first = (b as usize * (8 * l + 1) + bits - 16) % bits;
            (first / 32, (first / 32 + 1) % nw, (first / 32 + 2) % nw, (first % 32) as u32)
        });
        // codes j and j + 1 of a window: right shifts by 48 - b j
        let shifts: [int64x2_t; 4] = std::array::from_fn(|p| vcombine_s64(vdup_n_s64(-(48 - b * (2 * p as i64))), vdup_n_s64(-(48 - b * (2 * p as i64 + 1)))));
        let mask = vdupq_n_u32(0xffff);
        // four partial sums an output
        let mut acc = vec![0f32; m * width * 4];
        let mut tile = [0u32; 64];
        let zero = vdupq_n_f32(0.0);
        let (mut lo, mut hi) = ([zero; 32], [zero; 32]);
        for kt in 0..ktiles {
            for nt in cols.clone() {
                let base = (kt * ntiles + nt) * nw;
                for (i, t) in tile[..nw].iter_mut().enumerate() {
                    *t = words.word(base + i);
                }
                for (l, &(q0, q1, q2, sh)) in starts.iter().enumerate() {
                    let (a, bb, c) = (tile[q0] as u64, tile[q1] as u64, tile[q2] as u64);
                    let win = vdupq_n_u64(((a << 32 | bb) << sh) | ((c << sh) >> 32));
                    let code = |p: usize| vmovn_u64(vshlq_u64(win, shifts[p]));
                    lo[l] = values(vandq_u32(vcombine_u32(code(0), code(1)), mask));
                    hi[l] = values(vandq_u32(vcombine_u32(code(2), code(3)), mask));
                }
                for row in 0..m {
                    let q = &xq[(row * ktiles + kt) * 16..(row * ktiles + kt) * 16 + 16];
                    let x4 = [load(&q[0..]), load(&q[4..]), load(&q[8..]), load(&q[12..])];
                    let at = (row * width + (nt - cols.start) * 16) * 4;
                    let out = &mut acc[at..at + 64];
                    for c in 0..8 {
                        let (mut left, mut right) = (load(&out[4 * c..]), load(&out[4 * (c + 8)..]));
                        for (a, &x) in x4.iter().enumerate() {
                            left = vfmaq_f32(left, lo[4 * c + a], x);
                            right = vfmaq_f32(right, hi[4 * c + a], x);
                        }
                        store(left, &mut out[4 * c..]);
                        store(right, &mut out[4 * (c + 8)..]);
                    }
                }
            }
        }
        acc.chunks_exact(4).map(|p| (p[0] + p[1]) + (p[2] + p[3])).collect()
    }
}

/// [`positions`] of each bitrate, made once.
fn positions_of(tile_words: usize) -> &'static [(usize, usize, u32)] {
    static MADE: std::sync::OnceLock<Vec<Vec<(usize, usize, u32)>>> = std::sync::OnceLock::new();
    &MADE.get_or_init(|| (0..=128).map(|tw| if tw % 8 == 0 && tw >= 16 { positions(tw) } else { Vec::new() }).collect())[tile_words]
}

// ---------------------------------------------------------------------------
// On the GPU
// ---------------------------------------------------------------------------
