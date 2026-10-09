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
    words: Vec<u32>,
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
        let (k, n) = (self.t.k, self.t.n);
        let (ktiles, ntiles, nw) = (k / 16, n / 16, self.tile_words / 2);
        let pos = positions(self.tile_words);
        let book = codebook();
        // Each thread takes a run of tile columns, all rows: its sums are its own.
        let threads = threads.min(ntiles).max(1);
        let per = ntiles.div_ceil(threads);
        let mut y = vec![0f32; m * n];
        let parts: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|th| {
                    let (pos, words) = (&pos, &self.words);
                    scope.spawn(move || {
                        let (a, b) = (th * per, ((th + 1) * per).min(ntiles));
                        let width = b.saturating_sub(a) * 16;
                        let mut acc = vec![0f32; m * width];
                        let mut w = [0f32; 256];
                        for kt in 0..ktiles {
                            for nt in a..b {
                                let tile = &words[(kt * ntiles + nt) * nw..(kt * ntiles + nt + 1) * nw];
                                for (slot, &(w0, w1, sh)) in pos.iter().enumerate() {
                                    let pair = ((tile[w0] as u64) << 32) | tile[w1] as u64;
                                    w[slot] = book[((pair >> sh) & 0xffff) as usize];
                                }
                                for row in 0..m {
                                    let xs = &xh[row * k + kt * 16..row * k + kt * 16 + 16];
                                    let out = &mut acc[row * width + (nt - a) * 16..row * width + (nt - a) * 16 + 16];
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
                    })
                })
                .collect();
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
}

// ---------------------------------------------------------------------------
// On the GPU
// ---------------------------------------------------------------------------
