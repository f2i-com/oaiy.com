//! EXL3 projections (exllamav3's mul1 trellis, as OrcaSAQ ships them) without CUDA: the packed weights on the GPU
//! through WebGPU, decoded inside the matmul, or on the CPU beyond the budget or without a GPU.
//!
//! A projection is `y = U · (W · (V · x))`: the input mapped and scaled by `suh`, a Hadamard-128 transform, the
//! trellis-decoded matrix, a second Hadamard-128 transform, then scaled by `svh` and mapped back, each step rounded to
//! f16 where exllamav3 rounds. The two transforms and the maps are a few thousand operations a row and run on the host,
//! as the rest of this backend's activations do; the matmul, which reads every packed weight, runs in WGSL
//! ([`shader`]: one kernel for a decode step's single row, one for a prompt's many), or tile by tile on the CPU. Both decode a weight as `ggml_rs::exl3::Exl3Data::value` does, to the bit:
//! `f16((1024 + bytesum(code · 0x83dcd12d)) · 1774/2^18 − 10.3828125)`, whose product is exact in f32, so the one
//! rounding (round to nearest even, by hand) is the reference's.

use crate::{chunk_limit, Gpu, WgpuBackend};
use ggml_rs::exl3::{Exl3Data, PackedLinear};
use ggml_rs::{DeviceChain, DeviceVec, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 1/√128: the Hadamard transforms' normalisation.
const ISQRT128: f32 = 0.088_388_35;
/// Input rows one GPU pass multiplies: a prompt's rows decode each weight once a pass. 32 is what [`MANY`]'s two
/// rows a thread give a workgroup of 256.
const ROWS: usize = 32;

fn half(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// The unnormalised Walsh-Hadamard transform of each 128-wide block, as exllamav3's `had` kernel computes it.
fn had(x: &mut [f32]) {
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
struct Transform {
    k: usize,
    n: usize,
    suh: Vec<f32>,
    svh: Vec<f32>,
    input_map: Vec<u32>,
    output_map: Vec<u32>,
}

impl Transform {
    /// The input row as the matmul takes it: mapped, scaled, transformed and rounded.
    fn pre(&self, x: &[f32], out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = half(x[self.input_map[i] as usize]) * self.suh[i];
        }
        had(out);
        for v in out.iter_mut() {
            *v = half(*v * ISQRT128);
        }
    }

    /// The matmul's sums for one row as the caller takes them: rounded, transformed, scaled and mapped back.
    fn post(&self, y: &mut [f32], out: &mut [f32]) {
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
    fn rows<'a>(&self, x: &'a Tensor, host: &'a mut Option<Tensor>) -> (&'a [f32], usize, Vec<usize>) {
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
fn positions(tile_words: usize) -> Vec<(usize, usize, u32)> {
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
fn decode(code: u32) -> f32 {
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
    t: Transform,
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
    fn matmul(&self, xh: &[f32], m: usize, threads: usize) -> Vec<f32> {
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

/// The parts both kernels share: the parameters, the bindings, and where a thread's weight starts in its tile.
/// A one-row kernel's tile loop as a warp decodes it: lane `l` of a warp holds codes `8l..8l + 8` of a tile (the
/// trellis' order), which are weights `(r, c)` for `r` in `2 (l % 4) + {0, 1, 8, 9}` and `c` in `l / 4 + {0, 8}`; it
/// loads the four words its codes lie in and its four inputs once a tile, and sums its two columns' products (in
/// code order). A workgroup's eight warps take a split's tiles in turn, and column `c`'s sum is its four lanes' in
/// each warp, the warps in order. `x_at` is the WGSL of the input's index for row `r` of tile `kt`, `words_at` of a
/// tile's first word, `out` of the column's sum's place (`total` the sum, `c` the column).
fn one_lanes(x_at: &str, words_at: &str, out: &str) -> String {
    one_lanes_with(x_at, words_at, out, None)
}

/// The rates a tile may be written at, in 16-bit words a tile: 1 to 4 bits a weight by halves, and 5 to 8.
pub(crate) const RATES: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

/// What differs between [`one_lanes`]' forms ([`one_lanes_with`]): how a thread writes its table entry, what a lane
/// reads of the table, its loads of a tile's words, and its eight codes.
struct Lanes {
    place: String,
    held: String,
    loads: String,
    codes: Vec<String>,
}

fn lanes_general() -> Lanes {
    Lanes {
        place: "    places[t] = place(t, tw, first);\n".into(),
        held: "    var at: array<u32, 8>;\n".to_string() + &(0..8).map(|jj| format!("    at[{jj}] = places[8u * l + {jj}u];\n")).collect::<String>(),
        loads: (0..4).map(|i| format!("        let q{i} = words[tile + {}];\n", ["w0", "o1", "o2", "o3"][i])).collect(),
        codes: (0..8).map(|jj| format!("code_in(q0, q1, q2, q3, at[{jj}])")).collect(),
    }
}

/// Where a lane's code `j` starts after its first, in bits, at `tw` words a tile (as `window` places them).
fn code_offset(tw: usize, j: usize) -> usize {
    j * (tw / 16) + if tw % 16 == 8 { j.div_ceil(2) } else { 0 }
}

/// The furthest into its word any lane's first code starts, in bits, at `tw` words a tile.
fn furthest_start(tw: usize) -> usize {
    let total = tw * 16;
    (0..32).map(|l| ((8 * l + 1) * (tw / 16) + if tw % 16 == 8 { (8 * l + 1) / 2 } else { 0 } + total - 16) % total % 32).max().unwrap()
}

/// A lane's loop for tiles of `tw` words alone, each code's place written into the kernel. A lane's code `j` starts
/// `j b` bits after its first (and `(j + 1) / 2` more at a half rate), whatever the lane; only where the first code
/// starts in its word differs between lanes. So a lane shifts its words to its first code once a tile (`a0`, `a1`,
/// `a2`: the stream's bits 0, 32 and 64 on; `m0`, `m1`: bits 16 and 48 on), and a code is one of those shifted by a
/// constant, where the general loop picks each code's view of three by two selects and shifts it by a number it read.
/// It loads only the words some lane's codes reach (two at 3 and 4 bits a weight, where the general loop loads four).
/// The same codes, so the same sums.
fn lanes_fixed(tw: usize) -> Lanes {
    // code j's view (0: a0, 1: m0, 2: a1, 3: m1, 4: a2, 32 bits of the lane's stream from 16 times its number) and
    // how far the code is above the view's low end
    let at: Vec<(usize, usize)> = (0..8)
        .map(|j| {
            let f = code_offset(tw, j);
            let view = if f <= 16 { 0 } else { (f - 1) / 16 };
            (view, 16 * view + 16 - f)
        })
        .collect();
    let last = at.iter().map(|&(view, _)| view).max().unwrap();
    assert!(last <= 4, "a lane's codes lie within 96 bits of its first");
    // the words some lane reaches, and the aligned words the views are made of
    let words = (furthest_start(tw) + code_offset(tw, 7) + 16).div_ceil(32);
    let aligned = last.div_ceil(2) + 1;
    assert!((aligned..=4).contains(&words), "a lane's views lie in the words it loads");
    let mut loads: String = (0..words).map(|i| format!("        let q{i} = words[tile + {}];\n", ["w0", "o1", "o2", "o3"][i])).collect();
    for i in 0..aligned {
        // (a shift of 32 is no shift at all, so the next word's part goes down by one and then by `31 - into`)
        loads += &if i + 1 < words { format!("        let a{i} = (q{i} << into) | ((q{} >> 1u) >> rest);\n", i + 1) } else { format!("        let a{i} = q{i} << into;\n") };
    }
    for i in 0..2 {
        if at.iter().any(|&(view, _)| view == 2 * i + 1) {
            loads += &format!("        let m{i} = (a{i} << 16u) | (a{} >> 16u);\n", i + 1);
        }
    }
    Lanes {
        place: "    places[t] = 48u - window(t, tw).y;\n".into(),
        held: "    let into = places[8u * l];\n    let rest = 31u - into;\n".into(),
        loads,
        codes: at.iter().map(|&(view, shift)| format!("{} >> {shift}u", ["a0", "m0", "a1", "m1", "a2"][view])).collect(),
    }
}

fn one_lanes_with(x_at: &str, words_at: &str, out: &str, fixed: Option<usize>) -> String {
    let Lanes { place, held, loads, codes } = fixed.map_or_else(lanes_general, lanes_fixed);
    let codes: String = codes
        .iter()
        .enumerate()
        .map(|(jj, code)| {
            let acc = if jj < 4 { "lo" } else { "hi" };
            format!("            {acc} = {acc} + xv[{}] * decode_at({code});\n", jj % 4)
        })
        .collect();
    format!(
        r#"
var<workgroup> red: array<vec2<f32>, 256>;
// every code's place in a tile (`place`; in a kernel written for one rate, how far into its word it starts, read of
// a lane's first code alone), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
var<workgroup> firsts: array<u32, 32>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each: a weight is then a read where it
// was a conversion, a product and a rounding to f16
var<workgroup> values: array<f32, 1024>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

// The 16-bit window of code `i` of a tile of `tw` words: (its first word, its shift) as `weight` finds them.
fn window(i: u32, tw: u32) -> vec2<u32> {{
    let nw = tw / 2u;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec2<u32>(start / 32u, 48u - start % 32u);
}}

// Code `i`'s view and its shift, packed, for a lane whose first word is `w0`: which 32 bits of the lane's four
// words its window lies in (view 2 d: word d; view 2 d + 1: word d's low half and the next's high half), and how
// far the code is above the view's low end. A lane's eight codes start within 96 bits of its first word (8 bits
// a weight at most), so views 0 to 5.
fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    let o = 32u * ((wd.x + nw - w0) % nw) + (48u - wd.y);
    return (o >> 4u) | ((16u - (o & 15u)) << 8u);
}}

// A code from the four words `q0..q3` (from a lane's first) at its packed place: its view, shifted (its low 16
// bits: `decode_at` takes no more). The straddling views are the same for a lane's eight codes.
fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let odd = (at & 1u) == 1u;
    let p0 = select(q0, (q0 << 16u) | (q1 >> 16u), odd);
    let p1 = select(q1, (q1 << 16u) | (q2 >> 16u), odd);
    let p2 = select(q2, (q2 << 16u) | (q3 >> 16u), odd);
    let hi = (at >> 1u) & 3u;
    return select(select(p0, p1, hi == 1u), p2, hi == 2u) >> (at >> 8u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    return values[dot4U8Packed(hx, 0x01010101u)];
}}

fn lanes(t: u32, nt: u32, ntiles: u32, ks: u32, ke: u32, tw: u32, base: u32) -> vec2<f32> {{
    let warp = t / 32u;
    let l = t % 32u;
    let nw = tw / 2u;
    let rb = 2u * (l % 4u);
    let first = window(8u * (t / 8u), tw).x;
{place}    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
{held}    var lo = 0.0;
    var hi = 0.0;
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = {words_at};
{loads}        let xv = array<f32, 4>(x[{x0}], x[{x1}], x[{x8}], x[{x9}]);
        {{
{codes}        }}
    }}
    return vec2<f32>(lo, hi);
}}

// Column `c`'s sum of a workgroup's lanes (`red`, after its barrier): each warp's four lanes of it, the warps in order.
fn column(c: u32) -> f32 {{
    let half = c / 8u;
    let cl = c % 8u;
    var total = 0.0;
    for (var w = 0u; w < 8u; w = w + 1u) {{
        for (var q = 0u; q < 4u; q = q + 1u) {{
            let v = red[w * 32u + cl * 4u + q];
            total = total + select(v.x, v.y, half == 1u);
        }}
    }}
    return total;
}}
"#,
        x0 = x_at.replace("{r}", "rb"),
        x1 = x_at.replace("{r}", "rb + 1u"),
        x8 = x_at.replace("{r}", "rb + 8u"),
        x9 = x_at.replace("{r}", "rb + 9u"),
    ) + &format!("// {out}\n")
}

const COMMON: &str = r#"
struct Params {
    n: u32,
    k: u32,
    kt0: u32,
    kts: u32,
    tw: u32,
    rows: u32,
    splits: u32,
    slot0: u32,
};
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// f32 to f16 and back, rounding to nearest even: exact for the decoded weights, which are normal f16 values.
fn round_f16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}

// Where weight (r, c)'s 16-bit code sits in a tile of `tw` words: its two words and the shift (the same in every tile).
fn code_at(r: u32, c: u32, tw: u32) -> vec3<u32> {
    let nw = tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let j = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + j;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {
        end = end + (i + 1u) / 2u;
    }
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec3<u32>(start / 32u, (start / 32u + 1u) % nw, 48u - start % 32u);
}

// The weight whose code is in words `a` and `b` at shift `sh` (`code_at`), decoded as mul1.
fn decode_at(a: u32, b: u32, sh: u32) -> f32 {
    let code = select((a << (32u - sh)) | (b >> sh), a >> (sh - 32u), sh >= 32u);
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}

"#;

/// One input row (a decode step): a workgroup of 256 threads takes one 16-wide tile column and a run of its tile rows
/// (one split), its warps the tiles in turn as [`one_lanes`] has them. Each split writes its partial sums to a slot of
/// its own, which the host adds up.
fn one_source() -> String {
    let body = one_lanes("(p.kt0 + kt) * 16u + {r}", "(kt * ntiles + nt) * nw", "");
    format!(
        r#"
struct Params {{
    n: u32,
    k: u32,
    kt0: u32,
    kts: u32,
    tw: u32,
    rows: u32,
    splits: u32,
    slot0: u32,
}};
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
{body}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let ntiles = p.n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let s = wg.z;
    let per = (p.kts + p.splits - 1u) / p.splits;
    let ks = s * per;
    let ke = min(p.kts, ks + per);
    red[t] = lanes(t, nt, ntiles, ks, ke, p.tw, 0u);
    workgroupBarrier();
    if (t < 16u) {{
        part[(p.slot0 + s) * p.n + nt * 16u + t] = column(t);
    }}
}}
"#
    )
}

/// Up to [`ROWS`] input rows (a prompt): each tile's 256 weights are decoded once into the workgroup's memory, and
/// thread `(r, c)` sums column `c` for rows `r` and `r + 16`, sixteen products a tile each. Each split writes its
/// partial sums to a slot of its own, which the host adds up.
const MANY: &str = r#"
var<workgroup> tile: array<u32, 64>;
var<workgroup> xs: array<f32, 512>;
var<workgroup> wt: array<f32, 256>;

// Weight (r, c) of the tile in `tile`: its 16-bit code from the bit stream, decoded as mul1.
fn weight(r: u32, c: u32) -> f32 {
    let nw = p.tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let j = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + j;
    var end = (i + 1u) * (p.tw / 16u);
    if (p.tw % 16u == 8u) {
        end = end + (i + 1u) / 2u;
    }
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    let w0 = start / 32u;
    let sh = 48u - start % 32u;
    let a = tile[w0];
    let b = tile[(w0 + 1u) % nw];
    var code: u32;
    if (sh >= 32u) {
        code = a >> (sh - 32u);
    } else {
        code = (a << (32u - sh)) | (b >> sh);
    }
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let ntiles = p.n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {
        return;
    }
    let s = wg.z;
    let r = t / 16u;
    let c = t % 16u;
    let nw = p.tw / 2u;
    let per = (p.kts + p.splits - 1u) / p.splits;
    let ks = s * per;
    let ke = min(p.kts, ks + per);
    let a = r < p.rows;
    let b = r + 16u < p.rows;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var kt = ks; kt < ke; kt = kt + 1u) {
        if (t < nw) {
            tile[t] = words[(kt * ntiles + nt) * nw + t];
        }
        for (var q = t; q < p.rows * 16u; q = q + 256u) {
            xs[q] = x[(q / 16u) * p.k + (p.kt0 + kt) * 16u + (q % 16u)];
        }
        workgroupBarrier();
        wt[t] = weight(r, c);
        workgroupBarrier();
        if (a) {
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                acc0 = acc0 + xs[r * 16u + kk] * wt[kk * 16u + c];
            }
        }
        if (b) {
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                acc1 = acc1 + xs[(r + 16u) * 16u + kk] * wt[kk * 16u + c];
            }
        }
        workgroupBarrier();
    }
    if (a) {
        part[((p.slot0 + s) * p.rows + r) * p.n + nt * 16u + c] = acc0;
    }
    if (b) {
        part[((p.slot0 + s) * p.rows + r + 16u) * p.n + nt * 16u + c] = acc1;
    }
}
"#;

/// The kernel for one row, and the one for several, as WGSL.
pub fn shader(many: bool) -> String {
    if many {
        format!("{COMMON}{MANY}")
    } else {
        one_source()
    }
}

/// An EXL3 projection with its packed weights on the GPU, in buffers of whole tile rows below the binding limit.
pub struct Exl3Gpu {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    t: Transform,
    tile_words: usize,
    /// `(buffer, first tile row, tile rows, splits)`.
    chunks: Vec<(wgpu::Buffer, u32, u32, u32)>,
    shape: [usize; 2],
    nbytes: usize,
    used: Arc<AtomicU64>,
    /// What a chain needs of it, made when first chained.
    chain: std::sync::OnceLock<Exl3Chain>,
}

impl std::fmt::Debug for Exl3Gpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exl3Gpu({}x{}, {} words/tile, {} chunk(s))", self.shape[0], self.shape[1], self.tile_words, self.chunks.len())
    }
}

impl Drop for Exl3Gpu {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes as u64, Ordering::Relaxed);
    }
}

impl Exl3Gpu {
    /// Upload `data`, `max_tile_rows` tile rows a buffer at most (the binding limit otherwise; tests make it small).
    fn upload(backend: &WgpuBackend, data: Exl3Data, max_tile_rows: Option<usize>) -> Self {
        let gpu = Arc::clone(&backend.gpu);
        let (k, n, tw) = (data.suh.len(), data.svh.len(), data.tile_words);
        let (ktiles, ntiles, nw) = (k / 16, n / 16, tw / 2);
        // A tile row (every tile column of 16 input channels) is the unit a buffer holds: as many as the binding
        // limit allows, or `max_tile_rows` (tests make it small, as a small adapter's limit would).
        let row_bytes = ntiles * nw * 4;
        let limit = (chunk_limit(&gpu.limits) as usize / row_bytes).max(1);
        let per = max_tile_rows.map_or(limit, |m| m.min(limit)).max(1);
        let bytes: Vec<u8> = data.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut chunks = Vec::new();
        let mut first = 0;
        while first < ktiles {
            let rows = per.min(ktiles - first);
            let slice = &bytes[first * row_bytes..(first + rows) * row_bytes];
            let (buffer, _, _) = gpu.upload_rows(slice, slice.len(), 1).remove(0);
            // As the CUDA kernel splits: enough workgroups to fill a large GPU, at most 8 slots of partial sums.
            let splits = 4096usize.div_ceil(ntiles).next_power_of_two().min(8).min(rows).max(1);
            chunks.push((buffer, first as u32, rows as u32, splits as u32));
            first += rows;
        }
        let nbytes = data.words.len() * 4;
        Self {
            gpu,
            serial: Arc::clone(&backend.serial),
            t: Transform { k, n, suh: data.suh, svh: data.svh, input_map: data.input_map, output_map: data.output_map },
            tile_words: tw,
            chunks,
            shape: [n, k],
            nbytes,
            used: Arc::clone(&backend.used),
            chain: std::sync::OnceLock::new(),
        }
    }

    /// Record one pass into `enc`: `rows` (at most [`ROWS`]) prepared input rows. Its partial sums are in the
    /// returned buffer once `enc` has run ([`Recorded`]).
    fn record(&self, enc: &mut wgpu::CommandEncoder, xh: &[f32], rows: usize) -> Recorded {
        let (k, n) = (self.t.k, self.t.n);
        let gpu = &self.gpu;
        let pipeline = gpu.exl3_pipeline(rows > 1);
        let slots: usize = self.chunks.iter().map(|c| c.3 as usize).sum();
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let xbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-exl3-x"),
            size: (rows * k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&xbuf, 0, &bytes(xh));
        let psize = (slots * rows * n * 4) as u64;
        let pbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-exl3-partial"),
            size: psize,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ntiles = (n / 16) as u32;
        let mut groups = Vec::new();
        let mut slot0 = 0u32;
        for (buffer, first, tile_rows, splits) in &self.chunks {
            let params: Vec<u8> = [n as u32, k as u32, *first, *tile_rows, self.tile_words as u32, rows as u32, *splits, slot0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            slot0 += splits;
            let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-exl3-params"),
                size: params.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue().write_buffer(&ubuf, 0, &params);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-exl3"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: pbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
                ],
            });
            groups.push((group, *splits));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (group, splits) in &groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(ntiles.min(65535), ntiles.div_ceil(65535), *splits);
            }
        }
        Recorded { part: pbuf, size: psize, rows, n, slots }
    }

    /// One pass: `rows` (at most [`ROWS`]) prepared input rows, their matmul sums `[rows, n]`.
    fn pass(&self, xh: &[f32], rows: usize) -> Vec<f32> {
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let recorded = self.record(&mut enc, xh, rows);
        read_back(&self.gpu, enc, &[recorded]).pop().expect("one pass, one result")
    }
}

/// A pass recorded into an encoder: its partial sums' buffer, `slots` of `[rows, n]`.
struct Recorded {
    part: wgpu::Buffer,
    size: u64,
    rows: usize,
    n: usize,
    slots: usize,
}

/// Run `enc`, which recorded `passes`, and read every pass's sums back with one submit and one mapping: each one's
/// slots added up, `[rows, n]`.
fn read_back(gpu: &Gpu, mut enc: wgpu::CommandEncoder, passes: &[Recorded]) -> Vec<Vec<f32>> {
    let total: u64 = passes.iter().map(|r| r.size).sum();
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-exl3-read"),
        size: total.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut at = 0;
    for r in passes {
        enc.copy_buffer_to_buffer(&r.part, 0, &staging, at, r.size);
        at += r.size;
    }
    gpu.queue().submit([enc.finish()]);
    let raw = gpu.map_read(&staging, total.max(4));
    let all: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut at = 0;
    passes
        .iter()
        .map(|r| {
            let len = r.slots * r.rows * r.n;
            let mut y = vec![0f32; r.rows * r.n];
            for slot in all[at..at + len].chunks_exact(r.rows * r.n) {
                for (a, b) in y.iter_mut().zip(slot) {
                    *a += b;
                }
            }
            at += len;
            y
        })
        .collect()
}

impl PackedLinear for Exl3Gpu {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn nbytes(&self) -> usize {
        self.nbytes + (self.t.k + self.t.n) * 8
    }
    fn linear(&self, x: &Tensor) -> Tensor {
        let (k, n) = (self.t.k, self.t.n);
        let mut host = None;
        let (data, m, shape) = self.t.rows(x, &mut host);
        let mut out = vec![0f32; m * n];
        let mut xh = vec![0f32; ROWS * k];
        for start in (0..m).step_by(ROWS) {
            let rows = ROWS.min(m - start);
            for r in 0..rows {
                self.t.pre(&data[(start + r) * k..(start + r + 1) * k], &mut xh[r * k..(r + 1) * k]);
            }
            let mut y = self.pass(&xh[..rows * k], rows);
            for r in 0..rows {
                self.t.post(&mut y[r * n..(r + 1) * n], &mut out[(start + r) * n..(start + r + 1) * n]);
            }
        }
        Tensor::from_vec(out, shape)
    }
}

// ---------------------------------------------------------------------------
// In a chain
// ---------------------------------------------------------------------------

/// f32 to f16 and back as the host's `half` (the `half` crate's: to nearest even, subnormals, infinity past 65504),
/// for the chain's transforms: an activation is not always a normal f16 value, as a decoded weight is.
pub(crate) const HALF: &str = r#"
fn half(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    let sign = b & 0x80000000u;
    let a = b & 0x7fffffffu;
    if (a > 0x7f800000u) { return v; }
    if (a >= 0x477ff000u) { return bitcast<f32>(sign | 0x7f800000u); }
    if (a < 0x38800000u) {
        // below f16's normals: a multiple of 2^-24, to nearest even
        let q = round(bitcast<f32>(a) * 16777216.0) / 16777216.0;
        return bitcast<f32>(sign | bitcast<u32>(q));
    }
    return bitcast<f32>(sign | ((a + 0xfffu + ((a >> 13u) & 1u)) & 0xffffe000u));
}
"#;

/// The chain's EXL3 kernels take a job list: job `j` is matrix `jobs[2j]` of a group on row `jobs[2j + 1]` of its
/// input, its result row `j`. A projection's input transform for each job, a workgroup a (128-block, job): `x`'s row
/// gathered through the input map (unless `p[0].y`, the identity), rounded to f16 and scaled by `suh`, the Hadamard
/// transform of each 128-block, scaled by 1/sqrt(128) and rounded (as `Transform::pre`). `p[0]`: k, identity.
const G_PRE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> suh: array<f32>;
@group(0) @binding(2) var<storage, read> imap: array<u32>;
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> xh: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = p[0].x;
    let j = wg.y;
    let m = jobs[2u * j];
    let xr = jobs[2u * j + 1u];
    let i = wg.x * 128u + t;
    var src = i;
    if (p[0].y == 0u) { src = imap[m * k + i]; }
    sh[t] = half(x[xr * k + src]) * suh[m * k + i];
    workgroupBarrier();
    for (var s = 1u; s < 128u; s *= 2u) {
        if ((t & s) == 0u) {
            let a = sh[t];
            let b = sh[t + s];
            sh[t] = a + b;
            sh[t + s] = a - b;
        }
        workgroupBarrier();
    }
    xh[j * k + i] = half(sh[t] * bitcast<f32>(0x3db504f4u));
}
"#;

/// [`G_PRE`] of a SwiGLU's output computed as it is read: job `j`'s row `xr` is `silu(g) * u`, `g` row `p[0].z xr +
/// p[0].w` of `x` and `u` row `p[1].x xr + p[1].y` of `up` (an expert group's gate and up rows `2 xr` and `2 xr + 1` of
/// one vector; a shared expert's row `xr` of two), as the SwiGLU kernels compute it: a dispatch fewer.
const G_PRE_SWIGLU: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> suh: array<f32>;
@group(0) @binding(2) var<storage, read> imap: array<u32>;
@group(0) @binding(3) var<storage, read> jobs: array<u32>;
@group(0) @binding(4) var<storage, read> up: array<f32>;
@group(0) @binding(6) var<storage, read_write> xh: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let k = p[0].x;
    let j = wg.y;
    let m = jobs[2u * j];
    let xr = jobs[2u * j + 1u];
    let i = wg.x * 128u + t;
    var src = i;
    if (p[0].y == 0u) { src = imap[m * k + i]; }
    let g = x[(p[0].z * xr + p[0].w) * k + src];
    let u = up[(p[1].x * xr + p[1].y) * k + src];
    sh[t] = half((g / (1.0 + exp(-g))) * u) * suh[m * k + i];
    workgroupBarrier();
    for (var s = 1u; s < 128u; s *= 2u) {
        if ((t & s) == 0u) {
            let a = sh[t];
            let b = sh[t + s];
            sh[t] = a + b;
            sh[t + s] = a - b;
        }
        workgroupBarrier();
    }
    xh[j * k + i] = half(sh[t] * bitcast<f32>(0x3db504f4u));
}
"#;

/// The matmul of each job's transformed row (the host's one-row kernel's, [`one_source`]; the matrices a group's:
/// matrix `m`'s words from `m * p[1].x`): a workgroup a (tile column, job and split), its partial sums to `part[(j *
/// splits + s) * n..]`, the jobs from `p[1].y` (a pass of a long list: 65535 workgroups an axis). `p[0]`: n, k, tile
/// words, splits; `p[1]`: words a matrix, the pass's first job.
fn g_mm_source() -> String {
    g_mm_source_with(None)
}

/// [`g_mm_source`] for tiles of `tw` words alone ([`lanes_fixed`]).
fn g_mm_fixed_source(tw: usize) -> String {
    g_mm_source_with(Some(tw))
}

fn g_mm_source_with(fixed: Option<usize>) -> String {
    let body = one_lanes_with("j * k + kt * 16u + {r}", "base + (kt * ntiles + nt) * nw", "", fixed);
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;
var<private> j: u32;
var<private> k: u32;
{body}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    j = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    let base = jobs[2u * j] * p[1].x;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    red[t] = lanes(t, nt, ntiles, ks, ke, tw, base);
    workgroupBarrier();
    if (t < 16u) {{
        part[(j * splits + s) * n + nt * 16u + t] = column(t);
    }}
}}
"#
    )
}

/// [`MANY`]'s matmul in a chain, for a prompt's rows: `rows` (16, 32 or 64) jobs of one matrix at a time, from
/// `order` in blocks of `rows` (a block's jobs one matrix's, its unused places [`NONE`]; see [`many_order`]). Each
/// tile's 256 weights are decoded once into the workgroup's memory and thread `(r, c)` sums column `c` for the block's
/// jobs `r`, `r + 16`, .., sixteen products a tile each in `MANY`'s order (so a projection's rows are its own kernel's
/// bit for bit): a workgroup a (tile column, block and split), each job's partial sums to `part[(j * splits + s) *
/// n..]` as [`g_mm_source`]'s. The next tile's words and inputs load while this one's are summed, through two sets of the
/// workgroup's buffers. `p[0]`: n, k, tile words, splits; `p[1]`: words a matrix, the pass's first block.
pub(crate) fn g_many(rows: usize) -> String {
    assert!(matches!(rows, 16 | 32 | 64), "a block of 16, 32 or 64 rows");
    let m = rows / 16;
    let each = |f: &dyn Fn(usize) -> String| (0..m).map(f).collect::<Vec<_>>().join("\n");
    let ids = each(&|i| format!("    let j{i} = ids[r + {}u];", 16 * i));
    let regs = each(&|i| format!("    var xn{i} = 0.0;\n    var acc{i} = 0.0;"));
    let first = each(&|i| format!("        if (j{i} != 0xffffffffu) {{ xn{i} = x[j{i} * k + ks * 16u + c]; }}"));
    let store = each(&|i| format!("        xs[b][(r + {}u) * 4u + c / 4u][c % 4u] = xn{i};", 16 * i));
    let next = each(&|i| format!("            if (j{i} != 0xffffffffu) {{ xn{i} = x[j{i} * k + (kt + 1u) * 16u + c]; }}"));
    let sums = each(&|i| {
        format!(
            "            let x{i} = xs[b][(r + {}u) * 4u + q];\n            acc{i} = acc{i} + x{i}.x * w4.x;\n            acc{i} = acc{i} + x{i}.y * w4.y;\n            acc{i} = acc{i} + x{i}.z * w4.z;\n            acc{i} = acc{i} + x{i}.w * w4.w;",
            16 * i
        )
    });
    let out = each(&|i| format!("    if (j{i} != 0xffffffffu) {{ part[(j{i} * splits + s) * n + nt * 16u + c] = acc{i}; }}"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> tile: array<array<u32, 64>, 2>;
// a block's inputs, [row][16 of k]
var<workgroup> xs: array<array<vec4<f32>, {xs_len}>, 2>;
// the decoded tile, [column][16 of k] (20 apart, against bank conflicts)
var<workgroup> wt: array<array<vec4<f32>, 80>, 2>;
var<workgroup> ids: array<u32, {rows}>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

fn weight(r: u32, c: u32, tw: u32, buf: u32) -> f32 {{
    let nw = tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let jj = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + jj;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    let w0 = start / 32u;
    let sh = 48u - start % 32u;
    let a = tile[buf][w0];
    let b = tile[buf][(w0 + 1u) % nw];
    var code: u32;
    if (sh >= 32u) {{
        code = a >> (sh - 32u);
    }} else {{
        code = (a << (32u - sh)) | (b >> sh);
    }}
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let blk = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    if (t < {rows}u) {{
        ids[t] = order[blk * {rows}u + t];
    }}
    workgroupBarrier();
    let base = jobs[2u * ids[0]] * p[1].x;
    let r = t / 16u;
    let c = t % 16u;
    let nw = tw / 2u;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    // this thread's rows r + 16 i: it loads their inputs at column c, and sums their outputs at column c
{ids}
{regs}
    var wn = 0u;
    if (ks < ke) {{
        if (t < nw) {{ wn = words[base + (ks * ntiles + nt) * nw + t]; }}
{first}
    }}
    var b = 0u;
    for (var kt = ks; kt < ke; kt = kt + 1u) {{
        if (t < nw) {{ tile[b][t] = wn; }}
{store}
        workgroupBarrier();
        if (kt + 1u < ke) {{
            if (t < nw) {{ wn = words[base + ((kt + 1u) * ntiles + nt) * nw + t]; }}
{next}
        }}
        wt[b][c * 5u + r / 4u][r % 4u] = weight(r, c, tw, b);
        workgroupBarrier();
        for (var q = 0u; q < 4u; q = q + 1u) {{
            let w4 = wt[b][c * 5u + q];
{sums}
        }}
        b = 1u - b;
    }}
{out}
}}
"#,
        xs_len = rows * 4,
    )
}

/// [`g_many`] on the tensor cores (WGSL's cooperative matrices, f16 into f32), for blocks of `rows` (16, 32, 64 or
/// 128) jobs of one matrix: a workgroup 8 tile columns (128 outputs) of a block, `k` a tile row (16) at a time; each
/// step a warp decodes its tile column's 256 codes (a lane its 8, as the one-row kernel's lanes hold them, from the
/// four words they lie in) into the workgroup's memory as f16 (each an f16 exactly) and the block's inputs (f16 too,
/// as the input transform rounds them) beside them, the next step's words and inputs loaded as this one's are
/// multiplied; each warp its 16 outputs by the block's rows. Each job's sums to `part[j * n..]` (one split), through a
/// warp's staging (split `s` of `p[0].w` along k to its own part, `part[(j * splits + s) * n..]`, as [`g_mm_source`]'s).
/// The sums are a matmul's, not the one-row kernel's bit for bit. `p` as for [`g_many`].
pub(crate) fn g_coop(rows: usize) -> String {
    assert!(matches!(rows, 16 | 32 | 64 | 128), "a block of 16, 32, 64 or 128 rows");
    let f = rows / 16;
    let pairs = rows * 8;
    let each = |g: &dyn Fn(usize) -> String| (0..f).map(g).collect::<String>();
    let decl = each(&|i| format!("    var c{i} = coop_mat16x16<f32, C>();\n"));
    let mma = each(&|i| format!("        {{\n            let ib = cur + {}u * S2;\n            let bf = coopLoad<coop_mat16x16<f16, B>>(&xt[ib], s2);\n            c{i} = coopMultiplyAdd(af, bf, c{i});\n        }}\n", i * 16));
    let out = each(&|i| {
        format!(
            "    {{\n        let so = warp * 256u;\n        coopStore(c{i}, &stage[so], 16u);\n        workgroupBarrier();\n        for (var e = l; e < 256u; e += 32u) {{\n            let id = ids[{}u + e / 16u];\n            if (id != 0xffffffffu && live) {{ part[(id * splits + sp) * n + tc * 16u + e % 16u] = stage[so + e]; }}\n        }}\n        workgroupBarrier();\n    }}\n",
            i * 16
        )
    });
    // the inputs a thread loads a step: pairs `t + 256 i` of the block's rows by 16 of k
    let per = pairs.div_ceil(256);
    let xdecl: String = (0..per).map(|i| format!("    var xp{i} = vec2<f32>(0.0);\n")).collect();
    let xload: String = (0..per)
        .map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{\n                let id = ids[q / 8u];\n                xp{i} = vec2<f32>(0.0);\n                if (id != 0xffffffffu) {{ xp{i} = x2[(id * k + kn * 16u) / 2u + q % 8u]; }}\n            }}\n        }}\n", 256 * i))
        .collect();
    let xstore: String = (0..per)
        .map(|i| format!("        {{\n            let q = t + {}u;\n            if (q < {pairs}u) {{ xt[nb + (q / 8u) * S2 + q % 8u] = vec2<f16>(xp{i}); }}\n        }}\n", 256 * i))
        .collect();
    format!(
        r#"enable f16;
enable wgpu_cooperative_matrix;
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// a row's stride in the tiles, as f16 pairs: 16 of k and 8 against bank conflicts
const S2: u32 = 12u;
// two steps' weights [output][k] and inputs [row][k]
var<workgroup> wt: array<vec2<f16>, {wt_len}>;
var<workgroup> xt: array<vec2<f16>, {xt_len}>;
var<workgroup> stage: array<f32, 2048>;
// every code's place in a tile, and each lane's first word (the same in every tile)
var<workgroup> places: array<u32, 256>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each (as the one-row kernel's)
var<workgroup> values: array<f32, 1024>;
var<workgroup> firsts: array<u32, 32>;
var<workgroup> ids: array<u32, {rows}>;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

// The 16-bit window of code `i` of a tile of `tw` words: (its first word, its shift).
fn window(i: u32, tw: u32) -> vec2<u32> {{
    let nw = tw / 2u;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec2<u32>(start / 32u, 48u - start % 32u);
}}

// Code `i`'s view and its shift, packed, for a lane whose first word is `w0`: which 32 bits of the lane's four
// words its window lies in (view 2 d: word d; view 2 d + 1: word d's low half and the next's high half), and how
// far the code is above the view's low end. A lane's eight codes start within 96 bits of its first word (8 bits
// a weight at most), so views 0 to 5.
fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    let o = 32u * ((wd.x + nw - w0) % nw) + (48u - wd.y);
    return (o >> 4u) | ((16u - (o & 15u)) << 8u);
}}

// A code from the four words `q0..q3` (from a lane's first) at its packed place: its view, shifted (its low 16
// bits: `decode_at` takes no more). The straddling views are the same for a lane's eight codes.
fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let odd = (at & 1u) == 1u;
    let p0 = select(q0, (q0 << 16u) | (q1 >> 16u), odd);
    let p1 = select(q1, (q1 << 16u) | (q2 >> 16u), odd);
    let p2 = select(q2, (q2 << 16u) | (q3 >> 16u), odd);
    let hi = (at >> 1u) & 3u;
    return select(select(p0, p1, hi == 1u), p2, hi == 2u) >> (at >> 8u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    return values[dot4U8Packed(hx, 0x01010101u)];
}}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let nw = tw / 2u;
    let ntiles = n / 16u;
    // the block, and its split of k (`p[0].w` of them, none empty)
    let splits = p[0].w;
    let blk = p[1].y + wg.z / splits;
    let sp = wg.z % splits;
    let first = window(8u * (t / 8u), tw).x;
    places[t] = place(t, tw, first);
    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    if (t < {rows}u) {{
        ids[t] = order[blk * {rows}u + t];
    }}
    workgroupBarrier();
    // a block the order left unused (a GPU's grouping sizes the grid for the most blocks it could fill)
    if (workgroupUniformLoad(&ids[0]) == 0xffffffffu) {{
        return;
    }}
    let base = jobs[2u * ids[0]] * p[1].x;
    let warp = t / 32u;
    let l = t % 32u;
    // the warp's tile column (past the matrix: the last, its sums not stored), the lane's codes' places
    let tc = wg.x * 8u + warp;
    let live = tc < ntiles;
    let tcl = min(tc, ntiles - 1u);
    let w0 = firsts[l];
    let a0 = places[8u * l];
    let a1 = places[8u * l + 1u];
    let a2 = places[8u * l + 2u];
    let a3 = places[8u * l + 3u];
    let a4 = places[8u * l + 4u];
    let a5 = places[8u * l + 5u];
    let a6 = places[8u * l + 6u];
    let a7 = places[8u * l + 7u];
    // where its codes go: weights (k 2 (l % 4) + {{0, 1, 8, 9}}, output l / 4 + {{0, 8}}) of the warp's 16
    let wo0 = (warp * 16u + l / 4u) * S2 + l % 4u;
    let wo1 = wo0 + 8u * S2;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = sp * per;
    let ke = min(kts, ks + per);
{decl}{xdecl}    var q0 = 0u;
    var q1 = 0u;
    var q2 = 0u;
    var q3 = 0u;
    // the first step's
    {{
        let kn = ks;
        let wb = base + (kn * ntiles + tcl) * nw;
        q0 = words[wb + w0 % nw];
        q1 = words[wb + (w0 + 1u) % nw];
        q2 = words[wb + (w0 + 2u) % nw];
        q3 = words[wb + (w0 + 3u) % nw];
{xload}        let nb = (ks % 2u) * {half_wt}u;
        wt[nb + wo0] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a0))), f16(decode_at(code_in(q0, q1, q2, q3, a1))));
        wt[nb + wo0 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a2))), f16(decode_at(code_in(q0, q1, q2, q3, a3))));
        wt[nb + wo1] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a4))), f16(decode_at(code_in(q0, q1, q2, q3, a5))));
        wt[nb + wo1 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a6))), f16(decode_at(code_in(q0, q1, q2, q3, a7))));
        {{
            let nb = (ks % 2u) * {half_xt}u;
{xstore}        }}
    }}
    workgroupBarrier();
    for (var kt = ks; kt < ke; kt++) {{
        // the next step (the last's own again, into the buffer no one reads after): loaded before this one's are
        // multiplied, decoded and stored after, with no branch between
        let kn = min(kt + 1u, ke - 1u);
        let wb = base + (kn * ntiles + tcl) * nw;
        q0 = words[wb + w0 % nw];
        q1 = words[wb + (w0 + 1u) % nw];
        q2 = words[wb + (w0 + 2u) % nw];
        q3 = words[wb + (w0 + 3u) % nw];
{xload}        let cur = (kt % 2u) * {half_wt}u;
        let curx = (kt % 2u) * {half_xt}u;
        let s2 = S2;
        let ia = cur + warp * 16u * S2;
        let af = coopLoadT<coop_mat16x16<f16, A>>(&wt[ia], s2);
        {{
            let cur = curx;
{mma}        }}
        let nb = ((kt + 1u) % 2u) * {half_wt}u;
        wt[nb + wo0] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a0))), f16(decode_at(code_in(q0, q1, q2, q3, a1))));
        wt[nb + wo0 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a2))), f16(decode_at(code_in(q0, q1, q2, q3, a3))));
        wt[nb + wo1] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a4))), f16(decode_at(code_in(q0, q1, q2, q3, a5))));
        wt[nb + wo1 + 4u] = vec2<f16>(f16(decode_at(code_in(q0, q1, q2, q3, a6))), f16(decode_at(code_in(q0, q1, q2, q3, a7))));
        {{
            let nb = ((kt + 1u) % 2u) * {half_xt}u;
{xstore}        }}
        workgroupBarrier();
    }}
{out}}}
"#,
        wt_len = 2 * 128 * 12,
        xt_len = 2 * rows * 12,
        half_wt = 128 * 12,
        half_xt = rows * 12,
    )
}

/// The pipeline name of [`g_coop`]'s kernel for blocks of `rows`.
pub(crate) fn coop_name(rows: usize) -> &'static str {
    match rows {
        16 => "exl3-coop-16",
        32 => "exl3-coop-32",
        64 => "exl3-coop-64",
        _ => "exl3-coop-128",
    }
}

/// The most rows [`few_kernel`] takes a block.
pub(crate) const FEW_MAX: usize = 8;

/// The matmul for a few rows of one matrix (a check of drafted tokens, a short chunk): [`g_mm_source`]'s lanes, each
/// decoding its eight codes of a tile once and summing them against every row's inputs in the order the one-row
/// kernel sums them (so each row's sums are that kernel's bit for bit). A workgroup a (tile column, block and split);
/// a block is `rows` jobs of one matrix from `order` (its unused places [`NONE`]; a block with none ends at once), each
/// job's partial sums to `part[(j * splits + s) * n..]` as [`g_mm_source`]'s. `p[0]`: n, k, tile words, splits; `p[1]`:
/// words a matrix, the pass's first block.
///
/// Its kernel and the kernel's name for tiles of `tile_words`: written for that rate as the one-row kernel is
/// ([`lanes_fixed`]; the general form under OAIY_EXL3_GENERAL), each made when first met.
pub(crate) fn few_kernel(rows: usize, tile_words: usize) -> (&'static str, &'static str) {
    type Made = [[std::sync::OnceLock<String>; RATES.len() + 1]; FEW_MAX - 1];
    static NAMES: Made = [const { [const { std::sync::OnceLock::new() }; RATES.len() + 1] }; FEW_MAX - 1];
    static SOURCES: Made = [const { [const { std::sync::OnceLock::new() }; RATES.len() + 1] }; FEW_MAX - 1];
    assert!((2..=FEW_MAX).contains(&rows), "a block of 2 to {FEW_MAX} rows");
    let rate = RATES.iter().position(|&rate| rate == tile_words).filter(|_| !general_only());
    let at = rate.unwrap_or(RATES.len());
    let name = NAMES[rows - 2][at].get_or_init(|| match rate {
        Some(_) => format!("exl3-few-{rows}-{tile_words}"),
        None => format!("exl3-few-{rows}"),
    });
    (name, SOURCES[rows - 2][at].get_or_init(|| g_few_source(rows, rate.map(|_| tile_words))))
}

fn g_few_source(rows: usize, fixed: Option<usize>) -> String {
    let Lanes { place, held, loads, codes } = fixed.map_or_else(lanes_general, lanes_fixed);
    let codes: String = codes.iter().enumerate().map(|(jj, code)| format!("        let c{jj} = decode_at({code});\n")).collect();
    let each = |f: &dyn Fn(usize) -> String| (0..rows).map(f).collect::<Vec<_>>().join("\n");
    let ids = each(&|i| format!("    let j{i} = order[blk * {rows}u + {i}u];"));
    let regs = each(&|i| format!("    var lo{i} = 0.0;\n    var hi{i} = 0.0;"));
    let sums = each(&|i| {
        format!(
            "        if (j{i} != 0xffffffffu) {{
            let xo = (j{i} * k + kt * 16u + rb) / 2u;
            let xa = x2[xo];
            let xb = x2[xo + 4u];
            lo{i} = lo{i} + xa.x * c0;
            lo{i} = lo{i} + xa.y * c1;
            lo{i} = lo{i} + xb.x * c2;
            lo{i} = lo{i} + xb.y * c3;
            hi{i} = hi{i} + xa.x * c4;
            hi{i} = hi{i} + xa.y * c5;
            hi{i} = hi{i} + xb.x * c6;
            hi{i} = hi{i} + xb.y * c7;
        }}"
        )
    });
    let reds = each(&|i| format!("    red[{i}u * 256u + t] = vec2<f32>(lo{i}, hi{i});"));
    let outs = each(&|i| format!("        if (r == {i}u) {{ j = j{i}; }}"));
    format!(
        r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(3) var<storage, read> order: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> red: array<vec2<f32>, {red_len}>;
// every code's place in a tile (`place`; in a kernel written for one rate, how far into its word it starts, read of
// a lane's first code alone), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
// a code's value by its bytes' sum (0 to 1,020), the workgroup's threads four each (as the one-row kernel's)
var<workgroup> values: array<f32, 1024>;
var<workgroup> firsts: array<u32, 32>;
var<workgroup> lead: u32;

fn round_f16(v: f32) -> f32 {{
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}}

fn window(i: u32, tw: u32) -> vec2<u32> {{
    let nw = tw / 2u;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {{
        end = end + (i + 1u) / 2u;
    }}
    let start = (end + nw * 32u - 16u) % (nw * 32u);
    return vec2<u32>(start / 32u, 48u - start % 32u);
}}

fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    let o = 32u * ((wd.x + nw - w0) % nw) + (48u - wd.y);
    return (o >> 4u) | ((16u - (o & 15u)) << 8u);
}}

fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let odd = (at & 1u) == 1u;
    let p0 = select(q0, (q0 << 16u) | (q1 >> 16u), odd);
    let p1 = select(q1, (q1 << 16u) | (q2 >> 16u), odd);
    let p2 = select(q2, (q2 << 16u) | (q3 >> 16u), odd);
    let hi = (at >> 1u) & 3u;
    return select(select(p0, p1, hi == 1u), p2, hi == 2u) >> (at >> 8u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    return values[dot4U8Packed(hx, 0x01010101u)];
}}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {{
        return;
    }}
    let blk = p[1].y + wg.z / splits;
    let s = wg.z % splits;
    if (t == 0u) {{
        lead = order[blk * {rows}u];
    }}
    if (workgroupUniformLoad(&lead) == 0xffffffffu) {{
        return;
    }}
{ids}
    let base = jobs[2u * j0] * p[1].x;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    let warp = t / 32u;
    let l = t % 32u;
    let nw = tw / 2u;
    let rb = 2u * (l % 4u);
    let first = window(8u * (t / 8u), tw).x;
{place}    for (var i = 0u; i < 4u; i = i + 1u) {{
        let sum = 4u * t + i;
        values[sum] = round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
    }}
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
{held}{regs}
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = base + (kt * ntiles + nt) * nw;
{loads}{codes}{sums}
    }}
{reds}
    workgroupBarrier();
    // row r's column c: each warp's four lanes of it, the warps in order (as the one-row kernel's `column`)
    if (t < {rows}u * 16u) {{
        let r = t / 16u;
        let c = t % 16u;
        var j = 0xffffffffu;
{outs}
        if (j != 0xffffffffu) {{
            let half = c / 8u;
            let cl = c % 8u;
            var total = 0.0;
            for (var w = 0u; w < 8u; w = w + 1u) {{
                for (var q = 0u; q < 4u; q = q + 1u) {{
                    let v = red[r * 256u + w * 32u + cl * 4u + q];
                    total = total + select(v.x, v.y, half == 1u);
                }}
            }}
            part[(j * splits + s) * n + nt * 16u + c] = total;
        }}
    }}
}}
"#,
        red_len = rows * 256,
    )
}

/// What the EXL3 projections chained for a few rows share on a device (their bind groups kept, as a step's are): the
/// job list `[0, r]` and the order `r` of [`FEW_MAX`] rows, and the transformed inputs, partial sums and unmapped
/// outputs of a projection of up to [`FewScratch::K`] inputs, [`FewScratch::PART`] partial sums a row (outputs times
/// splits) and [`FewScratch::N`] outputs.
pub(crate) struct FewScratch {
    pub jobs: DeviceVec,
    pub order: DeviceVec,
    pub xh: DeviceVec,
    pub part: DeviceVec,
    pub yt: DeviceVec,
}

impl FewScratch {
    pub const K: usize = 16384;
    pub const PART: usize = 1 << 18;
    pub const N: usize = 1 << 18;

    pub(crate) fn new(b: &WgpuBackend) -> Self {
        let rows = FEW_MAX as u32;
        FewScratch {
            jobs: u32_vec(b, &(0..rows).flat_map(|r| [0, r]).collect::<Vec<_>>()),
            order: u32_vec(b, &(0..rows).collect::<Vec<_>>()),
            xh: ggml_rs::DeviceChain::vec(b, FEW_MAX * Self::K),
            part: ggml_rs::DeviceChain::vec(b, FEW_MAX * Self::PART),
            yt: ggml_rs::DeviceChain::vec(b, FEW_MAX * Self::N),
        }
    }

    /// Whether a projection of `k` inputs, `n` outputs and `splits` fits.
    pub(crate) fn fits(k: usize, n: usize, splits: usize) -> bool {
        k <= Self::K && n <= Self::N && n * splits <= Self::PART
    }
}

/// A check's routed experts in blocks for [`few_kernel`], from its down jobs ([`ROUTE_RANK`]'s: pair `j`'s expert, `p[0].x`
/// pairs, at most 256): each expert's pairs (at most `p[0].y`, the block's rows, as a row takes an expert once) a block
/// of their own in their list's order, the blocks in the order their experts first appear; the down jobs' order
/// (`order_d`, `p[0].x` blocks of `p[0].y`) and the gate and up jobs' (`order_gu`: expert block `b`'s gate jobs `2j` in
/// block `2b`, its up jobs `2j + 1` in `2b + 1`), the other places [`NONE`]. One workgroup.
pub(crate) const GROUP: &str = r#"
@group(0) @binding(0) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> order_gu: array<u32>;
@group(0) @binding(7) var<storage, read_write> order_d: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> ex: array<u32, 256>;
// 1 where a pair is its expert's first
var<workgroup> first: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let rows = p[0].y;
    for (var i = t; i < n * rows; i += 256u) {
        order_d[i] = 0xffffffffu;
        order_gu[2u * i] = 0xffffffffu;
        order_gu[2u * i + 1u] = 0xffffffffu;
    }
    if (t < n) {
        ex[t] = jobs[2u * t];
    }
    workgroupBarrier();
    var leader = t;
    var slot = 0u;
    if (t < n) {
        let e = ex[t];
        for (var r = 0u; r < t; r++) {
            if (ex[r] == e) {
                if (leader == t) { leader = r; }
                slot += 1u;
            }
        }
        first[t] = select(0u, 1u, leader == t);
    }
    storageBarrier();
    workgroupBarrier();
    if (t < n) {
        var blk = 0u;
        for (var r = 0u; r < leader; r++) { blk += first[r]; }
        order_d[blk * rows + slot] = t;
        order_gu[2u * blk * rows + slot] = 2u * t;
        order_gu[(2u * blk + 1u) * rows + slot] = 2u * t + 1u;
    }
}
"#;

/// How a group's jobs are taken: each a workgroup's (a step's, [`g_mm_source`]); a prompt's in blocks of one matrix
/// ([`g_many`], the order and its blocks); a check's few rows' grouped on the GPU in blocks of `rows` ([`few_kernel`], the
/// order and the most blocks it can have), each job's sums the one-job kernel's.
#[derive(Clone, Copy)]
enum Order<'a> {
    Jobs,
    /// The order, its blocks, and the jobs a block.
    Many(&'a DeviceVec, usize, usize),
    Few(&'a DeviceVec, usize, usize),
}

/// An unused place in a block of [`g_many`]'s job order.
pub(crate) const NONE: u32 = u32::MAX;

/// A job list's (`jobs`: pairs of matrix and input row) order for [`g_many`]: its jobs grouped by matrix (each
/// matrix's in their list's order), in blocks of `rows`, a block one matrix's and its unused places [`NONE`].
pub(crate) fn many_order(jobs: &[u32], rows: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..(jobs.len() / 2) as u32).collect();
    idx.sort_by_key(|&j| jobs[2 * j as usize]);
    let mut order = Vec::with_capacity(idx.len() + rows);
    for same in idx.chunk_by(|&a, &b| jobs[2 * a as usize] == jobs[2 * b as usize]) {
        for block in same.chunks(rows) {
            order.extend_from_slice(block);
            order.resize(order.len().next_multiple_of(rows), NONE);
        }
    }
    order
}

/// The pipeline name of [`g_many`]'s kernel for blocks of `rows`.
pub(crate) fn many_name(rows: usize) -> &'static str {
    match rows {
        16 => "exl3-many-16",
        32 => "exl3-many-32",
        _ => "exl3-many-64",
    }
}

/// Rows a block of a MoE layer's experts takes: a prompt routes a few rows to each, and a block decodes its expert's
/// weights whatever its rows (Qwen3.8-Flash-Next's chunk of 512, about 10 rows an expert: its experts 276 ms in blocks
/// of 32, 378 in blocks of 16; OAIY_EXL3_MOE_BLOCK=16 for those).
fn moe_block() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| match std::env::var("OAIY_EXL3_MOE_BLOCK").ok().and_then(|v| v.parse().ok()) {
        Some(16) => 16,
        _ => 32,
    })
}

/// Whether a prompt's EXL3 matmuls go through the tensor cores ([`g_coop`]): where the device has them.
pub(crate) fn coop_on(gpu: &Gpu) -> bool {
    gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
}

/// Rows a block of a prompt's experts' jobs takes: [`g_coop`]'s 16 (an expert's some 10 rows of a 512-row chunk in
/// one fragment of the tensor cores), else [`moe_block`]'s.
fn many_rows(gpu: &Gpu) -> usize {
    if coop_on(gpu) { 16 } else { moe_block() }
}

/// The jobs a block of a prompt's experts grouped on the GPU takes: about three times the jobs an expert has on
/// average (16 to 128), so a busy expert's tiles are decoded for as few blocks as a quiet one's empty places cost
/// columns (Qwen3.8-Flash-Next's chunk of 512, some 10 jobs an expert: blocks of 32 308 ms, of 16 337, of 64 339).
/// Blocks of 32 up to some 24 jobs an expert: a chunk of 1,024, 20 an expert, its experts' matmuls 118 ms with them
/// and 149 with blocks of 64 (a workgroup of 64 an SM, where two of 32).
pub(crate) fn moe_rows_for(pairs: usize, experts: usize) -> usize {
    let target = 3 * pairs / experts.max(1);
    if (16..=72).contains(&target) {
        return 32;
    }
    [16, 32, 64, 128].into_iter().find(|&b| b >= target).unwrap_or(128)
}

/// Each job's output transform, a workgroup a (128-block, job): its splits' partial sums added up (in order),
/// rounded to f16, the Hadamard transform of each 128-block, scaled by 1/sqrt(128) and `svh` and rounded (as
/// `Transform::post` before its output map). `p[0]`: n, splits.
const G_POST: &str = r#"
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> svh: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sh: array<f32, 128>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let splits = p[0].y;
    let j = wg.y;
    let m = jobs[2u * j];
    let c = wg.x * 128u + t;
    var v = 0.0;
    for (var s = 0u; s < splits; s++) { v += part[(j * splits + s) * n + c]; }
    sh[t] = half(v);
    workgroupBarrier();
    for (var st = 1u; st < 128u; st *= 2u) {
        if ((t & st) == 0u) {
            let a = sh[t];
            let b = sh[t + st];
            sh[t] = a + b;
            sh[t + st] = a - b;
        }
        workgroupBarrier();
    }
    y[j * n + c] = half(sh[t] * bitcast<f32>(0x3db504f4u) * svh[m * n + c]);
}
"#;

/// Each job's output through its matrix's output map: `y[j, i] = yt[j, omap[m, i]]`. `p[0]`: n.
const G_GATHER: &str = r#"
@group(0) @binding(0) var<storage, read> yt: array<f32>;
@group(0) @binding(1) var<storage, read> omap: array<u32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = p[0].x;
    let i = id.x;
    let j = id.y;
    if (i >= n) { return; }
    let m = jobs[2u * j];
    y[j * n + i] = yt[j * n + omap[m * n + i]];
}
"#;

/// The chain's kernels as WGSL: the input transform (of a SwiGLU's output, "pre-swiglu"), the matmul (a row a job; a
/// prompt's is [`g_many`]), the output transform and its map; each made once.
/// OAIY_EXL3_GENERAL: the one-row kernel's general form at every rate (to compare a rate's own with).
fn general_only() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OAIY_EXL3_GENERAL").is_some())
}

/// The chain's one-row kernel for tiles of `tile_words`, and its name: the one written for that rate
/// ([`lanes_fixed`]), each made when a tile of its rate is first met.
pub(crate) fn one_row_kernel(tile_words: usize) -> (&'static str, &'static str) {
    const NAMES: [&str; 11] = ["exl3-mm-16", "exl3-mm-24", "exl3-mm-32", "exl3-mm-40", "exl3-mm-48", "exl3-mm-56", "exl3-mm-64", "exl3-mm-80", "exl3-mm-96", "exl3-mm-112", "exl3-mm-128"];
    static SOURCES: [std::sync::OnceLock<String>; 11] = [const { std::sync::OnceLock::new() }; 11];
    match RATES.iter().position(|&rate| rate == tile_words) {
        Some(at) if !general_only() => (NAMES[at], SOURCES[at].get_or_init(|| g_mm_fixed_source(tile_words))),
        _ => ("exl3-mm", chain_shader("mm")),
    }
}

pub(crate) fn chain_shader(which: &str) -> &'static str {
    static SOURCES: [std::sync::OnceLock<String>; 5] = [const { std::sync::OnceLock::new() }; 5];
    let (at, make): (usize, fn() -> String) = match which {
        "pre" => (0, || format!("{HALF}{G_PRE}")),
        "pre-swiglu" => (1, || format!("{HALF}{G_PRE_SWIGLU}")),
        "mm" => (2, g_mm_source),
        "post" => (3, || format!("{HALF}{G_POST}")),
        _ => (4, || G_GATHER.to_string()),
    };
    SOURCES[at].get_or_init(make)
}

/// What a chain needs of a projection on the GPU, made when it is first chained: its transforms' tables (the maps
/// only where they are not the identity), and a one-row run's scratch and job table (one set, so a decode step's bind
/// groups are made once).
pub(crate) struct Exl3Chain {
    pub suh: DeviceVec,
    pub svh: DeviceVec,
    pub imap: Option<DeviceVec>,
    pub omap: Option<DeviceVec>,
    pub xh: DeviceVec,
    pub part: DeviceVec,
    pub yt: DeviceVec,
    pub jobs1: DeviceVec,
}

/// `values` (u32s) as a chain's vector (their bits).
pub(crate) fn u32_vec(b: &WgpuBackend, values: &[u32]) -> DeviceVec {
    let v = ggml_rs::DeviceChain::vec(b, values.len());
    upload_u32(b, &v, values);
    v
}

/// `values` (u32s) into a chain's vector `v` (their bits).
pub(crate) fn upload_u32(b: &WgpuBackend, v: &DeviceVec, values: &[u32]) {
    let as_f32: Vec<f32> = values.iter().map(|&w| f32::from_bits(w)).collect();
    ggml_rs::DeviceChain::upload(b, v, &as_f32);
}

impl Exl3Gpu {
    /// Its one buffer of words and its splits, if it has one buffer (a chain takes no other).
    pub(crate) fn single_chunk(&self) -> Option<(&wgpu::Buffer, u32)> {
        match self.chunks.as_slice() {
            [(buffer, 0, _, splits)] => Some((buffer, *splits)),
            _ => None,
        }
    }

    pub(crate) fn kn(&self) -> (usize, usize) {
        (self.t.k, self.t.n)
    }

    pub(crate) fn tile_words(&self) -> usize {
        self.tile_words
    }

    pub(crate) fn is_on(&self, gpu: &Arc<Gpu>) -> bool {
        Arc::ptr_eq(&self.gpu, gpu)
    }

    /// Its chain's tables and one-row scratch, made when first asked for.
    pub(crate) fn chain(&self, b: &WgpuBackend) -> &Exl3Chain {
        use ggml_rs::DeviceChain;
        self.chain.get_or_init(|| {
            let (k, n) = (self.t.k, self.t.n);
            let splits = self.single_chunk().map_or(1, |c| c.1) as usize;
            let up = |v: &[f32]| {
                let d = b.vec(v.len());
                DeviceChain::upload(b, &d, v);
                d
            };
            let identity = |m: &[u32]| m.iter().enumerate().all(|(i, &v)| v as usize == i);
            Exl3Chain {
                suh: up(&self.t.suh),
                svh: up(&self.t.svh),
                imap: (!identity(&self.t.input_map)).then(|| u32_vec(b, &self.t.input_map)),
                omap: (!identity(&self.t.output_map)).then(|| u32_vec(b, &self.t.output_map)),
                xh: b.vec(k),
                part: b.vec(splits * n),
                yt: b.vec(n),
                jobs1: u32_vec(b, &[0, 0]),
            }
        })
    }
}

/// Each row's experts summed in its own order, each weighted (as `Exl3MoeHost::forward`): `out[r, i] = sum over j < K
/// of w[r, j] * d[r K + j, i]`, then `+ w[r, K] * sh[r, i]` (the shared expert). `p[0]`: hidden, K, rows.
pub(crate) const WSUM_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read> sh: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let h = p[0].x;
    let kk = p[0].y;
    let i = id.x;
    if (i >= h * p[0].z) { return; }
    let r = i / h;
    let c = i % h;
    var acc = 0.0;
    for (var j = 0u; j < kk; j++) { acc += w[r * (kk + 1u) + j] * d[(r * kk + j) * h + c]; }
    acc += w[r * (kk + 1u) + kk] * sh[r * h + c];
    out[i] = acc;
}
"#;

/// [`WSUM_ROWS`] with each row's sum added to its streams (`xs[r, s] += post[r, s] * sum`, `p[0].w` streams), as
/// `ChainRecorder::stream_apply` writes it back. `p[0]`: hidden, K, rows, streams.
pub(crate) const WSUM_APPLY: &str = r#"
@group(0) @binding(0) var<storage, read> d: array<f32>;
@group(0) @binding(1) var<storage, read> sh: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> post: array<f32>;
@group(0) @binding(6) var<storage, read_write> xs: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let h = p[0].x;
    let kk = p[0].y;
    let streams = p[0].w;
    let i = id.x;
    if (i >= h * p[0].z) { return; }
    let r = i / h;
    let c = i % h;
    var acc = 0.0;
    for (var j = 0u; j < kk; j++) { acc += w[r * (kk + 1u) + j] * d[(r * kk + j) * h + c]; }
    acc += w[r * (kk + 1u) + kk] * sh[r * h + c];
    for (var s = 0u; s < streams; s++) {
        let at = (r * streams + s) * h + c;
        xs[at] = xs[at] + post[r * streams + s] * acc;
    }
}
"#;

/// The routing on the GPU, as `ggml_rs::exl3::route` routes, in two kernels ([`record_route`]). This one ranks: the
/// top `p[0].y` of a row's `p[0].x` routed logits by logit, a tie to the lower index (each logit placed by how many
/// beat it), and writes their down jobs by rank, `[e, j]` for pair `j = r k + rank` (the experts' down projections
/// take hidden row `j`). A workgroup 64 of a row's logits (the grid's second axis), four threads a logit, each every
/// fourth of the comparisons' quads: one workgroup a row gave a thread two or three logits' every comparison, some 390
/// turns of four, and a layer's routing was 14.6 us whatever its rows (0.7 ms of a Flash-Next step's 11). `p[0]`:
/// routed (at most 1024), k (at most 32).
pub(crate) const ROUTE_RANK: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> jd: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

// the row's logits four at a time (the comparisons' reads)
var<workgroup> l4: array<vec4<f32>, 256>;
var<workgroup> beat: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let r = wg.x;
    let lr = r * (n + 1u);
    let quads = (n + 3u) / 4u;
    // past the last logit, -inf (beats none)
    if (t < quads) {
        var v = vec4<f32>(bitcast<f32>(0xff800000u));
        for (var c = 0u; c < 4u; c++) {
            if (4u * t + c < n) { v[c] = logits[lr + 4u * t + c]; }
        }
        l4[t] = v;
    }
    workgroupBarrier();
    let i = wg.y * 64u + t / 4u;
    let live = i < n;
    var above = 0u;
    if (live) {
        // how many beat logit `i` (a higher logit, or as high at a lower index), among this thread's quads
        let v = logits[lr + i];
        for (var q = t % 4u; q < quads; q += 4u) {
            let u = l4[q];
            let j = 4u * q;
            above += select(0u, 1u, u.x > v || (u.x == v && j < i));
            above += select(0u, 1u, u.y > v || (u.y == v && j + 1u < i));
            above += select(0u, 1u, u.z > v || (u.z == v && j + 2u < i));
            above += select(0u, 1u, u.w > v || (u.w == v && j + 3u < i));
        }
    }
    beat[t] = above;
    workgroupBarrier();
    if (live && t % 4u == 0u) {
        let rank = beat[t] + beat[t + 1u] + beat[t + 2u] + beat[t + 3u];
        if (rank < k) {
            let pair = r * k + rank;
            jd[2u * pair] = i;
            jd[2u * pair + 1u] = pair;
        }
    }
}
"#;

/// [`ROUTE_RANK`]'s second kernel, a workgroup a row `r`, a thread an expert of its top k: their weights (a softmax
/// among themselves, their exponentials summed in rank order) and the shared expert's (the logit after the routed
/// ones, by its gate's sigmoid), `w` from `r (k + 1)`: the top k's, the shared one's last; and the row's jobs as the
/// grouped experts take them, gate and up `[2e, r, 2e + 1, r]` an expert (`jobs`, from pair `r k`). `p[0]`: routed, k
/// (at most 32).
pub(crate) const ROUTE_WEIGHTS: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> jd: array<u32>;
@group(0) @binding(6) var<storage, read_write> jobs: array<u32>;
@group(0) @binding(7) var<storage, read_write> w: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> ex: array<f32, 32>;

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) j: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let r = wg.x;
    let lr = r * (n + 1u);
    let mx = logits[lr + jd[2u * r * k]];
    var e = 0u;
    var v = 0.0;
    if (j < k) {
        e = jd[2u * (r * k + j)];
        v = exp(logits[lr + e] - mx);
    }
    ex[j] = v;
    workgroupBarrier();
    if (j < k) {
        var sum = 0.0;
        for (var i = 0u; i < k; i++) { sum += ex[i]; }
        w[r * (k + 1u) + j] = v / sum;
        let at = 4u * (r * k + j);
        jobs[at] = 2u * e;
        jobs[at + 1u] = r;
        jobs[at + 2u] = 2u * e + 1u;
        jobs[at + 3u] = r;
    }
    if (j == 0u) {
        w[r * (k + 1u) + k] = 1.0 / (1.0 + exp(-logits[lr + n]));
    }
}
"#;

/// `rows` rows' routing recorded from their router's `logits` (`[rows, routed + 1]`): each row's top `top_k` experts'
/// down jobs by rank into `st` ([`ROUTE_RANK`]), then their weights and gate and up jobs ([`ROUTE_WEIGHTS`]).
pub(crate) fn record_route(rec: &mut crate::chain::Recorder<'_>, logits: &wgpu::Buffer, st: &Step, routed: usize, top_k: usize, rows: usize) {
    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
    assert!(routed <= 1024 && top_k <= 32 && top_k <= routed, "moe: the top {top_k} of {routed} experts");
    let d = rec.gpu().dummy().clone();
    let drw = rec.gpu().dummy_rw().clone();
    let words = [routed as u32, top_k as u32];
    let jd = buf(&st.jobs_d);
    rec.dispatch_wide("moe-route-rank", ROUTE_RANK, [logits, &d, &d, &d, &d, &d, &jd, &drw], &words, (rows as u32, (routed as u32).div_ceil(64), 1));
    rec.dispatch_wide("moe-route-weights", ROUTE_WEIGHTS, [logits, &jd, &d, &d, &d, &d, &buf(&st.jobs_gu), &buf(&st.w)], &words, (rows as u32, 1, 1));
}

/// A prompt's routed jobs grouped by expert on the GPU, as [`many_order`] groups them on the host: each expert's down
/// jobs in blocks of `p[0].w` (a block one expert's, its unused places [`NONE`]; the blocks in the experts' order, a
/// job's place among its expert's as the atomics fall), and their gate and up jobs in blocks `2 b` and `2 b + 1`.
/// `od`: the down order (`p[0].y` blocks), then each expert's count, first block and filled places (`p[0].z`
/// experts); `og` the gate and up order. First every place unused and every count 0.
///
/// `p[1].x` not 0: an expert with that many jobs or fewer takes no block of those; its jobs go to a block of its own
/// of `p[1].x` places in a second order, the down one from `od[p[1].y]` (block `e` expert `e`'s) and the gate and up
/// one from `og[p[1].z]` (blocks `2 e` and `2 e + 1`): the few rows' kernel's, where the tensor cores' would decode
/// a matrix for a block of mostly empty rows.
pub(crate) const MANY_CLEAR: &str = r#"
@group(0) @binding(6) var<storage, read_write> og: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 256u;
    let places = p[0].w * p[0].y;
    if (i < 2u * places) { og[i] = 0xffffffffu; }
    if (i < places) { od[i] = 0xffffffffu; }
    if (i < 3u * p[0].z) { od[places + i] = 0u; }
    let few = p[1].x;
    if (few > 0u) {
        if (i < 2u * p[0].z * few) { og[p[1].z + i] = 0xffffffffu; }
        if (i < p[0].z * few) { od[p[1].y + i] = 0xffffffffu; }
    }
}
"#;

/// [`MANY_CLEAR`]'s second pass: each expert's jobs counted, a thread a pair. `p[0]`: the pairs, the blocks.
pub(crate) const MANY_COUNT: &str = r#"
@group(0) @binding(0) var<storage, read> jd: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= p[0].x) { return; }
    atomicAdd(&od[p[0].w * p[0].y + jd[2u * j]], 1u);
}
"#;

/// [`MANY_CLEAR`]'s third: each expert's first block, the blocks of the experts before it, one workgroup a thread an
/// expert (1024 at most). `p[0]`: the pairs, the blocks, the experts, the jobs a block.
pub(crate) const MANY_SCAN: &str = r#"
@group(0) @binding(7) var<storage, read_write> od: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sums: array<u32, 1024>;

@compute @workgroup_size(1024)
fn main(@builtin(local_invocation_index) e: u32) {
    let bs = p[0].w;
    let at = bs * p[0].y;
    let ne = p[0].z;
    var nb = 0u;
    if (e < ne) { nb = (od[at + e] + bs - 1u) / bs; }
    // (an expert of few jobs: none of these blocks, its own in the second order)
    if (p[1].x > 0u && e < ne && od[at + e] <= p[1].x) { nb = 0u; }
    sums[e] = nb;
    workgroupBarrier();
    for (var st = 1u; st < 1024u; st *= 2u) {
        var v = sums[e];
        if (e >= st) { v += sums[e - st]; }
        workgroupBarrier();
        sums[e] = v;
        workgroupBarrier();
    }
    if (e < ne) { od[at + ne + e] = sums[e] - nb; }
}
"#;

/// [`MANY_CLEAR`]'s last: each pair's down job into its expert's blocks, and its gate and up jobs (`2 j`, `2 j + 1`)
/// into theirs, a thread a pair. `p[0]`: the pairs, the blocks, the experts.
pub(crate) const MANY_SCATTER: &str = r#"
@group(0) @binding(0) var<storage, read> jd: array<u32>;
@group(0) @binding(6) var<storage, read_write> og: array<u32>;
@group(0) @binding(7) var<storage, read_write> od: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= p[0].x) { return; }
    let bs = p[0].w;
    let at = bs * p[0].y;
    let ne = p[0].z;
    let e = jd[2u * j];
    let pos = atomicAdd(&od[at + 2u * ne + e], 1u);
    let few = p[1].x;
    if (few > 0u && atomicLoad(&od[at + e]) <= few) {
        atomicStore(&od[p[1].y + e * few + pos], j);
        og[p[1].z + 2u * e * few + pos] = 2u * j;
        og[p[1].z + (2u * e + 1u) * few + pos] = 2u * j + 1u;
        return;
    }
    let b = atomicLoad(&od[at + ne + e]) + pos / bs;
    let slot = pos % bs;
    atomicStore(&od[b * bs + slot], j);
    og[2u * b * bs + slot] = 2u * j;
    og[(2u * b + 1u) * bs + slot] = 2u * j + 1u;
}
"#;

/// One kind of a layer's routed experts' projection as a group (their gate and up matrices, or their down ones): their
/// words in one buffer, a matrix's after another, and their transforms' tables (the maps the identity).
struct Group {
    words: wgpu::Buffer,
    suh: DeviceVec,
    svh: DeviceVec,
    k: usize,
    n: usize,
    tw: usize,
    /// Words a matrix.
    mwords: usize,
    splits: u32,
}

/// A decode step's scratch: one row, `top_k` experts (the bind groups of a step made once).
pub(crate) struct Step {
    pub(crate) top_k: usize,
    pub(crate) jobs_gu: DeviceVec,
    pub(crate) jobs_d: DeviceVec,
    pub(crate) w: DeviceVec,
    pub(crate) xh_gu: DeviceVec,
    pub(crate) part_gu: DeviceVec,
    pub(crate) out_gu: DeviceVec,
    pub(crate) xh_d: DeviceVec,
    pub(crate) part_d: DeviceVec,
    pub(crate) out_d: DeviceVec,
    pub(crate) sg: DeviceVec,
    pub(crate) su: DeviceVec,
    pub(crate) sd: DeviceVec,
    /// A check's jobs grouped by matrix ([`GROUP`]): gate and up, down.
    pub(crate) order_gu: DeviceVec,
    pub(crate) order_d: DeviceVec,
}

/// A MoE layer's experts on the GPU as groups (Qwen3.8-Flash-Next's 512 routed ones): their gate and up matrices in
/// one buffer (matrix `2e` expert `e`'s gate, `2e + 1` its up), their down matrices in another, each group's
/// transforms' tables beside it; the shared expert as projections of its own. A row's (a step's) experts run in a
/// few dispatches from a job list, their transforms on the GPU too, where a projection was a dispatch (and its
/// transforms the host's) in a layer's two round trips.
pub struct Exl3MoeGrouped {
    b: WgpuBackend,
    routed: usize,
    hidden: usize,
    ff: usize,
    gu: Group,
    down: Group,
    shared: [Exl3Gpu; 3],
}

impl std::fmt::Debug for Exl3MoeGrouped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exl3MoeGrouped({} routed experts of {}x{} as two groups, and the shared one)", self.routed, self.hidden, self.ff)
    }
}

impl Exl3MoeGrouped {
    /// The layer's experts as groups on `b`, if they can be: every routed expert's projections of one shape and
    /// bitrate, their maps the identity, each group within a binding, and all of them within the budget less `reserve`.
    /// None leaves them to [`Exl3MoeHost`].
    fn try_new(b: &WgpuBackend, experts: &[[Exl3Data; 3]], reserve: u64) -> Result<Option<Exl3MoeGrouped>, String> {
        let Some((shared, routed)) = experts.split_last() else { return Ok(None) };
        let Some(first) = routed.first() else { return Ok(None) };
        let (hidden, ff) = (first[0].suh.len(), first[0].svh.len());
        let identity = |m: &[u32]| m.iter().enumerate().all(|(i, &v)| v as usize == i);
        for e in routed.iter().chain([shared]) {
            for d in e.iter() {
                d.validate()?;
            }
        }
        let same = routed.iter().all(|e| {
            [(hidden, ff), (hidden, ff), (ff, hidden)].iter().zip(e.iter()).all(|(&(k, n), d)| d.suh.len() == k && d.svh.len() == n && d.tile_words == e[0].tile_words && identity(&d.input_map) && identity(&d.output_map))
                && e[0].tile_words == first[0].tile_words
                && e[2].tile_words == first[2].tile_words
        });
        let shapes = [(hidden, ff), (hidden, ff), (ff, hidden)];
        if !same || shared.iter().zip(shapes).any(|(d, (k, n))| d.suh.len() != k || d.svh.len() != n) {
            return Ok(None);
        }
        let limit = chunk_limit(&b.gpu.limits);
        let mwords = |k: usize, n: usize, tw: usize| k / 16 * n / 16 * (tw / 2);
        let (gu_words, d_words) = (mwords(hidden, ff, first[0].tile_words), mwords(ff, hidden, first[2].tile_words));
        let gu_bytes = (2 * routed.len() * gu_words * 4) as u64;
        let d_bytes = (routed.len() * d_words * 4) as u64;
        let shared_bytes: u64 = shared.iter().map(|d| d.words.len() as u64 * 4).sum();
        let tables = (routed.len() * 3 * (hidden + ff) * 4) as u64;
        let total = gu_bytes + d_bytes + shared_bytes + tables;
        if gu_bytes > limit || d_bytes > limit {
            return Ok(None);
        }
        let prev = b.used.fetch_add(total, Ordering::Relaxed);
        if prev + total > b.budget.saturating_sub(reserve) {
            b.used.fetch_sub(total, Ordering::Relaxed);
            return Ok(None);
        }
        // the groups' words and tables; the shared expert's projections count themselves, so their share is given back
        b.used.fetch_sub(shared_bytes, Ordering::Relaxed);
        let group = |which: &[usize], k: usize, n: usize, tw: usize| -> Group {
            let mw = mwords(k, n, tw);
            let mut bytes = Vec::with_capacity(routed.len() * which.len() * mw * 4);
            let (mut suh, mut svh) = (Vec::new(), Vec::new());
            for e in routed {
                for &p in which {
                    bytes.extend(e[p].words.iter().flat_map(|w| w.to_le_bytes()));
                    suh.extend_from_slice(&e[p].suh);
                    svh.extend_from_slice(&e[p].svh);
                }
            }
            let words = b.gpu.upload_rows(&bytes, bytes.len(), 1).remove(0).0;
            let up = |v: &[f32]| {
                use ggml_rs::DeviceChain;
                let d = b.vec(v.len());
                DeviceChain::upload(b, &d, v);
                d
            };
            let ntiles = n / 16;
            let splits = 4096usize.div_ceil(ntiles).next_power_of_two().min(8).min(k / 16).max(1) as u32;
            Group { words, suh: up(&suh), svh: up(&svh), k, n, tw, mwords: mw, splits }
        };
        let gu = group(&[0, 1], hidden, ff, first[0].tile_words);
        let down = group(&[2], ff, hidden, first[2].tile_words);
        let [sg, su, sd] = shared;
        let one = |d: &Exl3Data| Exl3Gpu::upload(b, Exl3Data { words: d.words.clone(), suh: d.suh.clone(), svh: d.svh.clone(), tile_words: d.tile_words, input_map: d.input_map.clone(), output_map: d.output_map.clone() }, None);
        let shared = [one(sg), one(su), one(sd)];
        if shared.iter().any(|s| s.single_chunk().is_none()) {
            return Ok(None);
        }
        // the routed groups' bytes stay counted as long as the layer lives
        Ok(Some(Exl3MoeGrouped { b: b.clone(), routed: routed.len(), hidden, ff, gu, down, shared }))
    }

    pub(crate) fn is_on(&self, gpu: &Arc<Gpu>) -> bool {
        Arc::ptr_eq(&self.b.gpu, gpu)
    }

    /// Scratch for `rows` rows of `top_k` experts, from `vec`: each job one row (`one`: a step's, a check's) the groups'
    /// splits, a prompt's (each expert's rows in blocks) one split.
    fn scratch(&self, vec: &mut dyn FnMut(usize) -> DeviceVec, rows: usize, top_k: usize, one: bool) -> Step {
        let (h, f) = (self.hidden, self.ff);
        let pairs = rows * top_k;
        let (sgu, sd) = if one { (self.gu.splits as usize, self.down.splits as usize) } else { (1, 1) };
        Step {
            top_k,
            jobs_gu: vec(4 * pairs),
            jobs_d: vec(2 * pairs),
            w: vec(rows * (top_k + 1)),
            xh_gu: vec(2 * pairs * h),
            part_gu: vec(2 * pairs * sgu * f),
            out_gu: vec(2 * pairs * f),
            xh_d: vec(pairs * f),
            part_d: vec(pairs * sd * h),
            out_d: vec(pairs * h),
            sg: vec(rows * f),
            su: vec(rows * f),
            sd: vec(rows * h),
            order_gu: vec(if one && rows > 1 { 2 * pairs * rows } else { 1 }),
            order_d: vec(if one && rows > 1 { pairs * rows } else { 1 }),
        }
    }

    /// The kept scratch of `rows` routed rows (a step's one, a check's few), each job one row: one for every layer of
    /// this shape on the device (a layer's experts are done before the next layer's start), its bind groups kept.
    fn step(&self, rows: usize, top_k: usize) -> Arc<Step> {
        let key = [rows, top_k, self.hidden, self.ff, self.gu.splits as usize, self.down.splits as usize];
        let mut s = self.b.gpu.moe_steps.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, st)) = s.iter().find(|(k, _)| *k == key) {
            return Arc::clone(st);
        }
        let st = Arc::new(self.scratch(&mut |n| self.b.vec(n), rows, top_k, true));
        s.push((key, Arc::clone(&st)));
        st
    }

    /// One group's jobs (`jobs`, `count` of them) on `x`: its input transforms, matmul and output transforms into `y`.
    /// `order`: a prompt's jobs in blocks of one matrix (and the blocks' count), each tile decoded once a block, in one
    /// split (a prompt has workgroups enough without, and its partial sums are the smaller).
    #[allow(clippy::too_many_arguments)]
    /// `pairs`: `x` the gate and up rows of each hidden row (`2 r` and `2 r + 1`), the input their SwiGLU.
    #[allow(clippy::too_many_arguments)]
    fn group_pass(&self, rec: &mut crate::chain::Recorder<'_>, g: &Group, x: &DeviceVec, pairs: bool, jobs: &DeviceVec, count: usize, order: Order<'_>, xh: &DeviceVec, part: &DeviceVec, y: &DeviceVec) {
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let (xb, jb, xhb, pb, yb) = (buf(x), buf(jobs), buf(xh), buf(part), buf(y));
        let (suh, svh) = (buf(&g.suh), buf(&g.svh));
        if pairs {
            rec.dispatch_wide("exl3-pre-swiglu", chain_shader("pre-swiglu"), [&xb, &suh, &d, &jb, &xb, &d, &xhb, &drw], &[g.k as u32, 1, 2, 0, 2, 1], ((g.k / 128) as u32, count as u32, 1));
        } else {
            rec.dispatch_wide("exl3-pre", chain_shader("pre"), [&xb, &suh, &d, &jb, &d, &d, &xhb, &drw], &[g.k as u32, 1], ((g.k / 128) as u32, count as u32, 1));
        }
        let ntiles = (g.n / 16) as u32;
        let splits = if matches!(order, Order::Many(..)) { 1 } else { g.splits };
        // as many jobs (or blocks) a pass as the grid's third axis takes
        let per = (65535 / splits) as usize;
        match order {
            Order::Many(order, blocks, rows) => {
                let ob = buf(order);
                for first in (0..blocks).step_by(per) {
                    let these = per.min(blocks - first) as u32;
                    if coop_on(rec.gpu()) {
                        // 8 tile columns a workgroup on the tensor cores
                        let src = g_coop(rows);
                        rec.dispatch_wide(coop_name(rows), &src, [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, 1, g.mwords as u32, first as u32], (ntiles.div_ceil(8), 1, these));
                    } else {
                        rec.dispatch_wide(many_name(rows), &g_many(rows), [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), these * splits));
                    }
                }
            }
            Order::Few(order, blocks, rows) => {
                let ob = buf(order);
                for first in (0..blocks).step_by(per) {
                    let these = per.min(blocks - first) as u32;
                    let (name, source) = few_kernel(rows, g.tw);
                    rec.dispatch_wide(name, source, [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), these * splits));
                }
            }
            Order::Jobs => {
                for first in (0..count).step_by(per) {
                    let jobs = per.min(count - first) as u32;
                    // (tiles of up to 4 bits a weight through the kernel that finds a code in three words: OAIY_EXL3_GENERAL
                    // keeps every tile on the general one)
                    let (name, source) = one_row_kernel(g.tw as usize);
                    rec.dispatch_wide(name, source, [&g.words, &xhb, &jb, &d, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, g.splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), jobs * g.splits));
                }
            }
        }
        rec.dispatch_wide("exl3-post", chain_shader("post"), [&pb, &svh, &jb, &d, &d, &d, &yb, &drw], &[g.n as u32, splits], ((g.n / 128) as u32, count as u32, 1));
    }

    /// Record `assign`'s experts for each row of `x` into `out` (see `ChainRecorder::moe_rows`).
    pub(crate) fn record(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, assign: &[Vec<(usize, f32)>]) {
        let rows = assign.len();
        let h = self.hidden;
        let top_k = assign.first().map_or(0, |a| a.len().saturating_sub(1));
        assert!(rows > 0 && top_k > 0 && assign.iter().all(|a| a.len() == top_k + 1 && a[top_k].0 == self.routed), "moe: each row's routed experts, then the shared one");
        assert!(x.len >= rows * h && out.len >= rows * h, "moe: {rows} rows of {h}");
        // the jobs: gate and up of each (row, expert) on its row of x, then down of each on its hidden row
        let mut jobs_gu = Vec::with_capacity(4 * rows * top_k);
        let mut jobs_d = Vec::with_capacity(2 * rows * top_k);
        let mut w = Vec::with_capacity(rows * (top_k + 1));
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, wt)) in a[..top_k].iter().enumerate() {
                jobs_gu.extend([2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32]);
                jobs_d.extend([e as u32, (r * top_k + j) as u32]);
                w.push(wt);
            }
            w.push(a[top_k].1);
        }
        let b = rec.backend().clone();
        // a step's one row: the kept scratch (its bind groups kept); else this call's (from the pool, given back when
        // the recording has run)
        let st = if rows == 1 && rec.keeps() { self.step(1, top_k) } else { Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, rows == 1)) };
        let up = |v: &DeviceVec, data: &[u32]| DeviceChain::upload(&b, v, &data.iter().map(|&u| f32::from_bits(u)).collect::<Vec<_>>());
        up(&st.jobs_gu, &jobs_gu);
        up(&st.jobs_d, &jobs_d);
        DeviceChain::upload(&b, &st.w, &w);
        // a prompt's rows: each expert's in blocks, a tile decoded once a block
        let block = many_rows(rec.gpu());
        let mut order = |jobs: &[u32]| {
            let o = many_order(jobs, block);
            let v = rec.scratch(o.len());
            up(&v, &o);
            (v, o.len() / block)
        };
        let orders = (rows > 1).then(|| (order(&jobs_gu), order(&jobs_d)));
        let (ogu, od) = match &orders {
            Some(((g, gn), (d, dn))) => (Order::Many(g, *gn, block), Order::Many(d, *dn, block)),
            None => (Order::Jobs, Order::Jobs),
        };
        self.run(rec, &st, x, out, rows, ogu, od, None);
    }

    /// `rows` rows' experts (a step's one, a check's few) routed on the GPU from the router's `logits` (`[rows, routed +
    /// 1]`) and recorded into `out` (see `ChainRecorder::moe_routed`): [`record_route`] writes the jobs and weights where
    /// [`Self::record`] uploads them, each job one row (so each row's sums are a step's). `into`: each row's sum added to
    /// its streams (the streams, their write weights, how many) where it would be `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_routed(&self, rec: &mut crate::chain::Recorder<'_>, x: &DeviceVec, out: &DeviceVec, logits: &DeviceVec, top_k: usize, rows: usize, into: Option<(&DeviceVec, &DeviceVec, usize)>) -> bool {
        // a prompt's rows (more than a check's) grouped by expert on the GPU, where the tensor cores take its blocks
        let many = rows > FEW_MAX && coop_on(rec.gpu());
        if self.routed > 1024 || top_k == 0 || top_k > 32.min(self.routed) || rows == 0 || (rows > 64 && !many) || rows > 65535 || logits.len < rows * (self.routed + 1) {
            return false;
        }
        assert!(x.len >= rows * self.hidden && (into.is_some() || out.len >= rows * self.hidden), "moe: {rows} rows of {}", self.hidden);
        // (a prompt's orders its own: the scratch's are a check's; its scratch the recording's, each layer's in turn)
        let st = if rec.keeps() && !many {
            self.step(rows, top_k)
        } else if many {
            let key = [rows, top_k, self.hidden, self.ff];
            match rec.moe_tmp.take() {
                Some((k, st)) if k == key => {
                    rec.moe_tmp = Some((k, Arc::clone(&st)));
                    st
                }
                _ => {
                    let st = Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, false));
                    rec.moe_tmp = Some((key, Arc::clone(&st)));
                    st
                }
            }
        } else {
            Arc::new(self.scratch(&mut |n| rec.scratch(n), rows, top_k, true))
        };
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        record_route(rec, &buf(logits), &st, self.routed, top_k, rows);
        let pairs = rows * top_k;
        if many {
            // blocks of as many jobs as an expert has on average (16 to 64: an expert's tiles decoded once a block,
            // what an empty place of it costs a fragment's columns); the most blocks the experts could fill (a
            // part-filled one each at most), the grid that wide
            let bs = moe_rows_for(pairs, self.routed);
            let blocks = pairs.div_ceil(bs) + self.routed;
            let (og, od) = (rec.scratch(2 * bs * blocks), rec.scratch(bs * blocks + 3 * self.routed));
            let words = [pairs as u32, blocks as u32, self.routed as u32, bs as u32];
            let clear = (2 * bs * blocks) as u32;
            let groups = clear.div_ceil(256);
            let jd = buf(&st.jobs_d);
            rec.dispatch_wide("moe-many-clear", MANY_CLEAR, [&d, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, (groups.min(65535), groups.div_ceil(65535), 1));
            rec.dispatch_wide("moe-many-count", MANY_COUNT, [&jd, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            rec.dispatch_wide("moe-many-scan", MANY_SCAN, [&d, &d, &d, &d, &d, &d, &drw, &buf(&od)], &words, (1, 1, 1));
            rec.dispatch_wide("moe-many-scatter", MANY_SCATTER, [&jd, &d, &d, &d, &d, &d, &buf(&og), &buf(&od)], &words, ((pairs as u32).div_ceil(256), 1, 1));
            self.run(rec, &st, x, out, rows, Order::Many(&og, 2 * blocks, bs), Order::Many(&od, blocks, bs), into);
            return true;
        }
        // a check's few rows: an expert the rows share decoded once for them (its jobs one block; OAIY_MOE_UNGROUPED:
        // a job each)
        let grouped = (2..=FEW_MAX).contains(&rows) && pairs <= 256 && std::env::var_os("OAIY_MOE_UNGROUPED").is_none();
        let (ogu, od) = if grouped {
            rec.dispatch_wide("moe-group", GROUP, [&buf(&st.jobs_d), &d, &d, &d, &d, &d, &buf(&st.order_gu), &buf(&st.order_d)], &[pairs as u32, rows as u32], (1, 1, 1));
            (Order::Few(&st.order_gu, 2 * pairs, rows), Order::Few(&st.order_d, pairs, rows))
        } else {
            (Order::Jobs, Order::Jobs)
        };
        self.run(rec, &st, x, out, rows, ogu, od, into);
        true
    }

    /// The experts' work once `st` holds the jobs and weights: gate and up, SwiGLU, down, the shared expert on every
    /// row, and each row's weighted sum (into `out`, or added to the streams `into` names); the gate and up jobs taken as
    /// `ogu` has them, the down jobs as `od`, each SwiGLU computed as its down projection reads it.
    #[allow(clippy::too_many_arguments)]
    fn run(&self, rec: &mut crate::chain::Recorder<'_>, st: &Step, x: &DeviceVec, out: &DeviceVec, rows: usize, ogu: Order<'_>, od: Order<'_>, into: Option<(&DeviceVec, &DeviceVec, usize)>) {
        use ggml_rs::ChainRecorder;
        let (h, top_k) = (self.hidden, st.top_k);
        let pairs = rows * top_k;
        let (jgu, jd, wv, xh_gu, part_gu, out_gu, xh_d, part_d, out_d, sg, su, sd) = (&st.jobs_gu, &st.jobs_d, &st.w, &st.xh_gu, &st.part_gu, &st.out_gu, &st.xh_d, &st.part_d, &st.out_d, &st.sg, &st.su, &st.sd);
        self.group_pass(rec, &self.gu, x, false, jgu, 2 * pairs, ogu, xh_gu, part_gu, out_gu);
        self.group_pass(rec, &self.down, out_gu, true, jd, pairs, od, xh_d, part_d, out_d);
        // the shared expert on every row
        rec.exl3_rows(&self.shared[0], x, sg, rows);
        rec.exl3_rows(&self.shared[1], x, su, rows);
        rec.exl3_rows_swiglu(&self.shared[2], sg, su, sd, rows);
        let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector").clone();
        let d = rec.gpu().dummy().clone();
        let drw = rec.gpu().dummy_rw().clone();
        match into {
            Some((xs, post, streams)) => {
                assert!(xs.len >= rows * streams * h && post.len >= rows * streams, "moe: {rows} rows' {streams} streams");
                rec.dispatch_wide("moe-wsum-apply", WSUM_APPLY, [&buf(out_d), &buf(sd), &buf(wv), &buf(post), &d, &d, &buf(xs), &drw], &[h as u32, top_k as u32, rows as u32, streams as u32], (((rows * h) as u32).div_ceil(256), 1, 1));
            }
            None => rec.dispatch_wide("moe-wsum-rows", WSUM_ROWS, [&buf(out_d), &buf(sd), &buf(wv), &d, &d, &d, &buf(out), &drw], &[h as u32, top_k as u32, rows as u32], (((rows * h) as u32).div_ceil(256), 1, 1)),
        }
    }
}

impl ggml_rs::exl3::Experts for Exl3MoeGrouped {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let h = self.hidden;
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.routed + 1;
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        let b = &self.b;
        let (xd, out) = (b.vec(rows * h), b.vec(rows * h));
        DeviceChain::upload(b, &xd, x.data());
        let mut rec = b.begin();
        rec.keep_groups(false);
        rec.moe_rows(self, &xd, &out, &assign);
        rec.read(&out);
        let y = rec.finish().pop().expect("the experts' sum");
        Tensor::from_vec(y, vec![rows, h])
    }
}

/// One projection of an expert: its packed weights on the GPU, or decoded on the CPU.
#[derive(Debug)]
enum Proj {
    Gpu(Exl3Gpu),
    Cpu(Exl3Cpu),
}

impl Proj {
    fn t(&self) -> &Transform {
        match self {
            Proj::Gpu(g) => &g.t,
            Proj::Cpu(c) => &c.t,
        }
    }
}

/// A MoE layer's EXL3 experts (the shared one last) without CUDA: each projection on the GPU while the weight budget
/// holds it, else decoded on the CPU, routed on the host exactly as `ggml_rs_cuda::exl3::Exl3Experts` routes.
///
/// A layer's work goes as two batches: every expert's gate and up projections for the rows routed to it, then every
/// down projection. The GPU's of a batch are recorded into one command encoder and read back together, one submit
/// for the lot (a decode step of Qwen3.8-Flash-Next would otherwise wait on 33 a layer); the CPU's run meanwhile, an
/// expert a thread.
pub struct Exl3MoeHost {
    experts: Vec<[Proj; 3]>,
    hidden: usize,
    ff: usize,
    gpu: Option<(Arc<Gpu>, Arc<Mutex<()>>)>,
}

impl std::fmt::Debug for Exl3MoeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let on_gpu = self.experts.iter().flatten().filter(|p| matches!(p, Proj::Gpu(_))).count();
        write!(f, "Exl3MoeHost({} experts of {}x{}, {} of {} projections on the GPU)", self.experts.len(), self.hidden, self.ff, on_gpu, 3 * self.experts.len())
    }
}

use ggml_rs::exl3::route;

impl Exl3MoeHost {
    fn new(experts: Vec<[Proj; 3]>, gpu: Option<(Arc<Gpu>, Arc<Mutex<()>>)>) -> Result<Self, String> {
        let first = experts.first().ok_or("no experts")?;
        let (hidden, ff) = (first[0].t().k, first[0].t().n);
        for e in &experts {
            let shapes = [e[0].t(), e[1].t(), e[2].t()].map(|t| (t.k, t.n));
            if shapes != [(hidden, ff), (hidden, ff), (ff, hidden)] {
                return Err("every expert needs gate and up of hidden -> ff, and down of ff -> hidden".into());
            }
        }
        Ok(Self { experts, hidden, ff, gpu })
    }

    /// Each job's projection applied to its prepared rows: `(expert, which projection, rows [count, k])`. The GPU's
    /// in one submit, the CPU's on threads meanwhile; each result post-transformed, `[count, n]`, in job order.
    fn batch(&self, jobs: &[(usize, usize, Vec<f32>)]) -> Vec<Vec<f32>> {
        let mut out: Vec<Option<Vec<f32>>> = (0..jobs.len()).map(|_| None).collect();
        let cpu_jobs: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(self.experts[jobs[i].0][jobs[i].1], Proj::Cpu(_))).collect();
        let gpu_jobs: Vec<usize> = (0..jobs.len()).filter(|&i| matches!(self.experts[jobs[i].0][jobs[i].1], Proj::Gpu(_))).collect();
        let finish = |i: usize, mut y: Vec<f32>| -> Vec<f32> {
            let t = self.experts[jobs[i].0][jobs[i].1].t();
            let rows = jobs[i].2.len() / t.k;
            let mut o = vec![0f32; rows * t.n];
            for (yr, or) in y.chunks_exact_mut(t.n).zip(o.chunks_exact_mut(t.n)) {
                t.post(yr, or);
            }
            o
        };
        std::thread::scope(|scope| {
            // The CPU's experts, one a thread, while the GPU works.
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).max(1);
            let per = cpu_jobs.len().div_ceil(threads).max(1);
            let cpu_work: Vec<_> = cpu_jobs
                .chunks(per)
                .map(|part| {
                    scope.spawn(move || {
                        part.iter()
                            .map(|&i| {
                                let Proj::Cpu(c) = &self.experts[jobs[i].0][jobs[i].1] else { unreachable!() };
                                (i, c.matmul(&jobs[i].2, jobs[i].2.len() / c.t.k, 1))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            if let (Some((gpu, serial)), false) = (&self.gpu, gpu_jobs.is_empty()) {
                let _one = serial.lock().unwrap_or_else(|p| p.into_inner());
                let mut enc = gpu.device.create_command_encoder(&Default::default());
                // A job of more rows than a pass takes goes as several passes.
                let mut passes = Vec::new();
                let mut owner = Vec::new();
                for &i in &gpu_jobs {
                    let Proj::Gpu(g) = &self.experts[jobs[i].0][jobs[i].1] else { unreachable!() };
                    let rows = jobs[i].2.len() / g.t.k;
                    for start in (0..rows).step_by(ROWS) {
                        let count = ROWS.min(rows - start);
                        passes.push(g.record(&mut enc, &jobs[i].2[start * g.t.k..(start + count) * g.t.k], count));
                        owner.push(i);
                    }
                }
                for (i, y) in owner.into_iter().zip(read_back(gpu, enc, &passes)) {
                    out[i].get_or_insert_with(Vec::new).extend(y);
                }
            }
            for w in cpu_work {
                for (i, y) in w.join().expect("an EXL3 expert worker panicked") {
                    out[i] = Some(y);
                }
            }
        });
        out.into_iter().enumerate().map(|(i, y)| finish(i, y.expect("every job ran"))).collect()
    }
}

impl ggml_rs::exl3::Experts for Exl3MoeHost {
    fn forward(&self, x: &Tensor, logits: &Tensor, top_k: usize) -> Tensor {
        let (h, f) = (self.hidden, self.ff);
        let x = x.to_host();
        let logits = logits.to_host();
        let rows = x.numel() / h;
        let width = self.experts.len();
        // Each row's (expert, weight) assignments, in the row's own order.
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits.data()[r * width..(r + 1) * width], top_k)).collect();
        // The rows routed to each expert, in row order: (row, its place among the row's assignments).
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); width];
        for (r, a) in assign.iter().enumerate() {
            for (j, &(e, _)) in a.iter().enumerate() {
                by_expert[e].push((r, j));
            }
        }
        let used: Vec<usize> = (0..width).filter(|&e| !by_expert[e].is_empty()).collect();
        // Gate and up, every expert's rows at once.
        let mut jobs = Vec::new();
        for &e in &used {
            for which in 0..2 {
                let t = self.experts[e][which].t();
                let mut xh = vec![0f32; by_expert[e].len() * h];
                for (slot, &(r, _)) in by_expert[e].iter().enumerate() {
                    t.pre(&x.data()[r * h..(r + 1) * h], &mut xh[slot * h..(slot + 1) * h]);
                }
                jobs.push((e, which, xh));
            }
        }
        let gu = self.batch(&jobs);
        // silu(gate) * up, then down, every expert's rows at once.
        let mut jobs = Vec::new();
        for (n, &e) in used.iter().enumerate() {
            let (g, u) = (&gu[2 * n], &gu[2 * n + 1]);
            let hidden: Vec<f32> = g.iter().zip(u).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
            let t = self.experts[e][2].t();
            let mut xh = vec![0f32; by_expert[e].len() * f];
            for slot in 0..by_expert[e].len() {
                t.pre(&hidden[slot * f..(slot + 1) * f], &mut xh[slot * f..(slot + 1) * f]);
            }
            jobs.push((e, 2, xh));
        }
        let down = self.batch(&jobs);
        // Each row's experts summed in its own order, each weighted: the same sum every run.
        let mut placed: Vec<Vec<Option<&[f32]>>> = assign.iter().map(|a| vec![None; a.len()]).collect();
        for (n, &e) in used.iter().enumerate() {
            for (slot, &(r, j)) in by_expert[e].iter().enumerate() {
                placed[r][j] = Some(&down[n][slot * h..(slot + 1) * h]);
            }
        }
        let mut out = vec![0f32; rows * h];
        for (r, row) in out.chunks_exact_mut(h).enumerate() {
            for (j, &(_, w)) in assign[r].iter().enumerate() {
                let y = placed[r][j].expect("every assignment computed");
                for (o, v) in row.iter_mut().zip(y) {
                    *o += w * v;
                }
            }
        }
        Tensor::from_vec(out, vec![rows, h])
    }
}

impl WgpuBackend {
    /// A MoE layer's EXL3 experts (`experts[e]` its gate, up and down; the shared one last): each projection on the
    /// GPU while the weight budget holds it, else on the CPU.
    pub fn exl3_experts(&self, experts: Vec<[Exl3Data; 3]>) -> Result<Box<dyn ggml_rs::exl3::Experts>, String> {
        self.exl3_experts_leaving(experts, 0)
    }

    /// As `exl3_experts`, leaving `reserve` bytes of the budget for the model's other matrices (loaded after). Routed
    /// experts of one shape and bitrate go up as groups (`Exl3MoeGrouped`) while the budget holds them all.
    pub fn exl3_experts_leaving(&self, experts: Vec<[Exl3Data; 3]>, reserve: u64) -> Result<Box<dyn ggml_rs::exl3::Experts>, String> {
        if std::env::var_os("OAIY_EXL3_UNGROUPED").is_none() {
            if let Some(g) = Exl3MoeGrouped::try_new(self, &experts, reserve)? {
                return Ok(Box::new(g));
            }
        }
        let mut out = Vec::with_capacity(experts.len());
        for e in experts {
            let [g, u, d] = e;
            out.push([self.proj(g, reserve)?, self.proj(u, reserve)?, self.proj(d, reserve)?]);
        }
        Ok(Box::new(Exl3MoeHost::new(out, Some((Arc::clone(&self.gpu), Arc::clone(&self.serial))))?))
    }

    fn proj(&self, data: Exl3Data, reserve: u64) -> Result<Proj, String> {
        data.validate()?;
        let nbytes = data.words.len() as u64 * 4;
        let fits_binding = (data.svh.len() / 16 * data.tile_words / 2 * 4) as u64 <= chunk_limit(&self.gpu.limits);
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if !fits_binding || prev + nbytes > self.budget.saturating_sub(reserve) {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return Ok(Proj::Cpu(Exl3Cpu::new(data)?));
        }
        Ok(Proj::Gpu(Exl3Gpu::upload(self, data, None)))
    }
}

impl WgpuBackend {
    /// An EXL3 projection: on the GPU while the weight budget holds it, else on the CPU.
    pub fn exl3(&self, data: Exl3Data) -> Result<Arc<dyn PackedLinear>, String> {
        self.exl3_with(data, None)
    }

    pub(crate) fn exl3_with(&self, data: Exl3Data, max_tile_rows: Option<usize>) -> Result<Arc<dyn PackedLinear>, String> {
        data.validate()?;
        let (ntiles, nw) = (data.svh.len() / 16, data.tile_words / 2);
        let nbytes = data.words.len() as u64 * 4;
        let fits_binding = (ntiles * nw * 4) as u64 <= chunk_limit(&self.gpu.limits);
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if !fits_binding || prev + nbytes > self.budget {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return Ok(Arc::new(Exl3Cpu::new(data)?));
        }
        Ok(Arc::new(Exl3Gpu::upload(self, data, max_tile_rows)))
    }
}

/// An EXL3 projection on the CPU, for a computer without a GPU.
pub fn exl3_cpu(data: Exl3Data) -> Result<Arc<dyn PackedLinear>, String> {
    Ok(Arc::new(Exl3Cpu::new(data)?))
}

/// A MoE layer's EXL3 experts on the CPU, for a computer without a GPU.
pub fn exl3_experts_cpu(experts: Vec<[Exl3Data; 3]>) -> Result<Box<dyn ggml_rs::exl3::Experts>, String> {
    let mut out = Vec::with_capacity(experts.len());
    for e in experts {
        let [g, u, d] = e;
        out.push([Proj::Cpu(Exl3Cpu::new(g)?), Proj::Cpu(Exl3Cpu::new(u)?), Proj::Cpu(Exl3Cpu::new(d)?)]);
    }
    Ok(Box::new(Exl3MoeHost::new(out, None)?))
}

#[cfg(test)]
mod tests;
