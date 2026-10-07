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
    let codes: String = (0..8)
        .map(|jj| {
            let acc = if jj < 4 { "lo" } else { "hi" };
            format!("            {acc} = {acc} + xv[{}] * decode_at(code_in(q0, q1, q2, q3, at[{jj}]));\n", jj % 4)
        })
        .collect();
    let places: String = (0..8).map(|jj| format!("    at[{jj}] = places[8u * l + {jj}u];\n")).collect();
    format!(
        r#"
var<workgroup> red: array<vec2<f32>, 256>;
// every code's place in a tile (`place`), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
var<workgroup> firsts: array<u32, 32>;

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

// Code `i`'s word after `w0` (0, 1 or 2) and its shift, packed.
fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    return ((wd.x + nw - w0) % nw) | (wd.y << 8u);
}}

// A code from the four words `q0..q3` (from a lane's first) at its packed place.
fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let d = at & 3u;
    let sh = at >> 8u;
    let a = select(select(q0, q1, d == 1u), q2, d >= 2u);
    let b = select(select(q1, q2, d == 1u), q3, d >= 2u);
    return select((a << (32u - sh)) | (b >> sh), a >> (sh - 32u), sh >= 32u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}}

fn lanes(t: u32, nt: u32, ntiles: u32, ks: u32, ke: u32, tw: u32, base: u32) -> vec2<f32> {{
    let warp = t / 32u;
    let l = t % 32u;
    let nw = tw / 2u;
    let rb = 2u * (l % 4u);
    let first = window(8u * (t / 8u), tw).x;
    places[t] = place(t, tw, first);
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
    var at: array<u32, 8>;
{places}    var lo = 0.0;
    var hi = 0.0;
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = {words_at};
        let q0 = words[tile + w0];
        let q1 = words[tile + o1];
        let q2 = words[tile + o2];
        let q3 = words[tile + o3];
        let xv = array<f32, 4>(x[{x0}], x[{x1}], x[{x8}], x[{x9}]);
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
    let body = one_lanes("j * k + kt * 16u + {r}", "base + (kt * ntiles + nt) * nw", "");
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

// Code `i`'s word after `w0` (0, 1 or 2) and its shift, packed.
fn place(i: u32, tw: u32, w0: u32) -> u32 {{
    let nw = tw / 2u;
    let wd = window(i, tw);
    return ((wd.x + nw - w0) % nw) | (wd.y << 8u);
}}

// A code from the four words `q0..q3` (from a lane's first) at its packed place.
fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let d = at & 3u;
    let sh = at >> 8u;
    let a = select(select(q0, q1, d == 1u), q2, d >= 2u);
    let b = select(select(q1, q2, d == 1u), q3, d >= 2u);
    return select((a << (32u - sh)) | (b >> sh), a >> (sh - 32u), sh >= 32u);
}}

fn decode_at(code: u32) -> f32 {{
    let hx = (code & 0xffffu) * 0x83dcd12du;
    let sum = dot4U8Packed(hx, 0x01010101u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
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

/// The most rows [`g_few`] takes a block.
pub(crate) const FEW_MAX: usize = 8;

/// The matmul for a few rows of one matrix (a check of drafted tokens, a short chunk): [`g_mm_source`]'s lanes, each
/// decoding its eight codes of a tile once and summing them against every row's inputs in the order the one-row
/// kernel sums them (so each row's sums are that kernel's bit for bit). A workgroup a (tile column, block and split);
/// a block is `rows` jobs of one matrix from `order` (its unused places [`NONE`]; a block with none ends at once), each
/// job's partial sums to `part[(j * splits + s) * n..]` as [`g_mm_source`]'s. `p[0]`: n, k, tile words, splits; `p[1]`:
/// words a matrix, the pass's first block.
pub(crate) fn g_few(rows: usize) -> &'static str {
    static SOURCES: [std::sync::OnceLock<String>; FEW_MAX - 1] = [const { std::sync::OnceLock::new() }; FEW_MAX - 1];
    assert!((2..=FEW_MAX).contains(&rows), "a block of 2 to {FEW_MAX} rows");
    SOURCES[rows - 2].get_or_init(|| g_few_source(rows))
}

fn g_few_source(rows: usize) -> String {
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
// every code's place in a tile (`place`), the workgroup's threads one each: the same in every tile
var<workgroup> places: array<u32, 256>;
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
    return ((wd.x + nw - w0) % nw) | (wd.y << 8u);
}}

fn code_in(q0: u32, q1: u32, q2: u32, q3: u32, at: u32) -> u32 {{
    let d = at & 3u;
    let sh = at >> 8u;
    let a = select(select(q0, q1, d == 1u), q2, d >= 2u);
    let b = select(select(q1, q2, d == 1u), q3, d >= 2u);
    return select((a << (32u - sh)) | (b >> sh), a >> (sh - 32u), sh >= 32u);
}}

fn decode_at(code: u32) -> f32 {{
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
    places[t] = place(t, tw, first);
    if (t % 8u == 0u) {{
        firsts[t / 8u] = first;
    }}
    workgroupBarrier();
    let w0 = firsts[l];
    let o1 = (w0 + 1u) % nw;
    let o2 = (w0 + 2u) % nw;
    let o3 = (w0 + 3u) % nw;
    let a0 = places[8u * l];
    let a1 = places[8u * l + 1u];
    let a2 = places[8u * l + 2u];
    let a3 = places[8u * l + 3u];
    let a4 = places[8u * l + 4u];
    let a5 = places[8u * l + 5u];
    let a6 = places[8u * l + 6u];
    let a7 = places[8u * l + 7u];
{regs}
    for (var kt = ks + warp; kt < ke; kt = kt + 8u) {{
        let tile = base + (kt * ntiles + nt) * nw;
        let q0 = words[tile + w0];
        let q1 = words[tile + o1];
        let q2 = words[tile + o2];
        let q3 = words[tile + o3];
        let c0 = decode_at(code_in(q0, q1, q2, q3, a0));
        let c1 = decode_at(code_in(q0, q1, q2, q3, a1));
        let c2 = decode_at(code_in(q0, q1, q2, q3, a2));
        let c3 = decode_at(code_in(q0, q1, q2, q3, a3));
        let c4 = decode_at(code_in(q0, q1, q2, q3, a4));
        let c5 = decode_at(code_in(q0, q1, q2, q3, a5));
        let c6 = decode_at(code_in(q0, q1, q2, q3, a6));
        let c7 = decode_at(code_in(q0, q1, q2, q3, a7));
{sums}
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

/// A check's routed experts in blocks for [`g_few`], from its down jobs ([`DOWN_JOBS`]'s: pair `j`'s expert, `p[0].x`
/// pairs, at most 256): each expert's pairs (at most `p[0].y`, the block's rows, as a row takes an expert once) a block
/// of their own in their list's order, the blocks in the order their experts first appear; the down jobs' order
/// (`order_d`, `p[0].x` blocks of `p[0].y`) and the gate and up jobs' (`order_gu`: expert block `b`'s gate jobs `2j` in
/// block `2b`, its up jobs `2j + 1` in `2b + 1`), the other places [`NONE`]. One workgroup.
const GROUP: &str = r#"
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
/// ([`g_many`], the order and its blocks); a check's few rows' grouped on the GPU in blocks of `rows` ([`g_few`], the
/// order and the most blocks it can have), each job's sums the one-job kernel's.
#[derive(Clone, Copy)]
enum Order<'a> {
    Jobs,
    /// The order, its blocks, and the jobs a block.
    Many(&'a DeviceVec, usize, usize),
    Few(&'a DeviceVec, usize, usize),
}

/// The pipeline name of [`g_few`]'s kernel for blocks of `rows`.
pub(crate) fn few_name(rows: usize) -> &'static str {
    ["exl3-few-2", "exl3-few-3", "exl3-few-4", "exl3-few-5", "exl3-few-6", "exl3-few-7", "exl3-few-8"][rows - 2]
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
fn moe_rows_for(pairs: usize, experts: usize) -> usize {
    let target = 3 * pairs / experts.max(1);
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
const WSUM_ROWS: &str = r#"
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
const WSUM_APPLY: &str = r#"
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

/// A decode step's routing on the GPU, as `ggml_rs::exl3::route` routes (a workgroup a row `r`): the top `p[0].y` of
/// the router's `p[0].x` routed logits by logit, a tie to the lower index (each logit placed by how many beat it),
/// their weights a softmax among themselves summed in their order, the shared expert (the logit after them) weighted
/// by its gate's sigmoid; then the row's jobs as the grouped experts take them: gate and up `[2e, r, 2e + 1, r]` an
/// expert (`jobs`, from pair `r k`), the weights `w` (from `r (k + 1)`: the top k's, the shared one's last). `p[0]`:
/// routed (at most 1024), k (at most 32).
const ROUTE: &str = r#"
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(6) var<storage, read_write> jobs: array<u32>;
@group(0) @binding(7) var<storage, read_write> w: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> l: array<f32, 1024>;
// the same, four at a time (the comparisons' reads)
var<workgroup> l4: array<vec4<f32>, 256>;
var<workgroup> top: array<u32, 32>;

// How many of l4's first `quads` beat logit `v` at index `i` (a higher logit, or as high at a lower index).
fn beaten(v: f32, i: u32, quads: u32) -> u32 {
    var above = 0u;
    for (var q = 0u; q < quads; q++) {
        let u = l4[q];
        let j = 4u * q;
        above += select(0u, 1u, u.x > v || (u.x == v && j < i));
        above += select(0u, 1u, u.y > v || (u.y == v && j + 1u < i));
        above += select(0u, 1u, u.z > v || (u.z == v && j + 2u < i));
        above += select(0u, 1u, u.w > v || (u.w == v && j + 3u < i));
    }
    return above;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let r = wg.x;
    let lr = r * (n + 1u);
    let quads = (n + 3u) / 4u;
    // past the last logit, -inf (beats none)
    for (var i = t; i < 4u * quads; i += 256u) {
        var v = bitcast<f32>(0xff800000u);
        if (i < n) {
            v = logits[lr + i];
        }
        l[i] = v;
        l4[i / 4u][i % 4u] = v;
    }
    workgroupBarrier();
    for (var i = t; i < n; i += 256u) {
        let above = beaten(l[i], i, quads);
        if (above < k) {
            top[above] = i;
        }
    }
    workgroupBarrier();
    if (t == 0u) {
        let mx = l[top[0]];
        var sum = 0.0;
        for (var j = 0u; j < k; j++) {
            sum += exp(l[top[j]] - mx);
        }
        for (var j = 0u; j < k; j++) {
            let e = top[j];
            w[r * (k + 1u) + j] = exp(l[e] - mx) / sum;
            let at = 4u * (r * k + j);
            jobs[at] = 2u * e;
            jobs[at + 1u] = r;
            jobs[at + 2u] = 2u * e + 1u;
            jobs[at + 3u] = r;
        }
        w[r * (k + 1u) + k] = 1.0 / (1.0 + exp(-logits[lr + n]));
    }
}
"#;

/// A prompt's routed jobs grouped by expert on the GPU, as [`many_order`] groups them on the host: each expert's down
/// jobs in blocks of `p[0].w` (a block one expert's, its unused places [`NONE`]; the blocks in the experts' order, a
/// job's place among its expert's as the atomics fall), and their gate and up jobs in blocks `2 b` and `2 b + 1`.
/// `od`: the down order (`p[0].y` blocks), then each expert's count, first block and filled places (`p[0].z`
/// experts); `og` the gate and up order. First every place unused and every count 0.
const MANY_CLEAR: &str = r#"
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
}
"#;

/// [`MANY_CLEAR`]'s second pass: each expert's jobs counted, a thread a pair. `p[0]`: the pairs, the blocks.
const MANY_COUNT: &str = r#"
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
const MANY_SCAN: &str = r#"
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
const MANY_SCATTER: &str = r#"
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
    let b = atomicLoad(&od[at + ne + e]) + pos / bs;
    let slot = pos % bs;
    atomicStore(&od[b * bs + slot], j);
    og[2u * b * bs + slot] = 2u * j;
    og[(2u * b + 1u) * bs + slot] = 2u * j + 1u;
}
"#;

/// The down projections' jobs of routed rows from their gate and up jobs ([`ROUTE`]'s): pair `j`'s expert `e` on
/// hidden row `j`, `[e, j]`. `p[0]`: the pairs (rows times k).
const DOWN_JOBS: &str = r#"
@group(0) @binding(0) var<storage, read> gu: array<u32>;
@group(0) @binding(6) var<storage, read_write> jobs: array<u32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
    for (var j = t; j < p[0].x; j += 64u) {
        jobs[2u * j] = gu[4u * j] / 2u;
        jobs[2u * j + 1u] = j;
    }
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
    top_k: usize,
    jobs_gu: DeviceVec,
    jobs_d: DeviceVec,
    w: DeviceVec,
    xh_gu: DeviceVec,
    part_gu: DeviceVec,
    out_gu: DeviceVec,
    xh_d: DeviceVec,
    part_d: DeviceVec,
    out_d: DeviceVec,
    sg: DeviceVec,
    su: DeviceVec,
    sd: DeviceVec,
    /// A check's jobs grouped by matrix ([`GROUP`]): gate and up, down.
    order_gu: DeviceVec,
    order_d: DeviceVec,
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
                    rec.dispatch_wide(few_name(rows), g_few(rows), [&g.words, &xhb, &jb, &ob, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), these * splits));
                }
            }
            Order::Jobs => {
                for first in (0..count).step_by(per) {
                    let jobs = per.min(count - first) as u32;
                    rec.dispatch_wide("exl3-mm", chain_shader("mm"), [&g.words, &xhb, &jb, &d, &d, &d, &pb, &drw], &[g.n as u32, g.k as u32, g.tw as u32, g.splits, g.mwords as u32, first as u32], (ntiles.min(65535), ntiles.div_ceil(65535), jobs * g.splits));
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
    /// 1]`) and recorded into `out` (see `ChainRecorder::moe_routed`): [`ROUTE`] writes the jobs and weights where
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
        rec.dispatch_wide("moe-route", ROUTE, [&buf(logits), &d, &d, &d, &d, &d, &buf(&st.jobs_gu), &buf(&st.w)], &[self.routed as u32, top_k as u32], (rows as u32, 1, 1));
        rec.dispatch_wide("moe-down-jobs", DOWN_JOBS, [&buf(&st.jobs_gu), &d, &d, &d, &d, &d, &buf(&st.jobs_d), &drw], &[(rows * top_k) as u32], (1, 1, 1));
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
mod tests {
    use super::*;
    use ggml_rs::exl3::mul1;

    /// An independent packing oracle: each weight's bits written MSB first into its tile's stream, and the dense
    /// matrix they decode to (the same construction as ggml-rs-cuda's EXL3 test, which checks the CUDA kernels).
    fn oracle(tw: usize, k: usize, n: usize) -> (Exl3Data, Vec<f32>) {
        let nw = tw / 2;
        let mut words = vec![0u32; k / 16 * n / 16 * nw];
        let mut dense = vec![0f32; k * n];
        for kt in 0..k / 16 {
            for nt in 0..n / 16 {
                let mut stream = vec![];
                for i in 0..256 {
                    let bits = tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                    let v = (i * 17 + kt * 43 + nt * 11 + 3) as u32 & ((1 << bits) - 1);
                    for bit in (0..bits).rev() {
                        stream.push((v >> bit) & 1);
                    }
                }
                let base = (kt * (n / 16) + nt) * nw;
                for (i, &bit) in stream.iter().enumerate() {
                    words[base + i / 32] |= bit << (31 - i % 32);
                }
                let mut end = 0;
                for i in 0..256 {
                    end += tw / 16 + if tw % 16 == 8 { i % 2 } else { 0 };
                    let mut code = 0;
                    for j in 0..16 {
                        code = (code << 1) | stream[(end + stream.len() - 16 + j) % stream.len()];
                    }
                    let lane = i / 8;
                    let j = i % 8;
                    let r = (lane % 4) * 2 + (j % 2) + if j & 2 != 0 { 8 } else { 0 };
                    let c = lane / 4 + if j & 4 != 0 { 8 } else { 0 };
                    dense[(kt * 16 + r) * n + nt * 16 + c] = mul1(code);
                }
            }
        }
        let data = Exl3Data {
            words,
            suh: (0..k).map(|i| if i % 3 == 0 { -0.5 } else { 0.5 }).collect(),
            svh: (0..n).map(|i| if i % 5 == 0 { -0.25 } else { 0.25 }).collect(),
            tile_words: tw,
            input_map: (0..k as u32).rev().collect(),
            output_map: (0..n as u32).rev().collect(),
        };
        (data, dense)
    }

    fn inputs(rows: usize, k: usize) -> Vec<f32> {
        (0..rows * k).map(|i| ((i % k * 7 + i / k * 13) % 23) as f32 / 32.0 - 0.25).collect()
    }

    /// The projection the oracle's dense matrix gives, step by step as exllamav3 rounds.
    fn expected(data: &Exl3Data, dense: &[f32], x: &[f32]) -> Vec<f32> {
        let (k, n) = (data.suh.len(), data.svh.len());
        let mut out = vec![];
        for row in x.chunks_exact(k) {
            let mut xh: Vec<f32> = data.input_map.iter().enumerate().map(|(i, &j)| half(row[j as usize]) * data.suh[i]).collect();
            had(&mut xh);
            for v in &mut xh {
                *v = half(*v * ISQRT128);
            }
            let mut y: Vec<f32> = (0..n).map(|j| half((0..k).map(|i| xh[i] * dense[i * n + j]).sum())).collect();
            had(&mut y);
            out.extend(data.output_map.iter().map(|&j| half(y[j as usize] * ISQRT128 * data.svh[j as usize])));
        }
        out
    }

    fn close(actual: &[f32], expected: &[f32], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 0.003, "{what} [{i}]: {a} != {e}");
        }
    }

    const RATES: [usize; 11] = [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128];

    #[test]
    fn the_cpu_projection_matches_the_independent_oracle_at_every_rate() {
        for tw in RATES {
            let (k, n) = (128, 256);
            let (data, dense) = oracle(tw, k, n);
            for i in 0..k {
                for j in 0..n {
                    assert_eq!(data.value(i, j), dense[i * n + j], "the oracle and Exl3Data::value disagree: tw={tw} k={i} n={j}");
                    let (w0, w1, sh) = positions(tw)[(i % 16) * 16 + j % 16];
                    let base = ((i / 16) * (n / 16) + j / 16) * (tw / 2);
                    let pair = ((data.words[base + w0] as u64) << 32) | data.words[base + w1] as u64;
                    assert_eq!(decode((pair >> sh) as u32 & 0xffff), dense[i * n + j], "decode: tw={tw} k={i} n={j}");
                }
            }
            for rows in [1, 5] {
                let x = inputs(rows, k);
                let want = expected(&data, &dense, &x);
                let cpu = Exl3Cpu::new(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
                let y = cpu.linear(&Tensor::from_vec(x, vec![rows, k]));
                assert_eq!(y.shape(), &[rows, n]);
                close(y.data(), &want, &format!("cpu tw={tw} rows={rows}"));
            }
        }
    }

    fn backend() -> Option<WgpuBackend> {
        match WgpuBackend::new(Some(1 << 30)) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping WebGPU EXL3 tests: {e}");
                None
            }
        }
    }

    #[test]
    fn the_gpu_projection_matches_the_independent_oracle_at_every_rate() {
        let Some(b) = backend() else { return };
        for tw in RATES {
            let (k, n) = (128, 256);
            let (data, dense) = oracle(tw, k, n);
            // More rows than one pass takes (32), and fewer.
            for rows in [1, 5, 37] {
                let x = inputs(rows, k);
                let want = expected(&data, &dense, &x);
                let w = b.exl3(Exl3Data { words: data.words.clone(), suh: data.suh.clone(), svh: data.svh.clone(), tile_words: tw, input_map: data.input_map.clone(), output_map: data.output_map.clone() }).unwrap();
                assert!(format!("{w:?}").starts_with("Exl3Gpu"), "on the GPU: {w:?}");
                let y = w.linear(&Tensor::from_vec(x, vec![rows, k]));
                close(y.data(), &want, &format!("gpu tw={tw} rows={rows}"));
            }
        }
    }

    #[test]
    fn a_projection_over_several_buffers_and_splits_agrees_with_the_cpu() {
        let Some(b) = backend() else { return };
        // A model-like width, buffers of 3 tile rows (as a small binding limit would make them), split work.
        let (k, n, tw) = (1024, 512, 48);
        let mut seed = 13_234_567u32;
        let words: Vec<u32> = (0..k / 16 * n / 16 * (tw / 2))
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                seed
            })
            .collect();
        let data = || Exl3Data {
            words: words.clone(),
            suh: (0..k).map(|i| if i % 3 == 0 { -0.25 } else { 0.25 }).collect(),
            svh: vec![0.125; n],
            tile_words: tw,
            input_map: (0..k as u32).rev().collect(),
            output_map: (0..n as u32).rev().collect(),
        };
        let gpu = b.exl3_with(data(), Some(3)).unwrap();
        assert!(format!("{gpu:?}").contains("chunk(s)") && !format!("{gpu:?}").contains(" 1 chunk"), "{gpu:?}");
        let cpu = exl3_cpu(data()).unwrap();
        let x = Tensor::from_vec((0..3 * k).map(|i| ((i * 17 % 73) as f32 - 36.0) / 37.0).collect(), vec![3, k]);
        let (g, c) = (gpu.linear(&x), cpu.linear(&x));
        close(g.data(), c.data(), "gpu vs cpu");
    }

    fn random_exl3(k: usize, n: usize, tw: usize, seed: u32) -> Exl3Data {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        };
        let words = (0..k / 16 * n / 16 * (tw / 2)).map(|_| next()).collect();
        let mut scale = |len: usize, mag: f32| (0..len).map(|_| if next() & 1 == 0 { mag } else { -mag }).collect::<Vec<f32>>();
        Exl3Data { suh: scale(k, 0.25), svh: scale(n, 0.125), words, tile_words: tw, input_map: (0..k as u32).collect(), output_map: (0..n as u32).collect() }
    }

    /// `count` experts plus the shared one, each `[gate, up, down]`.
    fn experts(count: usize, hidden: usize, ff: usize, tw: usize) -> Vec<[Exl3Data; 3]> {
        (0..=count as u32).map(|e| [random_exl3(hidden, ff, tw, 11 + e * 3), random_exl3(hidden, ff, tw, 12 + e * 3), random_exl3(ff, hidden, tw, 13 + e * 3)]).collect()
    }

    /// The MoE as its definition reads, from standalone projections: each row's top-k routed experts by logit (ties to
    /// the lower index), softmax-weighted, plus the shared expert by the sigmoid of its gate.
    fn reference(experts: Vec<[Exl3Data; 3]>, x: &[f32], logits: &[f32], top_k: usize) -> Vec<f32> {
        let p: Vec<[Exl3Cpu; 3]> = experts.into_iter().map(|[g, u, d]| [Exl3Cpu::new(g).unwrap(), Exl3Cpu::new(u).unwrap(), Exl3Cpu::new(d).unwrap()]).collect();
        let (h, width) = (p[0][0].t.k, p.len());
        let mut out = vec![];
        for (r, row) in x.chunks_exact(h).enumerate() {
            let l = &logits[r * width..(r + 1) * width];
            let mut idx: Vec<usize> = (0..width - 1).collect();
            idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
            let top = &idx[..top_k];
            let z: f32 = top.iter().map(|&e| (l[e] - l[top[0]]).exp()).sum();
            let mut picks: Vec<(usize, f32)> = top.iter().map(|&e| (e, (l[e] - l[top[0]]).exp() / z)).collect();
            picks.push((width - 1, 1.0 / (1.0 + (-l[width - 1]).exp())));
            let mut acc = vec![0f32; h];
            let xr = Tensor::from_vec(row.to_vec(), vec![1, h]);
            for (e, w) in picks {
                let g = p[e][0].linear(&xr);
                let u = p[e][1].linear(&xr);
                let hid: Vec<f32> = g.data().iter().zip(u.data()).map(|(&g, &u)| g / (1.0 + (-g).exp()) * u).collect();
                let f = hid.len();
                let d = p[e][2].linear(&Tensor::from_vec(hid, vec![1, f]));
                for (a, v) in acc.iter_mut().zip(d.data()) {
                    *a += w * v;
                }
            }
            out.extend(acc);
        }
        out
    }

    #[test]
    fn a_moe_layer_routes_and_mixes_as_its_definition_on_the_gpu_on_the_cpu_and_split_between_them() {
        let (count, hidden, ff, top_k) = (8, 256, 128, 3);
        for tw in [48, 80] {
            for rows in [1, 7] {
                let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
                let logits: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 29) as f32 - 14.0) / 7.0).collect();
                let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
                let xt = Tensor::from_vec(x.clone(), vec![rows, hidden]);
                let lt = Tensor::from_vec(logits.clone(), vec![rows, count + 1]);
                let cpu = exl3_experts_cpu(experts(count, hidden, ff, tw)).unwrap();
                close(cpu.forward(&xt, &lt, top_k).data(), &want, &format!("cpu moe tw={tw} rows={rows}"));
                let Some(b) = backend() else { continue };
                // the routed experts as groups (one shape and bitrate), their transforms on the GPU
                let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
                assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
                close(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("grouped moe tw={tw} rows={rows}"));
                if rows == 1 {
                    // chained, a step's kept scratch, twice
                    let assign = vec![route(&logits, top_k)];
                    let (xd, out) = (b.vec(hidden), b.vec(hidden));
                    DeviceChain::upload(&b, &xd, &x);
                    for _ in 0..2 {
                        let mut rec = b.begin();
                        rec.moe_rows(gpu.as_ref(), &xd, &out, &assign);
                        rec.read(&out);
                        close(&rec.finish().pop().unwrap(), &want, &format!("chained moe tw={tw}"));
                    }
                }
                drop(gpu);
                // projections a dispatch each, as a layer whose experts differ in shape or bitrate goes
                std::env::set_var("OAIY_EXL3_UNGROUPED", "1");
                let host = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
                std::env::remove_var("OAIY_EXL3_UNGROUPED");
                assert!(format!("{host:?}").contains(&format!("{} of {} projections on the GPU", 3 * (count + 1), 3 * (count + 1))), "{host:?}");
                close(host.forward(&xt, &lt, top_k).data(), &want, &format!("gpu moe tw={tw} rows={rows}"));
                drop(host);
                // A budget for some of the experts: the rest decode on the CPU, and the layer is the same.
                let one = (hidden * ff * tw / 128) as u64;
                let Ok(small) = WgpuBackend::new(Some(one * 10)) else { continue };
                let split = small.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
                assert!(format!("{split:?}").contains("10 of 27 projections on the GPU"), "{split:?}");
                close(split.forward(&xt, &lt, top_k).data(), &want, &format!("split moe tw={tw} rows={rows}"));
            }
        }
    }

    #[test]
    fn routing_takes_the_top_k_ties_to_the_lower_index_and_the_shared_expert_by_its_gate() {
        let r = route(&[1.0, 3.0, 3.0, 2.0, 0.0], 2);
        assert_eq!(r.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 4]);
        assert!((r[0].1 - 0.5).abs() < 1e-6 && (r[1].1 - 0.5).abs() < 1e-6, "{r:?}");
        assert!((r[2].1 - 0.5).abs() < 1e-6, "sigmoid(0) for the shared expert: {r:?}");
        let r = route(&[0.0, 0.0, 0.0, 5.0], 1);
        assert_eq!(r[0], (0, 1.0), "a three-way tie goes to the lowest index");
    }

    #[test]
    fn the_cpu_projection_spreads_any_width_over_any_threads() {
        // 40 tile columns over 32 threads: two each, so the last threads have none (it panicked on a prompt's last row).
        let (k, n, tw) = (256, 640, 48);
        let data = random_exl3(k, n, tw, 7);
        let x: Vec<f32> = (0..9 * k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 20.0).collect();
        let cpu = Exl3Cpu::new(data).unwrap();
        let mut xh = vec![0f32; 9 * k];
        for (row, out) in x.chunks_exact(k).zip(xh.chunks_exact_mut(k)) {
            cpu.t.pre(row, out);
        }
        let one = cpu.matmul(&xh, 9, 1);
        for threads in [2, 3, 7, 32, 64] {
            let many = cpu.matmul(&xh, 9, threads);
            close(&many, &one, &format!("{threads} threads"));
        }
    }

    /// How long a decode step's projection takes, and how much of it is waiting for the GPU (`--ignored --nocapture`).
    #[test]
    #[ignore = "timing"]
    fn time_a_decode_projection() {
        let Some(b) = backend() else { return };
        for (k, n) in [(2560, 10240), (6144, 2560), (2560, 640)] {
            let w = b.exl3(random_exl3(k, n, 48, 5)).unwrap();
            let x = Tensor::from_vec(vec![0.1; k], vec![1, k]);
            for _ in 0..5 {
                w.linear(&x);
            }
            let t = std::time::Instant::now();
            for _ in 0..50 {
                w.linear(&x);
            }
            let each = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
            // The same with nothing to compute: an empty submit and wait.
            let t = std::time::Instant::now();
            for _ in 0..50 {
                b.gpu.queue().submit([]);
                b.gpu.wait(None);
            }
            let idle = t.elapsed().as_secs_f64() * 1000.0 / 50.0;
            // As in a model: the host computes between calls (here 3 ms of spinning), so the GPU waits idle between them.
            let spin = |ms: f64| {
                let t = std::time::Instant::now();
                while t.elapsed().as_secs_f64() * 1000.0 < ms {
                    std::hint::spin_loop();
                }
            };
            let mut gapped = 0.0;
            for _ in 0..50 {
                spin(3.0);
                let t = std::time::Instant::now();
                w.linear(&x);
                gapped += t.elapsed().as_secs_f64() * 1000.0;
            }
            // And with every other core busy, as a model's host threads keep them.
            let stop = std::sync::atomic::AtomicBool::new(false);
            let busy = std::thread::scope(|s| {
                for _ in 1..std::thread::available_parallelism().map_or(4, |n| n.get()) {
                    s.spawn(|| {
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            std::hint::spin_loop();
                        }
                    });
                }
                let t = std::time::Instant::now();
                for _ in 0..50 {
                    w.linear(&x);
                }
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                t.elapsed().as_secs_f64() * 1000.0 / 50.0
            });
            eprintln!("{k}x{n}: {each:.3} ms a call back to back; {:.3} ms after 3 ms of host work; {busy:.3} ms with every core busy; an empty submit and wait {idle:.3} ms", gapped / 50.0);
        }
    }

    #[test]
    fn a_weight_beyond_the_budget_runs_on_the_cpu_and_the_budget_is_returned() {
        let Ok(b) = WgpuBackend::new(Some(4096)) else { return };
        let (data, _) = oracle(48, 128, 256);
        let w = b.exl3(data).unwrap();
        assert!(format!("{w:?}").starts_with("Exl3Cpu"), "{w:?}");
        assert_eq!(b.usage().0, 0);
        let Some(b) = backend() else { return };
        let (data, _) = oracle(48, 128, 256);
        let w = b.exl3(data).unwrap();
        assert!(b.usage().0 > 0);
        drop(w);
        assert_eq!(b.usage().0, 0, "dropping the weight returns its bytes");
    }

    /// A projection chained on the GPU, its transforms there too (the maps gathered, the Hadamard transforms and their
    /// f16 roundings), gives the projection's own answer (its transforms on the host) bit for bit: one row (a step's,
    /// its scratch kept), a few rows (a check of drafts: each row as one row alone), and a prompt's rows as its passes
    /// take them (32 at a time, a last lone row as one row), with and without maps, at 3 and 5 bits.
    #[test]
    fn a_chained_projection_matches_the_projection() {
        let Some(b) = backend() else { return };
        let (k, n) = (512usize, 384usize);
        for (tw, maps) in [(48usize, false), (80, true)] {
            let mut data = random_exl3(k, n, tw, 77 + tw as u32);
            if maps {
                data.input_map = (0..k as u32).map(|i| (i * 7 + 3) % k as u32).collect();
                data.output_map = (0..n as u32).rev().collect();
            }
            let w = b.exl3(data).unwrap();
            assert!(b.holds_exl3(w.as_ref()), "the adapter holds it");
            for rows in [1usize, 2, 3, 5, 8, 9, 32, 33, 70, 129] {
                let xs: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect();
                let want = if (2..=FEW_MAX).contains(&rows) {
                    let each: Vec<f32> = xs.chunks_exact(k).flat_map(|r| w.linear(&Tensor::from_vec(r.to_vec(), vec![1, k])).data().to_vec()).collect();
                    Tensor::from_vec(each, vec![rows, n])
                } else {
                    w.linear(&Tensor::from_vec(xs.clone(), vec![rows, k]))
                };
                let (x, y) = (b.vec(rows * k), b.vec(rows * n));
                DeviceChain::upload(&b, &x, &xs);
                // twice: the second from the kept scratch
                for _ in 0..2 {
                    let mut rec = b.begin();
                    rec.exl3_rows(w.as_ref(), &x, &y, rows);
                    rec.read(&y);
                    let got = rec.finish().pop().unwrap();
                    if rows > FEW_MAX && coop_on(&b.gpu) {
                        // a prompt's rows on the tensor cores: a matmul's sums (the same products, f16 exactly)
                        let dot: f64 = got.iter().zip(want.data()).map(|(a, e)| *a as f64 * *e as f64).sum();
                        let norm = |v: &[f32]| v.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
                        let cos = dot / (norm(&got) * norm(want.data()));
                        let top = want.data().iter().fold(0f32, |m, v| m.max(v.abs()));
                        let worst = got.iter().zip(want.data()).fold(0f32, |m, (a, e)| m.max((a - e).abs()));
                        assert!(cos > 0.99999 && worst <= 2e-3 * top, "tw={tw} maps={maps} rows={rows}: cosine {cos}, worst {worst} of {top}");
                        continue;
                    }
                    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                    assert_eq!(bits(&got), bits(want.data()), "tw={tw} maps={maps} rows={rows}");
                }
            }
        }
    }
    /// A chained projection's time for a few rows (`--ignored --nocapture`): Qwen3.8-Flash-Next's delta nets' qkv, z and
    /// output, its attention's q, and an expert's gate and down, at 3 bits, matrices in turn past the L2 as a model's
    /// layers are.
    #[test]
    #[ignore = "a measurement"]
    fn measure_chained_projections() {
        let Some(b) = backend() else { return };
        for (k, n, count) in [(2560usize, 10240usize, 8usize), (2560, 6144, 12), (6144, 2560, 12), (2560, 12288, 8), (2560, 640, 96), (640, 2560, 96)] {
            let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 900 + i)).unwrap()).collect();
            let bytes = (k / 16 * n / 16 * 24 * 4) as f64;
            let mut line = format!("[{n}, {k}]:");
            let mut one = 0.0;
            for rows in [1usize, 2, 3, 4, 8] {
                let (x, y) = (b.vec(rows * k), b.vec(rows * n));
                DeviceChain::upload(&b, &x, &(0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
                let reps = 4 * count;
                let run = || {
                    let mut rec = b.begin();
                    for i in 0..reps {
                        rec.exl3_rows(ws[i % count].as_ref(), &x, &y, rows);
                    }
                    rec.read_range(&y, 0, 1);
                    rec.finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                let each = t.elapsed().as_secs_f64() / 5.0 / reps as f64;
                if rows == 1 {
                    one = each;
                }
                line += &format!(" {rows} rows {:.1} us ({:.2}x, {:.0} GB/s);", each * 1e6, each / one, bytes / each / 1e9);
            }
            eprintln!("{line}");
        }
    }

    /// The few-rows kernel's cost by how many of a block's places hold jobs, against the one-job kernel's
    /// (`--ignored --nocapture`): an expert's gate (640 x 2560) and a delta net's qkv (10240 x 2560) at 3 bits, matrices
    /// in turn past the L2; and a dispatch of empty blocks.
    #[test]
    #[ignore = "a measurement"]
    fn measure_few_blocks() {
        let Some(b) = backend() else { return };
        for (k, n, count) in [(2560usize, 640usize, 96usize), (2560, 10240, 8)] {
            let ws: Vec<_> = (0..count as u32).map(|i| b.exl3(random_exl3(k, n, 48, 700 + i)).unwrap()).collect();
            let g: Vec<&Exl3Gpu> = ws.iter().map(|w| w.as_any().unwrap().downcast_ref::<Exl3Gpu>().unwrap()).collect();
            let splits = g[0].single_chunk().unwrap().1;
            let (xh, part) = (b.vec(8 * k), b.vec(8 * splits as usize * n));
            DeviceChain::upload(&b, &xh, &(0..8 * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
            let jobs = u32_vec(&b, &(0..8u32).flat_map(|r| [0, r]).collect::<Vec<_>>());
            let ntiles = (n / 16) as u32;
            let reps = 4 * count;
            let time = |name: &'static str, body: &str, order: Option<&DeviceVec>, blocks: u32| {
                let run = || {
                    use ggml_rs::ChainRecorder;
                    let mut rec = crate::chain::Recorder::new(&b);
                    let d = rec.gpu().dummy().clone();
                    let drw = rec.gpu().dummy_rw().clone();
                    let buf = |v: &DeviceVec| v.inner.downcast_ref::<wgpu::Buffer>().unwrap().clone();
                    for i in 0..reps {
                        let (words, _) = g[i % count].single_chunk().unwrap();
                        let ob = order.map(buf).unwrap_or_else(|| d.clone());
                        rec.dispatch_wide(name, body, [words, &buf(&xh), &buf(&jobs), &ob, &d, &d, &buf(&part), &drw], &[n as u32, k as u32, 48, splits, 0, 0], (ntiles, 1, blocks * splits));
                    }
                    rec.read_range(&part, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                t.elapsed().as_secs_f64() / 5.0 / reps as f64 * 1e6
            };
            let one = time("exl3-mm", &chain_shader("mm"), None, 1);
            let mut line = format!("[{n}, {k}]: a job {one:.1} us;");
            for (rows, used) in [(4usize, 1usize), (4, 2), (4, 4), (2, 1), (2, 2), (8, 1)] {
                let order: Vec<u32> = (0..rows as u32).map(|r| if (r as usize) < used { r } else { NONE }).collect();
                let ov = u32_vec(&b, &order);
                line += &format!(" few-{rows} with {used} {:.1} us;", time(few_name(rows), g_few(rows), Some(&ov), 1));
            }
            let empty = u32_vec(&b, &vec![NONE; 4 * 40]);
            line += &format!(" 40 empty blocks of 4 {:.1} us", time(few_name(4), g_few(4), Some(&empty), 40));
            eprintln!("{line}");
        }
    }

    /// A check's experts as the GPU takes them in one pass (`--ignored --nocapture`): 4 rows of 10 of 128 experts
    /// (Qwen3.8-Flash-Next's 2560 x 640 at 3 bits), their gate and up jobs a job a workgroup set, or grouped by matrix
    /// in blocks of 4 (no empty blocks), as the rows share none, some or all of them.
    #[test]
    #[ignore = "a measurement"]
    fn measure_grouped_check_experts() {
        let Some(b) = backend() else { return };
        let (count, hidden, ff, rows, k) = (128usize, 2560usize, 640usize, 4usize, 10usize);
        let moe = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
        let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
        let st = g.scratch(&mut |n| b.vec(n), rows, k, true);
        let x = b.vec(rows * hidden);
        DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        for (what, shared) in [("none shared", 0usize), ("3 of 10 shared", 3), ("all shared", 10)] {
            // row r's experts: the shared ones, then its own
            let picks: Vec<Vec<usize>> = (0..rows).map(|r| (0..k).map(|j| if j < shared { j } else { 10 + r * 25 + j }).collect()).collect();
            let jobs: Vec<u32> = picks.iter().enumerate().flat_map(|(r, p)| p.iter().flat_map(move |&e| [2 * e as u32, r as u32, 2 * e as u32 + 1, r as u32])).collect();
            upload_u32(&b, &st.jobs_gu, &jobs);
            let n = jobs.len() / 2;
            // blocks of a matrix's jobs, as GROUP makes them
            let mut seen: Vec<(u32, Vec<u32>)> = Vec::new();
            for q in 0..n {
                let m = jobs[2 * q];
                match seen.iter_mut().find(|(mm, _)| *mm == m) {
                    Some((_, list)) => list.push(q as u32),
                    None => seen.push((m, vec![q as u32])),
                }
            }
            let order: Vec<u32> = seen.iter().flat_map(|(_, list)| (0..rows).map(|i| list.get(i).copied().unwrap_or(NONE))).collect();
            upload_u32(&b, &st.order_gu, &order);
            let blocks = seen.len();
            let time = |few: bool| {
                let run = || {
                    use ggml_rs::ChainRecorder;
                    let mut rec = crate::chain::Recorder::new(&b);
                    for _ in 0..48 {
                        let o = if few { Order::Few(&st.order_gu, blocks, rows) } else { Order::Jobs };
                        g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, o, &st.xh_gu, &st.part_gu, &st.out_gu);
                    }
                    rec.read_range(&st.out_gu, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    run();
                }
                t.elapsed().as_secs_f64() / 5.0 / 48.0 * 1e6
            };
            eprintln!("{what}: {n} jobs, {blocks} matrices: a job each {:.1} us a pass, grouped {:.1} us", time(false), time(true));
        }
    }

    /// What a prompt's grouped experts' gate and up matmul takes on the tensor cores (`--ignored --nocapture`, with
    /// OAIY_CHAIN_PROFILE for each kernel's GPU time): 128 of Qwen3.8-Flash-Next's experts (2,560 by 640, 3 bits), 10,
    /// 20 or 40 jobs each (a chunk of 512, 1,024 or 2,048 rows over its 512), in blocks of 16 to 128; then the same
    /// blocks over 8 matrices (their words in the L2 throughout). On an RTX 5090 a block's time is its decode's, its
    /// rows nearly free: 10 jobs in blocks of 32 0.47 ms, 20 0.51; but blocks of 64 or 128 0.83 and 0.94 for the same
    /// blocks (their registers leave a workgroup an SM where 32's leave two: 32's padded to one an SM is 0.80), and the
    /// words in the L2 0.42 where 0.50. Neither skipping a block's empty fragments' multiply-adds nor the decode's
    /// conversion done in bits changed it.
    #[test]
    #[ignore = "a measurement"]
    fn measure_prompt_expert_blocks() {
        let Some(b) = backend() else { return };
        if !coop_on(&b.gpu) {
            return;
        }
        let (count, hidden, ff, tw, top_k) = (128usize, 2560usize, 640usize, 48usize, 10usize);
        let moe = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
        let g = moe.as_any().unwrap().downcast_ref::<Exl3MoeGrouped>().unwrap();
        for per in [10usize, 20, 40] {
            let rows = per * count / top_k;
            let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
            let x = b.vec(rows * hidden);
            DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
            // row r's experts r k to r k + k - 1 (of the count, around): `per` rows each
            let jobs: Vec<u32> = (0..rows)
                .flat_map(|r| {
                    (0..top_k).flat_map(move |j| {
                        let e = ((r * top_k + j) % count) as u32;
                        [2 * e, r as u32, 2 * e + 1, r as u32]
                    })
                })
                .collect();
            upload_u32(&b, &st.jobs_gu, &jobs);
            let n = jobs.len() / 2;
            for bs in [16usize, 32, 64, 128] {
                let order = many_order(&jobs, bs);
                let ob = b.vec(order.len());
                upload_u32(&b, &ob, &order);
                let blocks = order.len() / bs;
                let run = || {
                    let mut rec = crate::chain::Recorder::new(&b);
                    for _ in 0..8 {
                        g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
                    }
                    use ggml_rs::ChainRecorder;
                    rec.read_range(&st.out_gu, 0, 1);
                    Box::new(rec).finish();
                };
                run();
                let _ = crate::profile::take_kernels();
                let t = std::time::Instant::now();
                for _ in 0..3 {
                    run();
                }
                let ms = t.elapsed().as_secs_f64() / 24.0 * 1e3;
                let k = crate::profile::take_kernels();
                let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
                eprintln!("{per} jobs an expert ({rows} rows), blocks of {bs} ({blocks}): a pass {ms:.3} ms, its matmul {mm:.3} ms on the GPU");
            }
        }
        // the same blocks (256 of 10 jobs in places for 32) over 4 experts' matrices (their words in the L2 throughout)
        let (rows, bs, blocks, per) = (256usize, 32usize, 256usize, 10usize);
        let st = g.scratch(&mut |n| b.vec(n), rows, top_k, false);
        let x = b.vec(rows * hidden);
        DeviceChain::upload(&b, &x, &(0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect::<Vec<_>>());
        for (what, spread) in [("256 matrices", 256u32), ("8 matrices", 8)] {
            let jobs: Vec<u32> = (0..blocks * per).flat_map(|j| [((j / per) as u32) % spread, (j % rows) as u32]).collect();
            upload_u32(&b, &st.jobs_gu, &jobs);
            let order: Vec<u32> = (0..blocks).flat_map(|blk| (0..bs).map(move |i| if i < per { (blk * per + i) as u32 } else { NONE })).collect();
            let ob = b.vec(order.len());
            upload_u32(&b, &ob, &order);
            let n = blocks * per;
            let run = || {
                let mut rec = crate::chain::Recorder::new(&b);
                for _ in 0..8 {
                    g.group_pass(&mut rec, &g.gu, &x, false, &st.jobs_gu, n, Order::Many(&ob, blocks, bs), &st.xh_gu, &st.part_gu, &st.out_gu);
                }
                use ggml_rs::ChainRecorder;
                rec.read_range(&st.out_gu, 0, 1);
                Box::new(rec).finish();
            };
            run();
            let _ = crate::profile::take_kernels();
            for _ in 0..3 {
                run();
            }
            let k = crate::profile::take_kernels();
            let mm: f64 = k.iter().filter(|e| e.0.starts_with("exl3-coop")).map(|e| e.1).sum::<f64>() / 24.0;
            eprintln!("{blocks} blocks of {per} jobs over {what}: the matmul {mm:.3} ms on the GPU");
        }
    }

    /// A prompt's rows through grouped experts, each expert's in blocks of 32 (a tile decoded once a block): an expert
    /// with more rows than a block (two blocks), one with a single row, the rest a few each, as the definition gives.
    #[test]
    fn grouped_experts_take_a_prompts_rows_in_blocks_as_the_definition() {
        let Some(b) = backend() else { return };
        let (count, hidden, ff, top_k, rows) = (6, 256, 128, 2, 40);
        // expert 0 in the top two of rows 0..35, expert 5 only in row 39's, the others by a spread
        let logits: Vec<f32> = (0..rows)
            .flat_map(|r| {
                (0..=count).map(move |e| match e {
                    0 if r < 35 => 5.0,
                    5 if r == 39 => 5.0,
                    5 => -5.0,
                    e if e == count => 0.3,
                    e => ((r * 7 + e * 3) % 11) as f32 / 11.0,
                })
            })
            .collect();
        let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&logits[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
        let on = |e: usize| assign.iter().filter(|a| a[..top_k].iter().any(|p| p.0 == e)).count();
        assert!(on(0) > 32 && on(5) == 1, "expert 0 on {} rows, expert 5 on {}", on(0), on(5));
        let x: Vec<f32> = (0..rows * hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        for tw in [48, 80] {
            let want = reference(experts(count, hidden, ff, tw), &x, &logits, top_k);
            let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
            assert!(format!("{gpu:?}").contains("Exl3MoeGrouped"), "{gpu:?}");
            let got = gpu.forward(&Tensor::from_vec(x.clone(), vec![rows, hidden]), &Tensor::from_vec(logits.clone(), vec![rows, count + 1]), top_k);
            close(got.data(), &want, &format!("grouped moe of a prompt tw={tw}"));
        }
    }

    #[test]
    fn a_prompts_jobs_go_by_matrix_in_blocks_of_32() {
        // matrix 1 on 33 jobs, matrix 0 on 2, matrix 7 on 1: in their list's order within a matrix
        let mut jobs = vec![];
        for j in 0..36u32 {
            let m = match j {
                3 | 20 => 0,
                35 => 7,
                _ => 1,
            };
            jobs.extend([m, j]);
        }
        let order = many_order(&jobs, 32);
        assert_eq!(order.len(), 4 * 32, "a block for matrix 0, two for matrix 1, one for matrix 7");
        assert_eq!(&order[..3], &[3, 20, NONE]);
        assert!(order[2..32].iter().all(|&j| j == NONE));
        let ones: Vec<u32> = (0..35).filter(|&j| j != 3 && j != 20).collect();
        assert_eq!(&order[32..64], &ones[..32]);
        assert_eq!(&order[64..65], &ones[32..]);
        assert!(order[65..96].iter().all(|&j| j == NONE));
        assert_eq!(order[96], 35);
        assert!(order[97..].iter().all(|&j| j == NONE));
    }

    /// A prompt's experts routed and grouped by expert on the GPU (where the tensor cores take a prompt's blocks) give
    /// what routing and grouping them on the host gives, and the MoE's definition: rows more than a check's, experts
    /// on more rows than a block's 16 and on none, the sums into the streams as well.
    #[test]
    fn a_prompts_experts_routed_and_grouped_on_the_gpu_are_the_hosts() {
        let Some(b) = backend() else { return };
        if !coop_on(&b.gpu) {
            return;
        }
        let (count, hidden, ff, top_k) = (40, 256, 128, 6);
        let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
        for rows in [9usize, 40, 130] {
            let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
            // expert 3 on every row, expert 39 on none, the rest spread
            let ls: Vec<f32> = (0..rows * (count + 1))
                .map(|i| match i % (count + 1) {
                    3 => 4.0,
                    39 => -9.0,
                    e => (((i / (count + 1)) * 7 + e * 13) % 31) as f32 / 10.0 - 1.5,
                })
                .collect();
            let assign: Vec<Vec<(usize, f32)>> = (0..rows).map(|r| route(&ls[r * (count + 1)..(r + 1) * (count + 1)], top_k)).collect();
            let (xr, lr, on_gpu, on_host) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xr, &xs);
            DeviceChain::upload(&b, &lr, &ls);
            let mut rec = b.begin();
            rec.keep_groups(false);
            assert!(rec.moe_routed(gpu.as_ref(), &xr, &on_gpu, &lr, top_k, rows), "{rows} rows route on the GPU");
            rec.moe_rows(gpu.as_ref(), &xr, &on_host, &assign);
            rec.read(&on_gpu);
            rec.read(&on_host);
            let mut got = rec.finish();
            let (h, g) = (got.pop().unwrap(), got.pop().unwrap());
            let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
            for (i, (a, e)) in g.iter().zip(&h).enumerate() {
                assert!((a - e).abs() <= 1e-5 * scale, "{rows} rows [{i}]: {a} against {e}");
            }
            close(&g, &reference(experts(count, hidden, ff, 48), &xs, &ls, top_k), &format!("{rows} rows routed on the GPU against the definition"));
        }
    }

    /// A step's experts routed on the GPU from the router's logits give what routing on the host gives: the same
    /// experts (ties to the lower index; logits with few distinct values, so ties are common), their weights within an
    /// ulp of exp; and the MoE's definition.
    #[test]
    fn experts_routed_on_the_gpu_are_the_hosts() {
        let Some(b) = backend() else { return };
        let (count, hidden, ff, top_k) = (40, 256, 128, 6);
        let gpu = b.exl3_experts(experts(count, hidden, ff, 48)).unwrap();
        let x: Vec<f32> = (0..hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) / 60.0).collect();
        let xd = b.vec(hidden);
        DeviceChain::upload(&b, &xd, &x);
        let (on_host, on_gpu, ld) = (b.vec(hidden), b.vec(hidden), b.vec(count + 1));
        let mut seed = 7u32;
        for case in 0..12 {
            let levels = [4u32, 1 << 16][case % 2];
            let logits: Vec<f32> = (0..=count)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) % levels) as f32 / levels as f32 * 6.0 - 3.0
                })
                .collect();
            DeviceChain::upload(&b, &ld, &logits);
            for keep in [true, false] {
                let mut rec = b.begin();
                rec.keep_groups(keep);
                rec.moe_rows(gpu.as_ref(), &xd, &on_host, &[route(&logits, top_k)]);
                assert!(rec.moe_routed(gpu.as_ref(), &xd, &on_gpu, &ld, top_k, 1), "the grouped experts route on the GPU");
                rec.read(&on_host);
                rec.read(&on_gpu);
                let mut got = rec.finish();
                let (g, h) = (got.pop().unwrap(), got.pop().unwrap());
                let scale = h.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
                for (i, (a, e)) in g.iter().zip(&h).enumerate() {
                    assert!((a - e).abs() <= 1e-5 * scale, "case {case} keep {keep} [{i}]: {a} against {e}");
                }
                if case == 0 {
                    close(&g, &reference(experts(count, hidden, ff, 48), &x, &logits, top_k), "routed on the GPU against the definition");
                }
            }
        }
        // a check's few rows, each routed on its own: each row's sums its own step's
        for rows in [2usize, 3, 5] {
            let xs: Vec<f32> = (0..rows * hidden).map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0).collect();
            let ls: Vec<f32> = (0..rows * (count + 1)).map(|i| ((i * 53 % 89) as f32 - 44.0) / 15.0).collect();
            let (xr, lr, yr) = (b.vec(rows * hidden), b.vec(rows * (count + 1)), b.vec(rows * hidden));
            DeviceChain::upload(&b, &xr, &xs);
            DeviceChain::upload(&b, &lr, &ls);
            for keep in [true, false] {
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &xr, &yr, &lr, top_k, rows), "{rows} rows route on the GPU");
                rec.read(&yr);
                let got = rec.finish().pop().unwrap();
                for r in 0..rows {
                    let (x1, l1, y1) = (b.vec(hidden), b.vec(count + 1), b.vec(hidden));
                    DeviceChain::upload(&b, &x1, &xs[r * hidden..(r + 1) * hidden]);
                    DeviceChain::upload(&b, &l1, &ls[r * (count + 1)..(r + 1) * (count + 1)]);
                    let mut rec = b.begin();
                    rec.keep_groups(keep);
                    assert!(rec.moe_routed(gpu.as_ref(), &x1, &y1, &l1, top_k, 1));
                    rec.read(&y1);
                    let want = rec.finish().pop().unwrap();
                    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                    assert_eq!(bits(&got[r * hidden..(r + 1) * hidden]), bits(&want), "{rows} rows keep {keep}: row {r}");
                }
                // the sums added to each row's streams as a site's write-back adds them: the same bits as the sums,
                // then the write-back
                let streams = 3;
                let base: Vec<f32> = (0..rows * streams * hidden).map(|i| ((i * 13 % 47) as f32 - 23.0) / 9.0).collect();
                let post: Vec<f32> = (0..rows * streams).map(|i| (i as f32 - 2.0) / 3.0).collect();
                let (xa, xb, pd, sums) = (b.vec(base.len()), b.vec(base.len()), b.vec(post.len()), b.vec(rows * hidden));
                DeviceChain::upload(&b, &xa, &base);
                DeviceChain::upload(&b, &xb, &base);
                DeviceChain::upload(&b, &pd, &post);
                let mut rec = b.begin();
                rec.keep_groups(keep);
                assert!(rec.moe_routed(gpu.as_ref(), &xr, &sums, &lr, top_k, rows));
                rec.stream_apply(&xa, &sums, &pd, rows, streams, hidden);
                assert!(rec.moe_routed_into(gpu.as_ref(), &xr, &xb, &pd, &lr, top_k, rows, streams));
                rec.read(&xa);
                rec.read(&xb);
                let mut got = rec.finish();
                let (into, apart) = (got.pop().unwrap(), got.pop().unwrap());
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                assert_eq!(bits(&into), bits(&apart), "{rows} rows keep {keep}: into the streams");
            }
        }
    }

}
