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
use ggml_rs::{DeviceVec, Tensor};
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

var<workgroup> tile: array<u32, 64>;

// f32 to f16 and back, rounding to nearest even: exact for the decoded weights, which are normal f16 values.
fn round_f16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}

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
    let sum = (hx & 255u) + ((hx >> 8u) & 255u) + ((hx >> 16u) & 255u) + (hx >> 24u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}
"#;

/// One input row (a decode step): a workgroup of 256 threads takes one 16-wide tile column and a run of its tile rows
/// (one split); thread `(r, c)` decodes weight `(r, c)` of each tile into one sum, and the 16 threads of a column then
/// add theirs up. Each split writes its partial sums to a slot of its own, which the host adds up.
const ONE: &str = r#"
var<workgroup> xs: array<f32, 16>;
var<workgroup> red: array<f32, 256>;

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
    var acc = 0.0;
    for (var kt = ks; kt < ke; kt = kt + 1u) {
        if (t < nw) {
            tile[t] = words[(kt * ntiles + nt) * nw + t];
        }
        if (t < 16u) {
            xs[t] = x[(p.kt0 + kt) * 16u + t];
        }
        workgroupBarrier();
        acc = acc + xs[r] * weight(r, c);
        workgroupBarrier();
    }
    red[t] = acc;
    workgroupBarrier();
    if (r == 0u) {
        var total = 0.0;
        for (var q = 0u; q < 16u; q = q + 1u) {
            total = total + red[q * 16u + c];
        }
        part[(p.slot0 + s) * p.n + nt * 16u + c] = total;
    }
}
"#;

/// Up to [`ROWS`] input rows (a prompt): each tile's 256 weights are decoded once into the workgroup's memory, and
/// thread `(r, c)` sums column `c` for rows `r` and `r + 16`, sixteen products a tile each. Each split writes its
/// partial sums to a slot of its own, which the host adds up.
const MANY: &str = r#"
var<workgroup> xs: array<f32, 512>;
var<workgroup> wt: array<f32, 256>;

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
    format!("{COMMON}{}", if many { MANY } else { ONE })
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
        gpu.queue.write_buffer(&xbuf, 0, &bytes(xh));
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
            gpu.queue.write_buffer(&ubuf, 0, &params);
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
    gpu.queue.submit([enc.finish()]);
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
const HALF: &str = r#"
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

/// The matmul of each job's transformed row ([`ONE`]'s, the matrices a group's: matrix `m`'s words from `m *
/// p[1].x`): a workgroup a (tile column, job and split), its partial sums to `part[(j * splits + s) * n..]`.
/// `p[0]`: n, k, tile words, splits; `p[1]`: words a matrix.
const G_MM: &str = r#"
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> jobs: array<u32>;
@group(0) @binding(6) var<storage, read_write> part: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> tile: array<u32, 64>;
var<workgroup> xs: array<f32, 16>;
var<workgroup> red: array<f32, 256>;

fn round_f16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}

fn weight(r: u32, c: u32, tw: u32) -> f32 {
    let nw = tw / 2u;
    let lane = (r % 8u) / 2u + 4u * (c % 8u);
    let jj = (r % 2u) + 2u * (r / 8u) + 4u * (c / 8u);
    let i = lane * 8u + jj;
    var end = (i + 1u) * (tw / 16u);
    if (tw % 16u == 8u) {
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
    let sum = (hx & 255u) + ((hx >> 8u) & 255u) + ((hx >> 16u) & 255u) + (hx >> 24u);
    return round_f16(f32(1024u + sum) * 0.00676727294921875 - 10.3828125);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let n = p[0].x;
    let k = p[0].y;
    let tw = p[0].z;
    let splits = p[0].w;
    let ntiles = n / 16u;
    let nt = wg.x + wg.y * 65535u;
    if (nt >= ntiles) {
        return;
    }
    let j = wg.z / splits;
    let s = wg.z % splits;
    let base = jobs[2u * j] * p[1].x;
    let r = t / 16u;
    let c = t % 16u;
    let nw = tw / 2u;
    let kts = k / 16u;
    let per = (kts + splits - 1u) / splits;
    let ks = s * per;
    let ke = min(kts, ks + per);
    var acc = 0.0;
    for (var kt = ks; kt < ke; kt = kt + 1u) {
        if (t < nw) {
            tile[t] = words[base + (kt * ntiles + nt) * nw + t];
        }
        if (t < 16u) {
            xs[t] = x[j * k + kt * 16u + t];
        }
        workgroupBarrier();
        acc = acc + xs[r] * weight(r, c, tw);
        workgroupBarrier();
    }
    red[t] = acc;
    workgroupBarrier();
    if (r == 0u) {
        var total = 0.0;
        for (var q = 0u; q < 16u; q = q + 1u) {
            total = total + red[q * 16u + c];
        }
        part[(j * splits + s) * n + nt * 16u + c] = total;
    }
}
"#;

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

/// The chain's kernels as WGSL: the input transform, the matmul, the output transform and its map.
pub(crate) fn chain_shader(which: &str) -> String {
    match which {
        "pre" => format!("{HALF}{G_PRE}"),
        "mm" => G_MM.to_string(),
        "post" => format!("{HALF}{G_POST}"),
        _ => G_GATHER.to_string(),
    }
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
    use ggml_rs::DeviceChain;
    let v = b.vec(values.len());
    let as_f32: Vec<f32> = values.iter().map(|&w| f32::from_bits(w)).collect();
    DeviceChain::upload(b, &v, &as_f32);
    v
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

/// One row's experts and their weights, as the CUDA routing gives them: the `top_k` of the routed experts by logit
/// (a tie to the lower index), softmax-weighted among themselves, then the shared expert (index `routed`) weighted by
/// the sigmoid of its gate.
fn route(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let routed = logits.len() - 1;
    let mut order: Vec<usize> = (0..routed).collect();
    order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let top = &order[..top_k.min(routed)];
    let max = logits[top[0]];
    let sum: f32 = top.iter().map(|&e| (logits[e] - max).exp()).sum();
    let mut out: Vec<(usize, f32)> = top.iter().map(|&e| (e, (logits[e] - max).exp() / sum)).collect();
    out.push((routed, 1.0 / (1.0 + (-logits[routed]).exp())));
    out
}

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

    /// As `exl3_experts`, leaving `reserve` bytes of the budget for the model's other matrices (loaded after).
    pub fn exl3_experts_leaving(&self, experts: Vec<[Exl3Data; 3]>, reserve: u64) -> Result<Box<dyn ggml_rs::exl3::Experts>, String> {
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
                let gpu = b.exl3_experts(experts(count, hidden, ff, tw)).unwrap();
                assert!(format!("{gpu:?}").contains(&format!("{} of {} projections on the GPU", 3 * (count + 1), 3 * (count + 1))), "{gpu:?}");
                close(gpu.forward(&xt, &lt, top_k).data(), &want, &format!("gpu moe tw={tw} rows={rows}"));
                drop(gpu);
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
                b.gpu.queue.submit([]);
                let _ = b.gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
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
    /// f16 roundings), gives the projection's own answer (its transforms on the host): one row (a step's, its scratch
    /// kept) bit for bit, and three, with and without maps, at 3 and 5 bits.
    #[test]
    fn a_chained_projection_matches_the_projection() {
        use ggml_rs::{ChainRecorder, DeviceChain};
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
            for rows in [1usize, 3] {
                let xs: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 31.0).collect();
                let want = w.linear(&Tensor::from_vec(xs.clone(), vec![rows, k]));
                let (x, y) = (b.vec(rows * k), b.vec(rows * n));
                DeviceChain::upload(&b, &x, &xs);
                // twice: the second from the kept scratch
                for _ in 0..2 {
                    let mut rec = b.begin();
                    rec.exl3_rows(w.as_ref(), &x, &y, rows);
                    rec.read(&y);
                    let got = rec.finish().pop().unwrap();
                    // one row as the projection's own kernel sums it, bit for bit; several as its kernel for a
                    // prompt's rows does, in another order (an f16 step apart at most)
                    let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
                    if rows == 1 {
                        assert_eq!(bits(&got), bits(want.data()), "tw={tw} maps={maps} rows={rows}");
                    } else {
                        close(&got, want.data(), &format!("tw={tw} maps={maps} rows={rows}"));
                    }
                }
            }
        }
    }
}
