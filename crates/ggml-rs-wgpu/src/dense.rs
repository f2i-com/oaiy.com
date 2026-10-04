//! Block-scaled dense weights on the GPU, the kinds DeepSeek-V4.1 holds: fp8 e4m3 `[n, k]` with one scale per 32x32
//! tile and bf16 `[n, k]` (its trunk: attention projections, shared experts, router, Engram projection, head), and
//! MXFP4 `[n, k]` (e2m1, two to a byte, low nibble first) with one scale per row per 32 of `k` (its routed experts). The
//! caller (dsv41) quantizes the activation as the reference does and rounds the result; here
//! `y[t, r] = Σ_k x[t, k] · W[r, k]` in f32, every weight decoded exactly (a scale is an exact power of two, given as
//! f32 by the caller), so only the order of the sum differs from the CPU's.
//!
//! The rows are kept in buffers below the binding limit, whole 32-row tiles each (their scales beside their weights in
//! the same buffer). A call records a dispatch per buffer its rows touch and the copy back in one submit, and
//! [`forward_batch`] does that for several weights at once (a layer's experts): a kernel for a few tokens (a decode
//! step: 32 lanes a row, 8 rows a workgroup) and a tiled one for more (64 tokens by 64 rows, 32 of `k` a step).

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{chunk_limit, Gpu, WgpuBackend};

/// Tokens the decode kernel takes in one call; more go to the tiled one.
const FEW: usize = 8;

/// A dense weight as stored.
pub enum DenseData {
    /// e4m3 bytes `[n, k]` and one scale per 32x32 tile, `[ceil(n/32), k/32]`, as f32 (an exact power of two).
    Fp8 { w: Vec<u8>, scales: Vec<f32>, n: usize, k: usize },
    /// bf16 bits `[n, k]`.
    Bf16 { w: Vec<u16>, n: usize, k: usize },
    /// e2m1 nibbles `[n, k]`, two to a byte (low first), and one scale per row per 32 of k, `[n, k/32]`, as f32.
    Mxfp4 { w: Vec<u8>, scales: Vec<f32>, n: usize, k: usize },
}

/// Which of the three; also the kernels' `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Fp8 = 0,
    Bf16 = 1,
    Mxfp4 = 2,
}

impl DenseData {
    fn nk(&self) -> (usize, usize) {
        match self {
            DenseData::Fp8 { n, k, .. } | DenseData::Bf16 { n, k, .. } | DenseData::Mxfp4 { n, k, .. } => (*n, *k),
        }
    }
}

const COMMON: &str = r#"
@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
// k, tokens, the buffer's first row, its rows, the first row asked for, the rows asked for, where its scales start
// (words), and the kind (0 fp8, 1 bf16, 2 mxfp4).
@group(0) @binding(3) var<uniform> p: array<vec4<u32>, 2>;

fn fp8(b: u32) -> f32 {
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    // e4m3fn has no infinity: all ones is NaN, as the reference decodes it.
    if (e == 15u && m == 7u) { return bitcast<f32>(0x7fc00000u); }
    var v: f32;
    if (e == 0u) {
        v = f32(m) * 0.001953125;
    } else {
        v = bitcast<f32>(((e + 120u) << 23u) | (m << 20u));
    }
    return select(v, -v, (b & 128u) != 0u);
}

// e2m1: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives.
fn fp4(n: u32) -> f32 {
    let m = n & 7u;
    var v = f32(m) * 0.5;
    if (m >= 4u) { v = select(f32(m) - 2.0, 6.0, m == 7u); }
    return select(v, -v, (n & 8u) != 0u);
}

// Weights a word holds: 4 fp8, 2 bf16, 8 e2m1.
fn wide(kind: u32) -> u32 {
    return select(select(4u, 2u, kind == 1u), 8u, kind == 2u);
}

// The j-th weight of a word.
fn weight(word: u32, kind: u32, j: u32) -> f32 {
    if (kind == 0u) { return fp8((word >> (8u * j)) & 255u); }
    if (kind == 2u) { return fp4((word >> (4u * j)) & 15u); }
    return select(bitcast<f32>(word & 0xffff0000u), bitcast<f32>(word << 16u), j == 0u);
}

// The scale of the weights of row `local` (in this buffer) at k column `c`: a 32x32 tile's (fp8), a row's 32 (mxfp4),
// none (bf16).
fn scale(kind: u32, soff: u32, local: u32, c: u32, kb: u32) -> f32 {
    if (kind == 0u) { return bitcast<f32>(wbuf[soff + (local / 32u) * kb + c / 32u]); }
    if (kind == 2u) { return bitcast<f32>(wbuf[soff + local * kb + c / 32u]); }
    return 1.0;
}
"#;

/// The decode kernel: 8 rows a workgroup, 32 lanes a row, each lane a word in 32 of the row, every token at once.
const FEW_KERNEL: &str = r#"
var<workgroup> part: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let lane = li & 31u;
    let slot = li >> 5u;
    // The rows this buffer and the call share: [max(first, lo), min(first + rows, lo + count)).
    let start = max(first, lo);
    let row = start + wg.x * 8u + slot;
    let live = row < min(first + rows, lo + count);
    var acc: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) { acc[i] = 0.0; }
    if (live) {
        let local = row - first;
        let wd = wide(kind);
        let per = k / wd;
        let kb = k / 32u;
        for (var w = lane; w < per; w += 32u) {
            let word = wbuf[local * per + w];
            let c = w * wd;
            let s = scale(kind, soff, local, c, kb);
            for (var tt = 0u; tt < t; tt++) {
                var d = 0.0;
                for (var j = 0u; j < wd; j++) { d += weight(word, kind, j) * x[tt * k + c + j]; }
                acc[tt] += d * s;
            }
        }
    }
    for (var tt = 0u; tt < 8u; tt++) { part[(slot * 32u + lane) * 8u + tt] = acc[tt]; }
    workgroupBarrier();
    if (live && lane == 0u) {
        for (var tt = 0u; tt < t; tt++) {
            var sum = 0.0;
            for (var l = 0u; l < 32u; l++) { sum += part[(slot * 32u + l) * 8u + tt]; }
            y[tt * count + (row - lo)] = sum;
        }
    }
}
"#;

/// The prompt kernel: 64 tokens by 64 rows a workgroup, 32 of k a step; each thread 4 tokens by 4 rows.
const MANY_KERNEL: &str = r#"
var<workgroup> xs: array<f32, 2048>;
var<workgroup> wt: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let k = p[0].x; let t = p[0].y; let first = p[0].z; let rows = p[0].w;
    let lo = p[1].x; let count = p[1].y; let soff = p[1].z; let kind = p[1].w;
    let start = max(first, lo);
    let end = min(first + rows, lo + count);
    let r0 = start + wg.x * 64u;
    let t0 = wg.y * 64u;
    let tx = li & 15u;
    let ty = li >> 4u;
    let wd = wide(kind);
    let per = k / wd;
    // Words a row holds in a 32-wide step of k.
    let wps = 32u / wd;
    let kb = k / 32u;
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }
    for (var k0 = 0u; k0 < k; k0 += 32u) {
        // The tokens' 64 x 32: element e, token e / 32, k e % 32, kept k-major.
        for (var e = li; e < 2048u; e += 256u) {
            let tok = t0 + e / 32u;
            let kk = e % 32u;
            var v = 0.0;
            if (tok < t) { v = x[tok * k + k0 + kk]; }
            xs[kk * 64u + e / 32u] = v;
        }
        // The rows' 64 x 32, decoded and scaled: a word an element.
        for (var e = li; e < 64u * wps; e += 256u) {
            let r = e / wps;
            let q = e % wps;
            let row = r0 + r;
            let live = row < end;
            var word = 0u;
            var s = 0.0;
            if (live) {
                let local = row - first;
                word = wbuf[local * per + k0 / wd + q];
                s = scale(kind, soff, local, k0, kb);
            }
            for (var j = 0u; j < wd; j++) {
                wt[(q * wd + j) * 64u + r] = select(0.0, weight(word, kind, j) * s, live);
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 32u; kk++) {
            for (var a = 0u; a < 4u; a++) {
                let xv = xs[kk * 64u + ty * 4u + a];
                for (var b = 0u; b < 4u; b++) { acc[a * 4u + b] += xv * wt[kk * 64u + tx * 4u + b]; }
            }
        }
        workgroupBarrier();
    }
    for (var a = 0u; a < 4u; a++) {
        let tok = t0 + ty * 4u + a;
        for (var b = 0u; b < 4u; b++) {
            let row = r0 + tx * 4u + b;
            if (tok < t && row < end) { y[tok * count + (row - lo)] = acc[a * 4u + b]; }
        }
    }
}
"#;

fn shader(many: bool) -> String {
    format!("{COMMON}{}", if many { MANY_KERNEL } else { FEW_KERNEL })
}

/// A dense weight on the GPU.
pub struct DenseGpu {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    /// `(buffer, first row, rows, where its scales start in words)`; each buffer whole 32-row tiles.
    chunks: Vec<(wgpu::Buffer, u32, u32, u32)>,
    n: usize,
    k: usize,
    kind: Kind,
    nbytes: u64,
    used: Arc<AtomicU64>,
}

impl std::fmt::Debug for DenseGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DenseGpu({:?} [{}, {}], {} buffers)", self.kind, self.n, self.k, self.chunks.len())
    }
}

impl Drop for DenseGpu {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes, Ordering::Relaxed);
    }
}

/// One weight's part of a submit: its output buffer and size, read back after.
struct Pass {
    y: wgpu::Buffer,
    size: u64,
}

impl DenseGpu {
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// `[t, rows.len()]`: the sums of `x` (`[t, k]`, as given) against rows `rows` of the weight, in f32.
    pub fn forward(&self, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f32> {
        forward_batch(&[(self, x, t, rows)]).pop().expect("one weight, one result")
    }

    /// Record the dispatches for `x` (`[t, k]`) against rows `rows` into `pass`, writing a new output buffer.
    fn record(&self, enc: &mut wgpu::CommandEncoder, x: &[f32], t: usize, rows: Range<usize>) -> Pass {
        let (k, count) = (self.k, rows.len());
        assert_eq!(x.len(), t * k, "dense: the input is not [t, k]");
        assert!(rows.end <= self.n, "dense: rows past the weight");
        let gpu = &self.gpu;
        let many = t > FEW;
        let pipeline = gpu.named_pipeline(if many { "dense-many" } else { "dense-few" }, || shader(many));
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-dense-x"),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&xbuf, 0, &bytes);
        let size = (t * count * 4).max(4) as u64;
        let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-dense-y"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut groups = Vec::new();
        for (buffer, first, n_rows, soff) in &self.chunks {
            let (a, b) = ((*first as usize).max(rows.start), (*first as usize + *n_rows as usize).min(rows.end));
            if a >= b {
                continue;
            }
            let params: Vec<u8> = [k as u32, t as u32, *first, *n_rows, rows.start as u32, count as u32, *soff, self.kind as u32]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-dense-params"),
                size: params.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&ubuf, 0, &params);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("oaiy-dense"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
                ],
            });
            let grid = if many { ((b - a).div_ceil(64) as u32, t.div_ceil(64) as u32) } else { ((b - a).div_ceil(8) as u32, 1) };
            groups.push((group, grid));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (group, (gx, gy)) in &groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        Pass { y: ybuf, size }
    }
}

/// [`DenseGpu::forward`] for several weights of one GPU in one submit and one read back: each `(weight, x, t, rows)`
/// gives its `[t, rows.len()]`, in order. A layer's experts go this way, a round trip a stage rather than a matrix.
pub fn forward_batch(items: &[(&DenseGpu, &[f32], usize, Range<usize>)]) -> Vec<Vec<f32>> {
    let Some(first) = items.first() else { return Vec::new() };
    let gpu = &first.0.gpu;
    assert!(items.iter().all(|(w, ..)| Arc::ptr_eq(&w.gpu, gpu)), "dense: a batch on more than one GPU");
    let _one = first.0.serial.lock().unwrap_or_else(|p| p.into_inner());
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let passes: Vec<Option<Pass>> =
        items.iter().map(|(w, x, t, rows)| (*t > 0 && !rows.is_empty()).then(|| w.record(&mut enc, x, *t, rows.clone()))).collect();
    let total: u64 = passes.iter().flatten().map(|p| p.size).sum();
    if total == 0 {
        return items.iter().map(|_| Vec::new()).collect();
    }
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-dense-read"),
        size: total,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut at = 0;
    for p in passes.iter().flatten() {
        enc.copy_buffer_to_buffer(&p.y, 0, &staging, at, p.size);
        at += p.size;
    }
    gpu.queue.submit([enc.finish()]);
    let raw = gpu.map_read(&staging, total);
    let mut at = 0usize;
    items
        .iter()
        .zip(&passes)
        .map(|((_, _, t, rows), p)| {
            let Some(p) = p else { return Vec::new() };
            let len = t * rows.len();
            let out = raw[at..at + len * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            at += p.size as usize;
            out
        })
        .collect()
}

impl WgpuBackend {
    /// A dense weight on the GPU while the weight budget holds it: None beyond it (the caller keeps it on the CPU), or
    /// when `k` is not a multiple of 32 (the tiles' edge).
    pub fn dense(&self, data: DenseData) -> Result<Option<Arc<DenseGpu>>, String> {
        let (n, k) = data.nk();
        if k == 0 || n == 0 || k % 32 != 0 {
            return Ok(None);
        }
        let kb = k / 32;
        // Bytes a row of weights takes, and scale words a 32-row tile takes.
        let (kind, row_bytes, tile_scales) = match &data {
            DenseData::Fp8 { w, scales, .. } => {
                if w.len() != n * k || scales.len() != n.div_ceil(32) * kb {
                    return Err(format!("dense: fp8 [{n}, {k}] with {} bytes and {} scales", w.len(), scales.len()));
                }
                (Kind::Fp8, k, kb)
            }
            DenseData::Bf16 { w, .. } => {
                if w.len() != n * k {
                    return Err(format!("dense: bf16 [{n}, {k}] with {} values", w.len()));
                }
                (Kind::Bf16, k * 2, 0)
            }
            DenseData::Mxfp4 { w, scales, .. } => {
                if w.len() * 2 != n * k || scales.len() != n * kb {
                    return Err(format!("dense: mxfp4 [{n}, {k}] with {} bytes and {} scales", w.len(), scales.len()));
                }
                (Kind::Mxfp4, k / 2, 32 * kb)
            }
        };
        let nbytes = (n * row_bytes + n.div_ceil(32) * tile_scales * 4) as u64;
        let prev = self.used.fetch_add(nbytes, Ordering::Relaxed);
        if prev + nbytes > self.budget {
            self.used.fetch_sub(nbytes, Ordering::Relaxed);
            return Ok(None);
        }
        // Whole 32-row tiles a buffer, as many as the binding limit takes (their scales with them).
        let tile_bytes = 32 * row_bytes + tile_scales * 4;
        let per = ((chunk_limit(&self.gpu.limits) as usize / tile_bytes).max(1)) * 32;
        let gpu = &self.gpu;
        let mut chunks = Vec::new();
        let mut first = 0;
        while first < n {
            let rows = per.min(n - first);
            let mut bytes: Vec<u8> = Vec::with_capacity(rows * row_bytes + rows.div_ceil(32) * tile_scales * 4);
            let soff = match &data {
                DenseData::Fp8 { w, scales, .. } => {
                    bytes.extend_from_slice(&w[first * k..(first + rows) * k]);
                    let soff = (bytes.len() / 4) as u32;
                    for s in &scales[(first / 32) * kb..(first / 32 + rows.div_ceil(32)) * kb] {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    soff
                }
                DenseData::Bf16 { w, .. } => {
                    bytes.extend(w[first * k..(first + rows) * k].iter().flat_map(|v| v.to_le_bytes()));
                    0
                }
                DenseData::Mxfp4 { w, scales, .. } => {
                    bytes.extend_from_slice(&w[first * k / 2..(first + rows) * k / 2]);
                    let soff = (bytes.len() / 4) as u32;
                    for s in &scales[first * kb..(first + rows) * kb] {
                        bytes.extend_from_slice(&s.to_le_bytes());
                    }
                    soff
                }
            };
            let (buffer, _, _) = gpu.upload_rows(&bytes, bytes.len(), 1).remove(0);
            chunks.push((buffer, first as u32, rows as u32, soff));
            first += rows;
        }
        Ok(Some(Arc::new(DenseGpu { gpu: Arc::clone(gpu), serial: Arc::clone(&self.serial), chunks, n, k, kind, nbytes, used: Arc::clone(&self.used) })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's own decoding of an e4m3 byte (dsv41 formats::fp8_e4m3_to_f32), for the oracle.
    fn e4m3(b: u8) -> f32 {
        let (s, e, m) = ((b >> 7) as i32, ((b >> 3) & 15) as i32, (b & 7) as f32);
        let v = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
        if s == 1 {
            -v
        } else {
            v
        }
    }

    /// dsv41's FP4_VALUES.
    const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

    fn backend() -> Option<WgpuBackend> {
        WgpuBackend::new(Some(1 << 30)).ok()
    }

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn copy(data: &DenseData) -> DenseData {
        match data {
            DenseData::Fp8 { w, scales, n, k } => DenseData::Fp8 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
            DenseData::Bf16 { w, n, k } => DenseData::Bf16 { w: w.clone(), n: *n, k: *k },
            DenseData::Mxfp4 { w, scales, n, k } => DenseData::Mxfp4 { w: w.clone(), scales: scales.clone(), n: *n, k: *k },
        }
    }

    /// `[t, rows]` in f64: the oracle.
    fn oracle(data: &DenseData, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f64> {
        let (_, k) = data.nk();
        let w = |r: usize, c: usize| -> f64 {
            match data {
                DenseData::Fp8 { w, scales, .. } => e4m3(w[r * k + c]) as f64 * scales[(r / 32) * (k / 32) + c / 32] as f64,
                DenseData::Bf16 { w, .. } => f32::from_bits((w[r * k + c] as u32) << 16) as f64,
                DenseData::Mxfp4 { w, scales, .. } => {
                    let byte = w[(r * k + c) / 2];
                    let nib = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                    E2M1[nib as usize] as f64 * scales[r * (k / 32) + c / 32] as f64
                }
            }
        };
        let mut out = Vec::new();
        for tt in 0..t {
            for r in rows.clone() {
                out.push((0..k).map(|c| x[tt * k + c] as f64 * w(r, c)).sum());
            }
        }
        out
    }

    fn fp8_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        // Bytes that are not NaN (0x7f / 0xff), scales 2^-8 .. 2^1.
        let w = (0..n * k).map(|_| { let b = (next() & 255) as u8; if b & 0x7f == 0x7f { b ^ 1 } else { b } }).collect();
        let scales = (0..n.div_ceil(32) * (k / 32)).map(|_| 2f32.powi((next() % 10) as i32 - 8)).collect();
        DenseData::Fp8 { w, scales, n, k }
    }

    fn bf16_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        let w = (0..n * k).map(|_| ((((next() % 2000) as f32 / 1000.0) - 1.0).to_bits() >> 16) as u16).collect();
        DenseData::Bf16 { w, n, k }
    }

    fn mxfp4_data(n: usize, k: usize, seed: u64) -> DenseData {
        let mut next = rng(seed);
        let w = (0..n * k / 2).map(|_| (next() & 255) as u8).collect();
        let scales = (0..n * (k / 32)).map(|_| 2f32.powi((next() % 12) as i32 - 9)).collect();
        DenseData::Mxfp4 { w, scales, n, k }
    }

    fn input(t: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut next = rng(seed);
        (0..t * k).map(|_| ((next() % 2001) as f32 / 1000.0) - 1.0).collect()
    }

    fn close(got: &[f32], want: &[f64], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(((*g as f64) - w).abs() <= 2e-5 * scale + 1e-6, "{what}: [{i}] {g} against {w}");
        }
    }

    #[test]
    fn every_fp8_byte_decodes_as_the_reference_decodes_it() {
        let Some(b) = backend() else { return };
        // One row of the 256 bytes (and a second tile row), scale 1: a one-hot input reads each weight back alone.
        let (n, k) = (33, 256);
        // The two NaN bytes (0x7f, 0xff) left out: a NaN weight makes every sum it is in NaN, even times zero.
        let w: Vec<u8> = (0..n * k).map(|i| (i % 256) as u8).map(|b| if b & 0x7f == 0x7f { 0 } else { b }).collect();
        let data = DenseData::Fp8 { w: w.clone(), scales: vec![1.0; 2 * (k / 32)], n, k };
        let g = b.dense(data).unwrap().expect("on the GPU");
        // Every byte but the two NaNs (0x7f, 0xff), which no checkpoint holds.
        for col in (0usize..256).filter(|c| c & 0x7f != 0x7f) {
            let mut x = vec![0.0f32; k];
            x[col] = 1.0;
            let y = g.forward(&x, 1, 0..1);
            let want = e4m3(col as u8);
            // Exactly (by value: the sum of -0 with the +0 products beside it is +0).
            assert_eq!(y[0], want, "byte {col:#04x}");
        }
    }

    #[test]
    fn every_e2m1_nibble_decodes_as_the_reference_decodes_it() {
        let Some(b) = backend() else { return };
        // One row holding the 16 nibbles twice (k = 32), scale 1.
        let w: Vec<u8> = (0..16u8).map(|i| ((2 * i) % 16) | (((2 * i + 1) % 16) << 4)).collect();
        let g = b.dense(DenseData::Mxfp4 { w, scales: vec![1.0], n: 1, k: 32 }).unwrap().expect("on the GPU");
        for col in 0..32 {
            let mut x = vec![0.0f32; 32];
            x[col] = 1.0;
            assert_eq!(g.forward(&x, 1, 0..1)[0], E2M1[col % 16], "nibble {}", col % 16);
        }
    }

    #[test]
    fn each_kind_matches_the_oracle_for_a_decode_step_and_a_prompt() {
        let Some(b) = backend() else { return };
        for (n, k) in [(96, 64), (70, 160), (300, 512)] {
            for (data, kind) in [(fp8_data(n, k, n as u64), "fp8"), (bf16_data(n, k, k as u64), "bf16"), (mxfp4_data(n, k, (n * k) as u64), "mxfp4")] {
                let want_data = copy(&data);
                let g = b.dense(data).unwrap().expect("on the GPU");
                for t in [1usize, 3, 8, 9, 70, 130] {
                    let x = input(t, k, (t * 31 + n) as u64);
                    for rows in [0..n, 5..n.min(77), n - 1..n] {
                        let got = g.forward(&x, t, rows.clone());
                        close(&got, &oracle(&want_data, &x, t, rows.clone()), &format!("{kind} [{n}, {k}] t={t} rows {rows:?}"));
                    }
                }
            }
        }
    }

    #[test]
    fn a_batch_gives_each_weight_its_own_answer_in_one_submit() {
        let Some(b) = backend() else { return };
        let datas = [mxfp4_data(64, 96, 1), mxfp4_data(96, 64, 2), fp8_data(40, 32, 3), bf16_data(33, 64, 4)];
        let wants: Vec<DenseData> = datas.iter().map(copy).collect();
        let gs: Vec<Arc<DenseGpu>> = datas.into_iter().map(|d| b.dense(d).unwrap().unwrap()).collect();
        let shapes = [(5usize, 0..64usize), (12, 10..96), (1, 0..40), (0, 0..33)];
        let xs: Vec<Vec<f32>> = gs.iter().zip(&shapes).map(|(g, (t, _))| input(*t, g.k(), *t as u64 + 7)).collect();
        let items: Vec<(&DenseGpu, &[f32], usize, Range<usize>)> = gs.iter().zip(&xs).zip(&shapes).map(|((g, x), (t, rows))| (&**g, x.as_slice(), *t, rows.clone())).collect();
        let got = forward_batch(&items);
        for (i, ((want, x), (t, rows))) in wants.iter().zip(&xs).zip(&shapes).enumerate() {
            close(&got[i], &oracle(want, x, *t, rows.clone()), &format!("item {i}"));
        }
        assert!(got[3].is_empty(), "no tokens, no rows");
    }

    #[test]
    fn a_weight_beyond_the_budget_or_with_an_odd_k_stays_with_the_caller() {
        let Ok(small) = WgpuBackend::new(Some(1000)) else { return };
        assert!(small.dense(bf16_data(64, 64, 1)).unwrap().is_none());
        assert_eq!(small.usage().0, 0);
        let Some(b) = backend() else { return };
        assert!(b.dense(DenseData::Bf16 { w: vec![0; 2 * 48], n: 2, k: 48 }).unwrap().is_none());
        let g = b.dense(bf16_data(64, 64, 2)).unwrap().unwrap();
        assert!(b.usage().0 > 0);
        drop(g);
        assert_eq!(b.usage().0, 0);
    }
}
