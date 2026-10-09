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
//!
//! A routed expert's record can also be read as stored ([`RecordSlots`]): uploaded whole into a slot made once, its
//! MXFP4 matrices read in place with their e8m0 scales (a byte each), so a prompt's busy experts cost one write each
//! and no conversion on the host.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{chunk_limit, Gpu, WgpuBackend};

mod arena;
mod kernels;
mod records;
#[cfg(test)]
mod tests;

pub(crate) use arena::Arena;
pub use arena::{begin_units, forward_chained, forward_units, Chain, Pending, Unit};
use arena::{forget, forward_in_arena};
use kernels::shader;
pub use records::RecordSlots;

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

/// Which of the three, and MXFP4 as a record holds it (its scales e8m0 bytes, in the same buffer); also the kernels'
/// `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Fp8 = 0,
    Bf16 = 1,
    Mxfp4 = 2,
    Record = 3,
}

impl DenseData {
    fn nk(&self) -> (usize, usize) {
        match self {
            DenseData::Fp8 { n, k, .. } | DenseData::Bf16 { n, k, .. } | DenseData::Mxfp4 { n, k, .. } => (*n, *k),
        }
    }
}

/// A dense weight on the GPU.
pub struct DenseGpu {
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
    /// `(buffer, first row, rows, where its scales start, where its weights start)` (see the kernels' `p`); each
    /// buffer whole 32-row tiles.
    chunks: Vec<(wgpu::Buffer, u32, u32, u32, u32)>,
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
        // (a record's matrix is its slots' buffer, which the slots let go)
        if self.kind != Kind::Record {
            forget(&self.gpu, &mut self.chunks.iter().map(|c| &c.0));
        }
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

    /// The bytes `rows` of its rows hold (for the profile's count of what a call read).
    fn nbytes_of(&self, rows: usize) -> u64 {
        let per = match self.kind {
            Kind::Fp8 => 2,
            Kind::Bf16 => 4,
            Kind::Mxfp4 | Kind::Record => 1,
        };
        (rows * self.k * per / 2) as u64
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Whether `other` is on the same adapter, so the two can go in one [`forward_batch`].
    pub fn same_device(&self, other: &DenseGpu) -> bool {
        Arc::ptr_eq(&self.gpu, &other.gpu)
    }

    /// `[t, rows.len()]`: the sums of `x` (`[t, k]`, as given) against rows `rows` of the weight, in f32.
    pub fn forward(&self, x: &[f32], t: usize, rows: Range<usize>) -> Vec<f32> {
        forward_batch(&[(self, x, t, rows)]).pop().expect("one weight, one result")
    }

    /// Record the dispatches for `xbuf` (`[t, k]`, uploaded) against rows `rows` into `pass`, writing a new output
    /// buffer.
    fn record(&self, enc: &mut wgpu::CommandEncoder, xbuf: &wgpu::Buffer, t: usize, rows: Range<usize>) -> Pass {
        let (k, count) = (self.k, rows.len());
        assert!(rows.end <= self.n, "dense: rows past the weight");
        let gpu = &self.gpu;
        let many = t > FEW;
        let pipeline = gpu.named_pipeline(if many { "dense-many" } else { "dense-few" }, || shader(many));
        let size = (t * count * 4).max(4) as u64;
        let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-dense-y"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut groups = Vec::new();
        for (buffer, first, n_rows, soff, woff) in &self.chunks {
            let (a, b) = ((*first as usize).max(rows.start), (*first as usize + *n_rows as usize).min(rows.end));
            if a >= b {
                continue;
            }
            let params: Vec<u8> = [k as u32, t as u32, *first, *n_rows, rows.start as u32, count as u32, *soff, self.kind as u32, *woff, 0, 0, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let ubuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-dense-params"),
                size: params.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue().write_buffer(&ubuf, 0, &params);
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
    if let Some(out) = forward_in_arena(gpu, items) {
        return out;
    }
    let making = std::time::Instant::now();
    // Each input once, however many weights take it (an expert's gate and up take the same).
    let mut inputs: Vec<((*const f32, usize), wgpu::Buffer)> = Vec::new();
    for (w, x, t, rows) in items {
        assert_eq!(x.len(), t * w.k, "dense: the input is not [t, k]");
        if *t > 0 && !rows.is_empty() && !inputs.iter().any(|(key, _)| *key == (x.as_ptr(), x.len())) {
            inputs.push(((x.as_ptr(), x.len()), upload_f32(gpu, x)));
        }
    }
    let input = |x: &[f32]| &inputs.iter().find(|(key, _)| *key == (x.as_ptr(), x.len())).expect("every input uploaded").1;
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    let passes: Vec<Option<Pass>> =
        items.iter().map(|(w, x, t, rows)| (*t > 0 && !rows.is_empty()).then(|| w.record(&mut enc, input(x), *t, rows.clone()))).collect();
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
    crate::profile::add(&crate::profile::DENSE_MAKE, making);
    for (w, _, t, rows) in items {
        if *t > 0 && !rows.is_empty() {
            crate::profile::DENSE_BYTES[0].fetch_add(w.nbytes_of(rows.len()), Ordering::Relaxed);
            crate::profile::DENSE_BYTES[1].fetch_add(1, Ordering::Relaxed);
        }
    }
    let submitting = std::time::Instant::now();
    gpu.queue().submit([enc.finish()]);
    crate::profile::add(&crate::profile::DENSE_SUBMIT, submitting);
    let waiting = std::time::Instant::now();
    let raw = gpu.map_read(&staging, total);
    crate::profile::add(&crate::profile::DENSE_WAIT, waiting);
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

/// `x` in a new storage buffer.
fn upload_f32(gpu: &Gpu, x: &[f32]) -> wgpu::Buffer {
    let mut bytes = Vec::with_capacity(x.len() * 4);
    for v in x {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-dense-x"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue().write_buffer(&buf, 0, &bytes);
    buf
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
            chunks.push((buffer, first as u32, rows as u32, soff, 0));
            first += rows;
        }
        Ok(Some(Arc::new(DenseGpu { gpu: Arc::clone(gpu), serial: Arc::clone(&self.serial), chunks, n, k, kind, nbytes, used: Arc::clone(&self.used) })))
    }
}
