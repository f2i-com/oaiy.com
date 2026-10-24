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
use ggml_rs::exl3::{route, Exl3Data, PackedLinear};
use ggml_rs::{DeviceChain, DeviceVec, Tensor};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod host;
mod kernels;
mod moe;
mod routing;
mod slots;
mod source;
mod tiered;

// (what this file gave the crate and its users, and what the files make for each other)
pub use {host::*, moe::*, slots::*, source::*, tiered::*};
pub(crate) use {kernels::*, routing::*};

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
        let mut host = None;
        let (data, _, shape) = self.t.rows(x, &mut host);
        Tensor::from_vec(many(&[self], &[data]).pop().expect("one projection, one output"), shape)
    }
}

/// Partial sums one submit of [`exl3_many`] reads back at most: a prompt's passes past that go in more submits.
const MANY_BYTES: u64 = 256 << 20;

/// Several EXL3 projections on one GPU, `(projection, x)` each (`x` its host rows, `[m, k]`, any `m`), in as few
/// submits as their partial sums allow (one for a decode step's, [`MANY_BYTES`] of sums a read back): each one's output
/// `[m, n]`, as its `linear` gives it. None if any is not on the GPU the first is on. A model that runs its projections
/// one by one made a round trip of each (and of each [`ROWS`] rows of a prompt).
pub fn exl3_many(items: &[(&dyn PackedLinear, &[f32])]) -> Option<Vec<Vec<f32>>> {
    let gpus: Vec<&Exl3Gpu> = items.iter().map(|(l, _)| l.as_any()?.downcast_ref::<Exl3Gpu>()).collect::<Option<_>>()?;
    let first = *gpus.first()?;
    if !gpus.iter().all(|g| Arc::ptr_eq(&g.gpu, &first.gpu)) {
        return None;
    }
    let xs: Vec<&[f32]> = items.iter().map(|(_, x)| *x).collect();
    Some(many(&gpus, &xs))
}

/// Partial sums a recording of [`exl3_chained`] holds at most (bytes): a prompt's rows past them go in more recordings.
const CHAINED_BYTES: usize = 256 << 20;

/// [`exl3_many`] through a chained recording on `b`, the device every projection is on: each projection's transforms on
/// the device beside its matmul (a step's row through the projection's own kept buffers and bind groups; a prompt's rows
/// on the tensor cores in blocks of 128 where the device has them), the inputs uploaded and the outputs read back in one
/// submit. A prompt's rows go in as many recordings as their partial sums need ([`CHAINED_BYTES`]). None where one is
/// not on `b` or not in one buffer (a chain takes no other), or has no rows: the caller goes [`exl3_many`]'s way, whose
/// passes of [`ROWS`] rows each decoded every weight again (a 2,650-token prompt's trunk, 169 s).
pub fn exl3_chained(b: &WgpuBackend, items: &[(&dyn PackedLinear, &[f32])]) -> Option<Vec<Vec<f32>>> {
    use ggml_rs::ChainRecorder;
    let gpus: Vec<&Exl3Gpu> = items.iter().map(|(l, _)| l.as_any()?.downcast_ref::<Exl3Gpu>()).collect::<Option<_>>()?;
    if gpus.iter().zip(items).any(|(g, (_, x))| !g.is_on(&b.gpu) || g.single_chunk().is_none() || x.is_empty() || x.len() % g.t.k != 0) {
        return None;
    }
    let one = gpus.iter().zip(items).all(|(g, (_, x))| x.len() == g.t.k);
    // (item, first row, rows) of every piece, a recording's pieces until their partial sums would pass the bound
    let mut recordings: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new()];
    let mut held = 0usize;
    for (i, (g, (_, x))) in gpus.iter().zip(items).enumerate() {
        let (k, n) = g.kn();
        let rows = x.len() / k;
        let per_row = 8 * n * 4;
        let chunk = (CHAINED_BYTES / per_row / 128 * 128).max(128);
        for first in (0..rows).step_by(chunk) {
            let take = chunk.min(rows - first);
            if held > 0 && held + take * per_row > CHAINED_BYTES {
                recordings.push(Vec::new());
                held = 0;
            }
            recordings.last_mut().expect("a recording").push((i, first, take));
            held += take * per_row;
        }
    }
    let mut outs: Vec<Vec<f32>> = gpus.iter().zip(items).map(|(g, (_, x))| Vec::with_capacity(x.len() / g.t.k * g.t.n)).collect();
    for pieces in recordings {
        let mut rec = crate::chain::Recorder::new(b);
        rec.keep_groups(one);
        for &(i, first, rows) in &pieces {
            let (k, n) = gpus[i].kn();
            let xv = rec.scratch(rows * k);
            DeviceChain::upload(b, &xv, &items[i].1[first * k..(first + rows) * k]);
            let yv = rec.scratch(rows * n);
            rec.exl3_rows_of(items[i].0, &xv, None, &yv, rows);
            rec.read_range(&yv, 0, rows * n);
        }
        for (&(i, ..), y) in pieces.iter().zip(Box::new(rec).finish()) {
            outs[i].extend_from_slice(&y);
        }
    }
    Some(outs)
}

/// [`exl3_many`] of projections on one GPU: every [`ROWS`] rows of each a pass, the passes recorded into one encoder
/// until their sums come to [`MANY_BYTES`], each encoder one submit and one read back.
fn many(gpus: &[&Exl3Gpu], xs: &[&[f32]]) -> Vec<Vec<f32>> {
    let first = gpus[0];
    // (projection, first row, rows) of every pass
    let passes: Vec<(usize, usize, usize)> = gpus
        .iter()
        .zip(xs)
        .enumerate()
        .flat_map(|(i, (g, x))| {
            let m = x.len() / g.t.k;
            (0..m).step_by(ROWS).map(move |s| (i, s, ROWS.min(m - s)))
        })
        .collect();
    let mut outs: Vec<Vec<f32>> = gpus.iter().zip(xs).map(|(g, x)| vec![0f32; x.len() / g.t.k * g.t.n]).collect();
    let mut at = 0;
    while at < passes.len() {
        let (mut xhs, mut bytes, mut these) = (Vec::new(), 0u64, Vec::new());
        while at < passes.len() {
            let (i, s, rows) = passes[at];
            let g = gpus[i];
            let size = g.chunks.iter().map(|c| c.3 as u64).sum::<u64>() * (rows * g.t.n * 4) as u64;
            if !these.is_empty() && bytes + size > MANY_BYTES {
                break;
            }
            let k = g.t.k;
            let mut xh = vec![0f32; rows * k];
            for r in 0..rows {
                g.t.pre(&xs[i][(s + r) * k..(s + r + 1) * k], &mut xh[r * k..(r + 1) * k]);
            }
            xhs.push(xh);
            these.push(passes[at]);
            bytes += size;
            at += 1;
        }
        let ys = {
            let _one = first.serial.lock().unwrap_or_else(|p| p.into_inner());
            let mut enc = first.gpu.device.create_command_encoder(&Default::default());
            let recorded: Vec<Recorded> = these.iter().zip(&xhs).map(|(&(i, _, rows), xh)| gpus[i].record(&mut enc, xh, rows)).collect();
            read_back(&first.gpu, enc, &recorded)
        };
        for ((i, s, rows), mut y) in these.into_iter().zip(ys) {
            let n = gpus[i].t.n;
            for r in 0..rows {
                gpus[i].t.post(&mut y[r * n..(r + 1) * n], &mut outs[i][(s + r) * n..(s + r + 1) * n]);
            }
        }
    }
    outs
}

// ---------------------------------------------------------------------------
// In a chain
// ---------------------------------------------------------------------------

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
