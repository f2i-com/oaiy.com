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

mod attend;
mod group;
mod host;
mod kernels;
mod moe;
mod program;
mod routing;
mod slots;
mod source;
mod step_host;
mod team;
mod tiered;

// (what this file gave the crate and its users, and what the files make for each other)
pub use {group::*, host::*, moe::*, program::*, slots::*, source::*, step_host::*, tiered::*};
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
    let mats: Vec<(ChainMat<'_>, &[f32])> = items.iter().map(|&(l, x)| (ChainMat::Exl3(l), x)).collect();
    chained(b, &mats)
}

/// A matrix a chained recording ([`chained`]) multiplies: an EXL3 projection on the device, a matrix of f16 values
/// (`[n, k]`, made by `DeviceChain::vec_f16`: a checkpoint's f16 weights exactly), or one of BF16 values as a
/// checkpoint's bytes lie (`WgpuBackend::vec_of_bytes`), or one of f32 values (a model's hyper-connections' mixing
/// projections, which it stores so).
#[derive(Clone, Copy)]
pub enum ChainMat<'a> {
    Exl3(&'a dyn PackedLinear),
    F16 { w: &'a DeviceVec, n: usize, k: usize },
    Bf16 { w: &'a DeviceVec, n: usize, k: usize },
    F32 { w: &'a DeviceVec, n: usize, k: usize },
}

/// [`exl3_chained`] of EXL3 projections and f16 or BF16 matrices alike, all in its recordings (a model whose router and
/// some projections are BF16 beside EXL3 ones: a call fewer for each a layer than each kind's own). None as
/// [`exl3_chained`] is, or where a matrix's input is not whole rows of it.
pub fn chained(b: &WgpuBackend, items: &[(ChainMat<'_>, &[f32])]) -> Option<Vec<Vec<f32>>> {
    use ggml_rs::ChainRecorder;
    // each item's (k, n), its EXL3 projection checked as one this device holds in one buffer
    let shapes: Vec<(usize, usize)> = items.iter().map(|(m, x)| chain_shape(b, m).filter(|&(k, _)| !x.is_empty() && x.len() % k == 0)).collect::<Option<_>>()?;
    let one = shapes.iter().zip(items).all(|(&(k, _), (_, x))| x.len() == k);
    // a step's row: a program (its vectors kept from its first run, and their bind groups with them), the inputs several
    // take uploaded once
    if one {
        let mut steps: Vec<ProgramStep<'_>> = Vec::with_capacity(2 * items.len());
        let at: Vec<usize> = items.iter().map(|(_, x)| input_once(&mut steps, x)).collect();
        let muls: Vec<usize> = items
            .iter()
            .zip(&at)
            .map(|((m, _), &x)| {
                steps.push(ProgramStep::Mul { m: *m, x, rows: 1 });
                steps.len() - 1
            })
            .collect();
        let reads: Vec<(usize, usize, usize)> = muls.iter().zip(&shapes).map(|(&s, &(_, n))| (s, 0, n)).collect();
        return run_program(b, &steps, &reads);
    }
    // (item, first row, rows) of every piece, a recording's pieces until their partial sums would pass the bound
    let mut recordings: Vec<Vec<(usize, usize, usize)>> = vec![Vec::new()];
    let mut held = 0usize;
    for (i, (&(k, n), (_, x))) in shapes.iter().zip(items).enumerate() {
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
    let mut outs: Vec<Vec<f32>> = shapes.iter().zip(items).map(|(&(k, n), (_, x))| Vec::with_capacity(x.len() / k * n)).collect();
    for pieces in recordings {
        let mut rec = crate::chain::Recorder::new(b);
        rec.keep_groups(one);
        // Every piece's input and output taken and its input written before anything is recorded: a matmul's own scratch
        // goes back to the recording when it is done with it (an f16 matmul's partial sums), and an input taken after
        // that may be that very buffer, its upload (made before the recording runs) then written over by the matmul
        // that ran first. A router's partial sums over a prompt's 15 rows so made the shared expert's input garbage.
        let mut uploaded = Vec::new();
        let vecs: Vec<(DeviceVec, DeviceVec)> = pieces
            .iter()
            .map(|&(i, first, rows)| {
                let (k, n) = shapes[i];
                (upload_once(&mut rec, b, &mut uploaded, &items[i].1[first * k..(first + rows) * k]), rec.scratch(rows * n))
            })
            .collect();
        for (&(i, _, rows), (xv, yv)) in pieces.iter().zip(&vecs) {
            chain_mul(&mut rec, items[i].0, shapes[i], xv, yv, rows);
            rec.read_range(yv, 0, rows * shapes[i].1);
        }
        for (&(i, ..), y) in pieces.iter().zip(Box::new(rec).finish()) {
            outs[i].extend_from_slice(&y);
        }
    }
    Some(outs)
}

/// A step's rows (or a check's few: up to [`FEW_MAX`]) through two matrices in one chained recording on `b`: each of
/// `first`'s matrices of its own rows (`(matrix, x, rows)`: those of its outputs), each row's taken rounded to bf16 and
/// laid end to end on the device, then `then` of those rows; its sums, every row. The rounding is the host's to the bit
/// (to nearest even at bit 16), so the sums are those of the two calls made one after the other with the rows brought
/// back between (DeepSeek-V4.1's grouped `wo_a`, then `wo_b`: a round trip fewer a layer). None as [`chained`] is, or
/// where the rows laid end to end are not `then`'s input, or the inputs are of other rows' counts or more.
pub fn chained_after(b: &WgpuBackend, first: &[(ChainMat<'_>, &[f32], std::ops::Range<usize>)], then: ChainMat<'_>) -> Option<Vec<f32>> {
    let (k, n) = chain_shape(b, &then)?;
    let shapes: Vec<(usize, usize)> = first.iter().map(|(m, x, rows)| chain_shape(b, m).filter(|&(k, n)| !x.is_empty() && x.len() % k == 0 && rows.end <= n && !rows.is_empty())).collect::<Option<_>>()?;
    let t = first.first().map(|(_, x, _)| x.len() / shapes[0].0)?;
    if t > FEW_MAX || first.iter().zip(&shapes).any(|((_, x, _), &(k, _))| x.len() != t * k) || first.iter().map(|(_, _, rows)| rows.len()).sum::<usize>() != k {
        return None;
    }
    // (a program: its vectors kept from its first run, and their bind groups with them)
    let mut steps: Vec<ProgramStep<'_>> = Vec::with_capacity(3 * first.len() + 3);
    for (_, x, _) in first {
        steps.push(ProgramStep::Input(x));
    }
    for (i, (m, _, _)) in first.iter().enumerate() {
        steps.push(ProgramStep::Mul { m: *m, x: i, rows: t });
    }
    let rows_in = steps.len();
    steps.push(ProgramStep::Scratch(t * k));
    let mut at = 0;
    for (i, ((_, _, rows), &(_, ni))) in first.iter().zip(&shapes).enumerate() {
        steps.push(ProgramStep::CopyColsBf16 { src: first.len() + i, src_stride: ni, src_at: rows.start, dst: rows_in, dst_stride: k, dst_at: at, width: rows.len(), rows: t });
        at += rows.len();
    }
    steps.push(ProgramStep::Mul { m: then, x: rows_in, rows: t });
    run_program(b, &steps, &[(steps.len() - 1, 0, t * n)])?.pop()
}

/// `x` in a scratch vector of `rec`, uploaded once for every item of the recording that takes the same host rows (a
/// layer's projections of x: each its own upload before).
fn upload_once(rec: &mut crate::chain::Recorder<'_>, b: &WgpuBackend, uploaded: &mut Vec<((*const f32, usize), DeviceVec)>, x: &[f32]) -> DeviceVec {
    let key = (x.as_ptr(), x.len());
    if let Some((_, v)) = uploaded.iter().find(|(k, _)| *k == key) {
        return v.clone();
    }
    let v = rec.scratch(x.len());
    DeviceChain::upload(b, &v, x);
    uploaded.push((key, v.clone()));
    v
}

/// A chained recording's matrix's `(k, n)` on `b`: an EXL3 projection this device holds in one buffer, or a matrix of
/// floats; None for another.
fn chain_shape(b: &WgpuBackend, m: &ChainMat<'_>) -> Option<(usize, usize)> {
    match m {
        ChainMat::Exl3(l) => {
            let any = l.as_any()?;
            // (a GGUF's matrix, its blocks or its floats as f16, which a recorder's `exl3_rows_of` multiplies too)
            if let Some(q) = any.downcast_ref::<crate::quant::linear::QuantLinear>() {
                return q.is_on(&b.gpu).then(|| q.kn());
            }
            if let Some(h) = any.downcast_ref::<crate::quant::linear::HalfLinear>() {
                return h.is_on(&b.gpu).then(|| h.kn());
            }
            let g = any.downcast_ref::<Exl3Gpu>()?;
            (g.is_on(&b.gpu) && g.single_chunk().is_some()).then(|| g.kn())
        }
        ChainMat::F16 { n, k, .. } | ChainMat::Bf16 { n, k, .. } | ChainMat::F32 { n, k, .. } => Some((*k, *n)),
    }
}

/// `m` (of `(k, n)`) of `rows` rows of `x` into `y`, recorded.
fn chain_mul(rec: &mut crate::chain::Recorder<'_>, m: ChainMat<'_>, (k, n): (usize, usize), x: &DeviceVec, y: &DeviceVec, rows: usize) {
    use ggml_rs::ChainRecorder;
    match m {
        ChainMat::Exl3(l) => rec.exl3_rows_of(l, x, None, y, rows),
        ChainMat::F16 { w, .. } => rec.matmul_f16_rows_f32(w, n, k, x, y, rows),
        ChainMat::Bf16 { w, .. } => rec.matmul_bf16_rows_f32(w, n, k, x, y, rows),
        ChainMat::F32 { w, .. } => rec.matmul_f32_rows(w, n, k, x, y, rows),
    }
}

/// A step's rows (or a check's few: up to [`FEW_MAX`]) through two stages in one chained recording on `b`: `first`'s
/// matrices of their rows, as [`chained`] makes them, then `then`'s of the rows of `first[normed]`'s output normed as
/// DeepSeek-V4.1's host model norms its queries' low-rank row (`rmsnorm` of the output rounded to bf16, its sum of
/// squares in the host's order, the scale's square root and reciprocal correctly rounded, each value `bf16(weight *
/// (v * scale))`): the host's to the bit. `(first's outputs, then's)`, every row of each. None as [`chained`] is, or
/// where `then`'s inputs are not that output's width or the rows are more.
pub fn chained_normed(b: &WgpuBackend, first: &[(ChainMat<'_>, &[f32])], normed: usize, weight: &[f32], eps: f32, then: &[ChainMat<'_>]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let shapes: Vec<(usize, usize)> = first.iter().map(|(m, x)| chain_shape(b, m).filter(|&(k, _)| !x.is_empty() && x.len() % k == 0)).collect::<Option<_>>()?;
    let (k0, d) = *shapes.get(normed)?;
    let rows = first[normed].1.len() / k0;
    let thens: Vec<(usize, usize)> = then.iter().map(|m| chain_shape(b, m).filter(|&(k, _)| k == d)).collect::<Option<_>>()?;
    if rows > FEW_MAX || weight.len() != d || d > 1 << 20 {
        return None;
    }
    // (a program: its vectors kept from its first run, and their bind groups with them; inputs several take uploaded
    // once)
    let mut steps: Vec<ProgramStep<'_>> = Vec::with_capacity(2 * first.len() + then.len() + 1);
    let at: Vec<usize> = first.iter().map(|(_, x)| input_once(&mut steps, x)).collect();
    let muls: Vec<usize> = first
        .iter()
        .zip(&shapes)
        .zip(&at)
        .map(|(((m, x), &(k, _)), &xi)| {
            steps.push(ProgramStep::Mul { m: *m, x: xi, rows: x.len() / k });
            steps.len() - 1
        })
        .collect();
    let row = steps.len();
    steps.push(ProgramStep::Norm { x: muls[normed], weight, eps, rows, width: d });
    let thens_at: Vec<usize> = then
        .iter()
        .map(|m| {
            steps.push(ProgramStep::Mul { m: *m, x: row, rows });
            steps.len() - 1
        })
        .collect();
    let mut reads: Vec<(usize, usize, usize)> = muls.iter().zip(first.iter().zip(&shapes)).map(|(&s, ((_, x), &(k, n)))| (s, 0, x.len() / k * n)).collect();
    reads.extend(thens_at.iter().zip(&thens).map(|(&s, &(_, n))| (s, 0, rows * n)));
    let mut all = run_program(b, &steps, &reads)?;
    let after = all.split_off(first.len());
    Some((all, after))
}

/// [`chained_normed`]'s norm, a workgroup a row: `y = bf16(g * (v * r))` of each `v = bf16(x)` of the row, `r` the
/// reciprocal of the square root of `sum(v * v) / width + eps` (the row's width `p[0].x`, eps's bits `p[0].z`), the
/// squares added one after another in the host's order (each rounded as it is stored, so no fused multiply-add takes
/// them), the quotient, root and reciprocal correctly rounded by integer steps (the device's own are within ulps): the
/// host's to the bit.
const NORM_AS_HOST: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(6) var<storage, read_write> y: array<f32>;
@group(0) @binding(8) var<uniform> p: array<vec4<u32>, 2>;

var<workgroup> sq: array<vec4<f32>, 2048>;
var<workgroup> scale: f32;

fn bf16(v: f32) -> f32 {
    let b = bitcast<u32>(v);
    return bitcast<f32>((b + 0x7fffu + ((b >> 16u) & 1u)) & 0xffff0000u);
}

// the square root of a positive normal number, to nearest even: its mantissa's (times 2^26) digit by digit, two bits a
// step, the remainder below twice the root
fn sqrt_rn(a: f32) -> f32 {
    let bits = bitcast<u32>(a);
    var e = i32(bits >> 23u) - 150;
    var m = (bits & 0x7fffffu) | 0x800000u;
    if ((e & 1) != 0) {
        m = m << 1u;
        e = e - 1;
    }
    var rem = 0u;
    var root = 0u;
    for (var i = 25; i >= 0; i--) {
        var pair = 0u;
        if (i >= 13) { pair = (m >> u32(2 * i - 26)) & 3u; }
        rem = (rem << 2u) | pair;
        let trial = (root << 2u) | 1u;
        root = root << 1u;
        if (rem >= trial) {
            rem = rem - trial;
            root = root | 1u;
        }
    }
    // a root of 25 or 26 bits, kept to 24
    var shift = 1u;
    if (root >= 0x2000000u) { shift = 2u; }
    let guard = (root >> (shift - 1u)) & 1u;
    let sticky = rem != 0u || (root & ((1u << (shift - 1u)) - 1u)) != 0u;
    var r = root >> shift;
    if (guard == 1u && (sticky || (r & 1u) == 1u)) { r = r + 1u; }
    var be = e / 2 - 13 + i32(shift) + 150;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}

// a / b of positive normal numbers (a 0 too), to nearest even: the mantissas' quotient to 26 bits by long division, the
// remainder below b's mantissa, the rest a sticky bit (a quotient past the normal numbers the device's own)
fn div_rn(a: f32, b: f32) -> f32 {
    if (a == 0.0) { return 0.0; }
    let ab = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let mb = (bb & 0x7fffffu) | 0x800000u;
    var rem = (ab & 0x7fffffu) | 0x800000u;
    var q = 0u;
    for (var i = 0u; i < 26u; i++) {
        q = q << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
        rem = rem << 1u;
    }
    var shift = 1u;
    if (q >= 0x2000000u) { shift = 2u; }
    let guard = (q >> (shift - 1u)) & 1u;
    let sticky = rem != 0u || (q & ((1u << (shift - 1u)) - 1u)) != 0u;
    var r = q >> shift;
    if (guard == 1u && (sticky || (r & 1u) == 1u)) { r = r + 1u; }
    var be = i32(shift) - 25 + i32(ab >> 23u) - i32(bb >> 23u) + 150;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    if (be <= 0 || be >= 255) { return a / b; }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}

// 1 / s of a positive normal number, to nearest even: 2^48 over its mantissa by long division, the remainder below it
fn recip_rn(s: f32) -> f32 {
    let bits = bitcast<u32>(s);
    let es = i32(bits >> 23u);
    let m = (bits & 0x7fffffu) | 0x800000u;
    var rem = 0u;
    var q = 0u;
    for (var i = 0u; i < 49u; i++) {
        rem = rem << 1u;
        if (i == 0u) { rem = rem | 1u; }
        q = q << 1u;
        if (rem >= m) {
            rem = rem - m;
            q = q | 1u;
        }
    }
    if (q == 0x2000000u) {
        return bitcast<f32>(u32(254 - es) << 23u);
    }
    var r = q >> 1u;
    if ((q & 1u) == 1u && (rem != 0u || (r & 1u) == 1u)) { r = r + 1u; }
    var be = 253 - es;
    if (r == 0x1000000u) {
        r = 0x800000u;
        be = be + 1;
    }
    return bitcast<f32>((u32(be) << 23u) | (r & 0x7fffffu));
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let d = p[0].x;
    let base = wg.x * d;
    var acc = 0.0;
    // eight thousand squares at a time, each thread's four side by side (zeros past the row), thread 0 adding them in
    // order, eight reads of four at once ahead of their adds (a read of four at a time waited before its adds: some
    // 30 us a row of 5,120, the adds' own chain some 8)
    for (var c = 0u; c < d; c += 8192u) {
        for (var q = t; q < 2048u; q += 256u) {
            var v = vec4<f32>(0.0);
            for (var i = 0u; i < 4u; i++) {
                if (c + 4u * q + i < d) {
                    let e = bf16(x[base + c + 4u * q + i]);
                    v[i] = e * e;
                }
            }
            sq[q] = v;
        }
        workgroupBarrier();
        if (t == 0u) {
            let n = min(8192u, d - c);
            var j = 0u;
            for (; j + 8u <= n / 4u; j += 8u) {
                let s0 = sq[j];
                let s1 = sq[j + 1u];
                let s2 = sq[j + 2u];
                let s3 = sq[j + 3u];
                let s4 = sq[j + 4u];
                let s5 = sq[j + 5u];
                let s6 = sq[j + 6u];
                let s7 = sq[j + 7u];
                acc = acc + s0.x; acc = acc + s0.y; acc = acc + s0.z; acc = acc + s0.w;
                acc = acc + s1.x; acc = acc + s1.y; acc = acc + s1.z; acc = acc + s1.w;
                acc = acc + s2.x; acc = acc + s2.y; acc = acc + s2.z; acc = acc + s2.w;
                acc = acc + s3.x; acc = acc + s3.y; acc = acc + s3.z; acc = acc + s3.w;
                acc = acc + s4.x; acc = acc + s4.y; acc = acc + s4.z; acc = acc + s4.w;
                acc = acc + s5.x; acc = acc + s5.y; acc = acc + s5.z; acc = acc + s5.w;
                acc = acc + s6.x; acc = acc + s6.y; acc = acc + s6.z; acc = acc + s6.w;
                acc = acc + s7.x; acc = acc + s7.y; acc = acc + s7.z; acc = acc + s7.w;
            }
            for (; j < n / 4u; j++) {
                let s = sq[j];
                acc = acc + s.x;
                acc = acc + s.y;
                acc = acc + s.z;
                acc = acc + s.w;
            }
            for (var k = n / 4u * 4u; k < n; k++) { acc = acc + sq[k / 4u][k % 4u]; }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        scale = recip_rn(sqrt_rn(div_rn(acc, f32(d)) + bitcast<f32>(p[0].z)));
    }
    workgroupBarrier();
    let r = scale;
    for (var i = t; i < d; i += 256u) {
        y[base + i] = bf16(g[i] * (bf16(x[base + i]) * r));
    }
}
"#;

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
