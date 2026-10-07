//! A decode step's ops chained on the adapter ([`ggml_rs::chain`]): vectors kept in buffers here, the ops recorded and
//! run in one compute pass of one submit, only what the host asks for read back. The quantized matmuls are the GGUF
//! kernels (`shaders`, one row of `x`); RMSNorm, the residual add, the SwiGLU of a fused gate-up, RoPE, a store into a
//! cache and a decode step's attention have kernels of their own here, on the same bind group layout (weights, a
//! table or a cache at 0, the input at 1, the output at 2, the parameters at 3).

use std::sync::{Arc, Mutex};

use ggml_rs::chain::{ChainRecorder, DeltaNet, DeviceChain, DeviceVec};
use ggml_rs::{QuantizedTensor, Tensor};

use crate::{Gpu, WgpuBackend, WgpuQuant};

mod attention;
mod conv;
mod device;
mod kernels;
mod matmul;
mod ops;
mod packed;
mod recurrent;
#[cfg(test)]
mod tests;

pub(crate) use kernels::*;

/// A uniform's eight words, the rest of `words` zero.
fn words8(words: &[u32]) -> [u32; 8] {
    let mut all = [0u32; 8];
    all[..words.len()].copy_from_slice(words);
    all
}

fn buffer(v: &DeviceVec) -> &wgpu::Buffer {
    v.inner.downcast_ref::<wgpu::Buffer>().expect("a WebGPU chain's vector")
}

/// A tensor that is a chain's vector itself ([`DeviceChain::alias`]), read back when the host asks for it.
struct Aliased {
    v: DeviceVec,
    gpu: Arc<Gpu>,
    serial: Arc<Mutex<()>>,
}

impl std::fmt::Debug for Aliased {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Aliased({})", self.v.len)
    }
}

/// A buffer for a chain's vector of `len`.
/// A grid of `groups` workgroups of 256 for a kernel that indexes over two of its dimensions (`id.x + id.y * 65535 *
/// 256`): a dimension takes 65,535 at most (a chunk of 1,024 rows' FFN is 69,632).
fn grid(groups: u32) -> (u32, u32, u32) {
    (groups.min(65535), groups.div_ceil(65535).max(1), 1)
}

fn vec_buffer(gpu: &Gpu, len: usize) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oaiy-chain-vec"),
        size: (len.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl ggml_rs::DeviceStorage for Aliased {
    fn len(&self) -> usize {
        self.v.len
    }

    fn device_name(&self) -> &str {
        "webgpu"
    }

    fn copy_to_host(&self) -> Vec<f32> {
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let len = self.v.len;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-alias-read"),
            size: (len.max(1) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if len > 0 {
            enc.copy_buffer_to_buffer(buffer(&self.v), 0, &staging, 0, (len * 4) as u64);
        }
        self.gpu.queue().submit([enc.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.gpu.wait(None);
        let view = staging.slice(..).get_mapped_range().expect("webgpu: mapping a finished buffer");
        // (as bytes: every target wgpu runs on is little-endian)
        let mut host = vec![0f32; len];
        bytemuck::cast_slice_mut::<f32, u8>(&mut host).copy_from_slice(&view[..len * 4]);
        drop(view);
        staging.unmap();
        host
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_to_device(&self) -> Box<dyn ggml_rs::DeviceStorage> {
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let copy = vec_buffer(&self.gpu, self.v.len);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        if self.v.len > 0 {
            enc.copy_buffer_to_buffer(buffer(&self.v), 0, &copy, 0, (self.v.len * 4) as u64);
        }
        self.gpu.queue().submit([enc.finish()]);
        Box::new(Aliased { v: DeviceVec { len: self.v.len, inner: Arc::new(copy) }, gpu: Arc::clone(&self.gpu), serial: Arc::clone(&self.serial) })
    }
}

impl<'a> Recorder<'a> {
    /// A recording on `backend`, its bind groups kept (the crate's own measurements record kernels directly).
    #[cfg(test)]
    pub(crate) fn new(backend: &'a WgpuBackend) -> Self {
        Recorder { backend, dispatches: Vec::new(), reads: Vec::new(), keep: true, pooled: Vec::new(), spare: Vec::new(), q8: Vec::new(), x16: Vec::new(), att16: None, parts: None, exl3_tmp: None, moe_tmp: None, hold: false, held: Vec::new(), lists: Vec::new(), copied: 0, flushed: None, stamps: None, stamped: None, timed: Vec::new(), weight: 0.0 }
    }
}

/// One dispatch: its pipeline, bind group and grid.
use crate::Dispatch;

/// A bind group a chain makes again step after step: its pipeline, its three buffers and its parameters.
pub(crate) type GroupKey = (usize, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, [u32; 8]);

/// A bind group of the eight-buffer layout a chain makes again step after step (`Gpu::wide_layout`).
pub(crate) type WideKey = (usize, [wgpu::Buffer; 8], [u32; 8]);

/// Bind groups kept before the cache starts over (a cache grown from buffers that were replaced).
const KEEP_GROUPS: usize = 16384;

/// The work (FLOPs) a piece is submitted past ([`Recorder::weigh`]): 2^38 (some 0.3 s at a TFLOP a second), eight times
/// that on tensor cores; or `OAIY_PIECE_FLOPS`'s.
fn piece_flops(gpu: &crate::Gpu) -> f64 {
    static FLOPS: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    let set = *FLOPS.get_or_init(|| std::env::var("OAIY_PIECE_FLOPS").ok().and_then(|v| v.parse().ok()).filter(|&f: &f64| f > 0.0));
    // (a device that feeds its pieces a few at a time: the smaller ones, the one slow spell a card under a power limit
    // has through a long run of them 0.45 s shorter so, the 27B's 15,360 tokens 7.9 s the first time where 8.35, and
    // the run no slower after it)
    set.unwrap_or(if gpu.device.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) && !gpu.feeds() { (1u64 << 41) as f64 } else { (1u64 << 38) as f64 })
}

/// What a dispatch's workgroup counts for in a piece's work ([`piece_flops`]), whatever its kernel: its 256 values'
/// reads and writes, about what a thousand FLOPs a value take on the tensor cores (an elementwise op over 300 million
/// values a seventh of a piece: some 3 ms of 20, and under a throttled memory still well short of a second).
const WORKGROUP_FLOPS: f64 = 262_144.0;

/// Dispatches a piece of a run submits ([`Recorder::finish`]'s): 128, or `OAIY_PIECE`'s.
fn piece() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("OAIY_PIECE").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(128))
}

/// The most splits along k of a prompt's EXL3 projection on the tensor cores.
const COOP_SPLITS_MAX: usize = 8;

pub(crate) struct Recorder<'a> {
    backend: &'a WgpuBackend,
    /// Every op's dispatch, in order, run in one compute pass (a pass an op cost more than the ops).
    dispatches: Vec<Dispatch>,
    /// What to read back once they have run: the vector, the element it starts at, a staging buffer, the length.
    reads: Vec<(wgpu::Buffer, usize, wgpu::Buffer, usize)>,
    /// Whether bind groups are kept for the steps after ([`ChainRecorder::keep_groups`]).
    keep: bool,
    /// The scratch it took from the GPU's pool ([`Recorder::scratch`]), given back when it has run.
    pooled: Vec<(u64, wgpu::Buffer)>,
    /// Of that, what nothing recorded after reads (an input's f16 or int8 rows once the input is written): taken
    /// again before the pool's (a prompt's 27B chunk made 256 f16 copies, 3.6 GB held to its end).
    spare: Vec<(u64, wgpu::Buffer)>,
    /// Inputs of several rows quantized to int8 for the int8 kernels so far (the vector, its rows and width, the int8
    /// rows): each quantized once for the matmuls that read it, until something writes it.
    q8: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// Inputs of a prompt's rows as f16 for the tensor cores so far, as `q8`.
    x16: Vec<(wgpu::Buffer, usize, usize, DeviceVec)>,
    /// The f16 queries and cache rows of the tensor cores' attention: each attention's converted into them as it runs.
    att16: Option<(DeviceVec, DeviceVec)>,
    /// The parts of a tensor-core matmul split along k, each split matmul's in turn.
    parts: Option<DeviceVec>,
    /// A prompt's EXL3 projections' scratch (transformed inputs, partial sums, outputs before the map), each
    /// projection's in turn (a device's layers one recording: each its own, they would all be held to its end).
    exl3_tmp: Option<[DeviceVec; 3]>,
    /// A prompt's routed experts' scratch, each layer's in turn (its shape: rows, top k, hidden, ff).
    pub(crate) moe_tmp: Option<([usize; 4], std::sync::Arc<crate::exl3::Step>)>,
    /// Whether what is recorded waits for [`ChainRecorder::finish`] ([`ChainRecorder::hold`]), and the pieces encoded
    /// as it is (each its command buffer, submitted in turn when it is let go).
    hold: bool,
    held: Vec<wgpu::CommandBuffer>,
    /// The held pieces that go by the device's feed ([`Recorder::fed`]): each its dispatches, encoded at its turn.
    lists: Vec<Vec<Dispatch>>,
    /// The reads copied out by a flush (the first so many of `reads`), and that flush's submission: a recording
    /// waits for its own work, not what went after it (the next chunk's).
    copied: usize,
    flushed: Option<crate::Piece>,
    /// `OAIY_PIECE_STAMPS`: the pieces' timestamps, and how many pieces so far; once a flush has copied them out
    /// (with its reads: a finish waits for its own work, not for what went after), where to
    stamps: Option<(wgpu::QuerySet, u32)>,
    stamped: Option<wgpu::Buffer>,
    /// `OAIY_CHAIN_PROFILE`: each piece's kernels and where their timestamps are copied (read at the finish).
    timed: Vec<(Vec<Arc<wgpu::ComputePipeline>>, wgpu::Buffer)>,
    /// The work (FLOPs) the heavy ops have said of the piece so far ([`Recorder::weigh`]).
    weight: f64,
}

impl Recorder<'_> {
    /// A dispatch recorded; a piece of [`piece`]'s submitted as soon as it is recorded, so the GPU runs
    /// a run's first ops while the CPU records the rest. A later upload (`Queue::write_buffer`) lands after the pieces
    /// already submitted and before the ones after, as the recording's order has it.
    fn push(&mut self, d: Dispatch) {
        // (each workgroup some memory's worth of work whatever its kernel: an op over a large level's values is all
        // traffic, weighed by none, and a piece of 128 of them over a 3D decoder's 4.5 million voxels ran past the
        // OS's 2 s once the card's power limiter had its memory throttled: a lost device)
        let groups = d.2 .0 as f64 * d.2 .1 as f64 * d.2 .2 as f64;
        self.dispatches.push(d);
        self.weight += groups * WORKGROUP_FLOPS;
        if self.dispatches.len() >= piece() || self.weight >= piece_flops(self.gpu()) {
            self.submit_piece();
        }
    }

    /// The dispatches recorded so far encoded as a piece (one compute pass) and submitted (held: kept to be, see
    /// [`ChainRecorder::hold`]); profiled (`OAIY_CHAIN_PROFILE`), each its own pass between two timestamps (a piece
    /// at a time: the whole recording one submission would run past the OS's limit on one, Windows' 2 s).
    /// Whether this recording's pieces go by the device's feed as their dispatches (its pieces in flight are limited):
    /// not one that times its pieces or its kernels (their passes are its own to encode), nor one that has held
    /// command buffers already.
    fn fed(&self) -> bool {
        self.gpu().feeds() && self.held.is_empty() && !crate::profile::chain_on() && !crate::profile::pieces_on()
    }

    fn submit_piece(&mut self) {
        if !self.dispatches.is_empty() && self.fed() {
            let list = std::mem::take(&mut self.dispatches);
            self.weight = 0.0;
            if self.hold {
                self.lists.push(list);
            } else {
                self.gpu().feed(list);
            }
            return;
        }
        if !self.dispatches.is_empty() {
            let _one = self.backend.serial.lock().unwrap_or_else(|p| p.into_inner());
            let start = std::time::Instant::now();
            let mut piece = self.gpu().device.create_command_encoder(&Default::default());
            if crate::profile::chain_on() && self.gpu().device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
                let n = self.dispatches.len();
                let device = &self.gpu().device;
                let set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("oaiy-chain-profile"), ty: wgpu::QueryType::Timestamp, count: 2 * n as u32 });
                for (i, (pipeline, group, (x, y, z))) in self.dispatches.iter().enumerate() {
                    let timestamp_writes = Some(wgpu::ComputePassTimestampWrites { query_set: &set, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) });
                    let mut pass = piece.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(*x, *y, *z);
                }
                let bytes = 16 * n as u64;
                let resolved = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
                let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-profile-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
                piece.resolve_query_set(&set, 0..2 * n as u32, &resolved, 0);
                piece.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
                self.timed.push((self.dispatches.iter().map(|(p, _, _)| Arc::clone(p)).collect(), staging));
            } else {
                let i = self.next_stamp();
                let timestamp_writes = self.stamp_writes(i);
                let mut pass = piece.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes });
                for (pipeline, group, (x, y, z)) in &self.dispatches {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(*x, *y, *z);
                }
            }
            self.held.push(piece.finish());
            self.dispatches.clear();
            self.weight = 0.0;
            if !self.hold {
                let held = std::mem::take(&mut self.held);
                self.gpu().submit_piece(held);
            }
            crate::profile::add(&crate::profile::CHAIN_ENCODE, start);
        }
    }

    /// A heavy op's work (its FLOPs) added to the piece's: a piece past [`piece_flops`]'s is submitted then, so none
    /// runs long enough for the OS to reset the GPU (Windows: a submission past 2 s), however slow the GPU and big the
    /// work (a step of 4,096 tokens without tensor cores: 128 dispatches some 2.3 s).
    pub(crate) fn weigh(&mut self, flops: f64) {
        self.weight += flops;
        if self.weight >= piece_flops(self.gpu()) {
            self.submit_piece();
        }
    }

    /// `OAIY_PIECE_STAMPS`: the next piece's place among the recording's timestamps (up to 512 pieces).
    fn next_stamp(&mut self) -> Option<u32> {
        if !crate::profile::pieces_on() || !self.gpu().device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        if self.stamps.is_none() {
            let set = self.gpu().device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("oaiy-chain-pieces"), ty: wgpu::QueryType::Timestamp, count: 1024 });
            self.stamps = Some((set, 0));
        }
        let (_, n) = self.stamps.as_mut().expect("made");
        if *n >= 512 {
            return None;
        }
        *n += 1;
        Some(*n - 1)
    }

    /// The pieces' timestamps resolved and copied out by `enc` (where there are any): the buffer they are read from.
    fn resolve_pieces(&mut self, enc: &mut wgpu::CommandEncoder) -> Option<wgpu::Buffer> {
        let (set, n) = self.stamps.take().filter(|(_, n)| *n > 0)?;
        let device = &self.gpu().device;
        let bytes = 16 * n as u64;
        let resolved = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-pieces"), size: bytes, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-chain-pieces-read"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        enc.resolve_query_set(&set, 0..2 * n, &resolved, 0);
        enc.copy_buffer_to_buffer(&resolved, 0, &staging, 0, bytes);
        Some(staging)
    }

    /// Piece `i`'s pass's timestamps.
    fn stamp_writes(&self, i: Option<u32>) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let (set, _) = self.stamps.as_ref()?;
        let i = i?;
        Some(wgpu::ComputePassTimestampWrites { query_set: set, beginning_of_pass_write_index: Some(2 * i), end_of_pass_write_index: Some(2 * i + 1) })
    }

    /// A vector of `len` for this recording alone (a prompt's scratch): from the GPU's pool, given back when the
    /// recording has run, so nothing may keep it. Its values are whatever it last held.
    pub(crate) fn scratch(&mut self, len: usize) -> DeviceVec {
        let bytes = ((len.max(1) * 4) as u64).next_power_of_two().max(256);
        if let Some(i) = self.spare.iter().position(|(b, _)| *b == bytes) {
            let (_, b) = self.spare.swap_remove(i);
            return DeviceVec { len, inner: Arc::new(b) };
        }
        let b = self.gpu().pooled(bytes);
        self.pooled.push((bytes, b.clone()));
        DeviceVec { len, inner: Arc::new(b) }
    }

    /// The backend recorded on.
    pub(crate) fn backend(&self) -> &WgpuBackend {
        self.backend
    }

    /// Whether this recording keeps its bind groups.
    pub(crate) fn keeps(&self) -> bool {
        self.keep
    }

    pub(crate) fn gpu(&self) -> &Gpu {
        &self.backend.gpu
    }

    fn uniform(&self, words: &[u32]) -> wgpu::Buffer {
        let all = words8(words);
        let bytes: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf = self.gpu().device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-chain-params"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // (a new buffer: its write is in no waiting piece's way, so not after them as `queue`'s are)
        self.gpu().queue_raw.write_buffer(&buf, 0, &bytes);
        buf
    }

    /// A dispatch whose bind group is the same every step (its buffers and parameters): made once and kept.
    /// `b` written by what was just recorded: its int8 rows (if quantized) are stale.
    fn wrote(&mut self, b: &wgpu::Buffer) {
        let spare = &mut self.spare;
        let mut stale = |(x, _, _, v): &(wgpu::Buffer, usize, usize, DeviceVec)| {
            if x != b {
                return true;
            }
            // what read its rows is recorded: the scratch may be written again (wgpu orders the two)
            spare.push((buffer(v).size(), buffer(v).clone()));
            false
        };
        if !self.q8.is_empty() {
            self.q8.retain(&mut stale);
        }
        if !self.x16.is_empty() {
            self.x16.retain(&mut stale);
        }
    }

    pub(crate) fn dispatch_kept(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, words: &[u32], groups: (u32, u32, u32)) {
        self.wrote(at2);
        if !self.keep {
            let params = self.uniform(words);
            self.dispatch(pipeline, at0, at1, at2, &params, groups);
            return;
        }
        let key: GroupKey = (Arc::as_ptr(pipeline) as usize, at0.clone(), at1.clone(), at2.clone(), words8(words));
        let kept = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned();
        let group = match kept {
            Some(group) => group,
            None => {
                let params = self.uniform(words);
                let group = self.group(at0, at1, at2, &params);
                let mut groups = self.gpu().chain_groups.lock().unwrap_or_else(|p| p.into_inner());
                if groups.len() >= KEEP_GROUPS {
                    groups.clear();
                }
                groups.insert(key, group.clone());
                group
            }
        };
        self.push((Arc::clone(pipeline), group, groups));
    }

    fn group(&self, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer) -> wgpu::BindGroup {
        self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        })
    }

    fn dispatch(&mut self, pipeline: &Arc<wgpu::ComputePipeline>, at0: &wgpu::Buffer, at1: &wgpu::Buffer, at2: &wgpu::Buffer, params: &wgpu::Buffer, groups: (u32, u32, u32)) {
        self.wrote(at2);
        let group = self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("oaiy-chain"),
            layout: &self.gpu().layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: at0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: at1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
        self.push((Arc::clone(pipeline), group, groups));
    }

    pub(crate) fn named(&self, name: &'static str, body: &'static str) -> Arc<wgpu::ComputePipeline> {
        self.gpu().named_pipeline(name, || format!("{HEAD}{body}"))
    }

    /// A dispatch of an eight-buffer kernel (`Gpu::wide_layout`), its bind group kept as `dispatch_kept`'s.
    pub(crate) fn dispatch_wide(&mut self, name: &'static str, body: &str, bufs: [&wgpu::Buffer; 8], words: &[u32], groups: (u32, u32, u32)) {
        self.wrote(bufs[6]);
        self.wrote(bufs[7]);
        let pipeline = self.gpu().named_pipeline_wide(name, || body.to_string());
        let key: WideKey = (Arc::as_ptr(&pipeline) as usize, bufs.map(|b| b.clone()), words8(words));
        let kept = if self.keep { self.gpu().chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner()).get(&key).cloned() } else { None };
        let group = match kept {
            Some(group) => group,
            None => {
                let params = self.uniform(words);
                let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
                entries.push(wgpu::BindGroupEntry { binding: 8, resource: params.as_entire_binding() });
                let group = self.gpu().device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("oaiy-chain-wide"), layout: &self.gpu().wide_layout().0, entries: &entries });
                if self.keep {
                    let mut groups = self.gpu().chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner());
                    if groups.len() >= KEEP_GROUPS {
                        groups.clear();
                    }
                    groups.insert(key, group.clone());
                }
                group
            }
        };
        self.push((pipeline, group, groups));
    }
}
