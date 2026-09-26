//! WebGPU backend for ggml-rs: quantized weights on any GPU wgpu reaches
//! (Direct3D 12, Vulkan, Metal), for machines without CUDA.
//!
//! The load-bearing op of a GGUF model is `linear_q`: every projection reads a
//! quantized weight matrix. This backend uploads those matrices to the GPU in
//! their GGML block layout (so VRAM use equals the file's size) and runs the
//! matmul in WGSL (see [`shaders`]); everything else -- norms, RoPE, attention,
//! recurrent state -- runs on [`CpuBackend`], with activations on the host.
//! A projection is one upload of `x`, one dispatch and one read-back.
//!
//! Weights beyond the VRAM budget, or in a type without a shader, stay on the
//! host and take `CpuBackend`'s path: a model larger than the GPU still runs,
//! the rest of it on the CPU.

// VENDORED-LOCAL: this crate is nrob's addition beside ggml-rs-cuda.

pub mod shaders;

use ggml_quants::GgmlType;
use ggml_rs::{Backend, CpuBackend, QuantizedDeviceStorage, QuantizedTensor, RopeType, Tensor};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const GIB: u64 = 1 << 30;

/// Weight bytes one storage binding may cover: rows are split across buffers
/// below the adapter's binding limit (a 152k-vocab Q6_K output is ~640 MB).
fn chunk_limit(limits: &wgpu::Limits) -> u64 {
    (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size).min(1 << 30) & !3
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<GgmlType, Arc<wgpu::ComputePipeline>>>,
    limits: wgpu::Limits,
    /// Upload bytes written since the queue was last flushed.
    staged: AtomicU64,
}

/// A weight matrix on the GPU: its rows in one or more buffers.
struct WgpuQuant {
    gpu: Arc<Gpu>,
    dtype: GgmlType,
    /// `(buffer, first_row, rows)`; each buffer holds whole rows.
    chunks: Vec<(wgpu::Buffer, u32, u32)>,
    row_bytes: usize,
    nbytes: usize,
    used: Arc<AtomicU64>,
}

impl std::fmt::Debug for WgpuQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuQuant({:?}, {} bytes, {} chunk(s))", self.dtype, self.nbytes, self.chunks.len())
    }
}

impl Drop for WgpuQuant {
    fn drop(&mut self) {
        self.used.fetch_sub(self.nbytes as u64, Ordering::Relaxed);
    }
}

impl QuantizedDeviceStorage for WgpuQuant {
    fn nbytes(&self) -> usize {
        self.nbytes
    }
    fn dtype(&self) -> GgmlType {
        self.dtype
    }
    fn device_name(&self) -> &str {
        "webgpu"
    }
    fn copy_to_host(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nbytes);
        for (buffer, _, rows) in &self.chunks {
            let len = *rows as usize * self.row_bytes;
            out.extend_from_slice(&self.gpu.read(buffer, len as u64)[..len]);
        }
        out
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage> {
        let bytes = self.copy_to_host();
        self.used.fetch_add(self.nbytes as u64, Ordering::Relaxed);
        Box::new(WgpuQuant {
            gpu: Arc::clone(&self.gpu),
            dtype: self.dtype,
            chunks: self.gpu.upload_rows(&bytes, self.row_bytes, self.chunks.len()),
            row_bytes: self.row_bytes,
            nbytes: self.nbytes,
            used: Arc::clone(&self.used),
        })
    }
}

impl Gpu {
    /// Copy `len` bytes of `src` back to the host.
    fn read(&self, src: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let len = len.div_ceil(4) * 4;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nrob-readback"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, len);
        self.queue.submit([enc.finish()]);
        self.map_read(&staging, len)
    }

    fn map_read(&self, staging: &wgpu::Buffer, len: u64) -> Vec<u8> {
        let slice = staging.slice(..len);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("webgpu: device lost while waiting for a result");
        let out = slice.get_mapped_range().expect("webgpu: mapping a finished buffer").to_vec();
        staging.unmap();
        out
    }

    /// Upload whole rows into buffers below the binding limit.
    fn upload_rows(&self, bytes: &[u8], row_bytes: usize, _hint: usize) -> Vec<(wgpu::Buffer, u32, u32)> {
        let rows = bytes.len() / row_bytes;
        let per = ((chunk_limit(&self.limits) as usize) / row_bytes).max(1);
        let mut chunks = Vec::new();
        let mut r = 0;
        while r < rows {
            let n = per.min(rows - r);
            let data = &bytes[r * row_bytes..(r + n) * row_bytes];
            let size = (data.len() as u64).div_ceil(4) * 4;
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nrob-weight"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            if data.len() % 4 == 0 {
                self.queue.write_buffer(&buffer, 0, data);
            } else {
                let mut padded = data.to_vec();
                padded.resize(size as usize, 0);
                self.queue.write_buffer(&buffer, 0, &padded);
            }
            chunks.push((buffer, r as u32, n as u32));
            r += n;
            // `write_buffer` stages through host-visible memory that is only
            // recycled after a submission completes: flush as uploads pile up,
            // or loading a 16 GB model exhausts the staging pool.
            let pending = self.staged.fetch_add(size, Ordering::Relaxed) + size;
            if pending >= 256 << 20 {
                self.staged.store(0, Ordering::Relaxed);
                self.queue.submit([]);
                let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
            }
        }
        chunks
    }

    fn pipeline(&self, dtype: GgmlType) -> Option<Arc<wgpu::ComputePipeline>> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(&dtype) {
            return Some(Arc::clone(p));
        }
        let source = shaders::source(dtype)?;
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nrob-linear-q"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nrob-linear-q"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert(dtype, Arc::clone(&pipeline));
        Some(pipeline)
    }
}

/// Which adapter the backend runs on, for logs.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

pub struct WgpuBackend {
    cpu: CpuBackend,
    gpu: Arc<Gpu>,
    budget: u64,
    used: Arc<AtomicU64>,
    summary: AdapterSummary,
    /// One projection at a time: the dispatch and read-back share the queue.
    serial: Mutex<()>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend({} via {}, {} of {} GiB used)", self.summary.name, self.summary.backend,
            self.used.load(Ordering::Relaxed) / GIB, self.budget / GIB)
    }
}

impl WgpuBackend {
    /// Open the best adapter wgpu finds. `budget_bytes` caps the weights placed
    /// on it (WebGPU cannot report free memory); `None` picks a default by
    /// adapter type: 8 GiB discrete, 2 GiB integrated, none for software.
    pub fn new(budget_bytes: Option<u64>) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let info = adapter.get_info();
        let summary = AdapterSummary {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        };
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("nrob"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .map_err(|e| format!("WebGPU device on {}: {e}", info.name))?;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nrob-linear-q"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(
                    3,
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nrob-linear-q"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let budget = budget_bytes.unwrap_or(match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 8 * GIB,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2 * GIB,
            _ => 0,
        });
        Ok(Self {
            cpu: CpuBackend::new(),
            gpu: Arc::new(Gpu { device, queue, layout, pipeline_layout, pipelines: Mutex::new(HashMap::new()), limits, staged: AtomicU64::new(0) }),
            budget,
            used: Arc::new(AtomicU64::new(0)),
            summary,
            serial: Mutex::new(()),
        })
    }

    pub fn adapter(&self) -> &AdapterSummary {
        &self.summary
    }

    /// Bytes of weights placed on the GPU, and the budget.
    pub fn usage(&self) -> (u64, u64) {
        (self.used.load(Ordering::Relaxed), self.budget)
    }

    /// Whether `dtype` has a GPU kernel.
    pub fn supports(dtype: GgmlType) -> bool {
        shaders::layout(dtype).is_some()
    }

    fn upload(&self, w: QuantizedTensor) -> QuantizedTensor {
        let Some((elems, block_bytes, _)) = shaders::layout(w.dtype()) else { return w };
        if w.is_device() || w.rank() != 2 || w.dim(1) % elems as usize != 0 || self.gpu.pipeline(w.dtype()).is_none() {
            return w;
        }
        let row_bytes = w.dim(1) / elems as usize * block_bytes as usize;
        let nbytes = w.bytes().len();
        if nbytes != row_bytes * w.dim(0) || row_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return w;
        }
        // Reserve before uploading so concurrent loads cannot overshoot together.
        let prev = self.used.fetch_add(nbytes as u64, Ordering::Relaxed);
        if prev + nbytes as u64 > self.budget {
            self.used.fetch_sub(nbytes as u64, Ordering::Relaxed);
            return w;
        }
        let chunks = self.gpu.upload_rows(w.bytes(), row_bytes, 1);
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype: w.dtype(), chunks, row_bytes, nbytes, used: Arc::clone(&self.used) };
        QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec())
    }

    /// `y = x · Wᵀ` with `W` on the GPU.
    fn linear_gpu(&self, x: &Tensor, q: &WgpuQuant, shape: &[usize]) -> Tensor {
        let (n, k) = (shape[0], shape[1]);
        let host;
        let x = if x.is_device() {
            host = x.to_host();
            &host
        } else {
            x
        };
        let m = x.numel() / k;
        let mut out_shape = x.shape().to_vec();
        *out_shape.last_mut().expect("x has a last axis") = n;
        if m == 0 || n == 0 {
            return Tensor::from_vec(Vec::new(), out_shape);
        }
        // x and y are single bindings: a long prompt's rows go in batches that
        // stay under the adapter's binding and buffer limits.
        let limit = chunk_limit(&self.gpu.limits) as usize;
        let rows = (limit / (4 * k.max(n))).max(1);
        if m > rows {
            let data = x.data();
            let mut y = Vec::with_capacity(m * n);
            for start in (0..m).step_by(rows) {
                let count = rows.min(m - start);
                let part = Tensor::from_vec(data[start * k..(start + count) * k].to_vec(), vec![count, k]);
                y.extend_from_slice(self.linear_gpu(&part, q, shape).data());
            }
            return Tensor::from_vec(y, out_shape);
        }
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let gpu = &self.gpu;
        let pipeline = gpu.pipeline(q.dtype).expect("uploaded weights have a pipeline");
        let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let xbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("nrob-x"), size: (x.numel() * 4) as u64, usage, mapped_at_creation: false });
        gpu.queue.write_buffer(&xbuf, 0, &bytes(x.data()));
        let ysize = (m * n * 4) as u64;
        let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nrob-y"),
            size: ysize,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nrob-y-read"),
            size: ysize,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut groups = Vec::new();
        for (buffer, row0, rows) in &q.chunks {
            let params: Vec<u8> = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let pbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nrob-params"),
                size: params.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&pbuf, 0, &params);
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nrob-linear-q"),
                layout: &gpu.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: xbuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ybuf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pbuf.as_entire_binding() },
                ],
            });
            groups.push((group, *rows));
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (group, rows) in &groups {
                pass.set_bind_group(0, group, &[]);
                // Rows beyond 65535 wrap into the second grid axis.
                let gx = (*rows).min(65535);
                let gy = rows.div_ceil(65535);
                let gz = (m as u32).div_ceil(shaders::M_TILE);
                pass.dispatch_workgroups(gx, gy, gz);
            }
        }
        enc.copy_buffer_to_buffer(&ybuf, 0, &staging, 0, ysize);
        gpu.queue.submit([enc.finish()]);
        let raw = gpu.map_read(&staging, ysize);
        let y: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Tensor::from_vec(y, out_shape)
    }
}

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "webgpu"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn vram_status(&self) -> Option<(usize, usize)> {
        let (used, budget) = self.usage();
        Some((budget.saturating_sub(used) as usize, budget as usize))
    }
    fn to_device_quant(&self, w: QuantizedTensor) -> QuantizedTensor {
        self.upload(w)
    }
    fn try_to_device_quant(&self, w: QuantizedTensor, _safety_margin_bytes: usize) -> QuantizedTensor {
        // The budget is the whole weight allowance; activations live on the host.
        self.upload(w)
    }
    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        if let Some(q) = w.device_storage().and_then(|s| s.as_any().downcast_ref::<WgpuQuant>()) {
            return self.linear_gpu(x, q, w.shape());
        }
        self.cpu.linear_q(x, w)
    }

    // Everything else is CpuBackend's, forwarded explicitly so its optimized
    // overrides are kept rather than the trait's defaults.
    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        self.cpu.embed_lookup(table, tokens, embedding_dim)
    }
    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor {
        self.cpu.linear(x, w)
    }
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        self.cpu.rmsnorm(x, weight, eps)
    }
    fn softmax_last(&self, x: &mut Tensor) {
        self.cpu.softmax_last(x)
    }
    fn silu(&self, x: &Tensor) -> Tensor {
        self.cpu.silu(x)
    }
    fn gelu_approx(&self, x: &Tensor) -> Tensor {
        self.cpu.gelu_approx(x)
    }
    fn add_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.add_inplace(x, y)
    }
    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor) {
        self.cpu.mul_inplace(x, y)
    }
    fn rope(&self, x: &mut Tensor, positions: &[u32], head_dim: usize, rope_type: RopeType, theta: f32, freq_factors: Option<&[f32]>) {
        self.cpu.rope(x, positions, head_dim, rope_type, theta, freq_factors)
    }
    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor {
        self.cpu.repeat_kv(x, n_rep)
    }
    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor {
        self.cpu.bmm_qkt(q, k, scale, past)
    }
    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        self.cpu.bmm_av(scores, v)
    }
    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        self.cpu.argmax_last(x)
    }
}

#[cfg(test)]
mod tests;
