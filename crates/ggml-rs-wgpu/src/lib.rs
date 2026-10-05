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

// VENDORED-LOCAL: this crate is OAIY's addition beside ggml-rs-cuda.

pub mod chain;
pub mod dense;
pub mod exl3;
pub mod shaders;

/// Where a GGUF model's time goes on this backend: counters any thread adds to and a timing test reads and resets.
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// A quantized projection's call: all of it, and the part spent waiting for the GPU (submit to mapped).
    pub static LINEAR: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    pub static LINEAR_WAIT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
    /// The attention (on the CPU: the cache is on the host).
    pub static ATTENTION: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

    pub(crate) fn add(counter: &[AtomicU64; 2], start: Instant) {
        counter[0].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        counter[1].fetch_add(1, Ordering::Relaxed);
    }

    /// The counters since the last call, as one line, and reset.
    pub fn take_line() -> String {
        let take = |c: &[AtomicU64; 2]| (c[0].swap(0, Ordering::Relaxed) as f64 / 1e9, c[1].swap(0, Ordering::Relaxed));
        let (l, w, a) = (take(&LINEAR), take(&LINEAR_WAIT), take(&ATTENTION));
        format!("projections {:.3} s ({}), of it waiting for the GPU {:.3} s; attention {:.3} s ({})", l.0, l.1, w.0, a.0, a.1)
    }
}

use ggml_quants::GgmlType;
use ggml_rs::{Backend, CpuBackend, QuantizedDeviceStorage, QuantizedTensor, RopeType, Tensor};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const GIB: u64 = 1 << 30;

/// The adapter's own memory where its API says: Vulkan's largest device-local heap (a discrete card's VRAM). None on
/// Direct3D 12 and Metal.
fn device_memory(adapter: &wgpu::Adapter) -> Option<u64> {
    #[cfg(any(windows, target_os = "linux"))]
    {
        // SAFETY: the adapter's handles are only read, by a query that creates and frees nothing.
        let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
        let props = unsafe { a.shared_instance().raw_instance().get_physical_device_memory_properties(a.raw_physical_device()) };
        let heaps = &props.memory_heaps[..(props.memory_heap_count as usize).min(props.memory_heaps.len())];
        // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
        heaps.iter().filter(|h| h.flags.as_raw() & 1 != 0).map(|h| h.size).max()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = adapter;
        None
    }
}

/// The weights a discrete card with `memory` bytes holds by default: all but 4 GiB (the cache, the work buffers and
/// the rest of the computer's use of it), or half of a card under 8 GiB.
fn discrete_budget(memory: u64) -> u64 {
    memory.saturating_sub(4 * GIB).max(memory / 2)
}

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
    pipelines: Mutex<HashMap<(GgmlType, u8), Arc<wgpu::ComputePipeline>>>,
    /// The EXL3 matmul's pipelines (`exl3::shader`): for one row, and for several. Made when first used.
    exl3: Mutex<[Option<Arc<wgpu::ComputePipeline>>; 2]>,
    /// Other kernels' pipelines by name (`dense`), made when first used.
    named: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    /// A chain's bind groups that are the same step after step (`chain`): by pipeline, buffers and parameters.
    chain_groups: Mutex<HashMap<chain::GroupKey, wgpu::BindGroup>>,
    /// The layout of the chain's kernels of eight buffers (a gated delta net's: six read, two written, then the
    /// parameters), made when first used, and their bind groups as `chain_groups`.
    wide: std::sync::OnceLock<(wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    chain_groups_wide: Mutex<HashMap<chain::WideKey, wgpu::BindGroup>>,
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

impl WgpuQuant {
    /// The weights as the GPU holds them.
    fn gpu_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.nbytes);
        for (buffer, _, rows) in &self.chunks {
            let len = *rows as usize * self.row_bytes;
            out.extend_from_slice(&self.gpu.read(buffer, len as u64)[..len]);
        }
        out
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
        let out = self.gpu_bytes();
        // ggml's layout, where the GPU's blocks are padded
        match shaders::padded_block(self.dtype) {
            Some((host, gpu)) => shaders::pad_blocks(&out, gpu, host),
            None => out,
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage> {
        let bytes = self.gpu_bytes();
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
            label: Some("oaiy-readback"),
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
                label: Some("oaiy-weight"),
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

    fn exl3_pipeline(&self, many: bool) -> Arc<wgpu::ComputePipeline> {
        let mut slots = self.exl3.lock().unwrap_or_else(|p| p.into_inner());
        let slot = &mut slots[many as usize];
        if let Some(p) = slot.as_ref() {
            return Arc::clone(p);
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oaiy-exl3"),
            source: wgpu::ShaderSource::Wgsl(exl3::shader(many).into()),
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-exl3"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        *slot = Some(Arc::clone(&pipeline));
        pipeline
    }

    /// The pipeline `name`, made from the WGSL `source` gives the first time it is asked for.
    fn named_pipeline(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        self.named_pipeline_in(name, &self.pipeline_layout, source)
    }

    /// The layout of eight storage buffers (the first six read, the last two written) and the parameters at 8.
    fn wide_layout(&self) -> &(wgpu::BindGroupLayout, wgpu::PipelineLayout) {
        self.wide.get_or_init(|| {
            let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8).map(|b| storage(b, b < 6)).collect();
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            });
            let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("oaiy-chain-wide"), entries: &entries });
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("oaiy-chain-wide"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            (layout, pipeline_layout)
        })
    }

    /// A named kernel of eight buffers ([`Gpu::wide_layout`]).
    fn named_pipeline_wide(&self, name: &'static str, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let layout = &self.wide_layout().1;
        self.named_pipeline_in(name, layout, source)
    }

    fn named_pipeline_in(&self, name: &'static str, layout: &wgpu::PipelineLayout, source: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.named.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(name) {
            return Arc::clone(p);
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(source().into()) });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert(name, Arc::clone(&pipeline));
        pipeline
    }

    /// `dtype`'s kernel for `m` rows of `x`: the decode kernel for one, the one-row kernel for a few, the tiled one
    /// a prompt takes.
    fn pipeline(&self, dtype: GgmlType, m: usize) -> Option<Arc<wgpu::ComputePipeline>> {
        let kind = shaders::kind(dtype, m);
        let mut cache = self.pipelines.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(p) = cache.get(&(dtype, kind)) {
            return Some(Arc::clone(p));
        }
        let source = match kind {
            1 => shaders::source_many(dtype)?,
            2 => shaders::source_decode(dtype)?,
            3 => shaders::source_multi(dtype)?,
            _ => shaders::source(dtype)?,
        };
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("oaiy-linear-q"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("oaiy-linear-q"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        cache.insert((dtype, kind), Arc::clone(&pipeline));
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
    /// One projection at a time: the dispatch and read-back share the queue (EXL3 weights hold it too).
    serial: Arc<Mutex<()>>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend({} via {}, {} of {} GiB used)", self.summary.name, self.summary.backend,
            self.used.load(Ordering::Relaxed) / GIB, self.budget / GIB)
    }
}

impl WgpuBackend {
    /// Open the best adapter wgpu finds, or the one `OAIY_WEBGPU_ADAPTER` names
    /// (part of its name, any case: "radeon", "arc", "5090"), for a computer with
    /// more than one GPU. `budget_bytes` caps the weights placed on it (WebGPU
    /// cannot report free memory); `None` picks a default: a discrete card's
    /// memory less 4 GiB where Vulkan says how much it has (27.8 GiB of a 32 GB
    /// card), else 8 GiB; 2 GiB integrated, none for software.
    ///
    /// Vulkan, D3D12 and Metal only, unless `WGPU_BACKEND` names others: an
    /// instance with OpenGL too starts a WGL thread in NVIDIA's GL driver, and
    /// that thread's exit, as an instance drops, deadlocked on the loader lock
    /// against another thread opening Vulkan (one test run in about forty hung).
    pub fn new(budget_bytes: Option<u64>) -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let adapter = match std::env::var("OAIY_WEBGPU_ADAPTER").ok().map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()) {
            Some(wanted) => {
                let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
                let names: Vec<String> = adapters.iter().map(|a| { let i = a.get_info(); format!("{} ({:?})", i.name, i.backend) }).collect();
                adapters
                    .into_iter()
                    .find(|a| a.get_info().name.to_lowercase().contains(&wanted))
                    .ok_or_else(|| format!("OAIY_WEBGPU_ADAPTER={wanted}: no WebGPU adapter has that in its name; there are {}", names.join(", ")))?
            }
            None => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .map_err(|e| format!("no WebGPU adapter: {e}"))?,
        };
        let info = adapter.get_info();
        let summary = AdapterSummary {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        };
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("oaiy"),
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
            label: Some("oaiy-linear-q"),
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
            label: Some("oaiy-linear-q"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let budget = budget_bytes.unwrap_or(match info.device_type {
            wgpu::DeviceType::DiscreteGpu => device_memory(&adapter).map_or(8 * GIB, discrete_budget),
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2 * GIB,
            _ => 0,
        });
        Ok(Self {
            cpu: CpuBackend::new(),
            gpu: Arc::new(Gpu { device, queue, layout, pipeline_layout, pipelines: Mutex::new(HashMap::new()), exl3: Mutex::new([None, None]), named: Mutex::new(HashMap::new()), chain_groups: Mutex::new(HashMap::new()), wide: std::sync::OnceLock::new(), chain_groups_wide: Mutex::new(HashMap::new()), limits, staged: AtomicU64::new(0) }),
            budget,
            used: Arc::new(AtomicU64::new(0)),
            summary,
            serial: Arc::new(Mutex::new(())),
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
        if w.is_device() || w.rank() != 2 || w.dim(1) % elems as usize != 0 || self.gpu.pipeline(w.dtype(), 2).is_none() {
            return w;
        }
        let row_bytes = w.dim(1) / elems as usize * block_bytes as usize;
        // ggml's bytes, and the GPU's: Q3_K's blocks padded (`shaders::padded_block`)
        let padded = shaders::padded_block(w.dtype());
        let host_row = padded.map_or(row_bytes, |(host, gpu)| row_bytes / gpu * host);
        if w.bytes().len() != host_row * w.dim(0) || row_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return w;
        }
        let nbytes = row_bytes * w.dim(0);
        // Reserve before uploading so concurrent loads cannot overshoot together.
        let prev = self.used.fetch_add(nbytes as u64, Ordering::Relaxed);
        if prev + nbytes as u64 > self.budget {
            self.used.fetch_sub(nbytes as u64, Ordering::Relaxed);
            return w;
        }
        let chunks = match padded {
            Some((host, gpu)) => self.gpu.upload_rows(&shaders::pad_blocks(w.bytes(), host, gpu), row_bytes, 1),
            None => self.gpu.upload_rows(w.bytes(), row_bytes, 1),
        };
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype: w.dtype(), chunks, row_bytes, nbytes, used: Arc::clone(&self.used) };
        QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec())
    }

    /// `y = x · Wᵀ` with `W` on the GPU.
    fn linear_gpu(&self, x: &Tensor, q: &WgpuQuant, shape: &[usize]) -> Tensor {
        let start = std::time::Instant::now();
        let y = self.linear_gpu_inner(x, q, shape);
        profile::add(&profile::LINEAR, start);
        y
    }

    fn linear_gpu_inner(&self, x: &Tensor, q: &WgpuQuant, shape: &[usize]) -> Tensor {
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
                y.extend_from_slice(self.linear_gpu_inner(&part, q, shape).data());
            }
            return Tensor::from_vec(y, out_shape);
        }
        self.linear_gpu_batch(x, &[(q, shape)]).pop().expect("one weight, one result")
    }

    /// Several weights of one input, `x` (`[m, k]` on the host, few enough rows for one binding), in one submit and
    /// one read back: each weight's `[m, n]`, in order. The input goes up once.
    fn linear_gpu_batch(&self, x: &Tensor, ws: &[(&WgpuQuant, &[usize])]) -> Vec<Tensor> {
        let k = ws[0].1[1];
        let m = x.numel() / k;
        let _one = self.serial.lock().unwrap_or_else(|p| p.into_inner());
        let gpu = &self.gpu;
        let bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for f in v {
                out.extend_from_slice(&f.to_le_bytes());
            }
            out
        };
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let xbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("oaiy-x"), size: (x.numel() * 4) as u64, usage, mapped_at_creation: false });
        gpu.queue.write_buffer(&xbuf, 0, &bytes(x.data()));
        let sizes: Vec<u64> = ws.iter().map(|(_, shape)| (m * shape[0] * 4) as u64).collect();
        let total: u64 = sizes.iter().sum();
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-y-read"),
            size: total,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let mut ybufs = Vec::with_capacity(ws.len());
        for ((q, shape), &ysize) in ws.iter().zip(&sizes) {
            let n = shape[0];
            let pipeline = gpu.pipeline(q.dtype, m).expect("uploaded weights have a pipeline");
            let ybuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("oaiy-y"),
                size: ysize,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut groups = Vec::new();
            for (buffer, row0, rows) in &q.chunks {
                let params: Vec<u8> = [k as u32, n as u32, m as u32, *row0, *rows, q.row_bytes as u32, 0, 0]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let pbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("oaiy-params"),
                    size: params.len() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue.write_buffer(&pbuf, 0, &params);
                let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("oaiy-linear-q"),
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
                    let (gx, gy, gz) = shaders::grid(q.dtype, m, *rows);
                    pass.dispatch_workgroups(gx, gy, gz);
                }
            }
            ybufs.push(ybuf);
        }
        let mut at = 0;
        for (ybuf, &ysize) in ybufs.iter().zip(&sizes) {
            enc.copy_buffer_to_buffer(ybuf, 0, &staging, at, ysize);
            at += ysize;
        }
        let waiting = std::time::Instant::now();
        gpu.queue.submit([enc.finish()]);
        let slice = staging.slice(..total);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("webgpu: device lost while waiting for a result");
        profile::add(&profile::LINEAR_WAIT, waiting);
        // each output made straight from the mapped bytes (a copy of them first was a second pass over a prompt's
        // outputs, up to 131 MB a call)
        let raw = slice.get_mapped_range().expect("webgpu: mapping a finished buffer");
        let mut at = 0usize;
        let out = ws
            .iter()
            .zip(&sizes)
            .map(|((_, shape), &ysize)| {
                let y: Vec<f32> = raw[at..at + ysize as usize].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                at += ysize as usize;
                let mut out_shape = x.shape().to_vec();
                *out_shape.last_mut().expect("x has a last axis") = shape[0];
                Tensor::from_vec(y, out_shape)
            })
            .collect();
        drop(raw);
        staging.unmap();
        out
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
    /// Weights of one input that are all on this adapter, in one submit: a layer's q, k and v were three round trips
    /// a decode step.
    fn linear_q_many(&self, x: &Tensor, ws: &[&QuantizedTensor]) -> Vec<Tensor> {
        let on_gpu: Option<Vec<(&WgpuQuant, &[usize])>> =
            ws.iter().map(|w| Some((w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?, w.shape()))).collect();
        let host;
        let x = if x.is_device() {
            host = x.to_host();
            &host
        } else {
            x
        };
        match on_gpu {
            Some(on_gpu) if on_gpu.len() > 1 && !x.data().is_empty() => {
                let k = on_gpu[0].1[1];
                let m = x.numel() / k;
                let widest = on_gpu.iter().map(|(_, s)| s[0]).max().unwrap_or(0);
                let fits = m <= (chunk_limit(&self.gpu.limits) as usize / (4 * k.max(widest))).max(1);
                if on_gpu.iter().all(|(_, s)| s[1] == k && s[0] > 0) && fits {
                    let start = std::time::Instant::now();
                    let ys = self.linear_gpu_batch(x, &on_gpu);
                    profile::add(&profile::LINEAR, start);
                    return ys;
                }
                ws.iter().map(|w| self.linear_q(x, w)).collect()
            }
            _ => ws.iter().map(|w| self.linear_q(x, w)).collect(),
        }
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
    fn silu_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.silu_mul_split(fused, ff)
    }
    fn chain(&self) -> Option<&dyn ggml_rs::chain::DeviceChain> {
        Some(self)
    }
    fn gelu_approx_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        self.cpu.gelu_approx_mul_split(fused, ff)
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
    /// The CPU's fused attention (the cache is on the host): the default would copy the cache and take the softmax on
    /// one thread.
    fn attention(&self, q: &Tensor, k_buffer: &Tensor, v_buffer: &Tensor, kv_len: usize, scale: f32, past: usize, sliding_window: Option<usize>) -> Tensor {
        let start = std::time::Instant::now();
        let host = |t: &Tensor| if t.is_device() { t.to_host() } else { t.clone() };
        let out = if q.is_device() || k_buffer.is_device() || v_buffer.is_device() {
            self.cpu.attention(&host(q), &host(k_buffer), &host(v_buffer), kv_len, scale, past, sliding_window)
        } else {
            self.cpu.attention(q, k_buffer, v_buffer, kv_len, scale, past, sliding_window)
        };
        profile::add(&profile::ATTENTION, start);
        out
    }
    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        self.cpu.argmax_last(x)
    }
}

#[cfg(test)]
mod tests;

/// `(available, total)` bytes of the computer's memory, for a build without CUDA: what the portable engine sizes its
/// expert cache from (a streamed MoE model such as GLM-5.3-Flash), as the CUDA build does from
/// `ggml_rs_cuda::host_memory`. Without it the portable build assumed 32 GB on any computer, so a model of 190 GB read
/// most of each token's experts from the disk on a computer with 192 GB.
///
/// The same reader as `ggml_rs_cuda::host_memory`, here because this is the crate every portable build links and
/// `ggml-rs` and `llama-rs` keep no `unsafe` at all: on Windows it is one documented call. `None` on a platform it has
/// no reader for, so a caller keeps a fallback.
pub fn host_memory() -> Option<(usize, usize)> {
    host_memory_impl()
}

#[cfg(windows)]
fn host_memory_impl() -> Option<(usize, usize)> {
    // MEMORYSTATUSEX, exactly as documented: `length` must be set by the caller and every other field is written by
    // the call.
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `s` is a fully initialised MEMORYSTATUSEX of the size its own `length` field declares, and the call only
    // writes into it. The pointer is valid for the duration of the call and nothing retains it.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    (ok != 0).then_some((s.avail_phys as usize, s.total_phys as usize))
}

#[cfg(target_os = "linux")]
fn host_memory_impl() -> Option<(usize, usize)> {
    // /proc/meminfo reports both, in kB. MemAvailable is the kernel's own estimate of what a new allocation can have;
    // MemFree undercounts badly because of the page cache.
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |name: &str| -> Option<usize> {
        text.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<usize>().ok()).map(|kb| kb * 1024)
    };
    Some((field("MemAvailable:")?, field("MemTotal:")?))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn host_memory_impl() -> Option<(usize, usize)> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    /// Whatever the platform, the answer must be self-consistent or absent.
    #[test]
    fn host_memory_is_plausible_or_absent() {
        if let Some((free, total)) = super::host_memory() {
            assert!(total > (1 << 30), "total {total} is implausibly small");
            assert!(free <= total, "free {free} exceeds total {total}");
        }
    }
}
