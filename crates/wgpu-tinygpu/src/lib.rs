//! A wgpu adapter for an NVIDIA card that a Mac reaches only through tinygrad (its TinyGPU app): WebGPU's compute,
//! its kernels as CUDA.
//!
//! macOS has no driver for an NVIDIA card in a Thunderbolt enclosure, so no WebGPU adapter of wgpu's (Metal on a Mac)
//! sees one. tinygrad drives such a card itself; `tools/tinygpu/webgpu_server.py` holds it in one process and offers
//! what WebGPU's compute needs over a Unix socket (buffers, writes and reads, kernels, submissions). [`adapter`] is a
//! `wgpu::Adapter` (wgpu's `custom` backend) that asks it: a program written against wgpu's compute runs on the card as
//! it is. A compute pipeline's WGSL is made CUDA by `wgsl-cuda` and compiled by the server (nvcc); a command buffer's
//! dispatches, copies and clears go in one message at `Queue::submit`, run in order on the card's compute queue.
//!
//! Compute alone: render pipelines, textures, samplers, queries and acceleration structures are refused (a panic
//! naming them, as wgpu's own validation errors panic without a handler).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::ops::Range;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::custom::*;

// ---- the connection ----

const HELLO: u32 = 1;
const ALLOC: u32 = 2;
const FREE: u32 = 3;
const WRITE: u32 = 4;
const READ: u32 = 5;
const PROGRAM: u32 = 6;
const SUBMIT: u32 = 7;
const SYNC: u32 = 8;

/// The server's requests as this adapter makes them (tools/tinygpu/webgpu_server.py's PROTOCOL): a server of another
/// version would read a kernel or a dispatch other than it was sent, so it is refused at HELLO.
const PROTOCOL: u32 = 3;

/// The server's socket, one request at a time, and the buffers let go that are kept for the next of their size.
#[derive(Debug)]
struct Conn {
    stream: Mutex<UnixStream>,
    pool: Mutex<Pool>,
}

/// Buffers this program let go, kept by size for the next buffer of that size: a language model's token makes and
/// lets go dozens, each two requests to the server (a round trip each) and a launch of its clear there. A buffer taken
/// from it is cleared (WebGPU's buffers begin as zeros) before anything else reaches the server: in the same submission
/// as the next work where that comes next, else in one of its own first. Up to [`POOL_BYTES`] of buffers of
/// [`POOL_LARGEST`] or less; TINYGPU_NO_POOL=1 keeps none.
#[derive(Debug, Default)]
struct Pool {
    free: std::collections::HashMap<u64, Vec<u64>>,
    bytes: u64,
    /// Buffers taken from it, to be cleared (id, size) before the next request.
    clears: Vec<(u64, u64)>,
    off: bool,
}

const POOL_BYTES: u64 = 256 << 20;
const POOL_LARGEST: u64 = 64 << 20;

impl Conn {
    fn new(stream: UnixStream) -> Conn {
        let off = std::env::var("TINYGPU_NO_POOL").is_ok_and(|v| v == "1");
        Conn { stream: Mutex::new(stream), pool: Mutex::new(Pool { off, ..Pool::default() }) }
    }

    /// A buffer of `size` bytes: one the pool has (to be cleared), else a new one from the server.
    fn alloc(&self, size: u64) -> u64 {
        {
            let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(id) = pool.free.get_mut(&size).and_then(Vec::pop) {
                pool.bytes -= size;
                pool.clears.push((id, size));
                return id;
            }
        }
        u64_of(&self.must(ALLOC, &size.to_le_bytes()))
    }

    /// Buffer `id` (of `size` bytes) let go: into the pool where it has room, else freed on the server.
    fn release(&self, id: u64, size: u64) {
        {
            let mut pool = self.pool.lock().unwrap_or_else(|p| p.into_inner());
            if !pool.off && size <= POOL_LARGEST && pool.bytes + size <= POOL_BYTES {
                pool.bytes += size;
                pool.free.entry(size).or_default().push(id);
                return;
            }
        }
        let _ = self.call(FREE, &id.to_le_bytes());
    }

    fn call(&self, cmd: u32, payload: &[u8]) -> Result<Vec<u8>, String> {
        self.call_parts(cmd, &[payload])
    }

    /// A request whose payload is `parts`, one after another: sent as they are, not copied into one (a write's bytes
    /// are a model's weights, hundreds of megabytes a tensor). The pool's buffers taken since the last request are
    /// cleared first: at the head of a submission, or in a submission of their own before any other request.
    fn call_parts(&self, cmd: u32, parts: &[&[u8]]) -> Result<Vec<u8>, String> {
        let clears = std::mem::take(&mut self.pool.lock().unwrap_or_else(|p| p.into_inner()).clears);
        if !clears.is_empty() {
            let mut ops = Vec::with_capacity(25 * clears.len());
            for (id, size) in clears {
                ops.push(3u8);
                ops.extend(id.to_le_bytes());
                ops.extend(0u64.to_le_bytes());
                ops.extend(size.to_le_bytes());
            }
            if cmd == SUBMIT {
                let mut all = vec![ops.as_slice()];
                all.extend_from_slice(parts);
                return self.send(cmd, &all);
            }
            self.send(SUBMIT, &[&ops])?;
        }
        self.send(cmd, parts)
    }

    /// The request itself.
    fn send(&self, cmd: u32, parts: &[&[u8]]) -> Result<Vec<u8>, String> {
        let mut s = self.stream.lock().unwrap_or_else(|p| p.into_inner());
        let len: usize = parts.iter().map(|p| p.len()).sum();
        let mut head = Vec::with_capacity(12 + if len <= 64 << 10 { len } else { 0 });
        head.extend(cmd.to_le_bytes());
        head.extend((len as u64).to_le_bytes());
        // (a small request goes in one write, as it did)
        if len <= 64 << 10 {
            parts.iter().for_each(|p| head.extend_from_slice(p));
            s.write_all(&head).map_err(|e| format!("tinygpu: {e}"))?;
        } else {
            s.write_all(&head).map_err(|e| format!("tinygpu: {e}"))?;
            for p in parts {
                s.write_all(p).map_err(|e| format!("tinygpu: {e}"))?;
            }
        }
        let mut answer = [0u8; 12];
        s.read_exact(&mut answer).map_err(|e| format!("tinygpu: {e}"))?;
        let status = u32::from_le_bytes(answer[..4].try_into().unwrap());
        let n = u64::from_le_bytes(answer[4..].try_into().unwrap()) as usize;
        let mut out = vec![0u8; n];
        s.read_exact(&mut out).map_err(|e| format!("tinygpu: {e}"))?;
        if status != 0 {
            return Err(format!("tinygpu: {}", String::from_utf8_lossy(&out)));
        }
        Ok(out)
    }

    /// A call that cannot fail but by the server's error, which is the program's (wgpu panics on its errors too).
    fn must(&self, cmd: u32, payload: &[u8]) -> Vec<u8> {
        self.call(cmd, payload).unwrap_or_else(|e| panic!("{e}"))
    }
}

/// The socket's buffers made large: a Mac gives a local socket 8 KB each way, and a model's weights then cross it a
/// few kilobytes a call, at half what the card's link takes (1.4 GB/s, where 4 MB buffers give 2.4 and the link 2.6).
fn roomy(stream: &UnixStream) {
    use std::os::fd::AsRawFd;
    let size: libc::c_int = 4 << 20;
    for option in [libc::SO_SNDBUF, libc::SO_RCVBUF] {
        // (a smaller buffer only costs speed: what the system will not give is left as it is)
        unsafe {
            libc::setsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, option, (&size as *const libc::c_int).cast(), std::mem::size_of::<libc::c_int>() as libc::socklen_t);
        }
    }
}

fn u64_of(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
}

/// The default socket: where `tools/tinygpu/webgpu_server.py` listens unless told otherwise.
pub fn default_socket() -> PathBuf {
    std::env::var_os("TINYGPU_SOCKET").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/tinygpu-webgpu/server.sock")
    })
}

/// The card the server at `socket` holds, as a `wgpu::Adapter`.
pub fn adapter(socket: &Path) -> Result<wgpu::Adapter, String> {
    let stream = UnixStream::connect(socket).map_err(|e| format!("tinygpu: no server at {} ({e}): start tools/tinygpu/webgpu_server.py", socket.display()))?;
    roomy(&stream);
    let conn = Arc::new(Conn::new(stream));
    let hello = String::from_utf8_lossy(&conn.call(HELLO, &[])?).into_owned();
    let field = |key: &str| -> Option<String> {
        let at = hello.find(&format!("\"{key}\""))? + key.len() + 2;
        let rest = hello[at..].trim_start_matches([':', ' ']);
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..end].trim().trim_matches('"').to_string())
    };
    let protocol = field("protocol").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
    if protocol != PROTOCOL {
        return Err(format!("tinygpu: the server at {} speaks version {protocol} of its requests, this program {PROTOCOL}: run the webgpu_server.py of this program's version", socket.display()));
    }
    let arch = field("arch").unwrap_or_default();
    let memory = field("memory").and_then(|m| m.parse().ok()).unwrap_or(24u64 << 30);
    let name = field("name").unwrap_or_else(|| format!("NVIDIA {arch} via tinygrad (TinyGPU)"));
    Ok(wgpu::Adapter::from_custom(TgAdapter { conn, arch, name, memory }))
}

// ---- adapter and device ----

#[derive(Debug)]
struct TgAdapter {
    conn: Arc<Conn>,
    arch: String,
    name: String,
    memory: u64,
}

fn limits(memory: u64) -> wgpu::Limits {
    let mut l = wgpu::Limits::default();
    l.max_buffer_size = memory;
    l.max_storage_buffer_binding_size = (u32::MAX & !255) as _;
    l.max_uniform_buffer_binding_size = 65536;
    l.max_storage_buffers_per_shader_stage = 31;
    l.max_uniform_buffers_per_shader_stage = 12;
    l.max_bindings_per_bind_group = 1000;
    l.max_bind_groups = 4;
    l.max_dynamic_storage_buffers_per_pipeline_layout = 8;
    l.max_dynamic_uniform_buffers_per_pipeline_layout = 8;
    // (CUDA's static shared memory: 48 KB)
    l.max_compute_workgroup_storage_size = 48 << 10;
    l.max_compute_invocations_per_workgroup = 1024;
    l.max_compute_workgroup_size_x = 1024;
    l.max_compute_workgroup_size_y = 1024;
    l.max_compute_workgroup_size_z = 64;
    l.max_compute_workgroups_per_dimension = 65535;
    l.min_storage_buffer_offset_alignment = 32;
    l.min_uniform_buffer_offset_alignment = 256;
    l
}

fn info(name: &str, arch: &str) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: name.to_string(),
        vendor: 0x10de,
        device: 0,
        device_type: wgpu::DeviceType::DiscreteGpu,
        device_pci_bus_id: "tinygpu".into(),
        driver: "tinygrad (TinyGPU)".into(),
        driver_info: arch.to_string(),
        backend: wgpu::Backend::Noop,
        subgroup_min_size: 32,
        subgroup_max_size: 32,
        transient_saves_memory: None,
        limit_bucket: None,
    }
}

impl AdapterInterface for TgAdapter {
    fn request_device(&self, _desc: &wgpu::DeviceDescriptor<'_>) -> Pin<Box<dyn RequestDeviceFuture>> {
        let device = TgDevice { conn: Arc::clone(&self.conn), memory: self.memory, info: info(&self.name, &self.arch) };
        let queue = TgQueue { conn: Arc::clone(&self.conn), submitted: AtomicU64::new(0) };
        Box::pin(std::future::ready(Ok((DispatchDevice::custom(device), DispatchQueue::custom(queue)))))
    }
    fn is_surface_supported(&self, _surface: &DispatchSurface) -> bool {
        false
    }
    fn features(&self) -> wgpu::Features {
        features()
    }
    fn limits(&self) -> wgpu::Limits {
        limits(self.memory)
    }
    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }
    fn get_info(&self) -> wgpu::AdapterInfo {
        info(&self.name, &self.arch)
    }
    fn get_texture_format_features(&self, _format: wgpu::TextureFormat) -> wgpu::TextureFormatFeatures {
        unsupported("textures")
    }
    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        wgpu::PresentationTimestamp::INVALID_TIMESTAMP
    }
    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        // (wmma's 16 x 16 x 16 on the tensor cores: f16's into f32 sums and into f16 ones; `wgsl-cuda`'s fragments)
        if !features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            return Vec::new();
        }
        [wgpu::CooperativeScalarType::F32, wgpu::CooperativeScalarType::F16]
            .map(|sums| wgpu::wgt::CooperativeMatrixProperties { m_size: 16, n_size: 16, k_size: 16, ab_type: wgpu::CooperativeScalarType::F16, cr_type: sums, saturating_accumulation: false })
            .to_vec()
    }
}

/// f16 in kernels, and the tensor cores' cooperative matrices (`TINYGPU_NO_COOP` set: none).
fn features() -> wgpu::Features {
    if std::env::var_os("TINYGPU_NO_COOP").is_some() {
        wgpu::Features::SHADER_F16
    } else {
        wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX
    }
}

fn unsupported(what: &str) -> ! {
    panic!("tinygpu: {what} are not WebGPU compute, which is all the TinyGPU adapter does")
}

#[derive(Debug)]
struct TgDevice {
    conn: Arc<Conn>,
    memory: u64,
    info: wgpu::AdapterInfo,
}

#[derive(Debug)]
struct TgShaderModule {
    wgsl: String,
}
impl ShaderModuleInterface for TgShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn ShaderCompilationInfoFuture>> {
        Box::pin(std::future::ready(wgpu::CompilationInfo { messages: Vec::new() }))
    }
}

/// A layout's bindings that take a dynamic offset, by number.
#[derive(Debug)]
struct TgBindGroupLayout {
    dynamic: Vec<u32>,
}
impl BindGroupLayoutInterface for TgBindGroupLayout {}

#[derive(Debug)]
struct TgPipelineLayout;
impl PipelineLayoutInterface for TgPipelineLayout {}

/// A bind group: each binding's buffer, offset and size (None: to the buffer's end), and which take a dynamic offset
/// (in the layout's order).
#[derive(Debug)]
struct TgBindGroup {
    entries: HashMap<u32, (Arc<Allocation>, u64, Option<u64>)>,
    dynamic: Vec<u32>,
}
impl BindGroupInterface for TgBindGroup {}

#[derive(Debug)]
struct TgComputePipeline {
    program: u64,
    /// The kernel's bindings, in the order it takes them.
    bindings: Vec<(u32, u32)>,
}
impl ComputePipelineInterface for TgComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> DispatchBindGroupLayout {
        let _ = index;
        DispatchBindGroupLayout::custom(TgBindGroupLayout { dynamic: Vec::new() })
    }
}

impl DeviceInterface for TgDevice {
    fn features(&self) -> wgpu::Features {
        features()
    }
    fn limits(&self) -> wgpu::Limits {
        limits(self.memory)
    }
    fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.info.clone()
    }
    fn create_shader_module(&self, desc: wgpu::ShaderModuleDescriptor<'_>, _checks: wgpu::ShaderRuntimeChecks) -> DispatchShaderModule {
        let wgsl = match desc.source {
            wgpu::ShaderSource::Wgsl(src) => src.into_owned(),
            _ => panic!("tinygpu: a kernel is taken as WGSL"),
        };
        DispatchShaderModule::custom(TgShaderModule { wgsl })
    }
    unsafe fn create_shader_module_passthrough(&self, _desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>) -> DispatchShaderModule {
        panic!("tinygpu: a kernel is taken as WGSL")
    }
    fn create_bind_group_layout(&self, desc: &wgpu::BindGroupLayoutDescriptor<'_>) -> DispatchBindGroupLayout {
        let mut dynamic: Vec<u32> = desc
            .entries
            .iter()
            .filter(|e| matches!(e.ty, wgpu::BindingType::Buffer { has_dynamic_offset: true, .. }))
            .map(|e| e.binding)
            .collect();
        dynamic.sort_unstable();
        DispatchBindGroupLayout::custom(TgBindGroupLayout { dynamic })
    }
    fn create_bind_group(&self, desc: &wgpu::BindGroupDescriptor<'_>) -> DispatchBindGroup {
        let mut entries = HashMap::new();
        for e in desc.entries {
            match &e.resource {
                wgpu::BindingResource::Buffer(b) => {
                    let buf = b.buffer.as_custom::<TgBuffer>().expect("tinygpu: a buffer of this adapter's");
                    entries.insert(e.binding, (Arc::clone(&buf.mem), b.offset, b.size.map(|s| s.get())));
                }
                _ => unsupported("bindings other than buffers"),
            }
        }
        let dynamic = desc.layout.as_custom::<TgBindGroupLayout>().map(|l| l.dynamic.clone()).unwrap_or_default();
        DispatchBindGroup::custom(TgBindGroup { entries, dynamic })
    }
    fn create_pipeline_layout(&self, _desc: &wgpu::PipelineLayoutDescriptor<'_>) -> DispatchPipelineLayout {
        DispatchPipelineLayout::custom(TgPipelineLayout)
    }
    fn create_render_pipeline(&self, _desc: &wgpu::RenderPipelineDescriptor<'_>) -> DispatchRenderPipeline {
        unsupported("render pipelines")
    }
    fn create_mesh_pipeline(&self, _desc: &wgpu::MeshPipelineDescriptor<'_>) -> DispatchRenderPipeline {
        unsupported("mesh pipelines")
    }
    fn create_compute_pipeline(&self, desc: &wgpu::ComputePipelineDescriptor<'_>) -> DispatchComputePipeline {
        let module = desc.module.as_custom::<TgShaderModule>().expect("tinygpu: a shader module of this adapter's");
        let kernel = wgsl_cuda::translate(&module.wgsl, desc.entry_point).unwrap_or_else(|e| panic!("tinygpu: {}: {e}", desc.label.unwrap_or("a kernel")));
        let mut payload = Vec::new();
        for d in kernel.workgroup_size {
            payload.extend(d.to_le_bytes());
        }
        // (after its pointers the kernel takes the server's scratch where it has cooperative matrices, then each
        // binding's size: its indexes are kept in its arrays by them)
        payload.extend((kernel.bindings.len() as u32).to_le_bytes());
        payload.extend((kernel.scratch as u32).to_le_bytes());
        payload.extend(kernel.source.as_bytes());
        let program = u64_of(&self.conn.call(PROGRAM, &payload).unwrap_or_else(|e| panic!("{}: {e}", desc.label.unwrap_or("a kernel"))));
        let bindings = kernel.bindings.iter().map(|b| (b.group, b.binding)).collect();
        DispatchComputePipeline::custom(TgComputePipeline { program, bindings })
    }
    unsafe fn create_pipeline_cache(&self, _desc: &wgpu::PipelineCacheDescriptor<'_>) -> DispatchPipelineCache {
        unsupported("pipeline caches")
    }
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> DispatchBuffer {
        let id = self.conn.alloc(desc.size.max(4));
        let mapped = desc.mapped_at_creation.then(|| Mapping { start: 0, data: vec![0; desc.size as usize], write: true });
        let mem = Arc::new(Allocation { conn: Arc::clone(&self.conn), id, size: desc.size });
        DispatchBuffer::custom(TgBuffer { mem, size: desc.size, state: Arc::new(Mutex::new(mapped)) })
    }
    fn create_texture(&self, _desc: &wgpu::TextureDescriptor<'_>) -> DispatchTexture {
        unsupported("textures")
    }
    fn create_external_texture(&self, _desc: &wgpu::ExternalTextureDescriptor<'_>, _planes: &[&wgpu::TextureView]) -> DispatchExternalTexture {
        unsupported("textures")
    }
    fn create_blas(&self, _desc: &wgpu::CreateBlasDescriptor<'_>, _sizes: wgpu::BlasGeometrySizeDescriptors) -> (Option<u64>, DispatchBlas) {
        unsupported("acceleration structures")
    }
    fn create_tlas(&self, _desc: &wgpu::CreateTlasDescriptor<'_>) -> DispatchTlas {
        unsupported("acceleration structures")
    }
    fn create_sampler(&self, _desc: &wgpu::SamplerDescriptor<'_>) -> DispatchSampler {
        unsupported("samplers")
    }
    fn create_query_set(&self, _desc: &wgpu::QuerySetDescriptor<'_>) -> DispatchQuerySet {
        unsupported("query sets")
    }
    fn create_command_encoder(&self, _desc: &wgpu::CommandEncoderDescriptor<'_>) -> DispatchCommandEncoder {
        DispatchCommandEncoder::custom(TgEncoder { ops: Ops::default() })
    }
    fn create_render_bundle_encoder(&self, _desc: &wgpu::RenderBundleEncoderDescriptor<'_>) -> DispatchRenderBundleEncoder {
        unsupported("render bundles")
    }
    fn set_device_lost_callback(&self, _callback: BoxDeviceLostCallback) {}
    fn on_uncaptured_error(&self, _handler: Arc<dyn wgpu::UncapturedErrorHandler>) {}
    fn push_error_scope(&self, _filter: wgpu::ErrorFilter) -> u32 {
        0
    }
    fn pop_error_scope(&self, _index: u32) -> Pin<Box<dyn PopErrorScopeFuture>> {
        Box::pin(std::future::ready(None))
    }
    unsafe fn start_graphics_debugger_capture(&self) {}
    unsafe fn stop_graphics_debugger_capture(&self) {}
    fn poll(&self, poll_type: wgpu::wgt::PollType<u64>) -> Result<wgpu::PollStatus, wgpu::PollError> {
        // (every map's callback has been called by the time its map_async returned: a wait is the queue's)
        match poll_type {
            wgpu::wgt::PollType::Poll => Ok(wgpu::PollStatus::QueueEmpty),
            _ => {
                self.conn.must(SYNC, &[]);
                Ok(wgpu::PollStatus::WaitSucceeded)
            }
        }
    }
    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        wgpu::InternalCounters::default()
    }
    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        None
    }
    fn destroy(&self) {}
}

// ---- buffers ----

/// What of a buffer is mapped: its bytes from `start`, and whether they go back to the card at unmap.
#[derive(Debug)]
struct Mapping {
    start: u64,
    data: Vec<u8>,
    write: bool,
}

/// A buffer's memory on the card, freed there when the last that uses it lets it go: the program's handle, a bind
/// group, a command buffer not yet submitted (as wgpu's own buffers outlive their handle while work holds them).
#[derive(Debug)]
struct Allocation {
    conn: Arc<Conn>,
    id: u64,
    size: u64,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.conn.release(self.id, self.size.max(4));
    }
}

#[derive(Debug)]
struct TgBuffer {
    mem: Arc<Allocation>,
    size: u64,
    state: Arc<Mutex<Option<Mapping>>>,
}

fn read_range(conn: &Conn, id: u64, start: u64, len: u64) -> Vec<u8> {
    let mut p = Vec::with_capacity(24);
    p.extend(id.to_le_bytes());
    p.extend(start.to_le_bytes());
    p.extend(len.to_le_bytes());
    conn.must(READ, &p)
}

fn write_range(conn: &Conn, id: u64, start: u64, data: &[u8]) {
    let mut at = [0u8; 16];
    at[..8].copy_from_slice(&id.to_le_bytes());
    at[8..].copy_from_slice(&start.to_le_bytes());
    conn.call_parts(WRITE, &[&at, data]).unwrap_or_else(|e| panic!("{e}"));
}

impl BufferInterface for TgBuffer {
    fn map_async(&self, mode: wgpu::MapMode, range: Range<wgpu::BufferAddress>, callback: BufferMapCallback) {
        // (a buffer mapped for writing starts as it is, as WebGPU's does)
        let data = read_range(&self.mem.conn, self.mem.id, range.start, range.end - range.start);
        *self.state.lock().unwrap_or_else(|p| p.into_inner()) = Some(Mapping { start: range.start, data, write: mode == wgpu::MapMode::Write });
        callback(Ok(()));
    }
    fn get_mapped_range(&self, sub_range: Range<wgpu::BufferAddress>) -> Result<DispatchBufferMappedRange, wgpu::MapRangeError> {
        let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let m = state.as_ref().expect("tinygpu: a range of a buffer that is not mapped");
        let (a, b) = ((sub_range.start - m.start) as usize, (sub_range.end - m.start) as usize);
        Ok(DispatchBufferMappedRange::custom(TgMappedRange { data: m.data[a..b].to_vec(), back: m.write.then(|| (Arc::clone(&self.state), a)) }))
    }
    fn unmap(&self) {
        if let Some(m) = self.state.lock().unwrap_or_else(|p| p.into_inner()).take() {
            if m.write {
                write_range(&self.mem.conn, self.mem.id, m.start, &m.data);
            }
        }
    }
    fn destroy(&self) {}
}

/// A mapped range's bytes, a copy; written back to the buffer's mapping when dropped, where it is for writing.
#[derive(Debug)]
struct TgMappedRange {
    data: Vec<u8>,
    back: Option<(Arc<Mutex<Option<Mapping>>>, usize)>,
}

impl Drop for TgMappedRange {
    fn drop(&mut self) {
        if let Some((state, at)) = &self.back {
            if let Some(m) = state.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
                m.data[*at..*at + self.data.len()].copy_from_slice(&self.data);
            }
        }
    }
}

impl BufferMappedRangeInterface for TgMappedRange {
    fn len(&self) -> usize {
        self.data.len()
    }
    unsafe fn read_slice(&self) -> &[u8] {
        &self.data
    }
    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        wgpu::WriteOnly::from_mut(&mut self.data[..])
    }
}

// ---- commands ----

/// A command buffer's operations, as the server's SUBMIT takes them, and the buffers they use (kept until submitted).
#[derive(Debug, Default)]
struct Recorded {
    bytes: Vec<u8>,
    keep: Vec<Arc<Allocation>>,
}
type Ops = Arc<Mutex<Recorded>>;

#[derive(Debug)]
struct TgEncoder {
    ops: Ops,
}

#[derive(Debug)]
struct TgCommandBuffer {
    ops: Recorded,
}
impl CommandBufferInterface for TgCommandBuffer {}

fn buffer_of(b: &DispatchBuffer) -> &TgBuffer {
    b.as_custom::<TgBuffer>().expect("tinygpu: a buffer of this adapter's")
}

impl CommandEncoderInterface for TgEncoder {
    fn copy_buffer_to_buffer(&self, source: &DispatchBuffer, source_offset: wgpu::BufferAddress, destination: &DispatchBuffer, destination_offset: wgpu::BufferAddress, copy_size: Option<wgpu::BufferAddress>) {
        let (s, d) = (buffer_of(source), buffer_of(destination));
        let size = copy_size.unwrap_or(s.size - source_offset);
        let mut ops = self.ops.lock().unwrap_or_else(|p| p.into_inner());
        ops.bytes.push(2);
        for v in [s.mem.id, source_offset, d.mem.id, destination_offset, size] {
            ops.bytes.extend(v.to_le_bytes());
        }
        ops.keep.extend([Arc::clone(&s.mem), Arc::clone(&d.mem)]);
    }
    fn copy_buffer_to_texture(&self, _s: wgpu::TexelCopyBufferInfo<'_>, _d: wgpu::TexelCopyTextureInfo<'_>, _size: wgpu::Extent3d) {
        unsupported("textures")
    }
    fn copy_texture_to_buffer(&self, _s: wgpu::TexelCopyTextureInfo<'_>, _d: wgpu::TexelCopyBufferInfo<'_>, _size: wgpu::Extent3d) {
        unsupported("textures")
    }
    fn copy_texture_to_texture(&self, _s: wgpu::TexelCopyTextureInfo<'_>, _d: wgpu::TexelCopyTextureInfo<'_>, _size: wgpu::Extent3d) {
        unsupported("textures")
    }
    fn begin_compute_pass(&self, _desc: &wgpu::ComputePassDescriptor<'_>) -> DispatchComputePass {
        DispatchComputePass::custom(TgComputePass { ops: Arc::clone(&self.ops), pipeline: None, groups: Default::default() })
    }
    fn begin_render_pass(&self, _desc: &wgpu::RenderPassDescriptor<'_>) -> DispatchRenderPass {
        unsupported("render passes")
    }
    fn finish(&mut self) -> DispatchCommandBuffer {
        let ops = std::mem::take(&mut *self.ops.lock().unwrap_or_else(|p| p.into_inner()));
        DispatchCommandBuffer::custom(TgCommandBuffer { ops })
    }
    fn clear_texture(&self, _texture: &DispatchTexture, _range: &wgpu::ImageSubresourceRange) {
        unsupported("textures")
    }
    fn clear_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, size: Option<wgpu::BufferAddress>) {
        let b = buffer_of(buffer);
        let size = size.unwrap_or(b.size - offset);
        let mut ops = self.ops.lock().unwrap_or_else(|p| p.into_inner());
        ops.bytes.push(3);
        for v in [b.mem.id, offset, size] {
            ops.bytes.extend(v.to_le_bytes());
        }
        ops.keep.push(Arc::clone(&b.mem));
    }
    fn insert_debug_marker(&self, _label: &str) {}
    fn push_debug_group(&self, _label: &str) {}
    fn pop_debug_group(&self) {}
    fn write_timestamp(&self, _query_set: &DispatchQuerySet, _query_index: u32) {
        unsupported("timestamps")
    }
    fn resolve_query_set(&self, _q: &DispatchQuerySet, _first: u32, _count: u32, _d: &DispatchBuffer, _o: wgpu::BufferAddress) {
        unsupported("query sets")
    }
    fn mark_acceleration_structures_built<'a>(&self, _blas: &mut dyn Iterator<Item = &'a wgpu::Blas>, _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>) {
        unsupported("acceleration structures")
    }
    fn build_acceleration_structures<'a>(&self, _blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>, _tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>) {
        unsupported("acceleration structures")
    }
    fn transition_resources<'a>(&mut self, _b: &mut dyn Iterator<Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>>, _t: &mut dyn Iterator<Item = wgpu::wgt::TextureTransition<&'a DispatchTexture>>) {}
}

/// A bind group as set: its bindings' buffers and offsets, the dynamic offsets applied.
#[derive(Debug, Clone, Default)]
struct BoundGroup {
    entries: HashMap<u32, (Arc<Allocation>, u64, Option<u64>)>,
}

#[derive(Debug)]
struct TgComputePass {
    ops: Ops,
    pipeline: Option<(u64, Vec<(u32, u32)>)>,
    groups: [Option<BoundGroup>; 4],
}

impl Drop for TgComputePass {
    fn drop(&mut self) {}
}

impl ComputePassInterface for TgComputePass {
    fn set_pipeline(&mut self, pipeline: &DispatchComputePipeline) {
        let p = pipeline.as_custom::<TgComputePipeline>().expect("tinygpu: a pipeline of this adapter's");
        self.pipeline = Some((p.program, p.bindings.clone()));
    }
    fn set_bind_group(&mut self, index: u32, bind_group: Option<&DispatchBindGroup>, offsets: &[wgpu::DynamicOffset]) {
        let Some(g) = bind_group.and_then(|g| g.as_custom::<TgBindGroup>()) else {
            self.groups[index as usize] = None;
            return;
        };
        let mut entries = g.entries.clone();
        for (binding, extra) in g.dynamic.iter().zip(offsets) {
            if let Some(e) = entries.get_mut(binding) {
                e.1 += *extra as u64;
            }
        }
        self.groups[index as usize] = Some(BoundGroup { entries });
    }
    fn set_immediates(&mut self, _offset: u32, _data: &[u8]) {
        unsupported("immediates")
    }
    fn insert_debug_marker(&mut self, _label: &str) {}
    fn push_debug_group(&mut self, _label: &str) {}
    fn pop_debug_group(&mut self) {}
    fn write_timestamp(&mut self, _q: &DispatchQuerySet, _i: u32) {
        unsupported("timestamps")
    }
    fn begin_pipeline_statistics_query(&mut self, _q: &DispatchQuerySet, _i: u32) {
        unsupported("pipeline statistics")
    }
    fn end_pipeline_statistics_query(&mut self) {}
    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        let (program, bindings) = self.pipeline.as_ref().expect("tinygpu: a dispatch with no pipeline set");
        let mut ops = self.ops.lock().unwrap_or_else(|p| p.into_inner());
        ops.bytes.push(1);
        ops.bytes.extend(program.to_le_bytes());
        for v in [x, y, z, bindings.len() as u32] {
            ops.bytes.extend(v.to_le_bytes());
        }
        for (group, binding) in bindings {
            let (mem, offset, size) = self.groups[*group as usize]
                .as_ref()
                .and_then(|g| g.entries.get(binding))
                .unwrap_or_else(|| panic!("tinygpu: a dispatch with nothing bound at group {group}, binding {binding}"));
            ops.bytes.extend(mem.id.to_le_bytes());
            ops.bytes.extend(offset.to_le_bytes());
            ops.bytes.extend(size.unwrap_or(mem.size.saturating_sub(*offset)).to_le_bytes());
            ops.keep.push(Arc::clone(mem));
        }
    }
    fn dispatch_workgroups_indirect(&mut self, _buffer: &DispatchBuffer, _offset: wgpu::BufferAddress) {
        unsupported("indirect dispatches")
    }
    fn transition_resources<'a>(&mut self, _b: &mut dyn Iterator<Item = wgpu::wgt::BufferTransition<&'a DispatchBuffer>>, _t: &mut dyn Iterator<Item = wgpu::wgt::TextureTransition<&'a DispatchTextureView>>) {}
}

// ---- the queue ----

#[derive(Debug)]
struct TgQueue {
    conn: Arc<Conn>,
    submitted: AtomicU64,
}

#[derive(Debug)]
struct TgStaging {
    data: Vec<u8>,
}
impl QueueWriteBufferInterface for TgStaging {
    fn len(&self) -> usize {
        self.data.len()
    }
    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        wgpu::WriteOnly::from_mut(&mut self.data[..])
    }
}

impl QueueInterface for TgQueue {
    fn write_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, data: &[u8]) {
        write_range(&self.conn, buffer_of(buffer).mem.id, offset, data);
    }
    fn create_staging_buffer(&self, size: wgpu::BufferSize) -> Option<DispatchQueueWriteBuffer> {
        Some(DispatchQueueWriteBuffer::custom(TgStaging { data: vec![0; size.get() as usize] }))
    }
    fn validate_write_buffer(&self, _buffer: &DispatchBuffer, _offset: wgpu::BufferAddress, _size: wgpu::BufferSize) -> Option<()> {
        Some(())
    }
    fn write_staging_buffer(&self, buffer: &DispatchBuffer, offset: wgpu::BufferAddress, staging: &DispatchQueueWriteBuffer) {
        let s = staging.as_custom::<TgStaging>().expect("tinygpu: a staging buffer of this adapter's");
        write_range(&self.conn, buffer_of(buffer).mem.id, offset, &s.data);
    }
    fn write_texture(&self, _t: wgpu::TexelCopyTextureInfo<'_>, _d: &[u8], _l: wgpu::TexelCopyBufferLayout, _s: wgpu::Extent3d) {
        unsupported("textures")
    }
    fn submit(&self, command_buffers: &mut dyn Iterator<Item = DispatchCommandBuffer>) -> u64 {
        // (the command buffers, and the buffers they keep, are let go once the server has the work)
        let mut all = Vec::new();
        let mut held = Vec::new();
        for cb in command_buffers {
            if let Some(c) = cb.as_custom::<TgCommandBuffer>() {
                all.extend_from_slice(&c.ops.bytes);
            }
            held.push(cb);
        }
        if !all.is_empty() {
            self.conn.must(SUBMIT, &all);
        }
        self.submitted.fetch_add(1, Ordering::Relaxed) + 1
    }
    fn get_timestamp_period(&self) -> f32 {
        1.0
    }
    fn on_submitted_work_done(&self, callback: BoxSubmittedWorkDoneCallback) {
        self.conn.must(SYNC, &[]);
        callback();
    }
    fn compact_blas(&self, _blas: &DispatchBlas) -> (Option<u64>, DispatchBlas) {
        unsupported("acceleration structures")
    }
    fn present(&self, _detail: &DispatchSurfaceOutputDetail) {
        unsupported("surfaces")
    }
}
