//! `webgpu.h`, WebGPU's C API, for the NVIDIA card a Mac reaches only through tinygrad: a library any program written
//! against the standard header loads (wgpu-native's ABI, as `include/webgpu.h` and `include/wgpu.h` have it), its
//! compute run by the TinyGPU adapter (crates/wgpu-tinygpu) on the card `tools/tinygpu/webgpu_server.py` holds, or on
//! that server's emulator. Python's `wgpu` takes it as `WGPU_LIB_PATH=.../libwebgpu_tinygpu.dylib`.
//!
//! `src/webgpu.c` is the API: it reads the header's structs (so the C compiler lays them out, from the header itself),
//! keeps each object's references and the callbacks' futures, and asks this crate, in plain arguments (`include/tgw.h`),
//! for what wgpu does. Compute alone, as the adapter does: buffers, WGSL modules, compute pipelines, bind groups,
//! passes, copies, clears, the queue's writes and submissions, buffer mapping and error scopes.
//!
//! The adapter is the server's at `TINYGPU_SOCKET` (else ~/.cache/tinygpu-webgpu/server.sock);
//! `WEBGPU_TINYGPU_BACKEND=native` takes wgpu's own instead (Metal on a Mac), to check a program against both.
#![allow(clippy::missing_safety_doc)]

use std::collections::VecDeque;
use std::ffi::{c_char, c_void, CString};
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

pub const TGW_VALIDATION: u32 = 1;
pub const TGW_OUT_OF_MEMORY: u32 = 2;
pub const TGW_INTERNAL: u32 = 3;
pub const TGW_SHADER_F16: u64 = 1;
pub const TGW_TIMESTAMP_QUERY: u64 = 2;
pub const TGW_UNIFORM: u32 = 1;
pub const TGW_STORAGE: u32 = 2;
pub const TGW_READ_ONLY_STORAGE: u32 = 3;

#[repr(C)]
pub struct TgwLimits {
    max_bind_groups: u64,
    max_bindings_per_bind_group: u64,
    max_dynamic_uniform_buffers_per_pipeline_layout: u64,
    max_dynamic_storage_buffers_per_pipeline_layout: u64,
    max_storage_buffers_per_shader_stage: u64,
    max_uniform_buffers_per_shader_stage: u64,
    max_uniform_buffer_binding_size: u64,
    max_storage_buffer_binding_size: u64,
    min_uniform_buffer_offset_alignment: u64,
    min_storage_buffer_offset_alignment: u64,
    max_buffer_size: u64,
    max_compute_workgroup_storage_size: u64,
    max_compute_invocations_per_workgroup: u64,
    max_compute_workgroup_size_x: u64,
    max_compute_workgroup_size_y: u64,
    max_compute_workgroup_size_z: u64,
    max_compute_workgroups_per_dimension: u64,
    max_texture_dimension_1d: u64,
    max_texture_dimension_2d: u64,
    max_texture_dimension_3d: u64,
    max_texture_array_layers: u64,
    max_sampled_textures_per_shader_stage: u64,
    max_samplers_per_shader_stage: u64,
    max_storage_textures_per_shader_stage: u64,
    max_vertex_buffers: u64,
    max_vertex_attributes: u64,
    max_vertex_buffer_array_stride: u64,
    max_inter_stage_shader_variables: u64,
    max_color_attachments: u64,
    max_color_attachment_bytes_per_sample: u64,
    max_immediate_size: u64,
}

#[repr(C)]
pub struct TgwInfo {
    name: [c_char; 256],
    driver: [c_char; 256],
    vendor: [c_char; 64],
    vendor_id: u32,
    device_id: u32,
    backend: u32,
    discrete: u32,
    subgroup_min: u32,
    subgroup_max: u32,
}

#[repr(C)]
pub struct TgwLayoutEntry {
    binding: u32,
    kind: u32,
    has_dynamic_offset: u32,
    min_binding_size: u64,
}

#[repr(C)]
pub struct TgwGroupEntry {
    binding: u32,
    buffer: *mut TgwBuffer,
    offset: u64,
    size: u64,
}

pub struct TgwAdapter {
    adapter: wgpu::Adapter,
}

/// A device's error scopes (each its filter and first error) and the errors none took, oldest first.
#[derive(Default)]
struct Errors {
    scopes: Vec<(u32, Option<(u32, String)>)>,
    loose: VecDeque<(u32, String)>,
}

impl Errors {
    /// To the innermost scope of its kind, which keeps its first; else to the device's uncaptured errors.
    fn report(&mut self, kind: u32, message: String) {
        for (filter, first) in self.scopes.iter_mut().rev() {
            if *filter == kind {
                first.get_or_insert((kind, message));
                return;
            }
        }
        self.loose.push_back((kind, message));
    }
}

pub struct TgwDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    errors: Arc<Mutex<Errors>>,
}

impl TgwDevice {
    fn report(&self, kind: u32, message: String) {
        self.errors.lock().unwrap_or_else(|p| p.into_inner()).report(kind, message);
    }

    /// `f`, its panic (the TinyGPU adapter's way to fail, as wgpu's own without a handler) made the device's error.
    fn guard<T>(&self, f: impl FnOnce() -> T) -> Option<T> {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(v) => Some(v),
            Err(p) => {
                let message = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "a panic".into());
                self.report(TGW_VALIDATION, message);
                None
            }
        }
    }
}

enum View {
    Read(#[allow(dead_code)] wgpu::BufferView),
    Write(#[allow(dead_code)] wgpu::BufferViewMut),
}

pub struct TgwBuffer {
    buffer: wgpu::Buffer,
    views: Vec<View>,
}
pub struct TgwShader(wgpu::ShaderModule);
pub struct TgwBgl(wgpu::BindGroupLayout);
pub struct TgwPipelineLayout(wgpu::PipelineLayout);
pub struct TgwPipeline(wgpu::ComputePipeline);
pub struct TgwBindGroup(wgpu::BindGroup);
pub struct TgwEncoder(Option<wgpu::CommandEncoder>);
pub struct TgwPass(Option<wgpu::ComputePass<'static>>);
pub struct TgwCommandBuffer(Option<wgpu::CommandBuffer>);

fn boxed<T>(v: Option<T>) -> *mut T {
    v.map(|v| Box::into_raw(Box::new(v))).unwrap_or(std::ptr::null_mut())
}

unsafe fn free<T>(p: *mut T) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

unsafe fn text<'a>(p: *const c_char, len: usize) -> &'a str {
    if p.is_null() {
        return "";
    }
    std::str::from_utf8(std::slice::from_raw_parts(p as *const u8, len)).unwrap_or("")
}

fn c_string(s: String) -> *mut c_char {
    CString::new(s.replace('\0', " ")).map(CString::into_raw).unwrap_or(std::ptr::null_mut())
}

fn copy_into(dst: &mut [c_char], s: &str) {
    let n = s.len().min(dst.len() - 1);
    for (d, b) in dst.iter_mut().zip(s.as_bytes()[..n].iter()) {
        *d = *b as c_char;
    }
    dst[n] = 0;
}

#[no_mangle]
pub unsafe extern "C" fn tgw_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

// ---- adapter ----

#[no_mangle]
pub unsafe extern "C" fn tgw_adapter_open(error: *mut *mut c_char) -> *mut TgwAdapter {
    let opened = catch_unwind(|| {
        if std::env::var("WEBGPU_TINYGPU_BACKEND").as_deref() == Ok("native") {
            let options = wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() };
            pollster::block_on(wgpu::Instance::default().request_adapter(&options)).map_err(|e| e.to_string())
        } else {
            wgpu_tinygpu::adapter(&wgpu_tinygpu::default_socket())
        }
    })
    .unwrap_or_else(|_| Err("tinygpu: the adapter panicked".into()));
    match opened {
        Ok(adapter) => boxed(Some(TgwAdapter { adapter })),
        Err(e) => {
            if !error.is_null() {
                *error = c_string(e);
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_adapter_info(a: *mut TgwAdapter, out: *mut TgwInfo) {
    let info = (*a).adapter.get_info();
    let out = &mut *out;
    copy_into(&mut out.name, &info.name);
    copy_into(&mut out.driver, &format!("{} {}", info.driver, info.driver_info).trim().to_string());
    copy_into(&mut out.vendor, if info.vendor == 0x10de { "NVIDIA" } else if info.vendor == 0x106b { "Apple" } else { "" });
    out.vendor_id = info.vendor;
    out.device_id = info.device;
    out.backend = match info.backend {
        wgpu::Backend::Noop => 0,
        wgpu::Backend::Metal => 1,
        wgpu::Backend::Vulkan => 2,
        _ => 3,
    };
    out.discrete = (info.device_type == wgpu::DeviceType::DiscreteGpu) as u32;
    out.subgroup_min = info.subgroup_min_size;
    out.subgroup_max = info.subgroup_max_size;
}

fn limits_into(l: &wgpu::Limits, out: &mut TgwLimits) {
    *out = TgwLimits {
        max_bind_groups: l.max_bind_groups as u64,
        max_bindings_per_bind_group: l.max_bindings_per_bind_group as u64,
        max_dynamic_uniform_buffers_per_pipeline_layout: l.max_dynamic_uniform_buffers_per_pipeline_layout as u64,
        max_dynamic_storage_buffers_per_pipeline_layout: l.max_dynamic_storage_buffers_per_pipeline_layout as u64,
        max_storage_buffers_per_shader_stage: l.max_storage_buffers_per_shader_stage as u64,
        max_uniform_buffers_per_shader_stage: l.max_uniform_buffers_per_shader_stage as u64,
        max_uniform_buffer_binding_size: l.max_uniform_buffer_binding_size,
        max_storage_buffer_binding_size: l.max_storage_buffer_binding_size,
        min_uniform_buffer_offset_alignment: l.min_uniform_buffer_offset_alignment as u64,
        min_storage_buffer_offset_alignment: l.min_storage_buffer_offset_alignment as u64,
        max_buffer_size: l.max_buffer_size,
        max_compute_workgroup_storage_size: l.max_compute_workgroup_storage_size as u64,
        max_compute_invocations_per_workgroup: l.max_compute_invocations_per_workgroup as u64,
        max_compute_workgroup_size_x: l.max_compute_workgroup_size_x as u64,
        max_compute_workgroup_size_y: l.max_compute_workgroup_size_y as u64,
        max_compute_workgroup_size_z: l.max_compute_workgroup_size_z as u64,
        max_compute_workgroups_per_dimension: l.max_compute_workgroups_per_dimension as u64,
        max_texture_dimension_1d: l.max_texture_dimension_1d as u64,
        max_texture_dimension_2d: l.max_texture_dimension_2d as u64,
        max_texture_dimension_3d: l.max_texture_dimension_3d as u64,
        max_texture_array_layers: l.max_texture_array_layers as u64,
        max_sampled_textures_per_shader_stage: l.max_sampled_textures_per_shader_stage as u64,
        max_samplers_per_shader_stage: l.max_samplers_per_shader_stage as u64,
        max_storage_textures_per_shader_stage: l.max_storage_textures_per_shader_stage as u64,
        max_vertex_buffers: l.max_vertex_buffers as u64,
        max_vertex_attributes: l.max_vertex_attributes as u64,
        max_vertex_buffer_array_stride: l.max_vertex_buffer_array_stride as u64,
        max_inter_stage_shader_variables: l.max_inter_stage_shader_variables as u64,
        max_color_attachments: l.max_color_attachments as u64,
        max_color_attachment_bytes_per_sample: l.max_color_attachment_bytes_per_sample as u64,
        max_immediate_size: l.max_immediate_size as u64,
    };
}

#[no_mangle]
pub unsafe extern "C" fn tgw_adapter_limits(a: *mut TgwAdapter, out: *mut TgwLimits) {
    limits_into(&(*a).adapter.limits(), &mut *out);
}

fn feature_bits(f: wgpu::Features) -> u64 {
    let mut bits = 0;
    if f.contains(wgpu::Features::SHADER_F16) {
        bits |= TGW_SHADER_F16;
    }
    if f.contains(wgpu::Features::TIMESTAMP_QUERY) {
        bits |= TGW_TIMESTAMP_QUERY;
    }
    bits
}

#[no_mangle]
pub unsafe extern "C" fn tgw_adapter_features(a: *mut TgwAdapter) -> u64 {
    feature_bits((*a).adapter.features())
}

#[no_mangle]
pub unsafe extern "C" fn tgw_adapter_free(a: *mut TgwAdapter) {
    free(a)
}

// ---- device ----

#[no_mangle]
pub unsafe extern "C" fn tgw_device_request(a: *mut TgwAdapter, features: u64, error: *mut *mut c_char) -> *mut TgwDevice {
    let adapter = &(*a).adapter;
    let mut required = wgpu::Features::empty();
    if features & TGW_SHADER_F16 != 0 {
        required |= wgpu::Features::SHADER_F16;
    }
    if features & TGW_TIMESTAMP_QUERY != 0 {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    // (the adapter's limits, every one: a program asks for less, and gets what the card has)
    let desc = wgpu::DeviceDescriptor { label: None, required_features: required & adapter.features(), required_limits: adapter.limits(), ..Default::default() };
    let made = catch_unwind(AssertUnwindSafe(|| pollster::block_on(adapter.request_device(&desc)).map_err(|e| e.to_string())))
        .unwrap_or_else(|_| Err("tinygpu: the device panicked".into()));
    match made {
        Ok((device, queue)) => {
            let errors = Arc::new(Mutex::new(Errors::default()));
            let sink = Arc::clone(&errors);
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                let kind = match e {
                    wgpu::Error::OutOfMemory { .. } => TGW_OUT_OF_MEMORY,
                    wgpu::Error::Validation { .. } => TGW_VALIDATION,
                    wgpu::Error::Internal { .. } => TGW_INTERNAL,
                };
                sink.lock().unwrap_or_else(|p| p.into_inner()).report(kind, e.to_string());
            }));
            boxed(Some(TgwDevice { device, queue, errors }))
        }
        Err(e) => {
            if !error.is_null() {
                *error = c_string(e);
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_limits(d: *mut TgwDevice, out: *mut TgwLimits) {
    limits_into(&(*d).device.limits(), &mut *out);
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_features(d: *mut TgwDevice) -> u64 {
    feature_bits((*d).device.features())
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_push_scope(d: *mut TgwDevice, filter: u32) {
    (*d).errors.lock().unwrap_or_else(|p| p.into_inner()).scopes.push((filter, None));
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_pop_scope(d: *mut TgwDevice, kind: *mut u32, message: *mut *mut c_char) -> i32 {
    let popped = (*d).errors.lock().unwrap_or_else(|p| p.into_inner()).scopes.pop();
    match popped {
        None => -1,
        Some((_, first)) => {
            let (k, m) = first.map(|(k, m)| (k, c_string(m))).unwrap_or((0, std::ptr::null_mut()));
            *kind = k;
            *message = m;
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_take_error(d: *mut TgwDevice, kind: *mut u32) -> *mut c_char {
    match (*d).errors.lock().unwrap_or_else(|p| p.into_inner()).loose.pop_front() {
        Some((k, m)) => {
            *kind = k;
            c_string(m)
        }
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_poll(d: *mut TgwDevice, wait: i32) -> i32 {
    let d = &*d;
    let poll = if wait != 0 { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll };
    d.guard(|| d.device.poll(poll).map(|s| s.is_queue_empty()).unwrap_or(false)).unwrap_or(false) as i32
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_destroy(d: *mut TgwDevice) {
    (*d).device.destroy();
}

#[no_mangle]
pub unsafe extern "C" fn tgw_device_free(d: *mut TgwDevice) {
    free(d)
}

// ---- buffers ----

#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_create(d: *mut TgwDevice, size: u64, usage: u32, mapped_at_creation: i32) -> *mut TgwBuffer {
    let d = &*d;
    let desc = wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::from_bits_truncate(usage), mapped_at_creation: mapped_at_creation != 0 };
    boxed(d.guard(|| TgwBuffer { buffer: d.device.create_buffer(&desc), views: Vec::new() }))
}

/// Maps the range and waits for it (the adapter's maps are done as they are asked); 0 when mapped.
#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_map(d: *mut TgwDevice, b: *mut TgwBuffer, write: i32, offset: u64, size: u64) -> i32 {
    let (d, b) = (&*d, &mut *b);
    let mode = if write != 0 { wgpu::MapMode::Write } else { wgpu::MapMode::Read };
    let result = Arc::new(Mutex::new(None));
    let done = Arc::clone(&result);
    let ok = d.guard(|| {
        b.buffer.slice(offset..offset + size).map_async(mode, move |r| *done.lock().unwrap_or_else(|p| p.into_inner()) = Some(r));
        let _ = d.device.poll(wgpu::PollType::wait_indefinitely());
    });
    let outcome = result.lock().unwrap_or_else(|p| p.into_inner()).take();
    match (ok, outcome) {
        (Some(()), Some(Ok(()))) => 0,
        (_, Some(Err(e))) => {
            d.report(TGW_VALIDATION, e.to_string());
            1
        }
        _ => 1,
    }
}

/// The mapped range's bytes, valid until the buffer is unmapped.
#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_range(d: *mut TgwDevice, b: *mut TgwBuffer, offset: u64, size: u64, write: i32) -> *mut c_void {
    let (d, b) = (&*d, &mut *b);
    let slice = b.buffer.slice(offset..offset + size);
    let got = if write != 0 {
        slice.get_mapped_range_mut().map(|mut v| {
            let p = v.slice(..).as_raw_ptr().as_ptr() as *mut u8 as *mut c_void;
            b.views.push(View::Write(v));
            p
        })
    } else {
        slice.get_mapped_range().map(|v| {
            let p = v.as_ptr() as *mut c_void;
            b.views.push(View::Read(v));
            p
        })
    };
    got.unwrap_or_else(|e| {
        d.report(TGW_VALIDATION, e.to_string());
        std::ptr::null_mut()
    })
}

#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_unmap(b: *mut TgwBuffer) {
    let b = &mut *b;
    b.views.clear();
    b.buffer.unmap();
}

#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_destroy(b: *mut TgwBuffer) {
    let b = &mut *b;
    b.views.clear();
    b.buffer.destroy();
}

#[no_mangle]
pub unsafe extern "C" fn tgw_buffer_free(b: *mut TgwBuffer) {
    free(b)
}

// ---- modules, layouts, pipelines, bind groups ----

#[no_mangle]
pub unsafe extern "C" fn tgw_shader_create(d: *mut TgwDevice, wgsl: *const c_char, len: usize) -> *mut TgwShader {
    let d = &*d;
    let code = text(wgsl, len);
    boxed(d.guard(|| TgwShader(d.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) }))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_shader_free(s: *mut TgwShader) {
    free(s)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_bgl_create(d: *mut TgwDevice, entries: *const TgwLayoutEntry, n: usize) -> *mut TgwBgl {
    let d = &*d;
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n)
        .map(|i| {
            let e = &*entries.add(i);
            let ty = match e.kind {
                TGW_UNIFORM => wgpu::BufferBindingType::Uniform,
                TGW_READ_ONLY_STORAGE => wgpu::BufferBindingType::Storage { read_only: true },
                _ => wgpu::BufferBindingType::Storage { read_only: false },
            };
            wgpu::BindGroupLayoutEntry {
                binding: e.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: e.has_dynamic_offset != 0, min_binding_size: NonZeroU64::new(e.min_binding_size) },
                count: None,
            }
        })
        .collect();
    boxed(d.guard(|| TgwBgl(d.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries }))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_bgl_free(l: *mut TgwBgl) {
    free(l)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pipeline_layout_create(d: *mut TgwDevice, groups: *const *mut TgwBgl, n: usize) -> *mut TgwPipelineLayout {
    let d = &*d;
    let layouts: Vec<Option<&wgpu::BindGroupLayout>> = (0..n).map(|i| (*groups.add(i)).as_ref().map(|l| &l.0)).collect();
    boxed(d.guard(|| TgwPipelineLayout(d.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &layouts, immediate_size: 0 }))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pipeline_layout_free(l: *mut TgwPipelineLayout) {
    free(l)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pipeline_create(
    d: *mut TgwDevice,
    layout: *mut TgwPipelineLayout,
    s: *mut TgwShader,
    entry: *const c_char,
    entry_len: usize,
    keys: *const *const c_char,
    key_lens: *const usize,
    values: *const f64,
    n: usize,
) -> *mut TgwPipeline {
    let d = &*d;
    let entry = (!entry.is_null()).then(|| text(entry, entry_len));
    let constants: Vec<(&str, f64)> = (0..n).map(|i| (text(*keys.add(i), *key_lens.add(i)), *values.add(i))).collect();
    let desc = wgpu::ComputePipelineDescriptor {
        label: None,
        layout: layout.as_ref().map(|l| &l.0),
        module: &(*s).0,
        entry_point: entry,
        compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, zero_initialize_workgroup_memory: true },
        cache: None,
    };
    boxed(d.guard(|| TgwPipeline(d.device.create_compute_pipeline(&desc))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pipeline_bgl(d: *mut TgwDevice, p: *mut TgwPipeline, index: u32) -> *mut TgwBgl {
    let d = &*d;
    boxed(d.guard(|| TgwBgl((*p).0.get_bind_group_layout(index))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pipeline_free(p: *mut TgwPipeline) {
    free(p)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_bind_group_create(d: *mut TgwDevice, layout: *mut TgwBgl, entries: *const TgwGroupEntry, n: usize) -> *mut TgwBindGroup {
    let d = &*d;
    let entries: Vec<wgpu::BindGroupEntry> = (0..n)
        .map(|i| {
            let e = &*entries.add(i);
            let size = if e.size == u64::MAX { None } else { NonZeroU64::new(e.size) };
            wgpu::BindGroupEntry { binding: e.binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &(*e.buffer).buffer, offset: e.offset, size }) }
        })
        .collect();
    boxed(d.guard(|| TgwBindGroup(d.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &(*layout).0, entries: &entries }))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_bind_group_free(g: *mut TgwBindGroup) {
    free(g)
}

// ---- commands ----

#[no_mangle]
pub unsafe extern "C" fn tgw_encoder_create(d: *mut TgwDevice) -> *mut TgwEncoder {
    let d = &*d;
    boxed(d.guard(|| TgwEncoder(Some(d.device.create_command_encoder(&Default::default())))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_encoder_copy(d: *mut TgwDevice, e: *mut TgwEncoder, src: *mut TgwBuffer, src_offset: u64, dst: *mut TgwBuffer, dst_offset: u64, size: u64) {
    let d = &*d;
    if let Some(enc) = (*e).0.as_mut() {
        d.guard(|| enc.copy_buffer_to_buffer(&(*src).buffer, src_offset, &(*dst).buffer, dst_offset, Some(size)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_encoder_clear(d: *mut TgwDevice, e: *mut TgwEncoder, b: *mut TgwBuffer, offset: u64, size: u64) {
    let d = &*d;
    if let Some(enc) = (*e).0.as_mut() {
        d.guard(|| enc.clear_buffer(&(*b).buffer, offset, (size != u64::MAX).then_some(size)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_encoder_finish(d: *mut TgwDevice, e: *mut TgwEncoder) -> *mut TgwCommandBuffer {
    let d = &*d;
    let enc = (*e).0.take();
    boxed(enc.and_then(|enc| d.guard(|| TgwCommandBuffer(Some(enc.finish())))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_encoder_free(e: *mut TgwEncoder) {
    free(e)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_begin(d: *mut TgwDevice, e: *mut TgwEncoder) -> *mut TgwPass {
    let d = &*d;
    let Some(enc) = (*e).0.as_mut() else { return std::ptr::null_mut() };
    boxed(d.guard(|| TgwPass(Some(enc.begin_compute_pass(&Default::default()).forget_lifetime()))))
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_set_pipeline(d: *mut TgwDevice, p: *mut TgwPass, pipeline: *mut TgwPipeline) {
    if let (Some(pass), Some(pipeline)) = ((*p).0.as_mut(), pipeline.as_ref()) {
        (*d).guard(|| pass.set_pipeline(&pipeline.0));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_set_bind_group(d: *mut TgwDevice, p: *mut TgwPass, index: u32, g: *mut TgwBindGroup, offsets: *const u32, n: usize) {
    let offsets = if n == 0 { &[][..] } else { std::slice::from_raw_parts(offsets, n) };
    if let Some(pass) = (*p).0.as_mut() {
        (*d).guard(|| pass.set_bind_group(index, g.as_ref().map(|g| &g.0), offsets));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_dispatch(d: *mut TgwDevice, p: *mut TgwPass, x: u32, y: u32, z: u32) {
    if let Some(pass) = (*p).0.as_mut() {
        (*d).guard(|| pass.dispatch_workgroups(x, y, z));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_dispatch_indirect(d: *mut TgwDevice, p: *mut TgwPass, b: *mut TgwBuffer, offset: u64) {
    if let Some(pass) = (*p).0.as_mut() {
        (*d).guard(|| pass.dispatch_workgroups_indirect(&(*b).buffer, offset));
    }
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_end(p: *mut TgwPass) {
    drop((*p).0.take());
}

#[no_mangle]
pub unsafe extern "C" fn tgw_pass_free(p: *mut TgwPass) {
    free(p)
}

#[no_mangle]
pub unsafe extern "C" fn tgw_command_buffer_free(c: *mut TgwCommandBuffer) {
    free(c)
}

// ---- the queue ----

#[no_mangle]
pub unsafe extern "C" fn tgw_queue_submit(d: *mut TgwDevice, buffers: *const *mut TgwCommandBuffer, n: usize) {
    let d = &*d;
    let taken: Vec<wgpu::CommandBuffer> = (0..n).filter_map(|i| (*buffers.add(i)).as_mut().and_then(|c| c.0.take())).collect();
    d.guard(|| d.queue.submit(taken));
}

#[no_mangle]
pub unsafe extern "C" fn tgw_queue_write(d: *mut TgwDevice, b: *mut TgwBuffer, offset: u64, data: *const c_void, size: usize) {
    let d = &*d;
    if size == 0 {
        return;
    }
    let bytes = std::slice::from_raw_parts(data as *const u8, size);
    d.guard(|| d.queue.write_buffer(&(*b).buffer, offset, bytes));
}
