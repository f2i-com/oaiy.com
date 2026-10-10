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
pub mod quant_linear;
pub mod quant_moe;
pub mod quant_host;
pub mod shaders;

mod backend;
mod feed;
mod gpu;
mod memory;
pub mod profile;
mod quant;
mod upload;
mod watch;

// (what the crate's files make for each other, and for its users what the root gave them)
pub(crate) use {feed::*, quant::*, upload::*};
pub use {memory::*, watch::*};

use ggml_quants::GgmlType;
use ggml_rs::{Backend, CpuBackend, QuantizedDeviceStorage, QuantizedTensor, RopeType, Tensor};
use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const GIB: u64 = 1 << 30;
/// The scratch a GPU's pool keeps between chains' runs: a prompt's layer's (Qwen3.8-Flash-Next's at 512 rows about
/// 0.6 GiB), within the 4 GiB a card's budget leaves.
const POOL_BYTES: u64 = 3 * GIB / 2;
/// The scratch made new for chains' recordings (none of its size in its device's pool), in bytes, since the process
/// began: memory the system hands over zeroed, at some 3 GB a second on a card of its own, so a prompt whose scratch
/// is new is the slower by that.
static SCRATCH_MADE: AtomicU64 = AtomicU64::new(0);

/// The bytes of scratch made new since the process began, all devices' (a request's line says its own).
pub fn scratch_made() -> u64 {
    SCRATCH_MADE.load(Ordering::Relaxed)
}

/// The read-backs' staging a GPU keeps between chains' runs: two of a prompt's chunks' (some 70 MB each).
const STAGING_BYTES: u64 = GIB / 4;

/// Weight bytes one storage binding may cover: rows are split across buffers
/// below the adapter's binding limit (a 152k-vocab Q6_K output is ~640 MB).
fn chunk_limit(limits: &wgpu::Limits) -> u64 {
    (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size).min(1 << 30) & !3
}

/// The bytes of a WGSL module's workgroup variables, as its layout has them (None: it does not parse here).
fn workgroup_bytes(source: &str) -> Option<u64> {
    use wgpu::naga;
    let module = naga::front::wgsl::parse_str(source).ok()?;
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(module.to_ctx()).ok()?;
    Some(module.global_variables.iter().filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup).map(|(_, g)| layouter[g.ty].size as u64).sum())
}

/// The workgroup variables a kernel writes one component of a vector of (`v[i][c] = ..`, `v[i].x = ..`), by name.
///
/// WGSL lets a write to one component of a vector in memory read and write the whole vector, so threads that each
/// write their own component of one vector race (the WGSL spec, "Component Reference from Vector Memory View").
/// Apple's compiler takes that leave: on a Mac every tile staged into the workgroup's memory a component a thread came
/// out three quarters zeros, and Qwen3.5 answered "amon!!!!". Vulkan's and D3D12's compilers store the one component,
/// so it is seen only on a Mac; the crate's tests ask this of every kernel they make, so it is seen anywhere. A kernel
/// stages such values in an array of scalars and reads them four at a time.
#[cfg(test)]
fn lane_writes(source: &str) -> Vec<String> {
    use wgpu::naga::{self, Expression, Statement, TypeInner};
    fn walk(module: &naga::Module, f: &naga::Function, info: &naga::valid::FunctionInfo, block: &naga::Block, out: &mut Vec<String>) {
        for statement in block.iter() {
            match statement {
                Statement::Store { pointer, .. } => {
                    // a pointer to a component, taken from a pointer to a vector ..
                    let (Expression::Access { base, .. } | Expression::AccessIndex { base, .. }) = f.expressions[*pointer] else { continue };
                    let lane = match info[base].ty.inner_with(&module.types) {
                        TypeInner::Pointer { base: ty, .. } => matches!(module.types[*ty].inner, TypeInner::Vector { .. }),
                        TypeInner::ValuePointer { size, .. } => size.is_some(),
                        _ => false,
                    };
                    // .. into a workgroup variable
                    let mut root = base;
                    while let Expression::Access { base, .. } | Expression::AccessIndex { base, .. } = f.expressions[root] {
                        root = base;
                    }
                    if let (true, Expression::GlobalVariable(g)) = (lane, &f.expressions[root]) {
                        let global = &module.global_variables[*g];
                        if global.space == naga::AddressSpace::WorkGroup {
                            out.push(global.name.clone().unwrap_or_default());
                        }
                    }
                }
                Statement::Block(b) => walk(module, f, info, b, out),
                Statement::If { accept, reject, .. } => {
                    walk(module, f, info, accept, out);
                    walk(module, f, info, reject, out);
                }
                Statement::Loop { body, continuing, .. } => {
                    walk(module, f, info, body, out);
                    walk(module, f, info, continuing, out);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| walk(module, f, info, &c.body, out)),
                _ => {}
            }
        }
    }
    let Ok(module) = naga::front::wgsl::parse_str(source) else { return Vec::new() };
    let Ok(info) = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module) else { return Vec::new() };
    let mut out = Vec::new();
    for (handle, f) in module.functions.iter() {
        walk(&module, f, &info[handle], &f.body, &mut out);
    }
    for (i, entry) in module.entry_points.iter().enumerate() {
        walk(&module, &entry.function, info.get_entry_point(i), &entry.function.body, &mut out);
    }
    out.sort();
    out.dedup();
    out
}

/// A kernel's text with each packed dot product (`dot4I8Packed`, `dot4U8Packed`) called through a function of its
/// own, for Metal.
///
/// naga writes one of those for Metal as a temporary for each operand, named after the operand, and the dot product
/// of the temporaries. A kernel that uses one operand in several dot products (a row's four weights against each of
/// four tokens: the multi-token 8-bit kernels) gets that temporary declared once a use, in one block, and Apple's
/// compiler refuses the kernel: "redefinition of 'reinterpreted_packed_char4_e1455'". The language-model server then
/// ended as it made its first pipeline, on every Mac. Inside a function of its own a dot product's operands are the
/// function's two parameters, used once, whatever the caller passed; Metal's compiler puts the function back in line.
///
/// Other APIs' kernels are left as they are written: their text is what the speed measurements were made with.
fn packed_dots_wrapped(source: String) -> String {
    let mut out = source;
    for (builtin, wrapper, result) in [("dot4I8Packed", "oaiy_dot4_i8", "i32"), ("dot4U8Packed", "oaiy_dot4_u8", "u32")] {
        let call = format!("{builtin}(");
        if !out.contains(&call) {
            continue;
        }
        out = out.replace(&call, &format!("{wrapper}("));
        out.push_str(&format!("\nfn {wrapper}(a: u32, b: u32) -> {result} {{\n    return {builtin}(a, b);\n}}\n"));
    }
    out
}

struct Gpu {
    device: wgpu::Device,
    /// The side of the cooperative matrices' fragments the kernels use (the tensor cores'): 16 (Vulkan's, D3D12's: every
    /// tensor-core kernel), 8 (Metal's simdgroup matrices: the K-quants' and Q8_0's prompt matmuls alone), 0 (none).
    coop_tile: u32,
    /// The device's queue as it is; [`Gpu::queue`] for anyone's use of it but the feed's.
    queue_raw: wgpu::Queue,
    /// Why the device was lost, once its callback has said (a wait then fails rather than waiting on).
    lost: Arc<Mutex<Option<String>>>,
    /// The driver's calls a watchdog thread watches (a hung one ends the process).
    watch: Arc<Watch>,
    /// The chains' pieces on their way to the queue where the device's pieces in flight are limited ([`Gpu::feed`]),
    /// and how many may be in flight at once (0: as many as are recorded).
    feed: Arc<Feed>,
    in_flight_limit: Arc<std::sync::atomic::AtomicUsize>,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<(GgmlType, u8), Arc<wgpu::ComputePipeline>>>,
    /// The EXL3 matmul's pipelines (`exl3::shader`): for one row, and for several. Made when first used.
    exl3: Mutex<[Option<Arc<wgpu::ComputePipeline>>; 2]>,
    /// Other kernels' pipelines by name (`dense`), made when first used.
    named: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
    /// The pipelines' names by address, for a chain's profile.
    names: Mutex<HashMap<usize, &'static str>>,
    /// Chains' scratch buffers between their runs (bytes, buffer): a prompt's layer takes the last one's, where a new
    /// buffer is allocated and cleared before its first use.
    pool: Mutex<Vec<(u64, wgpu::Buffer)>>,
    /// Read-backs' staging buffers (host memory the GPU copies into) a chain's reads use again: a prompt's chunk read
    /// 64 MB of its cache's rows into new ones, each a wait of the OS's before the GPU could start the next.
    staging: Mutex<Vec<(u64, wgpu::Buffer)>>,
    /// A chain's bind groups that are the same step after step (`chain`): by pipeline, buffers and parameters.
    chain_groups: Mutex<HashMap<chain::GroupKey, wgpu::BindGroup>>,
    /// The layout of the chain's kernels of eight buffers (a gated delta net's: six read, two written, then the
    /// parameters), made when first used, and their bind groups as `chain_groups`.
    wide: std::sync::OnceLock<(wgpu::BindGroupLayout, wgpu::PipelineLayout)>,
    chain_groups_wide: Mutex<HashMap<chain::WideKey, wgpu::BindGroup>>,
    /// Small buffers for a kernel's bindings it does not use: one read, one written.
    dummy: std::sync::OnceLock<wgpu::Buffer>,
    dummy_rw: std::sync::OnceLock<wgpu::Buffer>,
    limits: wgpu::Limits,
    /// Upload bytes written since the queue was last flushed.
    staged: AtomicU64,
    /// The EXL3 projections' scratch for a few rows (a check of drafts), shared by them all, made when first used.
    few: std::sync::OnceLock<exl3::FewScratch>,
    /// Grouped experts' kept scratch of a step or a check (rows, top k, hidden, ff and the groups' splits): one for
    /// every layer of that shape.
    moe_steps: Mutex<Vec<([usize; 6], Arc<exl3::Step>)>>,
    /// How many units (SMs) the tensor cores' matmuls share out ([`Gpu::coop_units`]), counted when first asked.
    coop_units: std::sync::OnceLock<u32>,
    /// What a decode step's dense calls share ([`dense::Arena`]), made at the first.
    dense_arena: Mutex<Option<dense::Arena>>,
}

/// Which adapter the backend runs on, for logs.
#[derive(Clone, Debug)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    /// Where the adapter sits on the PCI bus, as the API says (two cards of one model differ only here).
    pub pci_bus_id: String,
    /// Its memory is the computer's own (an Apple-silicon Mac's GPU): weights on it and weights in RAM share one pool.
    pub unified: bool,
}

pub struct WgpuBackend {
    cpu: CpuBackend,
    gpu: Arc<Gpu>,
    budget: u64,
    used: Arc<AtomicU64>,
    /// Bytes of the weights it was asked to hold and had no room for within the budget: they stay on the host.
    left: Arc<AtomicU64>,
    summary: AdapterSummary,
    /// One projection at a time: the dispatch and read-back share the queue (EXL3 weights hold it too).
    serial: Arc<Mutex<()>>,
    /// The adapter it was opened on (its memory budget asked of it).
    raw_adapter: wgpu::Adapter,
}

/// Another handle on the same adapter (its device, its budget's count and its lock shared): what a model's parts keep
/// to record chains of their own (an expert group).
impl Clone for WgpuBackend {
    fn clone(&self) -> Self {
        Self { cpu: CpuBackend::new(), gpu: Arc::clone(&self.gpu), budget: self.budget, used: Arc::clone(&self.used), left: Arc::clone(&self.left), summary: self.summary.clone(), serial: Arc::clone(&self.serial), raw_adapter: self.raw_adapter.clone() }
    }
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBackend({} via {}, {} of {} GiB used)", self.summary.name, self.summary.backend,
            self.used.load(Ordering::Relaxed) / GIB, self.budget / GIB)
    }
}

impl WgpuBackend {
    /// A gated delta net's step on the GPU (the chain's conv and recurrence kernels, one submit), its recurrent state
    /// and conv window kept there between calls as tensors that are the GPU's own vectors (`DeviceChain::alias`): a
    /// state on the host (a new one, a restore) is taken up into one. None for a shape the kernels do not take; the
    /// caller runs the host's. Where a model calls the backend's step itself (Qwen3.8-Flash-Next, a Qwen3.5 prompt
    /// with an image), this takes the recurrence off the host: 7.8 ms a layer there for Flash-Next.
    #[allow(clippy::too_many_arguments)]
    fn delta_net_gpu(
        &self, mixed_qkv: &Tensor, z_in: &Tensor, beta_alpha: &Tensor, conv_weight: &Tensor, ssm_a: &Tensor, dt_bias: &Tensor, ssm_norm: &Tensor,
        conv_state: &mut Tensor, state: &mut Tensor, d: ggml_rs::DeltaNet,
    ) -> Option<Tensor> {
        use ggml_rs::DeviceChain;
        let seq = d.rows;
        let kern = conv_weight.dim(conv_weight.rank() - 1);
        let ch = 2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim;
        let supported = d.k_dim == d.v_dim && [16, 32, 64, 128].contains(&d.k_dim) && d.k_heads > 0 && (2..=8).contains(&kern) && seq > 0;
        if !supported || mixed_qkv.numel() != seq * ch || z_in.numel() != seq * d.v_heads * d.v_dim || beta_alpha.numel() != seq * 2 * d.v_heads
            || conv_weight.numel() != ch * kern
        {
            return None;
        }
        // the state and the conv window as this adapter's vectors: the ones they alias, or the host's taken up
        let adopt = |t: &mut Tensor, shape: Vec<usize>| -> ggml_rs::DeviceVec {
            let len: usize = shape.iter().product();
            if let Some(v) = self.aliased(t).filter(|v| v.len == len) {
                return v;
            }
            let host = t.to_host();
            let v = self.vec(len);
            if host.numel() == len {
                DeviceChain::upload(self, &v, host.data());
            }
            *t = self.alias(&v, shape);
            v
        };
        let st = adopt(state, vec![d.v_heads, d.v_dim, d.k_dim]);
        let cv = adopt(conv_state, vec![kern - 1, ch]);
        let up = |t: &Tensor| {
            let h = if t.is_device() { t.to_host() } else { t.clone() };
            let v = self.vec(h.numel());
            DeviceChain::upload(self, &v, h.data());
            v
        };
        let (qkv, z, ba, cw, a, dt, nm) = (up(mixed_qkv), up(z_in), up(beta_alpha), up(conv_weight), up(ssm_a), up(dt_bias), up(ssm_norm));
        let (conv_out, out) = (self.vec(seq * ch), self.vec(seq * d.v_heads * d.v_dim));
        let mut rec = self.begin();
        // these vectors are this call's: no bind groups kept to hold them
        rec.keep_groups(false);
        rec.ssm_conv(&qkv, &cw, &cv, &conv_out, seq, ch, kern);
        rec.delta_net(&conv_out, &z, &ba, &a, &dt, &nm, &st, &out, d);
        rec.read(&out);
        let got = rec.finish().pop().expect("the delta net's output");
        Some(Tensor::from_vec(got, vec![seq, d.v_heads * d.v_dim]))
    }

    /// Open the best adapter wgpu finds, or the one `OAIY_WEBGPU_ADAPTER` names
    /// (part of its name, any case: "radeon", "arc", "5090", or of its PCI bus id:
    /// "03:00" of two cards alike), for a computer with more than one GPU. `budget_bytes` caps the weights placed on it (WebGPU
    /// cannot report free memory); `None` picks a default: a discrete card's
    /// memory less 4 GiB where Vulkan says how much it has (27.8 GiB of a 32 GB
    /// card), else 8 GiB; an Apple-silicon Mac's GPU as a card with two thirds of
    /// the computer's memory (`unified_budget`); 2 GiB integrated, none for software.
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
                    .find(|a| {
                        let i = a.get_info();
                        i.name.to_lowercase().contains(&wanted) || i.device_pci_bus_id.to_lowercase().contains(&wanted)
                    })
                    .ok_or_else(|| format!("OAIY_WEBGPU_ADAPTER={wanted}: no WebGPU adapter has that in its name or PCI bus id; there are {}", names.join(", ")))?
            }
            None => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
            .map_err(|e| format!("no WebGPU adapter: {e}"))?,
        };
        Self::open(adapter, budget_bytes)
    }

    /// The computer's `index`-th GPU (0 the first) as a CUDA device index counts them: the discrete ones on the API
    /// `new` picks, in their order on the PCI bus (nvidia-smi's, and CUDA's of cards alike), else every adapter on it;
    /// `OAIY_WEBGPU_ADAPTER`, where set, naming one instead (as for `new`). A worker told which GPU to use (an image
    /// model kept off a chat model's).
    pub fn nth(index: usize, budget_bytes: Option<u64>) -> Result<Self, String> {
        if std::env::var("OAIY_WEBGPU_ADAPTER").is_ok_and(|s| !s.trim().is_empty()) {
            return Self::new(budget_bytes);
        }
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let best = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let api = best.get_info().backend;
        let all: Vec<wgpu::Adapter> = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY)).into_iter().filter(|a| a.get_info().backend == api).collect();
        let discrete = all.iter().any(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
        let mut list: Vec<wgpu::Adapter> = all.into_iter().filter(|a| !discrete || a.get_info().device_type == wgpu::DeviceType::DiscreteGpu).collect();
        list.sort_by_key(|a| a.get_info().device_pci_bus_id);
        let count = list.len();
        let adapter = list.into_iter().nth(index).ok_or_else(|| format!("GPU {index}: the computer has {count} on {api:?}"))?;
        Self::open(adapter, budget_bytes)
    }

    /// The computer's other discrete GPUs on this one's API, each opened as a backend of its own (a budget each, as
    /// `new` picks one): two cards of one model differ only in where they sit on the PCI bus. For a model whose
    /// weights do not fit one card (Qwen3.8-Flash-Next's 46 GB of experts).
    pub fn others(&self, budget_bytes: Option<u64>) -> Vec<WgpuBackend> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(desc.with_env());
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::PRIMARY));
        let mut out = Vec::new();
        for adapter in adapters {
            let info = adapter.get_info();
            let same_api = format!("{:?}", info.backend) == self.summary.backend;
            if !same_api || info.device_type != wgpu::DeviceType::DiscreteGpu || info.device_pci_bus_id == self.summary.pci_bus_id {
                continue;
            }
            if let Ok(b) = Self::open(adapter, budget_bytes) {
                out.push(b);
            }
        }
        out
    }

    /// A backend on `adapter`.
    fn open(adapter: wgpu::Adapter, budget_bytes: Option<u64>) -> Result<Self, String> {
        let info = adapter.get_info();
        let summary = AdapterSummary {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            pci_bus_id: info.device_pci_bus_id.clone(),
            unified: unified(&info),
        };
        let mut limits = adapter.limits();
        if let Some(v) = std::env::var_os("OAIY_PORTABLE_LIMITS") {
            // a workgroup's memory as given (bytes; else Apple's GPUs' 32,768, the least a native adapter has: WebGPU's
            // default, a browser's, is 16,384), and its invocations as WebGPU's defaults (256): other GPUs' limits (a
            // kernel past them refused, Gpu::shader), the rest the adapter's
            let d = wgpu::Limits::default();
            let bytes = v.to_str().and_then(|s| s.parse::<u32>().ok()).filter(|&b| b >= 1024).unwrap_or(32 << 10);
            limits.max_compute_workgroup_storage_size = limits.max_compute_workgroup_storage_size.min(bytes);
            limits.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup.min(d.max_compute_invocations_per_workgroup);
            limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.min(d.max_compute_workgroup_size_x);
            limits.max_compute_workgroup_size_y = limits.max_compute_workgroup_size_y.min(d.max_compute_workgroup_size_y);
            limits.max_compute_workgroup_size_z = limits.max_compute_workgroup_size_z.min(d.max_compute_workgroup_size_z);
            // and what else wgpu's Metal backend gives an Apple GPU less of than a card's driver does (wgpu-hal 30's
            // metal/adapter.rs): 29 buffers a stage (Metal's 31 less wgpu's own two), 65,535 workgroups a dimension,
            // a storage binding's offset a multiple of 32 and a uniform one's of 256 (macOS's). wgpu checks each of
            // them against what the device was asked for, on any backend: a kernel past one fails here as it would
            // there
            limits.max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage.min(29);
            limits.max_uniform_buffers_per_shader_stage = limits.max_uniform_buffers_per_shader_stage.min(29);
            limits.max_dynamic_storage_buffers_per_pipeline_layout = limits.max_dynamic_storage_buffers_per_pipeline_layout.min(29);
            limits.max_dynamic_uniform_buffers_per_pipeline_layout = limits.max_dynamic_uniform_buffers_per_pipeline_layout.min(29);
            limits.max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension.min(0xFFFF);
            limits.min_storage_buffer_offset_alignment = limits.min_storage_buffer_offset_alignment.max(32);
            limits.min_uniform_buffer_offset_alignment = limits.min_uniform_buffer_offset_alignment.max(256);
        }
        let timestamps = if profile::chain_on() || profile::pieces_on() { adapter.features() & wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        // the tensor cores' matrices (Vulkan's cooperative matrices) and f16 in shaders, where the adapter has them: a
        // prompt's matmuls through them
        let coop = adapter.features() & (wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16);
        // (the kernels' fragments are 16 x 16 x 16, f16 into f32 sums and f16 ones; an adapter with Metal's 8 x 8 alone
        // (an Apple GPU's simdgroup matrices) has the K-quants' prompt matmuls in those, `shaders::coop8_tiled`, and
        // none of the others: `Gpu::coop_tile`)
        let shapes = adapter.cooperative_matrix_properties();
        let shape = |size: u32, sums: wgpu::CooperativeScalarType| shapes.iter().any(|p| (p.m_size, p.n_size, p.k_size) == (size, size, size) && p.ab_type == wgpu::CooperativeScalarType::F16 && p.cr_type == sums);
        let tile = if shape(16, wgpu::CooperativeScalarType::F32) && shape(16, wgpu::CooperativeScalarType::F16) {
            16
        } else if shape(8, wgpu::CooperativeScalarType::F32) {
            8
        } else {
            0
        };
        let coop = if coop == wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16 && tile != 0 && std::env::var_os("OAIY_NO_COOP").is_none() { coop } else { wgpu::Features::empty() };
        let coop_tile = if coop.is_empty() { 0 } else { tile };
        // SAFETY: wgpu's cooperative matrices are an experimental feature (its implementation may misbehave where
        // misused); only the prompt kernels use them, each checked against the f32 kernels (OAIY_NO_COOP: none).
        let experimental = if coop.is_empty() { wgpu::ExperimentalFeatures::disabled() } else { unsafe { wgpu::ExperimentalFeatures::enabled() } };
        // storage buffers in the host's memory (mappable ones), where the adapter has them: weights a card has no
        // room for, read over the bus by the kernels themselves
        // (OAIY_NO_HOST_BUFFERS: not asked for)
        let mappable = if std::env::var_os("OAIY_NO_HOST_BUFFERS").is_some() { wgpu::Features::empty() } else { adapter.features() & wgpu::Features::MAPPABLE_PRIMARY_BUFFERS };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("oaiy"),
            required_features: timestamps | coop | mappable,
            required_limits: limits.clone(),
            experimental_features: experimental,
            ..Default::default()
        }))
        .map_err(|e| format!("WebGPU device on {}: {e}", info.name))?;
        // its loss noted (a TDR, a driver's reset): a wait for it fails then
        let lost = Arc::new(Mutex::new(None));
        {
            let lost = Arc::clone(&lost);
            device.set_device_lost_callback(move |reason, message| {
                *lost.lock().unwrap_or_else(|p| p.into_inner()) = Some(format!("{reason:?}: {message}"));
            });
        }
        {
            // (an error once it is lost said to be that: its new buffers are invalid, the first use of one the error)
            let lost = Arc::clone(&lost);
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                if let Some(why) = lost.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                    panic!("webgpu: the device was lost ({why}): {e}");
                }
                panic!("wgpu error: {e}");
            }));
        }
        let watch = Arc::new(Watch::default());
        {
            let (w, l, name) = (Arc::downgrade(&watch), Arc::downgrade(&lost), info.name.clone());
            std::thread::Builder::new()
                .name("oaiy-webgpu-watchdog".into())
                .spawn(move || watchdog(w, l, name))
                .map_err(|e| format!("WebGPU device on {}: its watchdog: {e}", info.name))?;
        }
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
            // (a Mac's own GPU: the computer's memory is its memory, so its share of that, not an integrated GPU's 2 GiB)
            wgpu::DeviceType::IntegratedGpu if summary.unified => host_unified_budget(Some(&adapter)).unwrap_or(2 * GIB),
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu => 2 * GIB,
            _ => 0,
        });
        Ok(Self {
            cpu: CpuBackend::new(),
            gpu: Arc::new(Gpu { device, coop_tile, queue_raw: queue, lost, watch, feed: Arc::new(Feed::default()), in_flight_limit: Arc::new(std::sync::atomic::AtomicUsize::new(0)), layout, pipeline_layout, pipelines: Mutex::new(HashMap::new()), exl3: Mutex::new([None, None]), named: Mutex::new(HashMap::new()), names: Mutex::new(HashMap::new()), pool: Mutex::new(Vec::new()), staging: Mutex::new(Vec::new()), chain_groups: Mutex::new(HashMap::new()), wide: std::sync::OnceLock::new(), chain_groups_wide: Mutex::new(HashMap::new()), dummy: std::sync::OnceLock::new(), dummy_rw: std::sync::OnceLock::new(), limits, staged: AtomicU64::new(0), few: std::sync::OnceLock::new(), moe_steps: Mutex::new(Vec::new()), coop_units: std::sync::OnceLock::new(), dense_arena: Mutex::new(None) }),
            budget,
            used: Arc::new(AtomicU64::new(0)),
            left: Arc::new(AtomicU64::new(0)),
            summary,
            serial: Arc::new(Mutex::new(())),
            raw_adapter: adapter,
        })
    }

    pub fn adapter(&self) -> &AdapterSummary {
        &self.summary
    }

    /// How fast the host's bytes go up to the card and come back (GB/s each, timed on 64 MB three times): a card on
    /// fewer PCIe lanes is the slower (Qwen3.8 27B's prompts over two cards put the one that moves more on the faster).
    pub fn transfer_rates(&self) -> (f64, f64) {
        let len = 64usize << 20;
        let data = vec![0u8; len];
        let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("oaiy-transfer-probe"),
            size: len as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let up = || {
            self.gpu.queue().write_buffer(&buf, 0, &data);
            let i = self.gpu.queue().submit([]);
            self.gpu.wait(Some(i));
        };
        up();
        let _ = self.gpu.read(&buf, len as u64);
        let t = std::time::Instant::now();
        for _ in 0..3 {
            up();
        }
        let up_rate = 3.0 * len as f64 / t.elapsed().as_secs_f64() / 1e9;
        let t = std::time::Instant::now();
        for _ in 0..3 {
            let _ = self.gpu.read(&buf, len as u64);
        }
        (up_rate, 3.0 * len as f64 / t.elapsed().as_secs_f64() / 1e9)
    }

    /// The OS's budget for this process on the card's memory, and its use of it now (Vulkan's): None where the API
    /// does not say.
    /// Wait for the GPU's work and let go of the buffers dropped since (wgpu frees them at a poll or submit: a stage's
    /// vectors dropped before the next stage makes its own would otherwise share the card with them).
    pub fn settle(&self) {
        self.gpu.wait(None);
    }

    /// Let go of what the device keeps between chains' runs (the kept bind groups and the vectors they hold, the
    /// scratch and staging pools), then [`Self::settle`]: the room for another model's work on the same card (a
    /// diffusion step's kept some 10 GB past its weights).
    pub fn release_cached(&self) {
        self.gpu.chain_groups.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.chain_groups_wide.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.pool.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.gpu.staging.lock().unwrap_or_else(|p| p.into_inner()).clear();
        self.settle();
    }

    /// Whether the adapter has cooperative matrices (the tensor cores' kernels: f16, NVFP4 and the K-quants' and Q8_0's
    /// for a prompt's rows; on Metal, 8 x 8, the K-quants' and Q8_0's alone).
    pub fn tensor_cores(&self) -> bool {
        self.gpu.coop_tile != 0
    }

    /// The most bytes one vector an op reads or writes may hold (the adapter's storage binding limit: 2 GB here).
    pub fn max_binding(&self) -> u64 {
        (self.gpu.limits.max_storage_buffer_binding_size as u64).min(self.gpu.limits.max_buffer_size)
    }

    pub fn memory_budget(&self) -> Option<(u64, u64)> {
        heap_budget(&self.raw_adapter)
    }

    /// The device's limits that a kernel meets, for a log: what a GPU no test of this crate has run on gives its
    /// kernels (an Apple GPU's are in wgpu-hal's Metal adapter; `OAIY_PORTABLE_LIMITS` holds another GPU to them).
    pub fn limits_line(&self) -> String {
        let l = &self.gpu.limits;
        let gib = |bytes: u64| bytes as f64 / GIB as f64;
        // (a Mac's: what Metal recommends the GPU hold at most, which its share for weights is taken from)
        let metal = metal_working_set(&self.raw_adapter).map_or(String::new(), |bytes| format!("; Metal recommends it hold at most {:.1} GiB", gib(bytes)));
        format!(
            "a buffer up to {:.1} GiB and a binding {:.1} GiB, {} storage buffers a kernel, a workgroup of {} threads with {} KB, {} workgroups a dimension, tensor cores {}{metal}",
            gib(l.max_buffer_size),
            gib(l.max_storage_buffer_binding_size as u64),
            l.max_storage_buffers_per_shader_stage,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroup_storage_size >> 10,
            l.max_compute_workgroups_per_dimension,
            match self.gpu.coop_tile {
                16 => "used",
                8 => "used as Metal's 8x8 simdgroup matrices (a prompt's K-quant and Q8_0 matmuls)",
                _ => "not used",
            },
        )
    }

    /// At most `pieces` of the chains' pieces on this device's queue at once from here on (0: as many as are recorded,
    /// as it is until asked), each encoded just before it is submitted ([`Feed`]): for a loop of short steps, or a
    /// prompt's chunks, that runs the GPU for seconds on end. A recording's pieces otherwise go to the GPU as they are
    /// recorded, a step's dozen queued within its first milliseconds; twelve such steps a second (MiniMax Music 3's
    /// transformer: 800 dispatches a step, 82 ms), or a prompt's chunks one after another (the 27B's 512 tokens in
    /// 0.2 s), and an RTX 5090 under a power limit (402 W of its 575) fell, after some 5 to 7 s and then every few, into
    /// seconds of running a tenth as fast (a matmul over large weights 12 to 17 times as long, an attention 2 to 3: its
    /// memory busy three times as much of the time, its clock no lower and its limiter less at work than before),
    /// where so fed it does so once, for a second or two, and settles (Candle's CUDA, a kernel at a time, never
    /// does). The pieces' number on the queue is the lesser part of it: with one, two or three on the queue but each
    /// encoded when it was recorded, the card fell so as before; two encoded at their turn cost a step or a chunk
    /// nothing, three some 6% of a long prompt's chunk. Not for long dispatches (Pixal3D's flows over 15,000 voxels,
    /// 1.2 s a pass: steady as they are, and with a limit the throttle found them now and then).
    pub fn pieces_in_flight_at_most(&self, pieces: usize) {
        self.gpu.in_flight_limit.store(pieces, Ordering::Relaxed);
    }

    /// Bytes of weights placed on the GPU, and the budget.
    pub fn usage(&self) -> (u64, u64) {
        (self.used.load(Ordering::Relaxed), self.budget)
    }

    /// Bytes of the weights a model asked this GPU to hold that it had no room for within its budget: they are on the
    /// host, multiplied by the CPU straight from the model's file as it is mapped (so the system reads them from the
    /// drive as they are used, and again where it has no memory to keep them in).
    pub fn left_on_host(&self) -> u64 {
        self.left.load(Ordering::Relaxed)
    }

    /// Whether this device's kernels can read weights from the host's memory ([`Gpu::host_buffer`]): a card of its own
    /// on Vulkan, whose backend puts such a buffer in the system's memory (an integrated GPU's memory is the
    /// system's already, and the other backends say nothing of where a mappable buffer goes).
    pub fn host_weights(&self) -> bool {
        self.gpu.device.features().contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) && self.summary.backend == "Vulkan" && self.summary.device_type == "DiscreteGpu"
    }

    /// [`ggml_rs::DeviceChain::copy_weight`]: `w`'s buffers as another adapter holds them (blocks padded alike) read
    /// back and put on this one, within its budget.
    pub(crate) fn copy_weight(&self, w: &QuantizedTensor) -> Option<QuantizedTensor> {
        let q = w.device_storage()?.as_any().downcast_ref::<WgpuQuant>()?;
        if Arc::ptr_eq(&q.gpu, &self.gpu) {
            return None;
        }
        let prev = self.used.fetch_add(q.nbytes as u64, Ordering::Relaxed);
        if prev + q.nbytes as u64 > self.budget {
            self.used.fetch_sub(q.nbytes as u64, Ordering::Relaxed);
            return None;
        }
        let chunks = self.gpu.upload_rows(&q.gpu_bytes(), q.row_bytes, 1);
        let storage = WgpuQuant { gpu: Arc::clone(&self.gpu), dtype: q.dtype, chunks, row_bytes: q.row_bytes, nbytes: q.nbytes, used: Arc::clone(&self.used) };
        Some(QuantizedTensor::from_device(Box::new(storage), w.shape().to_vec()))
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
        let host_row = padded.map_or(row_bytes, |(host, gpu, _)| row_bytes / gpu * host);
        if w.bytes().len() != host_row * w.dim(0) || row_bytes as u64 > chunk_limit(&self.gpu.limits) {
            return w;
        }
        let nbytes = row_bytes * w.dim(0);
        // Reserve before uploading so concurrent loads cannot overshoot together.
        let prev = self.used.fetch_add(nbytes as u64, Ordering::Relaxed);
        if prev + nbytes as u64 > self.budget {
            self.used.fetch_sub(nbytes as u64, Ordering::Relaxed);
            self.left.fetch_add(nbytes as u64, Ordering::Relaxed);
            return w;
        }
        let chunks = match padded {
            Some((host, gpu, at)) => self.gpu.upload_rows(&shaders::pad_blocks(w.bytes(), host, gpu, at), row_bytes, 1),
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
        gpu.queue().write_buffer(&xbuf, 0, &bytes(x.data()));
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
                gpu.queue().write_buffer(&pbuf, 0, &params);
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
        gpu.queue().submit([enc.finish()]);
        let slice = staging.slice(..total);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.wait(None);
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

#[cfg(test)]
mod tests;
