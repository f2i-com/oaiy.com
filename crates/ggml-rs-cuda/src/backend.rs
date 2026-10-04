//! CUDA backend implementation.
//!
//! v0.5: tensors stay on device between ops. Inputs that arrive as CPU
//! tensors are uploaded transparently the first time they're used; outputs
//! are device tensors that downstream ops consume directly. Compared to the
//! v0 per-op-transfer design this turns a forward pass on a small model from
//! O(layers × ops × n_h2d) host↔device round-trips into ~2 round-trips total
//! (token IDs in, last-row logits out).

#![allow(deprecated)] // memcpy_stod / memcpy_dtov / memcpy_dtod are the available cudarc names

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use ggml_quants::GgmlType;
use ggml_rs::backend::{Backend, RopeType};
use ggml_rs::quantized::{QuantizedDeviceStorage, QuantizedTensor};
use ggml_rs::tensor::{DeviceStorage, Tensor};
use thiserror::Error;

use crate::kernels::{kernel_image, KERNEL_NAMES};

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("cuda driver error: {0:?}")]
    Driver(cudarc::driver::DriverError),



    #[error("kernel `{0}` not found in compiled module")]
    MissingKernel(&'static str),
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self { Self::Driver(e) }
}

// ----- Device storage -------------------------------------------------------

/// Concrete `DeviceStorage` (F32) implementation for CUDA tensors.
pub struct CudaStorage {
    pub(crate) slice:  CudaSlice<f32>,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) name:   String,
}

/// Concrete `QuantizedDeviceStorage` for CUDA. Holds packed bytes + dtype.
pub struct CudaQuantStorage {
    pub(crate) bytes:  CudaSlice<u8>,
    pub(crate) dtype:  GgmlType,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) name:   String,
}

impl std::fmt::Debug for CudaQuantStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaQuantStorage")
            .field("device", &self.name)
            .field("dtype", &self.dtype)
            .field("nbytes", &self.bytes.len())
            .finish()
    }
}

impl QuantizedDeviceStorage for CudaQuantStorage {
    fn nbytes(&self) -> usize { self.bytes.len() }
    fn dtype(&self) -> GgmlType { self.dtype }
    fn device_name(&self) -> &str { &self.name }

    fn copy_to_host(&self) -> Vec<u8> {
        self.stream.memcpy_dtov(&self.bytes).expect("d2h failed")
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn clone_to_device(&self) -> Box<dyn QuantizedDeviceStorage> {
        // VENDORED-LOCAL: SAFETY: the following same-stream D2D copy initializes
        // every byte before the cloned storage can be read.
        let mut new_slice = unsafe { self.stream.alloc::<u8>(self.bytes.len()) }.expect("alloc failed");
        self.stream
            .memcpy_dtod(&self.bytes, &mut new_slice)
            .expect("d2d failed");
        Box::new(CudaQuantStorage {
            bytes: new_slice,
            dtype: self.dtype,
            stream: self.stream.clone(),
            name: self.name.clone(),
        })
    }
}

impl std::fmt::Debug for CudaStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("device", &self.name)
            .field("len", &self.slice.len())
            .finish()
    }
}

impl DeviceStorage for CudaStorage {
    fn len(&self) -> usize { self.slice.len() }
    fn device_name(&self) -> &str { &self.name }

    fn copy_to_host(&self) -> Vec<f32> {
        self.stream.memcpy_dtov(&self.slice).expect("d2h failed")
    }

    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn clone_to_device(&self) -> Box<dyn DeviceStorage> {
        // VENDORED-LOCAL: SAFETY: the following same-stream D2D copy initializes
        // every element before the cloned storage can be read.
        let mut new_slice = unsafe { self.stream.alloc::<f32>(self.slice.len()) }.expect("alloc failed");
        self.stream
            .memcpy_dtod(&self.slice, &mut new_slice)
            .expect("d2d failed");
        Box::new(CudaStorage {
            slice: new_slice,
            stream: self.stream.clone(),
            name: self.name.clone(),
        })
    }
}

// ----- Backend --------------------------------------------------------------

pub struct CudaBackend {
    weight_budget: usize,
    weight_resident: std::sync::atomic::AtomicUsize,
    streamed_bytes: std::sync::atomic::AtomicU64,
    // VENDORED-LOCAL: GPU-02 — `ctx`/`stream`/`name` widened to pub(crate) so
    // the transfer module (pinned staging + H2D overlap) can build on them.
    pub(crate) ctx:    Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    funcs:  HashMap<&'static str, CudaFunction>,
    // VENDORED-LOCAL: GPU-02 — dedicated non-blocking H2D transfer stream and
    // a recycling pool of completion events (steady-state decode must not
    // create a CUDA event per upload).
    //
    // The stream is created LAZILY on first transfer-API use, not in `new`.
    // (History: creating it eagerly flipped cudarc into multi-stream mode for
    // every model, and with cudarc's event tracking on — the default — every
    // kernel launch then paid `cuStreamWaitEvent` + `cuEventRecord` per
    // tensor argument, halving dense/resident decode on WDDM (75 → 45 tok/s
    // on qwen3-0.6b). Tracking is now disabled context-wide in `new` — see
    // PERF-02 there — so multi-stream mode no longer costs per-launch driver
    // calls; lazy init is kept so resident-only workloads still skip the
    // second stream, its `CudaContext::synchronize` on creation, and all
    // multi-stream bookkeeping entirely.)
    h2d:                   std::sync::OnceLock<Arc<CudaStream>>,
    pub(crate) event_pool: crate::transfer::EventPool,
    /// VENDORED-LOCAL: recently uploaded rope positions, by content: every layer of a forward
    /// ropes at the same positions, so they cross to the device once, not once a layer.
    rope_positions: std::sync::Mutex<Vec<(Vec<u32>, Arc<CudaSlice<u32>>)>>,
    // VENDORED-LOCAL: a private stream for recording graphs, never for running
    // inference. Its mutex prevents overlapping capture across projections.
    capture: std::sync::OnceLock<std::sync::Mutex<Arc<CudaStream>>>,
    /// VENDORED-LOCAL: a whole step captured into one graph (see `graph_begin`).
    graph: std::sync::Mutex<GraphRun>,
    pub(crate) name:   String,
}

/// VENDORED-LOCAL: the graph a step is captured into and replayed from, updated in place from
/// each new capture (the launches repeat step to step; their arguments change), and the arena
/// the captured step's temporaries come from.
#[derive(Default)]
struct GraphRun {
    exec: usize,
    arena: Option<CudaSlice<u8>>,
    /// What the captured step dropped, freed once it has run.
    deferred: Vec<cudarc::driver::sys::CUdeviceptr>,
    /// How much of the arena the last capture used.
    used: usize,
}

/// VENDORED-LOCAL: room for one captured step's temporaries, at first. Some grow with the
/// context (long attention's block scores), so the arena doubles whenever a step fills half.
const GRAPH_ARENA: usize = 256 << 20;

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let g = self.graph.get_mut().unwrap_or_else(|e| e.into_inner());
        if g.exec != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            // SAFETY: the exec is ours alone; the stream finishes any replay before its memory goes.
            unsafe { let _ = cudarc::driver::result::graph::exec_destroy(g.exec as cudarc::driver::sys::CUgraphExec); }
            g.exec = 0;
        }
    }
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("name", &self.name)
            .finish()
    }
}

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    fn delta_net_inner(
        &self,
        mixed_qkv:   &Tensor,
        z_in:        &Tensor,
        beta_alpha:  &Tensor,
        conv_weight: &Tensor,
        ssm_a:       &Tensor,
        dt_bias:     &Tensor,
        ssm_norm:    &Tensor,
        conv_state:  &mut Tensor,
        state:       &mut Tensor,
        seq:         usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_v_dim:  usize,
        head_k_dim:  usize,
        v_per_k:     usize,
        scale_q:     f32,
        eps:         f32,
        gate_mode:   i32,
    ) -> Tensor {
        let conv_dim    = mixed_qkv.numel() / seq;
        let conv_kernel = conv_weight.dim(1);
        let mqkv_dev = self.cuda_input(mixed_qkv);
        let z_dev    = self.cuda_input(z_in);
        let ba_dev   = self.cuda_input(beta_alpha);
        let cw_dev   = self.cuda_input(conv_weight);
        let sa_dev   = self.cuda_input(ssm_a);
        let dt_dev   = self.cuda_input(dt_bias);
        let nm_dev   = self.cuda_input(ssm_norm);
        let cs_dev   = self.cuda_input_mut(conv_state);
        let st_dev   = self.cuda_input_mut(state);

        // Allocate intermediate conv_out for ALL seq tokens (the loop kernels
        // write/read the full [seq, conv_dim] buffer rather than reusing one
        // token's worth — keeps the kernels single-launch).
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut conv_out = unsafe { self.stream.alloc::<f32>(seq * conv_dim) }.expect("output allocation");
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut output = unsafe { self.stream.alloc::<f32>(seq * num_v_heads * head_v_dim) }.expect("output allocation");

        let block_x: u32 = 256;
        let grid_x = ((conv_dim as u32) + block_x - 1) / block_x;
        let conv_cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let step_cfg = LaunchConfig {
            grid_dim: (num_v_heads as u32, 1, 1),
            block_dim: (head_v_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let cd_i = conv_dim as i32;
        let ck_i = conv_kernel as i32;
        let nvh = num_v_heads as i32;
        let nkh = num_k_heads as i32;
        let hvd = head_v_dim as i32;
        let hkd = head_k_dim as i32;
        let vpk = v_per_k as i32;

        // One launch per layer per direction (conv + step). The seq loop runs
        // INSIDE each kernel — seq*64KB of state I/O eliminated by holding the
        // per-head state row in registers across iterations.
        unsafe {
            self.stream
                .launch_builder(self.func("delta_net_conv1d_loop_f32"))
                .arg(mqkv_dev.as_ref())
                .arg(cs_dev)
                .arg(cw_dev.as_ref())
                .arg(&mut conv_out)
                .arg(&seq_i).arg(&cd_i).arg(&ck_i)
                .launch(conv_cfg)
                .expect("delta_net_conv1d_loop launch");
        }
        // VENDORED-LOCAL: long prefills benefit from distributing state rows
        // across SMs. Decode retains its single fused launch.
        if head_k_dim == 128 && head_v_dim == 128 {
            // SAFETY: every core/output element and state row has one writer.
            // Full warps own 128-wide rows; the second launch runs after the
            // first on this stream, and all buffers outlive both launches.
            let mut core =
                unsafe { self.stream.alloc::<f32>(seq * num_v_heads * 128) }.expect("delta core");
            // SAFETY: launch dimensions cover disjoint rows/elements, and the
            // ordered stream keeps the initialized core alive for normalization.
            unsafe {
                self.stream
                    .launch_builder(self.func("delta_net_rows_128_f32"))
                    .arg(&conv_out)
                    .arg(ba_dev.as_ref())
                    .arg(sa_dev.as_ref())
                    .arg(dt_dev.as_ref())
                    .arg(st_dev)
                    .arg(&mut core)
                    .arg(&seq_i)
                    .arg(&nvh)
                    .arg(&nkh)
                    .arg(&scale_q)
                    .arg(&eps)
                    .launch(LaunchConfig {
                        grid_dim: (num_v_heads as u32, 32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("delta rows");
                self.stream
                    .launch_builder(self.func("delta_net_norm_128_f32"))
                    .arg(&core)
                    .arg(z_dev.as_ref())
                    .arg(nm_dev.as_ref())
                    .arg(&mut output)
                    .arg(&eps)
                    .arg(&gate_mode)
                    .launch(LaunchConfig {
                        grid_dim: ((seq * num_v_heads) as u32, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                    .expect("delta norm");
            }
            return self.make_tensor(output, vec![seq, num_v_heads * head_v_dim]);
        }
        unsafe {
            self.stream
                .launch_builder(self.func("delta_net_step_loop_f32"))
                .arg(&conv_out)
                .arg(z_dev.as_ref())
                .arg(ba_dev.as_ref())
                .arg(sa_dev.as_ref())
                .arg(dt_dev.as_ref())
                .arg(nm_dev.as_ref())
                .arg(st_dev)
                .arg(&mut output)
                .arg(&seq_i)
                .arg(&nvh).arg(&nkh).arg(&hvd).arg(&hkd).arg(&vpk)
                .arg(&scale_q).arg(&eps).arg(&gate_mode)
                .launch(step_cfg)
                .expect("delta_net_step_loop launch");
        }

        self.make_tensor(output, vec![seq, num_v_heads * head_v_dim])
    }

    pub fn new(device_ordinal: usize) -> Result<Self, CudaError> {
        Self::with_stream(device_ordinal, false)
    }

    /// VENDORED-LOCAL: a backend on a stream of its own rather than the device's default one,
    /// so that its steps can be captured into graphs (`graph_begin`).
    pub fn new_graphable(device_ordinal: usize) -> Result<Self, CudaError> {
        Self::with_stream(device_ordinal, true)
    }

    fn with_stream(device_ordinal: usize, own: bool) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = if own { ctx.new_stream()? } else { ctx.default_stream() };

        // VENDORED-LOCAL: machine code for the installed GPU, compiled at build
        // time (build.rs), so EXL3 can sum four codebook bytes with one DP4A
        // instruction; older devices keep the scalar fallback, and no relaxed
        // floating-point compiler flags are used. No NVRTC at run time.
        let module = ctx.load_module_image(kernel_image(ctx.compute_capability()?))?;

        let mut funcs = HashMap::with_capacity(KERNEL_NAMES.len());
        for &n in KERNEL_NAMES {
            let f = module.load_function(n).map_err(|_| CudaError::MissingKernel(n))?;
            funcs.insert(n, f);
        }

        // VENDORED-LOCAL: GPU-02/PERF-02 — turn cudarc's per-slice event
        // tracking OFF for the whole context. With tracking on (the cudarc
        // default), every CudaSlice carries read/write events and every
        // `device_ptr`/`device_ptr_mut` extraction in multi-stream mode —
        // several per kernel launch — issues `cuStreamWaitEvent` +
        // `cuEventRecord` driver calls. Measured on WDDM (RTX 5090, 30B MoE
        // streaming decode): ~76µs per kernel-launch enqueue vs ~15µs with
        // tracking off — the launch thread, not the GPU, was the bottleneck
        // (3.8 → 5.3 tok/s from this change alone).
        //
        // Safety without tracking, by slice lifetime:
        // * resident weights / activations / KV live and die on the compute
        //   stream only — same-stream ordering already covers them;
        // * expert upload slots cross streams (H2D write on `h2d`, reads on
        //   the compute stream): the write→read edge is ordered explicitly
        //   by `upload_async`'s ticket + `wait_upload` (cuStreamWaitEvent),
        //   and the read→free edge by `CudaBackend::order_transfer_after_compute`,
        //   which the VRAM expert cache calls from `DeviceEntry::drop`;
        // * the pinned staging ring keeps its own per-buffer event guard
        //   (`PinnedHostSlice::stream_synced_slice` is unconditional in
        //   cudarc — it does not consult this flag).
        unsafe { ctx.disable_event_tracking() };

        Ok(Self {
            weight_budget: usize::MAX,
            weight_resident: Default::default(),
            streamed_bytes: Default::default(),
            ctx,
            stream,
            module,
            funcs,
            h2d: Default::default(), // created lazily — see the field comment
            event_pool: Default::default(),
            rope_positions: Default::default(),
            capture: Default::default(),
            graph: Default::default(),
            name: format!("cuda:{device_ordinal}"),
        })
    }

    /// Bound persistent packed weights; other mapped weights upload on demand.
    /// Dense activations/norms and temporary staging are outside this budget.
    pub fn with_weight_budget(mut self, bytes: usize) -> Self { self.weight_budget = bytes; self }
    pub fn weight_stats(&self) -> (usize, u64) {
        (self.weight_resident.load(std::sync::atomic::Ordering::Relaxed),
         self.streamed_bytes.load(std::sync::atomic::Ordering::Relaxed))
    }
    // VENDORED-LOCAL: GPU-02 — the H2D transfer stream, created on first use
    // (see the `h2d` field comment for why this must not happen in `new`).
    pub(crate) fn h2d(&self) -> &Arc<CudaStream> {
        self.h2d.get_or_init(|| {
            self.ctx.new_stream().expect("h2d transfer stream")
        })
    }

    /// VENDORED-LOCAL: start capturing a step (e.g. a decode step's layers on this device) into
    /// a graph instead of launching it: false (and nothing captured) on the default stream.
    /// Until `graph_end`, nothing may wait on the device (no reads back, no uploads from
    /// pageable memory); what the step allocates comes from an arena reused every step, so a
    /// tensor it makes must be done with before the next capture, and state kept across steps
    /// must be updated in place.
    pub fn graph_begin(&self) -> bool {
        if self.stream.cu_stream() == self.ctx.default_stream().cu_stream() {
            return false;
        }
        let mut g = self.graph.lock().unwrap_or_else(|e| e.into_inner());
        let size = g.arena.as_ref().map_or(0, |a| a.len());
        if size == 0 || 2 * g.used > size {
            let grown = (4 * g.used).max(GRAPH_ARENA).next_power_of_two();
            // The old arena goes in stream order, after the steps that used it.
            // SAFETY: scratch, written by the captured kernels before they read it.
            g.arena = Some(unsafe { self.stream.alloc::<u8>(grown) }.expect("graph arena"));
        }
        let arena = g.arena.as_ref().unwrap();
        let (base, _record) = cudarc::driver::DevicePtr::device_ptr(arena, &self.stream);
        // SAFETY: the arena outlives every graph captured with it: a grown one replaces it
        // only before a capture, and the exec is updated to the new addresses then.
        unsafe { self.stream.begin_graph(base, arena.len()) }.is_ok()
    }

    /// VENDORED-LOCAL: end the capture `graph_begin` started and run it.
    pub fn graph_end(&self) {
        self.graph_finish();
        self.graph_launch();
    }

    /// VENDORED-LOCAL: end the capture `graph_begin` started, ready for `graph_launch`: the
    /// previous step's graph updated to this one's arguments when the launches match, or a
    /// new one. (Between the two, e.g., the step's input can be written.)
    pub fn graph_finish(&self) {
        use cudarc::driver::{result, sys};
        let (graph, deferred, used) = self.stream.end_graph().expect("end of the step's capture");
        let mut g = self.graph.lock().unwrap_or_else(|e| e.into_inner());
        g.deferred.extend(deferred);
        g.used = used;
        // SAFETY: the graph and exec are this backend's alone; the exec replays on its stream.
        unsafe {
            let mut updated = false;
            if g.exec != 0 {
                let mut info = std::mem::zeroed::<sys::CUgraphExecUpdateResultInfo>();
                updated = sys::cuGraphExecUpdate_v2(g.exec as sys::CUgraphExec, graph, &mut info).result().is_ok();
                if !updated {
                    let _ = result::graph::exec_destroy(g.exec as sys::CUgraphExec);
                    g.exec = 0;
                }
            }
            if !updated {
                g.exec = result::graph::instantiate(graph, sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                    .expect("step graph") as usize;
            }
            let _ = result::graph::destroy(graph);
        }
    }

    /// VENDORED-LOCAL: run the step `graph_finish` readied.
    pub fn graph_launch(&self) {
        use cudarc::driver::{result, sys};
        let mut g = self.graph.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the exec is this backend's alone; it replays on its stream.
        unsafe { result::graph::launch(g.exec as sys::CUgraphExec, self.stream.cu_stream()) }.expect("step graph launch");
        // What the step dropped (e.g. its input) goes once the graph has run.
        let deferred = std::mem::take(&mut g.deferred);
        drop(g);
        self.stream.free_after(deferred);
    }

    /// VENDORED-LOCAL: where a tensor on this device lives, for `write_at`.
    pub fn device_address(&self, t: &Tensor) -> u64 {
        assert!(t.device_storage().is_some(), "device_address of a host tensor");
        let input = self.cuda_input(t);
        let (address, _record) = cudarc::driver::DevicePtr::device_ptr(input.as_ref(), &self.stream);
        address
    }

    /// VENDORED-LOCAL: copy `data` to `address` (from `device_address`), in stream order.
    pub fn write_at(&self, address: u64, data: &[f32]) {
        self.ctx.bind_to_thread().expect("bind");
        // SAFETY: the caller's address holds at least `data.len()` floats and lives until the
        // copy (queued on this stream) is done; pageable memory is staged before this returns.
        unsafe { cudarc::driver::result::memcpy_htod_async(address, data, self.stream.cu_stream()) }.expect("h2d write");
    }

    /// VENDORED-LOCAL: whether a step is being captured (`graph_begin`).
    pub fn graph_recording(&self) -> bool { self.stream.graph_recording() }

    /// VENDORED-LOCAL: upload rope positions ahead of a capture, which cannot.
    pub fn prime_rope_positions(&self, positions: &[u32]) { let _ = self.cached_positions(positions); }

    pub(crate) fn capture_stream(&self) -> std::sync::MutexGuard<'_, Arc<CudaStream>> {
        self.capture.get_or_init(|| std::sync::Mutex::new(self.ctx.new_stream().expect("graph capture stream")))
            .lock().unwrap_or_else(|e|e.into_inner())
    }

    pub fn context(&self) -> &Arc<CudaContext> { &self.ctx }

    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module's launches.
    pub(crate) fn func(&self, name: &'static str) -> &CudaFunction { &self.funcs[name] }

    /// VENDORED-LOCAL: `gemm_f32` (kernels.cu), the GEMM cuBLAS ran: for each of `batch`
    /// batches, `c[m*ldc + n] = alpha * Σ_k a[m*lda + k] · B(k, n)`, B(k, n) being
    /// `b[n*ldb + k]` when `b_nk` and `b[k*ldb + n]` otherwise, batch z's operands offset by
    /// z times their strides `(batch, sa, sb, sc)`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, (m, n, k): (usize, usize, usize),
            (lda, ldb, ldc): (usize, usize, usize), b_nk: bool, (batch, sa, sb, sc): (usize, usize, usize, usize), alpha: f32) {
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(128), (m as u32).div_ceil(128), batch as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mi, ni, ki, lda, ldb, ldc, b_nk) = (m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32, b_nk as i32);
        let (sa, sb, sc) = (sa as i64, sb as i64, sc as i64);
        // SAFETY: the callers size a, b and c for these shapes, strides and batches; every
        // element of C in range is written; one ordered stream.
        unsafe {
            self.stream.launch_builder(self.func("gemm_f32"))
                .arg(a).arg(b).arg(c)
                .arg(&mi).arg(&ni).arg(&ki).arg(&lda).arg(&ldb).arg(&ldc).arg(&b_nk)
                .arg(&sa).arg(&sb).arg(&sc).arg(&alpha)
                .launch(cfg)
                .expect("gemm_f32 launch");
        }
    }
    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module's launches.
    pub(crate) fn func_dyn(&self, name: &str) -> &CudaFunction {
        self.funcs.get(name).unwrap_or_else(|| panic!("CUDA kernel `{name}` not loaded"))
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. Which shapes take the split-K GEMV.
    ///
    /// Short and wide only: with `out` rows there are `out` warps, so below a few
    /// hundred rows the card sits idle while a long reduction runs in each one.
    /// Ordinary transformer weights are thousands of rows and keep the existing
    /// path, which is deliberate -- this must not move any other model's numbers.
    fn split_gemv_kernel(dtype: GgmlType, out: usize, in_: usize) -> Option<&'static str> {
        const MAX_OUT: usize = 320;
        const MIN_IN: usize = 2048;
        if out > MAX_OUT || in_ < MIN_IN {
            return None;
        }
        match dtype {
            GgmlType::Q8_0 if in_ % 32 == 0 => Some("gemv_split_q8_0_f32"),
            GgmlType::F32 => Some("gemv_split_f32"),
            _ => None,
        }
    }

    /// How many ways to split the reduction: enough warps to fill the card,
    /// without making each one so short that the launch dominates.
    fn gemv_splits(dtype: GgmlType, in_: usize) -> usize {
        let units = match dtype {
            GgmlType::Q8_0 => in_ / 32, // Q8_0 blocks
            _ => in_ / 32,              // one 32-wide strip per lane sweep
        };
        (units / 32).clamp(2, 32)
    }

    fn alloc(&self, n: usize) -> CudaSlice<f32> {
        self.stream.alloc_zeros::<f32>(n).expect("alloc failed")
    }

    /// VENDORED-LOCAL: an output buffer a kernel writes in full, left uninitialized: the zeroing
    /// `alloc` does is a memset launch of its own, and decode makes hundreds a token.
    fn alloc_uninit(&self, n: usize) -> CudaSlice<f32> {
        // SAFETY: only for outputs every element of which the following launch writes.
        unsafe { self.stream.alloc::<f32>(n.max(1)) }.expect("alloc failed")
    }

    fn alloc_u32(&self, n: usize) -> CudaSlice<u32> {
        self.stream.alloc_zeros::<u32>(n).expect("alloc failed")
    }

    fn upload_f32(&self, host: &[f32]) -> CudaSlice<f32> {
        self.stream.memcpy_stod(host).expect("h2d failed")
    }

    /// VENDORED-LOCAL: `positions` on the device, uploaded once while they stay in use (the
    /// last few kept; never written after upload, so sharing them is safe).
    /// VENDORED-LOCAL: attention over `width` keys per query row, split across blocks and then
    /// combined: the keys `sel` lists (-1 for none), or with no `sel` every key j that row r
    /// sees (j <= past + r).
    #[allow(clippy::too_many_arguments)]
    fn split_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, sel: Option<&Tensor>, width: usize, scale: f32, past: usize) -> Tensor {
        let n = q.dim(0);
        let (heads, d) = (q.dim(1), q.dim(2));
        let kv_heads = k.dim(1);
        // Enough blocks to fill the card: a few query rows split their keys finely.
        let splits = if n * heads >= 256 { (width / 256).clamp(1, 16) } else { width.div_ceil(64).clamp(1, 32) };
        let per = width.div_ceil(splits);
        let q_in = self.cuda_input(q);
        let k_in = self.cuda_input(k);
        let v_in = self.cuda_input(v);
        let s_in = sel.map(|s| self.cuda_input(s));
        let null: u64 = 0;
        let mut part = self.alloc_uninit(n * heads * splits * (d + 2));
        let mut out = self.alloc_uninit(n * heads * d);
        let (h_i, g_i, d_i, w_i, sp_i, p_i) = (heads as i32, kv_heads as i32, d as i32, width as i32, splits as i32, past as i32);
        unsafe {
            let mut launch = self.stream.launch_builder(self.func("qsa_sparse_partial_f32"));
            launch.arg(q_in.as_ref()).arg(k_in.as_ref()).arg(v_in.as_ref());
            match &s_in { Some(s) => { launch.arg(s.as_ref()); } None => { launch.arg(&null); } }
            launch.arg(&mut part).arg(&h_i).arg(&g_i).arg(&d_i).arg(&w_i).arg(&sp_i).arg(&scale).arg(&p_i)
                .launch(LaunchConfig { grid_dim: (n as u32, heads as u32, splits as u32), block_dim: (256, 1, 1), shared_mem_bytes: ((per + d) * 4) as u32 })
                .expect("qsa_sparse_partial launch");
            self.stream.launch_builder(self.func("qsa_sparse_combine_f32"))
                .arg(&part).arg(&mut out).arg(&h_i).arg(&d_i).arg(&sp_i)
                .launch(LaunchConfig { grid_dim: (n as u32, heads as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("qsa_sparse_combine launch");
        }
        self.make_tensor(out, vec![n, heads, d])
    }

    fn cached_positions(&self, positions: &[u32]) -> Arc<CudaSlice<u32>> {
        let mut cache = self.rope_positions.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, dev)) = cache.iter().find(|(host, _)| host.as_slice() == positions) {
            return dev.clone();
        }
        let dev = Arc::new(self.upload_u32(positions));
        if cache.len() >= 4 { cache.remove(0); }
        cache.push((positions.to_vec(), dev.clone()));
        dev
    }

    fn upload_u32(&self, host: &[u32]) -> CudaSlice<u32> {
        self.stream.memcpy_stod(host).expect("h2d failed")
    }

    /// Upload a packed-quant byte buffer + dtype as a `QuantizedTensor`
    /// backed by `CudaQuantStorage`.
    pub fn upload_quantized(&self, host_bytes: &[u8], shape: Vec<usize>, dtype: GgmlType)
        -> QuantizedTensor
    {
        let bytes = self.stream.memcpy_stod(host_bytes).expect("h2d quant");
        QuantizedTensor::from_device(
            Box::new(CudaQuantStorage {
                bytes,
                dtype,
                stream: self.stream.clone(),
                name: self.name.clone(),
            }),
            shape,
        )
    }

    /// Stream output-row tiles without expanding the packed weight matrix.
    /// Only the small result is assembled on host; subsequent ops upload it.
    fn linear_q_tiled(&self, x: &Tensor, w: &QuantizedTensor, tile_rows: usize) -> Tensor {
        let input = w.dim(1);
        let output = w.dim(0);
        let batch = x.numel() / input;
        let row_bytes = input / w.dtype().block_size() * w.dtype().type_size();
        let mut values = vec![0.0; batch * output];
        let mut first = 0;
        let mut rows = tile_rows.max(1);
        while first < output {
            let count = rows.min(output - first);
            let bytes = &w.bytes()[first * row_bytes..(first + count) * row_bytes];
            let device = match self.stream.memcpy_stod(bytes) {
                Ok(device) => device,
                Err(_) if count > 1 => { rows = (count / 2).max(1); continue; }
                Err(e) => panic!("insufficient VRAM even for one packed weight row; model state/activations must fit: {e:?}"),
            };
            self.streamed_bytes.fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            let tile = QuantizedTensor::from_device(Box::new(CudaQuantStorage {
                bytes: device, dtype: w.dtype(), stream: self.stream.clone(), name: self.name.clone(),
            }), vec![count, input]);
            let result = self.linear_q(x, &tile).to_host();
            for b in 0..batch {
                values[b * output + first..b * output + first + count]
                    .copy_from_slice(&result.data()[b * count..(b + 1) * count]);
            }
            first += count;
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = output;
        Tensor::from_vec(values, shape)
    }

    /// Borrow `&CudaSlice<u8>` for a quantized tensor that may be on host or
    /// device. Mirrors `cuda_input` for F32 tensors.
    fn cuda_quant_input<'a>(&self, w: &'a QuantizedTensor) -> CudaQuantInput<'a> {
        if let Some(s) = w.device_storage() {
            if let Some(c) = s.as_any().downcast_ref::<CudaQuantStorage>() {
                return CudaQuantInput::Borrowed(&c.bytes);
            }
            panic!("QuantizedTensor on `{}` passed to CudaBackend", s.device_name());
        }
        self.streamed_bytes.fetch_add(w.nbytes() as u64, std::sync::atomic::Ordering::Relaxed);
        let bytes = self.stream.memcpy_stod(w.bytes()).expect("h2d quant input");
        CudaQuantInput::Owned(bytes)
    }

    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module.
    pub(crate) fn make_tensor(&self, slice: CudaSlice<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::from_device(
            Box::new(CudaStorage {
                slice,
                stream: self.stream.clone(),
                name: self.name.clone(),
            }),
            shape,
        )
    }

    /// Get a `&CudaSlice<f32>` view of `t`. If `t` is already on this device
    /// we borrow zero-copy; otherwise we upload, return a borrow into a
    /// short-lived `CudaSlice` carried by the wrapper.
    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module.
    pub(crate) fn cuda_input<'a>(&self, t: &'a Tensor) -> CudaInput<'a> {
        if let Some(s) = t.device_storage() {
            if let Some(c) = s.as_any().downcast_ref::<CudaStorage>() {
                return CudaInput::Borrowed(&c.slice);
            }
            panic!("Tensor on `{}` passed to CudaBackend (only CPU + cuda are supported)",
                   s.device_name());
        }
        CudaInput::Owned(self.upload_f32(t.data()))
    }

    /// Mutable downcast — for in-place ops on a tensor we know is on this device.
    pub(crate) fn cuda_input_mut<'a>(&self, t: &'a mut Tensor) -> &'a mut CudaSlice<f32> {
        let s = t.device_storage_mut().expect(
            "mutating CUDA op called on non-device tensor; call backend.to_device(t) first",
        );
        let c = s.as_any_mut().downcast_mut::<CudaStorage>()
            .expect("tensor is on a different device");
        &mut c.slice
    }

    /// Round `n` up to the next power of 2 (with floor 1).
    fn pow2_ceil(n: usize) -> usize {
        if n <= 1 { return 1; }
        let mut p = 1usize;
        while p < n { p <<= 1; }
        p
    }

    // VENDORED-LOCAL: GPU-02 — launch the single-block busy-wait kernel for
    // `cycles` clock ticks on the compute stream. Diagnostic/test aid: keeps
    // the compute stream busy with negligible SM/HBM footprint so transfer
    // overlap can be measured without copy-engine throttling artifacts.
    pub fn spin(&self, cycles: i64) {
        let mut out = self.alloc(1);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(self.func("spin_f32"))
                .arg(&mut out)
                .arg(&cycles)
                .launch(cfg)
                .expect("spin launch");
        }
    }
}

fn pow2_ceil_u32(n: u32) -> u32 {
    if n <= 1 { return 1; }
    let mut p = 1u32;
    while p < n { p <<= 1; }
    p
}

// VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module.
pub(crate) enum CudaInput<'a> {
    Borrowed(&'a CudaSlice<f32>),
    Owned(CudaSlice<f32>),
}

impl<'a> CudaInput<'a> {
    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module's launches.
    pub(crate) fn as_ref(&self) -> &CudaSlice<f32> {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s,
        }
    }
}

enum CudaQuantInput<'a> {
    Borrowed(&'a CudaSlice<u8>),
    Owned(CudaSlice<u8>),
}

impl<'a> CudaQuantInput<'a> {
    fn as_ref(&self) -> &CudaSlice<u8> {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(s) => s,
        }
    }
}

// ----- Backend impl ---------------------------------------------------------

impl Backend for CudaBackend {
    fn streams_weights(&self) -> bool { self.weight_budget != usize::MAX }
    fn name(&self) -> &str { &self.name }

    // VENDORED-LOCAL: MOE-01 — downcast support for the grouped-MoE fast
    // path in llama-rs (see ggml-rs Backend::as_any).
    fn as_any(&self) -> &dyn Any { self }

    // VENDORED-LOCAL: MOE-01 — route on device, copy back only the compact
    // (ids, weights) mailbox instead of the full [seq, n_experts] logits.
    // Falls back to the host reference when the logits aren't device-
    // resident or the kernel's guards don't hold.
    fn moe_route_topk(&self, router_logits: &Tensor, top_k: usize) -> (Vec<u32>, Vec<f32>) {
        if let Some(r) = self.moe_route_device(router_logits, top_k) {
            return (r.ids_to_host(), r.weights_to_host());
        }
        let n_experts = router_logits.dim(router_logits.rank() - 1);
        let h = self.to_host(router_logits.clone());
        ggml_rs::backend::moe_route_topk_host(h.data(), top_k, n_experts)
    }

    // VENDORED-LOCAL: PERF-01 — explicit sync point for benchmark timing
    // boundaries (see ggml-rs Backend::synchronize).
    fn synchronize(&self) {
        self.stream.synchronize().expect("cuda synchronize");
    }

    fn vram_status(&self) -> Option<(usize, usize)> {
        // cuMemGetInfo returns free + total bytes for the CURRENT context, so on
        // a multi-GPU box it answers for whichever card this thread last touched
        // unless we bind first. Without the bind, two 5090s both reported 22 GB
        // free even though only one of them was holding the 5.97 GB trunk.
        // Cheap call (~1µs) so loaders can poll per-tensor without measurable
        // overhead.
        self.ctx.bind_to_thread().ok()?;
        cudarc::driver::result::mem_get_info().ok()
    }

    fn to_device_quant(&self, w: QuantizedTensor) -> QuantizedTensor {
        if let Some(s) = w.device_storage() {
            if s.as_any().downcast_ref::<CudaQuantStorage>().is_some() {
                return w;
            }
            panic!("to_device_quant: tensor is on `{}`, can't migrate to `{}`",
                   s.device_name(), self.name);
        }
        // A configured cap is an upper bound, not permission to exhaust VRAM.
        // Preserve 2 GiB for this model's state/temporary uploads at load time.
        if self.streams_weights() && self.vram_status().is_some_and(|(free,_)|
            free < w.nbytes().saturating_add(2usize << 30)) { return w; }
        if !reserve_weight(&self.weight_resident, self.weight_budget, w.nbytes()) { return w; }
        let shape = w.shape().to_vec();
        let dtype = w.dtype();
        // A concurrent GPU user can consume memory after the free-space check.
        // Keep the mapped packed tensor if its permanent upload no longer fits.
        let bytes = match self.stream.memcpy_stod(w.bytes()) {
            Ok(bytes) => bytes,
            Err(_) if self.streams_weights() => {
                self.weight_resident.fetch_sub(w.nbytes(), std::sync::atomic::Ordering::Relaxed);
                return w;
            }
            Err(e) => panic!("h2d quant: {e:?}"),
        };
        QuantizedTensor::from_device(Box::new(CudaQuantStorage {
            bytes, dtype, stream: self.stream.clone(), name: self.name.clone(),
        }), shape)
    }

    fn to_device(&self, t: Tensor) -> Tensor {
        if let Some(s) = t.device_storage() {
            // If already on this device, no-op; if on a different device, panic.
            if s.as_any().downcast_ref::<CudaStorage>().is_some() {
                return t;
            }
            panic!("to_device: tensor is on `{}`, can't migrate to `{}`",
                   s.device_name(), self.name);
        }
        // Was CPU — upload.
        crate::xfer::device(t.data().len() * 4);
        let slice = self.upload_f32(t.data());
        self.make_tensor(slice, t.shape().to_vec())
    }

    fn to_host(&self, t: Tensor) -> Tensor {
        if t.is_cpu() {
            return t;
        }
        // Every one of these is a synchronisation point: see `crate::xfer`.
        crate::xfer::host(t.numel() * 4);
        t.to_host()
    }

    fn embed_lookup(&self, table: &Tensor, tokens: &[u32], embedding_dim: usize) -> Tensor {
        let table_in = self.cuda_input(table);
        let tokens_dev = self.upload_u32(tokens);
        let mut out = self.alloc(tokens.len() * embedding_dim);

        let total = (tokens.len() * embedding_dim) as u32;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((total + block - 1) / block, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_tok = tokens.len() as i32;
        let d = embedding_dim as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("embed_lookup_f32"))
                .arg(table_in.as_ref())
                .arg(&tokens_dev)
                .arg(&mut out)
                .arg(&n_tok).arg(&d)
                .launch(cfg)
                .expect("embed_lookup launch");
        }
        self.make_tensor(out, vec![tokens.len(), embedding_dim])
    }

    fn last_row_to_host(&self, t: &Tensor) -> Tensor {
        let last = t.dim(t.rank() - 1);
        if let Some(s) = t.device_storage() {
            if let Some(c) = s.as_any().downcast_ref::<CudaStorage>() {
                let n = c.slice.len();
                let off = n - last;
                // Slice on device, copy that slice to host.
                let view = c.slice.slice(off..n);
                let host = self.stream.memcpy_dtov(&view).expect("d2h slice");
                return Tensor::from_vec(host, vec![last]);
            }
        }
        // Fallback: CPU last_row.
        let host = t.to_host();
        let n = host.numel();
        Tensor::from_vec(host.data()[n - last..].to_vec(), vec![last])
    }

    fn concat_to_host_flat(&self, tensors: &[&Tensor]) -> Vec<f32> {
        // Issue N async memcpy_dtoh calls into one pre-allocated host Vec, then
        // a single explicit synchronize at the end. For Qwen3.5 SSM this turns
        // 3 driver roundtrips (+ 3 internal stream syncs / Vec allocs) into 1.
        let total: usize = tensors.iter().map(|t| t.numel()).sum();
        let mut out: Vec<f32> = Vec::with_capacity(total);
        // Safety: we set_len up to capacity and then fill via memcpy_dtoh
        // before reading any element (validated by the explicit synchronize
        // before this function returns).
        unsafe { out.set_len(total); }
        let mut off = 0;
        for t in tensors {
            let n = t.numel();
            if let Some(s) = t.device_storage() {
                if let Some(c) = s.as_any().downcast_ref::<CudaStorage>() {
                    let dst = &mut out[off..off + n];
                    self.stream.memcpy_dtoh(&c.slice, dst).expect("d2h chunk");
                    off += n;
                    continue;
                }
            }
            // Fallback for already-host tensors.
            let h = t.to_host();
            out[off..off + n].copy_from_slice(h.data());
            off += n;
        }
        // One explicit sync to make sure all the queued d2h transfers have
        // landed before the host reads `out`.
        self.stream.synchronize().expect("stream sync");
        out
    }

    fn linear(&self, x: &Tensor, w: &Tensor) -> Tensor {
        // Row-major: x:[M, K], w:[N, K], y:[M, N], y[m,n] = Σ_k x[m,k] * w[n,k].
        //
        // Dispatch by problem size:
        //   * < ~100M FLOPs (tiny models, decode step): our naive 16×16 kernel
        //     wins (the line was measured against cuBLAS's per-call setup;
        //     the tiled GEMM that replaced it keeps it).
        //   * ≥ 100M FLOPs (real-world prefill on ~1B+ models): the tiled
        //     `gemm_f32` (VENDORED-LOCAL: it replaced cuBLAS Sgemm).
        //
        // The crossover was measured empirically on RTX 5090 / sm_120; we
        // expect to revisit when we add quantized matmul or change kernels.
        assert_eq!(w.rank(), 2);
        let in_ = x.dim(x.rank() - 1);
        let out = w.dim(0);
        assert_eq!(w.dim(1), in_);
        let m_rows = x.numel() / in_;

        let a = self.cuda_input(x);
        let b = self.cuda_input(w);
        let mut c = self.alloc_uninit(m_rows * out);

        // Decode-time GEMV fast path: same coop pattern as the quantized kernels,
        // but for dense F32 weights. Below the GEMM threshold the alternative
        // is the naive 16×16 kernel which only has 1 productive thread per warp
        // at M=1; the coop version uses all 32. Used by Qwen3.6 27B's F32
        // ssm_ba matmul (96×5120, 48× per token).
        // VENDORED-LOCAL: GLM-5.3-Flash. Short and wide: split the reduction, or
        // `out` warps is all the parallelism the card gets. glm5next's F32 router
        // is [4096 -> 288], 42 a token.
        if m_rows == 1 {
            if let Some(split_kernel) = Self::split_gemv_kernel(GgmlType::F32, out, in_) {
                let splits = Self::gemv_splits(GgmlType::F32, in_);
                const OUT_PER_BLOCK: u32 = 8;
                let mut partials = self.alloc_uninit(out * splits);
                let cfg = LaunchConfig {
                    grid_dim: ((out as u32).div_ceil(OUT_PER_BLOCK), splits as u32, 1),
                    block_dim: (32, OUT_PER_BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let (n_i, k_i, s_i) = (out as i32, in_ as i32, splits as i32);
                unsafe {
                    self.stream
                        .launch_builder(self.func_dyn(split_kernel))
                        .arg(a.as_ref())
                        .arg(b.as_ref())
                        .arg(&mut partials)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&s_i)
                        .launch(cfg)
                        .expect("split GEMV launch");
                }
                let rcfg = LaunchConfig::for_num_elems(out as u32);
                unsafe {
                    self.stream
                        .launch_builder(self.func("gemv_split_reduce_f32"))
                        .arg(&partials)
                        .arg(&mut c)
                        .arg(&n_i)
                        .arg(&s_i)
                        .launch(rcfg)
                        .expect("split GEMV reduce launch");
                }
                let mut shape = x.shape().to_vec();
                *shape.last_mut().unwrap() = out;
                return self.make_tensor(c, shape);
            }
        }

        if m_rows == 1 && in_ % 32 == 0 {
            const OUT_PER_BLOCK: u32 = 8;
            let block = (32u32, OUT_PER_BLOCK, 1u32);
            let cfg = LaunchConfig {
                grid_dim: ((out as u32 + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK, 1, 1),
                block_dim: block,
                shared_mem_bytes: 0,
            };
            let n_i = out as i32;
            let k_i = in_ as i32;
            unsafe {
                self.stream
                    .launch_builder(self.func("linear_f32_gemv_coop"))
                    .arg(a.as_ref())
                    .arg(b.as_ref())
                    .arg(&mut c)
                    .arg(&n_i).arg(&k_i)
                    .launch(cfg)
                    .expect("linear_f32_gemv_coop launch");
            }
            let mut shape = x.shape().to_vec();
            *shape.last_mut().unwrap() = out;
            return self.make_tensor(c, shape);
        }

        let flops = (m_rows as u64) * (out as u64) * (in_ as u64);
        if flops >= 100_000_000 {
            self.gemm(a.as_ref(), b.as_ref(), &mut c, (m_rows, out, in_), (in_, in_, out), true, (1, 0, 0, 0), 1.0);
        } else {
            let block = (16u32, 16u32, 1u32);
            let grid = (
                (out as u32 + block.0 - 1) / block.0,
                (m_rows as u32 + block.1 - 1) / block.1,
                1,
            );
            let cfg = LaunchConfig {
                grid_dim: grid,
                block_dim: block,
                shared_mem_bytes: 0,
            };
            let m_i = m_rows as i32; let n_i = out as i32; let k_i = in_ as i32;
            unsafe {
                self.stream
                    .launch_builder(self.func("linear_f32"))
                    .arg(a.as_ref())
                    .arg(b.as_ref())
                    .arg(&mut c)
                    .arg(&m_i).arg(&n_i).arg(&k_i)
                    .launch(cfg)
                    .expect("linear launch");
            }
        }

        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = out;
        self.make_tensor(c, shape)
    }

    // VENDORED-LOCAL: eliminate two output allocations, their zero fills, and
    // a separate add for each decode-time adapter. Prefill keeps the tiled GEMM.
    fn add_lora(&self, y: &mut Tensor, x: &Tensor, a: &Tensor, b: &Tensor) {
        let k = a.dim(1);
        let rank = a.dim(0);
        let n = b.dim(0);
        if x.numel() != k || rank > 1024 {
            let low = self.linear(x, a);
            let delta = self.linear(&low, b);
            self.add_inplace(y, &delta);
            return;
        }
        assert_eq!(b.shape(), &[n, rank]);
        assert_eq!(y.numel(), n);
        let input = self.cuda_input(x);
        let down = self.cuda_input(a);
        let up = self.cuda_input(b);
        let output = self.cuda_input_mut(y);
        let splits = k.div_ceil(1024).clamp(1, 16);
        let (ki, ri, ni, si) = (k as i32, rank as i32, n as i32, splits as i32);
        // SAFETY: the down kernel writes every rank*splits element, then the
        // up kernel reads it on the same ordered stream. Output is initialized
        // by the base projection; all tensors outlive both launches.
        unsafe {
            let mut partial = self.stream.alloc::<f32>(rank*splits).expect("LoRA scratch");
            self.stream.launch_builder(self.func("lora_down_f32"))
                .arg(input.as_ref()).arg(down.as_ref()).arg(&mut partial)
                .arg(&ki).arg(&ri).arg(&si)
                .launch(LaunchConfig { grid_dim: (rank as u32, splits as u32, 1), block_dim: (256,1,1), shared_mem_bytes: 0 })
                .expect("LoRA down projection");
            self.stream.launch_builder(self.func("lora_up_add_f32"))
                .arg(&partial).arg(up.as_ref()).arg(output)
                .arg(&ni).arg(&ri).arg(&si)
                .launch(LaunchConfig { grid_dim: (n.div_ceil(4) as u32,1,1), block_dim: (128,1,1), shared_mem_bytes: (rank*4) as u32 })
                .expect("LoRA up projection and add");
        }
    }

    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        let in_ = x.dim(x.rank() - 1);
        let out = w.dim(0);
        debug_assert_eq!(w.dim(1), in_);
        let m_rows = x.numel() / in_;

        // For offloaded weights, do not require the largest matrix to fit
        // in one allocation. Bound the temporary packed upload by free VRAM.
        if self.streams_weights() && w.is_cpu() && in_ % w.dtype().block_size() == 0 {
            if let Some((free, _)) = self.vram_status() {
                let scratch = (x.numel() + m_rows * out).saturating_mul(4).saturating_add(256 << 20);
                let room = free.saturating_sub(scratch) / 2;
                if w.nbytes() > room {
                    let row_bytes = in_ / w.dtype().block_size() * w.dtype().type_size();
                    let rows = (room.min(64 << 20) / row_bytes).max(1).min(out);
                    return self.linear_q_tiled(x, w, rows);
                }
            }
        }

        let kernel_name = match w.dtype() {
            GgmlType::Q8_0 => "linear_q8_0_f32",
            GgmlType::IQ4_NL => "linear_iq4_nl_f32",
            GgmlType::IQ4_XS => "linear_iq4_xs_f32",
            GgmlType::Q2_K => "linear_q2_k_f32",
            GgmlType::Q3_K => "linear_q3_k_f32",
            GgmlType::Q4_0 => "linear_q4_0_f32",
            GgmlType::Q4_1 => "linear_q4_1_f32",
            GgmlType::Q4_K => "linear_q4_k_f32",
            GgmlType::Q5_K => "linear_q5_k_f32",
            GgmlType::Q5_0 => "linear_q5_0_f32",
            GgmlType::Q5_1 => "linear_q5_1_f32",
            GgmlType::Q6_K => "linear_q6_k_f32",
            other => {
                // Unsupported quant kernel — fall back to default (dequant + linear).
                let _ = other;
                let bytes_host = if w.is_cpu() { w.bytes().to_vec() } else { w.to_host().bytes().to_vec() };
                let mut dequant = vec![0.0f32; w.numel()];
                ggml_quants::dequantize(w.dtype(), &bytes_host, &mut dequant)
                    .expect("linear_q fallback dequant");
                let w_dense = Tensor::from_vec(dequant, w.shape().to_vec());
                let w_dense = self.to_device(w_dense);
                return self.linear(x, &w_dense);
            }
        };

        let x_in = self.cuda_input(x);
        let w_in = self.cuda_quant_input(w);
        let mut c = self.alloc(m_rows * out);

        // Decode-time GEMV fast path: cooperative warp kernels for Q4_K (when
        // K % 1024 == 0) and Q6_K (when K % 256 == 0 — every K-quant model fits
        // by construction). Each warp (32 threads) computes one output element
        // with coalesced row reads and a shfl-xor reduction. Blocks pack 8 warps
        // = 256 threads, so grid is (N/8, 1, 1) with N/8 ≤ ~1500 → good SM
        // occupancy. Q6_K coverage is important because lm_head + per-layer
        // attn_v are usually Q6_K and lm_head is the largest single matmul per
        // decoded token (vocab × hidden_dim).
        if m_rows == 1 {
            let coop_kernel = match w.dtype() {
                GgmlType::Q4_K if in_ % 1024 == 0 => Some("linear_q4_k_gemv_coop_f32"),
                GgmlType::Q5_K if in_ % 1024 == 0 => Some("linear_q5_k_gemv_coop_f32"),
                GgmlType::Q6_K if in_ % 256 == 0  => Some("linear_q6_k_gemv_coop_f32"),
                GgmlType::Q5_0 if in_ % 32  == 0  => Some("linear_q5_0_gemv_coop_f32"),
                GgmlType::Q5_1 if in_ % 32  == 0  => Some("linear_q5_1_gemv_coop_f32"),
                GgmlType::Q8_0 if in_ % 32  == 0  => Some("linear_q8_0_gemv_coop_f32"),
                _ => None,
            };
            // VENDORED-LOCAL: GLM-5.3-Flash. A short, wide weight leaves the
            // coop kernels with almost no warps to run; split the reduction.
            // Only these shapes, so no other architecture's numerics move.
            if let Some(split_kernel) = Self::split_gemv_kernel(w.dtype(), out, in_) {
                let splits = Self::gemv_splits(w.dtype(), in_);
                const OUT_PER_BLOCK: u32 = 8;
                let mut partials = self.alloc(out * splits);
                let cfg = LaunchConfig {
                    grid_dim: (
                        (out as u32).div_ceil(OUT_PER_BLOCK),
                        splits as u32,
                        1,
                    ),
                    block_dim: (32, OUT_PER_BLOCK, 1),
                    shared_mem_bytes: 0,
                };
                let (n_i, k_i, s_i) = (out as i32, in_ as i32, splits as i32);
                unsafe {
                    self.stream
                        .launch_builder(self.func_dyn(split_kernel))
                        .arg(x_in.as_ref())
                        .arg(w_in.as_ref())
                        .arg(&mut partials)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&s_i)
                        .launch(cfg)
                        .expect("split GEMV launch");
                }
                let rcfg = LaunchConfig::for_num_elems(out as u32);
                unsafe {
                    self.stream
                        .launch_builder(self.func("gemv_split_reduce_f32"))
                        .arg(&partials)
                        .arg(&mut c)
                        .arg(&n_i)
                        .arg(&s_i)
                        .launch(rcfg)
                        .expect("split GEMV reduce launch");
                }
                let mut shape = x.shape().to_vec();
                *shape.last_mut().unwrap() = out;
                return self.make_tensor(c, shape);
            }
            if let Some(kernel) = coop_kernel {
                const OUT_PER_BLOCK: u32 = 8;
                let block = (32u32, OUT_PER_BLOCK, 1u32);
                let cfg = LaunchConfig {
                    grid_dim: ((out as u32 + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK, 1, 1),
                    block_dim: block,
                    shared_mem_bytes: 0,
                };
                let n_i = out as i32;
                let k_i = in_ as i32;
                unsafe {
                    self.stream
                        .launch_builder(self.func_dyn(kernel))
                        .arg(x_in.as_ref())
                        .arg(w_in.as_ref())
                        .arg(&mut c)
                        .arg(&n_i).arg(&k_i)
                        .launch(cfg)
                        .expect("coop GEMV launch");
                }
                let mut shape = x.shape().to_vec();
                *shape.last_mut().unwrap() = out;
                return self.make_tensor(c, shape);
            }
        }

        // Fallback: per-thread-one-output kernel for M>1 (prefill) and other
        // dtypes. Tried (256,1,1) for M=1 with the per-thread kernel and it
        // was slower than (16,16,1) — the extra block count compensates for
        // wasted threads. The right M=1 fix is the cooperative kernel above.
        let block = (16u32, 16u32, 1u32);
        let cfg = LaunchConfig {
            grid_dim: (
                (out as u32 + block.0 - 1) / block.0,
                (m_rows as u32 + block.1 - 1) / block.1,
                1,
            ),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let m_i = m_rows as i32;
        let n_i = out as i32;
        let k_i = in_ as i32;
        unsafe {
            self.stream
                .launch_builder(self.func_dyn(kernel_name))
                .arg(x_in.as_ref())
                .arg(w_in.as_ref())
                .arg(&mut c)
                .arg(&m_i).arg(&n_i).arg(&k_i)
                .launch(cfg)
                .expect("linear_q launch");
        }

        let mut shape = x.shape().to_vec();
        *shape.last_mut().unwrap() = out;
        self.make_tensor(c, shape)
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let x_in = self.cuda_input(x);
        let w_in = self.cuda_input(weight);
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut y = unsafe { self.stream.alloc::<f32>(x.numel()) }.expect("output allocation");

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("rmsnorm_f32"))
                .arg(x_in.as_ref()).arg(w_in.as_ref()).arg(&mut y)
                .arg(&n_rows_i).arg(&last_i).arg(&eps)
                .launch(cfg)
                .expect("rmsnorm launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn add_inplace_then_rmsnorm(&self, x: &mut Tensor, y: &Tensor, weight: &Tensor, eps: f32) -> Tensor {
        debug_assert_eq!(x.shape(), y.shape());
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        debug_assert_eq!(weight.numel(), last);

        let y_in = self.cuda_input(y);
        let w_in = self.cuda_input(weight);
        let x_dev = self.cuda_input_mut(x);
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut out = unsafe { self.stream.alloc::<f32>(n_rows * last) }.expect("output allocation");

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("add_inplace_then_rmsnorm_f32"))
                .arg(x_dev).arg(y_in.as_ref()).arg(w_in.as_ref()).arg(&mut out)
                .arg(&n_rows_i).arg(&last_i).arg(&eps)
                .launch(cfg)
                .expect("add_inplace_then_rmsnorm launch");
        }
        self.make_tensor(out, x.shape().to_vec())
    }

    fn layer_norm(&self, x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let x_in = self.cuda_input(x);
        let w_in = self.cuda_input(weight);
        let b_in = self.cuda_input(bias);
        let mut y = self.alloc(x.numel());

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("layer_norm_f32"))
                .arg(x_in.as_ref()).arg(w_in.as_ref()).arg(b_in.as_ref()).arg(&mut y)
                .arg(&n_rows_i).arg(&last_i).arg(&eps)
                .launch(cfg)
                .expect("layer_norm launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn add_inplace_broadcast_last(&self, x: &mut Tensor, bias: &Tensor) {
        debug_assert_eq!(bias.rank(), 1);
        let last = bias.numel();
        debug_assert_eq!(x.dim(x.rank() - 1), last);
        let n_rows = x.numel() / last;
        let bias_in = self.cuda_input(bias);
        let x_dev = self.cuda_input_mut(x);

        // 256 threads per block along the lane axis; multiple blocks if last > 256.
        let lane_bs: u32 = 256;
        let n_lane_blocks = ((last as u32) + lane_bs - 1) / lane_bs;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, n_lane_blocks, 1),
            block_dim: (lane_bs, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("add_bias_last_f32"))
                .arg(x_dev).arg(bias_in.as_ref())
                .arg(&n_rows_i).arg(&last_i)
                .launch(cfg)
                .expect("add_bias_last launch");
        }
    }

    fn rmsnorm_no_scale(&self, x: &Tensor, eps: f32) -> Tensor {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let x_in = self.cuda_input(x);
        let mut y = self.alloc(x.numel());

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("rmsnorm_no_scale_f32"))
                .arg(x_in.as_ref()).arg(&mut y)
                .arg(&n_rows_i).arg(&last_i).arg(&eps)
                .launch(cfg)
                .expect("rmsnorm_no_scale launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn softmax_last(&self, x: &mut Tensor) {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;

        // x must already be device-resident (caller's job).
        let x_dev = self.cuda_input_mut(x);

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("softmax_last_f32"))
                .arg(x_dev)
                .arg(&n_rows_i).arg(&last_i)
                .launch(cfg)
                .expect("softmax launch");
        }
    }

    fn silu(&self, x: &Tensor) -> Tensor {
        let n = x.numel();
        let x_in = self.cuda_input(x);
        let mut y = self.alloc_uninit(n);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("silu_f32"))
                .arg(x_in.as_ref()).arg(&mut y).arg(&n_i)
                .launch(cfg)
                .expect("silu launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn silu_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        debug_assert_eq!(a.shape(), b.shape());
        let n = a.numel();
        let a_in = self.cuda_input(a);
        let b_in = self.cuda_input(b);
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut y = unsafe { self.stream.alloc::<f32>(n) }.expect("output allocation");
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("silu_mul_f32"))
                .arg(a_in.as_ref()).arg(b_in.as_ref()).arg(&mut y).arg(&n_i)
                .launch(cfg)
                .expect("silu_mul launch");
        }
        self.make_tensor(y, a.shape().to_vec())
    }

    fn sigmoid(&self, x: &Tensor) -> Tensor {
        let n = x.numel();
        let x_in = self.cuda_input(x);
        let mut y = self.alloc_uninit(n);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("sigmoid_f32"))
                .arg(x_in.as_ref()).arg(&mut y).arg(&n_i)
                .launch(cfg)
                .expect("sigmoid launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn mul_sigmoid_inplace(&self, x: &mut Tensor, gate: &Tensor) {
        debug_assert_eq!(x.shape(), gate.shape());
        let n = x.numel();
        let g_in = self.cuda_input(gate);
        let x_dev = self.cuda_input_mut(x);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("mul_sigmoid_inplace_f32"))
                .arg(x_dev).arg(g_in.as_ref()).arg(&n_i)
                .launch(cfg)
                .expect("mul_sigmoid_inplace launch");
        }
    }

    fn gelu_approx(&self, x: &Tensor) -> Tensor {
        let n = x.numel();
        let x_in = self.cuda_input(x);
        let mut y = self.alloc(n);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("gelu_approx_f32"))
                .arg(x_in.as_ref()).arg(&mut y).arg(&n_i)
                .launch(cfg)
                .expect("gelu launch");
        }
        self.make_tensor(y, x.shape().to_vec())
    }

    fn gelu_approx_mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        debug_assert_eq!(a.shape(), b.shape());
        let n = a.numel();
        let a_in = self.cuda_input(a);
        let b_in = self.cuda_input(b);
        let mut y = self.alloc(n);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("gelu_approx_mul_f32"))
                .arg(a_in.as_ref()).arg(b_in.as_ref()).arg(&mut y).arg(&n_i)
                .launch(cfg)
                .expect("gelu_approx_mul launch");
        }
        self.make_tensor(y, a.shape().to_vec())
    }

    // VENDORED-LOCAL: GLM-5.3-Flash clamped SwiGLU.
    fn swiglu_clamped(&self, gate: &Tensor, up: &Tensor, limit: f32, after_silu: bool) -> Tensor {
        let n = gate.numel();
        debug_assert_eq!(n, up.numel());
        let g_in = self.cuda_input(gate);
        let u_in = self.cuda_input(up);
        let mut out = self.alloc(n);

        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (((n as u32) + block - 1) / block, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let after = i32::from(after_silu);
        unsafe {
            self.stream
                .launch_builder(self.func("swiglu_clamped_f32"))
                .arg(g_in.as_ref())
                .arg(u_in.as_ref())
                .arg(&mut out)
                .arg(&n_i)
                .arg(&limit)
                .arg(&after)
                .launch(cfg)
                .expect("swiglu_clamped launch");
        }
        self.make_tensor(out, gate.shape().to_vec())
    }

    // VENDORED-LOCAL: GLM-5.3-Flash clamped SwiGLU, fused layout.
    fn swiglu_clamped_split(
        &self,
        fused: &Tensor,
        ff: usize,
        limit: f32,
        after_silu: bool,
    ) -> Tensor {
        let seq = fused.numel() / (2 * ff);
        let f_in = self.cuda_input(fused);
        let mut out = self.alloc(seq * ff);

        let block_x = 256u32.min(ff as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (((ff as u32) + block_x - 1) / block_x, seq as u32, 1),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let ff_i = ff as i32;
        let after = i32::from(after_silu);
        unsafe {
            self.stream
                .launch_builder(self.func("swiglu_clamped_split_f32"))
                .arg(f_in.as_ref())
                .arg(&mut out)
                .arg(&seq_i)
                .arg(&ff_i)
                .arg(&limit)
                .arg(&after)
                .launch(cfg)
                .expect("swiglu_clamped_split launch");
        }
        self.make_tensor(out, vec![seq, ff])
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. A per-head stack of matvecs.
    fn batched_gemv(&self, w: &Tensor, x: &Tensor, b: usize, m: usize, k: usize) -> Tensor {
        let w_in = self.cuda_input(w);
        let x_in = self.cuda_input(x);
        let mut out = self.alloc(b * m);

        // 256 threads is 8 warps, so one block covers 8 output rows.
        const BLOCK: u32 = 256;
        let warps = BLOCK / 32;
        let cfg = LaunchConfig {
            grid_dim: ((m as u32).div_ceil(warps), b as u32, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (bi, mi, ki) = (b as i32, m as i32, k as i32);
        unsafe {
            self.stream
                .launch_builder(self.func("batched_gemv_f32"))
                .arg(w_in.as_ref())
                .arg(x_in.as_ref())
                .arg(&mut out)
                .arg(&bi)
                .arg(&mi)
                .arg(&ki)
                .launch(cfg)
                .expect("batched_gemv launch");
        }
        self.make_tensor(out, vec![b, m])
    }

    // VENDORED-LOCAL: GLM-5.3-Flash. One KDA delta-rule step, state in place.
    fn kda_delta_step(
        &self,
        state: &mut Tensor,
        qkvg: &Tensor,
        beta: &Tensor,
        n_head: usize,
        head_dim: usize,
    ) -> Tensor {
        let n = n_head * head_dim;
        debug_assert_eq!(state.numel(), n * head_dim);
        debug_assert_eq!(qkvg.numel(), 4 * n);
        let qk_in = self.cuda_input(qkvg);
        let b_in = self.cuda_input(beta);
        let mut out = self.alloc(n);
        let st = self.cuda_input_mut(state);

        const BLOCK: u32 = 256;
        let warps = BLOCK / 32;
        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(warps), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (nh, hd) = (n_head as i32, head_dim as i32);
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        unsafe {
            self.stream
                .launch_builder(self.func("kda_delta_step_f32"))
                .arg(st)
                .arg(qk_in.as_ref())
                .arg(b_in.as_ref())
                .arg(&mut out)
                .arg(&nh)
                .arg(&hd)
                .arg(&scale)
                .launch(cfg)
                .expect("kda_delta_step launch");
        }
        self.make_tensor(out, vec![n])
    }

    fn silu_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        let seq = fused.dim(0);
        debug_assert_eq!(fused.dim(fused.rank() - 1), 2 * ff);
        let f_in = self.cuda_input(fused);
        let mut out = self.alloc(seq * ff);

        let block_x = 256u32.min(ff as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((ff as u32) + block_x - 1) / block_x,
                seq as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let ff_i = ff as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("silu_mul_split_f32"))
                .arg(f_in.as_ref()).arg(&mut out)
                .arg(&seq_i).arg(&ff_i)
                .launch(cfg)
                .expect("silu_mul_split launch");
        }
        self.make_tensor(out, vec![seq, ff])
    }

    fn gelu_approx_mul_split(&self, fused: &Tensor, ff: usize) -> Tensor {
        let seq = fused.dim(0);
        debug_assert_eq!(fused.dim(fused.rank() - 1), 2 * ff);
        let f_in = self.cuda_input(fused);
        let mut out = self.alloc(seq * ff);

        let block_x = 256u32.min(ff as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((ff as u32) + block_x - 1) / block_x,
                seq as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let ff_i = ff as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("gelu_approx_mul_split_f32"))
                .arg(f_in.as_ref()).arg(&mut out)
                .arg(&seq_i).arg(&ff_i)
                .launch(cfg)
                .expect("gelu_approx_mul_split launch");
        }
        self.make_tensor(out, vec![seq, ff])
    }

    fn tanh_inplace(&self, x: &mut Tensor) {
        let n = x.numel();
        let x_dev = self.cuda_input_mut(x);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("tanh_inplace_f32"))
                .arg(x_dev).arg(&n_i)
                .launch(cfg)
                .expect("tanh launch");
        }
    }

    fn gaussian_topk_inplace(&self, x: &mut Tensor, std_multiplier: f32) {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let x_dev = self.cuda_input_mut(x);

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 4,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("gaussian_topk_inplace_f32"))
                .arg(x_dev).arg(&n_rows_i).arg(&last_i).arg(&std_multiplier)
                .launch(cfg)
                .expect("gaussian_topk launch");
        }
    }

    fn add_inplace(&self, x: &mut Tensor, y: &Tensor) {
        assert_eq!(x.shape(), y.shape());
        let n = x.numel();
        let y_in = self.cuda_input(y);
        let x_dev = self.cuda_input_mut(x);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("add_inplace_f32"))
                .arg(x_dev).arg(y_in.as_ref()).arg(&n_i)
                .launch(cfg)
                .expect("add launch");
        }
    }

    fn mul_inplace(&self, x: &mut Tensor, y: &Tensor) {
        assert_eq!(x.shape(), y.shape());
        let n = x.numel();
        let y_in = self.cuda_input(y);
        let x_dev = self.cuda_input_mut(x);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("mul_inplace_f32"))
                .arg(x_dev).arg(y_in.as_ref()).arg(&n_i)
                .launch(cfg)
                .expect("mul launch");
        }
    }

    fn mul_scalar_inplace(&self, x: &mut Tensor, s: f32) {
        let n = x.numel();
        let x_dev = self.cuda_input_mut(x);
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("mul_scalar_inplace_f32"))
                .arg(x_dev).arg(&s).arg(&n_i)
                .launch(cfg)
                .expect("mul_scalar launch");
        }
    }

    fn mul_inplace_broadcast_last(&self, x: &mut Tensor, w: &Tensor) {
        debug_assert_eq!(w.rank(), 1);
        let last = w.numel();
        debug_assert_eq!(x.dim(x.rank() - 1), last);
        let n_rows = x.numel() / last;

        let w_in = self.cuda_input(w);
        let x_dev = self.cuda_input_mut(x);

        let block_x = 256u32.min(last as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((last as u32) + block_x - 1) / block_x,
                n_rows as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("mul_inplace_broadcast_last_f32"))
                .arg(x_dev).arg(w_in.as_ref())
                .arg(&n_rows_i).arg(&last_i)
                .launch(cfg)
                .expect("mul_broadcast_last launch");
        }
    }

    fn mul_inplace_broadcast_axis0(&self, x: &mut Tensor, g: &Tensor) {
        let seq = x.dim(0);
        debug_assert_eq!(g.numel(), seq, "gate must have one entry per row");
        let inner: usize = x.shape().iter().skip(1).product::<usize>().max(1);

        let g_in = self.cuda_input(g);
        let x_dev = self.cuda_input_mut(x);

        let block_x = 256u32.min(inner as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((inner as u32) + block_x - 1) / block_x,
                seq as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let inner_i = inner as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("mul_inplace_broadcast_axis0_f32"))
                .arg(x_dev).arg(g_in.as_ref())
                .arg(&seq_i).arg(&inner_i)
                .launch(cfg)
                .expect("mul_inplace_broadcast_axis0 launch");
        }
    }

    fn rope_partial_neox(
        &self,
        x: &mut Tensor,
        positions: &[u32],
        head_dim: usize,
        rotated_dim: usize,
        theta: f32,
    ) {
        assert_eq!(x.rank(), 3);
        assert_eq!(x.dim(2), head_dim);
        assert!(rotated_dim <= head_dim && rotated_dim % 2 == 0);
        let seq = x.dim(0);
        let n_h = x.dim(1);
        assert_eq!(positions.len(), seq);

        let pos_dev = self.cached_positions(positions);
        let x_dev = self.cuda_input_mut(x);

        let half = (rotated_dim / 2) as u32;
        let block_x = 32u32.min(half).max(1);
        let cfg = LaunchConfig {
            grid_dim: ((half + block_x - 1) / block_x, n_h as u32, seq as u32),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let n_h_i = n_h as i32;
        let hd_i = head_dim as i32;
        let rot_i = rotated_dim as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("rope_partial_neox_f32"))
                .arg(x_dev).arg(pos_dev.as_ref())
                .arg(&seq_i).arg(&n_h_i).arg(&hd_i)
                .arg(&rot_i).arg(&theta)
                .launch(cfg)
                .expect("rope_partial_neox launch");
        }
    }

    // VENDORED-LOCAL: image positions stay on the GPU for vision and LM prefill.
    fn rope_axes(&self, x: &mut Tensor, positions: &[[u32;3]], rotated: usize,
        theta: f32, axes: &[usize], frequencies: &[usize], frequency_dim: usize) {
        assert_eq!(x.rank(),3);
        assert_eq!(positions.len(),x.dim(0));
        assert_eq!(axes.len(),rotated/2);
        assert_eq!(axes.len(),frequencies.len());
        assert!(rotated<=x.dim(2) && rotated%2==0 && frequency_dim>0);
        assert!(axes.iter().all(|&a|a<3));
        if x.numel()==0 || rotated==0 { return; }
        let pos=self.upload_u32(&positions.iter().flatten().copied().collect::<Vec<_>>());
        let map=self.upload_u32(&axes.iter().zip(frequencies).flat_map(|(&a,&f)|[a as u32,f as u32]).collect::<Vec<_>>());
        let (heads,width,half)=(x.dim(1) as i32,x.dim(2) as i32,(rotated/2) as i32);
        let count=(x.dim(0)*x.dim(1)*(rotated/2)) as i32;
        let freq_dim=frequency_dim as f32;
        let xd=self.cuda_input_mut(x);
        // SAFETY: each thread owns one disjoint rotation pair. Position/map
        // lengths and axes are validated above; buffers live on this stream.
        unsafe {
            self.stream.launch_builder(self.func("rope_axes_f32"))
                .arg(xd).arg(&pos).arg(&map).arg(&count).arg(&heads).arg(&width)
                .arg(&half).arg(&theta).arg(&freq_dim)
                .launch(LaunchConfig::for_num_elems(count as u32)).expect("axial rope");
        }
    }

    fn delta_net_step(
        &self,
        mixed_qkv:   &Tensor,
        z_in:        &Tensor,
        beta_alpha:  &Tensor,
        conv_weight: &Tensor,
        ssm_a:       &Tensor,
        dt_bias:     &Tensor,
        ssm_norm:    &Tensor,
        conv_state:  &mut Tensor,
        state:       &mut Tensor,
        seq:         usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_v_dim:  usize,
        head_k_dim:  usize,
        v_per_k:     usize,
        scale_q:     f32,
        eps:         f32,
    ) -> Tensor {
        self.delta_net_inner(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps, 0)
    }

    fn delta_net_step_sigmoid(
        &self,
        mixed_qkv:   &Tensor,
        z_in:        &Tensor,
        beta_alpha:  &Tensor,
        conv_weight: &Tensor,
        ssm_a:       &Tensor,
        dt_bias:     &Tensor,
        ssm_norm:    &Tensor,
        conv_state:  &mut Tensor,
        state:       &mut Tensor,
        seq:         usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_v_dim:  usize,
        head_k_dim:  usize,
        v_per_k:     usize,
        scale_q:     f32,
        eps:         f32,
    ) -> Tensor {
        self.delta_net_inner(mixed_qkv, z_in, beta_alpha, conv_weight, ssm_a, dt_bias, ssm_norm, conv_state, state, seq, num_v_heads, num_k_heads, head_v_dim, head_k_dim, v_per_k, scale_q, eps, 1)
    }

    fn qsa_pool(&self, raw: &Tensor, blocks: usize, ratio: usize) -> Tensor {
        let d = raw.numel() / raw.dim(0).max(1);
        let raw_in = self.cuda_input(raw);
        let mut out = self.alloc_uninit(blocks * d);
        if blocks > 0 {
            let (b_i, d_i, r_i) = (blocks as i32, d as i32, ratio as i32);
            unsafe {
                self.stream.launch_builder(self.func("qsa_pool_f32"))
                    .arg(raw_in.as_ref()).arg(&mut out).arg(&b_i).arg(&d_i).arg(&r_i)
                    .launch(LaunchConfig::for_num_elems((blocks * d) as u32)).expect("qsa_pool launch");
            }
        }
        self.make_tensor(out, vec![blocks, d])
    }

    fn qsa_block_scores(&self, q: &Tensor, pooled: &Tensor, first: usize, ratio: usize, scale: f32) -> Tensor {
        let rows = q.dim(0);
        let nb = pooled.dim(0);
        let d = pooled.numel() / nb.max(1);
        let heads = q.numel() / rows / d;
        let q_in = self.cuda_input(q);
        let p_in = self.cuda_input(pooled);
        let mut out = self.alloc_uninit(rows * nb);
        let (r_i, h_i, d_i, n_i, f_i, c_i) = (rows as i32, heads as i32, d as i32, nb as i32, first as i32, ratio as i32);
        unsafe {
            self.stream.launch_builder(self.func("qsa_block_scores_f32"))
                .arg(q_in.as_ref()).arg(p_in.as_ref()).arg(&mut out)
                .arg(&r_i).arg(&h_i).arg(&d_i).arg(&n_i).arg(&f_i).arg(&c_i).arg(&scale)
                .launch(LaunchConfig { grid_dim: (((rows * nb) as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("qsa_block_scores launch");
        }
        self.make_tensor(out, vec![rows, nb])
    }

    fn qsa_select(&self, scores: &Tensor, first: usize, ratio: usize, keep: usize) -> Tensor {
        let n = scores.dim(0);
        let nb = scores.numel() / n.max(1);
        let width = keep * ratio + ratio;
        let s_in = self.cuda_input(scores);
        let mut out = self.alloc_uninit(n * width);
        let (nb_i, f_i, r_i, k_i, w_i) = (nb as i32, first as i32, ratio as i32, keep as i32, width as i32);
        unsafe {
            self.stream.launch_builder(self.func("qsa_select_f32"))
                .arg(s_in.as_ref()).arg(&mut out).arg(&nb_i).arg(&f_i).arg(&r_i).arg(&k_i).arg(&w_i)
                .launch(LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 })
                .expect("qsa_select launch");
        }
        self.make_tensor(out, vec![n, width])
    }

    fn sparse_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, sel: &Tensor, scale: f32) -> Tensor {
        let width = sel.numel() / q.dim(0).max(1);
        self.split_attention(q, k, v, Some(sel), width, scale, 0)
    }

    fn gather_rows(&self, src: &Tensor, rows: &[u32]) -> Tensor {
        let d = src.numel() / src.dim(0).max(1);
        let src_in = self.cuda_input(src);
        let rows_dev = self.upload_u32(rows);
        let mut out = self.alloc_uninit(rows.len() * d);
        let n = (rows.len() * d) as u32;
        if n > 0 {
            let (m_i, d_i) = (rows.len() as i32, d as i32);
            unsafe {
                self.stream.launch_builder(self.func("gather_rows_f32"))
                    .arg(src_in.as_ref()).arg(&rows_dev).arg(&mut out).arg(&m_i).arg(&d_i)
                    .launch(LaunchConfig::for_num_elems(n)).expect("gather_rows launch");
            }
        }
        self.make_tensor(out, vec![rows.len(), d])
    }

    fn scatter_add_rows(&self, dst: &mut Tensor, src: &Tensor, rows: &[u32], weights: &[f32]) {
        let d = src.numel() / src.dim(0).max(1);
        let n = (rows.len() * d) as u32;
        if n == 0 { return; }
        let src_in = self.cuda_input(src);
        let rows_dev = self.upload_u32(rows);
        let w_dev = self.upload_f32(weights);
        let dst_dev = self.cuda_input_mut(dst);
        let (m_i, d_i) = (rows.len() as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("scatter_add_rows_f32"))
                .arg(dst_dev).arg(src_in.as_ref()).arg(&rows_dev).arg(&w_dev).arg(&m_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems(n)).expect("scatter_add_rows launch");
        }
    }

    fn split_cols(&self, x: &Tensor, a: usize) -> (Tensor, Tensor) {
        let rows = x.dim(0);
        let width = x.numel() / rows;
        let x_in = self.cuda_input(x);
        let mut l = self.alloc_uninit(rows * a);
        let mut r = self.alloc_uninit(rows * (width - a));
        let (r_i, w_i, a_i) = (rows as i32, width as i32, a as i32);
        unsafe {
            self.stream.launch_builder(self.func("split_cols_f32"))
                .arg(x_in.as_ref()).arg(&mut l).arg(&mut r).arg(&r_i).arg(&w_i).arg(&a_i)
                .launch(LaunchConfig::for_num_elems((rows * width) as u32)).expect("split_cols launch");
        }
        (self.make_tensor(l, vec![rows, a]), self.make_tensor(r, vec![rows, width - a]))
    }

    fn hc_apply_norm(&self, x: &mut Tensor, y: &Tensor, post: &Tensor, weight: &Tensor, streams: usize, eps: f32) -> Tensor {
        let rows = x.dim(0);
        let d = x.numel() / rows / streams;
        let y_in = self.cuda_input(y);
        let p_in = self.cuda_input(post);
        let w_in = self.cuda_input(weight);
        let mut out = self.alloc_uninit(rows * streams * d);
        let shape = x.shape().to_vec();
        let x_dev = self.cuda_input_mut(x);
        let (d_i, s_i) = (d as i32, streams as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_apply_norm_f32"))
                .arg(x_dev).arg(y_in.as_ref()).arg(p_in.as_ref()).arg(w_in.as_ref()).arg(&mut out).arg(&d_i).arg(&s_i).arg(&eps)
                .launch(LaunchConfig { grid_dim: ((rows * streams) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("hc_apply_norm launch");
        }
        self.make_tensor(out, shape)
    }

    fn hc_down_gates(&self, normed: &Tensor, down: &Tensor, rank: usize, writes: usize, streams: usize) -> (Tensor, Tensor) {
        let rows = normed.dim(0);
        // The fused kernel reads the weights once a row: a GEMM for a prompt's many rows.
        if rows > 8 {
            let mut t = self.linear(normed, &self.unpack_f16(down));
            let post = self.hc_gates(&mut t, rank, writes, streams);
            return (t, post);
        }
        let width = normed.numel() / rows;
        let cols = rank + writes;
        let n_in = self.cuda_input(normed);
        let w_in = self.cuda_input(down);
        let mut t = self.alloc_uninit(rows * cols);
        let mut post = self.alloc_uninit((rows * writes).max(1));
        let (wd, rk, wr, st) = (width as i32, rank as i32, writes as i32, streams as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_down_gates_f32"))
                .arg(n_in.as_ref()).arg(w_in.as_ref()).arg(&mut t).arg(&mut post).arg(&wd).arg(&rk).arg(&wr).arg(&st)
                .launch(LaunchConfig { grid_dim: (cols as u32, rows as u32, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 })
                .expect("hc_down_gates launch");
        }
        (self.make_tensor(t, vec![rows, cols]), self.make_tensor(post, vec![rows, writes.max(1)]))
    }

    fn unpack_f16(&self, packed: &Tensor) -> Tensor {
        let rows = packed.dim(0);
        let words = packed.numel();
        let p_in = self.cuda_input(packed);
        let mut out = self.alloc_uninit(2 * words);
        let w_i = words as i64;
        unsafe {
            self.stream.launch_builder(self.func("unpack_f16_f32"))
                .arg(p_in.as_ref()).arg(&mut out).arg(&w_i)
                .launch(LaunchConfig::for_num_elems(words as u32)).expect("unpack_f16 launch");
        }
        self.make_tensor(out, vec![rows, 2 * words / rows.max(1)])
    }

    fn hc_up_mix(&self, t: &Tensor, up: &Tensor, normed: &Tensor, streams: usize) -> Tensor {
        let rows = normed.dim(0);
        if rows > 8 {
            let logits = self.linear(t, &self.unpack_f16(up));
            return self.hc_mix(&logits, normed, streams);
        }
        let d = normed.numel() / rows / streams;
        let cols = t.numel() / rows;
        let t_in = self.cuda_input(t);
        let u_in = self.cuda_input(up);
        let n_in = self.cuda_input(normed);
        let mut out = self.alloc_uninit(rows * d);
        let (r_i, s_i, d_i, c_i) = (rows as i32, streams as i32, d as i32, cols as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_up_mix_f32"))
                .arg(t_in.as_ref()).arg(u_in.as_ref()).arg(n_in.as_ref()).arg(&mut out).arg(&r_i).arg(&s_i).arg(&d_i).arg(&c_i)
                .launch(LaunchConfig { grid_dim: (((rows * d) as u32).div_ceil(8), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("hc_up_mix launch");
        }
        self.make_tensor(out, vec![rows, d])
    }

    fn hc_norm(&self, x: &Tensor, weight: &Tensor, streams: usize, eps: f32) -> Tensor {
        let rows = x.dim(0);
        let d = x.numel() / rows / streams;
        let x_in = self.cuda_input(x);
        let w_in = self.cuda_input(weight);
        let mut out = self.alloc_uninit(rows * streams * d);
        let (d_i, s_i) = (d as i32, streams as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_norm_f32"))
                .arg(x_in.as_ref()).arg(w_in.as_ref()).arg(&mut out).arg(&d_i).arg(&s_i).arg(&eps)
                .launch(LaunchConfig { grid_dim: ((rows * streams) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("hc_norm launch");
        }
        self.make_tensor(out, x.shape().to_vec())
    }

    fn hc_gates(&self, t: &mut Tensor, rank: usize, writes: usize, streams: usize) -> Tensor {
        let rows = t.dim(0);
        let mut post = self.alloc((rows * writes).max(1));
        let t_dev = self.cuda_input_mut(t);
        let (r_i, k_i, w_i, s_i) = (rows as i32, rank as i32, writes as i32, streams as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_gates_f32"))
                .arg(t_dev).arg(&mut post).arg(&r_i).arg(&k_i).arg(&w_i).arg(&s_i)
                .launch(LaunchConfig::for_num_elems((rows * (rank + writes)) as u32)).expect("hc_gates launch");
        }
        self.make_tensor(post, vec![rows, writes.max(1)])
    }

    fn hc_mix(&self, logits: &Tensor, normed: &Tensor, streams: usize) -> Tensor {
        let rows = normed.dim(0);
        let d = normed.numel() / rows / streams;
        let l_in = self.cuda_input(logits);
        let n_in = self.cuda_input(normed);
        let mut out = self.alloc_uninit(rows * d);
        let (r_i, s_i, d_i) = (rows as i32, streams as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("hc_mix_f32"))
                .arg(l_in.as_ref()).arg(n_in.as_ref()).arg(&mut out).arg(&r_i).arg(&s_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems((rows * d) as u32)).expect("hc_mix launch");
        }
        self.make_tensor(out, vec![rows, d])
    }

    fn add_rows_scaled(&self, dst: &mut Tensor, src: &Tensor, scale: &Tensor) {
        let rows = src.dim(0);
        let d = src.numel() / rows.max(1);
        let n = (rows * d) as u32;
        if n == 0 { return; }
        let src_in = self.cuda_input(src);
        let g_in = self.cuda_input(scale);
        let dst_dev = self.cuda_input_mut(dst);
        let (r_i, d_i) = (rows as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("add_rows_scaled_f32"))
                .arg(dst_dev).arg(src_in.as_ref()).arg(g_in.as_ref()).arg(&r_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems(n)).expect("add_rows_scaled launch");
        }
    }

    fn stream_mean(&self, x: &Tensor, streams: usize) -> Tensor {
        let rows = x.dim(0);
        let d = x.numel() / rows / streams;
        let x_in = self.cuda_input(x);
        let mut out = self.alloc_uninit(rows * d);
        let (r_i, h_i, d_i) = (rows as i32, streams as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("stream_mean_f32"))
                .arg(x_in.as_ref()).arg(&mut out).arg(&r_i).arg(&h_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems((rows * d) as u32)).expect("stream_mean launch");
        }
        self.make_tensor(out, vec![rows, d])
    }

    fn ple_gate(&self, key: &Tensor, x: &Tensor, value: &Tensor, norm_key: &Tensor, norm_query: &Tensor, norm_conv: &Tensor, streams: usize, eps: f32) -> (Tensor, Tensor) {
        let rows = value.dim(0);
        let d = value.numel() / rows;
        let width = streams * d;
        let (k_in, x_in, v_in) = (self.cuda_input(key), self.cuda_input(x), self.cuda_input(value));
        let (nk, nq, nc) = (self.cuda_input(norm_key), self.cuda_input(norm_query), self.cuda_input(norm_conv));
        let mut gated = self.alloc_uninit(rows * width);
        let mut conv_in = self.alloc_uninit(rows * width);
        let (s_i, d_i) = (streams as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("ple_gate_f32"))
                .arg(k_in.as_ref()).arg(x_in.as_ref()).arg(v_in.as_ref()).arg(nk.as_ref()).arg(nq.as_ref()).arg(nc.as_ref())
                .arg(&mut gated).arg(&mut conv_in).arg(&s_i).arg(&d_i).arg(&eps)
                .launch(LaunchConfig { grid_dim: (rows as u32, streams as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                .expect("ple_gate launch");
        }
        (self.make_tensor(gated, vec![rows, width]), self.make_tensor(conv_in, vec![rows, width]))
    }

    fn ple_conv(&self, x: &mut Tensor, gated: &Tensor, conv_in: &Tensor, window: &mut Tensor, weight: &Tensor, kernel: usize, dilation: usize) {
        let rows = gated.dim(0);
        let width = gated.numel() / rows;
        let (g_in, c_in, wt) = (self.cuda_input(gated), self.cuda_input(conv_in), self.cuda_input(weight));
        let (r_i, w_i, k_i, d_i) = (rows as i32, width as i32, kernel as i32, dilation as i32);
        let win = self.cuda_input_mut(window) as *mut CudaSlice<f32>;
        let x_dev = self.cuda_input_mut(x);
        // SAFETY: `window` and `x` are distinct tensors; the pointer only splits the borrows.
        let win = unsafe { &mut *win };
        unsafe {
            self.stream.launch_builder(self.func("ple_conv_f32"))
                .arg(x_dev).arg(g_in.as_ref()).arg(c_in.as_ref()).arg(win).arg(wt.as_ref())
                .arg(&r_i).arg(&w_i).arg(&k_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems(width as u32)).expect("ple_conv launch");
        }
    }

    fn stream_apply(&self, x: &mut Tensor, y: &Tensor, post: &Tensor, streams: usize) {
        let rows = y.dim(0);
        let d = y.numel() / rows;
        let y_in = self.cuda_input(y);
        let p_in = self.cuda_input(post);
        let x_dev = self.cuda_input_mut(x);
        let (r_i, h_i, d_i) = (rows as i32, streams as i32, d as i32);
        unsafe {
            self.stream.launch_builder(self.func("stream_apply_f32"))
                .arg(x_dev).arg(y_in.as_ref()).arg(p_in.as_ref()).arg(&r_i).arg(&h_i).arg(&d_i)
                .launch(LaunchConfig::for_num_elems((rows * streams * d) as u32)).expect("stream_apply launch");
        }
    }

    fn split_qkv_3way(&self, qkv: &Tensor, d: usize) -> (Tensor, Tensor, Tensor) {
        let seq = qkv.dim(0);
        debug_assert_eq!(qkv.numel(), seq * 3 * d);
        let qkv_in = self.cuda_input(qkv);
        let mut q = self.alloc(seq * d);
        let mut k = self.alloc(seq * d);
        let mut v = self.alloc(seq * d);
        let total = (seq * d) as u32;
        let block: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((total + block - 1) / block, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let d_i = d as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("split_qkv_3way_f32"))
                .arg(qkv_in.as_ref())
                .arg(&mut q).arg(&mut k).arg(&mut v)
                .arg(&seq_i).arg(&d_i)
                .launch(cfg)
                .expect("split_qkv_3way launch");
        }
        (
            self.make_tensor(q, vec![seq, d]),
            self.make_tensor(k, vec![seq, d]),
            self.make_tensor(v, vec![seq, d]),
        )
    }

    fn split_q_and_gate(
        &self,
        q_full:  &Tensor,
        n_heads: usize,
        head_dim: usize,
    ) -> (Tensor, Tensor) {
        let seq = q_full.dim(0);
        debug_assert_eq!(q_full.numel(), seq * n_heads * 2 * head_dim);
        let q_in = self.cuda_input(q_full);
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut q_only = unsafe { self.stream.alloc::<f32>(seq * n_heads * head_dim) }.expect("output allocation");
        // VENDORED-LOCAL: SAFETY: the kernel below overwrites every output
        // element on this stream before it is exposed to a consumer.
        let mut q_gate = unsafe { self.stream.alloc::<f32>(seq * n_heads * head_dim) }.expect("output allocation");

        let total = (seq * n_heads * head_dim) as u32;
        let block: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: ((total + block - 1) / block, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let nh_i  = n_heads as i32;
        let hd_i  = head_dim as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("split_q_and_gate_f32"))
                .arg(q_in.as_ref())
                .arg(&mut q_only)
                .arg(&mut q_gate)
                .arg(&seq_i).arg(&nh_i).arg(&hd_i)
                .launch(cfg)
                .expect("split_q_and_gate launch");
        }
        (
            self.make_tensor(q_only, vec![seq, n_heads * head_dim]),
            self.make_tensor(q_gate, vec![seq, n_heads * head_dim]),
        )
    }

    fn rope(
        &self,
        x: &mut Tensor,
        positions: &[u32],
        head_dim: usize,
        rope_type: RopeType,
        theta: f32,
        freq_factors: Option<&[f32]>,
    ) {
        assert_eq!(x.rank(), 3);
        let seq = x.dim(0);
        let n_h = x.dim(1);
        assert_eq!(x.dim(2), head_dim);
        assert_eq!(positions.len(), seq);

        let pos_dev = self.upload_u32(positions);
        // Always upload a head_dim/2-long buffer: real factors or all-ones (no-op
        // divide). cudarc doesn't expose a null device pointer for slice args,
        // so the unified-buffer approach is simpler than threading a flag.
        let ff_buf = match freq_factors {
            Some(f) => {
                assert!(f.len() >= head_dim / 2, "freq_factors len {} < head_dim/2 {}", f.len(), head_dim / 2);
                self.stream.memcpy_stod(&f[..head_dim / 2]).expect("upload freq_factors")
            }
            None => self.stream.memcpy_stod(&vec![1.0f32; head_dim / 2]).expect("upload ones freq_factors"),
        };
        let x_dev = self.cuda_input_mut(x);

        let half = head_dim / 2;
        let block_x = 32u32.min(half as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((half as u32) + block_x - 1) / block_x,
                n_h as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let n_h_i = n_h as i32;
        let hd_i = head_dim as i32;
        let rt_i: i32 = match rope_type { RopeType::Normal => 0, RopeType::NeoX => 1 };
        unsafe {
            self.stream
                .launch_builder(self.func("rope_f32"))
                .arg(x_dev).arg(&pos_dev)
                .arg(&seq_i).arg(&n_h_i).arg(&hd_i)
                .arg(&rt_i).arg(&theta)
                .arg(&ff_buf)
                .launch(cfg)
                .expect("rope launch");
        }
    }

    fn repeat_kv(&self, x: &Tensor, n_rep: usize) -> Tensor {
        if n_rep == 1 {
            // Need to clone storage so caller-mutation doesn't ripple back.
            return x.clone();
        }
        assert_eq!(x.rank(), 3);
        let seq = x.dim(0);
        let n_kv = x.dim(1);
        let hd = x.dim(2);
        let x_in = self.cuda_input(x);
        let mut y = self.alloc(seq * n_kv * n_rep * hd);

        let block_x = 32u32.min(hd as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((hd as u32) + block_x - 1) / block_x,
                (n_kv * n_rep) as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let n_kv_i = n_kv as i32;
        let n_rep_i = n_rep as i32;
        let hd_i = hd as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("repeat_kv_f32"))
                .arg(x_in.as_ref()).arg(&mut y)
                .arg(&seq_i).arg(&n_kv_i).arg(&n_rep_i).arg(&hd_i)
                .launch(cfg)
                .expect("repeat_kv launch");
        }
        self.make_tensor(y, vec![seq, n_kv * n_rep, hd])
    }

    fn bmm_qkt(&self, q: &Tensor, k: &Tensor, scale: f32, past: usize) -> Tensor {
        let seq = q.dim(0);
        let n_h = q.dim(1);
        let hd = q.dim(2);
        let kv_len = k.dim(0);
        debug_assert_eq!(k.dim(1), n_h);
        debug_assert_eq!(k.dim(2), hd);

        let q_in = self.cuda_input(q);
        let k_in = self.cuda_input(k);
        let mut scores = self.alloc(seq * n_h * kv_len);

        let block_x = 32u32.min(kv_len as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((kv_len as u32) + block_x - 1) / block_x,
                n_h as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let n_h_i = n_h as i32;
        let kv_len_i = kv_len as i32;
        let hd_i = hd as i32;
        let past_i = past as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("bmm_qkt_f32"))
                .arg(q_in.as_ref()).arg(k_in.as_ref()).arg(&mut scores)
                .arg(&seq_i).arg(&n_h_i).arg(&kv_len_i).arg(&hd_i)
                .arg(&scale).arg(&past_i)
                .launch(cfg)
                .expect("bmm_qkt launch");
        }
        self.make_tensor(scores, vec![seq, n_h, kv_len])
    }

    fn bmm_av(&self, scores: &Tensor, v: &Tensor) -> Tensor {
        let seq = scores.dim(0);
        let n_h = scores.dim(1);
        let kv_len = scores.dim(2);
        let hd = v.dim(2);
        debug_assert_eq!(v.dim(0), kv_len);
        debug_assert_eq!(v.dim(1), n_h);

        let sc_in = self.cuda_input(scores);
        let v_in = self.cuda_input(v);
        let mut out = self.alloc(seq * n_h * hd);

        let block_x = 32u32.min(hd as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((hd as u32) + block_x - 1) / block_x,
                n_h as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_i = seq as i32;
        let n_h_i = n_h as i32;
        let kv_len_i = kv_len as i32;
        let hd_i = hd as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("bmm_av_f32"))
                .arg(sc_in.as_ref()).arg(v_in.as_ref()).arg(&mut out)
                .arg(&seq_i).arg(&n_h_i).arg(&kv_len_i).arg(&hd_i)
                .launch(cfg)
                .expect("bmm_av launch");
        }
        self.make_tensor(out, vec![seq, n_h, hd])
    }

    fn attention(
        &self,
        q: &Tensor,
        k_buffer: &Tensor,
        v_buffer: &Tensor,
        kv_len: usize,
        scale: f32,
        past: usize,
        sliding_window: Option<usize>,
    ) -> Tensor {
        let seq = q.dim(0);
        let n_h_q = q.dim(1);
        let hd = q.dim(2);
        let max_kv_len = k_buffer.dim(0);
        let n_h_kv = k_buffer.dim(1);
        debug_assert_eq!(k_buffer.dim(2), hd);
        debug_assert_eq!(v_buffer.dim(0), max_kv_len);
        debug_assert_eq!(v_buffer.dim(1), n_h_kv);
        debug_assert_eq!(v_buffer.dim(2), hd);
        debug_assert!(kv_len <= max_kv_len);
        debug_assert_eq!(n_h_q % n_h_kv, 0, "n_h_q must be a multiple of n_h_kv");

        // VENDORED-LOCAL: a few query rows (decode): the keys split over many blocks, a warp
        // a key, rather than a block per head walking all of them a thread a key.
        if seq <= 4 && sliding_window.is_none() && past + seq <= kv_len && kv_len <= 32768 {
            return self.split_attention(q, k_buffer, v_buffer, None, kv_len, scale, past);
        }

        // VENDORED-LOCAL: full bidirectional vision attention. A bounded score
        // matrix lets the tiled GEMM reuse tiles instead of rereading K/V per query.
        // Never select this path for causal decoding or long-context attention.
        if seq>=128 && seq==kv_len && n_h_q==n_h_kv && past>=kv_len
            && sliding_window.is_none() && seq.checked_mul(seq).and_then(|n|n.checked_mul(n_h_q)).is_some_and(|n|n<=96*1024*1024) {
            let q=self.cuda_input(q);
            let k=self.cuda_input(k_buffer);
            let v=self.cuda_input(v_buffer);
            // SAFETY: both GEMMs write every element of their outputs.
            let mut scores=unsafe { self.stream.alloc::<f32>(seq*seq*n_h_q) }.expect("vision scores");
            // SAFETY: the PV GEMM writes every output element.
            let mut out=unsafe { self.stream.alloc::<f32>(seq*n_h_q*hd) }.expect("vision attention output");
            // Head h: scores[h][i][j] = scale * sum_d q[i,h,d] k[j,h,d] (q and k rows heads*hd apart).
            let w=n_h_q*hd;
            self.gemm(q.as_ref(),k.as_ref(),&mut scores,(seq,seq,hd),(w,w,seq),true,(n_h_q,hd,hd,seq*seq),scale);
            let mut scores=self.make_tensor(scores,vec![n_h_q,seq,seq]);
            self.softmax_last(&mut scores);
            let scores=self.cuda_input(&scores);
            // out[i,h,d] = sum_j scores[h][i][j] v[j,h,d]: each head writes its own hd columns of every row.
            self.gemm(scores.as_ref(),v.as_ref(),&mut out,(seq,hd,seq),(seq,w,w),false,(n_h_q,seq*seq,hd,hd),1.0);
            return self.make_tensor(out,vec![seq,n_h_q,hd]);
        }

        let bs = pow2_ceil_u32(hd.min(256).max(32) as u32);
        let smem_bytes = (kv_len as u32 + bs) * 4;

        if smem_bytes > 48 * 1024 {
            let parts = kv_len.div_ceil(4096);
            let q_in = self.cuda_input(q);
            let k_in = self.cuda_input(k_buffer);
            let v_in = self.cuda_input(v_buffer);
            let mut partial = self.alloc(seq * n_h_q * parts * (hd + 2));
            let mut out = self.alloc(seq * n_h_q * hd);
            let (seq_i, nh_i, nk_i, len_i, hd_i, past_i, sw_i) =
                (seq as i32, n_h_q as i32, n_h_kv as i32, kv_len as i32,
                 hd as i32, past as i32, sliding_window.unwrap_or(0) as i32);
            let rows_i = (seq * n_h_q) as i32;
            let parts_i = parts as i32;
            // SAFETY: buffers belong to this context, sizes and grids match the
            // partition/merge layouts; both launches use the same ordered stream.
            unsafe {
                self.stream.launch_builder(self.func("attention_partition_f32"))
                    .arg(q_in.as_ref()).arg(k_in.as_ref()).arg(v_in.as_ref()).arg(&mut partial)
                    .arg(&seq_i).arg(&nh_i).arg(&nk_i).arg(&len_i).arg(&hd_i)
                    .arg(&scale).arg(&past_i).arg(&sw_i)
                    .launch(LaunchConfig { grid_dim: (n_h_q as u32, seq as u32, parts as u32),
                        block_dim: (bs, 1, 1), shared_mem_bytes: (4096 + bs) * 4 })
                    .expect("partition attention launch");
                self.stream.launch_builder(self.func("attention_merge_f32"))
                    .arg(&partial).arg(&mut out).arg(&rows_i).arg(&parts_i).arg(&hd_i)
                    .launch(LaunchConfig { grid_dim: (rows_i as u32, 1, 1),
                        block_dim: (bs, 1, 1), shared_mem_bytes: 0 })
                    .expect("merge attention launch");
            }
            return self.make_tensor(out, vec![seq, n_h_q, hd]);
        }

        let q_in = self.cuda_input(q);
        let k_in = self.cuda_input(k_buffer);
        let v_in = self.cuda_input(v_buffer);
        let mut out = self.alloc(seq * n_h_q * hd);

        let cfg = LaunchConfig {
            grid_dim: (n_h_q as u32, seq as u32, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: smem_bytes,
        };
        let seq_i = seq as i32;
        let n_h_q_i = n_h_q as i32;
        let n_h_kv_i = n_h_kv as i32;
        let max_kv_len_i = max_kv_len as i32;
        let kv_len_i = kv_len as i32;
        let hd_i = hd as i32;
        let past_i = past as i32;
        // 0 disables the sliding-window mask in the kernel; any positive value enables it.
        let sw_i: i32 = sliding_window.map(|w| w as i32).unwrap_or(0);
        unsafe {
            self.stream
                .launch_builder(self.func("attention_f32"))
                .arg(q_in.as_ref()).arg(k_in.as_ref()).arg(v_in.as_ref())
                .arg(&mut out)
                .arg(&seq_i).arg(&n_h_q_i).arg(&n_h_kv_i)
                .arg(&max_kv_len_i).arg(&kv_len_i).arg(&hd_i)
                .arg(&scale).arg(&past_i).arg(&sw_i)
                .launch(cfg)
                .expect("attention launch");
        }

        self.make_tensor(out, vec![seq, n_h_q, hd])
    }

    fn alloc_zeros(&self, shape: Vec<usize>) -> Tensor {
        let n: usize = shape.iter().product();
        let slice = self.alloc(n);
        self.make_tensor(slice, shape)
    }

    fn copy_axis0_into(&self, dst: &mut Tensor, start: usize, src: &Tensor) {
        debug_assert!(start <= dst.dim(0));
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        let off = start * inner;
        let n = src.numel();

        let src_in = self.cuda_input(src);
        let dst_dev = self.cuda_input_mut(dst);

        // Slice mutable view over [off..off+n] of dst, then d2d-copy from src.
        let mut dst_view = dst_dev.slice_mut(off..off + n);
        self.stream
            .memcpy_dtod(src_in.as_ref(), &mut dst_view)
            .expect("d2d copy_axis0_into");
    }

    fn slice_axis0(&self, src: &Tensor, end: usize) -> Tensor {
        debug_assert!(end <= src.dim(0));
        let mut shape = src.shape().to_vec();
        shape[0] = end;
        let inner: usize = src.shape().iter().skip(1).product::<usize>().max(1);
        let n = end * inner;

        let src_in = self.cuda_input(src);
        let mut new_slice = self.alloc(n);
        // Take the [0..n] prefix of src and copy to new buffer.
        let src_view = src_in.as_ref().slice(0..n);
        self.stream
            .memcpy_dtod(&src_view, &mut new_slice)
            .expect("d2d slice_axis0");
        self.make_tensor(new_slice, shape)
    }

    fn slice_axis0_range(&self, src: &Tensor, start: usize, count: usize) -> Tensor {
        debug_assert!(start + count <= src.dim(0));
        let mut shape = src.shape().to_vec();
        shape[0] = count;
        let inner: usize = src.shape().iter().skip(1).product::<usize>().max(1);
        let off = start * inner;
        let n = count * inner;

        let src_in = self.cuda_input(src);
        let mut new_slice = self.alloc_uninit(n);
        let src_view = src_in.as_ref().slice(off..off + n);
        self.stream
            .memcpy_dtod(&src_view, &mut new_slice)
            .expect("d2d slice_axis0_range");
        self.make_tensor(new_slice, shape)
    }

    fn slice_axis1_2d(&self, src: &Tensor, idx: usize) -> Tensor {
        debug_assert_eq!(src.rank(), 3);
        let d0 = src.dim(0);
        let d1 = src.dim(1);
        let d2 = src.dim(2);
        debug_assert!(idx < d1);

        let src_in = self.cuda_input(src);
        let mut out = self.alloc(d0 * d2);

        let block_x = 256u32.min(d2 as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((d2 as u32) + block_x - 1) / block_x,
                d0 as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let d0_i = d0 as i32;
        let d1_i = d1 as i32;
        let d2_i = d2 as i32;
        let idx_i = idx as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("slice_axis1_2d_f32"))
                .arg(src_in.as_ref()).arg(&mut out)
                .arg(&d0_i).arg(&d1_i).arg(&d2_i).arg(&idx_i)
                .launch(cfg)
                .expect("slice_axis1_2d launch");
        }
        self.make_tensor(out, vec![d0, d2])
    }

    fn add_to_axis0_range(&self, dst: &mut Tensor, start: usize, count: usize, src: &Tensor) {
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        debug_assert_eq!(src.numel(), inner, "broadcast src.numel() != dst inner");
        debug_assert!(start + count <= dst.dim(0));

        let src_in = self.cuda_input(src);
        let dst_dev = self.cuda_input_mut(dst);

        let block_x = 256u32.min(inner as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((inner as u32) + block_x - 1) / block_x,
                count as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let start_i = start as i32;
        let count_i = count as i32;
        let inner_i = inner as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("add_to_axis0_range_f32"))
                .arg(dst_dev).arg(src_in.as_ref())
                .arg(&start_i).arg(&count_i).arg(&inner_i)
                .launch(cfg)
                .expect("add_to_axis0_range launch");
        }
    }

    fn add_to_axis0_range_scaled(
        &self,
        dst: &mut Tensor,
        start: usize,
        count: usize,
        src: &Tensor,
        scale: f32,
    ) {
        let inner: usize = dst.shape().iter().skip(1).product::<usize>().max(1);
        debug_assert_eq!(src.numel(), inner, "broadcast src.numel() != dst inner");
        debug_assert!(start + count <= dst.dim(0));

        let src_in = self.cuda_input(src);
        let dst_dev = self.cuda_input_mut(dst);

        let block_x = 256u32.min(inner as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((inner as u32) + block_x - 1) / block_x,
                count as u32,
                1,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let start_i = start as i32;
        let count_i = count as i32;
        let inner_i = inner as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("add_to_axis0_range_scaled_f32"))
                .arg(dst_dev).arg(src_in.as_ref())
                .arg(&start_i).arg(&count_i).arg(&inner_i)
                .arg(&scale)
                .launch(cfg)
                .expect("add_to_axis0_range_scaled launch");
        }
    }

    fn altup_predict(&self, streams: &Tensor, coefs: &Tensor, n_alt: usize) -> Tensor {
        debug_assert_eq!(streams.dim(0), n_alt);
        let seq    = streams.dim(1);
        let hidden = streams.dim(2);

        let s_in = self.cuda_input(streams);
        let c_in = self.cuda_input(coefs);
        let mut out = self.alloc(n_alt * seq * hidden);

        let block_x = 32u32.min(hidden as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((hidden as u32) + block_x - 1) / block_x,
                n_alt as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_alt_i = n_alt as i32;
        let seq_i   = seq as i32;
        let hd_i    = hidden as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("altup_predict_f32"))
                .arg(s_in.as_ref()).arg(c_in.as_ref()).arg(&mut out)
                .arg(&n_alt_i).arg(&seq_i).arg(&hd_i)
                .launch(cfg)
                .expect("altup_predict launch");
        }
        self.make_tensor(out, vec![n_alt, seq, hidden])
    }

    fn altup_correct(
        &self,
        predictions: &Tensor,
        activated: &Tensor,
        coefs: &Tensor,
        n_alt: usize,
        active_idx: usize,
    ) -> Tensor {
        debug_assert_eq!(predictions.dim(0), n_alt);
        let seq    = predictions.dim(1);
        let hidden = predictions.dim(2);

        let p_in = self.cuda_input(predictions);
        let a_in = self.cuda_input(activated);
        let c_in = self.cuda_input(coefs);
        let mut out = self.alloc(n_alt * seq * hidden);

        let block_x = 32u32.min(hidden as u32).max(1);
        let cfg = LaunchConfig {
            grid_dim: (
                ((hidden as u32) + block_x - 1) / block_x,
                n_alt as u32,
                seq as u32,
            ),
            block_dim: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_alt_i = n_alt as i32;
        let seq_i   = seq as i32;
        let hd_i    = hidden as i32;
        let active_i = active_idx as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("altup_correct_f32"))
                .arg(p_in.as_ref()).arg(a_in.as_ref()).arg(c_in.as_ref()).arg(&mut out)
                .arg(&n_alt_i).arg(&seq_i).arg(&hd_i).arg(&active_i)
                .launch(cfg)
                .expect("altup_correct launch");
        }
        self.make_tensor(out, vec![n_alt, seq, hidden])
    }

    fn argmax_last(&self, x: &Tensor) -> Vec<u32> {
        let last = x.dim(x.rank() - 1);
        let n_rows = x.numel() / last;
        let x_in = self.cuda_input(x);
        let mut out = self.alloc_u32(n_rows);

        let bs = Self::pow2_ceil(last.min(256)) as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (bs, 1, 1),
            shared_mem_bytes: bs * 8,
        };
        let n_rows_i = n_rows as i32;
        let last_i = last as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("argmax_last_f32"))
                .arg(x_in.as_ref()).arg(&mut out)
                .arg(&n_rows_i).arg(&last_i)
                .launch(cfg)
                .expect("argmax launch");
        }
        self.stream.memcpy_dtov(&out).expect("d2h argmax")
    }
}

fn reserve_weight(used: &std::sync::atomic::AtomicUsize, limit: usize, bytes: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    used.fetch_update(Relaxed, Relaxed, |n| n.checked_add(bytes).filter(|&n|n <= limit)).is_ok()
}
#[cfg(test)]
mod observer_weight_budget_tests {
    use super::*;
    #[test]
    #[ignore = "explicit CUDA packed-tile regression"]
    fn streamed_q4_tiles_match_resident_weights() {
        let backend = CudaBackend::new(0).unwrap().with_weight_budget(0);
        let mut bytes = vec![0u8; 7 * 4 * 144];
        for (i, block) in bytes.chunks_mut(144).enumerate() {
            block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            block[4..16].fill(1);
            block[16..].fill((i % 13 + 1) as u8);
        }
        let w = QuantizedTensor::from_bytes_cpu(bytes.clone(),vec![7,1024],GgmlType::Q4_K);
        for batch in [1,3] {
            let x = Tensor::from_vec((0..batch*1024).map(|i|(i%17) as f32/17.0).collect(),vec![batch,1024]);
            let resident = backend.upload_quantized(w.bytes(),w.shape().to_vec(),w.dtype());
            let expected = backend.linear_q(&x,&resident).to_host();
            let actual = backend.linear_q_tiled(&x,&w,2);
            assert_eq!(actual.shape(),expected.shape());
            for (a,b) in actual.data().iter().zip(expected.data()) { assert!((a-b).abs() <= 1e-4 * b.abs().max(1.0)); }
        }
        assert_eq!(w.bytes(),bytes.as_slice());
    }
    #[test]
    fn packed_weight_reservation_never_exceeds_budget() {
        let used = std::sync::atomic::AtomicUsize::new(0);
        assert!(reserve_weight(&used, 100, 60));
        assert!(!reserve_weight(&used, 100, 41));
        assert!(reserve_weight(&used, 100, 40));
        assert!(!reserve_weight(&used, 100, usize::MAX));
        assert_eq!(used.load(std::sync::atomic::Ordering::Relaxed), 100);
    }
}
