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

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use ggml_quants::GgmlType;
use ggml_rs::backend::{Backend, RopeType};
use ggml_rs::quantized::{QuantizedDeviceStorage, QuantizedTensor};
use ggml_rs::tensor::{DeviceStorage, Tensor};
use thiserror::Error;

use crate::kernels::{KERNEL_NAMES, KERNEL_SRC};

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("cuda driver error: {0:?}")]
    Driver(cudarc::driver::DriverError),

    #[error("nvrtc compile error: {0:?}")]
    Nvrtc(cudarc::nvrtc::CompileError),

    #[error("cublas error: {0}")]
    Cublas(String),

    #[error("kernel `{0}` not found in compiled module")]
    MissingKernel(&'static str),
}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> Self { Self::Driver(e) }
}
impl From<cudarc::nvrtc::CompileError> for CudaError {
    fn from(e: cudarc::nvrtc::CompileError) -> Self { Self::Nvrtc(e) }
}
impl From<cudarc::cublas::result::CublasError> for CudaError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self { Self::Cublas(format!("{e:?}")) }
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
        let mut new_slice = self.stream
            .alloc_zeros::<u8>(self.bytes.len())
            .expect("alloc failed");
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
        let mut new_slice = self.stream
            .alloc_zeros::<f32>(self.slice.len())
            .expect("alloc failed");
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
    // VENDORED-LOCAL: GPU-02 — `ctx`/`stream`/`name` widened to pub(crate) so
    // the transfer module (pinned staging + H2D overlap) can build on them.
    pub(crate) ctx:    Arc<CudaContext>,
    pub(crate) stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    funcs:  HashMap<&'static str, CudaFunction>,
    blas:   CudaBlas,
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
    pub(crate) name:   String,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("name", &self.name)
            .finish()
    }
}

impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Result<Self, CudaError> {
        let ctx = CudaContext::new(device_ordinal)?;
        let stream = ctx.default_stream();

        let ptx = compile_ptx(KERNEL_SRC)?;
        let module = ctx.load_module(ptx)?;

        let mut funcs = HashMap::with_capacity(KERNEL_NAMES.len());
        for &n in KERNEL_NAMES {
            let f = module.load_function(n).map_err(|_| CudaError::MissingKernel(n))?;
            funcs.insert(n, f);
        }

        let blas = CudaBlas::new(stream.clone())?;

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
            ctx,
            stream,
            module,
            funcs,
            blas,
            h2d: Default::default(), // created lazily — see the field comment
            event_pool: Default::default(),
            name: format!("cuda:{device_ordinal}"),
        })
    }

    // VENDORED-LOCAL: GPU-02 — the H2D transfer stream, created on first use
    // (see the `h2d` field comment for why this must not happen in `new`).
    pub(crate) fn h2d(&self) -> &Arc<CudaStream> {
        self.h2d.get_or_init(|| {
            self.ctx.new_stream().expect("h2d transfer stream")
        })
    }

    pub fn context(&self) -> &Arc<CudaContext> { &self.ctx }

    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module's launches.
    pub(crate) fn func(&self, name: &'static str) -> &CudaFunction { &self.funcs[name] }
    // VENDORED-LOCAL: MOE-01/02 — pub(crate) for the moe module's launches.
    pub(crate) fn func_dyn(&self, name: &str) -> &CudaFunction {
        self.funcs.get(name).unwrap_or_else(|| panic!("CUDA kernel `{name}` not loaded"))
    }

    fn alloc(&self, n: usize) -> CudaSlice<f32> {
        self.stream.alloc_zeros::<f32>(n).expect("alloc failed")
    }

    fn alloc_u32(&self, n: usize) -> CudaSlice<u32> {
        self.stream.alloc_zeros::<u32>(n).expect("alloc failed")
    }

    fn upload_f32(&self, host: &[f32]) -> CudaSlice<f32> {
        self.stream.memcpy_stod(host).expect("h2d failed")
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

    /// Borrow `&CudaSlice<u8>` for a quantized tensor that may be on host or
    /// device. Mirrors `cuda_input` for F32 tensors.
    fn cuda_quant_input<'a>(&self, w: &'a QuantizedTensor) -> CudaQuantInput<'a> {
        if let Some(s) = w.device_storage() {
            if let Some(c) = s.as_any().downcast_ref::<CudaQuantStorage>() {
                return CudaQuantInput::Borrowed(&c.bytes);
            }
            panic!("QuantizedTensor on `{}` passed to CudaBackend", s.device_name());
        }
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
    fn cuda_input_mut<'a>(&self, t: &'a mut Tensor) -> &'a mut CudaSlice<f32> {
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
        // cuMemGetInfo returns free + total bytes for the current context.
        // Cheap call (~1µs) so loaders can poll per-tensor without measurable
        // overhead.
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
        let bytes = w.bytes().to_vec();
        let shape = w.shape().to_vec();
        let dtype = w.dtype();
        self.upload_quantized(&bytes, shape, dtype)
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
        let slice = self.upload_f32(t.data());
        self.make_tensor(slice, t.shape().to_vec())
    }

    fn to_host(&self, t: Tensor) -> Tensor {
        if t.is_cpu() { t } else { t.to_host() }
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
        //     wins because cuBLAS's per-call setup dominates the compute.
        //   * ≥ 100M FLOPs (real-world prefill on ~1B+ models): cuBLAS Sgemm.
        //     Column-major math: Y_cm[n,m] = Σ_k W_cm[k,n] * X_cm[k,m]
        //     = op_T(W_cm) · X_cm with m=N, n=M, k=K.
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
        let mut c = self.alloc(m_rows * out);

        // Decode-time GEMV fast path: same coop pattern as the quantized kernels,
        // but for dense F32 weights. Below the cuBLAS threshold the alternative
        // is the naive 16×16 kernel which only has 1 productive thread per warp
        // at M=1; the coop version uses all 32. Used by Qwen3.6 27B's F32
        // ssm_ba matmul (96×5120, 48× per token).
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
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_T,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: out as i32,
                n: m_rows as i32,
                k: in_ as i32,
                alpha: 1.0f32,
                beta: 0.0f32,
                lda: in_ as i32,
                ldb: in_ as i32,
                ldc: out as i32,
            };
            unsafe {
                self.blas.gemm(cfg, b.as_ref(), a.as_ref(), &mut c)
                    .expect("cublas sgemm");
            }
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

    fn linear_q(&self, x: &Tensor, w: &QuantizedTensor) -> Tensor {
        let in_ = x.dim(x.rank() - 1);
        let out = w.dim(0);
        debug_assert_eq!(w.dim(1), in_);
        let m_rows = x.numel() / in_;

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
        let mut out = self.alloc(n_rows * last);

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
        let mut y = self.alloc(n);
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
        let mut y = self.alloc(n);
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
        let mut y = self.alloc(n);
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

        let pos_dev = self.upload_u32(positions);
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
                .arg(x_dev).arg(&pos_dev)
                .arg(&seq_i).arg(&n_h_i).arg(&hd_i)
                .arg(&rot_i).arg(&theta)
                .launch(cfg)
                .expect("rope_partial_neox launch");
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
        let mut conv_out = self.alloc(seq * conv_dim);
        let mut output   = self.alloc(seq * num_v_heads * head_v_dim);

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
                .arg(&scale_q).arg(&eps)
                .launch(step_cfg)
                .expect("delta_net_step_loop launch");
        }

        self.make_tensor(output, vec![seq, num_v_heads * head_v_dim])
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
        let mut q_only = self.alloc(seq * n_heads * head_dim);
        let mut q_gate = self.alloc(seq * n_heads * head_dim);

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

        let bs = pow2_ceil_u32(hd.min(256).max(32) as u32);
        let smem_bytes = (kv_len as u32 + bs) * 4;

        // Out-of-shared-memory fallback: walk the default trait impl, which
        // applies the sliding-window mask on the host. Slow but correct.
        if smem_bytes > 48 * 1024 {
            let n_rep = n_h_q / n_h_kv;
            let k_pref = self.slice_axis0(k_buffer, kv_len);
            let v_pref = self.slice_axis0(v_buffer, kv_len);
            let k_full = self.repeat_kv(&k_pref, n_rep);
            let v_full = self.repeat_kv(&v_pref, n_rep);
            let mut scores = self.bmm_qkt(q, &k_full, scale, past);
            if let Some(w) = sliding_window {
                // Pull scores to host, apply mask, push back.
                let mut host = scores.to_host();
                let data = host.data_mut();
                for s in 0..seq {
                    let q_pos = past + s;
                    let lo = q_pos.saturating_sub(w - 1);
                    for h in 0..n_h_q {
                        let row_off = (s * n_h_q + h) * kv_len;
                        for t in 0..lo.min(kv_len) {
                            data[row_off + t] = f32::NEG_INFINITY;
                        }
                    }
                }
                scores = self.to_device(host);
            }
            self.softmax_last(&mut scores);
            return self.bmm_av(&scores, &v_full);
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
        let mut new_slice = self.alloc(n);
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
