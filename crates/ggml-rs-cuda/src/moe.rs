//! VENDORED-LOCAL: MOE-01 / MOE-02 — GPU-resident MoE routing and grouped
//! expert execution (docs/ROADMAP.md Phase 3).
//!
//! What this replaces at decode (seq=1), per MoE layer:
//!
//! ```text
//! before: router gemv → full-logits D2H (sync) → CPU top-k softmax →
//!         per expert: gate/up matvec, silu_mul_split, down matvec,
//!         scaled accumulate            (≈ 4·top_k + 2 launches + 1 sync)
//! after:  router gemv → moe_topk_softmax → moe_gate_up_act →
//!         moe_down_scale → moe_reduce_slots
//!                                      (5 launches, 0 syncs resident)
//! ```
//!
//! Routing ids/weights stay on the device for the resident path. The
//! streaming path copies back ONLY the compact ids mailbox (`top_k` u32s)
//! so the device expert cache knows which experts to stage; weights remain
//! device-side because the grouped down kernel consumes them there.
//!
//! The pointer tables ([`MoeDevicePlan`]) are what let one kernel launch
//! address a different weight allocation per expert: each entry is the raw
//! device address of one expert's packed fused-gate‖up (or down) bytes,
//! uploaded once per layer (resident) or per layer-forward (streaming, one
//! small H2D covering the layer's k staged experts).
//!
//! Numerics: the kernels replicate the coop-GEMV partitioning of the
//! reference `linear_q` path wherever that path uses a coop kernel, so the
//! grouped intermediate is bit-identical to the reference composition there
//! (see the numerics comment in `kernels.rs`). The Q4_K strided fallback
//! (K=768 down projection on Qwen3-30B) reassociates within the down dot
//! only; greedy token ids are verified unchanged on the real models.

#![allow(deprecated)] // memcpy_stod / memcpy_dtov are the available cudarc names

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use ggml_quants::GgmlType;
use ggml_rs::quantized::QuantizedTensor;
use ggml_rs::tensor::Tensor;

use crate::backend::{CudaBackend, CudaQuantStorage};

/// Largest `n_experts` the one-block routing kernel handles (dynamic shared
/// memory is n_experts × 4 B, well under the 48 KiB default limit).
pub const MOE_ROUTE_MAX_EXPERTS: usize = 8192;
/// Largest `top_k` the routing kernel handles (register-array bound).
pub const MOE_ROUTE_MAX_TOP_K: usize = 128;

/// Device-resident routing result: top-k expert ids + softmax weights for
/// `seq` tokens, produced by `moe_topk_softmax_f32`. Host copies are
/// explicit and tiny (the "mailbox") — nothing here forces a full-logits
/// D2H.
pub struct MoeRoutingDevice {
    pub ids:     CudaSlice<u32>,
    pub weights: CudaSlice<f32>,
    stream: Arc<CudaStream>,
    pub seq:   usize,
    pub top_k: usize,
}

impl std::fmt::Debug for MoeRoutingDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeRoutingDevice")
            .field("seq", &self.seq)
            .field("top_k", &self.top_k)
            .finish()
    }
}

impl MoeRoutingDevice {
    /// Mailbox D2H of the routed expert ids: `seq * top_k * 4` bytes. This
    /// is the only host round-trip the streaming path needs per MoE layer.
    pub fn ids_to_host(&self) -> Vec<u32> {
        self.stream.memcpy_dtov(&self.ids).expect("d2h moe ids")
    }

    /// Mailbox D2H of the routing weights. Only the reference (per-expert
    /// host-driven) dispatch loop needs these on the host; the grouped
    /// kernels read them from `self.weights` on device.
    pub fn weights_to_host(&self) -> Vec<f32> {
        self.stream.memcpy_dtov(&self.weights).expect("d2h moe weights")
    }
}

impl MoeRoutingDevice {
    /// VENDORED-LOCAL: GLM-5.3-Flash. A routing built from host ids and weights.
    ///
    /// `moe_route_device` computes the route on the card from the router logits,
    /// which is what the generic MoE path does. glm5next routes on the host --
    /// its router sits inside a hyper-connection pass that never leaves host
    /// memory -- so its ids and weights are uploaded instead: two copies of
    /// `top_k` values, against the eight 14 MB expert records the kernels then
    /// read without leaving the card.
    ///
    /// `ids` are whatever the pointer table is indexed by, so a table built over
    /// routing slots wants `0..k` here rather than the real expert ids.
    pub fn from_host(backend: &CudaBackend, ids: &[u32], weights: &[f32]) -> Self {
        assert_eq!(ids.len(), weights.len(), "MoeRoutingDevice: ids/weights length mismatch");
        assert!(!ids.is_empty(), "MoeRoutingDevice: empty route");
        Self {
            ids: backend.stream.memcpy_stod(ids).expect("h2d moe ids"),
            weights: backend.stream.memcpy_stod(weights).expect("h2d moe weights"),
            stream: Arc::clone(&backend.stream),
            seq: 1,
            top_k: ids.len(),
        }
    }
}

/// Per-layer tables for the grouped MoE kernels: one device pointer per
/// expert weight allocation, plus dtype/geometry the launcher checks once at
/// plan build. `ptrs` is `[gate_up entries | down entries]`, `ntab` entries
/// each — a resident layer indexes it by expert id (`ntab == n_experts`), a
/// streaming layer-forward by routing slot (`ntab == top_k`).
pub struct MoeDevicePlan {
    ptrs:    CudaSlice<u64>,
    ntab:    usize,
    /// Per-expert down-projection scales (Gemma 4 MoE); `None` elsewhere.
    scales:  Option<CudaSlice<f32>>,
    pub gate_up_dtype: GgmlType,
    pub down_dtype:    GgmlType,
    pub ff:     usize,
    pub hidden: usize,
    pub use_gelu: bool,
    /// VENDORED-LOCAL: GLM-5.3-Flash. The SwiGLU clamp, 0 for none.
    pub limit: f32,
    /// Whether that clamp applies after the SiLU rather than before it.
    pub after_silu: bool,
}

impl std::fmt::Debug for MoeDevicePlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeDevicePlan")
            .field("gate_up_dtype", &self.gate_up_dtype)
            .field("down_dtype", &self.down_dtype)
            .field("ff", &self.ff)
            .field("hidden", &self.hidden)
            .field("use_gelu", &self.use_gelu)
            .finish()
    }
}

impl MoeDevicePlan {
    /// Upload the pointer tables (one small H2D). `gate_up_ptrs` and
    /// `down_ptrs` must have the same length — the table entry count.
    pub fn new(
        backend: &CudaBackend,
        gate_up_ptrs: &[u64],
        down_ptrs: &[u64],
        scales: Option<&[f32]>,
        gate_up_dtype: GgmlType,
        down_dtype: GgmlType,
        ff: usize,
        hidden: usize,
        use_gelu: bool,
    ) -> Self {
        assert_eq!(
            gate_up_ptrs.len(),
            down_ptrs.len(),
            "MoeDevicePlan: gate_up/down table length mismatch"
        );
        assert!(!gate_up_ptrs.is_empty(), "MoeDevicePlan: empty pointer table");
        let mut both = Vec::with_capacity(gate_up_ptrs.len() + down_ptrs.len());
        both.extend_from_slice(gate_up_ptrs);
        both.extend_from_slice(down_ptrs);
        let ptrs = backend.stream.memcpy_stod(&both).expect("h2d moe ptr table");
        let scales =
            scales.map(|s| backend.stream.memcpy_stod(s).expect("h2d moe scales"));
        Self {
            ptrs,
            ntab: gate_up_ptrs.len(),
            scales,
            gate_up_dtype,
            down_dtype,
            ff,
            hidden,
            use_gelu,
            limit: 0.0,
            after_silu: false,
        }
    }

    /// VENDORED-LOCAL: GLM-5.3-Flash. Clamp the activation, as
    /// `Backend::swiglu_clamped` does. `limit <= 0` leaves it unclamped, which is
    /// what every other model here wants, so this is a step off the plain `new`
    /// rather than an argument on it.
    pub fn with_clamp(mut self, limit: f32, after_silu: bool) -> Self {
        self.limit = limit;
        self.after_silu = after_silu;
        self
    }
}

/// Raw device address of a quantized tensor's packed bytes, when they live
/// on a CUDA device. `None` for host-resident tensors — callers treat that
/// as "not eligible for grouped dispatch".
pub fn quant_device_ptr(qt: &QuantizedTensor) -> Option<u64> {
    let s = qt.device_storage()?;
    let c = s.as_any().downcast_ref::<CudaQuantStorage>()?;
    let (p, _guard) = c.bytes.device_ptr(c.bytes.stream());
    Some(p as u64)
}

/// True when the grouped kernels cover `dtype` at inner dim `k`.
/// Q4_K/Q6_K need whole 256-wide super-blocks; Q8_0 whole 32-wide blocks.
pub fn grouped_kernel_covers(dtype: GgmlType, k: usize) -> bool {
    match dtype {
        GgmlType::Q4_K | GgmlType::Q6_K => k % 256 == 0,
        GgmlType::Q8_0 => k % 32 == 0,
        _ => false,
    }
}

impl CudaBackend {
    /// MOE-01: run top-k + softmax over `router_logits` (`[seq, n_experts]`)
    /// on device. One block per token row; ids and weights stay on device.
    /// Returns `None` when the kernel's guards don't hold (logits not on
    /// this device, `n_experts`/`top_k` out of range) — callers fall back
    /// to the host reference path.
    pub fn moe_route_device(
        &self,
        router_logits: &Tensor,
        top_k: usize,
    ) -> Option<MoeRoutingDevice> {
        if router_logits.rank() != 2 {
            return None;
        }
        let seq = router_logits.dim(0);
        let n_experts = router_logits.dim(1);
        let k = top_k.min(n_experts);
        if seq == 0 || k == 0 || k > MOE_ROUTE_MAX_TOP_K || n_experts > MOE_ROUTE_MAX_EXPERTS
        {
            return None;
        }
        // Only route on-device when the logits are already there; uploading
        // them first would make the mailbox round trip strictly worse than
        // the host reference.
        let storage = router_logits.device_storage()?;
        storage.as_any().downcast_ref::<crate::backend::CudaStorage>()?;
        let logits_in = self.cuda_input(router_logits);

        let mut ids = self.stream.alloc_zeros::<u32>(seq * k).expect("alloc moe ids");
        let mut weights =
            self.stream.alloc_zeros::<f32>(seq * k).expect("alloc moe weights");

        let cfg = LaunchConfig {
            grid_dim: (seq as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n_experts * 4) as u32,
        };
        let n_experts_i = n_experts as i32;
        let k_i = k as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("moe_topk_softmax_f32"))
                .arg(logits_in.as_ref())
                .arg(&mut ids)
                .arg(&mut weights)
                .arg(&n_experts_i)
                .arg(&k_i)
                .launch(cfg)
                .expect("moe_topk_softmax launch");
        }
        Some(MoeRoutingDevice {
            ids,
            weights,
            stream: self.stream.clone(),
            seq,
            top_k: k,
        })
    }

    /// MOE-02: grouped expert FFN for one token (`x` is `[1, hidden]`).
    /// Three launches — gate_up+activation, down+route-scale into per-slot
    /// partials, fixed-order slot reduction — and no host round-trips:
    /// routing ids/weights and the per-expert weight pointers are all read
    /// from device memory. Returns the `[1, hidden]` MoE output.
    pub fn moe_grouped_ffn(
        &self,
        x: &Tensor,
        plan: &MoeDevicePlan,
        routing: &MoeRoutingDevice,
    ) -> Tensor {
        assert_eq!(routing.seq, 1, "moe_grouped_ffn: decode-only (seq == 1)");
        assert!(routing.top_k <= plan.ntab, "moe_grouped_ffn: table smaller than top_k");
        assert_eq!(x.numel(), plan.hidden, "moe_grouped_ffn: x is not [1, hidden]");
        let k = routing.top_k;
        let ff = plan.ff;
        let hidden = plan.hidden;

        let x_in = self.cuda_input(x);
        let mut act = self.stream.alloc_zeros::<f32>(k * ff).expect("alloc moe act");
        let mut partial =
            self.stream.alloc_zeros::<f32>(k * hidden).expect("alloc moe partial");
        let mut out = self.stream.alloc_zeros::<f32>(hidden).expect("alloc moe out");

        const OUT_PER_BLOCK: u32 = 8;
        let block = (32u32, OUT_PER_BLOCK, 1u32);
        let ff_i = ff as i32;
        let hidden_i = hidden as i32;
        let k_gate_i = hidden as i32; // gate_up inner dim = model hidden
        let k_down_i = ff as i32;     // down inner dim = expert ff
        let gelu_i: i32 = if plan.use_gelu { 1 } else { 0 };
        let limit_f: f32 = plan.limit;
        let after_silu_i: i32 = if plan.after_silu { 1 } else { 0 };
        let has_scales_i: i32 = if plan.scales.is_some() { 1 } else { 0 };
        let ntab_i = plan.ntab as i32;

        // gate_up + activation: grid (ceil(ff/8), k)
        let gu_kernel = match plan.gate_up_dtype {
            GgmlType::Q4_K => "moe_gate_up_act_q4_k_f32",
            GgmlType::Q6_K => "moe_gate_up_act_q6_k_f32",
            GgmlType::Q8_0 => "moe_gate_up_act_q8_0_f32",
            other => panic!("moe_grouped_ffn: no gate_up kernel for {other:?}"),
        };
        let gu_cfg = LaunchConfig {
            grid_dim: ((ff as u32 + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK, k as u32, 1),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(self.func_dyn(gu_kernel))
                .arg(x_in.as_ref())
                .arg(&plan.ptrs)
                .arg(&routing.ids)
                .arg(&mut act)
                .arg(&ff_i)
                .arg(&k_gate_i)
                .arg(&gelu_i)
                .arg(&0i32) // idx_base: resident tables index by expert id
                .arg(&limit_f)
                .arg(&after_silu_i)
                .launch(gu_cfg)
                .expect("moe gate_up launch");
        }

        // down + route-scale into per-slot partials: grid (ceil(hidden/8), k).
        // The down table starts `ntab` entries into `ptrs`; the kernel adds
        // idx_base to the expert id before indexing, so pass it through.
        let dn_kernel = match plan.down_dtype {
            GgmlType::Q4_K => "moe_down_scale_q4_k_f32",
            GgmlType::Q6_K => "moe_down_scale_q6_k_f32",
            GgmlType::Q8_0 => "moe_down_scale_q8_0_f32",
            other => panic!("moe_grouped_ffn: no down kernel for {other:?}"),
        };
        let dn_cfg = LaunchConfig {
            grid_dim: ((hidden as u32 + OUT_PER_BLOCK - 1) / OUT_PER_BLOCK, k as u32, 1),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        let weights_arg = &routing.weights;
        unsafe {
            let mut b = self.stream.launch_builder(self.func_dyn(dn_kernel));
            b.arg(&act)
                .arg(&plan.ptrs)
                .arg(&routing.ids)
                .arg(weights_arg);
            match &plan.scales {
                Some(s) => {
                    b.arg(s);
                }
                None => {
                    // No per-expert scales: any valid f32 pointer satisfies
                    // the ABI; the kernel never reads it (has_scales == 0).
                    b.arg(weights_arg);
                }
            }
            b.arg(&mut partial)
                .arg(&hidden_i)
                .arg(&k_down_i)
                .arg(&has_scales_i)
                .arg(&ntab_i)
                .launch(dn_cfg)
                .expect("moe down launch");
        }

        // Fixed-order slot reduction: out[i] = Σ_slot partial[slot, i].
        let red_block = 256u32;
        let red_cfg = LaunchConfig {
            grid_dim: ((hidden as u32 + red_block - 1) / red_block, 1, 1),
            block_dim: (red_block, 1, 1),
            shared_mem_bytes: 0,
        };
        let k_i = k as i32;
        unsafe {
            self.stream
                .launch_builder(self.func("moe_reduce_slots_f32"))
                .arg(&partial)
                .arg(&mut out)
                .arg(&k_i)
                .arg(&hidden_i)
                .launch(red_cfg)
                .expect("moe reduce launch");
        }

        self.make_tensor(out, vec![1, hidden])
    }
}
