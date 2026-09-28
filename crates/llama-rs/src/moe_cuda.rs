//! VENDORED-LOCAL: MOE-01 / MOE-02 — grouped CUDA fast path for MoE decode.
//!
//! Resident MoE layers (`MoeFfn.stream == None`) cache a [`GpuLayerPlan`]
//! on first use: static per-layer pointer tables over the device-resident
//! expert weights. A grouped decode forward is then, per layer:
//!
//! ```text
//! router gemv (existing linear op)
//! moe_topk_softmax_f32   → ids + weights, device-resident (MOE-01)
//! moe_gate_up_act_<q>    → fused gate/up matvec + SwiGLU/GeGLU (MOE-02a)
//! moe_down_scale_<q>     → fused down matvec + route-scale (MOE-02b)
//! moe_reduce_slots_f32   → fixed-order slot reduction (MOE-02c)
//! ```
//!
//! — 5 launches and no host synchronization, replacing ≈ 4·top_k + 2
//! launches and one full-logits D2H barrier. The routing decision never
//! leaves the device: the kernels index the layer's pointer table by the
//! on-device expert ids.
//!
//! Eligibility (anything else falls back to the reference per-expert loop in
//! `moe.rs`): CUDA backend, single-token decode, every expert a fused
//! packed-quant gate‖up + packed-quant down with uniform dtype/geometry
//! across the layer, and an inner dim the grouped kernels cover
//! ([`ggml_rs_cuda::grouped_kernel_covers`]).

use std::sync::Arc;

use ggml_quants::GgmlType;
use ggml_rs::{Backend, Tensor};
use ggml_rs_cuda::{grouped_kernel_covers, quant_device_ptr, CudaBackend, MoeDevicePlan};

use crate::loader::{FfnPair, Weight};
use crate::moe::{MoeFfn, MoeOptions};

/// Cached per-layer grouped plan (static pointer tables + geometry).
pub(crate) struct GpuLayerPlan {
    pub plan: MoeDevicePlan,
}

// VENDORED-LOCAL: A/B switch for benchmarks and debugging —
/// `OAIY_MOE_GROUPED=0` forces the reference per-expert loops (resident and
/// streaming alike). Read once per process; unset or any other value keeps
/// the grouped path on.
pub(crate) fn grouped_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !matches!(std::env::var("OAIY_MOE_GROUPED").as_deref(), Ok("0")))
}

impl std::fmt::Debug for GpuLayerPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLayerPlan").field("plan", &self.plan).finish()
    }
}

/// Build the layer's plan, or `None` when any expert weight isn't eligible
/// (split pair, dense weight, host-resident, mixed dtype/geometry, or an
/// uncovered inner dim). `None` is cached by the caller too — ineligibility
/// is a load-time property, not per-token.
fn build_plan(cuda: &CudaBackend, moe: &MoeFfn, opts: &MoeOptions<'_>) -> Option<GpuLayerPlan> {
    if moe.stream.is_some() {
        return None;
    }
    let n = moe.gate_up_experts.len();
    if n == 0 || n != moe.down_experts.len() {
        return None;
    }
    let mut gu_ptrs = Vec::with_capacity(n);
    let mut dn_ptrs = Vec::with_capacity(n);
    let mut gu_dt: Option<GgmlType> = None;
    let mut dn_dt: Option<GgmlType> = None;
    let mut ff = 0usize;
    let mut hidden = 0usize;
    for (pair, down) in moe.gate_up_experts.iter().zip(&moe.down_experts) {
        let FfnPair::Fused(Weight::Quant(gu)) = pair else { return None };
        let Weight::Quant(dn) = down else { return None };
        if gu.shape().len() != 2 || dn.shape().len() != 2 || gu.dim(0) % 2 != 0 {
            return None;
        }
        let f = gu.dim(0) / 2;
        let h = gu.dim(1);
        // down is [hidden, ff]: one row per output hidden unit.
        if dn.dim(0) != h || dn.dim(1) != f {
            return None;
        }
        if ff == 0 {
            ff = f;
            hidden = h;
        } else if ff != f || hidden != h {
            return None;
        }
        match gu_dt {
            Some(d) if d != gu.dtype() => return None,
            None => gu_dt = Some(gu.dtype()),
            _ => {}
        }
        match dn_dt {
            Some(d) if d != dn.dtype() => return None,
            None => dn_dt = Some(dn.dtype()),
            _ => {}
        }
        gu_ptrs.push(quant_device_ptr(gu)?);
        dn_ptrs.push(quant_device_ptr(dn)?);
    }
    let (gu_dt, dn_dt) = (gu_dt?, dn_dt?);
    if !grouped_kernel_covers(gu_dt, hidden) || !grouped_kernel_covers(dn_dt, ff) {
        return None;
    }
    Some(GpuLayerPlan {
        plan: MoeDevicePlan::new(
            cuda,
            &gu_ptrs,
            &dn_ptrs,
            opts.down_exps_scale_host,
            gu_dt,
            dn_dt,
            ff,
            hidden,
            opts.use_gelu,
        ),
    })
}

/// Grouped fast path for resident MoE decode. Returns `None` — the caller
/// runs the reference per-expert loop — when the backend isn't CUDA, this
/// isn't single-token decode, or the layer is ineligible (see above).
pub(crate) fn try_grouped_forward(
    backend: &dyn Backend,
    x: &Tensor,
    moe: &MoeFfn,
    router_logits: &Tensor,
    opts: &MoeOptions<'_>,
) -> Option<Tensor> {
    if x.rank() != 2 || x.dim(0) != 1 {
        return None;
    }
    if !grouped_enabled() {
        return None;
    }
    let cuda = backend.as_any().downcast_ref::<CudaBackend>()?;
    let cell = moe
        .gpu_plan
        .get_or_init(|| build_plan(cuda, moe, opts).map(Arc::new));
    let gpu = cell.as_ref()?;
    if gpu.plan.hidden != x.dim(1) {
        return None;
    }
    // MOE-01: routing stays on device; the grouped kernels read ids/weights
    // from `routing` directly. `None` when the kernel guards don't hold —
    // the reference path then routes via the mailbox instead.
    let routing = cuda.moe_route_device(router_logits, moe.top_k)?;
    Some(cuda.moe_grouped_ffn(x, &gpu.plan, &routing))
}
