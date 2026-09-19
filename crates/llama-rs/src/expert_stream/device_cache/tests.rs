//! CUDA-gated tests for the VRAM expert cache (CACHE-02 / STREAM-01).
//!
//! Every test skips silently when no CUDA device/driver is present, so the
//! suite stays green on GPU-less hosts; on this machine they run for real.
//! The fixtures are the same synthetic stacked-experts GGUF the host-side
//! streaming tests use (Q8_0, 2 layers x 4 experts), via
//! `expert_stream::tests`'s helpers.

use std::sync::Arc;

use ggml_rs::quantized::QuantizedTensor;
use ggml_rs::{Backend, Tensor};
use ggml_rs_cuda::CudaBackend;

use crate::expert_stream::tests::{
    assert_f32_bitwise, make_store, stacked_fixture, FF, HIDDEN, N_EXPERTS,
};
use crate::expert_stream::{record_plan, StreamShared};
use crate::moe::MoeOptions;

use super::*;

/// A CUDA backend, or None when this host has no usable device (tests skip).
fn cuda() -> Option<Arc<CudaBackend>> {
    CudaBackend::new(0).ok().map(Arc::new)
}

fn quant_of(w: &Weight) -> &QuantizedTensor {
    match w {
        Weight::Quant(q) => q,
        other => panic!("expected Quant weight, got {other:?}"),
    }
}

/// CACHE-02 acceptance: a VRAM hit performs ZERO H2D bytes, and a re-lookup
/// of the same expert returns the same entry (same device allocation — no
/// per-dispatch device allocation after warm-up).
#[test]
fn hit_is_zero_h2d_and_same_allocation() {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let plan = record_plan(make_store(&fx).layout(), 0);
    let dc = DeviceCache::new(backend, 8 * fx.rec_bytes, fx.rec_bytes).expect("cache");

    let rec = &fx.records[0][1];
    let a = dc.stage_and_admit(0, 1, rec, &plan).expect("stage");
    let after_stage = dc.stats();
    assert_eq!(after_stage.h2d_bytes as usize, a.bytes);
    assert_eq!(after_stage.misses, 0, "staging itself counts no lookup miss");

    dc.ensure_ready(&a);
    let b = dc.acquire(0, 1).expect("hit");
    let s = dc.stats();
    assert_eq!(s.hits, 1);
    assert_eq!(s.bytes_hit as usize, a.bytes);
    assert_eq!(
        s.h2d_bytes, after_stage.h2d_bytes,
        "a device hit uploaded bytes — the hit path must be zero-H2D"
    );
    assert!(
        Arc::ptr_eq(&a, &b),
        "hit returned a different entry: the same expert must keep one \
         stable device allocation across dispatches"
    );
    // The tickets drained on first use; a second ensure_ready is a no-op.
    dc.ensure_ready(&b);
    assert_eq!(dc.stats().waits_free + dc.stats().waits_pending, 2);
}

/// A device-resident entry computes bitwise-identically to the host upload
/// path: same packed bytes, same kernels.
#[test]
fn device_entry_matches_host_upload_numerically() {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let plan = record_plan(make_store(&fx).layout(), 0);
    let dc = DeviceCache::new(backend.clone(), 8 * fx.rec_bytes, fx.rec_bytes).expect("cache");

    let rec = &fx.records[0][2];
    let en = dc.stage_and_admit(0, 2, rec, &plan).expect("stage");
    dc.ensure_ready(&en);
    backend.synchronize();

    // The fixture's gate/up share dtype+shape, so the pair is Fused.
    let FfnPair::Fused(fused_w) = &en.pair else {
        panic!("fixture gate/up must fuse");
    };
    let fused_len = plan.gate.1 + plan.up.1;
    let host_fused = QuantizedTensor::from_bytes_cpu(
        rec[..fused_len].to_vec(),
        vec![plan.gate.3[0] + plan.up.3[0], plan.gate.3[1]],
        plan.gate.2,
    );
    let host_down = QuantizedTensor::from_bytes_cpu(
        rec[plan.down.0..plan.down.0 + plan.down.1].to_vec(),
        plan.down.3.to_vec(),
        plan.down.2,
    );

    let x = Tensor::from_vec(
        (0..HIDDEN).map(|i| (i as f32 - 16.0) / 16.0).collect(),
        vec![1, HIDDEN],
    );
    // The down projection consumes the activated [1, ff] intermediate.
    let act = Tensor::from_vec(
        (0..FF).map(|i| (i as f32 - 32.0) / 32.0).collect(),
        vec![1, FF],
    );
    let be: &dyn Backend = backend.as_ref();

    let dev_f = be.linear_q(&x, quant_of(fused_w)).to_host();
    let host_f = be.linear_q(&x, &host_fused).to_host();
    assert_f32_bitwise(dev_f.data(), host_f.data(), "fused gate/up matvec");

    let dev_d = be.linear_q(&act, quant_of(&en.down)).to_host();
    let host_d = be.linear_q(&act, &host_down).to_host();
    assert_f32_bitwise(dev_d.data(), host_d.data(), "down matvec");
}

/// Eviction under the byte budget: a third admission evicts the LFRU victim
/// and the byte accounting never exceeds the budget.
#[test]
fn eviction_under_byte_budget() {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let plan = record_plan(make_store(&fx).layout(), 0);
    let entry_bytes = plan.gate.1 + plan.up.1 + plan.down.1;
    let dc = DeviceCache::new(backend, 2 * entry_bytes, fx.rec_bytes).expect("cache");

    // Warm expert 0 twice so LFRU frequency protects it over expert 1.
    let a0 = dc.stage_and_admit(0, 0, &fx.records[0][0], &plan).expect("stage 0");
    dc.ensure_ready(&a0);
    dc.acquire(0, 0);
    dc.acquire(0, 0);
    drop(a0);
    dc.stage_and_admit(0, 1, &fx.records[0][1], &plan).expect("stage 1");
    dc.stage_and_admit(0, 2, &fx.records[0][2], &plan).expect("stage 2");

    let s = dc.stats();
    assert_eq!(s.evictions, 1, "third admission over a 2-entry budget");
    assert!(s.bytes_used as usize <= 2 * entry_bytes);
    assert_eq!(s.entries, 2);
    assert!(
        dc.acquire(0, 0).is_some(),
        "the frequent entry must survive (LFRU: frequency first)"
    );
    assert!(dc.acquire(0, 1).is_none(), "the coldest entry was evicted");
    assert!(dc.acquire(0, 2).is_some());
}

/// An entry a dispatch still references (strong_count > 1) is never chosen
/// as the eviction victim — the device-side spelling of the host cache's
/// live-lease rule.
#[test]
fn no_evict_while_referenced() {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let plan = record_plan(make_store(&fx).layout(), 0);
    let entry_bytes = plan.gate.1 + plan.up.1 + plan.down.1;
    let dc = DeviceCache::new(backend, 2 * entry_bytes, fx.rec_bytes).expect("cache");

    let held = dc.stage_and_admit(0, 0, &fx.records[0][0], &plan).expect("stage 0");
    dc.ensure_ready(&held);
    // Fill the rest of the budget and force an eviction while `held` is live.
    dc.stage_and_admit(0, 1, &fx.records[0][1], &plan).expect("stage 1");
    dc.stage_and_admit(0, 2, &fx.records[0][2], &plan).expect("stage 2");

    let s = dc.stats();
    assert_eq!(s.evictions, 1);
    assert!(
        dc.acquire(0, 0).is_some(),
        "an entry with a live compute reference must be skipped as victim"
    );
    drop(held);
    // Unreferenced now: the next admission must be able to evict again
    // (the exact victim is LFRU's choice — expert 0's two hits above made
    // it the *frequent* entry, so a colder one goes first).
    dc.stage_and_admit(0, 3, &fx.records[0][3], &plan).expect("stage 3");
    let s = dc.stats();
    assert_eq!(s.evictions, 2);
    assert!(s.bytes_used as usize <= 2 * entry_bytes);
    assert_eq!(s.entries, 2);
}

/// Dropping an entry whose upload may still be in flight (never waited,
/// evicted immediately) must be safe — the transfer API's stream-ordered
/// drop semantics cover it. This test crashes loudly if that contract
/// breaks.
#[test]
fn evicted_entry_with_unwatched_upload_drops_safely() {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let plan = record_plan(make_store(&fx).layout(), 0);
    let entry_bytes = plan.gate.1 + plan.up.1 + plan.down.1;
    let dc = DeviceCache::new(backend.clone(), entry_bytes, fx.rec_bytes).expect("cache");

    // Budget fits ONE entry: staging the second evicts the first while its
    // H2D copy may still be running (no ensure_ready in between).
    dc.stage_and_admit(0, 0, &fx.records[0][0], &plan).expect("stage 0");
    dc.stage_and_admit(0, 1, &fx.records[0][1], &plan).expect("stage 1");
    assert_eq!(dc.stats().evictions, 1);
    drop(dc);
    backend.synchronize();
}

/// End-to-end through `LayerStream::forward_with_logits`, eager mode:
///  * pass 2 runs entirely on device hits — device misses stay 0 and
///    `h2d_bytes` does not move (zero-H2D proof, and no new device
///    allocation: a miss is what allocates);
///  * device hits never touch the host cache (its counters do not move on
///    pass 2 — the "host cache disabled" case differs only in WHERE the
///    miss path reads from, which pass 1 already exercises);
///  * both passes produce bitwise-identical output to the host path.
fn forward_twice(eager: bool) {
    let Some(backend) = cuda() else { return };
    let fx = stacked_fixture();
    let rec = fx.rec_bytes;
    let shared = {
        let store = make_store(&fx);
        let s = StreamShared::new(store, 8 * rec, 0).expect("shared");
        let mut dc = DeviceCache::new(backend.clone(), 8 * rec, rec).expect("cache");
        dc.set_eager(eager);
        s.enable_device_cache_with(Arc::new(dc));
        s
    };

    // Reference: the same forward on a fresh shared state with NO device
    // cache (host-lease path, per-dispatch upload).
    let ref_shared = StreamShared::new(make_store(&fx), 8 * rec, 0).expect("ref shared");

    // Route to experts 1 and 3 (top-2): logits put them clearly on top.
    let mut logits = vec![0.0f32; N_EXPERTS];
    logits[1] = 5.0;
    logits[3] = 4.0;
    let router_logits = Tensor::from_vec(logits, vec![1, N_EXPERTS]);
    let x = Tensor::from_vec(
        (0..HIDDEN).map(|i| (i as f32 - 16.0) / 16.0).collect(),
        vec![1, HIDDEN],
    );
    let be: &dyn Backend = backend.as_ref();
    let opts = MoeOptions::default();

    let layer = shared.layer(0);
    let out1 = layer
        .forward_with_logits(be, &x, &router_logits, 2, &opts)
        .to_host();
    assert!(shared.error().is_none(), "pass 1: {}", shared.error().unwrap_or_default());

    let ref_out = ref_shared
        .layer(0)
        .forward_with_logits(be, &x, &router_logits, 2, &opts)
        .to_host();
    assert_f32_bitwise(out1.data(), ref_out.data(), "device path vs host path");

    let s1 = shared.device_cache_stats().expect("device stats");
    let host1 = shared.cache_stats();
    assert_eq!(s1.hits, 0, "pass 1 is cold");
    assert_eq!(s1.misses, 2, "top-2 routing stages two uploads");
    assert!(s1.h2d_bytes > 0);

    let out2 = layer
        .forward_with_logits(be, &x, &router_logits, 2, &opts)
        .to_host();
    assert_f32_bitwise(out1.data(), out2.data(), "pass 1 vs pass 2");

    let s2 = shared.device_cache_stats().expect("device stats");
    assert_eq!(s2.hits, 2, "pass 2 serves both experts from VRAM");
    assert_eq!(
        s2.misses, s1.misses,
        "pass 2 staged nothing new — a miss is what allocates a device \
         slot, so this also proves zero per-token device allocation once warm"
    );
    assert_eq!(
        s2.h2d_bytes, s1.h2d_bytes,
        "pass 2 uploaded zero bytes — VRAM hits are zero-H2D"
    );
    let host2 = shared.cache_stats();
    assert_eq!(
        host1.hits + host1.misses,
        host2.hits + host2.misses,
        "device hits must not touch the host cache at all"
    );
}

#[test]
fn forward_eager_hits_are_zero_h2d_and_bypass_host_cache() {
    forward_twice(true);
}

#[test]
fn forward_lazy_hits_are_zero_h2d_and_bypass_host_cache() {
    forward_twice(false);
}
