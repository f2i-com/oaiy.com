//! Wall time per phase of a forward pass, so a slow token can be attributed
//! rather than guessed at.
//!
//! Always compiled: it is one [`std::time::Instant::now`] per phase per layer,
//! about 250 calls a token against a token measured in milliseconds. Nothing
//! synchronises the device here, so a phase that only launches kernels will
//! look cheap and the phase that next reads a result will carry the wait --
//! which is the honest picture for a host-driven forward, where every
//! [`Mat::apply`] reads its own result back.

use std::sync::atomic::{AtomicU64, Ordering};

/// Nanoseconds in each phase since the last [`reset`].
pub static HC: AtomicU64 = AtomicU64::new(0);
pub static KDA: AtomicU64 = AtomicU64::new(0);
pub static MLA: AtomicU64 = AtomicU64::new(0);
pub static FFN: AtomicU64 = AtomicU64::new(0);
pub static HEAD: AtomicU64 = AtomicU64::new(0);

/// Inside [`KDA`]: the projections, the depthwise conv and its state shift,
/// the decay gate, and the delta-rule recurrence.
pub static KDA_PROJ: AtomicU64 = AtomicU64::new(0);
pub static KDA_CONV: AtomicU64 = AtomicU64::new(0);
pub static KDA_GATE: AtomicU64 = AtomicU64::new(0);
pub static KDA_STEP: AtomicU64 = AtomicU64::new(0);

/// Inside [`MLA`]: the projections, the sparse indexer, the query absorb
/// through `k_b`, and the attention through `v_b`.
// VENDORED-LOCAL: GLM-5.3-Flash. The FFN's three parts. Two rounds of
// optimising the routed experts moved the FFN 88.6 -> 57.5 ms, which is a lot
// less than the arithmetic said it should, so the question of which part of an
// FFN layer the time is in has to be measured rather than reasoned about.
pub static FFN_ROUTER: AtomicU64 = AtomicU64::new(0);
pub static FFN_ROUTED: AtomicU64 = AtomicU64::new(0);
pub static FFN_SHARED: AtomicU64 = AtomicU64::new(0);
// Inside the routed experts: resolving a route (three cache tiers, the LFRU
// ranking, the staging) against actually computing it. The routed experts are
// 48.8 ms of a 98.7 ms token and the parts that are accounted for -- ~20 ms of
// PCIe stall and ~12 ms of CPU tier -- do not add up to it.
pub static FFN_RESOLVE: AtomicU64 = AtomicU64::new(0);
pub static FFN_DISPATCH: AtomicU64 = AtomicU64::new(0);

pub static MLA_PROJ: AtomicU64 = AtomicU64::new(0);
pub static MLA_INDEX: AtomicU64 = AtomicU64::new(0);
pub static MLA_ABSORB: AtomicU64 = AtomicU64::new(0);
pub static MLA_ATTEND: AtomicU64 = AtomicU64::new(0);

/// The five top-level phases, in forward order. These sum to the token.
pub fn all() -> [(&'static str, &'static AtomicU64); 5] {
    [
        ("hyper-connections", &HC),
        ("KDA attention", &KDA),
        ("MLA attention", &MLA),
        ("FFN (router, shared, routed)", &FFN),
        ("output head", &HEAD),
    ]
}

/// Sub-phases of [`KDA`] and [`MLA`]. These sum to less than their parents:
/// what is left over is the projections and glue not counted here.
pub fn inner() -> [(&'static str, &'static AtomicU64); 13] {
    [
        ("  FFN router", &FFN_ROUTER),
        ("  FFN routed experts", &FFN_ROUTED),
        ("    of which: resolve", &FFN_RESOLVE),
        ("    of which: dispatch", &FFN_DISPATCH),
        ("  FFN shared expert", &FFN_SHARED),
        ("  KDA q/k/v projections", &KDA_PROJ),
        ("  KDA depthwise conv + shift", &KDA_CONV),
        ("  KDA decay gate", &KDA_GATE),
        ("  KDA delta-rule recurrence", &KDA_STEP),
        ("  MLA projections", &MLA_PROJ),
        ("  MLA sparse indexer", &MLA_INDEX),
        ("  MLA absorb through k_b", &MLA_ABSORB),
        ("  MLA attend through v_b", &MLA_ATTEND),
    ]
}

pub fn reset() {
    for (_, c) in all() {
        c.store(0, Ordering::Relaxed);
    }
    for (_, c) in inner() {
        c.store(0, Ordering::Relaxed);
    }
}

/// Add the time since `t` to `c`.
pub fn add(c: &AtomicU64, t: std::time::Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

pub fn ms(c: &AtomicU64) -> f64 {
    c.load(Ordering::Relaxed) as f64 / 1e6
}

/// Total over every phase.
pub fn total_ms() -> f64 {
    all().iter().map(|(_, c)| ms(c)).sum()
}
