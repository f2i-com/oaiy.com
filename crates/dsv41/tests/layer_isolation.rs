//! Per-layer error in isolation (teacher forcing) against the oracle's
//! golden file; bounds and method in `dsv41::golden`. Heavy, `#[ignore]`d:
//!
//!   cargo test -p dsv41 --release --test layer_isolation -- --ignored --nocapture
//!   DSV41_GOLDEN_NAME=golden_long.safetensors cargo test ... (247-token prompt)

mod common;

use dsv41::golden::{check, isolation, PhaseReport};
use dsv41::model::ModelOptions;

pub fn print(r: &PhaseReport) {
    for l in &r.layers {
        eprintln!(
            "  {} layer {:2}: out p50 {:.1e} p95 {:.1e} max {:.1e}  bf16-exact {:4.1}%  attn p95 {:.1e} max {:.1e}  moe p95 {:.1e}  route flips {}",
            r.phase,
            l.layer,
            l.p50,
            l.p95,
            l.max,
            100.0 * l.bf16_exact,
            l.attn_p95,
            l.attn_max,
            l.moe_p95,
            l.route_flips
        );
    }
    eprintln!(
        "  {}: {} of {} token-routes differ from the oracle ({:.2}%); logits rel-L2 {:.2e}",
        r.phase,
        r.route_flips,
        r.routes,
        100.0 * r.route_flips as f64 / r.routes.max(1) as f64,
        r.logits_rel
    );
}

#[test]
#[ignore]
fn layers_in_isolation() {
    let opts = ModelOptions { max_seq: 1024, expert_cache_bytes: 24 << 30, direct_io: false };
    let Some((g, mut model)) = common::setup(opts) else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let first: u32 = g.read_i64("generated_ids").unwrap()[0] as u32;
    let mut failures = Vec::new();
    for (phase, ids, start) in [("prefill", prompt.clone(), 0), ("decode0", vec![first], prompt.len())] {
        let r = isolation(&mut model, &g, phase, &ids, start).unwrap();
        print(&r);
        failures.extend(check(&r));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
