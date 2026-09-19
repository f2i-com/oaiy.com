//! Phase B gate (docs/DEEPSEEK_V41.md): the Rust backbone against the
//! oracle's golden dumps, layer by layer, then greedy generation token for
//! token. Heavy (loads ~11 GB of trunk, streams ~2,500 experts), so it is
//! `#[ignore]`d; run it with
//!
//!   cargo test -p dsv41 --release --test forward_oracle -- --ignored --nocapture
//!
//! Environment: DSV41_MODEL, DSV41_GOLDEN_DIR (holds golden.safetensors and
//! engram_meta.safetensors). Skips when either is missing.

mod common;

use std::collections::HashMap;
use std::time::Instant;

use dsv41::formats::f32_to_bf16;
use dsv41::model::{argmax, ModelOptions};
use dsv41::safetensors::StIndex;

struct Diff {
    rel_l2: f64,
    max_abs: f32,
    bf16_exact: f64,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "shape mismatch");
    let (mut num, mut den, mut max_abs, mut exact) = (0.0f64, 0.0f64, 0.0f32, 0usize);
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
        max_abs = max_abs.max((a - b).abs());
        exact += usize::from(f32_to_bf16(*a) == f32_to_bf16(*b));
    }
    Diff { rel_l2: (num / den.max(1e-30)).sqrt(), max_abs, bf16_exact: exact as f64 / got.len() as f64 }
}

fn compare(phase: &str, trace: &HashMap<String, Vec<f32>>, g: &StIndex, n_layers: usize) -> f64 {
    let hashes = &trace["engram_hashes"];
    let want: Vec<f32> = g.read_i64(&format!("{phase}.engram_hashes")).unwrap().iter().map(|&v| v as f32).collect();
    assert_eq!(hashes, &want, "{phase}: engram hashes differ");
    let e = diff(&trace["embed"], &g.read_f32(&format!("{phase}.embed")).unwrap());
    assert_eq!(e.max_abs, 0.0, "{phase}: embedding differs");
    let mut worst = 0.0f64;
    for l in 0..n_layers {
        let o = diff(&trace[&format!("layer{l:02}.out")], &g.read_f32(&format!("{phase}.layer{l:02}.out")).unwrap());
        let p = diff(&trace[&format!("layer{l:02}.pre_mix")], &g.read_f32(&format!("{phase}.layer{l:02}.pre_mix")).unwrap());
        eprintln!(
            "  {phase} layer {l:2}: out rel-L2 {:.2e}  max|d| {:.2e}  bf16-exact {:6.2}%   pre_mix max|d| {:.1e}",
            o.rel_l2,
            o.max_abs,
            100.0 * o.bf16_exact,
            p.max_abs
        );
        worst = worst.max(o.rel_l2);
    }
    worst
}

#[test]
#[ignore]
fn backbone_matches_oracle() {
    let opts = ModelOptions { max_seq: 1024, expert_cache_bytes: 24 << 30, direct_io: true };
    let Some((g, mut model)) = common::setup(opts) else { return };
    let prompt: Vec<u32> = g.read_i64("prompt_ids").unwrap().iter().map(|&v| v as u32).collect();
    let expected: Vec<u32> = g.read_i64("generated_ids").unwrap().iter().map(|&v| v as u32).collect();
    let n_layers = model.cfg.n_layers;

    // prefill
    let mut trace: HashMap<String, Vec<f32>> = HashMap::new();
    let t = Instant::now();
    let logits = model.forward(&prompt, 0, &mut |k, v| {
        trace.insert(k.to_string(), v.to_vec());
    })
    .unwrap();
    eprintln!("prefill of {} tokens: {:.1}s", prompt.len(), t.elapsed().as_secs_f64());
    let worst_prefill = compare("prefill", &trace, &g, n_layers);
    let ld = diff(&logits, &g.read_f32("prefill.logits").unwrap());
    eprintln!("  prefill logits: rel-L2 {:.2e} max|d| {:.2e}", ld.rel_l2, ld.max_abs);
    let mut got = vec![argmax(&logits)];

    // first decode step, compared layer by layer
    trace.clear();
    let t = Instant::now();
    let logits = model.forward(&got[..1], prompt.len(), &mut |k, v| {
        trace.insert(k.to_string(), v.to_vec());
    })
    .unwrap();
    eprintln!("decode step: {:.1}s", t.elapsed().as_secs_f64());
    let worst_decode = compare("decode0", &trace, &g, n_layers);
    let ld = diff(&logits, &g.read_f32("decode0.logits").unwrap());
    eprintln!("  decode0 logits: rel-L2 {:.2e} max|d| {:.2e}", ld.rel_l2, ld.max_abs);
    got.push(argmax(&logits));

    // the rest greedily, token for token against the oracle
    while got.len() < expected.len() {
        let pos = prompt.len() + got.len() - 1;
        let logits = model.forward(&got[got.len() - 1..], pos, &mut |_, _| {}).unwrap();
        got.push(argmax(&logits));
    }
    let stats = model.expert_cache().stats();
    eprintln!("generated {got:?}\n expected {expected:?}\n expert cache: {stats:?}");
    assert_eq!(got, expected, "greedy tokens differ from the oracle");
    // Free-running drift is NOT a bug signal on its own: summation-order
    // differences flip a bf16 ulp, fp8 activation quantization turns that into
    // ~1% per layer, and a near-tied router picks a different 6th expert now
    // and then — compounded over 40 layers that reaches ~15% relative on the
    // decode token while every greedy token still agrees. Per-layer correctness
    // is gated by `layer_isolation` (teacher forcing); this bound only catches
    // a catastrophic divergence.
    assert!(worst_prefill < 0.5 && worst_decode < 0.5, "layer drift too large: {worst_prefill} / {worst_decode}");
}
