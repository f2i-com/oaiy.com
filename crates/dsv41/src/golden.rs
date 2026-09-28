//! Comparison against the oracle's golden files (`tools/dsv41/oracle.py`),
//! shared by the CPU and GPU test suites so both are held to one standard.
//! Library code: returns reports, never prints.
//!
//! The per-layer check feeds each layer the oracle's own input (teacher
//! forcing), so a layer's error is its own. Tokens whose router picked a
//! different expert set than the oracle (near-ties) are counted, not bounded:
//! swapping one of six experts legitimately moves that token's output.

use std::collections::HashMap;

use oaiy_engine::Result;

use crate::formats::f32_to_bf16;
use crate::hc::HC;
use crate::model::Backbone;
use crate::safetensors::StIndex;

/// Apply the test-only config overrides the oracle recorded in the golden
/// file's metadata (e.g. a shrunk `index_topk` so a short prompt exercises
/// top-k selection).
pub fn apply_overrides(model: &mut dyn Backbone, g: &StIndex) -> Result<()> {
    let bad = |k: &str| oaiy_engine::Error::Format(format!("golden metadata {k} is not a number"));
    if let Some(v) = g.metadata("index_topk") {
        model.config_mut().index_topk = v.parse().map_err(|_| bad("index_topk"))?;
    }
    if let Some(v) = g.metadata("candidate_topk_blocks") {
        model.config_mut().candidate_topk_blocks = v.parse().map_err(|_| bad("candidate_topk_blocks"))?;
    }
    Ok(())
}

pub fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt()
}

fn percentile(mut v: Vec<f64>, p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * p).round() as usize]
}

#[derive(Debug, Clone)]
pub struct LayerReport {
    pub layer: usize,
    /// Per-token relative L2 of the layer output, over tokens routed like the oracle.
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
    pub bf16_exact: f64,
    pub attn_p95: f64,
    pub attn_max: f64,
    pub moe_p95: f64,
    pub route_flips: usize,
}

#[derive(Debug, Clone)]
pub struct PhaseReport {
    pub phase: String,
    pub layers: Vec<LayerReport>,
    pub route_flips: usize,
    pub routes: usize,
    pub logits_rel: f64,
}

/// Teacher-forced run of `phase` ("prefill" / "decode0") over `ids` at `start_pos`.
pub fn isolation(model: &mut dyn Backbone, g: &StIndex, phase: &str, ids: &[u32], start_pos: usize) -> Result<PhaseReport> {
    let (d, n_layers, t) = (model.config().dim, model.config().n_layers, ids.len());
    let teacher = |l: usize| -> Result<(Vec<f32>, Vec<[f32; HC]>)> {
        let h = g.read_f32(&format!("{phase}.layer{:02}.out", l - 1))?;
        let p = g.read_f32(&format!("{phase}.layer{:02}.pre_mix", l - 1))?;
        Ok((h, p.chunks_exact(HC).map(|c| [c[0], c[1], c[2], c[3]]).collect()))
    };
    let mut trace: HashMap<String, Vec<f32>> = HashMap::new();
    let logits = model.forward_traced(ids, start_pos, Some(&teacher), &mut |k, v| {
        trace.insert(k.to_string(), v.to_vec());
    })?;
    let got = |name: &str| -> Result<&Vec<f32>> {
        trace.get(name).ok_or_else(|| oaiy_engine::Error::Format(format!("model did not trace {name}")))
    };

    let mut layers = Vec::with_capacity(n_layers);
    let mut flips_total = 0;
    for l in 0..n_layers {
        let key = |name: &str| format!("{phase}.layer{l:02}.{name}");
        let mut flipped = vec![false; t];
        if g.get(&key("route_ids")).is_some() {
            let want = g.read_i64(&key("route_ids"))?;
            let have = got(&format!("layer{l:02}.route_ids"))?;
            let k = want.len() / t;
            for (i, f) in flipped.iter_mut().enumerate() {
                let mut a: Vec<i64> = have[i * k..(i + 1) * k].iter().map(|&v| v as i64).collect();
                let mut b = want[i * k..(i + 1) * k].to_vec();
                a.sort_unstable();
                b.sort_unstable();
                *f = a != b;
            }
        }
        let route_flips = flipped.iter().filter(|&&f| f).count();
        flips_total += route_flips;
        let per_tok = |name: &str, width: usize| -> Result<Vec<f64>> {
            if g.get(&key(name)).is_none() {
                return Ok(vec![]);
            }
            let (have, want) = (got(&format!("layer{l:02}.{name}"))?, g.read_f32(&key(name))?);
            Ok((0..t).filter(|&i| !flipped[i]).map(|i| rel_l2(&have[i * width..(i + 1) * width], &want[i * width..(i + 1) * width])).collect())
        };
        let out = per_tok("out", HC * d)?;
        let (have, want) = (got(&format!("layer{l:02}.out"))?, g.read_f32(&key("out"))?);
        let bf16_exact = have.iter().zip(&want).filter(|(a, b)| f32_to_bf16(**a) == f32_to_bf16(**b)).count() as f64 / have.len().max(1) as f64;
        let (attn, moe) = (per_tok("attn_out", d)?, per_tok("moe_out", d)?);
        layers.push(LayerReport {
            layer: l,
            p50: percentile(out.clone(), 0.5),
            p95: percentile(out.clone(), 0.95),
            max: percentile(out, 1.0),
            bf16_exact,
            attn_p95: percentile(attn.clone(), 0.95),
            attn_max: percentile(attn, 1.0),
            moe_p95: percentile(moe, 0.95),
            route_flips,
        });
    }
    Ok(PhaseReport {
        phase: phase.to_string(),
        layers,
        route_flips: flips_total,
        routes: t * n_layers,
        logits_rel: rel_l2(&logits, &g.read_f32(&format!("{phase}.logits"))?),
    })
}

/// Bounds calibrated 2026-09-19 on both golden files (see the Phase B notes
/// in docs/DEEPSEEK_V41.md). Returns a description of every violation.
pub fn check(r: &PhaseReport) -> Vec<String> {
    const MAX_P95_REL: f64 = 0.05;
    const MAX_REL: f64 = 0.15;
    const MAX_LOGITS_REL: f64 = 0.02;
    const MAX_ROUTE_FLIPS: f64 = 0.02;
    let mut bad = Vec::new();
    for l in &r.layers {
        if l.p95 > MAX_P95_REL || l.max > MAX_REL {
            bad.push(format!("{} layer {}: p95 {:.3} max {:.3} on unflipped tokens", r.phase, l.layer, l.p95, l.max));
        }
    }
    let rate = r.route_flips as f64 / r.routes.max(1) as f64;
    if rate > MAX_ROUTE_FLIPS {
        bad.push(format!("{}: route flip rate {rate:.4}", r.phase));
    }
    if r.logits_rel > MAX_LOGITS_REL {
        bad.push(format!("{}: logits error {:.3}", r.phase, r.logits_rel));
    }
    bad
}
