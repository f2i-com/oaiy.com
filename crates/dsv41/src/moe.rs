//! MoE: router, routed experts through the expert cache, shared expert
//! (reference `Gate`, `Expert`, `MoE`).
//!
//! Routing: `sqrt(softplus(x . W_gate))`; the correction bias picks the top
//! 6 but the weights come from the unbiased scores, normalized and scaled by
//! `route_scale`. Outputs accumulate in f32 in ascending expert id order
//! (the reference's loop order), then the shared expert, then one bf16 round.

use std::sync::Arc;

use nrob::ecache::Ecache;
use nrob::store::WeightStore;
use nrob::Result;

use crate::config::Config;
use crate::expert::expert_forward_batch;
use crate::formats::to_bf16;
use crate::linear::{load_vec, Out, Weight};
use crate::ops::{silu, softplus};
use crate::safetensors::StIndex;

/// Where routed expert records come from: a store behind a bounded cache.
pub struct Experts {
    pub store: Arc<dyn WeightStore>,
    pub cache: Ecache,
}

pub struct Moe {
    layer: u32,
    gate: Weight,
    bias: Vec<f32>,
    shared: [Weight; 3],
}

/// Routed experts one token was sent to, with their weights.
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Moe {
    pub fn load(idx: &StIndex, layer: usize) -> Result<Moe> {
        let p = format!("layers.{layer}.ffn");
        Ok(Moe {
            layer: layer as u32,
            gate: Weight::load(idx, &format!("{p}.gate"))?,
            bias: load_vec(idx, &format!("{p}.gate.bias"))?,
            shared: [
                Weight::load(idx, &format!("{p}.shared_experts.w1"))?,
                Weight::load(idx, &format!("{p}.shared_experts.w2"))?,
                Weight::load(idx, &format!("{p}.shared_experts.w3"))?,
            ],
        })
    }

    /// Router decision for `t` tokens of `x` (`[t, dim]`, bf16 values).
    pub fn route(&self, cfg: &Config, x: &[f32], t: usize) -> Vec<Route> {
        route(cfg, &self.gate.forward(x, t, Out::F32), &self.bias, t)
    }
}

/// Routing from the gate's f32 logits (`[t, n_experts]`) and correction bias:
/// the bias picks the top experts, the unbiased `sqrt(softplus)` scores
/// weight them (normalized, times `route_scale`). Ties: lower id first.
pub fn route(cfg: &Config, logits: &[f32], bias: &[f32], t: usize) -> Vec<Route> {
    let n = bias.len();
    (0..t)
        .map(|i| {
            let scores: Vec<f32> = logits[i * n..(i + 1) * n].iter().map(|&v| softplus(v).sqrt()).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| (scores[b] + bias[b]).total_cmp(&(scores[a] + bias[a])).then(a.cmp(&b)));
            order.truncate(cfg.n_activated_experts);
            let mut weights: Vec<f32> = order.iter().map(|&e| scores[e]).collect();
            if weights.len() > 1 {
                let s = weights.iter().sum::<f32>() + 1e-20;
                weights.iter_mut().for_each(|w| *w /= s);
            }
            weights.iter_mut().for_each(|w| *w *= cfg.route_scale);
            Route { experts: order.into_iter().map(|e| e as u32).collect(), weights }
        })
        .collect()
}

impl Moe {

    /// `x`: `[t, dim]` (bf16 values); returns `[t, dim]` bf16 and the routes taken.
    pub fn forward(&self, cfg: &Config, x: &[f32], t: usize, experts: &Experts) -> Result<(Vec<f32>, Vec<Route>)> {
        let d = cfg.dim;
        let routes = self.route(cfg, x, t);
        let mut y = vec![0.0f32; t * d];
        let mut used: Vec<u32> = routes.iter().flat_map(|r| r.experts.iter().copied()).collect();
        used.sort_unstable();
        used.dedup();
        for e in used {
            let rec = experts.cache.acquire(self.layer, e, experts.store.as_ref())?;
            // every token routed to e, in token order, as one batch
            let (mut toks, mut xs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
            for (i, r) in routes.iter().enumerate() {
                if let Some(k) = r.experts.iter().position(|&x| x == e) {
                    toks.push(i);
                    xs.extend_from_slice(&x[i * d..(i + 1) * d]);
                    ws.push(r.weights[k]);
                }
            }
            let out = expert_forward_batch(&rec, &xs, Some(&ws), cfg.swiglu_limit);
            for (j, &i) in toks.iter().enumerate() {
                for (acc, v) in y[i * d..(i + 1) * d].iter_mut().zip(&out[j * d..(j + 1) * d]) {
                    *acc += v;
                }
            }
        }
        let shared = self.shared_forward(cfg, x, t);
        for (acc, v) in y.iter_mut().zip(shared) {
            *acc = to_bf16(*acc + v);
        }
        Ok((y, routes))
    }

    /// The fp8 shared expert: same SwiGLU as a routed one, no route weight.
    fn shared_forward(&self, cfg: &Config, x: &[f32], t: usize) -> Vec<f32> {
        let [w1, w2, w3] = &self.shared;
        let gate = w1.forward(x, t, Out::Bf16);
        let up = w3.forward(x, t, Out::Bf16);
        let lim = cfg.swiglu_limit;
        let h: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let (g, u) = if lim > 0.0 { (g.min(lim), u.clamp(-lim, lim)) } else { (g, u) };
                to_bf16(silu(g) * u)
            })
            .collect();
        w2.forward(&h, t, Out::Bf16)
    }
}
