//! MoE: router, routed experts through the expert cache, shared expert
//! (reference `Gate`, `Expert`, `MoE`).
//!
//! Routing: `sqrt(softplus(x . W_gate))`; the correction bias picks the top
//! 6 but the weights come from the unbiased scores, normalized and scaled by
//! `route_scale`. Outputs accumulate in f32 in ascending expert id order
//! (the reference's loop order), then the shared expert, then one bf16 round.

use std::sync::Arc;

use oaiy_engine::ecache::Ecache;
use oaiy_engine::store::WeightStore;
use oaiy_engine::Result;

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
    /// Workers for a decode step's routed experts, all of a token's at once (the same results as one at a time);
    /// none: one after another on the caller's thread.
    pub pool: Option<crate::cpu_experts::CpuExperts>,
    /// Where a prompt's busy experts (those [`GPU_MIN_ROWS`] or more of its tokens chose) have their matmuls made, a
    /// group of [`GPU_GROUP`] a call: a GPU's (`crate::expert::ExpertsKernel`). None: all on the CPU.
    pub gpu: Option<Arc<dyn crate::expert::ExpertsKernel>>,
}

/// Tokens of a prompt an expert needs before a GPU is worth its record's upload (18.8 MB): with fewer, the CPU's
/// matmuls of a few rows are quicker than the copy.
pub const GPU_MIN_ROWS: usize = 8;
/// Experts a GPU takes in one call: their records' upload bounded (about 600 MB), the matmuls a stage's round trip.
pub const GPU_GROUP: usize = 32;
/// Threads reading a layer's expert records from the drive at once.
const READERS: usize = 8;

pub struct Moe {
    layer: u32,
    gate: Weight,
    bias: Vec<f32>,
    shared: [Weight; 3],
}

impl Moe {
    /// Its dense weights by name (the router and the shared expert), for [`crate::model::Model::offload`].
    pub(crate) fn weights_mut(&mut self) -> Vec<(&'static str, &mut Weight)> {
        let [w1, w2, w3] = &mut self.shared;
        vec![("gate", &mut self.gate), ("shared.w1", w1), ("shared.w2", w2), ("shared.w3", w3)]
    }
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
    route_mixed(cfg, logits, bias, None, t)
}

/// [`route`] where image tokens (`image_bias.1[i]`) pick their experts with
/// the vision correction bias `image_bias.0` (the checkpoint's `bias_vl`,
/// the reference `Gate`'s `image_mask` path).
pub fn route_mixed(cfg: &Config, logits: &[f32], bias: &[f32], image_bias: Option<(&[f32], &[bool])>, t: usize) -> Vec<Route> {
    let n = bias.len();
    (0..t)
        .map(|i| {
            let bias = match image_bias {
                Some((vl, mask)) if mask[i] => vl,
                _ => bias,
            };
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
        // Each expert's (tokens, outputs), every expert's computed before any is added, then added in ascending
        // expert order, as one at a time added them: the same sums either way.
        // Their records, read on several threads at once: an SSD serves a queue of reads several times faster than one
        // at a time (the cache reads a record outside its lock, and the store keeps a scratch buffer a read).
        let acquire = |e: u32| experts.cache.acquire(self.layer, e, experts.store.as_ref());
        let leases: Vec<oaiy_engine::ecache::HostLease> = if used.len() <= 1 {
            used.iter().map(|&e| acquire(e)).collect::<Result<Vec<_>>>()?
        } else {
            let readers = READERS.min(used.len());
            let got: Vec<Result<oaiy_engine::ecache::HostLease>> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..readers)
                    .map(|r| {
                        let used = &used;
                        let acquire = &acquire;
                        scope.spawn(move || (r..used.len()).step_by(readers).map(|i| (i, acquire(used[i]))).collect::<Vec<_>>())
                    })
                    .collect();
                let mut all: Vec<(usize, Result<oaiy_engine::ecache::HostLease>)> = handles.into_iter().flat_map(|h| h.join().expect("an expert reader panicked")).collect();
                all.sort_by_key(|(i, _)| *i);
                all.into_iter().map(|(_, r)| r).collect()
            });
            got.into_iter().collect::<Result<Vec<_>>>()?
        };
        let outs: Vec<(Vec<usize>, Vec<f32>)> = match (&experts.pool, t) {
            // A decode step: the token's experts at once on the workers.
            (Some(pool), 1) => {
                let r = &routes[0];
                let weights: Vec<f32> = used.iter().map(|e| r.weights[r.experts.iter().position(|x| x == e).expect("a used expert is routed")]).collect();
                let records: Vec<_> = leases.iter().map(|l| l.to_arc()).collect();
                pool.forward(&records, &weights, x, cfg.swiglu_limit).into_iter().map(|out| (vec![0], out)).collect()
            }
            // A prompt: each expert's tokens as one batch, the busy experts' on the GPU when there is one, the rest
            // spread over threads.
            _ => {
                // every token routed to each expert, in token order
                let gathered: Vec<(Vec<usize>, Vec<f32>, Vec<f32>)> = used
                    .iter()
                    .map(|e| {
                        let (mut toks, mut xs, mut ws) = (Vec::new(), Vec::new(), Vec::new());
                        for (i, r) in routes.iter().enumerate() {
                            if let Some(k) = r.experts.iter().position(|x| x == e) {
                                toks.push(i);
                                xs.extend_from_slice(&x[i * d..(i + 1) * d]);
                                ws.push(r.weights[k]);
                            }
                        }
                        (toks, xs, ws)
                    })
                    .collect();
                let mut done: Vec<Option<Vec<f32>>> = (0..used.len()).map(|_| None).collect();
                if let Some(gpu) = &experts.gpu {
                    let busy: Vec<usize> = (0..used.len()).filter(|&j| gathered[j].0.len() >= GPU_MIN_ROWS).collect();
                    for group in busy.chunks(GPU_GROUP) {
                        let jobs: Vec<crate::expert::ExpertJob<'_>> =
                            group.iter().map(|&j| crate::expert::ExpertJob { record: &leases[j], x: &gathered[j].1, weights: &gathered[j].2 }).collect();
                        for (&j, out) in group.iter().zip(gpu.forward(&jobs, cfg.swiglu_limit)) {
                            done[j] = Some(out);
                        }
                    }
                }
                let rest: Vec<usize> = (0..used.len()).filter(|&j| done[j].is_none()).collect();
                let batch = |j: usize| (j, expert_forward_batch(&leases[j], &gathered[j].1, Some(&gathered[j].2), cfg.swiglu_limit));
                let threads = if experts.pool.is_some() { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { 1 };
                let computed: Vec<(usize, Vec<f32>)> = if threads <= 1 || rest.len() <= 1 {
                    rest.iter().map(|&j| batch(j)).collect()
                } else {
                    let per = rest.len().div_ceil(threads);
                    std::thread::scope(|scope| {
                        let handles: Vec<_> = rest.chunks(per).map(|part| scope.spawn(move || part.iter().map(|&j| batch(j)).collect::<Vec<_>>())).collect();
                        handles.into_iter().flat_map(|h| h.join().expect("an expert worker panicked")).collect()
                    })
                };
                for (j, out) in computed {
                    done[j] = Some(out);
                }
                gathered.into_iter().zip(done).map(|((toks, _, _), out)| (toks, out.expect("every expert computed"))).collect()
            }
        };
        for (toks, out) in outs {
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
